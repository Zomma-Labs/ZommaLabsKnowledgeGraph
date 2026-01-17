"""
Test OpenAI text-embedding-3-large vs Voyage embeddings for fact retrieval.

This script:
1. Backfills facts with OpenAI embeddings (to embedding_openai property)
2. Runs test queries comparing both embedding models
3. Shows which facts each model retrieves

Usage:
    uv run python scripts/test_openai_embeddings.py --backfill  # First time only
    uv run python scripts/test_openai_embeddings.py --query "Which districts reported growth?"
"""

import sys
import os
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import OpenAIEmbeddings
from src.util.services import get_services
from src.util.llm_client import LLMClient


def get_openai_embeddings():
    """Get OpenAI text-embedding-3-large model."""
    return OpenAIEmbeddings(model="text-embedding-3-large")


def backfill_openai_embeddings(batch_size: int = 50):
    """Add OpenAI embeddings to FactNodes as embedding_openai property."""
    services = get_services()
    neo4j = services.neo4j
    openai_emb = get_openai_embeddings()

    # Count facts
    count_result = neo4j.query("""
        MATCH (f:FactNode)
        WHERE f.content IS NOT NULL
        RETURN count(f) as count
    """)
    total_count = count_result[0]["count"] if count_result else 0
    print(f"Found {total_count} FactNodes to embed with OpenAI")

    if total_count == 0:
        return

    # Process in batches
    processed = 0
    offset = 0

    while processed < total_count:
        batch = neo4j.query("""
            MATCH (f:FactNode)
            WHERE f.content IS NOT NULL
            RETURN f.uuid as uuid, f.content as content
            SKIP $offset
            LIMIT $batch_size
        """, {"offset": offset, "batch_size": batch_size})

        if not batch:
            break

        # Generate OpenAI embeddings
        contents = [row["content"] for row in batch]
        new_embeddings = openai_emb.embed_documents(contents)

        # Update with new property
        updates = [
            {"uuid": row["uuid"], "embedding_openai": emb}
            for row, emb in zip(batch, new_embeddings)
        ]

        neo4j.query("""
            UNWIND $updates AS u
            MATCH (f:FactNode {uuid: u.uuid})
            SET f.embedding_openai = u.embedding_openai
        """, {"updates": updates})

        processed += len(batch)
        offset += batch_size
        print(f"  Embedded {processed}/{total_count} facts with OpenAI...")

    print(f"Backfill complete! Added OpenAI embeddings to {processed} facts.")


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_with_voyage(query: str, top_k: int = 10):
    """Search facts using Voyage embeddings."""
    services = get_services()
    neo4j = services.neo4j
    voyage_emb = LLMClient.get_embeddings(model="voyage-3-large")

    query_embedding = voyage_emb.embed_query(query)

    results = neo4j.query("""
        MATCH (f:FactNode)
        WHERE f.embedding IS NOT NULL
        WITH f, gds.similarity.cosine(f.embedding, $query_embedding) AS score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN f.uuid as uuid, f.content as content, f.subject as subject,
               f.edge_type as edge_type, f.object as object, score
    """, {"query_embedding": query_embedding, "top_k": top_k})

    return results


def search_with_openai(query: str, top_k: int = 10):
    """Search facts using OpenAI embeddings."""
    services = get_services()
    neo4j = services.neo4j
    openai_emb = get_openai_embeddings()

    query_embedding = openai_emb.embed_query(query)

    results = neo4j.query("""
        MATCH (f:FactNode)
        WHERE f.embedding_openai IS NOT NULL
        WITH f, gds.similarity.cosine(f.embedding_openai, $query_embedding) AS score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN f.uuid as uuid, f.content as content, f.subject as subject,
               f.edge_type as edge_type, f.object as object, score
    """, {"query_embedding": query_embedding, "top_k": top_k})

    return results


def compare_retrievals(query: str, top_k: int = 10):
    """Compare Voyage vs OpenAI retrieval for the same query."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    # Get results from both
    print("\nSearching with Voyage...")
    voyage_results = search_with_voyage(query, top_k)

    print("Searching with OpenAI...")
    openai_results = search_with_openai(query, top_k)

    # Display Voyage results
    print(f"\n{'─'*40}")
    print("VOYAGE (voyage-3-large) TOP {top_k}:")
    print(f"{'─'*40}")
    for i, r in enumerate(voyage_results):
        print(f"\n[{i+1}] Score: {r['score']:.4f}")
        print(f"    {r['subject']} --[{r['edge_type']}]--> {r['object']}")
        print(f"    Content: {r['content'][:100]}...")

    # Display OpenAI results
    print(f"\n{'─'*40}")
    print(f"OPENAI (text-embedding-3-large) TOP {top_k}:")
    print(f"{'─'*40}")
    for i, r in enumerate(openai_results):
        print(f"\n[{i+1}] Score: {r['score']:.4f}")
        print(f"    {r['subject']} --[{r['edge_type']}]--> {r['object']}")
        print(f"    Content: {r['content'][:100]}...")

    # Compare overlap
    voyage_uuids = {r['uuid'] for r in voyage_results}
    openai_uuids = {r['uuid'] for r in openai_results}
    overlap = voyage_uuids & openai_uuids

    print(f"\n{'─'*40}")
    print("COMPARISON:")
    print(f"{'─'*40}")
    print(f"  Voyage unique facts: {len(voyage_uuids - openai_uuids)}")
    print(f"  OpenAI unique facts: {len(openai_uuids - voyage_uuids)}")
    print(f"  Overlap: {len(overlap)} / {top_k}")

    # Show facts only OpenAI found
    openai_only = openai_uuids - voyage_uuids
    if openai_only:
        print(f"\n  Facts ONLY found by OpenAI:")
        for r in openai_results:
            if r['uuid'] in openai_only:
                print(f"    - {r['subject']} --[{r['edge_type']}]--> {r['object']} (score: {r['score']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Test OpenAI vs Voyage embeddings")
    parser.add_argument("--backfill", action="store_true", help="Backfill OpenAI embeddings first")
    parser.add_argument("--query", type=str, help="Query to test")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for backfill")
    args = parser.parse_args()

    if args.backfill:
        backfill_openai_embeddings(args.batch_size)

    if args.query:
        compare_retrievals(args.query, args.top_k)
    elif not args.backfill:
        # Default test queries
        test_queries = [
            "Which districts reported slight to modest economic growth?",
            "What happened to manufacturing in the Chicago District?",
            "How did tariffs affect prices?",
        ]
        for q in test_queries:
            compare_retrievals(q, args.top_k)


if __name__ == "__main__":
    main()
