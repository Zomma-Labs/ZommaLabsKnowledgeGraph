"""
Backfill Fact Vectors to Qdrant
===============================

This script populates the Qdrant fact vector store with existing facts from Neo4j.
It reads all FactNodes, embeds them with voyage-3-large, and indexes them to Qdrant.

Usage:
    uv run src/scripts/backfill_fact_vectors.py
    uv run src/scripts/backfill_fact_vectors.py --group-id tenant1  # Specific tenant
    uv run src/scripts/backfill_fact_vectors.py --clear  # Clear Qdrant first
"""

import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.util.neo4j_client import Neo4jClient
from src.util.fact_vector_store import get_fact_store
from src.util.llm_client import get_dedup_embeddings


def backfill_facts(group_id: str = None, clear: bool = False, batch_size: int = 50):
    """
    Backfill facts from Neo4j to Qdrant.

    Args:
        group_id: Optional tenant ID filter. If None, processes all tenants.
        clear: If True, clears the Qdrant collection first.
        batch_size: Number of facts to process at a time.
    """
    neo4j = Neo4jClient()
    fact_store = get_fact_store()
    embeddings = get_dedup_embeddings()

    if clear:
        print("Clearing Qdrant fact collection...")
        fact_store.clear()

    # Query facts from Neo4j
    # We need to get the subject and object entities connected through the EpisodicNode
    print("Querying facts from Neo4j...")

    if group_id:
        query = """
        MATCH (s)-[r1]->(c:EpisodicNode {group_id: $gid})-[r2]->(o)
        WHERE (s:EntityNode OR s:TopicNode)
          AND (o:EntityNode OR o:TopicNode)
          AND s.group_id = $gid
          AND o.group_id = $gid
          AND r1.fact_id = r2.fact_id
          AND r1.fact_id IS NOT NULL
        RETURN DISTINCT
            r1.fact_id as fact_id,
            s.name as subject,
            o.name as object,
            type(r1) as edge_type,
            r1.content as content,
            c.group_id as group_id
        """
        params = {"gid": group_id}
    else:
        query = """
        MATCH (s)-[r1]->(c:EpisodicNode)-[r2]->(o)
        WHERE (s:EntityNode OR s:TopicNode)
          AND (o:EntityNode OR o:TopicNode)
          AND r1.fact_id = r2.fact_id
          AND r1.fact_id IS NOT NULL
        RETURN DISTINCT
            r1.fact_id as fact_id,
            s.name as subject,
            o.name as object,
            type(r1) as edge_type,
            r1.content as content,
            c.group_id as group_id
        """
        params = {}

    results = neo4j.query(query, params)
    print(f"Found {len(results)} facts to backfill")

    if not results:
        print("No facts found. Exiting.")
        neo4j.close()
        return

    # Process in batches
    total_indexed = 0
    for i in range(0, len(results), batch_size):
        batch = results[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} ({len(batch)} facts)...")

        # Get content texts for embedding
        texts = [r.get("content", "") or f"{r['subject']} {r['edge_type']} {r['object']}" for r in batch]

        # Embed the batch
        try:
            batch_embeddings = embeddings.embed_documents(texts)
        except Exception as e:
            print(f"Error embedding batch: {e}")
            continue

        # Prepare facts for Qdrant
        qdrant_facts = []
        for j, r in enumerate(batch):
            fact_id = r["fact_id"]
            if not fact_id:
                continue

            qdrant_facts.append({
                "fact_id": fact_id,
                "embedding": batch_embeddings[j],
                "group_id": r.get("group_id", "default"),
                "subject": r["subject"],
                "object": r["object"],
                "edge_type": r["edge_type"],
                "content": r.get("content", "") or f"{r['subject']} {r['edge_type']} {r['object']}"
            })

        # Index to Qdrant
        if qdrant_facts:
            fact_store.index_facts_batch(qdrant_facts)
            total_indexed += len(qdrant_facts)
            print(f"  Indexed {len(qdrant_facts)} facts (total: {total_indexed})")

    print(f"\nBackfill complete! Indexed {total_indexed} facts to Qdrant.")

    # Print stats
    stats = fact_store.get_stats()
    print(f"Qdrant collection stats: {stats}")

    # Close connections
    fact_store.close()
    neo4j.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill fact vectors to Qdrant")
    parser.add_argument("--group-id", type=str, help="Filter by tenant/group ID")
    parser.add_argument("--clear", action="store_true", help="Clear Qdrant collection first")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding")
    args = parser.parse_args()

    backfill_facts(
        group_id=args.group_id,
        clear=args.clear,
        batch_size=args.batch_size
    )
