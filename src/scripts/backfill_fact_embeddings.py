"""
Backfill Script: Re-embed FactNodes with voyage-finance-2

The fact embeddings may have been created with a different model.
This script re-embeds all facts with the current embedding model.

Usage:
    uv run python src/scripts/backfill_fact_embeddings.py [--batch-size N]
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.util.services import get_services
from src.util.llm_client import LLMClient


def backfill_fact_embeddings(batch_size: int = 50, model: str = "voyage-3-large"):
    """Re-embed all FactNodes with the specified embedding model."""
    services = get_services()
    neo4j = services.neo4j
    embeddings = LLMClient.get_embeddings(model=model)
    print(f"Using embedding model: {model}")

    # Count facts
    count_result = neo4j.query("""
        MATCH (f:FactNode)
        WHERE f.content IS NOT NULL
        RETURN count(f) as count
    """)
    total_count = count_result[0]["count"] if count_result else 0
    print(f"Found {total_count} FactNodes to re-embed")

    if total_count == 0:
        print("Nothing to backfill!")
        return

    # Process in batches
    processed = 0
    offset = 0

    while processed < total_count:
        # Fetch batch
        batch = neo4j.query("""
            MATCH (f:FactNode)
            WHERE f.content IS NOT NULL
            RETURN f.uuid as uuid, f.content as content
            SKIP $offset
            LIMIT $batch_size
        """, {"offset": offset, "batch_size": batch_size})

        if not batch:
            break

        # Generate new embeddings
        contents = [row["content"] for row in batch]
        new_embeddings = embeddings.embed_documents(contents)

        # Prepare updates
        updates = [
            {"uuid": row["uuid"], "embedding": emb}
            for row, emb in zip(batch, new_embeddings)
        ]

        # Bulk update
        neo4j.query("""
            UNWIND $updates AS u
            MATCH (f:FactNode {uuid: u.uuid})
            SET f.embedding = u.embedding
        """, {"updates": updates})

        processed += len(batch)
        offset += batch_size
        print(f"  Re-embedded {processed}/{total_count} facts...")

    print(f"Backfill complete! Re-embedded {processed} facts.")


def main():
    parser = argparse.ArgumentParser(description="Re-embed FactNodes")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--model", type=str, default="voyage-3-large", help="Embedding model")
    args = parser.parse_args()

    backfill_fact_embeddings(args.batch_size, args.model)


if __name__ == "__main__":
    main()
