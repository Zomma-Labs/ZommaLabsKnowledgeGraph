"""
Backfill Script: Re-embed TopicNodes and EntityNodes with voyage-3-large.

Standardizes all embeddings to use the same model as facts.

Usage:
    uv run python src/scripts/backfill_topic_entity_embeddings.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.util.services import get_services
from src.util.llm_client import LLMClient


def backfill_topic_embeddings(batch_size: int = 50):
    """Re-embed all TopicNodes with voyage-3-large."""
    services = get_services()
    neo4j = services.neo4j
    embeddings = LLMClient.get_embeddings(model="voyage-3-large")

    print("=== Backfilling TopicNode embeddings ===")

    # Count topics
    count_result = neo4j.query("""
        MATCH (t:TopicNode)
        WHERE t.name IS NOT NULL
        RETURN count(t) as count
    """)
    total_count = count_result[0]["count"] if count_result else 0
    print(f"Found {total_count} TopicNodes to re-embed")

    if total_count == 0:
        return

    processed = 0
    offset = 0

    while processed < total_count:
        batch = neo4j.query("""
            MATCH (t:TopicNode)
            WHERE t.name IS NOT NULL
            RETURN t.uuid as uuid, t.name as name
            SKIP $offset
            LIMIT $batch_size
        """, {"offset": offset, "batch_size": batch_size})

        if not batch:
            break

        # Generate new embeddings
        names = [row["name"] for row in batch]
        new_embeddings = embeddings.embed_documents(names)

        # Prepare updates
        updates = [
            {"uuid": row["uuid"], "embedding": emb}
            for row, emb in zip(batch, new_embeddings)
        ]

        # Bulk update
        neo4j.query("""
            UNWIND $updates AS u
            MATCH (t:TopicNode {uuid: u.uuid})
            SET t.embedding = u.embedding
        """, {"updates": updates})

        processed += len(batch)
        offset += batch_size
        print(f"  Re-embedded {processed}/{total_count} topics...")

    print(f"Topic backfill complete! Re-embedded {processed} topics.")


def backfill_entity_embeddings(batch_size: int = 50):
    """Re-embed all EntityNodes with voyage-3-large."""
    services = get_services()
    neo4j = services.neo4j
    embeddings = LLMClient.get_embeddings(model="voyage-3-large")

    print("\n=== Backfilling EntityNode name_embedding ===")

    # Count entities
    count_result = neo4j.query("""
        MATCH (e:EntityNode)
        WHERE e.name IS NOT NULL
        RETURN count(e) as count
    """)
    total_count = count_result[0]["count"] if count_result else 0
    print(f"Found {total_count} EntityNodes to re-embed")

    if total_count == 0:
        return

    processed = 0
    offset = 0

    while processed < total_count:
        batch = neo4j.query("""
            MATCH (e:EntityNode)
            WHERE e.name IS NOT NULL
            RETURN e.uuid as uuid, e.name as name
            SKIP $offset
            LIMIT $batch_size
        """, {"offset": offset, "batch_size": batch_size})

        if not batch:
            break

        # Generate new embeddings
        names = [row["name"] for row in batch]
        new_embeddings = embeddings.embed_documents(names)

        # Prepare updates
        updates = [
            {"uuid": row["uuid"], "embedding": emb}
            for row, emb in zip(batch, new_embeddings)
        ]

        # Bulk update
        neo4j.query("""
            UNWIND $updates AS u
            MATCH (e:EntityNode {uuid: u.uuid})
            SET e.name_embedding = u.embedding
        """, {"updates": updates})

        processed += len(batch)
        offset += batch_size
        print(f"  Re-embedded {processed}/{total_count} entities...")

    print(f"Entity backfill complete! Re-embedded {processed} entities.")


def main():
    print("Backfilling embeddings with voyage-3-large...\n")
    backfill_topic_embeddings()
    backfill_entity_embeddings()
    print("\nDone!")


if __name__ == "__main__":
    main()
