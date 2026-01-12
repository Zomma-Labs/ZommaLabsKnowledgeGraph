"""
Backfill Script: Add name_only_embedding to existing EntityNodes

This script adds the name_only_embedding field to all EntityNodes that
currently only have the name_embedding (name + summary) field.

Usage:
    uv run src/scripts/backfill_name_only_embeddings.py [--group-id GROUP_ID] [--batch-size N]
"""

import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.util.services import get_services


def backfill_name_only_embeddings(group_id: str = "default", batch_size: int = 100):
    """
    Add name_only_embedding to EntityNodes that don't have it.

    Args:
        group_id: Tenant ID to filter entities (default: "default")
        batch_size: Number of entities to process per batch
    """
    services = get_services()
    neo4j = services.neo4j
    embeddings = services.embeddings

    # 1. Count entities needing backfill
    count_result = neo4j.query("""
        MATCH (e:EntityNode {group_id: $group_id})
        WHERE e.name_only_embedding IS NULL
        RETURN count(e) as count
    """, {"group_id": group_id})

    total_count = count_result[0]["count"] if count_result else 0
    print(f"Found {total_count} EntityNodes needing name_only_embedding backfill")

    if total_count == 0:
        print("Nothing to backfill!")
        return

    # 2. Process in batches
    processed = 0
    while processed < total_count:
        # Fetch batch of entities
        batch = neo4j.query("""
            MATCH (e:EntityNode {group_id: $group_id})
            WHERE e.name_only_embedding IS NULL
            RETURN e.uuid as uuid, e.name as name
            LIMIT $batch_size
        """, {"group_id": group_id, "batch_size": batch_size})

        if not batch:
            break

        # Generate embeddings for names
        updates = []
        for row in batch:
            entity_uuid = row["uuid"]
            entity_name = row["name"]

            # Generate name-only embedding
            name_only_embedding = embeddings.embed_query(entity_name)
            updates.append({
                "uuid": entity_uuid,
                "name_only_embedding": name_only_embedding
            })

        # Bulk update
        if updates:
            neo4j.query("""
                UNWIND $updates AS u
                MATCH (e:EntityNode {uuid: u.uuid, group_id: $group_id})
                SET e.name_only_embedding = u.name_only_embedding
            """, {"updates": updates, "group_id": group_id})

        processed += len(batch)
        print(f"  Processed {processed}/{total_count} entities...")

    print(f"Backfill complete! Updated {processed} entities.")


def main():
    parser = argparse.ArgumentParser(description="Backfill name_only_embedding for EntityNodes")
    parser.add_argument("--group-id", default="default", help="Tenant/group ID to filter entities")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    args = parser.parse_args()

    backfill_name_only_embeddings(args.group_id, args.batch_size)


if __name__ == "__main__":
    main()
