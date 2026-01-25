"""
Backfill topic definitions from Qdrant ontology to Neo4j TopicNodes.

Usage:
    uv run src/scripts/backfill_topic_definitions.py
    uv run src/scripts/backfill_topic_definitions.py --group-id default
    uv run src/scripts/backfill_topic_definitions.py --dry-run
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from src.util.services import get_services

QDRANT_PATH = "./qdrant_topics"
COLLECTION_NAME = "topic_ontology"


def backfill_definitions(group_id: str = "default", dry_run: bool = False):
    """
    Backfill topic definitions from Qdrant ontology to Neo4j TopicNodes.

    Strategy:
    1. Get all TopicNodes from Neo4j that don't have definitions
    2. For each topic name, look up the definition in Qdrant ontology
    3. Update Neo4j with the definition
    """
    services = get_services()
    neo4j = services.neo4j
    embeddings = services.embeddings

    # Initialize Qdrant client
    qdrant = QdrantClient(path=QDRANT_PATH)

    if not qdrant.collection_exists(COLLECTION_NAME):
        print(f"Error: Qdrant collection '{COLLECTION_NAME}' does not exist")
        print("Run the topic ontology setup script first")
        return

    # Get all TopicNodes without definitions
    print(f"Finding TopicNodes without definitions (group_id={group_id})...")

    topics_without_def = neo4j.query("""
        MATCH (t:TopicNode {group_id: $gid})
        WHERE t.definition IS NULL OR t.definition = ''
        RETURN t.name as name, t.uuid as uuid
    """, {"gid": group_id})

    if not topics_without_def:
        print("All TopicNodes already have definitions!")
        return

    print(f"Found {len(topics_without_def)} TopicNodes without definitions")

    # Look up each topic in Qdrant
    updates = []
    not_found = []

    for topic in topics_without_def:
        name = topic["name"]

        # Embed the topic name and search Qdrant
        vector = embeddings.embed_query(name)

        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=1,
            with_payload=True,
            score_threshold=0.7  # High threshold for exact match
        ).points

        if results:
            hit = results[0]
            # Check if it's actually the same topic (label match)
            label = hit.payload.get("label", "")
            definition = hit.payload.get("definition", "")

            if label.lower() == name.lower() and definition:
                updates.append({
                    "uuid": topic["uuid"],
                    "name": name,
                    "definition": definition
                })
                print(f"  + {name}: {definition[:60]}...")
            else:
                # Try exact name match in payload
                exact_results = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter={
                        "must": [
                            {"key": "label", "match": {"value": name}}
                        ]
                    },
                    limit=1,
                    with_payload=True
                )[0]

                if exact_results:
                    definition = exact_results[0].payload.get("definition", "")
                    if definition:
                        updates.append({
                            "uuid": topic["uuid"],
                            "name": name,
                            "definition": definition
                        })
                        print(f"  + {name}: {definition[:60]}...")
                    else:
                        not_found.append(name)
                        print(f"  - {name}: No definition in ontology")
                else:
                    not_found.append(name)
                    print(f"  ? {name}: Not found in ontology")
        else:
            not_found.append(name)
            print(f"  ? {name}: No match in ontology")

    print(f"\nSummary:")
    print(f"  Found definitions: {len(updates)}")
    print(f"  Not found: {len(not_found)}")

    if not_found:
        print(f"\nTopics without definitions in ontology:")
        for name in not_found[:10]:
            print(f"    - {name}")
        if len(not_found) > 10:
            print(f"    ... and {len(not_found) - 10} more")

    if dry_run:
        print("\n[DRY RUN] Would update {len(updates)} TopicNodes")
        return

    if not updates:
        print("\nNo updates to make")
        return

    # Batch update Neo4j
    print(f"\nUpdating {len(updates)} TopicNodes in Neo4j...")

    neo4j.query("""
        UNWIND $updates AS u
        MATCH (t:TopicNode {uuid: u.uuid})
        SET t.definition = u.definition
    """, {"updates": updates})

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill topic definitions from Qdrant to Neo4j")
    parser.add_argument("--group-id", default="default", help="Group ID to update")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without making changes")
    args = parser.parse_args()

    backfill_definitions(group_id=args.group_id, dry_run=args.dry_run)
