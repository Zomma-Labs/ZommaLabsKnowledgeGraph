#!/usr/bin/env python3
"""
Diagnose Extraction vs Resolution Mismatch
==========================================

Checks if facts reference entities that weren't resolved.

Usage:
    uv run scripts/diagnose_extraction_resolution.py
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.util.services import get_services


def check_fact_entity_references(neo4j, group_id: str = "default"):
    """Check if facts reference entities that exist in the graph."""
    print("=" * 60)
    print("CHECKING FACT AND ENTITY RELATIONSHIPS")
    print("=" * 60)

    # Get all entity names
    entities = neo4j.query("""
        MATCH (e:EntityNode {group_id: $uid})
        RETURN e.name AS name
    """, {"uid": group_id})
    entity_names = {e["name"] for e in entities}
    entity_names_lower = {e["name"].lower() for e in entities}
    print(f"Total entity names: {len(entity_names)}")

    # Get all topic names
    topics = neo4j.query("""
        MATCH (t:TopicNode {group_id: $uid})
        RETURN t.name AS name
    """, {"uid": group_id})
    topic_names = {t["name"] for t in topics}
    topic_names_lower = {t["name"].lower() for t in topics}
    print(f"Total topic names: {len(topic_names)}")

    # Check all edge types that connect entities to chunks
    print("\n" + "=" * 60)
    print("ANALYZING ENTITY -> CHUNK EDGES")
    print("=" * 60)

    # Count edges from entities to chunks
    entity_to_chunk = neo4j.query("""
        MATCH (e:EntityNode {group_id: $uid})-[r]->(c:EpisodicNode)
        RETURN type(r) AS rel_type, count(*) AS cnt
        ORDER BY cnt DESC
        LIMIT 30
    """, {"uid": group_id})
    print("\nEdges EntityNode -> EpisodicNode:")
    total_entity_edges = 0
    for row in entity_to_chunk:
        print(f"  {row['rel_type']}: {row['cnt']}")
        total_entity_edges += row['cnt']
    print(f"Total: {total_entity_edges}")

    # Count edges from topics to chunks
    topic_to_chunk = neo4j.query("""
        MATCH (t:TopicNode {group_id: $uid})-[r]->(c:EpisodicNode)
        RETURN type(r) AS rel_type, count(*) AS cnt
        ORDER BY cnt DESC
        LIMIT 20
    """, {"uid": group_id})
    print("\nEdges TopicNode -> EpisodicNode:")
    total_topic_edges = 0
    for row in topic_to_chunk:
        print(f"  {row['rel_type']}: {row['cnt']}")
        total_topic_edges += row['cnt']
    print(f"Total: {total_topic_edges}")

    # Check chunk -> entity edges (_TARGET)
    print("\n" + "=" * 60)
    print("ANALYZING CHUNK -> ENTITY EDGES (_TARGET)")
    print("=" * 60)

    chunk_to_entity = neo4j.query("""
        MATCH (c:EpisodicNode {group_id: $uid})-[r]->(e:EntityNode)
        RETURN type(r) AS rel_type, count(*) AS cnt
        ORDER BY cnt DESC
        LIMIT 30
    """, {"uid": group_id})
    print("\nEdges EpisodicNode -> EntityNode:")
    total_target_edges = 0
    for row in chunk_to_entity:
        print(f"  {row['rel_type']}: {row['cnt']}")
        total_target_edges += row['cnt']
    print(f"Total: {total_target_edges}")

    # Check for facts where both subject and object exist
    print("\n" + "=" * 60)
    print("ANALYZING COMPLETE FACT PATTERNS")
    print("=" * 60)

    complete_facts = neo4j.query("""
        MATCH (subj)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(obj)
        WHERE (subj:EntityNode OR subj:TopicNode)
          AND (obj:EntityNode OR obj:TopicNode)
          AND r1.fact_id IS NOT NULL
          AND r1.fact_id = r2.fact_id
        RETURN count(DISTINCT r1.fact_id) AS complete_facts,
               count(DISTINCT c) AS chunks_with_facts
    """, {"uid": group_id})
    print(f"Complete Subject->Chunk->Object patterns: {complete_facts[0]['complete_facts']}")
    print(f"Chunks with complete patterns: {complete_facts[0]['chunks_with_facts']}")

    # Sample some incomplete patterns
    print("\n" + "=" * 60)
    print("SAMPLING INCOMPLETE FACT PATTERNS")
    print("=" * 60)

    # Facts with subject edge but no object edge
    incomplete = neo4j.query("""
        MATCH (subj)-[r1]->(c:EpisodicNode {group_id: $uid})
        WHERE (subj:EntityNode OR subj:TopicNode)
          AND r1.fact_id IS NOT NULL
          AND NOT EXISTS {
            MATCH (c)-[r2]->(obj)
            WHERE (obj:EntityNode OR obj:TopicNode)
              AND r2.fact_id = r1.fact_id
          }
        MATCH (f:FactNode {uuid: r1.fact_id})
        RETURN subj.name AS subject, type(r1) AS rel_type, r1.fact_id AS fact_id,
               f.content AS fact_content
        LIMIT 10
    """, {"uid": group_id})
    print(f"\nFacts with Subject->Chunk but no Chunk->Object: {len(incomplete)}")
    for row in incomplete[:5]:
        print(f"  Subject: {row['subject']}")
        print(f"  Rel: {row['rel_type']}")
        print(f"  Fact: {row['fact_content'][:80]}...")
        print()


def check_entity_lookup_pattern(neo4j, group_id: str = "default"):
    """Check how entity_lookup is built and what's missing."""
    print("\n" + "=" * 60)
    print("ANALYZING ENTITY LOOKUP BUILDING")
    print("=" * 60)

    # The entity_lookup is built from extraction.entities
    # Then matched against entity_lookup_global which comes from entity_registry.resolve()

    # Get entities that have edges to chunks
    entities_with_edges = neo4j.query("""
        MATCH (e:EntityNode {group_id: $uid})-[r]->(c:EpisodicNode)
        WHERE r.fact_id IS NOT NULL
        RETURN DISTINCT e.name AS name
    """, {"uid": group_id})
    print(f"Entities with fact edges: {len(entities_with_edges)}")

    # Get topics that have edges to chunks (non-DISCUSSES)
    topics_with_edges = neo4j.query("""
        MATCH (t:TopicNode {group_id: $uid})-[r]->(c:EpisodicNode)
        WHERE r.fact_id IS NOT NULL
        RETURN DISTINCT t.name AS name
    """, {"uid": group_id})
    print(f"Topics with fact edges (as subject): {len(topics_with_edges)}")

    # Check for DISCUSSES relationships (topics linked to chunks)
    discusses_count = neo4j.query("""
        MATCH (c:EpisodicNode {group_id: $uid})-[:DISCUSSES]->(t:TopicNode)
        RETURN count(*) AS cnt
    """, {"uid": group_id})
    print(f"DISCUSSES relationships (chunk -> topic): {discusses_count[0]['cnt']}")


def main():
    print("Initializing connections...")
    svc = get_services()
    neo4j = svc.neo4j
    group_id = "default"

    check_fact_entity_references(neo4j, group_id)
    check_entity_lookup_pattern(neo4j, group_id)


if __name__ == "__main__":
    main()
