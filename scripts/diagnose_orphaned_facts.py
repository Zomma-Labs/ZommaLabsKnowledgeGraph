#!/usr/bin/env python3
"""
Diagnose Orphaned Facts
=======================

Deep dive into why facts don't have proper Subject->Chunk->Object patterns.

Usage:
    uv run scripts/diagnose_orphaned_facts.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.util.services import get_services


def analyze_orphaned_facts(neo4j, group_id: str = "default"):
    """Analyze why facts are orphaned."""
    print("=" * 60)
    print("ANALYZING ORPHANED FACTS")
    print("=" * 60)

    # Get all facts
    all_facts = neo4j.query("""
        MATCH (f:FactNode {group_id: $uid})
        RETURN f.uuid AS fact_id, f.content AS content
    """, {"uid": group_id})
    print(f"Total facts: {len(all_facts)}")

    # Get facts that have Subject->Chunk->Object pattern
    connected_facts = neo4j.query("""
        MATCH (subj)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(obj)
        WHERE (subj:EntityNode OR subj:TopicNode)
          AND (obj:EntityNode OR obj:TopicNode)
          AND r1.fact_id IS NOT NULL
          AND r1.fact_id = r2.fact_id
        RETURN DISTINCT r1.fact_id AS fact_id
    """, {"uid": group_id})
    connected_set = {r["fact_id"] for r in connected_facts}
    print(f"Facts with Subject->Chunk->Object: {len(connected_set)}")

    # Get facts that only have CONTAINS_FACT connection
    only_contains = neo4j.query("""
        MATCH (c:EpisodicNode {group_id: $uid})-[:CONTAINS_FACT]->(f:FactNode {group_id: $uid})
        WHERE NOT EXISTS {
            MATCH (subj)-[r1 {fact_id: f.uuid}]->(c2:EpisodicNode)-[r2 {fact_id: f.uuid}]->(obj)
            WHERE (subj:EntityNode OR subj:TopicNode)
              AND (obj:EntityNode OR obj:TopicNode)
        }
        RETURN f.uuid AS fact_id, f.content AS content, c.uuid AS chunk_id
        LIMIT 20
    """, {"uid": group_id})
    print(f"\nSample of orphaned facts (have CONTAINS_FACT but no entity edges):")
    for row in only_contains[:10]:
        print(f"  [{row['fact_id'][:8]}...] {row['content'][:80]}...")
        print(f"    Chunk: {row['chunk_id'][:8]}...")

    # Check what relationships exist for these orphaned facts
    print("\n" + "=" * 60)
    print("CHECKING RELATIONSHIP TYPES")
    print("=" * 60)

    # Get all edge types that have fact_id
    edge_types = neo4j.query("""
        MATCH ()-[r]->()
        WHERE r.fact_id IS NOT NULL
        RETURN DISTINCT type(r) AS rel_type, count(*) AS cnt
        ORDER BY cnt DESC
        LIMIT 20
    """)
    print("\nRelationship types with fact_id property:")
    for row in edge_types:
        print(f"  {row['rel_type']}: {row['cnt']}")

    # Check if orphaned facts have ANY outgoing edges from entities
    print("\n" + "=" * 60)
    print("ANALYZING EDGE PATTERNS")
    print("=" * 60)

    # Check for edges TO EpisodicNode but missing edges FROM EpisodicNode
    partial_edges = neo4j.query("""
        MATCH (subj)-[r1]->(c:EpisodicNode {group_id: $uid})
        WHERE (subj:EntityNode OR subj:TopicNode)
          AND r1.fact_id IS NOT NULL
          AND NOT EXISTS {
            MATCH (c)-[r2]->(obj)
            WHERE r2.fact_id = r1.fact_id
              AND (obj:EntityNode OR obj:TopicNode)
          }
        RETURN subj.name AS subject, type(r1) AS rel_type, r1.fact_id AS fact_id, c.uuid AS chunk_id
        LIMIT 10
    """, {"uid": group_id})
    print(f"\nEdges that go Subject->Chunk but missing Chunk->Object: {len(partial_edges)}")
    for row in partial_edges[:5]:
        print(f"  {row['subject']} -[{row['rel_type']}]-> Chunk[{row['chunk_id'][:8]}...]")
        print(f"    fact_id: {row['fact_id'][:8]}...")

    # Check for _TARGET edges (should exist for all facts with subject edges)
    target_edges = neo4j.query("""
        MATCH ()-[r]->()
        WHERE type(r) ENDS WITH '_TARGET'
          AND r.fact_id IS NOT NULL
        RETURN count(*) AS cnt
    """)
    print(f"\nTotal _TARGET edges: {target_edges[0]['cnt']}")

    # Check for mismatched fact_ids
    print("\n" + "=" * 60)
    print("VERIFYING FACT_ID MATCHING")
    print("=" * 60)

    # Get sample of facts and check if they have matching r1.fact_id = r2.fact_id
    matching_check = neo4j.query("""
        MATCH (subj)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(obj)
        WHERE r1.fact_id IS NOT NULL
          AND r2.fact_id IS NOT NULL
          AND (subj:EntityNode OR subj:TopicNode)
          AND (obj:EntityNode OR obj:TopicNode)
        RETURN r1.fact_id = r2.fact_id AS match, count(*) AS cnt
    """, {"uid": group_id})
    print("fact_id matching between r1 and r2:")
    for row in matching_check:
        print(f"  Matching: {row['match']} - Count: {row['cnt']}")

    # Check entities that exist
    print("\n" + "=" * 60)
    print("ENTITY NODE CHECK")
    print("=" * 60)

    entity_count = neo4j.query("""
        MATCH (e:EntityNode {group_id: $uid})
        RETURN count(e) AS cnt
    """, {"uid": group_id})
    print(f"Total EntityNodes: {entity_count[0]['cnt']}")

    topic_count = neo4j.query("""
        MATCH (t:TopicNode {group_id: $uid})
        RETURN count(t) AS cnt
    """, {"uid": group_id})
    print(f"Total TopicNodes: {topic_count[0]['cnt']}")

    # Check entities with outgoing edges
    entities_with_edges = neo4j.query("""
        MATCH (e:EntityNode {group_id: $uid})-[r]->(c:EpisodicNode)
        WHERE r.fact_id IS NOT NULL
        RETURN count(DISTINCT e) AS cnt
    """, {"uid": group_id})
    print(f"Entities with outgoing edges to EpisodicNode: {entities_with_edges[0]['cnt']}")


def main():
    print("Initializing Neo4j connection...")
    svc = get_services()
    neo4j = svc.neo4j
    group_id = "default"

    analyze_orphaned_facts(neo4j, group_id)


if __name__ == "__main__":
    main()
