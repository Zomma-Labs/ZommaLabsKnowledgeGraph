#!/usr/bin/env python3
"""Check if topic resolution is causing the missing object edges."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.util.services import get_services

neo4j = get_services().neo4j

print("=" * 60)
print("CHECKING TOPIC RESOLUTION ISSUES")
print("=" * 60)

# Get the orphaned subject edges (subject exists but no matching object edge)
orphaned = neo4j.query('''
    MATCH (subj)-[r1]->(c:EpisodicNode {group_id: "default"})
    WHERE r1.fact_id IS NOT NULL
      AND (subj:EntityNode OR subj:TopicNode)
      AND NOT EXISTS {
        MATCH (c)-[r2]->(obj)
        WHERE r2.fact_id = r1.fact_id
          AND (obj:EntityNode OR obj:TopicNode)
      }
    MATCH (f:FactNode {uuid: r1.fact_id})
    RETURN subj.name as subject, labels(subj) as subject_labels,
           type(r1) as rel_type, r1.fact_id as fact_id,
           f.content as fact_content, c.content as chunk_content
    LIMIT 20
''')

print(f"\nOrphaned subject edges (missing object): {len(orphaned)}")
print("\nSamples:")
for r in orphaned[:10]:
    print(f"\n  Subject: {r['subject']} ({r['subject_labels']})")
    print(f"  Rel: {r['rel_type']}")
    print(f"  Fact: {r['fact_content'][:100]}...")

    # Try to infer what the object should be from the fact text
    fact = r['fact_content']
    # Look for patterns like "X caused Y" or "X in Y"
    print(f"  (Check fact for expected object)")

# Check if there are TopicNodes that exist but aren't being linked
print("\n" + "=" * 60)
print("CHECKING TOPIC NODE COVERAGE")
print("=" * 60)

topic_count = neo4j.query('''
    MATCH (t:TopicNode {group_id: "default"})
    RETURN count(*) as cnt
''')[0]["cnt"]
print(f"Total TopicNodes: {topic_count}")

topics_with_incoming = neo4j.query('''
    MATCH (c:EpisodicNode {group_id: "default"})-[r]->(t:TopicNode)
    WHERE r.fact_id IS NOT NULL
    RETURN count(DISTINCT t) as cnt
''')[0]["cnt"]
print(f"Topics with incoming fact edges (as object): {topics_with_incoming}")

topics_with_discusses = neo4j.query('''
    MATCH (c:EpisodicNode {group_id: "default"})-[:DISCUSSES]->(t:TopicNode)
    RETURN count(DISTINCT t) as cnt
''')[0]["cnt"]
print(f"Topics with DISCUSSES edges: {topics_with_discusses}")

# Sample topics that only have DISCUSSES but no fact edges
print("\n" + "=" * 60)
print("TOPICS ONLY CONNECTED VIA DISCUSSES (not as fact object)")
print("=" * 60)

discusses_only = neo4j.query('''
    MATCH (c:EpisodicNode {group_id: "default"})-[:DISCUSSES]->(t:TopicNode {group_id: "default"})
    WHERE NOT EXISTS {
        MATCH (c2:EpisodicNode)-[r]->(t)
        WHERE r.fact_id IS NOT NULL
    }
    RETURN DISTINCT t.name as topic
    LIMIT 20
''')
print(f"Topics only connected via DISCUSSES: {len(discusses_only)}")
for t in discusses_only[:10]:
    print(f"  - {t['topic']}")
