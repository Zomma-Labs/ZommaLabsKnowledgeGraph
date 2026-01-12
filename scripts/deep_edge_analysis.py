#!/usr/bin/env python3
"""Deep analysis of edge creation issues."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.util.services import get_services

neo4j = get_services().neo4j

print("=" * 60)
print("EDGE ANALYSIS")
print("=" * 60)

# 1. Get all unique fact_ids from edges
print("\n1. Checking fact_id distribution across edges...")

subject_fact_ids = neo4j.query('''
    MATCH (e)-[r]->(c:EpisodicNode {group_id: "default"})
    WHERE (e:EntityNode OR e:TopicNode)
      AND r.fact_id IS NOT NULL
    RETURN DISTINCT r.fact_id as fact_id
''')
subject_set = {r["fact_id"] for r in subject_fact_ids}

object_fact_ids = neo4j.query('''
    MATCH (c:EpisodicNode {group_id: "default"})-[r]->(e)
    WHERE (e:EntityNode OR e:TopicNode)
      AND r.fact_id IS NOT NULL
    RETURN DISTINCT r.fact_id as fact_id
''')
object_set = {r["fact_id"] for r in object_fact_ids}

print(f"Unique fact_ids in Subject->Chunk edges: {len(subject_set)}")
print(f"Unique fact_ids in Chunk->Object edges: {len(object_set)}")
print(f"fact_ids in BOTH (complete patterns): {len(subject_set & object_set)}")
print(f"fact_ids ONLY in Subject edges: {len(subject_set - object_set)}")
print(f"fact_ids ONLY in Object edges: {len(object_set - subject_set)}")

# 2. Check what happens to edges that have subject but no object
if subject_set - object_set:
    orphaned_subject = list(subject_set - object_set)[:5]
    print(f"\nSample facts with Subject edge but no Object edge:")
    for fid in orphaned_subject:
        # Get the subject edge details
        result = neo4j.query('''
            MATCH (subj)-[r]->(c:EpisodicNode {group_id: "default"})
            WHERE r.fact_id = $fid
              AND (subj:EntityNode OR subj:TopicNode)
            MATCH (f:FactNode {uuid: $fid})
            RETURN subj.name as subject, type(r) as rel_type, f.content as fact
        ''', {"fid": fid})
        if result:
            r = result[0]
            print(f"  Subject: {r['subject']}")
            print(f"  Rel: {r['rel_type']}")
            print(f"  Fact: {r['fact'][:60]}...")
            print()

# 3. Check edge creation pattern - are both edges being created with same fact_id?
print("\n3. Checking if edges come in pairs with same fact_id...")

pairs = neo4j.query('''
    MATCH (subj)-[r1]->(c:EpisodicNode {group_id: "default"})-[r2]->(obj)
    WHERE r1.fact_id IS NOT NULL
      AND (subj:EntityNode OR subj:TopicNode)
      AND (obj:EntityNode OR obj:TopicNode)
    RETURN r1.fact_id = r2.fact_id as same_fact_id, count(*) as cnt
''')
for p in pairs:
    print(f"  Same fact_id: {p['same_fact_id']} - Count: {p['cnt']}")

# 4. Check which entity types are involved
print("\n4. Entity types in edges...")
entity_types = neo4j.query('''
    MATCH (e)-[r]->(c:EpisodicNode {group_id: "default"})
    WHERE r.fact_id IS NOT NULL
    RETURN labels(e) as labels, count(*) as cnt
''')
print("Subject entity types:")
for et in entity_types:
    print(f"  {et['labels']}: {et['cnt']}")

object_types = neo4j.query('''
    MATCH (c:EpisodicNode {group_id: "default"})-[r]->(e)
    WHERE r.fact_id IS NOT NULL
    RETURN labels(e) as labels, count(*) as cnt
''')
print("Object entity types:")
for ot in object_types:
    print(f"  {ot['labels']}: {ot['cnt']}")

# 5. Check if there are duplicate subject edges for same fact
print("\n5. Checking for duplicate edges per fact...")
dups = neo4j.query('''
    MATCH (subj)-[r]->(c:EpisodicNode {group_id: "default"})
    WHERE r.fact_id IS NOT NULL
      AND (subj:EntityNode OR subj:TopicNode)
    WITH r.fact_id as fid, count(*) as edge_count
    WHERE edge_count > 1
    RETURN edge_count, count(fid) as num_facts
    ORDER BY edge_count DESC
''')
if dups:
    print("Facts with multiple subject edges:")
    for d in dups:
        print(f"  {d['edge_count']} edges: {d['num_facts']} facts")
else:
    print("No duplicate subject edges found")
