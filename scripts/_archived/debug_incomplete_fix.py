#!/usr/bin/env python3
"""Debug why there are still incomplete patterns after fix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services

neo4j = get_services().neo4j
group_id = sys.argv[1] if len(sys.argv) > 1 else "test_fix"

print("=" * 60)
print("Facts with ONLY Subject edge (no Object edge)")
print("=" * 60)

# Get facts with ONLY Subject edge
incomplete = neo4j.query('''
    MATCH (f:FactNode {group_id: $gid})
    OPTIONAL MATCH (subj)-[subj_rel {fact_id: f.uuid}]->(c:EpisodicNode)
    WHERE subj:EntityNode OR subj:TopicNode
    OPTIONAL MATCH (c)-[obj_rel {fact_id: f.uuid}]->(obj)
    WHERE obj:EntityNode OR obj:TopicNode
    WITH f, subj, subj_rel, c, obj
    WHERE subj IS NOT NULL AND obj IS NULL
    RETURN f.uuid as fact_uuid, f.content as content, f.subject as subject, f.object as object,
           subj.name as actual_subject, type(subj_rel) as rel_type, c.uuid as chunk_uuid
''', {"gid": group_id})

for row in incomplete:
    print(f"\nFact: {row['content'][:80]}...")
    print(f"  UUID: {row['fact_uuid'][:8]}...")
    print(f"  Expected subject: {row['subject']}")
    print(f"  Expected object: {row['object']}")
    print(f"  Actual subject: {row['actual_subject']}")
    print(f"  Rel type: {row['rel_type']}")

    # Check what target node should exist
    obj_name = row['object']
    # Check if it's an entity
    entity = neo4j.query('''
        MATCH (e:EntityNode {group_id: $gid})
        WHERE toLower(e.name) = toLower($name)
        RETURN e.name as name, e.uuid as uuid
    ''', {"gid": group_id, "name": obj_name})

    # Check if it's a topic
    topic = neo4j.query('''
        MATCH (t:TopicNode {group_id: $gid})
        WHERE toLower(t.name) = toLower($name)
        RETURN t.name as name, t.uuid as uuid
    ''', {"gid": group_id, "name": obj_name})

    if entity:
        print(f"  Object as EntityNode EXISTS: {entity[0]['name']} ({entity[0]['uuid'][:8]}...)")
    elif topic:
        print(f"  Object as TopicNode EXISTS: {topic[0]['name']} ({topic[0]['uuid'][:8]}...)")
    else:
        print(f"  Object node MISSING for: {obj_name}")

print("\n" + "=" * 60)
print("Facts with NO edges at all")
print("=" * 60)

no_edges = neo4j.query('''
    MATCH (f:FactNode {group_id: $gid})
    WHERE NOT EXISTS {
        MATCH ()-[r {fact_id: f.uuid}]->()
        WHERE type(r) <> 'CONTAINS_FACT'
    }
    RETURN f.uuid as fact_uuid, f.content as content, f.subject as subject, f.object as object
''', {"gid": group_id})

for row in no_edges:
    print(f"\nFact: {row['content'][:80]}...")
    print(f"  UUID: {row['fact_uuid'][:8]}...")
    print(f"  Subject: {row['subject']}")
    print(f"  Object: {row['object']}")
