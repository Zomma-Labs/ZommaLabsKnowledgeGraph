#!/usr/bin/env python3
"""Verify all facts have complete Subject->Chunk->Object patterns."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services

neo4j = get_services().neo4j
group_id = sys.argv[1] if len(sys.argv) > 1 else "test_fix"

# Get total facts
total_facts = neo4j.query('''
    MATCH (f:FactNode {group_id: $gid})
    RETURN count(*) as cnt
''', {"gid": group_id})[0]['cnt']

# Get facts with complete patterns (both Subject and Object edges)
complete_facts = neo4j.query('''
    MATCH (f:FactNode {group_id: $gid})
    MATCH (subj)-[subj_rel {fact_id: f.uuid}]->(c:EpisodicNode)
    WHERE subj:EntityNode OR subj:TopicNode
    MATCH (c)-[obj_rel {fact_id: f.uuid}]->(obj)
    WHERE obj:EntityNode OR obj:TopicNode
    RETURN count(DISTINCT f) as cnt
''', {"gid": group_id})[0]['cnt']

# Get facts with ONLY Subject edge (incomplete)
subject_only = neo4j.query('''
    MATCH (f:FactNode {group_id: $gid})
    OPTIONAL MATCH (subj)-[subj_rel {fact_id: f.uuid}]->(c:EpisodicNode)
    WHERE subj:EntityNode OR subj:TopicNode
    OPTIONAL MATCH (c)-[obj_rel {fact_id: f.uuid}]->(obj)
    WHERE obj:EntityNode OR obj:TopicNode
    WITH f, subj, obj
    WHERE subj IS NOT NULL AND obj IS NULL
    RETURN count(DISTINCT f) as cnt
''', {"gid": group_id})[0]['cnt']

# Get facts with NO edges at all
no_edges = neo4j.query('''
    MATCH (f:FactNode {group_id: $gid})
    WHERE NOT EXISTS {
        MATCH ()-[r {fact_id: f.uuid}]->()
        WHERE type(r) <> 'CONTAINS_FACT'
    }
    RETURN count(*) as cnt
''', {"gid": group_id})[0]['cnt']

print(f"Group ID: {group_id}")
print(f"Total facts: {total_facts}")
print(f"Complete patterns (Subject + Object): {complete_facts}")
print(f"Incomplete (Subject only): {subject_only}")
print(f"No edges at all: {no_edges}")
print()

if subject_only == 0 and complete_facts == total_facts:
    print("✓ All facts have complete Subject->Chunk->Object patterns!")
else:
    coverage = complete_facts / total_facts * 100 if total_facts > 0 else 0
    print(f"Coverage: {coverage:.1f}%")
    if subject_only > 0:
        print(f"✗ {subject_only} facts have incomplete patterns")
