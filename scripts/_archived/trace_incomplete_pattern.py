#!/usr/bin/env python3
"""Trace exactly why some facts have incomplete Subject->Chunk->Object patterns."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services

neo4j = get_services().neo4j

# Get facts where Subject edge exists but Object edge doesn't
incomplete = neo4j.query('''
    MATCH (f:FactNode {group_id: "default"})
    OPTIONAL MATCH (subj)-[subj_rel {fact_id: f.uuid}]->(c:EpisodicNode)
    WHERE subj:EntityNode OR subj:TopicNode
    OPTIONAL MATCH (c)-[obj_rel {fact_id: f.uuid}]->(obj)
    WHERE obj:EntityNode OR obj:TopicNode
    WITH f, subj, subj_rel, obj, obj_rel, c
    WHERE subj IS NOT NULL AND obj IS NULL
    RETURN f.uuid as fact_uuid, f.content as content,
           subj.name as subject_name, type(subj_rel) as subject_rel,
           c.uuid as chunk_uuid
''')

print(f"Facts with Subject edge but NO Object edge: {len(incomplete)}\n")

for row in incomplete:
    print(f"Fact: {row['content'][:80]}...")
    print(f"  UUID: {row['fact_uuid'][:8]}...")
    print(f"  Subject: {row['subject_name']} via {row['subject_rel']}")
    print(f"  Chunk: {row['chunk_uuid'][:8]}...")

    # Check ALL edges with this fact_id
    all_edges = neo4j.query('''
        MATCH (a)-[r {fact_id: $fid}]->(b)
        RETURN labels(a) as from_labels, a.name as from_name,
               type(r) as rel_type,
               labels(b) as to_labels, b.name as to_name, b.uuid as to_uuid
    ''', {"fid": row['fact_uuid']})

    print(f"  ALL edges with fact_id: {len(all_edges)}")
    for e in all_edges:
        print(f"    {e['from_labels']} {e['from_name']} -> [{e['rel_type']}] -> {e['to_labels']} {e['to_name'][:30] if e['to_name'] else e['to_uuid'][:8]}")

    print()
