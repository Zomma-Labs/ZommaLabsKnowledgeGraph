#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services

neo4j = get_services().neo4j

# Get a fact that has Subject but no Object
fid = "ba92aea4"  # From earlier trace - has Subject but no Object

# Find the full fact_id
full_fid = neo4j.query('''
    MATCH (f:FactNode {group_id: "default"})
    WHERE f.uuid STARTS WITH $prefix
    RETURN f.uuid as uuid, f.content as content
''', {"prefix": fid})[0]

print(f"Fact: {full_fid['content']}")
print(f"UUID: {full_fid['uuid']}")

# Get ALL edges with this fact_id
edges = neo4j.query('''
    MATCH (a)-[r {fact_id: $fid}]->(b)
    RETURN labels(a) as from_labels, a.name as from_name,
           type(r) as rel_type,
           labels(b) as to_labels, b.name as to_name
''', {"fid": full_fid['uuid']})

print(f"\nAll edges with this fact_id: {len(edges)}")
for e in edges:
    print(f"  ({e['from_labels']}) {e['from_name']} -[{e['rel_type']}]-> ({e['to_labels']}) {e['to_name']}")

# Check if there's a _TARGET edge that we might have missed
target_edges = neo4j.query('''
    MATCH (c:EpisodicNode)-[r]->(obj)
    WHERE r.fact_id = $fid
    RETURN type(r) as rel_type, obj.name as obj_name, labels(obj) as obj_labels
''', {"fid": full_fid['uuid']})

print(f"\nChunk->Object edges: {len(target_edges)}")
for e in target_edges:
    print(f"  -[{e['rel_type']}]-> {e['obj_name']} ({e['obj_labels']})")