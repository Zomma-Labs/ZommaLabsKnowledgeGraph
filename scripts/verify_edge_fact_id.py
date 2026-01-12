#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services

neo4j = get_services().neo4j

# Get the fact that has Subject edge but no Object edge
prefix = "ba92aea4"

fact = neo4j.query('''
    MATCH (f:FactNode {group_id: "default"})
    WHERE f.uuid STARTS WITH $prefix
    RETURN f.uuid as uuid, f.content as content
''', {"prefix": prefix})[0]

print(f"Fact UUID: {fact['uuid']}")
print(f"Content: {fact['content']}")

# Get the Subject edge with this fact_id
subj_edge = neo4j.query('''
    MATCH (subj)-[r {fact_id: $fid}]->(c:EpisodicNode)
    WHERE (subj:EntityNode OR subj:TopicNode)
    RETURN subj.name as name, type(r) as rel, r.fact_id as edge_fact_id
''', {"fid": fact['uuid']})

print(f"\nSubject edge:")
if subj_edge:
    e = subj_edge[0]
    print(f"  Subject: {e['name']}")
    print(f"  Rel type: {e['rel']}")
    print(f"  Edge fact_id: {e['edge_fact_id']}")
    print(f"  Matches fact UUID: {e['edge_fact_id'] == fact['uuid']}")
else:
    print("  No Subject edge found!")

# Verify the CONTAINS_FACT relationship
contains = neo4j.query('''
    MATCH (c:EpisodicNode)-[:CONTAINS_FACT]->(f:FactNode {uuid: $fid})
    RETURN c.uuid as chunk_uuid, c.content as chunk_content
''', {"fid": fact['uuid']})

print(f"\nCONTAINS_FACT relationship:")
if contains:
    print(f"  Chunk UUID: {contains[0]['chunk_uuid'][:8]}...")
else:
    print("  No CONTAINS_FACT found!")