#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services

neo4j = get_services().neo4j

# For facts with only 1 edge, let's see what's there
incomplete_with_edges = neo4j.query('''
    MATCH (c:EpisodicNode {group_id: "default"})-[:CONTAINS_FACT]->(f:FactNode)
    WHERE NOT EXISTS {
        MATCH (subj)-[r1 {fact_id: f.uuid}]->(c)-[r2 {fact_id: f.uuid}]->(obj)
        WHERE (subj:EntityNode OR subj:TopicNode)
          AND (obj:EntityNode OR obj:TopicNode)
    }
    RETURN f.uuid as fact_id, f.content as content
''')

print("Analyzing incomplete facts:\n")

for r in incomplete_with_edges:
    fid = r["fact_id"]
    content = r["content"]

    print(f"Fact: {content[:80]}...")

    # Check for Subject->Chunk edge
    subj_edge = neo4j.query('''
        MATCH (subj)-[r {fact_id: $fid}]->(c:EpisodicNode)
        WHERE (subj:EntityNode OR subj:TopicNode)
        RETURN subj.name as name, labels(subj) as labels, type(r) as rel_type
    ''', {"fid": fid})

    # Check for Chunk->Object edge
    obj_edge = neo4j.query('''
        MATCH (c:EpisodicNode)-[r {fact_id: $fid}]->(obj)
        WHERE (obj:EntityNode OR obj:TopicNode)
        RETURN obj.name as name, labels(obj) as labels, type(r) as rel_type
    ''', {"fid": fid})

    if subj_edge:
        s = subj_edge[0]
        print(f"  ✓ Subject: {s['name']} ({s['labels']}) -[{s['rel_type']}]->")
    else:
        print(f"  ✗ Subject: MISSING")

    if obj_edge:
        o = obj_edge[0]
        print(f"  ✓ Object: -[{o['rel_type']}]-> {o['name']} ({o['labels']})")
    else:
        print(f"  ✗ Object: MISSING")

    print()