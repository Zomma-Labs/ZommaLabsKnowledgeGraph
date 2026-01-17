#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services

neo4j = get_services().neo4j

# Get all edges from "The Beige Book" entity
edges = neo4j.query('''
    MATCH (e:EntityNode {name: "The Beige Book", group_id: "default"})-[r]->(c:EpisodicNode)
    WHERE r.fact_id IS NOT NULL
    RETURN r.fact_id as fact_id, type(r) as rel_type
''')

print(f"Edges from 'The Beige Book' to chunks: {len(edges)}")
for e in edges:
    fid = e['fact_id']
    rel = e['rel_type']

    # Get the fact content
    fact = neo4j.query('''
        MATCH (f:FactNode {uuid: $fid})
        RETURN f.content as content
    ''', {"fid": fid})
    content = fact[0]['content'] if fact else "NOT FOUND"

    # Check if there's a matching _TARGET edge
    target = neo4j.query('''
        MATCH (c:EpisodicNode)-[r {fact_id: $fid}]->(obj)
        WHERE (obj:EntityNode OR obj:TopicNode)
        RETURN obj.name as name, type(r) as rel_type
    ''', {"fid": fid})

    if target:
        print(f"\n✓ [{fid[:8]}...] {rel}")
        print(f"  Fact: {content[:60]}...")
        print(f"  Object: {target[0]['name']} via {target[0]['rel_type']}")
    else:
        print(f"\n✗ [{fid[:8]}...] {rel} - NO TARGET EDGE!")
        print(f"  Fact: {content[:80]}...")