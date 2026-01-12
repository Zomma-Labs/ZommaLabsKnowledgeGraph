#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services

neo4j = get_services().neo4j

# Find facts without complete patterns
incomplete = neo4j.query('''
    MATCH (c:EpisodicNode {group_id: "default"})-[:CONTAINS_FACT]->(f:FactNode)
    WHERE NOT EXISTS {
        MATCH (subj)-[r1 {fact_id: f.uuid}]->(c)-[r2 {fact_id: f.uuid}]->(obj)
        WHERE (subj:EntityNode OR subj:TopicNode)
          AND (obj:EntityNode OR obj:TopicNode)
    }
    RETURN f.uuid as fact_id, f.content as content
''')
print(f'Facts without complete patterns: {len(incomplete)}')
for r in incomplete:
    print(f'\n  [{r["fact_id"][:8]}...]')
    print(f'  Content: {r["content"]}')

# Check if these facts have ANY edges
print("\n\nChecking if these facts have any edges:")
for r in incomplete:
    fid = r["fact_id"]
    edges = neo4j.query('''
        MATCH ()-[r {fact_id: $fid}]->()
        RETURN type(r) as rel_type, startNode(r) as start, endNode(r) as end
    ''', {"fid": fid})
    print(f'\n  Fact [{fid[:8]}...]: {len(edges)} edges')
    for e in edges:
        print(f'    - {e["rel_type"]}')