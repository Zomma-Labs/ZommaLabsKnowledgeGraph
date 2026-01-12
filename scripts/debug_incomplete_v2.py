#!/usr/bin/env python3
"""Debug incomplete patterns - check what edges exist."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services

neo4j = get_services().neo4j
group_id = sys.argv[1] if len(sys.argv) > 1 else "test_fix"

# Get all facts
facts = neo4j.query('''
    MATCH (f:FactNode {group_id: $gid})
    RETURN f.uuid as uuid, f.content as content
''', {"gid": group_id})

print(f"Total facts: {len(facts)}\n")

for fact in facts:
    fid = fact['uuid']
    content = fact['content'][:70]

    # Get ALL edges with this fact_id
    edges = neo4j.query('''
        MATCH (a)-[r {fact_id: $fid}]->(b)
        RETURN labels(a)[0] as from_type, a.name as from_name,
               type(r) as rel_type,
               labels(b)[0] as to_type, b.name as to_name
    ''', {"fid": fid})

    # Categorize edges
    subj_edges = [e for e in edges if e['to_type'] == 'EpisodicNode']
    obj_edges = [e for e in edges if e['from_type'] == 'EpisodicNode' and e['to_type'] in ('EntityNode', 'TopicNode')]

    if len(subj_edges) > 0 and len(obj_edges) > 0:
        status = "✓ Complete"
    elif len(subj_edges) > 0 and len(obj_edges) == 0:
        status = "✗ Subject only"
    elif len(subj_edges) == 0 and len(obj_edges) > 0:
        status = "✗ Object only"
    else:
        status = "✗ No edges"

    print(f"{status}: {content}...")
    if status != "✓ Complete":
        print(f"  UUID: {fid[:8]}...")
        print(f"  Edges: {len(edges)}")
        for e in edges:
            print(f"    {e['from_type']}:{e['from_name'][:20] if e['from_name'] else '?'} -[{e['rel_type']}]-> {e['to_type']}:{e['to_name'][:20] if e['to_name'] else '?'}")
