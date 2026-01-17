#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services

neo4j = get_services().neo4j

# The 3 facts with missing objects
facts_with_missing = [
    ("ba92aea4", "economic conditions"),
    ("db733c8c", "regional information gathering"),
    ("cdf48f28", "views of Federal Reserve officials"),
]

print("Checking if Object nodes exist for missing edges:\n")

for prefix, expected_obj in facts_with_missing:
    # Get full fact
    fact = neo4j.query('''
        MATCH (f:FactNode {group_id: "default"})
        WHERE f.uuid STARTS WITH $prefix
        RETURN f.uuid as uuid, f.content as content
    ''', {"prefix": prefix})[0]

    print(f"Fact: {fact['content'][:70]}...")
    print(f"  Expected object: '{expected_obj}'")

    # Search for this as entity or topic
    entity = neo4j.query('''
        MATCH (e:EntityNode {group_id: "default"})
        WHERE toLower(e.name) CONTAINS toLower($name)
        RETURN e.name as name, e.uuid as uuid
    ''', {"name": expected_obj})

    topic = neo4j.query('''
        MATCH (t:TopicNode {group_id: "default"})
        WHERE toLower(t.name) CONTAINS toLower($name)
        RETURN t.name as name, t.uuid as uuid
    ''', {"name": expected_obj})

    if entity:
        print(f"  Found EntityNode: {entity[0]['name']} (uuid: {entity[0]['uuid'][:8]}...)")
    if topic:
        print(f"  Found TopicNode: {topic[0]['name']} (uuid: {topic[0]['uuid'][:8]}...)")
    if not entity and not topic:
        print(f"  ✗ No matching node found!")

    # Check if there's a _TARGET edge at all for this fact
    target_edge = neo4j.query('''
        MATCH ()-[r]->(obj)
        WHERE r.fact_id = $fid
          AND type(r) ENDS WITH '_TARGET'
        RETURN obj.name as name, obj.uuid as uuid, type(r) as rel
    ''', {"fid": fact['uuid']})

    if target_edge:
        print(f"  Found _TARGET edge to: {target_edge[0]['name']}")
    else:
        print(f"  ✗ No _TARGET edge exists for this fact")

    print()