#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services

neo4j = get_services().neo4j

# Facts that have Subject but missing Object edge
missing_objects = ["economic conditions", "regional information gathering", "views of Federal Reserve officials"]

print("Checking if TopicNodes exist for missing objects:\n")

for obj_name in missing_objects:
    print(f"Object: '{obj_name}'")

    # Check exact match
    exact = neo4j.query('''
        MATCH (t:TopicNode {group_id: "default"})
        WHERE t.name = $name
        RETURN t.name as name, t.uuid as uuid
    ''', {"name": obj_name})

    # Check case-insensitive
    case_insensitive = neo4j.query('''
        MATCH (t:TopicNode {group_id: "default"})
        WHERE toLower(t.name) = toLower($name)
        RETURN t.name as name, t.uuid as uuid
    ''', {"name": obj_name})

    # Check partial match
    partial = neo4j.query('''
        MATCH (t:TopicNode {group_id: "default"})
        WHERE toLower(t.name) CONTAINS toLower($name) OR toLower($name) CONTAINS toLower(t.name)
        RETURN t.name as name, t.uuid as uuid
    ''', {"name": obj_name})

    if exact:
        print(f"  ✓ Exact match: {exact[0]['name']} ({exact[0]['uuid'][:8]}...)")
    elif case_insensitive:
        print(f"  ~ Case-insensitive match: {case_insensitive[0]['name']} ({case_insensitive[0]['uuid'][:8]}...)")
    elif partial:
        print(f"  ? Partial match: {partial[0]['name']} ({partial[0]['uuid'][:8]}...)")
    else:
        print(f"  ✗ NO TopicNode exists for this object!")

    print()

# Also check if these are EntityNodes instead
print("\n" + "="*60)
print("Checking if these exist as EntityNodes:\n")

for obj_name in missing_objects:
    print(f"Object: '{obj_name}'")

    # Check case-insensitive
    entity = neo4j.query('''
        MATCH (e:EntityNode {group_id: "default"})
        WHERE toLower(e.name) CONTAINS toLower($name)
        RETURN e.name as name, e.uuid as uuid, e.entity_type as type
    ''', {"name": obj_name})

    if entity:
        for e in entity:
            print(f"  Found EntityNode: {e['name']} (type: {e['type']}, uuid: {e['uuid'][:8]}...)")
    else:
        print(f"  No EntityNode found")

    print()
