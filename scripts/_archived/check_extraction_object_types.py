#!/usr/bin/env python3
"""Check what object_type the extractor assigns to certain object names."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services
import json

neo4j = get_services().neo4j

# Get facts that have Subject edge but no Object edge
# These are the problematic ones
incomplete_facts = [
    "The Beige Book is intended to characterize the change in economic conditions since the last report",
    "The Beige Book can complement other forms of regional information gathering",
    "The Beige Book is not a commentary on the views of Federal Reserve officials",
    "Contacts are used by each Federal Reserve Bank specifically to provide accurate and objective information about economic conditions"
]

# Let's look at what subject/object these facts should have
# The extraction should have extracted these with object_type

# Let's check the facts in the DB
print("Checking FactNodes for expected subject/object patterns:\n")

for fact_content in incomplete_facts:
    # Find the fact
    facts = neo4j.query('''
        MATCH (f:FactNode {group_id: "default"})
        WHERE f.content STARTS WITH $prefix
        RETURN f.uuid as uuid, f.content as content, f.subject as subject, f.object as object, f.edge_type as edge_type
    ''', {"prefix": fact_content[:60]})

    if facts:
        f = facts[0]
        print(f"Fact: {f['content'][:70]}...")
        print(f"  Subject: {f['subject']}")
        print(f"  Object: {f['object']}")
        print(f"  Edge type: {f['edge_type']}")

        # Check if object exists as TopicNode
        topic = neo4j.query('''
            MATCH (t:TopicNode {group_id: "default"})
            WHERE toLower(t.name) = toLower($name)
            RETURN t.name as name, t.uuid as uuid
        ''', {"name": f['object']})

        if topic:
            print(f"  TopicNode exists: {topic[0]['name']} ({topic[0]['uuid'][:8]}...)")
        else:
            print(f"  TopicNode MISSING for: {f['object']}")

        # Check if object exists as EntityNode
        entity = neo4j.query('''
            MATCH (e:EntityNode {group_id: "default"})
            WHERE toLower(e.name) = toLower($name)
            RETURN e.name as name, e.uuid as uuid
        ''', {"name": f['object']})

        if entity:
            print(f"  EntityNode exists: {entity[0]['name']} ({entity[0]['uuid'][:8]}...)")
        else:
            print(f"  EntityNode MISSING for: {f['object']}")
    else:
        print(f"Fact not found: {fact_content[:50]}...")

    print()
