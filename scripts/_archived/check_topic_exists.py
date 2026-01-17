#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util.services import get_services
neo4j = get_services().neo4j

# Check if 'current economic conditions' exists with any case
results = neo4j.query('''
    MATCH (t:TopicNode {group_id: "default"})
    WHERE toLower(t.name) CONTAINS "current economic"
       OR toLower(t.name) CONTAINS "economic conditions"
    RETURN t.name as name
''')
print('Topics matching "current economic" or "economic conditions":')
for r in results:
    print(f'  - {repr(r["name"])}')

# Check what objects are expected in the orphaned facts
print("\nChecking Qdrant for fact metadata with subject/object...")
# Can't access Qdrant due to lock, but let's check Neo4j fact content

results = neo4j.query('''
    MATCH (f:FactNode {group_id: "default"})
    WHERE f.content CONTAINS "about current economic conditions"
       OR f.content CONTAINS "is about"
    RETURN f.content as content, f.uuid as uuid
    LIMIT 5
''')
print("\nFacts with 'about':")
for r in results:
    print(f'  - {r["content"][:100]}...')
