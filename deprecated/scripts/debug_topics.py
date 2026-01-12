import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.tools.neo4j_client import Neo4jClient

def debug_topics():
    client = Neo4jClient()
    print("--- 🔍 Searching for 'Labor' Topics ---")
    
    res = client.query("MATCH (t:TopicNode) WHERE toLower(t.name) CONTAINS 'labor' RETURN t.name, t.uuid, t.group_id")
    if res:
        for r in res:
            print(f"   - {r['t.name']} ({r['t.uuid']}) [Group: {r['t.group_id']}]")
    else:
        print("   ❌ No topics found containing 'labor'.")

    print("\n--- 🔍 Searching for 'Employment' Topics ---")
    res = client.query("MATCH (t:TopicNode) WHERE toLower(t.name) CONTAINS 'employment' RETURN t.name, t.uuid")
    if res:
        for r in res:
            print(f"   - {r['t.name']} ({r['t.uuid']})")
    else:
        print("   ❌ No topics found containing 'employment'.")

if __name__ == "__main__":
    debug_topics()
