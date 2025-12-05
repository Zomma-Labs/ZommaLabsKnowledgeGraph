import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.tools.neo4j_client import Neo4jClient

def debug_specific_nodes():
    client = Neo4jClient()
    print("--- 🔍 Inspecting Specific Nodes ---")
    
    # Check Boston
    print("\n1. Entity: Boston")
    res = client.query("""
    MATCH (e:EntityNode) 
    WHERE toLower(e.name) CONTAINS 'boston' 
    RETURN e.name, e.uuid, e.group_id, size(e.embedding) as embed_len
    """)
    for r in res:
        print(f"   - Name: {r['e.name']}")
        print(f"     UUID: {r['e.uuid']}")
        print(f"     Group: {r['e.group_id']}")
        print(f"     Embedding Length: {r['embed_len']}")

    # Check Labor Markets
    print("\n2. Topic: Labor Markets")
    res = client.query("""
    MATCH (t:TopicNode) 
    WHERE toLower(t.name) CONTAINS 'labor' 
    RETURN t.name, t.uuid, t.group_id, size(t.embedding) as embed_len
    """)
    for r in res:
        print(f"   - Name: {r['t.name']}")
        print(f"     UUID: {r['t.uuid']}")
        print(f"     Group: {r['t.group_id']}")
        print(f"     Embedding Length: {r['embed_len']}")

if __name__ == "__main__":
    debug_specific_nodes()
