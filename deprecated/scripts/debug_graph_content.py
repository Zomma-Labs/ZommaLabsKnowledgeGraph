import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.tools.neo4j_client import Neo4jClient

def debug_graph():
    client = Neo4jClient()
    
    print("--- 🔍 Graph Debug Info ---")
    
    # 1. Check Group IDs
    print("\n1. Group IDs present:")
    res = client.query("MATCH (n) RETURN distinct n.group_id as group_id, labels(n) as labels, count(n) as count")
    for r in res:
        print(f"   - {r['group_id']} ({r['labels']}): {r['count']}")
        
    # 2. Check Topics
    print("\n2. TopicNodes:")
    res = client.query("MATCH (t:TopicNode) RETURN t.name, t.uuid LIMIT 20")
    for r in res:
        print(f"   - {r['t.name']} ({r['t.uuid']})")
        
    # 3. Check Entities
    print("\n3. EntityNodes (sample):")
    res = client.query("MATCH (e:EntityNode) RETURN e.name, e.uuid LIMIT 20")
    for r in res:
        print(f"   - {r['e.name']} ({r['e.uuid']})")
        
    # 4. Check Hub Connections
    print("\n4. SectionNode Connections:")
    res = client.query("""
        MATCH (s:SectionNode)
        OPTIONAL MATCH (s)-[:REPRESENTS]->(e:EntityNode)
        OPTIONAL MATCH (s)-[:DISCUSSES]->(t:TopicNode)
        RETURN s.header_path, collect(distinct e.name) as entities, collect(distinct t.name) as topics LIMIT 10
    """)
    for r in res:
        print(f"   - Hub: {r['s.header_path']}")
        print(f"     Entities: {r['entities']}")
        print(f"     Topics: {r['topics']}")

if __name__ == "__main__":
    debug_graph()
