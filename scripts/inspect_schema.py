import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.neo4j_client import Neo4jClient

def inspect_schema():
    client = Neo4jClient()
    
    print("--- Node Labels ---")
    try:
        labels = client.query("CALL db.labels()")
        print([r['label'] for r in labels])
    except Exception as e:
        print(f"Error fetching labels: {e}")

    print("\n--- Relationship Types ---")
    try:
        rels = client.query("CALL db.relationshipTypes()")
        print([r['relationshipType'] for r in rels])
    except Exception as e:
        print(f"Error fetching relationship types: {e}")

    print("\n--- Property Keys ---")
    try:
        props = client.query("CALL db.propertyKeys()")
        print([r['propertyKey'] for r in props])
    except Exception as e:
        print(f"Error fetching property keys: {e}")
        
    print("\n--- Schema Visualization (Sample) ---")
    try:
        # Get a sample of connections to understand patterns
        sample = client.query("""
            MATCH (n)-[r]->(m) 
            RETURN labels(n) as start_node, type(r) as rel_type, labels(m) as end_node 
            LIMIT 20
        """)
        for row in sample:
            print(f"{row['start_node']} -[{row['rel_type']}]-> {row['end_node']}")
    except Exception as e:
        print(f"Error fetching sample: {e}")

if __name__ == "__main__":
    inspect_schema()
