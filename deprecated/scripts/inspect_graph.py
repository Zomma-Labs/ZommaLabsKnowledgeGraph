import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.tools.neo4j_client import Neo4jClient

def inspect():
    client = Neo4jClient()
    
    print("\n--- Node Labels ---")
    try:
        results = client.query("MATCH (n) RETURN DISTINCT labels(n) LIMIT 10")
        for r in results:
            print(r['labels(n)'])
    except Exception as e:
        print(f"Error getting labels: {e}")

    print("\n--- Entity URI Check ---")
    try:
        total = client.query("MATCH (n:Entity) RETURN count(n) as c")[0]['c']
        with_uri = client.query("MATCH (n:Entity) WHERE n.uri IS NOT NULL RETURN count(n) as c")[0]['c']
        print(f"Total Entities: {total}")
        print(f"Entities with URI: {with_uri}")
        
        if total > with_uri:
            print("Sample without URI:")
            results = client.query("MATCH (n:Entity) WHERE n.uri IS NULL RETURN n.name LIMIT 5")
            for r in results:
                print(r['n.name'])
    except Exception as e:
        print(f"Error checking URIs: {e}")

if __name__ == "__main__":
    inspect()
