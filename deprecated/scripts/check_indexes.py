import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.neo4j_client import Neo4jClient

def check_indexes():
    client = Neo4jClient()
    print("--- Indexes ---")
    try:
        indexes = client.query("SHOW INDEXES")
        for idx in indexes:
            print(f"Name: {idx['name']}, Type: {idx['type']}, Entity: {idx['entityType']}, Properties: {idx['properties']}")
    except Exception as e:
        print(f"Error fetching indexes: {e}")

if __name__ == "__main__":
    check_indexes()
