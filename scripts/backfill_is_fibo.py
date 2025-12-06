"""
Script to Backfill is_fibo Property.
Iterates through all Entity/Topic nodes.
- If uri starts with 'http', set is_fibo = true
- Else, set is_fibo = false
"""
import sys
import os
from src.tools.neo4j_client import Neo4jClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def backfill_is_fibo():
    print("🚀 Backfilling is_fibo property...")
    neo4j = Neo4jClient()
    
    # 1. Set True for FIBO URIs
    cypher_true = """
    MATCH (n)
    WHERE (n:EntityNode OR n:TopicNode) AND n.uri STARTS WITH "http"
    SET n.is_fibo = true
    """
    neo4j.query(cypher_true)
    print("✅ Marked FIBO nodes (uri starts with http) as is_fibo=true.")
    
    # 2. Set False for UUIDs (Local)
    cypher_false = """
    MATCH (n)
    WHERE (n:EntityNode OR n:TopicNode) AND n.is_fibo IS NULL
    SET n.is_fibo = false
    """
    neo4j.query(cypher_false)
    print("✅ Marked remaining nodes as is_fibo=false.")
    
    neo4j.close()

if __name__ == "__main__":
    backfill_is_fibo()
