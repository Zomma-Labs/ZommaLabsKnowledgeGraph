"""
Script to Backfill Missing Embeddings for EntityNodes and TopicNodes.
Iterates through all nodes, checks if `embedding` is null, generating and setting it if so.
Usage: uv run scripts/backfill_embeddings.py
"""
import sys
import os
import time
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.util.llm_client import get_embeddings
from src.tools.neo4j_client import Neo4jClient

def backfill():
    print("🚀 Starting Embedding Backfill...")
    neo4j = Neo4jClient()
    embeddings = get_embeddings()
    
    # 1. Find nodes with missing embeddings
    # We look for Entities or Topics where embedding is null
    # We need name and description to generate it.
    cypher_find = """
    MATCH (n)
    WHERE (n:EntityNode OR n:TopicNode) AND n.embedding IS NULL
    RETURN n.uuid as uuid, n.name as name, n.description as description, labels(n) as labels
    """
    
    print("🔎 Scanning graph for missing embeddings...")
    nodes = neo4j.query(cypher_find)
    print(f"   Found {len(nodes)} nodes needing backfill.")
    
    if not nodes:
        print("✅ Graph is fully embedded!")
        return

    # 2. Process in batches
    batch_size = 50
    updated_count = 0
    
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i+batch_size]
        
        # Prepare text to embed
        texts = []
        for node in batch:
            name = node.get('name', '')
            desc = node.get('description', '') or "Financial Concept"
            texts.append(f"{name}: {desc}")
            
        # Generate Embeddings
        try:
            vectors = embeddings.embed_documents(texts)
            
            # Update Graph
            for j, node in enumerate(batch):
                uuid_val = node['uuid']
                vector = vectors[j]
                
                # Update query
                # Use uuid to match
                cypher_update = """
                MATCH (n)
                WHERE n.uuid = $uuid
                SET n.embedding = $vector
                """
                neo4j.query(cypher_update, {"uuid": uuid_val, "vector": vector})
                
            updated_count += len(batch)
            print(f"   Saved batch {i//batch_size + 1} ({updated_count}/{len(nodes)})")
            
        except Exception as e:
            print(f"   ❌ Batch failed: {e}")
            
    print("✅ Backfill Complete.")
    neo4j.close()

if __name__ == "__main__":
    backfill()
