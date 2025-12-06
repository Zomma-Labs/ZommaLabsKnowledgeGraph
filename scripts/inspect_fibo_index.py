"""
Script to inspect the FIBO Qdrant index.
Queries for specific terms and prints the full payload to verify 'examples' are present.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import QdrantClient
from src.util.llm_client import get_embeddings

QDRANT_PATH = "./qdrant_fibo"
COLLECTION_NAME = "fibo_entities"

def inspect_term(term: str):
    print(f"\n🔍 Inspecting Term: '{term}'")
    client = QdrantClient(path=QDRANT_PATH)
    embeddings = get_embeddings()
    
    vector = embeddings.embed_query(term)
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=3,
        with_payload=True
    ).points
    
    for i, res in enumerate(results):
        payload = res.payload
        print(f"   {i+1}. {payload['label']} (Score: {res.score:.4f})")
        print(f"      Definition: {payload['definition'][:100]}...")
        print(f"      Examples:   {payload.get('examples', 'N/A')}")
        print("-" * 40)

if __name__ == "__main__":
    if not os.path.exists(QDRANT_PATH):
        print("❌ Qdrant index not found!")
        sys.exit(1)
        
    inspect_term("Inflation")
    inspect_term("Price")
    inspect_term("Contract")
