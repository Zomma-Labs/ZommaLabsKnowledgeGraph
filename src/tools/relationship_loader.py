import json
import os
from typing import List
from src.schemas.relationship import RelationshipDefinition
from src.util.vector_store import VectorStore

# Path to relationships data
DATA_PATH = "src/data/relationships.json"

def load_relationships():
    print("📚 Reading Relationship Definitions...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
        
    definitions = []
    for item in data:
        try:
            # Validate and create object
            rel_def = RelationshipDefinition(**item)
            definitions.append(rel_def)
        except Exception as e:
            print(f"   ⚠️ Skipping invalid item: {item.get('name', 'Unknown')} - {e}")
            
    print(f"   ✅ Loaded {len(definitions)} relationship definitions.")
    
    # Initialize Vector Store
    # Uses default path "./qdrant_relationships" as updated in vector_store.py
    vs = VectorStore()
    
    print("💾 Indexing to Qdrant...")
    vs.add_relationships(definitions)
    print("   ✅ Indexing Complete.")

if __name__ == "__main__":
    load_relationships()
