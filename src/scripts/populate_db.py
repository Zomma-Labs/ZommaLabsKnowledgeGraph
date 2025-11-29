import json
import os
import sys
from typing import List

# Add the project root to the python path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.schemas.relationship import RelationshipDefinition
from src.util.vector_store import VectorStore

def load_relationships(filepath: str) -> List[RelationshipDefinition]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    relationships = []
    for item in data:
        relationships.append(RelationshipDefinition(**item))
    return relationships

def main():
    # Path to relationships.json
    json_path = os.path.join(os.path.dirname(__file__), "../data/relationships.json")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    print(f"Loading relationships from {json_path}...")
    relationships = load_relationships(json_path)
    
    print("Initializing VectorStore...")
    # Ensure we are using the correct Qdrant path relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    qdrant_path = os.path.join(project_root, "qdrant_data")
    
    vs = VectorStore(path=qdrant_path)
    
    print("Adding relationships to VectorStore...")
    vs.add_relationships(relationships)
    
    print("Done!")

if __name__ == "__main__":
    main()
