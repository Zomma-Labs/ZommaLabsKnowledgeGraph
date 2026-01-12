import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.util.services import get_services

def clear_fibo():
    print("🧹 Cleaning FIBO Qdrant Database...")
    
    services = get_services()
    client = services.qdrant_fibo
    
    collection_name = "fibo_entities"
    
    if client.collection_exists(collection_name):
        print(f"   Found collection '{collection_name}'. Deleting...")
        client.delete_collection(collection_name)
        print("   ✅ Collection deleted.")
    else:
        print(f"   Collection '{collection_name}' does not exist.")
        
    services.close()

if __name__ == "__main__":
    clear_fibo()
