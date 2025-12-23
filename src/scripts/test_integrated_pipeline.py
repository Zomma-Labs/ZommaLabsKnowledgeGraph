import sys
import os
import asyncio
from typing import List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.workflows.main_pipeline import ingest_document
from src.schemas.document_types import Chunk

def create_mock_chunks() -> List[Chunk]:
    chunks = []
    # Create 3 chunks to be fast
    for i in range(3):
        chunks.append(Chunk(
            chunk_id=f"integrated_test_chunk_{i}",
            doc_id="integrated_doc_1",
            heading="Integrated Test Doc",
            body=f"This is chunk {i}. The document was published on November 10, 2024.",
            metadata={"doc_item_refs": [], "page_numbers": [1]}
        ))
    return chunks

async def run_integration_test():
    print("--- 🔄 Testing Integrated Pipeline (ingest_document) ---")
    
    chunks = create_mock_chunks()
    group_id = "integrated_test_tenant"
    
    try:
        await ingest_document(chunks, group_id)
        print("✅ Ingest function executed without raising exceptions.")
    except Exception as e:
        print(f"❌ Ingest function failed: {e}")
        return

    # Check Neo4j
    from src.tools.neo4j_client import Neo4jClient
    client = Neo4jClient()
    
    # Check Document Date
    print("\n🔍 Verifying Neo4j Data...")
    res = client.query("MATCH (d:DocumentNode {group_id: $gid}) RETURN d.document_date as Date", {"gid": group_id})
    if res:
        print(f"   Found Document Date: {res[0]['Date']}")
    else:
        print("   ❌ Document Node not found.")
        
    # Check if facts were created (implies pipeline ran)
    res_facts = client.query("MATCH (d:DocumentNode {group_id: $gid})-[:HAS_CHUNK]->()-[:MENTIONED_IN|ABOUT|PERFORMED]-() RETURN count(*) as Count", {"gid": group_id})
    print(f"   Graph Interactions Count: {res_facts[0]['Count'] if res_facts else 0}")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(run_integration_test())
