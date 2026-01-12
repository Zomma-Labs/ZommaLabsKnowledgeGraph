import sys
import os
import asyncio
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.temporal_extractor import TemporalExtractor
from src.schemas.document_types import Chunk
from src.workflows.main_pipeline import app
from src.tools.neo4j_client import Neo4jClient

# Mock data
def create_mock_chunks():
    chunks = []
    # Create 10 chunks
    for i in range(10):
        chunks.append(Chunk(
            chunk_id=f"chunk_{i}",
            doc_id="test_doc_1",
            heading="Test Heading",
            body=f"This is chunk {i}. The document was published on October 5, 2023.",
            metadata={"doc_item_refs": [], "page_numbers": [1]}
        ))
    return chunks

async def run_verification():
    print("--- 🕒 Verifying Temporal Extractor Flow ---")
    
    # 1. Setup
    extractor = TemporalExtractor()
    chunks = create_mock_chunks()
    title = "Test Financial Report Q3"
    
    # 2. Enrich Chunks
    print("running enrich_chunks...")
    enriched_chunks = extractor.enrich_chunks(chunks, title)
    
    doc_date = enriched_chunks[0].metadata.get("doc_date")
    print(f"Extracted Date: {doc_date}")
    
    if not doc_date:
        print("❌ Failed to extract date. Aborting.")
        return

    # 3. Run Pipeline with ONE enriched chunk
    target_chunk = enriched_chunks[0]
    
    # Construct input for pipeline
    inputs = {
        "chunk_text": target_chunk.body,
        "metadata": {
            "doc_id": target_chunk.doc_id,
            "filename": "test_doc_1.pdf",
            "chunk_id": target_chunk.chunk_id,
            "headings": [target_chunk.heading],
            "group_id": "test_temporal_tenant",
            "doc_date": target_chunk.metadata.get("doc_date") # CRITICAL: Passing the extracted date
        }
    }
    
    print("\n🚀 Running Pipeline on Enriched Chunk...")
    try:
        await app.ainvoke(inputs)
        print("Pipeline completed.")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return

    # 4. Verify in Neo4j
    print("\n🔍 Checking Neo4j...")
    client = Neo4jClient()
    query = """
    MATCH (d:DocumentNode {group_id: 'test_temporal_tenant'})
    RETURN d.name as Name, d.document_date as Date, d.created_at as CreatedAt
    """
    results = client.query(query)
    client.close()
    
    if not results:
        print("❌ DocumentNode not found!")
    else:
        record = results[0]
        # Neo4j datetime to string
        neo_date = record['Date']
        print(f"   Document Name: {record['Name']}")
        print(f"   Document Date (Neo4j): {neo_date}")
        
        # Verify match (approximate string check)
        if str(neo_date).startswith("2023-10-05") or str(neo_date).startswith("2023-10-05"):
             print("✅ SUCCESS: Date matches extracted date!")
        else:
             print(f"⚠️ MISMATCH: Expected 2023-10-05, got {neo_date}")

if __name__ == "__main__":
    asyncio.run(run_verification())
