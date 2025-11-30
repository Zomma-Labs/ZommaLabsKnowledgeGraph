import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.workflows.main_pipeline import app

async def test_fact_assembly():
    print("--- Testing Fact-as-Node Pipeline ---")
    
    # Input text with causal relationship
    chunk_text = "Because Apple released the new iPhone 15, the tech giant expects high sales."
    metadata = {"doc_date": "2023-09-22", "chunk_id": "test_fact_1", "section_header": "Business"}
    
    print(f"Input: {chunk_text}")
    
    # Run Pipeline
    inputs = {
        "chunk_text": chunk_text,
        "metadata": metadata
    }
    
    try:
        result = await app.ainvoke(inputs)
        print("\n✅ Pipeline Finished Successfully.")
        
        # Verify Results (State)
        facts = result.get("atomic_facts", [])
        links = result.get("causal_links", [])
        
        print(f"\nExtracted {len(facts)} facts:")
        for i, f in enumerate(facts):
            print(f"{i}. {f.fact}")
            
        print(f"\nExtracted {len(links)} causal links:")
        for l in links:
            print(f"{l.cause_index} -> {l.effect_index}: {l.reasoning}")
            
    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_fact_assembly())
