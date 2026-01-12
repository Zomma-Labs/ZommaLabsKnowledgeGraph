import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.atomizer import atomizer
from src.agents.entity_extractor import EntityExtractor

def test_extraction_pipeline():
    print("🧪 Testing Context-Aware Extraction Pipeline...")
    
    # 1. Test Data: Use a chunk where context is critical
    # "It" in the second sentence refers to "Apple".
    chunk_text = (
        "Apple released its quarterly earnings report yesterday. "
        "It reported a significant increase in services revenue, beating expectations. "
        "Tim Cook praised the team's effort."
    )
    metadata = {"doc_date": "2023-11-01"}
    
    print(f"\n📄 Input Chunk:\n{chunk_text}\n")
    
    # 2. Test Decompostion (Atomizer)
    print("--- Step 1: Decomposition (Atomizer) ---")
    try:
        propositions = atomizer(chunk_text, metadata)
        print(f"✅ Decomposed into {len(propositions)} propositions:")
        for idx, prop in enumerate(propositions):
            print(f"   {idx+1}. {prop}")
            
        # Verify Pronoun Resolution happened in Atomizer (Rule 1 of Atomizer)
        # Atomizer SHOULD replace 'It' with 'Apple' if it does its job well.
        # But wait, the user's request for context-aware extraction implies 
        # that sometimes Atomizer might miss it or we want to double check?
        # The prompt for Atomizer SAYS "Resolve pronouns". 
        # BUT EntityExtractor ALSO has context now.
        # If Atomizer resolves it, great. If not, EntityExtractor should catch it.
        # Let's see what happens.
    except Exception as e:
        print(f"❌ Atomizer Failed: {e}")
        return

    # 3. Test Entity Extraction (Context-Aware)
    print("\n--- Step 2: Entity Extraction (Context-Aware) ---")
    extractor = EntityExtractor()
    
    for prop in propositions:
        print(f"\n🔍 Extracting from: '{prop}'")
        try:
            # We pass the ORIGINAL chunk as context
            fact = extractor.extract(prop, chunk_text)
            
            print(f"   Subject: {fact.subject} ({fact.subject_type})")
            print(f"   Object:  {fact.object} ({fact.object_type})")
            print(f"   Topics:  {fact.topics}")
            
            # Validation Logic
            if "beat" in prop.lower() or "increase" in prop.lower():
                # This fact is about Apple/Revenue
                if "Apple" in fact.subject or "Revenue" in fact.subject or "It" not in fact.subject:
                     print("   ✅ Subject looks resolved/specific.")
                else:
                     print("   ⚠️ Subject might be vague.")
                     
        except Exception as e:
            print(f"   ❌ Extraction Failed: {e}")

if __name__ == "__main__":
    test_extraction_pipeline()
