import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.atomizer import atomizer

def test_atomizer_reflexion():
    print("--- Testing Atomizer with Reflexion ---")
    
    # Input text where "The tech giant" refers to "Apple"
    chunk_text = "Apple released the new iPhone 15. The tech giant expects high sales."
    metadata = {"doc_date": "2023-09-22", "chunk_id": "test_1", "section_header": "Business"}
    
    print(f"Input: {chunk_text}")
    
    facts = atomizer(chunk_text, metadata)
    
    print(f"\nExtracted {len(facts)} facts:")
    for i, fact in enumerate(facts):
        print(f"{i+1}. Subject: {fact.subject} | Fact: {fact.fact} | Object: {fact.object}")

if __name__ == "__main__":
    test_atomizer_reflexion()
