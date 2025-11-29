import json
import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.abspath("src"))

from agents.atomizer import atomizer

def main():
    # input_file = "src/chunker/SAVED/berkshirehathaway.jsonl"
    input_file = "src/chunker/SAVED/beigebook_20251015.jsonl"
    print(f"Reading from {input_file}...")
    
    chunks = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 10:
                continue
            if i >= 15:
                break
            chunks.append(json.loads(line))
            
    print(f"Loaded {len(chunks)} chunks.")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Processing Chunk {i+1} ---")
        chunk_text = chunk.get('body', '') # Using 'body' as the key
        # If 'body' key is missing, print keys to debug
        if not chunk_text:
             print(f"Keys in chunk: {chunk.keys()}")
             # Fallback
             chunk_text = chunk.get('text', '') or chunk.get('page_content', '')
        
        metadata = {k: v for k, v in chunk.items() if k != 'text' and k != 'page_content'}
        
        print(f"Input Text (first 100 chars): {chunk_text[:100]}...")
        
        try:
            facts = atomizer(chunk_text, metadata)
            print(f"Generated {len(facts)} atomic facts:")
            for j, fact in enumerate(facts):
                print(f"  {j+1}. {fact.fact}")
                print(f"     Entities: {fact.entities}")
                if fact.key_concepts:
                    print(f"     Key Concepts: {fact.key_concepts}")
                if fact.date_context:
                    print(f"     Date: {fact.date_context}")
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")

if __name__ == "__main__":
    main()
