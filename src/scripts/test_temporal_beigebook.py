import sys
import os
import json
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.temporal_extractor import TemporalExtractor
from src.schemas.document_types import Chunk

def load_chunks(filepath):
    chunks = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Basic reconstruction of Chunk object from JSON dict
            # We filter for 'text' chunks primarily for the extractor
            if data.get("metadata", {}).get("content_type") == "text":
                   chunks.append(Chunk(**data))
    return chunks

def run_test():
    filepath = "src/chunker/SAVED/beigebook_20251015.jsonl"
    print(f"--- 📖 Testing Temporal Extractor on {filepath} ---")
    
    if not os.path.exists(filepath):
        print("❌ File not found.")
        return

    chunks = load_chunks(filepath)
    print(f"Loaded {len(chunks)} text chunks.")
    
    extractor = TemporalExtractor()
    
    # We can use the public method provided by the class
    # extract_date takes lists of strings
    
    text_bodies = [c.body for c in chunks]
    first_6 = text_bodies[:6]
    last_6 = text_bodies[-6:]
    title = chunks[0].heading if chunks else "Beige Book Test"
    
    print("\n--- 🕵️‍♀️ Extracting Date ---")
    
    # Use the internal llm logic to see reasoning too, but the public method returns just string.
    # I'll just use the public method first as intended.
    
    date = extractor.extract_date(first_6, last_6, title)
    print(f"\n✅ RESULT: {date}")

if __name__ == "__main__":
    run_test()
