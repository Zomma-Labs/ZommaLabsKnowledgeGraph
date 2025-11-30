import sys
import os

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import json
from typing import Dict
from src.workflows.main_pipeline import app
from src.scripts.setup_graph_index import setup_index

# Path to saved chunks
CHUNKS_DIR = "src/chunker/SAVED"

async def main():
    print("🚀 Starting Pipeline Run...")
    
    # Ensure Vector Indices Exist
    print("\n⚙️ Setting up Vector Indices...")
    setup_index()
    print("✅ Indices Ready.\n")
    
    # 1. Load Chunks
    # We'll look for .jsonl files in the SAVED directory
    # Filter for Beige Book only as requested
    chunk_files = [f for f in os.listdir(CHUNKS_DIR) if f.endswith('.jsonl') and 'beigebook' in f.lower()]
    
    if not chunk_files:
        print("❌ No chunk files found in src/chunker/SAVED")
        return

    print(f"📂 Found {len(chunk_files)} chunk files: {chunk_files}")
    
    tasks = []
    
    for filename in chunk_files:
        filepath = os.path.join(CHUNKS_DIR, filename)
        print(f"\n📄 Processing file: {filename}")
        
        with open(filepath, 'r') as f:
            # Read line by line (each line is a chunk object)
            # Limit to first few chunks for testing to avoid huge costs/time
            for i, line in enumerate(f):
                # if i >= 3: # Limit to 3 chunks per file for test
                #     break
                
                chunk_data = json.loads(line)
                
                # Extract text and metadata
                # Based on standard chunker output:
                chunk_text = chunk_data.get("body", "")
                metadata = chunk_data.get("metadata", {})
                
                if not chunk_text:
                    print(f"   ⚠️ Skipping chunk {i}: No text found.")
                    continue
                    
                # Initialize State
                initial_state = {
                    "chunk_text": chunk_text,
                    "metadata": metadata,
                    "atomic_facts": [],
                    "resolved_entities": [],
                    "classified_relationships": [],
                    "errors": []
                }
                
                # Create Task
                tasks.append(process_chunk(i, initial_state))

    # Run all chunks in parallel
    if tasks:
        print(f"   🚀 Running {len(tasks)} chunks in parallel...")
        await asyncio.gather(*tasks)

async def process_chunk(i: int, state: Dict):
    print(f"   🔄 Starting Chunk {i}...")
    try:
        # app.ainvoke is async
        result = await app.ainvoke(state)
        
        if result.get("errors"):
            print(f"   ❌ Chunk {i} Errors: {result['errors']}")
        else:
            print(f"   ✅ Chunk {i} Completed Successfully.")
            
    except Exception as e:
        print(f"   ❌ Chunk {i} Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
