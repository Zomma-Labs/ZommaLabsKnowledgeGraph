import unittest
import json
import os
import asyncio
from src.workflows.main_pipeline import app

class TestMultiTenantPipeline(unittest.TestCase):
    def setUp(self):
        self.chunk_dir = "/home/rithv/Programming/Startups/ZommaLabsKG/src/chunker/SAVED"
        self.beigebook_file = os.path.join(self.chunk_dir, "beigebook_20251015.jsonl")
        self.berkshire_file = os.path.join(self.chunk_dir, "berkshirehathaway.jsonl")

    def load_chunks(self, filepath, limit=5):
        chunks = []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                chunks.append(json.loads(line))
        return chunks

    async def run_pipeline_for_chunk(self, chunk, group_id):
        state = {
            "chunk_text": chunk["body"],
            "metadata": chunk.get("metadata", {}),
            "group_id": group_id,
            "atomic_facts": [],
            "resolved_entities": [],
            "classified_relationships": [],
            "causal_links": [],
            "errors": []
        }
        # Add other fields from chunk to metadata if needed
        state["metadata"]["doc_id"] = chunk.get("doc_id")
        state["metadata"]["chunk_id"] = chunk.get("chunk_id")
        
        print(f"Running pipeline for chunk {chunk.get('chunk_id')} with group_id {group_id}")
        result = await app.ainvoke(state)
        return result

    def test_multi_tenant_execution(self):
        async def run_test():
            # Tenant A - Beige Book
            print("\n--- Processing Tenant A (Beige Book) ---")
            beigebook_chunks = self.load_chunks(self.beigebook_file, limit=3)
            for chunk in beigebook_chunks:
                result = await self.run_pipeline_for_chunk(chunk, "tenant_A")
                self.assertFalse(result.get("errors"), f"Pipeline failed for tenant_A chunk {chunk.get('chunk_id')}: {result.get('errors')}")
                # Check if group_id is preserved (though result might not have it, the side effects in Neo4j should)
                # Ideally we would query Neo4j here to verify, but the prompt says "this is initself a test so it doesnt need tests"
                # implying the execution itself is the test. I'll add basic assertions on the result state.
                self.assertTrue(result.get("episodic_uuid"), "Episodic UUID missing")

            # Tenant B - Berkshire Hathaway
            print("\n--- Processing Tenant B (Berkshire Hathaway) ---")
            berkshire_chunks = self.load_chunks(self.berkshire_file, limit=3)
            for chunk in berkshire_chunks:
                result = await self.run_pipeline_for_chunk(chunk, "tenant_B")
                self.assertFalse(result.get("errors"), f"Pipeline failed for tenant_B chunk {chunk.get('chunk_id')}: {result.get('errors')}")
                self.assertTrue(result.get("episodic_uuid"), "Episodic UUID missing")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_test())
        loop.close()

if __name__ == '__main__':
    unittest.main()
