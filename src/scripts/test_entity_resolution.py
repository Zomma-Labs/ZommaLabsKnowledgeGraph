import sys
import os
import asyncio
import uuid
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.workflows.main_pipeline import app
from src.tools.neo4j_client import Neo4jClient

async def run_pipeline(chunk_text: str, group_id: str) -> Dict[str, str]:
    """Runs the pipeline and returns a map of Entity Name -> UUID"""
    print(f"\n--- Running Pipeline for '{group_id}' ---")
    print(f"Input: {chunk_text}")
    
    inputs = {
        "chunk_text": chunk_text,
        "metadata": {"source_id": "test_resolution", "group_id": group_id}
    }
    
    try:
        result = await app.ainvoke(inputs)
        resolved = result.get("resolved_entities", [])
        
        entity_map = {}
        for res in resolved:
            if res.get("subject_label"):
                entity_map[res["subject_label"]] = res["subject_uri"]
            if res.get("object_label"):
                entity_map[res["object_label"]] = res["object_uri"]
                
        return entity_map
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return {}

async def test_entity_resolution():
    print("🚀 Starting Entity Resolution Test")
    
    # 1. Apple Inc. (Tenant A)
    print("\nStep 1: Apple Inc. (Tenant A)")
    map1 = await run_pipeline("Apple released the new iPhone 15.", "tenant_A")
    apple_inc_uuid = map1.get("Apple")
    print(f"Apple Inc. UUID: {apple_inc_uuid}")
    
    # 2. Apple Fruit (Tenant A)
    print("\nStep 2: Apple Fruit (Tenant A)")
    map2 = await run_pipeline("The apple is a tasty fruit.", "tenant_A")
    apple_fruit_uuid = map2.get("Apple") or map2.get("apple") # Case might vary
    print(f"Apple Fruit UUID: {apple_fruit_uuid}")
    
    if apple_inc_uuid != apple_fruit_uuid:
        print("✅ SUCCESS: Apple Inc. and Apple Fruit have DIFFERENT UUIDs.")
    else:
        print("❌ FAILURE: Apple Inc. and Apple Fruit have SAME UUID.")

    # 3. Apple Inc. Again (Tenant A) - Should Merge
    print("\nStep 3: Apple Inc. Again (Tenant A)")
    map3 = await run_pipeline("Apple posted record profits for the quarter.", "tenant_A")
    apple_inc_2_uuid = map3.get("Apple")
    print(f"Apple Inc. (Run 2) UUID: {apple_inc_2_uuid}")
    
    if apple_inc_uuid == apple_inc_2_uuid:
        print("✅ SUCCESS: Apple Inc. merged correctly.")
    else:
        print("❌ FAILURE: Apple Inc. did NOT merge.")

    # 4. Apple REIT (Tenant A)
    print("\nStep 4: Apple REIT (Tenant A)")
    map4 = await run_pipeline("Apple Hospitality REIT owns many hotels.", "tenant_A")
    apple_reit_uuid = map4.get("Apple Hospitality REIT") or map4.get("Apple") # Depends on extraction
    print(f"Apple REIT UUID: {apple_reit_uuid}")
    
    if apple_reit_uuid != apple_inc_uuid and apple_reit_uuid != apple_fruit_uuid:
        print("✅ SUCCESS: Apple REIT is distinct from Apple Inc. and Apple Fruit.")
    else:
        print("❌ FAILURE: Apple REIT collision.")

    # 5. Multi-Tenancy Check (Tenant B)
    print("\nStep 5: Apple Inc. (Tenant B)")
    map5 = await run_pipeline("Apple released the new iPhone 15.", "tenant_B")
    apple_inc_tenant_b_uuid = map5.get("Apple")
    print(f"Apple Inc. (Tenant B) UUID: {apple_inc_tenant_b_uuid}")
    
    if apple_inc_tenant_b_uuid != apple_inc_uuid:
        print("✅ SUCCESS: Tenant B Apple is distinct from Tenant A Apple.")
    else:
        print("❌ FAILURE: Tenant Isolation Failed.")

if __name__ == "__main__":
    asyncio.run(test_entity_resolution())
