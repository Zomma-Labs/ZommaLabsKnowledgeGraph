
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schemas.atomic_fact import AtomicFact
from src.workflows.main_pipeline import parallel_resolution_node, GraphState

# Mock State with Duplicate Facts and "Sales" as Subject
mock_state: GraphState = {
    "chunk_text": "Mock Text",
    "metadata": {"doc_id": "test_doc"},
    "group_id": "test_tenant_dedup", # specific test tenant
    "episodic_uuid": "mock_episode",
    "atomic_facts": [
        # Fact 1: Consumer Spending (Topic)
        AtomicFact(
            fact="Consumer spending increased in Q3.",
            subject="Consumer Spending",
            subject_type="Topic", # Correctly typed by (mock) Atomizer
            object=None,
            topics=["Consumer Spending"]
        ),
        # Fact 2: Duplicate Consumer Spending (Topic)
        AtomicFact(
            fact="Consumer spending was strong.",
            subject="Consumer Spending",
            subject_type="Topic",
            object=None,
            topics=["Consumer Spending"]
        ),
        # Fact 3: Sales (Topic acting as Subject)
        AtomicFact(
            fact="Sales rose by 5%.",
            subject="Sales",
            subject_type="Topic", # Correctly typed by (mock) Atomizer
            object=None,
            topics=["Sales"]
        ),
        # Fact 4: Apple (Entity)
        AtomicFact(
            fact="Apple released a new phone.",
            subject="Apple",
            subject_type="Entity",
            object="Phone",
            object_type="Entity"
        )
    ],
    "resolved_entities": [],
    "resolved_topics": [],
    "classified_relationships": [],
    "causal_links": [],
    "errors": []
}

async def run_test():
    print("--- Running Reproduction Test ---")
    
    # 1. Run Parallel Resolution
    # This calls the REAL pipeline function, which checks for graph candidates & creates UUIDs
    result = await parallel_resolution_node(mock_state)
    
    resolved_ents = result["resolved_entities"]
    resolved_topics = result["resolved_topics"]
    
    print("\n--- Results Analysis ---")
    
    # Check 1: Deduplication of "Consumer Spending"
    # Both facts should point to the SAME URI/UUID for Subject "Consumer Spending"
    cs_uuid_1 = resolved_ents[0]["subject_uri"]
    cs_uuid_2 = resolved_ents[1]["subject_uri"]
    
    print(f"Consumer Spending (Fact 1) UUID: {cs_uuid_1}")
    print(f"Consumer Spending (Fact 2) UUID: {cs_uuid_2}")
    
    if cs_uuid_1 == cs_uuid_2:
        print("✅ PASS: Consumer Spending deduped correctly.")
    else:
        print("❌ FAIL: Consumer Spending has different UUIDs!")
        
    # Check 2: Sales Type
    # Should have subject_type="Topic"
    sales_type = resolved_ents[2]["subject_type"]
    print(f"Sales (Fact 3) Type: {sales_type}")
    
    if sales_type == "Topic":
        print("✅ PASS: Sales correctly typed as Topic.")
    else:
        print(f"❌ FAIL: Sales typed as {sales_type}")
        
    # Check 3: Apple Type
    apple_type = resolved_ents[3]["subject_type"]
    if apple_type == "Entity":
         print("✅ PASS: Apple correctly typed as Entity.")
    else:
         print(f"❌ FAIL: Apple typed as {apple_type}")

if __name__ == "__main__":
    # Ensure we are in project root
    sys.path.append(os.getcwd())
    asyncio.run(run_test())
