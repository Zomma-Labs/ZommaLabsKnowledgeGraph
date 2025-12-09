import asyncio
import uuid
from src.util.services import get_services
from src.agents.graph_assembler import GraphAssembler
from src.agents.graph_enhancer import GraphEnhancer
from src.schemas.atomic_fact import AtomicFact

async def repro():
    services = get_services()
    assembler = GraphAssembler(services=services)
    neo4j = services.neo4j

    group_id = "repro_test_tenant_1"
    entity_name = f"ReproEntity_{uuid.uuid4().hex[:8]}"
    
    print(f"--- TESTING WITH ENTITY: {entity_name} ---")

    # 1. Simulate Resolution (Creation of Internal Node)
    # Current buggy behavior: Creates node with UUID but NO URI property (implicitly)
    # OR in main_pipeline it creates it with UUID.
    
    # We will manually create a "Buggy" node that mimics what main_pipeline currently does
    # (or what we suspect it does)
    # Actually, let's try to verify what main_pipeline DOES.
    # main_pipeline currently does:
    # cypher_atomic_create = ... MERGE (n:EntityNode {name: $name...}) ON CREATE SET n.uuid = $uuid ...
    # It does NOT set n.uri.

    node_uuid = str(uuid.uuid4())
    print(f"Step 1: Creating Node via Cypher (Simulating main_pipeline resolution). UUID: {node_uuid}")
    
    cypher_create = """
    MERGE (n:EntityNode {name: $name, group_id: $group_id})
    ON CREATE SET 
        n.uuid = $uuid, 
        n.summary = 'A test entity',
        n.created_at = datetime()
    RETURN n.uuid
    """
    neo4j.query(cypher_create, {
        "name": entity_name,
        "group_id": group_id,
        "uuid": node_uuid
    })

    # 2. Simulate Assembly
    # Assembler is called with `subject_uri` which is currently mapped to the UUID in main_pipeline
    # But assembler uses that value to query `uri` property?
    # Let's check assembler code again.
    # MERGE (s:EntityNode {uri: $subj_uri...})
    
    print("Step 2: Running Assembler (Simulating assemble_node)")
    
    fact = AtomicFact(fact="Test Fact", subject=entity_name, object="Something", subject_type="Entity", object_type="Entity")
    
    # In main_pipeline: subject_uuid=res_ent["subject_uuid"] which IS the UUID from resolution.
    # So we pass node_uuid as subject_uuid
    try:
        assembler.assemble_fact_node(
            fact_obj=fact,
            subject_uuid=node_uuid,
            subject_label=entity_name,
            object_uuid=None,
            object_label=None,
            episode_uuid=str(uuid.uuid4()),
            group_id=group_id,
            subject_summary="A test entity"
        )
    except Exception as e:
        print(f"Assembler failed: {e}")

    # 3. Check Results
    print("Step 3: Checking for Duplicates")
    result = neo4j.query("MATCH (n:EntityNode {name: $name, group_id: $group_id}) RETURN n.uuid, n.uri", {
        "name": entity_name,
        "group_id": group_id
    })
    
    print(f"Found {len(result)} nodes for name '{entity_name}':")
    for r in result:
        print(f" - UUID: {r.get('uuid')}, URI: {r.get('uri')}")
        
    if len(result) > 1:
        print("❌ FAIL: Duplicates detected!")
    elif len(result) == 1:
        # Check if URI was set? current behavior sets uri on the second one. 
        # If we fixed it, we expect 1 node and it relies on UUID.
        print("✅ SUCCESS: Single node found.")
    else:
        print("❓ WEIRD: No nodes found?")

    assembler.close()

if __name__ == "__main__":
    asyncio.run(repro())
