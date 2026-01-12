from src.agents.mcp_server import _resolve_entity_or_topic_logic, _get_chunk_logic, _explore_neighbors_logic
import json

def test_mcp_tools():
    # Use default_tenant for our real graph test
    test_user_id = "default_tenant" 
    
    print("\n--- 1. Testing Resolve Entity (Alphabet Inc.) ---")
    # Search for 'Alphabet' or related terms
    terms = _resolve_entity_or_topic_logic("Alphabet Inc.", "Entity", test_user_id)
    print(f"Resolved Terms: {terms}")
    
    if not terms:
        print("WARNING: No terms found. Using hardcoded 'Alphabet Inc.' for further tests.")
        entity_name = "Alphabet Inc."
    else:
        entity_name = terms[0]
        
    print(f"Using Entity: {entity_name}")

    print(f"\n--- 2. Testing Explore Neighbors for {entity_name} ---")
    neighbors = _explore_neighbors_logic(entity_name, test_user_id)
    for n in neighbors:
        print(n)
        
    if not neighbors:
        print("No neighbors found. Skipping Traversal Test.")
        return

    # Pick a connection to test get_chunk
    # Expect format: "Source --[edge]--> Target" or "Target <----[edge]---- Source"
    
    # Simple parser to find a valid test case
    test_case = None
    for n in neighbors:
        if "-->" in n:
            parts = n.split(" --[")
            source = parts[0].strip()
            rest = parts[1].split("]--> ")
            edge = rest[0].strip()
            target = rest[1].strip()
            test_case = (source, target, edge)
            break
        elif "<----" in n:
             # "Target <----[edge]---- Source"
             parts = n.split(" <----[")
             target_node = parts[0].strip() # This is the entity_name we started with
             rest = parts[1].split("]---- ")
             edge = rest[0].strip()
             source_node = rest[1].strip()
             
             # For get_chunk(e1, e2, edge), e1 is Source, e2 is Target
             test_case = (source_node, target_node, edge)
             break
             
    if test_case:
        e1, e2, edge = test_case
        print(f"\n--- 3. Testing Get Chunk ({e1} -[{edge}]-> {e2}) ---")
        chunk = _get_chunk_logic(e1, e2, edge, test_user_id)
        print(chunk)
    else:
        print("Could not parse a neighbor connection to test get_chunk.")

if __name__ == "__main__":
    test_mcp_tools()
