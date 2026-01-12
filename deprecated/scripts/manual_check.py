import sys
from src.agents.mcp_server import _resolve_entity_or_topic_logic, _explore_neighbors_logic, _get_chunk_logic

def print_separator(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def manual_check():
    user_id = "default_tenant"
    query = "Alphabet Inc."
    
    # 1. Resolve
    print_separator(f"STEP 1: Resolve Entity ('{query}')")
    resolved = _resolve_entity_or_topic_logic(query, "Entity", user_id)
    print(f"OUTPUT: {resolved}")
    
    if not resolved:
        print("❌ No entities found. Exiting.")
        return

    entity_name = resolved[0]
    print(f"✅ Selected Entity: {entity_name}")

    # 2. Explore
    print_separator(f"STEP 2: Explore Neighbors ('{entity_name}')")
    neighbors = _explore_neighbors_logic(entity_name, user_id)
    
    if not neighbors:
        print("❌ No neighbors found.")
        return
        
    for i, n in enumerate(neighbors):
        print(f"[{i}] {n}")
        
    # 3. Get Chunk
    # Try to pick a valid one automatically for demonstration
    print_separator("STEP 3: Get Chunk (Testing first valid connection)")
    
    target_conn = None
    e1, e2, edge = None, None, None
    
    for n in neighbors:
        # Check Outgoing: "Source --[EDGE]--> Target"
        if "-->" in n:
            parts = n.split(" --[")
            e1 = parts[0].strip()
            rest = parts[1].split("]--> ")
            edge = rest[0].strip()
            e2 = rest[1].strip()
            target_conn = n
            break
        # Check Incoming: "Target <----[EDGE]---- Source"
        elif "<----" in n:
            parts = n.split(" <----[")
            e2 = parts[0].strip() # Target is the entity we started with (e2 in (e1)-[]->(e2))
            rest = parts[1].split("]---- ")
            edge = rest[0].strip()
            e1 = rest[1].strip() # Source is e1
            target_conn = n
            break
    
    if target_conn:
        print(f"Testing Connection: {e1} --[{edge}]--> {e2}")
        chunk_output = _get_chunk_logic(e1, e2, edge, user_id)
        print("\n--- CHUNK OUTPUT START ---")
        print(chunk_output)
        print("--- CHUNK OUTPUT END ---")
    else:
        print("❌ Could not parse any neighbor string to test chunk retrieval.")

if __name__ == "__main__":
    manual_check()
