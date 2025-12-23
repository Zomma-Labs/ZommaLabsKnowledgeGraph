from src.util.services import get_services

def list_entities():
    services = get_services()
    query = "MATCH (n:EntityNode) WHERE n.group_id = 'default_tenant' RETURN labels(n) as labels, n.name as name LIMIT 20"
    results = services.neo4j.query(query)
    
    print(f"--- Entities for 'default_tenant' ---")
    if not results:
        print("No nodes found for 'default_tenant'. Checking ALL nodes...")
        query_all = "MATCH (n) RETURN n.group_id as group, labels(n) as labels, n.name as name LIMIT 10"
        results_all = services.neo4j.query(query_all)
        for row in results_all:
             print(f"[{row['group']}] {row['labels']} : {row['name']}")
    else:
        for row in results:
            print(f"{row['labels']} : {row['name']}")

if __name__ == "__main__":
    list_entities()
