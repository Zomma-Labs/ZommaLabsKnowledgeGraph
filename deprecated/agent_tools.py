from typing import List, Optional, Dict
from langchain_core.tools import tool
from src.util.vector_store import VectorStore

@tool
def lookup_entity(query: str) -> str:
    """
    Finds an existing entity in the Knowledge Graph by name.
    Use this to get the correct name and UUID for Cypher queries.
    
    Args:
        query: The name of the entity to look up (e.g., "Apple", "Minneapolis District").
        
    Returns:
        A list of matching entities found in the graph with their names and UUIDs.
    """
    from src.util.services import get_services
    services = get_services()
    
    # 1. Search the graph directly for existing nodes (Fuzzy/Contains match)
    print(f"   🔍 Searching Graph for '{query}'...")
    cypher = """
    MATCH (n:EntityNode) 
    WHERE toLower(n.name) CONTAINS toLower($query) 
    RETURN n.name, n.uuid, n.summary
    LIMIT 10
    UNION
    MATCH (n:TopicNode) 
    WHERE toLower(n.name) CONTAINS toLower($query) 
    RETURN n.name, n.uuid, n.summary
    LIMIT 5
    """
    try:
        results = services.neo4j.query(cypher, {"query": query})
        if results:
            output = f"Found {len(results)} existing nodes in the graph:\n"
            for r in results:
                summary = r.get('n.summary', '')
                summary_str = f" - {summary[:50]}..." if summary else ""
                output += f"- Name: '{r['n.name']}', UUID: '{r['n.uuid']}'{summary_str}\n"
            return output
    except Exception as e:
        return f"Error searching graph: {e}"

    # 2. If not found in graph, we can't query it.
    return "No matching entities found in the existing Knowledge Graph."

@tool
def lookup_relationship(description: str) -> str:
    """
    Finds relevant relationship types from the schema based on a natural language description.
    Use this to understand which relationship types (edge labels) to use in Cypher.
    
    Args:
        description: A description of the relationship or action (e.g., "companies hiring people", "wage pressure").
        
    Returns:
        A list of potential relationship types (edge labels) with descriptions.
    """
    from src.util.services import get_services
    services = get_services()
    
    vector_store = VectorStore(client=services.qdrant_relationships)
    candidates = vector_store.search_relationships(description, limit=5)
    if not candidates:
        return "No relevant relationship types found."
        
    result = "Potential Relationship Types (Edge Labels):\n"
    for c in candidates:
        result += f"- {c.name}: {c.description}\n"
    return result

@tool
def execute_cypher(query: str) -> str:
    """
    Executes a Read-Only Cypher query against the Neo4j database.
    Use this to retrieve data after you have looked up entity names and relationship types.
    
    Args:
        query: The Cypher query string.
        
    Returns:
        The query results as a string representation of the list of records.
    """
    from src.util.services import get_services
    services = get_services()
    
    # Safety check: Prevent write operations
    if any(keyword in query.upper() for keyword in ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"]):
        return "Error: Only READ-ONLY queries are allowed."
        
    try:
        results = services.neo4j.query(query)
        if not results:
            return "No records found."
        return str(results)
    except Exception as e:
        return f"Cypher Execution Error: {e}"

@tool
def search_graph_text(keywords: str) -> str:
    """
    Performs a full-text search on the content of chunks and facts in the graph.
    Use this when you are looking for general topics (e.g., "wage pressures", "hiring", "inflation") 
    that might not map to specific entities.
    
    Args:
        keywords: A string of keywords to search for (e.g., "wage pressures").
        
    Returns:
        A list of matching chunks with their content, header path, and connected entities.
    """
    from src.util.services import get_services
    services = get_services()
    
    # Search EpisodicNode content and FactNode content
    cypher = """
    MATCH (ep:EpisodicNode)
    WHERE toLower(ep.content) CONTAINS toLower($keywords)
    OPTIONAL MATCH (entity)-[r]->(ep)
    WHERE entity:EntityNode OR entity:TopicNode
    RETURN ep.content as chunk, ep.header_path as header, 
           collect(DISTINCT {name: entity.name, edge: type(r)}) as connections
    LIMIT 5
    UNION
    MATCH (f:FactNode)-[:MENTIONED_IN]->(ep:EpisodicNode)
    WHERE toLower(f.content) CONTAINS toLower($keywords)
    RETURN ep.content as chunk, ep.header_path as header, 
           [{name: f.content, edge: f.fact_type}] as connections
    LIMIT 5
    """
    
    try:
        results = services.neo4j.query(cypher, {"keywords": keywords})
        if not results:
            return "No matching content found."
        
        output = f"Found {len(results)} relevant chunks:\n\n"
        for r in results:
            header = r.get('header', 'N/A')
            chunk = r.get('chunk', '')[:200] + "..." if len(r.get('chunk', '')) > 200 else r.get('chunk', '')
            connections = r.get('connections', [])
            conn_str = ", ".join([f"{c['name']} ({c['edge']})" for c in connections if c.get('name')]) or "No direct entity connections"
            
            output += f"--- Header: {header} ---\n"
            output += f"Content: {chunk}\n"
            output += f"Connections: {conn_str}\n\n"
        return output
    except Exception as e:
        return f"Search Error: {e}"

