from typing import List, Optional, Dict
from langchain_core.tools import tool
from src.agents.FIBO_librarian import FIBOLibrarian
from src.agents.analyst import AnalystAgent
from src.tools.neo4j_client import Neo4jClient

# Initialize singletons for tools
_librarian = FIBOLibrarian()
_analyst = AnalystAgent()
_neo4j = Neo4jClient()

@tool
def lookup_entity(query: str) -> str:
    """
    Resolves a company name, organization, or economic concept to its FIBO URI.
    Use this to find the correct 'uri' to use in Cypher queries.
    
    Args:
        query: The name of the entity to look up (e.g., "Apple", "Wage Pressures").
        
    Returns:
        A string describing the best match found, including URI and Label, or "No match found".
    """
    match = _librarian.resolve(query)
    if match:
        return f"Match found: Label='{match['label']}', URI='{match['uri']}', Score={match['score']:.2f}"
    return "No match found in FIBO ontology."

@tool
def lookup_relationship(description: str) -> str:
    """
    Finds relevant relationship types from the schema based on a natural language description.
    Use this to understand which relationship types (edges) to use in Cypher.
    
    Args:
        description: A description of the relationship or action (e.g., "companies hiring people", "prices going up").
        
    Returns:
        A list of potential relationship types with descriptions.
    """
    candidates = _analyst.vector_store.search_relationships(description, limit=5)
    if not candidates:
        return "No relevant relationship types found."
        
    result = "Potential Relationship Types:\n"
    for c in candidates:
        result += f"- {c.name}: {c.description}\n"
    return result

@tool
def execute_cypher(query: str) -> str:
    """
    Executes a Read-Only Cypher query against the Neo4j database.
    Use this to retrieve data after you have looked up necessary URIs and Relationship Types.
    
    Args:
        query: The Cypher query string.
        
    Returns:
        The query results as a string representation of the list of records.
    """
    # Safety check: Prevent write operations
    if any(keyword in query.upper() for keyword in ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"]):
        return "Error: Only READ-ONLY queries are allowed."
        
    try:
        results = _neo4j.query(query)
        if not results:
            return "No records found."
        return str(results)
    except Exception as e:
        return f"Cypher Execution Error: {e}"

@tool
def search_graph_text(keywords: str) -> str:
    """
    Performs a full-text search on the 'fact' property of relationships in the graph.
    Use this when you are looking for general topics (e.g., "wage pressures", "hiring", "inflation") 
    that might not map to specific entities.
    
    Args:
        keywords: A string of keywords to search for (e.g., "wage pressures").
        
    Returns:
        A list of matching facts with their subject, object, and relationship type.
    """
    # Simple case-insensitive containment search
    # In a real production system, we'd use a FullText index
    cypher = f"""
    MATCH (s)-[r]->(o)
    WHERE toLower(r.fact) CONTAINS toLower($keywords)
    RETURN s.name, type(r), o.name, r.fact, r.date
    LIMIT 10
    """
    
    try:
        results = _neo4j.query(cypher, {"keywords": keywords})
        if not results:
            return "No matching facts found."
        
        output = "Found the following facts:\n"
        for r in results:
            output += f"- [{r['type(r)']}] {r['s.name']} -> {r['o.name']}: \"{r['r.fact']}\" (Date: {r['r.date']})\n"
        return output
    except Exception as e:
        return f"Search Error: {e}"
