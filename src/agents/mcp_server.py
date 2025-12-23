"""
ZommaLabs Knowledge Graph MCP Server
====================================

FastMCP server exposing graph retrieval tools for AI agents.

Tools:
    - resolve_entity_or_topic: Semantic search to find exact entity/topic names
    - explore_neighbors: Discover relationships connected to an entity
    - get_chunk: Retrieve source text evidence for a specific relationship

Run with: python -m src.agents.mcp_server
"""


from fastmcp import FastMCP, Context
from src.util.services import get_services

# Initialize FastMCP Server with instructions for AI agents
mcp = FastMCP(
    name="ZommaGraph",
    instructions="""
ZommaLabs Knowledge Graph - Graph Retrieval API
================================================

This server provides tools to navigate a financial knowledge graph and answer 
complex questions with ground-truth evidence from source documents.

## 3-Step Graph Retrieval Workflow

To answer a user question effectively, follow these steps IN ORDER:

### STEP 1: Resolve Entities (`resolve_entity_or_topic`)
- **Goal**: Map vague user terms (e.g., "Google", "wage pressure") to exact node names.
- **Why**: The graph requires exact matches. "Alphabet" won't work if the node is "Alphabet Inc.".
- **Example**: `resolve_entity_or_topic(query="Google", node_type="Entity")` → ["Alphabet Inc."]

### STEP 2: Explore Relationships (`explore_neighbors`)
- **Goal**: Discover what connects to your resolved entities.
- **Why**: You don't know the edge types or connected nodes yet. This gives you a map.
- **Example**: `explore_neighbors(entity_name="Alphabet Inc.")` → ["Alphabet Inc. --[INVESTED]-->  Waymo", ...]

### STEP 3: Retrieve Evidence (`get_chunk`)
- **Goal**: Get the actual source text (chunk) that justifies a specific relationship.
- **Why**: The edge tells you THAT something happened; the chunk tells you HOW, WHEN, and WHY.
- **Example**: `get_chunk(entity_one="Alphabet Inc.", entity_two="Waymo", edge_type="INVESTED")`

## Important Notes
- All queries are automatically scoped to your authorized tenant (data isolation).
- Always use exact entity names returned by resolve_entity_or_topic.
- If you can't find an entity, try different search terms or check for spelling variations.
"""
)

def get_user_id(ctx: Context) -> str:
    """
    Helper to extract user_id from Auth headers or Session.
    For local testing/development, it defaults to 'default_tenant' if no auth is present.
    """
    if not ctx:
        return "default_tenant"
    
    # Try to extract from request context (FastMCP specific) or fallback
    # Note: FastMCP Context object structure might vary, so we check standard places
    try:
        # Placeholder for actual auth extraction logic
        # user_id = ctx.meta.get("user_id")
        user_id = None 
        if not user_id:
            # For this phase, we default to 'default_tenant' as requested
            return "default_tenant"
        return user_id
    except Exception:
        return "default_tenant"

# --- Core Logic Functions (Testable) ---

def _resolve_entity_or_topic_logic(query: str, node_type: str, user_id: str) -> dict:
    """
    Core logic for resolving entities/topics.
    
    Returns a dict with:
        - found: bool indicating if any matches were found
        - results: list of matching entity/topic names
        - message: human-readable status message
    """
    services = get_services()
    
    # 1. Embed Query
    query_vector = services.embeddings.embed_query(query)
    
    # 2. Select index based on node_type
    index_name = "topic_embeddings" if node_type == "Topic" else "entity_embeddings"
    
    # 3. Vector Search (limit to top 10 for efficiency)
    results = services.neo4j.vector_search(
        index_name=index_name,
        query_vector=query_vector,
        top_k=20
    )
    
    valid_names = []
    
    for row in results:
        node = row.get('node')
        score = row.get('score', 0)
        
        # Defensive: skip malformed results
        if not node or not isinstance(node, dict):
            continue
        
        if score < 0.7:
            continue
            
        if node.get('group_id') != user_id:
            continue
        
        name = node.get('name')
        if name:
            valid_names.append(name)

    unique_names = sorted(list(set(valid_names)))
    
    if unique_names:
        return {
            "found": True,
            "results": unique_names,
            "message": f"Found {len(unique_names)} matching {node_type.lower()}(s) in the graph."
        }
    else:
        return {
            "found": False,
            "results": [],
            "message": f"NO MATCH FOUND: The {node_type.lower()} '{query}' does not exist in the knowledge graph. Do NOT search for variations - this {node_type.lower()} is simply not in the data."
        }


def _get_chunk_logic(entity_one: str, entity_two: str, edge_type: str, user_id: str) -> str:
    """Core logic for getting chunk content."""
    services = get_services()
    
    query = """
    MATCH (e1:EntityNode {name: $e1, group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(e2:EntityNode {name: $e2, group_id: $uid})
    WHERE type(r1) = $edge_type AND r1.fact_id = r2.fact_id
    RETURN c
    UNION
    MATCH (e1:EntityNode {name: $e1, group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(e2:TopicNode {name: $e2, group_id: $uid})
    WHERE type(r1) = $edge_type AND r1.fact_id = r2.fact_id
    RETURN c
    UNION
    MATCH (e1:TopicNode {name: $e1, group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(e2:EntityNode {name: $e2, group_id: $uid})
    WHERE type(r1) = $edge_type AND r1.fact_id = r2.fact_id
    RETURN c
    UNION
    MATCH (e1:TopicNode {name: $e1, group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(e2:TopicNode {name: $e2, group_id: $uid})
    WHERE type(r1) = $edge_type AND r1.fact_id = r2.fact_id
    RETURN c
    LIMIT 1
    """
    
    # Try original order first
    results = services.neo4j.query(query, {
        "e1": entity_one,
        "e2": entity_two,
        "edge_type": edge_type,
        "uid": user_id
    })
    
    # If no results, try swapping entities (handles LLM subject/object confusion)
    if not results:
        results = services.neo4j.query(query, {
            "e1": entity_two,
            "e2": entity_one,
            "edge_type": edge_type,
            "uid": user_id
        })
    
    if not results:
        return {
            "found": False,
            "chunk": None,
            "message": f"NO EVIDENCE FOUND: No source document connects '{entity_one}' to '{entity_two}' via [{edge_type}]. This specific relationship has no supporting evidence in the graph."
        }
        
    chunk_data = results[0]['c']
    
    formatted_output = f'''"""
DOCUMENT: {chunk_data.get('doc_id', 'N/A')}
CHUNK_id: {chunk_data.get('uuid', 'N/A')}
Header: {chunk_data.get('header_path', 'N/A')}
Date: {chunk_data.get('created_at', 'N/A')}

{chunk_data.get('content', '')}
"""'''
    
    return {
        "found": True,
        "chunk": formatted_output,
        "message": "Evidence retrieved successfully."
    }


def _explore_neighbors_logic(entity_name: str, user_id: str) -> dict:
    """
    Core logic for exploring neighbors.
    Returns a token-efficient grouped format instead of repeating entity name.
    """
    services = get_services()
    
    # Dictionaries to group by edge type
    outgoing = {}  # {edge_type: [target1, target2, ...]}
    incoming = {}  # {edge_type: [source1, source2, ...]}
    
    # 1. Outgoing relationships
    query_out = """
    MATCH (Start:EntityNode {name: $name, group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(Target:EntityNode {group_id: $uid})
    WHERE r1.fact_id = r2.fact_id
    RETURN type(r1) as edge, Target.name as target
    UNION
    MATCH (Start:EntityNode {name: $name, group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(Target:TopicNode {group_id: $uid})
    WHERE r1.fact_id = r2.fact_id
    RETURN type(r1) as edge, Target.name as target
    UNION
    MATCH (Start:TopicNode {name: $name, group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(Target:EntityNode {group_id: $uid})
    WHERE r1.fact_id = r2.fact_id
    RETURN type(r1) as edge, Target.name as target
    UNION
    MATCH (Start:TopicNode {name: $name, group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(Target:TopicNode {group_id: $uid})
    WHERE r1.fact_id = r2.fact_id
    RETURN type(r1) as edge, Target.name as target
    """
    results_out = services.neo4j.query(query_out, {"name": entity_name, "uid": user_id})
    for row in results_out:
        edge = row['edge']
        target = row['target']
        if edge not in outgoing:
            outgoing[edge] = []
        if target not in outgoing[edge]:
            outgoing[edge].append(target)
        
    # 2. Incoming relationships
    query_in = """
    MATCH (Source:EntityNode {group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(Target:EntityNode {name: $name, group_id: $uid})
    WHERE r1.fact_id = r2.fact_id
    RETURN Source.name as source, type(r1) as edge
    UNION
    MATCH (Source:EntityNode {group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(Target:TopicNode {name: $name, group_id: $uid})
    WHERE r1.fact_id = r2.fact_id
    RETURN Source.name as source, type(r1) as edge
    UNION
    MATCH (Source:TopicNode {group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(Target:EntityNode {name: $name, group_id: $uid})
    WHERE r1.fact_id = r2.fact_id
    RETURN Source.name as source, type(r1) as edge
    UNION
    MATCH (Source:TopicNode {group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(Target:TopicNode {name: $name, group_id: $uid})
    WHERE r1.fact_id = r2.fact_id
    RETURN Source.name as source, type(r1) as edge
    """
    results_in = services.neo4j.query(query_in, {"name": entity_name, "uid": user_id})
    for row in results_in:
        edge = row['edge']
        source = row['source']
        if edge not in incoming:
            incoming[edge] = []
        if source not in incoming[edge]:
            incoming[edge].append(source)
    
    # Check if any relationships found
    total_count = sum(len(v) for v in outgoing.values()) + sum(len(v) for v in incoming.values())
    
    if total_count == 0:
        return f"found: false\nentity: {entity_name}\nmessage: NO RELATIONSHIPS FOUND - stop exploring this entity"
    
    # Build TOON format response (Token-Oriented Object Notation)
    # Format: key: value with arrays as key[N]: item1,item2,item3
    lines = [
        "found: true",
        f"entity: {entity_name}",
    ]
    
    # Add outgoing relationships in TOON format
    if outgoing:
        lines.append("outgoing:")
        for edge_type, targets in outgoing.items():
            # TOON array syntax: key[count]: item1,item2,...
            lines.append(f"  {edge_type}[{len(targets)}]: {','.join(targets)}")
    
    # Add incoming relationships in TOON format  
    if incoming:
        lines.append("incoming:")
        for edge_type, sources in incoming.items():
            lines.append(f"  {edge_type}[{len(sources)}]: {','.join(sources)}")
    
    lines.append(f"total: {total_count}")
    
    return "\n".join(lines)


# --- MCP Tool Wrappers ---

@mcp.tool()
def resolve_entity_or_topic(query: str, node_type: str, ctx: Context) -> dict:
    """
    STEP 1: Resolve fuzzy user terms to strict Graph Entity Names.
    
    Use this tool FIRST. The graph requires exact node names. 
    This tool performs a semantic vector search to find the closest matching nodes.
    
    Args:
        query (str): The search text (e.g., "The search giant", "Mr. Page").
        node_type (str): The type of node to find. Use "Entity" for people/orgs, "Topic" for themes.
        ctx (Context): The request context (Injected automatically).

    Returns:
        dict: Contains 'found' (bool), 'results' (list of names), and 'message' (status). 
              If 'found' is False, do NOT search for variations - the entity is not in the graph.
    """
    return _resolve_entity_or_topic_logic(query, node_type, get_user_id(ctx))

@mcp.tool()
def get_chunk(entity_one: str, entity_two: str, edge_type: str, ctx: Context) -> dict:
    """
    STEP 3: Retrieve the specific text chunk (evidence) for a known relationship.
    
    Use this AFTER `explore_neighbors` has confirmed an edge exists.
    This tool traverses the specific path `(e1)-[edge]->(CHUNK)-[edge]->(e2)` and returns the chunk.
    
    Args:
        entity_one (str): The SOURCE entity name (exactly as returned by previous steps).
        entity_two (str): The TARGET entity name (exactly as returned by previous steps).
        edge_type (str): The active relationship name (e.g., "INVESTED", "SUED").
        ctx (Context): The request context (Injected automatically).

    Returns:
        dict: Contains 'found' (bool), 'chunk' (the formatted text or None), and 'message' (status).
              If 'found' is False, this relationship has no supporting evidence.
    """
    return _get_chunk_logic(entity_one, entity_two, edge_type, get_user_id(ctx))

@mcp.tool()
def explore_neighbors(entity_name: str, ctx: Context) -> str:
    """
    STEP 2: Explore the graph to find relationships connected to an Entity.
    
    Use this AFTER you have a resolved entity name.
    Returns a compact TOON-format view of connections grouped by relationship type.
    
    Args:
        entity_name (str): The canonical entity name (e.g., "Alphabet Inc.").
        ctx (Context): The request context (Injected automatically).

    Returns:
        str: TOON-format response with outgoing/incoming relationships grouped by type.
             Example: "outgoing:\\n  INVESTED[2]: Waymo,DeepMind\\nincoming:\\n  HIRED[1]: Larry Page"
             If found is false, stop exploring - this entity has no connections.
    """
    return _explore_neighbors_logic(entity_name, get_user_id(ctx))

if __name__ == "__main__":
    mcp.run()
