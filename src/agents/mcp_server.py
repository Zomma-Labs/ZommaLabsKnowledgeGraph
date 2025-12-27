"""
ZommaLabs Knowledge Graph MCP Server
====================================

FastMCP server exposing graph retrieval tools for AI agents.

Tools:
    - resolve_entity_or_topic: Semantic search to find exact entity/topic names
    - get_entity_info: Get entity metadata (summary, type, description)
    - explore_neighbors: Discover relationships connected to an entity
    - get_chunk: Retrieve source text evidence for a specific relationship
    - get_chunks: Batch retrieve multiple chunks at once
    - get_chunks_by_edge: Wildcard search - get all chunks for entity + edge type
    - think: Reasoning tool to analyze evidence before answering (prevents hallucination)

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

## Graph Retrieval Workflow

To answer a user question effectively, follow these steps:

### STEP 1: Resolve Entities (`resolve_entity_or_topic`)
- **Goal**: Map vague user terms (e.g., "Google", "wage pressure") to exact node names.
- **Why**: The graph requires exact matches. "Alphabet" won't work if the node is "Alphabet Inc.".
- **Example**: `resolve_entity_or_topic(query="Google", node_type="Entity")` → ["Alphabet Inc."]

### STEP 1b (Optional): Get Entity Info (`get_entity_info`)
- **Goal**: Get descriptive information about an entity (summary, type, what it is).
- **When**: Use this for "What is X?" questions or when you need entity descriptions.
- **Example**: `get_entity_info(entity_name="Alphabet Inc.")` → {"summary": "American multinational technology conglomerate...", "entity_type": "Organization"}

### STEP 2: Explore Relationships (`explore_neighbors`)
- **Goal**: Discover what connects to your resolved entities AND what edge types are available.
- **Why**: You don't know the edge types or connected nodes yet. This gives you a map.
- **Output**: Shows all edge types grouped by direction (outgoing/incoming) with connected entities.
- **Example**: `explore_neighbors(entity_name="Alphabet Inc.")` →
  ```
  outgoing:
    ACQUIRED[5]: Waymo,DeepMind,Wing...
    REACHED_MARKET_CAP[3]: $1 Trillion,$2 Trillion,$3 Trillion
  incoming:
    FOUNDED_BY[2]: Larry Page,Sergey Brin
  ```
- **IMPORTANT**: Use the edge types from this output for Step 3. If you see `REACHED_MARKET_CAP[3]`, you can use that edge type in `get_chunks_by_edge`.

### STEP 3: Retrieve Evidence (`get_chunk`, `get_chunks`, or `get_chunks_by_edge`)
- **Goal**: Get the actual source text (chunk) that justifies a specific relationship.
- **Why**: The edge tells you THAT something happened; the chunk tells you HOW, WHEN, and WHY.
- **Single**: `get_chunk(entity_one="Alphabet Inc.", entity_two="Waymo", edge_type="INVESTED")`
- **Batch**: `get_chunks(relationships=[["Alphabet Inc.", "INVESTED", "Waymo"], ["Sundar Pichai", "LEADS", "Google"]])` - PREFERRED when exploring multiple relationships!
- **Wildcard**: `get_chunks_by_edge(entity_name="Alphabet Inc.", edge_type="REACHED_MARKET_CAP")` - Use when you know the entity and edge type but NOT the other entity. Returns ALL matching chunks.

## Important Notes
- All queries are automatically scoped to your authorized tenant (data isolation).
- Always use exact entity names returned by resolve_entity_or_topic.
- If you can't find an entity, try different search terms or check for spelling variations.
- YOU MUST CALL RETRIEVE EVIDENCE (`get_chunk`, `get_chunks`, or `get_chunks_by_edge`) TO GET THE ACTUAL SOURCE TEXT TO ANSWER THE QUESTION
- DO NOT JUST ANSWER THE QUESTION BASED ON THE RELATIONSHIPS YOU DISCOVERED IN STEP 2
- PREFER `get_chunks` (batch) when you need to check multiple relationships - it's more efficient!
- USE `get_chunks_by_edge` when you see a relevant edge type in explore_neighbors but don't know the exact target entity

## Recommended Workflow Pattern

1. `resolve_entity_or_topic("Alphabet")` → Get exact name: "Alphabet Inc."
2. `explore_neighbors("Alphabet Inc.")` → See available edges:
   - outgoing: ACQUIRED[5], REACHED_MARKET_CAP[3], DECLARED_DIVIDEND[1]
3. Pick the relevant edge type and retrieve chunks:
   - If you need ALL market cap events: `get_chunks_by_edge("Alphabet Inc.", "REACHED_MARKET_CAP")`
   - If you need a specific one: `get_chunk("Alphabet Inc.", "$2 Trillion", "REACHED_MARKET_CAP")`
4. **REQUIRED: Call `think` to analyze the evidence** before answering:
   - List the specific facts found in the chunks
   - Quote relevant text that answers the question
   - Identify chunk ID for citation
5. Provide your final answer with proper citations.

## IMPORTANT: Using the Think Tool

After retrieving chunks, you MUST call the `think` tool to analyze the evidence:

```
think(thought="Analyzing the retrieved chunk for the question 'When did Alphabet reach $2T market cap?':
1. Found fact: 'Alphabet reached a $2 trillion market capitalization on January 23, 2024'
2. Chunk ID: abc123, Document: alphabet_10k_2024
3. This directly answers the question.
Citation: [DOC: alphabet_10k_2024, CHUNK: abc123]")
```

This prevents hallucination by forcing explicit fact extraction before answering.
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
        
        if score < 0.5:
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


def _get_chunks_logic(relationships: list[list[str]], user_id: str) -> dict:
    """
    Batch logic for getting multiple chunks at once.
    
    Args:
        relationships: List of tuples (entity_one, edge_type, entity_two)
        user_id: The tenant ID for scoping
    
    Returns:
        dict with found_count, total_requested, and results array
    """
    # Cap at 8 to prevent response explosion
    MAX_BATCH_SIZE = 8
    if len(relationships) > MAX_BATCH_SIZE:
        relationships = relationships[:MAX_BATCH_SIZE]
    
    results = []
    found_count = 0
    
    for rel in relationships:
        if len(rel) != 3:
            results.append({
                "relationship": str(rel),
                "found": False,
                "message": "Invalid tuple format. Expected (entity_one, edge_type, entity_two)"
            })
            continue
            
        entity_one, edge_type, entity_two = rel
        
        # Reuse existing single-chunk logic
        single_result = _get_chunk_logic(entity_one, entity_two, edge_type, user_id)
        
        if single_result.get("found"):
            found_count += 1
            results.append({
                "relationship": f"({entity_one}, {edge_type}, {entity_two})",
                "found": True,
                "chunk": single_result["chunk"]
            })
        else:
            results.append({
                "relationship": f"({entity_one}, {edge_type}, {entity_two})",
                "found": False,
                "message": single_result.get("message", "No evidence found")
            })
    
    return {
        "found_count": found_count,
        "total_requested": len(relationships),
        "results": results,
        "message": f"Found evidence for {found_count}/{len(relationships)} relationships."
    }


def _get_chunks_by_edge_logic(entity_name: str, edge_type: str, user_id: str, direction: str = "both") -> dict:
    """
    Get all chunks for a given entity and edge type, without needing to know the other entity.

    Args:
        entity_name: The entity to search from
        edge_type: The relationship type (e.g., "REACHED_MARKET_CAP", "ACQUIRED")
        user_id: Tenant ID
        direction: "outgoing" (entity is subject), "incoming" (entity is object), or "both"

    Returns:
        dict with found chunks and their connected entities
    """
    services = get_services()
    results = []

    # Query for outgoing relationships (entity is subject)
    if direction in ("outgoing", "both"):
        query_out = """
        MATCH (e1)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(e2)
        WHERE (e1:EntityNode OR e1:TopicNode)
          AND (e2:EntityNode OR e2:TopicNode)
          AND e1.name = $name
          AND e1.group_id = $uid
          AND e2.group_id = $uid
          AND type(r1) = $edge_type
          AND r1.fact_id = r2.fact_id
        RETURN e1.name as subject, type(r1) as edge, e2.name as object,
               c.uuid as chunk_id, c.content as content, c.header_path as header, c.doc_id as doc_id
        LIMIT 10
        """
        rows = services.neo4j.query(query_out, {
            "name": entity_name,
            "edge_type": edge_type,
            "uid": user_id
        })
        for row in rows:
            results.append({
                "direction": "outgoing",
                "subject": row["subject"],
                "edge": row["edge"],
                "object": row["object"],
                "chunk_id": row["chunk_id"],
                "header": row.get("header", "N/A"),
                "content": row.get("content", "")
            })

    # Query for incoming relationships (entity is object)
    if direction in ("incoming", "both"):
        query_in = """
        MATCH (e1)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(e2)
        WHERE (e1:EntityNode OR e1:TopicNode)
          AND (e2:EntityNode OR e2:TopicNode)
          AND e2.name = $name
          AND e1.group_id = $uid
          AND e2.group_id = $uid
          AND type(r1) = $edge_type
          AND r1.fact_id = r2.fact_id
        RETURN e1.name as subject, type(r1) as edge, e2.name as object,
               c.uuid as chunk_id, c.content as content, c.header_path as header, c.doc_id as doc_id
        LIMIT 10
        """
        rows = services.neo4j.query(query_in, {
            "name": entity_name,
            "edge_type": edge_type,
            "uid": user_id
        })
        for row in rows:
            # Avoid duplicates if same chunk found in both directions
            chunk_id = row["chunk_id"]
            if not any(r["chunk_id"] == chunk_id for r in results):
                results.append({
                    "direction": "incoming",
                    "subject": row["subject"],
                    "edge": row["edge"],
                    "object": row["object"],
                    "chunk_id": row["chunk_id"],
                    "header": row.get("header", "N/A"),
                    "content": row.get("content", "")
                })

    if not results:
        return {
            "found": False,
            "count": 0,
            "results": [],
            "message": f"No relationships found for '{entity_name}' with edge type [{edge_type}]. Try explore_neighbors first to see available edge types."
        }

    # Format chunks for output
    formatted_results = []
    for r in results:
        formatted_results.append({
            "relationship": f"{r['subject']} -[{r['edge']}]-> {r['object']}",
            "direction": r["direction"],
            "chunk": f'''"""
DOCUMENT: {r.get('doc_id', 'N/A')}
CHUNK_id: {r['chunk_id']}
Header: {r['header']}

{r['content']}
"""'''
        })

    return {
        "found": True,
        "count": len(formatted_results),
        "results": formatted_results,
        "message": f"Found {len(formatted_results)} chunk(s) for '{entity_name}' with edge [{edge_type}]."
    }


def _get_entity_info_logic(entity_name: str, user_id: str) -> dict:
    """
    Core logic for getting entity metadata including summary and type.
    """
    services = get_services()

    query = """
    MATCH (e {name: $name, group_id: $uid})
    WHERE e:EntityNode OR e:TopicNode
    RETURN e.name as name,
           e.summary as summary,
           e.entity_type as entity_type,
           labels(e) as labels
    LIMIT 1
    """

    results = services.neo4j.query(query, {"name": entity_name, "uid": user_id})

    if not results:
        return {
            "found": False,
            "message": f"Entity '{entity_name}' not found. Use resolve_entity_or_topic first to get exact names."
        }

    row = results[0]
    labels = row.get('labels', [])
    node_type = "Topic" if "TopicNode" in labels else "Entity"

    return {
        "found": True,
        "name": row.get('name'),
        "type": node_type,
        "entity_type": row.get('entity_type'),  # e.g., "Organization", "Person", "Place"
        "summary": row.get('summary') or "No summary available for this entity.",
        "message": "Entity information retrieved successfully."
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
def get_chunks(relationships: list[list[str]], ctx: Context) -> dict:
    """
    STEP 3 (BATCH): Retrieve multiple text chunks (evidence) for several relationships at once.
    
    Use this when you want to explore multiple potential relationships efficiently.
    This is PREFERRED over calling get_chunk multiple times - it's faster and uses fewer tokens.
    
    Args:
        relationships: A list of 3-element lists, each in format [entity_one, edge_type, entity_two].
                       Example: [["Alphabet Inc.", "INVESTED", "Waymo"], ["Google", "HIRED", "Sundar Pichai"]]
                       Maximum 8 relationships per call.
        ctx (Context): The request context (Injected automatically).

    Returns:
        dict: Contains:
            - found_count (int): Number of relationships with evidence
            - total_requested (int): Total relationships queried
            - results (list): Array of results, each with 'relationship', 'found', and 'chunk' or 'message'
            - message (str): Summary status
    """
    return _get_chunks_logic(relationships, get_user_id(ctx))

@mcp.tool()
def get_chunks_by_edge(entity_name: str, edge_type: str, ctx: Context, direction: str = "both") -> dict:
    """
    STEP 3 (WILDCARD): Retrieve all chunks for an entity with a specific edge type.

    Use this when you know the entity and relationship type but NOT the other entity.
    For example: "Find all chunks about Alphabet reaching market cap milestones"

    Args:
        entity_name (str): The entity name (e.g., "Alphabet Inc.").
        edge_type (str): The relationship type (e.g., "REACHED_MARKET_CAP", "ACQUIRED").
                         Use explore_neighbors first to see available edge types.
        ctx (Context): The request context (Injected automatically).
        direction (str): Where to search - "outgoing" (entity is subject),
                        "incoming" (entity is object), or "both" (default).

    Returns:
        dict: Contains:
            - found (bool): Whether any chunks were found
            - count (int): Number of chunks found
            - results (list): Array of {relationship, direction, chunk}
            - message (str): Summary status

    Example:
        get_chunks_by_edge("Alphabet Inc.", "REACHED_MARKET_CAP")
        → Returns all chunks about Alphabet reaching market cap milestones ($1T, $2T, $3T)
    """
    return _get_chunks_by_edge_logic(entity_name, edge_type, get_user_id(ctx), direction)

@mcp.tool()
def get_entity_info(entity_name: str, ctx: Context) -> dict:
    """
    Get detailed information about an entity including its summary and type.

    Use this tool when you need to answer "What is X?" questions or need descriptive
    information about an entity beyond just its relationships.

    Args:
        entity_name (str): The exact entity name (use resolve_entity_or_topic first if needed).
        ctx (Context): The request context (Injected automatically).

    Returns:
        dict: Contains:
            - found (bool): Whether the entity exists
            - name (str): The canonical entity name
            - type (str): "Entity" or "Topic"
            - entity_type (str): Specific type like "Organization", "Person", "Place"
            - summary (str): A description of the entity
            - message (str): Status message
    """
    return _get_entity_info_logic(entity_name, get_user_id(ctx))


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


@mcp.tool()
def think(thought: str) -> str:
    """
    Use this tool to think through the retrieved evidence before answering.

    Call this tool AFTER retrieving chunks to:
    1. List the specific facts found in the evidence
    2. Check if those facts answer the user's question
    3. Identify what citation to use

    This tool does not retrieve new information - it just provides space for reasoning.

    Args:
        thought (str): Your analysis of the retrieved evidence and reasoning about the answer.

    Returns:
        str: Acknowledgment that the thought was recorded.
    """
    return "Thought recorded. Now provide your final answer based ONLY on the facts you identified above."


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ZommaGraph MCP Server")
    parser.add_argument("--sse", action="store_true", help="Run as SSE server instead of stdio")
    parser.add_argument("--host", default="127.0.0.1", help="Host for SSE server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port for SSE server (default: 8765)")
    args = parser.parse_args()
    
    if args.sse:
        print(f"Starting ZommaGraph MCP Server (SSE) on http://{args.host}:{args.port}")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run()  # Default: stdio
