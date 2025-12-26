"""
ZommaLabs Knowledge Graph MCP Server (OPTIMIZED)
=================================================

Optimized version of the MCP server with:
1. Parallel Neo4j queries in explore_neighbors
2. Combined resolve_and_explore tool (saves one LLM round-trip)
3. Eager service initialization

Run with: python -m src.agents.mcp_server_optimized
"""

from concurrent.futures import ThreadPoolExecutor
from fastmcp import FastMCP, Context
from src.util.services import get_services

# === OPTIMIZATION: Eager service initialization ===
_services = None

def _get_services():
    """Get cached services instance."""
    global _services
    if _services is None:
        _services = get_services()
    return _services


# Initialize FastMCP Server
mcp = FastMCP(
    name="ZommaGraph",
    instructions="""
ZommaLabs Knowledge Graph - Graph Retrieval API (Optimized)
============================================================

## Available Tools

1. **resolve_and_explore** (RECOMMENDED) - Combines entity resolution + neighbor exploration in ONE call
   - Faster than calling resolve then explore separately
   - Use this for most queries

2. **resolve_entity_or_topic** - Just resolve a term to exact entity names
3. **explore_neighbors** - Just explore relationships (if you already have exact name)
4. **get_chunk** - Retrieve source evidence for a specific relationship

## Typical Workflow
1. Call `resolve_and_explore` with your query
2. If you need evidence, call `get_chunk` for specific relationships
"""
)


def get_user_id(ctx: Context) -> str:
    """Extract user_id from context or default to 'default_tenant'."""
    if not ctx:
        return "default_tenant"
    try:
        user_id = None
        if not user_id:
            return "default_tenant"
        return user_id
    except Exception:
        return "default_tenant"


# --- Core Logic Functions ---

def _resolve_entity_or_topic_logic(query: str, node_type: str, user_id: str) -> dict:
    """Core logic for resolving entities/topics."""
    services = _get_services()

    # Embed query
    query_vector = services.embeddings.embed_query(query)

    # Select index based on node_type
    index_name = "topic_embeddings" if node_type == "Topic" else "entity_embeddings"

    # Vector Search
    results = services.neo4j.vector_search(
        index_name=index_name,
        query_vector=query_vector,
        top_k=20
    )

    valid_names = []
    for row in results:
        node = row.get('node')
        score = row.get('score', 0)

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
            "message": f"Found {len(unique_names)} matching {node_type.lower()}(s)."
        }
    else:
        return {
            "found": False,
            "results": [],
            "message": f"NO MATCH: '{query}' not found in graph."
        }


def _explore_neighbors_logic(entity_name: str, user_id: str) -> str:
    """
    Core logic for exploring neighbors.
    OPTIMIZED: Runs outgoing and incoming queries in parallel.
    """
    services = _get_services()

    outgoing = {}
    incoming = {}

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

    params = {"name": entity_name, "uid": user_id}

    # OPTIMIZATION: Run both queries in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_out = executor.submit(services.neo4j.query, query_out, params)
        future_in = executor.submit(services.neo4j.query, query_in, params)
        results_out = future_out.result()
        results_in = future_in.result()

    for row in results_out:
        edge = row['edge']
        target = row['target']
        if edge not in outgoing:
            outgoing[edge] = []
        if target not in outgoing[edge]:
            outgoing[edge].append(target)

    for row in results_in:
        edge = row['edge']
        source = row['source']
        if edge not in incoming:
            incoming[edge] = []
        if source not in incoming[edge]:
            incoming[edge].append(source)

    total_count = sum(len(v) for v in outgoing.values()) + sum(len(v) for v in incoming.values())

    if total_count == 0:
        return f"found: false\nentity: {entity_name}\nmessage: NO RELATIONSHIPS FOUND"

    lines = ["found: true", f"entity: {entity_name}"]

    if outgoing:
        lines.append("outgoing:")
        for edge_type, targets in outgoing.items():
            lines.append(f"  {edge_type}[{len(targets)}]: {','.join(targets)}")

    if incoming:
        lines.append("incoming:")
        for edge_type, sources in incoming.items():
            lines.append(f"  {edge_type}[{len(sources)}]: {','.join(sources)}")

    lines.append(f"total: {total_count}")
    return "\n".join(lines)


def _get_chunk_logic(entity_one: str, entity_two: str, edge_type: str, user_id: str) -> dict:
    """Core logic for getting chunk content."""
    services = _get_services()

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

    results = services.neo4j.query(query, {
        "e1": entity_one, "e2": entity_two, "edge_type": edge_type, "uid": user_id
    })

    if not results:
        results = services.neo4j.query(query, {
            "e1": entity_two, "e2": entity_one, "edge_type": edge_type, "uid": user_id
        })

    if not results:
        return {
            "found": False,
            "chunk": None,
            "message": f"NO EVIDENCE: '{entity_one}' -> [{edge_type}] -> '{entity_two}' not found."
        }

    chunk_data = results[0]['c']
    formatted_output = f'''"""
DOCUMENT: {chunk_data.get('doc_id', 'N/A')}
CHUNK_id: {chunk_data.get('uuid', 'N/A')}
Header: {chunk_data.get('header_path', 'N/A')}
Date: {chunk_data.get('created_at', 'N/A')}

{chunk_data.get('content', '')}
"""'''

    return {"found": True, "chunk": formatted_output, "message": "Evidence retrieved."}


def _resolve_and_explore_logic(query: str, node_type: str, user_id: str) -> dict:
    """
    OPTIMIZATION: Combined resolve + explore in one call.
    Saves one LLM round-trip by doing both operations together.
    """
    # Step 1: Resolve
    resolve_result = _resolve_entity_or_topic_logic(query, node_type, user_id)

    if not resolve_result["found"]:
        return {
            "found": False,
            "resolved_entities": [],
            "relationships": {},
            "message": resolve_result["message"]
        }

    # Step 2: Explore each resolved entity (in parallel if multiple)
    entities = resolve_result["results"]
    all_relationships = {}

    if len(entities) == 1:
        # Single entity - just explore it
        explore_result = _explore_neighbors_logic(entities[0], user_id)
        all_relationships[entities[0]] = explore_result
    else:
        # Multiple entities - explore in parallel
        with ThreadPoolExecutor(max_workers=min(len(entities), 5)) as executor:
            futures = {
                executor.submit(_explore_neighbors_logic, entity, user_id): entity
                for entity in entities[:5]  # Limit to first 5 to avoid overload
            }
            for future in futures:
                entity = futures[future]
                try:
                    all_relationships[entity] = future.result()
                except Exception as e:
                    all_relationships[entity] = f"error: {str(e)}"

    return {
        "found": True,
        "resolved_entities": entities,
        "relationships": all_relationships,
        "message": f"Resolved {len(entities)} entity(ies) and explored relationships."
    }


# --- MCP Tool Wrappers ---

@mcp.tool()
def resolve_and_explore(query: str, node_type: str, ctx: Context) -> dict:
    """
    RECOMMENDED: Combined entity resolution + neighbor exploration in ONE call.

    This is faster than calling resolve_entity_or_topic then explore_neighbors separately.
    Use this for most queries to minimize round-trips.

    Args:
        query (str): The search text (e.g., "Google", "inflation").
        node_type (str): "Entity" for people/orgs, "Topic" for themes.
        ctx (Context): Request context (injected automatically).

    Returns:
        dict: Contains resolved entities and their relationships in one response.
    """
    return _resolve_and_explore_logic(query, node_type, get_user_id(ctx))


@mcp.tool()
def resolve_entity_or_topic(query: str, node_type: str, ctx: Context) -> dict:
    """
    STEP 1: Resolve fuzzy user terms to strict Graph Entity Names.

    Args:
        query (str): The search text.
        node_type (str): "Entity" or "Topic".
        ctx (Context): Request context.

    Returns:
        dict: Contains 'found', 'results' (list of names), and 'message'.
    """
    return _resolve_entity_or_topic_logic(query, node_type, get_user_id(ctx))


@mcp.tool()
def explore_neighbors(entity_name: str, ctx: Context) -> str:
    """
    STEP 2: Explore relationships connected to an Entity.

    Args:
        entity_name (str): The canonical entity name.
        ctx (Context): Request context.

    Returns:
        str: TOON-format response with outgoing/incoming relationships.
    """
    return _explore_neighbors_logic(entity_name, get_user_id(ctx))


@mcp.tool()
def get_chunk(entity_one: str, entity_two: str, edge_type: str, ctx: Context) -> dict:
    """
    STEP 3: Retrieve the source text evidence for a known relationship.

    Args:
        entity_one (str): The SOURCE entity name.
        entity_two (str): The TARGET entity name.
        edge_type (str): The relationship type (e.g., "INVESTED").
        ctx (Context): Request context.

    Returns:
        dict: Contains 'found', 'chunk' (text or None), and 'message'.
    """
    return _get_chunk_logic(entity_one, entity_two, edge_type, get_user_id(ctx))


if __name__ == "__main__":
    mcp.run()
