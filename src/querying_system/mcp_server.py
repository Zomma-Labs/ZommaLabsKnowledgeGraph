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


import atexit
import signal
import sys

from fastmcp import FastMCP, Context
from src.util.services import get_services
from src.util.fact_vector_store import get_fact_store, FactVectorStore
from src.util.llm_client import get_dedup_embeddings


def cleanup():
    """Release Qdrant lock on shutdown."""
    FactVectorStore.reset()


# Register cleanup for normal exit
atexit.register(cleanup)


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM by cleaning up and exiting."""
    cleanup()
    sys.exit(0)


# Only register signal handlers in main thread (avoids errors when imported)
import threading
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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
- **Goal**: Map vague user terms (e.g., "tech company", "wage pressure") to exact node names.
- **Why**: The graph requires exact matches. A partial name won't work if the node uses the full legal name.
- **Example**: `resolve_entity_or_topic(query="[company name]", node_type="Entity")` → ["[Exact Company Name]"]

### STEP 1b (Optional): Get Entity Info (`get_entity_info`)
- **Goal**: Get descriptive information about an entity (summary, type, what it is).
- **When**: Use this for "What is X?" questions or when you need entity descriptions.
- **Example**: `get_entity_info(entity_name="[Entity Name]")` → {"summary": "Description of the entity...", "entity_type": "Organization"}

### STEP 2: Explore Relationships (`explore_neighbors`)
- **Goal**: Discover what connects to your resolved entities AND what edge types are available.
- **Why**: You don't know the edge types or connected nodes yet. This gives you a map.
- **Output**: Returns a list of edges with temporal annotations (dates help you filter temporally).
- **query_hint is REQUIRED**: Pass `""` (empty string) for all edges, or a search query to filter semantically.
  - Example: `explore_neighbors(entity_name="Alphabet Inc.", query_hint="")` → All edges with dates
  - Example: `explore_neighbors(entity_name="Alphabet Inc.", query_hint="market cap milestones")` → Filtered + ranked
- **Format** (query_hint=""):
  ```
  edges:
  - ACQUIRED → Google Nest | 2018-02
  - REACHED_MILESTONE → Market Valuation | doc: 2024-10-15
  - FOUNDED_BY ← Larry Page | 1998
  ```
- **Format** (with query_hint): Same format with relevance scores: `(0.92)`
- **Temporal Annotations**: Pipe separator shows date - either from text or `doc: YYYY-MM-DD` (document date).
- **IMPORTANT**: Use the edge types from this output for Step 3.

### STEP 3: Retrieve Evidence (`get_chunk`, `get_chunks`, or `get_chunks_by_edge`)
- **Goal**: Get the actual source text (chunk) that justifies a specific relationship.
- **Why**: The edge tells you THAT something happened; the chunk tells you HOW, WHEN, and WHY.
- **Single**: `get_chunk(entity_one="[Entity1]", entity_two="[Entity2]", edge_type="[EDGE_TYPE]")`
- **Batch**: `get_chunks(relationships=[["[Entity1]", "[EDGE_TYPE]", "[Entity2]"], ...])` - PREFERRED when exploring multiple relationships!
- **Wildcard**: `get_chunks_by_edge(entity_name="[Entity]", edge_type="[EDGE_TYPE]")` - Use when you know the entity and edge type but NOT the other entity. Returns ALL matching chunks.

## Important Notes
- All queries are automatically scoped to your authorized tenant (data isolation).
- Always use exact entity names returned by resolve_entity_or_topic.
- If you can't find an entity, try different search terms or check for spelling variations.
- YOU MUST CALL RETRIEVE EVIDENCE (`get_chunk`, `get_chunks`, or `get_chunks_by_edge`) TO GET THE ACTUAL SOURCE TEXT TO ANSWER THE QUESTION
- DO NOT JUST ANSWER THE QUESTION BASED ON THE RELATIONSHIPS YOU DISCOVERED IN STEP 2
- PREFER `get_chunks` (batch) when you need to check multiple relationships - it's more efficient!
- USE `get_chunks_by_edge` when you see a relevant edge type in explore_neighbors but don't know the exact target entity

## Recommended Workflow Pattern

1. `resolve_entity_or_topic("[search term]")` → Get exact name: "[Exact Entity Name]"
2. `explore_neighbors("[Exact Entity Name]", query_hint="")` → See all edges with dates:
   - `- EDGE_TYPE → target [fact_date: ...]`
   - Use `query_hint="your query"` to filter and rank semantically
3. Pick the relevant edge type and retrieve chunks:
   - If you need ALL events of a type: `get_chunks_by_edge("[Entity]", "[EDGE_TYPE]")`
   - If you need a specific one: `get_chunk("[Entity1]", "[Entity2]", "[EDGE_TYPE]")`
4. **REQUIRED: Call `think` to analyze the evidence** before answering:
   - List the specific facts found in the chunks
   - Quote relevant text that answers the question
   - Identify chunk ID for citation
5. Provide your final answer with proper citations.

## IMPORTANT: Using the Think Tool

After retrieving chunks, you MUST call the `think` tool to analyze the evidence:

```
think(thought="Analyzing the retrieved chunk for the question '[user question]':
1. Found fact: '[specific fact from chunk]'
2. Chunk ID: [chunk_id], Document: [doc_name]
3. This directly answers the question.
Citation: [DOC: doc_name, CHUNK: chunk_id]")
```

This prevents hallucination by forcing explicit fact extraction before answering.
"""
)

def get_user_id(ctx: Context) -> str:
    """
    Helper to extract user_id from Auth headers or Session.
    For local testing/development, it defaults to 'default' if no auth is present.
    """
    if not ctx:
        return "default"

    # Try to extract from request context (FastMCP specific) or fallback
    # Note: FastMCP Context object structure might vary, so we check standard places
    try:
        # Placeholder for actual auth extraction logic
        # user_id = ctx.meta.get("user_id")
        user_id = None
        if not user_id:
            # For this phase, we default to 'default' to match pipeline default
            return "default"
        return user_id
    except Exception:
        return "default"

# --- Core Logic Functions (Testable) ---

def _resolve_entity_or_topic_logic(query: str, node_type: str, user_id: str, context: str = "") -> dict:
    """
    Core logic for resolving entities/topics.

    Args:
        query: The search text (entity or topic name)
        node_type: "Entity" or "Topic"
        user_id: Tenant ID for scoping
        context: Optional description to improve matching. When provided, searches
                 the name+summary index. When empty, searches the name-only index.

    Returns a dict with:
        - found: bool indicating if any matches were found
        - results: list of matching entity/topic names
        - message: human-readable status message
    """
    services = get_services()

    # 1. Choose embedding text and index based on context
    if node_type == "Topic":
        # Topics only have one index
        embed_text = query
        index_name = "topic_embeddings"
    elif context:
        # With context: embed "name: context" and use name+summary index
        embed_text = f"{query}: {context}"
        index_name = "entity_name_embeddings"
    else:
        # Without context: embed just name and use name-only index
        embed_text = query
        index_name = "entity_name_only_embeddings"

    query_vector = services.embeddings.embed_query(embed_text)

    # 2. Vector Search
    results = services.neo4j.vector_search(
        index_name=index_name,
        query_vector=query_vector,
        top_k=20
    )
    
    # Use dict to deduplicate while preserving score-based ordering
    # (vector search returns results sorted by score, highest first)
    seen_names = {}

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
        if name and name not in seen_names:
            seen_names[name] = score

    # Preserve score-based ordering (first occurrence = highest score)
    unique_names = list(seen_names.keys())
    
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


def _explore_neighbors_logic(entity_name: str, user_id: str) -> str:
    """
    Core logic for exploring neighbors.
    Returns edge list format with temporal annotations.
    """
    services = get_services()

    edges = []  # List of edge dicts

    # 1. Outgoing relationships with date info
    query_out = """
    MATCH (Start)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(Target)
    WHERE (Start:EntityNode OR Start:TopicNode)
      AND (Target:EntityNode OR Target:TopicNode)
      AND Start.name = $name
      AND Start.group_id = $uid
      AND Target.group_id = $uid
      AND r1.fact_id = r2.fact_id
    OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)
    RETURN type(r1) as edge, Target.name as target, r1.date_context as fact_date, d.document_date as doc_date
    """
    results_out = services.neo4j.query(query_out, {"name": entity_name, "uid": user_id})
    for row in results_out:
        edges.append({
            "edge": row["edge"],
            "target": row["target"],
            "direction": "outgoing",
            "fact_date": row.get("fact_date"),
            "doc_date": row.get("doc_date")
        })

    # 2. Incoming relationships with date info
    query_in = """
    MATCH (Source)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(Target)
    WHERE (Source:EntityNode OR Source:TopicNode)
      AND (Target:EntityNode OR Target:TopicNode)
      AND Target.name = $name
      AND Source.group_id = $uid
      AND Target.group_id = $uid
      AND r1.fact_id = r2.fact_id
    OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)
    RETURN Source.name as source, type(r1) as edge, r1.date_context as fact_date, d.document_date as doc_date
    """
    results_in = services.neo4j.query(query_in, {"name": entity_name, "uid": user_id})
    for row in results_in:
        edges.append({
            "edge": row["edge"],
            "target": row["source"],
            "direction": "incoming",
            "fact_date": row.get("fact_date"),
            "doc_date": row.get("doc_date")
        })

    # Check if any relationships found
    if not edges:
        return f"found: false\nentity: {entity_name}\nmessage: NO RELATIONSHIPS FOUND - stop exploring this entity"

    # Build edge list format with temporal annotations
    lines = [
        "found: true",
        f"entity: {entity_name}",
        "edges:"
    ]

    # Deduplicate by (edge, target, direction) keeping first occurrence
    seen = set()
    for e in edges:
        key = (e["edge"], e["target"], e["direction"])
        if key in seen:
            continue
        seen.add(key)

        # Format temporal annotation with pipe separator
        fact_date = e.get("fact_date")
        doc_date = e.get("doc_date")
        # Skip unhelpful date contexts
        if fact_date and fact_date.strip() and "Document date" not in fact_date and "not specified" not in fact_date.lower():
            temporal = f" | {fact_date}"
        elif doc_date:
            temporal = f" | doc: {doc_date}"
        else:
            temporal = ""

        # Direction arrow
        arrow = "→" if e["direction"] == "outgoing" else "←"

        lines.append(f"- {e['edge']} {arrow} {e['target']}{temporal}")

    lines.append(f"total: {len(seen)}")
    return "\n".join(lines)


def _explore_neighbors_semantic_logic(entity_name: str, user_id: str, query_hint: str) -> str:
    """
    Semantic version of explore_neighbors that ranks facts by relevance to query_hint.

    Uses the Qdrant fact vector store to find facts where the entity is subject or object,
    ranked by semantic similarity to the query_hint.

    Args:
        entity_name: The entity to explore neighbors for
        user_id: Tenant ID for scoping
        query_hint: Natural language description of what kind of relationships to find

    Returns:
        Edge list format with temporal annotations and relevance scores
    """
    services = get_services()

    try:
        # Get embeddings for the query hint
        embeddings = get_dedup_embeddings()
        query_embedding = embeddings.embed_query(query_hint)

        # Search facts for this entity
        fact_store = get_fact_store()
        facts = fact_store.search_facts_for_entity(
            entity_name=entity_name,
            query_embedding=query_embedding,
            group_id=user_id,
            top_k=15
        )
    except RuntimeError as e:
        if "already accessed" in str(e):
            # Fall back to graph traversal if Qdrant is locked
            return _explore_neighbors_logic(entity_name, user_id)
        raise

    if not facts:
        return f"found: false\nentity: {entity_name}\nquery: \"{query_hint}\"\nmessage: NO MATCHING FACTS - no facts found for this entity matching your query"

    # Query Neo4j to get document dates for these facts
    fact_ids = [f["fact_id"] for f in facts]
    date_query = """
    UNWIND $fact_ids AS fid
    MATCH (c:EpisodicNode {group_id: $uid})<-[r {fact_id: fid}]-()
    OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)
    RETURN fid as fact_id, r.date_context as fact_date, d.document_date as doc_date
    """
    date_results = services.neo4j.query(date_query, {"fact_ids": fact_ids, "uid": user_id})
    date_map = {r["fact_id"]: {"fact_date": r.get("fact_date"), "doc_date": r.get("doc_date")} for r in date_results}

    # Build edge list with scores and temporal annotations
    lines = [
        "found: true",
        f"entity: {entity_name}",
        f"query: \"{query_hint}\"",
        "edges:"
    ]

    # Sort by score descending
    facts_sorted = sorted(facts, key=lambda f: f["score"], reverse=True)

    seen = set()
    for fact in facts_sorted:
        edge_type = fact["edge_type"]
        direction = fact["direction"]
        target = fact["object"] if direction == "outgoing" else fact["subject"]
        score = fact["score"]

        key = (edge_type, target, direction)
        if key in seen:
            continue
        seen.add(key)

        # Get temporal info with pipe separator
        dates = date_map.get(fact["fact_id"], {})
        fact_date = dates.get("fact_date")
        doc_date = dates.get("doc_date")

        # Skip unhelpful date contexts
        if fact_date and fact_date.strip() and "Document date" not in fact_date and "not specified" not in fact_date.lower():
            temporal = f" | {fact_date}"
        elif doc_date:
            temporal = f" | doc: {doc_date}"
        else:
            temporal = ""

        # Direction arrow
        arrow = "→" if direction == "outgoing" else "←"

        lines.append(f"- {edge_type} {arrow} {target}{temporal} ({score:.2f})")

    lines.append(f"total: {len(seen)}")
    return "\n".join(lines)


def _expand_query(query: str) -> list[str]:
    """
    Generate query variations for more robust search.
    Returns the original query plus 2-3 variations.
    """
    from src.util.llm_client import get_nano_llm
    from pydantic import BaseModel, Field
    from typing import List

    class QueryExpansion(BaseModel):
        variations: List[str] = Field(description="2-3 alternative phrasings of the query")

    try:
        llm = get_nano_llm().with_structured_output(QueryExpansion)
        result = llm.invoke(f"""Generate 2-3 alternative search queries for finding facts about:
"{query}"

Create variations that:
1. Use different keywords/synonyms
2. Rephrase the question as a statement
3. Extract key entities/concepts

Keep each variation concise (under 10 words).""")

        # Return original + variations, deduplicated
        all_queries = [query] + result.variations
        return list(dict.fromkeys(all_queries))[:4]  # Max 4 queries
    except Exception:
        # Fallback: just return original
        return [query]


def _search_relationships_logic(
    query: str,
    user_id: str,
    top_k: int = 10,
    date_from: str = None,
    date_to: str = None
) -> dict:
    """
    Search for relationships/facts by semantic similarity with auto query expansion.

    Use this when you don't have a specific entity to search from, but want to find
    facts matching a description (e.g., "shutdown subsidiary", "merged into").

    Args:
        query: Description of the relationship to search for
        user_id: Tenant ID for scoping
        top_k: Number of results to return
        date_from: Optional start date filter (YYYY-MM-DD format)
        date_to: Optional end date filter (YYYY-MM-DD format)

    Returns:
        dict with matching facts and their connected entities, sorted by date (newest first)
    """
    from src.util.llm_client import LLMClient
    services = get_services()

    # Use voyage-3-large for fact search (better general semantic matching)
    fact_embeddings = LLMClient.get_embeddings(model="voyage-3-large")

    # Expand query into variations for more robust search
    query_variations = _expand_query(query)

    # Search all variations and combine results
    all_results = []
    seen_facts_global = set()

    for q in query_variations:
        query_vector = fact_embeddings.embed_query(q)

        # Vector search on FactNode with document date info
        results = services.neo4j.query("""
            CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
            YIELD node, score
            WHERE node.group_id = $uid AND score > 0.3

            // Find the chunk and entities connected via this fact
            OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})-[r2 {fact_id: node.uuid}]->(obj)
            WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)

            // Get document date from parent document
            OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

            RETURN node.content as fact, score,
                   subj.name as subject, type(r1) as edge_type, obj.name as object,
                   c.uuid as chunk_id, c.header_path as header,
                   c.document_date as chunk_date, d.document_date as doc_date
            ORDER BY score DESC
        """, {"vec": query_vector, "uid": user_id, "top_k": top_k * 2})

        # Process results from this query variation
        for row in results:
            fact = row.get("fact")
            if not fact or fact in seen_facts_global:
                continue
            seen_facts_global.add(fact)

            # Get document date (prefer chunk date, fall back to doc date)
            doc_date = row.get("chunk_date") or row.get("doc_date")

            # Convert datetime to string if needed
            if doc_date and hasattr(doc_date, 'strftime'):
                doc_date = doc_date.strftime('%Y-%m-%d')
            elif doc_date and hasattr(doc_date, 'isoformat'):
                doc_date = doc_date.isoformat()[:10]

            # Apply date filtering if specified
            if date_from and doc_date and doc_date < date_from:
                continue
            if date_to and doc_date and doc_date > date_to:
                continue

            all_results.append({
                "fact": fact,
                "score": round(row.get("score", 0), 3),
                "subject": row.get("subject"),
                "edge_type": row.get("edge_type"),
                "object": row.get("object"),
                "chunk_id": row.get("chunk_id"),
                "header": row.get("header"),
                "document_date": doc_date
            })

    # Sort by score (best matches first), then by date
    all_results.sort(
        key=lambda x: (x.get("score", 0), x.get("document_date") or "0000-00-00"),
        reverse=True
    )

    # Take top_k results
    formatted_results = all_results[:top_k]

    if not formatted_results:
        return {
            "found": False,
            "results": [],
            "message": f"No relationships found matching '{query}' (searched {len(query_variations)} variations)"
        }

    return {
        "found": len(formatted_results) > 0,
        "count": len(formatted_results),
        "results": formatted_results,
        "message": f"Found {len(formatted_results)} relationships matching '{query}'"
    }


# --- MCP Tool Wrappers ---

@mcp.tool()
def resolve_entity_or_topic(query: str, node_type: str, ctx: Context, context: str = "") -> dict:
    """
    STEP 1: Resolve fuzzy user terms to strict Graph Entity Names.

    Use this tool FIRST. The graph requires exact node names.
    This tool performs a semantic vector search to find the closest matching nodes.

    Args:
        query (str): The search text (e.g., "Ruth Porat", "Alphabet").
        node_type (str): The type of node to find. Use "Entity" for people/orgs, "Topic" for themes.
        ctx (Context): The request context (Injected automatically).
        context (str): Optional description to improve matching accuracy.
                       Provide a brief description of the entity you're looking for.
                       Example: "President and Chief Investment Officer of Alphabet"
                       When provided, enables semantic matching against entity descriptions.
                       When omitted, matches directly against entity names.

    Returns:
        dict: Contains 'found' (bool), 'results' (list of names), and 'message' (status).
              If 'found' is False, do NOT search for variations - the entity is not in the graph.
    """
    return _resolve_entity_or_topic_logic(query, node_type, get_user_id(ctx), context)

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
def explore_neighbors(entity_name: str, query_hint: str, ctx: Context) -> str:
    """
    STEP 2: Explore the graph to find relationships connected to an Entity.

    Use this AFTER you have a resolved entity name.
    Returns a list of edges with temporal annotations (date_context or document_date).

    Args:
        entity_name (str): The canonical entity name (e.g., "Alphabet Inc.").
        query_hint (str): REQUIRED - Pass "" (empty string) for all edges, or a query string to filter.
                         Examples:
                         - "": Returns all edges with temporal annotations
                         - "market cap milestones": Semantic filter + ranking with scores
                         - "acquisitions": Filter to acquisition-related edges
        ctx (Context): The request context (Injected automatically).

    Returns:
        str: Edge list format with temporal annotations.
             Format: "- EDGE_TYPE → target | date (score)" or "| doc: YYYY-MM-DD"
             If found is false, stop exploring - this entity has no connections.
    """
    user_id = get_user_id(ctx)

    # If query_hint provided (not empty), use semantic search for better ranking
    if query_hint and query_hint.strip():
        return _explore_neighbors_semantic_logic(entity_name, user_id, query_hint)

    # Otherwise, use the graph traversal (returns all edges)
    return _explore_neighbors_logic(entity_name, user_id)


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


@mcp.tool()
def search_relationships(
    query: str,
    ctx: Context,
    date_from: str = None,
    date_to: str = None
) -> dict:
    """
    Search for relationships/facts directly by semantic similarity on fact content.

    WHEN TO USE:
    Use this when the question asks about an event/action but doesn't name specific entities.
    This tool searches the fact text directly, bypassing entity-based navigation.

    HOW IT WORKS:
    - Embeds your query and searches against all fact embeddings
    - Returns matching facts with their subject, edge_type, object, and chunk_id
    - Results include document_date and are sorted by date (newest first)
    - Scores indicate semantic similarity (higher = better match)

    TEMPORAL BEHAVIOR:
    - Results are sorted by date (newest first) by default
    - Each result includes a 'document_date' field showing when the source was written
    - Use date_from/date_to to filter by time period when needed
    - If no date filter is specified, you get all results and can see the dates

    SEARCH STRATEGY:
    - Use descriptive phrases that match how facts are written
    - Include key action words and context from the question
    - Results show the entities involved, which you can then explore further

    Args:
        query (str): Descriptive phrase matching the relationship/event you're looking for.
        ctx (Context): The request context (Injected automatically).
        date_from (str): Optional start date filter (YYYY-MM-DD). Only return facts from this date onwards.
        date_to (str): Optional end date filter (YYYY-MM-DD). Only return facts up to this date.

    Returns:
        dict: Contains 'found' (bool), 'count' (int), 'results' (list).
              Each result: {fact, score, subject, edge_type, object, chunk_id, header, document_date}
    """
    return _search_relationships_logic(query, get_user_id(ctx), date_from=date_from, date_to=date_to)


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
