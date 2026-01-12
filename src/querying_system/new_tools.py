"""
General-Purpose Tools to Improve KG Retrieval
==============================================

These tools address retrieval gaps without being domain-specific.
"""

from src.util.services import get_services
from src.util.llm_client import get_dedup_embeddings, LLMClient


def _search_chunks_filtered(
    query: str,
    user_id: str,
    header_contains: str = "",
    header_not_contains: str = "",
    top_k: int = 10
) -> dict:
    """
    Semantic search on chunks with header filtering.

    This is the key missing capability - searching with structural filters.

    Args:
        query: Semantic search query
        user_id: Tenant ID
        header_contains: Only include chunks where header_path contains this string
        header_not_contains: Exclude chunks where header_path contains this string
        top_k: Number of results

    Examples:
        - header_contains="Summary" → Get summary sections
        - header_not_contains="Federal Reserve Bank" → Get national sections only
        - header_contains="Labor Markets" → Get labor market sections across all districts
    """
    services = get_services()

    # Build the filter clause
    filters = ["c.group_id = $uid"]
    if header_contains:
        filters.append("c.header_path CONTAINS $header_contains")
    if header_not_contains:
        filters.append("NOT c.header_path CONTAINS $header_not_contains")

    filter_clause = " AND ".join(filters)

    # Get query embedding
    embeddings = LLMClient.get_embeddings(model="voyage-3-large")
    query_vector = embeddings.embed_query(query)

    # Vector search with filters
    cypher = f"""
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k * 3, $vec)
        YIELD node as c, score
        WHERE {filter_clause} AND score > 0.3
        RETURN c.uuid as chunk_id,
               c.header_path as header,
               c.content as content,
               score
        ORDER BY score DESC
        LIMIT $top_k
    """

    results = services.neo4j.query(cypher, {
        "vec": query_vector,
        "uid": user_id,
        "header_contains": header_contains,
        "header_not_contains": header_not_contains,
        "top_k": top_k
    })

    if not results:
        return {
            "found": False,
            "results": [],
            "message": f"No chunks found matching query with filters"
        }

    formatted = []
    for row in results:
        formatted.append({
            "header": row["header"],
            "content": row["content"],
            "score": round(row["score"], 3),
            "chunk_id": row["chunk_id"]
        })

    return {
        "found": True,
        "count": len(formatted),
        "results": formatted
    }


def _search_facts_grouped(
    query: str,
    user_id: str,
    group_by: str = "header_prefix",
    top_k: int = 20
) -> dict:
    """
    Search facts and group results by a field.

    Useful for "which X have Y" questions - finds all matches grouped by source.

    Args:
        query: Semantic search query
        user_id: Tenant ID
        group_by: How to group results
            - "header_prefix": Group by first part of header (e.g., district name)
            - "document": Group by source document
            - "entity": Group by subject entity
        top_k: Total results before grouping

    Returns:
        Results grouped by the specified field with evidence for each group
    """
    services = get_services()
    embeddings = LLMClient.get_embeddings(model="voyage-3-large")
    query_vector = embeddings.embed_query(query)

    # Determine grouping expression
    if group_by == "header_prefix":
        group_expr = "split(c.header_path, ' > ')[0]"
    elif group_by == "document":
        group_expr = "c.doc_id"
    elif group_by == "entity":
        group_expr = "subj.name"
    else:
        group_expr = "split(c.header_path, ' > ')[0]"

    cypher = f"""
        CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
        YIELD node, score
        WHERE node.group_id = $uid AND score > 0.4

        // Find the chunk and entities
        MATCH (subj)-[r1 {{fact_id: node.uuid}}]->(c:EpisodicNode {{group_id: $uid}})
        WHERE (subj:EntityNode OR subj:TopicNode)

        WITH node, score, c, subj, {group_expr} as group_key

        RETURN group_key,
               collect({{
                   fact: node.content,
                   score: score,
                   header: c.header_path,
                   subject: subj.name
               }})[0..3] as evidence,
               max(score) as best_score,
               count(*) as match_count
        ORDER BY best_score DESC
    """

    results = services.neo4j.query(cypher, {
        "vec": query_vector,
        "uid": user_id,
        "top_k": top_k
    })

    if not results:
        return {
            "found": False,
            "groups": [],
            "message": f"No facts found matching '{query}'"
        }

    groups = []
    for row in results:
        groups.append({
            "group": row["group_key"],
            "match_count": row["match_count"],
            "best_score": round(row["best_score"], 3),
            "evidence": row["evidence"]
        })

    return {
        "found": True,
        "query": query,
        "group_by": group_by,
        "count": len(groups),
        "groups": groups
    }


def _list_unique_headers(user_id: str, prefix: str = "") -> dict:
    """
    List unique header paths/sections in the knowledge graph.

    Helps researcher understand document structure before searching.

    Args:
        user_id: Tenant ID
        prefix: Optional prefix to filter headers

    Returns:
        List of unique header paths
    """
    services = get_services()

    if prefix:
        cypher = """
            MATCH (c:EpisodicNode {group_id: $uid})
            WHERE c.header_path STARTS WITH $prefix
            RETURN DISTINCT c.header_path as header
            ORDER BY header
            LIMIT 50
        """
        results = services.neo4j.query(cypher, {"uid": user_id, "prefix": prefix})
    else:
        cypher = """
            MATCH (c:EpisodicNode {group_id: $uid})
            RETURN DISTINCT c.header_path as header
            ORDER BY header
            LIMIT 100
        """
        results = services.neo4j.query(cypher, {"uid": user_id})

    headers = [row["header"] for row in results]

    # Also extract unique top-level sections
    top_level = list(set(h.split(" > ")[0] for h in headers if " > " in h))

    return {
        "found": len(headers) > 0,
        "headers": headers,
        "top_level_sections": sorted(top_level),
        "count": len(headers)
    }
