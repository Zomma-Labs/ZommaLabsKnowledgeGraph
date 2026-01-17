"""
Proposed New Tools to Address KG Query Weaknesses
==================================================

Based on analysis of incorrect answers, these tools address:
1. Multi-district comparison queries (Q51, Q52)
2. Granularity issues - finding summaries vs details (Q15, Q30)
3. Intersection queries - finding entities matching multiple criteria
"""

from typing import Optional
from src.util.services import get_services
from src.util.llm_client import get_dedup_embeddings


# === TOOL 1: Search by Document Section ===

def search_by_header(
    header_pattern: str,
    query: str,
    user_id: str,
    top_k: int = 10
) -> dict:
    """
    Search chunks by header/section path with semantic filtering.

    PURPOSE: Find information at the right level of granularity.
    - Use "OverallEconomicActivity" to get national summaries
    - Use "Federal Reserve Bank of Chicago" to get district-specific info

    Args:
        header_pattern: Pattern to match in header_path (e.g., "OverallEconomicActivity",
                        "Federal Reserve Bank of", "LaborMarkets")
        query: Semantic query to filter/rank results
        user_id: Tenant ID
        top_k: Number of results

    Returns:
        Matching chunks with their content and headers
    """
    services = get_services()
    embeddings = get_dedup_embeddings()

    # Get query embedding for semantic ranking
    query_vector = embeddings.embed_query(query)

    # Search chunks matching header pattern
    results = services.neo4j.query("""
        MATCH (c:EpisodicNode {group_id: $uid})
        WHERE c.header_path CONTAINS $header_pattern
        WITH c,
             gds.similarity.cosine($query_vec, c.embedding) as score
        WHERE score > 0.3
        RETURN c.uuid as chunk_id,
               c.header_path as header,
               c.content as content,
               score
        ORDER BY score DESC
        LIMIT $top_k
    """, {
        "uid": user_id,
        "header_pattern": header_pattern,
        "query_vec": query_vector,
        "top_k": top_k
    })

    if not results:
        return {
            "found": False,
            "results": [],
            "message": f"No chunks found matching header '{header_pattern}' and query '{query}'"
        }

    formatted = []
    for row in results:
        formatted.append({
            "header": row["header"],
            "content": row["content"][:500] + "..." if len(row.get("content", "")) > 500 else row.get("content", ""),
            "score": round(row["score"], 3),
            "chunk_id": row["chunk_id"]
        })

    return {
        "found": True,
        "count": len(formatted),
        "results": formatted,
        "message": f"Found {len(formatted)} chunks matching header pattern"
    }


# === TOOL 2: List Districts ===

def list_districts(user_id: str) -> dict:
    """
    List all Federal Reserve districts in the knowledge graph.

    PURPOSE: Enable systematic multi-district queries.
    The user can then iterate through districts for comparison queries.

    Returns:
        List of district names and their summary info
    """
    services = get_services()

    # Find all district entities
    results = services.neo4j.query("""
        MATCH (e:EntityNode {group_id: $uid})
        WHERE e.name CONTAINS 'Federal Reserve Bank'
           OR e.name CONTAINS 'District'
        RETURN DISTINCT e.name as name, e.summary as summary
        ORDER BY e.name
    """, {"uid": user_id})

    # Also get districts from chunk headers
    header_results = services.neo4j.query("""
        MATCH (c:EpisodicNode {group_id: $uid})
        WHERE c.header_path CONTAINS 'Federal Reserve Bank'
        WITH c.header_path as header
        RETURN DISTINCT
            CASE
                WHEN header CONTAINS 'Boston' THEN 'Federal Reserve Bank of Boston'
                WHEN header CONTAINS 'New York' THEN 'Federal Reserve Bank of New York'
                WHEN header CONTAINS 'Philadelphia' THEN 'Federal Reserve Bank of Philadelphia'
                WHEN header CONTAINS 'Cleveland' THEN 'Federal Reserve Bank of Cleveland'
                WHEN header CONTAINS 'Richmond' THEN 'Federal Reserve Bank of Richmond'
                WHEN header CONTAINS 'Atlanta' THEN 'Federal Reserve Bank of Atlanta'
                WHEN header CONTAINS 'Chicago' THEN 'Federal Reserve Bank of Chicago'
                WHEN header CONTAINS 'St. Louis' THEN 'Federal Reserve Bank of St. Louis'
                WHEN header CONTAINS 'Minneapolis' THEN 'Federal Reserve Bank of Minneapolis'
                WHEN header CONTAINS 'Kansas City' THEN 'Federal Reserve Bank of Kansas City'
                WHEN header CONTAINS 'Dallas' THEN 'Federal Reserve Bank of Dallas'
                WHEN header CONTAINS 'San Francisco' THEN 'Federal Reserve Bank of San Francisco'
                ELSE null
            END as district
        ORDER BY district
    """, {"uid": user_id})

    districts = []
    seen = set()

    for row in results:
        name = row.get("name")
        if name and name not in seen:
            seen.add(name)
            districts.append({
                "name": name,
                "summary": row.get("summary", "")[:200] if row.get("summary") else ""
            })

    for row in header_results:
        name = row.get("district")
        if name and name not in seen:
            seen.add(name)
            districts.append({"name": name, "summary": ""})

    return {
        "found": len(districts) > 0,
        "count": len(districts),
        "districts": sorted(districts, key=lambda x: x["name"]),
        "message": f"Found {len(districts)} Federal Reserve districts"
    }


# === TOOL 3: Search Within District ===

def search_district(
    district_name: str,
    query: str,
    user_id: str,
    top_k: int = 5
) -> dict:
    """
    Search for facts/chunks within a specific Federal Reserve district.

    PURPOSE: Enable district-specific queries for comparison tasks.

    Args:
        district_name: District name (e.g., "Chicago", "New York", "Federal Reserve Bank of Chicago")
        query: Semantic query for what to find
        user_id: Tenant ID
        top_k: Number of results

    Returns:
        Facts and chunks from that district matching the query
    """
    services = get_services()
    embeddings = get_dedup_embeddings()

    # Normalize district name
    district_pattern = district_name
    if "Federal Reserve" not in district_name:
        district_pattern = f"Federal Reserve Bank of {district_name}"

    query_vector = embeddings.embed_query(query)

    # Search chunks in this district
    results = services.neo4j.query("""
        MATCH (c:EpisodicNode {group_id: $uid})
        WHERE c.header_path CONTAINS $district
        WITH c
        // Get facts from this chunk
        OPTIONAL MATCH (subj)-[r1]->(c)-[r2]->(obj)
        WHERE (subj:EntityNode OR subj:TopicNode)
          AND (obj:EntityNode OR obj:TopicNode)
          AND r1.fact_id = r2.fact_id
        OPTIONAL MATCH (f:FactNode {uuid: r1.fact_id, group_id: $uid})
        WITH c, f, subj, obj, type(r1) as edge_type
        RETURN c.uuid as chunk_id,
               c.header_path as header,
               c.content as content,
               collect(DISTINCT {
                   fact: f.content,
                   subject: subj.name,
                   edge: edge_type,
                   object: obj.name
               }) as facts
        LIMIT $top_k
    """, {
        "uid": user_id,
        "district": district_pattern,
        "top_k": top_k
    })

    if not results:
        return {
            "found": False,
            "district": district_name,
            "results": [],
            "message": f"No information found for district '{district_name}'"
        }

    formatted = []
    for row in results:
        facts = [f for f in row.get("facts", []) if f.get("fact")]
        formatted.append({
            "header": row["header"],
            "content": row["content"][:500] + "..." if len(row.get("content", "")) > 500 else row.get("content", ""),
            "facts": facts[:5],  # Limit facts per chunk
            "chunk_id": row["chunk_id"]
        })

    return {
        "found": True,
        "district": district_name,
        "count": len(formatted),
        "results": formatted,
        "message": f"Found {len(formatted)} results in {district_name}"
    }


# === TOOL 4: Find Districts Matching Criteria ===

def find_districts_matching(
    criteria_query: str,
    user_id: str
) -> dict:
    """
    Find all districts that match a semantic criteria.

    PURPOSE: Answer questions like "Which districts saw economic decline?"
    Returns list of districts with evidence.

    Args:
        criteria_query: Description of what to look for (e.g., "economic activity declined")
        user_id: Tenant ID

    Returns:
        List of districts matching the criteria with evidence snippets
    """
    services = get_services()
    embeddings = get_dedup_embeddings()

    query_vector = embeddings.embed_query(criteria_query)

    # Search facts across all districts
    results = services.neo4j.query("""
        CALL db.index.vector.queryNodes('fact_embeddings', 50, $vec)
        YIELD node, score
        WHERE node.group_id = $uid AND score > 0.5

        // Find the chunk and its district
        MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
        WHERE (subj:EntityNode OR subj:TopicNode)

        WITH node, score, c,
             CASE
                WHEN c.header_path CONTAINS 'Boston' THEN 'Boston'
                WHEN c.header_path CONTAINS 'New York' THEN 'New York'
                WHEN c.header_path CONTAINS 'Philadelphia' THEN 'Philadelphia'
                WHEN c.header_path CONTAINS 'Cleveland' THEN 'Cleveland'
                WHEN c.header_path CONTAINS 'Richmond' THEN 'Richmond'
                WHEN c.header_path CONTAINS 'Atlanta' THEN 'Atlanta'
                WHEN c.header_path CONTAINS 'Chicago' THEN 'Chicago'
                WHEN c.header_path CONTAINS 'St. Louis' THEN 'St. Louis'
                WHEN c.header_path CONTAINS 'Minneapolis' THEN 'Minneapolis'
                WHEN c.header_path CONTAINS 'Kansas City' THEN 'Kansas City'
                WHEN c.header_path CONTAINS 'Dallas' THEN 'Dallas'
                WHEN c.header_path CONTAINS 'San Francisco' THEN 'San Francisco'
                WHEN c.header_path CONTAINS 'OverallEconomicActivity' THEN 'National Summary'
                ELSE 'Unknown'
             END as district

        WHERE district <> 'Unknown'

        RETURN district,
               collect({fact: node.content, score: score, header: c.header_path})[0..3] as evidence,
               max(score) as best_score
        ORDER BY best_score DESC
    """, {"vec": query_vector, "uid": user_id})

    if not results:
        return {
            "found": False,
            "query": criteria_query,
            "districts": [],
            "message": f"No districts found matching '{criteria_query}'"
        }

    districts = []
    for row in results:
        districts.append({
            "district": row["district"],
            "score": round(row["best_score"], 3),
            "evidence": row["evidence"]
        })

    return {
        "found": True,
        "query": criteria_query,
        "count": len(districts),
        "districts": districts,
        "message": f"Found {len(districts)} districts matching criteria"
    }


# === TOOL 5: Find Intersection (Districts with Multiple Criteria) ===

def find_districts_intersection(
    criteria_1: str,
    criteria_2: str,
    user_id: str
) -> dict:
    """
    Find districts that match BOTH criteria.

    PURPOSE: Answer questions like "Which districts mentioned both X AND Y?"

    Args:
        criteria_1: First criteria (e.g., "tariff impacts on manufacturing")
        criteria_2: Second criteria (e.g., "labor shortages from immigration")
        user_id: Tenant ID

    Returns:
        Districts matching both criteria with evidence for each
    """
    # Find districts matching each criteria
    result_1 = find_districts_matching(criteria_1, user_id)
    result_2 = find_districts_matching(criteria_2, user_id)

    if not result_1["found"] or not result_2["found"]:
        return {
            "found": False,
            "criteria_1": criteria_1,
            "criteria_2": criteria_2,
            "districts": [],
            "message": "Could not find districts matching one or both criteria"
        }

    # Find intersection
    districts_1 = {d["district"]: d for d in result_1["districts"]}
    districts_2 = {d["district"]: d for d in result_2["districts"]}

    intersection = []
    for district in districts_1:
        if district in districts_2:
            intersection.append({
                "district": district,
                "criteria_1_evidence": districts_1[district]["evidence"],
                "criteria_1_score": districts_1[district]["score"],
                "criteria_2_evidence": districts_2[district]["evidence"],
                "criteria_2_score": districts_2[district]["score"],
                "combined_score": (districts_1[district]["score"] + districts_2[district]["score"]) / 2
            })

    # Sort by combined score
    intersection.sort(key=lambda x: x["combined_score"], reverse=True)

    return {
        "found": len(intersection) > 0,
        "criteria_1": criteria_1,
        "criteria_2": criteria_2,
        "count": len(intersection),
        "districts": intersection,
        "all_matching_criteria_1": list(districts_1.keys()),
        "all_matching_criteria_2": list(districts_2.keys()),
        "message": f"Found {len(intersection)} districts matching BOTH criteria"
    }


# === TOOL 6: Get National Summary ===

def get_national_summary(
    topic: str,
    user_id: str
) -> dict:
    """
    Get the national-level summary for a topic.

    PURPOSE: Answer high-level questions without getting lost in district details.
    Searches specifically in national summary sections.

    Args:
        topic: Topic to find summary for (e.g., "economic activity", "labor markets", "prices")
        user_id: Tenant ID

    Returns:
        National summary content for the topic
    """
    services = get_services()
    embeddings = get_dedup_embeddings()

    query_vector = embeddings.embed_query(topic)

    # Search in national/overall sections only
    results = services.neo4j.query("""
        MATCH (c:EpisodicNode {group_id: $uid})
        WHERE c.header_path CONTAINS 'OverallEconomicActivity'
           OR c.header_path CONTAINS 'LaborMarkets'
           OR c.header_path CONTAINS 'Prices'
           OR (c.header_path CONTAINS 'Summary' AND NOT c.header_path CONTAINS 'Federal Reserve Bank')
        RETURN c.uuid as chunk_id,
               c.header_path as header,
               c.content as content
        LIMIT 20
    """, {"uid": user_id})

    if not results:
        return {
            "found": False,
            "topic": topic,
            "message": "No national summary sections found"
        }

    # Rank by relevance to topic
    ranked = []
    for row in results:
        content = row.get("content", "")
        # Simple keyword matching for now
        relevance = sum(1 for word in topic.lower().split() if word in content.lower())
        ranked.append({
            "header": row["header"],
            "content": content,
            "relevance": relevance,
            "chunk_id": row["chunk_id"]
        })

    ranked.sort(key=lambda x: x["relevance"], reverse=True)

    return {
        "found": True,
        "topic": topic,
        "results": ranked[:5],
        "message": f"Found {len(ranked)} national summary sections"
    }
