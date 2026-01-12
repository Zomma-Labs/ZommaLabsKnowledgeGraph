"""
Deterministic Retrieval System
==============================

Multi-strategy parallel search with RRF (Reciprocal Rank Fusion).
Inspired by Graphiti's approach - no LLM decisions in retrieval.

Three search strategies run in parallel:
1. Vector Search - Semantic similarity on fact embeddings
2. Keyword Search - BM25 full-text search on fact content
3. Graph Traversal - Entity-based neighbor exploration

Results are fused using RRF to produce consistent, comprehensive rankings.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from src.util.services import get_services


@dataclass
class RetrievedEvidence:
    """A single piece of retrieved evidence."""
    fact_id: str
    content: str  # The fact text
    subject: str
    edge_type: str
    object: str
    chunk_id: Optional[str] = None
    chunk_content: Optional[str] = None
    chunk_header: Optional[str] = None
    doc_id: Optional[str] = None
    document_date: Optional[str] = None
    # Scores from different strategies
    vector_score: float = 0.0
    keyword_score: float = 0.0
    graph_score: float = 0.0
    # Final fused score
    rrf_score: float = 0.0
    # Which strategies found this
    found_by: list = field(default_factory=list)


def rrf_score(rank: int, k: int = 60) -> float:
    """
    Reciprocal Rank Fusion score.

    Args:
        rank: 1-indexed rank (1 = best)
        k: Smoothing constant (default 60, standard in literature)

    Returns:
        RRF contribution: 1/(k + rank)
    """
    return 1.0 / (k + rank)


def fuse_results(
    vector_results: list[dict],
    keyword_results: list[dict],
    graph_results: list[dict],
    k: int = 60
) -> list[RetrievedEvidence]:
    """
    Fuse results from multiple strategies using RRF.

    Args:
        vector_results: Results from vector search [{fact_id, content, score, ...}]
        keyword_results: Results from keyword search [{fact_id, content, score, ...}]
        graph_results: Results from graph traversal [{fact_id, content, ...}]
        k: RRF smoothing constant

    Returns:
        List of RetrievedEvidence sorted by fused RRF score
    """
    # Track all evidence by fact_id
    evidence_map: dict[str, RetrievedEvidence] = {}

    # Process vector results
    for rank, result in enumerate(vector_results, start=1):
        fact_id = result.get("fact_id")
        if not fact_id:
            continue

        if fact_id not in evidence_map:
            evidence_map[fact_id] = RetrievedEvidence(
                fact_id=fact_id,
                content=result.get("content", ""),
                subject=result.get("subject", ""),
                edge_type=result.get("edge_type", ""),
                object=result.get("object", ""),
                chunk_id=result.get("chunk_id"),
                chunk_content=result.get("chunk_content"),
                chunk_header=result.get("chunk_header"),
                doc_id=result.get("doc_id"),
                document_date=result.get("document_date"),
            )

        evidence_map[fact_id].vector_score = result.get("score", 0.0)
        evidence_map[fact_id].rrf_score += rrf_score(rank, k)
        evidence_map[fact_id].found_by.append("vector")

    # Process keyword results
    for rank, result in enumerate(keyword_results, start=1):
        fact_id = result.get("fact_id")
        if not fact_id:
            continue

        if fact_id not in evidence_map:
            evidence_map[fact_id] = RetrievedEvidence(
                fact_id=fact_id,
                content=result.get("content", ""),
                subject=result.get("subject", ""),
                edge_type=result.get("edge_type", ""),
                object=result.get("object", ""),
                chunk_id=result.get("chunk_id"),
                chunk_content=result.get("chunk_content"),
                chunk_header=result.get("chunk_header"),
                doc_id=result.get("doc_id"),
                document_date=result.get("document_date"),
            )

        evidence_map[fact_id].keyword_score = result.get("score", 0.0)
        evidence_map[fact_id].rrf_score += rrf_score(rank, k)
        if "keyword" not in evidence_map[fact_id].found_by:
            evidence_map[fact_id].found_by.append("keyword")

    # Process graph results
    for rank, result in enumerate(graph_results, start=1):
        fact_id = result.get("fact_id")
        if not fact_id:
            continue

        if fact_id not in evidence_map:
            evidence_map[fact_id] = RetrievedEvidence(
                fact_id=fact_id,
                content=result.get("content", ""),
                subject=result.get("subject", ""),
                edge_type=result.get("edge_type", ""),
                object=result.get("object", ""),
                chunk_id=result.get("chunk_id"),
                chunk_content=result.get("chunk_content"),
                chunk_header=result.get("chunk_header"),
                doc_id=result.get("doc_id"),
                document_date=result.get("document_date"),
            )

        evidence_map[fact_id].graph_score = result.get("score", 1.0)  # Graph doesn't have scores, use 1.0
        evidence_map[fact_id].rrf_score += rrf_score(rank, k)
        if "graph" not in evidence_map[fact_id].found_by:
            evidence_map[fact_id].found_by.append("graph")

    # Sort by RRF score descending
    results = sorted(evidence_map.values(), key=lambda x: x.rrf_score, reverse=True)
    return results


def extract_keywords(query: str) -> str:
    """
    Extract keywords from query for BM25 search.
    Removes common stop words and creates a Lucene-compatible query.
    """
    # Common stop words to filter
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
        'if', 'or', 'because', 'until', 'while', 'what', 'which', 'who',
        'whom', 'this', 'that', 'these', 'those', 'am', 'it', 'its', 'i',
        'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs',
        'themselves', 'about', 'tell', 'describe', 'explain', 'give'
    }

    # Extract words, filter stop words
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    # Return as space-separated for Lucene
    return ' '.join(keywords)


class DeterministicRetriever:
    """
    Deterministic multi-strategy retriever with RRF fusion.

    Usage:
        retriever = DeterministicRetriever(group_id="default")
        results = await retriever.search("What happened to wages in Boston?", top_k=10)
    """

    def __init__(self, group_id: str = "default"):
        self.group_id = group_id
        self.services = get_services()

    async def _vector_search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Strategy 1: Vector search on fact embeddings.

        Embeds the query and finds semantically similar facts.
        """
        # Get embedding
        query_vector = self.services.embeddings.embed_query(query)

        # Vector search with chunk/document enrichment
        results = self.services.neo4j.query("""
            CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
            YIELD node, score
            WHERE node.group_id = $uid AND score > 0.25

            // Find connected entities and chunk
            OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})-[r2 {fact_id: node.uuid}]->(obj)
            WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)

            // Get document info
            OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

            RETURN node.uuid as fact_id,
                   node.content as content,
                   score,
                   subj.name as subject,
                   type(r1) as edge_type,
                   obj.name as object,
                   c.uuid as chunk_id,
                   c.content as chunk_content,
                   c.header_path as chunk_header,
                   d.name as doc_id,
                   d.document_date as document_date
            ORDER BY score DESC
        """, {"vec": query_vector, "uid": self.group_id, "top_k": top_k})

        return [dict(r) for r in results]

    async def _keyword_search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Strategy 2: BM25 keyword search on fact content.

        Extracts keywords and searches using Neo4j full-text index.
        """
        keywords = extract_keywords(query)
        if not keywords:
            return []

        # Full-text search with chunk/document enrichment
        results = self.services.neo4j.query("""
            CALL db.index.fulltext.queryNodes('fact_fulltext', $keywords)
            YIELD node, score
            WHERE node.group_id = $uid

            // Find connected entities and chunk
            OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})-[r2 {fact_id: node.uuid}]->(obj)
            WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)

            // Get document info
            OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

            RETURN node.uuid as fact_id,
                   node.content as content,
                   score,
                   subj.name as subject,
                   type(r1) as edge_type,
                   obj.name as object,
                   c.uuid as chunk_id,
                   c.content as chunk_content,
                   c.header_path as chunk_header,
                   d.name as doc_id,
                   d.document_date as document_date
            ORDER BY score DESC
            LIMIT $top_k
        """, {"keywords": keywords, "uid": self.group_id, "top_k": top_k})

        return [dict(r) for r in results]

    async def _graph_traversal(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Strategy 3: Entity-based graph traversal.

        1. Find entities mentioned in the query using vector search
        2. Get their direct relationships (facts)
        """
        # First, find relevant entities
        query_vector = self.services.embeddings.embed_query(query)

        entity_results = self.services.neo4j.query("""
            CALL db.index.vector.queryNodes('entity_name_only_embeddings', 5, $vec)
            YIELD node, score
            WHERE node.group_id = $uid AND score > 0.5
            RETURN node.name as name, score
            ORDER BY score DESC
        """, {"vec": query_vector, "uid": self.group_id})

        if not entity_results:
            # Fallback: try full-text search on entity names
            keywords = extract_keywords(query)
            if keywords:
                entity_results = self.services.neo4j.query("""
                    CALL db.index.fulltext.queryNodes('entity_fulltext', $keywords)
                    YIELD node, score
                    WHERE node.group_id = $uid
                    RETURN node.name as name, score
                    ORDER BY score DESC
                    LIMIT 5
                """, {"keywords": keywords, "uid": self.group_id})

        if not entity_results:
            return []

        # Get facts for these entities
        entity_names = [r["name"] for r in entity_results]

        results = self.services.neo4j.query("""
            UNWIND $entities as entity_name

            // Outgoing relationships
            MATCH (e {name: entity_name, group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(target)
            WHERE (e:EntityNode OR e:TopicNode) AND (target:EntityNode OR target:TopicNode)
              AND r1.fact_id = r2.fact_id

            // Get fact node
            OPTIONAL MATCH (f:FactNode {uuid: r1.fact_id, group_id: $uid})

            // Get document info
            OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

            RETURN DISTINCT r1.fact_id as fact_id,
                   COALESCE(f.content, e.name + ' ' + type(r1) + ' ' + target.name) as content,
                   1.0 as score,
                   e.name as subject,
                   type(r1) as edge_type,
                   target.name as object,
                   c.uuid as chunk_id,
                   c.content as chunk_content,
                   c.header_path as chunk_header,
                   d.name as doc_id,
                   d.document_date as document_date

            UNION

            // Incoming relationships
            UNWIND $entities as entity_name
            MATCH (source)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(e {name: entity_name, group_id: $uid})
            WHERE (e:EntityNode OR e:TopicNode) AND (source:EntityNode OR source:TopicNode)
              AND r1.fact_id = r2.fact_id

            // Get fact node
            OPTIONAL MATCH (f:FactNode {uuid: r1.fact_id, group_id: $uid})

            // Get document info
            OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

            RETURN DISTINCT r1.fact_id as fact_id,
                   COALESCE(f.content, source.name + ' ' + type(r1) + ' ' + e.name) as content,
                   1.0 as score,
                   source.name as subject,
                   type(r1) as edge_type,
                   e.name as object,
                   c.uuid as chunk_id,
                   c.content as chunk_content,
                   c.header_path as chunk_header,
                   d.name as doc_id,
                   d.document_date as document_date

            LIMIT $top_k
        """, {"entities": entity_names, "uid": self.group_id, "top_k": top_k})

        return [dict(r) for r in results]

    async def search(self, query: str, top_k: int = 10) -> list[RetrievedEvidence]:
        """
        Execute multi-strategy search with RRF fusion.

        Args:
            query: The user's question
            top_k: Number of results to return

        Returns:
            List of RetrievedEvidence sorted by RRF score
        """
        # Run all three strategies in parallel
        vector_task = self._vector_search(query, top_k=top_k * 2)
        keyword_task = self._keyword_search(query, top_k=top_k * 2)
        graph_task = self._graph_traversal(query, top_k=top_k * 2)

        vector_results, keyword_results, graph_results = await asyncio.gather(
            vector_task, keyword_task, graph_task
        )

        # Fuse results with RRF
        fused = fuse_results(vector_results, keyword_results, graph_results)

        return fused[:top_k]

    def search_sync(self, query: str, top_k: int = 10) -> list[RetrievedEvidence]:
        """Synchronous wrapper for search."""
        return asyncio.run(self.search(query, top_k))

    def format_evidence_for_llm(self, evidence: list[RetrievedEvidence]) -> str:
        """
        Format retrieved evidence for LLM consumption.

        Returns a structured text block that the LLM can use to answer questions.
        """
        if not evidence:
            return "No relevant evidence found in the knowledge graph."

        lines = [
            "# Retrieved Evidence",
            f"Found {len(evidence)} relevant facts.\n"
        ]

        for i, e in enumerate(evidence, start=1):
            strategies = ", ".join(e.found_by)
            lines.append(f"## Evidence {i} (RRF: {e.rrf_score:.4f}, found by: {strategies})")
            lines.append(f"**Fact:** {e.content}")
            lines.append(f"**Relationship:** {e.subject} -[{e.edge_type}]-> {e.object}")
            if e.document_date:
                lines.append(f"**Date:** {e.document_date}")
            if e.chunk_header:
                lines.append(f"**Source Section:** {e.chunk_header}")
            if e.doc_id:
                lines.append(f"**Document:** {e.doc_id}")
            if e.chunk_id:
                lines.append(f"**Chunk ID:** {e.chunk_id}")
            if e.chunk_content:
                # Truncate chunk content if too long
                content = e.chunk_content[:500] + "..." if len(e.chunk_content) > 500 else e.chunk_content
                lines.append(f"**Context:**\n```\n{content}\n```")
            lines.append("")

        return "\n".join(lines)


# Convenience function for direct use
def search_kg(query: str, group_id: str = "default", top_k: int = 10) -> list[RetrievedEvidence]:
    """
    Deterministic knowledge graph search.

    Runs three parallel search strategies and fuses with RRF.
    No LLM decisions - pure mechanical retrieval.

    Args:
        query: The user's question
        group_id: Tenant ID for multi-tenant isolation
        top_k: Number of results to return

    Returns:
        List of RetrievedEvidence sorted by fused RRF score
    """
    retriever = DeterministicRetriever(group_id=group_id)
    return retriever.search_sync(query, top_k)
