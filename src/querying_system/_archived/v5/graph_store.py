"""
V5 GraphStore: Unified interface for all graph queries.

Centralizes all Neo4j and Qdrant operations, eliminating duplicate Cypher patterns.
All queries respect the chunk-centric design: Entity -> EpisodicNode -> Entity with fact_id linking.
"""

import asyncio
import os
from typing import Optional, Union
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.util.services import get_services
from src.util.llm_client import get_nano_gpt_llm
from src.querying_system.shared.schemas import EntityHint
from .schemas import RawFact, ResolvedEntity, ResolvedTopic, ResolvedContext

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[GraphStore] {msg}")


# =============================================================================
# Resolution LLM Schemas
# =============================================================================

class ResolvedNode(BaseModel):
    """A resolved graph node from LLM verification."""
    name: str = Field(..., description="Node name in the graph")
    match_reason: str = Field(default="", description="Why this matches")


class ResolutionResult(BaseModel):
    """LLM output for resolution verification."""
    resolved_nodes: list[ResolvedNode] = Field(default_factory=list)
    no_match: bool = Field(default=False)


RESOLUTION_SYSTEM_PROMPT = """You are resolving query terms to knowledge graph nodes.

Given a query term, the original question context, and candidate nodes, determine which candidates match.

RULES:
- Use the question context to understand what the query term refers to
- Match based on semantic equivalence in the context of the question
- Generic/plural terms should match all relevant specific instances
- A term can match multiple candidates
- If no candidates match, set no_match to true"""

RESOLUTION_USER_PROMPT = """QUESTION CONTEXT: {question}

QUERY TERM: {term}

CANDIDATE NODES:
{candidates}

Which candidates match the query term?"""


# =============================================================================
# GraphStore
# =============================================================================

class GraphStore:
    """
    Unified interface for all graph queries.

    Centralizes:
    - Entity/topic resolution (vector search + LLM verification)
    - Scoped fact retrieval (facts connected to specific entities/topics)
    - Global fact retrieval (vector + keyword search)
    - Graph expansion (1-hop from entities)
    - Chunk content fetching
    """

    def __init__(self, group_id: str = "default"):
        self.group_id = group_id
        self.services = get_services()
        self.resolution_llm = get_nano_gpt_llm()
        self.structured_resolver = self.resolution_llm.with_structured_output(ResolutionResult)

    # =========================================================================
    # Resolution
    # =========================================================================

    async def resolve(
        self,
        entity_hints: Union[list[str], list[EntityHint]],
        topic_hints: Union[list[str], list[EntityHint]],
        question_context: str = "",
        top_k_candidates: int = 20,
    ) -> ResolvedContext:
        """
        Resolve entity and topic hints to actual graph nodes.

        Uses vector search to find candidates, then LLM to verify matches.
        """
        log(f"Resolving {len(entity_hints)} entities, {len(topic_hints)} topics")

        # Run entity and topic resolution in parallel
        entity_task = self._resolve_entities(entity_hints, question_context, top_k_candidates)
        topic_task = self._resolve_topics(topic_hints, question_context, top_k_candidates)

        entities, topics = await asyncio.gather(entity_task, topic_task)

        log(f"Resolved to {len(entities)} entities, {len(topics)} topics")

        return ResolvedContext(entities=entities, topics=topics)

    async def _resolve_entities(
        self,
        hints: Union[list[str], list[EntityHint]],
        question: str,
        top_k: int,
    ) -> list[ResolvedEntity]:
        """Resolve entity hints to EntityNode names."""
        if not hints:
            return []

        results = []
        for hint in hints:
            embed_text, display_name = self._hint_to_embed_text(hint)
            candidates = await self._get_entity_candidates(embed_text, top_k)

            if not candidates:
                log(f"No entity candidates for: {display_name}")
                continue

            resolved_names = await self._verify_candidates(display_name, candidates, question)
            for name in resolved_names:
                results.append(ResolvedEntity(
                    original_hint=display_name,
                    resolved_name=name,
                ))

        # Dedupe by resolved_name
        seen = set()
        deduped = []
        for r in results:
            if r.resolved_name not in seen:
                seen.add(r.resolved_name)
                deduped.append(r)

        return deduped

    async def _resolve_topics(
        self,
        hints: Union[list[str], list[EntityHint]],
        question: str,
        top_k: int,
    ) -> list[ResolvedTopic]:
        """Resolve topic hints to TopicNode names."""
        if not hints:
            return []

        results = []
        for hint in hints:
            embed_text, display_name = self._hint_to_embed_text(hint)
            candidates = await self._get_topic_candidates(embed_text, top_k)

            if not candidates:
                log(f"No topic candidates for: {display_name}")
                continue

            resolved_names = await self._verify_candidates(display_name, candidates, question)
            for name in resolved_names:
                results.append(ResolvedTopic(
                    original_hint=display_name,
                    resolved_name=name,
                ))

        # Dedupe
        seen = set()
        deduped = []
        for r in results:
            if r.resolved_name not in seen:
                seen.add(r.resolved_name)
                deduped.append(r)

        return deduped

    def _hint_to_embed_text(self, hint: Union[str, EntityHint]) -> tuple[str, str]:
        """Convert hint to (embed_text, display_name)."""
        if isinstance(hint, EntityHint):
            return f"{hint.name}: {hint.definition}", hint.name
        return hint, hint

    async def _get_entity_candidates(self, embed_text: str, top_k: int) -> list[str]:
        """Vector search for entity candidates."""
        embedding = await asyncio.to_thread(
            self.services.embeddings.embed_query, embed_text
        )

        def _query():
            results = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('entity_name_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > 0.3
                RETURN DISTINCT node.name as name, score
                ORDER BY score DESC
                """,
                {"vec": embedding, "uid": self.group_id, "top_k": top_k}
            )
            return [r["name"] for r in results]

        return await asyncio.to_thread(_query)

    async def _get_topic_candidates(self, embed_text: str, top_k: int) -> list[str]:
        """Vector search for topic candidates."""
        embedding = await asyncio.to_thread(
            self.services.embeddings.embed_query, embed_text
        )

        def _query():
            results = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('topic_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > 0.3
                RETURN DISTINCT node.name as name, score
                ORDER BY score DESC
                """,
                {"vec": embedding, "uid": self.group_id, "top_k": top_k}
            )
            return [r["name"] for r in results]

        return await asyncio.to_thread(_query)

    async def _verify_candidates(
        self,
        hint: str,
        candidates: list[str],
        question: str,
    ) -> list[str]:
        """LLM verification of candidate matches."""
        if not candidates:
            return []

        candidates_text = "\n".join(f"- {name}" for name in candidates)
        prompt = RESOLUTION_USER_PROMPT.format(
            question=question,
            term=hint,
            candidates=candidates_text,
        )

        try:
            result = await asyncio.to_thread(
                self.structured_resolver.invoke,
                [
                    ("system", RESOLUTION_SYSTEM_PROMPT),
                    ("human", prompt),
                ]
            )

            if result.no_match:
                log(f"LLM found no matches for '{hint}'")
                return []

            resolved = [node.name for node in result.resolved_nodes]
            log(f"Resolved '{hint}' -> {resolved}")
            return resolved

        except Exception as e:
            log(f"Resolution error for '{hint}': {e}")
            # Fallback to top candidates
            return candidates[:3]

    # =========================================================================
    # Scoped Fact Retrieval
    # =========================================================================

    async def search_entity_facts(
        self,
        entity_name: str,
        query_embedding: list[float],
        threshold: float = 0.3,
        top_k: int = 500,
    ) -> list[RawFact]:
        """
        Search facts connected to a specific entity.

        Uses the chunk-centric pattern:
        (Entity) -[r1 {fact_id}]-> (EpisodicNode) -[r2 {fact_id}]-> (Target)
        """
        log(f"Searching facts for entity: {entity_name}")

        def _query():
            # Search as subject
            subject_results = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                MATCH (subj:EntityNode {name: $entity_name, group_id: $uid})
                      -[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
                      -[r2 {fact_id: node.uuid}]->(obj)
                WHERE (obj:EntityNode OR obj:TopicNode) AND obj.group_id = $uid

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT
                    node.uuid as fact_id,
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
                """,
                {
                    "vec": query_embedding,
                    "uid": self.group_id,
                    "entity_name": entity_name,
                    "threshold": threshold,
                    "top_k": top_k,
                }
            )

            # Search as object
            object_results = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
                      -[r2 {fact_id: node.uuid}]->(obj:EntityNode {name: $entity_name, group_id: $uid})
                WHERE (subj:EntityNode OR subj:TopicNode) AND subj.group_id = $uid

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT
                    node.uuid as fact_id,
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
                """,
                {
                    "vec": query_embedding,
                    "uid": self.group_id,
                    "entity_name": entity_name,
                    "threshold": threshold,
                    "top_k": top_k,
                }
            )

            return subject_results + object_results

        results = await asyncio.to_thread(_query)

        # Convert to RawFact and dedupe
        facts = self._results_to_facts(results, source=f"scoped:{entity_name}")
        log(f"Found {len(facts)} facts for entity {entity_name}")
        return facts

    async def search_topic_facts(
        self,
        topic_name: str,
        query_embedding: list[float],
        threshold: float = 0.3,
        top_k: int = 500,
    ) -> list[RawFact]:
        """Search facts connected to a specific topic."""
        log(f"Searching facts for topic: {topic_name}")

        def _query():
            # Topics connect via DISCUSSES relationship
            results = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                MATCH (c:EpisodicNode {group_id: $uid})-[:DISCUSSES]->(t:TopicNode {name: $topic_name, group_id: $uid})

                // Get facts from this chunk
                MATCH (subj)-[r1 {fact_id: node.uuid}]->(c)-[r2 {fact_id: node.uuid}]->(obj)
                WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)
                  AND subj.group_id = $uid AND obj.group_id = $uid

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT
                    node.uuid as fact_id,
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
                """,
                {
                    "vec": query_embedding,
                    "uid": self.group_id,
                    "topic_name": topic_name,
                    "threshold": threshold,
                    "top_k": top_k,
                }
            )
            return results

        results = await asyncio.to_thread(_query)
        facts = self._results_to_facts(results, source=f"scoped:{topic_name}")
        log(f"Found {len(facts)} facts for topic {topic_name}")
        return facts

    # =========================================================================
    # Global Fact Retrieval
    # =========================================================================

    async def search_all_facts_vector(
        self,
        query_embedding: list[float],
        top_k: int = 30,
        threshold: float = 0.25,
    ) -> list[RawFact]:
        """Global vector search across all facts."""
        log(f"Global vector search (top_k={top_k})")

        def _query():
            results = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
                               -[r2 {fact_id: node.uuid}]->(obj)
                WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)
                  AND subj.group_id = $uid AND obj.group_id = $uid

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT
                    node.uuid as fact_id,
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
                """,
                {
                    "vec": query_embedding,
                    "uid": self.group_id,
                    "threshold": threshold,
                    "top_k": top_k,
                }
            )
            return results

        results = await asyncio.to_thread(_query)
        facts = self._results_to_facts(results, source="global_vector")
        log(f"Global vector search found {len(facts)} facts")
        return facts

    async def search_all_facts_keyword(
        self,
        keywords: list[str],
        top_k: int = 30,
    ) -> list[RawFact]:
        """Global keyword/fulltext search across all facts."""
        if not keywords:
            return []

        keyword_string = " OR ".join(keywords)
        log(f"Global keyword search: {keyword_string[:50]}...")

        def _query():
            results = self.services.neo4j.query(
                """
                CALL db.index.fulltext.queryNodes('fact_fulltext', $keywords)
                YIELD node, score
                WHERE node.group_id = $uid

                OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
                               -[r2 {fact_id: node.uuid}]->(obj)
                WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)
                  AND subj.group_id = $uid AND obj.group_id = $uid

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT
                    node.uuid as fact_id,
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
                """,
                {
                    "keywords": keyword_string,
                    "uid": self.group_id,
                    "top_k": top_k,
                }
            )
            return results

        results = await asyncio.to_thread(_query)
        facts = self._results_to_facts(results, source="global_keyword")
        log(f"Global keyword search found {len(facts)} facts")
        return facts

    # =========================================================================
    # Expansion
    # =========================================================================

    async def expand_from_entity(
        self,
        entity_name: str,
        query_embedding: list[float],
        max_facts: int = 5,
    ) -> list[RawFact]:
        """
        1-hop expansion: get facts connected to an entity's neighbors.

        Used for gap-driven expansion when we need more context.
        """
        log(f"Expanding from entity: {entity_name}")

        def _query():
            results = self.services.neo4j.query(
                """
                // Get facts where entity is involved
                MATCH (e:EntityNode {name: $entity_name, group_id: $uid})
                      -[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(target)
                WHERE r1.fact_id = r2.fact_id
                  AND (target:EntityNode OR target:TopicNode)
                  AND target.group_id = $uid

                // Get the fact node
                MATCH (f:FactNode {uuid: r1.fact_id, group_id: $uid})

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT
                    f.uuid as fact_id,
                    f.content as content,
                    1.0 as score,
                    e.name as subject,
                    type(r1) as edge_type,
                    target.name as object,
                    c.uuid as chunk_id,
                    c.content as chunk_content,
                    c.header_path as chunk_header,
                    d.name as doc_id,
                    d.document_date as document_date
                LIMIT $max_facts
                """,
                {
                    "entity_name": entity_name,
                    "uid": self.group_id,
                    "max_facts": max_facts,
                }
            )
            return results

        results = await asyncio.to_thread(_query)
        facts = self._results_to_facts(results, source=f"expansion:{entity_name}")
        log(f"Expansion found {len(facts)} facts from {entity_name}")
        return facts

    # =========================================================================
    # Chunk Content
    # =========================================================================

    async def fetch_chunk_content(
        self,
        chunk_ids: list[str],
    ) -> dict[str, str]:
        """Fetch chunk content by IDs."""
        if not chunk_ids:
            return {}

        def _query():
            results = self.services.neo4j.query(
                """
                UNWIND $chunk_ids as cid
                MATCH (c:EpisodicNode {uuid: cid, group_id: $uid})
                RETURN c.uuid as chunk_id, c.content as content
                """,
                {"chunk_ids": chunk_ids, "uid": self.group_id}
            )
            return {r["chunk_id"]: r["content"] for r in results}

        return await asyncio.to_thread(_query)

    # =========================================================================
    # Embedding
    # =========================================================================

    async def embed_text(self, text: str) -> list[float]:
        """Embed a text string."""
        return await asyncio.to_thread(
            self.services.embeddings.embed_query, text
        )

    async def batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple texts in a single API call."""
        if not texts:
            return []
        if len(texts) == 1:
            return [await self.embed_text(texts[0])]
        return await asyncio.to_thread(
            self.services.embeddings.embed_documents, texts
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _results_to_facts(
        self,
        results: list[dict],
        source: str,
    ) -> list[RawFact]:
        """Convert Neo4j results to RawFact objects, deduped by fact_id."""
        seen = set()
        facts = []

        for r in results:
            fact_id = r.get("fact_id")
            if not fact_id or fact_id in seen:
                continue
            seen.add(fact_id)

            facts.append(RawFact(
                fact_id=fact_id,
                content=r.get("content") or "",
                subject=r.get("subject") or "",
                edge_type=r.get("edge_type") or "",
                object=r.get("object") or "",
                chunk_id=r.get("chunk_id") or "",
                chunk_content=r.get("chunk_content"),
                chunk_header=r.get("chunk_header") or "",
                doc_id=r.get("doc_id") or "",
                document_date=r.get("document_date") or "",
                vector_score=r.get("score") or 0.0,
                source=source,
            ))

        return facts
