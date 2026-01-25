"""
V7 GraphStore: Knowledge graph interface with wide-net resolution and chunk-centric retrieval.

V7 Design Principles:
- Wide-net resolution: One hint can resolve to MANY graph nodes (one-to-many)
- Chunk-centric retrieval: Returns EpisodicNode content, not just FactNodes
- 1-hop expansion: Traverse to neighbor entities for additional context
- Global search: Vector search across all chunks for broad coverage

Key differences from V6:
- Resolution returns summaries with entities (for context)
- Chunk retrieval is primary (not fact retrieval)
- 1-hop neighbor expansion is explicit
"""

import asyncio
import os
from typing import Optional, Union

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from src.util.services import get_services
from .schemas import (
    ResolvedEntity,
    ResolvedTopic,
    ResolvedContext,
    RetrievedChunk,
    RetrievedFact,
    RetrievedEntity,
)
from .prompts import (
    ENTITY_RESOLUTION_SYSTEM_PROMPT,
    TOPIC_RESOLUTION_SYSTEM_PROMPT,
    RESOLUTION_USER_PROMPT,
    format_candidates_for_resolution,
)

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[V7 GraphStore] {msg}")


# =============================================================================
# LLM Resolution Schemas
# =============================================================================

class ResolvedNode(BaseModel):
    """A node selected by the LLM during resolution."""
    name: str = Field(..., description="The exact name of the node in the graph")
    reason: str = Field(default="", description="Why this node matches the query term")


class ResolutionResult(BaseModel):
    """LLM output for resolution - can match many nodes."""
    resolved_nodes: list[ResolvedNode] = Field(
        default_factory=list,
        description="List of matching nodes from candidates"
    )
    no_match: bool = Field(
        default=False,
        description="True if no candidates are relevant"
    )


# =============================================================================
# GraphStore
# =============================================================================

class GraphStore:
    """
    V7 Knowledge graph interface with wide-net resolution and chunk-centric retrieval.

    Features:
    - Wide-net entity/topic resolution (one hint -> many nodes)
    - Chunk retrieval via entity/topic connections
    - 1-hop expansion for neighbor entities
    - FactNode retrieval for structured knowledge
    - Global chunk search for broad coverage
    """

    def __init__(self, group_id: str = "default"):
        self.group_id = group_id
        self.services = get_services()

        # Resolution LLM - use gemini-3-flash-preview as specified in V7Config
        self.resolution_llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0,
        )
        self.structured_resolver = self.resolution_llm.with_structured_output(ResolutionResult)

        # Resolution cache to avoid redundant LLM calls across sub-queries
        self._entity_cache: dict[str, list[tuple[str, str]]] = {}  # hint -> [(name, summary)]
        self._topic_cache: dict[str, list[tuple[str, str]]] = {}   # hint -> [(name, definition)]

    # =========================================================================
    # Resolution Methods
    # =========================================================================

    async def resolve_entities(
        self,
        hints: list[str],
        question_context: str,
        threshold: float = 0.3,
        top_k_candidates: int = 30,
    ) -> list[ResolvedEntity]:
        """
        Resolve entity hints to graph entities with wide-net matching.

        One hint can resolve to MANY entities (e.g., "tech companies" -> Apple, Microsoft, etc.)

        Args:
            hints: Entity names/descriptions to resolve
            question_context: The original question for context
            threshold: Minimum vector similarity for candidates
            top_k_candidates: Max candidates to retrieve per hint

        Returns:
            List of ResolvedEntity with resolved_name and summary
        """
        if not hints:
            return []

        log(f"Resolving {len(hints)} entity hints")

        async def resolve_single_hint(hint: str) -> list[ResolvedEntity]:
            """Resolve a single hint to entities."""
            # Check cache first
            cache_key = hint.lower().strip()
            if cache_key in self._entity_cache:
                log(f"Entity cache hit for: {hint}")
                return [
                    ResolvedEntity(
                        original_hint=hint,
                        resolved_name=name,
                        summary=summary,
                        confidence=1.0,
                    )
                    for name, summary in self._entity_cache[cache_key]
                ]

            # Get candidates via vector search
            candidates = await self._get_entity_candidates(hint, top_k_candidates, threshold)

            if not candidates:
                log(f"No entity candidates for: {hint}")
                self._entity_cache[cache_key] = []  # Cache empty result
                return []

            # LLM picks matching candidates (one-to-many)
            resolved = await self._resolve_with_llm(
                hint=hint,
                candidates=candidates,
                question=question_context,
                system_prompt=ENTITY_RESOLUTION_SYSTEM_PROMPT,
            )

            # Cache result
            self._entity_cache[cache_key] = resolved

            return [
                ResolvedEntity(
                    original_hint=hint,
                    resolved_name=name,
                    summary=summary,
                    confidence=1.0,  # LLM-verified
                )
                for name, summary in resolved
            ]

        # Resolve all hints in parallel
        hint_results = await asyncio.gather(
            *[resolve_single_hint(hint) for hint in hints],
            return_exceptions=True
        )

        # Flatten results, skip exceptions
        results = []
        for r in hint_results:
            if isinstance(r, Exception):
                log(f"Entity resolution error: {r}")
            else:
                results.extend(r)

        # Dedupe by resolved_name
        seen = set()
        deduped = []
        for r in results:
            if r.resolved_name not in seen:
                seen.add(r.resolved_name)
                deduped.append(r)

        log(f"Resolved {len(hints)} hints to {len(deduped)} entities")
        return deduped

    async def resolve_topics(
        self,
        hints: list[str],
        question_context: str,
        threshold: float = 0.3,
        top_k_candidates: int = 20,
    ) -> list[ResolvedTopic]:
        """
        Resolve topic hints to graph topics with wide-net matching.

        Args:
            hints: Topic names/descriptions to resolve
            question_context: The original question for context
            threshold: Minimum vector similarity for candidates
            top_k_candidates: Max candidates to retrieve per hint

        Returns:
            List of ResolvedTopic with resolved_name and definition
        """
        if not hints:
            return []

        log(f"Resolving {len(hints)} topic hints")

        async def resolve_single_hint(hint: str) -> list[ResolvedTopic]:
            """Resolve a single hint to topics."""
            # Check cache first
            cache_key = hint.lower().strip()
            if cache_key in self._topic_cache:
                log(f"Topic cache hit for: {hint}")
                return [
                    ResolvedTopic(
                        original_hint=hint,
                        resolved_name=name,
                        definition=definition,
                        confidence=1.0,
                    )
                    for name, definition in self._topic_cache[cache_key]
                ]

            # Get candidates via vector search
            candidates = await self._get_topic_candidates(hint, top_k_candidates, threshold)

            if not candidates:
                log(f"No topic candidates for: {hint}")
                self._topic_cache[cache_key] = []  # Cache empty result
                return []

            # LLM picks matching candidates
            resolved = await self._resolve_with_llm(
                hint=hint,
                candidates=candidates,
                question=question_context,
                system_prompt=TOPIC_RESOLUTION_SYSTEM_PROMPT,
            )

            # Cache result
            self._topic_cache[cache_key] = resolved

            return [
                ResolvedTopic(
                    original_hint=hint,
                    resolved_name=name,
                    definition=definition,
                    confidence=1.0,
                )
                for name, definition in resolved
            ]

        # Resolve all hints in parallel
        hint_results = await asyncio.gather(
            *[resolve_single_hint(hint) for hint in hints],
            return_exceptions=True
        )

        # Flatten results, skip exceptions
        results = []
        for r in hint_results:
            if isinstance(r, Exception):
                log(f"Topic resolution error: {r}")
            else:
                results.extend(r)

        # Dedupe by resolved_name
        seen = set()
        deduped = []
        for r in results:
            if r.resolved_name not in seen:
                seen.add(r.resolved_name)
                deduped.append(r)

        log(f"Resolved {len(hints)} hints to {len(deduped)} topics")
        return deduped

    async def _get_entity_candidates(
        self,
        embed_text: str,
        top_k: int,
        threshold: float,
    ) -> list[tuple[str, str, float]]:
        """
        Vector search for entity candidates.

        Returns:
            List of (name, summary, score) tuples
        """
        embedding = await self.embed_text(embed_text)

        def _query():
            results = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('entity_name_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold
                RETURN node.name as name, node.summary as summary, score
                ORDER BY score DESC
                """,
                {
                    "vec": embedding,
                    "uid": self.group_id,
                    "top_k": top_k,
                    "threshold": threshold,
                }
            )
            return [(r["name"], r.get("summary") or "", r["score"]) for r in results]

        return await asyncio.to_thread(_query)

    async def _get_topic_candidates(
        self,
        embed_text: str,
        top_k: int,
        threshold: float,
    ) -> list[tuple[str, str, float]]:
        """
        Vector search for topic candidates.

        Returns:
            List of (name, definition, score) tuples
        """
        embedding = await self.embed_text(embed_text)

        def _query():
            results = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('topic_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold
                RETURN node.name as name, score
                ORDER BY score DESC
                """,
                {
                    "vec": embedding,
                    "uid": self.group_id,
                    "top_k": top_k,
                    "threshold": threshold,
                }
            )
            # Definition field not currently populated in Neo4j TopicNodes
            return [(r["name"], "", r["score"]) for r in results]

        return await asyncio.to_thread(_query)

    async def _resolve_with_llm(
        self,
        hint: str,
        candidates: list[tuple[str, str, float]],
        question: str,
        system_prompt: str,
    ) -> list[tuple[str, str]]:
        """
        Use LLM to select matching candidates from vector search results.

        Returns:
            List of (name, summary/definition) tuples for matched candidates
        """
        if not candidates:
            return []

        # Format candidates for prompt
        candidates_text = format_candidates_for_resolution(candidates)

        prompt = RESOLUTION_USER_PROMPT.format(
            question=question,
            term=hint,
            candidates=candidates_text,
        )

        try:
            result = await asyncio.to_thread(
                self.structured_resolver.invoke,
                [
                    ("system", system_prompt),
                    ("human", prompt),
                ]
            )

            if result.no_match:
                log(f"LLM found no matches for '{hint}'")
                return []

            # Build name -> summary map from candidates
            candidate_map = {name: summary for name, summary, _ in candidates}

            # Return matched names with their summaries
            matched = []
            for node in result.resolved_nodes:
                if node.name in candidate_map:
                    matched.append((node.name, candidate_map[node.name]))
                else:
                    # Try case-insensitive match
                    for cand_name, cand_summary, _ in candidates:
                        if cand_name.lower() == node.name.lower():
                            matched.append((cand_name, cand_summary))
                            break

            log(f"Resolved '{hint}' -> {[m[0] for m in matched]}")
            return matched

        except Exception as e:
            log(f"Resolution error for '{hint}': {e}")
            # Fallback to top 3 candidates by score
            return [(name, summary) for name, summary, _ in candidates[:3]]

    # =========================================================================
    # Chunk Retrieval
    # =========================================================================

    async def get_entity_chunks(
        self,
        entity_name: str,
        query_embedding: list[float],
        threshold: float = 0.3,
        top_k: int = 50,
    ) -> list[RetrievedChunk]:
        """
        Get chunks connected to an entity, ranked by query relevance.

        Uses the chunk-centric pattern where entities connect through EpisodicNodes.
        """
        log(f"Getting chunks for entity: {entity_name}")

        def _query():
            # Get chunks where entity appears (as subject or object of any relationship)
            results = self.services.neo4j.query(
                """
                // Find chunks connected to entity via any relationship
                MATCH (e:EntityNode {name: $entity_name, group_id: $uid})
                      -[r]->(c:EpisodicNode {group_id: $uid})
                WHERE r.fact_id IS NOT NULL

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                WITH c, d, collect(DISTINCT type(r)) as rel_types

                // Score chunks by embedding similarity
                CALL (c) {
                    CALL db.index.vector.queryNodes('fact_embeddings', 1000, $vec)
                    YIELD node, score
                    WHERE node.group_id = $uid
                    WITH node, score
                    WHERE EXISTS {
                        MATCH (subj)-[rel {fact_id: node.uuid}]->(c2:EpisodicNode {uuid: c.uuid})
                    }
                    RETURN max(score) as chunk_score
                }

                WITH c, d, rel_types, chunk_score
                WHERE chunk_score IS NULL OR chunk_score > $threshold

                RETURN DISTINCT
                    c.uuid as chunk_id,
                    c.content as content,
                    c.header_path as header_path,
                    d.name as doc_id,
                    d.document_date as document_date,
                    coalesce(chunk_score, 0.5) as score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                {
                    "entity_name": entity_name,
                    "uid": self.group_id,
                    "vec": query_embedding,
                    "threshold": threshold,
                    "top_k": top_k,
                }
            )
            return results

        try:
            results = await asyncio.to_thread(_query)
        except Exception as e:
            log(f"Entity chunk query failed, using fallback: {e}")
            # Fallback: simpler query without vector scoring in subquery
            results = await self._get_entity_chunks_simple(entity_name, top_k)

        chunks = self._results_to_chunks(results, source=f"entity:{entity_name}")
        log(f"Found {len(chunks)} chunks for entity {entity_name}")
        return chunks

    async def _get_entity_chunks_simple(
        self,
        entity_name: str,
        top_k: int,
    ) -> list[dict]:
        """Simpler fallback query for entity chunks without complex subquery."""
        def _query():
            results = self.services.neo4j.query(
                """
                MATCH (e:EntityNode {name: $entity_name, group_id: $uid})
                      -[r]->(c:EpisodicNode {group_id: $uid})
                WHERE r.fact_id IS NOT NULL

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT
                    c.uuid as chunk_id,
                    c.content as content,
                    c.header_path as header_path,
                    d.name as doc_id,
                    d.document_date as document_date,
                    0.5 as score
                LIMIT $top_k
                """,
                {
                    "entity_name": entity_name,
                    "uid": self.group_id,
                    "top_k": top_k,
                }
            )
            return results

        return await asyncio.to_thread(_query)

    async def get_topic_chunks(
        self,
        topic_name: str,
        top_k: int = 30,
    ) -> list[RetrievedChunk]:
        """
        Get chunks that discuss a specific topic.

        Topics connect to chunks via DISCUSSES relationship.
        """
        log(f"Getting chunks for topic: {topic_name}")

        def _query():
            results = self.services.neo4j.query(
                """
                MATCH (c:EpisodicNode {group_id: $uid})-[:DISCUSSES]->(t:TopicNode {name: $topic_name, group_id: $uid})

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT
                    c.uuid as chunk_id,
                    c.content as content,
                    c.header_path as header_path,
                    d.name as doc_id,
                    d.document_date as document_date,
                    0.6 as score
                LIMIT $top_k
                """,
                {
                    "topic_name": topic_name,
                    "uid": self.group_id,
                    "top_k": top_k,
                }
            )
            return results

        results = await asyncio.to_thread(_query)
        chunks = self._results_to_chunks(results, source=f"topic:{topic_name}")
        log(f"Found {len(chunks)} chunks for topic {topic_name}")
        return chunks

    # =========================================================================
    # 1-Hop Expansion
    # =========================================================================

    async def get_1hop_neighbors(
        self,
        entity_name: str,
        max_neighbors: int = 10,
    ) -> list[RetrievedEntity]:
        """
        Get entities connected to this entity via any relationship.

        Returns neighboring entities for potential expansion.
        """
        log(f"Getting 1-hop neighbors for: {entity_name}")

        def _query():
            results = self.services.neo4j.query(
                """
                // Get neighbors where entity is subject
                MATCH (e:EntityNode {name: $entity_name, group_id: $uid})
                      -[r]->(c:EpisodicNode {group_id: $uid})
                      -[r2]->(neighbor:EntityNode {group_id: $uid})
                WHERE r.fact_id = r2.fact_id AND neighbor.name <> $entity_name

                RETURN DISTINCT
                    neighbor.name as name,
                    neighbor.summary as summary,
                    count(*) as connection_count

                UNION

                // Get neighbors where entity is object
                MATCH (neighbor:EntityNode {group_id: $uid})
                      -[r]->(c:EpisodicNode {group_id: $uid})
                      -[r2]->(e:EntityNode {name: $entity_name, group_id: $uid})
                WHERE r.fact_id = r2.fact_id AND neighbor.name <> $entity_name

                RETURN DISTINCT
                    neighbor.name as name,
                    neighbor.summary as summary,
                    count(*) as connection_count

                ORDER BY connection_count DESC
                LIMIT $max_neighbors
                """,
                {
                    "entity_name": entity_name,
                    "uid": self.group_id,
                    "max_neighbors": max_neighbors,
                }
            )
            return results

        results = await asyncio.to_thread(_query)

        neighbors = []
        seen = set()
        for r in results:
            name = r.get("name")
            if name and name not in seen:
                seen.add(name)
                neighbors.append(RetrievedEntity(
                    name=name,
                    summary=r.get("summary") or "",
                    entity_type="ENTITY",  # EntityNode doesn't store type
                ))

        log(f"Found {len(neighbors)} neighbors for {entity_name}")
        return neighbors

    async def get_neighbor_chunks(
        self,
        neighbor_names: list[str],
        query_embedding: list[float],
        top_k_per_neighbor: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Get chunks from neighbor entities, ranked by query relevance.

        Used after get_1hop_neighbors to retrieve relevant content from neighbors.
        """
        if not neighbor_names:
            return []

        log(f"Getting chunks from {len(neighbor_names)} neighbors")

        # Query chunks for each neighbor in parallel
        async def _get_neighbor_chunks(name: str) -> list[RetrievedChunk]:
            return await self.get_entity_chunks(
                entity_name=name,
                query_embedding=query_embedding,
                threshold=0.25,
                top_k=top_k_per_neighbor,
            )

        tasks = [_get_neighbor_chunks(name) for name in neighbor_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_chunks = []
        for result in results:
            if isinstance(result, Exception):
                log(f"Neighbor chunk error: {result}")
                continue
            all_chunks.extend(result)

        # Dedupe by chunk_id, keep highest score
        chunk_map = {}
        for chunk in all_chunks:
            if chunk.chunk_id not in chunk_map or chunk.vector_score > chunk_map[chunk.chunk_id].vector_score:
                chunk_map[chunk.chunk_id] = chunk

        chunks = sorted(chunk_map.values(), key=lambda c: c.vector_score, reverse=True)
        log(f"Found {len(chunks)} chunks from neighbors")
        return chunks

    # =========================================================================
    # FactNode Retrieval
    # =========================================================================

    async def get_entity_facts(
        self,
        entity_name: str,
        query_embedding: list[float],
        threshold: float = 0.3,
        top_k: int = 30,
    ) -> list[RetrievedFact]:
        """
        Get facts involving a specific entity, ranked by query relevance.

        Uses the chunk-centric pattern:
        (Entity) -[r1 {fact_id}]-> (EpisodicNode) -[r2 {fact_id}]-> (Target)
        """
        log(f"Getting facts for entity: {entity_name}")

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

                RETURN DISTINCT
                    node.uuid as fact_id,
                    node.content as content,
                    subj.name as subject,
                    type(r1) as edge_type,
                    obj.name as object,
                    c.uuid as chunk_id,
                    score
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

                RETURN DISTINCT
                    node.uuid as fact_id,
                    node.content as content,
                    subj.name as subject,
                    type(r1) as edge_type,
                    obj.name as object,
                    c.uuid as chunk_id,
                    score
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

        # Convert to RetrievedFact and dedupe
        facts = self._results_to_facts(results)
        log(f"Found {len(facts)} facts for entity {entity_name}")
        return facts

    # =========================================================================
    # Global Search
    # =========================================================================

    async def global_chunk_search(
        self,
        query_embedding: list[float],
        top_k: int = 50,
        threshold: float = 0.25,
    ) -> list[RetrievedChunk]:
        """
        Global vector search across all chunks (via fact embeddings).

        Used as a fallback when entity/topic resolution yields few results.
        """
        log(f"Global chunk search (top_k={top_k})")

        def _query():
            results = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                // Get the chunk containing this fact
                MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
                WHERE subj.group_id = $uid

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT
                    c.uuid as chunk_id,
                    c.content as content,
                    c.header_path as header_path,
                    d.name as doc_id,
                    d.document_date as document_date,
                    max(score) as score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                {
                    "vec": query_embedding,
                    "uid": self.group_id,
                    "threshold": threshold,
                    "top_k": top_k * 2,  # Get more facts to aggregate to chunks
                }
            )
            return results

        results = await asyncio.to_thread(_query)
        chunks = self._results_to_chunks(results, source="global")
        log(f"Global search found {len(chunks)} chunks")
        return chunks

    # =========================================================================
    # Embedding
    # =========================================================================

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
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

    def _results_to_chunks(
        self,
        results: list[dict],
        source: str,
    ) -> list[RetrievedChunk]:
        """Convert Neo4j results to RetrievedChunk objects, deduped by chunk_id."""
        seen = set()
        chunks = []

        for r in results:
            chunk_id = r.get("chunk_id")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)

            content = r.get("content")
            if not content:
                continue

            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=content,
                header_path=r.get("header_path") or "",
                doc_id=r.get("doc_id") or "",
                document_date=r.get("document_date") or "",
                vector_score=r.get("score") or 0.0,
                source=source,
            ))

        return chunks

    def _results_to_facts(self, results: list[dict]) -> list[RetrievedFact]:
        """Convert Neo4j results to RetrievedFact objects, deduped by fact_id."""
        seen = set()
        facts = []

        for r in results:
            fact_id = r.get("fact_id")
            if not fact_id or fact_id in seen:
                continue
            seen.add(fact_id)

            facts.append(RetrievedFact(
                fact_id=fact_id,
                content=r.get("content") or "",
                subject=r.get("subject") or "",
                edge_type=r.get("edge_type") or "",
                object=r.get("object") or "",
                chunk_id=r.get("chunk_id") or "",
                vector_score=r.get("score") or 0.0,
            ))

        return facts
