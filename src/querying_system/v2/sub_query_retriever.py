"""
V2 Sub-Query Parallel Retriever.

Implements Deep Research pattern: spawn parallel mini-retrievers per sub_query,
combining results with cross-query boosting.

Key design:
1. Each sub_query gets its own resolution context (more targeted matching)
2. Threshold-based retrieval (sim > 0.7) instead of top_k limits
3. Facts found by multiple sub-queries get boosted
4. Provenance tracking via found_by_queries
"""

import asyncio
import os
import time
from typing import Union

from src.querying_system.shared.schemas import (
    SubQuery,
    ScoredFact,
    SubQueryResult,
    ParallelRetrievalResult,
    EntityHint,
)
from src.util.services import get_services
from .resolver import Resolver, ResolvedEntities

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

# Cross-query boost for facts found by multiple sub-queries
CROSS_QUERY_BOOST = 0.15

# Similarity threshold for retrieval
SIMILARITY_THRESHOLD = 0.7

# Max concurrent sub-query retrievers
MAX_CONCURRENCY = 5


def log(msg: str):
    if VERBOSE:
        print(f"[SubQueryRetriever] {msg}")


class SubQueryRetriever:
    """
    Mini-retriever for a single sub-query.

    Flow:
    1. Resolve entities using sub_query.query_text as context
    2. Threshold retrieve using sub_query.query_text embedding (sim > 0.7)
    3. Tag facts with provenance
    """

    def __init__(self, group_id: str = "default"):
        self.group_id = group_id
        self.services = get_services()
        self.resolver = Resolver(group_id=group_id)

    async def retrieve(
        self,
        sub_query: SubQuery,
        question: str,
    ) -> SubQueryResult:
        """
        Execute retrieval for a single sub-query.

        Args:
            sub_query: The sub-query to process
            question: Original question (for additional context)

        Returns:
            SubQueryResult with facts, resolved entities, and timing
        """
        start_time = time.time()
        log(f"Processing sub-query: {sub_query.query_text}")

        # Step 1: Resolve entities using sub_query.query_text as context
        resolution_start = time.time()

        # Convert entity_hints (list[str]) to EntityHint objects with definitions
        entity_hints = [
            EntityHint(
                name=hint,
                definition=f"Entity related to: {sub_query.target_info}"
            )
            for hint in sub_query.entity_hints
        ]

        resolved = await self.resolver.resolve(
            entity_hints=entity_hints,
            topic_hints=[],  # Sub-queries focus on explicit entity hints
            question=sub_query.query_text,  # Use sub_query as context, not full question
            top_k_candidates=15,
        )
        resolution_time_ms = int((time.time() - resolution_start) * 1000)

        log(f"Resolved {len(resolved.entity_nodes)} entities, {len(resolved.topic_nodes)} topics")

        # Step 2: Threshold-based retrieval using sub_query.query_text embedding
        retrieval_start = time.time()
        facts = await self._threshold_retrieve(
            resolved=resolved,
            query_text=sub_query.query_text,
        )
        retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

        # Step 3: Tag facts with provenance
        query_tag = f"subquery:{sub_query.query_text}"
        for fact in facts:
            fact.found_by_queries.append(query_tag)

        log(f"Retrieved {len(facts)} facts for sub-query in {retrieval_time_ms}ms")

        return SubQueryResult(
            sub_query_text=sub_query.query_text,
            target_info=sub_query.target_info,
            facts=facts,
            resolved_entities=resolved.entity_nodes,
            resolved_topics=resolved.topic_nodes,
            retrieval_time_ms=retrieval_time_ms,
            resolution_time_ms=resolution_time_ms,
        )

    async def _threshold_retrieve(
        self,
        resolved: ResolvedEntities,
        query_text: str,
    ) -> list[ScoredFact]:
        """
        Threshold-based retrieval: get ALL facts above similarity threshold.

        Args:
            resolved: Resolved entity and topic nodes
            query_text: Text to embed for similarity search

        Returns:
            List of facts above threshold, sorted by score
        """
        if not resolved.entity_nodes and not resolved.topic_nodes:
            return []

        # Embed the sub-query text (more targeted than full question)
        query_embedding = await asyncio.to_thread(
            self.services.embeddings.embed_query, query_text
        )

        # Build parallel search tasks
        tasks = []
        node_names = []

        for entity_name in resolved.entity_nodes:
            tasks.append(self._search_entity_facts(entity_name, query_embedding))
            node_names.append(entity_name)

        for topic_name in resolved.topic_nodes:
            tasks.append(self._search_topic_facts(topic_name, query_embedding))
            node_names.append(topic_name)

        # Execute in parallel
        all_facts_lists = await asyncio.gather(*tasks)

        # Merge and dedupe
        all_facts: dict[str, ScoredFact] = {}
        for node_name, facts in zip(node_names, all_facts_lists):
            for fact in facts:
                if fact.fact_id not in all_facts:
                    all_facts[fact.fact_id] = fact
                else:
                    # Fact found via multiple nodes - small boost
                    all_facts[fact.fact_id].cross_query_boost += 0.05

        # Sort by score
        sorted_facts = sorted(
            all_facts.values(),
            key=lambda f: f.vector_score + f.cross_query_boost,
            reverse=True
        )

        return sorted_facts

    async def _search_entity_facts(
        self,
        entity_name: str,
        query_embedding: list[float],
    ) -> list[ScoredFact]:
        """Search facts where entity is subject or object, above threshold."""
        def _query():
            # Query facts where entity is the subject
            results_subject = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', 500, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                MATCH (subj:EntityNode {name: $entity_name, group_id: $uid})
                      -[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
                      -[r2 {fact_id: node.uuid}]->(obj)
                WHERE obj:EntityNode OR obj:TopicNode

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT node.uuid as fact_id,
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
                    "threshold": SIMILARITY_THRESHOLD,
                    "entity_name": entity_name
                }
            )

            # Query facts where entity is the object
            results_object = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', 500, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
                      -[r2 {fact_id: node.uuid}]->(obj:EntityNode {name: $entity_name, group_id: $uid})
                WHERE subj:EntityNode OR subj:TopicNode

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT node.uuid as fact_id,
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
                    "threshold": SIMILARITY_THRESHOLD,
                    "entity_name": entity_name
                }
            )

            return results_subject + results_object

        results = await asyncio.to_thread(_query)

        # Dedupe by fact_id
        seen = {}
        for r in results:
            fact_id = r.get("fact_id", "")
            if not fact_id:
                continue
            if fact_id not in seen or r.get("score", 0) > seen[fact_id].vector_score:
                seen[fact_id] = ScoredFact(
                    fact_id=fact_id,
                    content=r.get("content", ""),
                    subject=r.get("subject", ""),
                    edge_type=r.get("edge_type", ""),
                    object=r.get("object", ""),
                    chunk_id=r.get("chunk_id"),
                    chunk_content=r.get("chunk_content"),
                    chunk_header=r.get("chunk_header"),
                    doc_id=r.get("doc_id"),
                    document_date=r.get("document_date"),
                    vector_score=r.get("score", 0.0),
                    rrf_score=r.get("score", 0.0),
                )

        return list(seen.values())

    async def _search_topic_facts(
        self,
        topic_name: str,
        query_embedding: list[float],
    ) -> list[ScoredFact]:
        """Search facts where topic is subject or object, above threshold."""
        def _query():
            # Query facts where topic is subject
            results_subject = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', 500, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                MATCH (subj:TopicNode {name: $topic_name, group_id: $uid})
                      -[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
                      -[r2 {fact_id: node.uuid}]->(obj)
                WHERE obj:EntityNode OR obj:TopicNode

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT node.uuid as fact_id,
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
                    "threshold": SIMILARITY_THRESHOLD,
                    "topic_name": topic_name
                }
            )

            # Query facts where topic is object
            results_object = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', 500, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
                      -[r2 {fact_id: node.uuid}]->(obj:TopicNode {name: $topic_name, group_id: $uid})
                WHERE subj:EntityNode OR subj:TopicNode

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT node.uuid as fact_id,
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
                    "threshold": SIMILARITY_THRESHOLD,
                    "topic_name": topic_name
                }
            )

            return results_subject + results_object

        results = await asyncio.to_thread(_query)

        # Dedupe by fact_id
        seen = {}
        for r in results:
            fact_id = r.get("fact_id", "")
            if not fact_id:
                continue
            if fact_id not in seen or r.get("score", 0) > seen[fact_id].vector_score:
                seen[fact_id] = ScoredFact(
                    fact_id=fact_id,
                    content=r.get("content", ""),
                    subject=r.get("subject", ""),
                    edge_type=r.get("edge_type", ""),
                    object=r.get("object", ""),
                    chunk_id=r.get("chunk_id"),
                    chunk_content=r.get("chunk_content"),
                    chunk_header=r.get("chunk_header"),
                    doc_id=r.get("doc_id"),
                    document_date=r.get("document_date"),
                    vector_score=r.get("score", 0.0),
                    rrf_score=r.get("score", 0.0),
                )

        return list(seen.values())


class ParallelSubQueryOrchestrator:
    """
    Orchestrates parallel sub-query execution.

    Flow:
    1. Spawn SubQueryRetriever per sub_query (semaphore-limited)
    2. Gather all results
    3. Combine with deduplication + cross-query boosting
    """

    def __init__(self, group_id: str = "default"):
        self.group_id = group_id
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def retrieve(
        self,
        sub_queries: list[SubQuery],
        question: str,
    ) -> ParallelRetrievalResult:
        """
        Execute parallel retrieval for all sub-queries.

        Args:
            sub_queries: List of sub-queries from the splitter
            question: Original question

        Returns:
            ParallelRetrievalResult with combined facts
        """
        start_time = time.time()

        if not sub_queries:
            log("No sub-queries to process")
            return ParallelRetrievalResult()

        log(f"Orchestrating {len(sub_queries)} parallel sub-query retrievals...")

        # Create tasks for each sub-query
        tasks = [
            self._process_sub_query(sq, question)
            for sq in sub_queries
        ]

        # Execute in parallel (semaphore-limited)
        sub_query_results = await asyncio.gather(*tasks)

        # Combine results with deduplication and cross-query boosting
        combined_facts, boosted_ids = self._combine_results(sub_query_results)

        total_time_ms = int((time.time() - start_time) * 1000)

        log(f"Parallel retrieval complete: {len(combined_facts)} unique facts, "
            f"{len(boosted_ids)} cross-query boosted, {total_time_ms}ms total")

        return ParallelRetrievalResult(
            sub_query_results=list(sub_query_results),
            combined_facts=combined_facts,
            cross_query_boosted_fact_ids=boosted_ids,
            total_retrieval_time_ms=total_time_ms,
        )

    async def _process_sub_query(
        self,
        sub_query: SubQuery,
        question: str,
    ) -> SubQueryResult:
        """Process a single sub-query with semaphore limiting."""
        async with self.semaphore:
            retriever = SubQueryRetriever(group_id=self.group_id)
            return await retriever.retrieve(sub_query, question)

    def _combine_results(
        self,
        sub_query_results: tuple[SubQueryResult, ...],
    ) -> tuple[list[ScoredFact], set[str]]:
        """
        Combine facts from all sub-queries with deduplication and boosting.

        Facts found by multiple sub-queries get a cross-query boost.

        Returns:
            (combined_facts, set of boosted fact_ids)
        """
        # Track fact occurrences
        fact_occurrences: dict[str, list[ScoredFact]] = {}

        for result in sub_query_results:
            for fact in result.facts:
                if fact.fact_id not in fact_occurrences:
                    fact_occurrences[fact.fact_id] = []
                fact_occurrences[fact.fact_id].append(fact)

        # Merge and apply cross-query boost
        combined: dict[str, ScoredFact] = {}
        boosted_ids: set[str] = set()

        for fact_id, occurrences in fact_occurrences.items():
            # Use the occurrence with highest vector score as base
            best = max(occurrences, key=lambda f: f.vector_score)

            # Merge found_by_queries from all occurrences
            all_queries = set()
            for occ in occurrences:
                all_queries.update(occ.found_by_queries)
            best.found_by_queries = list(all_queries)

            # Apply cross-query boost if found by multiple sub-queries
            num_sub_queries = len([q for q in all_queries if q.startswith("subquery:")])
            if num_sub_queries > 1:
                boost = CROSS_QUERY_BOOST * (num_sub_queries - 1)
                best.cross_query_boost += boost
                boosted_ids.add(fact_id)
                log(f"Cross-query boost +{boost:.2f} for fact found by {num_sub_queries} sub-queries")

            combined[fact_id] = best

        # Sort by total score
        sorted_facts = sorted(
            combined.values(),
            key=lambda f: f.vector_score + f.cross_query_boost,
            reverse=True
        )

        return sorted_facts, boosted_ids
