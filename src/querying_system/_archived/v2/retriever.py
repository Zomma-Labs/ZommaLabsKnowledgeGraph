"""
V2 Retriever: GNN-inspired scoped + global retrieval.

Architecture (from GNN perspective):
1. Scoped Search: For each resolved node, similarity search around that node
   using the question embedding. This is like message passing from neighbors.
2. Global Search: Vector + keyword search on all facts using the question.
   This captures broader context not connected to resolved entities.

Both searches return facts that will be scored separately by the pipeline.
"""

import asyncio
import os
from dataclasses import dataclass, field

from src.querying_system.shared.schemas import QueryDecomposition, ScoredFact
from src.util.services import get_services
from src.util.deterministic_retrieval import extract_keywords
from src.util.fact_vector_store import get_fact_store
from .resolver import ResolvedEntities

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[RetrieverV2] {msg}")


@dataclass
class ScopedResult:
    """Facts from scoped search around a resolved node."""
    node_name: str
    node_type: str  # "entity" or "topic"
    facts: list[ScoredFact] = field(default_factory=list)
    unique_connected_entities: list[str] = field(default_factory=list)  # First N unique entities from facts


@dataclass
class RetrievalResult:
    """Separates scoped and global results for independent scoring."""
    scoped_results: list[ScopedResult]  # Per-node results
    global_facts: list[ScoredFact]      # From global search

    @property
    def all_scoped_facts(self) -> list[ScoredFact]:
        """Flatten all scoped facts."""
        facts = []
        seen = set()
        for result in self.scoped_results:
            for fact in result.facts:
                if fact.fact_id not in seen:
                    facts.append(fact)
                    seen.add(fact.fact_id)
        return facts


class GNNRetrieverV2:
    """
    GNN-inspired retriever with separate scoped and global searches.

    Key insight: Scoped search (around resolved nodes) gives high-precision
    facts. Global search gives coverage. Both are scored separately to
    let the LLM decide what's useful.
    """

    def __init__(self, group_id: str = "default"):
        self.group_id = group_id
        self.services = get_services()

    async def retrieve_scoped(
        self,
        resolved: ResolvedEntities,
        question: str,
        top_k_per_node: int = 20,
        top_n_unique_entities: int = 10,
    ) -> list[ScopedResult]:
        """
        Phase 3a: Scoped search around each resolved node.

        For each entity/topic node:
        1. Embed the question (provides semantic context for what we're looking for)
        2. Search facts connected to that node, ranked by question similarity

        This is like GNN message passing - gathering info from neighbors.
        """
        if not resolved.entity_nodes and not resolved.topic_nodes:
            log("No resolved nodes for scoped search")
            return []

        # Embed the question once (used for all searches)
        question_embedding = await asyncio.to_thread(
            self.services.embeddings.embed_query, question
        )

        # Build tasks and metadata for parallel execution
        tasks = []
        node_info: list[tuple[str, str]] = []  # (name, type)

        for entity_name in resolved.entity_nodes:
            log(f"Queuing scoped search for entity: {entity_name}")
            tasks.append(self._search_node_neighbors(
                node_name=entity_name,
                query_embedding=question_embedding,
                top_k=top_k_per_node,
            ))
            node_info.append((entity_name, "entity"))

        for topic_name in resolved.topic_nodes:
            log(f"Queuing scoped search for topic: {topic_name}")
            tasks.append(self._search_node_neighbors(
                node_name=topic_name,
                query_embedding=question_embedding,
                top_k=top_k_per_node,
            ))
            node_info.append((topic_name, "topic"))

        # Run ALL searches in parallel
        log(f"Running {len(tasks)} scoped searches in parallel...")
        all_facts_lists = await asyncio.gather(*tasks)

        # Build results from parallel execution
        results: list[ScopedResult] = []
        for (node_name, node_type), facts in zip(node_info, all_facts_lists):
            # Extract unique connected entities (first N unique subjects/objects)
            unique_entities = self._extract_unique_entities(
                facts, exclude_node=node_name, top_n=top_n_unique_entities
            )
            results.append(ScopedResult(
                node_name=node_name,
                node_type=node_type,
                facts=facts,
                unique_connected_entities=unique_entities,
            ))

        total_facts = sum(len(r.facts) for r in results)
        log(f"Scoped search: {len(results)} nodes, {total_facts} total facts")
        return results

    def _extract_unique_entities(
        self,
        facts: list[ScoredFact],
        exclude_node: str,
        top_n: int = 10,
    ) -> list[str]:
        """
        Extract first N unique entities from facts in ranking order.

        For enumeration questions, this provides a quick view of
        "which entities are connected to this node".
        """
        seen = set()
        unique = []
        exclude_lower = exclude_node.lower()

        for fact in facts:
            # Check subject
            if fact.subject and fact.subject.lower() != exclude_lower:
                if fact.subject not in seen:
                    seen.add(fact.subject)
                    unique.append(fact.subject)
                    if len(unique) >= top_n:
                        break

            # Check object
            if fact.object and fact.object.lower() != exclude_lower:
                if fact.object not in seen:
                    seen.add(fact.object)
                    unique.append(fact.object)
                    if len(unique) >= top_n:
                        break

        return unique

    async def retrieve_global(
        self,
        question: str,
        decomposition: QueryDecomposition,
        top_k: int = 20,
    ) -> list[ScoredFact]:
        """
        Phase 5: Global search for broader context.

        Uses both vector and keyword search on all facts.
        Captures information not connected to resolved entities.
        """
        log("Global search starting...")

        # Embed the question
        question_embedding = await asyncio.to_thread(
            self.services.embeddings.embed_query, question
        )

        # Run vector and keyword searches in parallel
        vector_task = self._global_vector_search(question_embedding, top_k)
        keyword_task = self._global_keyword_search(question, top_k)

        vector_results, keyword_results = await asyncio.gather(
            vector_task, keyword_task
        )

        # Merge and deduplicate
        all_facts: dict[str, ScoredFact] = {}

        for fact in vector_results:
            if fact.fact_id not in all_facts:
                fact.found_by_queries.append("global_vector")
                all_facts[fact.fact_id] = fact

        for fact in keyword_results:
            if fact.fact_id not in all_facts:
                fact.found_by_queries.append("global_keyword")
                all_facts[fact.fact_id] = fact
            else:
                # Boost facts found by both methods
                all_facts[fact.fact_id].found_by_queries.append("global_keyword")
                all_facts[fact.fact_id].cross_query_boost = 0.2

        log(f"Global search: {len(all_facts)} unique facts")
        return list(all_facts.values())

    async def _search_node_neighbors(
        self,
        node_name: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[ScoredFact]:
        """
        Search facts connected to a node, ranked by question similarity.

        Uses Qdrant fact vector store for semantic ranking.
        Falls back to Neo4j graph traversal if Qdrant unavailable.
        """
        try:
            fact_store = get_fact_store()
            results = fact_store.search_facts_for_entity(
                entity_name=node_name,
                query_embedding=query_embedding,
                group_id=self.group_id,
                top_k=top_k,
            )

            facts = []
            for r in results:
                fact = ScoredFact(
                    fact_id=r["fact_id"],
                    content=r["content"],
                    subject=r["subject"],
                    edge_type=r["edge_type"],
                    object=r["object"],
                    vector_score=r["score"],
                    rrf_score=r["score"],
                )
                fact.found_by_queries.append(f"scoped:{node_name}")
                facts.append(fact)

            return facts

        except Exception as e:
            log(f"Qdrant search failed for {node_name}: {e}, falling back to Neo4j")
            return await self._neo4j_node_search(node_name, top_k)

    async def _neo4j_node_search(
        self, node_name: str, top_k: int
    ) -> list[ScoredFact]:
        """Fallback: Get facts from Neo4j graph traversal."""
        def _query():
            return self.services.neo4j.query(
                """
                MATCH (n {name: $node_name, group_id: $uid})
                WHERE n:EntityNode OR n:TopicNode

                MATCH (n)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(target)
                WHERE r1.fact_id = r2.fact_id AND (target:EntityNode OR target:TopicNode)

                MATCH (f:FactNode {uuid: r1.fact_id, group_id: $uid})

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT f.uuid as fact_id,
                       f.content as content,
                       1.0 as score,
                       n.name as subject,
                       type(r1) as edge_type,
                       target.name as object,
                       c.uuid as chunk_id,
                       c.content as chunk_content,
                       c.header_path as chunk_header,
                       d.name as doc_id,
                       d.document_date as document_date
                LIMIT $top_k
                """,
                {"node_name": node_name, "uid": self.group_id, "top_k": top_k},
            )

        results = await asyncio.to_thread(_query)

        facts = []
        for r in results:
            fact = ScoredFact(
                fact_id=r.get("fact_id", ""),
                content=r.get("content", ""),
                subject=r.get("subject", ""),
                edge_type=r.get("edge_type", ""),
                object=r.get("object", ""),
                chunk_id=r.get("chunk_id"),
                chunk_content=r.get("chunk_content"),
                chunk_header=r.get("chunk_header"),
                doc_id=r.get("doc_id"),
                document_date=r.get("document_date"),
                vector_score=1.0,
                rrf_score=1.0,
            )
            fact.found_by_queries.append(f"scoped:{node_name}")
            facts.append(fact)

        return facts

    async def _global_vector_search(
        self, embedding: list[float], top_k: int
    ) -> list[ScoredFact]:
        """Global vector search on fact embeddings."""
        def _query():
            return self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > 0.25

                OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})-[r2 {fact_id: node.uuid}]->(obj)
                WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)

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
                """,
                {"vec": embedding, "uid": self.group_id, "top_k": top_k},
            )

        results = await asyncio.to_thread(_query)

        return [
            ScoredFact(
                fact_id=r.get("fact_id", ""),
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
            for r in results
        ]

    async def _global_keyword_search(
        self, question: str, top_k: int
    ) -> list[ScoredFact]:
        """Global keyword search on fact content."""
        keywords = extract_keywords(question)
        if not keywords:
            return []

        def _query():
            return self.services.neo4j.query(
                """
                CALL db.index.fulltext.queryNodes('fact_fulltext', $keywords)
                YIELD node, score
                WHERE node.group_id = $uid

                OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})-[r2 {fact_id: node.uuid}]->(obj)
                WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)

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
                """,
                {"keywords": keywords, "uid": self.group_id, "top_k": top_k},
            )

        results = await asyncio.to_thread(_query)

        return [
            ScoredFact(
                fact_id=r.get("fact_id", ""),
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
            for r in results
        ]
