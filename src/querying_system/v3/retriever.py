"""
V3 Retriever: Threshold-based retrieval (no k-limit).

Key difference from V2:
- Instead of top_k per entity, retrieves ALL facts with similarity > threshold
- No artificial cutoff - gets everything semantically relevant
- Simpler, lets the scoring phase handle relevance ranking
"""

import asyncio
import os
from dataclasses import dataclass, field

from src.querying_system.shared.schemas import ScoredFact
from src.util.services import get_services
from src.querying_system.v2.resolver import ResolvedEntities

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[RetrieverV3] {msg}")


@dataclass
class RetrievalResultV3:
    """Facts retrieved via threshold-based search."""
    facts: list[ScoredFact] = field(default_factory=list)
    facts_per_entity: dict[str, int] = field(default_factory=dict)  # entity -> count


class ThresholdRetriever:
    """
    Threshold-based retriever: gets ALL facts above similarity threshold.

    No top_k limit - retrieves everything semantically relevant to the question
    for each resolved entity.
    """

    def __init__(self, group_id: str = "default", similarity_threshold: float = 0.7):
        self.group_id = group_id
        self.similarity_threshold = similarity_threshold
        self.services = get_services()

    async def retrieve(
        self,
        resolved: ResolvedEntities,
        question: str,
    ) -> RetrievalResultV3:
        """
        Retrieve all facts above similarity threshold for resolved entities.

        Args:
            resolved: Resolved entity and topic nodes from Phase 2
            question: Original question (used for embedding)

        Returns:
            RetrievalResultV3 with all facts above threshold
        """
        if not resolved.entity_nodes and not resolved.topic_nodes:
            log("No resolved nodes for retrieval")
            return RetrievalResultV3()

        # Embed the question once
        question_embedding = await asyncio.to_thread(
            self.services.embeddings.embed_query, question
        )

        # Build tasks for parallel execution
        tasks = []
        node_names = []

        for entity_name in resolved.entity_nodes:
            log(f"Queuing threshold search for entity: {entity_name}")
            tasks.append(self._search_entity_facts(
                entity_name=entity_name,
                query_embedding=question_embedding,
            ))
            node_names.append(entity_name)

        for topic_name in resolved.topic_nodes:
            log(f"Queuing threshold search for topic: {topic_name}")
            tasks.append(self._search_topic_facts(
                topic_name=topic_name,
                query_embedding=question_embedding,
            ))
            node_names.append(topic_name)

        # Run all searches in parallel
        log(f"Running {len(tasks)} threshold searches in parallel...")
        all_facts_lists = await asyncio.gather(*tasks)

        # Merge and dedupe
        all_facts: dict[str, ScoredFact] = {}
        facts_per_entity: dict[str, int] = {}

        for node_name, facts in zip(node_names, all_facts_lists):
            facts_per_entity[node_name] = len(facts)
            for fact in facts:
                if fact.fact_id not in all_facts:
                    all_facts[fact.fact_id] = fact
                else:
                    # Boost facts found via multiple entities
                    all_facts[fact.fact_id].cross_query_boost += 0.1
                    all_facts[fact.fact_id].found_by_queries.extend(fact.found_by_queries)

        # Sort by vector score descending
        sorted_facts = sorted(
            all_facts.values(),
            key=lambda f: f.vector_score + f.cross_query_boost,
            reverse=True
        )

        log(f"Retrieved {len(sorted_facts)} unique facts above threshold {self.similarity_threshold}")
        for node, count in facts_per_entity.items():
            log(f"  {node}: {count} facts")

        return RetrievalResultV3(
            facts=sorted_facts,
            facts_per_entity=facts_per_entity
        )

    async def _search_entity_facts(
        self,
        entity_name: str,
        query_embedding: list[float],
    ) -> list[ScoredFact]:
        """
        Search facts where entity is subject or object, filtered by similarity threshold.

        Uses Neo4j vector search with threshold filtering (no top_k limit).
        """
        def _query():
            # Query facts where entity is the subject
            results_subject = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', 500, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                // Find facts where this entity is the subject
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
                    "threshold": self.similarity_threshold,
                    "entity_name": entity_name
                }
            )

            # Query facts where entity is the object
            results_object = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', 500, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                // Find facts where this entity is the object
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
                    "threshold": self.similarity_threshold,
                    "entity_name": entity_name
                }
            )

            return results_subject + results_object

        results = await asyncio.to_thread(_query)

        # Dedupe by fact_id (might appear in both subject and object queries)
        seen = {}
        for r in results:
            fact_id = r.get("fact_id", "")
            if not fact_id:
                continue
            if fact_id not in seen or r.get("score", 0) > seen[fact_id].vector_score:
                fact = ScoredFact(
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
                fact.found_by_queries.append(f"entity:{entity_name}")
                seen[fact_id] = fact

        return list(seen.values())

    async def _search_topic_facts(
        self,
        topic_name: str,
        query_embedding: list[float],
    ) -> list[ScoredFact]:
        """
        Search facts where topic is subject or object, filtered by similarity threshold.

        TopicNodes work the same as EntityNodes - they connect to EpisodicNodes
        via fact relationships with fact_id properties.
        """
        def _query():
            # Query facts where topic is subject
            results_subject = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', 500, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                // Find facts where this topic is the subject
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
                    "threshold": self.similarity_threshold,
                    "topic_name": topic_name
                }
            )

            # Query facts where topic is object
            results_object = self.services.neo4j.query(
                """
                CALL db.index.vector.queryNodes('fact_embeddings', 500, $vec)
                YIELD node, score
                WHERE node.group_id = $uid AND score > $threshold

                // Find facts where this topic is the object
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
                    "threshold": self.similarity_threshold,
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
                fact = ScoredFact(
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
                fact.found_by_queries.append(f"topic:{topic_name}")
                seen[fact_id] = fact

        return list(seen.values())
