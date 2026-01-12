"""
Phase 2c: Conditional Graph Expansion.

1-hop expansion from entities marked with should_expand=True.
Only triggered for CAUSAL and COMPARISON question types.
"""

import asyncio
import os

from .schemas import ScoredFact, EvidencePool, QuestionType, QueryDecomposition
from src.util.services import get_services

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[GraphExpander] {msg}")


class GraphExpander:
    """
    Conditional 1-hop graph expander.

    Expands from entities in high-scoring facts that have should_expand=True.
    This helps find causal chains and comparison dimensions.

    Example:
        Initial: Tariffs -[INCREASED]-> Cost Structure
        Expansion from "Cost Structure" finds:
            Cost Structure -[CONTRIBUTED_TO_LIMITING]-> Employment
        Result: Complete causal chain Tariffs -> Cost Structure -> Employment
    """

    def __init__(self, group_id: str = "default"):
        self.group_id = group_id
        self.services = get_services()

    async def expand(
        self,
        evidence_pool: EvidencePool,
        decomposition: QueryDecomposition,
        top_k_per_entity: int = 5,
        max_entities_to_expand: int = 5,
    ) -> EvidencePool:
        """
        Perform 1-hop expansion from should_expand facts.

        Now runs for ANY question type - the LLM scorer controls expansion
        via the should_expand flag on individual facts.

        Args:
            evidence_pool: Current evidence pool
            decomposition: Query decomposition for context
            top_k_per_entity: Facts to retrieve per expanded entity
            max_entities_to_expand: Limit entities to expand

        Returns:
            Updated EvidencePool with expansion results
        """
        # Expansion is now LLM-controlled via should_expand flag
        # No longer restricted by question type
        log(f"Checking expansion for {decomposition.question_type.value}")

        # Find entities to expand from
        entities_to_expand = set()
        for fact in evidence_pool.scored_facts:
            if fact.should_expand and fact.final_score > 0.3:
                # Add both subject and object as candidates
                if fact.subject:
                    entities_to_expand.add(fact.subject)
                if fact.object:
                    entities_to_expand.add(fact.object)

        # Limit expansion
        entities_to_expand = list(entities_to_expand)[:max_entities_to_expand]

        if not entities_to_expand:
            log("No entities marked for expansion")
            return evidence_pool

        log(f"Expanding from {len(entities_to_expand)} entities: {entities_to_expand}")

        # Parallel 1-hop expansion
        expansion_tasks = [
            self._expand_entity(entity, top_k_per_entity)
            for entity in entities_to_expand
        ]

        results = await asyncio.gather(*expansion_tasks)

        # Add new facts to pool
        existing_fact_ids = {f.fact_id for f in evidence_pool.scored_facts}
        new_facts = []

        for entity_facts in results:
            for fact in entity_facts:
                if fact.fact_id and fact.fact_id not in existing_fact_ids:
                    # Mark as expansion result with lower base score
                    fact.found_by_queries.append("graph_expansion")
                    fact.rrf_score = 0.3  # Lower base score
                    fact.final_score = 0.3
                    new_facts.append(fact)
                    existing_fact_ids.add(fact.fact_id)

        log(f"Expansion added {len(new_facts)} new facts")

        evidence_pool.scored_facts.extend(new_facts)
        evidence_pool.expansion_performed = True

        return evidence_pool

    async def _expand_entity(
        self, entity_name: str, top_k: int
    ) -> list[ScoredFact]:
        """Get 1-hop neighbors from an entity."""

        def _query():
            return self.services.neo4j.query(
                """
                // Outgoing relationships
                MATCH (e {name: $entity_name, group_id: $uid})-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(target)
                WHERE (e:EntityNode OR e:TopicNode) AND (target:EntityNode OR target:TopicNode)
                  AND r1.fact_id = r2.fact_id

                OPTIONAL MATCH (f:FactNode {uuid: r1.fact_id, group_id: $uid})
                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT r1.fact_id as fact_id,
                       COALESCE(f.content, e.name + ' ' + type(r1) + ' ' + target.name) as content,
                       0.3 as score,
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
                MATCH (source)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(e {name: $entity_name, group_id: $uid})
                WHERE (e:EntityNode OR e:TopicNode) AND (source:EntityNode OR source:TopicNode)
                  AND r1.fact_id = r2.fact_id

                OPTIONAL MATCH (f:FactNode {uuid: r1.fact_id, group_id: $uid})
                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT r1.fact_id as fact_id,
                       COALESCE(f.content, source.name + ' ' + type(r1) + ' ' + e.name) as content,
                       0.3 as score,
                       source.name as subject,
                       type(r1) as edge_type,
                       e.name as object,
                       c.uuid as chunk_id,
                       c.content as chunk_content,
                       c.header_path as chunk_header,
                       d.name as doc_id,
                       d.document_date as document_date

                LIMIT $top_k
                """,
                {
                    "entity_name": entity_name,
                    "uid": self.group_id,
                    "top_k": top_k,
                },
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
                vector_score=r.get("score", 0.3),
                rrf_score=0.3,
            )
            for r in results
        ]
