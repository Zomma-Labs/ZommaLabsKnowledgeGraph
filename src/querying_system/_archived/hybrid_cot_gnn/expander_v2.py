"""
Phase 2b: Graph Expansion from Scoped Results Only (V2).

Key difference from v1:
- Only expands from scoped_facts (entity+topic search results)
- Does NOT expand from global fallback results
- Expanded facts are returned separately for later LLM scoring
"""

import asyncio
import os

from .schemas import ScoredFact, QueryDecomposition
from src.util.services import get_services

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[GraphExpanderV2] {msg}")


class GraphExpanderV2:
    """
    Graph expander that only expands from scoped (high-confidence) facts.

    Key insight: Scoped facts are structurally connected to our entities/topics,
    so expanding from them follows meaningful graph paths. Global facts may not
    have meaningful graph connections.
    """

    def __init__(self, group_id: str = "default"):
        self.group_id = group_id
        self.services = get_services()

    async def expand_from_scoped(
        self,
        scoped_facts: dict[str, ScoredFact],
        decomposition: QueryDecomposition,
        top_k_per_entity: int = 5,
        max_entities_to_expand: int = 10,
    ) -> list[ScoredFact]:
        """
        Perform 1-hop expansion ONLY from scoped facts.

        Args:
            scoped_facts: Facts from entity+topic scoped search (high confidence)
            decomposition: Query decomposition for context
            top_k_per_entity: Facts to retrieve per expanded entity
            max_entities_to_expand: Limit entities to expand

        Returns:
            List of new facts found via expansion (NOT yet scored by LLM)
        """
        if not scoped_facts:
            log("No scoped facts to expand from")
            return []

        # Collect entities from scoped facts
        entities_to_expand = set()
        for fact in scoped_facts.values():
            if fact.subject:
                entities_to_expand.add(fact.subject)
            if fact.object:
                entities_to_expand.add(fact.object)

        entities_to_expand = list(entities_to_expand)[:max_entities_to_expand]

        if not entities_to_expand:
            log("No entities to expand from")
            return []

        log(f"Expanding from {len(entities_to_expand)} scoped entities: {entities_to_expand}")

        # Parallel 1-hop expansion
        expansion_tasks = [
            self._expand_entity(entity, top_k_per_entity)
            for entity in entities_to_expand
        ]

        results = await asyncio.gather(*expansion_tasks)

        # Collect new facts (not already in scoped)
        existing_fact_ids = set(scoped_facts.keys())
        new_facts = []

        for entity_facts in results:
            for fact in entity_facts:
                if fact.fact_id and fact.fact_id not in existing_fact_ids:
                    fact.found_by_queries.append("scoped_expansion")
                    new_facts.append(fact)
                    existing_fact_ids.add(fact.fact_id)

        log(f"Expansion found {len(new_facts)} new facts from scoped entities")

        return new_facts

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
                       0.5 as score,
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
                       0.5 as score,
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
                vector_score=r.get("score", 0.5),
                rrf_score=0.5,
            )
            for r in results
        ]
