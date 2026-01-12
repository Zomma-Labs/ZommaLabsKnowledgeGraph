"""
V2 Pipeline: GNN-Inspired Knowledge Graph Query Pipeline.

Two modes:
1. Classic (use_parallel_subqueries=False):
    Question → Decompose → Resolve → Scoped Search → Score → Expand → Synthesize

2. Parallel Sub-Query (use_parallel_subqueries=True):
    Question → Decompose → Split → Parallel Sub-Query Retrieval → Score → Synthesize

    Each sub-query gets:
    - Its own entity resolution (using sub-query context)
    - Threshold-based retrieval (sim > 0.7)
    - Facts tagged with provenance

    Cross-query boosting: Facts found by multiple sub-queries get score boost.

LLM Calls:
- Classic: 5 (decompose + resolve + scoped_score + global_score + synthesize)
- Parallel: 4+ (decompose + split + N*resolve + synthesize)
"""

import asyncio
import argparse
import os
import time

from src.querying_system.shared.schemas import (
    PipelineResult,
    StructuredAnswer,
    QueryDecomposition,
    EvidencePool,
    ScoredFact,
    SubQueryResult,
    ParallelRetrievalResult,
)
from src.querying_system.shared.decomposer import QueryDecomposer
from src.querying_system.shared.scorer import FactScorer
from src.querying_system.shared.synthesizer import Synthesizer
from src.querying_system.shared.entity_drilldown import EntityDrillDown
from src.querying_system.shared.schemas import QuestionType
from .resolver import Resolver
from .retriever import GNNRetrieverV2
from .expander import GraphExpanderV2
from .query_splitter import QuerySplitter
from .sub_query_retriever import ParallelSubQueryOrchestrator
from src.util.services import get_services

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[PipelineV2] {msg}")


class GNNPipelineV2:
    """
    GNN-Inspired Query Pipeline.

    Key insight: Separate scoped and global searches, score each independently,
    then combine. This gives high precision (scoped) + high coverage (global).
    """

    def __init__(
        self,
        group_id: str = "default",
        top_k_per_node: int = 20,
        top_k_global: int = 20,
        max_facts_to_score: int = 50,
        top_k_evidence: int = 20,
        max_entities_to_expand: int = 10,
        top_n_unique_entities: int = 30,  # For enumeration - first N unique connected entities
        use_parallel_subqueries: bool = True,  # Enable parallel sub-query retrieval
    ):
        self.group_id = group_id
        self.top_k_per_node = top_k_per_node
        self.top_k_global = top_k_global
        self.max_facts_to_score = max_facts_to_score
        self.top_n_unique_entities = top_n_unique_entities
        self.top_k_evidence = top_k_evidence
        self.max_entities_to_expand = max_entities_to_expand
        self.use_parallel_subqueries = use_parallel_subqueries

        # Initialize components
        self.decomposer = QueryDecomposer()
        self.resolver = Resolver(group_id=group_id)
        self.retriever = GNNRetrieverV2(group_id=group_id)
        self.scoped_scorer = FactScorer()  # For scoring scoped facts
        self.global_scorer = FactScorer()  # For scoring global facts
        self.expander = GraphExpanderV2(group_id=group_id)
        self.entity_drilldown = EntityDrillDown()  # For ENUMERATION drill-down
        self.synthesizer = Synthesizer()
        self.services = get_services()

        # Parallel sub-query components
        self.query_splitter = QuerySplitter()
        self.subquery_orchestrator = ParallelSubQueryOrchestrator(group_id=group_id)

    async def query_async(self, question: str) -> PipelineResult:
        """
        Execute the GNN-inspired pipeline.

        Two modes:
        1. Classic: Decompose → Resolve → Scoped+Global Search → Score → Expand → Synthesize
        2. Parallel Sub-Query: Decompose → Split → Parallel Retrieval → Score → Synthesize
        """
        total_start = time.time()
        mode = "parallel-subquery" if self.use_parallel_subqueries else "classic"
        log(f"Starting V2 pipeline ({mode}) for: {question}")

        # Phase 1: Decomposition (gpt-5.1)
        log("Phase 1: Decomposition")
        decomposition, decomp_time = self.decomposer.decompose(question)
        log(
            f"  Entities: {decomposition.entity_hints}, "
            f"Topics: {decomposition.topic_hints}, "
            f"Time: {decomp_time}ms"
        )

        # Route to appropriate retrieval strategy
        if self.use_parallel_subqueries:
            return await self._query_parallel_subqueries(
                question=question,
                decomposition=decomposition,
                decomp_time=decomp_time,
                total_start=total_start,
            )
        else:
            return await self._query_classic(
                question=question,
                decomposition=decomposition,
                decomp_time=decomp_time,
                total_start=total_start,
            )

    async def _query_parallel_subqueries(
        self,
        question: str,
        decomposition: QueryDecomposition,
        decomp_time: int,
        total_start: float,
    ) -> PipelineResult:
        """
        Parallel sub-query retrieval path.

        Flow:
        1. Split question into sub-queries (LLM)
        2. Execute each sub-query in parallel (resolve + threshold retrieve)
        3. Combine with cross-query boosting
        4. Score combined facts
        5. Synthesize answer
        """
        # Phase 2: Split into sub-queries (gpt-5.1)
        log("Phase 2: Query Splitting")
        sub_queries, split_time = await self.query_splitter.split(question, decomposition)
        log(f"  Split into {len(sub_queries)} sub-queries in {split_time}ms")
        for sq in sub_queries:
            log(f"    - {sq.query_text}")

        # Phase 3: Parallel sub-query retrieval
        log("Phase 3: Parallel Sub-Query Retrieval")
        retrieval_start = time.time()
        parallel_result = await self.subquery_orchestrator.retrieve(
            sub_queries=sub_queries,
            question=question,
        )
        retrieval_time = int((time.time() - retrieval_start) * 1000)

        log(f"  Retrieved {len(parallel_result.combined_facts)} unique facts")
        log(f"  Cross-query boosted: {len(parallel_result.cross_query_boosted_fact_ids)}")
        log(f"  Time: {retrieval_time}ms")

        # Phase 4: Score combined facts
        log("Phase 4: Scoring")
        scoring_start = time.time()

        scored_facts, scoring_time = await asyncio.to_thread(
            self.scoped_scorer.score,
            question=question,
            decomposition=decomposition,
            facts=parallel_result.combined_facts,
            max_facts_to_score=self.max_facts_to_score,
        )
        parallel_scoring_time = int((time.time() - scoring_start) * 1000)
        log(f"  Scored {len(scored_facts)} facts in {parallel_scoring_time}ms")

        # Fetch chunk content for top facts
        await self._fetch_chunk_content(scored_facts[:self.top_k_evidence])

        # Build evidence pool
        evidence_pool = EvidencePool(
            scored_facts=scored_facts,
            coverage_map={},
            entities_found=list(
                set(f.subject for f in scored_facts if f.subject)
                | set(f.object for f in scored_facts if f.object)
            ),
            expansion_performed=False,
        )

        # Phase 5: Synthesis (gpt-5.1)
        log("Phase 5: Synthesis")
        answer, synthesis_time = self.synthesizer.synthesize(
            question=question,
            decomposition=decomposition,
            evidence_pool=evidence_pool,
            top_k_evidence=self.top_k_evidence,
        )
        log(f"  Confidence: {answer.confidence:.2f}, Time: {synthesis_time}ms")

        # Populate timing
        answer.decomposition_time_ms = decomp_time + split_time
        answer.retrieval_time_ms = retrieval_time
        answer.scoring_time_ms = parallel_scoring_time
        answer.expansion_time_ms = 0
        answer.synthesis_time_ms = synthesis_time

        total_time = int((time.time() - total_start) * 1000)
        log(f"V2 Pipeline (parallel-subquery) complete in {total_time}ms")

        return PipelineResult(
            question=question,
            answer=answer,
            decomposition=decomposition,
            evidence_pool=evidence_pool,
        )

    async def _query_classic(
        self,
        question: str,
        decomposition: QueryDecomposition,
        decomp_time: int,
        total_start: float,
    ) -> PipelineResult:
        """
        Classic retrieval path (original V2 behavior).

        Flow:
        1. Resolve entities/topics to graph nodes
        2. Scoped + Global search in parallel
        3. Score both sets
        4. Expand from high-scoring scoped facts
        5. Combine all facts
        6. Synthesize
        """

        # Phase 2: Resolution (gpt-5-mini)
        log("Phase 2: Resolution")
        resolution_start = time.time()
        resolved = await self.resolver.resolve(
            entity_hints=decomposition.entity_hints,
            topic_hints=decomposition.topic_hints,
            question=question,
        )
        resolution_time = int((time.time() - resolution_start) * 1000)
        log(
            f"  Resolved: {len(resolved.entity_nodes)} entities, "
            f"{len(resolved.topic_nodes)} topics, "
            f"Time: {resolution_time}ms"
        )

        # Phase 3 & 6: Parallel Retrieval (scoped + global)
        log("Phase 3+6: Parallel Retrieval (scoped + global)")
        retrieval_start = time.time()

        scoped_task = self.retriever.retrieve_scoped(
            resolved=resolved,
            question=question,
            top_k_per_node=self.top_k_per_node,
            top_n_unique_entities=self.top_n_unique_entities,
        )
        global_task = self.retriever.retrieve_global(
            question=question,
            decomposition=decomposition,
            top_k=self.top_k_global,
        )

        scoped_results, global_facts = await asyncio.gather(scoped_task, global_task)

        retrieval_time = int((time.time() - retrieval_start) * 1000)
        scoped_facts = []
        unique_entities_by_node: dict[str, list[str]] = {}
        for result in scoped_results:
            scoped_facts.extend(result.facts)
            # Collect unique entities for ENUMERATION questions
            if result.unique_connected_entities:
                unique_entities_by_node[result.node_name] = result.unique_connected_entities
        log(f"  Scoped: {len(scoped_facts)} facts, Global: {len(global_facts)} facts, Time: {retrieval_time}ms")
        if unique_entities_by_node:
            log(f"  Unique entities found: {sum(len(v) for v in unique_entities_by_node.values())} across {len(unique_entities_by_node)} nodes")
            for node, entities in list(unique_entities_by_node.items())[:3]:  # Show first 3
                log(f"    {node}: {entities[:5]}")

        # Phase 4 & 7: Parallel Scoring (scoped + global)
        log("Phase 4+7: Parallel Scoring (scoped + global)")
        scoring_start = time.time()

        # Run both scoring operations in parallel using threads
        async def score_scoped():
            return await asyncio.to_thread(
                self.scoped_scorer.score,
                question=question,
                decomposition=decomposition,
                facts=scoped_facts,
                max_facts_to_score=self.max_facts_to_score,
                unique_entities_by_node=unique_entities_by_node,  # For ENUMERATION
            )

        async def score_global():
            return await asyncio.to_thread(
                self.global_scorer.score,
                question=question,
                decomposition=decomposition,
                facts=global_facts,
                max_facts_to_score=self.max_facts_to_score,
            )

        (scored_scoped, scoped_scoring_time), (scored_global, global_scoring_time) = await asyncio.gather(
            score_scoped(), score_global()
        )

        parallel_scoring_time = int((time.time() - scoring_start) * 1000)
        expand_count = sum(1 for f in scored_scoped if f.should_expand)
        log(
            f"  Scoped scored: {len(scored_scoped)}, To expand: {expand_count}, "
            f"Global scored: {len(scored_global)}, Time: {parallel_scoring_time}ms"
        )

        # Phase 5: Expansion (from facts marked should_expand)
        log("Phase 5: Expansion")
        expansion_start = time.time()
        expand_facts = {f.fact_id: f for f in scored_scoped if f.should_expand}

        expanded_facts = await self.expander.expand_from_scoped(
            scoped_facts=expand_facts,
            decomposition=decomposition,
            max_entities_to_expand=self.max_entities_to_expand,
        )
        expansion_time = int((time.time() - expansion_start) * 1000)
        log(f"  Expanded: {len(expanded_facts)} new facts, Time: {expansion_time}ms")

        # Phase 6: Entity Drill-Down (ENUMERATION only)
        drilldown_facts: list[ScoredFact] = []
        drilldown_time = 0
        if decomposition.question_type == QuestionType.ENUMERATION and unique_entities_by_node:
            log("Phase 6: Entity Drill-Down (ENUMERATION)")
            drilldown_start = time.time()

            # Get all current facts for context
            current_facts = scored_scoped + scored_global + expanded_facts

            # Let agent select entities to drill down on
            selected_entities = self.entity_drilldown.select_entities(
                question=question,
                decomposition=decomposition,
                unique_entities_by_node=unique_entities_by_node,
                current_facts=current_facts,
            )

            if selected_entities:
                log(f"  Drilling down on {len(selected_entities)} entities...")
                # Fetch additional facts for selected entities
                drilldown_facts = await self._fetch_entity_facts(
                    entities=selected_entities,
                    question=question,
                    existing_fact_ids={f.fact_id for f in current_facts},
                )
                log(f"  Found {len(drilldown_facts)} additional facts from drill-down")

            drilldown_time = int((time.time() - drilldown_start) * 1000)
            log(f"  Drill-down time: {drilldown_time}ms")

        # Phase 8: Combine all facts
        log("Phase 8: Combining results")
        all_facts: dict[str, ScoredFact] = {}

        # Add scoped facts (highest priority)
        for fact in scored_scoped:
            all_facts[fact.fact_id] = fact

        # Add expanded facts
        for fact in expanded_facts:
            if fact.fact_id not in all_facts:
                all_facts[fact.fact_id] = fact

        # Add drill-down facts (for ENUMERATION)
        for fact in drilldown_facts:
            if fact.fact_id not in all_facts:
                all_facts[fact.fact_id] = fact

        # Add global facts (fills gaps)
        for fact in scored_global:
            if fact.fact_id not in all_facts:
                all_facts[fact.fact_id] = fact
            else:
                # Boost facts found by both scoped and global
                all_facts[fact.fact_id].cross_query_boost += 0.15

        # Fetch chunk content for top facts (if missing)
        combined_facts = list(all_facts.values())
        combined_facts.sort(key=lambda f: f.final_score, reverse=True)
        await self._fetch_chunk_content(combined_facts[:self.top_k_evidence])

        log(f"  Combined: {len(combined_facts)} total facts")

        # Build evidence pool
        evidence_pool = EvidencePool(
            scored_facts=combined_facts,
            coverage_map={},
            entities_found=list(
                set(f.subject for f in combined_facts if f.subject)
                | set(f.object for f in combined_facts if f.object)
            ),
            expansion_performed=len(expanded_facts) > 0,
        )

        # Phase 9: Synthesis (gpt-5.1)
        log("Phase 9: Synthesis")
        # For ENUMERATION, increase evidence to capture all items
        synthesis_top_k = self.top_k_evidence
        if decomposition.question_type == QuestionType.ENUMERATION:
            synthesis_top_k = min(40, len(combined_facts))  # Up to 40 for enumeration
            log(f"  ENUMERATION: Using top_k_evidence={synthesis_top_k}")

        answer, synthesis_time = self.synthesizer.synthesize(
            question=question,
            decomposition=decomposition,
            evidence_pool=evidence_pool,
            top_k_evidence=synthesis_top_k,
            unique_entities_by_node=unique_entities_by_node,  # For ENUMERATION
        )
        log(f"  Confidence: {answer.confidence:.2f}, Time: {synthesis_time}ms")

        # Populate timing
        answer.decomposition_time_ms = decomp_time
        answer.retrieval_time_ms = retrieval_time + resolution_time
        answer.expansion_time_ms = expansion_time
        answer.scoring_time_ms = parallel_scoring_time
        answer.synthesis_time_ms = synthesis_time

        total_time = int((time.time() - total_start) * 1000)
        log(f"V2 Pipeline complete in {total_time}ms")

        return PipelineResult(
            question=question,
            answer=answer,
            decomposition=decomposition,
            evidence_pool=evidence_pool,
        )

    async def _fetch_chunk_content(self, facts: list[ScoredFact]) -> None:
        """Fetch chunk content for facts that are missing it."""
        facts_needing_chunks = [f for f in facts if not f.chunk_content and f.fact_id]

        if not facts_needing_chunks:
            return

        log(f"Fetching chunks for {len(facts_needing_chunks)} facts...")

        def _query():
            fact_ids = [f.fact_id for f in facts_needing_chunks]
            return self.services.neo4j.query(
                """
                UNWIND $fact_ids as fid
                MATCH (f:FactNode {uuid: fid, group_id: $uid})

                MATCH (subj)-[r1 {fact_id: fid}]->(c:EpisodicNode {group_id: $uid})-[r2 {fact_id: fid}]->(obj)
                WHERE (subj:EntityNode OR subj:TopicNode) AND (obj:EntityNode OR obj:TopicNode)

                OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                RETURN DISTINCT fid as fact_id,
                       c.uuid as chunk_id,
                       c.content as chunk_content,
                       c.header_path as chunk_header,
                       d.name as doc_id,
                       d.document_date as document_date
                """,
                {"fact_ids": [f.fact_id for f in facts_needing_chunks], "uid": self.group_id},
            )

        results = await asyncio.to_thread(_query)

        # Map results to facts
        chunk_map = {r["fact_id"]: r for r in results}
        for fact in facts_needing_chunks:
            if fact.fact_id in chunk_map:
                r = chunk_map[fact.fact_id]
                fact.chunk_id = r.get("chunk_id")
                fact.chunk_content = r.get("chunk_content")
                fact.chunk_header = r.get("chunk_header")
                fact.doc_id = r.get("doc_id")
                fact.document_date = r.get("document_date")

    async def _fetch_entity_facts(
        self,
        entities: list[str],
        question: str,
        existing_fact_ids: set[str],
        top_k_per_entity: int = 10,
    ) -> list[ScoredFact]:
        """
        Fetch additional facts for specific entities (drill-down).

        Used for ENUMERATION questions to get complete coverage.
        """
        if not entities:
            return []

        # Embed question for ranking
        question_embedding = await asyncio.to_thread(
            self.services.embeddings.embed_query, question
        )

        all_facts: list[ScoredFact] = []

        # Query facts for each entity in parallel
        async def fetch_for_entity(entity_name: str) -> list[ScoredFact]:
            def _query():
                return self.services.neo4j.query(
                    """
                    MATCH (n {name: $entity_name, group_id: $uid})
                    WHERE n:EntityNode OR n:TopicNode

                    MATCH (n)-[r1]->(c:EpisodicNode {group_id: $uid})-[r2]->(target)
                    WHERE r1.fact_id = r2.fact_id
                      AND (target:EntityNode OR target:TopicNode)
                      AND NOT r1.fact_id IN $existing_ids

                    MATCH (f:FactNode {uuid: r1.fact_id, group_id: $uid})

                    OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)

                    RETURN DISTINCT f.uuid as fact_id,
                           f.content as content,
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
                    {
                        "entity_name": entity_name,
                        "uid": self.group_id,
                        "existing_ids": list(existing_fact_ids),
                        "top_k": top_k_per_entity,
                    },
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
                    vector_score=0.6,  # Default score for drill-down facts
                    rrf_score=0.6,
                    final_score=0.6,
                )
                fact.found_by_queries.append(f"drilldown:{entity_name}")
                facts.append(fact)

            return facts

        # Fetch all in parallel
        tasks = [fetch_for_entity(entity) for entity in entities]
        results = await asyncio.gather(*tasks)

        # Flatten and dedupe
        seen = set()
        for entity_facts in results:
            for fact in entity_facts:
                if fact.fact_id not in seen:
                    seen.add(fact.fact_id)
                    all_facts.append(fact)

        return all_facts

    def query(self, question: str) -> PipelineResult:
        """Synchronous wrapper."""
        return asyncio.run(self.query_async(question))


def query_gnn_v2(
    question: str,
    group_id: str = "default",
) -> PipelineResult:
    """Convenience function for V2 pipeline."""
    pipeline = GNNPipelineV2(group_id=group_id)
    return pipeline.query(question)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GNN-Inspired Query Pipeline V2"
    )
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("-g", "--group-id", default="default")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "true"

    result = query_gnn_v2(args.question, args.group_id)

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 80)
        print("QUESTION:", result.question)
        print("=" * 80)
        print("\nQUESTION TYPE:", result.decomposition.question_type.value)
        print(f"ENTITIES: {result.decomposition.entity_hints}")
        print(f"TOPICS: {result.decomposition.topic_hints}")
        print("\nANSWER:")
        print(result.answer.answer)
        print("\n" + "-" * 80)
        print(f"Confidence: {result.answer.confidence:.2f}")
        print(f"Facts retrieved: {len(result.evidence_pool.scored_facts)}")
        print(f"Expansion: {'Yes' if result.evidence_pool.expansion_performed else 'No'}")
        print(f"\nTiming:")
        print(f"  Decomposition: {result.answer.decomposition_time_ms}ms")
        print(f"  Retrieval:     {result.answer.retrieval_time_ms}ms")
        print(f"  Expansion:     {result.answer.expansion_time_ms}ms")
        print(f"  Scoring:       {result.answer.scoring_time_ms}ms")
        print(f"  Synthesis:     {result.answer.synthesis_time_ms}ms")
        print(f"  Total:         {result.answer.total_time_ms}ms")


if __name__ == "__main__":
    main()
