"""
V3 Pipeline: Simplified threshold-based retrieval.

Architecture:
    Question → Decompose → Resolve → Threshold Retrieve → Synthesize

Key differences from V2:
- No top_k limit: retrieves ALL facts above similarity threshold (0.7)
- No expansion phase: threshold captures everything relevant
- No drill-down agent: deterministic retrieval
- No separate scoped/global scoring: single pass

LLM Calls: 3 total (decompose + resolve + synthesize)
"""

import asyncio
import argparse
import os
import time

from src.querying_system.shared.schemas import (
    PipelineResult,
    StructuredAnswer,
    EvidencePool,
    QuestionType,
)
from src.querying_system.shared.decomposer import QueryDecomposer
from src.querying_system.shared.synthesizer import Synthesizer
from src.querying_system.v2.resolver import Resolver
from .retriever import ThresholdRetriever
from src.util.services import get_services

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[PipelineV3] {msg}")


class ThresholdPipelineV3:
    """
    Simplified pipeline with threshold-based retrieval.

    No top_k limits - gets ALL facts above similarity threshold.
    Simpler, faster, more deterministic.
    """

    def __init__(
        self,
        group_id: str = "default",
        similarity_threshold: float = 0.7,
        top_k_evidence: int = 40,  # Max facts to synthesize
    ):
        self.group_id = group_id
        self.similarity_threshold = similarity_threshold
        self.top_k_evidence = top_k_evidence

        # Initialize components
        self.decomposer = QueryDecomposer()
        self.resolver = Resolver(group_id=group_id)
        self.retriever = ThresholdRetriever(
            group_id=group_id,
            similarity_threshold=similarity_threshold
        )
        self.synthesizer = Synthesizer()
        self.services = get_services()

    async def query_async(self, question: str) -> PipelineResult:
        """
        Execute the simplified threshold-based pipeline.

        Flow:
        1. Decompose → entities, topics
        2. Resolve → actual graph nodes
        3. Threshold Retrieve → ALL facts with similarity > threshold
        4. Synthesize → answer
        """
        total_start = time.time()
        log(f"Starting V3 pipeline for: {question[:80]}...")

        # Phase 1: Decomposition
        log("Phase 1: Decomposition")
        decomposition, decomp_time = self.decomposer.decompose(question)
        log(
            f"  Entities: {decomposition.entity_hints}, "
            f"Topics: {decomposition.topic_hints}, "
            f"Type: {decomposition.question_type.value}, "
            f"Time: {decomp_time}ms"
        )

        # Phase 2: Resolution
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
        log(f"  Entity nodes: {resolved.entity_nodes}")
        log(f"  Topic nodes: {resolved.topic_nodes}")

        # Phase 3: Threshold-based Retrieval
        log(f"Phase 3: Threshold Retrieval (similarity > {self.similarity_threshold})")
        retrieval_start = time.time()
        retrieval_result = await self.retriever.retrieve(
            resolved=resolved,
            question=question,
        )
        retrieval_time = int((time.time() - retrieval_start) * 1000)
        log(
            f"  Retrieved: {len(retrieval_result.facts)} facts above threshold, "
            f"Time: {retrieval_time}ms"
        )
        for entity, count in retrieval_result.facts_per_entity.items():
            log(f"    {entity}: {count} facts")

        # Fetch chunk content for top facts
        facts = retrieval_result.facts
        await self._fetch_chunk_content(facts[:self.top_k_evidence])

        # Build evidence pool
        evidence_pool = EvidencePool(
            scored_facts=facts,
            coverage_map={},
            entities_found=list(
                set(f.subject for f in facts if f.subject)
                | set(f.object for f in facts if f.object)
            ),
            expansion_performed=False,
        )

        # Phase 4: Synthesis
        log("Phase 4: Synthesis")
        synthesis_top_k = self.top_k_evidence
        if decomposition.question_type == QuestionType.ENUMERATION:
            # For enumeration, use more facts
            synthesis_top_k = min(60, len(facts))
            log(f"  ENUMERATION: Using top_k_evidence={synthesis_top_k}")

        answer, synthesis_time = self.synthesizer.synthesize(
            question=question,
            decomposition=decomposition,
            evidence_pool=evidence_pool,
            top_k_evidence=synthesis_top_k,
        )
        log(f"  Confidence: {answer.confidence:.2f}, Time: {synthesis_time}ms")

        # Populate timing
        answer.decomposition_time_ms = decomp_time
        answer.retrieval_time_ms = retrieval_time + resolution_time
        answer.scoring_time_ms = 0  # No separate scoring in V3
        answer.expansion_time_ms = 0  # No expansion in V3
        answer.synthesis_time_ms = synthesis_time

        total_time = int((time.time() - total_start) * 1000)
        log(f"V3 Pipeline complete in {total_time}ms")

        return PipelineResult(
            question=question,
            answer=answer,
            decomposition=decomposition,
            evidence_pool=evidence_pool,
        )

    async def _fetch_chunk_content(self, facts: list) -> None:
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

    def query(self, question: str) -> PipelineResult:
        """Synchronous wrapper."""
        return asyncio.run(self.query_async(question))


def query_v3(
    question: str,
    group_id: str = "default",
    similarity_threshold: float = 0.7,
) -> PipelineResult:
    """Convenience function for V3 pipeline."""
    pipeline = ThresholdPipelineV3(
        group_id=group_id,
        similarity_threshold=similarity_threshold,
    )
    return pipeline.query(question)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Threshold-based Query Pipeline V3"
    )
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("-g", "--group-id", default="default")
    parser.add_argument("-t", "--threshold", type=float, default=0.7,
                        help="Similarity threshold (default: 0.7)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "true"

    result = query_v3(args.question, args.group_id, args.threshold)

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
        print(f"\nFACTS RETRIEVED: {len(result.evidence_pool.scored_facts)}")
        print(f"SIMILARITY THRESHOLD: {args.threshold}")
        print("\nANSWER:")
        print(result.answer.answer)
        print("\n" + "-" * 80)
        print(f"Confidence: {result.answer.confidence:.2f}")
        print(f"\nTiming:")
        print(f"  Decomposition: {result.answer.decomposition_time_ms}ms")
        print(f"  Retrieval:     {result.answer.retrieval_time_ms}ms")
        print(f"  Synthesis:     {result.answer.synthesis_time_ms}ms")
        print(f"  Total:         {result.answer.total_time_ms}ms")


if __name__ == "__main__":
    main()
