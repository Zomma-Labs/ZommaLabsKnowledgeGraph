"""
Hybrid CoT-GNN Query Pipeline V2.

Flow:
    Question → Decompose → Resolve → Retrieve → Score → Expand → Synthesize

1. Decompose: Extract entities, topics, relationships from question (gpt-5.1)
2. Resolve: Match extracted terms to actual graph nodes (gpt-5-mini)
3. Retrieve: Vector search localized around resolved nodes
4. Score: Rank facts for relevance, mark for expansion (gpt-5-mini)
5. Expand: Follow graph edges from marked facts
6. Synthesize: Answer using top facts + chunks (gpt-5.1)

LLM Calls: 4 total (decompose + resolve + score + synthesize)
"""

import asyncio
import argparse
import os
import time

from .schemas import (
    PipelineResult,
    StructuredAnswer,
    QueryDecomposition,
    EvidencePool,
    ScoredFact,
)
from .decomposer import QueryDecomposer
from .resolver import Resolver
from .retriever_v2 import HybridRetrieverV2
from .scorer import FactScorer
from .expander_v2 import GraphExpanderV2
from .synthesizer import Synthesizer
from src.util.services import get_services

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[PipelineV2] {msg}")


class HybridCoTGNNPipelineV2:
    """
    V2 Pipeline with resolution-based retrieval.

    Flow: Decompose → Resolve → Retrieve → Score → Expand → Synthesize
    """

    def __init__(
        self,
        group_id: str = "default",
        top_k_per_node: int = 10,
        max_facts_to_score: int = 50,
        top_k_evidence: int = 20,
        max_entities_to_expand: int = 10,
    ):
        self.group_id = group_id
        self.top_k_per_node = top_k_per_node
        self.max_facts_to_score = max_facts_to_score
        self.top_k_evidence = top_k_evidence
        self.max_entities_to_expand = max_entities_to_expand

        # Initialize components
        self.decomposer = QueryDecomposer()
        self.resolver = Resolver(group_id=group_id)
        self.retriever = HybridRetrieverV2(group_id=group_id)
        self.scorer = FactScorer()
        self.expander = GraphExpanderV2(group_id=group_id)
        self.synthesizer = Synthesizer()
        self.services = get_services()

    async def query_async(self, question: str) -> PipelineResult:
        """
        Execute the V2 pipeline.

        Flow:
        1. Decompose: Extract entities, topics, relationships
        2. Resolve: Match to actual graph nodes
        3. Retrieve: Vector search around resolved nodes
        4. Score: Rank facts, mark for expansion
        5. Expand: Follow graph edges
        6. Synthesize: Answer from top facts
        """
        total_start = time.time()
        log(f"Starting V2 pipeline for: {question[:80]}...")

        # Phase 1: Decomposition (gpt-5.1)
        log("Phase 1: Decomposition")
        decomposition, decomp_time = self.decomposer.decompose(question)
        log(
            f"  Type: {decomposition.question_type.value}, "
            f"Entities: {decomposition.entity_hints}, "
            f"Topics: {decomposition.topic_hints}, "
            f"Time: {decomp_time}ms"
        )

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
            f"  Resolved entities: {resolved.entity_nodes}, "
            f"Resolved topics: {resolved.topic_nodes}, "
            f"Time: {resolution_time}ms"
        )

        # Phase 3: Retrieval (get facts connected to resolved nodes)
        log("Phase 3: Retrieval")
        retrieval_start = time.time()

        # Get facts connected to resolved nodes using question-based vector search
        all_facts = await self.retriever.retrieve_from_resolved(
            resolved=resolved,
            decomposition=decomposition,
            question=question,
            top_k_per_node=self.top_k_per_node,
        )
        retrieval_time = int((time.time() - retrieval_start) * 1000)
        log(f"  Retrieved: {len(all_facts)} facts, Time: {retrieval_time}ms")

        # Phase 4: Scoring (gpt-5-mini)
        log("Phase 4: Scoring")
        scored_facts, scoring_time = self.scorer.score(
            question=question,
            decomposition=decomposition,
            facts=list(all_facts.values()),
            max_facts_to_score=self.max_facts_to_score,
        )
        log(
            f"  Scored: {len(scored_facts)} facts, "
            f"should_expand: {sum(1 for f in scored_facts if f.should_expand)}, "
            f"Time: {scoring_time}ms"
        )

        # Phase 5: Expansion
        log("Phase 5: Expansion")
        expansion_start = time.time()
        expand_facts = {f.fact_id: f for f in scored_facts if f.should_expand}

        expanded_facts = await self.expander.expand_from_scoped(
            scoped_facts=expand_facts,
            decomposition=decomposition,
            max_entities_to_expand=self.max_entities_to_expand,
        )
        expansion_time = int((time.time() - expansion_start) * 1000)
        log(f"  Expanded: {len(expanded_facts)} new facts, Time: {expansion_time}ms")

        # Merge expanded facts
        for fact in expanded_facts:
            if fact.fact_id not in all_facts:
                all_facts[fact.fact_id] = fact
                scored_facts.append(fact)

        # Build evidence pool
        evidence_pool = EvidencePool(
            scored_facts=scored_facts,
            coverage_map={},
            entities_found=list(
                set(f.subject for f in scored_facts if f.subject)
                | set(f.object for f in scored_facts if f.object)
            ),
            expansion_performed=len(expanded_facts) > 0,
        )

        # Phase 6: Synthesis (gpt-5.1)
        log("Phase 6: Synthesis")
        answer, synthesis_time = self.synthesizer.synthesize(
            question=question,
            decomposition=decomposition,
            evidence_pool=evidence_pool,
            top_k_evidence=self.top_k_evidence,
        )
        log(f"  Confidence: {answer.confidence:.2f}, Time: {synthesis_time}ms")

        # Populate timing
        answer.decomposition_time_ms = decomp_time
        answer.retrieval_time_ms = retrieval_time + resolution_time
        answer.expansion_time_ms = expansion_time
        answer.scoring_time_ms = scoring_time
        answer.synthesis_time_ms = synthesis_time

        total_time = int((time.time() - total_start) * 1000)
        log(f"V2 Pipeline complete in {total_time}ms")

        return PipelineResult(
            question=question,
            answer=answer,
            decomposition=decomposition,
            evidence_pool=evidence_pool,
        )

    def query(self, question: str) -> PipelineResult:
        """Synchronous wrapper."""
        return asyncio.run(self.query_async(question))


def query_hybrid_cot_v2(
    question: str,
    group_id: str = "default",
) -> PipelineResult:
    """Convenience function for V2 pipeline."""
    pipeline = HybridCoTGNNPipelineV2(group_id=group_id)
    return pipeline.query(question)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid CoT-GNN Query Pipeline V2"
    )
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("-g", "--group-id", default="default")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "true"

    result = query_hybrid_cot_v2(args.question, args.group_id)

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
