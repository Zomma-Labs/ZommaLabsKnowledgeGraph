"""
Hybrid CoT-GNN Query Pipeline.

Combines Chain-of-Thought decomposition with GNN-inspired graph expansion
for high-quality knowledge graph querying.

Architecture:
    Question → Decompose (gpt-5.2) → Retrieve (scoped entity+topic) → Score (gpt-5-mini)
             → [Expand if LLM marks should_expand] → Synthesize (gpt-5.2, using chunk context)

Key improvements:
1. Entity+topic scoped retrieval (not global search)
2. Global fallback only if scoped search yields <10 results
3. Expansion for ANY question type (LLM-controlled via should_expand flag)
4. Synthesis uses chunk content for full context, not just atomic facts

LLM Calls: 3 total (2 main + 1 mini)
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
    QuestionType,
)
from src.querying_system.shared.decomposer import QueryDecomposer
from src.querying_system.shared.scorer import FactScorer
from src.querying_system.shared.synthesizer import Synthesizer
from .retriever import HybridRetriever
from .expander import GraphExpander

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[Pipeline] {msg}")


class HybridCoTGNNPipeline:
    """
    Main orchestrator for the Hybrid CoT-GNN query pipeline.

    Features:
    - Chain-of-thought query decomposition
    - Parallel multi-strategy retrieval
    - LLM-based fact scoring with gpt-5-mini
    - Conditional 1-hop graph expansion for CAUSAL/COMPARISON
    - Question-type-aware answer synthesis
    """

    def __init__(
        self,
        group_id: str = "default",
        enable_expansion: bool = True,
        top_k_per_query: int = 15,
        max_facts_to_score: int = 50,
        top_k_evidence: int = 20,
    ):
        """
        Initialize the pipeline.

        Args:
            group_id: Tenant/group ID for multi-tenant isolation
            enable_expansion: Whether to perform graph expansion for CAUSAL/COMPARISON
            top_k_per_query: Results per sub-query in retrieval
            max_facts_to_score: Maximum facts to send to LLM scorer
            top_k_evidence: Facts to include in synthesis context
        """
        self.group_id = group_id
        self.enable_expansion = enable_expansion
        self.top_k_per_query = top_k_per_query
        self.max_facts_to_score = max_facts_to_score
        self.top_k_evidence = top_k_evidence

        # Initialize components
        self.decomposer = QueryDecomposer()
        self.retriever = HybridRetriever(group_id=group_id)
        self.scorer = FactScorer()
        self.expander = GraphExpander(group_id=group_id)
        self.synthesizer = Synthesizer()

    async def query_async(self, question: str) -> PipelineResult:
        """
        Execute the full pipeline asynchronously.

        Args:
            question: Natural language question

        Returns:
            PipelineResult with answer, decomposition, and evidence
        """
        total_start = time.time()
        log(f"Starting pipeline for: {question[:80]}...")

        # Phase 1: Decomposition (gpt-5.2)
        log("Phase 1: Decomposition")
        decomposition, decomp_time = self.decomposer.decompose(question)
        log(
            f"  Type: {decomposition.question_type.value}, "
            f"Sub-queries: {len(decomposition.sub_queries)}, "
            f"Time: {decomp_time}ms"
        )

        # Phase 2a: Parallel Retrieval (deterministic)
        log("Phase 2a: Retrieval")
        retrieval_start = time.time()
        evidence_pool = await self.retriever.retrieve(
            decomposition, top_k_per_query=self.top_k_per_query
        )
        retrieval_time = int((time.time() - retrieval_start) * 1000)
        log(f"  Facts: {len(evidence_pool.scored_facts)}, Time: {retrieval_time}ms")

        # Phase 2b: LLM Scoring (gpt-5-mini)
        log("Phase 2b: Scoring")
        scored_facts, scoring_time = self.scorer.score(
            question=question,
            decomposition=decomposition,
            facts=evidence_pool.scored_facts,
            max_facts_to_score=self.max_facts_to_score,
        )
        evidence_pool.scored_facts = scored_facts
        log(f"  Scored: {len(scored_facts)} facts, Time: {scoring_time}ms")

        # Phase 2c: Conditional Expansion (LLM-controlled via should_expand flag)
        expansion_time = 0
        if self.enable_expansion:
            log("Phase 2c: Graph Expansion")
            expansion_start = time.time()
            evidence_pool = await self.expander.expand(
                evidence_pool, decomposition
            )
            expansion_time = int((time.time() - expansion_start) * 1000)
            log(
                f"  Expanded: {evidence_pool.expansion_performed}, "
                f"Facts: {len(evidence_pool.scored_facts)}, "
                f"Time: {expansion_time}ms"
            )

        # Phase 3: Synthesis (gpt-5.2)
        log("Phase 3: Synthesis")
        answer, synthesis_time = self.synthesizer.synthesize(
            question=question,
            decomposition=decomposition,
            evidence_pool=evidence_pool,
            top_k_evidence=self.top_k_evidence,
        )
        log(f"  Confidence: {answer.confidence:.2f}, Time: {synthesis_time}ms")

        # Populate timing
        answer.decomposition_time_ms = decomp_time
        answer.retrieval_time_ms = retrieval_time
        answer.scoring_time_ms = scoring_time
        answer.expansion_time_ms = expansion_time
        answer.synthesis_time_ms = synthesis_time

        total_time = int((time.time() - total_start) * 1000)
        log(f"Pipeline complete in {total_time}ms")

        return PipelineResult(
            question=question,
            answer=answer,
            decomposition=decomposition,
            evidence_pool=evidence_pool,
        )

    def query(self, question: str) -> PipelineResult:
        """
        Synchronous wrapper for query_async.

        Args:
            question: Natural language question

        Returns:
            PipelineResult with answer, decomposition, and evidence
        """
        return asyncio.run(self.query_async(question))


def query_hybrid_cot(
    question: str,
    group_id: str = "default",
    enable_expansion: bool = True,
) -> PipelineResult:
    """
    Convenience function to run a single query.

    Args:
        question: Natural language question
        group_id: Tenant/group ID
        enable_expansion: Enable graph expansion for CAUSAL/COMPARISON

    Returns:
        PipelineResult
    """
    pipeline = HybridCoTGNNPipeline(
        group_id=group_id,
        enable_expansion=enable_expansion,
    )
    return pipeline.query(question)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid CoT-GNN Query Pipeline for Knowledge Graph"
    )
    parser.add_argument("question", help="Natural language question to answer")
    parser.add_argument(
        "-g", "--group-id",
        default="default",
        help="Tenant/group ID (default: 'default')"
    )
    parser.add_argument(
        "--no-expansion",
        action="store_true",
        help="Disable graph expansion"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "true"

    result = query_hybrid_cot(
        question=args.question,
        group_id=args.group_id,
        enable_expansion=not args.no_expansion,
    )

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 80)
        print("QUESTION:", result.question)
        print("=" * 80)
        print("\nQUESTION TYPE:", result.decomposition.question_type.value)
        print("\nSUB-QUERIES:")
        for sq in result.decomposition.sub_queries:
            print(f"  - {sq.query_text}")
        print("\nANSWER:")
        print(result.answer.answer)
        print("\n" + "-" * 80)
        print(f"Confidence: {result.answer.confidence:.2f}")
        print(f"Facts retrieved: {len(result.evidence_pool.scored_facts)}")
        print(f"Expansion: {'Yes' if result.evidence_pool.expansion_performed else 'No'}")
        if result.answer.gaps:
            print(f"Coverage gaps: {', '.join(result.answer.gaps)}")
        print(f"\nTiming:")
        print(f"  Decomposition: {result.answer.decomposition_time_ms}ms")
        print(f"  Retrieval:     {result.answer.retrieval_time_ms}ms")
        print(f"  Scoring:       {result.answer.scoring_time_ms}ms")
        print(f"  Expansion:     {result.answer.expansion_time_ms}ms")
        print(f"  Synthesis:     {result.answer.synthesis_time_ms}ms")
        print(f"  Total:         {result.answer.total_time_ms}ms")
        print("=" * 80)


if __name__ == "__main__":
    main()
