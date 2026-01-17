"""
Main Structured KG-RAG Pipeline
Orchestrates query analysis, retrieval, and answer generation.
"""

import asyncio
import time
from typing import Optional

from .models import QueryPlan, RetrievalResult, PipelineResult
from .query_analyzer import QueryAnalyzer, RuleBasedQueryAnalyzer
from .retriever import Retriever
from .answer_generator import AnswerGenerator


class StructuredKGRAG:
    """
    Main pipeline class for Structured KG-RAG.

    Flow:
    1. Query Analysis (1 LLM call) → QueryPlan
    2. Deterministic Retrieval (0 LLM calls) → RetrievalResult
    3. Answer Generation (1 LLM call) → Final Answer

    Total: 2 LLM calls per query (vs 10-30+ in agent-based system)
    """

    def __init__(
        self,
        user_id: str = "default",
        use_rule_based_analyzer: bool = False,
    ):
        """
        Initialize the pipeline.

        Args:
            user_id: User/tenant ID for graph scoping
            use_rule_based_analyzer: If True, use rule-based analyzer (no LLM call for step 1)
        """
        self.user_id = user_id

        # Initialize components
        if use_rule_based_analyzer:
            self.analyzer = RuleBasedQueryAnalyzer()
        else:
            self.analyzer = QueryAnalyzer()

        self.retriever = Retriever(user_id=user_id)
        self.generator = AnswerGenerator()

    def query(
        self,
        question: str,
        verbose: bool = False,
    ) -> PipelineResult:
        """
        Answer a question using the structured pipeline.

        Args:
            question: The user's natural language question
            verbose: If True, print intermediate steps

        Returns:
            PipelineResult with full details including answer
        """
        total_start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"QUESTION: {question}")
            print(f"{'='*60}\n")

        # Step 1: Query Analysis
        if verbose:
            print(">>> STEP 1: Analyzing query...")

        plan, analysis_time = self.analyzer.analyze(question)

        if verbose:
            print(f"    Query Type: {plan.query_type.value}")
            print(f"    Entities: {plan.anchor_entities}")
            print(f"    Relationship: {plan.target_relationship}")
            print(f"    Confidence: {plan.confidence}")
            print(f"    Time: {analysis_time}ms\n")

        # Step 2: Retrieval
        if verbose:
            print(">>> STEP 2: Retrieving evidence...")

        retrieval, retrieval_time = self.retriever.retrieve(plan)

        if verbose:
            print(f"    Pattern: {retrieval.retrieval_pattern_used}")
            print(f"    Chunks Found: {len(retrieval.chunks)}")
            resolved_names = [
                e.resolved_name for e in retrieval.resolved_entities if e.resolved_name
            ]
            print(f"    Entities Resolved: {resolved_names}")
            if retrieval.warnings:
                print(f"    Warnings: {retrieval.warnings}")
            print(f"    Time: {retrieval_time}ms\n")

        # Step 3: Answer Generation
        if verbose:
            print(">>> STEP 3: Generating answer...")

        answer, generation_time = self.generator.generate(
            question=question,
            plan=plan,
            retrieval=retrieval,
        )

        if verbose:
            print(f"    Time: {generation_time}ms\n")
            print(f"{'='*60}")
            print("ANSWER:")
            print(f"{'='*60}")
            print(answer)

        # Build result
        result = PipelineResult(
            question=question,
            answer=answer,
            plan=plan,
            retrieval=retrieval,
            analysis_time_ms=analysis_time,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
        )

        return result

    async def query_async(
        self,
        question: str,
        verbose: bool = False,
    ) -> PipelineResult:
        """Async version of query - runs sync code in thread pool."""
        return await asyncio.to_thread(self.query, question, verbose)

    def batch_query(
        self,
        questions: list[str],
        verbose: bool = False,
    ) -> list[PipelineResult]:
        """
        Process multiple questions sequentially.

        Args:
            questions: List of questions to answer
            verbose: If True, print progress

        Returns:
            List of PipelineResults
        """
        results = []

        for i, question in enumerate(questions):
            if verbose:
                print(f"\n[{i+1}/{len(questions)}] Processing...")

            result = self.query(question, verbose=verbose)
            results.append(result)

        return results


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """CLI for testing the pipeline."""
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is the Beige Book?"

    pipeline = StructuredKGRAG(user_id="default")

    result = pipeline.query(question, verbose=True)

    print(f"\n{'='*60}")
    print("METRICS:")
    print(f"{'='*60}")
    print(f"Total Time: {result.total_time_ms}ms")
    print(f"  - Analysis: {result.analysis_time_ms}ms")
    print(f"  - Retrieval: {result.retrieval_time_ms}ms")
    print(f"  - Generation: {result.generation_time_ms}ms")
    print(f"Chunks Retrieved: {len(result.retrieval.chunks)}")
    print(f"Fallback Used: {result.retrieval.fallback_used}")


if __name__ == "__main__":
    main()
