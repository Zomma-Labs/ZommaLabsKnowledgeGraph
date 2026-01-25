#!/usr/bin/env python3
"""
Evaluate V6 Knowledge Graph Pipeline on BeigeOA dataset.

This script evaluates the V6 querying system (knowledge graph with deep research)
against the Beige Book Q&A dataset using an LLM judge.

The V6 system uses:
- Neo4j knowledge graph with entity/fact/topic nodes
- Query decomposition into sub-queries
- Threshold-only retrieval (no LLM scoring)
- OpenAI text-embedding-3-large (3072 dims)

Usage:
    uv run python -m testing.eval_v6 [--limit 5] [--concurrency 5]

Options:
    --limit            Limit number of questions to evaluate (default: all)
    --concurrency      Number of concurrent evaluations (default: 3)
    --qa-file          Path to Q&A dataset (default: eval/Biege_OA.json)
    --group-id         Neo4j group ID for multi-tenant isolation (default: default)
    --threshold        Relevance threshold for fact retrieval (default: 0.65)
    --disable-drilldown   Disable entity drilldown for ENUMERATION questions
    --disable-refinement  Disable answer refinement loop
"""

import argparse
import asyncio
import sys
import time

from testing.common import (
    load_qa_dataset,
    save_eval_results,
    print_eval_summary,
    EvalResult,
    EvalSummary,
    LLMJudge,
)
from src.querying_system.v6 import query_v6, ResearcherConfig

# Default file paths
DEFAULT_QA_FILE = "eval/Biege_OA.json"


async def evaluate_question(
    judge: LLMJudge,
    qa: dict,
    group_id: str,
    config: ResearcherConfig,
) -> EvalResult:
    """Evaluate a single question using V6 pipeline.

    Args:
        judge: The LLM judge for evaluation.
        qa: The Q&A pair dict with id, question, answer.
        group_id: Neo4j group ID for multi-tenant isolation.
        config: ResearcherConfig for V6 pipeline.

    Returns:
        EvalResult with verdict and timing.
    """
    start_time = time.perf_counter()

    # Query the V6 system
    try:
        result = await query_v6(
            question=qa["question"],
            group_id=group_id,
            config=config,
        )
        answer = result.answer

        # Convert evidence to chunk format for manual review
        chunks = [
            {
                "fact_id": ev.fact_id,
                "content": ev.content,
                "subject": ev.subject,
                "edge_type": ev.edge_type,
                "object": ev.object,
                "source_doc": ev.source_doc,
                "document_date": ev.document_date,
                "score": ev.score,
            }
            for ev in result.evidence
        ]
    except Exception as e:
        # On error, return empty answer (will be judged as abstained/incorrect)
        answer = f"Error: {e}"
        chunks = []

    # Judge the answer
    verdict, reasoning = await judge.judge(
        question=qa["question"],
        expected_answer=qa["answer"],
        system_answer=answer,
    )

    timing_ms = int((time.perf_counter() - start_time) * 1000)

    return EvalResult(
        question_id=qa["id"],
        question=qa["question"],
        expected_answer=qa["answer"],
        system_answer=answer,
        verdict=verdict,
        judge_reasoning=reasoning,
        retrieved_chunks=chunks,
        timing_ms=timing_ms,
    )


async def run_evaluation(
    qa_file: str = DEFAULT_QA_FILE,
    limit: int | None = None,
    concurrency: int = 3,
    group_id: str = "default",
    threshold: float = 0.65,
    enable_drilldown: bool = True,
    enable_refinement: bool = True,
) -> tuple[list[EvalResult], EvalSummary]:
    """Run the full evaluation.

    Args:
        qa_file: Path to the Q&A dataset JSON file.
        limit: Optional limit on number of questions to evaluate.
        concurrency: Maximum concurrent evaluations.
        group_id: Neo4j group ID for multi-tenant isolation.
        threshold: Relevance threshold for fact retrieval.
        enable_drilldown: Enable entity drilldown for ENUMERATION questions.
        enable_refinement: Enable answer refinement loop.

    Returns:
        Tuple of (results list, summary).
    """
    # Load Q&A dataset
    print(f"Loading Q&A dataset from {qa_file}...")
    qa_dataset = load_qa_dataset(qa_file)
    total_questions = len(qa_dataset)

    if limit:
        qa_dataset = qa_dataset[:limit]
        print(f"Limited to {len(qa_dataset)} of {total_questions} questions")
    else:
        print(f"Loaded {total_questions} questions")

    # Initialize V6 config
    print("\nInitializing V6 Knowledge Graph Pipeline...")
    print(f"  Group ID: {group_id}")
    print(f"  Relevance threshold: {threshold}")
    print(f"  Entity drilldown: {'enabled' if enable_drilldown else 'disabled'}")
    print(f"  Refinement loop: {'enabled' if enable_refinement else 'disabled'}")

    config = ResearcherConfig(
        relevance_threshold=threshold,
        enable_entity_drilldown=enable_drilldown,
        enable_refinement_loop=enable_refinement,
        enable_gap_expansion=True,
        enable_llm_fact_filter=True,
    )

    # Initialize judge
    print("\nInitializing LLM judge...")
    judge = LLMJudge()

    # Track progress
    completed = 0
    total = len(qa_dataset)
    results: list[EvalResult] = []

    # Semaphore for concurrency control
    # V6 is more resource-intensive, so default concurrency is lower
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_with_progress(qa: dict) -> EvalResult:
        nonlocal completed
        async with semaphore:
            result = await evaluate_question(judge, qa, group_id, config)
            completed += 1
            # Print progress
            verdict_str = result.verdict.value
            print(
                f"[{completed}/{total}] Q{result.question_id}: {verdict_str} "
                f"({result.timing_ms}ms)"
            )
            return result

    # Run all evaluations concurrently
    print(f"\nEvaluating {total} questions with concurrency={concurrency}...")
    print("=" * 60)

    eval_start = time.perf_counter()
    results = await asyncio.gather(*[
        evaluate_with_progress(qa) for qa in qa_dataset
    ])
    eval_duration = time.perf_counter() - eval_start

    print("=" * 60)
    print(f"Evaluation completed in {eval_duration:.1f}s")

    # Create summary
    summary = EvalSummary.from_results("v6_knowledge_graph", list(results))

    return list(results), summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate V6 Knowledge Graph Pipeline on BeigeOA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate all questions
    uv run python -m testing.eval_v6

    # Quick test with 5 questions
    uv run python -m testing.eval_v6 --limit 5

    # Higher concurrency (V6 is resource-intensive)
    uv run python -m testing.eval_v6 --concurrency 5

    # Use a different group ID
    uv run python -m testing.eval_v6 --group-id beige_book

    # Disable optional features for speed
    uv run python -m testing.eval_v6 --disable-drilldown --disable-refinement
""",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of questions to evaluate",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of concurrent evaluations (default: 3, V6 is resource-intensive)",
    )
    parser.add_argument(
        "--qa-file",
        type=str,
        default=DEFAULT_QA_FILE,
        help=f"Path to Q&A dataset JSON (default: {DEFAULT_QA_FILE})",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="default",
        help="Neo4j group ID for multi-tenant isolation (default: default)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Relevance threshold for fact retrieval (default: 0.65)",
    )
    parser.add_argument(
        "--disable-drilldown",
        action="store_true",
        help="Disable entity drilldown for ENUMERATION questions",
    )
    parser.add_argument(
        "--disable-refinement",
        action="store_true",
        help="Disable answer refinement loop",
    )
    args = parser.parse_args()

    # Run evaluation
    try:
        results, summary = asyncio.run(
            run_evaluation(
                qa_file=args.qa_file,
                limit=args.limit,
                concurrency=args.concurrency,
                group_id=args.group_id,
                threshold=args.threshold,
                enable_drilldown=not args.disable_drilldown,
                enable_refinement=not args.disable_refinement,
            )
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)

    # Save results
    config = {
        "system": "v6_knowledge_graph",
        "embedding_model": "text-embedding-3-large",
        "synthesis_model": "gpt-5.1",
        "group_id": args.group_id,
        "threshold": args.threshold,
        "enable_drilldown": not args.disable_drilldown,
        "enable_refinement": not args.disable_refinement,
        "qa_file": args.qa_file,
    }
    output_file = save_eval_results(results, summary, config)

    # Print summary
    print_eval_summary(summary)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
