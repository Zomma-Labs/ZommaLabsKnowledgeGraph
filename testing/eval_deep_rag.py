#!/usr/bin/env python3
"""
Evaluate Deep RAG (multi-query + RRF) on BeigeOA dataset.

This script evaluates the DeepRAG system (query decomposition + parallel vector
search + Reciprocal Rank Fusion) against the Beige Book Q&A dataset using an
LLM judge.

Usage:
    uv run python -m testing.eval_deep_rag [--limit 5] [--concurrency 5]

Options:
    --limit        Limit number of questions to evaluate (default: all 75)
    --concurrency  Number of concurrent evaluations (default: 5)
    --top-k        Number of fused chunks for synthesis (default: 20)
    --per-query-k  Number of chunks per sub-query search (default: 20)
    --rrf-k        RRF constant k (default: 60)
    --qa-file      Path to Q&A dataset (default: eval/Biege_OA.json)
    --chunk-file   Path to chunks JSONL (default: src/chunker/SAVED/BeigeBook_20251015.jsonl)
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime

from testing.common import (
    load_qa_dataset,
    save_eval_results,
    print_eval_summary,
    EvalResult,
    EvalSummary,
    LLMJudge,
)
from testing.systems import DeepRAG

# Default file paths
DEFAULT_CHUNK_FILE = "src/chunker/SAVED/BeigeBook_20251015.jsonl"
DEFAULT_QA_FILE = "eval/Biege_OA.json"


async def evaluate_question(
    rag: DeepRAG,
    judge: LLMJudge,
    qa: dict,
    top_k: int,
    per_query_k: int,
) -> EvalResult:
    """Evaluate a single question.

    Args:
        rag: The DeepRAG instance to query.
        judge: The LLM judge for evaluation.
        qa: The Q&A pair dict with id, question, answer.
        top_k: Number of fused chunks for synthesis.
        per_query_k: Number of chunks per sub-query search.

    Returns:
        EvalResult with verdict and timing.
    """
    start_time = time.perf_counter()

    # Query the RAG system
    answer, chunks = await rag.query(qa["question"], top_k=top_k, per_query_k=per_query_k)

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
    chunk_file: str = DEFAULT_CHUNK_FILE,
    qa_file: str = DEFAULT_QA_FILE,
    limit: int | None = None,
    concurrency: int = 5,
    top_k: int = 20,
    per_query_k: int = 20,
) -> tuple[list[EvalResult], EvalSummary]:
    """Run the full evaluation.

    Args:
        chunk_file: Path to the JSONL file containing chunks.
        qa_file: Path to the Q&A dataset JSON file.
        limit: Optional limit on number of questions to evaluate.
        concurrency: Maximum concurrent evaluations.
        top_k: Number of fused chunks for synthesis.
        per_query_k: Number of chunks per sub-query search.

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

    # Initialize DeepRAG (this embeds all chunks if not cached)
    print(f"\nInitializing DeepRAG with chunks from {chunk_file}...")
    rag = DeepRAG(chunk_file)

    # Initialize judge
    print("\nInitializing LLM judge...")
    judge = LLMJudge()

    # Track progress
    completed = 0
    total = len(qa_dataset)
    results: list[EvalResult] = []

    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_with_progress(qa: dict) -> EvalResult:
        nonlocal completed
        async with semaphore:
            result = await evaluate_question(rag, judge, qa, top_k, per_query_k)
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
    print(f"Parameters: top_k={top_k}, per_query_k={per_query_k}")
    print("=" * 60)

    eval_start = time.perf_counter()
    results = await asyncio.gather(*[
        evaluate_with_progress(qa) for qa in qa_dataset
    ])
    eval_duration = time.perf_counter() - eval_start

    print("=" * 60)
    print(f"Evaluation completed in {eval_duration:.1f}s")

    # Create summary
    summary = EvalSummary.from_results("deep_rag", list(results))

    return list(results), summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Deep RAG (multi-query + RRF) on BeigeOA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate all questions
    uv run python -m testing.eval_deep_rag

    # Quick test with 5 questions
    uv run python -m testing.eval_deep_rag --limit 5

    # Higher concurrency
    uv run python -m testing.eval_deep_rag --concurrency 10

    # Custom retrieval parameters
    uv run python -m testing.eval_deep_rag --top-k 25 --per-query-k 30

    # Custom RRF constant
    uv run python -m testing.eval_deep_rag --rrf-k 80

    # Custom chunk file
    uv run python -m testing.eval_deep_rag --chunk-file path/to/chunks.jsonl
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
        default=20,
        help="Number of concurrent evaluations (default: 20)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of fused chunks for synthesis (default: 20)",
    )
    parser.add_argument(
        "--per-query-k",
        type=int,
        default=20,
        help="Number of chunks per sub-query search (default: 20)",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant k (default: 60, as recommended in literature)",
    )
    parser.add_argument(
        "--qa-file",
        type=str,
        default=DEFAULT_QA_FILE,
        help=f"Path to Q&A dataset JSON (default: {DEFAULT_QA_FILE})",
    )
    parser.add_argument(
        "--chunk-file",
        type=str,
        default=DEFAULT_CHUNK_FILE,
        help=f"Path to chunks JSONL (default: {DEFAULT_CHUNK_FILE})",
    )
    args = parser.parse_args()

    # Run evaluation
    try:
        results, summary = asyncio.run(
            run_evaluation(
                chunk_file=args.chunk_file,
                qa_file=args.qa_file,
                limit=args.limit,
                concurrency=args.concurrency,
                top_k=args.top_k,
                per_query_k=args.per_query_k,
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
        "embedding_model": "text-embedding-3-large",
        "synthesis_model": "gpt-5.1",
        "decomposition_model": "gpt-5.1",
        "top_k": args.top_k,
        "per_query_k": args.per_query_k,
        "rrf_k": args.rrf_k,
        "chunk_file": args.chunk_file,
        "qa_file": args.qa_file,
    }
    output_file = save_eval_results(results, summary, config)

    # Print summary
    print_eval_summary(summary)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
