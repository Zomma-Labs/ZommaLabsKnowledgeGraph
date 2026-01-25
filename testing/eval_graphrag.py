#!/usr/bin/env python3
"""
Evaluate Microsoft GraphRAG on BeigeOA dataset.

This script evaluates Microsoft's GraphRAG system (community-based knowledge graph)
against the Beige Book Q&A dataset using an LLM judge.

Usage:
    uv run python -m testing.eval_graphrag [--limit 5] [--concurrency 5]

Options:
    --limit          Limit number of questions to evaluate (default: all 75)
    --concurrency    Number of concurrent evaluations (default: 20)
    --index-path     Path to GraphRAG index (default: ./graphrag_beige)
    --search-type    Search mode: "global" or "local" (default: global)
    --community-level  Community hierarchy level for retrieval (default: 2)
    --qa-file        Path to Q&A dataset (default: eval/Biege_OA.json)

Note: Requires a pre-built GraphRAG index. Run setup_graphrag.py first.
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
from testing.systems import GraphRAGSystem, GraphRAGIndexNotFoundError

# Default file paths
DEFAULT_INDEX_PATH = "./graphrag_beige"
DEFAULT_QA_FILE = "eval/Biege_OA.json"


async def evaluate_question(
    rag: GraphRAGSystem,
    judge: LLMJudge,
    qa: dict,
) -> EvalResult:
    """Evaluate a single question.

    Args:
        rag: The GraphRAGSystem instance to query.
        judge: The LLM judge for evaluation.
        qa: The Q&A pair dict with id, question, answer.

    Returns:
        EvalResult with verdict and timing.
    """
    start_time = time.perf_counter()

    # Query the RAG system
    answer, evidence = await rag.query(qa["question"])

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
        retrieved_chunks=evidence,  # Contains community reports or entities
        timing_ms=timing_ms,
    )


async def run_evaluation(
    index_path: str = DEFAULT_INDEX_PATH,
    qa_file: str = DEFAULT_QA_FILE,
    search_type: str = "global",
    community_level: int = 2,
    limit: int | None = None,
    concurrency: int = 5,
) -> tuple[list[EvalResult], EvalSummary] | tuple[list, None]:
    """Run the full evaluation.

    Args:
        index_path: Path to the GraphRAG index directory.
        qa_file: Path to the Q&A dataset JSON file.
        search_type: Type of search - "global" or "local".
        community_level: Community hierarchy level for retrieval.
        limit: Optional limit on number of questions to evaluate.
        concurrency: Maximum concurrent evaluations.

    Returns:
        Tuple of (results list, summary), or ([], None) if index not found.
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

    # Initialize GraphRAGSystem (validates index exists)
    print(f"\nInitializing GraphRAG from {index_path}...")
    print(f"Search type: {search_type}, Community level: {community_level}")
    try:
        rag = GraphRAGSystem(
            index_path=index_path,
            search_type=search_type,
            community_level=community_level,
        )
    except GraphRAGIndexNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print(
            "\nTo build the GraphRAG index, run:",
            file=sys.stderr,
        )
        print(
            "    uv run python -m testing.setup.setup_graphrag --chunk-file <your_chunks.jsonl>",
            file=sys.stderr,
        )
        return [], None

    # Print index info
    info = rag.index_info
    print(f"\nIndex info:")
    print(f"  Root path: {info.get('root_path', 'N/A')}")
    print(f"  Search type: {info.get('search_type', 'N/A')}")
    if 'entities' in info:
        print(f"  Entities: {info['entities']}")
    if 'community_reports' in info:
        print(f"  Community Reports: {info['community_reports']}")

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
            print(f"[Starting] Q{qa['id']}: {qa['question'][:50]}...", flush=True)
            result = await evaluate_question(rag, judge, qa)
            completed += 1
            # Print progress
            verdict_str = result.verdict.value
            print(
                f"[{completed}/{total}] Q{result.question_id}: {verdict_str} "
                f"({result.timing_ms}ms)",
                flush=True,
            )
            return result

    # Run all evaluations concurrently
    print(f"\nEvaluating {total} questions with concurrency={concurrency}...")
    print(f"Using {search_type} search mode")
    print("=" * 60)

    eval_start = time.perf_counter()
    results = await asyncio.gather(*[
        evaluate_with_progress(qa) for qa in qa_dataset
    ])
    eval_duration = time.perf_counter() - eval_start

    print("=" * 60)
    print(f"Evaluation completed in {eval_duration:.1f}s")

    # Create summary with search type in the name
    system_name = f"graphrag_{search_type}"
    summary = EvalSummary.from_results(system_name, list(results))

    return list(results), summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Microsoft GraphRAG on BeigeOA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate all questions with global search
    uv run python -m testing.eval_graphrag

    # Quick test with 5 questions
    uv run python -m testing.eval_graphrag --limit 5

    # Use local search mode
    uv run python -m testing.eval_graphrag --search-type local

    # Higher concurrency
    uv run python -m testing.eval_graphrag --concurrency 10

    # Custom index path
    uv run python -m testing.eval_graphrag --index-path ./my_graphrag_index

    # Adjust community level (lower = broader, higher = more specific)
    uv run python -m testing.eval_graphrag --community-level 1
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
        "--index-path",
        type=str,
        default=DEFAULT_INDEX_PATH,
        help=f"Path to GraphRAG index (default: {DEFAULT_INDEX_PATH})",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        choices=["global", "local"],
        default="global",
        help="Search mode: 'global' uses community summaries, 'local' uses entities (default: global)",
    )
    parser.add_argument(
        "--community-level",
        type=int,
        default=2,
        help="Community hierarchy level for retrieval (default: 2). Lower = broader, higher = more specific.",
    )
    parser.add_argument(
        "--qa-file",
        type=str,
        default=DEFAULT_QA_FILE,
        help=f"Path to Q&A dataset JSON (default: {DEFAULT_QA_FILE})",
    )
    args = parser.parse_args()

    # Run evaluation
    try:
        results, summary = asyncio.run(
            run_evaluation(
                index_path=args.index_path,
                qa_file=args.qa_file,
                search_type=args.search_type,
                community_level=args.community_level,
                limit=args.limit,
                concurrency=args.concurrency,
            )
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)

    # Check if evaluation ran (index might not exist)
    if summary is None:
        print("\nEvaluation could not run. Please build the GraphRAG index first.")
        sys.exit(1)

    # Save results
    config = {
        "system": "graphrag",
        "search_type": args.search_type,
        "community_level": args.community_level,
        "index_path": args.index_path,
        "llm_model": "gpt-5.1",
        "embedding_model": "text-embedding-3-large",
        "qa_file": args.qa_file,
    }
    output_file = save_eval_results(results, summary, config)

    # Print summary
    print_eval_summary(summary)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
