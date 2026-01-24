#!/usr/bin/env python3
"""
Run all RAG evaluations and generate comparison.

This script runs all RAG system evaluations (SimpleRAG, DeepRAG, GraphRAG, V6)
and generates a comparison report showing how each system performed on the same
questions.

Usage:
    uv run python -m testing.run_all [--limit 5] [--skip-graphrag] [--skip-v6]

Options:
    --limit          Limit number of questions to evaluate (default: all)
    --skip-graphrag  Skip GraphRAG evaluation (useful if index not built)
    --skip-v6        Skip V6 Knowledge Graph evaluation (useful if Neo4j not populated)
    --concurrency    Number of concurrent evaluations per system (default: 5)
    --qa-file        Path to Q&A dataset (default: eval/Biege_OA.json)
    --chunk-file     Path to chunks JSONL (default: src/chunker/SAVED/BeigeBook_20251015.jsonl)
    --index-path     Path to GraphRAG index (default: ./graphrag_beige)
    --group-id       Neo4j group ID for V6 (default: default)
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from testing.common import EvalResult, EvalSummary, JudgeVerdict

# Import run_evaluation functions from each evaluation module
from testing.eval_simple_rag import run_evaluation as run_simple_rag
from testing.eval_deep_rag import run_evaluation as run_deep_rag
from testing.eval_graphrag import run_evaluation as run_graphrag
from testing.eval_v6 import run_evaluation as run_v6

# Default file paths
DEFAULT_CHUNK_FILE = "src/chunker/SAVED/BeigeBook_20251015.jsonl"
DEFAULT_QA_FILE = "eval/Biege_OA.json"
DEFAULT_INDEX_PATH = "./graphrag_beige"
DEFAULT_OUTPUT_DIR = "eval"


async def run_all_evaluations(
    limit: int | None = None,
    skip_graphrag: bool = False,
    skip_v6: bool = False,
    concurrency: int = 5,
    chunk_file: str = DEFAULT_CHUNK_FILE,
    qa_file: str = DEFAULT_QA_FILE,
    index_path: str = DEFAULT_INDEX_PATH,
    group_id: str = "default",
) -> dict[str, tuple[list[EvalResult], EvalSummary]]:
    """Run all evaluations and collect results.

    Args:
        limit: Optional limit on number of questions to evaluate.
        skip_graphrag: Whether to skip GraphRAG evaluation.
        skip_v6: Whether to skip V6 Knowledge Graph evaluation.
        concurrency: Maximum concurrent evaluations per system.
        chunk_file: Path to the JSONL file containing chunks.
        qa_file: Path to the Q&A dataset JSON file.
        index_path: Path to the GraphRAG index directory.
        group_id: Neo4j group ID for V6 multi-tenant isolation.

    Returns:
        Dictionary mapping system name to (results, summary) tuple.
    """
    all_results: dict[str, tuple[list[EvalResult], EvalSummary]] = {}

    # Run SimpleRAG
    print("\n" + "=" * 60)
    print("Running Simple RAG Evaluation...")
    print("=" * 60)
    try:
        simple_results, simple_summary = await run_simple_rag(
            chunk_file=chunk_file,
            qa_file=qa_file,
            limit=limit,
            concurrency=concurrency,
        )
        all_results["simple_rag"] = (simple_results, simple_summary)
        print(f"\nSimple RAG: {simple_summary.correct}/{simple_summary.total_questions} correct")
    except Exception as e:
        print(f"Error running Simple RAG evaluation: {e}", file=sys.stderr)

    # Run DeepRAG
    print("\n" + "=" * 60)
    print("Running Deep RAG Evaluation...")
    print("=" * 60)
    try:
        deep_results, deep_summary = await run_deep_rag(
            chunk_file=chunk_file,
            qa_file=qa_file,
            limit=limit,
            concurrency=concurrency,
        )
        all_results["deep_rag"] = (deep_results, deep_summary)
        print(f"\nDeep RAG: {deep_summary.correct}/{deep_summary.total_questions} correct")
    except Exception as e:
        print(f"Error running Deep RAG evaluation: {e}", file=sys.stderr)

    # Run GraphRAG (optional - may not have index)
    if not skip_graphrag:
        print("\n" + "=" * 60)
        print("Running GraphRAG Evaluation...")
        print("=" * 60)
        try:
            graph_results, graph_summary = await run_graphrag(
                index_path=index_path,
                qa_file=qa_file,
                limit=limit,
                concurrency=concurrency,
            )
            if graph_summary is not None:
                all_results["graphrag"] = (graph_results, graph_summary)
                print(f"\nGraphRAG: {graph_summary.correct}/{graph_summary.total_questions} correct")
            else:
                print("\nGraphRAG evaluation skipped (index not found)")
        except Exception as e:
            print(f"Error running GraphRAG evaluation: {e}", file=sys.stderr)
    else:
        print("\n" + "=" * 60)
        print("Skipping GraphRAG Evaluation (--skip-graphrag flag set)")
        print("=" * 60)

    # Run V6 Knowledge Graph (optional - may not have Neo4j populated)
    if not skip_v6:
        print("\n" + "=" * 60)
        print("Running V6 Knowledge Graph Evaluation...")
        print("=" * 60)
        try:
            # V6 is more resource-intensive, use lower concurrency
            v6_concurrency = min(concurrency, 3)
            v6_results, v6_summary = await run_v6(
                qa_file=qa_file,
                limit=limit,
                concurrency=v6_concurrency,
                group_id=group_id,
            )
            all_results["v6_knowledge_graph"] = (v6_results, v6_summary)
            print(f"\nV6: {v6_summary.correct}/{v6_summary.total_questions} correct")
        except Exception as e:
            print(f"Error running V6 evaluation: {e}", file=sys.stderr)
    else:
        print("\n" + "=" * 60)
        print("Skipping V6 Knowledge Graph Evaluation (--skip-v6 flag set)")
        print("=" * 60)

    return all_results


def build_per_question_comparison(
    results: dict[str, tuple[list[EvalResult], EvalSummary]]
) -> list[dict[str, Any]]:
    """Build per-question comparison across all systems.

    Args:
        results: Dictionary mapping system name to (results, summary) tuple.

    Returns:
        List of dicts with per-question comparison data.
    """
    if not results:
        return []

    # Get the first system's results as the baseline for question data
    first_system = list(results.keys())[0]
    first_results = results[first_system][0]

    # Build question ID to result mapping for each system
    system_maps: dict[str, dict[int, EvalResult]] = {}
    for system_name, (eval_results, _) in results.items():
        system_maps[system_name] = {r.question_id: r for r in eval_results}

    # Build comparison list
    comparison = []
    for result in first_results:
        question_data: dict[str, Any] = {
            "question_id": result.question_id,
            "question": result.question,
            "expected_answer": result.expected_answer,
        }

        # Add each system's result
        for system_name in results.keys():
            if result.question_id in system_maps[system_name]:
                sys_result = system_maps[system_name][result.question_id]
                question_data[system_name] = {
                    "verdict": sys_result.verdict.value,
                    "answer": sys_result.system_answer,
                    "timing_ms": sys_result.timing_ms,
                }
            else:
                question_data[system_name] = {
                    "verdict": "missing",
                    "answer": "",
                    "timing_ms": 0,
                }

        comparison.append(question_data)

    return comparison


def generate_comparison(
    results: dict[str, tuple[list[EvalResult], EvalSummary]],
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Generate comparison report and save to file.

    Args:
        results: Dictionary mapping system name to (results, summary) tuple.
        output_dir: Directory to save the comparison report.

    Returns:
        Path to the saved comparison file.
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = output_path / f"comparison_{timestamp}.json"

    # Build summaries dict
    summaries = {}
    for system_name, (_, summary) in results.items():
        summaries[system_name] = {
            "total_questions": summary.total_questions,
            "correct": summary.correct,
            "partially_correct": summary.partially_correct,
            "abstained": summary.abstained,
            "incorrect": summary.incorrect,
            "correct_pct": summary.correct_pct,
            "partially_correct_pct": summary.partially_correct_pct,
            "abstained_pct": summary.abstained_pct,
            "incorrect_pct": summary.incorrect_pct,
            "accuracy_pct": summary.accuracy_pct,
            "avg_time_ms": summary.avg_time_ms,
        }

    # Build per-question comparison
    per_question = build_per_question_comparison(results)

    # Build comparison analysis
    analysis = analyze_comparison(results, per_question)

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "systems_compared": list(results.keys()),
        "summaries": summaries,
        "analysis": analysis,
        "per_question_comparison": per_question,
    }

    # Save
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    return str(comparison_file)


def analyze_comparison(
    results: dict[str, tuple[list[EvalResult], EvalSummary]],
    per_question: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze comparison data to find interesting insights.

    Args:
        results: Dictionary mapping system name to (results, summary) tuple.
        per_question: Per-question comparison data.

    Returns:
        Analysis dict with insights.
    """
    if not results or not per_question:
        return {}

    systems = list(results.keys())
    analysis: dict[str, Any] = {}

    # Questions where all systems agree
    all_correct = []
    all_incorrect = []
    disagreements = []

    for q in per_question:
        verdicts = [q.get(sys, {}).get("verdict") for sys in systems]
        verdicts = [v for v in verdicts if v and v != "missing"]

        if all(v == "correct" for v in verdicts):
            all_correct.append(q["question_id"])
        elif all(v == "incorrect" for v in verdicts):
            all_incorrect.append(q["question_id"])
        elif len(set(verdicts)) > 1:
            disagreements.append({
                "question_id": q["question_id"],
                "question": q["question"][:100] + "..." if len(q["question"]) > 100 else q["question"],
                "verdicts": {sys: q.get(sys, {}).get("verdict", "missing") for sys in systems},
            })

    analysis["all_systems_correct"] = {
        "count": len(all_correct),
        "question_ids": all_correct,
    }
    analysis["all_systems_incorrect"] = {
        "count": len(all_incorrect),
        "question_ids": all_incorrect,
    }
    analysis["disagreements"] = {
        "count": len(disagreements),
        "details": disagreements[:20],  # Limit to first 20 for readability
    }

    # Per-system unique successes (correct when others are not)
    for system in systems:
        unique_successes = []
        for q in per_question:
            sys_verdict = q.get(system, {}).get("verdict")
            other_verdicts = [
                q.get(s, {}).get("verdict")
                for s in systems
                if s != system
            ]
            if sys_verdict == "correct" and all(v != "correct" for v in other_verdicts if v):
                unique_successes.append(q["question_id"])

        analysis[f"{system}_unique_successes"] = {
            "count": len(unique_successes),
            "question_ids": unique_successes,
        }

    return analysis


def print_comparison_table(results: dict[str, tuple[list[EvalResult], EvalSummary]]) -> None:
    """Print a nice comparison table to console.

    Args:
        results: Dictionary mapping system name to (results, summary) tuple.
    """
    if not results:
        print("\nNo results to compare.")
        return

    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)

    # Header
    print(
        f"{'System':<15} "
        f"{'Correct':<12} "
        f"{'Partial':<12} "
        f"{'Abstained':<12} "
        f"{'Incorrect':<12} "
        f"{'Accuracy':<10} "
        f"{'Avg Time':<10}"
    )
    print("-" * 100)

    # Rows
    for system_name, (_, summary) in results.items():
        print(
            f"{system_name:<15} "
            f"{summary.correct:>3} ({summary.correct_pct:5.1f}%) "
            f"{summary.partially_correct:>3} ({summary.partially_correct_pct:5.1f}%) "
            f"{summary.abstained:>3} ({summary.abstained_pct:5.1f}%) "
            f"{summary.incorrect:>3} ({summary.incorrect_pct:5.1f}%) "
            f"{summary.accuracy_pct:>6.1f}%   "
            f"{summary.avg_time_ms:>6.0f}ms"
        )

    print("=" * 100)

    # Best system by accuracy
    if results:
        best_system = max(results.keys(), key=lambda s: results[s][1].accuracy_pct)
        best_accuracy = results[best_system][1].accuracy_pct
        print(f"\nBest System: {best_system} ({best_accuracy:.1f}% accuracy)")


def print_analysis_summary(comparison_file: str) -> None:
    """Print a summary of the analysis section from the comparison file.

    Args:
        comparison_file: Path to the comparison JSON file.
    """
    with open(comparison_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    analysis = data.get("analysis", {})
    if not analysis:
        return

    print("\n" + "-" * 100)
    print("ANALYSIS")
    print("-" * 100)

    all_correct = analysis.get("all_systems_correct", {})
    all_incorrect = analysis.get("all_systems_incorrect", {})
    disagreements = analysis.get("disagreements", {})

    print(f"Questions all systems got CORRECT:   {all_correct.get('count', 0)}")
    print(f"Questions all systems got INCORRECT: {all_incorrect.get('count', 0)}")
    print(f"Questions with disagreement:         {disagreements.get('count', 0)}")

    # Print unique successes for each system
    systems = data.get("systems_compared", [])
    if systems:
        print("\nUnique successes (correct when all others failed):")
        for system in systems:
            key = f"{system}_unique_successes"
            if key in analysis:
                count = analysis[key].get("count", 0)
                print(f"  {system}: {count}")

    # Show some disagreement details
    if disagreements.get("details"):
        print("\nNotable disagreements (first 5):")
        for detail in disagreements["details"][:5]:
            print(f"  Q{detail['question_id']}: {detail['question']}")
            for sys, verdict in detail["verdicts"].items():
                print(f"    - {sys}: {verdict}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all RAG evaluations and generate comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all evaluations
    uv run python -m testing.run_all

    # Quick test with 5 questions
    uv run python -m testing.run_all --limit 5

    # Skip GraphRAG if index not built
    uv run python -m testing.run_all --skip-graphrag

    # Skip V6 if Neo4j not populated
    uv run python -m testing.run_all --skip-v6

    # Run only SimpleRAG and DeepRAG
    uv run python -m testing.run_all --skip-graphrag --skip-v6

    # Higher concurrency
    uv run python -m testing.run_all --concurrency 10

    # Use a specific Neo4j group for V6
    uv run python -m testing.run_all --group-id beige_book

    # Custom files
    uv run python -m testing.run_all --chunk-file path/to/chunks.jsonl --qa-file path/to/qa.json
""",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of questions to evaluate per system",
    )
    parser.add_argument(
        "--skip-graphrag",
        action="store_true",
        help="Skip GraphRAG evaluation",
    )
    parser.add_argument(
        "--skip-v6",
        action="store_true",
        help="Skip V6 Knowledge Graph evaluation",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="default",
        help="Neo4j group ID for V6 multi-tenant isolation (default: default)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of concurrent evaluations per system (default: 20)",
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
    parser.add_argument(
        "--index-path",
        type=str,
        default=DEFAULT_INDEX_PATH,
        help=f"Path to GraphRAG index (default: {DEFAULT_INDEX_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Track total time
    total_start = time.perf_counter()

    # Run evaluations
    try:
        results = asyncio.run(
            run_all_evaluations(
                limit=args.limit,
                skip_graphrag=args.skip_graphrag,
                skip_v6=args.skip_v6,
                concurrency=args.concurrency,
                chunk_file=args.chunk_file,
                qa_file=args.qa_file,
                index_path=args.index_path,
                group_id=args.group_id,
            )
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)

    total_duration = time.perf_counter() - total_start

    # Generate comparison
    if results:
        comparison_file = generate_comparison(results, args.output_dir)
        print_comparison_table(results)
        print_analysis_summary(comparison_file)
        print(f"\nComparison saved to: {comparison_file}")
        print(f"Total evaluation time: {total_duration:.1f}s")
    else:
        print("\nNo evaluations completed successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
