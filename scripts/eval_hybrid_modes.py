#!/usr/bin/env python3
"""
Evaluate Hybrid Pipeline Modes
==============================

Compares:
1. DETERMINISTIC - Pure multi-strategy retrieval
2. HYBRID - Deterministic base + targeted exploration
3. AGENT - Original LLM-controlled (for comparison)

Tests consistency and accuracy on Beige Book Q&A.

Usage:
    uv run scripts/eval_hybrid_modes.py --mode deterministic
    uv run scripts/eval_hybrid_modes.py --mode all --runs 3
"""

import sys
import os
import json
import asyncio
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.querying_system.deep_research.hybrid_pipeline import (
    HybridDeepResearchPipeline,
    ResearchMode,
)


def load_qa_pairs(filepath: str = "Biege_OA.json", num_questions: int = 5) -> list[dict]:
    """Load Q&A pairs from the Beige Book JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['qa_pairs'][:num_questions]


def evaluate_answer(generated: str, expected: str) -> dict:
    """Simple keyword coverage evaluation."""
    stop_words = {'about', 'after', 'before', 'being', 'between', 'could',
                  'during', 'would', 'should', 'these', 'those', 'their',
                  'there', 'where', 'which', 'while', 'other', 'through'}

    expected_words = set(
        word.lower().strip('.,;:()')
        for word in expected.split()
        if len(word) > 4 and word.lower() not in stop_words
    )

    generated_lower = generated.lower()
    found_terms = [term for term in expected_words if term in generated_lower]
    coverage = len(found_terms) / len(expected_words) if expected_words else 0

    return {"coverage": coverage, "found": len(found_terms), "total": len(expected_words)}


async def evaluate_mode(
    mode: ResearchMode,
    qa_pairs: list[dict],
    user_id: str = "default",
    verbose: bool = False
) -> list[dict]:
    """Evaluate a single mode on all questions."""
    pipeline = HybridDeepResearchPipeline(user_id=user_id, mode=mode)
    results = []

    for qa in qa_pairs:
        print(f"\n  Q{qa['id']}: {qa['question'][:50]}...")

        start = datetime.now()
        result = await pipeline.query_async(qa['question'], verbose=verbose)
        elapsed = (datetime.now() - start).total_seconds()

        eval_result = evaluate_answer(result.answer, qa['answer'])

        results.append({
            "question_id": qa['id'],
            "question": qa['question'],
            "expected": qa['answer'],
            "generated": result.answer,
            "coverage": eval_result['coverage'],
            "elapsed_seconds": elapsed,
            "num_findings": len(result.findings),
            "research_time_ms": result.research_time_ms,
            "synthesis_time_ms": result.synthesis_time_ms,
        })

        status = "✓" if eval_result['coverage'] > 0.5 else "✗"
        print(f"    {status} Coverage: {eval_result['coverage']*100:.0f}% | Time: {elapsed:.1f}s")

    return results


async def run_consistency_test(
    mode: ResearchMode,
    question: str,
    num_runs: int = 3,
    user_id: str = "default"
) -> dict:
    """Test if the same question returns consistent results across runs."""
    print(f"\n  Testing consistency over {num_runs} runs...")

    answers = []
    coverages = []

    for run in range(num_runs):
        pipeline = HybridDeepResearchPipeline(user_id=user_id, mode=mode)
        result = await pipeline.query_async(question, verbose=False)
        answers.append(result.answer)
        print(f"    Run {run+1}: {len(result.answer)} chars")

    # Check if answers are similar (not identical due to LLM synthesis variance)
    # But the retrieved evidence should be the same
    first_len = len(answers[0])
    length_variance = max(abs(len(a) - first_len) for a in answers)

    return {
        "num_runs": num_runs,
        "length_variance": length_variance,
        "consistent": length_variance < 500  # Allow some synthesis variance
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["deterministic", "hybrid", "agent", "all"], default="deterministic")
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for consistency test")
    parser.add_argument("--user-id", default="default")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("HYBRID PIPELINE MODE EVALUATION")
    print("=" * 70)
    print(f"Questions: {args.num_questions}")
    print(f"Runs: {args.runs}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    qa_pairs = load_qa_pairs(num_questions=args.num_questions)

    if args.mode == "all":
        modes = [ResearchMode.DETERMINISTIC, ResearchMode.HYBRID, ResearchMode.AGENT]
    else:
        modes = [ResearchMode(args.mode)]

    all_results = {}

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"MODE: {mode.value.upper()}")
        print("=" * 70)

        # Run evaluation
        if args.runs > 1:
            # Multiple runs - test consistency
            run_results = []
            for run in range(args.runs):
                print(f"\n--- Run {run+1}/{args.runs} ---")
                results = await evaluate_mode(mode, qa_pairs, args.user_id, args.verbose)
                run_results.append(results)

            # Calculate consistency
            avg_coverages = []
            for i in range(len(qa_pairs)):
                coverages = [run_results[r][i]['coverage'] for r in range(args.runs)]
                avg_coverages.append(sum(coverages) / len(coverages))
                variance = max(coverages) - min(coverages)
                print(f"  Q{i+1} coverage variance: {variance*100:.1f}%")

            all_results[mode.value] = {
                "avg_coverage": sum(avg_coverages) / len(avg_coverages),
                "runs": run_results,
            }
        else:
            # Single run
            results = await evaluate_mode(mode, qa_pairs, args.user_id, args.verbose)
            avg_coverage = sum(r['coverage'] for r in results) / len(results)
            avg_time = sum(r['elapsed_seconds'] for r in results) / len(results)

            all_results[mode.value] = {
                "avg_coverage": avg_coverage,
                "avg_time": avg_time,
                "results": results,
            }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    for mode_name, data in all_results.items():
        print(f"\n{mode_name.upper()}:")
        print(f"  Average Coverage: {data['avg_coverage']*100:.1f}%")
        if 'avg_time' in data:
            print(f"  Average Time: {data['avg_time']:.1f}s")

    # Save results
    output_file = f"eval_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_questions": args.num_questions,
            "num_runs": args.runs,
            "results": all_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
