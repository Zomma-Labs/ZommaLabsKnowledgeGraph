#!/usr/bin/env python3
"""Score GraphRAG results using LLM judge."""

import sys
import json
import asyncio
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from testing.common import LLMJudge, EvalSummary, JudgeVerdict


async def judge_results(results_file: str, concurrency: int = 10):
    """Judge GraphRAG results."""

    # Load results
    with open(results_file) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results from {results_file}")

    # Initialize judge
    judge = LLMJudge(model="gpt-4o-mini")

    # Prepare judgments
    judgments = [
        (r["question"], r["expected"], r["graphrag_answer"])
        for r in results
    ]

    print(f"Judging {len(judgments)} answers with concurrency={concurrency}...")
    start = time.time()

    verdicts = await judge.batch_judge(judgments, max_concurrency=concurrency)

    elapsed = time.time() - start
    print(f"Judging completed in {elapsed:.1f}s")

    # Update results with verdicts
    for i, (verdict, reasoning) in enumerate(verdicts):
        results[i]["verdict"] = verdict.value
        results[i]["judge_reasoning"] = reasoning

    # Calculate summary
    correct = sum(1 for r in results if r["verdict"] == "correct")
    partial = sum(1 for r in results if r["verdict"] == "partially_correct")
    abstained = sum(1 for r in results if r["verdict"] == "abstained")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")
    total = len(results)

    summary = {
        "total": total,
        "correct": correct,
        "partially_correct": partial,
        "abstained": abstained,
        "incorrect": incorrect,
        "correct_pct": round(100 * correct / total, 1),
        "partial_pct": round(100 * partial / total, 1),
        "abstained_pct": round(100 * abstained / total, 1),
        "incorrect_pct": round(100 * incorrect / total, 1),
        "accuracy_pct": round(100 * (correct + partial) / total, 1),
        "avg_time_ms": round(sum(r["time_ms"] for r in results) / total, 0),
    }

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total questions: {total}")
    print(f"Correct:         {correct} ({summary['correct_pct']}%)")
    print(f"Partial:         {partial} ({summary['partial_pct']}%)")
    print(f"Abstained:       {abstained} ({summary['abstained_pct']}%)")
    print(f"Incorrect:       {incorrect} ({summary['incorrect_pct']}%)")
    print(f"Accuracy:        {summary['accuracy_pct']}% (correct + partial)")
    print(f"Avg time:        {summary['avg_time_ms']}ms")

    # Show incorrect answers
    incorrect_results = [r for r in results if r["verdict"] == "incorrect"]
    if incorrect_results:
        print("\n" + "="*60)
        print(f"INCORRECT ANSWERS ({len(incorrect_results)})")
        print("="*60)
        for r in incorrect_results[:5]:  # Show first 5
            print(f"\nQ{r['question_id']}: {r['question'][:60]}...")
            print(f"Expected: {r['expected'][:100]}...")
            print(f"Got: {r['graphrag_answer'][:100]}...")
            print(f"Reason: {r['judge_reasoning'][:150]}...")

    # Save judged results
    output = {
        "metadata": {
            "system": "graphrag",
            "source_file": results_file,
            "judge_model": "gpt-4o-mini",
        },
        "summary": summary,
        "results": results,
    }

    output_file = results_file.replace(".json", "_judged.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nJudged results saved to: {output_file}")
    return summary


if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else "eval/graphrag_beige_test_20260125_122418.json"
    asyncio.run(judge_results(results_file))
