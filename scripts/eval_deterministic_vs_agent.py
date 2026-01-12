#!/usr/bin/env python3
"""
Evaluation: Deterministic Retrieval vs Agent-Based Retrieval
============================================================

This script compares:
1. Deterministic retrieval + simple LLM synthesis
2. The results from multiple runs

The goal is to show that deterministic retrieval produces consistent results
while agent-based retrieval varies.

Usage:
    uv run scripts/eval_deterministic_vs_agent.py
"""

import sys
import os
import json
import asyncio
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.util.deterministic_retrieval import DeterministicRetriever
from src.util.llm_client import get_llm


# Evaluation questions with expected answers
EVAL_QUESTIONS = [
    {
        "question": "What happened to wages in Boston?",
        "expected_keywords": ["wage", "boston", "increase", "decrease", "growth", "pressure"],
    },
    {
        "question": "Which districts reported declining manufacturing activity?",
        "expected_keywords": ["manufacturing", "decline", "district"],
    },
    {
        "question": "What are the trends in consumer spending?",
        "expected_keywords": ["consumer", "spending", "retail"],
    },
]


async def answer_with_deterministic_retrieval(
    question: str,
    group_id: str = "default",
    top_k: int = 10
) -> dict:
    """
    Answer a question using deterministic retrieval + LLM synthesis.

    The retrieval is deterministic (no LLM decisions).
    Only the final synthesis uses an LLM.
    """
    # 1. Deterministic retrieval
    retriever = DeterministicRetriever(group_id=group_id)
    evidence = await retriever.search(question, top_k=top_k)

    # 2. Format evidence for LLM
    evidence_text = retriever.format_evidence_for_llm(evidence)

    # 3. Simple LLM synthesis (the ONLY LLM call)
    llm = get_llm()

    synthesis_prompt = f"""You are a financial analyst answering questions based ONLY on the provided evidence.

QUESTION: {question}

EVIDENCE:
{evidence_text}

INSTRUCTIONS:
1. Answer the question using ONLY the evidence provided above
2. If the evidence doesn't contain enough information, say so
3. Cite specific facts from the evidence
4. Be concise and factual

ANSWER:"""

    response = llm.invoke(synthesis_prompt)
    answer = response.content if hasattr(response, 'content') else str(response)

    return {
        "question": question,
        "answer": answer,
        "num_evidence": len(evidence),
        "evidence_ids": [e.fact_id for e in evidence],
        "strategies_used": list(set(s for e in evidence for s in e.found_by)),
        "top_rrf_score": evidence[0].rrf_score if evidence else 0,
    }


async def run_consistency_eval(num_runs: int = 3, group_id: str = "default"):
    """
    Run the same questions multiple times and check for consistency.
    """
    print("=" * 70)
    print("DETERMINISTIC RETRIEVAL CONSISTENCY EVALUATION")
    print("=" * 70)
    print(f"Runs: {num_runs}")
    print(f"Questions: {len(EVAL_QUESTIONS)}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    results_by_question = {q["question"]: [] for q in EVAL_QUESTIONS}

    for run_num in range(1, num_runs + 1):
        print(f"\n--- RUN {run_num} ---")

        for q in EVAL_QUESTIONS:
            question = q["question"]
            print(f"\nQ: {question}")

            result = await answer_with_deterministic_retrieval(question, group_id)

            results_by_question[question].append(result)

            print(f"  Evidence found: {result['num_evidence']}")
            print(f"  Strategies: {result['strategies_used']}")
            print(f"  Answer preview: {result['answer'][:100]}...")

    # Analyze consistency
    print("\n" + "=" * 70)
    print("CONSISTENCY ANALYSIS")
    print("=" * 70)

    consistency_scores = []

    for question, runs in results_by_question.items():
        print(f"\nQ: {question}")

        # Compare evidence IDs across runs
        evidence_sets = [set(r["evidence_ids"]) for r in runs]
        first_set = evidence_sets[0]

        identical_count = sum(1 for s in evidence_sets if s == first_set)
        consistency = identical_count / len(runs)
        consistency_scores.append(consistency)

        print(f"  Retrieval consistency: {consistency * 100:.0f}% ({identical_count}/{len(runs)} identical)")

        # Show any differences
        if consistency < 1.0:
            all_ids = set().union(*evidence_sets)
            for i, s in enumerate(evidence_sets, 1):
                missing = all_ids - s
                if missing:
                    print(f"    Run {i} missing: {len(missing)} facts")

        # Compare answer lengths (rough proxy for answer consistency)
        answer_lengths = [len(r["answer"]) for r in runs]
        length_variance = max(answer_lengths) - min(answer_lengths)
        print(f"  Answer length variance: {length_variance} chars")

    # Overall summary
    avg_consistency = sum(consistency_scores) / len(consistency_scores)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Average retrieval consistency: {avg_consistency * 100:.1f}%")

    if avg_consistency == 1.0:
        print("\n PERFECT: Retrieval is fully deterministic!")
    elif avg_consistency > 0.9:
        print("\n GOOD: Retrieval is mostly deterministic (>90%)")
    else:
        print("\n WARNING: Significant variance detected")

    # Save results
    output_file = f"eval_deterministic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_runs": num_runs,
            "avg_consistency": avg_consistency,
            "results": {q: [r for r in runs] for q, runs in results_by_question.items()},
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return avg_consistency


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--group-id", default="default", help="Tenant group ID")
    args = parser.parse_args()

    asyncio.run(run_consistency_eval(args.runs, args.group_id))


if __name__ == "__main__":
    main()
