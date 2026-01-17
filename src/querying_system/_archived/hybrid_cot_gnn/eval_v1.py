"""
Evaluate Hybrid CoT-GNN V1 Pipeline on Full QA List
====================================================

Usage:
    uv run python -m src.querying_system.hybrid_cot_gnn.eval_v1 --concurrency 5
    uv run python -m src.querying_system.hybrid_cot_gnn.eval_v1 --limit 10
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.querying_system.hybrid_cot_gnn import HybridCoTGNNPipeline
from src.util.llm_client import get_critique_llm


class JudgeVerdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


class JudgeResult(BaseModel):
    verdict: JudgeVerdict
    reasoning: str


JUDGE_PROMPT = """You are an impartial judge evaluating whether an AI agent's answer correctly addresses a question about the Federal Reserve Beige Book.

## Verdict Options:
- **correct**: The agent's answer contains all key facts from the expected answer
- **partially_correct**: The agent's answer contains some but not all key facts, or has minor inaccuracies
- **incorrect**: The agent's answer is wrong, contradicts the expected answer, or misses the main point

## Guidelines:
- Focus on factual accuracy, not exact wording
- The agent may provide MORE detail than the expected answer - that's fine if the core facts are correct
- District names can be referred to by name (e.g., "Chicago") or number (e.g., "Seventh District")
- Be fair but rigorous"""


def get_judge():
    llm = get_critique_llm()
    return llm.with_structured_output(JudgeResult)


async def evaluate_question(pipeline: HybridCoTGNNPipeline, judge, question: str, expected: str, q_id: int, category: str) -> dict:
    """Evaluate a single question."""
    agent_start = time.time()

    try:
        result = await pipeline.query_async(question)
        answer = result.answer.answer
        num_facts = len(result.evidence_pool.scored_facts)
        confidence = result.answer.confidence
        expansion = result.evidence_pool.expansion_performed
        decomp_time = result.answer.decomposition_time_ms
        retrieval_time = result.answer.retrieval_time_ms
        scoring_time = result.answer.scoring_time_ms
        expansion_time = result.answer.expansion_time_ms
        synthesis_time = result.answer.synthesis_time_ms
    except Exception as e:
        answer = f"Error: {e}"
        num_facts = 0
        confidence = 0
        expansion = False
        decomp_time = retrieval_time = scoring_time = expansion_time = synthesis_time = 0

    agent_time = time.time() - agent_start

    # Judge
    judge_start = time.time()
    try:
        judge_result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: judge.invoke([
                SystemMessage(content=JUDGE_PROMPT),
                HumanMessage(content=f"""## QUESTION
{question}

## EXPECTED ANSWER
{expected}

## AGENT'S ANSWER
{answer}

Evaluate whether the agent's answer is correct.""")
            ])
        )
    except Exception as e:
        judge_result = JudgeResult(
            verdict=JudgeVerdict.INCORRECT,
            reasoning=f"Judge error: {e}"
        )
    judge_time = time.time() - judge_start

    return {
        "question_id": q_id,
        "category": category,
        "question": question,
        "expected": expected,
        "answer": answer,
        "verdict": judge_result.verdict.value,
        "reasoning": judge_result.reasoning,
        "num_facts": num_facts,
        "confidence": confidence,
        "expansion_performed": expansion,
        "agent_time_sec": agent_time,
        "decomp_time_ms": decomp_time,
        "retrieval_time_ms": retrieval_time,
        "scoring_time_ms": scoring_time,
        "expansion_time_ms": expansion_time,
        "synthesis_time_ms": synthesis_time,
        "judge_time_sec": judge_time
    }


async def run_evaluation(qa_pairs: list, max_concurrent: int = 5):
    """Run evaluation with concurrency."""
    total = len(qa_pairs)

    print(f"\n{'='*70}", flush=True)
    print(f"HYBRID COT-GNN V1 EVALUATION", flush=True)
    print(f"Questions: {total} (concurrency: {max_concurrent})", flush=True)
    print(f"{'='*70}\n", flush=True)

    pipeline = HybridCoTGNNPipeline(group_id="default")
    judge = get_judge()

    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * total
    completed = [0]

    async def eval_with_semaphore(i, pair):
        async with semaphore:
            q_id = pair.get("id", i + 1)
            question = pair["question"]
            expected = pair["answer"]
            category = pair.get("category", "Unknown")

            result = await evaluate_question(pipeline, judge, question, expected, q_id, category)
            results[i] = result

            completed[0] += 1
            verdict = result["verdict"]
            icon = "✓" if verdict == "correct" else "~" if verdict == "partially_correct" else "✗"
            print(f"[{completed[0]}/{total}] Q{q_id}: {icon} ({result['agent_time_sec']:.1f}s, {result['num_facts']} facts) - {category}", flush=True)

            return result

    await asyncio.gather(*[eval_with_semaphore(i, pair) for i, pair in enumerate(qa_pairs)])

    return results


def print_summary(results: list):
    """Print summary."""
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    partial = sum(1 for r in results if r["verdict"] == "partially_correct")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")

    avg_time = sum(r["agent_time_sec"] for r in results) / total
    avg_facts = sum(r["num_facts"] for r in results) / total
    avg_confidence = sum(r["confidence"] for r in results) / total

    print(f"\n{'='*70}")
    print("HYBRID COT-GNN V1 RESULTS")
    print(f"{'='*70}")
    print(f"\n  Total Questions: {total}")
    print(f"\n  ✓ Correct:          {correct:3d} ({correct/total*100:5.1f}%)")
    print(f"  ~ Partially Correct: {partial:3d} ({partial/total*100:5.1f}%)")
    print(f"  ✗ Incorrect:        {incorrect:3d} ({incorrect/total*100:5.1f}%)")
    print(f"\n  Strict Accuracy:  {correct/total*100:.1f}%")
    print(f"  Lenient Accuracy: {(correct+partial)/total*100:.1f}%")
    print(f"\n  Avg Time: {avg_time:.1f}s")
    print(f"  Avg Facts: {avg_facts:.1f}")
    print(f"  Avg Confidence: {avg_confidence:.2f}")

    # By category
    print(f"\n  By Category:")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "partial": 0, "incorrect": 0}
        if r["verdict"] == "correct":
            categories[cat]["correct"] += 1
        elif r["verdict"] == "partially_correct":
            categories[cat]["partial"] += 1
        else:
            categories[cat]["incorrect"] += 1

    for cat in sorted(categories.keys()):
        counts = categories[cat]
        total_cat = sum(counts.values())
        correct_cat = counts["correct"]
        pct = correct_cat / total_cat * 100
        print(f"    {cat}: {correct_cat}/{total_cat} ({pct:.0f}%)")

    print(f"{'='*70}")


def save_results(results: list, output_path: str):
    """Save results to JSON."""
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    partial = sum(1 for r in results if r["verdict"] == "partially_correct")

    data = {
        "timestamp": datetime.now().isoformat(),
        "system": "hybrid_cot_gnn_v1",
        "total_questions": total,
        "summary": {
            "correct": correct,
            "partially_correct": partial,
            "incorrect": total - correct - partial,
            "strict_accuracy": correct / total,
            "lenient_accuracy": (correct + partial) / total,
            "avg_time_sec": sum(r["agent_time_sec"] for r in results) / total,
            "avg_facts": sum(r["num_facts"] for r in results) / total,
            "avg_confidence": sum(r["confidence"] for r in results) / total,
        },
        "results": results
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file", default="Biege_OA.json")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent questions")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    qa_file = args.qa_file if os.path.isabs(args.qa_file) else os.path.join(project_root, args.qa_file)

    with open(qa_file) as f:
        qa_data = json.load(f)

    qa_pairs = qa_data.get("qa_pairs", qa_data)

    # Apply start and limit
    qa_pairs = qa_pairs[args.start:]
    if args.limit:
        qa_pairs = qa_pairs[:args.limit]

    results = await run_evaluation(qa_pairs, max_concurrent=args.concurrency)
    print_summary(results)

    output = args.output or f"eval_hybrid_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, output)


if __name__ == "__main__":
    asyncio.run(main())
