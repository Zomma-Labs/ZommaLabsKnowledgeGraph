"""
Evaluate V7 Knowledge Graph Pipeline with LLM-as-a-Judge
=========================================================

Usage:
    uv run scripts/eval_v7_pipeline.py                      # Run all 75 questions
    uv run scripts/eval_v7_pipeline.py --limit 10           # Limit to first 10
    uv run scripts/eval_v7_pipeline.py --concurrency 5      # Adjust parallelism
    uv run scripts/eval_v7_pipeline.py --qa-file Alphabet_QA.json  # Different QA file
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.querying_system.v7 import query_v7, V7Config
from src.util.llm_client import get_critique_llm


class JudgeVerdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    ABSTAIN = "abstain"  # Agent properly refused to answer


class JudgeResult(BaseModel):
    verdict: JudgeVerdict
    reasoning: str


JUDGE_PROMPT = """You are an impartial judge evaluating whether an AI agent's answer correctly addresses a question.

## Verdict Options:
- **correct**: The agent's answer contains ALL key facts from the expected answer. Additional detail, context, or comprehensiveness is NEVER penalized.
- **partially_correct**: The agent's answer contains SOME but not all key facts, OR is missing specific items from an enumeration (e.g., lists 4 of 5 districts).
- **incorrect**: The agent's answer is factually WRONG or directly CONTRADICTS the expected answer on a key point.
- **abstain**: The agent explicitly refused to answer (e.g., "Sorry, I could not retrieve the information").

## Critical Guidelines:

**COMPREHENSIVENESS IS GOOD, NOT BAD:**
- If the agent provides the expected answer PLUS additional accurate details = CORRECT
- If the agent answers a general question with information from multiple districts/sources = CORRECT
- Adding contextual information (dates, years, explanations) is HELPFUL, not an error
- Example: Expected "tax credit expired in September", Agent says "September 2025" = CORRECT (adding year is clarification)

**For summary vs detailed answers:**
- If expected says "X was unchanged" and agent explains "X showed mixed signals with some up, some down, netting to little change" = CORRECT (same conclusion, more detail)
- A nuanced answer that explains WHY something happened is BETTER than a summary

**For enumeration questions (lists of districts, items, etc.):**
- Getting ALL items = correct
- Getting MOST items but missing 1-2 = partially_correct (NOT incorrect)
- Getting NONE or completely wrong items = incorrect
- Adding EXTRA valid items beyond the expected list = correct (comprehensiveness)

**INCORRECT is reserved for genuine factual errors:**
- Only mark "incorrect" if the answer CONTRADICTS the expected answer
- "Unchanged" vs "softening/declining" = INCORRECT (contradiction)
- "Mixed conditions" vs "uniformly negative" = INCORRECT (contradiction)
- Missing some details but getting the main point right = PARTIALLY_CORRECT, not incorrect
- Providing more comprehensive coverage than expected = CORRECT, not incorrect"""


def get_judge():
    llm = get_critique_llm()
    return llm.with_structured_output(JudgeResult)


async def evaluate_question(judge, question: str, expected: str, q_id: int, config: V7Config) -> dict:
    """Evaluate a single question."""
    agent_start = time.time()
    result = None  # Initialize for chunk extraction later

    try:
        result = await query_v7(question, config=config)
        answer = result.answer
        confidence = result.confidence
        num_sub_answers = len(result.sub_answers)
        decomposition_time = result.decomposition_time_ms
        resolution_time = result.resolution_time_ms
        retrieval_time = result.retrieval_time_ms
        synthesis_time = result.synthesis_time_ms
    except Exception as e:
        answer = f"Error: {e}"
        confidence = 0.0
        num_sub_answers = 0
        decomposition_time = 0
        resolution_time = 0
        retrieval_time = 0
        synthesis_time = 0

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

    # Extract chunk info for debugging (if available)
    chunks_summary = []
    if result is not None and hasattr(result, 'sub_answers'):
        for sa in result.sub_answers:
            if hasattr(sa, 'context') and sa.context:
                ctx = sa.context
                sa_chunks = {
                    "sub_query": sa.sub_query,
                    "high_relevance": [
                        {"chunk_id": c.chunk_id, "header": c.header_path, "score": c.vector_score}
                        for c in (ctx.high_relevance_chunks or [])[:5]  # Top 5 for brevity
                    ],
                    "topic_chunks": [
                        {"chunk_id": c.chunk_id, "header": c.header_path}
                        for c in (ctx.topic_chunks or [])[:3]
                    ],
                    "entities_found": sa.entities_found or [],
                }
                chunks_summary.append(sa_chunks)

    return {
        "question_id": q_id,
        "question": question,
        "expected": expected,
        "answer": answer,
        "verdict": judge_result.verdict.value,
        "reasoning": judge_result.reasoning,
        "confidence": confidence,
        "num_sub_answers": num_sub_answers,
        "chunks_summary": chunks_summary,  # For debugging retrieval
        "agent_time_sec": agent_time,
        "decomposition_time_ms": decomposition_time,
        "resolution_time_ms": resolution_time,
        "retrieval_time_ms": retrieval_time,
        "synthesis_time_ms": synthesis_time,
        "judge_time_sec": judge_time
    }


async def run_evaluation(qa_pairs: list, config: V7Config, max_concurrent: int = 5):
    """Run evaluation with concurrency."""
    total = len(qa_pairs)

    print(f"\n{'='*70}")
    print(f"V7 KNOWLEDGE GRAPH PIPELINE EVALUATION")
    print(f"Questions: {total} (concurrency: {max_concurrent})")
    print(f"Config: entity_threshold={config.entity_threshold}, 1hop={config.enable_1hop_expansion}, global={config.enable_global_search}")
    print(f"{'='*70}\n")

    judge = get_judge()

    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * total
    completed = [0]

    async def eval_with_semaphore(i, pair):
        async with semaphore:
            q_id = pair.get("id", i + 1)
            question = pair["question"]
            expected = pair["answer"]

            result = await evaluate_question(judge, question, expected, q_id, config)
            results[i] = result

            completed[0] += 1
            verdict = result["verdict"]
            icon = {"correct": "+", "partially_correct": "~", "incorrect": "-", "abstain": "?"}.get(verdict, "?")
            print(f"[{completed[0]}/{total}] Q{q_id}: {icon} ({result['agent_time_sec']:.1f}s, conf={result['confidence']:.2f})")

            return result

    await asyncio.gather(*[eval_with_semaphore(i, pair) for i, pair in enumerate(qa_pairs)])

    return results


def print_summary(results: list):
    """Print summary."""
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    partial = sum(1 for r in results if r["verdict"] == "partially_correct")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")
    abstain = sum(1 for r in results if r["verdict"] == "abstain")

    avg_time = sum(r["agent_time_sec"] for r in results) / total
    avg_confidence = sum(r["confidence"] for r in results) / total

    print(f"\n{'='*70}")
    print("V7 KNOWLEDGE GRAPH PIPELINE RESULTS")
    print(f"{'='*70}")
    print(f"\n  Total Questions: {total}")
    print(f"\n  + Correct:           {correct:3d} ({correct/total*100:5.1f}%)")
    print(f"  ~ Partially Correct: {partial:3d} ({partial/total*100:5.1f}%)")
    print(f"  - Incorrect:         {incorrect:3d} ({incorrect/total*100:5.1f}%)")
    print(f"  ? Abstain:           {abstain:3d} ({abstain/total*100:5.1f}%)")
    print(f"\n  Strict Accuracy:  {correct/total*100:.1f}%")
    print(f"  Lenient Accuracy: {(correct+partial)/total*100:.1f}%")
    print(f"\n  Avg Time: {avg_time:.1f}s")
    print(f"  Avg Confidence: {avg_confidence:.2f}")
    print(f"{'='*70}")


def save_results(results: list, output_path: str):
    """Save results to JSON."""
    total = len(results)
    data = {
        "timestamp": datetime.now().isoformat(),
        "system": "v7_knowledge_graph",
        "total_questions": total,
        "summary": {
            "correct": sum(1 for r in results if r["verdict"] == "correct"),
            "partially_correct": sum(1 for r in results if r["verdict"] == "partially_correct"),
            "incorrect": sum(1 for r in results if r["verdict"] == "incorrect"),
            "abstain": sum(1 for r in results if r["verdict"] == "abstain"),
            "strict_accuracy": sum(1 for r in results if r["verdict"] == "correct") / total,
            "lenient_accuracy": sum(1 for r in results if r["verdict"] in ["correct", "partially_correct"]) / total,
            "avg_time_sec": sum(r["agent_time_sec"] for r in results) / total,
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
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent questions")
    parser.add_argument("--entity-threshold", type=float, default=0.3, help="Entity resolution threshold")
    parser.add_argument("--no-1hop", action="store_true", help="Disable 1-hop expansion")
    parser.add_argument("--no-global", action="store_true", help="Disable global search")
    args = parser.parse_args()

    # Build config
    config = V7Config(
        entity_threshold=args.entity_threshold,
        topic_threshold=args.entity_threshold,
        enable_1hop_expansion=not args.no_1hop,
        enable_global_search=not args.no_global,
    )

    # Load QA file
    project_root = Path(__file__).parent.parent
    qa_file = args.qa_file if os.path.isabs(args.qa_file) else project_root / args.qa_file

    with open(qa_file) as f:
        qa_data = json.load(f)

    qa_pairs = qa_data.get("qa_pairs", qa_data)

    if args.limit:
        qa_pairs = qa_pairs[:args.limit]

    results = await run_evaluation(qa_pairs, config, max_concurrent=args.concurrency)
    print_summary(results)

    output = args.output or f"eval/eval_v7_knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, output)


if __name__ == "__main__":
    asyncio.run(main())
