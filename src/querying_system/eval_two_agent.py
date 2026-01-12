"""
Evaluate Two-Agent QA System
============================

Runs all questions through the two-agent system with an LLM judge.

Usage:
    uv run python -m src.querying_system.eval_two_agent [--limit N] [--batch-size N]
"""

import os
import sys
import json
import asyncio
import argparse
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.util.llm_client import get_critique_llm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.querying_system.two_agent_system import TwoAgentQA


# === CONFIGURATION ===
DEFAULT_QA_FILE = "Alphabet_QA.json"
DEFAULT_BATCH_SIZE = 3  # Concurrent questions


class JudgeVerdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    UNANSWERABLE = "unanswerable"


class JudgeResult(BaseModel):
    verdict: JudgeVerdict
    reasoning: str
    key_facts_matched: list[str] = Field(default_factory=list)
    key_facts_missing: list[str] = Field(default_factory=list)


@dataclass
class EvalResult:
    question_id: int
    question: str
    expected_answer: str
    agent_answer: str
    verdict: JudgeVerdict
    reasoning: str
    agent_time_sec: float = 0.0
    judge_time_sec: float = 0.0
    key_facts_matched: list[str] = field(default_factory=list)
    key_facts_missing: list[str] = field(default_factory=list)
    search_trace: list[dict] = field(default_factory=list)


JUDGE_PROMPT = """You are an impartial judge evaluating whether an AI agent's answer correctly addresses a question.

## Verdict Options:
- **correct**: The agent's answer contains all key facts from the expected answer
- **partially_correct**: The agent's answer contains some but not all key facts
- **incorrect**: The agent's answer is wrong or contradicts the expected answer
- **unanswerable**: The agent correctly indicated the information was not found

Be fair but rigorous. Focus on factual accuracy, not exact wording."""


def get_judge():
    """Create the LLM judge using GPT-5.1."""
    llm = get_critique_llm()
    return llm.with_structured_output(JudgeResult)


async def evaluate_question(qa: TwoAgentQA, judge, question: str, expected: str, q_id: int) -> EvalResult:
    """Evaluate a single question."""
    # Get agent answer with trace
    agent_start = time.time()
    search_trace = []
    try:
        result = await qa.query(question, verbose=False, return_trace=True)
        agent_answer, search_trace = result
    except Exception as e:
        agent_answer = f"Error: {e}"
    agent_time = time.time() - agent_start

    # Judge the answer
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
{agent_answer}

Evaluate whether the agent's answer is correct.""")
            ])
        )
    except Exception as e:
        judge_result = JudgeResult(
            verdict=JudgeVerdict.INCORRECT,
            reasoning=f"Judge error: {e}"
        )
    judge_time = time.time() - judge_start

    return EvalResult(
        question_id=q_id,
        question=question,
        expected_answer=expected,
        agent_answer=agent_answer,
        verdict=judge_result.verdict,
        reasoning=judge_result.reasoning,
        agent_time_sec=agent_time,
        judge_time_sec=judge_time,
        key_facts_matched=judge_result.key_facts_matched,
        key_facts_missing=judge_result.key_facts_missing,
        search_trace=search_trace
    )


async def run_evaluation(qa_pairs: list[dict], limit: Optional[int] = None, batch_size: int = DEFAULT_BATCH_SIZE):
    """Run evaluation on all questions."""
    pairs = qa_pairs[:limit] if limit else qa_pairs
    total = len(pairs)

    print(f"\n{'='*70}")
    print(f"TWO-AGENT SYSTEM EVALUATION")
    print(f"Questions: {total}, Batch size: {batch_size}")
    print(f"{'='*70}\n")

    # Initialize QA system once
    qa = TwoAgentQA()
    await qa.initialize()

    judge = get_judge()
    results = []

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = pairs[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"[Batch {batch_num}/{total_batches}] Processing Q{batch_start+1}-Q{batch_end}...")

        # Run batch concurrently
        batch_start_time = time.time()
        tasks = [
            evaluate_question(qa, judge, p["question"], p["answer"], p.get("id", batch_start + i + 1))
            for i, p in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*tasks)
        batch_time = time.time() - batch_start_time

        # Print batch results
        for r in batch_results:
            results.append(r)
            icon = {"correct": "[PASS]", "partially_correct": "[PART]", "incorrect": "[FAIL]", "unanswerable": "[N/A]"}
            print(f"  Q{r.question_id}: {icon.get(r.verdict.value, '[?]')} ({r.agent_time_sec:.1f}s)")

            if r.verdict != JudgeVerdict.CORRECT:
                print(f"       Q: {r.question[:60]}...")
                print(f"       Expected: {r.expected_answer[:60]}...")
                print(f"       Got: {r.agent_answer[:100]}...")
                print(f"       Reason: {r.reasoning[:80]}...")
                # Show search trace for debugging
                if r.search_trace:
                    print(f"       --- Search Trace ({len(r.search_trace)} steps) ---")
                    for step in r.search_trace[:10]:  # Limit to first 10 steps
                        if step["type"] == "tool_call":
                            args_str = ", ".join(f"{k}={repr(v)[:30]}" for k, v in step["args"].items())
                            print(f"         → {step['tool']}({args_str})")
                        elif step["type"] == "tool_result":
                            print(f"         ← {step['tool']}: {step['summary'][:60]}")
                    if len(r.search_trace) > 10:
                        print(f"         ... ({len(r.search_trace) - 10} more steps)")

        print(f"  Batch time: {batch_time:.1f}s\n")

    return results


def print_summary(results: list[EvalResult]):
    """Print evaluation summary."""
    total = len(results)
    correct = sum(1 for r in results if r.verdict == JudgeVerdict.CORRECT)
    partial = sum(1 for r in results if r.verdict == JudgeVerdict.PARTIALLY_CORRECT)
    incorrect = sum(1 for r in results if r.verdict == JudgeVerdict.INCORRECT)

    avg_agent_time = sum(r.agent_time_sec for r in results) / total
    avg_judge_time = sum(r.judge_time_sec for r in results) / total

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal Questions: {total}")
    print(f"\nVerdicts:")
    print(f"  Correct:           {correct:3d} ({correct/total*100:5.1f}%)")
    print(f"  Partially Correct: {partial:3d} ({partial/total*100:5.1f}%)")
    print(f"  Incorrect:         {incorrect:3d} ({incorrect/total*100:5.1f}%)")
    print(f"\nAccuracy:")
    print(f"  Strict:  {correct/total*100:5.1f}%")
    print(f"  Lenient: {(correct+partial)/total*100:5.1f}%")
    print(f"\nTiming:")
    print(f"  Avg Agent: {avg_agent_time:.1f}s")
    print(f"  Avg Judge: {avg_judge_time:.1f}s")
    print(f"  Total:     {sum(r.agent_time_sec + r.judge_time_sec for r in results):.1f}s")
    print(f"{'='*70}")


def save_results(results: list[EvalResult], output_path: str):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "system": "two_agent",
        "total_questions": len(results),
        "summary": {
            "correct": sum(1 for r in results if r.verdict == JudgeVerdict.CORRECT),
            "partially_correct": sum(1 for r in results if r.verdict == JudgeVerdict.PARTIALLY_CORRECT),
            "incorrect": sum(1 for r in results if r.verdict == JudgeVerdict.INCORRECT),
        },
        "results": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "expected": r.expected_answer,
                "answer": r.agent_answer,
                "verdict": r.verdict.value,
                "reasoning": r.reasoning,
                "agent_time": r.agent_time_sec,
                "judge_time": r.judge_time_sec,
                "search_trace": r.search_trace,
            }
            for r in results
        ]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate Two-Agent QA System")
    parser.add_argument("--qa-file", default=DEFAULT_QA_FILE, help="Q&A JSON file")
    parser.add_argument("--limit", "-l", type=int, help="Limit questions")
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--output", "-o", help="Output JSON file")
    args = parser.parse_args()

    # Load QA pairs
    qa_file = args.qa_file
    if not os.path.isabs(qa_file):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        qa_file = os.path.join(project_root, qa_file)

    with open(qa_file) as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", data)
    print(f"Loaded {len(qa_pairs)} Q&A pairs")

    # Run evaluation
    results = await run_evaluation(qa_pairs, args.limit, args.batch_size)

    # Print summary
    print_summary(results)

    # Save results
    output = args.output or f"eval_two_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, output)


if __name__ == "__main__":
    asyncio.run(main())
