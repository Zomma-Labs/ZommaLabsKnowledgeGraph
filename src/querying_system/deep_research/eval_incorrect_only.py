"""
Evaluate Deep Research on Previously Incorrect Questions
=========================================================

Runs only the questions that were marked incorrect in a previous eval
through the deep research pipeline to see if it does better.

Usage:
    uv run python -m src.querying_system.deep_research.eval_incorrect_only [--prev-eval FILE]
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.querying_system.deep_research import DeepResearchPipeline


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
    prev_verdict: str = ""
    num_findings: int = 0
    agent_time_sec: float = 0.0
    research_time_ms: int = 0
    synthesis_time_ms: int = 0
    judge_time_sec: float = 0.0


JUDGE_PROMPT = """You are an impartial judge evaluating whether an AI agent's answer correctly addresses a question.

## Verdict Options:
- **correct**: The agent's answer contains all key facts from the expected answer
- **partially_correct**: The agent's answer contains some but not all key facts
- **incorrect**: The agent's answer is wrong or contradicts the expected answer
- **unanswerable**: The agent correctly indicated the information was not found

Be fair but rigorous. Focus on factual accuracy, not exact wording."""


def get_judge():
    """Create the LLM judge."""
    llm = get_critique_llm()
    return llm.with_structured_output(JudgeResult)


def load_incorrect_questions(prev_eval_file: str, qa_file: str) -> list[dict]:
    """Load questions that were incorrect in previous eval."""

    # Load previous eval results
    with open(prev_eval_file) as f:
        prev_results = json.load(f)

    # Find incorrect question IDs
    incorrect_ids = set()
    for r in prev_results.get("results", []):
        if r.get("verdict", "").lower() == "incorrect":
            incorrect_ids.add(r["question_id"])

    print(f"Found {len(incorrect_ids)} incorrect questions in previous eval: {sorted(incorrect_ids)}")

    # Load QA pairs
    with open(qa_file) as f:
        qa_data = json.load(f)

    qa_pairs = qa_data.get("qa_pairs", qa_data)

    # Filter to only incorrect questions
    incorrect_pairs = []
    for pair in qa_pairs:
        if pair.get("id") in incorrect_ids:
            incorrect_pairs.append(pair)

    return incorrect_pairs, prev_results


async def evaluate_question(pipeline: DeepResearchPipeline, judge, question: str, expected: str, q_id: int, prev_verdict: str) -> EvalResult:
    """Evaluate a single question."""
    agent_start = time.time()
    num_findings = 0
    research_time = 0
    synthesis_time = 0

    try:
        result = await pipeline.query_async(question, verbose=False)
        agent_answer = result.answer
        num_findings = len(result.findings)
        research_time = result.research_time_ms
        synthesis_time = result.synthesis_time_ms
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
        prev_verdict=prev_verdict,
        num_findings=num_findings,
        agent_time_sec=agent_time,
        research_time_ms=research_time,
        synthesis_time_ms=synthesis_time,
        judge_time_sec=judge_time,
    )


async def run_evaluation(qa_pairs: list[dict], prev_results: dict):
    """Run evaluation on incorrect questions."""
    total = len(qa_pairs)

    # Get previous verdicts for comparison
    prev_verdicts = {}
    for r in prev_results.get("results", []):
        prev_verdicts[r["question_id"]] = r.get("verdict", "unknown")

    print(f"\n{'='*70}")
    print(f"DEEP RESEARCH - RETRY INCORRECT QUESTIONS")
    print(f"Questions to retry: {total}")
    print(f"{'='*70}\n")

    pipeline = DeepResearchPipeline(user_id="default")
    judge = get_judge()
    results = []

    for i, pair in enumerate(qa_pairs):
        q_id = pair.get("id", i + 1)
        question = pair["question"]
        expected = pair["answer"]
        prev_verdict = prev_verdicts.get(q_id, "unknown")

        print(f"[{i+1}/{total}] Q{q_id}: {question[:50]}...", end=" ", flush=True)

        result = await evaluate_question(pipeline, judge, question, expected, q_id, prev_verdict)
        results.append(result)

        # Show improvement indicator
        new_verdict = result.verdict.value
        if new_verdict == "correct":
            icon = "✅ FIXED"
        elif new_verdict == "partially_correct":
            icon = "⚠️  PARTIAL"
        else:
            icon = "❌ STILL WRONG"

        print(f"[{icon}] ({result.agent_time_sec:.1f}s, {result.num_findings} findings)")

        if result.verdict != JudgeVerdict.CORRECT:
            print(f"      Reason: {result.reasoning[:100]}...")

    return results


def print_summary(results: list[EvalResult]):
    """Print evaluation summary."""
    total = len(results)
    correct = sum(1 for r in results if r.verdict == JudgeVerdict.CORRECT)
    partial = sum(1 for r in results if r.verdict == JudgeVerdict.PARTIALLY_CORRECT)
    incorrect = sum(1 for r in results if r.verdict == JudgeVerdict.INCORRECT)

    avg_agent_time = sum(r.agent_time_sec for r in results) / total if total > 0 else 0
    avg_findings = sum(r.num_findings for r in results) / total if total > 0 else 0

    print(f"\n{'='*70}")
    print("SUMMARY: DEEP RESEARCH vs TWO-AGENT on INCORRECT QUESTIONS")
    print(f"{'='*70}")
    print(f"\nQuestions Retried: {total}")
    print(f"\nDeep Research Results:")
    print(f"  ✅ Now Correct:          {correct:3d} ({correct/total*100:5.1f}%)")
    print(f"  ⚠️  Partially Correct:   {partial:3d} ({partial/total*100:5.1f}%)")
    print(f"  ❌ Still Incorrect:      {incorrect:3d} ({incorrect/total*100:5.1f}%)")
    print(f"\nImprovement Rate: {(correct+partial)/total*100:.1f}% (were all incorrect)")
    print(f"\nTiming:")
    print(f"  Avg Agent Time: {avg_agent_time:.1f}s")
    print(f"  Avg Findings:   {avg_findings:.1f}")
    print(f"{'='*70}")

    # List which questions improved
    print(f"\n{'='*70}")
    print("QUESTION-BY-QUESTION COMPARISON")
    print(f"{'='*70}")
    for r in results:
        if r.verdict == JudgeVerdict.CORRECT:
            print(f"  Q{r.question_id}: incorrect → ✅ CORRECT")
        elif r.verdict == JudgeVerdict.PARTIALLY_CORRECT:
            print(f"  Q{r.question_id}: incorrect → ⚠️  PARTIAL")
        else:
            print(f"  Q{r.question_id}: incorrect → ❌ STILL INCORRECT")


def save_results(results: list[EvalResult], output_path: str):
    """Save results to JSON."""
    total = len(results)
    data = {
        "timestamp": datetime.now().isoformat(),
        "system": "deep_research_retry_incorrect",
        "total_questions": total,
        "summary": {
            "correct": sum(1 for r in results if r.verdict == JudgeVerdict.CORRECT),
            "partially_correct": sum(1 for r in results if r.verdict == JudgeVerdict.PARTIALLY_CORRECT),
            "incorrect": sum(1 for r in results if r.verdict == JudgeVerdict.INCORRECT),
            "avg_agent_time_sec": sum(r.agent_time_sec for r in results) / total if total > 0 else 0,
            "avg_findings": sum(r.num_findings for r in results) / total if total > 0 else 0,
        },
        "results": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "expected": r.expected_answer,
                "answer": r.agent_answer,
                "prev_verdict": r.prev_verdict,
                "new_verdict": r.verdict.value,
                "improved": r.verdict.value in ["correct", "partially_correct"],
                "reasoning": r.reasoning,
                "num_findings": r.num_findings,
                "agent_time": r.agent_time_sec,
                "research_time_ms": r.research_time_ms,
                "synthesis_time_ms": r.synthesis_time_ms,
            }
            for r in results
        ]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate Deep Research on Incorrect Questions")
    parser.add_argument("--prev-eval", default="eval_two_agent_20260107_000831.json",
                        help="Previous eval JSON with incorrect questions")
    parser.add_argument("--qa-file", default="Biege_OA.json", help="Q&A JSON file")
    parser.add_argument("--output", "-o", help="Output JSON file")
    args = parser.parse_args()

    # Resolve paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    prev_eval_file = args.prev_eval
    if not os.path.isabs(prev_eval_file):
        prev_eval_file = os.path.join(project_root, prev_eval_file)

    qa_file = args.qa_file
    if not os.path.isabs(qa_file):
        qa_file = os.path.join(project_root, qa_file)

    # Load incorrect questions
    qa_pairs, prev_results = load_incorrect_questions(prev_eval_file, qa_file)

    if not qa_pairs:
        print("No incorrect questions found!")
        return

    # Run evaluation
    results = await run_evaluation(qa_pairs, prev_results)

    # Print summary
    print_summary(results)

    # Save results
    output = args.output or f"eval_deep_retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, output)


if __name__ == "__main__":
    asyncio.run(main())
