"""
Evaluate Deep Research KG-RAG Pipeline
======================================

Runs questions through the deep research pipeline with an LLM judge.

Usage:
    uv run python -m src.querying_system.deep_research.eval_deep [--limit N]
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


# === CONFIGURATION ===
DEFAULT_QA_FILE = "Biege_OA.json"


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
    num_findings: int = 0
    agent_time_sec: float = 0.0
    research_time_ms: int = 0
    synthesis_time_ms: int = 0
    judge_time_sec: float = 0.0
    key_facts_matched: list[str] = field(default_factory=list)
    key_facts_missing: list[str] = field(default_factory=list)


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


async def evaluate_question(pipeline: DeepResearchPipeline, judge, question: str, expected: str, q_id: int) -> EvalResult:
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
        num_findings=num_findings,
        agent_time_sec=agent_time,
        research_time_ms=research_time,
        synthesis_time_ms=synthesis_time,
        judge_time_sec=judge_time,
        key_facts_matched=judge_result.key_facts_matched,
        key_facts_missing=judge_result.key_facts_missing,
    )


async def run_evaluation(qa_pairs: list[dict], limit: Optional[int] = None):
    """Run evaluation on questions."""
    pairs = qa_pairs[:limit] if limit else qa_pairs
    total = len(pairs)

    print(f"\n{'='*70}")
    print(f"DEEP RESEARCH PIPELINE EVALUATION")
    print(f"Questions: {total}")
    print(f"{'='*70}\n")

    # Initialize pipeline
    pipeline = DeepResearchPipeline(user_id="default")
    judge = get_judge()
    results = []

    # Process sequentially
    for i, pair in enumerate(pairs):
        q_id = pair.get("id", i + 1)
        question = pair["question"]
        expected = pair["answer"]

        print(f"[{i+1}/{total}] Q{q_id}: {question[:50]}...", end=" ", flush=True)

        result = await evaluate_question(pipeline, judge, question, expected, q_id)
        results.append(result)

        icon = {"correct": "PASS", "partially_correct": "PART", "incorrect": "FAIL", "unanswerable": "N/A"}
        print(f"[{icon.get(result.verdict.value, '?')}] ({result.agent_time_sec:.1f}s, {result.num_findings} findings)")

        if result.verdict != JudgeVerdict.CORRECT:
            print(f"      Expected: {expected[:80]}...")
            print(f"      Got: {result.agent_answer[:100]}...")

    return results


def print_summary(results: list[EvalResult]):
    """Print evaluation summary."""
    total = len(results)
    correct = sum(1 for r in results if r.verdict == JudgeVerdict.CORRECT)
    partial = sum(1 for r in results if r.verdict == JudgeVerdict.PARTIALLY_CORRECT)
    incorrect = sum(1 for r in results if r.verdict == JudgeVerdict.INCORRECT)
    unanswerable = sum(1 for r in results if r.verdict == JudgeVerdict.UNANSWERABLE)

    avg_agent_time = sum(r.agent_time_sec for r in results) / total
    avg_research = sum(r.research_time_ms for r in results) / total
    avg_synthesis = sum(r.synthesis_time_ms for r in results) / total
    avg_findings = sum(r.num_findings for r in results) / total

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal Questions: {total}")
    print(f"\nVerdicts:")
    print(f"  Correct:           {correct:3d} ({correct/total*100:5.1f}%)")
    print(f"  Partially Correct: {partial:3d} ({partial/total*100:5.1f}%)")
    print(f"  Incorrect:         {incorrect:3d} ({incorrect/total*100:5.1f}%)")
    print(f"  Unanswerable:      {unanswerable:3d} ({unanswerable/total*100:5.1f}%)")
    print(f"\nAccuracy:")
    print(f"  Strict:  {correct/total*100:5.1f}%")
    print(f"  Lenient: {(correct+partial)/total*100:5.1f}%")
    print(f"\nTiming (averages):")
    print(f"  Total Agent:  {avg_agent_time:.1f}s")
    print(f"  - Research:   {avg_research/1000:.1f}s")
    print(f"  - Synthesis:  {avg_synthesis/1000:.1f}s")
    print(f"\nResearch Stats:")
    print(f"  Avg Findings: {avg_findings:.1f}")
    print(f"{'='*70}")


def save_results(results: list[EvalResult], output_path: str):
    """Save results to JSON."""
    total = len(results)
    data = {
        "timestamp": datetime.now().isoformat(),
        "system": "deep_research",
        "total_questions": total,
        "summary": {
            "correct": sum(1 for r in results if r.verdict == JudgeVerdict.CORRECT),
            "partially_correct": sum(1 for r in results if r.verdict == JudgeVerdict.PARTIALLY_CORRECT),
            "incorrect": sum(1 for r in results if r.verdict == JudgeVerdict.INCORRECT),
            "unanswerable": sum(1 for r in results if r.verdict == JudgeVerdict.UNANSWERABLE),
            "avg_agent_time_sec": sum(r.agent_time_sec for r in results) / total,
            "avg_findings": sum(r.num_findings for r in results) / total,
        },
        "results": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "expected": r.expected_answer,
                "answer": r.agent_answer,
                "verdict": r.verdict.value,
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
    parser = argparse.ArgumentParser(description="Evaluate Deep Research Pipeline")
    parser.add_argument("--qa-file", default=DEFAULT_QA_FILE, help="Q&A JSON file")
    parser.add_argument("--limit", "-l", type=int, help="Limit questions")
    parser.add_argument("--output", "-o", help="Output JSON file")
    args = parser.parse_args()

    # Load QA pairs
    qa_file = args.qa_file
    if not os.path.isabs(qa_file):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        qa_file = os.path.join(project_root, qa_file)

    with open(qa_file) as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", data)
    print(f"Loaded {len(qa_pairs)} Q&A pairs")

    # Run evaluation
    results = await run_evaluation(qa_pairs, args.limit)

    # Print summary
    print_summary(results)

    # Save results
    output = args.output or f"eval_deep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, output)


if __name__ == "__main__":
    asyncio.run(main())
