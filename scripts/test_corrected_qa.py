"""
Test the corrected QA questions to verify KG answers.

Usage:
    uv run python scripts/test_corrected_qa.py
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.querying_system.deep_research import DeepResearchPipeline
from src.util.llm_client import get_critique_llm


# Questions we fixed
FIXED_QUESTION_IDS = [8, 35, 36, 68, 70, 72, 75]


class JudgeVerdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


class JudgeResult(BaseModel):
    verdict: JudgeVerdict
    reasoning: str
    key_facts_matched: list[str] = Field(default_factory=list)
    key_facts_missing: list[str] = Field(default_factory=list)


JUDGE_PROMPT = """You are an impartial judge evaluating whether an AI agent's answer correctly addresses a question.

## Verdict Options:
- **correct**: The agent's answer contains all key facts from the expected answer
- **partially_correct**: The agent's answer contains some but not all key facts
- **incorrect**: The agent's answer is wrong or contradicts the expected answer

Be fair but rigorous. Focus on factual accuracy, not exact wording."""


def get_judge():
    llm = get_critique_llm()
    return llm.with_structured_output(JudgeResult)


@dataclass
class Result:
    q_id: int
    question: str
    expected: str
    got: str
    verdict: str
    reasoning: str
    findings: int


async def evaluate_question(pipeline, judge, q_id: int, question: str, expected: str) -> Result:
    """Evaluate a single question."""
    print(f"  Querying KG...", end=" ", flush=True)

    try:
        result = await pipeline.query_async(question, verbose=False)
        answer = result.answer
        findings = len(result.findings)
    except Exception as e:
        answer = f"Error: {e}"
        findings = 0

    print(f"({findings} findings)", end=" ", flush=True)

    # Judge
    print("Judging...", end=" ", flush=True)
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
        verdict = judge_result.verdict.value
        reasoning = judge_result.reasoning
    except Exception as e:
        verdict = "error"
        reasoning = str(e)

    return Result(
        q_id=q_id,
        question=question,
        expected=expected,
        got=answer,
        verdict=verdict,
        reasoning=reasoning,
        findings=findings
    )


async def main():
    # Load QA pairs
    with open("Biege_OA.json") as f:
        data = json.load(f)

    qa_pairs = data["qa_pairs"]
    fixed_pairs = [p for p in qa_pairs if p["id"] in FIXED_QUESTION_IDS]

    print("=" * 70)
    print("TESTING CORRECTED QA QUESTIONS")
    print(f"Testing {len(fixed_pairs)} questions that were corrected")
    print("=" * 70)

    pipeline = DeepResearchPipeline(user_id="default")
    judge = get_judge()
    results = []

    for pair in fixed_pairs:
        q_id = pair["id"]
        question = pair["question"]
        expected = pair["answer"]

        print(f"\n[Q{q_id}] {question[:60]}...")

        result = await evaluate_question(pipeline, judge, q_id, question, expected)
        results.append(result)

        icon = {"correct": "✅", "partially_correct": "⚠️", "incorrect": "❌"}.get(result.verdict, "?")
        print(f"{icon} {result.verdict.upper()}")

        if result.verdict != "correct":
            print(f"    Expected: {expected[:100]}...")
            print(f"    Got: {result.got[:100]}...")
            print(f"    Reasoning: {result.reasoning[:150]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    correct = sum(1 for r in results if r.verdict == "correct")
    partial = sum(1 for r in results if r.verdict == "partially_correct")
    incorrect = sum(1 for r in results if r.verdict == "incorrect")

    print(f"\nTotal: {len(results)}")
    print(f"  ✅ Correct:           {correct}")
    print(f"  ⚠️  Partially Correct: {partial}")
    print(f"  ❌ Incorrect:         {incorrect}")

    # Previously these were all "incorrect" - now show improvement
    print(f"\nPrevious: All 7 marked incorrect (0% accuracy)")
    print(f"After correction: {correct + partial}/{len(results)} at least partially correct ({(correct+partial)/len(results)*100:.0f}% lenient accuracy)")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "questions_tested": FIXED_QUESTION_IDS,
        "results": [
            {
                "id": r.q_id,
                "question": r.question,
                "expected": r.expected,
                "got": r.got,
                "verdict": r.verdict,
                "reasoning": r.reasoning,
                "findings": r.findings
            }
            for r in results
        ],
        "summary": {
            "correct": correct,
            "partially_correct": partial,
            "incorrect": incorrect
        }
    }

    output_file = f"eval_corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
