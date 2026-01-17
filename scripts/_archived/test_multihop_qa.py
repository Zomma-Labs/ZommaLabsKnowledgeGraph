"""Test the multi-hop questions that were failing."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from src.querying_system.deep_research import DeepResearchPipeline
from src.util.llm_client import get_critique_llm
from pydantic import BaseModel, Field
from enum import Enum


class JudgeVerdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


class JudgeResult(BaseModel):
    verdict: JudgeVerdict
    reasoning: str


JUDGE_PROMPT = """You are an impartial judge evaluating whether an AI agent's answer correctly addresses a question.

## Verdict Options:
- **correct**: The agent's answer contains all key facts from the expected answer
- **partially_correct**: The agent's answer contains some but not all key facts
- **incorrect**: The agent's answer is wrong or contradicts the expected answer

Be fair but rigorous. Focus on factual accuracy, not exact wording."""


# Multi-hop questions that were failing
MULTIHOP_QUESTIONS = [
    {
        "id": 51,
        "question": "In districts where economic activity declined, what happened to manufacturing?",
        "expected": "In the four districts where economic activity declined (New York, Minneapolis, Kansas City, San Francisco): New York saw manufacturing hold steady after a summer uptick; Minneapolis reported manufacturing was flat; Kansas City saw manufacturing output growth moderate; and San Francisco reported manufacturing activity remained stable. Overall, manufacturing in declining districts was flat to slightly moderating."
    },
    {
        "id": 52,
        "question": "Which districts mentioned both tariff impacts on manufacturing AND labor shortages from immigration policies?",
        "expected": "The St. Louis District reported both: contacts mentioned that immigration policies resulted in labor shortages, and the national summary noted tariff-induced input cost increases affected manufacturing across many districts including St. Louis. The Chicago District also mentioned both tariff impacts on manufacturing (with manufacturers noting changing tariff policies made planning difficult) and labor supply strains in manufacturing due to immigration policy changes."
    }
]


async def main():
    print("=" * 70)
    print("TESTING MULTI-HOP QUESTIONS WITH COT BRIEF")
    print("=" * 70)

    pipeline = DeepResearchPipeline(user_id="default")
    judge_llm = get_critique_llm().with_structured_output(JudgeResult)

    for qa in MULTIHOP_QUESTIONS:
        print(f"\n[Q{qa['id']}] {qa['question'][:60]}...")
        print("-" * 60)

        # Run the query
        result = await pipeline.query_async(qa["question"], verbose=True)

        print(f"\n>>> KG Answer:\n{result.answer[:500]}...")
        print(f"\n>>> Expected:\n{qa['expected'][:300]}...")

        # Judge
        judge_result = judge_llm.invoke([
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=f"""## QUESTION
{qa['question']}

## EXPECTED ANSWER
{qa['expected']}

## AGENT'S ANSWER
{result.answer}

Evaluate whether the agent's answer is correct.""")
        ])

        icon = {"correct": "✅", "partially_correct": "⚠️", "incorrect": "❌"}[judge_result.verdict.value]
        print(f"\n>>> Verdict: {icon} {judge_result.verdict.value.upper()}")
        print(f">>> Reasoning: {judge_result.reasoning[:200]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
