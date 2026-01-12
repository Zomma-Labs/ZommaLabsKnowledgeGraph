"""
Compare Single-Agent vs Two-Agent QA Systems
=============================================

Uses the existing Alphabet_QA.json and evaluation framework to compare
accuracy between the single-agent and two-agent systems.
"""

import os
import sys
import json
import asyncio
import argparse
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field


# === CONFIGURATION ===
DEFAULT_QA_FILE = "Alphabet_QA.json"
DEFAULT_LIMIT = 10  # Limit for faster testing


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
    chunks_retrieved: int = 0
    system_name: str = ""


JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating whether an AI agent's answer correctly addresses a question.

## Verdict Options:
- **correct**: The agent's answer contains all key facts from the expected answer
- **partially_correct**: The agent's answer contains some but not all key facts
- **incorrect**: The agent's answer is wrong or contradicts the expected answer
- **unanswerable**: The agent correctly indicated the information was not found

Be fair but rigorous. Focus on factual accuracy, not exact wording."""


def get_judge_llm():
    """Create the LLM judge."""
    from src.util.llm_client import get_llm
    llm = get_llm(temperature=0)
    return llm.with_structured_output(JudgeResult)


def count_chunks(messages) -> int:
    """Count chunks retrieved from messages."""
    import re
    count = 0
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            count += len(re.findall(r'CHUNK_id:', content))
    return count


async def test_single_agent(question: str) -> tuple[str, float, int, list]:
    """Test the single-agent system."""
    from src.querying_system.kg_agent import make_graph

    start = time.time()
    try:
        graph = await make_graph()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 25}
        )

        messages = result["messages"]
        chunks = count_chunks(messages)

        # Get final answer
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(b.get('text', '') for b in content if isinstance(b, dict))
                answer = content
                break

        return answer, time.time() - start, chunks, messages
    except Exception as e:
        return f"Error: {e}", time.time() - start, 0, []


async def test_two_agent(question: str) -> tuple[str, float, int, list]:
    """Test the two-agent system."""
    from src.querying_system.two_agent_system import TwoAgentQA

    start = time.time()
    try:
        qa = TwoAgentQA()
        answer = await qa.query(question, verbose=False)

        # Count chunks from citations in answer
        import re
        chunks = len(set(re.findall(r'CHUNK[_\s]*(?:id)?[:\s]*(\S+)', answer, re.IGNORECASE)))

        return answer, time.time() - start, chunks, []
    except Exception as e:
        return f"Error: {e}", time.time() - start, 0, []


async def judge_answer(judge_llm, question: str, expected: str, agent_answer: str) -> JudgeResult:
    """Have the LLM judge evaluate the answer."""
    prompt = f"""## QUESTION
{question}

## EXPECTED ANSWER
{expected}

## AGENT'S ANSWER
{agent_answer}

Evaluate whether the agent's answer is correct."""

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: judge_llm.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
        )
        return result
    except Exception as e:
        return JudgeResult(
            verdict=JudgeVerdict.INCORRECT,
            reasoning=f"Judge error: {e}",
            key_facts_matched=[],
            key_facts_missing=[]
        )


async def run_comparison(qa_pairs: list[dict], limit: int = 10, verbose: bool = False):
    """Run comparison between both systems."""

    pairs = qa_pairs[:limit]
    judge_llm = get_judge_llm()

    single_results = []
    two_agent_results = []

    print(f"\n{'='*80}")
    print(f"COMPARING SINGLE-AGENT vs TWO-AGENT SYSTEMS")
    print(f"Testing {len(pairs)} questions")
    print(f"{'='*80}\n")

    for i, qa in enumerate(pairs, 1):
        q_id = qa.get("id", i)
        question = qa["question"]
        expected = qa["answer"]

        print(f"\n[{i}/{len(pairs)}] Q{q_id}: {question[:60]}...")

        # Test single agent
        print("  Single-Agent: ", end="", flush=True)
        s_answer, s_time, s_chunks, _ = await test_single_agent(question)
        s_judge = await judge_answer(judge_llm, question, expected, s_answer)
        single_results.append(EvalResult(
            question_id=q_id,
            question=question,
            expected_answer=expected,
            agent_answer=s_answer,
            verdict=s_judge.verdict,
            reasoning=s_judge.reasoning,
            agent_time_sec=s_time,
            chunks_retrieved=s_chunks,
            system_name="single"
        ))
        verdict_icon = {"correct": "✓", "partially_correct": "~", "incorrect": "✗", "unanswerable": "?"}
        print(f"{verdict_icon.get(s_judge.verdict.value, '?')} ({s_time:.1f}s, {s_chunks} chunks)")

        # Test two-agent
        print("  Two-Agent:    ", end="", flush=True)
        t_answer, t_time, t_chunks, _ = await test_two_agent(question)
        t_judge = await judge_answer(judge_llm, question, expected, t_answer)
        two_agent_results.append(EvalResult(
            question_id=q_id,
            question=question,
            expected_answer=expected,
            agent_answer=t_answer,
            verdict=t_judge.verdict,
            reasoning=t_judge.reasoning,
            agent_time_sec=t_time,
            chunks_retrieved=t_chunks,
            system_name="two_agent"
        ))
        print(f"{verdict_icon.get(t_judge.verdict.value, '?')} ({t_time:.1f}s, {t_chunks} chunks)")

        if verbose or s_judge.verdict != t_judge.verdict:
            print(f"\n  Expected: {expected[:100]}...")
            print(f"  Single:   {s_answer[:100]}...")
            print(f"  Two:      {t_answer[:100]}...")

    # Summary
    print_comparison_summary(single_results, two_agent_results)

    return single_results, two_agent_results


def print_comparison_summary(single_results: list[EvalResult], two_agent_results: list[EvalResult]):
    """Print comparison summary."""
    total = len(single_results)

    def calc_stats(results):
        correct = sum(1 for r in results if r.verdict == JudgeVerdict.CORRECT)
        partial = sum(1 for r in results if r.verdict == JudgeVerdict.PARTIALLY_CORRECT)
        incorrect = sum(1 for r in results if r.verdict == JudgeVerdict.INCORRECT)
        avg_time = sum(r.agent_time_sec for r in results) / len(results)
        avg_chunks = sum(r.chunks_retrieved for r in results) / len(results)
        return correct, partial, incorrect, avg_time, avg_chunks

    s_correct, s_partial, s_incorrect, s_time, s_chunks = calc_stats(single_results)
    t_correct, t_partial, t_incorrect, t_time, t_chunks = calc_stats(two_agent_results)

    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Metric':<25} | {'Single-Agent':<15} | {'Two-Agent':<15} | {'Winner':<10}")
    print("-" * 70)

    # Accuracy (strict)
    s_acc = s_correct / total * 100
    t_acc = t_correct / total * 100
    winner = "Single" if s_acc > t_acc else ("Two" if t_acc > s_acc else "Tie")
    print(f"{'Correct':<25} | {s_correct}/{total} ({s_acc:.0f}%){'':<5} | {t_correct}/{total} ({t_acc:.0f}%){'':<5} | {winner}")

    # Accuracy (lenient)
    s_lenient = (s_correct + s_partial) / total * 100
    t_lenient = (t_correct + t_partial) / total * 100
    winner = "Single" if s_lenient > t_lenient else ("Two" if t_lenient > s_lenient else "Tie")
    print(f"{'Correct+Partial':<25} | {s_correct+s_partial}/{total} ({s_lenient:.0f}%){'':<3} | {t_correct+t_partial}/{total} ({t_lenient:.0f}%){'':<3} | {winner}")

    # Incorrect
    winner = "Single" if s_incorrect < t_incorrect else ("Two" if t_incorrect < s_incorrect else "Tie")
    print(f"{'Incorrect':<25} | {s_incorrect}/{total}{'':<11} | {t_incorrect}/{total}{'':<11} | {winner}")

    # Time
    winner = "Single" if s_time < t_time else ("Two" if t_time < s_time else "Tie")
    print(f"{'Avg Time (sec)':<25} | {s_time:<15.1f} | {t_time:<15.1f} | {winner}")

    # Chunks
    winner = "Single" if s_chunks > t_chunks else ("Two" if t_chunks > s_chunks else "Tie")
    print(f"{'Avg Chunks Retrieved':<25} | {s_chunks:<15.1f} | {t_chunks:<15.1f} | {winner}")

    print(f"\n{'='*80}")

    # Per-question comparison
    print("\nPer-Question Breakdown:")
    print(f"{'Q#':<5} | {'Single':<12} | {'Two-Agent':<12} | {'Better':<10}")
    print("-" * 45)

    single_wins = 0
    two_wins = 0
    ties = 0

    verdict_order = {JudgeVerdict.CORRECT: 3, JudgeVerdict.PARTIALLY_CORRECT: 2, JudgeVerdict.UNANSWERABLE: 1, JudgeVerdict.INCORRECT: 0}

    for s, t in zip(single_results, two_agent_results):
        s_score = verdict_order[s.verdict]
        t_score = verdict_order[t.verdict]

        if s_score > t_score:
            better = "Single"
            single_wins += 1
        elif t_score > s_score:
            better = "Two"
            two_wins += 1
        else:
            better = "Tie"
            ties += 1

        print(f"Q{s.question_id:<4} | {s.verdict.value:<12} | {t.verdict.value:<12} | {better}")

    print(f"\nSingle-Agent wins: {single_wins}, Two-Agent wins: {two_wins}, Ties: {ties}")


async def main():
    parser = argparse.ArgumentParser(description="Compare Single vs Two-Agent QA Systems")
    parser.add_argument("--qa-file", default=DEFAULT_QA_FILE, help="Path to Q&A JSON file")
    parser.add_argument("--limit", "-l", type=int, default=DEFAULT_LIMIT, help="Limit questions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--output", "-o", help="Save results to JSON")
    args = parser.parse_args()

    # Load QA file
    qa_file = args.qa_file
    if not os.path.isabs(qa_file):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        qa_file = os.path.join(project_root, qa_file)

    if not os.path.exists(qa_file):
        print(f"Error: Q&A file not found: {qa_file}")
        sys.exit(1)

    with open(qa_file) as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", data)
    print(f"Loaded {len(qa_pairs)} Q&A pairs from {qa_file}")

    # Run comparison
    single_results, two_results = await run_comparison(qa_pairs, args.limit, args.verbose)

    # Save results if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(single_results),
            "single_agent": [
                {
                    "question_id": r.question_id,
                    "verdict": r.verdict.value,
                    "time_sec": r.agent_time_sec,
                    "chunks": r.chunks_retrieved,
                    "answer": r.agent_answer[:500]
                }
                for r in single_results
            ],
            "two_agent": [
                {
                    "question_id": r.question_id,
                    "verdict": r.verdict.value,
                    "time_sec": r.agent_time_sec,
                    "chunks": r.chunks_retrieved,
                    "answer": r.agent_answer[:500]
                }
                for r in two_results
            ]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
