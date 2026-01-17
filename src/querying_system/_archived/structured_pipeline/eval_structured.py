"""
Evaluate Structured KG-RAG Pipeline
===================================

Runs all questions through the structured pipeline with an LLM judge.

Usage:
    uv run python -m src.querying_system.structured_pipeline.eval_structured [--limit N]
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

from src.querying_system.structured_pipeline import StructuredKGRAG


# === CONFIGURATION ===
DEFAULT_QA_FILE = "Biege_OA.json"
DEFAULT_BATCH_SIZE = 5  # Concurrent questions


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
    query_type: str = ""
    agent_time_sec: float = 0.0
    analysis_time_ms: int = 0
    retrieval_time_ms: int = 0
    generation_time_ms: int = 0
    judge_time_sec: float = 0.0
    chunks_retrieved: int = 0
    fallback_used: bool = False
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
    """Create the LLM judge using GPT-5.1."""
    llm = get_critique_llm()
    return llm.with_structured_output(JudgeResult)


def evaluate_question(pipeline: StructuredKGRAG, judge, question: str, expected: str, q_id: int) -> EvalResult:
    """Evaluate a single question."""
    # Get agent answer
    agent_start = time.time()
    query_type = ""
    chunks_retrieved = 0
    fallback_used = False
    analysis_time = 0
    retrieval_time = 0
    generation_time = 0

    try:
        result = pipeline.query(question, verbose=False)
        agent_answer = result.answer
        query_type = result.plan.query_type.value
        chunks_retrieved = len(result.retrieval.chunks)
        fallback_used = result.retrieval.fallback_used
        analysis_time = result.analysis_time_ms
        retrieval_time = result.retrieval_time_ms
        generation_time = result.generation_time_ms
    except Exception as e:
        agent_answer = f"Error: {e}"

    agent_time = time.time() - agent_start

    # Judge the answer
    judge_start = time.time()
    try:
        judge_result = judge.invoke([
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=f"""## QUESTION
{question}

## EXPECTED ANSWER
{expected}

## AGENT'S ANSWER
{agent_answer}

Evaluate whether the agent's answer is correct.""")
        ])
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
        query_type=query_type,
        agent_time_sec=agent_time,
        analysis_time_ms=analysis_time,
        retrieval_time_ms=retrieval_time,
        generation_time_ms=generation_time,
        judge_time_sec=judge_time,
        chunks_retrieved=chunks_retrieved,
        fallback_used=fallback_used,
        key_facts_matched=judge_result.key_facts_matched,
        key_facts_missing=judge_result.key_facts_missing,
    )


def run_evaluation(qa_pairs: list[dict], limit: Optional[int] = None):
    """Run evaluation on all questions (synchronous)."""
    pairs = qa_pairs[:limit] if limit else qa_pairs
    total = len(pairs)

    print(f"\n{'='*70}")
    print(f"STRUCTURED KG-RAG PIPELINE EVALUATION")
    print(f"Questions: {total}")
    print(f"{'='*70}\n")

    # Initialize pipeline
    pipeline = StructuredKGRAG(user_id="default")
    judge = get_judge()
    results = []

    # Process sequentially (to avoid rate limits and ensure clean output)
    for i, pair in enumerate(pairs):
        q_id = pair.get("id", i + 1)
        question = pair["question"]
        expected = pair["answer"]

        print(f"[{i+1}/{total}] Q{q_id}: {question[:50]}...", end=" ", flush=True)

        result = evaluate_question(pipeline, judge, question, expected, q_id)
        results.append(result)

        icon = {"correct": "PASS", "partially_correct": "PART", "incorrect": "FAIL", "unanswerable": "N/A"}
        print(f"[{icon.get(result.verdict.value, '?')}] ({result.agent_time_sec:.1f}s, {result.chunks_retrieved} chunks)")

        if result.verdict != JudgeVerdict.CORRECT:
            print(f"      Expected: {expected[:80]}...")
            print(f"      Got: {result.agent_answer[:100]}...")
            print(f"      Type: {result.query_type}, Fallback: {result.fallback_used}")

    return results


def print_summary(results: list[EvalResult]):
    """Print evaluation summary."""
    total = len(results)
    correct = sum(1 for r in results if r.verdict == JudgeVerdict.CORRECT)
    partial = sum(1 for r in results if r.verdict == JudgeVerdict.PARTIALLY_CORRECT)
    incorrect = sum(1 for r in results if r.verdict == JudgeVerdict.INCORRECT)
    unanswerable = sum(1 for r in results if r.verdict == JudgeVerdict.UNANSWERABLE)

    avg_agent_time = sum(r.agent_time_sec for r in results) / total
    avg_analysis = sum(r.analysis_time_ms for r in results) / total
    avg_retrieval = sum(r.retrieval_time_ms for r in results) / total
    avg_generation = sum(r.generation_time_ms for r in results) / total
    avg_chunks = sum(r.chunks_retrieved for r in results) / total
    fallback_count = sum(1 for r in results if r.fallback_used)

    # Count by query type
    query_types = {}
    for r in results:
        qt = r.query_type or "unknown"
        if qt not in query_types:
            query_types[qt] = {"total": 0, "correct": 0}
        query_types[qt]["total"] += 1
        if r.verdict == JudgeVerdict.CORRECT:
            query_types[qt]["correct"] += 1

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
    print(f"  - Analysis:   {avg_analysis:.0f}ms")
    print(f"  - Retrieval:  {avg_retrieval:.0f}ms")
    print(f"  - Generation: {avg_generation:.0f}ms")
    print(f"\nRetrieval Stats:")
    print(f"  Avg Chunks: {avg_chunks:.1f}")
    print(f"  Fallbacks:  {fallback_count}/{total} ({fallback_count/total*100:.1f}%)")
    print(f"\nBy Query Type:")
    for qt, stats in sorted(query_types.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {qt:20s}: {stats['correct']:2d}/{stats['total']:2d} ({acc:5.1f}%)")
    print(f"{'='*70}")


def save_results(results: list[EvalResult], output_path: str):
    """Save results to JSON."""
    total = len(results)
    data = {
        "timestamp": datetime.now().isoformat(),
        "system": "structured_pipeline",
        "total_questions": total,
        "summary": {
            "correct": sum(1 for r in results if r.verdict == JudgeVerdict.CORRECT),
            "partially_correct": sum(1 for r in results if r.verdict == JudgeVerdict.PARTIALLY_CORRECT),
            "incorrect": sum(1 for r in results if r.verdict == JudgeVerdict.INCORRECT),
            "unanswerable": sum(1 for r in results if r.verdict == JudgeVerdict.UNANSWERABLE),
            "avg_agent_time_sec": sum(r.agent_time_sec for r in results) / total,
            "avg_chunks": sum(r.chunks_retrieved for r in results) / total,
        },
        "results": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "expected": r.expected_answer,
                "answer": r.agent_answer,
                "verdict": r.verdict.value,
                "reasoning": r.reasoning,
                "query_type": r.query_type,
                "agent_time": r.agent_time_sec,
                "analysis_time_ms": r.analysis_time_ms,
                "retrieval_time_ms": r.retrieval_time_ms,
                "generation_time_ms": r.generation_time_ms,
                "chunks_retrieved": r.chunks_retrieved,
                "fallback_used": r.fallback_used,
            }
            for r in results
        ]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Structured KG-RAG Pipeline")
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
    results = run_evaluation(qa_pairs, args.limit)

    # Print summary
    print_summary(results)

    # Save results
    output = args.output or f"eval_structured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, output)


if __name__ == "__main__":
    main()
