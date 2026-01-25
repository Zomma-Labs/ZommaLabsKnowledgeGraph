"""
Debug V7 Pipeline - Investigate slow and incorrect questions.

Runs specific questions with verbose logging to understand:
1. Why some queries take 800+ seconds
2. Why partial/incorrect answers occur

Usage:
    uv run scripts/debug_v7_questions.py
    uv run scripts/debug_v7_questions.py --question 15
    uv run scripts/debug_v7_questions.py --question 36 --verbose
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.querying_system.v7 import V7Pipeline, V7Config

# Questions that were problematic in overnight run
PROBLEM_QUESTIONS = {
    # Incorrect
    15: "slow + incorrect",
    30: "slow + incorrect",
    36: "very slow (834s) + incorrect",
    # Very slow (but partial/correct)
    21: "slow (398s)",
    27: "slow (496s)",
    29: "slow (397s)",
    39: "slow (436s)",
    # Partials to investigate
    1: "partial - first question",
    6: "partial",
    11: "partial",
}

def load_qa_data():
    """Load QA dataset."""
    qa_path = Path(__file__).parent.parent / "eval" / "Biege_OA.json"
    with open(qa_path) as f:
        data = json.load(f)
    # Return as dict keyed by question ID
    return {q["id"]: q for q in data["qa_pairs"]}


async def debug_question(question_id: int, question: str, expected: str, verbose: bool = True):
    """Run a single question with detailed timing."""

    if verbose:
        os.environ["VERBOSE"] = "true"
    else:
        os.environ["VERBOSE"] = "false"

    print(f"\n{'='*70}")
    print(f"QUESTION {question_id}: {question[:80]}...")
    print(f"{'='*70}")
    print(f"Expected: {expected[:200]}...")
    print()

    pipeline = V7Pipeline()

    start = time.time()
    result = await pipeline.query(question)
    total_time = time.time() - start

    print(f"\n{'='*70}")
    print("RESULT SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Decomposition: {result.decomposition_time_ms}ms")
    print(f"Resolution: {result.resolution_time_ms}ms")
    print(f"Retrieval: {result.retrieval_time_ms}ms")
    print(f"Synthesis: {result.synthesis_time_ms}ms")
    print()
    print(f"Sub-answers: {len(result.sub_answers)}")
    for i, sa in enumerate(result.sub_answers):
        print(f"\n  Sub-answer {i+1}:")
        print(f"    Query: {sa.sub_query[:60]}...")
        print(f"    Confidence: {sa.confidence}")
        print(f"    Entities found: {sa.entities_found}")
        if hasattr(sa, 'thinking') and sa.thinking:
            print(f"    Thinking: {sa.thinking[:200]}...")
        print(f"    Answer: {sa.answer[:200]}...")

    print(f"\n{'='*70}")
    print("FINAL ANSWER")
    print(f"{'='*70}")
    print(result.answer)

    # Analyze where time was spent
    print(f"\n{'='*70}")
    print("TIME ANALYSIS")
    print(f"{'='*70}")
    decomp_pct = result.decomposition_time_ms / (total_time * 1000) * 100
    resol_pct = result.resolution_time_ms / (total_time * 1000) * 100
    retr_pct = result.retrieval_time_ms / (total_time * 1000) * 100
    synth_pct = result.synthesis_time_ms / (total_time * 1000) * 100
    other_pct = 100 - decomp_pct - resol_pct - retr_pct - synth_pct

    print(f"  Decomposition: {result.decomposition_time_ms/1000:.1f}s ({decomp_pct:.1f}%)")
    print(f"  Resolution:    {result.resolution_time_ms/1000:.1f}s ({resol_pct:.1f}%)")
    print(f"  Retrieval:     {result.retrieval_time_ms/1000:.1f}s ({retr_pct:.1f}%)")
    print(f"  Synthesis:     {result.synthesis_time_ms/1000:.1f}s ({synth_pct:.1f}%)")
    print(f"  Other/overhead:{other_pct:.1f}%")

    if result.resolution_time_ms > 100000:  # > 100s
        print("\n  ⚠️  RESOLUTION is the bottleneck!")
        print("     This means entity/topic resolution is taking too long.")
        print("     Likely cause: too many LLM calls for wide-net resolution.")

    if result.retrieval_time_ms > 100000:
        print("\n  ⚠️  RETRIEVAL is the bottleneck!")
        print("     This means chunk/fact retrieval is taking too long.")
        print("     Likely cause: too many entities causing massive 1-hop expansion.")

    return result


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", type=int, help="Specific question ID to debug")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--all-problems", action="store_true", help="Run all problem questions")
    args = parser.parse_args()

    qa_data = load_qa_data()

    if args.question:
        # Run specific question
        q = qa_data[args.question]
        await debug_question(args.question, q["question"], q["answer"], verbose=args.verbose)

    elif args.all_problems:
        # Run all problem questions
        for qid, reason in PROBLEM_QUESTIONS.items():
            print(f"\n\n{'#'*70}")
            print(f"# Q{qid}: {reason}")
            print(f"{'#'*70}")
            q = qa_data[qid]
            try:
                await debug_question(qid, q["question"], q["answer"], verbose=False)
            except Exception as e:
                print(f"ERROR on Q{qid}: {e}")
            print("\n" + "="*70 + "\n")

    else:
        # Default: run the worst offender (Q36)
        print("Running Q36 (the 834s query) to debug...")
        q = qa_data[36]
        await debug_question(36, q["question"], q["answer"], verbose=args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
