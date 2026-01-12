#!/usr/bin/env python3
"""
Test Deterministic Retrieval System
====================================

This script tests the new multi-strategy retrieval with RRF fusion
and compares it against multiple runs for consistency.

Usage:
    uv run scripts/test_deterministic_retrieval.py
    uv run scripts/test_deterministic_retrieval.py --runs 5  # Test consistency over 5 runs
"""

import sys
import os
import json
import asyncio
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.util.deterministic_retrieval import DeterministicRetriever, search_kg


# Test questions - these should match what you've been testing with
TEST_QUESTIONS = [
    "What happened to wages in Boston?",
    "Which districts reported declining manufacturing activity?",
    "What are the trends in consumer spending?",
    "How did employment change across districts?",
    "What sectors showed growth?",
]


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_evidence_summary(evidence: list, max_items: int = 5):
    """Print a summary of retrieved evidence."""
    if not evidence:
        print("  No evidence found")
        return

    for i, e in enumerate(evidence[:max_items], start=1):
        strategies = ", ".join(e.found_by)
        print(f"  {i}. [RRF: {e.rrf_score:.4f}] ({strategies})")
        print(f"     {e.subject} -[{e.edge_type}]-> {e.object}")
        if e.content:
            content_preview = e.content[:80] + "..." if len(e.content) > 80 else e.content
            print(f"     \"{content_preview}\"")
        print()

    if len(evidence) > max_items:
        print(f"  ... and {len(evidence) - max_items} more results")


async def test_single_query(retriever: DeterministicRetriever, query: str, verbose: bool = True):
    """Test a single query and return results."""
    if verbose:
        print(f"\nQuery: \"{query}\"")
        print("-" * 40)

    results = await retriever.search(query, top_k=10)

    if verbose:
        print(f"Found {len(results)} results")
        print_evidence_summary(results)

    return results


async def test_consistency(query: str, group_id: str, num_runs: int = 3):
    """
    Test if the same query returns the same results across multiple runs.

    This is the key test - deterministic retrieval should give identical results.
    """
    print_header(f"CONSISTENCY TEST: {num_runs} runs")
    print(f"Query: \"{query}\"")

    all_results = []

    for run in range(num_runs):
        # Create fresh retriever each time
        retriever = DeterministicRetriever(group_id=group_id)
        results = await retriever.search(query, top_k=10)

        # Extract fact IDs for comparison
        fact_ids = [e.fact_id for e in results]
        all_results.append(fact_ids)

        print(f"\nRun {run + 1}: Found {len(results)} results")
        print(f"  Top 5 fact IDs: {fact_ids[:5]}")

    # Check consistency
    print("\n" + "-" * 40)
    print("CONSISTENCY CHECK:")

    # Compare all runs to first run
    first_run = all_results[0]
    all_identical = True

    for i, run_results in enumerate(all_results[1:], start=2):
        if run_results == first_run:
            print(f"  Run {i} vs Run 1: IDENTICAL")
        else:
            print(f"  Run {i} vs Run 1: DIFFERENT")
            # Show what's different
            set1, set2 = set(first_run), set(run_results)
            only_in_1 = set1 - set2
            only_in_2 = set2 - set1
            if only_in_1:
                print(f"    Only in Run 1: {list(only_in_1)[:3]}")
            if only_in_2:
                print(f"    Only in Run {i}: {list(only_in_2)[:3]}")
            all_identical = False

    if all_identical:
        print("\n DETERMINISTIC: All runs returned identical results!")
    else:
        print("\n WARNING: Results varied between runs")

    return all_identical


async def test_strategy_coverage(retriever: DeterministicRetriever, query: str):
    """
    Test which strategies contribute to the results.

    Good retrieval should have contributions from multiple strategies.
    """
    print_header("STRATEGY COVERAGE TEST")
    print(f"Query: \"{query}\"")

    results = await retriever.search(query, top_k=20)

    # Count strategy contributions
    strategy_counts = defaultdict(int)
    multi_strategy_count = 0

    for e in results:
        for strategy in e.found_by:
            strategy_counts[strategy] += 1
        if len(e.found_by) > 1:
            multi_strategy_count += 1

    print(f"\nTotal results: {len(results)}")
    print(f"\nStrategy contributions:")
    for strategy, count in sorted(strategy_counts.items()):
        pct = (count / len(results) * 100) if results else 0
        print(f"  {strategy}: {count} ({pct:.1f}%)")

    print(f"\nMulti-strategy results: {multi_strategy_count} ({multi_strategy_count/len(results)*100:.1f}%)")

    # Show RRF score distribution
    if results:
        scores = [e.rrf_score for e in results]
        print(f"\nRRF Score distribution:")
        print(f"  Max: {max(scores):.4f}")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Avg: {sum(scores)/len(scores):.4f}")

    return strategy_counts


async def test_all_questions(group_id: str):
    """Test all questions and summarize results."""
    print_header("TESTING ALL QUESTIONS")

    retriever = DeterministicRetriever(group_id=group_id)
    results_summary = []

    for question in TEST_QUESTIONS:
        results = await test_single_query(retriever, question, verbose=True)
        results_summary.append({
            "question": question,
            "num_results": len(results),
            "strategies": list(set(s for e in results for s in e.found_by)),
            "top_rrf": results[0].rrf_score if results else 0,
        })

    print_header("SUMMARY")
    for item in results_summary:
        print(f"\nQ: {item['question']}")
        print(f"   Results: {item['num_results']}, Strategies: {item['strategies']}, Top RRF: {item['top_rrf']:.4f}")


async def run_full_test(group_id: str = "default", consistency_runs: int = 3):
    """Run the full test suite."""
    print_header("DETERMINISTIC RETRIEVAL TEST SUITE")
    print(f"Group ID: {group_id}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # 1. Test all questions
    await test_all_questions(group_id)

    # 2. Test consistency on first question
    print("\n")
    await test_consistency(TEST_QUESTIONS[0], group_id, num_runs=consistency_runs)

    # 3. Test strategy coverage on first question
    retriever = DeterministicRetriever(group_id=group_id)
    await test_strategy_coverage(retriever, TEST_QUESTIONS[0])

    print_header("TEST COMPLETE")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test deterministic retrieval")
    parser.add_argument("--group-id", default="default", help="Tenant group ID")
    parser.add_argument("--runs", type=int, default=3, help="Number of consistency test runs")
    parser.add_argument("--query", type=str, help="Test a specific query")
    args = parser.parse_args()

    if args.query:
        # Test single query
        retriever = DeterministicRetriever(group_id=args.group_id)
        results = asyncio.run(retriever.search(args.query, top_k=10))
        print_header(f"Results for: {args.query}")
        print_evidence_summary(results, max_items=10)

        # Also print formatted output
        print("\n" + "-" * 40)
        print("FORMATTED FOR LLM:")
        print("-" * 40)
        print(retriever.format_evidence_for_llm(results))
    else:
        # Run full test suite
        asyncio.run(run_full_test(args.group_id, args.runs))


if __name__ == "__main__":
    main()
