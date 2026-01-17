#!/usr/bin/env python
"""
Integration test for V5 Pipeline.

Run with:
    uv run python scripts/test_v5_pipeline.py

Or with verbose output:
    VERBOSE=true uv run python scripts/test_v5_pipeline.py
"""

import asyncio
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set verbose for testing
os.environ.setdefault("VERBOSE", "true")

from src.querying_system.v5 import V5Pipeline, ResearcherConfig


# Test questions covering different types
TEST_QUESTIONS = [
    # Enumeration - slight to modest growth
    "Which districts reported slight to modest economic growth?",

    # Enumeration - slight softening or decline
    "Which districts reported a slight softening or decline in economic activity?",
]


async def test_single_question(pipeline: V5Pipeline, question: str) -> dict:
    """Test a single question and return results."""
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print('='*60)

    try:
        result = await pipeline.query(question)

        print(f"\nANSWER ({result.confidence:.2f} confidence):")
        print(result.answer)

        print(f"\nSUB-QUERIES ({len(result.sub_answers)}):")
        for i, sa in enumerate(result.sub_answers, 1):
            print(f"  {i}. {sa.sub_query}")
            print(f"     -> {sa.answer[:100]}..." if len(sa.answer) > 100 else f"     -> {sa.answer}")
            print(f"     (confidence: {sa.confidence:.2f}, facts: {len(sa.facts_used)})")

        print(f"\nEVIDENCE ({len(result.evidence)} facts):")
        for i, ev in enumerate(result.evidence[:5], 1):
            print(f"  {i}. [{ev.score:.2f}] {ev.subject} {ev.edge_type} {ev.object}")
            print(f"     {ev.content[:80]}...")

        if result.gaps:
            print(f"\nGAPS: {result.gaps}")

        print(f"\nTIMING:")
        print(f"  Decomposition: {result.decomposition_time_ms}ms")
        print(f"  Research: {result.research_time_ms}ms")
        print(f"  Synthesis: {result.synthesis_time_ms}ms")
        print(f"  TOTAL: {result.total_time_ms}ms")

        return {
            "question": question,
            "success": True,
            "answer": result.answer,
            "confidence": result.confidence,
            "num_sub_queries": len(result.sub_answers),
            "num_evidence": len(result.evidence),
            "total_time_ms": result.total_time_ms,
        }

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "question": question,
            "success": False,
            "error": str(e),
        }


async def test_config_comparison():
    """Test different configurations for A/B comparison."""
    print("\n" + "="*60)
    print("CONFIGURATION COMPARISON")
    print("="*60)

    configs = {
        "full": ResearcherConfig(
            enable_global_search=True,
            enable_gap_expansion=True,
            enable_entity_drilldown=True,
        ),
        "no_drilldown": ResearcherConfig(
            enable_global_search=True,
            enable_gap_expansion=True,
            enable_entity_drilldown=False,
        ),
        "no_gap_expansion": ResearcherConfig(
            enable_global_search=True,
            enable_gap_expansion=False,
            enable_entity_drilldown=True,
        ),
        "minimal": ResearcherConfig(
            enable_global_search=True,
            enable_gap_expansion=False,
            enable_entity_drilldown=False,
        ),
    }

    # Test with enumeration question (where drill-down matters)
    question = "Which Federal Reserve districts reported employment growth?"

    results = {}
    for name, config in configs.items():
        print(f"\n--- Config: {name} ---")
        pipeline = V5Pipeline(config=config)
        result = await pipeline.query(question)
        results[name] = {
            "confidence": result.confidence,
            "num_evidence": len(result.evidence),
            "time_ms": result.total_time_ms,
            "answer_length": len(result.answer),
        }
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Evidence: {len(result.evidence)} facts")
        print(f"  Time: {result.total_time_ms}ms")

    print("\n--- Summary ---")
    for name, r in results.items():
        print(f"  {name}: conf={r['confidence']:.2f}, ev={r['num_evidence']}, time={r['time_ms']}ms")


async def main():
    """Run integration tests."""
    print("V5 Pipeline Integration Test")
    print("="*60)

    # Test with default config
    pipeline = V5Pipeline()

    # Run all test questions
    results = []
    for question in TEST_QUESTIONS:  # Run all enumeration questions
        result = await test_single_question(pipeline, question)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    success_count = sum(1 for r in results if r.get("success"))
    print(f"Passed: {success_count}/{len(results)}")

    avg_time = sum(r.get("total_time_ms", 0) for r in results if r.get("success")) / max(success_count, 1)
    print(f"Avg time: {avg_time:.0f}ms")

    # Config comparison (optional, comment out for quick tests)
    # await test_config_comparison()

    # Save results
    output_file = f"test_v5_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
