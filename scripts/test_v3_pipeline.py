#!/usr/bin/env python3
"""
Test script for V3 threshold-based pipeline.

Usage:
    VERBOSE=true uv run scripts/test_v3_pipeline.py
    uv run scripts/test_v3_pipeline.py --threshold 0.6
"""

import argparse
import asyncio
import os
import json
from datetime import datetime

# Enable verbose logging
os.environ["VERBOSE"] = os.getenv("VERBOSE", "true")

from src.querying_system.v3.pipeline import ThresholdPipelineV3


TEST_QUESTIONS = [
    # Enumeration
    "Which Federal Reserve districts reported economic growth in October 2024?",
    # Factual
    "What happened to employment in the Boston district?",
    # Comparison
    "How did consumer spending differ between Richmond and Atlanta?",
]


async def test_single_question(
    pipeline: ThresholdPipelineV3,
    question: str,
) -> dict:
    """Test a single question and return results."""
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print("="*80)

    result = await pipeline.query_async(question)

    print(f"\nQUESTION TYPE: {result.decomposition.question_type.value}")
    print(f"ENTITIES: {result.decomposition.entity_hints}")
    print(f"TOPICS: {result.decomposition.topic_hints}")
    print(f"\nFACTS RETRIEVED: {len(result.evidence_pool.scored_facts)}")

    # Show fact breakdown by similarity score
    facts = result.evidence_pool.scored_facts
    if facts:
        scores = [f.vector_score for f in facts]
        print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"  Avg score: {sum(scores)/len(scores):.3f}")

        # Show top 5 facts
        print(f"\nTOP 5 FACTS:")
        for i, fact in enumerate(facts[:5]):
            print(f"  [{i+1}] {fact.subject} -[{fact.edge_type}]-> {fact.object}")
            print(f"      {fact.content[:100]}...")
            print(f"      Score: {fact.vector_score:.3f}")

    print(f"\nANSWER:")
    print(result.answer.answer)

    print(f"\n{'-'*80}")
    print(f"Confidence: {result.answer.confidence:.2f}")
    print(f"Timing:")
    print(f"  Decomposition: {result.answer.decomposition_time_ms}ms")
    print(f"  Retrieval:     {result.answer.retrieval_time_ms}ms")
    print(f"  Synthesis:     {result.answer.synthesis_time_ms}ms")
    print(f"  Total:         {result.answer.total_time_ms}ms")

    return {
        "question": question,
        "question_type": result.decomposition.question_type.value,
        "facts_retrieved": len(facts),
        "answer": result.answer.answer,
        "confidence": result.answer.confidence,
        "total_time_ms": result.answer.total_time_ms,
    }


async def main():
    parser = argparse.ArgumentParser(description="Test V3 Pipeline")
    parser.add_argument("-t", "--threshold", type=float, default=0.7,
                        help="Similarity threshold (default: 0.7)")
    parser.add_argument("-g", "--group-id", default="default")
    parser.add_argument("-q", "--question", help="Single question to test")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"# V3 Pipeline Test - Threshold: {args.threshold}")
    print(f"{'#'*80}")

    pipeline = ThresholdPipelineV3(
        group_id=args.group_id,
        similarity_threshold=args.threshold,
    )

    questions = [args.question] if args.question else TEST_QUESTIONS
    results = []

    for q in questions:
        result = await test_single_question(pipeline, q)
        results.append(result)

    # Summary
    print(f"\n\n{'#'*80}")
    print("# SUMMARY")
    print(f"{'#'*80}")
    print(f"Threshold: {args.threshold}")
    print(f"Questions tested: {len(results)}")
    print(f"Avg facts retrieved: {sum(r['facts_retrieved'] for r in results) / len(results):.1f}")
    print(f"Avg time: {sum(r['total_time_ms'] for r in results) / len(results):.0f}ms")

    if args.save:
        filename = f"eval_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump({
                "threshold": args.threshold,
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())
