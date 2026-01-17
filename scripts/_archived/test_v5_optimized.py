"""Test V5 pipeline on 10 Beige Book QA questions in parallel."""

import asyncio
import json
import time

from src.querying_system.v5.pipeline import V5Pipeline
from src.querying_system.v5.schemas import ResearcherConfig


async def run_question(pipeline: V5Pipeline, q: dict) -> dict:
    """Run a single question and return result."""
    start = time.time()
    try:
        result = await pipeline.query(q["question"])
        elapsed = time.time() - start
        return {
            "id": q["id"],
            "question": q["question"],
            "expected": q["answer"],
            "actual": result.answer,
            "confidence": result.confidence,
            "time_s": elapsed,
            "num_facts": len(result.evidence),
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "id": q["id"],
            "question": q["question"],
            "expected": q["answer"],
            "actual": None,
            "confidence": 0,
            "time_s": elapsed,
            "num_facts": 0,
            "error": str(e),
        }


async def main():
    # Load questions
    with open("Biege_OA.json") as f:
        data = json.load(f)

    # First 10 questions
    questions = data["qa_pairs"][:10]

    print("=" * 80)
    print("V5 PIPELINE TEST - 10 QUESTIONS IN PARALLEL")
    print("=" * 80)

    # Initialize pipeline
    config = ResearcherConfig(
        refinement_confidence_threshold=0.85,
    )
    pipeline = V5Pipeline(group_id="default", config=config)

    total_start = time.time()

    # Run all 10 questions in parallel
    tasks = [run_question(pipeline, q) for q in questions]
    results = await asyncio.gather(*tasks)

    total_time = time.time() - total_start

    # Print results
    for r in results:
        print(f"\n{'─' * 80}")
        print(f"Q{r['id']}: {r['question'][:60]}...")
        print(f"{'─' * 80}")
        if r["error"]:
            print(f"❌ ERROR: {r['error']}")
        else:
            print(f"📝 ANSWER ({r['time_s']:.1f}s, conf={r['confidence']:.2f}):")
            ans = r["actual"] or ""
            print(ans[:300] + "..." if len(ans) > 300 else ans)
            print(f"\n✅ EXPECTED: {r['expected'][:200]}...")

    print(f"\n{'=' * 80}")
    print(f"TOTAL TIME: {total_time:.1f}s for {len(questions)} questions (parallel)")
    print(f"AVG TIME: {sum(r['time_s'] for r in results) / len(results):.1f}s per question")
    print("=" * 80)

    # Save results
    with open("test_v5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to test_v5_results.json")


if __name__ == "__main__":
    asyncio.run(main())
