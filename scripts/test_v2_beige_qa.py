#!/usr/bin/env python3
"""Test v2 querying system on first 10 Beige Book QA pairs with parallel execution."""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.querying_system.v2.pipeline import GNNPipelineV2


async def process_question(pipeline: GNNPipelineV2, qa: dict, idx: int, total: int, semaphore: asyncio.Semaphore):
    """Process a single question with semaphore for concurrency control."""
    async with semaphore:
        print(f"\n[{idx}/{total}] Starting: {qa['question'][:60]}...", flush=True)

        try:
            start = time.time()
            result = await pipeline.query_async(qa["question"])
            elapsed = time.time() - start

            answer_text = result.answer.answer
            evidence_count = len(result.evidence_pool.scored_facts) if result.evidence_pool else 0

            print(f"[{idx}/{total}] Done in {elapsed:.1f}s | Conf: {result.answer.confidence:.2f} | Evidence: {evidence_count}", flush=True)

            timing = {
                "decomposition_ms": result.answer.decomposition_time_ms,
                "retrieval_ms": result.answer.retrieval_time_ms,
                "scoring_ms": result.answer.scoring_time_ms,
                "expansion_ms": result.answer.expansion_time_ms,
                "synthesis_ms": result.answer.synthesis_time_ms,
                "total_ms": result.answer.total_time_ms,
            }

            return {
                "id": qa["id"],
                "question": qa["question"],
                "category": qa["category"],
                "expected": qa["answer"],
                "generated": answer_text,
                "confidence": result.answer.confidence,
                "timing": timing,
                "evidence_count": evidence_count,
                "gaps": result.answer.gaps
            }

        except Exception as e:
            import traceback
            print(f"[{idx}/{total}] ERROR: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return {
                "id": qa["id"],
                "question": qa["question"],
                "category": qa["category"],
                "expected": qa["answer"],
                "generated": None,
                "error": str(e)
            }


async def run_evaluation(concurrency: int = 3):
    """Run evaluation with parallel question processing."""
    # Load QA pairs
    qa_path = Path(__file__).parent.parent / "Biege_OA.json"
    with open(qa_path) as f:
        data = json.load(f)

    qa_pairs = data["qa_pairs"][:10]  # First 10 questions

    print(f"Running {len(qa_pairs)} questions with concurrency={concurrency}", flush=True)
    print("=" * 80, flush=True)

    # Initialize pipeline
    pipeline = GNNPipelineV2(group_id="default")

    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    # Create tasks for all questions
    start_time = time.time()
    tasks = [
        process_question(pipeline, qa, i+1, len(qa_pairs), semaphore)
        for i, qa in enumerate(qa_pairs)
    ]

    # Run all tasks concurrently (semaphore controls actual parallelism)
    results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    # Save results
    output_path = Path(__file__).parent.parent / f"eval_v2_beige_10.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    successful = [r for r in results if 'error' not in r]
    print(f"Completed: {len(successful)}/{len(results)}", flush=True)
    print(f"Total wall-clock time: {total_time:.1f}s", flush=True)
    print(f"Results saved to: {output_path}", flush=True)

    if successful:
        avg_query_time = sum(r["timing"].get("total_ms", 0) for r in successful) / len(successful) / 1000
        avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        avg_evidence = sum(r["evidence_count"] for r in successful) / len(successful)
        print(f"\nPer-query metrics:", flush=True)
        print(f"  Avg query time: {avg_query_time:.2f}s", flush=True)
        print(f"  Avg confidence: {avg_confidence:.2f}", flush=True)
        print(f"  Avg evidence: {avg_evidence:.1f} facts", flush=True)

        # Speedup calculation
        sequential_estimate = avg_query_time * len(successful)
        speedup = sequential_estimate / total_time if total_time > 0 else 1
        print(f"\n  Sequential estimate: {sequential_estimate:.1f}s", flush=True)
        print(f"  Parallel speedup: {speedup:.2f}x", flush=True)

    # Print per-question summary
    print(f"\n{'='*80}", flush=True)
    print("PER-QUESTION RESULTS:", flush=True)
    print(f"{'='*80}", flush=True)
    for r in results:
        status = "OK" if "error" not in r else "ERR"
        time_s = r.get("timing", {}).get("total_ms", 0) / 1000 if "timing" in r else 0
        conf = r.get("confidence", 0)
        print(f"Q{r['id']:2d} [{status}] {time_s:6.1f}s | Conf: {conf:.2f} | {r['question'][:50]}...", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--concurrency", type=int, default=3, help="Number of parallel queries")
    args = parser.parse_args()

    asyncio.run(run_evaluation(concurrency=args.concurrency))
