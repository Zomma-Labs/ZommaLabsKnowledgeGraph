"""Test Deep Research Pipeline on first 10 questions from Biege_OA.json"""

import asyncio
import json
import time
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.querying_system.deep_research import DeepResearchPipeline


async def run_single_question(pipeline: DeepResearchPipeline, qa_pair: dict) -> dict:
    """Run a single question and return results."""
    qid = qa_pair["id"]
    question = qa_pair["question"]
    expected = qa_pair["answer"]

    print(f"[Q{qid}] Starting: {question[:60]}...")
    start = time.time()

    try:
        result = await pipeline.query_async(question, verbose=False)
        elapsed = time.time() - start

        # Calculate confidence from findings
        if result.findings:
            avg_conf = sum(f.confidence for f in result.findings) / len(result.findings)
        else:
            avg_conf = 0.5

        print(f"[Q{qid}] Done in {elapsed:.1f}s (findings={len(result.findings)}, avg_conf={avg_conf:.2f})")

        return {
            "id": qid,
            "question": question,
            "expected": expected,
            "predicted": result.answer,
            "confidence": avg_conf,
            "num_findings": len(result.findings),
            "research_brief": result.research_brief,
            "findings": [
                {
                    "topic": f.topic,
                    "finding": f.finding,
                    "confidence": f.confidence,
                    "evidence_chunks": f.evidence_chunks,
                }
                for f in result.findings
            ],
            "time_ms": int(elapsed * 1000),
            "research_time_ms": result.research_time_ms,
            "synthesis_time_ms": result.synthesis_time_ms,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"[Q{qid}] ERROR in {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return {
            "id": qid,
            "question": question,
            "expected": expected,
            "predicted": None,
            "confidence": 0.0,
            "num_findings": 0,
            "research_brief": None,
            "findings": [],
            "time_ms": int(elapsed * 1000),
            "research_time_ms": 0,
            "synthesis_time_ms": 0,
            "error": str(e),
        }


async def main():
    # Load questions
    qa_path = Path(__file__).parent.parent / "Biege_OA.json"
    with open(qa_path) as f:
        data = json.load(f)

    qa_pairs = data["qa_pairs"][:10]  # First 10

    print(f"\n{'='*80}")
    print(f"Deep Research Pipeline Test - First 10 Questions")
    print(f"{'='*80}\n")

    # Create pipeline (shared across queries)
    pipeline = DeepResearchPipeline(
        user_id="default",
        max_supervisor_iterations=3,
        max_concurrent_researchers=8,
    )

    # Run all in parallel
    total_start = time.time()

    tasks = [run_single_question(pipeline, qa) for qa in qa_pairs]
    results = await asyncio.gather(*tasks)

    total_time = time.time() - total_start

    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")

    for r in results:
        print(f"\n--- Q{r['id']}: {r['question'][:70]}{'...' if len(r['question']) > 70 else ''} ---")
        print(f"Findings: {r['num_findings']} | Conf: {r['confidence']:.2f} | Time: {r['time_ms']}ms")

        if r['error']:
            print(f"ERROR: {r['error']}")
            continue

        print(f"\nExpected:\n{r['expected'][:300]}{'...' if len(r['expected']) > 300 else ''}")
        print(f"\nPredicted:\n{r['predicted'][:300] if r['predicted'] else 'None'}{'...' if r['predicted'] and len(r['predicted']) > 300 else ''}")

    # Summary stats
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    successful = [r for r in results if not r['error']]
    errors = [r for r in results if r['error']]

    print(f"Total questions: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Errors: {len(errors)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per question: {sum(r['time_ms'] for r in results) / len(results) / 1000:.1f}s")

    if successful:
        avg_conf = sum(r['confidence'] for r in successful) / len(successful)
        avg_findings = sum(r['num_findings'] for r in successful) / len(successful)
        print(f"Avg confidence: {avg_conf:.2f}")
        print(f"Avg findings count: {avg_findings:.1f}")

    # Save results
    output_path = Path(__file__).parent.parent / "deep_research_test_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "total_time_s": total_time,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
