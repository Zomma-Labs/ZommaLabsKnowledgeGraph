"""Test V6 Pipeline on first 10 questions from Biege_OA.json"""

import asyncio
import json
import time
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.querying_system.v6 import query_v6, ResearcherConfig


async def run_single_question(qa_pair: dict, config: ResearcherConfig) -> dict:
    """Run a single question and return results."""
    qid = qa_pair["id"]
    question = qa_pair["question"]
    expected = qa_pair["answer"]

    print(f"[Q{qid}] Starting: {question[:60]}...")
    start = time.time()

    try:
        result = await query_v6(question, config=config)
        elapsed = time.time() - start

        print(f"[Q{qid}] Done in {elapsed:.1f}s (conf={result.confidence:.2f})")

        # Save actual evidence for analysis
        evidence_list = [
            {
                "fact_id": ev.fact_id,
                "content": ev.content,
                "subject": ev.subject,
                "object": ev.object,
                "score": ev.score,
                "source_doc": ev.source_doc,
            }
            for ev in result.evidence
        ]

        return {
            "id": qid,
            "question": question,
            "expected": expected,
            "predicted": result.answer,
            "confidence": result.confidence,
            "question_type": result.question_type,
            "num_evidence": len(result.evidence),
            "evidence": evidence_list,  # Now includes actual chunks
            "num_sub_answers": len(result.sub_answers),
            "time_ms": int(elapsed * 1000),
            "gaps": result.gaps,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"[Q{qid}] ERROR in {elapsed:.1f}s: {e}")
        return {
            "id": qid,
            "question": question,
            "expected": expected,
            "predicted": None,
            "confidence": 0.0,
            "question_type": None,
            "num_evidence": 0,
            "num_sub_answers": 0,
            "time_ms": int(elapsed * 1000),
            "gaps": [],
            "error": str(e),
        }


async def main():
    # Load questions
    qa_path = Path(__file__).parent.parent / "Biege_OA.json"
    with open(qa_path) as f:
        data = json.load(f)

    qa_pairs = data["qa_pairs"][:10]  # First 10

    print(f"\n{'='*80}")
    print(f"V6 Pipeline Test - First 10 Questions")
    print(f"{'='*80}\n")

    # Configure pipeline
    config = ResearcherConfig(
        relevance_threshold=0.65,
        enable_gap_expansion=True,
        enable_entity_drilldown=True,
        enable_refinement_loop=True,
    )

    # Run all in parallel
    total_start = time.time()

    tasks = [run_single_question(qa, config) for qa in qa_pairs]
    results = await asyncio.gather(*tasks)

    total_time = time.time() - total_start

    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")

    for r in results:
        print(f"\n--- Q{r['id']}: {r['question'][:70]}{'...' if len(r['question']) > 70 else ''} ---")
        print(f"Type: {r['question_type']} | Conf: {r['confidence']:.2f} | Evidence: {r['num_evidence']} | Time: {r['time_ms']}ms")

        if r['error']:
            print(f"ERROR: {r['error']}")
            continue

        print(f"\nExpected:\n{r['expected'][:300]}{'...' if len(r['expected']) > 300 else ''}")
        print(f"\nPredicted:\n{r['predicted'][:300]}{'...' if len(r['predicted']) > 300 else ''}")

        if r['gaps']:
            print(f"\nGaps: {r['gaps']}")

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
        avg_evidence = sum(r['num_evidence'] for r in successful) / len(successful)
        print(f"Avg confidence: {avg_conf:.2f}")
        print(f"Avg evidence count: {avg_evidence:.1f}")

    # Save results
    output_path = Path(__file__).parent.parent / "v6_test_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "total_time_s": total_time,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
