"""
Compare V1 vs V2 pipelines side by side.

V1: Expand from any scored fact (including global search results)
V2: Expand ONLY from scoped results, then LLM scores ALL facts together
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable verbose logging
os.environ["VERBOSE"] = "true"

from src.querying_system.hybrid_cot_gnn import query_hybrid_cot
from src.querying_system.hybrid_cot_gnn.pipeline_v2 import query_hybrid_cot_v2


def compare(question: str):
    """Run both pipelines and compare results."""
    print("=" * 100)
    print(f"QUESTION: {question}")
    print("=" * 100)

    # Run V1
    print("\n" + "=" * 50)
    print("V1 PIPELINE (expand from any scored fact)")
    print("=" * 50)
    start = time.time()
    v1_result = query_hybrid_cot(question)
    v1_time = time.time() - start

    print(f"\nV1 Results:")
    print(f"  Facts retrieved: {len(v1_result.evidence_pool.scored_facts)}")
    print(f"  Expansion: {v1_result.evidence_pool.expansion_performed}")
    print(f"  Confidence: {v1_result.answer.confidence:.2f}")
    print(f"  Total time: {v1_time:.1f}s")

    # Run V2
    print("\n" + "=" * 50)
    print("V2 PIPELINE (expand ONLY from scoped results)")
    print("=" * 50)
    start = time.time()
    v2_result = query_hybrid_cot_v2(question)
    v2_time = time.time() - start

    print(f"\nV2 Results:")
    print(f"  Facts retrieved: {len(v2_result.evidence_pool.scored_facts)}")
    print(f"  Expansion: {v2_result.evidence_pool.expansion_performed}")
    print(f"  Confidence: {v2_result.answer.confidence:.2f}")
    print(f"  Total time: {v2_time:.1f}s")

    # Compare answers
    print("\n" + "=" * 100)
    print("ANSWER COMPARISON")
    print("=" * 100)

    print("\n--- V1 ANSWER (first 1500 chars) ---")
    print(v1_result.answer.answer[:1500])
    print("...")

    print("\n--- V2 ANSWER (first 1500 chars) ---")
    print(v2_result.answer.answer[:1500])
    print("...")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Metric':<30} {'V1':<20} {'V2':<20}")
    print("-" * 70)
    print(f"{'Facts retrieved':<30} {len(v1_result.evidence_pool.scored_facts):<20} {len(v2_result.evidence_pool.scored_facts):<20}")
    print(f"{'Expansion performed':<30} {str(v1_result.evidence_pool.expansion_performed):<20} {str(v2_result.evidence_pool.expansion_performed):<20}")
    print(f"{'Confidence':<30} {v1_result.answer.confidence:<20.2f} {v2_result.answer.confidence:<20.2f}")
    print(f"{'Total time (s)':<30} {v1_time:<20.1f} {v2_time:<20.1f}")
    print(f"{'Decomposition (ms)':<30} {v1_result.answer.decomposition_time_ms:<20} {v2_result.answer.decomposition_time_ms:<20}")
    print(f"{'Retrieval (ms)':<30} {v1_result.answer.retrieval_time_ms:<20} {v2_result.answer.retrieval_time_ms:<20}")
    print(f"{'Expansion (ms)':<30} {v1_result.answer.expansion_time_ms:<20} {v2_result.answer.expansion_time_ms:<20}")
    print(f"{'Scoring (ms)':<30} {v1_result.answer.scoring_time_ms:<20} {v2_result.answer.scoring_time_ms:<20}")
    print(f"{'Synthesis (ms)':<30} {v1_result.answer.synthesis_time_ms:<20} {v2_result.answer.synthesis_time_ms:<20}")

    return v1_result, v2_result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        # Default test question
        question = "Which retail categories saw gains versus declines, and how does this reflect consumer behavior patterns?"

    compare(question)
