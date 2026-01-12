#!/usr/bin/env python3
"""
Integration test for Entity Deduplication System (src/util/entity_dedup.py)

Tests the embedding + LLM hybrid approach with real embeddings and Claude Sonnet.
Uses comprehensive financial entity test data with known ground truth.

Usage:
    uv run python src/scripts/test_entity_dedup_integration.py
    uv run python src/scripts/test_entity_dedup_integration.py --num-entities 200
"""

import argparse
import os
import sys
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from src.util.services import get_services
from src.util.entity_dedup import DeferredDeduplicationManager

# Import test data
from src.scripts.test_entity_deduplicator import ENTITY_GROUPS, ENTITY_SUMMARIES


# =============================================================================
# Test Data Creation
# =============================================================================

def create_test_entities(num_entities: int = 100):
    """Create test entities with ground truth mapping."""
    all_entities = []
    ground_truth = {}

    for group in ENTITY_GROUPS:
        canonical = group[0]
        for name in group:
            summary = ENTITY_SUMMARIES.get(name, "")
            all_entities.append({"name": name, "summary": summary})
            ground_truth[name] = canonical

    random.shuffle(all_entities)

    if len(all_entities) > num_entities:
        all_entities = all_entities[:num_entities]
        ground_truth = {k: v for k, v in ground_truth.items() if k in [e["name"] for e in all_entities]}

    return all_entities, ground_truth


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_dedup(result_mapping: dict, ground_truth: dict) -> dict:
    """
    Evaluate deduplication using precision/recall/F1 on entity pairs.

    - Precision: Of all pairs we merged, how many were correct?
    - Recall: Of all pairs that should be merged, how many did we get?
    """
    # Build groups from result mapping
    result_groups = {}
    for entity, canonical in result_mapping.items():
        result_groups.setdefault(canonical, []).append(entity)

    # Build groups from ground truth
    truth_groups = {}
    for entity, canonical in ground_truth.items():
        truth_groups.setdefault(canonical, []).append(entity)

    # Count pairs
    correct_pairs = 0
    total_result_pairs = 0
    total_truth_pairs = 0

    # Count correct pairs in result groups
    for canonical, entities in result_groups.items():
        entities_in_truth = [e for e in entities if e in ground_truth]
        n = len(entities_in_truth)
        total_result_pairs += n * (n - 1) // 2
        for i, e1 in enumerate(entities_in_truth):
            for e2 in entities_in_truth[i+1:]:
                if ground_truth[e1] == ground_truth[e2]:
                    correct_pairs += 1

    # Count expected pairs from ground truth
    for canonical, entities in truth_groups.items():
        entities_in_result = [e for e in entities if e in result_mapping]
        n = len(entities_in_result)
        total_truth_pairs += n * (n - 1) // 2

    # Calculate metrics
    precision = correct_pairs / max(total_result_pairs, 1)
    recall = correct_pairs / max(total_truth_pairs, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    wrong_merges = total_result_pairs - correct_pairs

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_pairs": correct_pairs,
        "wrong_merges": wrong_merges,
        "total_entities": len(result_mapping),
        "unique_canonicals": len(set(result_mapping.values())),
        "expected_canonicals": len(set(v for k, v in ground_truth.items() if k in result_mapping))
    }


# =============================================================================
# Main Test
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test entity deduplication system")
    parser.add_argument("--num-entities", "-n", type=int, default=100, help="Number of entities to test")
    parser.add_argument("--threshold", "-t", type=float, default=0.85, help="Similarity threshold")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "true"

    print("="*70)
    print("ENTITY DEDUPLICATION INTEGRATION TEST")
    print("System: Embedding similarity + LLM verification")
    print("="*70)

    # Create test entities
    print(f"\nCreating {args.num_entities} test entities...")
    entities, ground_truth = create_test_entities(args.num_entities)
    expected_groups = len(set(ground_truth.values()))
    print(f"  {len(entities)} entities, {expected_groups} expected canonical groups")

    # Initialize services
    print("\nInitializing services...")
    services = get_services()

    print(f"  LLM: GPT-5.2 (via services.llm)")
    print(f"  Embeddings: voyage-3-large (via services.dedup_embeddings)")

    # Reset and get dedup manager
    DeferredDeduplicationManager.reset()
    manager = DeferredDeduplicationManager.get_instance()

    # Generate embeddings
    print("\nGenerating embeddings...")
    start_embed = time.time()
    texts = [f"{e['name']}: {e['summary']}" for e in entities]
    entity_embeddings = services.dedup_embeddings.embed_documents(texts)
    embed_time = time.time() - start_embed
    print(f"  Generated {len(entity_embeddings)} embeddings in {embed_time:.1f}s")

    # Register entities
    print("\nRegistering entities...")
    for i, (entity, emb) in enumerate(zip(entities, entity_embeddings)):
        manager.register_entity(
            uuid=f"entity-{i:03d}",
            name=entity["name"],
            node_type="Entity",
            summary=entity["summary"],
            embedding=emb,
            group_id="test"
        )

    # Run deduplication
    print(f"\nRunning deduplication (threshold={args.threshold})...")
    print("-"*70)
    start_dedup = time.time()
    stats = manager.cluster_and_remap(similarity_threshold=args.threshold)
    dedup_time = time.time() - start_dedup

    print(f"\n  Components found: {stats['components_found']}")
    print(f"  LLM calls: {stats['llm_calls']}")
    print(f"  Distinct entities: {stats['distinct_entities']}")
    print(f"  Duplicates merged: {stats['duplicates_merged']}")
    print(f"  Time: {dedup_time:.1f}s")

    # Build result mapping
    result_mapping = {}
    for uuid, entity in manager._pending_entities.items():
        canonical_uuid = manager.get_remapped_uuid(uuid)
        canonical_entity = manager._pending_entities[canonical_uuid]
        result_mapping[entity.name] = canonical_entity.name

    # Evaluate
    metrics = evaluate_dedup(result_mapping, ground_truth)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall: {metrics['recall']:.1%}")
    print(f"  F1 Score: {metrics['f1']:.1%}")
    print(f"  Correct pairs: {metrics['correct_pairs']}")
    print(f"  Wrong merges: {metrics['wrong_merges']}")
    print(f"  Unique canonicals: {metrics['unique_canonicals']} (expected: {metrics['expected_canonicals']})")
    print(f"  Total time: {embed_time + dedup_time:.1f}s")

    # Show merged groups
    print("\n" + "-"*70)
    print("MERGED GROUPS (top 15):")
    print("-"*70)
    result_groups = {}
    for entity, canonical in result_mapping.items():
        result_groups.setdefault(canonical, []).append(entity)

    merged = [(c, e) for c, e in result_groups.items() if len(e) > 1]
    for canonical, members in sorted(merged, key=lambda x: -len(x[1]))[:15]:
        entities_in_truth = [e for e in members if e in ground_truth]
        if entities_in_truth:
            truth_canonicals = set(ground_truth[e] for e in entities_in_truth)
            status = "✓" if len(truth_canonicals) == 1 else "✗"
        else:
            status = "?"
        print(f"  {status} {canonical}: {members}")

    # Show wrong merges
    print("\n" + "-"*70)
    print("WRONG MERGES:")
    print("-"*70)
    wrong = []
    for canonical, members in result_groups.items():
        if len(members) > 1:
            entities_in_truth = [e for e in members if e in ground_truth]
            if len(entities_in_truth) > 1:
                truth_canonicals = set(ground_truth[e] for e in entities_in_truth)
                if len(truth_canonicals) > 1:
                    wrong.append((canonical, entities_in_truth, truth_canonicals))

    if wrong:
        for canonical, members, truths in wrong[:10]:
            print(f"  ✗ Merged: {members}")
            print(f"    Should be: {truths}")
    else:
        print("  None!")

    # Show missed merges (entities that should be together but aren't)
    print("\n" + "-"*70)
    print("MISSED MERGES (sample):")
    print("-"*70)

    # Build truth groups
    truth_groups = {}
    for entity, canonical in ground_truth.items():
        truth_groups.setdefault(canonical, []).append(entity)

    missed_count = 0
    for truth_canonical, truth_members in truth_groups.items():
        members_in_result = [e for e in truth_members if e in result_mapping]
        if len(members_in_result) > 1:
            result_canonicals = set(result_mapping[e] for e in members_in_result)
            if len(result_canonicals) > 1:
                missed_count += 1
                if missed_count <= 5:
                    print(f"  ✗ Should be together: {members_in_result}")
                    print(f"    But mapped to: {result_canonicals}")

    if missed_count == 0:
        print("  None!")
    elif missed_count > 5:
        print(f"  ... and {missed_count - 5} more missed merges")

    # Cleanup
    DeferredDeduplicationManager.reset()

    # Return exit code based on F1
    if metrics['f1'] >= 0.9:
        print("\n✓ PASSED (F1 >= 90%)")
        return 0
    elif metrics['f1'] >= 0.8:
        print("\n~ ACCEPTABLE (F1 >= 80%)")
        return 0
    else:
        print("\n✗ FAILED (F1 < 80%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
