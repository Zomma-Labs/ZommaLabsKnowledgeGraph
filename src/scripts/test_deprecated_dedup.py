#!/usr/bin/env python3
"""
Test the deprecated deduplication system with our current test data.
Compares embedding+LLM hybrid approach vs pure LLM approach.
"""

import os
import sys
import random
import time

os.environ["VERBOSE"] = "true"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from src.util.services import get_services

# Import from deprecated folder - add the full path
deprecated_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "deprecated")
sys.path.insert(0, deprecated_path)

# Also need to fix the import inside deferred_dedup which uses src.util.services
# So we import it directly here
from deprecated.util.deferred_dedup import DeferredDeduplicationManager

# Import test data from our current test
from src.scripts.test_entity_deduplicator import ENTITY_GROUPS, ENTITY_SUMMARIES


def create_test_entities(num_entities: int = 100):
    """Create test entities with ground truth."""
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


def evaluate_dedup(result_mapping: dict, ground_truth: dict):
    """Evaluate using same metrics as our current test."""
    result_groups = {}
    for entity, canonical in result_mapping.items():
        result_groups.setdefault(canonical, []).append(entity)

    truth_groups = {}
    for entity, canonical in ground_truth.items():
        truth_groups.setdefault(canonical, []).append(entity)

    correct_pairs = 0
    total_result_pairs = 0
    total_truth_pairs = 0

    for canonical, entities in result_groups.items():
        entities_in_truth = [e for e in entities if e in ground_truth]
        n = len(entities_in_truth)
        total_result_pairs += n * (n - 1) // 2
        for i, e1 in enumerate(entities_in_truth):
            for e2 in entities_in_truth[i+1:]:
                if ground_truth[e1] == ground_truth[e2]:
                    correct_pairs += 1

    for canonical, entities in truth_groups.items():
        entities_in_result = [e for e in entities if e in result_mapping]
        n = len(entities_in_result)
        total_truth_pairs += n * (n - 1) // 2

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
    }


def main():
    print("="*70)
    print("DEPRECATED DEDUP SYSTEM TEST")
    print("Using embedding similarity + LLM verification")
    print("="*70)

    # Create test entities
    print("\nCreating test entities...")
    entities, ground_truth = create_test_entities(100)
    print(f"  {len(entities)} entities, {len(set(ground_truth.values()))} expected groups")

    # Get services
    print("\nInitializing services...")
    services = get_services()
    embeddings_client = services.embeddings

    # Reset and get manager
    DeferredDeduplicationManager.reset()
    manager = DeferredDeduplicationManager.get_instance()

    # Override LLM to use Claude Sonnet 4.5 (faster than Gemini)
    manager._llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

    # Generate embeddings
    print("\nGenerating embeddings...")
    start_embed = time.time()
    texts = [f"{e['name']}: {e['summary']}" for e in entities]
    entity_embeddings = embeddings_client.embed_documents(texts)
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
    print("\nRunning deduplication (threshold=0.70)...")
    print("-"*70)
    start_dedup = time.time()
    stats = manager.cluster_and_remap(similarity_threshold=0.70)
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
    print(f"  Unique canonicals: {metrics['unique_canonicals']}")
    print(f"  Total time: {embed_time + dedup_time:.1f}s")

    # Show merged groups
    print("\n" + "-"*70)
    print("MERGED GROUPS:")
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
            print(f"    Should be separate: {truths}")
    else:
        print("  None!")

    # Cleanup
    DeferredDeduplicationManager.reset()


if __name__ == "__main__":
    main()
