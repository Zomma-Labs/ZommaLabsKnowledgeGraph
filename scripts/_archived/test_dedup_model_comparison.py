#!/usr/bin/env python3
"""
Compare entity deduplication quality: GPT-5.1 vs GPT-5.2
Uses the comprehensive 100+ entity test dataset with ground truth.
"""

import os
import sys
import random
import time
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Import test data
from src.scripts.test_entity_deduplicator import ENTITY_GROUPS, ENTITY_SUMMARIES
from src.util.llm_client import get_dedup_embeddings


# =============================================================================
# Pydantic Models (same as entity_dedup.py)
# =============================================================================

class DistinctEntity(BaseModel):
    canonical_name: str = Field(description="The best/most complete name for this entity")
    member_indices: List[int] = Field(description="List of indices (0-based) from the input that refer to this same entity")
    merged_summary: str = Field(description="Combined summary incorporating information from all members")


class DeduplicationResult(BaseModel):
    distinct_entities: List[DistinctEntity] = Field(description="List of distinct real-world entities")


# =============================================================================
# Union-Find for Connected Components
# =============================================================================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_components(self) -> Dict[int, List[int]]:
        components = defaultdict(list)
        for i in range(len(self.parent)):
            components[self.find(i)].append(i)
        return dict(components)


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
            summary = ENTITY_SUMMARIES.get(name, f"Entity: {name}")
            all_entities.append({"name": name, "summary": summary})
            ground_truth[name] = canonical

    random.seed(42)  # Reproducible
    random.shuffle(all_entities)

    if len(all_entities) > num_entities:
        all_entities = all_entities[:num_entities]
        ground_truth = {e["name"]: ground_truth[e["name"]] for e in all_entities}

    return all_entities, ground_truth


# =============================================================================
# Core Deduplication Logic (copied from entity_dedup.py)
# =============================================================================

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)


def build_connected_components(embeddings: np.ndarray, threshold: float = 0.85) -> Dict[int, List[int]]:
    n = len(embeddings)
    if n < 2:
        return {0: list(range(n))} if n == 1 else {}

    similarity = compute_similarity_matrix(embeddings)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] > threshold:
                uf.union(i, j)

    return uf.get_components()


def llm_dedupe_batch(llm, entities: List[dict]) -> List[DistinctEntity]:
    """Process a batch of entities through LLM to identify distinct entities."""
    structured_llm = llm.with_structured_output(DeduplicationResult)

    entity_list = "\n".join([
        f"{i}. Name: \"{e['name']}\"\n   Definition: {e['summary']}"
        for i, e in enumerate(entities)
    ])

    prompt = f"""You are deduplicating entities extracted from financial documents for a knowledge graph.

TASK: Group entity names that refer to THE SAME real-world entity.
This is NOT grouping similar or related entities - only TRUE duplicates (same entity, different names).

If unsure, do NOT merge. False negatives are better than false positives.

ENTITIES:
{entity_list}

MERGE - same entity, different names:
- Ticker ↔ company: "AAPL" = "Apple Inc." = "Apple"
- Abbreviation ↔ full name: "Fed" = "Federal Reserve" = "The Fed"
- Person name variants: "Tim Cook" = "Timothy D. Cook"

DO NOT MERGE - related but different entities:
- Parent ≠ subsidiary: "Alphabet" ≠ "Google" ≠ "YouTube"
- Competitors: "Goldman Sachs" ≠ "Morgan Stanley"
- Person ≠ their company: "Tim Cook" ≠ "Apple"
- Regional banks ≠ central entity: "Federal Reserve Bank of New York" ≠ "Federal Reserve"

DECISION TEST: If you swapped one name for the other in a sentence, would the meaning stay exactly the same?

IMPORTANT:
- Every input index (0 to {len(entities) - 1}) must appear in exactly ONE group
- When in doubt, keep entities separate

Return the distinct real-world entities found."""

    try:
        result = structured_llm.invoke([("human", prompt)])

        # Validate coverage
        covered = set()
        for de in result.distinct_entities:
            covered.update(de.member_indices)

        expected = set(range(len(entities)))
        missing = expected - covered
        if missing:
            for idx in missing:
                result.distinct_entities.append(DistinctEntity(
                    canonical_name=entities[idx]["name"],
                    member_indices=[idx],
                    merged_summary=entities[idx]["summary"]
                ))

        return result.distinct_entities

    except Exception as e:
        print(f"      LLM error: {e}")
        return [
            DistinctEntity(canonical_name=e["name"], member_indices=[i], merged_summary=e["summary"])
            for i, e in enumerate(entities)
        ]


def run_dedup_with_model(model_name: str, entities: List[dict], embeddings: np.ndarray, threshold: float = 0.85) -> Tuple[dict, dict]:
    """Run full deduplication pipeline with a specific model."""
    llm = ChatOpenAI(model=model_name, temperature=0)

    # Build connected components
    components = build_connected_components(embeddings, threshold)
    multi_components = [c for c in components.values() if len(c) > 1]

    stats = {
        "model": model_name,
        "components": len(multi_components),
        "llm_calls": 0,
        "time": 0,
    }

    # UUID remap (index -> canonical index)
    uuid_remap = {}

    start = time.time()

    for comp_id, member_indices in components.items():
        comp_entities = [entities[i] for i in member_indices]

        if len(comp_entities) == 1:
            continue

        stats["llm_calls"] += 1
        distinct = llm_dedupe_batch(llm, comp_entities)

        # Apply remapping
        for de in distinct:
            if len(de.member_indices) > 1:
                canonical_idx = member_indices[de.member_indices[0]]
                for local_idx in de.member_indices[1:]:
                    global_idx = member_indices[local_idx]
                    uuid_remap[global_idx] = canonical_idx

    stats["time"] = time.time() - start

    # Build result mapping: entity_name -> canonical_name
    result_mapping = {}
    for i, entity in enumerate(entities):
        canonical_idx = uuid_remap.get(i, i)
        result_mapping[entity["name"]] = entities[canonical_idx]["name"]

    return result_mapping, stats


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_dedup(result_mapping: dict, ground_truth: dict) -> dict:
    """Evaluate using precision/recall/F1 on entity pairs."""
    result_groups = defaultdict(list)
    for entity, canonical in result_mapping.items():
        result_groups[canonical].append(entity)

    truth_groups = defaultdict(list)
    for entity, canonical in ground_truth.items():
        truth_groups[canonical].append(entity)

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

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_pairs": correct_pairs,
        "wrong_merges": total_result_pairs - correct_pairs,
        "missed_merges": total_truth_pairs - correct_pairs,
        "unique_canonicals": len(set(result_mapping.values())),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    NUM_ENTITIES = 100
    MODELS = ["gpt-5.1", "gpt-5.2"]
    THRESHOLD = 0.85

    print("=" * 70)
    print("ENTITY DEDUPLICATION MODEL COMPARISON")
    print("=" * 70)
    print(f"Entities: {NUM_ENTITIES}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Similarity threshold: {THRESHOLD}")

    # Create test data
    print(f"\nCreating test entities...")
    entities, ground_truth = create_test_entities(NUM_ENTITIES)
    expected_groups = len(set(ground_truth.values()))
    print(f"  {len(entities)} entities, {expected_groups} expected canonical groups")

    # Generate embeddings (shared across models)
    print(f"\nGenerating embeddings...")
    embeddings_client = get_dedup_embeddings()
    texts = [f"{e['name']}: {e['summary']}" for e in entities]
    start = time.time()
    embeddings = np.array(embeddings_client.embed_documents(texts))
    embed_time = time.time() - start
    print(f"  Generated {len(embeddings)} embeddings in {embed_time:.1f}s")

    # Test each model
    all_results = {}

    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"TESTING: {model}")
        print("=" * 70)

        result_mapping, stats = run_dedup_with_model(model, entities, embeddings, THRESHOLD)
        metrics = evaluate_dedup(result_mapping, ground_truth)

        all_results[model] = {
            "stats": stats,
            "metrics": metrics,
            "mapping": result_mapping,
        }

        print(f"\n  Components found: {stats['components']}")
        print(f"  LLM calls: {stats['llm_calls']}")
        print(f"  Time: {stats['time']:.1f}s")
        print(f"\n  Precision: {metrics['precision']:.1%}")
        print(f"  Recall: {metrics['recall']:.1%}")
        print(f"  F1 Score: {metrics['f1']:.1%}")
        print(f"  Correct pairs: {metrics['correct_pairs']}")
        print(f"  Wrong merges: {metrics['wrong_merges']}")
        print(f"  Missed merges: {metrics['missed_merges']}")
        print(f"  Unique canonicals: {metrics['unique_canonicals']} (expected: {expected_groups})")

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Model':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Time':<10} {'LLM Calls':<10}")
    print("-" * 68)
    for model in MODELS:
        m = all_results[model]["metrics"]
        s = all_results[model]["stats"]
        print(f"{model:<12} {m['precision']:.1%}        {m['recall']:.1%}        {m['f1']:.1%}        {s['time']:.1f}s       {s['llm_calls']}")

    # Show differences
    print("\n" + "-" * 70)
    print("DIFFERENCES IN GROUPING:")
    print("-" * 70)

    map1 = all_results[MODELS[0]]["mapping"]
    map2 = all_results[MODELS[1]]["mapping"]

    diffs = []
    for entity in map1:
        if entity in map2 and map1[entity] != map2[entity]:
            diffs.append((entity, map1[entity], map2[entity]))

    if diffs:
        print(f"Found {len(diffs)} entities mapped differently:")
        for entity, c1, c2 in diffs[:20]:
            print(f"  {entity}:")
            print(f"    {MODELS[0]}: → {c1}")
            print(f"    {MODELS[1]}: → {c2}")
    else:
        print("  No differences! Both models produced identical groupings.")


if __name__ == "__main__":
    main()
