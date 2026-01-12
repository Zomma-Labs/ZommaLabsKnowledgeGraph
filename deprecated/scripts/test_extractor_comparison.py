"""
Test Script: Extractor Comparison

Compares the OLD extraction pipeline (atomizer → entity_extractor) with the
NEW rearchitected extractor on problematic chunks from EVAL_ISSUES.md.

Target: Chunk alphabet_chunk_0024 containing:
- Nest Labs (merged into Google)
- Chronicle Security (merged into Google Cloud)
- Sidewalk Labs (absorbed into Google)

Usage:
    uv run src/scripts/test_extractor_comparison.py
    uv run src/scripts/test_extractor_comparison.py --verbose
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import asyncio
from pathlib import Path
from typing import Optional


# Target entities we expect to find
TARGET_ENTITIES = {
    "Nest Labs": "merged into Google",
    "Chronicle Security": "merged into Google Cloud",
    "Sidewalk Labs": "absorbed into Google",
}


def load_chunk(chunk_id: str = "alphabet_chunk_0024") -> Optional[dict]:
    """Load a specific chunk from alphabet.jsonl."""
    jsonl_path = Path(__file__).parent.parent / "chunker" / "SAVED" / "alphabet.jsonl"

    if not jsonl_path.exists():
        print(f"ERROR: File not found: {jsonl_path}")
        return None

    with open(jsonl_path, "r") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk.get("chunk_id") == chunk_id:
                return chunk

    print(f"ERROR: Chunk '{chunk_id}' not found in {jsonl_path}")
    return None


def run_old_pipeline(chunk: dict, verbose: bool = False) -> dict:
    """
    Run the OLD pipeline: atomizer → entity_extractor.
    Returns dict with facts and relations.
    """
    from src.agents.atomizer import atomizer
    from src.agents.entity_extractor import EntityExtractor

    chunk_text = chunk["body"]
    header_path = " > ".join(chunk.get("breadcrumbs", []))
    metadata = {
        "chunk_id": chunk["chunk_id"],
        "section_header": header_path,
        "doc_date": None,
    }

    print("\n--- OLD PIPELINE (atomizer → entity_extractor) ---")

    # Step 1: Atomize
    print("Running atomizer...")
    facts = atomizer(chunk_text, metadata)
    print(f"Atomizer produced {len(facts)} facts:")
    for i, fact in enumerate(facts, 1):
        print(f"  {i}. {fact[:100]}..." if len(fact) > 100 else f"  {i}. {fact}")

    # Step 2: Extract entities from each fact
    print("\nRunning entity extractor...")
    extractor = EntityExtractor()
    all_relations = []

    for fact in facts:
        relations = extractor.extract_with_reflexion(fact, chunk_text, header_path)
        all_relations.extend(relations)
        if verbose:
            for rel in relations:
                print(f"    {rel.subject} --[{rel.relationship_description}]--> {rel.object}")

    print(f"\nEntity Extraction Results ({len(all_relations)} relations):")
    for rel in all_relations:
        print(f"  Subject: {rel.subject} ({rel.subject_type})")
        print(f"  Object: {rel.object} ({rel.object_type})")
        print(f"  Relationship: {rel.relationship_description}")
        print()

    return {
        "facts": facts,
        "relations": all_relations,
    }


def run_new_pipeline(chunk: dict, verbose: bool = False) -> dict:
    """
    Run the NEW rearchitected pipeline (V1).
    Returns dict with extracted facts.
    """
    from src.rearchitected.agents.extractor import Extractor

    chunk_text = chunk["body"]
    header_path = " > ".join(chunk.get("breadcrumbs", []))

    print("\n--- V1 PIPELINE (rearchitected extractor) ---")

    extractor = Extractor()
    result = extractor.extract(chunk_text, header_path)

    print(f"Extracted {len(result.facts)} facts:")
    for i, fact in enumerate(result.facts, 1):
        print(f"\n  {i}. Fact: {fact.fact[:80]}..." if len(fact.fact) > 80 else f"\n  {i}. Fact: {fact.fact}")
        print(f"     Subject: {fact.subject} ({fact.subject_type})")
        print(f"     Object: {fact.object} ({fact.object_type})")
        print(f"     Relationship: {fact.relationship}")
        if fact.date_context:
            print(f"     Date: {fact.date_context}")
        if fact.topics:
            print(f"     Topics: {', '.join(fact.topics)}")

    return {
        "facts": result.facts,
    }


def run_v2_pipeline(chunk: dict, verbose: bool = False) -> dict:
    """
    Run the V2 chain-of-thought extractor.
    Returns dict with entities and facts.
    """
    from src.rearchitected.agents.extractor_v2 import ExtractorV2

    chunk_text = chunk["body"]
    header_path = " > ".join(chunk.get("breadcrumbs", []))

    print("\n--- V2 PIPELINE (chain-of-thought extractor) ---")

    extractor = ExtractorV2()
    result = extractor.extract(chunk_text, header_path)

    # Show enumerated entities (the "thinking" step)
    print(f"\nStep 1 - Enumerated {len(result.entities)} entities:")
    for i, entity in enumerate(result.entities, 1):
        print(f"  {i}. {entity.name} ({entity.entity_type})")
        if verbose and entity.summary:
            print(f"     Summary: {entity.summary}")

    # Show facts derived from entities
    print(f"\nStep 2 - Generated {len(result.facts)} relationships:")
    for i, fact in enumerate(result.facts, 1):
        print(f"\n  {i}. Fact: {fact.fact[:80]}..." if len(fact.fact) > 80 else f"\n  {i}. Fact: {fact.fact}")
        print(f"     Subject: {fact.subject} ({fact.subject_type})")
        print(f"     Object: {fact.object} ({fact.object_type})")
        print(f"     Relationship: {fact.relationship}")
        if fact.date_context:
            print(f"     Date: {fact.date_context}")
        if fact.topics:
            print(f"     Topics: {', '.join(fact.topics)}")

    return {
        "entities": result.entities,
        "facts": result.facts,
    }


def check_target_entities(pipeline_name: str, data: dict) -> dict:
    """
    Check if target entities were found and with what relationship.
    Returns dict mapping entity name -> (found, relationship_type)
    """
    results = {}

    if pipeline_name == "old":
        relations = data.get("relations", [])
        for entity_name in TARGET_ENTITIES:
            found = False
            rel_type = None
            for rel in relations:
                # Check both subject and object
                obj_name = rel.object or ""
                if entity_name.lower() in rel.subject.lower() or entity_name.lower() in obj_name.lower():
                    found = True
                    rel_type = rel.relationship_description
                    break
            results[entity_name] = (found, rel_type)

    elif pipeline_name in ("new", "v1", "v2"):
        facts = data.get("facts", [])
        for entity_name in TARGET_ENTITIES:
            found = False
            rel_type = None
            for fact in facts:
                # Check subject and object
                if entity_name.lower() in fact.subject.lower() or entity_name.lower() in fact.object.lower():
                    found = True
                    rel_type = fact.relationship
                    break
            results[entity_name] = (found, rel_type)

    return results


def check_entity_enumeration(data: dict) -> dict:
    """
    Check if target entities were enumerated in V2's entity list.
    Returns dict mapping entity name -> (found, entity_type)
    """
    results = {}
    entities = data.get("entities", [])

    for entity_name in TARGET_ENTITIES:
        found = False
        entity_type = None
        for entity in entities:
            if entity_name.lower() in entity.name.lower():
                found = True
                entity_type = entity.entity_type
                break
        results[entity_name] = (found, entity_type)

    return results


def print_comparison_summary(old_results: dict, v1_results: dict, v2_results: dict, v2_enum: dict):
    """Print side-by-side comparison of entity extraction results."""
    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)

    # Header
    print(f"\n{'Entity':<20} | {'Old Pipeline':<18} | {'V1 Extractor':<18} | {'V2 (CoT)':<18}")
    print("-" * 20 + "-+-" + "-" * 18 + "-+-" + "-" * 18 + "-+-" + "-" * 18)

    for entity_name, expected_rel in TARGET_ENTITIES.items():
        old_found, old_rel = old_results.get(entity_name, (False, None))
        v1_found, v1_rel = v1_results.get(entity_name, (False, None))
        v2_found, v2_rel = v2_results.get(entity_name, (False, None))

        # Format results
        old_str = (old_rel[:16] if old_rel and len(old_rel) > 16 else old_rel) if old_found else "MISSING"
        v1_str = (v1_rel[:16] if v1_rel and len(v1_rel) > 16 else v1_rel) if v1_found else "MISSING"
        v2_str = (v2_rel[:16] if v2_rel and len(v2_rel) > 16 else v2_rel) if v2_found else "MISSING"

        print(f"{entity_name:<20} | {old_str:<18} | {v1_str:<18} | {v2_str:<18}")

    # V2 Entity Enumeration Check
    print("\n" + "-" * 90)
    print("V2 ENTITY ENUMERATION (Chain-of-Thought Step 1)")
    print("-" * 90)
    for entity_name in TARGET_ENTITIES:
        found, entity_type = v2_enum.get(entity_name, (False, None))
        status = f"ENUMERATED ({entity_type})" if found else "NOT ENUMERATED"
        symbol = "\u2713" if found else "\u2717"
        print(f"  {symbol} {entity_name}: {status}")

    print()


def main(verbose: bool = False, skip_old: bool = False):
    """Main test runner."""
    print("=" * 90)
    print("EXTRACTOR COMPARISON TEST (Old vs V1 vs V2)")
    print("=" * 90)

    # Load the problematic chunk
    chunk = load_chunk("alphabet_chunk_0024")
    if not chunk:
        return

    print(f"\nLoaded chunk: {chunk['chunk_id']}")
    print(f"Header: {' > '.join(chunk.get('breadcrumbs', []))}")
    print(f"Body preview: {chunk['body'][:150]}...")

    # Run old pipeline (optional - it's slow)
    if not skip_old:
        old_data = run_old_pipeline(chunk, verbose)
        old_check = check_target_entities("old", old_data)

        print("\nTarget Entity Check (Old Pipeline):")
        for entity_name, (found, rel_type) in old_check.items():
            status = "FOUND" if found else "MISSING"
            rel_info = f" (relationship: {rel_type})" if rel_type else ""
            symbol = "\u2713" if found else "\u2717"
            print(f"  {symbol} {entity_name}: {status}{rel_info}")
    else:
        old_check = {name: (False, None) for name in TARGET_ENTITIES}
        print("\n--- OLD PIPELINE skipped (use --include-old to run) ---")

    # Run V1 pipeline
    v1_data = run_new_pipeline(chunk, verbose)
    v1_check = check_target_entities("v1", v1_data)

    print("\nTarget Entity Check (V1 Pipeline):")
    for entity_name, (found, rel_type) in v1_check.items():
        status = "FOUND" if found else "MISSING"
        rel_info = f" (relationship: {rel_type})" if rel_type else ""
        symbol = "\u2713" if found else "\u2717"
        print(f"  {symbol} {entity_name}: {status}{rel_info}")

    # Run V2 pipeline (chain-of-thought)
    v2_data = run_v2_pipeline(chunk, verbose)
    v2_check = check_target_entities("v2", v2_data)
    v2_enum = check_entity_enumeration(v2_data)

    print("\nTarget Entity Check (V2 Pipeline):")
    for entity_name, (found, rel_type) in v2_check.items():
        status = "FOUND" if found else "MISSING"
        rel_info = f" (relationship: {rel_type})" if rel_type else ""
        symbol = "\u2713" if found else "\u2717"
        print(f"  {symbol} {entity_name}: {status}{rel_info}")

    # Print comparison
    print_comparison_summary(old_check, v1_check, v2_check, v2_enum)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare old vs V1 vs V2 extractor on problematic chunks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed extraction output")
    parser.add_argument("--skip-old", action="store_true", help="Skip the slow old pipeline")
    parser.add_argument("--include-old", action="store_true", help="Include the old pipeline (default: skip)")
    args = parser.parse_args()

    # Default to skipping old pipeline (it's slow)
    skip_old = not args.include_old if args.include_old else (not args.skip_old if not args.skip_old else True)
    # Simpler: skip old by default unless --include-old
    skip_old = not args.include_old

    main(verbose=args.verbose, skip_old=skip_old)
