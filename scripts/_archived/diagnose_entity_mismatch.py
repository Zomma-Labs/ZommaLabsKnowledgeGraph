#!/usr/bin/env python3
"""
Diagnose Entity Mismatch in Pipeline
====================================

Checks if fact subjects/objects match enumerated entities during extraction.

Usage:
    uv run scripts/diagnose_entity_mismatch.py
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.extractor_v2 import ExtractorV2


def test_extraction_matching():
    """Run a test extraction and check entity matching."""
    print("=" * 60)
    print("TESTING EXTRACTION ENTITY MATCHING")
    print("=" * 60)

    # Sample Beige Book text
    test_chunk = """
New York
Economic activity grew modestly in the latest reporting period. Labor demand remained modest on balance, though there were pockets of strength among skilled trades. Wage growth was moderate, while price pressures eased slightly. Consumer spending was mixed, with retail sales flat and leisure activity ticking up. Manufacturing activity declined modestly. The real estate sector was mixed.
"""

    extractor = ExtractorV2()
    result = extractor.extract(
        chunk_text=test_chunk,
        header_path="New York > Summary of Economic Activity",
        document_date="2025-10-15"
    )

    print(f"\nExtracted {len(result.entities)} entities:")
    entity_names = set()
    for entity in result.entities:
        print(f"  - {entity.name} ({entity.entity_type})")
        entity_names.add(entity.name)

    print(f"\nExtracted {len(result.facts)} facts:")
    unmatched_subjects = []
    unmatched_objects = []

    for fact in result.facts:
        subject_match = fact.subject in entity_names
        object_match = fact.object in entity_names

        # Check if subject_type/object_type is Topic
        is_topic_subject = fact.subject_type.lower() == "topic"
        is_topic_object = fact.object_type.lower() == "topic"

        print(f"\n  Fact: {fact.fact[:60]}...")
        print(f"    Subject: '{fact.subject}' (type={fact.subject_type}, match={subject_match or is_topic_subject})")
        print(f"    Object: '{fact.object}' (type={fact.object_type}, match={object_match or is_topic_object})")
        print(f"    Relationship: {fact.relationship}")

        if not subject_match and not is_topic_subject:
            unmatched_subjects.append({
                "name": fact.subject,
                "type": fact.subject_type,
                "fact": fact.fact[:50]
            })

        if not object_match and not is_topic_object:
            unmatched_objects.append({
                "name": fact.object,
                "type": fact.object_type,
                "fact": fact.fact[:50]
            })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Entities: {len(entity_names)}")
    print(f"Facts: {len(result.facts)}")
    print(f"Unmatched subjects (not in entities): {len(unmatched_subjects)}")
    print(f"Unmatched objects (not in entities): {len(unmatched_objects)}")

    if unmatched_subjects:
        print("\nUnmatched subjects:")
        for u in unmatched_subjects:
            print(f"  - '{u['name']}' ({u['type']}) from: {u['fact']}...")

    if unmatched_objects:
        print("\nUnmatched objects:")
        for u in unmatched_objects:
            print(f"  - '{u['name']}' ({u['type']}) from: {u['fact']}...")

    # This is the key: if subjects/objects don't appear in entities list,
    # they won't be in entity_lookup and relationships won't be created
    if unmatched_subjects or unmatched_objects:
        print("\n⚠️  ISSUE FOUND: Facts reference entities not in enumeration list!")
        print("   This causes entity_lookup.get() to return None")
        print("   => Subject->Chunk->Object edges are NOT created")


def main():
    test_extraction_matching()


if __name__ == "__main__":
    main()
