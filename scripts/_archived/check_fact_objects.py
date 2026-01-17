#!/usr/bin/env python3
"""Check what objects are in the facts and why they're not linking."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.extractor_v2 import ExtractorV2

# Test with the same type of content that's failing
test_chunk = """
The Beige Book
Summary of Commentary on Current Economic Conditions
by Federal Reserve District

The Beige Book is a Federal Reserve System publication about current economic conditions across the 12 Federal Reserve Districts. It characterizes regional economic conditions and prospects based on a variety of mostly qualitative information.
"""

extractor = ExtractorV2()
result = extractor.extract(
    chunk_text=test_chunk,
    header_path="",
    document_date="2025-01-01"
)

print("=" * 60)
print("EXTRACTION RESULTS")
print("=" * 60)

print(f"\nEntities ({len(result.entities)}):")
for e in result.entities:
    print(f"  - {e.name} (type={e.entity_type})")

print(f"\nFacts ({len(result.facts)}):")
for f in result.facts:
    print(f"\n  Fact: {f.fact[:80]}...")
    print(f"  Subject: '{f.subject}' (type={f.subject_type})")
    print(f"  Object: '{f.object}' (type={f.object_type})")
    print(f"  Relationship: {f.relationship}")

    # Check if subject/object are in entities
    entity_names = {e.name for e in result.entities}
    subj_in_entities = f.subject in entity_names
    obj_in_entities = f.object in entity_names

    print(f"  Subject in entities: {subj_in_entities}")
    print(f"  Object in entities: {obj_in_entities}")

    if not obj_in_entities and f.object_type.lower() != "topic":
        print(f"  ⚠️ Object '{f.object}' is NOT in entities and NOT a Topic!")
