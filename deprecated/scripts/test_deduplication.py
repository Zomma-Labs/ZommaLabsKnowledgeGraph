"""
Test script for the deferred deduplication algorithm.

Uses real Voyage embeddings and realistic financial document entities.

Usage:
    uv run src/scripts/test_deduplication.py
"""

import os
import sys

# Enable verbose logging for deduplication module
os.environ["VERBOSE"] = "true"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.util.services import get_services
from src.util.deferred_dedup import DeferredDeduplicationManager


def main():
    print("="*70)
    print("DEDUPLICATION TEST - Real Embeddings")
    print("="*70)

    services = get_services()
    embeddings = services.embeddings

    # Reset manager
    DeferredDeduplicationManager.reset()
    manager = DeferredDeduplicationManager.get_instance()

    # Realistic entities from a Beige Book-like document
    # These simulate what would be extracted from different chunks
    test_entities = [
        # ===== Federal Reserve variants (should merge to 1) =====
        ("Federal Reserve", "The Federal Reserve System is the central banking system of the United States"),
        ("The Fed", "The Fed is the central bank of the United States responsible for monetary policy"),
        ("Federal Reserve System", "Central bank of the US that conducts monetary policy"),

        # ===== Dallas Fed variants (should merge to 1, but SEPARATE from Federal Reserve) =====
        ("Dallas Fed", "Federal Reserve Bank of Dallas serving the Eleventh Federal Reserve District"),
        ("Federal Reserve Bank Of Dallas", "One of 12 regional Federal Reserve Banks, headquartered in Dallas, Texas"),
        ("Dallas Federal Reserve", "Regional Fed bank covering Texas, northern Louisiana, and southern New Mexico"),

        # ===== Manufacturing sector (should merge) =====
        ("Manufacturing", "The manufacturing sector produces goods through labor and machinery"),
        ("Manufacturing Sector", "Industrial sector involving production of finished goods"),
        ("Manufacturing Activity", "Economic activity related to the production of goods"),

        # ===== Inflation variants (should merge) =====
        ("Inflation", "The rate at which the general level of prices for goods and services rises"),
        ("Price Inflation", "Increase in the general price level of goods and services over time"),
        ("Inflationary Pressures", "Economic forces that cause prices to rise"),

        # ===== Employment topics (should merge) =====
        ("Employment", "The state of having paid work or the number of people with jobs"),
        ("Labor Market", "The supply and demand for labor where employees provide supply and employers demand"),
        ("Employment Conditions", "The state of job availability and workforce participation"),

        # ===== Interest rates (should merge) =====
        ("Interest Rates", "The cost of borrowing money, expressed as a percentage"),
        ("Rates", "Short for interest rates, the price of borrowing"),

        # ===== Distinct entities (should stay separate) =====
        ("Jerome Powell", "Chair of the Federal Reserve Board since February 2018"),
        ("Lorie Logan", "President and CEO of the Federal Reserve Bank of Dallas since 2022"),
        ("Texas", "State in the South Central region of the United States"),
        ("Houston", "Most populous city in Texas and fourth-most populous in the United States"),

        # ===== Tricky cases =====
        ("Economic Activity", "The production, distribution, and consumption of goods and services"),
        ("Economic Growth", "Increase in the production of economic goods and services over time"),
        ("Consumer Spending", "Total money spent by consumers on goods and services"),
        ("Retail Sales", "Sales of goods to consumers through retail channels"),
    ]

    print(f"\n📝 Test entities: {len(test_entities)}")
    print("-"*70)
    for name, summary in test_entities:
        print(f"  • {name}")

    # Generate real embeddings
    print(f"\n🔄 Generating embeddings with voyage-finance-2...")
    texts = [f"{name}: {summary}" for name, summary in test_entities]
    entity_embeddings = embeddings.embed_documents(texts)
    print(f"   Generated {len(entity_embeddings)} embeddings (dim={len(entity_embeddings[0])})")

    # Register entities
    print(f"\n📥 Registering entities...")
    for i, ((name, summary), emb) in enumerate(zip(test_entities, entity_embeddings)):
        manager.register_entity(
            uuid=f"entity-{i:03d}",
            name=name,
            node_type="Entity",
            summary=summary,
            embedding=emb,
            group_id="beige-book-test"
        )

    # Run deduplication
    print(f"\n🔍 Running deduplication (threshold=0.70)...")
    print("-"*70)
    stats = manager.cluster_and_remap(similarity_threshold=0.70)

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"   Components with potential duplicates: {stats['components_found']}")
    print(f"   LLM calls made: {stats['llm_calls']}")
    print(f"   Distinct entities: {stats['distinct_entities']}")
    print(f"   Duplicates merged: {stats['duplicates_merged']}")

    # Show what was merged
    print("\n" + "-"*70)
    print("UUID REMAPPING (what got merged):")
    print("-"*70)

    merged_groups = {}  # canonical_uuid -> list of original names
    for orig_uuid, canonical_uuid in manager._uuid_remap.items():
        orig_entity = manager._pending_entities[orig_uuid]
        canonical_entity = manager._pending_entities[canonical_uuid]

        if canonical_uuid not in merged_groups:
            merged_groups[canonical_uuid] = {
                "canonical_name": canonical_entity.name,
                "members": [canonical_entity.name]
            }
        merged_groups[canonical_uuid]["members"].append(orig_entity.name)

    for canonical_uuid, group in merged_groups.items():
        print(f"\n  ✅ {group['canonical_name']}")
        for member in group['members']:
            if member != group['canonical_name']:
                print(f"     ← {member}")

    # Show canonical entities (what will be written to Neo4j)
    print("\n" + "-"*70)
    print("CANONICAL ENTITIES (will be written to Neo4j):")
    print("-"*70)

    canonical_entities = [
        e for uuid, e in manager._pending_entities.items()
        if uuid not in manager._uuid_remap
    ]

    for e in sorted(canonical_entities, key=lambda x: x.name):
        print(f"  • {e.name}")
        print(f"    {e.summary[:80]}...")

    # Verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    def check_merged(entity_names: list, expected_canonical: str) -> bool:
        """Check if all entity names map to the same canonical."""
        uuids = []
        for i, (name, _) in enumerate(test_entities):
            if name in entity_names:
                uuids.append(f"entity-{i:03d}")

        canonical_uuids = set(manager.get_remapped_uuid(u) for u in uuids)
        return len(canonical_uuids) == 1

    def check_separate(entity_names: list) -> bool:
        """Check that these entities are NOT merged together."""
        uuids = []
        for i, (name, _) in enumerate(test_entities):
            if name in entity_names:
                uuids.append(f"entity-{i:03d}")

        canonical_uuids = set(manager.get_remapped_uuid(u) for u in uuids)
        return len(canonical_uuids) == len(uuids)

    checks = [
        ("Federal Reserve variants merged",
         check_merged(["Federal Reserve", "The Fed", "Federal Reserve System"], "Federal Reserve")),

        ("Dallas Fed variants merged",
         check_merged(["Dallas Fed", "Federal Reserve Bank Of Dallas", "Dallas Federal Reserve"], "Dallas Fed")),

        ("Federal Reserve ≠ Dallas Fed (separate)",
         check_separate(["Federal Reserve", "Dallas Fed"])),

        ("Manufacturing variants merged",
         check_merged(["Manufacturing", "Manufacturing Sector", "Manufacturing Activity"], "Manufacturing")),

        ("Inflation variants merged",
         check_merged(["Inflation", "Price Inflation", "Inflationary Pressures"], "Inflation")),

        ("People remain separate",
         check_separate(["Jerome Powell", "Lorie Logan"])),

        ("Locations remain separate",
         check_separate(["Texas", "Houston"])),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {check_name}")
        if not result:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review the merges above")
    print("="*70)

    # Cleanup
    DeferredDeduplicationManager.reset()


if __name__ == "__main__":
    main()
