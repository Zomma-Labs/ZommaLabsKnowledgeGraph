"""
Test deduplication with a wide range of tech companies.

Tests the algorithm's ability to:
1. Merge variants of the same company
2. Keep different companies separate (even if embeddings are similar)

Usage:
    uv run src/scripts/test_dedup_tech.py
"""

import os
import sys

# Enable verbose logging
os.environ["VERBOSE"] = "true"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.util.services import get_services
from src.util.deferred_dedup import DeferredDeduplicationManager


def main():
    print("="*70)
    print("TECH COMPANIES DEDUPLICATION TEST")
    print("="*70)

    services = get_services()
    embeddings = services.embeddings

    # Reset manager
    DeferredDeduplicationManager.reset()
    manager = DeferredDeduplicationManager.get_instance()

    # Tech companies - variants that SHOULD merge, plus distinct companies that should NOT
    test_entities = [
        # ===== Apple variants (should merge to 1) =====
        ("Apple", "Apple Inc. is an American multinational technology company that designs consumer electronics"),
        ("Apple Inc.", "Apple Inc. is a technology company known for iPhone, iPad, and Mac computers"),
        ("Apple Inc", "American tech giant Apple Inc, maker of iPhones and MacBooks"),
        ("AAPL", "Apple Inc. (NASDAQ: AAPL) is a consumer electronics and software company"),

        # ===== Microsoft variants (should merge to 1) =====
        ("Microsoft", "Microsoft Corporation is an American multinational technology company"),
        ("Microsoft Corporation", "Microsoft Corp develops software including Windows and Office"),
        ("MSFT", "Microsoft Corporation (NASDAQ: MSFT) is a software and cloud computing company"),
        ("Microsoft Corp", "Microsoft Corp is the maker of Windows, Azure, and Xbox"),

        # ===== Google/Alphabet variants (should merge to 1) =====
        ("Google", "Google is a technology company known for search, advertising, and cloud services"),
        ("Alphabet", "Alphabet Inc. is the parent company of Google and other subsidiaries"),
        ("Alphabet Inc.", "Alphabet Inc. is a holding company that owns Google, Waymo, and DeepMind"),
        ("GOOGL", "Alphabet Inc. (NASDAQ: GOOGL) is the parent company of Google"),

        # ===== Amazon variants (should merge to 1) =====
        ("Amazon", "Amazon.com Inc. is an e-commerce and cloud computing company"),
        ("Amazon.com", "Amazon.com is the world's largest online retailer and AWS cloud provider"),
        ("Amazon Inc.", "Amazon Inc. operates e-commerce, cloud computing (AWS), and streaming services"),
        ("AMZN", "Amazon.com Inc. (NASDAQ: AMZN) is an e-commerce and technology company"),

        # ===== Meta/Facebook variants (should merge to 1) =====
        ("Meta", "Meta Platforms Inc. is a social media and technology company"),
        ("Meta Platforms", "Meta Platforms operates Facebook, Instagram, and WhatsApp"),
        ("Facebook", "Facebook, now Meta Platforms, is a social networking company"),
        ("META", "Meta Platforms Inc. (NASDAQ: META) is a social media technology company"),

        # ===== NVIDIA variants (should merge to 1) =====
        ("NVIDIA", "NVIDIA Corporation designs graphics processing units and AI chips"),
        ("Nvidia", "Nvidia is a semiconductor company known for GPUs and AI accelerators"),
        ("NVIDIA Corporation", "NVIDIA Corporation is a chipmaker specializing in graphics and AI computing"),
        ("NVDA", "NVIDIA Corporation (NASDAQ: NVDA) is a semiconductor and AI chip company"),

        # ===== Tesla variants (should merge to 1) =====
        ("Tesla", "Tesla Inc. is an electric vehicle and clean energy company"),
        ("Tesla Inc.", "Tesla Inc. manufactures electric cars, batteries, and solar products"),
        ("Tesla Motors", "Tesla Motors, now Tesla Inc., is an electric vehicle manufacturer"),
        ("TSLA", "Tesla Inc. (NASDAQ: TSLA) is an electric vehicle and energy company"),

        # ===== Netflix variants (should merge to 1) =====
        ("Netflix", "Netflix Inc. is a streaming entertainment service company"),
        ("Netflix Inc.", "Netflix Inc. provides subscription streaming of movies and TV shows"),
        ("NFLX", "Netflix Inc. (NASDAQ: NFLX) is a streaming media company"),

        # ===== Smaller distinct companies (should stay separate) =====
        ("Palantir", "Palantir Technologies is a software company specializing in data analytics"),
        ("Snowflake", "Snowflake Inc. is a cloud computing company offering data warehousing"),
        ("Datadog", "Datadog Inc. provides monitoring and analytics for cloud applications"),
        ("CrowdStrike", "CrowdStrike Holdings provides cloud-delivered endpoint protection"),
        ("Salesforce", "Salesforce Inc. is a cloud-based CRM software company"),
        ("Adobe", "Adobe Inc. is a software company known for Photoshop and Creative Cloud"),
        ("Oracle", "Oracle Corporation is a database and enterprise software company"),
        ("IBM", "International Business Machines Corporation provides enterprise technology"),
        ("Intel", "Intel Corporation is a semiconductor chip manufacturer"),
        ("AMD", "Advanced Micro Devices Inc. designs CPUs and GPUs"),
        ("Qualcomm", "Qualcomm Inc. designs wireless telecommunications products and semiconductors"),
        ("Cisco", "Cisco Systems Inc. manufactures networking hardware and software"),
    ]

    print(f"\n📝 Test entities: {len(test_entities)}")
    print("-"*70)

    # Group display
    groups = {
        "Apple": ["Apple", "Apple Inc.", "Apple Inc", "AAPL"],
        "Microsoft": ["Microsoft", "Microsoft Corporation", "MSFT", "Microsoft Corp"],
        "Google/Alphabet": ["Google", "Alphabet", "Alphabet Inc.", "GOOGL"],
        "Amazon": ["Amazon", "Amazon.com", "Amazon Inc.", "AMZN"],
        "Meta": ["Meta", "Meta Platforms", "Facebook", "META"],
        "NVIDIA": ["NVIDIA", "Nvidia", "NVIDIA Corporation", "NVDA"],
        "Tesla": ["Tesla", "Tesla Inc.", "Tesla Motors", "TSLA"],
        "Netflix": ["Netflix", "Netflix Inc.", "NFLX"],
    }

    for group_name, members in groups.items():
        print(f"\n  {group_name} variants:")
        for m in members:
            print(f"    • {m}")

    print(f"\n  Distinct companies (should NOT merge):")
    distinct = ["Palantir", "Snowflake", "Datadog", "CrowdStrike", "Salesforce",
                "Adobe", "Oracle", "IBM", "Intel", "AMD", "Qualcomm", "Cisco"]
    for d in distinct:
        print(f"    • {d}")

    # Generate embeddings
    print(f"\n🔄 Generating embeddings...")
    texts = [f"{name}: {summary}" for name, summary in test_entities]
    entity_embeddings = embeddings.embed_documents(texts)
    print(f"   Generated {len(entity_embeddings)} embeddings (dim={len(entity_embeddings[0])})")

    # Register entities
    print(f"\n📥 Registering entities...")
    for i, ((name, summary), emb) in enumerate(zip(test_entities, entity_embeddings)):
        manager.register_entity(
            uuid=f"tech-{i:03d}",
            name=name,
            node_type="Entity",
            summary=summary,
            embedding=emb,
            group_id="tech-test"
        )

    # Run deduplication
    print(f"\n🔍 Running deduplication (threshold=0.70)...")
    print("-"*70)
    stats = manager.cluster_and_remap(similarity_threshold=0.70)

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"   Total input entities: {len(test_entities)}")
    print(f"   Components with potential duplicates: {stats['components_found']}")
    print(f"   LLM calls made: {stats['llm_calls']}")
    print(f"   Distinct entities: {stats['distinct_entities']}")
    print(f"   Duplicates merged: {stats['duplicates_merged']}")

    # Expected: 8 merged groups + 12 distinct = 20 distinct entities
    expected_distinct = 8 + 12  # 8 company groups + 12 individual companies
    print(f"   Expected distinct: ~{expected_distinct}")

    # Show merge history
    print("\n" + "-"*70)
    print("MERGE HISTORY:")
    print("-"*70)

    for record in manager.get_merge_history():
        print(f"\n  ✅ {record.canonical_name}")
        for name in record.merged_names:
            if name != record.canonical_name:
                print(f"     ← {name}")

    # Verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    def get_canonical_for_name(entity_name: str) -> str:
        """Get the canonical UUID for an entity by name."""
        for i, (name, _) in enumerate(test_entities):
            if name == entity_name:
                uuid = f"tech-{i:03d}"
                return manager.get_remapped_uuid(uuid)
        return None

    def check_merged(entity_names: list) -> bool:
        """Check if all entity names map to the same canonical."""
        canonical_uuids = set(get_canonical_for_name(n) for n in entity_names)
        return len(canonical_uuids) == 1

    def check_separate(entity_names: list) -> bool:
        """Check that these entities are NOT merged together."""
        canonical_uuids = [get_canonical_for_name(n) for n in entity_names]
        return len(set(canonical_uuids)) == len(canonical_uuids)

    checks = [
        # Variants should merge
        ("Apple variants merged", check_merged(["Apple", "Apple Inc.", "Apple Inc", "AAPL"])),
        ("Microsoft variants merged", check_merged(["Microsoft", "Microsoft Corporation", "MSFT", "Microsoft Corp"])),
        ("Google/Alphabet variants merged", check_merged(["Google", "Alphabet", "Alphabet Inc.", "GOOGL"])),
        ("Amazon variants merged", check_merged(["Amazon", "Amazon.com", "Amazon Inc.", "AMZN"])),
        ("Meta/Facebook variants merged", check_merged(["Meta", "Meta Platforms", "Facebook", "META"])),
        ("NVIDIA variants merged", check_merged(["NVIDIA", "Nvidia", "NVIDIA Corporation", "NVDA"])),
        ("Tesla variants merged", check_merged(["Tesla", "Tesla Inc.", "Tesla Motors", "TSLA"])),
        ("Netflix variants merged", check_merged(["Netflix", "Netflix Inc.", "NFLX"])),

        # Big tech should be separate from each other
        ("Apple ≠ Microsoft (separate)", check_separate(["Apple", "Microsoft"])),
        ("Google ≠ Amazon (separate)", check_separate(["Google", "Amazon"])),
        ("Meta ≠ Netflix (separate)", check_separate(["Meta", "Netflix"])),
        ("Tesla ≠ NVIDIA (separate)", check_separate(["Tesla", "NVIDIA"])),

        # Chip companies should be separate
        ("Intel ≠ AMD ≠ NVIDIA (separate)", check_separate(["Intel", "AMD", "NVIDIA"])),
        ("Qualcomm ≠ Intel (separate)", check_separate(["Qualcomm", "Intel"])),

        # Cloud/SaaS companies should be separate
        ("Salesforce ≠ Oracle (separate)", check_separate(["Salesforce", "Oracle"])),
        ("Snowflake ≠ Datadog (separate)", check_separate(["Snowflake", "Datadog"])),

        # Security companies should be separate
        ("CrowdStrike ≠ Palantir (separate)", check_separate(["CrowdStrike", "Palantir"])),
    ]

    passed = 0
    failed = 0
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {check_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print("\n" + "="*70)
    print(f"SUMMARY: {passed}/{len(checks)} checks passed")
    if failed == 0:
        print("🎉 ALL CHECKS PASSED!")
    else:
        print(f"⚠️  {failed} CHECKS FAILED - Review the merges above")
    print("="*70)

    # Cleanup
    DeferredDeduplicationManager.reset()


if __name__ == "__main__":
    main()
