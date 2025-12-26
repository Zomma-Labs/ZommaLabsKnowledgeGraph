"""
Test script for batch entity summary generation.
Tests the new batch_extract_summaries method in GraphEnhancer.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.graph_enhancer import GraphEnhancer
from src.util.services import get_services

def test_batch_summaries():
    """Test batch summary extraction with sample entities and chunk text."""

    # Sample chunk text (financial context)
    chunk_text = """
    Apple Inc. announced record quarterly earnings of $123 billion in Q4 2023.
    The Federal Reserve raised interest rates by 25 basis points to combat inflation.
    The S&P 500 index fell 2.3% following the announcement.
    Goldman Sachs upgraded their price target for Tesla to $250.
    The Consumer Price Index (CPI) rose 3.2% year-over-year.
    Microsoft acquired Activision Blizzard for $69 billion.
    The European Central Bank maintained its current policy stance.
    NVIDIA stock surged 15% after reporting strong AI chip demand.
    JP Morgan Chase reported net interest income of $22.9 billion.
    The unemployment rate remained steady at 3.8%.
    """

    # Sample entity names (mix of companies, institutions, indices, metrics)
    entity_names = [
        "Apple Inc.",
        "Federal Reserve",
        "S&P 500",
        "Goldman Sachs",
        "Tesla",
        "Consumer Price Index",
        "Microsoft",
        "Activision Blizzard",
        "European Central Bank",
        "NVIDIA",
        "JP Morgan Chase",
        "unemployment rate",
        "interest rates",
        "inflation",
        "AI chip demand"
    ]

    print(f"Testing batch summary extraction for {len(entity_names)} entities...")
    print(f"\nChunk text:\n{chunk_text}\n")
    print(f"Entities to summarize:\n{entity_names}\n")

    # Initialize GraphEnhancer
    services = get_services()
    enhancer = GraphEnhancer(services=services)

    # Test with default batch size (15)
    print("=" * 80)
    print("TEST 1: Default batch size (15 entities in 1 batch)")
    print("=" * 80)

    summaries = enhancer.batch_extract_summaries(
        entity_names=entity_names,
        context_text=chunk_text,
        batch_size=15
    )

    print(f"\n✅ Generated summaries for {len(summaries)} entities:")
    for name, summary in summaries.items():
        print(f"  • {name}: {summary}")

    # Test with smaller batch size to verify batching logic
    print("\n" + "=" * 80)
    print("TEST 2: Smaller batch size (5 entities per batch = 3 batches)")
    print("=" * 80)

    summaries_small_batch = enhancer.batch_extract_summaries(
        entity_names=entity_names,
        context_text=chunk_text,
        batch_size=5
    )

    print(f"\n✅ Generated summaries for {len(summaries_small_batch)} entities:")
    for name, summary in summaries_small_batch.items():
        print(f"  • {name}: {summary}")

    # Verify all entities got summaries
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    missing_entities = set(entity_names) - set(summaries.keys())
    if missing_entities:
        print(f"❌ Missing summaries for: {missing_entities}")
    else:
        print("✅ All entities received summaries")

    # Check summary quality (should be brief, 1-sentence)
    print("\n📊 Summary Quality Check:")
    for name, summary in summaries.items():
        word_count = len(summary.split())
        sentence_count = summary.count('.') + summary.count('!') + summary.count('?')
        status = "✅" if sentence_count <= 2 and word_count < 20 else "⚠️"
        print(f"  {status} {name}: {word_count} words, {sentence_count} sentence(s)")

if __name__ == "__main__":
    test_batch_summaries()
