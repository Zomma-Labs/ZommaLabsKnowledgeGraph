"""
Test entity resolution with OpenAI embeddings.

Tests how vague entities like "districts" resolve to specific entities
like the 12 Federal Reserve districts.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import numpy as np
from src.util.llm_client import get_embeddings

# The 12 Federal Reserve Districts
FED_DISTRICTS = [
    "Boston Federal Reserve District",
    "New York Federal Reserve District",
    "Philadelphia Federal Reserve District",
    "Cleveland Federal Reserve District",
    "Richmond Federal Reserve District",
    "Atlanta Federal Reserve District",
    "Chicago Federal Reserve District",
    "St. Louis Federal Reserve District",
    "Minneapolis Federal Reserve District",
    "Kansas City Federal Reserve District",
    "Dallas Federal Reserve District",
    "San Francisco Federal Reserve District",
]

# Vague and specific queries to test
TEST_QUERIES = [
    # Vague queries
    "districts",
    "the districts",
    "Federal Reserve districts",
    "Fed districts",
    "regional districts",

    # Specific queries
    "Boston district",
    "Boston Fed",
    "Boston Federal Reserve",
    "New York district",
    "Chicago district",

    # Related but different
    "economic regions",
    "banking regions",
    "US regions",

    # Unrelated
    "inflation",
    "unemployment",
    "interest rates",
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def main():
    print("=" * 80)
    print("OpenAI Embeddings Entity Resolution Test")
    print("Model: text-embedding-3-large (3072 dimensions)")
    print("=" * 80)

    embeddings = get_embeddings()

    # Embed all Fed districts
    print("\n1. Embedding 12 Federal Reserve Districts...")
    district_embeddings = embeddings.embed_documents(FED_DISTRICTS)
    print(f"   Embedding dimension: {len(district_embeddings[0])}")

    # Embed test queries
    print("\n2. Embedding test queries...")
    query_embeddings = embeddings.embed_documents(TEST_QUERIES)

    # Calculate similarities
    print("\n3. Calculating similarities...")
    print("\n" + "=" * 80)
    print("RESULTS: Query -> Top 3 District Matches")
    print("=" * 80)

    for i, query in enumerate(TEST_QUERIES):
        query_emb = query_embeddings[i]

        # Calculate similarity to each district
        similarities = []
        for j, district in enumerate(FED_DISTRICTS):
            sim = cosine_similarity(query_emb, district_embeddings[j])
            similarities.append((district, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Print results
        print(f"\nQuery: \"{query}\"")
        top_score = similarities[0][1]
        second_score = similarities[1][1]
        gap = top_score - second_score

        for district, sim in similarities[:3]:
            short_name = district.replace(" Federal Reserve District", "")
            print(f"  {sim:.4f} - {short_name}")

        # Analysis
        avg_sim = np.mean([s[1] for s in similarities])
        print(f"  [Avg: {avg_sim:.4f}, Top-2 Gap: {gap:.4f}]")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY: Score Distribution Analysis")
    print("=" * 80)

    # Group queries by type
    vague_queries = TEST_QUERIES[:5]
    specific_queries = TEST_QUERIES[5:10]
    related_queries = TEST_QUERIES[10:13]
    unrelated_queries = TEST_QUERIES[13:]

    for group_name, group_queries in [
        ("Vague (districts, Fed districts)", vague_queries),
        ("Specific (Boston district, etc)", specific_queries),
        ("Related (economic regions, etc)", related_queries),
        ("Unrelated (inflation, etc)", unrelated_queries),
    ]:
        print(f"\n{group_name}:")

        group_scores = []
        for query in group_queries:
            idx = TEST_QUERIES.index(query)
            query_emb = query_embeddings[idx]

            sims = [cosine_similarity(query_emb, district_embeddings[j])
                    for j in range(len(FED_DISTRICTS))]

            top_score = max(sims)
            avg_score = np.mean(sims)
            group_scores.append((query, top_score, avg_score))

        for query, top, avg in group_scores:
            print(f"  {query[:30]:30s} top={top:.4f} avg={avg:.4f}")

    # Inter-district similarity (are districts distinguishable from each other?)
    print("\n" + "=" * 80)
    print("DISTRICT DISTINCTIVENESS: Inter-district similarities")
    print("=" * 80)

    inter_sims = []
    for i in range(len(FED_DISTRICTS)):
        for j in range(i + 1, len(FED_DISTRICTS)):
            sim = cosine_similarity(district_embeddings[i], district_embeddings[j])
            inter_sims.append(sim)

    print(f"\nInter-district similarity stats:")
    print(f"  Min:  {min(inter_sims):.4f}")
    print(f"  Max:  {max(inter_sims):.4f}")
    print(f"  Mean: {np.mean(inter_sims):.4f}")
    print(f"  Std:  {np.std(inter_sims):.4f}")

    # Show most and least similar district pairs
    pairs = []
    for i in range(len(FED_DISTRICTS)):
        for j in range(i + 1, len(FED_DISTRICTS)):
            sim = cosine_similarity(district_embeddings[i], district_embeddings[j])
            pairs.append((FED_DISTRICTS[i], FED_DISTRICTS[j], sim))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\nMost similar district pairs:")
    for d1, d2, sim in pairs[:3]:
        d1_short = d1.replace(" Federal Reserve District", "")
        d2_short = d2.replace(" Federal Reserve District", "")
        print(f"  {sim:.4f} - {d1_short} <-> {d2_short}")

    print(f"\nLeast similar district pairs:")
    for d1, d2, sim in pairs[-3:]:
        d1_short = d1.replace(" Federal Reserve District", "")
        d2_short = d2.replace(" Federal Reserve District", "")
        print(f"  {sim:.4f} - {d1_short} <-> {d2_short}")

    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    # Calculate if vague queries have good separation from unrelated
    vague_avg = np.mean([
        max([cosine_similarity(query_embeddings[TEST_QUERIES.index(q)], d)
             for d in district_embeddings])
        for q in vague_queries
    ])

    unrelated_avg = np.mean([
        max([cosine_similarity(query_embeddings[TEST_QUERIES.index(q)], d)
             for d in district_embeddings])
        for q in unrelated_queries
    ])

    print(f"\nVague queries avg top score:     {vague_avg:.4f}")
    print(f"Unrelated queries avg top score: {unrelated_avg:.4f}")
    print(f"Separation gap:                  {vague_avg - unrelated_avg:.4f}")

    if vague_avg - unrelated_avg > 0.15:
        print("\n✅ Good separation - vague district queries score much higher than unrelated")
    else:
        print("\n⚠️  Weak separation - may have trouble distinguishing vague from unrelated")


if __name__ == "__main__":
    asyncio.run(main())
