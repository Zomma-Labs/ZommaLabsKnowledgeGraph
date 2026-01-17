#!/usr/bin/env python3
"""
Compare entity deduplication quality across GPT models (5.1 vs 5.2).
"""

import time
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# Same schema as entity_dedup.py
class DistinctEntity(BaseModel):
    canonical_name: str = Field(description="The best/most complete name for this entity")
    member_indices: List[int] = Field(description="List of indices (0-based) from the input that refer to this same entity")
    merged_summary: str = Field(description="Combined summary incorporating information from all members")


class DeduplicationResult(BaseModel):
    distinct_entities: List[DistinctEntity] = Field(description="List of distinct real-world entities")


# Test cases: groups of entities that may or may not be duplicates
TEST_CASES = [
    {
        "name": "Company name variants",
        "entities": [
            {"name": "Apple Inc.", "summary": "Technology company that makes iPhones and Macs"},
            {"name": "Apple", "summary": "Consumer electronics company headquartered in Cupertino"},
            {"name": "AAPL", "summary": "Stock ticker for Apple on NASDAQ"},
            {"name": "Apple Computer", "summary": "Former name of Apple Inc."},
        ],
        "expected_groups": 1,  # All same company
    },
    {
        "name": "Parent vs Subsidiary",
        "entities": [
            {"name": "Alphabet Inc.", "summary": "Parent company of Google, headquartered in Mountain View"},
            {"name": "Google", "summary": "Search engine and technology company owned by Alphabet"},
            {"name": "YouTube", "summary": "Video streaming platform owned by Google"},
            {"name": "DeepMind", "summary": "AI research lab owned by Alphabet"},
        ],
        "expected_groups": 4,  # All different entities
    },
    {
        "name": "Person name variants",
        "entities": [
            {"name": "Tim Cook", "summary": "CEO of Apple Inc."},
            {"name": "Timothy D. Cook", "summary": "Chief Executive Officer of Apple"},
            {"name": "Timothy Cook", "summary": "Apple's CEO since 2011"},
            {"name": "Cook", "summary": "Apple executive"},
        ],
        "expected_groups": 1,  # All same person
    },
    {
        "name": "Similar but different companies",
        "entities": [
            {"name": "Goldman Sachs", "summary": "Investment bank headquartered in New York"},
            {"name": "Morgan Stanley", "summary": "Investment bank and financial services company"},
            {"name": "JPMorgan Chase", "summary": "Largest bank in the United States"},
            {"name": "JP Morgan", "summary": "Major investment bank, part of JPMorgan Chase"},
        ],
        "expected_groups": 3,  # GS, MS, and JPM/JP Morgan (last two are same)
    },
    {
        "name": "Federal Reserve variants",
        "entities": [
            {"name": "Federal Reserve", "summary": "Central bank of the United States"},
            {"name": "The Fed", "summary": "US central banking system"},
            {"name": "Fed", "summary": "Federal Reserve System"},
            {"name": "Federal Reserve Bank of New York", "summary": "One of 12 regional Federal Reserve Banks"},
            {"name": "FOMC", "summary": "Federal Open Market Committee, sets monetary policy"},
        ],
        "expected_groups": 3,  # Fed/Federal Reserve/The Fed are same; NY Fed is different; FOMC is different
    },
    {
        "name": "Mixed entities from Beige Book",
        "entities": [
            {"name": "Employment", "summary": "Labor market conditions and job levels"},
            {"name": "employment levels", "summary": "Number of jobs in the economy"},
            {"name": "Labor Markets", "summary": "Overall employment and wage conditions"},
            {"name": "Wages", "summary": "Worker compensation levels"},
            {"name": "wage increases", "summary": "Growth in worker pay"},
            {"name": "layoffs", "summary": "Job cuts by employers"},
        ],
        "expected_groups": 4,  # Employment/employment levels; Labor Markets; Wages/wage increases; layoffs
    },
]

MODELS = ["gpt-5.1", "gpt-5.2"]


def build_dedup_prompt(entities: list) -> str:
    """Build the deduplication prompt."""
    entity_list = "\n".join([
        f"{i}. Name: \"{e['name']}\"\n   Definition: {e['summary']}"
        for i, e in enumerate(entities)
    ])

    return f"""You are deduplicating entities extracted from financial documents for a knowledge graph.

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
- General concept ≠ specific metric: "Labor Markets" ≠ "Employment"

DECISION TEST: If you swapped one name for the other in a sentence, would the meaning stay exactly the same?

IMPORTANT:
- Every input index (0 to {len(entities) - 1}) must appear in exactly ONE group
- When in doubt, keep entities separate

Return the distinct real-world entities found."""


def run_dedup_with_model(model_name: str, entities: list) -> dict:
    """Run deduplication with a specific model."""
    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(DeduplicationResult)

    prompt = build_dedup_prompt(entities)

    start = time.time()
    try:
        result = structured_llm.invoke([("human", prompt)])
        elapsed = time.time() - start

        return {
            "model": model_name,
            "time": elapsed,
            "groups": len(result.distinct_entities),
            "details": [
                {
                    "canonical": de.canonical_name,
                    "members": de.member_indices,
                    "member_names": [entities[i]["name"] for i in de.member_indices]
                }
                for de in result.distinct_entities
            ]
        }
    except Exception as e:
        return {"model": model_name, "error": str(e), "time": time.time() - start}


def main():
    print("=" * 80)
    print("ENTITY DEDUPLICATION MODEL COMPARISON")
    print("=" * 80)
    print(f"\nModels: {', '.join(MODELS)}")
    print(f"Test cases: {len(TEST_CASES)}")

    all_results = []

    for test in TEST_CASES:
        print(f"\n\n{'#'*80}")
        print(f"TEST: {test['name']}")
        print(f"Expected groups: {test['expected_groups']}")
        print(f"{'#'*80}")

        print("\nEntities:")
        for i, e in enumerate(test["entities"]):
            print(f"  {i}. {e['name']}")

        results = {}
        for model in MODELS:
            print(f"\n  Testing {model}...")
            result = run_dedup_with_model(model, test["entities"])
            results[model] = result

            if "error" in result:
                print(f"    ERROR: {result['error']}")
            else:
                print(f"    Groups found: {result['groups']} (expected: {test['expected_groups']})")
                print(f"    Time: {result['time']:.2f}s")

        # Compare results
        print(f"\n  --- COMPARISON ---")
        print(f"  {'Model':<12} {'Groups':<8} {'Correct?':<10} {'Time':<8}")
        print(f"  {'-'*38}")

        for model in MODELS:
            r = results[model]
            if "error" in r:
                print(f"  {model:<12} ERROR")
            else:
                correct = "✓" if r["groups"] == test["expected_groups"] else "✗"
                print(f"  {model:<12} {r['groups']:<8} {correct:<10} {r['time']:.2f}s")

        # Show groupings
        print(f"\n  --- GROUPINGS ---")
        for model in MODELS:
            r = results[model]
            if "error" not in r:
                print(f"\n  {model}:")
                for g in r["details"]:
                    members = ", ".join(g["member_names"])
                    print(f"    → {g['canonical']}: [{members}]")

        all_results.append({"test": test["name"], "expected": test["expected_groups"], "results": results})

    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for model in MODELS:
        correct = 0
        total_time = 0

        for ar in all_results:
            r = ar["results"].get(model, {})
            if "error" not in r:
                if r["groups"] == ar["expected"]:
                    correct += 1
                total_time += r["time"]

        print(f"\n{model}:")
        print(f"  Correct: {correct}/{len(TEST_CASES)}")
        print(f"  Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
