#!/usr/bin/env python3
"""
Compare extraction quality across GPT models (5.1 vs 5.2).
"""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from src.schemas.extraction import ChainOfThoughtResult
from src.agents.extractor_v2 import (
    COT_EXTRACTION_SYSTEM_PROMPT,
    COT_EXTRACTION_USER_PROMPT,
)

# Test chunks from Beige Book
TEST_CHUNKS = [
    {
        "id": "beige_summary",
        "header_path": "Beige Book > Summary of Economic Activity",
        "text": """Summary of Economic Activity
Economic activity ticked up overall, led by modest increases in consumer spending and manufacturing sales. Commercial real estate activity increased slightly, beating expectations. Software and IT services firms reported small increases in revenues and strong demand. Home sales were flat, while housing inventory increased notably. Employment was unchanged, but layoffs became more common, and wage increases remained modest. Prices rose at a moderate pace, as contacts mentioned tariff-related and other cost pressures. The outlook among business contacts brightened somewhat overall, with most now either neutral or cautiously optimistic. Non-business contacts, in contrast, expressed growing concerns about the economic security of lower-income households."""
    },
    {
        "id": "labor_markets",
        "header_path": "Beige Book > Labor Markets",
        "text": """Labor Markets
Employment levels were largely stable in recent weeks, and demand for labor was generally muted across Districts and sectors. In most Districts, more employers reported lowering head counts through layoffs and attrition, with contacts citing weaker demand, elevated economic uncertainty, and, in some cases, increased investment in artificial intelligence technologies. Employers that hired preferred part-time workers over offering full-time employment opportunities. Nevertheless, labor supply tightened in several Districts due to recent changes to immigration policies. Wages grew across all reporting Districts, generally at a modest to moderate pace, and labor cost pressures intensified in recent weeks due to outsized increases in employer-sponsored health insurance expenses."""
    },
    {
        "id": "prices",
        "header_path": "Beige Book > Prices",
        "text": """Prices
Prices rose further during the reporting period. Several District reports indicated that input costs increased, with contacts frequently citing rising prices for insurance, health care, and technology solutions. Tariff-induced input cost increases were reported across many Districts, but the extent of those higher costs passing through to final prices varied. Some firms facing tariff-induced cost pressures kept their selling prices largely unchanged to preserve market share and in response to pushback from price-sensitive clients. However, there were also reports of firms in manufacturing and retail trades fully passing higher import costs along to their customers. Waning demand in some markets reportedly pushed prices down for some materials, such as steel in the Sixth District and lumber in the Twelfth District."""
    }
]

MODELS = ["gpt-5.1", "gpt-5.2"]


def extract_with_model(model_name: str, chunk: dict) -> dict:
    """Run extraction with a specific model."""
    print(f"\n  Testing {model_name}...")

    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_extractor = llm.with_structured_output(ChainOfThoughtResult, include_raw=True)

    user_prompt = COT_EXTRACTION_USER_PROMPT.format(
        header_path=chunk["header_path"],
        document_date="2025-10-15",
        chunk_text=chunk["text"]
    )
    messages = [
        ("system", COT_EXTRACTION_SYSTEM_PROMPT),
        ("human", user_prompt)
    ]

    start = time.time()
    try:
        response = structured_extractor.invoke(messages)
        elapsed = time.time() - start

        if response.get("parsing_error"):
            return {"error": response["parsing_error"], "time": elapsed, "model": model_name}

        parsed = response.get("parsed")
        if parsed is None:
            return {"error": "Parsed result is None", "time": elapsed, "model": model_name}

        return {
            "model": model_name,
            "time": elapsed,
            "entities": [{"name": e.name, "type": e.entity_type, "summary": e.summary} for e in parsed.entities],
            "facts": [{"fact": f.fact, "subject": f.subject, "object": f.object, "rel": f.relationship} for f in parsed.facts],
            "entity_count": len(parsed.entities),
            "fact_count": len(parsed.facts)
        }
    except Exception as e:
        return {"error": str(e), "time": time.time() - start, "model": model_name}


def compare_results(results: list[dict], chunk_id: str):
    """Print comparison of results."""
    print(f"\n{'='*80}")
    print(f"COMPARISON FOR: {chunk_id}")
    print('='*80)

    # Summary table
    print(f"\n{'Model':<15} {'Entities':<10} {'Facts':<10} {'Time (s)':<10}")
    print("-" * 45)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<15} ERROR: {r['error'][:30]}...")
        else:
            print(f"{r['model']:<15} {r['entity_count']:<10} {r['fact_count']:<10} {r['time']:.2f}")

    # Entity comparison
    print(f"\n--- ENTITIES BY MODEL ---")
    for r in results:
        if "error" not in r:
            print(f"\n{r['model']}:")
            for e in r["entities"]:
                print(f"  - {e['name']} ({e['type']})")

    # Fact comparison
    print(f"\n--- FACTS BY MODEL ---")
    for r in results:
        if "error" not in r:
            print(f"\n{r['model']} ({r['fact_count']} facts):")
            for i, f in enumerate(r["facts"][:10], 1):  # Limit to first 10
                print(f"  {i}. {f['subject']} --[{f['rel']}]--> {f['object']}")
                print(f"     \"{f['fact'][:100]}...\"" if len(f['fact']) > 100 else f"     \"{f['fact']}\"")


def main():
    print("=" * 80)
    print("GPT MODEL COMPARISON FOR ENTITY EXTRACTION")
    print("=" * 80)
    print(f"\nModels: {', '.join(MODELS)}")
    print(f"Test chunks: {len(TEST_CHUNKS)}")

    all_results = {}

    for chunk in TEST_CHUNKS:
        print(f"\n\n{'#'*80}")
        print(f"CHUNK: {chunk['id']}")
        print(f"{'#'*80}")
        print(f"\nText preview: {chunk['text'][:200]}...")

        results = []
        for model in MODELS:
            result = extract_with_model(model, chunk)
            results.append(result)

        compare_results(results, chunk["id"])
        all_results[chunk["id"]] = results

    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for model in MODELS:
        total_entities = 0
        total_facts = 0
        total_time = 0
        errors = 0

        for chunk_id, results in all_results.items():
            for r in results:
                if r["model"] == model:
                    if "error" in r:
                        errors += 1
                    else:
                        total_entities += r["entity_count"]
                        total_facts += r["fact_count"]
                    total_time += r["time"]

        print(f"\n{model}:")
        print(f"  Total entities: {total_entities}")
        print(f"  Total facts: {total_facts}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
