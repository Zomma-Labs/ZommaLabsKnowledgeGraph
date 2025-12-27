"""
Full Pipeline A/B Test: Atomizer → Entity Extractor
====================================================

Tests all 4 combinations:
- Atomizer (lite) → Entity Extractor (lite)
- Atomizer (lite) → Entity Extractor (flash)
- Atomizer (flash) → Entity Extractor (lite)
- Atomizer (flash) → Entity Extractor (flash)

Shows downstream edges for the problematic chunk.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from src.schemas.financial_relation import FinancialRelationList

# The problematic chunk
CHUNK_TEXT = """Structure
Former subsidiaries include Nest Labs, which was merged into Google in February 2018,[38l and Chronicle Security, which was merged with Google Cloud in June 2019.[39] S Sidewalk Labs was absorbed into Google in 2021 following CEO Daniel L. Doctoroff's departure from the company due to a suspected ALS diagnosis."""

HEADER_PATH = "Alphabet > Structure"
METADATA = {"doc_date": "2024-01-01", "section_header": HEADER_PATH}

# Expected correct extractions
EXPECTED_EDGES = [
    ("Google", "merged/acquired", "Nest Labs"),
    ("Google Cloud", "merged/acquired", "Chronicle Security"),
    ("Google", "absorbed", "Sidewalk Labs"),
]

class PropositionList(BaseModel):
    propositions: List[str] = Field(description="List of atomic facts")


def run_atomizer(chunk_text: str, metadata: dict, model: str) -> List[str]:
    """Run atomizer with specified model."""
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    structured_llm = llm.with_structured_output(PropositionList)

    system_prompt = (
        "You are an expert Text Decomposer. Split the input text into 'Atomic Facts'.\n\n"
        "Rules:\n"
        "1. DE-CONTEXTUALIZE: Resolve pronouns to specific names.\n"
        "2. TEMPORAL GROUNDING: Convert relative dates to absolute.\n"
        "3. EXACT DATE PRESERVATION: Keep dates exactly as written.\n"
        "4. ATOMICITY: Each fact = one simple sentence.\n"
        "5. COMPLETENESS: Each fact must be self-contained and meaningful.\n"
        f"\nMETADATA: {metadata}"
    )

    response = structured_llm.invoke([
        ("system", system_prompt),
        ("human", chunk_text)
    ])
    return response.propositions


def run_entity_extractor(fact: str, chunk_text: str, header: str, model: str) -> List[dict]:
    """Run entity extraction with specified model."""
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    structured_llm = llm.with_structured_output(FinancialRelationList)

    prompt = (
        f"HEADER: {header}\n"
        f"CHUNK: \"{chunk_text}\"\n\n"
        f"FACT TO ANALYZE: \"{fact}\"\n\n"

        f"GOAL: Extract relationships from the FACT.\n\n"

        f"=== CRITICAL: SEMANTIC ROLE ASSIGNMENT ===\n\n"
        f"Subject and Object must be assigned based on SEMANTIC ROLES, not word order:\n\n"
        f"- SUBJECT = the AGENT (who performs, initiates, or causes the action)\n"
        f"- OBJECT = the PATIENT (who receives, undergoes, or is affected by the action)\n\n"
        f"PASSIVE VOICE WARNING:\n"
        f"In passive constructions, the grammatical subject is often the semantic PATIENT.\n"
        f"You must identify the true AGENT regardless of word order.\n\n"
        f"Ask yourself: 'Who is DOING the action to whom?'\n"
        f"The doer = Subject. The receiver = Object.\n"
    )

    try:
        result = structured_llm.invoke([("human", prompt)])
        return [
            {
                "subject": r.subject,
                "relation": r.relationship_description,
                "object": r.object,
                "date": r.date_context
            }
            for r in result.relations
        ]
    except Exception as e:
        return [{"error": str(e)}]


def check_edge_correctness(edge: dict) -> str:
    """Check if an edge has correct direction."""
    subj = edge.get("subject", "").lower()
    obj = edge.get("object", "").lower()

    # Check for correct directions
    if "google" in subj and "nest" in obj:
        return "✓ CORRECT"
    if "google cloud" in subj and "chronicle" in obj:
        return "✓ CORRECT"
    if "google" in subj and "sidewalk" in obj:
        return "✓ CORRECT"

    # Check for wrong directions
    if "nest" in subj and "google" in obj:
        return "✗ WRONG DIRECTION"
    if "chronicle" in subj and "google" in obj:
        return "✗ WRONG DIRECTION"
    if "sidewalk" in subj and "google" in obj:
        return "✗ WRONG DIRECTION"

    return "? OTHER"


def run_pipeline(atomizer_model: str, extractor_model: str) -> dict:
    """Run full pipeline and return results."""
    # Step 1: Atomize
    facts = run_atomizer(CHUNK_TEXT, METADATA, atomizer_model)

    # Step 2: Extract entities from each fact
    all_edges = []
    for fact in facts:
        edges = run_entity_extractor(fact, CHUNK_TEXT, HEADER_PATH, extractor_model)
        for edge in edges:
            edge["source_fact"] = fact
            all_edges.append(edge)

    return {
        "facts": facts,
        "edges": all_edges
    }


def print_results(name: str, results: dict):
    """Pretty print results."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    print(f"\n📝 FACTS EXTRACTED ({len(results['facts'])}):")
    for i, fact in enumerate(results['facts'], 1):
        print(f"  {i}. {fact}")

    print(f"\n🔗 EDGES EXTRACTED ({len(results['edges'])}):")
    correct = 0
    wrong = 0
    for edge in results['edges']:
        if "error" in edge:
            print(f"  ERROR: {edge['error']}")
            continue

        status = check_edge_correctness(edge)
        if "CORRECT" in status:
            correct += 1
        elif "WRONG" in status:
            wrong += 1

        date_str = f" [{edge.get('date', '')}]" if edge.get('date') else ""
        print(f"  {edge['subject']} -> {edge['relation']} -> {edge['object']}{date_str}  {status}")

    print(f"\n📊 SCORE: {correct} correct, {wrong} wrong direction")
    return correct, wrong


if __name__ == "__main__":
    print("="*70)
    print("FULL PIPELINE A/B TEST")
    print("="*70)
    print(f"\nChunk: {CHUNK_TEXT[:100]}...")
    print(f"\nExpected edges (correct direction):")
    for e in EXPECTED_EDGES:
        print(f"  {e[0]} -> {e[1]} -> {e[2]}")

    results_summary = []

    # Test all 4 combinations
    configs = [
        ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite", "A: Lite → Lite (Current)"),
        ("gemini-2.5-flash-lite", "gemini-2.5-flash", "B: Lite → Flash"),
        ("gemini-2.5-flash", "gemini-2.5-flash-lite", "C: Flash → Lite"),
        ("gemini-2.5-flash", "gemini-2.5-flash", "D: Flash → Flash"),
    ]

    for atomizer_model, extractor_model, name in configs:
        print(f"\n\n{'#'*70}")
        print(f"Running: {name}")
        print(f"  Atomizer: {atomizer_model}")
        print(f"  Extractor: {extractor_model}")
        print(f"{'#'*70}")

        results = run_pipeline(atomizer_model, extractor_model)
        correct, wrong = print_results(name, results)
        results_summary.append((name, correct, wrong, len(results['facts']), len(results['edges'])))

    # Final summary
    print("\n\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Config':<25} {'Correct':<10} {'Wrong':<10} {'Facts':<10} {'Edges':<10}")
    print("-"*70)
    for name, correct, wrong, facts, edges in results_summary:
        print(f"{name:<25} {correct:<10} {wrong:<10} {facts:<10} {edges:<10}")

    print("\n" + "="*70)
