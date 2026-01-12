"""
Compare V1 vs V2 pipelines on challenging multi-hop questions.
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable verbose logging
os.environ["VERBOSE"] = "true"

from src.querying_system.hybrid_cot_gnn import query_hybrid_cot
from src.querying_system.hybrid_cot_gnn.pipeline_v2 import query_hybrid_cot_v2


# Select challenging multi-hop questions
TEST_QUESTIONS = [
    {
        "id": 51,
        "category": "Multi-hop: District + Sector",
        "question": "In districts where economic activity declined, what happened to manufacturing?",
        "expected": "In the four districts where economic activity declined (New York, Minneapolis, Kansas City, San Francisco): New York saw manufacturing hold steady after a summer uptick; Minneapolis reported manufacturing was flat; Kansas City saw manufacturing output growth moderate; and San Francisco reported manufacturing activity remained stable.",
    },
    {
        "id": 52,
        "category": "Multi-hop: Policy + Sector Impact",
        "question": "Which districts mentioned both tariff impacts on manufacturing AND labor shortages from immigration policies?",
        "expected": "St. Louis and Chicago districts reported both: tariff-induced input cost increases affected manufacturing, and immigration policies resulted in labor shortages.",
    },
    {
        "id": 55,
        "category": "Multi-hop: Consumer + Income",
        "question": "How did consumer spending patterns differ between income groups, and what drove those differences?",
        "expected": "Higher-income individuals maintained strong spending on luxury travel and accommodation. Lower- and middle-income households sought discounts and promotions due to rising prices and elevated economic uncertainty.",
    },
    {
        "id": 58,
        "category": "Multi-hop: Agriculture + Trade",
        "question": "How did trade-related concerns affect the agriculture sector across different districts?",
        "expected": "Minneapolis District: concerned about China's elimination of soybean purchases. St. Louis: agriculture conditions strained and deteriorated. Chicago: soybean prices lower due to absence of new-crop exports to China.",
    },
    {
        "id": 67,
        "category": "Multi-hop: Outlook + Uncertainty",
        "question": "What specific factors contributed to economic uncertainty across different districts?",
        "expected": "Tariff policy changes made planning difficult for manufacturers. Federal grant uncertainty affected nonprofits. Immigration policy changes created labor supply concerns. Potential government shutdown highlighted as downside risk.",
    },
    {
        "id": 15,
        "category": "Leisure (previously failed)",
        "question": "How did demand from domestic consumers for leisure and hospitality change?",
        "expected": "Demand by domestic consumers was largely unchanged.",
    },
]


def evaluate_answer(answer: str, expected: str) -> dict:
    """Simple keyword matching to evaluate answer quality."""
    answer_lower = answer.lower()
    expected_lower = expected.lower()

    # Extract key phrases from expected
    key_phrases = []
    if "largely unchanged" in expected_lower:
        key_phrases.append("largely unchanged")
    if "new york" in expected_lower:
        key_phrases.append("new york")
    if "st. louis" in expected_lower or "st louis" in expected_lower:
        key_phrases.append(("st. louis", "st louis", "stlouis"))
    if "chicago" in expected_lower:
        key_phrases.append("chicago")
    if "minneapolis" in expected_lower:
        key_phrases.append("minneapolis")
    if "kansas city" in expected_lower:
        key_phrases.append("kansas city")
    if "san francisco" in expected_lower:
        key_phrases.append("san francisco")
    if "tariff" in expected_lower:
        key_phrases.append("tariff")
    if "immigration" in expected_lower:
        key_phrases.append("immigration")
    if "soybean" in expected_lower:
        key_phrases.append("soybean")
    if "china" in expected_lower:
        key_phrases.append("china")
    if "higher-income" in expected_lower or "luxury" in expected_lower:
        key_phrases.append(("luxury", "higher-income", "high-income", "wealthy"))
    if "lower-income" in expected_lower or "middle-income" in expected_lower:
        key_phrases.append(("lower-income", "middle-income", "low-income", "discount"))
    if "federal grant" in expected_lower or "nonprofit" in expected_lower:
        key_phrases.append(("federal grant", "nonprofit", "non-profit"))
    if "government shutdown" in expected_lower:
        key_phrases.append(("government shutdown", "shutdown"))

    # Count matches
    matches = 0
    for phrase in key_phrases:
        if isinstance(phrase, tuple):
            if any(p in answer_lower for p in phrase):
                matches += 1
        else:
            if phrase in answer_lower:
                matches += 1

    total = len(key_phrases) if key_phrases else 1
    score = matches / total if total > 0 else 0

    return {
        "score": score,
        "matches": matches,
        "total_phrases": total,
    }


def run_comparison(question_data: dict):
    """Run both V1 and V2 on a question and compare."""
    q = question_data["question"]
    expected = question_data["expected"]

    print("\n" + "=" * 100)
    print(f"Q{question_data['id']}: {q}")
    print(f"Category: {question_data['category']}")
    print("=" * 100)

    results = {}

    # Run V1
    print("\n--- V1 PIPELINE ---")
    start = time.time()
    try:
        v1_result = query_hybrid_cot(q)
        v1_time = time.time() - start
        v1_eval = evaluate_answer(v1_result.answer.answer, expected)
        results["v1"] = {
            "answer": v1_result.answer.answer,
            "confidence": v1_result.answer.confidence,
            "time": v1_time,
            "facts": len(v1_result.evidence_pool.scored_facts),
            "expansion": v1_result.evidence_pool.expansion_performed,
            "eval_score": v1_eval["score"],
            "eval_matches": v1_eval["matches"],
        }
        print(f"V1: {v1_eval['matches']}/{v1_eval['total_phrases']} key phrases, {v1_time:.1f}s")
    except Exception as e:
        results["v1"] = {"error": str(e)}
        print(f"V1 ERROR: {e}")

    # Run V2
    print("\n--- V2 PIPELINE ---")
    start = time.time()
    try:
        v2_result = query_hybrid_cot_v2(q)
        v2_time = time.time() - start
        v2_eval = evaluate_answer(v2_result.answer.answer, expected)
        results["v2"] = {
            "answer": v2_result.answer.answer,
            "confidence": v2_result.answer.confidence,
            "time": v2_time,
            "facts": len(v2_result.evidence_pool.scored_facts),
            "expansion": v2_result.evidence_pool.expansion_performed,
            "eval_score": v2_eval["score"],
            "eval_matches": v2_eval["matches"],
        }
        print(f"V2: {v2_eval['matches']}/{v2_eval['total_phrases']} key phrases, {v2_time:.1f}s")
    except Exception as e:
        results["v2"] = {"error": str(e)}
        print(f"V2 ERROR: {e}")

    # Print full answers
    print("\n--- EXPECTED ---")
    print(expected)

    if "answer" in results.get("v1", {}):
        print("\n--- V1 ANSWER (FULL) ---")
        print(results["v1"]["answer"])

    if "answer" in results.get("v2", {}):
        print("\n--- V2 ANSWER (FULL) ---")
        print(results["v2"]["answer"])

    return results


def main():
    print("=" * 100)
    print("V1 vs V2 Pipeline Comparison Test")
    print("=" * 100)

    all_results = []

    for q_data in TEST_QUESTIONS:
        results = run_comparison(q_data)
        results["id"] = q_data["id"]
        results["question"] = q_data["question"]
        results["expected"] = q_data["expected"]
        all_results.append(results)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'ID':<5} {'Category':<35} {'V1 Score':<12} {'V2 Score':<12} {'V1 Time':<10} {'V2 Time':<10} {'Winner':<8}")
    print("-" * 100)

    v1_wins = 0
    v2_wins = 0
    ties = 0

    for r in all_results:
        v1_score = r.get("v1", {}).get("eval_score", 0)
        v2_score = r.get("v2", {}).get("eval_score", 0)
        v1_time = r.get("v1", {}).get("time", 999)
        v2_time = r.get("v2", {}).get("time", 999)

        if v1_score > v2_score:
            winner = "V1"
            v1_wins += 1
        elif v2_score > v1_score:
            winner = "V2"
            v2_wins += 1
        else:
            winner = "TIE"
            ties += 1

        q_data = next((q for q in TEST_QUESTIONS if q["id"] == r["id"]), {})
        cat = q_data.get("category", "")[:33]

        print(f"{r['id']:<5} {cat:<35} {v1_score:<12.2f} {v2_score:<12.2f} {v1_time:<10.1f} {v2_time:<10.1f} {winner:<8}")

    print("-" * 100)
    print(f"V1 Wins: {v1_wins}, V2 Wins: {v2_wins}, Ties: {ties}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"eval_v1_v2_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
