#!/usr/bin/env python3
"""
Test Deterministic Retrieval on Beige Book Q&A
==============================================

Tests the first N questions from Biege_OA.json using deterministic retrieval.

Usage:
    uv run scripts/test_beige_book_qa.py
    uv run scripts/test_beige_book_qa.py --num-questions 10
"""

import sys
import os
import json
import asyncio
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.util.deterministic_retrieval import DeterministicRetriever
from src.util.llm_client import get_llm


def load_qa_pairs(filepath: str, num_questions: int = 5) -> list[dict]:
    """Load Q&A pairs from the Beige Book JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['qa_pairs'][:num_questions]


async def answer_question(
    question: str,
    retriever: DeterministicRetriever,
    top_k: int = 15
) -> dict:
    """
    Answer a question using deterministic retrieval + LLM synthesis.
    """
    # 1. Deterministic retrieval
    evidence = await retriever.search(question, top_k=top_k)

    # 2. Format evidence for LLM
    evidence_text = retriever.format_evidence_for_llm(evidence)

    # 3. LLM synthesis
    llm = get_llm()

    synthesis_prompt = f"""You are a financial analyst answering questions about the Federal Reserve Beige Book.
Answer based ONLY on the provided evidence. Be specific and cite facts from the evidence.

QUESTION: {question}

EVIDENCE:
{evidence_text}

INSTRUCTIONS:
1. Answer the question using ONLY the evidence provided above
2. If the evidence doesn't contain enough information to fully answer, say what you can answer and what's missing
3. Be specific - mention district names, percentages, and specific facts when available
4. Keep your answer concise but complete

ANSWER:"""

    response = llm.invoke(synthesis_prompt)
    answer = response.content if hasattr(response, 'content') else str(response)

    return {
        "answer": answer,
        "num_evidence": len(evidence),
        "evidence_facts": [e.content for e in evidence[:5]],  # Top 5 facts
        "strategies_used": list(set(s for e in evidence for s in e.found_by)),
    }


def evaluate_answer(generated: str, expected: str) -> dict:
    """
    Simple evaluation: check if key terms from expected answer appear in generated.
    """
    # Extract key terms (words > 4 chars, excluding common words)
    stop_words = {'about', 'after', 'before', 'being', 'between', 'could',
                  'during', 'would', 'should', 'these', 'those', 'their',
                  'there', 'where', 'which', 'while', 'other', 'through'}

    expected_words = set(
        word.lower().strip('.,;:()')
        for word in expected.split()
        if len(word) > 4 and word.lower() not in stop_words
    )

    generated_lower = generated.lower()

    found_terms = [term for term in expected_words if term in generated_lower]
    missing_terms = [term for term in expected_words if term not in generated_lower]

    coverage = len(found_terms) / len(expected_words) if expected_words else 0

    return {
        "coverage": coverage,
        "found_terms": found_terms[:10],  # Limit for display
        "missing_terms": missing_terms[:10],
        "key_term_count": len(expected_words),
    }


async def run_evaluation(num_questions: int = 5, group_id: str = "default"):
    """Run the evaluation on first N questions."""
    print("=" * 70)
    print("BEIGE BOOK Q&A EVALUATION - DETERMINISTIC RETRIEVAL")
    print("=" * 70)
    print(f"Questions: {num_questions}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load Q&A pairs
    qa_pairs = load_qa_pairs("Biege_OA.json", num_questions)

    # Initialize retriever
    retriever = DeterministicRetriever(group_id=group_id)

    results = []
    total_coverage = 0

    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n{'='*70}")
        print(f"QUESTION {i}/{num_questions}: {qa['question']}")
        print(f"Category: {qa['category']}")
        print("-" * 70)

        # Get answer
        result = await answer_question(qa['question'], retriever)

        # Evaluate
        eval_result = evaluate_answer(result['answer'], qa['answer'])
        total_coverage += eval_result['coverage']

        print(f"\nEXPECTED ANSWER:")
        print(f"  {qa['answer'][:200]}...")

        print(f"\nGENERATED ANSWER:")
        print(f"  {result['answer'][:300]}...")

        print(f"\nRETRIEVAL STATS:")
        print(f"  Evidence found: {result['num_evidence']}")
        print(f"  Strategies: {result['strategies_used']}")

        print(f"\nEVALUATION:")
        print(f"  Coverage: {eval_result['coverage']*100:.1f}%")
        print(f"  Found terms: {eval_result['found_terms'][:5]}")
        if eval_result['missing_terms']:
            print(f"  Missing terms: {eval_result['missing_terms'][:5]}")

        print(f"\nTOP EVIDENCE RETRIEVED:")
        for j, fact in enumerate(result['evidence_facts'][:3], 1):
            print(f"  {j}. {fact[:100]}...")

        results.append({
            "question_id": qa['id'],
            "question": qa['question'],
            "category": qa['category'],
            "expected": qa['answer'],
            "generated": result['answer'],
            "coverage": eval_result['coverage'],
            "num_evidence": result['num_evidence'],
            "strategies": result['strategies_used'],
        })

    # Summary
    avg_coverage = total_coverage / num_questions

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Average coverage: {avg_coverage*100:.1f}%")
    print(f"Questions evaluated: {num_questions}")

    # Per-question summary
    print(f"\nPer-question coverage:")
    for r in results:
        status = "✓" if r['coverage'] > 0.5 else "✗"
        print(f"  {status} Q{r['question_id']}: {r['coverage']*100:.0f}% - {r['question'][:50]}...")

    # Save results
    output_file = f"eval_beige_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_questions": num_questions,
            "avg_coverage": avg_coverage,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions to test")
    parser.add_argument("--group-id", default="default", help="Tenant group ID")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.num_questions, args.group_id))


if __name__ == "__main__":
    main()
