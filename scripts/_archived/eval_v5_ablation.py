#!/usr/bin/env python
"""
V5 Ablation Study: A/B testing different configurations.

Tests the impact of each feature flag on accuracy:
- enable_global_search: Always run global search alongside scoped
- enable_gap_expansion: LLM-guided expansion when gaps detected
- enable_entity_drilldown: Extra retrieval for ENUMERATION questions

Run with:
    uv run python scripts/eval_v5_ablation.py

With specific questions file:
    uv run python scripts/eval_v5_ablation.py --questions eval/test_questions.json
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Optional

from src.querying_system.v5 import V5Pipeline, ResearcherConfig
from src.util.llm_client import get_critique_llm


# Default test questions if none provided
DEFAULT_QUESTIONS = [
    {
        "question": "What economic conditions did Boston report?",
        "type": "factual",
        "expected_keywords": ["Boston", "economic", "conditions"],
    },
    {
        "question": "How do inflation trends in Boston compare to New York?",
        "type": "comparison",
        "expected_keywords": ["Boston", "New York", "inflation"],
    },
    {
        "question": "Which Federal Reserve districts reported employment growth?",
        "type": "enumeration",
        "expected_keywords": ["employment", "growth", "districts"],
    },
    {
        "question": "What changes occurred in manufacturing from October to November?",
        "type": "temporal",
        "expected_keywords": ["manufacturing", "October", "November", "changes"],
    },
]


# Configuration presets for A/B testing
CONFIGS = {
    "full": ResearcherConfig(
        enable_global_search=True,
        enable_gap_expansion=True,
        enable_entity_drilldown=True,
    ),
    "no_drilldown": ResearcherConfig(
        enable_global_search=True,
        enable_gap_expansion=True,
        enable_entity_drilldown=False,
    ),
    "no_gap_expansion": ResearcherConfig(
        enable_global_search=True,
        enable_gap_expansion=False,
        enable_entity_drilldown=True,
    ),
    "scoped_only": ResearcherConfig(
        enable_global_search=False,
        enable_gap_expansion=True,
        enable_entity_drilldown=True,
    ),
    "minimal": ResearcherConfig(
        enable_global_search=True,
        enable_gap_expansion=False,
        enable_entity_drilldown=False,
    ),
}


class Judge:
    """LLM-based answer quality judge."""

    def __init__(self):
        self.llm = get_critique_llm()

    async def score(
        self,
        question: str,
        answer: str,
        expected_keywords: list[str],
    ) -> dict:
        """
        Score an answer on multiple dimensions.

        Returns:
            {
                "relevance": float 0-1,
                "completeness": float 0-1,
                "accuracy": float 0-1,
                "overall": float 0-1,
                "reasoning": str
            }
        """
        prompt = f"""Score this answer to a financial question.

QUESTION: {question}

ANSWER: {answer}

EXPECTED KEYWORDS/CONCEPTS: {', '.join(expected_keywords)}

Score on these dimensions (0-1):
1. Relevance: Does the answer address the question?
2. Completeness: Does it cover all aspects of the question?
3. Accuracy: Is the information factually sound (based on citations)?
4. Overall: Combined quality score

Return JSON:
{{
    "relevance": 0.0-1.0,
    "completeness": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "overall": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                [("human", prompt)]
            )

            # Parse JSON from response
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Find JSON in response
            import re
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return scores

            # Fallback scores
            return {
                "relevance": 0.5,
                "completeness": 0.5,
                "accuracy": 0.5,
                "overall": 0.5,
                "reasoning": "Failed to parse judge response",
            }

        except Exception as e:
            return {
                "relevance": 0.0,
                "completeness": 0.0,
                "accuracy": 0.0,
                "overall": 0.0,
                "reasoning": f"Judge error: {e}",
            }


async def evaluate_config(
    config_name: str,
    config: ResearcherConfig,
    questions: list[dict],
    judge: Judge,
) -> dict:
    """Evaluate a single configuration on all questions."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}")
    print(f"{'='*60}")

    pipeline = V5Pipeline(config=config)
    results = []

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['question'][:50]}...")

        try:
            # Query
            result = await pipeline.query(q["question"])

            # Judge
            scores = await judge.score(
                q["question"],
                result.answer,
                q.get("expected_keywords", []),
            )

            results.append({
                "question": q["question"],
                "type": q.get("type", "unknown"),
                "answer": result.answer,
                "confidence": result.confidence,
                "num_evidence": len(result.evidence),
                "time_ms": result.total_time_ms,
                "scores": scores,
                "success": True,
            })

            print(f"   Overall: {scores['overall']:.2f}, Time: {result.total_time_ms}ms")

        except Exception as e:
            results.append({
                "question": q["question"],
                "type": q.get("type", "unknown"),
                "error": str(e),
                "success": False,
            })
            print(f"   ERROR: {e}")

    # Aggregate metrics
    successful = [r for r in results if r.get("success")]
    if successful:
        avg_overall = sum(r["scores"]["overall"] for r in successful) / len(successful)
        avg_time = sum(r["time_ms"] for r in successful) / len(successful)
        avg_evidence = sum(r["num_evidence"] for r in successful) / len(successful)
    else:
        avg_overall = 0.0
        avg_time = 0.0
        avg_evidence = 0.0

    return {
        "config_name": config_name,
        "config": {
            "enable_global_search": config.enable_global_search,
            "enable_gap_expansion": config.enable_gap_expansion,
            "enable_entity_drilldown": config.enable_entity_drilldown,
        },
        "num_questions": len(questions),
        "num_success": len(successful),
        "avg_overall_score": avg_overall,
        "avg_time_ms": avg_time,
        "avg_evidence": avg_evidence,
        "results": results,
    }


async def main():
    parser = argparse.ArgumentParser(description="V5 Ablation Study")
    parser.add_argument("--questions", type=str, help="Path to questions JSON file")
    parser.add_argument("--configs", type=str, nargs="+",
                       choices=list(CONFIGS.keys()),
                       default=list(CONFIGS.keys()),
                       help="Configs to test")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    # Load questions
    if args.questions:
        with open(args.questions) as f:
            questions = json.load(f)
    else:
        questions = DEFAULT_QUESTIONS

    print(f"Testing {len(questions)} questions across {len(args.configs)} configurations")

    # Initialize judge
    judge = Judge()

    # Evaluate each config
    all_results = {}
    for config_name in args.configs:
        config = CONFIGS[config_name]
        result = await evaluate_config(config_name, config, questions, judge)
        all_results[config_name] = result

    # Summary
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)

    print(f"\n{'Config':<20} {'Score':>8} {'Time':>10} {'Evidence':>10}")
    print("-"*50)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["avg_overall_score"], reverse=True):
        print(f"{name:<20} {r['avg_overall_score']:>8.2f} {r['avg_time_ms']:>8.0f}ms {r['avg_evidence']:>10.1f}")

    # Feature impact analysis
    print("\n--- Feature Impact ---")
    if "full" in all_results and "no_drilldown" in all_results:
        impact = all_results["full"]["avg_overall_score"] - all_results["no_drilldown"]["avg_overall_score"]
        print(f"Entity drilldown: {impact:+.3f} on score")

    if "full" in all_results and "no_gap_expansion" in all_results:
        impact = all_results["full"]["avg_overall_score"] - all_results["no_gap_expansion"]["avg_overall_score"]
        print(f"Gap expansion: {impact:+.3f} on score")

    if "full" in all_results and "scoped_only" in all_results:
        impact = all_results["full"]["avg_overall_score"] - all_results["scoped_only"]["avg_overall_score"]
        print(f"Global search: {impact:+.3f} on score")

    # Save results
    output_file = args.output or f"eval_v5_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
