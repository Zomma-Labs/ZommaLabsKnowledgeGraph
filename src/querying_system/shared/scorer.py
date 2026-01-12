"""
Phase 2b: LLM Batch Scoring.

Uses gpt-5-mini to batch-score candidate facts for:
1. Relevance to the question (0-1)
2. Should-expand flag (for CAUSAL/COMPARISON questions)
"""

import os
import time

from .schemas import (
    ScoredFact,
    QueryDecomposition,
    QuestionType,
    BatchScoringResult,
)
from .prompts import SCORING_PROMPT
from src.util.llm_client import get_nano_gpt_llm

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[FactScorer] {msg}")


class FactScorer:
    """
    Batch LLM scorer using gpt-5-mini.

    Scores all candidate facts in a single LLM call for efficiency.
    Returns relevance scores and expansion flags.
    """

    def __init__(self, llm=None):
        if llm is None:
            llm = get_nano_gpt_llm()  # gpt-5-mini - cheap and fast

        self.llm = llm
        self.structured_scorer = llm.with_structured_output(
            BatchScoringResult, include_raw=True
        )

    def score(
        self,
        question: str,
        decomposition: QueryDecomposition,
        facts: list[ScoredFact],
        max_facts_to_score: int = 30,
        unique_entities_by_node: dict[str, list[str]] | None = None,  # For ENUMERATION
    ) -> tuple[list[ScoredFact], int]:
        """
        Batch score candidate facts.

        Args:
            question: Original user question
            decomposition: Query decomposition for context
            facts: Candidate facts to score
            max_facts_to_score: Limit to prevent token overflow

        Returns:
            tuple of (scored_facts, elapsed_time_ms)
        """
        start_time = time.time()

        if not facts:
            return [], int((time.time() - start_time) * 1000)

        # Limit facts to score (take top by current scores)
        facts_to_score = sorted(
            facts,
            key=lambda f: f.rrf_score + f.cross_query_boost,
            reverse=True,
        )[:max_facts_to_score]

        log(f"Scoring {len(facts_to_score)} facts with gpt-5-mini...")

        # Format facts for LLM
        facts_text = self._format_facts_for_scoring(facts_to_score)

        # Determine if expansion is relevant
        expansion_relevant = decomposition.question_type in [
            QuestionType.CAUSAL,
            QuestionType.COMPARISON,
        ]

        expansion_hint = (
            "Mark should_expand=true for facts whose entities might have relevant connected facts (causal chains, related comparisons)."
            if expansion_relevant
            else "Set should_expand=false for all facts (not needed for this question type)."
        )

        # For ENUMERATION questions, add unique entities context
        enumeration_context = ""
        if (
            decomposition.question_type == QuestionType.ENUMERATION
            and unique_entities_by_node
        ):
            enumeration_context = "\n\nUNIQUE ENTITIES FOUND (for enumeration):\n"
            for node_name, entities in unique_entities_by_node.items():
                if entities:
                    enumeration_context += f"  Connected to '{node_name}': {', '.join(entities)}\n"
            enumeration_context += "\nNote: These are the unique entities connected to the search nodes. For enumeration questions, ensure you identify all relevant items.\n"

        prompt = SCORING_PROMPT.format(
            question=question,
            required_info="\n".join(
                f"- {info}" for info in decomposition.required_info
            ),
            facts_text=facts_text + enumeration_context,
            expansion_hint=expansion_hint,
        )

        messages = [("human", prompt)]

        try:
            response = self.structured_scorer.invoke(messages)

            if response.get("parsing_error") or response.get("parsed") is None:
                log(f"Scoring parsing error, using fallback")
                return self._apply_fallback_scores(facts_to_score), int(
                    (time.time() - start_time) * 1000
                )

            result = response["parsed"]

            # Apply scores to facts
            scored_count = 0
            for score in result.scores:
                if 0 <= score.fact_index < len(facts_to_score):
                    fact = facts_to_score[score.fact_index]
                    fact.llm_relevance = score.relevance
                    fact.should_expand = score.should_expand
                    scored_count += 1

                    # Calculate final score
                    fact.final_score = (
                        0.5 * (fact.rrf_score + fact.cross_query_boost)
                        + 0.5 * fact.llm_relevance
                    )

            # Handle any unscored facts
            for fact in facts_to_score:
                if fact.llm_relevance == 0.0 and fact.final_score == 0.0:
                    fact.llm_relevance = 0.5
                    fact.final_score = fact.rrf_score + fact.cross_query_boost

            # Sort by final score
            facts_to_score.sort(key=lambda f: f.final_score, reverse=True)

            elapsed = int((time.time() - start_time) * 1000)
            log(
                f"Scored {scored_count} facts in {elapsed}ms, "
                f"expand_flags={sum(1 for f in facts_to_score if f.should_expand)}"
            )

            return facts_to_score, elapsed

        except Exception as e:
            log(f"Scoring error: {e}")
            return self._apply_fallback_scores(facts_to_score), int(
                (time.time() - start_time) * 1000
            )

    def _format_facts_for_scoring(self, facts: list[ScoredFact]) -> str:
        """Format facts as numbered list for LLM."""
        lines = []
        for i, fact in enumerate(facts):
            lines.append(
                f"[{i}] {fact.subject} -[{fact.edge_type}]-> {fact.object}"
            )
            lines.append(f"    FACT: {fact.content}")
            if fact.document_date:
                lines.append(f"    DATE: {fact.document_date}")
            lines.append("")
        return "\n".join(lines)

    def _apply_fallback_scores(
        self, facts: list[ScoredFact]
    ) -> list[ScoredFact]:
        """Apply uniform scores when LLM fails."""
        for fact in facts:
            fact.llm_relevance = 0.5
            fact.should_expand = False
            fact.final_score = fact.rrf_score + fact.cross_query_boost
        return facts
