"""
Phase 3: Answer Synthesis.

Uses gpt-5.1 to generate a well-cited, question-type-aware answer
from the scored evidence pool.
"""

import os
import time

from .schemas import (
    ScoredFact,
    EvidencePool,
    QueryDecomposition,
    StructuredAnswer,
)
from .prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_PROMPT,
    get_question_type_instructions,
)
from src.util.llm_client import get_critique_llm

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[Synthesizer] {msg}")


class Synthesizer:
    """
    Evidence-based answer synthesizer using gpt-5.2.

    Features:
    - Question-type-aware formatting (comparison: side-by-side, causal: chain)
    - Coverage gap detection
    - Confidence calculation based on evidence quality
    """

    def __init__(self, llm=None):
        if llm is None:
            llm = get_critique_llm()  # gpt-5.1 for quality synthesis

        self.llm = llm

    def synthesize(
        self,
        question: str,
        decomposition: QueryDecomposition,
        evidence_pool: EvidencePool,
        top_k_evidence: int = 15,
        unique_entities_by_node: dict[str, list[str]] | None = None,  # For ENUMERATION
    ) -> tuple[StructuredAnswer, int]:
        """
        Synthesize a final answer from the evidence pool.

        Args:
            question: Original user question
            decomposition: Query decomposition for context
            evidence_pool: Scored and optionally expanded evidence
            top_k_evidence: Number of top facts to include in context

        Returns:
            tuple of (StructuredAnswer, elapsed_time_ms)
        """
        start_time = time.time()

        # Select top evidence
        top_facts = sorted(
            evidence_pool.scored_facts,
            key=lambda f: f.final_score,
            reverse=True,
        )[:top_k_evidence]

        if not top_facts:
            log("No evidence found, returning empty answer")
            return StructuredAnswer(
                answer="I couldn't find any relevant information to answer this question.",
                evidence_refs=[],
                confidence=0.0,
                gaps=decomposition.required_info,
            ), int((time.time() - start_time) * 1000)

        log(f"Synthesizing from {len(top_facts)} facts...")

        # Format evidence for LLM
        evidence_text = self._format_evidence(top_facts)

        # Detect coverage gaps
        gaps = self._detect_gaps(decomposition, evidence_pool)
        gaps_note = ""
        if gaps:
            gaps_note = f"\n\nNOTE: The following required information may not be fully covered:\n"
            gaps_note += "\n".join(f"- {g}" for g in gaps)

        # For ENUMERATION questions, add unique entities summary
        from .schemas import QuestionType
        enumeration_note = ""
        if (
            decomposition.question_type == QuestionType.ENUMERATION
            and unique_entities_by_node
        ):
            enumeration_note = "\n\nENTITIES FOUND FOR ENUMERATION:\n"
            for node_name, entities in unique_entities_by_node.items():
                if entities:
                    enumeration_note += f"  Related to '{node_name}': {', '.join(entities)}\n"
            enumeration_note += "Use this list to ensure your enumeration is complete.\n"

        # Get type-specific instructions
        type_instructions = get_question_type_instructions(
            decomposition.question_type
        )

        prompt = SYNTHESIS_USER_PROMPT.format(
            question=question,
            required_info="\n".join(
                f"- {info}" for info in decomposition.required_info
            ),
            type_instructions=type_instructions,
            evidence=evidence_text + enumeration_note,
            gaps_note=gaps_note,
        )

        messages = [
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            ("human", prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            answer_text = response.content if hasattr(response, 'content') else str(response)

            # Calculate confidence
            confidence = self._calculate_confidence(top_facts, gaps, decomposition)

            elapsed = int((time.time() - start_time) * 1000)
            log(f"Synthesized in {elapsed}ms, confidence={confidence:.2f}")

            return StructuredAnswer(
                answer=answer_text,
                evidence_refs=[f.chunk_id for f in top_facts if f.chunk_id],
                confidence=confidence,
                gaps=gaps,
            ), elapsed

        except Exception as e:
            log(f"Synthesis error: {e}")
            return StructuredAnswer(
                answer=f"An error occurred while generating the answer: {str(e)}",
                evidence_refs=[],
                confidence=0.0,
                gaps=decomposition.required_info,
            ), int((time.time() - start_time) * 1000)

    def _format_evidence(self, facts: list[ScoredFact]) -> str:
        """
        Format facts for LLM context with provenance.

        Key change: Include chunk_content for full context, not just atomic facts.
        This prevents misinterpretation and provides surrounding context.
        """
        lines = []

        # Group facts by chunk to avoid repeating chunk content
        chunk_facts: dict[str, list[ScoredFact]] = {}
        for fact in facts:
            chunk_id = fact.chunk_id or "no_chunk"
            if chunk_id not in chunk_facts:
                chunk_facts[chunk_id] = []
            chunk_facts[chunk_id].append(fact)

        evidence_num = 1
        for chunk_id, chunk_fact_list in chunk_facts.items():
            # Get chunk info from first fact
            first_fact = chunk_fact_list[0]

            lines.append(f"[EVIDENCE {evidence_num}]")

            # Source info
            source_parts = []
            if first_fact.chunk_header:
                source_parts.append(f"Section: {first_fact.chunk_header}")
            if first_fact.doc_id:
                source_parts.append(f"Document: {first_fact.doc_id}")
            if first_fact.document_date:
                source_parts.append(f"Date: {first_fact.document_date}")

            if source_parts:
                lines.append(f"  Source: {', '.join(source_parts)}")

            # Include full chunk content for context (no truncation)
            if first_fact.chunk_content:
                lines.append(f"  Full Context:")
                lines.append(f"    {first_fact.chunk_content}")

            # List the specific facts extracted from this chunk
            lines.append(f"  Key Facts from this section:")
            for fact in chunk_fact_list:
                lines.append(f"    - {fact.subject} -[{fact.edge_type}]-> {fact.object}")
                lines.append(f"      {fact.content}")

            # Best relevance score for this chunk
            best_score = max(f.final_score for f in chunk_fact_list)
            lines.append(f"  Relevance Score: {best_score:.2f}")
            lines.append("")
            evidence_num += 1

        return "\n".join(lines)

    def _detect_gaps(
        self,
        decomposition: QueryDecomposition,
        evidence_pool: EvidencePool,
    ) -> list[str]:
        """Detect which required_info items lack evidence."""
        gaps = []

        for info in decomposition.required_info:
            covering_fact_ids = evidence_pool.coverage_map.get(info, [])

            if not covering_fact_ids:
                gaps.append(info)
            else:
                # Check if covering facts have good scores
                covering_facts = [
                    f for f in evidence_pool.scored_facts
                    if f.fact_id in covering_fact_ids
                ]
                max_score = max((f.final_score for f in covering_facts), default=0)
                if max_score < 0.3:
                    gaps.append(f"{info} (low confidence evidence)")

        return gaps

    def _calculate_confidence(
        self,
        top_facts: list[ScoredFact],
        gaps: list[str],
        decomposition: QueryDecomposition,
    ) -> float:
        """
        Calculate answer confidence based on evidence quality.

        Formula:
        - Base: Average final score of top facts
        - Penalty: -0.15 per gap
        - Bonus: +0.1 for multi-query coverage (facts found by multiple sub-queries)
        """
        if not top_facts:
            return 0.0

        # Base: average score
        avg_score = sum(f.final_score for f in top_facts) / len(top_facts)

        # Gap penalty
        gap_penalty = len(gaps) * 0.15

        # Multi-query bonus: if facts were found by multiple sub-queries
        multi_query_facts = sum(1 for f in top_facts if len(f.found_by_queries) > 1)
        multi_query_bonus = min(0.2, multi_query_facts * 0.05)

        # Combined
        confidence = avg_score - gap_penalty + multi_query_bonus

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))
