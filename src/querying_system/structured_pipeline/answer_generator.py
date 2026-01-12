"""
Step 3: Answer Generator
Synthesizes a grounded answer from retrieved chunks.
"""

import time

from .models import QueryPlan, RetrievalResult
from .prompts import (
    ANSWER_GENERATOR_SYSTEM_PROMPT,
    NO_EVIDENCE_RESPONSE_TEMPLATE,
)
from src.util.llm_client import get_llm


class AnswerGenerator:
    """
    Generates grounded answers from retrieved evidence.
    Uses a single LLM call with the evidence as context.
    """

    def __init__(self):
        """Initialize with gpt-5.1 for quality answer generation."""
        self.llm = get_llm()

    def generate(
        self,
        question: str,
        plan: QueryPlan,
        retrieval: RetrievalResult,
        max_chunks: int = 5,
    ) -> tuple[str, int]:
        """
        Generate an answer from retrieved evidence.

        Args:
            question: The original user question
            plan: The query plan used
            retrieval: The retrieval results with chunks
            max_chunks: Maximum number of chunks to include in context

        Returns:
            tuple of (answer_string, elapsed_time_ms)
        """
        start_time = time.time()

        # Handle no evidence case
        if not retrieval.chunks:
            answer = self._generate_no_evidence_response(plan)
            elapsed_ms = int((time.time() - start_time) * 1000)
            return answer, elapsed_ms

        # Format evidence for context
        evidence = retrieval.get_context_for_llm(max_chunks=max_chunks)

        # Format entities for display
        entities_searched = (
            ", ".join(
                [e.resolved_name or e.query for e in retrieval.resolved_entities]
            )
            or "None"
        )

        # Build the prompt
        user_content = f"""## Question
{question}

## Retrieved Evidence
{evidence}

## Query Analysis
- Query Type: {plan.query_type.value}
- Entities Searched: {entities_searched}
- Retrieval Pattern: {retrieval.retrieval_pattern_used}

Please provide a comprehensive answer based ONLY on the evidence above. Cite every claim with [CHUNK: chunk_id]."""

        messages = [
            {"role": "system", "content": ANSWER_GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Call LLM
        response = self.llm.invoke(messages)
        answer = response.content

        elapsed_ms = int((time.time() - start_time) * 1000)

        return answer, elapsed_ms

    def _generate_no_evidence_response(self, plan: QueryPlan) -> str:
        """Generate a helpful response when no evidence was found."""
        entities_searched = ", ".join(plan.anchor_entities) or "None"
        relationships_searched = plan.target_relationship or "All relationships"
        search_terms = ", ".join(plan.fallback_search_terms) or "None"

        return NO_EVIDENCE_RESPONSE_TEMPLATE.format(
            entities_searched=entities_searched,
            relationships_searched=relationships_searched,
            search_terms=search_terms,
        )
