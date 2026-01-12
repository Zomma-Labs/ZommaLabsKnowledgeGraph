"""
Phase 1: Query Decomposition.

Single LLM call (gpt-5.1) that extracts from the question:
1. Entities explicitly mentioned (not enumerated from knowledge)
2. Topics/concepts mentioned
3. Relationships with modifiers
4. Sub-queries for retrieval
5. Question type classification
"""

import os
import time

from .schemas import QueryDecomposition, QuestionType, SubQuery
from .prompts import DECOMPOSITION_SYSTEM_PROMPT, DECOMPOSITION_USER_PROMPT
from src.util.llm_client import get_critique_llm

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    if VERBOSE:
        print(f"[QueryDecomposer] {msg}")


class QueryDecomposer:
    """
    Chain-of-thought query decomposer.

    Key innovation: For comparison questions like "Compare inflation in Boston vs NY",
    generates combinatorial sub-queries:
    - ["inflation Boston", "inflation New York"]

    This enables parallel retrieval and cross-query boosting.
    """

    def __init__(self, llm=None):
        if llm is None:
            llm = get_critique_llm()  # gpt-5.1 for quality decomposition

        self.llm = llm
        self.structured_decomposer = llm.with_structured_output(
            QueryDecomposition, include_raw=True
        )

    def decompose(self, question: str) -> tuple[QueryDecomposition, int]:
        """
        Decompose a question into structured retrieval plan.

        Args:
            question: User's natural language question

        Returns:
            tuple of (QueryDecomposition, elapsed_time_ms)
        """
        start_time = time.time()
        log(f"Decomposing: {question[:80]}...")

        messages = [
            ("system", DECOMPOSITION_SYSTEM_PROMPT),
            ("human", DECOMPOSITION_USER_PROMPT.format(question=question)),
        ]

        try:
            response = self.structured_decomposer.invoke(messages)

            if response.get("parsing_error"):
                log(f"Parsing error: {response['parsing_error']}")
                return self._create_fallback(question), int(
                    (time.time() - start_time) * 1000
                )

            parsed = response.get("parsed")
            if parsed is None:
                log("No parsed result, using fallback")
                return self._create_fallback(question), int(
                    (time.time() - start_time) * 1000
                )

            elapsed = int((time.time() - start_time) * 1000)
            log(
                f"Decomposed in {elapsed}ms: type={parsed.question_type.value}, "
                f"sub_queries={len(parsed.sub_queries)}, entities={parsed.entity_hints}"
            )
            return parsed, elapsed

        except Exception as e:
            log(f"Decomposition error: {e}")
            return self._create_fallback(question, str(e)), int(
                (time.time() - start_time) * 1000
            )

    def _create_fallback(
        self, question: str, error: str = ""
    ) -> QueryDecomposition:
        """Create basic decomposition when LLM fails."""
        log("Creating fallback decomposition")

        # Extract capitalized words as potential entities
        words = question.split()
        entities = [
            w.strip("?.,!")
            for w in words
            if w and w[0].isupper() and len(w) > 2
        ]
        stop_words = {
            "What",
            "Who",
            "When",
            "Where",
            "How",
            "Which",
            "The",
            "Compare",
            "Why",
            "Did",
            "Does",
            "Do",
            "Is",
            "Are",
            "Was",
            "Were",
        }
        entities = [e for e in entities if e not in stop_words]

        # Detect question type from keywords
        q_lower = question.lower()
        if any(w in q_lower for w in ["compare", "versus", " vs ", "differ", "difference"]):
            q_type = QuestionType.COMPARISON
        elif any(w in q_lower for w in ["why", "cause", "because", "led to", "affect", "effect", "result"]):
            q_type = QuestionType.CAUSAL
        elif any(w in q_lower for w in ["which", "list", "what are", "how many"]):
            q_type = QuestionType.ENUMERATION
        elif any(w in q_lower for w in ["change", "trend", "over time", "since", "from", "to"]):
            q_type = QuestionType.TEMPORAL
        else:
            q_type = QuestionType.FACTUAL

        return QueryDecomposition(
            required_info=[question[:100]],
            sub_queries=[
                SubQuery(
                    query_text=question[:100],
                    target_info="Answer to the question",
                    entity_hints=entities[:3],
                )
            ],
            entity_hints=entities[:5],
            topic_hints=[],
            temporal_scope=None,
            question_type=q_type,
            confidence=0.3,
            reasoning=f"Fallback decomposition. {error}" if error else "Fallback decomposition",
        )
