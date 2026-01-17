"""
Step 1: Query Analyzer
Classifies the user's question and creates a structured retrieval plan.
"""

import json
import time
import re

from .models import QueryPlan, QueryType
from .prompts import QUERY_ANALYZER_SYSTEM_PROMPT
from src.util.llm_client import get_nano_gpt_llm


class QueryAnalyzer:
    """
    Analyzes user questions and produces structured retrieval plans.

    Uses a single LLM call with structured output.
    """

    def __init__(self):
        """Initialize the analyzer with gpt-5-mini for fast, cheap classification."""
        self.llm = get_nano_gpt_llm().with_structured_output(QueryPlan)

    def analyze(self, question: str) -> tuple[QueryPlan, int]:
        """
        Analyze a question and return a retrieval plan.

        Args:
            question: The user's natural language question

        Returns:
            tuple of (QueryPlan, elapsed_time_ms)
        """
        start_time = time.time()

        try:
            messages = [
                {"role": "system", "content": QUERY_ANALYZER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nOutput the JSON retrieval plan:"},
            ]

            plan = self.llm.invoke(messages)

            # Validate query_type
            if not isinstance(plan.query_type, QueryType):
                plan.query_type = QueryType.UNKNOWN

        except Exception as e:
            # Fallback plan if analysis fails
            plan = self._create_fallback_plan(question, f"Analysis error: {e}")

        elapsed_ms = int((time.time() - start_time) * 1000)

        return plan, elapsed_ms

    def _create_fallback_plan(self, question: str, error: str) -> QueryPlan:
        """
        Create a fallback plan when LLM analysis fails.
        Uses simple heuristics to create a reasonable plan.
        """
        # Extract potential entity names (capitalized words)
        words = question.split()
        potential_entities = [
            w.strip("?.,!")
            for w in words
            if w and w[0].isupper() and len(w) > 2
        ]

        # Filter out common words
        stop_words = {"What", "Who", "When", "Where", "How", "Which", "The", "This", "That"}
        potential_entities = [e for e in potential_entities if e not in stop_words]

        # Create search terms from the question
        fallback_terms = [
            question[:100],  # First 100 chars
            " ".join(potential_entities) if potential_entities else question[:50],
        ]

        # Determine query type from keywords
        question_lower = question.lower()
        if any(w in question_lower for w in ["compare", "versus", " vs ", "difference"]):
            query_type = QueryType.COMPARISON
        elif any(w in question_lower for w in ["trend", "overall", "main", "general"]):
            query_type = QueryType.GLOBAL_THEME
        elif any(w in question_lower for w in ["in 20", "last year", "recent"]):
            query_type = QueryType.TEMPORAL
        elif potential_entities:
            query_type = QueryType.ENTITY_RELATIONSHIP
        else:
            query_type = QueryType.GLOBAL_THEME

        return QueryPlan(
            query_type=query_type,
            anchor_entities=potential_entities[:3],
            target_relationship=None,
            fallback_search_terms=fallback_terms,
            confidence=0.3,
            reasoning=f"Fallback plan due to: {error}",
        )


class RuleBasedQueryAnalyzer:
    """
    Rule-based query analyzer that doesn't use LLM.
    Faster and free, but less accurate for complex queries.
    """

    # District name mappings
    DISTRICT_ALIASES = {
        "first": ["boston", "first district"],
        "second": ["new york", "second district"],
        "third": ["philadelphia", "third district"],
        "fourth": ["cleveland", "fourth district"],
        "fifth": ["richmond", "fifth district"],
        "sixth": ["atlanta", "sixth district"],
        "seventh": ["chicago", "seventh district"],
        "eighth": ["st. louis", "st louis", "eighth district"],
        "ninth": ["minneapolis", "ninth district"],
        "tenth": ["kansas city", "tenth district"],
        "eleventh": ["dallas", "eleventh district"],
        "twelfth": ["san francisco", "twelfth district"],
    }

    def analyze(self, question: str) -> tuple[QueryPlan, int]:
        """Analyze using rules instead of LLM."""
        start = time.time()

        question_lower = question.lower()

        # Detect query type
        query_type = self._detect_query_type(question_lower)

        # Extract entities
        entities = self._extract_entities(question)

        # Expand district aliases
        entities = self._expand_district_aliases(entities, question_lower)

        # Detect relationship
        rel_type, direction = self._detect_relationship(question_lower)

        # Detect temporal
        temporal = self._detect_temporal(question_lower)

        # Create fallback terms
        fallback_terms = self._create_fallback_terms(question, entities)

        plan = QueryPlan(
            query_type=query_type,
            anchor_entities=entities,
            target_relationship=rel_type,
            relationship_direction=direction,
            temporal_filter=temporal,
            fallback_search_terms=fallback_terms,
            confidence=0.6,
            reasoning="Rule-based analysis",
        )

        elapsed_ms = int((time.time() - start) * 1000)
        return plan, elapsed_ms

    def _detect_query_type(self, question_lower: str) -> QueryType:
        """Classify query type based on patterns."""
        comparison_patterns = ["compare", "versus", " vs ", "difference between", "differ"]
        global_patterns = ["main theme", "overall", "trend", "across district", "general"]
        temporal_patterns = ["in 20", "last year", "recent", "this year", "october", "november"]

        for pattern in comparison_patterns:
            if pattern in question_lower:
                return QueryType.COMPARISON

        for pattern in global_patterns:
            if pattern in question_lower:
                return QueryType.GLOBAL_THEME

        for pattern in temporal_patterns:
            if pattern in question_lower:
                return QueryType.TEMPORAL

        return QueryType.ENTITY_RELATIONSHIP

    def _extract_entities(self, question: str) -> list[str]:
        """Extract potential entity names."""
        stop_words = {"What", "Who", "When", "Where", "How", "Which", "The", "This", "That", "These", "According"}

        words = question.split()
        entities = []
        current_entity = []

        for word in words:
            clean_word = word.strip("?.,!\"'()")
            if clean_word and clean_word[0].isupper() and clean_word not in stop_words:
                current_entity.append(clean_word)
            elif current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []

        if current_entity:
            entities.append(" ".join(current_entity))

        return entities[:5]

    def _expand_district_aliases(self, entities: list[str], question_lower: str) -> list[str]:
        """Expand district references to include city names."""
        expanded = list(entities)

        for district_num, aliases in self.DISTRICT_ALIASES.items():
            # Check if any alias appears in question
            for alias in aliases:
                if alias in question_lower:
                    # Add the city name (first alias) if not already present
                    city = aliases[0]
                    if city.title() not in [e.lower() for e in expanded]:
                        expanded.append(city.title())
                    break

        return expanded

    def _detect_relationship(self, question_lower: str) -> tuple[str | None, str | None]:
        """Detect relationship type from keywords."""
        keyword_map = {
            "acquire": ("ACQUIRED", "outgoing"),
            "bought": ("ACQUIRED", "outgoing"),
            "invest": ("INVESTED_IN", "outgoing"),
            "hire": ("HIRED", "outgoing"),
            "partner": ("PARTNERED_WITH", "both"),
        }

        for keyword, (rel_type, direction) in keyword_map.items():
            if keyword in question_lower:
                return rel_type, direction

        return None, "both"

    def _detect_temporal(self, question_lower: str) -> dict | None:
        """Detect temporal constraints."""
        year_match = re.search(r"in (20\d{2})", question_lower)
        if year_match:
            year = year_match.group(1)
            return {"start": f"{year}-01-01", "end": f"{year}-12-31"}

        if "october 2025" in question_lower:
            return {"start": "2025-10-01", "end": "2025-10-31"}

        if "last year" in question_lower:
            return {"relative": "last_year"}

        if "recent" in question_lower:
            return {"relative": "recent"}

        return None

    def _create_fallback_terms(self, question: str, entities: list[str]) -> list[str]:
        """Create search terms for fallback."""
        terms = []

        # Key phrases from question
        key_phrases = []
        q_lower = question.lower()

        # Economic terms
        econ_terms = ["economic activity", "employment", "wage", "price", "manufacturing",
                      "consumer spending", "real estate", "banking", "energy", "agriculture"]
        for term in econ_terms:
            if term in q_lower:
                key_phrases.append(term)

        terms.extend(key_phrases[:2])

        # Add entity-based terms
        for entity in entities[:2]:
            terms.append(entity)

        # Add question fragment if short
        if len(question) <= 100:
            terms.append(question)

        return terms[:4]
