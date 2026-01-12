"""
MODULE: Extractor Agent
DESCRIPTION: Single agent that atomizes text and extracts entities/relationships.

Applies Vercel's "80% tool removal" lesson:
- Merged atomizer + entity_extractor into ONE agent
- Financial analyst perspective in prompts
- Single reflexion step (not nested loops)
- Free-form relationships (no 68-type enum)

LLM Calls: 2-3 per chunk
  - Extraction: 1 call
  - Critique: 1 call
  - Re-extract (if needed): 0-1 call
"""

import os
from typing import Optional, TYPE_CHECKING

from src.schemas.extraction import (
    ExtractedFact,
    ExtractionResult,
    CritiqueResult,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

# Control verbose output
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(f"[Extractor] {msg}")


# ============================================================================
# PROMPTS - Financial Analyst Perspective (System + User separation)
# ============================================================================

# --- EXTRACTION PROMPTS ---

EXTRACTION_SYSTEM_PROMPT = """You are a financial analyst building a knowledge graph to track entity interactions and market influences.

Your goal: Extract facts that would help analysts answer questions like:
- "What companies did Microsoft acquire?"
- "How did inflation affect consumer spending?"
- "What leadership changes occurred at tech companies?"
- "Who are the major shareholders of Tesla?"

EXTRACTION INSTRUCTIONS:
For each distinct interaction or relationship in the text:

1. FACT: Write a complete, self-contained atomic proposition
   - Include WHO, WHAT, WHEN if available
   - Should be understandable without reading the original text
   - Example: "Apple Inc. acquired Beats Electronics for $3 billion in May 2014"

2. ENTITIES: Identify the subject and object
   - Subject: The entity performing the action
   - Object: The entity being acted upon or related to
   - Types: Company, Person, Organization, Location, Product, Topic
   - Include a 1-2 sentence summary for each entity

3. RELATIONSHIP: Describe HOW they interact (free-form, be specific)
   - Use active verbs: "acquired", "invested in", "partnered with", "reported"
   - Include nuance: "acquired majority stake in" vs "acquired"
   - Be precise: "appointed as CEO" not just "hired"

4. CONTEXT: Preserve temporal and financial details
   - Dates: "Q3 2024", "November 15, 2024", "fiscal year 2024"
   - Amounts: "$3 billion", "15% increase"

5. TOPICS: Identify related financial themes
   - M&A, Earnings, IPO, Labor Market, Inflation, Interest Rates, etc.

IMPORTANT RULES:
- Each fact must be SELF-CONTAINED (understandable without context)
- Subsidiaries are SEPARATE entities (AWS and Amazon are different entities)
- One fact per relationship (if A acquired B and C, create two facts)
- Preserve exact entity names from the text

Think like an analyst: What would someone search for? What connections matter?"""

EXTRACTION_USER_PROMPT = """DOCUMENT CONTEXT:
Header Path: {header_path}

CHUNK TEXT:
{chunk_text}

Extract all facts from this text following the instructions."""


# --- CRITIQUE PROMPTS ---

CRITIQUE_SYSTEM_PROMPT = """You are a senior financial analyst reviewing extraction results for accuracy.

REVIEW CHECKLIST:

1. COMPLETENESS: Did we capture all material entity interactions?
   - Every company, person, organization mentioned in a relationship
   - Every action, acquisition, partnership, filing, etc.
   - Every financial metric or event

2. ACCURACY: Are extractions correct?
   - Entity names match the text exactly
   - Entity types are correct (Company vs Person vs Organization)
   - Relationships accurately describe the interaction
   - No hallucinated information

3. CONTEXT: Is financial context preserved?
   - Dates and time periods
   - Dollar amounts and percentages
   - Subsidiaries kept as SEPARATE entities (AWS != Amazon)

4. SELF-CONTAINMENT: Can each fact be understood alone?
   - No pronouns without antecedents
   - No "the company" without naming it
   - Complete propositions

If ALL criteria are met, mark as APPROVED.
If ANY issues found, provide SPECIFIC corrections with what's wrong and how to fix it."""

CRITIQUE_USER_PROMPT = """ORIGINAL TEXT:
{chunk_text}

EXTRACTED FACTS:
{facts_summary}

Review these extractions for accuracy and completeness."""


# --- RE-EXTRACTION PROMPTS ---

REEXTRACT_USER_PROMPT = """A senior analyst has reviewed your previous extraction and found issues.

ORIGINAL TEXT:
{chunk_text}

DOCUMENT CONTEXT:
Header Path: {header_path}

PREVIOUS EXTRACTION ISSUES:
{critique}

SPECIFIC CORRECTIONS NEEDED:
{corrections}

MISSED FACTS TO ADD:
{missed_facts}

Please re-extract, addressing ALL the issues above. Follow the same extraction rules:
- Each fact must be self-contained
- Preserve exact entity names
- Subsidiaries are separate entities
- Include dates, amounts, and context"""


class Extractor:
    """
    Single agent that atomizes text and extracts entities/relationships.
    Uses financial analyst perspective and single reflexion step.
    """

    def __init__(self, llm: Optional["BaseChatModel"] = None, max_retries: int = 2):
        if llm is None:
            from src.util.services import get_services
            llm = get_services().llm
        self.llm = llm
        self.max_retries = max_retries
        # Use include_raw=True to detect parsing errors
        self.structured_extractor = llm.with_structured_output(ExtractionResult, include_raw=True)
        self.structured_critic = llm.with_structured_output(CritiqueResult, include_raw=True)

    def extract(self, chunk_text: str, header_path: str = "") -> ExtractionResult:
        """
        Main extraction method with single reflexion step.

        Args:
            chunk_text: The text to extract from
            header_path: Document header path for context

        Returns:
            ExtractionResult with list of ExtractedFact
        """
        # Step 1: Initial extraction
        log(f"Extracting from chunk ({len(chunk_text)} chars)")
        result = self._extract(chunk_text, header_path)
        log(f"Initial extraction: {len(result.facts)} facts")

        if not result.facts:
            log("No facts extracted, skipping critique")
            return result

        # Step 2: Critique
        critique = self._critique(chunk_text, result)

        if critique.is_approved:
            log("Extraction approved by critic")
            return result

        # Step 3: Re-extract with corrections (bounded - max 1 retry)
        log(f"Critique found issues, re-extracting...")
        log(f"Issues: {critique.critique}")
        result = self._reextract(chunk_text, header_path, critique)
        log(f"Re-extraction: {len(result.facts)} facts")

        return result

    def _extract(self, chunk_text: str, header_path: str) -> ExtractionResult:
        """Perform initial extraction with retry on parsing failures."""
        user_prompt = EXTRACTION_USER_PROMPT.format(
            header_path=header_path or "Unknown",
            chunk_text=chunk_text
        )
        messages = [
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("human", user_prompt)
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.structured_extractor.invoke(messages)
                # Response format: {"raw": AIMessage, "parsed": ExtractionResult, "parsing_error": Exception}

                if response.get("parsing_error"):
                    log(f"Attempt {attempt + 1}: Parsing error - {response['parsing_error']}")
                    continue  # Retry

                parsed = response.get("parsed")
                if parsed is not None:
                    return parsed

                log(f"Attempt {attempt + 1}: Parsed result is None, retrying...")

            except Exception as e:
                log(f"Attempt {attempt + 1}: Exception - {e}")

        log(f"All {self.max_retries} extraction attempts failed")
        return ExtractionResult(facts=[])

    def _critique(self, chunk_text: str, result: ExtractionResult) -> CritiqueResult:
        """Critique the extraction results."""
        # Format facts for review
        facts_summary = self._format_facts_for_review(result.facts)

        user_prompt = CRITIQUE_USER_PROMPT.format(
            chunk_text=chunk_text,
            facts_summary=facts_summary
        )
        messages = [
            ("system", CRITIQUE_SYSTEM_PROMPT),
            ("human", user_prompt)
        ]

        try:
            response = self.structured_critic.invoke(messages)
            # Response format: {"raw": AIMessage, "parsed": CritiqueResult, "parsing_error": Exception}

            if response.get("parsing_error") or response.get("parsed") is None:
                log(f"Critique parsing failed, assuming approved")
                return CritiqueResult(is_approved=True)

            return response["parsed"]
        except Exception as e:
            log(f"Critique error: {e}")
            # On error, assume approved to avoid infinite loops
            return CritiqueResult(is_approved=True)

    def _reextract(
        self,
        chunk_text: str,
        header_path: str,
        critique: CritiqueResult
    ) -> ExtractionResult:
        """Re-extract with critique corrections (with retry)."""
        corrections_text = "\n".join(f"- {c}" for c in critique.corrections) if critique.corrections else "None specified"
        missed_text = "\n".join(f"- {m}" for m in critique.missed_facts) if critique.missed_facts else "None specified"

        user_prompt = REEXTRACT_USER_PROMPT.format(
            chunk_text=chunk_text,
            header_path=header_path or "Unknown",
            critique=critique.critique or "General quality issues",
            corrections=corrections_text,
            missed_facts=missed_text
        )
        # Re-use the extraction system prompt for consistent behavior
        messages = [
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("human", user_prompt)
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.structured_extractor.invoke(messages)

                if response.get("parsing_error"):
                    log(f"Re-extract attempt {attempt + 1}: Parsing error - {response['parsing_error']}")
                    continue

                parsed = response.get("parsed")
                if parsed is not None:
                    return parsed

                log(f"Re-extract attempt {attempt + 1}: Parsed result is None, retrying...")

            except Exception as e:
                log(f"Re-extract attempt {attempt + 1}: Exception - {e}")

        log(f"All {self.max_retries} re-extraction attempts failed")
        return ExtractionResult(facts=[])

    def _format_facts_for_review(self, facts: list[ExtractedFact]) -> str:
        """Format facts for the critic to review."""
        lines = []
        for i, fact in enumerate(facts, 1):
            lines.append(f"{i}. FACT: {fact.fact}")
            lines.append(f"   Subject: {fact.subject} ({fact.subject_type})")
            lines.append(f"   Object: {fact.object} ({fact.object_type})")
            lines.append(f"   Relationship: {fact.relationship}")
            if fact.date_context:
                lines.append(f"   Date: {fact.date_context}")
            if fact.topics:
                lines.append(f"   Topics: {', '.join(fact.topics)}")
            lines.append("")
        return "\n".join(lines)


# Convenience function for direct use
def extract_facts(chunk_text: str, header_path: str = "") -> ExtractionResult:
    """
    Convenience function to extract facts from text.

    Args:
        chunk_text: The text to extract from
        header_path: Document header path for context

    Returns:
        ExtractionResult with list of ExtractedFact
    """
    extractor = Extractor()
    return extractor.extract(chunk_text, header_path)
