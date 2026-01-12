"""
MODULE: Extractor V2 (Chain-of-Thought)
DESCRIPTION: Experimental extractor that uses structured chain-of-thought.

Key difference from V1:
- Single LLM call with TWO-STEP structured output
- Step 1: Enumerate ALL entities in the text
- Step 2: Generate relationships between enumerated entities

Hypothesis: Forcing entity enumeration before relationship generation
may improve entity coverage (e.g., catch Chronicle Security).

LLM Calls: 2-3 per chunk (same as V1)
  - Extraction: 1 call (with CoT structure)
  - Critique: 1 call
  - Re-extract (if needed): 0-1 call
"""

import os
from typing import Optional, List, TYPE_CHECKING

from src.schemas.extraction import (
    ExtractedFact,
    ExtractionResult,
    CritiqueResult,
    EnumeratedEntity,
    ChainOfThoughtResult,
)
from src.util.llm_client import get_critique_llm

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

# Control verbose output
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


def log(msg: str):
    """Print only if VERBOSE mode is enabled."""
    if VERBOSE:
        print(f"[ExtractorV2] {msg}")


# ============================================================================
# PROMPTS - Chain-of-Thought Extraction
# ============================================================================

COT_EXTRACTION_SYSTEM_PROMPT = """You are a financial analyst building a knowledge graph. Extract information in TWO STEPS:

STEP 1 - ENUMERATE ENTITIES:
First, list EVERY named entity mentioned in the text AND the Section header:
- Companies, subsidiaries, divisions
- People (by full name)
- Organizations (non-corporate: Fed, SEC, etc.)
- Locations
- Products or services
- Metric categories as Topics (e.g., "Market Valuation", "Revenue", "R&D Expenditure", "Market Share")

NOTE: The Section header (format: "X > Y") shows Y is a subheader of X. Extract entities from the header too.

For each entity, provide:
- name: Exact name as it appears in the text
- entity_type: Company, Person, Organization, Location, Product, Topic
- summary: 1-2 sentence description based on the text

STEP 2 - GENERATE RELATIONSHIPS:
Using ONLY the entities from Step 1, identify explicit relationships:
- Subject and Object MUST come from your entity list
- Write a complete, self-contained fact for each relationship
- Relationship: normalized action verb (e.g., "reached milestone", "has value", "acquired")
- Only include relationships EXPLICITLY stated in the text

HANDLING NUMERIC METRICS:
When text mentions specific numbers, percentages, or rankings about an entity:
- Subject: The entity being measured
- Object: The metric CATEGORY as a Topic (not the specific number)
- Relationship: Normalized verb describing the action
- Fact: The complete statement including the specific values

DATE CONTEXT (REQUIRED FOR EVERY FACT):
- If a specific date/time is mentioned in the text for this fact, use it (e.g., "Q3 2024", "August 5, 2024")
- If no specific date mentioned, use "Document date: {document_date}" as fallback
- ALL facts MUST have date_context populated - this enables temporal queries
- NEVER leave date_context empty

IMPORTANT RULES:
- Ensure you extract ALL events/important information from the chunk
- Subsidiaries are SEPARATE entities (AWS != Amazon)
- Each relationship = one atomic fact
- Preserve exact entity names from the text
- Don't infer relationships that aren't directly stated
- Specific numbers, percentages, and dates are NOT entities - they belong in the fact text
- Use Topic entities for metric categories that numbers refer to
- Subject and Object are CLEAN entity names without any descriptors or qualifiers
- Descriptors and qualifiers belong in the relationship field, not in subject/object names
- Numbers and values are evidence/data - they belong only in the fact description, not in the relationship or subject/object
- No parentheses in relationship or subject/object names

EXCLUSIONS - DO NOT EXTRACT AS ENTITIES:
- URLs (e.g., https://..., www....)
- Specific dollar amounts, percentages, or rankings (these go in the fact text only)
- Dates and time periods (these go in the fact text only)
- Article titles or headlines from reference/citation sections
- Citation metadata (archive URLs, retrieval dates, "(PDF)" markers)
- Author names from citations/references (unless they are subjects of the main text)
- Page titles like "Company Name on Forbes" or "Company Name on Wikipedia"
- Any text that is clearly bibliographic/reference metadata"""

COT_EXTRACTION_USER_PROMPT = """DOCUMENT CONTEXT:
Document Date: {document_date}

CHUNK TEXT:
Section: {header_path}

{chunk_text}

First enumerate ALL entities (including any entities mentioned in the Section header - the ">" denotes subheader hierarchy), then generate relationships between them.
Remember: ALL facts MUST have date_context. Use "{document_date}" as fallback if no specific date in text."""


# --- CRITIQUE PROMPTS (same as V1) ---

CRITIQUE_SYSTEM_PROMPT = """You are a senior financial analyst reviewing extraction results for accuracy.

REVIEW CHECKLIST:

1. ENTITY COVERAGE: Were ALL named entities captured in the enumeration?
   - Every company, person, organization mentioned
   - Including subsidiaries, divisions, products

2. SENTENCE-LEVEL COMPLETENESS (CRITICAL):
   Go through EACH sentence in the original text and verify:
   - Does every clause/claim in that sentence have a corresponding extracted fact?
   - Compound sentences (with "and", "while", "but", "although") often contain MULTIPLE facts
   - Example: "X increased, while Y decreased" = TWO facts, not one
   - If ANY clause is missing a fact, flag it as a missed_fact

3. RELATIONSHIP COMPLETENESS: Did we capture all material interactions?
   - Every action, acquisition, partnership, filing, etc.
   - Every financial metric or event
   - Every relationship

4. ACCURACY: Are extractions correct?
   - Entity names match the text exactly
   - Relationships accurately describe the interaction
   - No hallucinated information

5. SELF-CONTAINMENT: Can each fact be understood alone?
   - No pronouns without antecedents
   - Complete propositions

6. CLEAN STRUCTURE: Are subjects, objects, and relationships properly formed?
   - Subject and Object are CLEAN entity names without any descriptors or qualifiers
   - Descriptors and qualifiers belong in the relationship field, not in subject/object names
   - Numbers and values belong only in the fact description, not in the relationship or subject/object
   - No parentheses in relationship or subject/object names

7. EXCLUSION CHECK: Flag any entities that should NOT have been extracted:
   - URLs (https://..., www....)
   - Article titles from reference/citation sections
   - Citation metadata (archive URLs, author names from references)
   - Page titles like "Company on Forbes" or "Company on Wikipedia"
   - Bibliographic/reference metadata

If ALL criteria are met, mark as APPROVED.
If ANY issues found (including invalid entities that should be excluded), provide SPECIFIC corrections."""

CRITIQUE_USER_PROMPT = """ORIGINAL TEXT:
{chunk_text}

ENUMERATED ENTITIES:
{entities_summary}

EXTRACTED FACTS:
{facts_summary}

Review both the entity enumeration and the relationships for accuracy."""


# --- RE-EXTRACTION PROMPTS ---

REEXTRACT_USER_PROMPT = """A senior analyst has reviewed your extraction and found issues.

ORIGINAL TEXT:
{chunk_text}

DOCUMENT CONTEXT:
Header Path: {header_path}
Document Date: {document_date}

PREVIOUS EXTRACTION ISSUES:
{critique}

SPECIFIC CORRECTIONS NEEDED:
{corrections}

MISSED ENTITIES/FACTS TO ADD:
{missed_facts}

Please re-extract following the two-step process:
1. First enumerate ALL entities (including any missed ones)
2. Then generate relationships between them

Address ALL the issues above.
Remember: ALL facts MUST have date_context. Use "{document_date}" as fallback if no specific date in text."""


class ExtractorV2:
    """
    Chain-of-thought extractor: enumerate entities, then generate relationships.
    All in ONE LLM call with structured output.
    """

    def __init__(self, llm: Optional["BaseChatModel"] = None, max_retries: int = 2):
        if llm is None:
            from src.util.services import get_services
            llm = get_services().llm
        self.llm = llm
        self.max_retries = max_retries

        # Use main LLM (gpt-5.2) for extraction
        self.structured_extractor = llm.with_structured_output(ChainOfThoughtResult, include_raw=True)

        # Use gpt-5.1 for critique (slightly cheaper, still capable)
        self.critique_llm = get_critique_llm()
        self.structured_critic = self.critique_llm.with_structured_output(CritiqueResult, include_raw=True)

    def extract(self, chunk_text: str, header_path: str = "", document_date: str = "") -> ChainOfThoughtResult:
        """
        Main extraction method with chain-of-thought and reflexion.

        Args:
            chunk_text: The text to extract from
            header_path: Document header path for context
            document_date: Document date (YYYY-MM-DD) for temporal fallback

        Returns:
            ChainOfThoughtResult with entities and facts
        """
        # Step 1: Initial extraction with CoT
        log(f"Extracting from chunk ({len(chunk_text)} chars)")
        result = self._extract(chunk_text, header_path, document_date)
        log(f"Enumerated {len(result.entities)} entities, extracted {len(result.facts)} facts")

        if not result.entities and not result.facts:
            log("No entities or facts extracted, skipping critique")
            return result

        # Step 2: Critique
        critique = self._critique(chunk_text, result)

        if critique.is_approved:
            log("Extraction approved by critic")
            return result

        # Step 3: Re-extract with corrections (bounded - max 1 retry)
        log(f"Critique found issues, re-extracting...")
        log(f"Issues: {critique.critique}")
        result = self._reextract(chunk_text, header_path, document_date, critique)
        log(f"Re-extraction: {len(result.entities)} entities, {len(result.facts)} facts")

        return result

    def extract_to_result(self, chunk_text: str, header_path: str = "") -> ExtractionResult:
        """
        Extract and return ExtractionResult for compatibility with V1.
        Discards the entity enumeration, keeps only facts.
        """
        cot_result = self.extract(chunk_text, header_path)
        return ExtractionResult(facts=cot_result.facts)

    def _extract(self, chunk_text: str, header_path: str, document_date: str) -> ChainOfThoughtResult:
        """Perform initial extraction with chain-of-thought."""
        user_prompt = COT_EXTRACTION_USER_PROMPT.format(
            header_path=header_path or "Unknown",
            document_date=document_date or "Unknown",
            chunk_text=chunk_text
        )
        messages = [
            ("system", COT_EXTRACTION_SYSTEM_PROMPT),
            ("human", user_prompt)
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.structured_extractor.invoke(messages)

                if response.get("parsing_error"):
                    log(f"Attempt {attempt + 1}: Parsing error - {response['parsing_error']}")
                    continue

                parsed = response.get("parsed")
                if parsed is not None:
                    return parsed

                log(f"Attempt {attempt + 1}: Parsed result is None, retrying...")

            except Exception as e:
                log(f"Attempt {attempt + 1}: Exception - {e}")

        log(f"All {self.max_retries} extraction attempts failed")
        return ChainOfThoughtResult(entities=[], facts=[])

    def _critique(self, chunk_text: str, result: ChainOfThoughtResult) -> CritiqueResult:
        """Critique both the entity enumeration and extracted facts."""
        entities_summary = self._format_entities_for_review(result.entities)
        facts_summary = self._format_facts_for_review(result.facts)

        user_prompt = CRITIQUE_USER_PROMPT.format(
            chunk_text=chunk_text,
            entities_summary=entities_summary,
            facts_summary=facts_summary
        )
        messages = [
            ("system", CRITIQUE_SYSTEM_PROMPT),
            ("human", user_prompt)
        ]

        try:
            response = self.structured_critic.invoke(messages)

            if response.get("parsing_error") or response.get("parsed") is None:
                log(f"Critique parsing failed, assuming approved")
                return CritiqueResult(is_approved=True)

            return response["parsed"]
        except Exception as e:
            log(f"Critique error: {e}")
            return CritiqueResult(is_approved=True)

    def _reextract(
        self,
        chunk_text: str,
        header_path: str,
        document_date: str,
        critique: CritiqueResult
    ) -> ChainOfThoughtResult:
        """Re-extract with critique corrections."""
        corrections_text = "\n".join(f"- {c}" for c in critique.corrections) if critique.corrections else "None specified"
        missed_text = "\n".join(f"- {m}" for m in critique.missed_facts) if critique.missed_facts else "None specified"

        user_prompt = REEXTRACT_USER_PROMPT.format(
            chunk_text=chunk_text,
            header_path=header_path or "Unknown",
            document_date=document_date or "Unknown",
            critique=critique.critique or "General quality issues",
            corrections=corrections_text,
            missed_facts=missed_text
        )
        messages = [
            ("system", COT_EXTRACTION_SYSTEM_PROMPT),
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
        return ChainOfThoughtResult(entities=[], facts=[])

    def _format_entities_for_review(self, entities: List[EnumeratedEntity]) -> str:
        """Format enumerated entities for the critic to review."""
        if not entities:
            return "No entities enumerated."

        lines = []
        for i, entity in enumerate(entities, 1):
            lines.append(f"{i}. {entity.name} ({entity.entity_type})")
            if entity.summary:
                lines.append(f"   Summary: {entity.summary}")
        return "\n".join(lines)

    def _format_facts_for_review(self, facts: List[ExtractedFact]) -> str:
        """Format facts for the critic to review."""
        if not facts:
            return "No facts extracted."

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
def extract_facts_v2(chunk_text: str, header_path: str = "", document_date: str = "") -> ChainOfThoughtResult:
    """
    Convenience function to extract facts using chain-of-thought.

    Args:
        chunk_text: The text to extract from
        header_path: Document header path for context
        document_date: Document date (YYYY-MM-DD) for temporal fallback

    Returns:
        ChainOfThoughtResult with entities and facts
    """
    extractor = ExtractorV2()
    return extractor.extract(chunk_text, header_path, document_date)
