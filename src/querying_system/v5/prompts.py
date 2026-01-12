"""
V5 Prompts: All LLM prompts for the Entity-Anchored Deep Research pipeline.

Centralized for easy maintenance and consistency.
"""

# =============================================================================
# Fact Scoring
# =============================================================================

SCORING_SYSTEM_PROMPT = """You are scoring facts for relevance to a question.

For each fact, assess:
1. How directly it answers the question (0-1)
2. Whether its connected entities might have more relevant information (should_expand)

Be strict: only mark facts as highly relevant (>0.7) if they directly address the question.
Mark should_expand=true for facts that reference entities likely to have more context."""

SCORING_USER_PROMPT = """QUESTION: {question}
TARGET INFO: {target_info}

FACTS TO SCORE:
{facts}

Score each fact's relevance (0-1) and whether to expand from its entities."""


# =============================================================================
# Combined Scoring + Gap Detection (Single LLM Call)
# =============================================================================

SCORING_AND_GAP_SYSTEM_PROMPT = """You are scoring facts for relevance AND identifying information gaps in a single pass.

TASK 1 - SCORE EACH FACT:
For each fact, assess:
1. relevance (0-1): How directly it answers the question
2. should_expand: Whether its connected entities might have more relevant information

Be strict: only mark facts as highly relevant (>0.7) if they directly address the question.

TASK 2 - IDENTIFY GAPS:
After scoring, analyze whether the facts sufficiently answer the question:
- What specific information is still missing?
- Which entities might have the missing information?

Be conservative: only identify clear gaps, not minor details."""

SCORING_AND_GAP_USER_PROMPT = """QUESTION: {question}
TARGET INFO: {target_info}

FACTS TO SCORE:
{facts}

1. Score each fact's relevance (0-1) and whether to expand from its entities.
2. Then analyze: Are these facts sufficient? What gaps exist?"""


# =============================================================================
# Gap Detection (Standalone - kept for backwards compatibility)
# =============================================================================

GAP_DETECTION_SYSTEM_PROMPT = """You are analyzing whether retrieved facts sufficiently answer a question.

Identify gaps in the information:
- What specific information is still missing?
- Which entities might have the missing information?

Be conservative: only identify clear gaps, not minor details."""

GAP_DETECTION_USER_PROMPT = """TARGET INFORMATION NEEDED: {target_info}

FACTS RETRIEVED:
{facts_summary}

Analyze: Are these facts sufficient? What gaps exist?

Return JSON:
{{
  "gaps": [
    {{"missing": "what specific info is missing", "expand_from": "entity name to get more info from"}}
  ],
  "sufficient": true/false
}}"""


# =============================================================================
# Sub-Answer Synthesis
# =============================================================================

SUB_ANSWER_SYSTEM_PROMPT = """You are synthesizing an answer to a focused sub-query based on retrieved facts.

Guidelines:
- Answer ONLY what the sub-query asks
- Cite evidence using [Source: doc_name, date] format
- If facts are insufficient, acknowledge uncertainty
- Keep the answer focused and concise (2-4 sentences for simple queries)
- Include specific numbers, dates, and details from the facts"""

SUB_ANSWER_USER_PROMPT = """SUB-QUERY: {sub_query}
TARGET INFO: {target_info}

EVIDENCE:
{evidence}

Synthesize a focused answer to this sub-query based on the evidence above."""


# =============================================================================
# Final Synthesis
# =============================================================================

FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are combining sub-query answers into a comprehensive final answer.

Guidelines:
- Integrate all sub-answers coherently
- Maintain citations from sub-answers
- For COMPARISON questions: structure as side-by-side comparison
- For ENUMERATION questions: use bullet points
- For CAUSAL questions: show cause-effect relationships
- For TEMPORAL questions: organize chronologically
- Acknowledge any gaps in information
- Be comprehensive but concise"""

FINAL_SYNTHESIS_USER_PROMPT = """ORIGINAL QUESTION: {question}
QUESTION TYPE: {question_type}

SUB-QUERY ANSWERS:
{sub_answers}

{gap_notes}

Synthesize a comprehensive answer to the original question by combining the sub-query answers."""


# =============================================================================
# Entity Drill-Down
# =============================================================================

DRILLDOWN_SYSTEM_PROMPT = """You are selecting entities that need more information for a complete answer.

For ENUMERATION questions, we need comprehensive coverage of all relevant entities.
Select entities from the current facts that likely have additional relevant information."""

DRILLDOWN_USER_PROMPT = """QUESTION: {question}

ENTITIES FOUND IN CURRENT FACTS:
{entities}

Which entities likely have more relevant information to answer the question comprehensively?
Return a list of entity names to retrieve more facts for."""


# =============================================================================
# Helper Functions
# =============================================================================

def format_facts_for_scoring(facts: list, max_facts: int = 30) -> str:
    """Format facts for LLM scoring prompt."""
    lines = []
    for i, fact in enumerate(facts[:max_facts]):
        lines.append(
            f"[{i}] {fact.subject} -[{fact.edge_type}]-> {fact.object}\n"
            f"    FACT: {fact.content}\n"
            f"    DATE: {fact.document_date or 'unknown'}"
        )
    return "\n\n".join(lines)


def format_facts_for_gap_detection(facts: list, max_facts: int = 15) -> str:
    """Format facts summary for gap detection prompt."""
    lines = []
    for fact in facts[:max_facts]:
        lines.append(f"- {fact.subject} {fact.edge_type} {fact.object}: {fact.content[:150]}")
    return "\n".join(lines)


def format_evidence_for_synthesis(facts: list, max_facts: int = 15) -> str:
    """Format facts as evidence for synthesis prompts."""
    lines = []
    for i, fact in enumerate(facts[:max_facts], 1):
        source = f"Source: {fact.doc_id or 'unknown'}"
        date = f", {fact.document_date}" if fact.document_date else ""
        header = f" ({fact.chunk_header})" if fact.chunk_header else ""

        lines.append(
            f"[Evidence {i}] {source}{date}{header}\n"
            f"  Fact: {fact.subject} {fact.edge_type} {fact.object}\n"
            f"  Content: {fact.content}"
        )

        # Include chunk context if available
        if fact.chunk_content:
            # Truncate chunk content for prompt efficiency
            chunk_preview = fact.chunk_content[:300]
            if len(fact.chunk_content) > 300:
                chunk_preview += "..."
            lines.append(f"  Context: {chunk_preview}")

    return "\n\n".join(lines)


def format_sub_answers_for_final(sub_answers: list) -> str:
    """Format sub-answers for final synthesis prompt."""
    lines = []
    for i, sa in enumerate(sub_answers, 1):
        lines.append(
            f"### Sub-Query {i}: {sa.sub_query}\n"
            f"**Target**: {sa.target_info}\n"
            f"**Answer**: {sa.answer}\n"
            f"**Confidence**: {sa.confidence:.2f}\n"
            f"**Entities Mentioned**: {', '.join(sa.entities_found) if sa.entities_found else 'none'}"
        )
    return "\n\n".join(lines)


def get_question_type_instructions(question_type: str) -> str:
    """Get type-specific synthesis instructions."""
    instructions = {
        "comparison": (
            "Structure your answer as a side-by-side comparison. "
            "For each aspect, show how the entities differ. "
            'Example: "In Boston, X [Source: ...]. In contrast, New York showed Y [Source: ...]."'
        ),
        "enumeration": (
            "Structure your answer as a comprehensive list. "
            "Include all relevant items found in the evidence. "
            "Use bullet points for clarity."
        ),
        "causal": (
            "Structure your answer to show cause-effect relationships. "
            "Explain what factors led to what outcomes. "
            'Use phrases like "This was caused by...", "As a result..."'
        ),
        "temporal": (
            "Structure your answer chronologically. "
            "Show how things changed over time. "
            "Include specific dates and time periods."
        ),
        "factual": (
            "Provide a direct, factual answer. "
            "Include specific details and numbers from the evidence. "
            "Cite sources for key claims."
        ),
    }
    return instructions.get(question_type.lower(), instructions["factual"])


# =============================================================================
# Vagueness Detection (Refinement Loop)
# =============================================================================

VAGUENESS_DETECTION_SYSTEM_PROMPT = """You are detecting vague references or information gaps in an answer.

Look for:
- Quantified but unnamed: numbers without corresponding names
- Generic references: "some", "various", "certain", "several" without specifics
- Unspecified lists: "other", "remaining", "a few" without enumeration
- Information discrepancies: if evidence mentions a count but the answer lists a different number of items, investigate to find the missing ones

These are VAGUE if the question asks for specifics but the answer only gives counts or generic descriptions without names.

These are NOT VAGUE:
- When the answer explicitly names the items
- When the question doesn't ask for specifics
- When the source data genuinely doesn't contain the names"""

VAGUENESS_DETECTION_USER_PROMPT = """QUESTION: {question}

EVIDENCE USED:
{evidence}

ANSWER TO CHECK:
{answer}

Compare the evidence against the answer. Look for:
1. Vague references in the answer that could be made specific
2. Counts in evidence that don't match items listed in answer (e.g., evidence says "four X" but answer only names three)
3. Information in evidence not reflected in the answer

For each gap found, suggest targeted search queries to find the missing specifics."""

REFINEMENT_SYNTHESIS_SYSTEM_PROMPT = """You are refining an answer by incorporating newly found specific information.

Guidelines:
- Replace vague references with specific names/details from the new evidence
- Maintain the original answer's structure and citations
- If the new evidence confirms specific names, list them explicitly
- If the new evidence is still vague, keep the original phrasing but note the limitation"""

REFINEMENT_SYNTHESIS_USER_PROMPT = """ORIGINAL QUESTION: {question}

ORIGINAL ANSWER (with vague references):
{original_answer}

VAGUE REFERENCES TO RESOLVE:
{vague_references}

NEW EVIDENCE FOUND:
{new_evidence}

Refine the answer by replacing vague references with specific details from the new evidence.
If specific details were found, list them explicitly.
If the new evidence doesn't provide specifics, acknowledge the limitation."""
