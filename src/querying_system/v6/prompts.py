"""
V6 Prompts: All LLM prompts for the Threshold-Only Deep Research pipeline.

V6 CHANGES:
- REMOVED: SCORING_SYSTEM_PROMPT, SCORING_USER_PROMPT
- REMOVED: SCORING_AND_GAP_SYSTEM_PROMPT, SCORING_AND_GAP_USER_PROMPT
- REMOVED: format_facts_for_scoring()
- KEPT: All synthesis, drilldown, gap detection, vagueness prompts
"""

# =============================================================================
# Gap Detection (Standalone - used when threshold filter returns too few facts)
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
# Helper Functions (V6: removed format_facts_for_scoring)
# =============================================================================

def format_facts_for_gap_detection(facts: list, max_facts: int = 15) -> str:
    """Format facts summary for gap detection prompt."""
    lines = []
    for fact in facts[:max_facts]:
        lines.append(f"- {fact.subject} {fact.edge_type} {fact.object}: {fact.content[:150]}")
    return "\n".join(lines)


def format_evidence_for_synthesis(facts: list, max_facts: int = 15, max_chunks: int = 10) -> str:
    """
    Format evidence for synthesis by grouping facts into unique chunks.

    Key insight: Facts are used for RETRIEVAL (finding relevant chunks),
    but synthesis should see FULL CHUNKS for complete context.

    This ensures details that didn't score high individually (like "14,000 documents")
    are still visible to the synthesis LLM if other facts from the same chunk scored high.
    """
    # Group facts by chunk_id to get unique chunks
    chunks_seen = {}  # chunk_id -> {chunk_content, header, doc_id, date, facts}

    for fact in facts[:max_facts]:
        chunk_id = fact.chunk_id or fact.chunk_header or "unknown"

        if chunk_id not in chunks_seen:
            chunks_seen[chunk_id] = {
                "content": fact.chunk_content,
                "header": fact.chunk_header,
                "doc_id": fact.doc_id,
                "date": fact.document_date,
                "facts": [],
                "max_score": fact.final_score if hasattr(fact, 'final_score') else 0,
            }

        # Track facts from this chunk
        chunks_seen[chunk_id]["facts"].append(fact)

        # Track highest score for sorting
        score = fact.final_score if hasattr(fact, 'final_score') else 0
        if score > chunks_seen[chunk_id]["max_score"]:
            chunks_seen[chunk_id]["max_score"] = score

    # Sort chunks by max fact score
    sorted_chunks = sorted(
        chunks_seen.items(),
        key=lambda x: x[1]["max_score"],
        reverse=True
    )[:max_chunks]

    # Format output - FULL chunks with their facts
    lines = []
    for i, (chunk_id, chunk_data) in enumerate(sorted_chunks, 1):
        source = f"Source: {chunk_data['doc_id'] or 'unknown'}"
        date = f", {chunk_data['date']}" if chunk_data['date'] else ""
        header = chunk_data['header'] or "unknown section"

        lines.append(f"[Chunk {i}] {source}{date}")
        lines.append(f"Section: {header}")

        # Include FULL chunk content (not truncated!)
        if chunk_data["content"]:
            lines.append(f"Content:\n{chunk_data['content']}")
        else:
            # Fallback: list individual facts if chunk content not available
            lines.append("Facts from this section:")
            for fact in chunk_data["facts"]:
                lines.append(f"  - {fact.content}")

        lines.append("")  # Blank line between chunks

    return "\n".join(lines)


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


# =============================================================================
# LLM Fact Filter (Precision Improvement)
# =============================================================================

FACT_FILTER_SYSTEM_PROMPT = """You are filtering facts to find only those that directly answer a question.

Be STRICT about relevance:
- A fact is relevant ONLY if it directly helps answer the specific question asked
- Exclude tangentially related information

Key distinctions:
1. Match the exact metric asked: If the question asks about metric X, facts about a different metric Y are NOT relevant (even if Y is related to X)
2. Match the scope: If asking about "overall" or aggregate measures, exclude facts about sub-components or partial measures
3. For "which/list" questions: Only include facts that directly satisfy the criteria - not facts about related but different properties

Return the indices of ONLY the relevant facts."""

FACT_FILTER_USER_PROMPT = """QUESTION: {question}

FACTS TO FILTER:
{facts}

Which facts are directly relevant to answering this question?
Return the indices (0-based) of relevant facts only."""

FACT_FILTER_CRITIQUE_SYSTEM_PROMPT = """You are reviewing a fact filtering decision for errors.

Check for:
1. MISSED FACTS: Facts that should have been included but weren't
2. WRONG INCLUSIONS: Facts that were included but don't actually answer the question

Be thorough - the goal is high precision AND high recall."""

FACT_FILTER_CRITIQUE_USER_PROMPT = """QUESTION: {question}

ALL FACTS:
{facts}

FACTS SELECTED AS RELEVANT (indices): {selected_indices}

Review this selection. Were any relevant facts missed? Were any irrelevant facts incorrectly included?"""
