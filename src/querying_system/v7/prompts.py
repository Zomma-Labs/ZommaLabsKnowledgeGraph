"""
V7 Prompts: All LLM prompts for the GraphRAG-aligned query pipeline.

V7 Design Principles:
- Entity/topic resolution casts a WIDE NET (one-to-many matching)
- Structured context sections for sub-answer synthesis
- Question-type-specific final synthesis formatting
"""

# =============================================================================
# Entity Resolution Prompts
# =============================================================================

ENTITY_RESOLUTION_SYSTEM_PROMPT = """You are resolving query terms to entities in a knowledge graph.

Your goal is to CAST A WIDE NET - find ALL entities that could be relevant to the query term.

Key principles:
1. ONE-TO-MANY MATCHING: A single query term may map to MULTIPLE entities
   - "tech companies" -> Apple, Microsoft, Google, Amazon, etc.
   - "Federal Reserve officials" -> Jerome Powell, multiple Fed governors
   - "inflation" -> may match entities discussing inflation across regions

2. INCLUDE PARTIAL MATCHES: If the term partially matches an entity, include it
   - "Apple" should match "Apple Inc." and "Apple Park"
   - "Boston" should match "Federal Reserve Bank of Boston" and "Boston Metro Area"

3. PREFER RECALL OVER PRECISION: It's better to include marginally relevant entities
   than to miss important ones. The retrieval system will filter later.

4. CONSIDER ALIASES AND VARIATIONS: Match different forms of the same concept
   - "Fed" -> "Federal Reserve", "Federal Reserve System", regional Fed banks

Return entities that should be searched for relevant information."""

TOPIC_RESOLUTION_SYSTEM_PROMPT = """You are resolving query terms to topics in a knowledge graph ontology.

Your goal is to CAST A WIDE NET - find ALL topics that could provide relevant context.

Key principles:
1. ONE-TO-MANY MATCHING: A query term may relate to MULTIPLE topics
   - "economic conditions" -> Inflation, Employment, GDP Growth, Consumer Spending
   - "financial performance" -> Revenue, Profitability, Cash Flow, Debt

2. HIERARCHICAL MATCHING: Include both specific and general topics
   - Query about "housing prices" -> Real Estate, Housing Market, Inflation, Consumer Economy

3. THEMATIC CONNECTIONS: Include topics that provide contextual understanding
   - "Fed rate decisions" -> Monetary Policy, Interest Rates, Inflation, Economic Outlook

4. PREFER RECALL OVER PRECISION: Better to include tangentially related topics
   than to miss important thematic context.

Return topics that should be searched for relevant chunks."""

RESOLUTION_USER_PROMPT = """QUESTION: {question}

TERM TO RESOLVE: {term}

CANDIDATE MATCHES (name, summary, similarity score):
{candidates}

Select ALL candidates that could be relevant to answering the question.
Consider the term's role in the question context - not just string matching.

Return the names of relevant candidates. If none are relevant, return an empty list."""


# =============================================================================
# Sub-Answer Synthesis Prompts
# =============================================================================

SUB_ANSWER_SYSTEM_PROMPT = """You are synthesizing an answer to a sub-query using evidence from a knowledge graph.

Answer the sub-query based on the provided context. Be direct and specific.

Guidelines:
- Answer what the sub-query asks - include specific numbers, dates, names
- Cite sources using [Source: doc_name, date] format
- If the context contains the answer, extract and report it
- Only say you cannot answer if the context truly contains no relevant information

Use the 'thinking' field to briefly note which evidence supports your answer."""

SUB_ANSWER_USER_PROMPT = """SUB-QUERY: {sub_query}
TARGET INFORMATION: {target_info}

{context}

Synthesize a focused answer to this sub-query based on the evidence above.
Be specific and cite your sources."""


# =============================================================================
# Final Synthesis Prompts
# =============================================================================

FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are combining sub-query answers into a comprehensive final answer.

Your task is to integrate multiple sub-answers into a coherent, well-structured response.

General guidelines:
- Integrate all sub-answers coherently - don't just concatenate them
- Maintain and propagate citations from sub-answers
- Resolve any contradictions between sub-answers explicitly
- Acknowledge gaps where sub-answers indicate insufficient information
- Be comprehensive but avoid redundancy

QUESTION-TYPE-SPECIFIC FORMATTING:

For COMPARISON questions:
- Structure as side-by-side comparison
- For each aspect, show how entities/periods differ
- Use parallel structure for clarity
- Example: "In Boston, X [Source]. In contrast, New York showed Y [Source]."

For ENUMERATION questions:
- Use clear bullet points or numbered lists
- Ensure comprehensive coverage of all items found
- Group related items logically
- Indicate if the list may be incomplete

For CAUSAL questions:
- Show cause-effect relationships explicitly
- Use connective phrases: "This was caused by...", "As a result...", "Leading to..."
- Distinguish between correlation and causation where evidence permits

For TEMPORAL questions:
- Organize chronologically
- Include specific dates and time periods
- Show progression and changes over time
- Highlight key turning points

For FACTUAL questions:
- Lead with the direct answer
- Follow with supporting evidence
- Include specific numbers and details
- Cite sources for all key claims"""

FINAL_SYNTHESIS_USER_PROMPT = """ORIGINAL QUESTION: {question}
QUESTION TYPE: {question_type}

SUB-QUERY ANSWERS:
{sub_answers}

{type_instructions}

Synthesize a comprehensive answer to the original question by integrating the sub-query answers above."""


# =============================================================================
# Helper Functions
# =============================================================================

def format_candidates_for_resolution(candidates: list[tuple[str, str, float]]) -> str:
    """
    Format candidate matches for the resolution prompt.

    Args:
        candidates: List of (name, summary, score) tuples from vector search

    Returns:
        Formatted string for inclusion in prompt
    """
    if not candidates:
        return "No candidates found."

    lines = []
    for i, (name, summary, score) in enumerate(candidates, 1):
        # Truncate long summaries
        summary_display = summary[:200] + "..." if len(summary) > 200 else summary
        if not summary_display:
            summary_display = "(no summary available)"
        lines.append(f"{i}. {name} (score: {score:.3f})")
        lines.append(f"   Summary: {summary_display}")

    return "\n".join(lines)


def format_sub_answers_for_final(sub_answers: list) -> str:
    """
    Format sub-answers for the final synthesis prompt.

    Args:
        sub_answers: List of SubAnswer objects

    Returns:
        Formatted string for inclusion in prompt
    """
    if not sub_answers:
        return "No sub-answers available."

    lines = []
    for i, sa in enumerate(sub_answers, 1):
        # Handle both dataclass objects and dicts
        if hasattr(sa, 'sub_query'):
            sub_query = sa.sub_query
            target_info = sa.target_info
            answer = sa.answer
            confidence = sa.confidence
            entities_found = sa.entities_found if hasattr(sa, 'entities_found') else []
        else:
            # Dict fallback
            sub_query = sa.get('sub_query', sa.get('query', ''))
            target_info = sa.get('target_info', '')
            answer = sa.get('answer', '')
            confidence = sa.get('confidence', 0.0)
            entities_found = sa.get('entities_found', [])

        lines.append(f"### Sub-Query {i}: {sub_query}")
        lines.append(f"**Target**: {target_info}")
        lines.append(f"**Answer**: {answer}")
        lines.append(f"**Confidence**: {confidence:.2f}")

        if entities_found:
            lines.append(f"**Entities Mentioned**: {', '.join(entities_found)}")
        else:
            lines.append("**Entities Mentioned**: none")

        lines.append("")  # Blank line between sub-answers

    return "\n".join(lines)


def get_question_type_instructions(question_type: str) -> str:
    """
    Get type-specific synthesis instructions for the final synthesis prompt.

    Args:
        question_type: The classified question type (e.g., "COMPARISON", "ENUMERATION")

    Returns:
        Instruction string for the specific question type
    """
    instructions = {
        "comparison": (
            "FORMATTING INSTRUCTIONS:\n"
            "Structure your answer as a side-by-side comparison. "
            "For each aspect discussed, show how the entities or time periods differ. "
            "Use parallel structure to make differences clear.\n"
            'Example format: "Regarding X: In Boston, [finding] [Source]. '
            'In contrast, New York showed [finding] [Source]."'
        ),
        "enumeration": (
            "FORMATTING INSTRUCTIONS:\n"
            "Structure your answer as a comprehensive bulleted or numbered list. "
            "Include ALL relevant items found across the sub-answers. "
            "Group related items together and use consistent formatting. "
            "If the list may be incomplete, note this explicitly."
        ),
        "causal": (
            "FORMATTING INSTRUCTIONS:\n"
            "Structure your answer to clearly show cause-effect relationships. "
            "Explain what factors led to what outcomes, using connective phrases like "
            '"This was caused by...", "As a result...", "Which in turn led to...". '
            "Distinguish between established causation and correlation."
        ),
        "temporal": (
            "FORMATTING INSTRUCTIONS:\n"
            "Structure your answer chronologically to show progression over time. "
            "Include specific dates and time periods where available. "
            "Highlight key turning points and transitions. "
            "Show how conditions evolved from earlier to later periods."
        ),
        "factual": (
            "FORMATTING INSTRUCTIONS:\n"
            "Lead with a direct, clear answer to the question. "
            "Follow with supporting evidence and specific details. "
            "Include numbers, dates, and proper names where relevant. "
            "Cite sources for all key factual claims."
        ),
        "analytical": (
            "FORMATTING INSTRUCTIONS:\n"
            "Structure your answer to show analysis and reasoning. "
            "Present evidence, then draw conclusions. "
            "Acknowledge multiple perspectives if the evidence suggests them. "
            "Distinguish between what the evidence shows and inferences drawn."
        ),
    }

    # Normalize the question type for lookup
    normalized_type = question_type.lower().strip() if question_type else "factual"

    return instructions.get(normalized_type, instructions["factual"])


def format_context_sections(
    entities: list = None,
    high_relevance_chunks: list = None,
    facts: list = None,
    topic_chunks: list = None,
    low_relevance_chunks: list = None,
) -> str:
    """
    Format structured context for sub-answer synthesis.

    This creates the structured context format expected by SUB_ANSWER_SYSTEM_PROMPT:
    - PRIMARY EVIDENCE (entity summaries)
    - STRUCTURED CONTEXT (high-relevance chunks)
    - THEMATIC CONTEXT (topic chunks)
    - SUPPORTING EVIDENCE (low-relevance chunks, facts)

    Args:
        entities: List of RetrievedEntity objects
        high_relevance_chunks: List of RetrievedChunk objects
        facts: List of RetrievedFact objects
        topic_chunks: List of RetrievedChunk objects
        low_relevance_chunks: List of RetrievedChunk objects

    Returns:
        Formatted context string
    """
    sections = []

    # PRIMARY EVIDENCE - Entity summaries
    if entities:
        entity_lines = ["## PRIMARY EVIDENCE (Entity Information)"]
        for e in entities:
            if hasattr(e, 'summary') and e.summary:
                entity_lines.append(f"- **{e.name}** ({e.entity_type}): {e.summary}")
            elif hasattr(e, 'name'):
                entity_lines.append(f"- **{e.name}** ({getattr(e, 'entity_type', 'UNKNOWN')})")
        sections.append("\n".join(entity_lines))

    # STRUCTURED CONTEXT - High-relevance chunks
    if high_relevance_chunks:
        chunk_lines = ["## STRUCTURED CONTEXT (High-Relevance Sources)"]
        for c in high_relevance_chunks:
            header = f"[{c.header_path}]" if hasattr(c, 'header_path') and c.header_path else ""
            date = f" ({c.document_date})" if hasattr(c, 'document_date') and c.document_date else ""
            source = getattr(c, 'source', '')
            content = c.content if hasattr(c, 'content') else str(c)

            chunk_lines.append(f"\n### {header}{date}")
            if source:
                chunk_lines.append(f"*Retrieved via: {source}*")
            chunk_lines.append(content)
        sections.append("\n".join(chunk_lines))

    # THEMATIC CONTEXT - Topic-related chunks
    if topic_chunks:
        chunk_lines = ["## THEMATIC CONTEXT (Topic-Related Sources)"]
        for c in topic_chunks:
            header = f"[{c.header_path}]" if hasattr(c, 'header_path') and c.header_path else ""
            date = f" ({c.document_date})" if hasattr(c, 'document_date') and c.document_date else ""
            content = c.content if hasattr(c, 'content') else str(c)

            chunk_lines.append(f"\n### {header}{date}")
            chunk_lines.append(content)
        sections.append("\n".join(chunk_lines))

    # SUPPORTING EVIDENCE - Facts and low-relevance chunks
    supporting_lines = []

    if facts:
        supporting_lines.append("## SUPPORTING EVIDENCE (Extracted Facts)")
        for f in facts:
            if hasattr(f, 'content'):
                supporting_lines.append(f"- {f.subject} {f.edge_type} {f.object}: {f.content}")
            else:
                supporting_lines.append(f"- {f}")

    if low_relevance_chunks:
        if supporting_lines:
            supporting_lines.append("")  # Blank line
        supporting_lines.append("## ADDITIONAL CONTEXT (Lower Relevance)")
        for c in low_relevance_chunks:
            header = f"[{c.header_path}]" if hasattr(c, 'header_path') and c.header_path else ""
            date = f" ({c.document_date})" if hasattr(c, 'document_date') and c.document_date else ""
            content = c.content if hasattr(c, 'content') else str(c)

            supporting_lines.append(f"\n### {header}{date}")
            supporting_lines.append(content)

    if supporting_lines:
        sections.append("\n".join(supporting_lines))

    if not sections:
        return "No context available."

    return "\n\n".join(sections)
