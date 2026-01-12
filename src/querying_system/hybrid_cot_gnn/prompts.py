"""
All prompts for the Hybrid CoT-GNN Query Pipeline.
Centralized for easy tuning and version control.
"""

from .schemas import QuestionType


# =============================================================================
# PHASE 1: DECOMPOSITION PROMPTS
# =============================================================================

DECOMPOSITION_SYSTEM_PROMPT = """You are a query parser extracting structure from questions for knowledge graph retrieval.

STEP 1 - EXTRACT ENTITIES:
Extract named things EXPLICITLY mentioned in the question text.
- Include only what is written - do not enumerate or expand from your knowledge
- Generic references like "districts" or "companies" stay generic
- Specific names like "Boston" or "Federal Reserve" are extracted as-is

STEP 2 - EXTRACT TOPICS:
Extract concepts and themes mentioned in the question.
- Economic concepts: growth, inflation, wages, employment, spending
- Sector concepts: manufacturing, retail, housing, services

STEP 3 - EXTRACT RELATIONSHIPS:
Extract action verbs and their modifiers as a single relationship phrase.
- Include qualifiers: "increased slightly", "declined modestly", "remained unchanged"
- Include manner: "reported", "experienced", "saw"
- The relationship captures HOW entities relate to topics

STEP 4 - IDENTIFY TEMPORAL SCOPE:
Extract any time references mentioned in the question.
- Specific dates, quarters, months, years
- Relative references: "recent", "previous", "last"

STEP 5 - CLASSIFY QUESTION TYPE:
- FACTUAL: Questions about specific facts or states
- COMPARISON: Questions comparing multiple entities
- CAUSAL: Questions about cause and effect
- TEMPORAL: Questions about change over time
- ENUMERATION: Questions asking which/what entities match criteria

STEP 6 - GENERATE SUB-QUERIES:
Create keyword-focused search queries combining extracted elements.
- Combine entities + topics + relationship keywords
- For comparison questions, generate separate queries per entity mentioned

IMPORTANT RULES:
- Extract ONLY what appears in the question text
- Do NOT add entities from your knowledge
- Do NOT enumerate or expand generic references
- Preserve exact phrasing for relationships and modifiers
- Sub-queries should be keyword phrases, not full sentences"""


DECOMPOSITION_USER_PROMPT = """Analyze this question and output the structured decomposition:

QUESTION: {question}

Think step-by-step:
1. What distinct pieces of information are needed?
2. What focused searches would find each piece?
3. What entities and topics should be looked up directly?
4. Is there a time constraint?
5. What type of question is this?

Output the structured decomposition."""


# =============================================================================
# PHASE 2b: SCORING PROMPT
# =============================================================================

SCORING_PROMPT = """Score the following facts for relevance to the question.

QUESTION: {question}

REQUIRED INFORMATION:
{required_info}

CANDIDATE FACTS:
{facts_text}

For each fact, provide:
- fact_index: The [index] number from the list
- relevance: 0.0-1.0 score (1.0 = directly answers the question)
- should_expand: {expansion_hint}

Scoring guidelines:
- 1.0: Directly answers a required_info item
- 0.7-0.9: Strongly relevant context or supporting evidence
- 0.4-0.6: Somewhat relevant, provides background
- 0.1-0.3: Tangentially related
- 0.0: Not relevant at all

Output JSON with scores array."""


# =============================================================================
# PHASE 3: SYNTHESIS PROMPTS
# =============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are a financial research synthesizer. Your job is to create accurate, well-cited answers from evidence.

## Core Principles

1. USE ONLY THE PROVIDED EVIDENCE
   - Do not use outside knowledge
   - Do not make assumptions beyond what the evidence states
   - If evidence is insufficient, say so clearly

2. CITE EVERY CLAIM
   - Format: [Source: header_path, Date: date]
   - Every factual statement must have a citation
   - Multiple citations OK for synthesized statements

3. BE PRECISE
   - Use exact numbers/dates from evidence
   - Don't round or estimate unless evidence is unclear
   - Quote directly when precision matters

4. MATCH FORMAT TO QUESTION TYPE
   - Follow the type-specific instructions provided

5. ACKNOWLEDGE GAPS
   - If required information is missing, note what couldn't be found
   - Distinguish between "no information" and "information not found"
"""


SYNTHESIS_USER_PROMPT = """Answer the following question using ONLY the provided evidence.

QUESTION: {question}

REQUIRED INFORMATION:
{required_info}

{type_instructions}

EVIDENCE:
{evidence}
{gaps_note}

Provide a comprehensive answer with citations for each claim."""


def get_question_type_instructions(question_type: QuestionType) -> str:
    """Get synthesis instructions specific to question type."""

    instructions = {
        QuestionType.FACTUAL: """
FORMAT INSTRUCTIONS (FACTUAL):
- Start with a direct answer
- Follow with supporting details
- Each claim needs a citation""",
        QuestionType.COMPARISON: """
FORMAT INSTRUCTIONS (COMPARISON):
- Structure as side-by-side comparison
- Cover the same aspects for each entity being compared
- Highlight similarities and differences
- Use parallel structure when possible
- Example: "In Boston, X [Source: ...]. In contrast, New York showed Y [Source: ...]." """,
        QuestionType.CAUSAL: """
FORMAT INSTRUCTIONS (CAUSAL):
- Trace the chain of cause and effect
- Identify the root causes if evident
- Explain the mechanism/relationship between cause and effect
- Be careful not to infer causation where only correlation exists
- Look for relationships like: A led to B, which caused C""",
        QuestionType.TEMPORAL: """
FORMAT INSTRUCTIONS (TEMPORAL):
- Organize chronologically when appropriate
- Note specific dates/periods for each fact
- Highlight changes and trends over time
- If comparing periods, make the comparison explicit""",
        QuestionType.ENUMERATION: """
FORMAT INSTRUCTIONS (ENUMERATION):
- Use a numbered or bulleted list
- Be exhaustive - include all items found in evidence
- Provide brief context for each item
- If the list might be incomplete, note that""",
    }

    return instructions.get(question_type, instructions[QuestionType.FACTUAL])
