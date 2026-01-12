"""
Query Analyzer: CoT-style analysis of questions before research.

Extracts structured information to guide the supervisor and researchers:
- Entities and topics mentioned
- Question type and scope
- Search strategy recommendations
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from src.util.llm_client import get_llm


class QueryAnalysis(BaseModel):
    """Structured analysis of a user question."""

    # What to search for
    entities: list[str] = Field(
        description="Specific named entities mentioned or implied (people, orgs, places, products)"
    )
    topics: list[str] = Field(
        description="Abstract topics/themes/concepts to search for (sectors, economic indicators, etc.)"
    )

    # Question classification
    question_type: Literal[
        "specific_lookup",    # "What is X?" "Who did Y?"
        "enumeration",        # "Which districts..." "List all..."
        "comparison",         # "Compare X and Y" "How does X differ from Y"
        "aggregation",        # "How many..." "What were the overall..."
        "multi_criteria",     # "Which X had BOTH A and B"
        "temporal",           # "How did X change over time"
    ] = Field(description="Type of question being asked")

    scope: Literal[
        "specific",           # About one specific entity/district
        "multi_entity",       # About multiple specific entities
        "cross_entity",       # Comparing/aggregating across entities
        "global_summary",     # National/overall summary level
    ] = Field(description="Scope of the answer needed")

    # Search guidance
    search_strategy: str = Field(
        description="Brief description of how to approach searching for this answer"
    )

    key_criteria: list[str] = Field(
        description="The key criteria/conditions that need to be searched for"
    )

    # For multi-criteria questions
    requires_intersection: bool = Field(
        default=False,
        description="True if the question requires finding entities matching MULTIPLE criteria"
    )

    # Granularity hint
    needs_summary_level: bool = Field(
        default=False,
        description="True if the question asks for high-level summary rather than details"
    )


QUERY_ANALYSIS_PROMPT = """Analyze this question to guide knowledge graph research.

Think step-by-step:

1. **What entities are mentioned or implied?**
   - Named people, organizations, places, products
   - Be specific (e.g., "Federal Reserve Bank of Chicago" not just "Chicago")

2. **What topics/themes are relevant?**
   - Economic concepts, sectors, indicators
   - Use terms likely to appear in financial documents

3. **What TYPE of question is this?**
   - specific_lookup: Looking up info about one thing
   - enumeration: "Which X..." "List all..."
   - comparison: Comparing multiple things
   - aggregation: Summary/count across things
   - multi_criteria: Finding things matching MULTIPLE conditions
   - temporal: About change over time

4. **What SCOPE is needed?**
   - specific: Answer about one entity
   - multi_entity: Answer about several specific entities
   - cross_entity: Aggregate/compare across entities
   - global_summary: Overall/national level answer

5. **What's the search strategy?**
   - How should researchers approach finding this info?
   - What searches should run in what order?

6. **What are the key criteria to search for?**
   - Break down compound conditions into separate searchable terms

7. **Does this require finding an INTERSECTION?**
   - If question says "both X AND Y", we need to find things matching both

8. **Does this need summary-level info?**
   - Questions about "overall", "national", "general" conditions need summaries, not details

QUESTION: {question}

Provide your analysis:"""


def analyze_query(question: str) -> QueryAnalysis:
    """
    Analyze a question to extract structured guidance for research.

    Args:
        question: The user's question

    Returns:
        QueryAnalysis with entities, topics, question type, scope, and search strategy
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(QueryAnalysis)

    response = structured_llm.invoke([
        SystemMessage(content="You are an expert at analyzing questions to guide knowledge graph research."),
        HumanMessage(content=QUERY_ANALYSIS_PROMPT.format(question=question))
    ])

    return response


# === Helper to format analysis for prompts ===

def format_analysis_for_supervisor(analysis: QueryAnalysis) -> str:
    """Format the analysis into guidance for the supervisor."""
    lines = [
        "## Query Analysis",
        "",
        f"**Question Type:** {analysis.question_type}",
        f"**Scope:** {analysis.scope}",
        "",
        "**Entities to search:** " + ", ".join(analysis.entities) if analysis.entities else "None specific",
        "**Topics to search:** " + ", ".join(analysis.topics) if analysis.topics else "None specific",
        "",
        "**Key criteria:**",
    ]

    for criterion in analysis.key_criteria:
        lines.append(f"  - {criterion}")

    lines.append("")
    lines.append(f"**Search Strategy:** {analysis.search_strategy}")

    if analysis.requires_intersection:
        lines.append("")
        lines.append("⚠️ **INTERSECTION REQUIRED:** Find sources matching ALL criteria, not just one.")
        lines.append("   Search each criterion separately, then find the overlap.")

    if analysis.needs_summary_level:
        lines.append("")
        lines.append("⚠️ **SUMMARY LEVEL:** Look for national/overall summaries, not district-level details.")

    return "\n".join(lines)


def format_analysis_for_researcher(analysis: QueryAnalysis, research_topic: str) -> list[str]:
    """
    Generate search hints for a researcher based on the analysis.

    Returns list of search terms/hints.
    """
    hints = []

    # Add relevant entities
    hints.extend(analysis.entities)

    # Add relevant topics
    hints.extend(analysis.topics)

    # Add key criteria as search hints
    hints.extend(analysis.key_criteria)

    # Deduplicate while preserving order
    seen = set()
    unique_hints = []
    for h in hints:
        if h.lower() not in seen:
            seen.add(h.lower())
            unique_hints.append(h)

    return unique_hints[:10]  # Limit to top 10 hints
