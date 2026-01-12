"""
Researcher Agent: Investigates specific topics in the knowledge graph.

Each researcher is spawned by the supervisor to investigate a specific aspect.
Uses graph tools to gather evidence, then compresses findings.
"""

import asyncio
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .state import ResearcherState, ResearchFinding
from .prompts import RESEARCHER_SYSTEM_PROMPT, COMPRESS_RESEARCH_PROMPT
from src.util.llm_client import get_llm, get_nano_llm

# Import existing MCP logic functions for graph operations
from src.querying_system.mcp_server import (
    _resolve_entity_or_topic_logic,
    _get_entity_info_logic,
    _explore_neighbors_logic,
    _explore_neighbors_semantic_logic,
    _get_chunk_logic,
    _get_chunks_by_edge_logic,
    _search_relationships_logic,
)


# === Graph Tools for Researcher ===

# Global user_id - will be set by create_researcher_graph
_CURRENT_USER_ID = "default"


def _get_user_id():
    return _CURRENT_USER_ID


@tool
def resolve_entity(query: str, context: str = "") -> str:
    """
    Find and match an entity name in the knowledge graph.
    Use for: people, organizations, places, products, specific named things.

    Args:
        query: The entity name or description to search for
        context: Optional context to help with resolution

    Returns:
        Matched entity names and alternatives
    """
    result = _resolve_entity_or_topic_logic(query, "Entity", _get_user_id(), context)
    if result["found"]:
        matches = result["results"][:10]
        return f"Found entities: {', '.join(matches)}"
    return f"No entity found matching '{query}'"


@tool
def resolve_topic(query: str) -> str:
    """
    Find and match a topic/theme in the knowledge graph.
    Use for: abstract concepts, themes, categories, sectors, industries.

    Args:
        query: The topic name or description to search for

    Returns:
        Matched topic names and alternatives
    """
    result = _resolve_entity_or_topic_logic(query, "Topic", _get_user_id(), "")
    if result["found"]:
        matches = result["results"][:10]
        return f"Found topics: {', '.join(matches)}"
    return f"No topic found matching '{query}'"


@tool
def explore_neighbors(entity_name: str, query_hint: str = "") -> str:
    """
    Explore relationships and connections from an entity.

    Args:
        entity_name: The resolved entity name to explore from
        query_hint: Optional hint to focus on relevant relationships

    Returns:
        List of relationships and connected entities
    """
    if query_hint:
        result = _explore_neighbors_semantic_logic(entity_name, _get_user_id(), query_hint)
    else:
        result = _explore_neighbors_logic(entity_name, _get_user_id())
    return result


@tool
def get_entity_info(entity_name: str) -> str:
    """
    Get metadata and summary for an entity.

    Args:
        entity_name: The entity to get info about

    Returns:
        Entity type, summary, and other metadata
    """
    result = _get_entity_info_logic(entity_name, _get_user_id())
    if result.get("found"):
        return f"Entity: {entity_name}\nType: {result.get('entity_type', 'Unknown')}\nSummary: {result.get('summary', 'No summary')}"
    return f"No info found for '{entity_name}'"


@tool
def get_chunks(entity_name: str, edge_type: str = "", direction: str = "both") -> str:
    """
    Retrieve evidence chunks for an entity's relationships.

    Args:
        entity_name: The entity to get chunks for
        edge_type: Optional specific relationship type
        direction: 'outgoing', 'incoming', or 'both'

    Returns:
        Evidence chunks with content and metadata
    """
    if edge_type:
        result = _get_chunks_by_edge_logic(entity_name, edge_type, _get_user_id(), direction)
    else:
        # Get all chunks for the entity
        result = _get_chunks_by_edge_logic(entity_name, "", _get_user_id(), "both")

    if not result.get("found"):
        return f"No chunks found for '{entity_name}'"

    chunks = []
    for item in result["results"][:5]:
        chunk_str = item.get("chunk", "")
        chunks.append(f"---\n{chunk_str}\n---")

    return "\n".join(chunks) if chunks else "No chunks found"


@tool
def search_facts(query: str, top_k: int = 10, date_from: str = "", date_to: str = "") -> str:
    """
    Semantic search for facts and statements in the knowledge graph.

    Results include document dates and are sorted by date (newest first).
    Use date_from/date_to to filter by time period when needed.

    Args:
        query: The search query
        top_k: Number of results to return
        date_from: Optional start date filter (YYYY-MM-DD). Only return facts from this date onwards.
        date_to: Optional end date filter (YYYY-MM-DD). Only return facts up to this date.

    Returns:
        Matching facts with dates, relevance scores, and source headers.
        Format: [DATE | Header] fact text (score: X.XX)
    """
    # Convert empty strings to None for the logic function
    date_from_param = date_from if date_from else None
    date_to_param = date_to if date_to else None

    result = _search_relationships_logic(query, _get_user_id(), top_k, date_from=date_from_param, date_to=date_to_param)
    if not result.get("found"):
        return f"No facts found matching '{query}'"

    facts = []
    for item in result["results"][:top_k]:
        fact_text = item.get('fact', '')
        score = item.get("score", 0)
        header = item.get("header", "")
        doc_date = item.get("document_date", "")

        # Build prefix with date and header for temporal/source context
        prefix_parts = []
        if doc_date:
            prefix_parts.append(doc_date)
        if header:
            prefix_parts.append(header)

        if prefix_parts:
            prefix = " | ".join(prefix_parts)
            fact_str = f"- [{prefix}] {fact_text} (score: {score:.2f})"
        else:
            fact_str = f"- {fact_text} (score: {score:.2f})"
        facts.append(fact_str)

    return "\n".join(facts) if facts else "No facts found"


# === Researcher Tools List ===

RESEARCHER_TOOLS = [resolve_entity, resolve_topic, explore_neighbors, get_entity_info, get_chunks, search_facts]


async def run_researcher(
    research_topic: str,
    search_hints: list[str] = None,
    user_id: str = "default",
    max_iterations: int = 5
) -> ResearchFinding:
    """
    Run a researcher to investigate a specific topic.

    This is a simpler, more direct implementation that doesn't use LangGraph
    to avoid state management complexity.

    Args:
        research_topic: The topic/question to investigate
        search_hints: Optional hints for entities/terms to search
        user_id: User/tenant ID for graph scoping
        max_iterations: Maximum tool call iterations

    Returns:
        ResearchFinding with compressed results
    """
    global _CURRENT_USER_ID
    _CURRENT_USER_ID = user_id

    llm = get_llm()
    llm_with_tools = llm.bind_tools(RESEARCHER_TOOLS)
    compress_llm = get_nano_llm()

    # Build initial messages
    hints_str = ", ".join(search_hints) if search_hints else "None"
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=f"Research topic: {research_topic}\n\nSearch hints: {hints_str}\n\nUse the tools to gather evidence about this topic.")
    ]

    collected_evidence = []

    # Research loop
    for iteration in range(max_iterations):
        # Get LLM response
        response = await asyncio.to_thread(llm_with_tools.invoke, messages)
        messages.append(response)

        # Check for tool calls
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            break

        # Execute tool calls
        tool_map = {t.name: t for t in RESEARCHER_TOOLS}

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in tool_map:
                try:
                    result = await asyncio.to_thread(
                        tool_map[tool_name].invoke,
                        tool_args
                    )

                    # Collect non-trivial evidence
                    if result and len(str(result)) > 50:
                        collected_evidence.append(f"[{tool_name}({tool_args})]\n{result}")

                    tool_message = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                except Exception as e:
                    tool_message = ToolMessage(
                        content=f"Error: {e}",
                        tool_call_id=tool_call["id"]
                    )
            else:
                tool_message = ToolMessage(
                    content=f"Unknown tool: {tool_name}",
                    tool_call_id=tool_call["id"]
                )

            messages.append(tool_message)

    # Compress findings
    evidence_text = "\n\n".join(collected_evidence) if collected_evidence else "No evidence collected"

    compress_prompt = f"""{COMPRESS_RESEARCH_PROMPT}

Research Topic: {research_topic}

Evidence Gathered:
{evidence_text}

Synthesize a finding for this research topic."""

    compress_response = await asyncio.to_thread(
        compress_llm.invoke,
        [HumanMessage(content=compress_prompt)]
    )

    finding = ResearchFinding(
        topic=research_topic,
        finding=compress_response.content,
        confidence=0.8 if collected_evidence else 0.3,
        evidence_chunks=[],
        raw_content=evidence_text  # Full evidence, no truncation
    )

    return finding
