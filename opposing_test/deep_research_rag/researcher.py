"""
Researcher that uses simple RAG instead of knowledge graph.
"""

import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from .rag_store import RAGStore
from src.util.llm_client import get_llm, get_nano_llm


class ResearchFinding(BaseModel):
    topic: str
    finding: str
    confidence: float
    raw_content: str = ""


RESEARCHER_PROMPT = """You are a research assistant investigating a specific topic.

Your task: Search for relevant information and gather evidence about your assigned topic.

## Available Tool

- `search_chunks`: Semantic search over document chunks. Returns relevant text passages with their source headers.

## Process

1. Break down your topic into key search terms
2. Run multiple searches with different phrasings
3. Extract relevant facts from the results
4. Note which sources (headers) the information comes from

After gathering evidence, summarize your findings."""


COMPRESS_PROMPT = """Compress your research findings into a structured summary.

From the evidence gathered, extract:
1. The key finding/answer to your research topic
2. Your confidence level (0-1)
3. The most important supporting evidence with source attribution

Be concise. Focus on facts directly relevant to the research topic."""


# Global RAG store
_rag_store = None


def get_rag_store():
    global _rag_store
    if _rag_store is None:
        _rag_store = RAGStore()
    return _rag_store


@tool
def search_chunks(query: str, num_results: int = 10) -> str:
    """
    Search for relevant document chunks.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        Matching chunks with headers and relevance scores
    """
    store = get_rag_store()
    results = store.search(query, k=num_results)

    if not results:
        return f"No results found for '{query}'"

    output = []
    for r in results:
        header = r.get("header", "Unknown")
        text = r.get("text", "")[:500]  # Truncate for readability
        score = r.get("score", 0)
        output.append(f"[{header}] (score: {score:.2f})\n{text}")

    return "\n\n---\n\n".join(output)


RESEARCHER_TOOLS = [search_chunks]


async def run_researcher(
    research_topic: str,
    search_hints: list[str] = None,
    max_iterations: int = 5
) -> ResearchFinding:
    """
    Run a researcher to investigate a topic using simple RAG.
    """
    # Ensure RAG store is initialized (blocking call - run in thread)
    await asyncio.to_thread(get_rag_store)

    llm = get_llm()
    llm_with_tools = llm.bind_tools(RESEARCHER_TOOLS)
    compress_llm = get_nano_llm()

    hints_str = ", ".join(search_hints) if search_hints else "None"
    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"Research topic: {research_topic}\n\nSearch hints: {hints_str}\n\nUse the search tool to gather evidence.")
    ]

    collected_evidence = []

    for iteration in range(max_iterations):
        response = await asyncio.to_thread(llm_with_tools.invoke, messages)
        messages.append(response)

        if not hasattr(response, "tool_calls") or not response.tool_calls:
            break

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

                    if result and len(str(result)) > 50:
                        collected_evidence.append(f"[{tool_name}({tool_args})]\\n{result}")

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

    compress_prompt = f"""{COMPRESS_PROMPT}

Research Topic: {research_topic}

Evidence Gathered:
{evidence_text}

Synthesize a finding for this research topic."""

    compress_response = await asyncio.to_thread(
        compress_llm.invoke,
        [HumanMessage(content=compress_prompt)]
    )

    return ResearchFinding(
        topic=research_topic,
        finding=compress_response.content,
        confidence=0.8 if collected_evidence else 0.3,
        raw_content=evidence_text
    )
