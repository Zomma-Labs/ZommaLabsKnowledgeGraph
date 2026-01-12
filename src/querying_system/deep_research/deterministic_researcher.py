"""
Deterministic Researcher: Uses multi-strategy retrieval instead of LLM-controlled tools.

Key difference from original researcher:
- Original: LLM decides which tools to call and in what order
- This: Deterministic retrieval runs ALL strategies, returns fused results

This eliminates variance from tool selection decisions.
"""

import asyncio
from typing import Optional

from .state import ResearchFinding
from .prompts import COMPRESS_RESEARCH_PROMPT
from src.util.llm_client import get_nano_llm
from src.util.deterministic_retrieval import DeterministicRetriever


async def run_deterministic_researcher(
    research_topic: str,
    search_hints: list[str] = None,
    user_id: str = "default",
    top_k: int = 15
) -> ResearchFinding:
    """
    Run a deterministic researcher that uses multi-strategy retrieval.

    Instead of LLM choosing which tools to call, this:
    1. Runs deterministic retrieval (vector + keyword + graph)
    2. Fuses results with RRF
    3. Compresses findings

    Args:
        research_topic: The topic/question to investigate
        search_hints: Optional hints to expand the search query
        user_id: User/tenant ID for graph scoping
        top_k: Number of results to retrieve

    Returns:
        ResearchFinding with compressed results
    """
    retriever = DeterministicRetriever(group_id=user_id)

    # Build search query - combine topic with hints for better coverage
    search_query = research_topic
    if search_hints:
        # Add hints as additional context
        hints_text = " ".join(search_hints)
        search_query = f"{research_topic} {hints_text}"

    # Run deterministic retrieval (vector + keyword + graph in parallel)
    evidence = await retriever.search(search_query, top_k=top_k)

    # Format evidence for compression
    evidence_parts = []
    for e in evidence:
        strategies = ", ".join(e.found_by)
        evidence_parts.append(
            f"[{strategies}] {e.subject} -[{e.edge_type}]-> {e.object}\n"
            f"Fact: {e.content}\n"
            f"Source: {e.chunk_header or 'N/A'}\n"
            f"Date: {e.document_date or 'N/A'}"
        )

    evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "No evidence found"

    # Compress findings using cheap LLM
    compress_llm = get_nano_llm()

    compress_prompt = f"""{COMPRESS_RESEARCH_PROMPT}

Research Topic: {research_topic}

Evidence Gathered (via deterministic multi-strategy retrieval):
{evidence_text}

Synthesize a finding for this research topic. Be specific and cite facts."""

    from langchain_core.messages import HumanMessage
    compress_response = await asyncio.to_thread(
        compress_llm.invoke,
        [HumanMessage(content=compress_prompt)]
    )

    # Calculate confidence based on evidence quality
    if not evidence:
        confidence = 0.2
    elif len(evidence) < 3:
        confidence = 0.5
    elif any(len(e.found_by) > 1 for e in evidence):
        # Higher confidence if results found by multiple strategies
        confidence = 0.9
    else:
        confidence = 0.7

    return ResearchFinding(
        topic=research_topic,
        finding=compress_response.content,
        confidence=confidence,
        evidence_chunks=[e.chunk_id for e in evidence if e.chunk_id],
        raw_content=evidence_text
    )


async def run_deterministic_research_batch(
    topics: list[dict],
    user_id: str = "default",
    top_k: int = 15,
    max_concurrent: int = 5
) -> list[ResearchFinding]:
    """
    Run multiple deterministic researchers in parallel.

    Args:
        topics: List of dicts with 'topic' and optional 'hints'
        user_id: User/tenant ID
        top_k: Results per topic
        max_concurrent: Max concurrent retrievals

    Returns:
        List of ResearchFindings
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_research(topic_info: dict) -> ResearchFinding:
        async with semaphore:
            return await run_deterministic_researcher(
                research_topic=topic_info.get("topic", ""),
                search_hints=topic_info.get("hints", []),
                user_id=user_id,
                top_k=top_k
            )

    tasks = [bounded_research(t) for t in topics]
    return await asyncio.gather(*tasks)
