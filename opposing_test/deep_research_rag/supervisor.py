"""
Supervisor for RAG-based deep research.
Simple loop implementation (no LangGraph) to avoid state issues.
"""

import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from .researcher import run_researcher, ResearchFinding
from src.util.llm_client import get_llm, get_nano_llm


SUPERVISOR_PROMPT = """You are a lead researcher coordinating investigation.

Your role: Plan and delegate research to answer the user's question thoroughly.

## Data-Driven Research
Search the actual data and let the results inform your understanding.

## Decompose Complex Questions
For questions with multiple criteria, break into simple searches and find the intersection.

## Available Tools

- `conduct_research(research_topic, search_hints)`: Spawn a researcher to investigate a topic
- `research_complete()`: Signal that enough research has been gathered

Call `conduct_research` MULTIPLE TIMES for parallel investigations.

## Process

1. Spawn 2-4 researchers with DIFFERENT search angles
2. After receiving findings, assess completeness
3. When confident, call `research_complete`"""


BRIEF_PROMPT = """Transform the user's question into a focused research brief.

The research brief should:
1. Clarify what information is needed
2. Identify the question type (enumeration, comparison, lookup, etc.)
3. Stay neutral about terminology

Output a clear, focused research brief."""


@tool
def conduct_research(research_topic: str, search_hints: list[str] = None) -> str:
    """Spawn a researcher to investigate a specific topic."""
    return f"Researching: {research_topic}"


@tool
def research_complete() -> str:
    """Signal that enough research has been gathered."""
    return "Research complete"


SUPERVISOR_TOOLS = [conduct_research, research_complete]


async def run_supervisor(
    question: str,
    max_iterations: int = 3,
    max_concurrent: int = 8
) -> tuple[list[ResearchFinding], str, int]:
    """Run the supervisor to orchestrate research using simple loop."""
    import time
    start = time.time()

    llm = get_llm()
    llm_with_tools = llm.bind_tools(SUPERVISOR_TOOLS)
    brief_llm = get_nano_llm()

    # Write research brief
    brief_prompt = f"{BRIEF_PROMPT}\n\nUser Question: {question}\n\nWrite a clear research brief:"
    brief_response = await asyncio.to_thread(brief_llm.invoke, [HumanMessage(content=brief_prompt)])
    research_brief = brief_response.content

    # Initialize conversation
    messages = [HumanMessage(content=question)]
    all_findings = []

    for iteration in range(max_iterations):
        # Build system prompt with current findings
        findings_summary = ""
        if all_findings:
            findings_summary = "\n\nCurrent findings:\n"
            for f in all_findings:
                findings_summary += f"- {f.topic}: {f.finding[:200]}...\n"

        system_content = SUPERVISOR_PROMPT
        system_content += f"\n\nResearch Brief:\n{research_brief}"
        if findings_summary:
            system_content += findings_summary

        # Get supervisor response
        full_messages = [SystemMessage(content=system_content)] + messages
        response = await asyncio.to_thread(llm_with_tools.invoke, full_messages)
        messages.append(response)

        # Check for tool calls
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            break

        # Check for research_complete
        should_stop = False
        for tc in response.tool_calls:
            if tc["name"] == "research_complete":
                should_stop = True
                break

        if should_stop:
            break

        # Execute research tasks in parallel
        research_tasks = []
        for tc in response.tool_calls:
            if tc["name"] == "conduct_research":
                args = tc["args"]
                research_tasks.append({
                    "id": tc["id"],
                    "topic": args.get("research_topic", ""),
                    "hints": args.get("search_hints", [])
                })

        if research_tasks:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_task(task):
                async with semaphore:
                    finding = await run_researcher(
                        research_topic=task["topic"],
                        search_hints=task["hints"],
                        max_iterations=5
                    )
                    return task["id"], finding

            results = await asyncio.gather(*[run_task(t) for t in research_tasks])

            # Add tool messages and collect findings
            for call_id, finding in results:
                all_findings.append(finding)
                messages.append(ToolMessage(
                    content=f"Topic: {finding.topic}\nFinding: {finding.finding}\nConfidence: {finding.confidence}",
                    tool_call_id=call_id
                ))

    elapsed_ms = int((time.time() - start) * 1000)
    return all_findings, research_brief, elapsed_ms
