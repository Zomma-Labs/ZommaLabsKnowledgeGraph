"""
Supervisor Agent: Orchestrates research by delegating to researcher sub-agents.

The supervisor:
1. Analyzes what information is needed
2. Spawns researchers for different aspects
3. Aggregates findings
4. Decides when research is complete
"""

import asyncio
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from .state import SupervisorState, ResearchFinding, ConductResearch, ResearchComplete
from .prompts import SUPERVISOR_SYSTEM_PROMPT, BRIEF_WRITER_PROMPT
from .researcher import run_researcher
from src.util.llm_client import get_llm, get_nano_llm


class ResearchPlanItem(BaseModel):
    """Planned research topic with optional search hints."""
    topic: str = Field(..., description="Specific topic/question to research")
    hints: list[str] = Field(default_factory=list, description="Entities/terms to search for")


class ResearchPlan(BaseModel):
    """Structured plan for targeted research."""
    topics: list[ResearchPlanItem] = Field(default_factory=list)


PLAN_RESEARCH_PROMPT = """You are planning targeted research topics for a knowledge graph.

Break the question into 2-4 specific research topics that can be investigated independently.
Each topic should be concise and focused. Include optional search hints (entities or terms).

Return JSON with this structure:
{
  "topics": [
    {"topic": "specific research topic", "hints": ["entity1", "term2"]}
  ]
}

QUESTION: {question}
"""


# === Supervisor Tools ===

# Placeholder - actual execution happens in the tools node
@tool
def conduct_research(research_topic: str, search_hints: list[str] = None) -> str:
    """
    Spawn a researcher to investigate a specific topic.

    Args:
        research_topic: The specific topic/question to research
        search_hints: Optional entities or terms to search for

    Returns:
        Research findings for the topic
    """
    # This is a placeholder - actual execution in supervisor_tools node
    return f"Researching: {research_topic}"


@tool
def research_complete() -> str:
    """Signal that enough research has been gathered to answer the question."""
    return "Research complete"


SUPERVISOR_TOOLS = [conduct_research, research_complete]


async def plan_research(question: str, user_id: str = "default") -> list[dict]:
    """
    Create a lightweight research plan without running the supervisor graph.

    Returns list of dicts: [{"topic": "...", "hints": [...]}, ...]
    """
    planner = get_nano_llm().with_structured_output(ResearchPlan)
    prompt = PLAN_RESEARCH_PROMPT.format(question=question)

    try:
        result = await asyncio.to_thread(
            planner.invoke,
            [HumanMessage(content=prompt)]
        )
        topics = []
        for item in result.topics:
            topic = (item.topic or "").strip()
            if not topic:
                continue
            topics.append({"topic": topic, "hints": item.hints or []})
        return topics
    except Exception:
        # Fallback: single-topic plan
        return [{"topic": question, "hints": []}]


def create_supervisor_graph(user_id: str = "default", max_iterations: int = 3, max_concurrent: int = 3):
    """Create the supervisor graph that orchestrates research."""

    llm = get_llm()
    llm_with_tools = llm.bind_tools(SUPERVISOR_TOOLS)
    brief_llm = get_nano_llm()

    def write_brief(state: SupervisorState) -> dict:
        """Transform user question into a focused research brief."""
        messages = state["messages"]
        user_question = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break

        prompt = f"""{BRIEF_WRITER_PROMPT}

User Question: {user_question}

Write a clear research brief:"""

        response = brief_llm.invoke([HumanMessage(content=prompt)])

        return {"research_brief": response.content}

    def supervisor_node(state: SupervisorState) -> dict:
        """Main supervisor decision node."""
        messages = state["messages"]

        # Build context with current findings
        findings_summary = ""
        if state.get("findings"):
            findings_summary = "\n\nCurrent findings:\n"
            for f in state["findings"]:
                findings_summary += f"- {f.topic}: {f.finding[:200]}... (confidence: {f.confidence})\n"

        # Add system prompt and context
        system_content = SUPERVISOR_SYSTEM_PROMPT
        if state.get("research_brief"):
            system_content += f"\n\nResearch Brief:\n{state['research_brief']}"
        if findings_summary:
            system_content += findings_summary

        full_messages = [SystemMessage(content=system_content)] + list(messages)

        response = llm_with_tools.invoke(full_messages)

        return {
            "messages": [response],
            "iterations": state["iterations"] + 1
        }

    async def supervisor_tools(state: SupervisorState) -> dict:
        """Execute supervisor tool calls - spawns researchers."""
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if not last_message or not hasattr(last_message, "tool_calls"):
            return {}

        new_findings = []
        tool_messages = []

        # Collect research tasks
        research_tasks = []
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "conduct_research":
                args = tool_call["args"]
                research_tasks.append({
                    "id": tool_call["id"],
                    "topic": args.get("research_topic", ""),
                    "hints": args.get("search_hints", [])
                })
            elif tool_call["name"] == "research_complete":
                tool_messages.append(ToolMessage(
                    content="Research marked as complete",
                    tool_call_id=tool_call["id"]
                ))

        # Run researchers in parallel (up to max_concurrent)
        if research_tasks:
            async def run_task(task):
                finding = await run_researcher(
                    research_topic=task["topic"],
                    search_hints=task["hints"],
                    user_id=user_id,
                    max_iterations=5
                )
                return task["id"], finding

            # Limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def bounded_task(task):
                async with semaphore:
                    return await run_task(task)

            results = await asyncio.gather(*[bounded_task(t) for t in research_tasks])

            for call_id, finding in results:
                new_findings.append(finding)
                tool_messages.append(ToolMessage(
                    content=f"Research complete.\n\nTopic: {finding.topic}\nFinding: {finding.finding}\nConfidence: {finding.confidence}",
                    tool_call_id=call_id
                ))

        return {
            "messages": tool_messages,
            "findings": new_findings
        }

    def should_continue(state: SupervisorState) -> Literal["tools", "end"]:
        """Decide whether to continue or end."""
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        # Check iteration limit
        if state["iterations"] >= state.get("max_iterations", max_iterations):
            return "end"

        # Check for tool calls
        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Check if research_complete was called
            for tc in last_message.tool_calls:
                if tc["name"] == "research_complete":
                    return "end"
            return "tools"

        return "end"

    # Build graph
    graph = StateGraph(SupervisorState)

    graph.add_node("write_brief", write_brief)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tools", supervisor_tools)

    graph.set_entry_point("write_brief")
    graph.add_edge("write_brief", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {"tools": "tools", "end": END}
    )

    graph.add_edge("tools", "supervisor")

    return graph.compile()


async def run_supervisor(
    question: str,
    user_id: str = "default",
    max_iterations: int = 3,
    max_concurrent: int = 3
) -> tuple[list[ResearchFinding], str, int]:
    """
    Run the supervisor to orchestrate research.

    Args:
        question: The user's question
        user_id: User/tenant ID
        max_iterations: Max supervisor iterations
        max_concurrent: Max concurrent researchers

    Returns:
        tuple of (findings, research_brief, iterations)
    """
    import time
    start = time.time()

    graph = create_supervisor_graph(user_id, max_iterations, max_concurrent)

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "research_brief": "",
        "findings": [],
        "iterations": 0,
        "max_iterations": max_iterations
    }

    result = await graph.ainvoke(initial_state)

    elapsed_ms = int((time.time() - start) * 1000)

    return (
        result.get("findings", []),
        result.get("research_brief", ""),
        elapsed_ms
    )
