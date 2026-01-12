"""
Deep Research LangGraph - For LangGraph Studio
===============================================

Exposes the Deep Research pipeline as a LangGraph graph for use with
LangGraph Studio and langgraph dev.

Run with: langgraph dev
"""

import asyncio
import sys
import os

# Ensure project root is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from src.querying_system.deep_research.state import ResearchFinding
from src.querying_system.deep_research.supervisor import create_supervisor_graph, run_supervisor
from src.querying_system.deep_research.synthesizer import Synthesizer
from src.querying_system.deep_research.prompts import SYNTHESIZER_SYSTEM_PROMPT
from src.util.llm_client import get_llm


# === Deep Research State ===
from typing import Optional
from pydantic import Field

class DeepResearchState(MessagesState):
    """State for the deep research graph."""
    research_brief: Optional[str] = Field(default="")
    findings: Optional[list] = Field(default_factory=list)
    research_complete: Optional[bool] = Field(default=False)


# === Graph Nodes ===

async def research_node(state: DeepResearchState) -> dict:
    """
    Run the full research phase: supervisor spawns researchers.
    """
    # Get the user's question from the last human message
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    if not question:
        return {
            "messages": [AIMessage(content="I didn't receive a question. Please ask me something about the knowledge graph.")],
            "research_brief": "",
            "findings": [],
            "research_complete": True
        }

    # Run supervisor to gather research
    findings, research_brief, _ = await run_supervisor(
        question=question,
        user_id="default",  # TODO: Extract from config
        max_iterations=3,
        max_concurrent=5
    )

    return {
        "messages": [],
        "research_brief": research_brief or "",
        "findings": findings or [],
        "research_complete": True
    }


def synthesize_node(state: DeepResearchState) -> dict:
    """
    Synthesize findings into a final answer.
    """
    # Get the user's question
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    findings = state.get("findings", [])
    research_brief = state.get("research_brief", "")

    # Format findings for synthesis
    findings_text = ""
    if findings:
        for i, f in enumerate(findings, 1):
            findings_text += f"""
### Finding {i}: {f.topic}
**Result:** {f.finding}
**Confidence:** {f.confidence}
**Evidence:**
{f.raw_content if f.raw_content else 'No raw evidence'}

---
"""
    else:
        findings_text = "No research findings available."

    # Synthesize
    llm = get_llm()

    prompt = f"""Original Question: {question}

Research Brief: {research_brief}

Research Findings:
{findings_text}

Based on the research findings above, provide a comprehensive answer to the original question.
Include specific facts and cite your sources where possible."""

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    return {
        "messages": [AIMessage(content=response.content)],
        "research_brief": research_brief or "",
        "findings": findings or [],
        "research_complete": True
    }


def should_synthesize(state: DeepResearchState) -> Literal["synthesize", "end"]:
    """Route based on whether research is complete."""
    if state.get("research_complete"):
        return "synthesize"
    return "end"


# === Build the Graph ===

def create_deep_research_graph():
    """Create the deep research LangGraph."""

    graph = StateGraph(DeepResearchState)

    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("synthesize", synthesize_node)

    # Add edges
    graph.add_edge(START, "research")
    graph.add_conditional_edges(
        "research",
        should_synthesize,
        {"synthesize": "synthesize", "end": END}
    )
    graph.add_edge("synthesize", END)

    return graph.compile()


# === Entry Point for LangGraph Studio ===

# This is what langgraph.json points to
graph = create_deep_research_graph()


# === CLI Testing ===

async def test_graph():
    """Test the graph directly."""
    g = create_deep_research_graph()

    question = "Which districts reported slight to modest economic growth?"
    print(f"\nQuestion: {question}")
    print("=" * 60)

    result = await g.ainvoke({
        "messages": [HumanMessage(content=question)],
        "research_brief": "",
        "findings": [],
        "research_complete": False
    })

    # Print final answer
    final_msg = result["messages"][-1]
    print("\nAnswer:")
    print("-" * 60)
    print(final_msg.content)


if __name__ == "__main__":
    asyncio.run(test_graph())
