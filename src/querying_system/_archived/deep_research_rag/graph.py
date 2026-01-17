"""
Deep Research RAG (No KG) - For Comparison
==========================================

Same multi-agent architecture as deep_research_kg, but uses
regular vector search on chunks instead of Knowledge Graph.

This is the CONTROL - compare against deep_research_kg to show
that the Knowledge Graph is what makes the difference.

Weaknesses to highlight:
- Researchers just get text chunks, no entity structure
- Can't traverse relationships
- Can't do multi-hop reasoning
- Retrieves by text similarity only
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Optional
from pydantic import Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from src.util.services import get_services
from src.util.llm_client import get_llm, get_nano_llm


# === State ===

class DeepResearchRAGState(MessagesState):
    """State for deep research RAG."""
    research_brief: Optional[str] = Field(default="")
    findings: Optional[list] = Field(default_factory=list)
    research_complete: Optional[bool] = Field(default=False)


# === RAG-only Researcher (no KG) ===

async def rag_researcher(topic: str, user_id: str = "default") -> dict:
    """
    Research a topic using only vector search on chunks.
    No KG structure, no entity resolution, no relationship traversal.
    """
    services = get_services()

    # Just vector search on chunks - basic RAG
    query_vector = services.embeddings.embed_query(topic)

    results = services.neo4j.query("""
        CALL db.index.vector.queryNodes('fact_embeddings', 15, $vec)
        YIELD node, score
        WHERE score > 0.25

        // Get chunk content only - NO entity/relationship structure
        OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode)-[r2 {fact_id: node.uuid}]->(obj)

        RETURN c.content as chunk, c.header_path as header, score
        ORDER BY score DESC
        LIMIT 10
    """, {"vec": query_vector})

    # Format as raw text (no structure)
    chunks = []
    for r in results:
        chunk = r.get("chunk", "")
        header = r.get("header", "")
        if chunk:
            chunks.append(f"[{header}]\n{chunk[:400]}")

    evidence = "\n\n---\n\n".join(chunks) if chunks else "No relevant chunks found."

    # Compress finding
    llm = get_nano_llm()
    compress_prompt = f"""Research topic: {topic}

Retrieved text chunks:
{evidence}

Summarize what you found about this topic in 2-3 sentences. If the chunks don't contain relevant information, say so."""

    response = llm.invoke([HumanMessage(content=compress_prompt)])

    return {
        "topic": topic,
        "finding": response.content,
        "evidence": evidence
    }


# === Supervisor (same as KG version, but uses RAG researcher) ===

async def plan_and_research(state: DeepResearchRAGState) -> dict:
    """
    Plan research topics and investigate using RAG-only retrieval.
    """
    # Get question
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    if not question:
        return {
            "messages": [AIMessage(content="Please ask a question.")],
            "research_brief": "",
            "findings": [],
            "research_complete": True
        }

    # Plan research topics (same as KG version)
    llm = get_nano_llm()
    plan_prompt = f"""Question: {question}

Break this question into 2-3 specific research topics to investigate.
Return just the topics, one per line."""

    plan_response = llm.invoke([HumanMessage(content=plan_prompt)])
    topics = [t.strip() for t in plan_response.content.strip().split("\n") if t.strip()][:3]

    if not topics:
        topics = [question]

    # Research each topic using RAG (no KG)
    findings = []
    for topic in topics:
        finding = await rag_researcher(topic)
        findings.append(finding)

    return {
        "messages": [],
        "research_brief": f"Researched {len(topics)} topics using vector search on document chunks.",
        "findings": findings,
        "research_complete": True
    }


def synthesize(state: DeepResearchRAGState) -> dict:
    """Synthesize findings into answer."""
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    findings = state.get("findings") or []

    # Format findings
    findings_text = ""
    for i, f in enumerate(findings, 1):
        if isinstance(f, dict):
            findings_text += f"""
### Research {i}: {f.get('topic', 'Unknown')}
{f.get('finding', 'No finding')}

Evidence:
{str(f.get('evidence', 'No evidence'))[:500]}...

---
"""

    if not findings_text:
        findings_text = "No research findings available."

    llm = get_llm()
    prompt = f"""Question: {question}

Research Findings (from vector search on document chunks):
{findings_text}

Based on these findings, answer the question. Be specific and note any limitations."""

    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Answer based only on the provided research findings."),
        HumanMessage(content=prompt)
    ])

    return {
        "messages": [AIMessage(content=response.content)],
        "research_brief": state.get("research_brief") or "",
        "findings": findings,
        "research_complete": True
    }


def should_synthesize(state: DeepResearchRAGState) -> str:
    if state.get("research_complete"):
        return "synthesize"
    return "end"


# === Build Graph ===

def create_graph():
    graph = StateGraph(DeepResearchRAGState)

    graph.add_node("research", plan_and_research)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", should_synthesize, {"synthesize": "synthesize", "end": END})
    graph.add_edge("synthesize", END)

    return graph.compile()


graph = create_graph()


if __name__ == "__main__":
    async def test():
        g = create_graph()
        result = await g.ainvoke({
            "messages": [HumanMessage(content="Which districts reported economic growth?")]
        })
        print(result["messages"][-1].content)

    asyncio.run(test())
