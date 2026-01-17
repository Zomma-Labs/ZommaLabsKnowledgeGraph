"""
Simple RAG Graph - For LangGraph Studio
========================================

A simple retrieve-and-answer graph using deterministic retrieval.
No agents, no tool selection - just:
1. Retrieve relevant facts
2. Generate answer

This is faster and more consistent than the deep research approach.

Run with: langgraph dev
"""

import sys
import os

# Ensure project root is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Optional
from pydantic import Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from src.util.deterministic_retrieval import DeterministicRetriever
from src.util.llm_client import get_llm


# === State ===

class SimpleRAGState(MessagesState):
    """State for simple RAG - just messages."""
    pass  # MessagesState already has messages with defaults


# === System Prompt ===

SIMPLE_RAG_PROMPT = """You are a Financial Knowledge Graph Assistant.

You will be given a question and relevant evidence retrieved from a knowledge graph.
Answer the question based ONLY on the provided evidence.

Rules:
1. Only use facts from the evidence - do not make things up
2. If the evidence doesn't contain the answer, say so
3. Be specific - cite document names and dates when available
4. Be concise but complete
"""


# === Graph Node ===

async def retrieve_and_answer(state: SimpleRAGState) -> dict:
    """
    Single node that retrieves evidence and generates an answer.
    """
    # Get the user's question
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    if not question:
        return {
            "messages": [AIMessage(content="Please ask a question about the knowledge graph.")]
        }

    # Step 1: Deterministic retrieval
    retriever = DeterministicRetriever(group_id="default")
    evidence = await retriever.search(question, top_k=15)

    # Step 2: Format evidence
    evidence_text = retriever.format_evidence_for_llm(evidence)

    # Step 3: Generate answer
    llm = get_llm()

    prompt = f"""Question: {question}

{evidence_text}

Based on the evidence above, answer the question. Be specific and cite sources."""

    response = llm.invoke([
        SystemMessage(content=SIMPLE_RAG_PROMPT),
        HumanMessage(content=prompt)
    ])

    return {
        "messages": [AIMessage(content=response.content)]
    }


# === Build Graph ===

def create_simple_rag_graph():
    """Create the simple RAG graph."""
    graph = StateGraph(SimpleRAGState)

    graph.add_node("retrieve_and_answer", retrieve_and_answer)

    graph.add_edge(START, "retrieve_and_answer")
    graph.add_edge("retrieve_and_answer", END)

    return graph.compile()


# === Entry Point for LangGraph Studio ===

graph = create_simple_rag_graph()


# === CLI Testing ===

if __name__ == "__main__":
    import asyncio

    async def test():
        g = create_simple_rag_graph()
        result = await g.ainvoke({
            "messages": [HumanMessage(content="What is the Beige Book?")]
        })
        print(result["messages"][-1].content)

    asyncio.run(test())
