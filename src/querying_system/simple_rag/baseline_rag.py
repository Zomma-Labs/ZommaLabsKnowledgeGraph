"""
Baseline Simple RAG - NO Knowledge Graph
=========================================

This is a "dumb" baseline RAG that just:
1. Embeds the question
2. Vector searches raw document chunks
3. Stuffs them into context and generates answer

This is what most basic RAG systems do. Use this to compare against
the Deep Research KG system to show why KG is better.

Weaknesses of this approach (to highlight in demo):
- No entity resolution (can't connect "Tim Cook" to "Apple CEO")
- No relationship understanding (just text similarity)
- No multi-hop reasoning (can't follow chains of facts)
- Retrieves redundant/irrelevant chunks
- Can't answer "which districts..." type questions well
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from src.util.services import get_services
from src.util.llm_client import get_llm


class BaselineRAGState(MessagesState):
    """State for baseline RAG."""
    pass


BASELINE_PROMPT = """You are a helpful assistant. Answer the question based on the provided context.

If the context doesn't contain enough information, say so.
"""


async def baseline_retrieve_and_answer(state: BaselineRAGState) -> dict:
    """
    Baseline RAG: Just vector search on chunks, no KG.
    """
    # Get question
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    if not question:
        return {"messages": [AIMessage(content="Please ask a question.")]}

    services = get_services()

    # Simple vector search on EpisodicNode (raw chunks) - NO KG STRUCTURE
    query_vector = services.embeddings.embed_query(question)

    # Just grab chunks by vector similarity - this is what basic RAG does
    results = services.neo4j.query("""
        CALL db.index.vector.queryNodes('fact_embeddings', 10, $vec)
        YIELD node, score
        WHERE score > 0.3

        // Just get the chunk content - no entity/relationship structure
        OPTIONAL MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode)-[r2 {fact_id: node.uuid}]->(obj)

        RETURN c.content as chunk, c.header_path as header, score
        ORDER BY score DESC
        LIMIT 10
    """, {"vec": query_vector})

    # Format as simple context (no structure, just text chunks)
    context_parts = []
    for i, r in enumerate(results, 1):
        chunk = r.get("chunk", "")
        header = r.get("header", "")
        if chunk:
            context_parts.append(f"[Chunk {i}] {header}\n{chunk[:500]}...")

    context = "\n\n".join(context_parts) if context_parts else "No relevant chunks found."

    # Generate answer
    llm = get_llm()

    prompt = f"""Question: {question}

Context (retrieved chunks):
{context}

Answer the question based on the context above."""

    response = llm.invoke([
        SystemMessage(content=BASELINE_PROMPT),
        HumanMessage(content=prompt)
    ])

    return {"messages": [AIMessage(content=response.content)]}


def create_baseline_rag_graph():
    graph = StateGraph(BaselineRAGState)
    graph.add_node("answer", baseline_retrieve_and_answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile()


graph = create_baseline_rag_graph()
