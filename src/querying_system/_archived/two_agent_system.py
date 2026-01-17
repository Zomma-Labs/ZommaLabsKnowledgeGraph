"""
Two-Agent Knowledge Graph QA System
====================================

Architecture:
1. SearchAgent - Exhaustively searches the graph for all relevant evidence
2. AnswerAgent - Synthesizes a grounded answer from collected chunks

The separation ensures:
- Search is thorough (agent can iterate without pressure to answer)
- Answers are grounded (AnswerAgent only sees collected evidence, not the graph)
"""

import os
import asyncio
from dataclasses import dataclass, field
from typing import Annotated, Sequence
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel

from src.util.llm_client import get_critique_llm

# === CONFIGURATION ===
MAX_SEARCH_ITERATIONS = 15  # More iterations for thorough search
MCP_SSE_URL = "http://127.0.0.1:8765/sse"
USE_SSE = True  # Set to True to use persistent SSE server


# === DATA STRUCTURES ===

@dataclass
class CollectedEvidence:
    """Evidence collected by the SearchAgent."""
    query: str
    chunks: list[dict] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)
    relationships_explored: list[str] = field(default_factory=list)
    search_summary: str = ""


class SearchState(MessagesState):
    """State for the SearchAgent."""
    original_query: str
    collected_chunks: list[dict]
    entities_found: list[str]
    relationships_explored: list[str]
    iteration_count: int


class AnswerState(MessagesState):
    """State for the AnswerAgent."""
    original_query: str
    evidence: str  # Formatted evidence from SearchAgent


# === SEARCH AGENT ===

SEARCH_AGENT_PROMPT = """You are a Knowledge Graph Search Specialist. Your job is to DISCOVER answers from the graph, not guess them.

## Critical Rule: No Outside Knowledge

**NEVER guess what the answer might be based on your training data.**

Instead, you must DISCOVER answers by:
1. Starting from entities that ARE mentioned in the question
2. Using semantic search (query_hint) to find relevant edges
3. Following those edges to find chunks containing the answer

If a question asks "What company inspired X?" - do NOT search for companies you think might be the answer.
Instead, search FROM "X" with query_hint="inspired modeled based on" and let the graph tell you.

## Tools
1. **resolve_entity_or_topic** - Find exact entity/topic names in the graph
2. **explore_neighbors** - Find relationships from an entity. The query_hint parameter does semantic filtering.
3. **search_relationships** - Search facts directly by description. Use when no entity to start from.
4. **get_chunk** / **get_chunks** / **get_chunks_by_edge** - Retrieve evidence chunks
5. **get_entity_info** - Get entity summaries
6. **think** - Plan your search and analyze results

## Search Strategy

### Step 1: Identify Anchor Entities OR Use Direct Fact Search
Use **think** to identify entities EXPLICITLY in the question:
- Named companies, people, places → resolve these, then explore_neighbors
- If NO specific entities in the question → use **search_relationships** to find matching facts directly
- search_relationships returns facts with their subject/object entities, which you can then explore

### Step 2: Semantic Edge Search
From your anchor entity, use explore_neighbors with query_hint containing KEY TERMS from the question.
The query_hint does semantic filtering - use words and phrases from the question itself.

### Step 3: Retrieve and Verify Chunks
Get chunks from promising edges. Use **think** to verify:
- Does this chunk contain the specific information asked?
- If NOT → try different query_hint terms or explore other edges

### Step 4: Iterate with Different Terms
If initial search fails, try:
- Synonyms and related terms in query_hint
- Temporal terms if question has dates/years
- Different anchor entities discovered in the graph

## What NOT To Do
- Do NOT resolve entities that are likely ANSWERS (you'd be guessing)
- Do NOT search for entities based on your outside knowledge
- Do NOT give up after one failed search - try multiple query_hints

## When to Stop
- Collected chunks contain information that answers the question
- You've tried multiple query_hints and anchor entities
- Further searches return no new relevant chunks

## Output Format
```
=== SEARCH COMPLETE ===
Verified Chunks: [count]
Search Approaches Tried: [list query_hints used]
```
"""


def count_tool_calls(messages: Sequence[BaseMessage]) -> int:
    """Count tool calls in message history."""
    return sum(1 for msg in messages if isinstance(msg, ToolMessage))


async def create_search_agent(mcp_client: MultiServerMCPClient):
    """Create the SearchAgent that collects evidence."""

    tools = await mcp_client.get_tools()

    model = get_critique_llm()
    model_with_tools = model.bind_tools(tools)

    def call_model(state: SearchState) -> dict:
        """Call the model for search decisions."""
        messages = state["messages"]

        # Add system prompt if first message
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            messages = [SystemMessage(content=SEARCH_AGENT_PROMPT)] + list(messages)

        # Check iteration limit
        tool_call_count = count_tool_calls(messages)
        if tool_call_count >= MAX_SEARCH_ITERATIONS:
            force_stop_msg = SystemMessage(
                content=f"SYSTEM: You have made {tool_call_count} searches. Summarize what you found and stop."
            )
            messages = list(messages) + [force_stop_msg]
            response = model.invoke(messages)
        else:
            response = model_with_tools.invoke(messages)

        return {"messages": [response]}

    def should_continue(state: SearchState) -> str:
        """Determine whether to continue searching."""
        messages = state["messages"]
        last_message = messages[-1]

        tool_call_count = count_tool_calls(messages)
        if tool_call_count >= MAX_SEARCH_ITERATIONS:
            return END

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END

    # Build graph
    builder = StateGraph(SearchState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile()


# === ANSWER AGENT ===

ANSWER_AGENT_PROMPT = """You are a Financial Research Analyst. Your job is to synthesize a clear, accurate answer from the evidence provided.

## Critical Rules

1. **ONLY use the evidence provided below** - Do not use external knowledge
2. **Every claim must cite its source** - Use format [DOC: doc_id, CHUNK: chunk_id]
3. **If evidence is insufficient, say so** - "The evidence does not contain information about X"
4. **Be precise about what the evidence says** - Don't extrapolate or assume

## Answer Structure

1. **Direct Answer**: Start with a clear, concise answer to the question
2. **Supporting Evidence**: Quote or paraphrase relevant chunks with citations
3. **Caveats**: Note any limitations or gaps in the evidence
4. **Confidence**: Indicate how well-supported the answer is

## Citation Format
Every factual claim must include: [DOC: document_id, CHUNK: chunk_id]

Example:
"Alphabet reached a $2 trillion market cap in April 2024 [DOC: alphabet_10k_2024, CHUNK: abc123]."

## If Evidence is Insufficient
If the collected evidence doesn't answer the question, respond:
"Based on the available evidence, I cannot answer this question because [reason]. The evidence contains information about [what was found], but not about [what was needed]."

Now analyze the evidence and answer the user's question.
"""


async def create_answer_agent():
    """Create the AnswerAgent that synthesizes answers from evidence."""

    # Use GPT-5.1 for answer synthesis
    model = get_critique_llm()

    def generate_answer(state: AnswerState) -> dict:
        """Generate final answer from collected evidence."""
        messages = [
            SystemMessage(content=ANSWER_AGENT_PROMPT),
            HumanMessage(content=f"""## User Question
{state['original_query']}

## Collected Evidence
{state['evidence']}

Please provide a comprehensive answer based ONLY on the evidence above.""")
        ]

        response = model.invoke(messages)
        return {"messages": [response]}

    # Simple single-node graph (no tools needed)
    builder = StateGraph(AnswerState)
    builder.add_node("answer", generate_answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)

    return builder.compile()


# === ORCHESTRATOR ===

def extract_chunks_from_messages(messages: Sequence[BaseMessage]) -> list[dict]:
    """Extract collected chunks from search agent messages."""
    chunks = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content
            # Look for chunk patterns in tool responses
            if isinstance(content, str) and "CHUNK_id:" in content:
                chunks.append({
                    "tool_name": getattr(msg, 'name', 'unknown'),
                    "content": content
                })
            elif isinstance(content, dict):
                if content.get("found") and content.get("chunk"):
                    chunks.append({
                        "tool_name": getattr(msg, 'name', 'unknown'),
                        "content": content["chunk"]
                    })
                elif content.get("results"):
                    for result in content["results"]:
                        if isinstance(result, dict) and result.get("chunk"):
                            chunks.append({
                                "tool_name": getattr(msg, 'name', 'unknown'),
                                "content": result["chunk"]
                            })
    return chunks


def extract_search_trace(messages: Sequence[BaseMessage]) -> list[dict]:
    """Extract tool calls and results from search agent messages for debugging."""
    trace = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            # Check for tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    trace.append({
                        "type": "tool_call",
                        "tool": tc.get("name", "unknown"),
                        "args": tc.get("args", {})
                    })
            # Check for think/reasoning content
            elif msg.content and len(msg.content) < 500:
                trace.append({
                    "type": "reasoning",
                    "content": msg.content[:200]
                })
        elif isinstance(msg, ToolMessage):
            # Summarize tool result
            content = msg.content
            if isinstance(content, dict):
                summary = f"found={content.get('found', '?')}"
                if 'count' in content:
                    summary += f", count={content['count']}"
                if 'results' in content:
                    summary += f", results={len(content['results'])}"
            elif isinstance(content, str):
                summary = content[:150] + "..." if len(content) > 150 else content
            else:
                summary = str(content)[:150]

            trace.append({
                "type": "tool_result",
                "tool": getattr(msg, 'name', 'unknown'),
                "summary": summary
            })

    return trace


def format_evidence_for_answer_agent(chunks: list[dict], search_summary: str) -> str:
    """Format collected evidence for the answer agent."""
    if not chunks:
        return f"No evidence chunks were collected.\n\nSearch Summary:\n{search_summary}"

    evidence_parts = [f"### Collected Evidence ({len(chunks)} chunks)\n"]

    for i, chunk in enumerate(chunks, 1):
        evidence_parts.append(f"--- Chunk {i} ---")
        evidence_parts.append(chunk["content"])
        evidence_parts.append("")

    evidence_parts.append(f"\n### Search Summary\n{search_summary}")

    return "\n".join(evidence_parts)


class TwoAgentQA:
    """Orchestrates the two-agent QA system."""

    def __init__(self):
        self.mcp_client = None
        self.search_agent = None
        self.answer_agent = None
        self._initialized = False

    async def initialize(self):
        """Initialize MCP client and agents."""
        if self._initialized:
            return

        # Initialize MCP client - use SSE for persistent server
        if USE_SSE:
            self.mcp_client = MultiServerMCPClient(
                {
                    "zommagraph": {
                        "url": MCP_SSE_URL,
                        "transport": "sse",
                    }
                }
            )
        else:
            self.mcp_client = MultiServerMCPClient(
                {
                    "zommagraph": {
                        "command": "python",
                        "args": ["-m", "src.querying_system.mcp_server"],
                        "transport": "stdio",
                        "env": {
                            **os.environ,
                            "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        }
                    }
                }
            )

        # Create agents
        self.search_agent = await create_search_agent(self.mcp_client)
        self.answer_agent = await create_answer_agent()
        self._initialized = True

    async def query(self, question: str, verbose: bool = False, return_trace: bool = False):
        """
        Answer a question using the two-agent system.

        Args:
            question: The user's question
            verbose: Whether to print intermediate steps
            return_trace: If True, returns (answer, trace_info) tuple

        Returns:
            The final answer with citations, or (answer, trace) if return_trace=True
        """
        await self.initialize()

        if verbose:
            print(f"\n{'='*60}")
            print(f"QUESTION: {question}")
            print(f"{'='*60}\n")

        # === Phase 1: Search ===
        if verbose:
            print(">>> PHASE 1: SearchAgent collecting evidence...")

        search_prompt = f"""Find ALL evidence in the knowledge graph that could help answer this question:

"{question}"

Be thorough - explore all relevant entities and relationships. Collect every chunk that might be relevant."""

        search_result = await self.search_agent.ainvoke(
            {
                "messages": [HumanMessage(content=search_prompt)],
                "original_query": question,
                "collected_chunks": [],
                "entities_found": [],
                "relationships_explored": [],
                "iteration_count": 0
            },
            config={"recursion_limit": 30}
        )

        # Extract chunks from search
        chunks = extract_chunks_from_messages(search_result["messages"])

        # Get search summary from last AI message
        search_summary = ""
        for msg in reversed(search_result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                search_summary = msg.content
                break

        if verbose:
            print(f"\n>>> SearchAgent found {len(chunks)} chunks")
            print(f">>> Search summary: {search_summary[:200]}...")

        # === Phase 2: Answer ===
        if verbose:
            print("\n>>> PHASE 2: AnswerAgent synthesizing response...")

        evidence = format_evidence_for_answer_agent(chunks, search_summary)

        answer_result = await self.answer_agent.ainvoke(
            {
                "messages": [],
                "original_query": question,
                "evidence": evidence
            }
        )

        # Extract final answer
        final_answer = ""
        for msg in answer_result["messages"]:
            if isinstance(msg, AIMessage):
                final_answer = msg.content
                break

        if verbose:
            print(f"\n{'='*60}")
            print("FINAL ANSWER:")
            print(f"{'='*60}")
            print(final_answer)

        if return_trace:
            trace = extract_search_trace(search_result["messages"])
            return final_answer, trace

        return final_answer

    async def close(self):
        """Clean up resources."""
        if self.mcp_client:
            # MCP client cleanup if needed
            pass


# === CLI / Testing ===

async def main():
    """Test the two-agent system."""
    import sys

    # Get question from args or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What market cap milestones has Alphabet reached?"

    qa = TwoAgentQA()
    try:
        answer = await qa.query(question, verbose=True)
        print("\n" + "="*60)
        print("RESULT:")
        print("="*60)
        print(answer)
    finally:
        await qa.close()


if __name__ == "__main__":
    asyncio.run(main())
