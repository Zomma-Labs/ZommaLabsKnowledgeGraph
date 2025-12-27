"""
Diagnostic Script for QA Evaluation Failures
=============================================

This script analyzes why certain questions fail by:
1. Loading failed questions from eval results
2. Running the full agent workflow with verbose logging
3. Showing exactly what chunks are retrieved (or not retrieved)
4. Identifying the root cause of each failure

Usage:
    uv run src/scripts/diagnose_failures.py
    uv run src/scripts/diagnose_failures.py --results eval_results_20251226_014902.json
    uv run src/scripts/diagnose_failures.py --question-id 5
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# === CONFIGURATION ===
MAX_TOOL_ITERATIONS = 15
MCP_SSE_URL = "http://127.0.0.1:8765/sse"


def get_chat_model(temperature: float = 0):
    """Get an LLM with fallback support."""
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
    if os.getenv("XAI_API_KEY"):
        from langchain_xai import ChatXAI
        return ChatXAI(model="grok-4-1-fast-non-reasoning", temperature=temperature)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=temperature)
    raise ValueError("No LLM API key found")


AGENT_SYSTEM_PROMPT = """You are a Knowledge Graph Research Agent. You answer questions ONLY using data from a graph database.

## CRITICAL GROUNDING RULES

1. **USE ONLY RETRIEVED DATA**: Your answer must come ONLY from chunks and entities returned by tool calls.
   - NEVER use your prior knowledge or training data to fill in facts
   - NEVER cite sources that weren't returned by tools
   - Specific dates, numbers, and names must come from retrieved chunks

2. **SEARCH THOROUGHLY**: If your first search doesn't find the answer:
   - Try different search terms (synonyms, related concepts)
   - Search for related entities that might have the connection
   - Use get_chunks to retrieve multiple relationship evidences at once

3. **COMPLETE THE WORKFLOW**: You MUST follow all 3 steps:
   - resolve_entity_or_topic → find exact entity names
   - explore_neighbors → discover relationships
   - get_chunk or get_chunks → **REQUIRED** to get actual source text with specific details

4. **SYNTHESIZE ANSWERS**:
   - NEVER return raw tool output to the user
   - Extract the relevant information from chunks
   - Format a proper natural language answer"""


def count_tool_calls(messages) -> int:
    return sum(1 for msg in messages if isinstance(msg, ToolMessage))


def extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and 'text' in block:
                text_parts.append(block['text'])
            elif isinstance(block, str):
                text_parts.append(block)
        return '\n'.join(text_parts) if text_parts else str(content)
    return str(content)


async def run_agent_with_logging(question: str, verbose: bool = True, transport: str = "stdio"):
    """Run the agent with full logging of all tool calls and responses."""

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if transport == "sse":
        mcp_client = MultiServerMCPClient({
            "zommagraph": {
                "url": MCP_SSE_URL,
                "transport": "sse",
            }
        })
    else:
        mcp_client = MultiServerMCPClient({
            "zommagraph": {
                "command": "python",
                "args": ["-m", "src.agents.mcp_server"],
                "transport": "stdio",
                "env": {**os.environ, "PYTHONPATH": project_root}
            }
        })

    tools = await mcp_client.get_tools()
    model = get_chat_model(temperature=0)
    model_with_tools = model.bind_tools(tools)

    # Tracking data
    tool_calls_log = []
    chunks_retrieved = []

    def call_model(state: MessagesState) -> dict:
        messages = state["messages"]
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + list(messages)

        tool_call_count = count_tool_calls(messages)
        if tool_call_count >= MAX_TOOL_ITERATIONS:
            force_stop_msg = SystemMessage(
                content=f"SYSTEM: You have made {tool_call_count} tool calls. Provide your final answer now."
            )
            messages = list(messages) + [force_stop_msg]
            response = model.invoke(messages)
        else:
            response = model_with_tools.invoke(messages)

        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        tool_call_count = count_tool_calls(messages)

        if tool_call_count >= MAX_TOOL_ITERATIONS:
            return END
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    graph = builder.compile()

    # Run with streaming to capture all messages
    if verbose:
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}\n")

    all_messages = []

    try:
        async for event in graph.astream(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 30},
            stream_mode="messages"
        ):
            msg, metadata = event
            all_messages.append(msg)

            # Log tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_call_entry = {
                        "tool": tc['name'],
                        "args": tc['args']
                    }
                    tool_calls_log.append(tool_call_entry)
                    if verbose:
                        print(f"📞 TOOL CALL: {tc['name']}")
                        print(f"   Args: {json.dumps(tc['args'], indent=2)}")

            # Log tool responses
            if isinstance(msg, ToolMessage):
                content = msg.content
                if verbose:
                    print(f"\n📦 TOOL RESPONSE ({msg.name}):")
                    # Truncate long responses
                    if len(str(content)) > 1500:
                        print(f"   {str(content)[:1500]}...")
                    else:
                        print(f"   {content}")
                    print()

                # Track chunks retrieved
                if msg.name in ['get_chunk', 'get_chunks']:
                    try:
                        if isinstance(content, str):
                            parsed = json.loads(content) if content.startswith('{') else content
                        else:
                            parsed = content
                        chunks_retrieved.append({
                            "tool": msg.name,
                            "response": parsed
                        })
                    except:
                        chunks_retrieved.append({"tool": msg.name, "response": content})

        # Get final answer
        final_answer = None
        for msg in reversed(all_messages):
            if hasattr(msg, 'content') and msg.content:
                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                    final_answer = extract_text_content(msg.content)
                    break

        if verbose:
            print(f"\n{'='*80}")
            print("FINAL ANSWER:")
            print(f"{'='*80}")
            print(final_answer or "No answer generated")
            print()

        return {
            "question": question,
            "answer": final_answer,
            "tool_calls": tool_calls_log,
            "chunks_retrieved": chunks_retrieved,
            "total_tool_calls": len(tool_calls_log)
        }

    except Exception as e:
        if verbose:
            print(f"ERROR: {e}")
        return {
            "question": question,
            "answer": f"Error: {e}",
            "tool_calls": tool_calls_log,
            "chunks_retrieved": chunks_retrieved,
            "total_tool_calls": len(tool_calls_log),
            "error": str(e)
        }


def analyze_failure(result: dict, expected: str) -> dict:
    """Analyze why a question failed."""

    issues = []

    # Check if chunks were retrieved
    if not result.get("chunks_retrieved"):
        issues.append("NO_CHUNKS_RETRIEVED - Agent never called get_chunk/get_chunks")

    # Check if answer is raw JSON
    answer = result.get("answer", "")
    if answer and (answer.startswith("{") or answer.startswith("found:")):
        issues.append("RAW_OUTPUT_RETURNED - Agent returned raw tool output instead of synthesizing answer")

    # Check tool call patterns
    tool_names = [tc["tool"] for tc in result.get("tool_calls", [])]

    if "resolve_entity_or_topic" in tool_names and "explore_neighbors" not in tool_names:
        issues.append("STOPPED_AT_RESOLVE - Agent stopped after resolve without exploring")

    if "explore_neighbors" in tool_names and "get_chunk" not in tool_names and "get_chunks" not in tool_names:
        issues.append("STOPPED_AT_EXPLORE - Agent stopped after explore without getting chunks")

    # Check for failed searches
    for tc in result.get("tool_calls", []):
        if tc["tool"] == "resolve_entity_or_topic":
            query = tc["args"].get("query", "")
            # Check if search term seems too specific
            if len(query.split()) > 3:
                issues.append(f"OVERLY_SPECIFIC_SEARCH - Searched for '{query}' which may be too specific")

    return {
        "issues": issues,
        "tool_sequence": tool_names,
        "chunks_count": len(result.get("chunks_retrieved", [])),
        "suggestion": get_suggestion(issues)
    }


def get_suggestion(issues: list) -> str:
    """Get improvement suggestions based on issues."""
    suggestions = []

    if "NO_CHUNKS_RETRIEVED" in issues:
        suggestions.append("Agent must call get_chunk/get_chunks to retrieve source evidence")

    if "RAW_OUTPUT_RETURNED" in issues:
        suggestions.append("Agent must synthesize a natural language answer from tool outputs")

    if "STOPPED_AT_RESOLVE" in issues:
        suggestions.append("After resolving entities, agent must explore_neighbors to find relationships")

    if "STOPPED_AT_EXPLORE" in issues:
        suggestions.append("After exploring, agent must get_chunk to retrieve specific facts (dates, numbers)")

    if any("OVERLY_SPECIFIC" in i for i in issues):
        suggestions.append("Use broader search terms, try synonyms or related concepts")

    return "; ".join(suggestions) if suggestions else "No specific suggestion"


async def main():
    parser = argparse.ArgumentParser(description="Diagnose QA evaluation failures")
    parser.add_argument("--results", default="eval_results_20251226_014902.json",
                        help="Evaluation results JSON file")
    parser.add_argument("--question-id", "-q", type=int, help="Analyze specific question ID")
    parser.add_argument("--only-failed", "-f", action="store_true", default=True,
                        help="Only analyze failed questions")
    parser.add_argument("--limit", "-l", type=int, default=5,
                        help="Limit number of questions to analyze")
    parser.add_argument("--transport", "-t", choices=["stdio", "sse"], default="stdio",
                        help="Transport type: 'stdio' (spawns process) or 'sse' (connects to running server)")
    args = parser.parse_args()

    # Load results
    results_path = args.results
    if not os.path.isabs(results_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_path = os.path.join(project_root, results_path)

    with open(results_path) as f:
        eval_data = json.load(f)

    results = eval_data.get("results", [])

    # Filter to failed questions
    if args.question_id:
        to_analyze = [r for r in results if r["question_id"] == args.question_id]
    elif args.only_failed:
        to_analyze = [r for r in results if r["verdict"] in ["incorrect", "partially_correct"]]
    else:
        to_analyze = results

    to_analyze = to_analyze[:args.limit]

    print(f"\n{'#'*80}")
    print(f"DIAGNOSING {len(to_analyze)} QUESTIONS")
    print(f"{'#'*80}\n")

    diagnoses = []

    for item in to_analyze:
        q_id = item["question_id"]
        question = item["question"]
        expected = item["expected_answer"]
        original_verdict = item["verdict"]

        print(f"\n{'='*80}")
        print(f"Question {q_id} (Original verdict: {original_verdict})")
        print(f"Expected: {expected}")
        print(f"{'='*80}")

        # Run agent with logging
        result = await run_agent_with_logging(question, verbose=True, transport=args.transport)

        # Analyze the failure
        analysis = analyze_failure(result, expected)

        print(f"\n📊 ANALYSIS:")
        print(f"   Issues: {analysis['issues']}")
        print(f"   Tool sequence: {' → '.join(analysis['tool_sequence'])}")
        print(f"   Chunks retrieved: {analysis['chunks_count']}")
        print(f"   Suggestion: {analysis['suggestion']}")

        diagnoses.append({
            "question_id": q_id,
            "question": question,
            "expected": expected,
            "original_verdict": original_verdict,
            "agent_answer": result.get("answer"),
            "analysis": analysis,
            "tool_calls": result.get("tool_calls", []),
            "chunks_retrieved": result.get("chunks_retrieved", [])
        })

    # Summary
    print(f"\n\n{'#'*80}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'#'*80}")

    all_issues = []
    for d in diagnoses:
        all_issues.extend(d["analysis"]["issues"])

    from collections import Counter
    issue_counts = Counter(all_issues)

    print("\nMost Common Issues:")
    for issue, count in issue_counts.most_common():
        print(f"  {count}x - {issue}")

    # Save detailed diagnoses
    output_path = f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(diagnoses, f, indent=2, default=str)

    print(f"\nDetailed diagnoses saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
