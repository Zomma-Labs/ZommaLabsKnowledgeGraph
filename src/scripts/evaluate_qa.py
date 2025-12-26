"""
QA Evaluation Script
====================

Evaluates the Knowledge Graph agent against a set of Q&A pairs.
Uses an LLM judge to determine if the agent's answers are correct.

Features:
- Batched agent queries for speed
- Batched judge evaluations for speed
- Timing metrics per question and overall

Usage:
    uv run src/scripts/evaluate_qa.py [--qa-file path/to/qa.json] [--limit N] [--verbose] [--batch-size N]
"""

import os
import sys
import json
import asyncio
import argparse
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


# === CONFIGURATION ===
MAX_TOOL_ITERATIONS = 10
DEFAULT_QA_FILE = "Alphabet_QA.json"
DEFAULT_BATCH_SIZE = 5  # Number of concurrent agent queries

# MCP Server options
MCP_SERVERS = {
    "original": "src.agents.mcp_server",
    "optimized": "src.agents.mcp_server_optimized",
}

# SSE Server URL (for persistent server mode)
MCP_SSE_URL = "http://127.0.0.1:8765/sse"


def get_chat_model(temperature: float = 0):
    """Get an LLM with fallback support for different providers."""
    # Try Gemini first (best for grounded responses)
    # if os.getenv("GOOGLE_API_KEY"):
    #     from langchain_google_genai import ChatGoogleGenerativeAI
    #     return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)

    # Try Grok
    if os.getenv("XAI_API_KEY"):
        from langchain_xai import ChatXAI
        return ChatXAI(model="grok-4-1-fast-non-reasoning", temperature=temperature)

    # Try OpenAI
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=temperature)

    raise ValueError(
        "No LLM API key found. Set one of: GOOGLE_API_KEY, XAI_API_KEY, or OPENAI_API_KEY"
    )


class JudgeVerdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    UNANSWERABLE = "unanswerable"  # When agent correctly says info not found


class JudgeResult(BaseModel):
    """Structured output from the LLM judge."""
    verdict: JudgeVerdict = Field(description="The verdict on the answer")
    reasoning: str = Field(description="Brief explanation of the verdict")
    key_facts_matched: list[str] = Field(default_factory=list, description="Key facts from expected answer that were present")
    key_facts_missing: list[str] = Field(default_factory=list, description="Key facts from expected answer that were missing")


@dataclass
class EvalResult:
    """Result of evaluating a single Q&A pair."""
    question_id: int
    question: str
    expected_answer: str
    agent_answer: str
    verdict: JudgeVerdict
    reasoning: str
    question_type: str
    difficulty: str
    agent_time_sec: float = 0.0  # Time for agent to answer
    judge_time_sec: float = 0.0  # Time for judge to evaluate
    key_facts_matched: list[str] = field(default_factory=list)
    key_facts_missing: list[str] = field(default_factory=list)
    error: Optional[str] = None


# === AGENT SYSTEM PROMPT ===
AGENT_SYSTEM_PROMPT = """You are a Knowledge Graph Research Agent. You must answer questions by searching a graph database. 
Follow the tool instructions provided to you"""

# AGENT_SYSTEM_PROMPT = """You are an autonomous Knowledge Graph Research Agent. You answer questions by searching a graph database.

# ## CRITICAL: YOU ARE AUTONOMOUS
# - Do NOT ask the user questions like "Would you like me to explore...?"
# - Do NOT stop after one tool call - complete the entire workflow
# - Do NOT ask for permission - just search and provide the answer
# - You must provide a COMPLETE ANSWER in your final response

# ## Workflow (complete ALL steps automatically)
# 1. **resolve_entity_or_topic** - Find the exact entity name
# 2. **explore_neighbors** - See ALL relationships for that entity  
# 3. **get_chunk** - Get source text with specific details (dates, numbers, etc.)
# 4. **Provide your answer** - Include the specific information requested

# ## Tools
# - resolve_entity_or_topic(query, node_type) - Find entities/topics
# - explore_neighbors(entity_name) - See relationships
# - get_chunk(entity_one, entity_two, edge_type) - Get source evidence

# ## Example (CORRECT behavior)
# Q: "What is COMPANY?"
# 1. resolve_entity_or_topic("COMPANY", "Entity") → found
# 2. explore_neighbors("COMPANY") → shows relationships and what kind of company it is
# 3. get_chunk("COMPANY", "some_entity", "RELATED_TO") → gets detailed description
# 4. ANSWER: "COMPANY is an American multinational technology conglomerate..." (from chunks)

# ## Rules
# - Use ONLY information from tool responses
# - Always complete the full workflow before answering
# - Provide a direct, complete answer"""


# === JUDGE SYSTEM PROMPT ===
JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating whether an AI agent's answer correctly addresses a question.

You will be given:
1. A QUESTION
2. The EXPECTED ANSWER (ground truth)
3. The AGENT'S ANSWER (what the AI produced)

Your task is to determine if the agent's answer is correct.

## Verdict Options:
- **correct**: The agent's answer contains all key facts from the expected answer (wording can differ)
- **partially_correct**: The agent's answer contains some but not all key facts
- **incorrect**: The agent's answer is wrong or contradicts the expected answer
- **unanswerable**: The agent correctly indicated the information was not found in its knowledge base (this is acceptable if the KB doesn't have the data)

## Important Guidelines:
- Focus on FACTUAL accuracy, not exact wording
- Dates, numbers, and names must be accurate
- The agent may provide additional context - this is fine as long as core facts are correct
- If the agent says "not found" but the expected answer exists, consider whether the KB might genuinely not have this info
- Be fair but rigorous

Provide your verdict along with brief reasoning."""


def count_tool_calls(messages) -> int:
    """Count the number of tool calls in the message history."""
    from langchain_core.messages import ToolMessage
    return sum(1 for msg in messages if isinstance(msg, ToolMessage))


async def create_agent_graph(mcp_server: str = "original", transport: str = "stdio"):
    """Create the KG agent with MCP tools.

    Args:
        mcp_server: Which MCP server to use ("original" or "optimized")
        transport: Transport type - "stdio" (spawns process) or "sse" (connects to running server)
    """
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Select MCP server module
    server_module = MCP_SERVERS.get(mcp_server, MCP_SERVERS["original"])
    
    if transport == "sse":
        print(f"  Connecting to MCP server via SSE: {MCP_SSE_URL}")
        mcp_client = MultiServerMCPClient(
            {
                "zommagraph": {
                    "url": MCP_SSE_URL,
                    "transport": "sse",
                }
            }
        )
    else:
        print(f"  Using MCP server (stdio): {server_module}")
        mcp_client = MultiServerMCPClient(
            {
                "zommagraph": {
                    "command": "python",
                    "args": ["-m", server_module],
                    "transport": "stdio",
                    "env": {
                        **os.environ,
                        "PYTHONPATH": project_root
                    }
                }
            }
        )

    # Get tools from MCP server
    tools = await mcp_client.get_tools()

    # Initialize the LLM (with fallback support)
    model = get_chat_model(temperature=0)
    print(f"  Using LLM: {model.__class__.__name__}")
    model_with_tools = model.bind_tools(tools)

    def call_model(state: MessagesState) -> dict:
        """Call the model with the current state."""
        messages = state["messages"]

        # Add system prompt if this is the first message
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + list(messages)

        # Check if we've exceeded the iteration limit
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
        """Determine whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        tool_call_count = count_tool_calls(messages)
        if tool_call_count >= MAX_TOOL_ITERATIONS:
            return END

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END

    # Build the graph
    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile(), mcp_client


def create_judge_llm():
    """Create the LLM judge (uses same provider as agent)."""
    llm = get_chat_model(temperature=0)
    return llm.with_structured_output(JudgeResult)


async def ask_agent(graph, question: str) -> tuple[str, float]:
    """Ask the agent a question and return its answer with timing."""
    start = time.time()
    try:
        response = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 25}
        )

        # Get the final AI message
        for msg in reversed(response["messages"]):
            if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_calls'):
                return msg.content, time.time() - start
            if hasattr(msg, 'content') and msg.content and hasattr(msg, 'tool_calls') and not msg.tool_calls:
                return msg.content, time.time() - start

        return "No response generated", time.time() - start
    except Exception as e:
        return f"Error: {str(e)}", time.time() - start


async def judge_answer_async(judge_llm, question: str, expected: str, agent_answer: str) -> tuple[JudgeResult, float]:
    """Have the LLM judge evaluate the agent's answer (async with timing)."""
    prompt = f"""## QUESTION
{question}

## EXPECTED ANSWER
{expected}

## AGENT'S ANSWER
{agent_answer}

Evaluate whether the agent's answer is correct."""

    start = time.time()
    try:
        # Run sync invoke in executor to make it async
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: judge_llm.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
        )
        return result, time.time() - start
    except Exception as e:
        return JudgeResult(
            verdict=JudgeVerdict.INCORRECT,
            reasoning=f"Judge error: {str(e)}",
            key_facts_matched=[],
            key_facts_missing=[]
        ), time.time() - start


async def evaluate_qa_pairs(
    qa_pairs: list[dict],
    verbose: bool = False,
    limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    mcp_server: str = "original",
    transport: str = "stdio"
) -> list[EvalResult]:
    """Evaluate all Q&A pairs with batching for speed."""

    print("Initializing agent...")
    graph, mcp_client = await create_agent_graph(mcp_server=mcp_server, transport=transport)
    judge_llm = create_judge_llm()

    pairs_to_eval = qa_pairs[:limit] if limit else qa_pairs
    total = len(pairs_to_eval)

    print(f"\nEvaluating {total} questions (batch size: {batch_size})...\n")
    print("-" * 80)

    results = []
    total_start = time.time()

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = pairs_to_eval[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\n[Batch {batch_num}/{total_batches}] Processing questions {batch_start + 1}-{batch_end}...")

        # Phase 1: Ask all questions in batch concurrently
        agent_start = time.time()
        agent_tasks = [ask_agent(graph, qa["question"]) for qa in batch]
        agent_results = await asyncio.gather(*agent_tasks)
        agent_batch_time = time.time() - agent_start
        print(f"  Agent answers: {agent_batch_time:.1f}s ({agent_batch_time/len(batch):.1f}s/q avg)")

        # Phase 2: Judge all answers in batch concurrently
        judge_start = time.time()
        judge_tasks = [
            judge_answer_async(judge_llm, qa["question"], qa["answer"], agent_answer)
            for qa, (agent_answer, _) in zip(batch, agent_results)
        ]
        judge_results = await asyncio.gather(*judge_tasks)
        judge_batch_time = time.time() - judge_start
        print(f"  Judge evals:   {judge_batch_time:.1f}s ({judge_batch_time/len(batch):.1f}s/q avg)")

        # Combine results
        for i, qa in enumerate(batch):
            agent_answer, agent_time = agent_results[i]
            judge_result, judge_time = judge_results[i]

            q_id = qa.get("id", batch_start + i + 1)
            question = qa["question"]
            expected = qa["answer"]
            q_type = qa.get("type", "unknown")
            difficulty = qa.get("difficulty", "unknown")

            result = EvalResult(
                question_id=q_id,
                question=question,
                expected_answer=expected,
                agent_answer=agent_answer,
                verdict=judge_result.verdict,
                reasoning=judge_result.reasoning,
                question_type=q_type,
                difficulty=difficulty,
                agent_time_sec=agent_time,
                judge_time_sec=judge_time,
                key_facts_matched=judge_result.key_facts_matched,
                key_facts_missing=judge_result.key_facts_missing
            )
            results.append(result)

            # Print individual result
            verdict_symbol = {
                JudgeVerdict.CORRECT: "[PASS]",
                JudgeVerdict.PARTIALLY_CORRECT: "[PARTIAL]",
                JudgeVerdict.INCORRECT: "[FAIL]",
                JudgeVerdict.UNANSWERABLE: "[N/A]"
            }
            global_idx = batch_start + i + 1
            print(f"  Q{q_id}: {verdict_symbol.get(judge_result.verdict, '[?]')} ({agent_time:.1f}s)")

            # Always show details for non-correct answers
            is_not_correct = judge_result.verdict != JudgeVerdict.CORRECT
            if is_not_correct or verbose:
                print(f"       Question: {question}")
                print(f"       Expected: {expected}")
                print(f"       Agent:    {agent_answer[:500]}{'...' if len(agent_answer) > 500 else ''}")
                print(f"       Reason:   {judge_result.reasoning}")
                if judge_result.key_facts_matched:
                    print(f"       ✓ Matched: {', '.join(judge_result.key_facts_matched)}")
                if judge_result.key_facts_missing:
                    print(f"       ✗ Missing: {', '.join(judge_result.key_facts_missing)}")
                print()  # Blank line for readability

    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"Total evaluation time: {total_time:.1f}s ({total_time/total:.1f}s/question avg)")

    return results


def print_summary(results: list[EvalResult]):
    """Print evaluation summary statistics."""
    total = len(results)

    # Count verdicts
    counts = {v: 0 for v in JudgeVerdict}
    for r in results:
        counts[r.verdict] += 1

    # Calculate accuracy
    correct = counts[JudgeVerdict.CORRECT]
    partial = counts[JudgeVerdict.PARTIALLY_CORRECT]
    incorrect = counts[JudgeVerdict.INCORRECT]
    unanswerable = counts[JudgeVerdict.UNANSWERABLE]

    strict_accuracy = correct / total * 100 if total > 0 else 0
    lenient_accuracy = (correct + partial) / total * 100 if total > 0 else 0

    # Timing stats
    agent_times = [r.agent_time_sec for r in results]
    judge_times = [r.judge_time_sec for r in results]
    total_times = [r.agent_time_sec + r.judge_time_sec for r in results]

    avg_agent = sum(agent_times) / len(agent_times) if agent_times else 0
    avg_judge = sum(judge_times) / len(judge_times) if judge_times else 0
    avg_total = sum(total_times) / len(total_times) if total_times else 0

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal Questions: {total}")

    print(f"\n📊 Verdicts:")
    print(f"  Correct:           {correct:3d} ({correct/total*100:5.1f}%)")
    print(f"  Partially Correct: {partial:3d} ({partial/total*100:5.1f}%)")
    print(f"  Incorrect:         {incorrect:3d} ({incorrect/total*100:5.1f}%)")
    print(f"  Unanswerable:      {unanswerable:3d} ({unanswerable/total*100:5.1f}%)")

    print(f"\n🎯 Accuracy:")
    print(f"  Strict (correct only):     {strict_accuracy:5.1f}%")
    print(f"  Lenient (correct+partial): {lenient_accuracy:5.1f}%")

    print(f"\n⏱️  Timing (per question):")
    print(f"  Agent avg:  {avg_agent:5.1f}s")
    print(f"  Judge avg:  {avg_judge:5.1f}s")
    print(f"  Total avg:  {avg_total:5.1f}s")
    print(f"  Total time: {sum(total_times):.1f}s (sequential would be ~{sum(agent_times) + sum(judge_times):.1f}s)")

    # Breakdown by type
    types = {}
    for r in results:
        t = r.question_type
        if t not in types:
            types[t] = {"total": 0, "correct": 0, "times": []}
        types[t]["total"] += 1
        types[t]["times"].append(r.agent_time_sec)
        if r.verdict == JudgeVerdict.CORRECT:
            types[t]["correct"] += 1

    print(f"\n📈 Accuracy by Question Type:")
    for t, stats in sorted(types.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        avg_t = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
        print(f"  {t:20s}: {stats['correct']:2d}/{stats['total']:2d} ({acc:5.1f}%) - avg {avg_t:.1f}s")

    # Breakdown by difficulty
    difficulties = {}
    for r in results:
        d = r.difficulty
        if d not in difficulties:
            difficulties[d] = {"total": 0, "correct": 0, "times": []}
        difficulties[d]["total"] += 1
        difficulties[d]["times"].append(r.agent_time_sec)
        if r.verdict == JudgeVerdict.CORRECT:
            difficulties[d]["correct"] += 1

    print(f"\n📈 Accuracy by Difficulty:")
    for d, stats in sorted(difficulties.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        avg_t = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
        print(f"  {d:20s}: {stats['correct']:2d}/{stats['total']:2d} ({acc:5.1f}%) - avg {avg_t:.1f}s")

    print("=" * 80)


def save_results(results: list[EvalResult], output_path: str):
    """Save detailed results to JSON."""
    # Calculate timing stats
    agent_times = [r.agent_time_sec for r in results]
    judge_times = [r.judge_time_sec for r in results]

    data = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(results),
        "timing": {
            "avg_agent_time_sec": sum(agent_times) / len(agent_times) if agent_times else 0,
            "avg_judge_time_sec": sum(judge_times) / len(judge_times) if judge_times else 0,
            "avg_total_time_sec": (sum(agent_times) + sum(judge_times)) / len(results) if results else 0,
        },
        "summary": {
            "correct": sum(1 for r in results if r.verdict == JudgeVerdict.CORRECT),
            "partially_correct": sum(1 for r in results if r.verdict == JudgeVerdict.PARTIALLY_CORRECT),
            "incorrect": sum(1 for r in results if r.verdict == JudgeVerdict.INCORRECT),
            "unanswerable": sum(1 for r in results if r.verdict == JudgeVerdict.UNANSWERABLE),
        },
        "results": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "expected_answer": r.expected_answer,
                "agent_answer": r.agent_answer,
                "verdict": r.verdict.value,
                "reasoning": r.reasoning,
                "question_type": r.question_type,
                "difficulty": r.difficulty,
                "agent_time_sec": r.agent_time_sec,
                "judge_time_sec": r.judge_time_sec,
                "key_facts_matched": r.key_facts_matched,
                "key_facts_missing": r.key_facts_missing,
            }
            for r in results
        ]
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate KG Agent against Q&A pairs")
    parser.add_argument("--qa-file", default=DEFAULT_QA_FILE, help="Path to Q&A JSON file")
    parser.add_argument("--limit", type=int, help="Limit number of questions to evaluate")
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for concurrent queries")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--server", "-s", choices=["original", "optimized"], default="original",
                        help="MCP server version to use (default: original)")
    parser.add_argument("--transport", "-t", choices=["stdio", "sse"], default="stdio",
                        help="Transport type: 'stdio' (spawns process) or 'sse' (connects to running server)")
    args = parser.parse_args()

    # Load Q&A pairs
    qa_file = args.qa_file
    if not os.path.isabs(qa_file):
        # Look in project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        qa_file = os.path.join(project_root, qa_file)

    if not os.path.exists(qa_file):
        print(f"Error: Q&A file not found: {qa_file}")
        sys.exit(1)

    with open(qa_file) as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", data)  # Handle both formats
    print(f"Loaded {len(qa_pairs)} Q&A pairs from {qa_file}")

    # Run evaluation
    results = await evaluate_qa_pairs(
        qa_pairs,
        verbose=args.verbose,
        limit=args.limit,
        batch_size=args.batch_size,
        mcp_server=args.server,
        transport=args.transport
    )

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output:
        save_results(results, args.output)
    else:
        # Default output path
        output_path = os.path.join(
            os.path.dirname(qa_file),
            f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        save_results(results, output_path)


if __name__ == "__main__":
    asyncio.run(main())
