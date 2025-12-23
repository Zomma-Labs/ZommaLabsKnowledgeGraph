"""
ZommaLabs Knowledge Graph Agent
===============================

LangGraph agent that uses the ZommaGraph MCP server to answer questions
about the financial knowledge graph.

Run with: langgraph dev
Test with: LangGraph Studio Web UI
"""

import os
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

# === AGENT CONFIGURATION ===
# Maximum number of tool calls before forcing a response
MAX_TOOL_ITERATIONS = 10

# System prompt for the Knowledge Graph agent with early-stopping guidance
SYSTEM_PROMPT = """You are a Financial Knowledge Graph Research Assistant powered by ZommaLabs.

## CRITICAL RULE: ONLY USE INFORMATION FROM TOOL RESPONSES
You must ONLY use information that is explicitly returned by the tools. 
DO NOT use any external knowledge, assumptions, or information you may have been trained on.
If a tool does not return specific information, you DO NOT know that information.

## Available Tools
1. **resolve_entity_or_topic** - ALWAYS use this first to find exact entity/topic names
2. **explore_neighbors** - Use this to discover relationships connected to entities  
3. **get_chunk** - Use this to retrieve source evidence for specific relationships

## Workflow
For each question:
1. First, resolve any entities or topics mentioned in the question
2. Then explore their relationships to understand connections
3. Finally, retrieve the source chunks to get detailed evidence

## MANDATORY CITATION FORMAT
Every factual claim in your response MUST include a citation in this format:
- [DOC: document_id, CHUNK: chunk_id]

Example: "The Federal Reserve raised interest rates in March 2024 [DOC: fed_minutes_2024, CHUNK: abc123]."

If you cannot cite a specific chunk for a claim, DO NOT make that claim.

## When to Stop Searching
- If `resolve_entity_or_topic` returns 'found: false', the entity does NOT exist - STOP
- If `explore_neighbors` returns 'found: false', the entity has NO relationships - STOP
- After 2-3 failed search attempts, conclude with what you found (or didn't find)
- DO NOT keep searching with variations endlessly

## Response Rules
- NEVER make claims without citing the specific chunk that contains that information
- If information is not available, state: "This information was not found in the knowledge graph"
- List what you searched for and what was missing
- It's better to say "not found" than to fabricate information

Be thorough but efficient. If the data isn't there, report that and move on.
"""


def get_mcp_server_path() -> str:
    """Get the absolute path to the MCP server."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "mcp_server.py")


def count_tool_calls(messages: Sequence[BaseMessage]) -> int:
    """Count the number of tool calls in the message history."""
    count = 0
    for msg in messages:
        if isinstance(msg, ToolMessage):
            count += 1
    return count


async def make_graph():
    """
    Create and return the LangGraph agent with MCP tools.
    
    This function is the entry point for the LangGraph API server.
    It connects to the MCP server and creates a ReAct-style agent.
    """
    # Initialize the MCP client connecting to our ZommaGraph server
    mcp_client = MultiServerMCPClient(
        {
            "zommagraph": {
                "command": "python",
                "args": ["-m", "src.agents.mcp_server"],
                "transport": "stdio",
                "env": {
                    **os.environ,
                    "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                }
            }
        }
    )
    
    # Get tools from MCP server
    tools = await mcp_client.get_tools()
    
    # Initialize the LLM - Using xAI Grok for fast, efficient responses
    # Requires XAI_API_KEY environment variable
    model = ChatXAI(
        model="grok-4-1-fast-non-reasoning",
        temperature=0,
    )
    # Alternatives:
    # model = ChatOpenAI(model="gpt-4o", temperature=0)
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    # Bind tools to model
    model_with_tools = model.bind_tools(tools)
    
    # Define the call_model node
    def call_model(state: MessagesState) -> dict:
        """Call the model with the current state."""
        messages = state["messages"]
        
        # Add system prompt if this is the first message
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        
        # Check if we've exceeded the iteration limit
        tool_call_count = count_tool_calls(messages)
        if tool_call_count >= MAX_TOOL_ITERATIONS:
            # Force a final response without tools
            from langchain_core.messages import SystemMessage
            force_stop_msg = SystemMessage(
                content=f"SYSTEM: You have made {tool_call_count} tool calls. You MUST now provide your final answer based on what you've found. Do NOT make any more tool calls."
            )
            messages = list(messages) + [force_stop_msg]
            # Use model without tools to force a text response
            response = model.invoke(messages)
        else:
            response = model_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    # Custom routing function that respects iteration limits
    def should_continue(state: MessagesState) -> str:
        """Determine whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check iteration limit
        tool_call_count = count_tool_calls(messages)
        if tool_call_count >= MAX_TOOL_ITERATIONS:
            return END
        
        # Use the standard tools_condition logic
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
    
    # Build the graph
    builder = StateGraph(MessagesState)
    
    # Add nodes
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    
    # Add edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )
    builder.add_edge("tools", "agent")
    
    # Compile the graph with recursion limit as a safety net
    graph = builder.compile()
    
    return graph


# For direct testing (not through LangGraph server)
async def test_agent():
    """Test the agent directly."""
    graph = await make_graph()
    
    # Test query
    response = await graph.ainvoke(
        {"messages": [HumanMessage(content="What entities are related to inflation in the knowledge graph?")]},
        config={"recursion_limit": 25}  # Additional safety limit
    )
    
    # Print the final response
    for msg in response["messages"]:
        print(f"\n{msg.__class__.__name__}: {msg.content}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"  Tool calls: {msg.tool_calls}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
