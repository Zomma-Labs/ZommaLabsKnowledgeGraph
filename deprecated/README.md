# Deprecated Query Files

These files were moved here on 2025-12-22 as they have been superseded by the new MCP Server approach.

## Why Deprecated?

The original query system used:
- **LangGraph Plan-Execute pattern** (`query_workflow.py`) - Complex multi-step agent
- **LangChain @tool decorators** (`agent_tools.py`) - Tools for the above workflow
- **Simple QueryAgent** (`query_agent.py`) - Single-pass vector search + LLM synthesis

These approaches had **stale schema references** (using `Entity` instead of `EntityNode`, `uri` instead of `uuid`, etc.) and didn't properly leverage the hypergraph model with EpisodicNode hubs.

## New Approach

The new **MCP Server** (`src/agents/mcp_server.py`) provides:
- FastMCP tool exposure for AI agents
- Built-in multi-tenancy via `group_id` scoping
- Correct schema using `EntityNode`, `EpisodicNode`, `fact_id` matching
- 3 focused tools: `resolve_entity_or_topic`, `explore_neighbors`, `get_chunk`
- Testable core logic separated from tool wrappers

## Files in this folder

| File | Original Location | Purpose |
|------|-------------------|---------|
| `query_workflow.py` | `src/workflows/` | LangGraph Plan-Execute query agent |
| `agent_tools.py` | `src/tools/` | LangChain tools for query_workflow |
| `query_agent.py` | `src/agents/` | Simple single-pass query agent |
| `test_query_agent.py` | `scripts/` | Test script for QueryAgent |
| `test_query_service.py` | `src/scripts/` | Test script for query service |

## Can I delete these?

Yes, once you've confirmed the MCP Server meets all your needs, you can safely delete this folder.
