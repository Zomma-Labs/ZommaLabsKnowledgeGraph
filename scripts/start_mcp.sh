#!/bin/bash
# MCP Server startup script for Claude Desktop
# This script ensures proper environment initialization

# Load environment variables from common locations
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

if [ -f "$HOME/.profile" ]; then
    source "$HOME/.profile"
fi

# Load .env file if it exists
if [ -f "/home/rithv/Programming/Startups/ZommaLabsKG/.env" ]; then
    set -a
    source "/home/rithv/Programming/Startups/ZommaLabsKG/.env"
    set +a
fi

# Ensure uv is in PATH (common install locations)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Change to project directory
cd /home/rithv/Programming/Startups/ZommaLabsKG

# Log startup for debugging (comment out in production)
echo "Starting MCP server at $(date)" >> /tmp/mcp_server.log
echo "PATH: $PATH" >> /tmp/mcp_server.log
echo "GOOGLE_API_KEY set: $([ -n \"$GOOGLE_API_KEY\" ] && echo 'yes' || echo 'no')" >> /tmp/mcp_server.log
echo "NEO4J_URI: $NEO4J_URI" >> /tmp/mcp_server.log

# Run the MCP server
exec uv run python -m src.querying_system.mcp_server
