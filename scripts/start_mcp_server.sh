#!/bin/bash
# Start the ZommaGraph MCP Server in SSE mode
#
# Usage:
#   ./scripts/start_mcp_server.sh          # Start server (foreground)
#   ./scripts/start_mcp_server.sh --daemon # Start server (background)
#   ./scripts/start_mcp_server.sh --stop   # Stop background server

set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

PID_FILE="$PROJECT_ROOT/.mcp_server.pid"
LOG_FILE="$PROJECT_ROOT/logs/mcp_server.log"

HOST="${MCP_HOST:-127.0.0.1}"
PORT="${MCP_PORT:-8765}"

start_foreground() {
    echo "🚀 Starting ZommaGraph MCP Server on http://$HOST:$PORT"
    echo "   Press Ctrl+C to stop"
    echo ""
    uv run python -m src.agents.mcp_server --sse --host "$HOST" --port "$PORT"
}

start_daemon() {
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "⚠️  Server already running (PID: $OLD_PID)"
            echo "   Use: ./scripts/start_mcp_server.sh --stop"
            exit 1
        fi
    fi
    
    mkdir -p "$(dirname "$LOG_FILE")"
    
    echo "🚀 Starting ZommaGraph MCP Server (daemon mode)"
    echo "   URL:  http://$HOST:$PORT"
    echo "   Logs: $LOG_FILE"
    
    nohup uv run python -m src.agents.mcp_server --sse --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    
    sleep 2
    
    if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "✅ Server started (PID: $(cat "$PID_FILE"))"
    else
        echo "❌ Server failed to start. Check logs: $LOG_FILE"
        exit 1
    fi
}

stop_daemon() {
    if [ ! -f "$PID_FILE" ]; then
        echo "⚠️  No PID file found. Server may not be running."
        exit 0
    fi
    
    PID=$(cat "$PID_FILE")
    
    if kill -0 "$PID" 2>/dev/null; then
        echo "🛑 Stopping server (PID: $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "✅ Server stopped"
    else
        echo "⚠️  Server not running (stale PID file)"
        rm -f "$PID_FILE"
    fi
}

case "${1:-}" in
    --daemon|-d)
        start_daemon
        ;;
    --stop|-s)
        stop_daemon
        ;;
    *)
        start_foreground
        ;;
esac
