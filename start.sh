#!/usr/bin/env bash
# start.sh — Launch the Bytecamp LLM Council (macOS / Linux)
# Usage: chmod +x start.sh && ./start.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================"
echo " Bytecamp LLM Council — Startup"
echo "======================================"

# ── 1. Install council backend dependencies ───────────────────────────────────
echo "[1/3] Installing council backend dependencies..."
cd "$PROJECT_ROOT/llm-council-master"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt
# Also install root-level deps (core/ imports openai, pydantic, etc.)
pip install -q -r "$PROJECT_ROOT/requirements.txt"

# ── 2. Start Council FastAPI backend (port 8001) ──────────────────────────────
echo "[2/3] Starting Council API on http://localhost:8001 ..."
cd "$PROJECT_ROOT/llm-council-master"
osascript -e "tell application \"Terminal\" to do script \"
  cd '$PROJECT_ROOT/llm-council-master' && \
  source .venv/bin/activate && \
  PYTHONPATH='$PROJECT_ROOT' python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
\"" 2>/dev/null || \
  (PYTHONPATH="$PROJECT_ROOT" python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload &)

# ── 3. Start Vite frontend (port 5173) ────────────────────────────────────────
FRONTEND_DIR="$PROJECT_ROOT/llm-council-master/frontend"
if [ -d "$FRONTEND_DIR" ]; then
  echo "[3/3] Starting frontend on http://localhost:5173 ..."
  osascript -e "tell application \"Terminal\" to do script \"
    cd '$FRONTEND_DIR' && npm run dev
  \"" 2>/dev/null || (cd "$FRONTEND_DIR" && npm run dev &)
else
  echo "[3/3] No frontend directory found — skipping."
fi

echo ""
echo "======================================"
echo " Services:"
echo "  Council API  →  http://localhost:8001"
echo "  API Docs     →  http://localhost:8001/docs"
echo "  Frontend     →  http://localhost:5173"
echo "======================================"
