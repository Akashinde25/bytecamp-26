@echo off
REM start.bat — Launch the Bytecamp LLM Council (Windows)
REM Usage: double-click or run from cmd

echo ======================================
echo  Bytecamp LLM Council — Startup
echo ======================================

SET PROJECT_ROOT=%~dp0

REM ── 1. Start Council FastAPI backend (port 8001) ──────────────────────────
echo [1/2] Starting Council API on http://localhost:8001 ...
start "LLM Council Backend" cmd /k ^
  "cd /d "%PROJECT_ROOT%llm-council-master" && ^
   (if not exist .venv python -m venv .venv) && ^
   .\.venv\Scripts\activate.bat && ^
   pip install -q -r requirements.txt && ^
   pip install -q -r ..\requirements.txt && ^
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload"

REM ── 2. Start Vite frontend (port 5173) ────────────────────────────────────
IF EXIST "%PROJECT_ROOT%llm-council-master\frontend" (
  echo [2/2] Starting frontend on http://localhost:5173 ...
  start "LLM Council Frontend" cmd /k ^
    "cd /d "%PROJECT_ROOT%llm-council-master\frontend" && npm run dev"
) ELSE (
  echo [2/2] No frontend directory found — skipping.
)

echo.
echo ======================================
echo  Services:
echo   Council API  -^>  http://localhost:8001
echo   API Docs     -^>  http://localhost:8001/docs
echo   Frontend     -^>  http://localhost:5173
echo ======================================
