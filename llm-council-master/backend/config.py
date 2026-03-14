"""
llm-council-master/backend/config.py

Central model registry for the LLM Council.
─────────────────────────────────────────────
TO CHANGE MODELS: edit COUNCIL_MODELS and/or CHAIRMAN_MODEL here.
No other file needs to be modified.
"""

from __future__ import annotations

# ── Council members (Stage 1 brainstorm) ──────────────────────────────────────
# Add, remove, or swap any OpenRouter-compatible model identifier.
# Format: "provider/model-name"  →  https://openrouter.ai/models
COUNCIL_MODELS: list[str] = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-8b-instruct:free",
]

# ── Chairman (Stage 3 synthesis) ──────────────────────────────────────────────
CHAIRMAN_MODEL: str = "openai/gpt-4o"

# ── OpenRouter endpoint ───────────────────────────────────────────────────────
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_ENDPOINT: str = f"{OPENROUTER_BASE_URL}/chat/completions"

# ── Generation parameters ─────────────────────────────────────────────────────
COUNCIL_TEMPERATURE: float = 0.7   # creativity for brainstorm stage
CHAIRMAN_TEMPERATURE: float = 0.4  # more deterministic for synthesis
MAX_TOKENS_COUNCIL: int = 800
MAX_TOKENS_CHAIRMAN: int = 1200

# ── HTTP timeouts (seconds) ───────────────────────────────────────────────────
REQUEST_TIMEOUT: float = 45.0
