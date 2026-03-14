"""
llm-council-master/backend/openrouter.py

Async HTTP client for the OpenRouter API.
Provides fire-and-forget fan-out to multiple models in parallel.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from backend.config import (
    CHAIRMAN_TEMPERATURE,
    COUNCIL_TEMPERATURE,
    MAX_TOKENS_CHAIRMAN,
    MAX_TOKENS_COUNCIL,
    OPENROUTER_CHAT_ENDPOINT,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        logger.warning("OPENROUTER_API_KEY is not set — requests will fail.")
    return key


def _build_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173",   # Required by OpenRouter
        "X-Title": "Bytecamp LLM Council",
    }


async def _call_model(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Send a single chat-completion request to OpenRouter.
    Returns the assistant's text content, or an error string on failure.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = await client.post(
            OPENROUTER_CHAT_ENDPOINT,
            headers=_build_headers(),
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    except httpx.HTTPStatusError as exc:
        logger.error("HTTP %s from OpenRouter for model %s: %s",
                     exc.response.status_code, model, exc.response.text[:200])
        return f"[ERROR] HTTP {exc.response.status_code} for model {model}"

    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error calling model %s: %s", model, exc)
        return f"[ERROR] {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def query_model(
    model: str,
    messages: list[dict[str, str]],
    is_chairman: bool = False,
) -> str:
    """
    Query a single OpenRouter model and return the response text.

    Args:
        model:        OpenRouter model identifier, e.g. 'openai/gpt-4o-mini'.
        messages:     Chat messages list in OpenAI format.
        is_chairman:  If True, uses chairman temperature/token settings.

    Returns:
        The model's response text (or an error string on failure).
    """
    temperature = CHAIRMAN_TEMPERATURE if is_chairman else COUNCIL_TEMPERATURE
    max_tokens = MAX_TOKENS_CHAIRMAN if is_chairman else MAX_TOKENS_COUNCIL

    async with httpx.AsyncClient() as client:
        return await _call_model(client, model, messages, temperature, max_tokens)


async def query_models_parallel(
    models: list[str],
    messages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Fan-out the same prompt to multiple models concurrently.

    Args:
        models:   List of OpenRouter model identifiers.
        messages: Shared chat messages to send to all models.

    Returns:
        List of dicts: [{"model": str, "response": str}, ...]
        Order mirrors the input `models` list.
        Failed model calls produce an error-prefixed response string
        rather than raising, so one bad model never blocks the others.
    """
    async with httpx.AsyncClient() as client:
        tasks = [
            _call_model(client, model, messages, COUNCIL_TEMPERATURE, MAX_TOKENS_COUNCIL)
            for model in models
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=False)

    return [
        {"model": model, "response": response}
        for model, response in zip(models, responses)
    ]
