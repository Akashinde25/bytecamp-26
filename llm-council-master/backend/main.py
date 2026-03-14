"""
llm-council-master/backend/main.py

FastAPI server for the LLM Council.
Runs on port 8001.

Endpoints:
  POST /council/query  — run the 3-stage council pipeline
  GET  /health         — liveness probe
  GET  /council/models — list currently configured models
"""

from __future__ import annotations

import logging

import os
from pathlib import Path

from dotenv import load_dotenv

# Explicitly load council .env (contains OPENROUTER_API_KEY)
_COUNCIL_DIR = Path(__file__).resolve().parent.parent  # llm-council-master/
load_dotenv(_COUNCIL_DIR / ".env", override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import CHAIRMAN_MODEL, COUNCIL_MODELS
from backend.council import run_full_council

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Bytecamp LLM Council API",
    description=(
        "3-stage negotiation council: LLM brainstorm → "
        "deterministic Python scoring → Chairman synthesis."
    ),
    version="1.0.0",
)

# Allow Vite dev server (localhost:5173) and any localhost variant
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CouncilQueryRequest(BaseModel):
    query: str = Field(..., description="The negotiation question to send to the council.")
    domain: str = Field(
        default="cloud",
        description="Domain to use as context (must match a file in /domains/).",
    )


class Stage1Entry(BaseModel):
    model: str
    label: str
    response: str


class Stage2Entry(BaseModel):
    model: str
    label: str
    ranking: str
    score: float
    buyer_score: float
    seller_score: float
    proposed_deal: dict


class Stage3Result(BaseModel):
    model: str
    response: str


class CouncilResponse(BaseModel):
    stage1: list[Stage1Entry]
    stage2: list[Stage2Entry]
    stage3: Stage3Result
    metadata: dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    """Liveness probe."""
    return {"status": "ok", "service": "bytecamp-llm-council"}


@app.get("/council/models", tags=["Council"])
async def list_models():
    """Return the currently configured council models and chairman."""
    return {
        "council_models": COUNCIL_MODELS,
        "chairman_model": CHAIRMAN_MODEL,
    }


@app.post("/council/query", response_model=CouncilResponse, tags=["Council"])
async def council_query(request: CouncilQueryRequest):
    """
    Run the full 3-stage LLM Council pipeline.

    - Stage 1: 4 LLMs brainstorm strategies with embedded JSON deal proposals.
    - Stage 2: BytecampAgentEngine scores each deal deterministically.
    - Stage 3: Chairman LLM synthesizes the final recommendation.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info("Council query received | domain=%s | query=%.80s…", request.domain, request.query)

    try:
        stage1, stage2, stage3, metadata = await run_full_council(
            user_query=request.query,
            domain=request.domain,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Council pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Council pipeline error: {exc}") from exc

    return CouncilResponse(
        stage1=[Stage1Entry(**e) for e in stage1],
        stage2=[Stage2Entry(**e) for e in stage2],
        stage3=Stage3Result(**stage3),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=True)
