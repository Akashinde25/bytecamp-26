"""
backend/main.py

FastAPI application for the domain-agnostic multi-agent negotiation simulator.

Endpoints
---------
POST   /negotiation/start                 — Start a new negotiation session.
GET    /negotiation/{session_id}/status   — Live status of a running session.
GET    /negotiation/{session_id}/result   — Final result of a completed session.
WS     /negotiation/{session_id}/live     — WebSocket stream of round updates.
GET    /domains                           — List available domain configs.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is on the path so core.* imports resolve correctly
import os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core.environment import initialize_session, list_available_domains
from core.negotiation import NegotiationSession

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# In-memory stores
# --------------------------------------------------------------------------- #

sessions: dict[str, NegotiationSession] = {}
round_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Negotiation simulator starting …")
    yield
    logger.info("Negotiation simulator shutting down.")


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Multi-Agent Negotiation Simulator",
    description=(
        "A domain-agnostic simulator where autonomous GPT-4o agents negotiate "
        "over shared resources with live WebSocket streaming."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / Response models
# --------------------------------------------------------------------------- #

class StartNegotiationRequest(BaseModel):
    """Request body for POST /negotiation/start."""

    domain: str
    session_id: str


class StartNegotiationResponse(BaseModel):
    """Immediate acknowledgement returned when a session is kicked off."""

    session_id: str
    status: str
    domain: str


# --------------------------------------------------------------------------- #
# Background negotiation runner
# --------------------------------------------------------------------------- #

async def run_negotiation(session_id: str) -> None:
    """
    Run a negotiation session asynchronously and push each round log
    plus the final completion message into the session's asyncio.Queue.

    This function is called via `asyncio.create_task` so it does not block
    the HTTP response.
    """
    session = sessions.get(session_id)
    if session is None:
        logger.error("run_negotiation: session '%s' not found", session_id)
        return

    queue = round_queues.get(session_id)
    if queue is None:
        logger.error("run_negotiation: queue for '%s' not found", session_id)
        return

    # Wire the session's internal queue to the shared store so the WS consumer
    # can poll it.  (NegotiationSession creates its own Queue by default; we
    # replace it with the one registered in round_queues.)
    session._queue = queue

    logger.info("Background task: starting negotiation for session '%s'", session_id)
    try:
        await session.run()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Negotiation error for session '%s': %s", session_id, exc)
        error_payload: dict[str, Any] = {
            "type": "error",
            "session_id": session_id,
            "message": str(exc),
        }
        await queue.put(error_payload)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/", summary="Health check / root", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint — confirms the API is running."""
    return {
        "name": "Multi-Agent Negotiation Simulator",
        "version": "1.0.0",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/domains", summary="List available negotiation domains")
async def get_domains() -> dict[str, list[str]]:
    """Return the names of all JSON domain configs found in /domains/."""
    return {"domains": list_available_domains()}


@app.post(
    "/negotiation/start",
    response_model=StartNegotiationResponse,
    summary="Start a new negotiation session",
)
async def start_negotiation(
    request: StartNegotiationRequest,
    background_tasks: BackgroundTasks,
) -> StartNegotiationResponse:
    """
    Validate the domain, create a NegotiationSession, register a streaming
    queue, and kick off the negotiation as a background asyncio task.

    Returns immediately with `{"session_id", "status": "started", "domain"}`.
    """
    available = list_available_domains()
    if request.domain not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain '{request.domain}'. Available: {available}",
        )

    if request.session_id in sessions:
        raise HTTPException(
            status_code=409,
            detail=f"Session '{request.session_id}' already exists.",
        )

    try:
        session = initialize_session(request.domain, request.session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Register session and its streaming queue
    sessions[request.session_id] = session
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    round_queues[request.session_id] = queue

    # Launch negotiation loop without blocking this request
    asyncio.create_task(run_negotiation(request.session_id))

    logger.info("Session '%s' created for domain '%s'", request.session_id, request.domain)

    return StartNegotiationResponse(
        session_id=request.session_id,
        status="started",
        domain=request.domain,
    )


@app.get(
    "/negotiation/{session_id}/status",
    summary="Get live status of a negotiation session",
)
async def get_status(session_id: str) -> dict[str, Any]:
    """
    Return the current round number and per-agent status snapshots.

    Returns
    -------
    ```json
    {
      "session_id": "...",
      "domain": "...",
      "current_round": 3,
      "agent_statuses": [{"name": ..., "role": ..., "status": ..., "current_offer": ...}],
      "is_complete": false
    }
    ```
    """
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session.status_payload()


@app.get(
    "/negotiation/{session_id}/result",
    summary="Get final result of a negotiation session",
)
async def get_result(session_id: str) -> dict[str, Any]:
    """
    Return the complete result including deal terms, metrics, and all round logs.

    Returns
    -------
    ```json
    {
      "session_id": "...",
      "domain": "...",
      "status": "converged | forced | in_progress",
      "deal": {...},
      "metrics": {"rounds_to_converge": ..., "agent_satisfaction_scores": {...}},
      "rounds_log": [...]
    }
    ```
    """
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session.result_payload()


# --------------------------------------------------------------------------- #
# WebSocket stream
# --------------------------------------------------------------------------- #

@app.websocket("/negotiation/{session_id}/live")
async def negotiation_live(websocket: WebSocket, session_id: str) -> None:
    """
    Stream live round updates for a negotiation session over WebSocket.

    Each negotiation round produces a JSON message:
    ```json
    {"type": "round", "round": 1, "offers": {...}, "proposed_deal": {...}, ...}
    ```
    When the session ends a final message is sent:
    ```json
    {"type": "complete", "result": {"deal": {...}, "metrics": {...}}}
    ```
    """
    await websocket.accept()
    logger.info("WebSocket connected for session '%s'", session_id)

    queue = round_queues.get(session_id)
    if queue is None:
        await websocket.send_json(
            {"type": "error", "message": f"Session '{session_id}' not found."}
        )
        await websocket.close()
        return

    try:
        while True:
            try:
                # Poll every 0.5 s so we can detect client disconnects quickly
                message: dict[str, Any] = await asyncio.wait_for(
                    queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                # Nothing in the queue yet — keep waiting
                continue

            await websocket.send_json(message)

            if message.get("type") in ("complete", "error"):
                logger.info(
                    "WebSocket stream ended for session '%s' (type=%s)",
                    session_id,
                    message.get("type"),
                )
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session '%s'", session_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("WebSocket error for session '%s': %s", session_id, exc)
    finally:
        await websocket.close()
