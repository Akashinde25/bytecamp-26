"""
scenarios/test_integration.py

Integration test suite for the multi-agent negotiation simulator.

Tests
-----
1. GET /domains                          — returns at least one domain
2. POST /negotiation/start               — returns 200 with correct shape
3. POST /negotiation/start (bad domain)  — returns 400
4. POST /negotiation/start (duplicate)   — returns 409
5. GET  /negotiation/{id}/status         — returns correct shape
6. WS   /negotiation/{id}/live           — receives round + complete messages
7. GET  /negotiation/{id}/result         — valid result after WS completes
8. GET  /negotiation/{id}/status (404)   — unknown session returns 404
9. GET  /negotiation/{id}/result (404)   — unknown session returns 404

Usage
-----
    # With server running:
    uvicorn backend.main:app --reload --port 8000

    # In another terminal:
    python scenarios/test_integration.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
    import websockets
except ImportError:
    print("[ERROR] Missing test dependencies. Run: pip install httpx websockets")
    sys.exit(1)


BASE_URL = "http://localhost:8000"
WS_BASE  = "ws://localhost:8000"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _pass(name: str) -> None:
    print(f"  \033[92m✔ PASS\033[0m  {name}")


def _fail(name: str, reason: str) -> None:
    print(f"  \033[91m✘ FAIL\033[0m  {name}: {reason}")


def _header(title: str) -> None:
    print(f"\n\033[1m{'─' * 60}\033[0m")
    print(f"\033[1m  {title}\033[0m")
    print(f"\033[1m{'─' * 60}\033[0m")


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #

async def test_list_domains(client: httpx.AsyncClient) -> None:
    _header("Test 1 — GET /domains")
    r = await client.get(f"{BASE_URL}/domains")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "domains" in data, "Response missing 'domains' key"
    assert len(data["domains"]) > 0, "No domains returned"
    _pass(f"domains={data['domains']}")


async def test_start_negotiation(
    client: httpx.AsyncClient,
    session_id: str,
    domain: str = "logistics",
) -> None:
    _header("Test 2 — POST /negotiation/start")
    payload = {"domain": domain, "session_id": session_id}
    r = await client.post(f"{BASE_URL}/negotiation/start", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["session_id"] == session_id
    assert data["status"] == "started"
    assert data["domain"] == domain
    _pass(f"session_id={session_id} status=started")


async def test_bad_domain(client: httpx.AsyncClient) -> None:
    _header("Test 3 — POST /negotiation/start (invalid domain)")
    payload = {"domain": "nonexistent_domain", "session_id": f"bad-{uuid.uuid4().hex[:6]}"}
    r = await client.post(f"{BASE_URL}/negotiation/start", json=payload)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    _pass("correctly rejected unknown domain with 400")


async def test_duplicate_session(
    client: httpx.AsyncClient,
    session_id: str,
) -> None:
    _header("Test 4 — POST /negotiation/start (duplicate session_id)")
    payload = {"domain": "logistics", "session_id": session_id}
    r = await client.post(f"{BASE_URL}/negotiation/start", json=payload)
    assert r.status_code == 409, f"Expected 409, got {r.status_code}"
    _pass("correctly rejected duplicate session with 409")


async def test_status_endpoint(
    client: httpx.AsyncClient,
    session_id: str,
) -> None:
    _header("Test 5 — GET /negotiation/{id}/status")
    r = await client.get(f"{BASE_URL}/negotiation/{session_id}/status")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    for key in ("session_id", "domain", "current_round", "agent_statuses", "is_complete"):
        assert key in data, f"Missing key: {key}"
    for agent in data["agent_statuses"]:
        for akey in ("name", "role", "status", "current_offer"):
            assert akey in agent, f"Agent missing key: {akey}"
    _pass(
        f"round={data['current_round']} agents={[a['name'] for a in data['agent_statuses']]}"
    )


async def test_websocket_stream(session_id: str) -> dict[str, Any]:
    """Connect to the WS stream, collect all messages, return final result."""
    _header("Test 6 — WS /negotiation/{id}/live")
    uri = f"{WS_BASE}/negotiation/{session_id}/live"

    round_messages: list[dict] = []
    final_result: dict[str, Any] = {}

    async with websockets.connect(uri, ping_interval=None) as ws:
        async for raw in ws:
            msg: dict[str, Any] = json.loads(raw)
            if msg.get("type") == "round":
                round_messages.append(msg)
                print(f"    ↓ round {msg.get('round')} received")
            elif msg.get("type") == "complete":
                final_result = msg.get("result", {})
                print(f"    ↓ complete message received")
                break

    assert len(round_messages) > 0, "No round messages received"
    assert "deal" in final_result, "Final result missing 'deal'"
    assert "metrics" in final_result, "Final result missing 'metrics'"
    _pass(
        f"received {len(round_messages)} round(s) + complete message"
    )
    return final_result


async def test_result_endpoint(
    client: httpx.AsyncClient,
    session_id: str,
) -> None:
    _header("Test 7 — GET /negotiation/{id}/result")
    r = await client.get(f"{BASE_URL}/negotiation/{session_id}/result")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    for key in ("session_id", "domain", "status", "deal", "metrics", "rounds_log"):
        assert key in data, f"Missing key: {key}"
    assert data["status"] in ("converged", "forced", "in_progress")
    _pass(f"status={data['status']} rounds={len(data['rounds_log'])}")


async def test_status_404(client: httpx.AsyncClient) -> None:
    _header("Test 8 — GET /negotiation/{id}/status (unknown session)")
    r = await client.get(f"{BASE_URL}/negotiation/ghost-session-xyz/status")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    _pass("correctly returned 404 for unknown session")


async def test_result_404(client: httpx.AsyncClient) -> None:
    _header("Test 9 — GET /negotiation/{id}/result (unknown session)")
    r = await client.get(f"{BASE_URL}/negotiation/ghost-session-xyz/result")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    _pass("correctly returned 404 for unknown session")


# --------------------------------------------------------------------------- #
# Test runner
# --------------------------------------------------------------------------- #

async def run_all_tests() -> None:
    print("\n\033[1m\033[94m" + "=" * 60)
    print("  Multi-Agent Negotiation Simulator — Integration Tests")
    print("=" * 60 + "\033[0m")

    session_id = f"integration-{uuid.uuid4().hex[:8]}"
    passed = 0
    failed = 0

    async with httpx.AsyncClient(timeout=300.0) as client:
        tests = [
            ("List domains",        test_list_domains(client)),
            ("Start negotiation",   test_start_negotiation(client, session_id)),
            ("Bad domain → 400",    test_bad_domain(client)),
            ("Duplicate → 409",     test_duplicate_session(client, session_id)),
            ("Status endpoint",     test_status_endpoint(client, session_id)),
        ]

        for name, coro in tests:
            try:
                await coro
                passed += 1
            except (AssertionError, Exception) as exc:
                _fail(name, str(exc))
                failed += 1

        # WebSocket test (blocks until negotiation completes)
        try:
            await test_websocket_stream(session_id)
            passed += 1
        except Exception as exc:
            _fail("WebSocket stream", str(exc))
            failed += 1

        post_ws_tests = [
            ("Result endpoint",     test_result_endpoint(client, session_id)),
            ("Status 404",          test_status_404(client)),
            ("Result 404",          test_result_404(client)),
        ]
        for name, coro in post_ws_tests:
            try:
                await coro
                passed += 1
            except (AssertionError, Exception) as exc:
                _fail(name, str(exc))
                failed += 1

    print(f"\n{'─' * 60}")
    colour = "\033[92m" if failed == 0 else "\033[91m"
    print(
        f"\033[1m{colour}  Results: {passed} passed, {failed} failed\033[0m"
    )
    print(f"{'─' * 60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
