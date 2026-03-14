"""
llm-council-master/backend/council.py

THE INTEGRATION HUB — 3-stage LLM Council orchestration.

Stage 1: 4 LLMs brainstorm negotiation strategies (OpenRouter fan-out).
Stage 2: BytecampAgentEngine scores each strategy deterministically via
         Python satisfaction_score() — no additional LLM calls.
Stage 3: Chairman LLM synthesizes the math-backed strategies into a
         final human-readable recommendation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

# ── Path bootstrap: allow importing from project root /core ───────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load root-level .env (contains OPENAI_API_KEY etc.)
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
# Also load council-level .env (contains OPENROUTER_API_KEY)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from backend.config import CHAIRMAN_MODEL, COUNCIL_MODELS  # noqa: E402
from backend.openrouter import query_model, query_models_parallel  # noqa: E402

# Import core engine — available because _PROJECT_ROOT is in sys.path
from core.environment import initialize_session  # noqa: E402

logger = logging.getLogger(__name__)

# Response labels A–D for Stage 1 output anonymisation
_LABELS = list("ABCD")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def extract_json_deal(text: str) -> dict[str, Any]:
    """
    Rip the first JSON object block out of a raw LLM text response.

    Searches for the outermost `{...}` and attempts to parse it.
    Returns an empty dict if nothing parseable is found.
    """
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, Exception):  # noqa: BLE001
        pass
    return {}


def _build_domain_context(session: Any) -> str:
    """Serialize the session's resource pool and agent roster for prompt injection."""
    if session is None:
        return ""
    lines = [
        "\n\nDOMAIN RULES & CONSTRAINTS:",
        f"Resource Pool: {json.dumps(session.resource_pool)}",
        "Roles:",
    ]
    for agent in session.agents:
        lines.append(f"  - {agent.name} ({agent.role}): {agent.objective}")
        if agent.constraints:
            lines.append(f"    Constraints: {json.dumps(agent.constraints)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1 — Brainstorm (LLMs propose strategies + embedded JSON deal)
# ---------------------------------------------------------------------------

async def stage1_collect_responses(
    user_query: str,
    session: Any = None,
) -> list[dict[str, Any]]:
    """
    Fan-out the negotiation question to all council models in parallel.
    Each model is asked to propose a strategy AND embed a JSON deal block.

    Returns:
        List of dicts: [{"model": str, "label": str, "response": str}, ...]
    """
    domain_context = _build_domain_context(session)
    enriched_query = f"{user_query}{domain_context}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert negotiation strategist. "
                "Analyse the scenario, propose an optimal negotiation strategy, "
                "and embed your proposed deal terms as a JSON object in your response "
                '(e.g. {"budget_limit": 5.0, "requested_units": 40, "price_per_unit": 0.10}). '
                "The JSON block must be the only JSON in your response."
            ),
        },
        {"role": "user", "content": enriched_query},
    ]

    logger.info("Stage 1: querying %d council models in parallel.", len(COUNCIL_MODELS))
    raw_results = await query_models_parallel(COUNCIL_MODELS, messages)

    # Attach A/B/C/D labels (anonymised for Stage 2 fairness)
    results: list[dict[str, Any]] = []
    for i, item in enumerate(raw_results):
        label = _LABELS[i] if i < len(_LABELS) else str(i + 1)
        results.append(
            {
                "model": item["model"],
                "label": f"Response {label}",
                "response": item["response"],
            }
        )
        logger.debug("Stage 1 [%s] %s: %.120s…", label, item["model"], item["response"])

    return results


# ---------------------------------------------------------------------------
# Stage 2 — Deterministic scoring (BytecampAgentEngine replaces LLM voting)
# ---------------------------------------------------------------------------

async def stage2_collect_rankings(
    user_query: str,
    stage1_results: list[dict[str, Any]],
    session: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Score each Stage 1 strategy using the Python Agent satisfaction_score().
    No LLM calls are made in this stage.

    Returns:
        (stage2_results, label_to_model)  where:
          stage2_results  — sorted list of evaluation dicts (best score first)
          label_to_model  — mapping from "Response X" label to actual model name
    """
    label_to_model: dict[str, str] = {r["label"]: r["model"] for r in stage1_results}

    # Retrieve buyer and seller from the session (fall back gracefully)
    buyer = None
    seller = None
    if session:
        buyer = next((a for a in session.agents if a.role == "buyer"), None)
        seller = next((a for a in session.agents if a.role == "seller"), None)
        if buyer is None and session.agents:
            buyer = session.agents[0]
        if seller is None and len(session.agents) > 1:
            seller = session.agents[1]

    logger.info("Stage 2: scoring %d strategies via BytecampAgentEngine.", len(stage1_results))

    stage2_results: list[dict[str, Any]] = []

    for item in stage1_results:
        label = item["label"]
        response_text = item["response"]

        # 1. Rip JSON deal block from LLM text
        proposed_deal = extract_json_deal(response_text)

        # 2. Score deterministically
        b_score = buyer.satisfaction_score(proposed_deal) if buyer else 0.5
        s_score = seller.satisfaction_score(proposed_deal) if seller else 0.5
        avg_score = round((b_score + s_score) / 2, 4)

        buyer_name = buyer.name if buyer else "Buyer"
        seller_name = seller.name if seller else "Seller"

        eval_text = (
            f"Agent Engine Evaluation for {label}:\n"
            f"Detected Deal: {json.dumps(proposed_deal) if proposed_deal else 'No JSON deal found'}\n"
            f"{buyer_name} (Buyer) Satisfaction: {b_score}\n"
            f"{seller_name} (Seller) Satisfaction: {s_score}\n"
            f"Average Objective Score: {avg_score}"
        )

        logger.info("Stage 2 [%s]: buyer=%.4f seller=%.4f avg=%.4f", label, b_score, s_score, avg_score)

        stage2_results.append(
            {
                "model": "BytecampAgentEngine",
                "label": label,
                "ranking": eval_text,
                "score": avg_score,
                "buyer_score": b_score,
                "seller_score": s_score,
                "proposed_deal": proposed_deal,
            }
        )

    # Sort best → worst
    stage2_results.sort(key=lambda x: x["score"], reverse=True)

    # Append final ranking summary
    ranking_str = "FINAL RANKING:\n" + "\n".join(
        [f"{i + 1}. {r['label']} (score={r['score']})" for i, r in enumerate(stage2_results)]
    )
    for r in stage2_results:
        r["ranking"] += f"\n\n{ranking_str}"

    return stage2_results, label_to_model


# ---------------------------------------------------------------------------
# Stage 3 — Chairman synthesis
# ---------------------------------------------------------------------------

async def stage3_synthesize_final(
    user_query: str,
    stage1_results: list[dict[str, Any]],
    stage2_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    The Chairman LLM reads Stage 1 strategies + Stage 2 math scores and
    produces the final, authoritative negotiation recommendation.

    Returns:
        {"model": str, "response": str}
    """
    # Build Stage 1 summary
    stage1_summary = "\n\n".join(
        [f"--- {r['label']} ({r['model']}) ---\n{r['response']}" for r in stage1_results]
    )

    # Build Stage 2 summary (scores only, no model names for objectivity)
    stage2_summary = "\n\n".join([r["ranking"] for r in stage2_results])

    chairman_prompt = f"""You are the Head Negotiator and final decision-maker.

NEGOTIATION SCENARIO:
{user_query}

STAGE 1 — Proposed Strategies from Council Members:
{stage1_summary}

STAGE 2 — Objective Engine Evaluation (Bytecamp Agent Math Scores):
{stage2_summary}

Your task:
1. Identify the strategy that best balances buyer and seller satisfaction (highest average score).
2. Synthesize a final, actionable negotiation recommendation that incorporates the top-scoring strategy.
3. Briefly explain WHY this strategy scored highest mathematically.
4. Provide the recommended deal terms as a final JSON object.

Be concise, authoritative, and data-driven. Emphasize the strategies that scored highest in the engine computations."""

    messages = [{"role": "user", "content": chairman_prompt}]

    logger.info("Stage 3: Chairman (%s) synthesizing final recommendation.", CHAIRMAN_MODEL)
    response = await query_model(CHAIRMAN_MODEL, messages, is_chairman=True)

    return {"model": CHAIRMAN_MODEL, "response": response}


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

async def run_full_council(
    user_query: str,
    domain: str = "cloud",
) -> tuple[list[dict], list[dict], dict[str, Any], dict[str, Any]]:
    """
    Execute the full 3-stage LLM Council pipeline.

    Args:
        user_query: The negotiation question from the user.
        domain:     Domain to load from /domains/{domain}.json (default "cloud").

    Returns:
        (stage1_results, stage2_results, stage3_result, metadata)
    """
    # ── Load Python negotiation engine ────────────────────────────────────────
    session = None
    try:
        session = initialize_session(domain=domain, session_id="council_run")
        logger.info("Council session loaded | domain=%s | agents=%d", domain, len(session.agents))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load domain '%s': %s — proceeding without context.", domain, exc)

    # ── Stage 1: Brainstorm ───────────────────────────────────────────────────
    stage1_results = await stage1_collect_responses(user_query, session)

    # ── Stage 2: Engine evaluation (pure Python, no LLM) ─────────────────────
    stage2_results, label_to_model = await stage2_collect_rankings(
        user_query, stage1_results, session
    )

    # ── Stage 3: Chairman synthesis ───────────────────────────────────────────
    stage3_result = await stage3_synthesize_final(user_query, stage1_results, stage2_results)

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata: dict[str, Any] = {
        "domain": domain,
        "council_models": COUNCIL_MODELS,
        "chairman_model": CHAIRMAN_MODEL,
        "label_to_model": label_to_model,
        "top_strategy": stage2_results[0]["label"] if stage2_results else None,
        "top_score": stage2_results[0]["score"] if stage2_results else None,
    }

    return stage1_results, stage2_results, stage3_result, metadata
