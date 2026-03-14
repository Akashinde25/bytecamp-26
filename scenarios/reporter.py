"""
scenarios/reporter.py

Utility functions for pretty-printing negotiation session data in the terminal.
Designed for use in run_demo.py and test_integration.py.
"""

from __future__ import annotations

import json
from typing import Any


# --------------------------------------------------------------------------- #
# ANSI colour helpers
# --------------------------------------------------------------------------- #

class _Colours:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE  = "\033[97m"


C = _Colours()

_ROLE_COLOUR: dict[str, str] = {
    "buyer":    C.CYAN,
    "seller":   C.YELLOW,
    "mediator": C.MAGENTA,
}


def _role_badge(role: str) -> str:
    colour = _ROLE_COLOUR.get(role.lower(), C.WHITE)
    return f"{colour}[{role.upper()}]{C.RESET}"


def _separator(char: str = "─", width: int = 72) -> str:
    return char * width


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def print_session_header(session_id: str, domain: str) -> None:
    """Print a prominent header for a new session."""
    print()
    print(f"{C.BOLD}{C.BLUE}{_separator('═')}{C.RESET}")
    print(
        f"{C.BOLD}{C.WHITE}  🤝  Multi-Agent Negotiation Simulator{C.RESET}"
    )
    print(f"  Session : {C.CYAN}{session_id}{C.RESET}")
    print(f"  Domain  : {C.GREEN}{domain.upper()}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{_separator('═')}{C.RESET}")
    print()


def print_round_log(round_log: dict[str, Any]) -> None:
    """Pretty-print a single round log dict received from the WebSocket stream."""
    if round_log.get("type") != "round":
        return

    rnd = round_log.get("round", "?")
    converged = round_log.get("converged", False)

    status_icon = f"{C.GREEN}✔ CONVERGED{C.RESET}" if converged else f"{C.YELLOW}⟳ ONGOING{C.RESET}"
    print(f"{C.BOLD}  Round {rnd}  {status_icon}{C.RESET}")
    print(f"  {_separator()}")

    offers: dict[str, Any] = round_log.get("offers", {})
    for agent_name, offer in offers.items():
        units = offer.get("requested_units", "?")
        price = offer.get("price_per_unit", "?")
        concede = "↓" if offer.get("concession") else " "
        reasoning = offer.get("reasoning", "")
        print(
            f"    {C.BOLD}{agent_name:<16}{C.RESET} "
            f"units={C.CYAN}{units}{C.RESET}  "
            f"price={C.YELLOW}{price}{C.RESET}  "
            f"{concede}  {C.DIM}{reasoning[:60]}{C.RESET}"
        )

    proposed: dict[str, Any] = round_log.get("proposed_deal", {})
    if proposed:
        p_units = proposed.get("requested_units", "?")
        p_price = proposed.get("price_per_unit", "?")
        print(
            f"\n    {C.MAGENTA}Proposed deal{C.RESET} → "
            f"units={C.CYAN}{p_units}{C.RESET}  price={C.YELLOW}{p_price}{C.RESET}"
        )

    evals: dict[str, bool] = round_log.get("evaluations", {})
    if evals:
        eval_str = "  ".join(
            f"{name}: {C.GREEN}✔{C.RESET}" if accepted else f"{name}: {C.RED}✘{C.RESET}"
            for name, accepted in evals.items()
        )
        print(f"    Evaluations → {eval_str}")

    print()


def print_final_result(result: dict[str, Any]) -> None:
    """Print the final deal and metrics after a session completes."""
    deal = result.get("deal", {})
    metrics = result.get("metrics", {})
    status = deal.get("status", "unknown")

    status_colour = C.GREEN if status == "converged" else C.YELLOW
    status_label = "CONVERGED ✔" if status == "converged" else "FORCED COMPROMISE ⚡"

    print(f"\n{C.BOLD}{C.BLUE}{_separator('═')}{C.RESET}")
    print(f"  {status_colour}{C.BOLD}RESULT: {status_label}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{_separator('═')}{C.RESET}")

    print(f"\n{C.BOLD}  Final Deal:{C.RESET}")
    for k, v in deal.items():
        if k != "status":
            print(f"    {k:<24} {C.CYAN}{v}{C.RESET}")

    print(f"\n{C.BOLD}  Metrics:{C.RESET}")
    print(f"    Rounds to converge : {C.YELLOW}{metrics.get('rounds_to_converge', '?')}{C.RESET}")
    print(f"    Total rounds       : {metrics.get('total_rounds', '?')}")

    sat_scores: dict[str, float] = metrics.get("agent_satisfaction_scores", {})
    if sat_scores:
        print(f"\n{C.BOLD}  Agent Satisfaction Scores:{C.RESET}")
        for agent, score in sat_scores.items():
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            score_colour = C.GREEN if score >= 0.7 else (C.YELLOW if score >= 0.4 else C.RED)
            print(
                f"    {agent:<16} {score_colour}[{bar}] {score:.2%}{C.RESET}"
            )

    print(f"\n{C.BOLD}{C.BLUE}{_separator('═')}{C.RESET}\n")


def print_session_summary(status_payload: dict[str, Any]) -> None:
    """Print a compact status snapshot (used for polling status endpoint)."""
    print(f"\n{C.BOLD}  Session Status ── {status_payload.get('session_id')}{C.RESET}")
    print(f"  Domain  : {status_payload.get('domain')}")
    print(f"  Round   : {status_payload.get('current_round')}")
    print(f"  Complete: {status_payload.get('is_complete')}")
    for agent in status_payload.get("agent_statuses", []):
        role = agent.get("role", "?")
        print(
            f"    {_role_badge(role)} {agent['name']:<16} "
            f"status={C.DIM}{agent.get('status', '?')}{C.RESET}"
        )
    print()
