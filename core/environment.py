"""
core/environment.py

Domain config loader and session initializer for the multi-agent negotiation system.

Public API:
    load_domain(domain_name)       -> dict with agents, resource_pool, domain
    initialize_session(domain_name) -> ready NegotiationSession
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.agent import Agent
from core.negotiation import NegotiationSession

logger = logging.getLogger(__name__)

# Resolve the /domains directory relative to this file's project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOMAINS_DIR = _PROJECT_ROOT / "domains"

# Required top-level keys every domain JSON must contain
_REQUIRED_KEYS = {"domain", "resource", "participants", "resource_pool"}

# Required keys inside each participant definition
_REQUIRED_PARTICIPANT_KEYS = {"name", "role", "objective", "strategy"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_domain_config(config: dict[str, Any], source: str) -> None:
    """
    Raise ValueError if the domain config is missing required keys or
    if any participant definition is incomplete.
    """
    missing_top = _REQUIRED_KEYS - set(config.keys())
    if missing_top:
        raise ValueError(
            f"Domain config '{source}' is missing required top-level keys: {missing_top}"
        )

    participants = config["participants"]
    if not isinstance(participants, list) or len(participants) == 0:
        raise ValueError(
            f"Domain config '{source}': 'participants' must be a non-empty list."
        )

    for idx, participant in enumerate(participants):
        missing_p = _REQUIRED_PARTICIPANT_KEYS - set(participant.keys())
        if missing_p:
            raise ValueError(
                f"Domain config '{source}': participant[{idx}] "
                f"(name={participant.get('name', '?')!r}) "
                f"is missing keys: {missing_p}"
            )


def _build_agent(participant: dict[str, Any]) -> Agent:
    """
    Construct an Agent from a participant config dict.
    Unknown extra keys are ignored safely.
    """
    return Agent(
        name=participant["name"],
        role=participant["role"],
        objective=participant["objective"],
        constraints=participant.get("constraints", {}),
        strategy=participant.get("strategy", "balanced"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_domain(domain_name: str) -> dict[str, Any]:
    """
    Load and validate a domain configuration from /domains/{domain_name}.json.

    Args:
        domain_name: Filename stem, e.g. "logistics", "cloud", "finance".

    Returns:
        {
            "agents":        list[Agent],
            "resource_pool": dict,
            "domain":        str,
            "resource":      str,   # human-readable resource label
        }

    Raises:
        FileNotFoundError: if the domain JSON file does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
        ValueError: if required keys are missing or participants are malformed.
    """
    json_path = _DOMAINS_DIR / f"{domain_name}.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Domain config not found: '{json_path}'. "
            f"Available domains: {_list_available_domains()}"
        )

    logger.info("Loading domain config from: %s", json_path)

    with json_path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = json.load(fh)

    _validate_domain_config(config, source=str(json_path))

    agents = [_build_agent(p) for p in config["participants"]]

    logger.info(
        "Loaded domain '%s' | resource: '%s' | agents: %s",
        config["domain"],
        config["resource"],
        [a.name for a in agents],
    )

    return {
        "agents": agents,
        "resource_pool": config["resource_pool"],
        "domain": config["domain"],
        "resource": config["resource"],
    }


def initialize_session(
    domain_name: str,
    max_rounds: int = 10,
) -> NegotiationSession:
    """
    Load a domain config and return a ready-to-run NegotiationSession.

    Args:
        domain_name: Filename stem of the domain JSON, e.g. "logistics".
        max_rounds:  Override the default round cap (default 10).

    Returns:
        A fully configured NegotiationSession instance.
    """
    loaded = load_domain(domain_name)

    session = NegotiationSession(
        agents=loaded["agents"],
        resource_pool=loaded["resource_pool"],
        domain=loaded["domain"],
        max_rounds=max_rounds,
    )

    logger.info(
        "Session initialized | domain=%s | agents=%d | max_rounds=%d",
        loaded["domain"],
        len(loaded["agents"]),
        max_rounds,
    )

    return session


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _list_available_domains() -> list[str]:
    """Return names of all .json files present in the domains directory."""
    if not _DOMAINS_DIR.exists():
        return []
    return [p.stem for p in sorted(_DOMAINS_DIR.glob("*.json"))]


def list_domains() -> list[str]:
    """Public helper — returns available domain names."""
    return _list_available_domains()
