"""
core/environment.py

Responsible for loading domain configuration JSON files and instantiating
a fully-configured NegotiationSession ready to run.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from core.agent import Agent, AgentConfig
from core.negotiation import NegotiationSession

load_dotenv()

logger = logging.getLogger(__name__)

# Resolve /domains/ relative to the project root (one level up from /core/)
_PROJECT_ROOT: Path = Path(__file__).parent.parent
DOMAIN_DIR: Path = _PROJECT_ROOT / "domains"


# --------------------------------------------------------------------------- #
# Domain config schema
# --------------------------------------------------------------------------- #

class DomainConfig(BaseModel):
    """Pydantic schema for a top-level domain JSON file."""

    domain: str
    resource: str
    resource_pool: dict[str, Any] = Field(default_factory=dict)
    participants: list[AgentConfig]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def list_available_domains() -> list[str]:
    """
    Return the names of all available domain configs (without .json extension).

    Example: ["logistics", "cloud", "finance"]
    """
    if not DOMAIN_DIR.exists():
        logger.warning("Domains directory not found at %s", DOMAIN_DIR)
        return []

    return [
        p.stem
        for p in DOMAIN_DIR.iterdir()
        if p.is_file() and p.suffix == ".json"
    ]


def initialize_session(domain: str, session_id: str) -> NegotiationSession:
    """
    Load a domain config file, validate it, and return a ready-to-run
    NegotiationSession with fully instantiated Agent objects.

    Parameters
    ----------
    domain     : Domain name matching a file in /domains/ (e.g. "logistics").
    session_id : Unique identifier for the new session.

    Returns
    -------
    NegotiationSession (not yet started — caller must call `.run()`).

    Raises
    ------
    FileNotFoundError   : No JSON file found for the requested domain.
    ValidationError     : Domain JSON does not conform to DomainConfig schema.
    """
    domain_path: Path = DOMAIN_DIR / f"{domain}.json"

    if not domain_path.exists():
        raise FileNotFoundError(
            f"Domain config not found: {domain_path}. "
            f"Available domains: {list_available_domains()}"
        )

    logger.info("Loading domain config from %s", domain_path)

    with domain_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)

    try:
        config = DomainConfig(**raw)
    except ValidationError as exc:
        logger.error("Domain config validation failed: %s", exc)
        raise

    api_key: str | None = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    client = AsyncOpenAI(api_key=api_key)

    agents: list[Agent] = [
        Agent(config=participant, resource=config.resource, client=client)
        for participant in config.participants
    ]

    session = NegotiationSession(
        session_id=session_id,
        domain=config.domain,
        agents=agents,
        resource_pool=dict(config.resource_pool),
    )

    logger.info(
        "Initialized session '%s' for domain '%s' with %d agents",
        session_id,
        domain,
        len(agents),
    )

    return session
