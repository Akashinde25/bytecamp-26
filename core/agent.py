"""
core/agent.py

Defines the Agent abstraction for the negotiation engine.
Each agent uses GPT-4o to reason about its role, constraints,
and negotiation history to produce offers and evaluate deals.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Pydantic config schema (mirrors the JSON participant block)
# --------------------------------------------------------------------------- #

class AgentConfig(BaseModel):
    """Validated participant block loaded from a domain JSON file."""

    name: str
    role: str  # "buyer" | "seller" | "mediator"
    objective: str
    constraints: dict[str, Any] = Field(default_factory=dict)
    strategy: str


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class Agent:
    """
    An autonomous negotiation participant powered by GPT-4o.

    Attributes
    ----------
    name        : Display name of the agent.
    role        : "buyer", "seller", or "mediator".
    objective   : Natural-language goal for this agent.
    constraints : Domain-specific numeric / string constraints.
    strategy    : High-level negotiation style hint (e.g. "aggressive").
    current_offer : Most recent offer produced by this agent.
    status      : "idle" | "negotiating" | "satisfied" | "rejected".
    """

    def __init__(
        self,
        config: AgentConfig,
        resource: str,
        client: AsyncOpenAI,
    ) -> None:
        self.name: str = config.name
        self.role: str = config.role
        self.objective: str = config.objective
        self.constraints: dict[str, Any] = config.constraints
        self.strategy: str = config.strategy
        self.resource: str = resource

        self.current_offer: dict[str, Any] = {}
        self.status: str = "idle"

        self._client: AsyncOpenAI = client

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    async def make_offer(
        self,
        resource_pool: dict[str, Any],
        round_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Ask GPT-4o to generate a negotiation offer given the current context.

        Parameters
        ----------
        resource_pool  : Available resources in the session.
        round_history  : Flattened list of all previous round logs.

        Returns
        -------
        A JSON-serialisable offer dict that is also stored in `current_offer`.
        """
        self.status = "negotiating"

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(resource_pool, round_history)

        logger.debug("[%s] requesting offer from GPT-4o …", self.name)

        response = await self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        raw = response.choices[0].message.content or "{}"
        offer: dict[str, Any] = json.loads(raw)
        self.current_offer = offer

        logger.info("[%s] offer: %s", self.name, offer)
        return offer

    async def evaluate_deal(
        self,
        proposed_deal: dict[str, Any],
    ) -> bool:
        """
        Ask GPT-4o whether a proposed deal satisfies this agent's constraints.

        Parameters
        ----------
        proposed_deal : The aggregated deal being proposed.

        Returns
        -------
        True if the agent accepts, False otherwise.
        """
        system_prompt = (
            f"You are {self.name}, a {self.role} in a negotiation.\n"
            f"Your objective: {self.objective}\n"
            f"Your constraints: {json.dumps(self.constraints)}\n\n"
            "Evaluate whether the proposed deal satisfies your constraints. "
            "Reply with a JSON object with a single key 'accept' set to true or false."
        )
        user_prompt = (
            f"Proposed deal: {json.dumps(proposed_deal)}\n"
            "Does this deal satisfy your constraints?"
        )

        response = await self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        raw = response.choices[0].message.content or '{"accept": false}'
        result: dict[str, Any] = json.loads(raw)
        accepted: bool = bool(result.get("accept", False))

        self.status = "satisfied" if accepted else "negotiating"
        logger.info("[%s] deal evaluation → accept=%s", self.name, accepted)
        return accepted

    def satisfaction_score(self, deal: dict[str, Any]) -> float:
        """
        Heuristic satisfaction score in [0.0, 1.0] based on constraint proximity.

        For numeric constraints the score is a ratio; non-numeric constraints
        default to 0.5 (neutral).  The mediator always returns 1.0 (neutral).
        """
        if self.role == "mediator":
            return 1.0

        scores: list[float] = []

        for key, limit in self.constraints.items():
            if not isinstance(limit, (int, float)):
                scores.append(0.5)
                continue

            actual = deal.get(key, deal.get("allocation", {}).get(self.name))
            if actual is None:
                scores.append(0.5)
                continue

            try:
                actual_f = float(actual)
                # Normalise: distance from limit expressed as a satisfaction %
                if limit == 0:
                    scores.append(1.0 if actual_f == 0 else 0.0)
                else:
                    ratio = actual_f / limit
                    # Clamp to [0, 1]
                    scores.append(max(0.0, min(1.0, ratio)))
            except (TypeError, ValueError):
                scores.append(0.5)

        return round(sum(scores) / len(scores), 4) if scores else 0.5

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _build_system_prompt(self) -> str:
        return (
            f"You are {self.name}, an autonomous AI negotiation agent.\n"
            f"Role       : {self.role}\n"
            f"Resource   : {self.resource}\n"
            f"Objective  : {self.objective}\n"
            f"Constraints: {json.dumps(self.constraints)}\n"
            f"Strategy   : {self.strategy}\n\n"
            "Produce a negotiation offer as a valid JSON object. "
            "The offer MUST contain:\n"
            "  - 'requested_units': numeric amount of the resource you want/offer\n"
            "  - 'price_per_unit' : numeric price you propose\n"
            "  - 'reasoning'      : brief string explaining your position\n"
            "  - 'concession'     : boolean, true if you are moving toward compromise\n"
            "Stay within your constraints. Be strategic but realistic."
        )

    def _build_user_prompt(
        self,
        resource_pool: dict[str, Any],
        round_history: list[dict[str, Any]],
    ) -> str:
        history_str = (
            json.dumps(round_history[-5:], indent=2)  # last 5 rounds for brevity
            if round_history
            else "No history yet. This is the first round."
        )
        return (
            f"Current resource pool: {json.dumps(resource_pool)}\n\n"
            f"Negotiation history (last 5 rounds):\n{history_str}\n\n"
            "Make your offer now."
        )

    # ---------------------------------------------------------------------- #
    # Representation
    # ---------------------------------------------------------------------- #

    def to_status_dict(self) -> dict[str, Any]:
        """Return a serialisable status snapshot."""
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "current_offer": self.current_offer,
        }
