"""
core/agent.py

Generic Agent class for the domain-agnostic multi-agent negotiation system.
Each agent represents an independent participant (buyer, seller, mediator, etc.)
with its own objectives, constraints, and negotiation strategy.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

# Load environment variables from .env (no-op if already loaded or file absent)
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Role = str          # e.g. "buyer", "seller", "mediator", "supplier", "transporter"
Strategy = Literal["aggressive", "cooperative", "balanced"]
Status = Literal["pending", "accepted", "rejected", "countered"]


# ---------------------------------------------------------------------------
# Agent Model
# ---------------------------------------------------------------------------

class Agent(BaseModel):
    # Allow direct field mutation (status, current_offer updated after LLM call)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """
    A domain-agnostic negotiation agent.

    Fields are intentionally generic so the same class can represent any
    participant in any negotiation domain loaded from a JSON config.
    """

    name: str = Field(..., description="Unique identifier / display name for the agent.")
    role: str = Field(..., description="Role in the negotiation, e.g. 'buyer', 'seller', 'mediator'.")
    objective: str = Field(
        ...,
        description="High-level objective driving the agent's decisions, "
                    "e.g. 'minimize_cost', 'maximize_utilization', 'maximize_profit'.",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Hard limits the agent must respect, "
                    "e.g. {'max_price': 100, 'min_quantity': 5}.",
    )
    strategy: Strategy = Field(
        default="balanced",
        description=(
            "Negotiation style: "
            "'aggressive' (push hard for own terms), "
            "'cooperative' (seek mutual benefit), "
            "'balanced' (blend of both)."
        ),
    )
    current_offer: dict[str, Any] = Field(
        default_factory=dict,
        description="The most recent offer this agent has made or received.",
    )
    status: Status = Field(
        default="pending",
        description="Current negotiation status for this agent.",
    )

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def build_prompt(self, offer: dict[str, Any], domain: str, round_number: int) -> str:
        """
        Dynamically build an LLM instruction prompt for this agent.

        The prompt encodes:
        - Who the agent is (name, role, objective, strategy, constraints)
        - The current negotiation context (domain, round, incoming offer)
        - The exact JSON response schema the LLM must follow

        Args:
            offer:        The incoming offer dict from the counterparty.
            domain:       Domain label, e.g. "logistics", "cloud", "finance".
            round_number: Current round index (1-based).

        Returns:
            A fully-formed prompt string to be sent to the LLM.
        """
        strategy_descriptions: dict[str, str] = {
            "aggressive": (
                "You negotiate assertively and prioritise your own terms. "
                "Issue counteroffers that strongly favour your constraints. "
                "Accept only if the offer clearly meets all your requirements."
            ),
            "cooperative": (
                "You seek mutually beneficial outcomes. "
                "Be willing to make concessions on secondary constraints "
                "to reach an agreement that works for both sides."
            ),
            "balanced": (
                "You blend firmness with flexibility. "
                "Protect your core constraints but show moderate willingness "
                "to meet the counterparty halfway on secondary points."
            ),
        }

        strategy_guidance = strategy_descriptions.get(
            self.strategy, strategy_descriptions["balanced"]
        )

        constraints_str = (
            json.dumps(self.constraints, indent=2)
            if self.constraints
            else "None specified."
        )

        offer_str = (
            json.dumps(offer, indent=2)
            if offer
            else "No offer has been made yet — you may open the negotiation."
        )

        prompt = textwrap.dedent(f"""
            You are **{self.name}**, a negotiation agent in a **{domain}** scenario.

            ## Your Profile
            - **Role**: {self.role}
            - **Objective**: {self.objective}
            - **Strategy**: {self.strategy} — {strategy_guidance}
            - **Constraints** (hard limits you must respect):
            ```json
            {constraints_str}
            ```

            ## Negotiation Context
            - **Round**: {round_number}
            - **Incoming offer from counterparty**:
            ```json
            {offer_str}
            ```

            ## Your Task
            Carefully evaluate the incoming offer against your objective and constraints.
            Decide one of the following:
            - **Accept** — the offer satisfies your constraints and objective.
            - **Reject** — the offer is completely unacceptable with no room for a counter.
            - **Counteroffer** — propose revised terms that better serve your objective
              while leaving room for further negotiation.

            ## Response Format
            Respond with **only** a valid JSON object — no markdown fences, no extra text.
            The JSON must contain exactly these keys:

            {{
              "decision": "Accept" | "Reject" | "Counteroffer",
              "counteroffer": {{...}},
              "reasoning": "<one or two sentences explaining your decision>"
            }}

            Rules:
            - If `decision` is "Accept" or "Reject", set `counteroffer` to `null`.
            - If `decision` is "Counteroffer", `counteroffer` must be a non-empty dict
              with the same keys as the incoming offer, adjusted to your preferred terms.
            - Keep `reasoning` concise (max 2 sentences).
            - Never violate your constraints in a counteroffer.
            - Output **only** the JSON — nothing else.
        """).strip()

        return prompt

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------

    def evaluate(self, offer: dict[str, Any], domain: str, round_number: int) -> dict[str, Any]:
        """
        Ask the LLM to evaluate an incoming offer and decide: Accept / Reject / Counteroffer.

        Steps:
            1. Build the prompt via build_prompt().
            2. Send the prompt to OpenAI GPT-4o (chat completion).
            3. Parse the JSON response.
            4. Update self.current_offer and self.status.
            5. Return the parsed response dict.

        On any error (API failure, invalid JSON, missing keys) the method falls
        back to a safe "Reject" response so the negotiation loop can continue.

        Args:
            offer:        Incoming offer dict from the counterparty.
            domain:       Domain label, e.g. "logistics", "cloud", "finance".
            round_number: 1-based round index.

        Returns:
            dict with keys: decision (str), counteroffer (dict | None), reasoning (str)
        """
        FALLBACK: dict[str, Any] = {
            "decision": "Reject",
            "counteroffer": None,
            "reasoning": "Fallback: could not obtain a valid LLM response.",
        }

        # ---- 1. Build prompt ------------------------------------------------
        prompt = self.build_prompt(offer=offer, domain=domain, round_number=round_number)

        # ---- 2. Call OpenAI GPT-4o ------------------------------------------
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "[%s] OPENAI_API_KEY not set in environment / .env — returning fallback.",
                self.name,
            )
            return FALLBACK

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strategic negotiation agent. "
                            "Always respond with a single, valid JSON object only — "
                            "no markdown, no extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,          # modest creativity; repeatable but not robotic
                max_tokens=512,
                response_format={"type": "json_object"},   # GPT-4o JSON mode
            )
            raw_content: str = response.choices[0].message.content or ""
            logger.debug("[%s] Raw LLM response: %s", self.name, raw_content)

        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] OpenAI API call failed: %s", self.name, exc)
            return FALLBACK

        # ---- 3. Parse JSON --------------------------------------------------
        try:
            parsed: dict[str, Any] = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[%s] JSON parse error (%s) — raw content: %r — using fallback.",
                self.name, exc, raw_content,
            )
            return FALLBACK

        # ---- 4. Validate required keys & normalise --------------------------
        decision = parsed.get("decision", "")
        valid_decisions = {"Accept", "Reject", "Counteroffer"}

        if decision not in valid_decisions:
            logger.warning(
                "[%s] Unexpected decision value %r — using fallback.", self.name, decision
            )
            return FALLBACK

        counteroffer: dict[str, Any] | None = parsed.get("counteroffer") or None
        reasoning: str = parsed.get("reasoning", "No reasoning provided.")

        result: dict[str, Any] = {
            "decision": decision,
            "counteroffer": counteroffer,
            "reasoning": reasoning,
        }

        # ---- 5. Update agent state ------------------------------------------
        if decision == "Accept":
            self.status = "accepted"
            self.current_offer = offer          # the accepted offer
        elif decision == "Reject":
            self.status = "rejected"
        elif decision == "Counteroffer" and counteroffer:
            self.status = "countered"
            self.current_offer = counteroffer  # agent's own counter is the new offer

        logger.info(
            "[%s] Round %d — decision: %s | reasoning: %s",
            self.name, round_number, decision, reasoning,
        )

        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset mutable state between negotiation sessions."""
        self.current_offer = {}
        self.status = "pending"

    def __str__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r}, status={self.status!r})"
