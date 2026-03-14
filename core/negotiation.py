"""
core/negotiation.py

NegotiationSession orchestrates a multi-round negotiation between a set of
domain-agnostic Agent instances over a shared resource pool.

Flow:
  Round 1  — each non-mediator agent proposes an initial offer from constraints.
  Round N  — each non-mediator agent evaluates the best available offer and responds.
  Converge — all non-mediators Accept → return agreed deal.
  Forced   — max_rounds hit → mediator (or automatic average) forces a compromise.
"""

from __future__ import annotations

import logging
import statistics
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field

from core.agent import Agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Round log entry (typed for clarity)
# ---------------------------------------------------------------------------

class RoundEntry(BaseModel):
    """A single logged event within a negotiation round."""
    round: int
    agent_name: str
    offer: dict[str, Any]
    decision: str
    reasoning: str


# ---------------------------------------------------------------------------
# NegotiationSession
# ---------------------------------------------------------------------------

class NegotiationSession(BaseModel):
    """
    Manages the full lifecycle of a multi-agent negotiation.

    Attributes:
        agents:        All participants, including any mediator agent(s).
        resource_pool: Shared environment state (read-only context for agents).
        domain:        Domain label forwarded to each agent's prompt.
        max_rounds:    Hard cap on negotiation rounds before mediator intervenes.
    """

    agents: list[Agent]
    resource_pool: dict[str, Any] = Field(default_factory=dict)
    domain: str
    max_rounds: int = 10

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _negotiators(self) -> list[Agent]:
        """Return all non-mediator agents."""
        return [a for a in self.agents if a.role.lower() != "mediator"]

    def _mediators(self) -> list[Agent]:
        """Return mediator agents (may be empty)."""
        return [a for a in self.agents if a.role.lower() == "mediator"]

    def _initial_offer(self, agent: Agent) -> dict[str, Any]:
        """
        Derive an agent's opening offer directly from their constraints.
        For numeric constraints we use the value as a starting bid;
        non-numeric values are included as-is.
        """
        offer: dict[str, Any] = {}
        for key, val in agent.constraints.items():
            # Strip directional prefixes so offer keys are clean domain terms
            # e.g.  "max_price" → "price",  "min_quantity" → "quantity"
            clean_key = key
            for prefix in ("max_", "min_", "target_", "desired_"):
                if key.startswith(prefix):
                    clean_key = key[len(prefix):]
                    break
            offer[clean_key] = val
        return offer if offer else deepcopy(agent.constraints)

    @staticmethod
    def _best_offer(offers: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """
        Select the 'best' current offer to present to an evaluating agent.
        Simple heuristic: return the offer with the numerically smallest
        total value sum (favours the buyer perspective as a neutral seed).
        If values are non-numeric, fall back to the first offer.
        """
        if not offers:
            return {}
        try:
            return min(
                offers.values(),
                key=lambda o: sum(v for v in o.values() if isinstance(v, (int, float))),
            )
        except (TypeError, ValueError):
            return next(iter(offers.values()))

    # ------------------------------------------------------------------
    # Satisfaction score
    # ------------------------------------------------------------------

    @staticmethod
    def _satisfaction_score(agent: Agent, final_deal: dict[str, Any]) -> float:
        """
        Compute a 0–1 satisfaction score for an agent given the final deal.

        For each numeric constraint:
          - "max_X" (upper bound): score = clamp(1 - (deal_val - limit) / limit, 0, 1)
          - "min_X" (lower bound): score = clamp(deal_val / limit, 0, 1)
          - Exact match: 1.0 if equal, else scaled proximity.

        Non-numeric constraints always contribute 1.0 (can't be measured).
        Returns the mean over all constraints, or 0.5 if no constraints exist.
        """
        if not agent.constraints:
            return 0.5

        scores: list[float] = []
        for constraint_key, limit in agent.constraints.items():
            if not isinstance(limit, (int, float)):
                scores.append(1.0)
                continue

            # Resolve the deal key (strip directional prefix)
            deal_key = constraint_key
            for prefix in ("max_", "min_", "target_", "desired_"):
                if constraint_key.startswith(prefix):
                    deal_key = constraint_key[len(prefix):]
                    break

            deal_val = final_deal.get(deal_key)
            if deal_val is None or not isinstance(deal_val, (int, float)):
                scores.append(0.5)
                continue

            if constraint_key.startswith("max_"):
                # Lower is better for the agent
                if deal_val <= limit:
                    score = 1.0
                else:
                    score = max(0.0, 1.0 - (deal_val - limit) / (limit or 1))
            elif constraint_key.startswith("min_"):
                # Higher is better for the agent
                if deal_val >= limit:
                    score = 1.0
                else:
                    score = max(0.0, deal_val / (limit or 1))
            else:
                # Proximity: exact match = 1.0
                score = max(0.0, 1.0 - abs(deal_val - limit) / (limit or 1))

            scores.append(min(score, 1.0))

        return round(statistics.mean(scores), 4) if scores else 0.5

    # ------------------------------------------------------------------
    # Mediator forced compromise
    # ------------------------------------------------------------------

    def _mediator_compromise(
        self, latest_offers: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Average all numeric fields across all latest offers.
        Non-numeric fields take the value from the first offer (majority fallback).
        If a mediator agent exists, it also evaluates this aggregate via LLM.
        """
        if not latest_offers:
            return {}

        all_offers = list(latest_offers.values())
        keys = set()
        for o in all_offers:
            keys.update(o.keys())

        compromise: dict[str, Any] = {}
        for key in keys:
            numeric_vals = [
                o[key] for o in all_offers
                if key in o and isinstance(o[key], (int, float))
            ]
            if numeric_vals:
                compromise[key] = round(statistics.mean(numeric_vals), 4)
            else:
                # Take non-numeric value from the first offer that has it
                for o in all_offers:
                    if key in o:
                        compromise[key] = o[key]
                        break

        # Optionally let the mediator agent refine via LLM
        mediators = self._mediators()
        if mediators:
            mediator = mediators[0]
            logger.info("Mediator %s is refining the forced compromise.", mediator.name)
            try:
                result = mediator.evaluate(
                    offer=compromise,
                    domain=self.domain,
                    round_number=self.max_rounds,
                )
                # If mediator counters, use that; otherwise keep average
                if result["decision"] == "Counteroffer" and result.get("counteroffer"):
                    compromise = result["counteroffer"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Mediator LLM call failed (%s); using arithmetic average.", exc)

        return compromise

    # ------------------------------------------------------------------
    # Main negotiation loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """
        Execute the negotiation and return a structured result.

        Returns:
            {
                "status":      "converged" | "forced",
                "deal":        dict,          # final agreed terms
                "rounds_log":  list[dict],    # full per-event history
                "metrics":     {
                    "rounds_to_converge": int,
                    "agent_satisfaction": {agent_name: float (0-1)},
                    "total_events":       int,
                }
            }
        """
        negotiators = self._negotiators()
        if not negotiators:
            raise ValueError("NegotiationSession requires at least one non-mediator agent.")

        rounds_log: list[dict[str, Any]] = []
        # Map: agent_name → their latest offer
        latest_offers: dict[str, dict[str, Any]] = {}

        # ── Round 1: initial proposals ─────────────────────────────────
        logger.info("=== Negotiation START | domain=%s | agents=%s ===",
                    self.domain, [a.name for a in negotiators])

        for agent in negotiators:
            offer = self._initial_offer(agent)
            agent.current_offer = offer
            latest_offers[agent.name] = offer

            entry: dict[str, Any] = {
                "round": 1,
                "agent_name": agent.name,
                "offer": deepcopy(offer),
                "decision": "Propose",
                "reasoning": "Initial offer derived from agent constraints.",
            }
            rounds_log.append(entry)
            logger.info("Round 1 | %s proposes: %s", agent.name, offer)

        # ── Rounds 2 … max_rounds ──────────────────────────────────────
        converged = False
        final_round = 1

        for round_num in range(2, self.max_rounds + 1):
            final_round = round_num
            accept_count = 0

            for agent in negotiators:
                # Build the best available offer from *other* agents
                others_offers = {
                    name: off
                    for name, off in latest_offers.items()
                    if name != agent.name
                }
                best = self._best_offer(others_offers) if others_offers else agent.current_offer

                # Ask the LLM to decide
                result = agent.evaluate(
                    offer=best,
                    domain=self.domain,
                    round_number=round_num,
                )

                decision: str = result["decision"]
                counteroffer: dict[str, Any] | None = result.get("counteroffer")
                reasoning: str = result.get("reasoning", "")

                # Update latest_offers with the agent's new counter (if any)
                if decision == "Counteroffer" and counteroffer:
                    latest_offers[agent.name] = counteroffer
                elif decision == "Accept":
                    accept_count += 1
                    # Agent accepts the best offer → adopt it as their own latest
                    latest_offers[agent.name] = best

                entry = {
                    "round": round_num,
                    "agent_name": agent.name,
                    "offer": deepcopy(latest_offers[agent.name]),
                    "decision": decision,
                    "reasoning": reasoning,
                }
                rounds_log.append(entry)
                logger.info(
                    "Round %d | %s → %s | reasoning: %s",
                    round_num, agent.name, decision, reasoning,
                )

            # Check convergence: all non-mediators accepted
            if accept_count == len(negotiators):
                converged = True
                logger.info("=== CONVERGED at round %d ===", round_num)
                break

        # ── Determine final deal ───────────────────────────────────────
        if converged:
            # Agreed deal = the offer that everyone accepted (latest from any agent)
            deal = deepcopy(next(iter(latest_offers.values())))
            status = "converged"
        else:
            logger.info("=== MAX ROUNDS REACHED — forcing mediator compromise ===")
            deal = self._mediator_compromise(latest_offers)
            status = "forced"

            # Log the forced deal
            rounds_log.append({
                "round": self.max_rounds,
                "agent_name": "MEDIATOR",
                "offer": deepcopy(deal),
                "decision": "ForcedCompromise",
                "reasoning": "Max rounds reached; arithmetic average of all final offers applied.",
            })

        # ── Compute metrics ────────────────────────────────────────────
        agent_satisfaction = {
            a.name: self._satisfaction_score(a, deal)
            for a in negotiators
        }
        metrics: dict[str, Any] = {
            "rounds_to_converge": final_round,
            "agent_satisfaction": agent_satisfaction,
            "total_events": len(rounds_log),
        }

        logger.info("=== Negotiation END | status=%s | deal=%s | metrics=%s ===",
                    status, deal, metrics)

        return {
            "status": status,
            "deal": deal,
            "rounds_log": rounds_log,
            "metrics": metrics,
        }
