"""
core/negotiation.py

Defines NegotiationSession — the central stateful object that orchestrates
multiple negotiation rounds, detects convergence, forces compromise via the
mediator, and computes final metrics.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.agent import Agent

logger = logging.getLogger(__name__)

MAX_ROUNDS: int = 10


# --------------------------------------------------------------------------- #
# NegotiationSession
# --------------------------------------------------------------------------- #

@dataclass
class NegotiationSession:
    """
    Stateful container for a single negotiation session.

    Attributes
    ----------
    session_id    : Unique identifier for this session.
    domain        : Name of the domain (e.g. "logistics").
    agents        : All participant agents (buyers, sellers, mediators).
    resource_pool : Shared resource pool dict from the domain config.
    current_round : Index of the most recently completed round (0 = not started).
    round_logs    : Ordered list of round result dicts.
    deal          : Final agreed or forced deal, empty until complete.
    metrics       : Computed metrics after session ends.
    is_complete   : True once the session has finished.
    _queue        : asyncio.Queue for streaming updates to WebSocket clients.
    """

    session_id: str
    domain: str
    agents: list[Agent]
    resource_pool: dict[str, Any]
    current_round: int = 0
    round_logs: list[dict[str, Any]] = field(default_factory=list)
    deal: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue)  # type: ignore[type-arg]

    # ---------------------------------------------------------------------- #
    # Main negotiation loop
    # ---------------------------------------------------------------------- #

    async def run(self) -> None:
        """
        Execute the full negotiation lifecycle.

        Each round:
        1. Buyers and sellers produce offers via GPT-4o.
        2. The mediator proposes a synthesised deal.
        3. All non-mediator agents evaluate the deal.
        4. If all accept → converged; stop.
        5. If MAX_ROUNDS reached → force compromise.
        """
        logger.info("[%s] session started (domain=%s)", self.session_id, self.domain)

        buyers_sellers = [a for a in self.agents if a.role != "mediator"]
        mediators = [a for a in self.agents if a.role == "mediator"]

        for round_num in range(1, MAX_ROUNDS + 1):
            self.current_round = round_num
            logger.info("[%s] round %d …", self.session_id, round_num)

            # ---- Step 1: gather offers ------------------------------------ #
            offer_tasks = [
                a.make_offer(self.resource_pool, self.round_logs)
                for a in buyers_sellers
            ]
            offers: list[dict[str, Any]] = await asyncio.gather(*offer_tasks)

            agent_offers: dict[str, dict[str, Any]] = {
                a.name: offer
                for a, offer in zip(buyers_sellers, offers)
            }

            # ---- Step 2: mediator synthesises ------------------------------ #
            proposed_deal: dict[str, Any] = {}
            if mediators:
                mediator = mediators[0]
                proposed_deal = await self._mediator_synthesise(
                    mediator, agent_offers, round_num
                )
            else:
                # No mediator: use average of offers as the proposal
                proposed_deal = self._average_offers(agent_offers)

            # ---- Step 3: agents evaluate the proposed deal ----------------- #
            eval_tasks = [a.evaluate_deal(proposed_deal) for a in buyers_sellers]
            evaluations: list[bool] = await asyncio.gather(*eval_tasks)
            all_satisfied = all(evaluations)

            # ---- Step 4: build and store round log ------------------------- #
            round_log: dict[str, Any] = {
                "type": "round",
                "round": round_num,
                "offers": agent_offers,
                "proposed_deal": proposed_deal,
                "evaluations": {
                    a.name: accepted
                    for a, accepted in zip(buyers_sellers, evaluations)
                },
                "converged": all_satisfied,
            }
            self.round_logs.append(round_log)

            # Push to streaming queue
            await self._queue.put(round_log)

            # ---- Step 5: check convergence --------------------------------- #
            if all_satisfied:
                logger.info("[%s] converged at round %d", self.session_id, round_num)
                self.deal = {**proposed_deal, "status": "converged"}
                break

            # ---- Step 6: check max rounds ---------------------------------- #
            if round_num == MAX_ROUNDS:
                logger.warning(
                    "[%s] max rounds reached, forcing compromise", self.session_id
                )
                forced = self._force_compromise(agent_offers)
                self.deal = {**forced, "status": "forced"}
                # Update agent statuses
                for a in buyers_sellers:
                    a.status = "satisfied"

        self._compute_metrics()
        self.is_complete = True

        # Signal completion to streaming consumer
        completion_message: dict[str, Any] = {
            "type": "complete",
            "result": {
                "deal": self.deal,
                "metrics": self.metrics,
            },
        }
        await self._queue.put(completion_message)
        logger.info("[%s] session complete", self.session_id)

    # ---------------------------------------------------------------------- #
    # Mediator synthesis
    # ---------------------------------------------------------------------- #

    async def _mediator_synthesise(
        self,
        mediator: Agent,
        agent_offers: dict[str, dict[str, Any]],
        round_num: int,
    ) -> dict[str, Any]:
        """
        Ask the mediator agent to synthesise a deal from all current offers.
        Falls back to the arithmetic average if GPT-4o returns an unusable object.
        """
        mediator_deal = await mediator.make_offer(
            self.resource_pool,
            self.round_logs + [{"round": round_num, "offers": agent_offers}],
        )
        # Ensure the synthesised deal has at least the standard keys
        if "requested_units" not in mediator_deal:
            mediator_deal = self._average_offers(agent_offers)
        return mediator_deal

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _average_offers(
        agent_offers: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute numeric-average deal when no mediator is present."""
        if not agent_offers:
            return {}

        numeric_keys: set[str] = set()
        for offer in agent_offers.values():
            for k, v in offer.items():
                if isinstance(v, (int, float)):
                    numeric_keys.add(k)

        averaged: dict[str, Any] = {}
        for key in numeric_keys:
            vals = [
                float(o[key])
                for o in agent_offers.values()
                if key in o and isinstance(o[key], (int, float))
            ]
            averaged[key] = round(sum(vals) / len(vals), 4) if vals else 0.0

        return averaged

    def _force_compromise(
        self,
        agent_offers: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Force a compromise by averaging numeric fields across all latest offers.
        Called only when MAX_ROUNDS is exhausted without convergence.
        """
        return self._average_offers(agent_offers)

    def _compute_metrics(self) -> None:
        """Populate self.metrics once the session ends."""
        status = self.deal.get("status", "in_progress")
        rounds_to_converge = (
            self.current_round if status == "converged" else MAX_ROUNDS
        )

        satisfaction_scores: dict[str, float] = {
            a.name: a.satisfaction_score(self.deal)
            for a in self.agents
        }

        self.metrics = {
            "rounds_to_converge": rounds_to_converge,
            "agent_satisfaction_scores": satisfaction_scores,
            "total_rounds": self.current_round,
        }

    # ---------------------------------------------------------------------- #
    # Status helpers
    # ---------------------------------------------------------------------- #

    def status_payload(self) -> dict[str, Any]:
        """Return the payload for GET /negotiation/{session_id}/status."""
        return {
            "session_id": self.session_id,
            "domain": self.domain,
            "current_round": self.current_round,
            "agent_statuses": [a.to_status_dict() for a in self.agents],
            "is_complete": self.is_complete,
        }

    def result_payload(self) -> dict[str, Any]:
        """Return the payload for GET /negotiation/{session_id}/result."""
        deal_status = self.deal.get("status", "in_progress")
        return {
            "session_id": self.session_id,
            "domain": self.domain,
            "status": deal_status,
            "deal": self.deal,
            "metrics": self.metrics,
            "rounds_log": self.round_logs,
        }
