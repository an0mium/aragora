"""Metaculus → AGT-05 reputation bridge (AGT-05 / #6066).

Flag: ARAGORA_REPUTATION_FLOW_ENABLED (default off).
Outcome: resolution=1.0→"yes", 0.0→"no", else→"inconclusive".
Companion to :mod:`aragora.reputation.bridge` (synthetic-GitHub path).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from aragora.reputation.types import (
    DOMAIN_PREDICTION_MARKET,
    ClaimOutcome,
    ResolvedClaim,
    StakeableClaim,
    StakePolicy,
)

if TYPE_CHECKING:
    from aragora.connectors.prediction_markets.metaculus import MetaculusQuestion

_RESOLUTION_SOURCE = "metaculus"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def reputation_flow_enabled() -> bool:
    """Return True when ``ARAGORA_REPUTATION_FLOW_ENABLED`` is truthy."""
    return os.environ.get("ARAGORA_REPUTATION_FLOW_ENABLED", "").strip().lower() in _TRUTHY


def _metaculus_outcome(resolution: float | None) -> ClaimOutcome:
    if resolution == 1.0:
        return "yes"
    if resolution == 0.0:
        return "no"
    return "inconclusive"


def bridge_from_metaculus_question(
    question: "MetaculusQuestion",
    agent_id: str,
    predicted_probability: float,
    *,
    stake_units: int = 1,
    stake_policy: StakePolicy = "forfeit_on_loss",
    resolution_source: str = _RESOLUTION_SOURCE,
    submitted_at: str | None = None,
    require_enabled: bool = True,
) -> tuple[StakeableClaim, ResolvedClaim]:
    """Return (StakeableClaim, ResolvedClaim) ready for settlement.settle_claim."""
    if require_enabled and not reputation_flow_enabled():
        raise RuntimeError(
            "ARAGORA_REPUTATION_FLOW_ENABLED is not set; "
            "set it to '1' or 'true' to use the Metaculus reputation bridge"
        )
    if not question.is_resolved:
        raise ValueError(
            f"MetaculusQuestion {question.question_id!r} is not resolved "
            f"(active_state={question.active_state!r})"
        )
    if not (0.0 <= predicted_probability <= 1.0):
        raise ValueError(
            f"predicted_probability must be in [0.0, 1.0]; got {predicted_probability!r}"
        )
    if stake_units < 1:
        raise ValueError(f"stake_units must be >= 1; got {stake_units!r}")

    _now = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    ts = submitted_at or _now
    resolved_at = question.resolve_time or _now

    claim = StakeableClaim.create(
        agent_id=agent_id,
        domain=DOMAIN_PREDICTION_MARKET,
        statement=question.title,
        position="yes" if predicted_probability >= 0.5 else "no",
        predicted_probability=predicted_probability,
        stake_units=stake_units,
        stake_policy=stake_policy,
        resolution_source=resolution_source,
        resolution_id=str(question.question_id),
        provenance={
            "question_id": question.question_id,
            "question_type": question.question_type,
            "close_time": question.close_time,
            "resolve_time": question.resolve_time,
            "submitted_at": ts,
        },
        created_at=ts,
    )
    resolved = ResolvedClaim(
        claim_id=claim.claim_id,
        outcome=_metaculus_outcome(question.resolution),
        resolved_at=resolved_at,
        resolution_source=resolution_source,
        evidence={
            "question_id": question.question_id,
            "resolution": question.resolution,
            "community_q2": question.community_q2,
            "active_state": question.active_state,
        },
    )
    return claim, resolved


__all__ = ["bridge_from_metaculus_question", "reputation_flow_enabled"]
