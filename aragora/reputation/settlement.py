"""AGT-05 settlement — compute ReputationDelta from StakeableClaim + ResolvedClaim.

The two scoring rules implemented here match the AGT-05 spec in
``docs/plans/SKIN_IN_THE_GAME_REPUTATION.md``:

- ``brier_proper`` — incentive-compatible for probabilistic claims.
  ``delta = stake * (1 - 2 * brier)`` where
  ``brier = (predicted_probability - realized) ** 2``. A perfectly
  calibrated 0/1 prediction returns +stake; a maximally wrong one
  returns -stake; the symmetric break-even point is a prediction of
  0.5, which returns half of the stake.
- ``binary`` — for domains where a probability is not reported and
  a typed ``position`` is compared to the outcome. Returns +stake if
  correct, -stake if wrong.

Inconclusive and invalid outcomes always return zero delta: under
``refund_on_inconclusive`` semantics the stake is returned; under
``forfeit_on_loss`` the caller may separately decide to forfeit.
This module only computes the delta.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from aragora.reputation.types import (
    ReputationDelta,
    ResolvedClaim,
    ScoringRule,
    StakeableClaim,
)


class SettlementError(RuntimeError):
    """Raised when a settlement cannot be computed."""


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _delta_id(agent_id: str, domain: str, claim_id: str, resolution_id: str) -> str:
    material = f"{agent_id}|{domain}|{claim_id}|{resolution_id}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"rep_{digest}"


def settle_claim(
    claim: StakeableClaim,
    resolved: ResolvedClaim,
    *,
    scoring_rule: ScoringRule = "brier_proper",
    decay_half_life_days: float | None = 30.0,
    applied_at: str | None = None,
) -> ReputationDelta:
    """Compute the reputation delta for a resolved claim.

    Raises :class:`SettlementError` if the claim and resolution do not
    refer to the same ``claim_id``, or if the scoring rule is not
    supported for the given claim shape.
    """
    if claim.claim_id != resolved.claim_id:
        raise SettlementError(
            f"claim_id mismatch: claim={claim.claim_id!r} vs resolved={resolved.claim_id!r}"
        )

    reason: dict[str, Any] = {
        "claim_id": claim.claim_id,
        "resolution_id": claim.resolution_id,
        "resolution_source": claim.resolution_source,
        "scoring_rule": scoring_rule,
        "outcome": resolved.outcome,
        "stake_units": claim.stake_units,
    }

    if resolved.outcome in {"inconclusive", "invalid"}:
        delta = 0.0
    elif scoring_rule == "brier_proper":
        if claim.predicted_probability is None:
            raise SettlementError("brier_proper requires predicted_probability on the claim")
        realized = 1.0 if resolved.outcome == "yes" else 0.0
        brier = (claim.predicted_probability - realized) ** 2
        # Symmetric payout around break-even at Brier=0.5
        payout_fraction = 1.0 - 2.0 * brier
        delta = payout_fraction * float(claim.stake_units)
        reason["brier"] = brier
        reason["payout_fraction"] = payout_fraction
    elif scoring_rule == "binary":
        correct = (
            (resolved.outcome == "yes" and claim.position == "yes")
            or (resolved.outcome == "no" and claim.position == "no")
            or (resolved.outcome == claim.position)
        )
        delta = float(claim.stake_units) * (1.0 if correct else -1.0)
        reason["correct"] = correct
    else:  # pragma: no cover - exhaustiveness
        raise SettlementError(f"unsupported scoring rule: {scoring_rule!r}")

    return ReputationDelta(
        delta_id=_delta_id(claim.agent_id, claim.domain, claim.claim_id, claim.resolution_id),
        agent_id=claim.agent_id,
        domain=claim.domain,
        claim_id=claim.claim_id,
        resolution_id=claim.resolution_id,
        delta=delta,
        scoring_rule=scoring_rule,
        applied_at=applied_at or _utc_now_iso(),
        decay_half_life_days=decay_half_life_days,
        reason=reason,
    )


__all__ = ["SettlementError", "settle_claim"]
