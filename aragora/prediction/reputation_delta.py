"""ReputationDelta — AGT-05 sub-deliverable 5: Brier-based reputation delta computation.

Converts resolved StakeableClaim objects into per-agent reputation deltas.
Feed point for the ERC-8004 registry write path (future slice).

Feature flag: ARAGORA_PREDICTION_MARKETS_ENABLED (same gate as stakeable_claim.py).
No live dispatch, no blockchain writes, no external API calls.

Advances: issue #6066 (AGT-05).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from aragora.prediction.stakeable_claim import ResolutionStatus, StakeableClaim

_ENV_FLAG = "ARAGORA_PREDICTION_MARKETS_ENABLED"


def _flag_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "").lower() in {"1", "true", "yes", "on"}


def _require_enabled() -> None:
    if not _flag_enabled():
        raise RuntimeError(f"Prediction markets are disabled. Set {_ENV_FLAG}=1 to enable.")


@dataclass(frozen=True)
class ReputationDelta:
    """Reputation change for one agent on one resolved claim.

    delta = 0.25 - brier_score  (calibrated: 0 at no-skill baseline prob=0.5)
    brier_score = (agent_probability - outcome)**2; outcome 1.0=YES 0.0=NO

    Range: perfect prediction → +0.25; maximally wrong → -0.75.
    """

    agent_id: str
    claim_id: str
    delta: float            # calibrated reputation change in [-0.75, 0.25]
    brier_score: float      # raw Brier score in [0.0, 1.0]
    resolved_yes: bool      # True = RESOLVED_YES, False = RESOLVED_NO
    agent_probability: float
    computed_at: str        # ISO-8601 UTC

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "claim_id": self.claim_id,
            "delta": self.delta,
            "brier_score": self.brier_score,
            "resolved_yes": self.resolved_yes,
            "agent_probability": self.agent_probability,
            "computed_at": self.computed_at,
        }


def compute_reputation_deltas(
    claims: list[StakeableClaim],
    *,
    window_days: int = 90,
    cutoff_dt: datetime | None = None,
    require_enabled: bool = True,
) -> list[ReputationDelta]:
    """Return per-agent reputation deltas for resolved claims in a rolling window.

    Only RESOLVED_YES / RESOLVED_NO claims whose ``expiry`` falls within
    [cutoff_dt - window_days, cutoff_dt] are processed; all others are skipped.
    Claims with no agent positions are also skipped.
    """
    if require_enabled:
        _require_enabled()

    if cutoff_dt is None:
        cutoff_dt = datetime.now(UTC)
    window_start = cutoff_dt - timedelta(days=window_days)
    now_str = cutoff_dt.isoformat()
    _resolved = {ResolutionStatus.RESOLVED_YES, ResolutionStatus.RESOLVED_NO}

    deltas: list[ReputationDelta] = []
    for claim in claims:
        if claim.resolution_status not in _resolved:
            continue
        if claim.resolution_value is None or not claim.positions:
            continue
        try:
            expiry_dt = datetime.fromisoformat(claim.expiry)
        except ValueError:
            continue
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=UTC)
        if not (window_start <= expiry_dt <= cutoff_dt):
            continue

        outcome = 1.0 if claim.resolution_value else 0.0
        resolved_yes = claim.resolution_status == ResolutionStatus.RESOLVED_YES
        for agent_id, prob in claim.positions.items():
            prob = float(prob)
            brier = (prob - outcome) ** 2
            deltas.append(
                ReputationDelta(
                    agent_id=agent_id,
                    claim_id=claim.claim_id,
                    delta=round(0.25 - brier, 6),
                    brier_score=round(brier, 6),
                    resolved_yes=resolved_yes,
                    agent_probability=prob,
                    computed_at=now_str,
                )
            )
    return deltas
