"""AGT-05 stale-policy seed (shadow only).

A *stale* claim is one whose resolution would still type-check but whose
information value has decayed past a threshold. The settlement module
already supports time-decay via ``decay_half_life_days``. This module
adds an *explicit, named* policy surface that future settlement,
leaderboard, and audit code can call without each call site rolling its
own age math.

This is shadow-only:

- It defines :class:`StalePolicy` with conservative defaults that match
  the existing 30-day settlement half-life.
- It exposes :func:`is_stale` as the public predicate.
- It returns a :class:`StaleDecision` carrying the reason, the age in
  days, and the policy fingerprint, so audit trails are
  machine-greppable.
- No production code path calls into this module yet.

A future PR can wire this into ``settle_claim``, the calibration
leaderboard, and the rev-4 staging scorecard. Until then, this seed
exists only so the policy surface is reviewable in isolation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

# Default tuning. These mirror the half-life used in
# ``aragora.reputation.settlement.settle_claim`` so a claim that is
# materially decayed under the existing scoring rule is also flagged
# stale by this predicate.
DEFAULT_FRESH_DAYS: Final[float] = 7.0
DEFAULT_STALE_DAYS: Final[float] = 30.0
DEFAULT_HARD_LIMIT_DAYS: Final[float] = 180.0


@dataclass(frozen=True)
class StalePolicy:
    """Bounds for the stale-claim predicate.

    Attributes:
        fresh_days: Claims younger than this are always fresh.
        stale_days: Claims older than this are always stale.
        hard_limit_days: Claims older than this are *expired* and
            should not even be settled.
    """

    fresh_days: float = DEFAULT_FRESH_DAYS
    stale_days: float = DEFAULT_STALE_DAYS
    hard_limit_days: float = DEFAULT_HARD_LIMIT_DAYS

    def __post_init__(self) -> None:
        if not (0 < self.fresh_days <= self.stale_days <= self.hard_limit_days):
            raise ValueError(
                "StalePolicy bounds must satisfy "
                "0 < fresh_days <= stale_days <= hard_limit_days; got "
                f"fresh={self.fresh_days}, stale={self.stale_days}, "
                f"hard_limit={self.hard_limit_days}"
            )

    def fingerprint(self) -> str:
        """Stable 12-char hash of the policy parameters.

        Useful for tagging audit rows so a future change in defaults can
        be tracked in receipts without a schema migration.
        """
        material = json.dumps(
            {
                "fresh_days": self.fresh_days,
                "stale_days": self.stale_days,
                "hard_limit_days": self.hard_limit_days,
            },
            sort_keys=True,
        )
        return f"sp_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:12]}"


@dataclass(frozen=True)
class StaleDecision:
    """The outcome of a stale-policy evaluation.

    Attributes:
        is_stale: True iff the claim should be treated as stale.
        is_expired: True iff the claim is past the hard-limit and
            should not be settled at all.
        age_days: Age of the claim in days at the evaluation moment.
        bucket: One of ``"fresh"``, ``"stale"``, ``"expired"``.
        policy_fingerprint: Stable hash of the policy parameters.
    """

    is_stale: bool
    is_expired: bool
    age_days: float
    bucket: str
    policy_fingerprint: str


def _age_days(claim_iso: str, now_iso: str) -> float:
    claim_at = datetime.fromisoformat(claim_iso.replace("Z", "+00:00"))
    if claim_at.tzinfo is None:
        claim_at = claim_at.replace(tzinfo=timezone.utc)
    now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta = (now - claim_at).total_seconds() / 86400.0
    if delta < 0:
        # Future-dated claims are treated as fresh; we never lie about age.
        return 0.0
    return delta


def is_stale(
    *,
    claim_iso: str,
    now_iso: str | None = None,
    policy: StalePolicy | None = None,
) -> StaleDecision:
    """Decide whether a claim is fresh, stale, or expired.

    Args:
        claim_iso: ISO-8601 timestamp at which the claim was made.
        now_iso: Optional override for the evaluation moment; defaults
            to ``datetime.now(timezone.utc)``.
        policy: Optional :class:`StalePolicy`; defaults to the
            conservative module-level defaults.

    Returns:
        A :class:`StaleDecision`. The decision is *advisory* — call
        sites are responsible for whatever action it implies (skip
        settlement, mark a leaderboard cell, append to an audit row).
    """
    if not claim_iso:
        raise ValueError("claim_iso must be a non-empty ISO-8601 string")
    if now_iso is None:
        now_iso = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    p = policy or StalePolicy()
    age = _age_days(claim_iso, now_iso)
    if age > p.hard_limit_days:
        return StaleDecision(
            is_stale=True,
            is_expired=True,
            age_days=age,
            bucket="expired",
            policy_fingerprint=p.fingerprint(),
        )
    if age >= p.stale_days:
        return StaleDecision(
            is_stale=True,
            is_expired=False,
            age_days=age,
            bucket="stale",
            policy_fingerprint=p.fingerprint(),
        )
    return StaleDecision(
        is_stale=False,
        is_expired=False,
        age_days=age,
        bucket="fresh",
        policy_fingerprint=p.fingerprint(),
    )


__all__ = [
    "DEFAULT_FRESH_DAYS",
    "DEFAULT_STALE_DAYS",
    "DEFAULT_HARD_LIMIT_DAYS",
    "StalePolicy",
    "StaleDecision",
    "is_stale",
]
