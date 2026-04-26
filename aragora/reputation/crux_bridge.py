"""Bridge from AGT-01 CruxSet positions to AGT-05 reputation types.

When a crux that appeared in a :class:`~aragora.reasoning.cruxset.CruxSet`
is later resolved — by debate consensus, an operator decision, or an
external oracle — each agent that staked a position on that crux deserves
a reputation delta.  This module provides:

* :class:`CruxPositionRecord` — an agent's staked position on one crux.
* :class:`CruxResolutionEvent` — the oracle's determination of which
  crux side won (or that the crux is inconclusive).
* :func:`bridge_from_crux_position` — converts the pair into a
  (:class:`~aragora.reputation.types.StakeableClaim`,
  :class:`~aragora.reputation.types.ResolvedClaim`) pair that feeds
  directly into :func:`~aragora.reputation.settlement.settle_claim`.

Feature-gating: this module is part of the ``DOMAIN_CRUX_RESOLUTION``
reputation flow.  It does **not** wire into any live debate or dispatch
path.  The connection from CruxDetector → CruxSet emission → position
recording → resolution → settlement is dormant until the AGT-* upper-layer
gate opens per ``docs/status/NEXT_STEPS_CANONICAL.md``.

Prior art: :mod:`aragora.reputation.bridge` handles
``DOMAIN_PREDICTION_MARKET``; this module adds ``DOMAIN_CRUX_RESOLUTION``
using the same (position, event) → (StakeableClaim, ResolvedClaim) pattern.
The comment in :mod:`aragora.reputation.bridge` notes that "the CruxSet
bridge lives in a follow-up PR once DIC-17 is wired" — DIC-17 is now
shipped in :mod:`aragora.epistemic.followup`, so this is that PR.

Scoring note: callers should pass ``scoring_rule="binary"`` to
:func:`~aragora.reputation.settlement.settle_claim`.  Every staker's
``claim.position`` is normalised to ``"yes"`` (they claim their side will
prevail); ``resolved.outcome`` carries the per-agent verdict so the binary
rule rewards correct stakers and penalises incorrect ones symmetrically.

Advances: AGT-05 (#6066).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aragora.reputation.types import (
    DOMAIN_CRUX_RESOLUTION,
    ClaimOutcome,
    ResolvedClaim,
    StakeableClaim,
)

if TYPE_CHECKING:
    from aragora.reasoning.cruxset import Crux

# Sentinel for an inconclusive resolution — no side was validated.
_INCONCLUSIVE_SIDE = ""


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _position_id(agent_id: str, crux_id: str, cruxset_id: str, side: str) -> str:
    material = json.dumps(
        {
            "agent_id": agent_id,
            "crux_id": crux_id,
            "cruxset_id": cruxset_id,
            "side": side,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"cp_{digest}"


@dataclass(frozen=True)
class CruxPositionRecord:
    """An agent's staked position on a single crux in a CruxSet.

    Analogous to :class:`~aragora.markets.types.MarketPosition` for
    synthetic GitHub markets.  The agent commits ``stake_units`` of
    compute-credit to the claim that their chosen ``side`` is the correct
    resolution of ``crux_id`` within ``cruxset_id``.

    Use :meth:`create` to build instances with a content-addressed
    ``position_id``.
    """

    position_id: str
    agent_id: str
    crux_id: str
    cruxset_id: str
    side: str
    stake_units: int
    submitted_at: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.agent_id).strip():
            raise ValueError("agent_id must be non-empty")
        if not str(self.crux_id).strip():
            raise ValueError("crux_id must be non-empty")
        if not str(self.cruxset_id).strip():
            raise ValueError("cruxset_id must be non-empty")
        if not str(self.side).strip():
            raise ValueError("side must be non-empty")
        if self.stake_units < 1:
            raise ValueError(f"stake_units must be >= 1, got {self.stake_units}")

    @classmethod
    def create(
        cls,
        *,
        agent_id: str,
        crux_id: str,
        cruxset_id: str,
        side: str,
        stake_units: int,
        submitted_at: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> "CruxPositionRecord":
        """Build a CruxPositionRecord with a content-addressed position_id."""
        position_id = _position_id(agent_id, crux_id, cruxset_id, side)
        return cls(
            position_id=position_id,
            agent_id=agent_id,
            crux_id=crux_id,
            cruxset_id=cruxset_id,
            side=side,
            stake_units=stake_units,
            submitted_at=submitted_at or _utc_now_iso(),
            provenance=dict(provenance or {}),
        )


@dataclass(frozen=True)
class CruxResolutionEvent:
    """Oracle determination of which side of a crux won.

    Analogous to :class:`~aragora.markets.types.ResolutionEvent` for
    synthetic GitHub markets.  The oracle declares a ``winning_side``
    (e.g. ``"for"`` or ``"against"``) or leaves it empty to signal that
    the crux is inconclusive.

    Per-agent outcomes are computed in :func:`bridge_from_crux_position`:
    agents on the winning side receive ``"yes"``, others ``"no"``, and all
    agents receive ``"inconclusive"`` when ``winning_side`` is empty.

    Use :meth:`resolved` or :meth:`inconclusive` factory methods.
    """

    crux_id: str
    cruxset_id: str
    winning_side: str
    resolution_source: str
    resolved_at: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.crux_id).strip():
            raise ValueError("crux_id must be non-empty")
        if not str(self.cruxset_id).strip():
            raise ValueError("cruxset_id must be non-empty")
        if not str(self.resolution_source).strip():
            raise ValueError("resolution_source must be non-empty")

    @property
    def is_inconclusive(self) -> bool:
        """True when no winning side was determined."""
        return self.winning_side == _INCONCLUSIVE_SIDE

    @classmethod
    def resolved(
        cls,
        *,
        crux_id: str,
        cruxset_id: str,
        winning_side: str,
        resolution_source: str,
        resolved_at: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> "CruxResolutionEvent":
        """Build a resolution event with a specific winning side."""
        if not str(winning_side).strip():
            raise ValueError(
                "winning_side must be non-empty for a resolved event; "
                "use .inconclusive() for no-winner outcomes"
            )
        return cls(
            crux_id=crux_id,
            cruxset_id=cruxset_id,
            winning_side=winning_side,
            resolution_source=resolution_source,
            resolved_at=resolved_at or _utc_now_iso(),
            evidence=dict(evidence or {}),
        )

    @classmethod
    def inconclusive(
        cls,
        *,
        crux_id: str,
        cruxset_id: str,
        resolution_source: str,
        resolved_at: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> "CruxResolutionEvent":
        """Build an inconclusive resolution event (no winning side)."""
        return cls(
            crux_id=crux_id,
            cruxset_id=cruxset_id,
            winning_side=_INCONCLUSIVE_SIDE,
            resolution_source=resolution_source,
            resolved_at=resolved_at or _utc_now_iso(),
            evidence=dict(evidence or {}),
        )


def bridge_from_crux_position(
    position: "CruxPositionRecord",
    crux: "Crux",
    resolution: "CruxResolutionEvent",
) -> tuple[StakeableClaim, ResolvedClaim]:
    """Convert a crux position + resolution into an AGT-05 claim/resolved pair.

    The returned tuple feeds directly into
    :func:`aragora.reputation.settlement.settle_claim` with
    ``scoring_rule="binary"``.

    Scoring convention: ``claim.position`` is always ``"yes"`` because
    every staker is claiming "my side will prevail."  The per-agent verdict
    lives in ``resolved.outcome``:

    * ``"yes"``         — agent was on the winning side
    * ``"no"``          — agent was on the losing side
    * ``"inconclusive"`` — resolution declared no winner

    This mirrors :func:`aragora.reputation.bridge.bridge_from_market_position`,
    which normalises a continuous probability to ``"yes"``/``"no"`` via the
    0.5 threshold.

    Cross-checks:

    * ``position.crux_id`` must equal ``crux.crux_id``
    * ``position.crux_id`` / ``position.cruxset_id`` must equal
      ``resolution.crux_id`` / ``resolution.cruxset_id``
    """
    if position.crux_id != crux.crux_id:
        raise ValueError(
            f"position.crux_id={position.crux_id!r} does not match crux.crux_id={crux.crux_id!r}"
        )
    if position.crux_id != resolution.crux_id:
        raise ValueError(
            f"position.crux_id={position.crux_id!r} does not match "
            f"resolution.crux_id={resolution.crux_id!r}"
        )
    if position.cruxset_id != resolution.cruxset_id:
        raise ValueError(
            f"position.cruxset_id={position.cruxset_id!r} does not match "
            f"resolution.cruxset_id={resolution.cruxset_id!r}"
        )

    if resolution.is_inconclusive:
        agent_outcome: ClaimOutcome = "inconclusive"
    elif position.side == resolution.winning_side:
        agent_outcome = "yes"
    else:
        agent_outcome = "no"

    resolution_id = f"{resolution.cruxset_id}:{resolution.crux_id}"

    claim = StakeableClaim.create(
        agent_id=position.agent_id,
        domain=DOMAIN_CRUX_RESOLUTION,
        statement=crux.statement,
        position="yes",  # normalised: staker claims their side will prevail
        predicted_probability=None,  # binary stance — use scoring_rule="binary"
        stake_units=position.stake_units,
        stake_policy="forfeit_on_loss",
        resolution_source=resolution.resolution_source,
        resolution_id=resolution_id,
        provenance={
            "position_id": position.position_id,
            "crux_id": position.crux_id,
            "cruxset_id": position.cruxset_id,
            "agent_side": position.side,
            "winning_side": resolution.winning_side,
            "load_bearing_score": crux.load_bearing_score,
        },
        created_at=position.submitted_at,
    )

    resolved = ResolvedClaim(
        claim_id=claim.claim_id,
        outcome=agent_outcome,
        resolved_at=resolution.resolved_at,
        resolution_source=resolution.resolution_source,
        evidence=dict(resolution.evidence),
    )
    return claim, resolved


__all__ = [
    "CruxPositionRecord",
    "CruxResolutionEvent",
    "bridge_from_crux_position",
]
