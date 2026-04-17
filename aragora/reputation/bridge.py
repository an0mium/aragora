"""Bridges from AGT-04 market events to AGT-05 reputation types.

This module converts a (:class:`aragora.markets.MarketPosition`,
:class:`aragora.markets.Market`, :class:`aragora.markets.ResolutionEvent`)
tuple — the shape AGT-04 produces when a synthetic GitHub market
resolves — into the unified AGT-05 (:class:`StakeableClaim`,
:class:`ResolvedClaim`) pair that the settlement function consumes.

The CruxSet bridge lives in a follow-up PR once DIC-17 (failed-claim /
open-crux → bounded follow-up) is wired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aragora.reputation.types import (
    DOMAIN_PREDICTION_MARKET,
    ResolvedClaim,
    StakeableClaim,
)

if TYPE_CHECKING:
    from aragora.markets.types import (
        Market,
        MarketPosition,
        ResolutionEvent,
    )


def bridge_from_market_position(
    position: "MarketPosition",
    market: "Market",
    resolution: "ResolutionEvent",
    *,
    resolution_source: str = "synthetic_github",
) -> tuple[StakeableClaim, ResolvedClaim]:
    """Convert an AGT-04 market position + resolution to AGT-05 claim + resolved.

    The returned tuple feeds directly into
    :func:`aragora.reputation.settlement.settle_claim`.

    Mapping choices:

    - ``domain`` is always ``DOMAIN_PREDICTION_MARKET`` — even for
      synthetic GitHub markets, the structure is identical to external
      prediction markets (probability-over-binary-outcome).
    - ``position`` is ``"yes"`` if ``predicted_probability >= 0.5``
      else ``"no"``. This lets binary scoring work too if a caller
      prefers that rule.
    - ``predicted_probability`` is carried through for Brier scoring.
    - ``resolution_id`` is the market_id (stable across runs because
      markets are content-addressed).
    - ``provenance`` carries the market target (repo, number/ref),
      position_id, and market_id so operators can audit the path.
    """
    if position.market_id != market.market_id:
        raise ValueError(
            f"market_id mismatch: position={position.market_id!r} vs market={market.market_id!r}"
        )
    if resolution.market_id != market.market_id:
        raise ValueError(
            f"market_id mismatch: resolution={resolution.market_id!r} vs market={market.market_id!r}"
        )

    derived_position = "yes" if position.probability >= 0.5 else "no"
    claim = StakeableClaim.create(
        agent_id=position.agent_id,
        domain=DOMAIN_PREDICTION_MARKET,
        statement=market.description or market.market_id,
        position=derived_position,
        predicted_probability=position.probability,
        stake_units=position.stake,
        stake_policy="forfeit_on_loss",
        resolution_source=resolution_source,
        resolution_id=market.market_id,
        provenance={
            "market_id": market.market_id,
            "position_id": position.position_id,
            "question_kind": market.question_kind,
            "target": dict(market.target),
            "submitted_at": position.submitted_at,
        },
        created_at=position.submitted_at,
    )

    resolved = ResolvedClaim(
        claim_id=claim.claim_id,
        outcome=resolution.outcome,
        resolved_at=resolution.resolved_at,
        resolution_source=resolution.resolution_source,
        evidence=dict(resolution.evidence),
    )
    return claim, resolved


__all__ = ["bridge_from_market_position"]
