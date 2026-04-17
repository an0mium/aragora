"""Tests for aragora.reputation.bridge — AGT-04 → AGT-05 conversion."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aragora.markets.types import Market, MarketPosition, ResolutionEvent as MarketResolution
from aragora.reputation.bridge import bridge_from_market_position
from aragora.reputation.settlement import settle_claim
from aragora.reputation.types import DOMAIN_PREDICTION_MARKET


def _market_position_resolution(
    *,
    probability: float = 0.7,
    stake: int = 40,
    outcome: str = "yes",
) -> tuple[Market, MarketPosition, MarketResolution]:
    market = Market.create(
        question_kind="pr_merge",
        target={"repo": "synaptent/aragora", "number": 5959},
        description="will #5959 merge",
        resolution_window_days=7,
        created_at=datetime(2026, 4, 10, tzinfo=UTC),
    )
    position = MarketPosition.create(
        market_id=market.market_id,
        agent_id="alice",
        probability=probability,
        stake=stake,
        submitted_at=datetime(2026, 4, 11, tzinfo=UTC),
    )
    resolution = (
        MarketResolution.yes(
            market_id=market.market_id,
            resolution_source="github_pr_state",
            resolved_at=datetime(2026, 4, 17, tzinfo=UTC),
            evidence={"merged": True},
        )
        if outcome == "yes"
        else MarketResolution.no(
            market_id=market.market_id,
            resolution_source="github_pr_state",
            resolved_at=datetime(2026, 4, 17, tzinfo=UTC),
            evidence={"merged": False},
        )
        if outcome == "no"
        else MarketResolution.inconclusive(
            market_id=market.market_id,
            resolution_source="github_pr_state",
            resolved_at=datetime(2026, 4, 17, tzinfo=UTC),
            evidence={"state": "OPEN"},
        )
    )
    return market, position, resolution


class TestBridgeFromMarketPosition:
    def test_produces_prediction_market_claim(self) -> None:
        market, position, resolution = _market_position_resolution()
        claim, resolved = bridge_from_market_position(position, market, resolution)
        assert claim.domain == DOMAIN_PREDICTION_MARKET
        assert claim.agent_id == "alice"
        assert claim.predicted_probability == pytest.approx(0.7)
        assert claim.stake_units == 40
        assert claim.resolution_id == market.market_id
        assert claim.position == "yes"  # probability >= 0.5

    def test_position_below_half_maps_to_no(self) -> None:
        market, position, resolution = _market_position_resolution(probability=0.3)
        claim, _ = bridge_from_market_position(position, market, resolution)
        assert claim.position == "no"

    def test_provenance_carries_market_context(self) -> None:
        market, position, resolution = _market_position_resolution()
        claim, _ = bridge_from_market_position(position, market, resolution)
        assert claim.provenance["market_id"] == market.market_id
        assert claim.provenance["position_id"] == position.position_id
        assert claim.provenance["question_kind"] == "pr_merge"
        assert claim.provenance["target"]["repo"] == "synaptent/aragora"

    def test_resolved_claim_carries_outcome_and_evidence(self) -> None:
        market, position, resolution = _market_position_resolution(outcome="yes")
        claim, resolved = bridge_from_market_position(position, market, resolution)
        assert resolved.claim_id == claim.claim_id
        assert resolved.outcome == "yes"
        assert resolved.evidence["merged"] is True

    def test_market_id_mismatch_raises(self) -> None:
        market, position, resolution = _market_position_resolution()
        # Construct a divergent resolution
        bad_resolution = MarketResolution.yes(
            market_id="mkt_other",
            resolution_source="github_pr_state",
            resolved_at=datetime(2026, 4, 17, tzinfo=UTC),
        )
        with pytest.raises(ValueError):
            bridge_from_market_position(position, market, bad_resolution)


class TestEndToEndFlow:
    def test_market_win_via_brier_proper_returns_positive_delta(self) -> None:
        # Probability 0.9 on YES, outcome YES → Brier=0.01, +48 expected
        market, position, resolution = _market_position_resolution(
            probability=0.9, stake=50, outcome="yes"
        )
        claim, resolved = bridge_from_market_position(position, market, resolution)
        delta = settle_claim(claim, resolved, scoring_rule="brier_proper")
        assert delta.delta == pytest.approx(49.0)  # (1 - 2*0.01) * 50
        assert delta.domain == DOMAIN_PREDICTION_MARKET

    def test_market_loss_returns_negative_delta(self) -> None:
        # Probability 0.9 on YES, outcome NO → Brier=0.81, -31 expected (rounded)
        market, position, resolution = _market_position_resolution(
            probability=0.9, stake=50, outcome="no"
        )
        claim, resolved = bridge_from_market_position(position, market, resolution)
        delta = settle_claim(claim, resolved, scoring_rule="brier_proper")
        assert delta.delta == pytest.approx(-31.0)  # (1 - 2*0.81) * 50

    def test_inconclusive_market_returns_zero_delta(self) -> None:
        market, position, resolution = _market_position_resolution(outcome="inconclusive")
        claim, resolved = bridge_from_market_position(position, market, resolution)
        delta = settle_claim(claim, resolved, scoring_rule="brier_proper")
        assert delta.delta == 0.0
