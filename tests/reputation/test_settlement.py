"""Tests for aragora.reputation.settlement."""

from __future__ import annotations

import pytest

from aragora.reputation.settlement import SettlementError, settle_claim
from aragora.reputation.types import (
    DOMAIN_DEBATE_POSITION,
    DOMAIN_PREDICTION_MARKET,
    ResolvedClaim,
    StakeableClaim,
)


def _claim(
    *,
    probability: float | None = 0.7,
    position: str = "yes",
    stake: int = 50,
    domain: str = DOMAIN_PREDICTION_MARKET,
) -> StakeableClaim:
    return StakeableClaim.create(
        agent_id="alice",
        domain=domain,
        statement="X will happen",
        position=position,
        stake_units=stake,
        resolution_source="synthetic_github",
        resolution_id="mkt_x",
        predicted_probability=probability,
    )


def _resolved(claim: StakeableClaim, outcome: str) -> ResolvedClaim:
    return ResolvedClaim(
        claim_id=claim.claim_id,
        outcome=outcome,  # type: ignore[arg-type]
        resolved_at="2026-04-24T12:00:00Z",
        resolution_source="synthetic_github",
    )


class TestBrierProper:
    def test_perfect_prediction_returns_plus_stake(self) -> None:
        claim = _claim(probability=1.0, stake=50)
        resolved = _resolved(claim, "yes")
        delta = settle_claim(claim, resolved, scoring_rule="brier_proper")
        assert delta.delta == pytest.approx(50.0)
        assert delta.scoring_rule == "brier_proper"
        assert delta.reason["brier"] == pytest.approx(0.0)

    def test_maximally_wrong_returns_minus_stake(self) -> None:
        claim = _claim(probability=0.0, stake=50)
        resolved = _resolved(claim, "yes")
        delta = settle_claim(claim, resolved, scoring_rule="brier_proper")
        assert delta.delta == pytest.approx(-50.0)

    def test_calibrated_prior_breaks_even_at_half_stake(self) -> None:
        claim = _claim(probability=0.5, stake=50)
        resolved = _resolved(claim, "yes")
        delta = settle_claim(claim, resolved, scoring_rule="brier_proper")
        # Brier=0.25 → fraction=0.5 → delta=25
        assert delta.delta == pytest.approx(25.0)

    def test_requires_probability(self) -> None:
        claim = _claim(probability=None, position="yes")
        resolved = _resolved(claim, "yes")
        with pytest.raises(SettlementError):
            settle_claim(claim, resolved, scoring_rule="brier_proper")


class TestBinary:
    def test_correct_binary_position_returns_plus_stake(self) -> None:
        claim = _claim(probability=None, position="yes", domain=DOMAIN_DEBATE_POSITION)
        resolved = _resolved(claim, "yes")
        delta = settle_claim(claim, resolved, scoring_rule="binary")
        assert delta.delta == pytest.approx(50.0)
        assert delta.reason["correct"] is True

    def test_wrong_binary_position_returns_minus_stake(self) -> None:
        claim = _claim(probability=None, position="no", domain=DOMAIN_DEBATE_POSITION)
        resolved = _resolved(claim, "yes")
        delta = settle_claim(claim, resolved, scoring_rule="binary")
        assert delta.delta == pytest.approx(-50.0)
        assert delta.reason["correct"] is False


class TestInconclusive:
    def test_inconclusive_outcome_zero_delta(self) -> None:
        claim = _claim(probability=0.7, stake=50)
        resolved = _resolved(claim, "inconclusive")
        delta = settle_claim(claim, resolved, scoring_rule="brier_proper")
        assert delta.delta == 0.0

    def test_invalid_outcome_zero_delta(self) -> None:
        claim = _claim(probability=0.7, stake=50)
        resolved = _resolved(claim, "invalid")
        delta = settle_claim(claim, resolved, scoring_rule="brier_proper")
        assert delta.delta == 0.0


class TestMismatch:
    def test_claim_id_mismatch_raises(self) -> None:
        claim = _claim()
        resolved = ResolvedClaim(
            claim_id="clm_other",
            outcome="yes",
            resolved_at="2026-04-24T12:00:00Z",
            resolution_source="x",
        )
        with pytest.raises(SettlementError):
            settle_claim(claim, resolved, scoring_rule="brier_proper")


class TestDeltaProvenance:
    def test_delta_id_is_deterministic(self) -> None:
        claim = _claim(probability=0.7)
        resolved = _resolved(claim, "yes")
        a = settle_claim(claim, resolved)
        b = settle_claim(claim, resolved)
        assert a.delta_id == b.delta_id

    def test_reason_carries_full_provenance(self) -> None:
        claim = _claim(probability=0.7, stake=50)
        resolved = _resolved(claim, "yes")
        delta = settle_claim(claim, resolved, scoring_rule="brier_proper")
        assert delta.reason["claim_id"] == claim.claim_id
        assert delta.reason["resolution_id"] == claim.resolution_id
        assert delta.reason["stake_units"] == 50
        assert delta.reason["scoring_rule"] == "brier_proper"
