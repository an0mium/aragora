"""Tests for aragora.reputation.types."""

from __future__ import annotations

import pytest

from aragora.reputation.types import (
    DOMAIN_PREDICTION_MARKET,
    ReputationDelta,
    ResolvedClaim,
    StakeableClaim,
)


class TestStakeableClaim:
    def test_create_builds_valid_claim(self) -> None:
        claim = StakeableClaim.create(
            agent_id="alice",
            domain=DOMAIN_PREDICTION_MARKET,
            statement="will PR #5959 merge",
            position="yes",
            stake_units=50,
            resolution_source="synthetic_github",
            resolution_id="mkt_pr_merge_abc",
            predicted_probability=0.72,
        )
        assert claim.agent_id == "alice"
        assert claim.domain == DOMAIN_PREDICTION_MARKET
        assert claim.predicted_probability == pytest.approx(0.72)
        assert claim.claim_id.startswith("clm_")

    def test_identical_inputs_produce_same_claim_id(self) -> None:
        a = StakeableClaim.create(
            agent_id="alice",
            domain=DOMAIN_PREDICTION_MARKET,
            statement="will PR #1 merge",
            position="yes",
            stake_units=10,
            resolution_source="synthetic_github",
            resolution_id="mkt_x",
        )
        b = StakeableClaim.create(
            agent_id="alice",
            domain=DOMAIN_PREDICTION_MARKET,
            statement="will PR #1 merge",
            position="yes",
            stake_units=10,
            resolution_source="synthetic_github",
            resolution_id="mkt_x",
        )
        assert a.claim_id == b.claim_id

    def test_rejects_unknown_domain(self) -> None:
        with pytest.raises(ValueError):
            StakeableClaim.create(
                agent_id="alice",
                domain="made_up_domain",
                statement="x",
                position="yes",
                stake_units=10,
                resolution_source="x",
                resolution_id="y",
            )

    def test_rejects_invalid_probability(self) -> None:
        with pytest.raises(ValueError):
            StakeableClaim.create(
                agent_id="alice",
                domain=DOMAIN_PREDICTION_MARKET,
                statement="x",
                position="yes",
                stake_units=10,
                resolution_source="x",
                resolution_id="y",
                predicted_probability=1.5,
            )

    def test_rejects_non_positive_stake(self) -> None:
        with pytest.raises(ValueError):
            StakeableClaim.create(
                agent_id="alice",
                domain=DOMAIN_PREDICTION_MARKET,
                statement="x",
                position="yes",
                stake_units=0,
                resolution_source="x",
                resolution_id="y",
            )

    def test_rejects_empty_agent_or_statement(self) -> None:
        with pytest.raises(ValueError):
            StakeableClaim.create(
                agent_id=" ",
                domain=DOMAIN_PREDICTION_MARKET,
                statement="x",
                position="yes",
                stake_units=1,
                resolution_source="x",
                resolution_id="y",
            )

    def test_json_roundtrip(self) -> None:
        claim = StakeableClaim.create(
            agent_id="alice",
            domain=DOMAIN_PREDICTION_MARKET,
            statement="will X happen",
            position="yes",
            stake_units=25,
            resolution_source="manifold",
            resolution_id="mkt_m1",
            predicted_probability=0.4,
            provenance={"debate_id": "d1"},
        )
        rt = StakeableClaim.from_json(claim.to_json())
        assert rt == claim


class TestResolvedClaim:
    def test_rejects_unknown_outcome(self) -> None:
        with pytest.raises(ValueError):
            ResolvedClaim(
                claim_id="clm_x",
                outcome="maybe",  # type: ignore[arg-type]
                resolved_at="2026-04-17T12:00:00Z",
                resolution_source="x",
            )

    def test_requires_claim_id(self) -> None:
        with pytest.raises(ValueError):
            ResolvedClaim(
                claim_id=" ",
                outcome="yes",
                resolved_at="2026-04-17T12:00:00Z",
                resolution_source="x",
            )

    def test_json_roundtrip(self) -> None:
        resolved = ResolvedClaim(
            claim_id="clm_x",
            outcome="yes",
            resolved_at="2026-04-17T12:00:00Z",
            resolution_source="synthetic_github",
            evidence={"state": "MERGED", "merged": True},
        )
        rt = ResolvedClaim.from_json(resolved.to_json())
        assert rt == resolved


class TestReputationDelta:
    def test_json_roundtrip(self) -> None:
        delta = ReputationDelta(
            delta_id="rep_abcdef",
            agent_id="alice",
            domain=DOMAIN_PREDICTION_MARKET,
            claim_id="clm_x",
            resolution_id="mkt_x",
            delta=12.5,
            scoring_rule="brier_proper",
            applied_at="2026-04-17T12:05:00Z",
            decay_half_life_days=30.0,
            reason={"brier": 0.09, "stake_units": 50},
        )
        rt = ReputationDelta.from_json(delta.to_json())
        assert rt == delta
