"""Tests for aragora.reputation.claim_verifier_bridge (AGT-05 #6066).

Verifies that bridge_from_claim_result converts every ClaimStatus value
to the correct AGT-05 ClaimOutcome, populates provenance correctly, and
produces claim_ids that are stable across calls with the same inputs.

No network dependency; all ClaimResult objects are constructed directly.
"""

from __future__ import annotations

import pytest

from aragora.epistemic.claim_verifier import ClaimResult, ClaimStatus
from aragora.reputation.claim_verifier_bridge import (
    bridge_from_claim_result,
    settle_from_claim_result,
)
from aragora.reputation.types import (
    DOMAIN_EPISTEMIC_CLAIM,
    ReputationDelta,
    ResolvedClaim,
    StakeableClaim,
)


def _make_result(
    claim_id: str,
    status: ClaimStatus,
    *,
    message: str = "ok",
    elapsed_ms: float = 1.0,
) -> ClaimResult:
    return ClaimResult(
        claim_id=claim_id,
        status=status,
        message=message,
        severity="info",
        allowed_action="report_only",
        elapsed_ms=elapsed_ms,
        detail={},
    )


class TestBridgeOutcomeMapping:
    """Every ClaimStatus maps to the expected ClaimOutcome."""

    @pytest.mark.parametrize(
        "status,expected_outcome",
        [
            (ClaimStatus.PASS, "yes"),
            (ClaimStatus.FAIL, "no"),
            (ClaimStatus.STALE, "no"),  # stale → failure
            (ClaimStatus.UNSUPPORTED, "inconclusive"),
            (ClaimStatus.ERROR, "inconclusive"),
        ],
    )
    def test_outcome(self, status: ClaimStatus, expected_outcome: str) -> None:
        result = _make_result("test.claim.id", status)
        stakeable, resolved = bridge_from_claim_result(result, agent_id="agent-a")
        assert resolved.outcome == expected_outcome

    def test_stale_is_no_not_inconclusive(self) -> None:
        result = _make_result("stale.claim", ClaimStatus.STALE)
        _, resolved = bridge_from_claim_result(result, agent_id="agent-b")
        assert resolved.outcome == "no", (
            "stale evidence must be treated as failure, not inconclusive"
        )


class TestBridgeTypes:
    """Return values are the correct types with consistent claim_ids."""

    def test_returns_tuple_of_correct_types(self) -> None:
        result = _make_result("my.claim", ClaimStatus.PASS)
        pair = bridge_from_claim_result(result, agent_id="agent-c")
        assert isinstance(pair, tuple) and len(pair) == 2
        stakeable, resolved = pair
        assert isinstance(stakeable, StakeableClaim)
        assert isinstance(resolved, ResolvedClaim)

    def test_claim_ids_match(self) -> None:
        result = _make_result("link.claim", ClaimStatus.PASS)
        stakeable, resolved = bridge_from_claim_result(result, agent_id="agent-d")
        assert stakeable.claim_id == resolved.claim_id

    def test_domain_is_epistemic_claim(self) -> None:
        result = _make_result("domain.claim", ClaimStatus.PASS)
        stakeable, _ = bridge_from_claim_result(result, agent_id="agent-e")
        assert stakeable.domain == DOMAIN_EPISTEMIC_CLAIM

    def test_position_is_yes_and_binary_scoring(self) -> None:
        result = _make_result("binary.claim", ClaimStatus.PASS)
        stakeable, _ = bridge_from_claim_result(result, agent_id="agent-f")
        assert stakeable.position == "yes"
        assert stakeable.predicted_probability is None
        assert stakeable.stake_policy == "forfeit_on_loss"


class TestBridgeProvenance:
    """Provenance and evidence carry the original claim metadata."""

    def test_stakeable_provenance_contains_claim_id(self) -> None:
        result = _make_result("prov.claim", ClaimStatus.FAIL)
        stakeable, _ = bridge_from_claim_result(result, agent_id="agent-i")
        assert stakeable.provenance["claim_id"] == "prov.claim"

    def test_resolved_evidence_contains_status(self) -> None:
        result = _make_result("ev.claim", ClaimStatus.STALE, message="too old")
        _, resolved = bridge_from_claim_result(result, agent_id="agent-j")
        assert resolved.evidence["status"] == "stale"
        assert resolved.evidence["message"] == "too old"

    def test_resolved_evidence_contains_elapsed_ms(self) -> None:
        result = _make_result("elap.claim", ClaimStatus.PASS, elapsed_ms=42.5)
        _, resolved = bridge_from_claim_result(result, agent_id="agent-k")
        assert resolved.evidence["elapsed_ms"] == pytest.approx(42.5)

    def test_resolution_source_default(self) -> None:
        result = _make_result("src.claim", ClaimStatus.PASS)
        stakeable, resolved = bridge_from_claim_result(result, agent_id="agent-l")
        assert stakeable.resolution_source == "dic14_claim_verifier"
        assert resolved.resolution_source == "dic14_claim_verifier"

    def test_resolution_source_override(self) -> None:
        result = _make_result("custom.claim", ClaimStatus.PASS)
        stakeable, resolved = bridge_from_claim_result(
            result, agent_id="agent-m", resolution_source="custom_runner"
        )
        assert stakeable.resolution_source == "custom_runner"
        assert resolved.resolution_source == "custom_runner"


class TestBridgeStability:
    """Content-addressed claim_id is stable given the same inputs."""

    def test_same_inputs_same_claim_id(self) -> None:
        result = _make_result("stable.claim", ClaimStatus.PASS)
        s1, _ = bridge_from_claim_result(result, agent_id="agent-n")
        s2, _ = bridge_from_claim_result(result, agent_id="agent-n")
        assert s1.claim_id == s2.claim_id

    def test_different_agents_different_claim_ids(self) -> None:
        result = _make_result("diff.claim", ClaimStatus.PASS)
        s1, _ = bridge_from_claim_result(result, agent_id="agent-p")
        s2, _ = bridge_from_claim_result(result, agent_id="agent-q")
        assert s1.claim_id != s2.claim_id

    def test_different_resolution_ids_different_claim_ids(self) -> None:
        r1 = _make_result("claim.alpha", ClaimStatus.PASS)
        r2 = _make_result("claim.beta", ClaimStatus.PASS)
        s1, _ = bridge_from_claim_result(r1, agent_id="agent-r")
        s2, _ = bridge_from_claim_result(r2, agent_id="agent-r")
        assert s1.claim_id != s2.claim_id


class TestBridgeStakeUnits:
    """stake_units parameter is forwarded correctly."""

    def test_default_stake_units(self) -> None:
        result = _make_result("su.default", ClaimStatus.PASS)
        stakeable, _ = bridge_from_claim_result(result, agent_id="agent-s")
        assert stakeable.stake_units == 1

    def test_custom_stake_units(self) -> None:
        result = _make_result("su.custom", ClaimStatus.PASS)
        stakeable, _ = bridge_from_claim_result(result, agent_id="agent-t", stake_units=10)
        assert stakeable.stake_units == 10

    def test_stake_units_below_minimum_raises(self) -> None:
        result = _make_result("su.bad", ClaimStatus.PASS)
        with pytest.raises(ValueError, match="stake_units"):
            bridge_from_claim_result(result, agent_id="agent-u", stake_units=0)


class TestSettleFromClaimResult:
    """settle_from_claim_result: bridge + binary settlement in one call."""

    def test_returns_reputation_delta(self) -> None:
        result = _make_result("settle.pass", ClaimStatus.PASS)
        delta = settle_from_claim_result(result, agent_id="agent-v")
        assert isinstance(delta, ReputationDelta)

    def test_pass_yields_positive_delta(self) -> None:
        result = _make_result("settle.win", ClaimStatus.PASS, elapsed_ms=5.0)
        delta = settle_from_claim_result(result, agent_id="agent-w", stake_units=3)
        assert delta.delta == pytest.approx(3.0)

    def test_fail_yields_negative_delta(self) -> None:
        result = _make_result("settle.lose", ClaimStatus.FAIL)
        delta = settle_from_claim_result(result, agent_id="agent-x", stake_units=2)
        assert delta.delta == pytest.approx(-2.0)

    def test_stale_yields_negative_delta(self) -> None:
        result = _make_result("settle.stale", ClaimStatus.STALE)
        delta = settle_from_claim_result(result, agent_id="agent-y", stake_units=1)
        assert delta.delta == pytest.approx(-1.0)

    def test_inconclusive_yields_zero_delta(self) -> None:
        result = _make_result("settle.inconc", ClaimStatus.UNSUPPORTED)
        delta = settle_from_claim_result(result, agent_id="agent-z")
        assert delta.delta == pytest.approx(0.0)

    def test_scoring_rule_is_binary(self) -> None:
        result = _make_result("settle.rule", ClaimStatus.PASS)
        delta = settle_from_claim_result(result, agent_id="agent-aa")
        assert delta.scoring_rule == "binary"

    def test_domain_is_epistemic_claim(self) -> None:
        result = _make_result("settle.domain", ClaimStatus.PASS)
        delta = settle_from_claim_result(result, agent_id="agent-ab")
        assert delta.domain == DOMAIN_EPISTEMIC_CLAIM

    def test_decay_half_life_forwarded(self) -> None:
        result = _make_result("settle.decay", ClaimStatus.PASS)
        delta = settle_from_claim_result(result, agent_id="agent-ac", decay_half_life_days=90.0)
        assert delta.decay_half_life_days == pytest.approx(90.0)
