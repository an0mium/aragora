"""Tests for aragora.reputation.anchor — AGT-05 on-chain anchoring."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from aragora.reputation.anchor import (
    DEFAULT_VALUE_DECIMALS,
    INT128_MAX,
    INT128_MIN,
    AnchorError,
    AnchorReceipt,
    anchor_delta,
    anchoring_enabled,
    compute_feedback_hash,
    compute_feedback_value,
    delta_to_feedback_args,
    enable_anchoring,
)
from aragora.reputation.settlement import settle_claim
from aragora.reputation.types import (
    DOMAIN_PREDICTION_MARKET,
    ResolvedClaim,
    StakeableClaim,
)


def _make_delta(*, probability: float = 0.9, stake: int = 50, outcome: str = "yes"):
    claim = StakeableClaim.create(
        agent_id="alice",
        domain=DOMAIN_PREDICTION_MARKET,
        statement="will X happen",
        position="yes",
        stake_units=stake,
        resolution_source="synthetic_github",
        resolution_id="mkt_x",
        predicted_probability=probability,
    )
    resolved = ResolvedClaim(
        claim_id=claim.claim_id,
        outcome=outcome,  # type: ignore[arg-type]
        resolved_at="2026-04-24T12:00:00Z",
        resolution_source="synthetic_github",
    )
    return settle_claim(claim, resolved, scoring_rule="brier_proper")


class _StubRegistry:
    """Stub matching ReputationRegistryContract.give_feedback signature."""

    def __init__(self, tx_hash: str = "0xdeadbeef", *, raises: Exception | None = None):
        self._tx_hash = tx_hash
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def give_feedback(
        self,
        agent_id: int,
        value: int,
        signer: Any,
        value_decimals: int = 0,
        tag1: str = "",
        tag2: str = "",
        endpoint: str = "",
        feedback_uri: str = "",
        feedback_hash: bytes = b"\x00" * 32,
    ) -> str:
        if self._raises is not None:
            raise self._raises
        self.calls.append(
            {
                "agent_id": agent_id,
                "value": value,
                "signer": signer,
                "value_decimals": value_decimals,
                "tag1": tag1,
                "tag2": tag2,
                "endpoint": endpoint,
                "feedback_uri": feedback_uri,
                "feedback_hash": feedback_hash,
            }
        )
        return self._tx_hash


class TestValueEncoding:
    def test_positive_delta_scales_by_decimals(self) -> None:
        delta = _make_delta(probability=0.9, outcome="yes")  # delta ≈ +49 (stake=50)
        value = compute_feedback_value(delta, value_decimals=6)
        # delta is ~49.0 → 49.0 * 1e6 ≈ 49000000
        assert 48_000_000 <= value <= 50_000_000

    def test_negative_delta_preserves_sign(self) -> None:
        delta = _make_delta(probability=0.9, outcome="no")  # Brier=0.81 → ~-31
        value = compute_feedback_value(delta, value_decimals=6)
        assert value < 0
        assert -32_000_000 <= value <= -30_000_000

    def test_zero_decimals_truncates_to_int(self) -> None:
        delta = _make_delta(probability=0.9, outcome="yes")  # ~+49
        value = compute_feedback_value(delta, value_decimals=0)
        assert value in {49, 48}  # rounding, but integer

    def test_decimals_bounds_enforced(self) -> None:
        delta = _make_delta()
        with pytest.raises(AnchorError):
            compute_feedback_value(delta, value_decimals=-1)
        with pytest.raises(AnchorError):
            compute_feedback_value(delta, value_decimals=19)


class TestFeedbackHash:
    def test_hash_is_32_bytes_sha256(self) -> None:
        delta = _make_delta()
        digest = compute_feedback_hash(delta)
        assert isinstance(digest, bytes)
        assert len(digest) == 32
        # Manually recompute
        canonical = json.dumps(delta.to_json(), sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).digest()
        assert digest == expected

    def test_hash_is_deterministic(self) -> None:
        delta = _make_delta()
        assert compute_feedback_hash(delta) == compute_feedback_hash(delta)

    def test_hash_differs_across_distinct_deltas(self) -> None:
        a = _make_delta(probability=0.9)
        b = _make_delta(probability=0.1)
        assert compute_feedback_hash(a) != compute_feedback_hash(b)


class TestDeltaToFeedbackArgs:
    def test_required_fields_present(self) -> None:
        delta = _make_delta()
        args = delta_to_feedback_args(delta, agent_id=42, value_decimals=6)
        assert args["agent_id"] == 42
        assert args["value_decimals"] == 6
        assert args["tag1"] == DOMAIN_PREDICTION_MARKET
        assert args["tag2"] == "brier_proper"
        assert isinstance(args["feedback_hash"], bytes)
        assert len(args["feedback_hash"]) == 32

    def test_rejects_negative_agent_id(self) -> None:
        delta = _make_delta()
        with pytest.raises(AnchorError):
            delta_to_feedback_args(delta, agent_id=-1)


class TestAnchorDeltaDryRun:
    def test_default_is_dry_run(self, monkeypatch) -> None:
        monkeypatch.delenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", raising=False)
        delta = _make_delta()
        receipt = anchor_delta(delta, agent_id=42)
        assert receipt.dry_run is True
        assert receipt.tx_hash is None
        assert receipt.error is None
        assert receipt.provenance["delta_id"] == delta.delta_id
        assert receipt.tag1 == DOMAIN_PREDICTION_MARKET

    def test_explicit_dry_run_true_honored(self, monkeypatch) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", "1")
        delta = _make_delta()
        receipt = anchor_delta(
            delta,
            agent_id=42,
            registry=_StubRegistry(),
            signer=object(),
            dry_run=True,
        )
        assert receipt.dry_run is True

    def test_flag_off_forces_dry_run_even_with_registry(self, monkeypatch) -> None:
        monkeypatch.delenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", raising=False)
        delta = _make_delta()
        registry = _StubRegistry()
        receipt = anchor_delta(
            delta,
            agent_id=42,
            registry=registry,
            signer=object(),
            dry_run=False,  # caller wants live, but flag is off
        )
        assert receipt.dry_run is True
        assert receipt.tx_hash is None
        assert registry.calls == []

    def test_missing_registry_forces_dry_run_even_with_flag(self, monkeypatch) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", "1")
        delta = _make_delta()
        receipt = anchor_delta(delta, agent_id=42, dry_run=False)
        assert receipt.dry_run is True

    def test_missing_signer_forces_dry_run(self, monkeypatch) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", "1")
        delta = _make_delta()
        receipt = anchor_delta(
            delta, agent_id=42, registry=_StubRegistry(), signer=None, dry_run=False
        )
        assert receipt.dry_run is True


class TestAnchorDeltaLive:
    def test_live_submission_records_tx_hash(self, monkeypatch) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", "1")
        delta = _make_delta()
        registry = _StubRegistry(tx_hash="0xabc123")
        receipt = anchor_delta(
            delta,
            agent_id=42,
            registry=registry,
            signer=object(),
            dry_run=False,
        )
        assert receipt.dry_run is False
        assert receipt.tx_hash == "0xabc123"
        assert receipt.error is None
        assert len(registry.calls) == 1
        call = registry.calls[0]
        assert call["agent_id"] == 42
        assert call["tag1"] == DOMAIN_PREDICTION_MARKET
        assert call["tag2"] == "brier_proper"
        assert call["value_decimals"] == DEFAULT_VALUE_DECIMALS
        assert len(call["feedback_hash"]) == 32

    def test_chain_error_captured_not_raised(self, monkeypatch) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", "1")
        delta = _make_delta()
        registry = _StubRegistry(raises=RuntimeError("rpc timeout"))
        receipt = anchor_delta(
            delta,
            agent_id=42,
            registry=registry,
            signer=object(),
            dry_run=False,
        )
        assert receipt.dry_run is False
        assert receipt.tx_hash is None
        assert receipt.error is not None
        assert "rpc timeout" in receipt.error
        assert "RuntimeError" in receipt.error


class TestFeatureFlag:
    def test_flag_default_off(self, monkeypatch) -> None:
        monkeypatch.delenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", raising=False)
        assert anchoring_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
    def test_truthy_values_enable(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", value)
        assert anchoring_enabled() is True

    def test_enable_helper(self, monkeypatch) -> None:
        monkeypatch.delenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", raising=False)
        assert anchoring_enabled() is False
        enable_anchoring()
        try:
            assert anchoring_enabled() is True
        finally:
            monkeypatch.delenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", raising=False)


class TestAnchorReceiptSerialization:
    def test_receipt_to_json_roundtrip(self, monkeypatch) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_ANCHORING_ENABLED", "1")
        delta = _make_delta()
        registry = _StubRegistry(tx_hash="0xfeed")
        receipt = anchor_delta(
            delta,
            agent_id=42,
            registry=registry,
            signer=object(),
            feedback_uri="ipfs://Qm...",
            dry_run=False,
        )
        payload = receipt.to_json()
        assert payload["tx_hash"] == "0xfeed"
        assert payload["agent_id"] == 42
        assert payload["tag1"] == DOMAIN_PREDICTION_MARKET
        assert payload["feedback_uri"] == "ipfs://Qm..."
        # All values JSON-serializable
        json.dumps(payload)


class TestValueClamp:
    """Regression guard for int128 overflow protection."""

    def test_overflow_clamped_to_int128_max(self) -> None:
        # Build a delta whose scaled value overflows int128
        from aragora.reputation.types import ReputationDelta

        bogus = ReputationDelta(
            delta_id="rep_overflow",
            agent_id="alice",
            domain="prediction_market",
            claim_id="clm_x",
            resolution_id="mkt_x",
            delta=float(10**35),  # scaled by 1e6 overflows int128 (max ~1.7e38)
            scoring_rule="brier_proper",
            applied_at="2026-04-17T12:00:00Z",
            decay_half_life_days=30.0,
            reason={},
        )
        value = compute_feedback_value(bogus, value_decimals=6)
        assert value == INT128_MAX

    def test_underflow_clamped_to_int128_min(self) -> None:
        from aragora.reputation.types import ReputationDelta

        bogus = ReputationDelta(
            delta_id="rep_underflow",
            agent_id="alice",
            domain="prediction_market",
            claim_id="clm_x",
            resolution_id="mkt_x",
            delta=float(-(10**35)),
            scoring_rule="brier_proper",
            applied_at="2026-04-17T12:00:00Z",
            decay_half_life_days=30.0,
            reason={},
        )
        value = compute_feedback_value(bogus, value_decimals=6)
        assert value == INT128_MIN
