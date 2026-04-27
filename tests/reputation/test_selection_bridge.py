"""Tests for aragora.reputation.selection_bridge (AGT-05)."""

from __future__ import annotations

import os
import pytest

from aragora.reputation.selection_bridge import (
    NEUTRAL_BRIER,
    ReputationBridgeConfig,
    ReputationCalibrationBridge,
    reputation_flow_enabled,
)
from aragora.reputation.store import ReputationStore
from aragora.reputation.types import (
    DOMAIN_PREDICTION_MARKET,
    DOMAIN_DEBATE_POSITION,
    ReputationDelta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_APPLIED_AT = "2026-04-01T00:00:00Z"


def _make_delta(
    agent_id: str,
    delta: float,
    *,
    domain: str = DOMAIN_PREDICTION_MARKET,
    decay_half_life_days: float | None = None,
    applied_at: str = _APPLIED_AT,
) -> ReputationDelta:
    return ReputationDelta(
        delta_id=f"rep_{agent_id}_{delta}",
        agent_id=agent_id,
        domain=domain,
        claim_id="clm_test",
        resolution_id="res_test",
        delta=delta,
        scoring_rule="brier_proper",
        applied_at=applied_at,
        decay_half_life_days=decay_half_life_days,
        reason={},
    )


def _store_with_deltas(
    agent_id: str, scores: list[float], domain: str = DOMAIN_PREDICTION_MARKET
) -> ReputationStore:
    store = ReputationStore()
    for s in scores:
        store.record_delta(_make_delta(agent_id, s, domain=domain))
    return store


# ---------------------------------------------------------------------------
# Flag tests
# ---------------------------------------------------------------------------


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARAGORA_REPUTATION_FLOW_ENABLED", raising=False)
    assert reputation_flow_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "YES"])
def test_enabled_truthy_values(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", val)
    assert reputation_flow_enabled() is True


def test_disabled_falsy_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "0")
    assert reputation_flow_enabled() is False


# ---------------------------------------------------------------------------
# Bridge disabled → always neutral
# ---------------------------------------------------------------------------


def test_returns_neutral_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARAGORA_REPUTATION_FLOW_ENABLED", raising=False)
    store = _store_with_deltas("agent-a", [10.0] * 10)
    bridge = ReputationCalibrationBridge(store)
    assert bridge.get_brier_score("agent-a") == NEUTRAL_BRIER


def test_returns_neutral_when_store_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    bridge = ReputationCalibrationBridge(store=None)
    assert bridge.get_brier_score("agent-a") == NEUTRAL_BRIER


def test_batch_returns_neutral_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARAGORA_REPUTATION_FLOW_ENABLED", raising=False)
    bridge = ReputationCalibrationBridge()
    result = bridge.get_brier_scores_batch(["a", "b"])
    assert result == {"a": NEUTRAL_BRIER, "b": NEUTRAL_BRIER}


# ---------------------------------------------------------------------------
# min_samples guard
# ---------------------------------------------------------------------------


def test_neutral_when_fewer_than_min_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    cfg = ReputationBridgeConfig(min_samples=5)
    store = _store_with_deltas("agent-b", [10.0, 10.0, 10.0])  # only 3 deltas
    bridge = ReputationCalibrationBridge(store, cfg)
    assert bridge.get_brier_score("agent-b") == NEUTRAL_BRIER


def test_non_neutral_at_exactly_min_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    cfg = ReputationBridgeConfig(min_samples=5, apply_decay=False)
    store = _store_with_deltas("agent-c", [10.0] * 5)
    bridge = ReputationCalibrationBridge(store, cfg)
    score = bridge.get_brier_score("agent-c")
    assert score != NEUTRAL_BRIER


# ---------------------------------------------------------------------------
# Score → Brier mapping
# ---------------------------------------------------------------------------


def test_positive_score_lowers_brier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Good reputation → pseudo-Brier below neutral."""
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    cfg = ReputationBridgeConfig(min_samples=1, score_scale=100.0, apply_decay=False)
    store = _store_with_deltas("agent-d", [50.0])  # sum=50, /scale=0.5, clip to max_shift
    bridge = ReputationCalibrationBridge(store, cfg)
    brier = bridge.get_brier_score("agent-d")
    assert brier < NEUTRAL_BRIER


def test_negative_score_raises_brier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad reputation → pseudo-Brier above neutral."""
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    cfg = ReputationBridgeConfig(min_samples=1, score_scale=100.0, apply_decay=False)
    store = _store_with_deltas("agent-e", [-50.0])
    bridge = ReputationCalibrationBridge(store, cfg)
    brier = bridge.get_brier_score("agent-e")
    assert brier > NEUTRAL_BRIER


def test_zero_score_returns_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    cfg = ReputationBridgeConfig(min_samples=1, apply_decay=False)
    store = _store_with_deltas("agent-f", [0.0])
    bridge = ReputationCalibrationBridge(store, cfg)
    assert bridge.get_brier_score("agent-f") == pytest.approx(NEUTRAL_BRIER)


def test_score_clipped_to_low_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extremely positive score cannot push Brier below low_clip."""
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    cfg = ReputationBridgeConfig(
        min_samples=1, score_scale=1.0, low_clip=0.1, high_clip=0.9, apply_decay=False
    )
    store = _store_with_deltas("agent-g", [10_000.0])  # normalised=10000, way above cap
    bridge = ReputationCalibrationBridge(store, cfg)
    brier = bridge.get_brier_score("agent-g")
    assert brier == pytest.approx(0.1)


def test_score_clipped_to_high_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extremely negative score cannot push Brier above high_clip."""
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    cfg = ReputationBridgeConfig(
        min_samples=1, score_scale=1.0, low_clip=0.1, high_clip=0.9, apply_decay=False
    )
    store = _store_with_deltas("agent-h", [-10_000.0])
    bridge = ReputationCalibrationBridge(store, cfg)
    brier = bridge.get_brier_score("agent-h")
    assert brier == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Domain filtering
# ---------------------------------------------------------------------------


def test_domain_filter_uses_only_matching_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    cfg = ReputationBridgeConfig(min_samples=1, apply_decay=False)
    store = ReputationStore()
    # 5 positive deltas in prediction_market, 5 very negative in debate_position
    for _ in range(5):
        store.record_delta(_make_delta("agent-i", 100.0, domain=DOMAIN_PREDICTION_MARKET))
    for _ in range(5):
        store.record_delta(_make_delta("agent-i", -100.0, domain=DOMAIN_DEBATE_POSITION))
    bridge = ReputationCalibrationBridge(store, cfg)
    brier_pm = bridge.get_brier_score("agent-i", domain=DOMAIN_PREDICTION_MARKET)
    brier_dp = bridge.get_brier_score("agent-i", domain=DOMAIN_DEBATE_POSITION)
    assert brier_pm < NEUTRAL_BRIER
    assert brier_dp > NEUTRAL_BRIER


def test_domain_filter_returns_neutral_below_min_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    cfg = ReputationBridgeConfig(min_samples=3, apply_decay=False)
    store = ReputationStore()
    # Only 2 deltas in prediction_market
    for _ in range(2):
        store.record_delta(_make_delta("agent-j", 50.0, domain=DOMAIN_PREDICTION_MARKET))
    bridge = ReputationCalibrationBridge(store, cfg)
    assert bridge.get_brier_score("agent-j", domain=DOMAIN_PREDICTION_MARKET) == NEUTRAL_BRIER


# ---------------------------------------------------------------------------
# Unknown agent
# ---------------------------------------------------------------------------


def test_unknown_agent_returns_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    bridge = ReputationCalibrationBridge(ReputationStore())
    assert bridge.get_brier_score("agent-unknown") == NEUTRAL_BRIER


# ---------------------------------------------------------------------------
# Batch method
# ---------------------------------------------------------------------------


def test_batch_returns_per_agent_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
    cfg = ReputationBridgeConfig(min_samples=1, apply_decay=False)
    store = ReputationStore()
    store.record_delta(_make_delta("agent-k", 50.0))
    store.record_delta(_make_delta("agent-l", -50.0))
    bridge = ReputationCalibrationBridge(store, cfg)
    result = bridge.get_brier_scores_batch(["agent-k", "agent-l", "agent-unknown"])
    assert result["agent-k"] < NEUTRAL_BRIER
    assert result["agent-l"] > NEUTRAL_BRIER
    assert result["agent-unknown"] == NEUTRAL_BRIER


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_zero_min_samples() -> None:
    with pytest.raises(ValueError, match="min_samples"):
        ReputationBridgeConfig(min_samples=0)


def test_config_rejects_zero_score_scale() -> None:
    with pytest.raises(ValueError, match="score_scale"):
        ReputationBridgeConfig(score_scale=0.0)


def test_config_rejects_invalid_clip_range() -> None:
    with pytest.raises(ValueError, match="low_clip"):
        ReputationBridgeConfig(low_clip=0.5, neutral_brier=0.5, high_clip=0.9)
