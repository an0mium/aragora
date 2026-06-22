"""Tests for aragora.metrics.viah_signals (AGT-06 SD-2 / SD-3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aragora.metrics.viah_signals import (
    DEFAULT_BRIER_THRESHOLD,
    count_crux_resolutions_correct,
    count_predictions_above_brier_threshold,
)
from aragora.reputation.store import ReputationStore
from aragora.reputation.types import (
    DOMAIN_CRUX_RESOLUTION,
    DOMAIN_PREDICTION_MARKET,
    ReputationDelta,
)

_FLAG = "ARAGORA_VIAH_TREND_ENABLED"
_NOW = datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC)
_RECENT = (_NOW - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
_OLD = (_NOW - timedelta(hours=200)).isoformat().replace("+00:00", "Z")


def _crux_delta(agent: str = "a", delta: float = 1.0, at: str = _RECENT) -> ReputationDelta:
    return ReputationDelta(
        delta_id=f"rep_crux_{agent}_{at[:10]}",
        agent_id=agent,
        domain=DOMAIN_CRUX_RESOLUTION,
        claim_id=f"c_{agent}",
        resolution_id=f"r_{agent}",
        delta=delta,
        scoring_rule="binary",
        applied_at=at,
        decay_half_life_days=30.0,
        reason={"correct": delta > 0},
    )


def _mkt_delta(agent: str = "a", brier: float = 0.1, at: str = _RECENT) -> ReputationDelta:
    stake, pf = 10, 1.0 - 2.0 * brier
    return ReputationDelta(
        delta_id=f"rep_mkt_{agent}_{at[:10]}",
        agent_id=agent,
        domain=DOMAIN_PREDICTION_MARKET,
        claim_id=f"cm_{agent}",
        resolution_id=f"rm_{agent}",
        delta=stake * pf,
        scoring_rule="brier_proper",
        applied_at=at,
        decay_half_life_days=30.0,
        reason={"brier": brier, "payout_fraction": pf, "stake_units": stake},
    )


# --- Flag gating (both functions) ---


def test_flag_off_crux_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    s = ReputationStore()
    s.record_delta(_crux_delta(delta=5.0))
    assert count_crux_resolutions_correct(s, now=_NOW) == 0


def test_flag_off_pred_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    s = ReputationStore()
    s.record_delta(_mkt_delta(brier=0.01))
    assert count_predictions_above_brier_threshold(s, now=_NOW) == 0


@pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
def test_truthy_flag_enables_crux(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv(_FLAG, val)
    s = ReputationStore()
    s.record_delta(_crux_delta(delta=2.0))
    assert count_crux_resolutions_correct(s, now=_NOW) == 1


# --- count_crux_resolutions_correct ---


def test_crux_empty_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    assert count_crux_resolutions_correct(ReputationStore(), now=_NOW) == 0


def test_crux_positive_delta_in_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    s = ReputationStore()
    s.record_delta(_crux_delta(delta=1.0))
    assert count_crux_resolutions_correct(s, now=_NOW) == 1


@pytest.mark.parametrize("delta", [-1.0, 0.0])
def test_crux_non_positive_delta_not_counted(monkeypatch: pytest.MonkeyPatch, delta: float) -> None:
    monkeypatch.setenv(_FLAG, "1")
    s = ReputationStore()
    s.record_delta(_crux_delta(delta=delta))
    assert count_crux_resolutions_correct(s, now=_NOW) == 0


def test_crux_old_delta_not_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    s = ReputationStore()
    s.record_delta(_crux_delta(delta=3.0, at=_OLD))
    assert count_crux_resolutions_correct(s, window_hours=168.0, now=_NOW) == 0


def test_crux_multiple_agents_and_window_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    s = ReputationStore()
    for agent in ["a", "b", "c"]:
        s.record_delta(_crux_delta(agent=agent, delta=1.0, at=_RECENT))
    s.record_delta(_crux_delta(agent="d", delta=-1.0, at=_RECENT))
    s.record_delta(_crux_delta(agent="e", delta=2.0, at=_OLD))
    assert count_crux_resolutions_correct(s, window_hours=168.0, now=_NOW) == 3


def test_crux_ignores_prediction_market_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    s = ReputationStore()
    s.record_delta(_mkt_delta(brier=0.01))
    assert count_crux_resolutions_correct(s, now=_NOW) == 0


# --- count_predictions_above_brier_threshold ---


def test_pred_empty_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    assert count_predictions_above_brier_threshold(ReputationStore(), now=_NOW) == 0


def test_pred_good_brier_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    s = ReputationStore()
    s.record_delta(_mkt_delta(brier=DEFAULT_BRIER_THRESHOLD))
    assert count_predictions_above_brier_threshold(s, now=_NOW) == 1


def test_pred_brier_above_threshold_not_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    s = ReputationStore()
    s.record_delta(_mkt_delta(brier=0.40))
    assert count_predictions_above_brier_threshold(s, now=_NOW) == 0


def test_pred_old_delta_not_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    s = ReputationStore()
    s.record_delta(_mkt_delta(brier=0.01, at=_OLD))
    assert count_predictions_above_brier_threshold(s, window_hours=168.0, now=_NOW) == 0


def test_pred_no_brier_key_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    s = ReputationStore()
    s.record_delta(
        ReputationDelta(
            delta_id="rep_nob",
            agent_id="x",
            domain=DOMAIN_PREDICTION_MARKET,
            claim_id="c1",
            resolution_id="r1",
            delta=5.0,
            scoring_rule="binary",
            applied_at=_RECENT,
            decay_half_life_days=None,
            reason={"correct": True},
        )
    )
    assert count_predictions_above_brier_threshold(s, now=_NOW) == 0


def test_pred_custom_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    s = ReputationStore()
    s.record_delta(_mkt_delta(agent="a", brier=0.10))
    s.record_delta(_mkt_delta(agent="b", brier=0.30))
    assert count_predictions_above_brier_threshold(s, brier_threshold=0.15, now=_NOW) == 1
    assert count_predictions_above_brier_threshold(s, brier_threshold=0.35, now=_NOW) == 2


def test_pred_invalid_threshold_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    with pytest.raises(ValueError, match="brier_threshold"):
        count_predictions_above_brier_threshold(ReputationStore(), brier_threshold=1.5, now=_NOW)


def test_pred_ignores_crux_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "1")
    s = ReputationStore()
    s.record_delta(_crux_delta(delta=5.0))
    assert count_predictions_above_brier_threshold(s, now=_NOW) == 0
