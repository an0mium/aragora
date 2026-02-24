"""Tests for aragora.streaming.reconnection."""

from __future__ import annotations

import time

import pytest

from aragora.streaming.reconnection import (
    ConnectionQualityScore,
    ReconnectionConfig,
    ReconnectionContext,
    ReconnectionManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> ReconnectionConfig:
    defaults = {
        "initial_delay": 0.01,
        "max_delay": 0.1,
        "max_attempts": 5,
        "backoff_factor": 2.0,
        "jitter_factor": 0.0,  # Deterministic for tests
    }
    defaults.update(kwargs)
    return ReconnectionConfig(**defaults)


# ---------------------------------------------------------------------------
# ReconnectionContext
# ---------------------------------------------------------------------------


class TestReconnectionContext:
    """Test reconnection context delay calculation and state tracking."""

    def test_initial_state(self):
        cfg = _make_config()
        ctx = ReconnectionContext("d1", cfg)
        assert ctx.debate_id == "d1"
        assert ctx.attempt == 0
        assert ctx.exhausted is False
        assert ctx.connected is False
        assert ctx.last_seen_seq == 0

    def test_next_delay_exponential_backoff(self):
        cfg = _make_config(
            initial_delay=1.0,
            max_delay=100.0,
            backoff_factor=2.0,
            jitter_factor=0.0,
        )
        ctx = ReconnectionContext("d1", cfg)
        assert ctx.next_delay() == pytest.approx(1.0)   # attempt 0: 1*2^0=1
        assert ctx.next_delay() == pytest.approx(2.0)   # attempt 1: 1*2^1=2
        assert ctx.next_delay() == pytest.approx(4.0)   # attempt 2: 1*2^2=4
        assert ctx.next_delay() == pytest.approx(8.0)   # attempt 3: 1*2^3=8

    def test_next_delay_capped_at_max(self):
        cfg = _make_config(
            initial_delay=1.0,
            max_delay=3.0,
            backoff_factor=2.0,
            jitter_factor=0.0,
            max_attempts=10,
        )
        ctx = ReconnectionContext("d1", cfg)
        ctx.next_delay()  # 1
        ctx.next_delay()  # 2
        delay = ctx.next_delay()  # Would be 4 but capped at 3
        assert delay == pytest.approx(3.0)

    def test_next_delay_with_jitter(self):
        cfg = _make_config(
            initial_delay=1.0,
            max_delay=100.0,
            backoff_factor=2.0,
            jitter_factor=0.2,
        )
        ctx = ReconnectionContext("d1", cfg)
        delays = [ctx.next_delay() for _ in range(3)]
        # With 20% jitter on 1.0, should be in [0.8, 1.2]
        assert 0.5 <= delays[0] <= 1.5

    def test_exhausted_after_max_attempts(self):
        cfg = _make_config(max_attempts=3)
        ctx = ReconnectionContext("d1", cfg)
        for _ in range(3):
            ctx.next_delay()
        assert ctx.exhausted is True

    def test_on_connected(self):
        cfg = _make_config()
        ctx = ReconnectionContext("d1", cfg)
        ctx.next_delay()  # attempt 1
        ctx.on_connected()
        assert ctx.connected is True

    def test_on_failure(self):
        cfg = _make_config()
        ctx = ReconnectionContext("d1", cfg)
        ctx.next_delay()
        ctx.on_failure()
        assert ctx.connected is False

    def test_on_disconnected_resets_attempt_counter(self):
        cfg = _make_config()
        ctx = ReconnectionContext("d1", cfg)
        ctx.next_delay()  # attempt becomes 1
        ctx.on_connected()
        ctx.on_disconnected()
        assert ctx.connected is False
        assert ctx.attempt == 0

    def test_last_seen_seq_tracks_max(self):
        cfg = _make_config()
        ctx = ReconnectionContext("d1", cfg)
        ctx.last_seen_seq = 10
        ctx.last_seen_seq = 5  # Should not decrease
        assert ctx.last_seen_seq == 10
        ctx.last_seen_seq = 15
        assert ctx.last_seen_seq == 15

    def test_reconnect_count_incremented(self):
        cfg = _make_config()
        ctx = ReconnectionContext("d1", cfg)
        ctx.next_delay()
        ctx.next_delay()  # attempt > 1
        ctx.on_connected()
        assert ctx.reconnect_count == 1


# ---------------------------------------------------------------------------
# ConnectionQualityScore
# ---------------------------------------------------------------------------


class TestConnectionQualityScore:
    """Test quality score computation."""

    def test_perfect_quality_score(self):
        cfg = _make_config()
        ctx = ReconnectionContext("d1", cfg)
        ctx.on_connected()
        time.sleep(0.02)
        score = ctx.get_quality_score()
        assert score.score > 0.9
        assert score.reconnect_penalty == 0.0

    def test_quality_degrades_with_reconnects(self):
        cfg = _make_config()
        ctx = ReconnectionContext("d1", cfg)
        ctx.on_connected()
        # Simulate many reconnects
        ctx._reconnect_count = 10
        score = ctx.get_quality_score()
        assert score.reconnect_penalty == 1.0  # 10/10 = 1.0

    def test_quality_degrades_with_high_latency(self):
        cfg = _make_config()
        ctx = ReconnectionContext("d1", cfg)
        ctx.on_connected()
        for _ in range(20):
            ctx.record_latency(1500.0)  # High latency
        score = ctx.get_quality_score()
        assert score.latency_penalty > 0.5

    def test_quality_score_to_dict(self):
        cfg = _make_config()
        ctx = ReconnectionContext("d1", cfg)
        ctx.on_connected()
        d = ctx.get_quality_score().to_dict()
        assert "score" in d
        assert "reconnect_penalty" in d
        assert "latency_penalty" in d
        assert "uptime_ratio" in d

    def test_context_to_dict(self):
        cfg = _make_config()
        ctx = ReconnectionContext("d1", cfg)
        d = ctx.to_dict()
        assert d["debate_id"] == "d1"
        assert "quality" in d


# ---------------------------------------------------------------------------
# ReconnectionManager
# ---------------------------------------------------------------------------


class TestReconnectionManager:
    """Test the reconnection manager."""

    def test_create_context(self):
        mgr = ReconnectionManager(config=_make_config())
        ctx = mgr.create_context("d1")
        assert ctx.debate_id == "d1"

    def test_create_context_returns_same_for_same_debate(self):
        mgr = ReconnectionManager(config=_make_config())
        ctx1 = mgr.create_context("d1")
        ctx2 = mgr.create_context("d1")
        assert ctx1 is ctx2

    def test_get_context_returns_none_for_unknown(self):
        mgr = ReconnectionManager(config=_make_config())
        assert mgr.get_context("d1") is None

    def test_remove_context(self):
        mgr = ReconnectionManager(config=_make_config())
        mgr.create_context("d1")
        mgr.remove_context("d1")
        assert mgr.get_context("d1") is None

    def test_get_all_quality_scores(self):
        mgr = ReconnectionManager(config=_make_config())
        mgr.create_context("d1").on_connected()
        mgr.create_context("d2").on_connected()
        scores = mgr.get_all_quality_scores()
        assert "d1" in scores
        assert "d2" in scores

    def test_get_summary(self):
        mgr = ReconnectionManager(config=_make_config())
        mgr.create_context("d1").on_connected()
        summary = mgr.get_summary()
        assert summary["active_contexts"] == 1
        assert summary["connected_count"] == 1
        assert summary["exhausted_count"] == 0

    def test_clear(self):
        mgr = ReconnectionManager(config=_make_config())
        mgr.create_context("d1")
        mgr.create_context("d2")
        mgr.clear()
        assert mgr.get_summary()["active_contexts"] == 0

    def test_config_accessible(self):
        cfg = _make_config(max_attempts=20)
        mgr = ReconnectionManager(config=cfg)
        assert mgr.config.max_attempts == 20
