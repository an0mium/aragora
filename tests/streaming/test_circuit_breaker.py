"""Tests for aragora.streaming.circuit_breaker."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from aragora.streaming.circuit_breaker import (
    StreamCircuitBreaker,
    StreamCircuitBreakerConfig,
    StreamCircuitState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_breaker(
    failure_threshold: int = 3,
    failure_window_seconds: float = 60.0,
    cooldown_seconds: float = 0.1,
    half_open_success_threshold: int = 2,
    **kwargs,
) -> StreamCircuitBreaker:
    """Create a breaker with short cooldown for fast tests."""
    return StreamCircuitBreaker(
        config=StreamCircuitBreakerConfig(
            failure_threshold=failure_threshold,
            failure_window_seconds=failure_window_seconds,
            cooldown_seconds=cooldown_seconds,
            half_open_success_threshold=half_open_success_threshold,
            **kwargs,
        ),
    )


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------


class TestStreamCircuitBreakerStates:
    """Test the CLOSED -> OPEN -> HALF_OPEN -> CLOSED lifecycle."""

    def test_initial_state_is_closed(self):
        breaker = _make_breaker()
        assert breaker.get_state("debate-1") == StreamCircuitState.CLOSED

    def test_can_send_when_closed(self):
        breaker = _make_breaker()
        assert breaker.can_send("debate-1") is True

    def test_failures_below_threshold_stay_closed(self):
        breaker = _make_breaker(failure_threshold=5)
        for _ in range(4):
            breaker.record_failure("debate-1")
        assert breaker.get_state("debate-1") == StreamCircuitState.CLOSED
        assert breaker.can_send("debate-1") is True

    def test_failures_at_threshold_opens_circuit(self):
        breaker = _make_breaker(failure_threshold=3)
        for i in range(3):
            opened = breaker.record_failure("debate-1")
            if i < 2:
                assert opened is False
        assert opened is True
        assert breaker.get_state("debate-1") == StreamCircuitState.OPEN

    def test_open_circuit_rejects_sends(self):
        breaker = _make_breaker(failure_threshold=2, cooldown_seconds=100.0)
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        assert breaker.can_send("d1") is False

    def test_open_transitions_to_half_open_after_cooldown(self):
        breaker = _make_breaker(failure_threshold=2, cooldown_seconds=0.05)
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        assert breaker.get_state("d1") == StreamCircuitState.OPEN
        time.sleep(0.06)
        assert breaker.get_state("d1") == StreamCircuitState.HALF_OPEN

    def test_half_open_allows_trial_sends(self):
        breaker = _make_breaker(failure_threshold=2, cooldown_seconds=0.05)
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        time.sleep(0.06)
        assert breaker.can_send("d1") is True

    def test_half_open_closes_after_enough_successes(self):
        breaker = _make_breaker(
            failure_threshold=2,
            cooldown_seconds=0.05,
            half_open_success_threshold=2,
        )
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        time.sleep(0.06)
        # Trial sends
        assert breaker.can_send("d1") is True
        breaker.record_success("d1")
        assert breaker.get_state("d1") == StreamCircuitState.HALF_OPEN
        breaker.record_success("d1")
        assert breaker.get_state("d1") == StreamCircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        breaker = _make_breaker(
            failure_threshold=2,
            cooldown_seconds=0.05,
        )
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        time.sleep(0.06)
        breaker.can_send("d1")  # Triggers half-open
        breaker.record_failure("d1")
        assert breaker.get_state("d1") == StreamCircuitState.OPEN

    def test_success_in_closed_clears_failure_timestamps(self):
        breaker = _make_breaker(failure_threshold=5)
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        breaker.record_success("d1")
        stats = breaker.get_stats("d1")
        assert stats["recent_failures"] == 0


# ---------------------------------------------------------------------------
# Per-debate isolation
# ---------------------------------------------------------------------------


class TestStreamCircuitBreakerIsolation:
    """Test that circuits are isolated per debate."""

    def test_different_debates_are_independent(self):
        breaker = _make_breaker(failure_threshold=2, cooldown_seconds=100.0)
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        assert breaker.can_send("d1") is False
        assert breaker.can_send("d2") is True

    def test_failure_in_one_debate_does_not_affect_another(self):
        breaker = _make_breaker(failure_threshold=2)
        for _ in range(10):
            breaker.record_failure("d1")
        assert breaker.get_state("d2") == StreamCircuitState.CLOSED
        assert breaker.get_stats("d2")["total_failures"] == 0

    def test_get_all_stats_returns_all_debates(self):
        breaker = _make_breaker()
        breaker.record_failure("d1")
        breaker.record_success("d2")
        all_stats = breaker.get_all_stats()
        assert "d1" in all_stats
        assert "d2" in all_stats


# ---------------------------------------------------------------------------
# Metrics emission
# ---------------------------------------------------------------------------


class TestStreamCircuitBreakerMetrics:
    """Test metrics emission on state transitions."""

    def test_metrics_callback_called_on_open(self):
        cb = MagicMock()
        breaker = StreamCircuitBreaker(
            config=StreamCircuitBreakerConfig(
                failure_threshold=2,
                cooldown_seconds=100.0,
            ),
            metrics_callback=cb,
        )
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        cb.assert_called()
        args = cb.call_args[0]
        assert args[0] == "d1"
        assert args[1] == StreamCircuitState.CLOSED
        assert args[2] == StreamCircuitState.OPEN

    def test_metrics_callback_called_on_close(self):
        cb = MagicMock()
        breaker = StreamCircuitBreaker(
            config=StreamCircuitBreakerConfig(
                failure_threshold=2,
                cooldown_seconds=0.05,
                half_open_success_threshold=1,
            ),
            metrics_callback=cb,
        )
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        time.sleep(0.06)
        breaker.can_send("d1")  # Triggers half-open
        breaker.record_success("d1")
        # Should have transitions: closed->open, open->half_open, half_open->closed
        assert cb.call_count >= 3

    def test_metrics_callback_error_does_not_break_breaker(self):
        cb = MagicMock(side_effect=RuntimeError("boom"))
        breaker = StreamCircuitBreaker(
            config=StreamCircuitBreakerConfig(failure_threshold=1),
            metrics_callback=cb,
        )
        # Should not raise
        breaker.record_failure("d1")
        assert breaker.get_state("d1") == StreamCircuitState.OPEN


# ---------------------------------------------------------------------------
# Reset and removal
# ---------------------------------------------------------------------------


class TestStreamCircuitBreakerReset:
    """Test reset and removal operations."""

    def test_reset_single_debate(self):
        breaker = _make_breaker(failure_threshold=2, cooldown_seconds=100.0)
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        breaker.reset("d1")
        assert breaker.get_state("d1") == StreamCircuitState.CLOSED

    def test_reset_all(self):
        breaker = _make_breaker(failure_threshold=2, cooldown_seconds=100.0)
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        breaker.record_failure("d2")
        breaker.record_failure("d2")
        breaker.reset()
        assert breaker.get_state("d1") == StreamCircuitState.CLOSED
        assert breaker.get_state("d2") == StreamCircuitState.CLOSED

    def test_remove_debate(self):
        breaker = _make_breaker()
        breaker.record_failure("d1")
        breaker.remove("d1")
        stats = breaker.get_stats("d1")
        assert stats["total_failures"] == 0


# ---------------------------------------------------------------------------
# Stats and config
# ---------------------------------------------------------------------------


class TestStreamCircuitBreakerStats:
    """Test stats and config access."""

    def test_stats_track_totals(self):
        breaker = _make_breaker(failure_threshold=10)
        for _ in range(3):
            breaker.record_failure("d1")
        for _ in range(2):
            breaker.record_success("d1")
        stats = breaker.get_stats("d1")
        assert stats["total_failures"] == 3
        assert stats["total_successes"] == 2
        assert stats["state"] == "closed"

    def test_config_accessible(self):
        config = StreamCircuitBreakerConfig(failure_threshold=7)
        breaker = StreamCircuitBreaker(config=config)
        assert breaker.config.failure_threshold == 7

    def test_rejected_count_incremented(self):
        breaker = _make_breaker(failure_threshold=1, cooldown_seconds=100.0)
        breaker.record_failure("d1")
        breaker.can_send("d1")  # rejected
        breaker.can_send("d1")  # rejected
        stats = breaker.get_stats("d1")
        assert stats["total_rejected"] == 2


# ---------------------------------------------------------------------------
# Windowed failure counting
# ---------------------------------------------------------------------------


class TestStreamCircuitBreakerWindow:
    """Test that failures outside the window are not counted."""

    def test_old_failures_are_pruned(self):
        breaker = _make_breaker(
            failure_threshold=3,
            failure_window_seconds=0.05,
        )
        breaker.record_failure("d1")
        breaker.record_failure("d1")
        time.sleep(0.06)
        # Old failures should have been pruned, third failure should not trip
        breaker.record_failure("d1")
        assert breaker.get_state("d1") == StreamCircuitState.CLOSED
