"""Comprehensive tests for the circuit breaker v2 module.

Tests cover all public components:
- CircuitState enum
- CircuitBreakerOpenError exception
- CircuitBreakerConfig dataclass defaults and custom values
- CircuitBreakerStats and to_dict()
- BaseCircuitBreaker state transitions, can_execute, record_success/failure, reset, get_stats
- Failure rate threshold and sliding window
- State change callbacks
- with_circuit_breaker async decorator
- with_circuit_breaker_sync sync decorator
- Global registry: get_circuit_breaker, reset_all_circuit_breakers, get_all_circuit_breakers
- Backward compatibility: failures property, get_status(), can_proceed()
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from aragora.resilience.circuit_breaker_v2 import (
    BaseCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerStats,
    CircuitState,
    _circuit_breakers,
    get_all_circuit_breakers,
    get_circuit_breaker,
    reset_all_circuit_breakers,
    with_circuit_breaker,
    with_circuit_breaker_sync,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_global_registry():
    """Clear the global circuit breaker registry before and after each test."""
    _circuit_breakers.clear()
    yield
    _circuit_breakers.clear()


@pytest.fixture()
def default_cb() -> BaseCircuitBreaker:
    """Return a BaseCircuitBreaker with default config."""
    return BaseCircuitBreaker("test_cb")


@pytest.fixture()
def low_threshold_cb() -> BaseCircuitBreaker:
    """Return a BaseCircuitBreaker with low thresholds for easy testing."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        cooldown_seconds=10.0,
        half_open_max_requests=2,
    )
    return BaseCircuitBreaker("low_threshold", config)


# ===========================================================================
# 1. CircuitState enum values
# ===========================================================================


class TestCircuitState:
    def test_enum_values(self):
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.OPEN == "open"
        assert CircuitState.HALF_OPEN == "half_open"

    def test_enum_members(self):
        assert set(CircuitState) == {
            CircuitState.CLOSED,
            CircuitState.OPEN,
            CircuitState.HALF_OPEN,
        }

    def test_is_string_subclass(self):
        assert isinstance(CircuitState.CLOSED, str)

    def test_value_attribute(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


# ===========================================================================
# 2. CircuitBreakerOpenError
# ===========================================================================


class TestCircuitBreakerOpenError:
    def test_default_message(self):
        err = CircuitBreakerOpenError()
        assert str(err) == "Circuit breaker is open"
        assert err.circuit_name is None
        assert err.cooldown_remaining is None

    def test_custom_message(self):
        err = CircuitBreakerOpenError("Custom msg")
        assert str(err) == "Custom msg"

    def test_attributes(self):
        err = CircuitBreakerOpenError("open", circuit_name="svc", cooldown_remaining=12.5)
        assert err.circuit_name == "svc"
        assert err.cooldown_remaining == 12.5

    def test_is_exception(self):
        assert issubclass(CircuitBreakerOpenError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            raise CircuitBreakerOpenError("boom", circuit_name="x", cooldown_remaining=5.0)
        assert exc_info.value.circuit_name == "x"
        assert exc_info.value.cooldown_remaining == 5.0


# ===========================================================================
# 3. CircuitBreakerConfig defaults
# ===========================================================================


class TestCircuitBreakerConfig:
    def test_defaults(self):
        cfg = CircuitBreakerConfig()
        assert cfg.failure_threshold == 5
        assert cfg.success_threshold == 3
        assert cfg.cooldown_seconds == 60.0
        assert cfg.half_open_max_requests == 3
        assert cfg.failure_rate_threshold is None
        assert cfg.window_size == 60.0
        assert cfg.excluded_exceptions == ()
        assert cfg.on_state_change is None

    def test_custom_values(self):
        cb_func = MagicMock()
        cfg = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            cooldown_seconds=120.0,
            half_open_max_requests=1,
            failure_rate_threshold=0.8,
            window_size=30.0,
            excluded_exceptions=(ValueError, TypeError),
            on_state_change=cb_func,
        )
        assert cfg.failure_threshold == 10
        assert cfg.success_threshold == 5
        assert cfg.cooldown_seconds == 120.0
        assert cfg.half_open_max_requests == 1
        assert cfg.failure_rate_threshold == 0.8
        assert cfg.window_size == 30.0
        assert cfg.excluded_exceptions == (ValueError, TypeError)
        assert cfg.on_state_change is cb_func


# ===========================================================================
# 4. CircuitBreakerStats.to_dict()
# ===========================================================================


class TestCircuitBreakerStats:
    def test_to_dict_basic(self):
        stats = CircuitBreakerStats(
            state=CircuitState.CLOSED,
            failure_count=2,
            success_count=10,
            last_failure_time=1000.0,
            last_success_time=1001.0,
            consecutive_failures=0,
            consecutive_successes=3,
            total_requests=12,
            total_failures=2,
            cooldown_remaining=None,
        )
        d = stats.to_dict()
        assert d["state"] == "closed"
        assert d["failure_count"] == 2
        assert d["success_count"] == 10
        assert d["last_failure_time"] == 1000.0
        assert d["last_success_time"] == 1001.0
        assert d["consecutive_failures"] == 0
        assert d["consecutive_successes"] == 3
        assert d["total_requests"] == 12
        assert d["total_failures"] == 2
        assert d["cooldown_remaining"] is None

    def test_to_dict_with_cooldown(self):
        stats = CircuitBreakerStats(
            state=CircuitState.OPEN,
            failure_count=5,
            success_count=0,
            last_failure_time=1000.0,
            last_success_time=None,
            consecutive_failures=5,
            consecutive_successes=0,
            total_requests=5,
            total_failures=5,
            cooldown_remaining=42.0,
        )
        d = stats.to_dict()
        assert d["state"] == "open"
        assert d["cooldown_remaining"] == 42.0

    def test_to_dict_returns_all_keys(self):
        stats = CircuitBreakerStats(
            state=CircuitState.HALF_OPEN,
            failure_count=0,
            success_count=0,
            last_failure_time=None,
            last_success_time=None,
            consecutive_failures=0,
            consecutive_successes=0,
            total_requests=0,
            total_failures=0,
        )
        expected_keys = {
            "state",
            "failure_count",
            "success_count",
            "last_failure_time",
            "last_success_time",
            "consecutive_failures",
            "consecutive_successes",
            "total_requests",
            "total_failures",
            "cooldown_remaining",
        }
        assert set(stats.to_dict().keys()) == expected_keys


# ===========================================================================
# 5. BaseCircuitBreaker state transitions
# ===========================================================================


class TestStateTransitions:
    """Test CLOSED->OPEN->HALF_OPEN->CLOSED lifecycle."""

    def test_initial_state_is_closed(self, default_cb: BaseCircuitBreaker):
        assert default_cb.state == CircuitState.CLOSED
        assert default_cb.is_closed is True
        assert default_cb.is_open is False
        assert default_cb.is_half_open is False

    def test_closed_to_open_after_failure_threshold(self, low_threshold_cb: BaseCircuitBreaker):
        """Circuit opens after consecutive failures reach failure_threshold."""
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))

        assert low_threshold_cb.state == CircuitState.OPEN
        assert low_threshold_cb.is_open is True

    def test_open_to_half_open_after_cooldown(self, low_threshold_cb: BaseCircuitBreaker):
        """After cooldown elapses, circuit transitions to HALF_OPEN."""
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))
        assert low_threshold_cb.state == CircuitState.OPEN

        # Simulate time passing beyond cooldown
        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            mock_time.time.return_value = (
                low_threshold_cb._opened_at + low_threshold_cb.config.cooldown_seconds + 1
            )
            assert low_threshold_cb.state == CircuitState.HALF_OPEN
            assert low_threshold_cb.is_half_open is True

    def test_half_open_to_closed_after_success_threshold(
        self, low_threshold_cb: BaseCircuitBreaker
    ):
        """Circuit closes after success_threshold successes in HALF_OPEN."""
        # Open the circuit
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))
        assert low_threshold_cb.state == CircuitState.OPEN

        # Force into HALF_OPEN by simulating cooldown expiry
        opened_at = low_threshold_cb._opened_at
        fake_time = opened_at + low_threshold_cb.config.cooldown_seconds + 1

        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            mock_time.time.return_value = fake_time
            # Trigger state check
            assert low_threshold_cb.can_execute() is True  # transitions to HALF_OPEN

        # Now in HALF_OPEN, record successes (success_threshold=2)
        low_threshold_cb.record_success()
        assert low_threshold_cb.state == CircuitState.HALF_OPEN
        low_threshold_cb.record_success()
        assert low_threshold_cb.state == CircuitState.CLOSED
        assert low_threshold_cb.is_closed is True

    def test_half_open_failure_reopens_circuit(self, low_threshold_cb: BaseCircuitBreaker):
        """A failure in HALF_OPEN reopens the circuit if threshold is met."""
        # Open the circuit
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))

        opened_at = low_threshold_cb._opened_at
        fake_time = opened_at + low_threshold_cb.config.cooldown_seconds + 1

        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            mock_time.time.return_value = fake_time
            assert low_threshold_cb.can_execute() is True  # HALF_OPEN

        # Now consecutive_failures was reset when transitioning to HALF_OPEN?
        # Actually _transition_to HALF_OPEN resets consecutive_successes but not failures.
        # The consecutive_failures remain from the previous OPEN transition.
        # But _transition_to OPEN does not reset consecutive_failures;
        # _transition_to CLOSED does. Let's check actual behavior:
        # When we went CLOSED -> OPEN, consecutive_failures was at 3.
        # When we went OPEN -> HALF_OPEN, consecutive_successes is reset to 0.
        # So consecutive_failures is still 3. Recording one more failure makes it 4,
        # which exceeds threshold 3 -> reopens.
        low_threshold_cb.record_failure(RuntimeError("fail"))
        # The consecutive_failures will be 4 (3 from before + 1), which >= 3
        assert low_threshold_cb.state == CircuitState.OPEN

    def test_not_enough_failures_stays_closed(self, low_threshold_cb: BaseCircuitBreaker):
        """Failures below threshold keep circuit closed."""
        low_threshold_cb.record_failure(RuntimeError("fail"))
        low_threshold_cb.record_failure(RuntimeError("fail"))
        assert low_threshold_cb.state == CircuitState.CLOSED

    def test_success_between_failures_resets_count(self, low_threshold_cb: BaseCircuitBreaker):
        """A success resets consecutive failure count, preventing opening."""
        low_threshold_cb.record_failure(RuntimeError("fail"))
        low_threshold_cb.record_failure(RuntimeError("fail"))
        low_threshold_cb.record_success()  # resets consecutive_failures
        low_threshold_cb.record_failure(RuntimeError("fail"))
        low_threshold_cb.record_failure(RuntimeError("fail"))
        assert low_threshold_cb.state == CircuitState.CLOSED  # never hit 3 consecutive


# ===========================================================================
# 6. can_execute() in each state
# ===========================================================================


class TestCanExecute:
    def test_closed_returns_true(self, default_cb: BaseCircuitBreaker):
        assert default_cb.can_execute() is True

    def test_open_returns_false(self, low_threshold_cb: BaseCircuitBreaker):
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))
        assert low_threshold_cb.can_execute() is False

    def test_half_open_allows_up_to_max_requests(self, low_threshold_cb: BaseCircuitBreaker):
        """HALF_OPEN allows half_open_max_requests (2), then rejects."""
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))

        opened_at = low_threshold_cb._opened_at
        fake_time = opened_at + low_threshold_cb.config.cooldown_seconds + 1

        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            mock_time.time.return_value = fake_time
            # First request transitions to HALF_OPEN and is allowed
            assert low_threshold_cb.can_execute() is True
            # Second request is allowed (half_open_max_requests=2, already used 1)
            assert low_threshold_cb.can_execute() is True
            # Third request should be rejected
            assert low_threshold_cb.can_execute() is False


# ===========================================================================
# 7. record_success()
# ===========================================================================


class TestRecordSuccess:
    def test_resets_consecutive_failures(self, default_cb: BaseCircuitBreaker):
        default_cb.record_failure(RuntimeError("fail"))
        default_cb.record_failure(RuntimeError("fail"))
        assert default_cb.failures == 2
        default_cb.record_success()
        assert default_cb.failures == 0

    def test_increments_success_count(self, default_cb: BaseCircuitBreaker):
        default_cb.record_success()
        default_cb.record_success()
        stats = default_cb.get_stats()
        assert stats.success_count == 2
        assert stats.consecutive_successes == 2

    def test_transitions_half_open_to_closed(self, low_threshold_cb: BaseCircuitBreaker):
        """success_threshold consecutive successes in HALF_OPEN closes the circuit."""
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))

        opened_at = low_threshold_cb._opened_at
        fake_time = opened_at + low_threshold_cb.config.cooldown_seconds + 1

        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            mock_time.time.return_value = fake_time
            low_threshold_cb.can_execute()  # transition to HALF_OPEN

        assert low_threshold_cb.state == CircuitState.HALF_OPEN
        low_threshold_cb.record_success()
        assert low_threshold_cb.state == CircuitState.HALF_OPEN
        low_threshold_cb.record_success()
        assert low_threshold_cb.state == CircuitState.CLOSED

    def test_updates_last_success_time(self, default_cb: BaseCircuitBreaker):
        assert default_cb.get_stats().last_success_time is None
        default_cb.record_success()
        assert default_cb.get_stats().last_success_time is not None


# ===========================================================================
# 8. record_failure()
# ===========================================================================


class TestRecordFailure:
    def test_increments_counters(self, default_cb: BaseCircuitBreaker):
        default_cb.record_failure(RuntimeError("oops"))
        stats = default_cb.get_stats()
        assert stats.failure_count == 1
        assert stats.consecutive_failures == 1
        assert stats.total_failures == 1
        assert stats.total_requests == 1

    def test_opens_circuit_at_threshold(self, low_threshold_cb: BaseCircuitBreaker):
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))
        assert low_threshold_cb.state == CircuitState.OPEN

    def test_excluded_exceptions_not_counted(self):
        cfg = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        cb = BaseCircuitBreaker("excl_test", cfg)

        # ValueError is excluded
        cb.record_failure(ValueError("excluded"))
        assert cb.failures == 0
        stats = cb.get_stats()
        assert stats.failure_count == 0
        assert stats.total_requests == 0

        # RuntimeError is not excluded
        cb.record_failure(RuntimeError("counted"))
        assert cb.failures == 1

    def test_excluded_exception_subclass(self):
        """Subclasses of excluded exceptions are also excluded (isinstance)."""
        cfg = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(OSError,),
        )
        cb = BaseCircuitBreaker("sub_excl", cfg)

        # FileNotFoundError is subclass of OSError
        cb.record_failure(FileNotFoundError("excluded too"))
        assert cb.failures == 0

    def test_none_exception_counts_as_failure(self, default_cb: BaseCircuitBreaker):
        """Passing None as exception still records the failure."""
        default_cb.record_failure(None)
        assert default_cb.failures == 1

    def test_resets_consecutive_successes(self, default_cb: BaseCircuitBreaker):
        default_cb.record_success()
        default_cb.record_success()
        assert default_cb.get_stats().consecutive_successes == 2
        default_cb.record_failure(RuntimeError("fail"))
        assert default_cb.get_stats().consecutive_successes == 0

    def test_updates_last_failure_time(self, default_cb: BaseCircuitBreaker):
        assert default_cb.get_stats().last_failure_time is None
        default_cb.record_failure(RuntimeError("fail"))
        assert default_cb.get_stats().last_failure_time is not None


# ===========================================================================
# 9. Failure rate threshold
# ===========================================================================


class TestFailureRateThreshold:
    def test_opens_circuit_when_rate_exceeded(self):
        """Circuit opens when failure rate >= failure_rate_threshold."""
        cfg = CircuitBreakerConfig(
            failure_threshold=100,  # high so count-based won't trigger
            failure_rate_threshold=0.6,
            window_size=60.0,
        )
        cb = BaseCircuitBreaker("rate_test", cfg)

        # 2 successes, 2 failures => rate = 2/4 = 0.5 < 0.6
        cb.record_success()
        cb.record_success()
        cb.record_failure(RuntimeError("f1"))
        cb.record_failure(RuntimeError("f2"))
        assert cb.state == CircuitState.CLOSED  # rate = 0.5, not >= 0.6

        cb.record_failure(RuntimeError("f3"))
        assert cb.state == CircuitState.OPEN  # rate = 3/5 = 0.6 >= 0.6

    def test_opens_circuit_at_exact_threshold(self):
        """Circuit opens when failure rate equals the threshold (>= comparison)."""
        cfg = CircuitBreakerConfig(
            failure_threshold=100,
            failure_rate_threshold=0.5,
            window_size=60.0,
        )
        cb = BaseCircuitBreaker("rate_exact", cfg)

        cb.record_success()
        cb.record_failure(RuntimeError("f1"))
        # rate = 1/2 = 0.5 >= 0.5 -> opens
        assert cb.state == CircuitState.OPEN

    def test_rate_not_exceeded_stays_closed(self):
        cfg = CircuitBreakerConfig(
            failure_threshold=100,
            failure_rate_threshold=0.8,
            window_size=60.0,
        )
        cb = BaseCircuitBreaker("rate_ok", cfg)

        cb.record_success()
        cb.record_success()
        cb.record_success()
        cb.record_failure(RuntimeError("f"))
        # rate = 1/4 = 0.25 < 0.8
        assert cb.state == CircuitState.CLOSED


# ===========================================================================
# 10. Sliding window pruning
# ===========================================================================


class TestSlidingWindow:
    def test_old_results_pruned(self):
        """Results older than window_size are removed."""
        cfg = CircuitBreakerConfig(
            failure_threshold=100,
            failure_rate_threshold=0.5,
            window_size=10.0,
        )
        cb = BaseCircuitBreaker("window_test", cfg)

        base_time = 1000.0

        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            # Record 3 failures at base_time
            mock_time.time.return_value = base_time
            cb.record_failure(RuntimeError("old1"))
            cb.record_failure(RuntimeError("old2"))
            cb.record_failure(RuntimeError("old3"))
            # rate = 3/3 = 1.0 >= 0.5, but consecutive_failures=3 < 100
            # Rate threshold triggers -> OPEN
            assert cb.state == CircuitState.OPEN

        # Reset for a clean test of pruning
        cb.reset()

        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            # Record 2 failures at base_time
            mock_time.time.return_value = base_time
            cb.record_failure(RuntimeError("old1"))
            cb.record_failure(RuntimeError("old2"))
            # rate = 2/2 = 1.0 >= 0.5 -> OPEN
            assert cb.state == CircuitState.OPEN

        cb.reset()

        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            # Record 1 failure at base_time (old, will be pruned)
            mock_time.time.return_value = base_time
            cb.record_failure(RuntimeError("old"))

            # Now advance time beyond window, record a success
            mock_time.time.return_value = base_time + 11.0  # beyond 10s window
            cb.record_success()

            # The old failure should be pruned. Recent results = [success]
            # rate = 0/1 = 0.0, under threshold
            assert len(cb._recent_results) == 1
            assert cb._recent_results[0][1] is True  # the success


# ===========================================================================
# 11. reset()
# ===========================================================================


class TestReset:
    def test_reset_from_open(self, low_threshold_cb: BaseCircuitBreaker):
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))
        assert low_threshold_cb.state == CircuitState.OPEN

        low_threshold_cb.reset()
        assert low_threshold_cb.state == CircuitState.CLOSED
        assert low_threshold_cb.failures == 0
        stats = low_threshold_cb.get_stats()
        assert stats.failure_count == 0
        assert stats.success_count == 0
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0

    def test_reset_clears_recent_results(self, default_cb: BaseCircuitBreaker):
        default_cb.record_failure(RuntimeError("fail"))
        default_cb.record_success()
        assert len(default_cb._recent_results) > 0
        default_cb.reset()
        assert len(default_cb._recent_results) == 0

    def test_reset_from_closed_no_callback(self):
        """Resetting when already CLOSED should not fire callback."""
        callback = MagicMock()
        cfg = CircuitBreakerConfig(on_state_change=callback)
        cb = BaseCircuitBreaker("reset_closed", cfg)
        cb.reset()
        callback.assert_not_called()

    def test_reset_from_open_fires_callback(self):
        callback = MagicMock()
        cfg = CircuitBreakerConfig(
            failure_threshold=2,
            on_state_change=callback,
        )
        cb = BaseCircuitBreaker("reset_open", cfg)

        cb.record_failure(RuntimeError("f1"))
        cb.record_failure(RuntimeError("f2"))
        # Callback was called for CLOSED -> OPEN
        callback.reset_mock()

        cb.reset()
        callback.assert_called_once_with("reset_open", CircuitState.OPEN, CircuitState.CLOSED)

    def test_reset_clears_opened_at(self, low_threshold_cb: BaseCircuitBreaker):
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))
        assert low_threshold_cb._opened_at is not None
        low_threshold_cb.reset()
        assert low_threshold_cb._opened_at is None


# ===========================================================================
# 12. get_stats() including cooldown_remaining
# ===========================================================================


class TestGetStats:
    def test_initial_stats(self, default_cb: BaseCircuitBreaker):
        stats = default_cb.get_stats()
        assert stats.state == CircuitState.CLOSED
        assert stats.failure_count == 0
        assert stats.success_count == 0
        assert stats.last_failure_time is None
        assert stats.last_success_time is None
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0
        assert stats.total_requests == 0
        assert stats.total_failures == 0
        assert stats.cooldown_remaining is None

    def test_stats_after_operations(self, default_cb: BaseCircuitBreaker):
        default_cb.record_success()
        default_cb.record_failure(RuntimeError("fail"))
        default_cb.record_success()

        stats = default_cb.get_stats()
        assert stats.total_requests == 3
        assert stats.success_count == 2
        assert stats.failure_count == 1
        assert stats.total_failures == 1
        assert stats.consecutive_successes == 1
        assert stats.consecutive_failures == 0

    def test_cooldown_remaining_when_open(self):
        cfg = CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=30.0)
        cb = BaseCircuitBreaker("cd_test", cfg)

        cb.record_failure(RuntimeError("f1"))
        cb.record_failure(RuntimeError("f2"))
        assert cb.state == CircuitState.OPEN

        opened_at = cb._opened_at
        fake_time = opened_at + 10.0  # 10 seconds elapsed

        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            mock_time.time.return_value = fake_time
            stats = cb.get_stats()
            assert stats.cooldown_remaining is not None
            assert abs(stats.cooldown_remaining - 20.0) < 0.1

    def test_cooldown_remaining_none_when_closed(self, default_cb: BaseCircuitBreaker):
        stats = default_cb.get_stats()
        assert stats.cooldown_remaining is None


# ===========================================================================
# 13. State change callback
# ===========================================================================


class TestStateChangeCallback:
    def test_callback_on_open(self):
        callback = MagicMock()
        cfg = CircuitBreakerConfig(
            failure_threshold=2,
            on_state_change=callback,
        )
        cb = BaseCircuitBreaker("cb_callback", cfg)

        cb.record_failure(RuntimeError("f1"))
        cb.record_failure(RuntimeError("f2"))

        callback.assert_called_once_with("cb_callback", CircuitState.CLOSED, CircuitState.OPEN)

    def test_callback_on_half_open(self):
        callback = MagicMock()
        cfg = CircuitBreakerConfig(
            failure_threshold=2,
            cooldown_seconds=5.0,
            on_state_change=callback,
        )
        cb = BaseCircuitBreaker("cb_ho", cfg)

        cb.record_failure(RuntimeError("f1"))
        cb.record_failure(RuntimeError("f2"))
        callback.reset_mock()

        opened_at = cb._opened_at
        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            mock_time.time.return_value = opened_at + 6.0
            _ = cb.state  # triggers transition

        callback.assert_called_once_with("cb_ho", CircuitState.OPEN, CircuitState.HALF_OPEN)

    def test_callback_on_close_from_half_open(self):
        callback = MagicMock()
        cfg = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            cooldown_seconds=5.0,
            on_state_change=callback,
        )
        cb = BaseCircuitBreaker("cb_close", cfg)

        cb.record_failure(RuntimeError("f1"))
        cb.record_failure(RuntimeError("f2"))

        opened_at = cb._opened_at
        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            mock_time.time.return_value = opened_at + 6.0
            cb.can_execute()  # HALF_OPEN

        callback.reset_mock()
        cb.record_success()  # success_threshold=1 -> CLOSED

        callback.assert_called_once_with("cb_close", CircuitState.HALF_OPEN, CircuitState.CLOSED)

    def test_callback_exception_is_caught(self):
        """Callback errors should not propagate."""

        def bad_callback(name, old, new):
            raise RuntimeError("callback error")

        cfg = CircuitBreakerConfig(
            failure_threshold=2,
            on_state_change=bad_callback,
        )
        cb = BaseCircuitBreaker("cb_bad", cfg)

        # Should not raise even though callback throws
        cb.record_failure(RuntimeError("f1"))
        cb.record_failure(RuntimeError("f2"))
        assert cb.state == CircuitState.OPEN


# ===========================================================================
# 14. with_circuit_breaker async decorator
# ===========================================================================


class TestWithCircuitBreakerAsync:
    @pytest.mark.asyncio
    async def test_success_passes_through(self):
        @with_circuit_breaker("async_ok")
        async def good_func():
            return 42

        result = await good_func()
        assert result == 42

    @pytest.mark.asyncio
    async def test_failure_records_and_reraises(self):
        cb = BaseCircuitBreaker("async_fail", CircuitBreakerConfig(failure_threshold=10))

        @with_circuit_breaker("async_fail", circuit_breaker=cb)
        async def bad_func():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await bad_func()

        assert cb.failures == 1

    @pytest.mark.asyncio
    async def test_open_circuit_raises_error(self):
        cfg = CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=60.0)
        cb = BaseCircuitBreaker("async_open", cfg)

        @with_circuit_breaker("async_open", circuit_breaker=cb)
        async def func():
            raise RuntimeError("fail")

        # Trigger failures to open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await func()

        assert cb.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await func()

        assert exc_info.value.circuit_name == "async_open"
        assert exc_info.value.cooldown_remaining is not None

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        @with_circuit_breaker("async_meta")
        async def documented_func():
            """My docstring."""
            return 1

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "My docstring."

    @pytest.mark.asyncio
    async def test_passes_arguments_through(self):
        @with_circuit_breaker("async_args")
        async def add(a, b, c=0):
            return a + b + c

        result = await add(1, 2, c=3)
        assert result == 6

    @pytest.mark.asyncio
    async def test_uses_provided_config(self):
        cfg = CircuitBreakerConfig(failure_threshold=1)

        @with_circuit_breaker("async_cfg", config=cfg)
        async def func():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            await func()

        # After 1 failure, circuit should be open
        with pytest.raises(CircuitBreakerOpenError):
            await func()


# ===========================================================================
# 15. with_circuit_breaker_sync decorator
# ===========================================================================


class TestWithCircuitBreakerSync:
    def test_success_passes_through(self):
        @with_circuit_breaker_sync("sync_ok")
        def good_func():
            return 42

        assert good_func() == 42

    def test_failure_records_and_reraises(self):
        cb = BaseCircuitBreaker("sync_fail", CircuitBreakerConfig(failure_threshold=10))

        @with_circuit_breaker_sync("sync_fail", circuit_breaker=cb)
        def bad_func():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            bad_func()

        assert cb.failures == 1

    def test_open_circuit_raises_error(self):
        cfg = CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=60.0)
        cb = BaseCircuitBreaker("sync_open", cfg)

        @with_circuit_breaker_sync("sync_open", circuit_breaker=cb)
        def func():
            raise RuntimeError("fail")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                func()

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            func()

        assert exc_info.value.circuit_name == "sync_open"
        assert exc_info.value.cooldown_remaining is not None

    def test_preserves_function_metadata(self):
        @with_circuit_breaker_sync("sync_meta")
        def documented_func():
            """My docstring."""
            return 1

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "My docstring."

    def test_passes_arguments_through(self):
        @with_circuit_breaker_sync("sync_args")
        def add(a, b, c=0):
            return a + b + c

        assert add(1, 2, c=3) == 6

    def test_uses_provided_config(self):
        cfg = CircuitBreakerConfig(failure_threshold=1)

        @with_circuit_breaker_sync("sync_cfg", config=cfg)
        def func():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            func()

        with pytest.raises(CircuitBreakerOpenError):
            func()


# ===========================================================================
# 16. Global registry
# ===========================================================================


class TestGlobalRegistry:
    def test_get_circuit_breaker_creates_new(self):
        cb = get_circuit_breaker("new_cb")
        assert isinstance(cb, BaseCircuitBreaker)
        assert cb.name == "new_cb"

    def test_get_circuit_breaker_returns_existing(self):
        cb1 = get_circuit_breaker("shared")
        cb2 = get_circuit_breaker("shared")
        assert cb1 is cb2

    def test_get_circuit_breaker_with_threshold(self):
        cb = get_circuit_breaker("thresh", failure_threshold=10, cooldown_seconds=30.0)
        assert cb.config.failure_threshold == 10
        assert cb.config.cooldown_seconds == 30.0

    def test_get_circuit_breaker_with_config(self):
        cfg = CircuitBreakerConfig(
            failure_threshold=7,
            success_threshold=4,
        )
        cb = get_circuit_breaker("cfg_cb", config=cfg)
        assert cb.config.failure_threshold == 7
        assert cb.config.success_threshold == 4

    def test_get_circuit_breaker_ignores_params_for_existing(self):
        """Once created, subsequent calls with different params return the same instance."""
        cb1 = get_circuit_breaker("fixed", failure_threshold=3)
        cb2 = get_circuit_breaker("fixed", failure_threshold=99)
        assert cb1 is cb2
        assert cb2.config.failure_threshold == 3  # original config preserved

    def test_get_all_circuit_breakers(self):
        get_circuit_breaker("a")
        get_circuit_breaker("b")
        all_cbs = get_all_circuit_breakers()
        assert "a" in all_cbs
        assert "b" in all_cbs
        assert len(all_cbs) == 2

    def test_get_all_returns_copy(self):
        get_circuit_breaker("x")
        all_cbs = get_all_circuit_breakers()
        all_cbs["y"] = BaseCircuitBreaker("y")
        # Original registry should not be affected
        assert "y" not in get_all_circuit_breakers()

    def test_reset_all_circuit_breakers(self):
        cb1 = get_circuit_breaker("r1", failure_threshold=2)
        cb2 = get_circuit_breaker("r2", failure_threshold=2)

        cb1.record_failure(RuntimeError("f1"))
        cb1.record_failure(RuntimeError("f2"))
        cb2.record_failure(RuntimeError("f1"))
        cb2.record_failure(RuntimeError("f2"))

        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

        reset_all_circuit_breakers()

        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED

    def test_reset_all_preserves_registry(self):
        """reset_all resets state but doesn't remove breakers from the registry."""
        get_circuit_breaker("keep_me")
        reset_all_circuit_breakers()
        all_cbs = get_all_circuit_breakers()
        assert "keep_me" in all_cbs

    def test_default_threshold_and_cooldown(self):
        """When no threshold/cooldown passed, defaults are 5 and 60.0."""
        cb = get_circuit_breaker("defaults")
        assert cb.config.failure_threshold == 5
        assert cb.config.cooldown_seconds == 60.0


# ===========================================================================
# 17. Backward compatibility
# ===========================================================================


class TestBackwardCompatibility:
    def test_failures_property(self, default_cb: BaseCircuitBreaker):
        assert default_cb.failures == 0
        default_cb.record_failure(RuntimeError("fail"))
        assert default_cb.failures == 1
        default_cb.record_success()
        assert default_cb.failures == 0

    def test_get_status(self, default_cb: BaseCircuitBreaker):
        assert default_cb.get_status() == "closed"

    def test_get_status_open(self, low_threshold_cb: BaseCircuitBreaker):
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))
        assert low_threshold_cb.get_status() == "open"

    def test_can_proceed_alias(self, default_cb: BaseCircuitBreaker):
        assert default_cb.can_proceed() is True
        assert default_cb.can_proceed() == default_cb.can_execute()

    def test_can_proceed_false_when_open(self, low_threshold_cb: BaseCircuitBreaker):
        for _ in range(3):
            low_threshold_cb.record_failure(RuntimeError("fail"))
        assert low_threshold_cb.can_proceed() is False

    def test_failure_threshold_property(self, default_cb: BaseCircuitBreaker):
        assert default_cb.failure_threshold == 5

    def test_cooldown_seconds_property(self, default_cb: BaseCircuitBreaker):
        assert default_cb.cooldown_seconds == 60.0

    def test_failure_threshold_custom(self):
        cfg = CircuitBreakerConfig(failure_threshold=10, cooldown_seconds=120.0)
        cb = BaseCircuitBreaker("compat", cfg)
        assert cb.failure_threshold == 10
        assert cb.cooldown_seconds == 120.0


# ===========================================================================
# Additional edge case tests
# ===========================================================================


class TestEdgeCases:
    def test_circuit_breaker_name(self, default_cb: BaseCircuitBreaker):
        assert default_cb.name == "test_cb"

    def test_default_config_created_when_none(self):
        cb = BaseCircuitBreaker("no_config")
        assert cb.config is not None
        assert cb.config.failure_threshold == 5

    def test_multiple_rapid_failures_and_successes(self):
        cfg = CircuitBreakerConfig(failure_threshold=3, success_threshold=2)
        cb = BaseCircuitBreaker("rapid", cfg)

        cb.record_failure(RuntimeError("f"))
        cb.record_failure(RuntimeError("f"))
        cb.record_success()
        cb.record_failure(RuntimeError("f"))
        cb.record_failure(RuntimeError("f"))
        cb.record_success()
        # Pattern: F F S F F S -> consecutive never reaches 3
        assert cb.state == CircuitState.CLOSED

    def test_total_requests_tracked(self):
        cb = BaseCircuitBreaker("totals")
        for _ in range(5):
            cb.record_success()
        for _ in range(3):
            cb.record_failure(RuntimeError("fail"))

        stats = cb.get_stats()
        assert stats.total_requests == 8
        assert stats.total_failures == 3

    def test_concurrent_access_does_not_corrupt_state(self):
        """Basic thread-safety test: concurrent operations should not crash."""
        import threading

        cfg = CircuitBreakerConfig(failure_threshold=100)
        cb = BaseCircuitBreaker("thread_test", cfg)
        errors = []

        def record_ops():
            try:
                for _ in range(50):
                    cb.record_success()
                    cb.record_failure(RuntimeError("f"))
                    _ = cb.can_execute()
                    _ = cb.get_stats()
                    _ = cb.state
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_ops) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread-safety errors: {errors}"
        # Just verify we can still read state without error
        assert cb.state in {
            CircuitState.CLOSED,
            CircuitState.OPEN,
            CircuitState.HALF_OPEN,
        }

    def test_get_status_half_open(self):
        cfg = CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=5.0)
        cb = BaseCircuitBreaker("ho_status", cfg)
        cb.record_failure(RuntimeError("f1"))
        cb.record_failure(RuntimeError("f2"))

        opened_at = cb._opened_at
        with patch("aragora.resilience.circuit_breaker_v2.time") as mock_time:
            mock_time.time.return_value = opened_at + 6.0
            assert cb.get_status() == "half_open"
