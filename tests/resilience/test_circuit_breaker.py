"""
Tests for the core CircuitBreaker implementation.

Tests cover:
- Single-entity mode: record_failure, record_success, can_proceed
- Multi-entity mode: is_available, filter_available_entities
- State transitions: closed -> open -> half-open -> closed
- Cooldown behavior and timing
- from_config and from_dict serialization
- protected_call async context manager
- protected_call_sync sync context manager
- CircuitOpenError
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from aragora.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    _set_metrics_callback,
)
from aragora.resilience_config import CircuitBreakerConfig


# ============================================================================
# CircuitOpenError Tests
# ============================================================================


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_error_message(self):
        """Test error message includes circuit name and cooldown."""
        err = CircuitOpenError("my-circuit", 15.3)
        assert "my-circuit" in str(err)
        assert "15.3" in str(err)

    def test_attributes(self):
        """Test error attributes are set correctly."""
        err = CircuitOpenError("api-call", 42.5)
        assert err.circuit_name == "api-call"
        assert err.cooldown_remaining == 42.5

    def test_is_exception(self):
        """Test CircuitOpenError is an Exception subclass."""
        assert issubclass(CircuitOpenError, Exception)


# ============================================================================
# CircuitBreaker Defaults Tests
# ============================================================================


class TestCircuitBreakerDefaults:
    """Tests for CircuitBreaker default values."""

    def test_default_name(self):
        cb = CircuitBreaker()
        assert cb.name == "default"

    def test_default_failure_threshold(self):
        cb = CircuitBreaker()
        assert cb.failure_threshold == 3

    def test_default_cooldown_seconds(self):
        cb = CircuitBreaker()
        assert cb.cooldown_seconds == 60.0

    def test_default_half_open_success_threshold(self):
        cb = CircuitBreaker()
        assert cb.half_open_success_threshold == 2

    def test_default_half_open_max_calls(self):
        cb = CircuitBreaker()
        assert cb.half_open_max_calls == 3

    def test_default_state_closed(self):
        cb = CircuitBreaker()
        assert cb.get_status() == "closed"
        assert not cb.is_open

    def test_default_failures_zero(self):
        cb = CircuitBreaker()
        assert cb.failures == 0

    def test_recovery_timeout_alias(self):
        """Test recovery_timeout sets cooldown_seconds."""
        cb = CircuitBreaker(recovery_timeout=120.0)
        assert cb.cooldown_seconds == 120.0
        assert cb.reset_timeout == 120.0


# ============================================================================
# Single-Entity Mode Tests
# ============================================================================


class TestSingleEntityMode:
    """Tests for single-entity mode circuit breaker operations."""

    def test_record_failure_below_threshold(self):
        """Failures below threshold do not open circuit."""
        cb = CircuitBreaker(failure_threshold=3)
        result = cb.record_failure()
        assert result is False
        assert cb.failures == 1
        assert cb.get_status() == "closed"

    def test_record_failure_reaches_threshold(self):
        """Reaching failure threshold opens circuit."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        result = cb.record_failure()  # 3rd failure - threshold met
        assert result is True
        assert cb.is_open
        assert cb.get_status() == "open"

    def test_record_failure_already_open(self):
        """Additional failures after opening don't re-trigger."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()  # Opens
        result = cb.record_failure()  # Already open
        assert result is False

    def test_record_success_resets_failures(self):
        """Success resets failure count when circuit is closed."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 2
        cb.record_success()
        assert cb.failures == 0

    def test_record_success_closes_open_circuit(self):
        """Success closes an open circuit."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        cb.record_success()
        assert not cb.is_open
        assert cb.get_status() == "closed"
        assert cb.failures == 0

    def test_can_proceed_when_closed(self):
        """can_proceed returns True when circuit is closed."""
        cb = CircuitBreaker()
        assert cb.can_proceed() is True

    def test_can_proceed_when_open(self):
        """can_proceed returns False when circuit is open."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.can_proceed() is False

    def test_can_proceed_after_cooldown(self):
        """can_proceed returns True after cooldown expires (resets circuit)."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.can_proceed() is False
        time.sleep(0.02)
        assert cb.can_proceed() is True
        assert cb.get_status() == "closed"

    def test_cooldown_remaining(self):
        """Test cooldown_remaining returns correct value."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)
        assert cb.cooldown_remaining() == 0.0  # Not open

        cb.record_failure()
        cb.record_failure()
        remaining = cb.cooldown_remaining()
        assert 55.0 < remaining <= 60.0  # Just opened

    def test_cooldown_remaining_expired(self):
        """Test cooldown_remaining returns 0 after expiry."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.cooldown_remaining() == 0.0

    def test_is_open_setter(self):
        """Test is_open setter manually opens/closes circuit."""
        cb = CircuitBreaker()
        cb.is_open = True
        assert cb.is_open
        assert cb.get_status() == "open"

        cb.is_open = False
        assert not cb.is_open
        assert cb.get_status() == "closed"
        assert cb.failures == 0


# ============================================================================
# Multi-Entity Mode Tests
# ============================================================================


class TestMultiEntityMode:
    """Tests for multi-entity mode circuit breaker operations."""

    def test_entity_failure_below_threshold(self):
        """Entity failures below threshold do not open circuit."""
        cb = CircuitBreaker(failure_threshold=3)
        result = cb.record_failure("agent-1")
        assert result is False

    def test_entity_failure_reaches_threshold(self):
        """Entity reaching failure threshold opens its circuit."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("agent-1")
        result = cb.record_failure("agent-1")
        assert result is True
        assert not cb.is_available("agent-1")

    def test_different_entities_independent(self):
        """Different entities have independent failure counts."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")  # Opens agent-1
        assert not cb.is_available("agent-1")
        assert cb.is_available("agent-2")  # agent-2 still available

    def test_entity_success_in_half_open(self):
        """Entity success in half-open state works toward closing."""
        cb = CircuitBreaker(
            failure_threshold=2, cooldown_seconds=0.01, half_open_success_threshold=2
        )
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")  # Opens

        time.sleep(0.02)  # Wait for cooldown
        assert cb.is_available("agent-1")  # Half-open

        cb.record_success("agent-1")
        cb.record_success("agent-1")  # Should close
        assert cb.is_available("agent-1")
        assert cb.get_status("agent-1") == "closed"

    def test_entity_success_resets_failures(self):
        """Entity success resets failure count when circuit is closed."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("agent-1")
        assert cb._failures.get("agent-1", 0) == 1
        cb.record_success("agent-1")
        assert cb._failures.get("agent-1", 0) == 0

    def test_entity_status(self):
        """Test get_status for entity."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.01)
        assert cb.get_status("agent-1") == "closed"

        cb.record_failure("agent-1")
        cb.record_failure("agent-1")
        assert cb.get_status("agent-1") == "open"

        time.sleep(0.02)
        assert cb.get_status("agent-1") == "half-open"

    def test_filter_available_entities(self):
        """Test filtering available entities."""
        cb = CircuitBreaker(failure_threshold=2)

        class MockAgent:
            def __init__(self, name):
                self.name = name

        agents = [MockAgent("a"), MockAgent("b"), MockAgent("c")]
        cb.record_failure("a")
        cb.record_failure("a")  # Opens 'a'

        available = cb.filter_available_entities(agents)
        assert len(available) == 2
        assert all(a.name != "a" for a in available)

    def test_filter_available_agents_alias(self):
        """Test filter_available_agents is an alias."""
        cb = CircuitBreaker(failure_threshold=2)

        class MockAgent:
            def __init__(self, name):
                self.name = name

        agents = [MockAgent("x")]
        result = cb.filter_available_agents(agents)
        assert len(result) == 1

    def test_cooldown_remaining_entity(self):
        """Test cooldown_remaining for entity."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)
        assert cb.cooldown_remaining("agent-1") == 0.0

        cb.record_failure("agent-1")
        cb.record_failure("agent-1")
        remaining = cb.cooldown_remaining("agent-1")
        assert 55.0 < remaining <= 60.0

    def test_get_all_status(self):
        """Test get_all_status for all tracked entities."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")  # Opens
        cb.record_failure("agent-2")  # 1 failure only

        status = cb.get_all_status()
        assert "agent-1" in status
        assert "agent-2" in status
        assert status["agent-1"]["status"] == "open"
        assert status["agent-2"]["status"] == "closed"


# ============================================================================
# State Transition Tests
# ============================================================================


class TestStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_closed_to_open(self):
        """Test transition from closed to open on threshold failure."""
        cb = CircuitBreaker(failure_threshold=2)
        assert cb.state == "closed"
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

    def test_open_to_half_open(self):
        """Test transition from open to half-open after cooldown."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.02)
        assert cb.state == "half-open"

    def test_half_open_to_closed_on_success(self):
        """Test transition from half-open to closed on success."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == "half-open"
        cb.record_success()
        assert cb.state == "closed"


# ============================================================================
# Serialization Tests
# ============================================================================


class TestSerialization:
    """Tests for CircuitBreaker serialization (to_dict/from_dict)."""

    def test_to_dict_default(self):
        """Test to_dict with default state."""
        cb = CircuitBreaker()
        d = cb.to_dict()
        assert "config" in d
        assert "single_mode" in d
        assert "entity_mode" in d
        assert d["config"]["failure_threshold"] == 3
        assert d["config"]["cooldown_seconds"] == 60.0
        assert d["single_mode"]["is_open"] is False

    def test_to_dict_with_failures(self):
        """Test to_dict captures failure state."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        d = cb.to_dict()
        assert d["single_mode"]["failures"] == 2

    def test_to_dict_open_circuit(self):
        """Test to_dict captures open state."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        d = cb.to_dict()
        assert d["single_mode"]["is_open"] is True
        assert d["single_mode"]["open_for_seconds"] > 0

    def test_to_dict_entity_mode(self):
        """Test to_dict captures entity failures."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")
        d = cb.to_dict()
        assert d["entity_mode"]["failures"]["agent-1"] == 2

    def test_from_dict_round_trip(self):
        """Test from_dict restores state from to_dict."""
        original = CircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)
        original.record_failure()
        original.record_failure()
        original.record_failure("entity-x")

        data = original.to_dict()
        restored = CircuitBreaker.from_dict(
            data,
            failure_threshold=5,
            cooldown_seconds=30.0,
        )

        assert restored._single_failures == 2
        assert restored._failures.get("entity-x", 0) == 1

    def test_from_dict_open_circuit(self):
        """Test from_dict restores open circuit within cooldown."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=120.0)
        cb.record_failure()
        cb.record_failure()
        data = cb.to_dict()

        restored = CircuitBreaker.from_dict(
            data,
            failure_threshold=2,
            cooldown_seconds=120.0,
        )
        assert restored._single_open_at > 0  # Still open


# ============================================================================
# from_config Tests
# ============================================================================


class TestFromConfig:
    """Tests for CircuitBreaker.from_config class method."""

    def test_from_config_basic(self):
        """Test from_config creates circuit breaker from config."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            timeout_seconds=120.0,
            success_threshold=3,
            half_open_max_calls=5,
        )
        cb = CircuitBreaker.from_config(config, name="test-service")
        assert cb.name == "test-service"
        assert cb.failure_threshold == 10
        assert cb.cooldown_seconds == 120.0
        assert cb.half_open_success_threshold == 3
        assert cb.half_open_max_calls == 5

    def test_from_config_stores_config(self):
        """Test from_config stores the original config."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker.from_config(config)
        assert cb.config is config

    def test_from_config_default_name(self):
        """Test from_config uses default name if not provided."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker.from_config(config)
        assert cb.name == "default"


# ============================================================================
# Reset Tests
# ============================================================================


class TestReset:
    """Tests for CircuitBreaker reset functionality."""

    def test_reset_all(self):
        """Test reset clears all state."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure("entity-a")
        cb.record_failure("entity-a")

        cb.reset()
        assert cb.failures == 0
        assert not cb.is_open
        assert cb.get_status() == "closed"
        assert cb.is_available("entity-a")

    def test_reset_specific_entity(self):
        """Test reset for specific entity."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("a")
        cb.record_failure("a")
        cb.record_failure("b")
        cb.record_failure("b")

        cb.reset("a")
        assert cb.is_available("a")
        assert not cb.is_available("b")  # Still open


# ============================================================================
# Async Protected Call Tests
# ============================================================================


class TestProtectedCallAsync:
    """Tests for async protected_call context manager."""

    @pytest.mark.asyncio
    async def test_success_records_success(self):
        """Test successful call records success."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()  # 1 failure

        async with cb.protected_call():
            pass  # Success

        assert cb.failures == 0

    @pytest.mark.asyncio
    async def test_exception_records_failure(self):
        """Test exception records failure and re-raises."""
        cb = CircuitBreaker(failure_threshold=5)
        with pytest.raises(ValueError, match="test error"):
            async with cb.protected_call():
                raise ValueError("test error")

        assert cb.failures == 1

    @pytest.mark.asyncio
    async def test_open_circuit_raises_error(self):
        """Test protected_call raises CircuitOpenError when open."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)
        cb.record_failure()
        cb.record_failure()

        with pytest.raises(CircuitOpenError) as exc_info:
            async with cb.protected_call():
                pass

        assert exc_info.value.circuit_name == "circuit"

    @pytest.mark.asyncio
    async def test_entity_mode(self):
        """Test protected_call with entity."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")

        with pytest.raises(CircuitOpenError) as exc_info:
            async with cb.protected_call(entity="agent-1"):
                pass

        assert "agent-1" in exc_info.value.circuit_name

    @pytest.mark.asyncio
    async def test_cancelled_error_not_recorded(self):
        """Test asyncio.CancelledError does not record failure."""
        cb = CircuitBreaker(failure_threshold=5)
        with pytest.raises(asyncio.CancelledError):
            async with cb.protected_call():
                raise asyncio.CancelledError()

        assert cb.failures == 0  # Not recorded as failure


# ============================================================================
# Sync Protected Call Tests
# ============================================================================


class TestProtectedCallSync:
    """Tests for sync protected_call_sync context manager."""

    def test_success_records_success(self):
        """Test successful call records success."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()

        with cb.protected_call_sync():
            pass

        assert cb.failures == 0

    def test_exception_records_failure(self):
        """Test exception records failure and re-raises."""
        cb = CircuitBreaker(failure_threshold=5)
        with pytest.raises(RuntimeError, match="sync error"):
            with cb.protected_call_sync():
                raise RuntimeError("sync error")

        assert cb.failures == 1

    def test_open_circuit_raises_error(self):
        """Test protected_call_sync raises CircuitOpenError when open."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)
        cb.record_failure()
        cb.record_failure()

        with pytest.raises(CircuitOpenError):
            with cb.protected_call_sync():
                pass


# ============================================================================
# Execute Tests
# ============================================================================


class TestExecute:
    """Tests for CircuitBreaker.execute method."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test execute with successful async function."""
        cb = CircuitBreaker()

        async def my_func(x):
            return x * 2

        result = await cb.execute(my_func, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """Test execute records failure on exception."""
        cb = CircuitBreaker(failure_threshold=5)

        async def failing_func():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await cb.execute(failing_func)

        assert cb.failures == 1

    @pytest.mark.asyncio
    async def test_execute_open_circuit(self):
        """Test execute raises CircuitOpenError when circuit is open."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)
        cb.record_failure()
        cb.record_failure()

        async def my_func():
            return "hello"

        with pytest.raises(CircuitOpenError):
            await cb.execute(my_func)


# ============================================================================
# Metrics Callback Tests
# ============================================================================


class TestMetricsCallback:
    """Tests for metrics callback integration."""

    def test_metrics_emitted_on_open(self):
        """Test metrics callback is called when circuit opens."""
        calls = []

        def callback(name, state):
            calls.append((name, state))

        _set_metrics_callback(callback)
        try:
            cb = CircuitBreaker(failure_threshold=2)
            cb.record_failure()
            cb.record_failure()
            assert any(state == 1 for _, state in calls)  # 1 = open
        finally:
            _set_metrics_callback(None)

    def test_metrics_emitted_on_close(self):
        """Test metrics callback is called when circuit closes."""
        calls = []

        def callback(name, state):
            calls.append((name, state))

        _set_metrics_callback(callback)
        try:
            cb = CircuitBreaker(failure_threshold=2)
            cb.record_failure()
            cb.record_failure()
            cb.record_success()
            assert any(state == 0 for _, state in calls)  # 0 = closed
        finally:
            _set_metrics_callback(None)

    def test_metrics_callback_error_handled(self):
        """Test errors in metrics callback are handled gracefully."""

        def bad_callback(name, state):
            raise RuntimeError("callback error")

        _set_metrics_callback(bad_callback)
        try:
            cb = CircuitBreaker(failure_threshold=2)
            # Should not raise
            cb.record_failure()
            cb.record_failure()
        finally:
            _set_metrics_callback(None)
