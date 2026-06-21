"""
Tests for Nomic Loop Error Recovery System.

Tests cover:
- RecoveryStrategy enum
- RecoveryDecision dataclass
- CircuitBreaker class - state transitions, failure tracking, timeout
- CircuitBreakerRegistry - management of multiple circuit breakers
- calculate_backoff - exponential backoff calculation
- RecoveryManager - error classification and recovery decisions
- recovery_handler - async handler for RECOVERY state
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from aragora.nomic.recovery import (
    RecoveryStrategy,
    RecoveryDecision,
    CircuitBreaker,
    CircuitBreakerRegistry,
    RecoveryManager,
    calculate_backoff,
    recovery_handler,
)
from aragora.nomic.states import NomicState, StateContext
from aragora.nomic.events import Event, EventType


# =============================================================================
# Tests: RecoveryStrategy Enum
# =============================================================================


class TestRecoveryStrategy:
    """Tests for RecoveryStrategy enum."""

    def test_all_strategies_exist(self):
        """Test that all expected strategies are defined."""
        assert RecoveryStrategy.RETRY
        assert RecoveryStrategy.SKIP
        assert RecoveryStrategy.ROLLBACK
        assert RecoveryStrategy.RESTART
        assert RecoveryStrategy.PAUSE
        assert RecoveryStrategy.FAIL

    def test_strategies_are_unique(self):
        """Test that all strategies have unique values."""
        values = [s.value for s in RecoveryStrategy]
        assert len(values) == len(set(values))


# =============================================================================
# Tests: RecoveryDecision Dataclass
# =============================================================================


class TestRecoveryDecision:
    """Tests for RecoveryDecision dataclass."""

    def test_create_basic_decision(self):
        """Test creating a basic recovery decision."""
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RETRY,
            target_state=NomicState.CONTEXT,
            delay_seconds=5.0,
            reason="Test retry",
        )

        assert decision.strategy == RecoveryStrategy.RETRY
        assert decision.target_state == NomicState.CONTEXT
        assert decision.delay_seconds == 5.0
        assert decision.reason == "Test retry"
        assert decision.requires_human is False

    def test_default_values(self):
        """Test default values for optional fields."""
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.FAIL,
        )

        assert decision.target_state is None
        assert decision.delay_seconds == 0
        assert decision.reason == ""
        assert decision.requires_human is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.PAUSE,
            target_state=NomicState.DEBATE,
            delay_seconds=10.0,
            reason="Test pause",
            requires_human=True,
        )

        d = decision.to_dict()
        assert d["strategy"] == "PAUSE"
        assert d["target_state"] == "DEBATE"
        assert d["delay_seconds"] == 10.0
        assert d["reason"] == "Test pause"
        assert d["requires_human"] is True

    def test_to_dict_without_target_state(self):
        """Test serialization when target_state is None."""
        decision = RecoveryDecision(strategy=RecoveryStrategy.FAIL)
        d = decision.to_dict()
        assert d["target_state"] is None


# =============================================================================
# Tests: CircuitBreaker
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Test that circuit starts closed."""
        cb = CircuitBreaker("test")
        assert cb.is_open is False
        assert cb._state == "closed"

    def test_record_success_keeps_closed(self):
        """Test that successes keep circuit closed."""
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_success()
        assert cb.is_open is False
        assert cb._failures == 0

    def test_record_failure_increments_count(self):
        """Test that failures increment the count."""
        cb = CircuitBreaker("test", failure_threshold=5)
        cb.record_failure()
        assert cb._failures == 1
        assert cb.is_open is False  # Not yet at threshold

    def test_circuit_opens_at_threshold(self):
        """Test that circuit opens when failure threshold is reached."""
        cb = CircuitBreaker("test", failure_threshold=3)

        cb.record_failure()  # 1
        cb.record_failure()  # 2
        assert cb.is_open is False

        cb.record_failure()  # 3 - threshold reached
        assert cb.is_open is True
        assert cb._state == "open"

    def test_success_resets_failure_count(self):
        """Test that success resets the failure count."""
        cb = CircuitBreaker("test", failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert cb._failures == 0
        assert cb.is_open is False

    def test_success_closes_circuit(self):
        """Test that success closes an open circuit."""
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()  # Opens circuit
        assert cb.is_open is True

        # Simulate timeout and success
        cb._state = "half-open"
        cb.record_success()

        assert cb._state == "closed"
        assert cb.is_open is False

    def test_manual_reset(self):
        """Test manual reset of circuit breaker."""
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.is_open is True

        cb.reset()
        assert cb._failures == 0
        assert cb._state == "closed"
        assert cb._last_failure_time is None
        assert cb.is_open is False

    def test_half_open_after_timeout(self):
        """Test that circuit goes half-open after reset timeout."""
        cb = CircuitBreaker("test", failure_threshold=1, reset_timeout_seconds=0)
        cb.record_failure()  # Opens circuit
        assert cb._state == "open"

        # After timeout (0 seconds), checking is_open should transition to half-open
        time.sleep(0.01)
        assert cb.is_open is False  # Should be half-open now, allowing one call
        assert cb._state == "half-open"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        cb = CircuitBreaker("test-cb", failure_threshold=5)
        cb.record_failure()

        d = cb.to_dict()
        assert d["name"] == "test-cb"
        assert d["state"] == "closed"
        assert d["failures"] == 1
        assert d["failure_threshold"] == 5
        assert d["last_failure"] is not None


# =============================================================================
# Tests: CircuitBreakerRegistry
# =============================================================================


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry class."""

    def test_get_or_create_creates_new(self):
        """Test that get_or_create creates a new circuit breaker."""
        registry = CircuitBreakerRegistry()
        cb = registry.get_or_create("new-cb", failure_threshold=5)

        assert cb is not None
        assert cb.name == "new-cb"
        assert cb.failure_threshold == 5

    def test_get_or_create_returns_existing(self):
        """Test that get_or_create returns existing circuit breaker."""
        registry = CircuitBreakerRegistry()
        cb1 = registry.get_or_create("existing-cb")
        cb1.record_failure()

        cb2 = registry.get_or_create("existing-cb")
        assert cb2 is cb1
        assert cb2._failures == 1

    def test_get_returns_none_for_nonexistent(self):
        """Test that get returns None for non-existent breaker."""
        registry = CircuitBreakerRegistry()
        assert registry.get("nonexistent") is None

    def test_all_open_returns_open_circuits(self):
        """Test getting all open circuit breakers."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("cb1", failure_threshold=1)
        cb2 = registry.get_or_create("cb2", failure_threshold=1)
        cb3 = registry.get_or_create("cb3", failure_threshold=1)

        cb1.record_failure()  # Opens
        cb3.record_failure()  # Opens

        open_names = registry.all_open()
        assert "cb1" in open_names
        assert "cb2" not in open_names
        assert "cb3" in open_names

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("cb1", failure_threshold=1)
        cb2 = registry.get_or_create("cb2", failure_threshold=1)

        cb1.record_failure()
        cb2.record_failure()

        assert len(registry.all_open()) == 2

        registry.reset_all()

        assert len(registry.all_open()) == 0

    def test_to_dict(self):
        """Test serialization of registry."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("cb1")
        registry.get_or_create("cb2")

        d = registry.to_dict()
        assert "cb1" in d
        assert "cb2" in d


# =============================================================================
# Tests: calculate_backoff
# =============================================================================


class TestCalculateBackoff:
    """Tests for exponential backoff calculation."""

    def test_first_attempt_is_base_delay(self):
        """Test that first attempt uses base delay."""
        delay = calculate_backoff(1, base_delay=1.0, jitter=False)
        assert delay == 1.0

    def test_exponential_growth(self):
        """Test exponential growth of delays."""
        d1 = calculate_backoff(1, base_delay=1.0, jitter=False)
        d2 = calculate_backoff(2, base_delay=1.0, jitter=False)
        d3 = calculate_backoff(3, base_delay=1.0, jitter=False)

        assert d1 == 1.0
        assert d2 == 2.0
        assert d3 == 4.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        delay = calculate_backoff(100, base_delay=1.0, max_delay=60.0, jitter=False)
        assert delay == 60.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness within bounds."""
        # Run multiple times to verify jitter behavior
        delays = [calculate_backoff(1, base_delay=10.0, jitter=True) for _ in range(100)]

        # All should be within 25% of base
        assert all(7.5 <= d <= 12.5 for d in delays)

        # Should have some variation (not all same)
        assert len(set(delays)) > 1

    def test_custom_base_delay(self):
        """Test custom base delay."""
        delay = calculate_backoff(1, base_delay=5.0, jitter=False)
        assert delay == 5.0


# =============================================================================
# Tests: RecoveryManager
# =============================================================================


class TestRecoveryManager:
    """Tests for RecoveryManager class."""

    @pytest.fixture
    def manager(self):
        """Create a RecoveryManager instance."""
        return RecoveryManager()

    @pytest.fixture
    def context(self):
        """Create a StateContext for testing."""
        return StateContext(
            cycle_id="test-cycle",
            current_state=NomicState.RECOVERY,
            previous_state=NomicState.CONTEXT,
        )

    def test_init_with_custom_registry(self):
        """Test initialization with custom circuit breaker registry."""
        registry = CircuitBreakerRegistry()
        manager = RecoveryManager(circuit_breakers=registry)
        assert manager.circuit_breakers is registry

    def test_transient_error_detection_timeout(self, manager):
        """Test that TimeoutError is detected as transient."""
        error = TimeoutError("Connection timed out")
        assert manager._is_transient_error(error) is True

    def test_transient_error_detection_connection(self, manager):
        """Test that ConnectionError is detected as transient."""
        error = ConnectionError("Connection refused")
        assert manager._is_transient_error(error) is True

    def test_transient_error_detection_by_message(self, manager):
        """Test transient error detection by message content."""
        error = Exception("Rate limit exceeded, please retry")
        assert manager._is_transient_error(error) is True

        error = Exception("HTTP 429 Too Many Requests")
        assert manager._is_transient_error(error) is True

        error = Exception("HTTP 503 Service Unavailable")
        assert manager._is_transient_error(error) is True

    def test_non_transient_error_detection(self, manager):
        """Test that non-transient errors are correctly classified."""
        error = ValueError("Invalid argument")
        assert manager._is_transient_error(error) is False

        error = KeyError("Missing key")
        assert manager._is_transient_error(error) is False

    def test_decide_recovery_transient_error_retries(self, manager, context):
        """Test that transient errors trigger retry."""
        error = TimeoutError("Request timed out")
        context.retry_counts = {"CONTEXT": 0}

        decision = manager.decide_recovery(
            state=NomicState.CONTEXT,
            error=error,
            context=context,
        )

        assert decision.strategy == RecoveryStrategy.RETRY
        assert decision.target_state == NomicState.CONTEXT
        assert decision.delay_seconds > 0

    def test_decide_recovery_multiple_open_circuits_pauses(self, manager, context):
        """Test that multiple open circuits trigger pause."""
        # Open two circuits
        cb1 = manager.circuit_breakers.get_or_create("agent1", failure_threshold=1)
        cb2 = manager.circuit_breakers.get_or_create("agent2", failure_threshold=1)
        cb1.record_failure()
        cb2.record_failure()

        error = Exception("Some error")
        decision = manager.decide_recovery(
            state=NomicState.CONTEXT,
            error=error,
            context=context,
        )

        assert decision.strategy == RecoveryStrategy.PAUSE
        assert decision.requires_human is True

    def test_decide_recovery_non_critical_skips(self, manager, context):
        """Test that non-critical states can be skipped."""
        error = ValueError("Persistent error")
        context.retry_counts = {"DESIGN": 10}  # Exhausted retries
        context.previous_state = NomicState.DESIGN

        decision = manager.decide_recovery(
            state=NomicState.DESIGN,
            error=error,
            context=context,
        )

        assert decision.strategy == RecoveryStrategy.SKIP
        assert decision.target_state == NomicState.IMPLEMENT

    def test_decide_recovery_critical_state_fails(self, manager, context):
        """Test that critical states fail after exhausted retries."""
        error = ValueError("Persistent error")
        # DEBATE is critical with max_retries=1
        context.retry_counts = {"DEBATE": 10}
        context.previous_state = NomicState.DEBATE

        decision = manager.decide_recovery(
            state=NomicState.DEBATE,
            error=error,
            context=context,
        )

        assert decision.strategy == RecoveryStrategy.FAIL

    def test_history_is_recorded(self, manager, context):
        """Test that recovery decisions are recorded in history."""
        error = Exception("Test error")
        manager.decide_recovery(
            state=NomicState.CONTEXT,
            error=error,
            context=context,
        )

        history = manager.get_history()
        assert len(history) == 1
        assert history[0]["state"] == "CONTEXT"
        assert "decision" in history[0]

    def test_clear_history(self, manager, context):
        """Test clearing recovery history."""
        error = Exception("Test error")
        manager.decide_recovery(NomicState.CONTEXT, error, context)

        manager.clear_history()
        assert len(manager.get_history()) == 0

    def test_skip_map_design_to_implement(self, manager):
        """Test skip target for DESIGN state."""
        target = manager._get_skip_target(NomicState.DESIGN)
        assert target == NomicState.IMPLEMENT

    def test_skip_map_implement_to_verify(self, manager):
        """Test skip target for IMPLEMENT state."""
        target = manager._get_skip_target(NomicState.IMPLEMENT)
        assert target == NomicState.VERIFY

    def test_skip_map_verify_to_commit(self, manager):
        """Test skip target for VERIFY state."""
        target = manager._get_skip_target(NomicState.VERIFY)
        assert target == NomicState.COMMIT

    def test_skip_map_commit_to_completed(self, manager):
        """Test skip target for COMMIT state."""
        target = manager._get_skip_target(NomicState.COMMIT)
        assert target == NomicState.COMPLETED

    def test_skip_map_unknown_returns_none(self, manager):
        """Test skip target for state without mapping."""
        target = manager._get_skip_target(NomicState.CONTEXT)
        assert target is None


# =============================================================================
# Tests: recovery_handler (async)
# =============================================================================


class TestRecoveryHandler:
    """Tests for the async recovery_handler function."""

    @pytest.fixture
    def context(self):
        """Create a StateContext for testing."""
        return StateContext(
            cycle_id="test-cycle",
            current_state=NomicState.RECOVERY,
            previous_state=NomicState.CONTEXT,
            retry_counts={},
        )

    @pytest.fixture
    def manager(self):
        """Create a RecoveryManager."""
        return RecoveryManager()

    @pytest.mark.asyncio
    async def test_handler_returns_next_state_for_retry(self, context, manager):
        """Test handler returns retry target state."""
        event = Event(
            event_type=EventType.ERROR,
            error_message="Connection timeout",
            error_type="TimeoutError",
        )

        next_state, result = await recovery_handler(context, event, manager)

        assert next_state == NomicState.CONTEXT  # Retry same state
        assert "decision" in result
        assert result["recovered_from"] == "CONTEXT"

    @pytest.mark.asyncio
    async def test_handler_returns_failed_for_critical_exhausted(self, context, manager):
        """Test handler returns FAILED for exhausted critical state."""
        context.previous_state = NomicState.DEBATE  # Critical state
        context.retry_counts = {"DEBATE": 100}

        event = Event(
            event_type=EventType.ERROR,
            error_message="Persistent error",
            error_type="ValueError",
        )

        next_state, result = await recovery_handler(context, event, manager)

        assert next_state == NomicState.FAILED

    @pytest.mark.asyncio
    async def test_handler_returns_paused_for_multiple_circuits(self, context, manager):
        """Test handler returns PAUSED when multiple circuits open."""
        # Open multiple circuits
        cb1 = manager.circuit_breakers.get_or_create("agent1", failure_threshold=1)
        cb2 = manager.circuit_breakers.get_or_create("agent2", failure_threshold=1)
        cb1.record_failure()
        cb2.record_failure()

        event = Event(
            event_type=EventType.ERROR,
            error_message="Error",
            error_type="Exception",
        )

        next_state, result = await recovery_handler(context, event, manager)

        assert next_state == NomicState.PAUSED
        assert result["decision"]["requires_human"] is True

    @pytest.mark.asyncio
    async def test_handler_respects_delay(self, context, manager):
        """Test that handler waits for delay before returning."""
        event = Event(
            event_type=EventType.ERROR,
            error_message="Rate limit",
            error_type="TimeoutError",
        )

        start = time.time()

        # Patch calculate_backoff to return small delay
        with patch("aragora.nomic.recovery.calculate_backoff", return_value=0.1):
            await recovery_handler(context, event, manager)

        elapsed = time.time() - start
        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_handler_uses_previous_state_as_fallback(self, context, manager):
        """Test handler uses previous_state as failed state."""
        context.previous_state = NomicState.DESIGN

        event = Event(
            event_type=EventType.ERROR,
            error_message="Test error",
        )

        next_state, result = await recovery_handler(context, event, manager)

        assert result["recovered_from"] == "DESIGN"

    @pytest.mark.asyncio
    async def test_handler_handles_missing_error_info(self, context, manager):
        """Test handler handles missing error info gracefully."""
        event = Event(event_type=EventType.ERROR)  # No error_message or error_type

        next_state, result = await recovery_handler(context, event, manager)

        assert result["original_error"] == "Unknown error"


# =============================================================================
# Tests: Integration / Edge Cases
# =============================================================================


class TestRecoveryIntegration:
    """Integration tests for recovery system."""

    def test_full_recovery_flow(self):
        """Test complete recovery flow from error to decision."""
        manager = RecoveryManager()
        context = StateContext(
            cycle_id="integration-test",
            current_state=NomicState.RECOVERY,
            previous_state=NomicState.IMPLEMENT,
            retry_counts={"IMPLEMENT": 0},
        )

        # First failure - should retry
        error = TimeoutError("Network timeout")
        decision = manager.decide_recovery(NomicState.IMPLEMENT, error, context)
        assert decision.strategy == RecoveryStrategy.RETRY

        # Update retry count
        context.retry_counts["IMPLEMENT"] = 1

        # Second failure - should fail (IMPLEMENT is critical with max_retries=1)
        decision = manager.decide_recovery(NomicState.IMPLEMENT, error, context)
        assert decision.strategy == RecoveryStrategy.FAIL

    def test_circuit_breaker_affects_recovery(self):
        """Test that circuit breaker state affects recovery decisions."""
        manager = RecoveryManager()
        context = StateContext(
            cycle_id="circuit-test",
            current_state=NomicState.RECOVERY,
            previous_state=NomicState.CONTEXT,
        )

        # Initially no open circuits - normal retry
        error = Exception("Error")
        decision = manager.decide_recovery(NomicState.CONTEXT, error, context)
        # Should be some retry-based decision

        # Open two circuits
        cb1 = manager.circuit_breakers.get_or_create("agent1", failure_threshold=1)
        cb2 = manager.circuit_breakers.get_or_create("agent2", failure_threshold=1)
        cb1.record_failure()
        cb2.record_failure()

        # Now should pause
        decision = manager.decide_recovery(NomicState.CONTEXT, error, context)
        assert decision.strategy == RecoveryStrategy.PAUSE

    def test_backoff_increases_with_retries(self):
        """Test that backoff delay increases with retry count."""
        delays = []
        for i in range(5):
            delay = calculate_backoff(i + 1, base_delay=1.0, jitter=False)
            delays.append(delay)

        # Each delay should be >= previous
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1]
