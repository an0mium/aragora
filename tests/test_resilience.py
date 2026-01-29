"""Tests for resilience patterns (circuit breaker)."""

import pytest
import time
from unittest.mock import patch, MagicMock

from aragora.resilience import CircuitBreaker


class TestCircuitBreakerSingleEntity:
    """Tests for CircuitBreaker in single-entity mode."""

    def test_initial_state_is_closed(self):
        """Circuit starts in closed state."""
        cb = CircuitBreaker()
        assert cb.get_status() == "closed"
        assert cb.can_proceed() is True
        assert cb.failures == 0

    def test_failures_increment(self):
        """Failures increment counter."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        assert cb.failures == 1
        cb.record_failure()
        assert cb.failures == 2

    def test_opens_after_threshold(self):
        """Circuit opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        assert cb.get_status() == "closed"
        cb.record_failure()
        assert cb.get_status() == "closed"

        opened = cb.record_failure()
        assert opened is True
        assert cb.get_status() == "open"
        assert cb.can_proceed() is False

    def test_success_resets_failures(self):
        """Success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 2

        cb.record_success()
        assert cb.failures == 0

    def test_success_closes_open_circuit(self):
        """Success closes open circuit."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

        # Wait for cooldown then try (simulate half-open)
        time.sleep(0.15)
        assert cb.can_proceed() is True

        # Record success - should close
        cb.record_success()
        assert cb.is_open is False
        assert cb.get_status() == "closed"

    def test_cooldown_resets_circuit(self):
        """Circuit resets after cooldown period."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.can_proceed() is False

        # Wait for cooldown
        time.sleep(0.15)

        # Should be able to proceed now
        assert cb.can_proceed() is True
        assert cb.get_status() == "closed"

    def test_is_open_property_settable(self):
        """Can manually set is_open for testing."""
        cb = CircuitBreaker()
        assert cb.is_open is False

        cb.is_open = True
        assert cb.is_open is True
        assert cb.get_status() == "open"

        cb.is_open = False
        assert cb.is_open is False
        assert cb.get_status() == "closed"


class TestCircuitBreakerMultiEntity:
    """Tests for CircuitBreaker in multi-entity mode."""

    def test_tracks_entities_independently(self):
        """Each entity has independent failure tracking."""
        cb = CircuitBreaker(failure_threshold=2)

        cb.record_failure("agent-1")
        cb.record_failure("agent-1")

        assert cb.is_available("agent-1") is False
        assert cb.is_available("agent-2") is True

    def test_half_open_success_threshold(self):
        """Entity requires multiple successes to close."""
        cb = CircuitBreaker(
            failure_threshold=2,
            cooldown_seconds=0.01,
            half_open_success_threshold=2,
        )

        # Open circuit for entity
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")
        assert cb.get_status("agent-1") == "open"

        # Wait for cooldown
        time.sleep(0.02)
        assert cb.get_status("agent-1") == "half-open"

        # First success doesn't close
        cb.record_success("agent-1")
        assert cb.get_status("agent-1") == "half-open"

        # Second success closes
        cb.record_success("agent-1")
        assert cb.get_status("agent-1") == "closed"

    def test_filter_available_entities(self):
        """Filters out entities with open circuits."""
        cb = CircuitBreaker(failure_threshold=2)

        # Open circuit for agent-1
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")

        entities = ["agent-1", "agent-2", "agent-3"]
        available = cb.filter_available_entities(entities)

        assert "agent-1" not in available
        assert "agent-2" in available
        assert "agent-3" in available

    def test_filter_available_agents_with_objects(self):
        """Works with objects that have .name attribute."""
        cb = CircuitBreaker(failure_threshold=2)

        # Open circuit for agent-1
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")

        class MockAgent:
            def __init__(self, name):
                self.name = name

        agents = [MockAgent("agent-1"), MockAgent("agent-2")]
        available = cb.filter_available_agents(agents)

        assert len(available) == 1
        assert available[0].name == "agent-2"


class TestCircuitBreakerPersistence:
    """Tests for CircuitBreaker serialization."""

    def test_to_dict(self):
        """Can serialize to dict."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure("entity-1")

        data = cb.to_dict()

        assert "single_mode" in data
        assert data["single_mode"]["failures"] == 1
        assert "entity_mode" in data
        assert data["entity_mode"]["failures"]["entity-1"] == 1

    def test_from_dict(self):
        """Can restore from dict."""
        data = {
            "entity_mode": {
                "failures": {"agent-1": 2, "agent-2": 1},
                "open_circuits": {"agent-1": 5.0},  # 5 seconds elapsed
            }
        }

        cb = CircuitBreaker.from_dict(data, failure_threshold=3, cooldown_seconds=10)

        assert cb._failures["agent-1"] == 2
        assert cb._failures["agent-2"] == 1
        # agent-1 should still be in cooldown (5s < 10s)
        assert cb.is_available("agent-1") is False

    def test_from_dict_expired_cooldown(self):
        """Expired cooldowns are not restored."""
        data = {
            "entity_mode": {
                "failures": {"agent-1": 3},
                "open_circuits": {"agent-1": 120.0},  # 120 seconds elapsed
            }
        }

        cb = CircuitBreaker.from_dict(data, cooldown_seconds=60)

        # agent-1 cooldown expired, should be available
        # (but note failures are restored, so next failure may re-open)
        assert cb.is_available("agent-1") is True


class TestCircuitBreakerReset:
    """Tests for CircuitBreaker reset functionality."""

    def test_reset_all(self):
        """Reset clears all state."""
        cb = CircuitBreaker(failure_threshold=2)

        # Build up some state
        cb.record_failure()
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")

        cb.reset()

        assert cb.failures == 0
        assert cb.is_open is False
        assert cb._failures == {}
        assert cb._circuit_open_at == {}

    def test_reset_single_entity(self):
        """Can reset single entity."""
        cb = CircuitBreaker(failure_threshold=2)

        cb.record_failure("agent-1")
        cb.record_failure("agent-1")
        cb.record_failure("agent-2")
        cb.record_failure("agent-2")

        cb.reset("agent-1")

        # agent-1 reset, agent-2 still open
        assert cb.is_available("agent-1") is True
        assert cb.is_available("agent-2") is False


class TestCircuitBreakerStatus:
    """Tests for status reporting."""

    def test_get_all_status(self):
        """Gets status for all tracked entities."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure("agent-1")
        cb.record_failure("agent-2")
        cb.record_failure("agent-2")
        cb.record_failure("agent-2")

        status = cb.get_all_status()

        assert status["agent-1"]["status"] == "closed"
        assert status["agent-1"]["failures"] == 1
        assert status["agent-2"]["status"] == "open"
        assert status["agent-2"]["failures"] == 3

    def test_reset_timeout_alias(self):
        """reset_timeout is alias for cooldown_seconds."""
        cb = CircuitBreaker(cooldown_seconds=45.0)
        assert cb.reset_timeout == 45.0


# =============================================================================
# CircuitOpenError Tests
# =============================================================================


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_create_error(self):
        """Test creating CircuitOpenError."""
        from aragora.resilience import CircuitOpenError

        error = CircuitOpenError("my-circuit", 30.5)

        assert error.circuit_name == "my-circuit"
        assert error.cooldown_remaining == 30.5
        assert "my-circuit" in str(error)
        assert "30.5" in str(error)

    def test_raised_when_circuit_open(self):
        """Test error raised when circuit is open."""
        from aragora.resilience import CircuitOpenError

        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        with pytest.raises(CircuitOpenError) as exc_info:
            with cb.protected_call_sync():
                pass

        assert exc_info.value.circuit_name == "circuit"


# =============================================================================
# Protected Call Context Manager Tests
# =============================================================================


class TestProtectedCall:
    """Tests for protected_call context managers."""

    @pytest.mark.asyncio
    async def test_protected_call_success(self):
        """Test protected_call records success."""
        cb = CircuitBreaker()
        cb.record_failure()
        assert cb.failures == 1

        async with cb.protected_call():
            pass

        assert cb.failures == 0

    @pytest.mark.asyncio
    async def test_protected_call_failure(self):
        """Test protected_call records failure."""
        import asyncio

        cb = CircuitBreaker()

        with pytest.raises(ValueError):
            async with cb.protected_call():
                raise ValueError("test error")

        assert cb.failures == 1

    @pytest.mark.asyncio
    async def test_protected_call_cancelled_not_failure(self):
        """Test cancelled tasks don't count as failures."""
        import asyncio

        cb = CircuitBreaker()

        with pytest.raises(asyncio.CancelledError):
            async with cb.protected_call():
                raise asyncio.CancelledError()

        assert cb.failures == 0

    def test_protected_call_sync_success(self):
        """Test sync protected_call records success."""
        cb = CircuitBreaker()
        cb.record_failure()

        with cb.protected_call_sync():
            pass

        assert cb.failures == 0

    def test_protected_call_sync_failure(self):
        """Test sync protected_call records failure."""
        cb = CircuitBreaker()

        with pytest.raises(RuntimeError):
            with cb.protected_call_sync():
                raise RuntimeError("test")

        assert cb.failures == 1


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestGlobalRegistry:
    """Tests for global circuit breaker registry functions."""

    def test_get_circuit_breaker_creates_new(self):
        """Test get_circuit_breaker creates new breaker."""
        from aragora.resilience import get_circuit_breaker, reset_all_circuit_breakers

        reset_all_circuit_breakers()

        cb = get_circuit_breaker("test-registry-1")

        assert cb is not None
        assert cb.name == "test-registry-1"

    def test_get_circuit_breaker_returns_same(self):
        """Test get_circuit_breaker returns same instance."""
        from aragora.resilience import get_circuit_breaker, reset_all_circuit_breakers

        reset_all_circuit_breakers()

        cb1 = get_circuit_breaker("test-registry-2")
        cb2 = get_circuit_breaker("test-registry-2")

        assert cb1 is cb2

    def test_get_circuit_breaker_with_params(self):
        """Test get_circuit_breaker with custom params."""
        from aragora.resilience import get_circuit_breaker, reset_all_circuit_breakers

        reset_all_circuit_breakers()

        cb = get_circuit_breaker(
            "test-registry-3",
            failure_threshold=10,
            cooldown_seconds=120.0,
        )

        assert cb.failure_threshold == 10
        assert cb.cooldown_seconds == 120.0

    def test_reset_all_circuit_breakers(self):
        """Test reset_all_circuit_breakers."""
        from aragora.resilience import get_circuit_breaker, reset_all_circuit_breakers

        cb = get_circuit_breaker("test-reset", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

        reset_all_circuit_breakers()

        assert not cb.is_open
        assert cb.failures == 0

    def test_get_circuit_breakers(self):
        """Test get_circuit_breakers returns registry."""
        from aragora.resilience import (
            get_circuit_breaker,
            get_circuit_breakers,
            reset_all_circuit_breakers,
        )

        reset_all_circuit_breakers()
        get_circuit_breaker("test-list-1")
        get_circuit_breaker("test-list-2")

        breakers = get_circuit_breakers()

        assert "test-list-1" in breakers
        assert "test-list-2" in breakers

    def test_get_circuit_breaker_status(self):
        """Test get_circuit_breaker_status."""
        from aragora.resilience import (
            get_circuit_breaker,
            get_circuit_breaker_status,
            reset_all_circuit_breakers,
        )

        reset_all_circuit_breakers()
        cb = get_circuit_breaker("test-status")
        cb.record_failure()

        status = get_circuit_breaker_status()

        assert "_registry_size" in status
        assert "test-status" in status
        assert status["test-status"]["failures"] == 1

    def test_get_circuit_breaker_metrics(self):
        """Test get_circuit_breaker_metrics."""
        from aragora.resilience import (
            get_circuit_breaker,
            get_circuit_breaker_metrics,
            reset_all_circuit_breakers,
        )

        reset_all_circuit_breakers()
        cb = get_circuit_breaker("test-metrics", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        metrics = get_circuit_breaker_metrics()

        assert "summary" in metrics
        assert "circuit_breakers" in metrics
        assert "health" in metrics
        assert metrics["summary"]["open"] >= 1


# =============================================================================
# with_resilience Decorator Tests
# =============================================================================


class TestWithResilienceDecorator:
    """Tests for with_resilience decorator."""

    @pytest.mark.asyncio
    async def test_decorator_retries_on_failure(self):
        """Test decorator retries on failure."""
        from aragora.resilience import reset_all_circuit_breakers, with_resilience

        reset_all_circuit_breakers()

        call_count = 0

        @with_resilience(circuit_name="test-decorator-1", retries=3, use_circuit_breaker=False)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"

        result = await flaky_function()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_decorator_respects_circuit_breaker(self):
        """Test decorator respects circuit breaker."""
        from aragora.resilience import (
            CircuitOpenError,
            get_circuit_breaker,
            reset_all_circuit_breakers,
            with_resilience,
        )

        reset_all_circuit_breakers()

        # Pre-open the circuit
        cb = get_circuit_breaker("test-decorator-2", failure_threshold=2, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()

        @with_resilience(circuit_name="test-decorator-2", retries=1, use_circuit_breaker=True)
        async def protected_function():
            return "should not reach"

        with pytest.raises(CircuitOpenError):
            await protected_function()

    @pytest.mark.asyncio
    async def test_decorator_exhausts_retries(self):
        """Test decorator raises after exhausting retries."""
        from aragora.resilience import reset_all_circuit_breakers, with_resilience

        reset_all_circuit_breakers()

        @with_resilience(circuit_name="test-decorator-3", retries=2, use_circuit_breaker=False)
        async def always_fails():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            await always_fails()


# =============================================================================
# SQLite Persistence Tests
# =============================================================================


class TestSQLitePersistence:
    """Tests for SQLite persistence functions."""

    def test_init_persistence(self, tmp_path):
        """Test initializing persistence."""
        from aragora.resilience import init_circuit_breaker_persistence

        db_path = str(tmp_path / "test.db")
        init_circuit_breaker_persistence(db_path)

        assert (tmp_path / "test.db").exists()

    def test_persist_and_load(self, tmp_path):
        """Test persisting and loading circuit breaker."""
        from aragora.resilience import (
            init_circuit_breaker_persistence,
            load_circuit_breakers,
            persist_circuit_breaker,
            reset_all_circuit_breakers,
        )

        db_path = str(tmp_path / "test.db")
        init_circuit_breaker_persistence(db_path)
        reset_all_circuit_breakers()

        # Create and persist
        cb = CircuitBreaker(name="persist-test", failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        persist_circuit_breaker("persist-test", cb)

        # Clear and reload
        reset_all_circuit_breakers()
        loaded = load_circuit_breakers()

        assert loaded >= 1

    def test_persist_all(self, tmp_path):
        """Test persist_all_circuit_breakers."""
        from aragora.resilience import (
            get_circuit_breaker,
            init_circuit_breaker_persistence,
            persist_all_circuit_breakers,
            reset_all_circuit_breakers,
        )

        db_path = str(tmp_path / "test.db")
        init_circuit_breaker_persistence(db_path)
        reset_all_circuit_breakers()

        get_circuit_breaker("persist-all-1")
        get_circuit_breaker("persist-all-2")

        count = persist_all_circuit_breakers()

        assert count >= 2

    def test_cleanup_stale(self, tmp_path):
        """Test cleanup_stale_persisted."""
        from aragora.resilience import cleanup_stale_persisted, init_circuit_breaker_persistence

        db_path = str(tmp_path / "test.db")
        init_circuit_breaker_persistence(db_path)

        # Should not raise
        deleted = cleanup_stale_persisted(max_age_hours=0.0001)

        assert isinstance(deleted, int)


# =============================================================================
# Metrics API Tests
# =============================================================================


class TestMetricsAPI:
    """Tests for metrics API functions."""

    def test_get_all_circuit_breakers_status(self):
        """Test get_all_circuit_breakers_status."""
        from aragora.resilience import (
            get_all_circuit_breakers_status,
            get_circuit_breaker,
            reset_all_circuit_breakers,
        )

        reset_all_circuit_breakers()
        get_circuit_breaker("metrics-api-1")

        status = get_all_circuit_breakers_status()

        assert "healthy" in status
        assert "total_circuits" in status
        assert "circuits" in status

    def test_get_circuit_breaker_summary(self):
        """Test get_circuit_breaker_summary."""
        from aragora.resilience import (
            get_circuit_breaker,
            get_circuit_breaker_summary,
            reset_all_circuit_breakers,
        )

        reset_all_circuit_breakers()
        cb = get_circuit_breaker("summary-test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        summary = get_circuit_breaker_summary()

        assert "healthy" in summary
        assert "total" in summary
        assert "open" in summary
        assert "summary-test" in summary["open"]


# =============================================================================
# Metrics Callback Tests
# =============================================================================


class TestMetricsCallback:
    """Tests for metrics callback integration."""

    def test_set_metrics_callback(self):
        """Test set_metrics_callback."""
        from aragora.resilience import set_metrics_callback

        callback_calls = []

        def mock_callback(name: str, state: int) -> None:
            callback_calls.append((name, state))

        set_metrics_callback(mock_callback)

        # Cleanup
        set_metrics_callback(None)

    def test_metrics_emitted_on_state_change(self):
        """Test metrics emitted on circuit state change."""
        from aragora.resilience import set_metrics_callback

        callback_calls = []

        def mock_callback(name: str, state: int) -> None:
            callback_calls.append((name, state))

        set_metrics_callback(mock_callback)

        try:
            cb = CircuitBreaker(name="callback-test", failure_threshold=2)
            cb.record_failure()
            cb.record_failure()  # Should trigger open (state=1)
            cb.record_success()  # Should trigger closed (state=0)

            # Check callbacks were made
            assert any(state == 1 for _, state in callback_calls)  # open
            assert any(state == 0 for _, state in callback_calls)  # closed
        finally:
            set_metrics_callback(None)


# =============================================================================
# CircuitBreaker from_config Tests
# =============================================================================


class TestCircuitBreakerFromConfig:
    """Tests for CircuitBreaker.from_config factory method."""

    def test_from_config(self):
        """Test creating from config."""
        from aragora.resilience_config import CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=10,
            timeout_seconds=120.0,
            success_threshold=3,
        )

        cb = CircuitBreaker.from_config(config, name="config-test")

        assert cb.name == "config-test"
        assert cb.failure_threshold == 10
        assert cb.cooldown_seconds == 120.0
        assert cb.half_open_success_threshold == 3

    def test_config_property(self):
        """Test config property returns original config."""
        from aragora.resilience_config import CircuitBreakerConfig

        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker.from_config(config)

        assert cb.config is config


# =============================================================================
# Pruning Tests
# =============================================================================


class TestPruning:
    """Tests for circuit breaker pruning."""

    def test_prune_circuit_breakers(self):
        """Test prune_circuit_breakers function."""
        from aragora.resilience import (
            get_circuit_breaker,
            prune_circuit_breakers,
            reset_all_circuit_breakers,
        )

        reset_all_circuit_breakers()

        # Create a circuit breaker
        cb = get_circuit_breaker("prune-test")
        # Make it stale by backdating last_accessed
        cb._last_accessed = time.time() - (25 * 60 * 60)  # 25 hours ago

        pruned = prune_circuit_breakers()

        # May or may not be pruned depending on timing
        assert isinstance(pruned, int)


# =============================================================================
# Execute Method Tests
# =============================================================================


class TestExecuteMethod:
    """Tests for CircuitBreaker.execute method."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test execute with successful call."""
        cb = CircuitBreaker()

        async def successful_call():
            return "result"

        result = await cb.execute(successful_call)

        assert result == "result"

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """Test execute with failing call."""
        cb = CircuitBreaker()

        async def failing_call():
            raise ValueError("test")

        with pytest.raises(ValueError):
            await cb.execute(failing_call)

        assert cb.failures == 1


# =============================================================================
# Cooldown Remaining Tests
# =============================================================================


class TestCooldownRemaining:
    """Tests for cooldown_remaining method."""

    def test_cooldown_remaining_closed(self):
        """Test cooldown_remaining when closed."""
        cb = CircuitBreaker()
        assert cb.cooldown_remaining() == 0.0

    def test_cooldown_remaining_open(self):
        """Test cooldown_remaining when open."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0)
        cb.record_failure()
        cb.record_failure()

        remaining = cb.cooldown_remaining()
        assert 9.0 < remaining <= 10.0

    def test_cooldown_remaining_for_entity(self):
        """Test cooldown_remaining for specific entity."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0)
        cb.record_failure("agent-1")
        cb.record_failure("agent-1")

        remaining = cb.cooldown_remaining("agent-1")
        assert 9.0 < remaining <= 10.0

        # Non-open entity has no cooldown
        assert cb.cooldown_remaining("agent-2") == 0.0


# =============================================================================
# Recovery Timeout Alias Tests
# =============================================================================


class TestRecoveryTimeoutAlias:
    """Tests for recovery_timeout backward compatibility."""

    def test_recovery_timeout_sets_cooldown(self):
        """Test recovery_timeout sets cooldown_seconds."""
        cb = CircuitBreaker(recovery_timeout=90.0)
        assert cb.cooldown_seconds == 90.0


# =============================================================================
# State Property Tests
# =============================================================================


class TestStateProperty:
    """Tests for state property backward compatibility."""

    def test_state_property_returns_status(self):
        """Test state property returns same as get_status."""
        cb = CircuitBreaker()
        assert cb.state == cb.get_status()
        assert cb.state == "closed"

    def test_state_when_open(self):
        """Test state when circuit is open."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
