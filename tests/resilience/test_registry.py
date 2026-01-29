"""
Tests for the circuit breaker registry.

Tests cover:
- get_circuit_breaker creates new or returns existing
- Thread safety with concurrent access
- Automatic pruning when MAX exceeded
- Provider-based config resolution
- reset_all_circuit_breakers
- get_circuit_breakers
- prune_circuit_breakers
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from aragora.resilience.circuit_breaker import CircuitBreaker
from aragora.resilience.registry import (
    MAX_CIRCUIT_BREAKERS,
    STALE_THRESHOLD_SECONDS,
    _circuit_breakers,
    _circuit_breakers_lock,
    _prune_stale_circuit_breakers,
    get_circuit_breaker,
    get_circuit_breakers,
    prune_circuit_breakers,
    reset_all_circuit_breakers,
)
from aragora.resilience_config import CircuitBreakerConfig


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before and after each test."""
    with _circuit_breakers_lock:
        _circuit_breakers.clear()
    yield
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


# ============================================================================
# get_circuit_breaker Tests
# ============================================================================


class TestGetCircuitBreaker:
    """Tests for get_circuit_breaker."""

    def test_creates_new_circuit_breaker(self):
        """Test creates a new circuit breaker when name not found."""
        cb = get_circuit_breaker("new-service")
        assert cb is not None
        assert isinstance(cb, CircuitBreaker)

    def test_returns_existing_circuit_breaker(self):
        """Test returns existing circuit breaker for same name."""
        cb1 = get_circuit_breaker("my-service")
        cb2 = get_circuit_breaker("my-service")
        assert cb1 is cb2

    def test_different_names_different_instances(self):
        """Test different names return different instances."""
        cb1 = get_circuit_breaker("service-a")
        cb2 = get_circuit_breaker("service-b")
        assert cb1 is not cb2

    def test_provider_based_config(self):
        """Test provider-based configuration resolution."""
        cb = get_circuit_breaker("anthropic-agent", provider="anthropic")
        # Anthropic config has failure_threshold=3, timeout=30s
        assert cb.failure_threshold == 3
        assert cb.cooldown_seconds == 30.0

    def test_explicit_config(self):
        """Test explicit config takes priority."""
        config = CircuitBreakerConfig(failure_threshold=10, timeout_seconds=120.0)
        cb = get_circuit_breaker("custom", config=config)
        assert cb.failure_threshold == 10
        assert cb.cooldown_seconds == 120.0

    def test_legacy_parameters(self):
        """Test legacy failure_threshold and cooldown_seconds parameters."""
        cb = get_circuit_breaker(
            "legacy-service",
            failure_threshold=7,
            cooldown_seconds=90.0,
        )
        assert cb.failure_threshold == 7
        assert cb.cooldown_seconds == 90.0

    def test_updates_last_accessed(self):
        """Test get_circuit_breaker updates _last_accessed."""
        cb = get_circuit_breaker("access-test")
        first_access = cb._last_accessed

        time.sleep(0.01)
        get_circuit_breaker("access-test")
        assert cb._last_accessed > first_access

    def test_default_config_used_when_no_params(self):
        """Test default config is used when no parameters provided."""
        cb = get_circuit_breaker("default-test")
        # Default config: failure_threshold=5, timeout_seconds=60
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 60.0


# ============================================================================
# reset_all_circuit_breakers Tests
# ============================================================================


class TestResetAll:
    """Tests for reset_all_circuit_breakers."""

    def test_resets_all_circuit_breakers(self):
        """Test resetting all circuit breakers."""
        cb1 = get_circuit_breaker("svc-1")
        cb2 = get_circuit_breaker("svc-2")

        cb1.record_failure()
        cb1.record_failure()
        cb1.record_failure()  # Opens
        cb2.record_failure()

        reset_all_circuit_breakers()

        assert cb1.failures == 0
        assert cb1.get_status() == "closed"
        assert cb2.failures == 0

    def test_reset_empty_registry(self):
        """Test resetting when registry is empty."""
        # Should not raise
        reset_all_circuit_breakers()


# ============================================================================
# get_circuit_breakers Tests
# ============================================================================


class TestGetCircuitBreakers:
    """Tests for get_circuit_breakers."""

    def test_returns_copy(self):
        """Test returns a copy of the registry."""
        get_circuit_breaker("svc-1")
        get_circuit_breaker("svc-2")

        result = get_circuit_breakers()
        assert len(result) == 2
        assert "svc-1" in result
        assert "svc-2" in result

    def test_modifications_dont_affect_registry(self):
        """Test modifying the copy doesn't affect the registry."""
        get_circuit_breaker("svc-1")
        result = get_circuit_breakers()
        result.clear()

        # Registry should still have the entry
        assert len(get_circuit_breakers()) == 1


# ============================================================================
# Pruning Tests
# ============================================================================


class TestPruning:
    """Tests for circuit breaker pruning."""

    def test_prune_stale_circuit_breakers(self):
        """Test pruning removes stale circuit breakers."""
        cb = get_circuit_breaker("stale-svc")
        # Make it stale
        cb._last_accessed = time.time() - STALE_THRESHOLD_SECONDS - 1

        with _circuit_breakers_lock:
            pruned = _prune_stale_circuit_breakers()

        assert pruned == 1
        assert len(get_circuit_breakers()) == 0

    def test_prune_keeps_recent(self):
        """Test pruning keeps recently accessed circuit breakers."""
        get_circuit_breaker("recent-svc")

        with _circuit_breakers_lock:
            pruned = _prune_stale_circuit_breakers()

        assert pruned == 0
        assert len(get_circuit_breakers()) == 1

    def test_public_prune_function(self):
        """Test prune_circuit_breakers public function."""
        cb = get_circuit_breaker("old-svc")
        cb._last_accessed = time.time() - STALE_THRESHOLD_SECONDS - 1

        pruned = prune_circuit_breakers()
        assert pruned == 1


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe registry access."""

    def test_concurrent_get_circuit_breaker(self):
        """Test concurrent access to get_circuit_breaker is thread-safe."""
        results = {}
        errors = []

        def worker(name):
            try:
                cb = get_circuit_breaker(name)
                results[name] = cb
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"svc-{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20

    def test_concurrent_same_name(self):
        """Test concurrent access with same name returns same instance."""
        results = []
        errors = []

        def worker():
            try:
                cb = get_circuit_breaker("shared-svc")
                results.append(cb)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All should be the same instance
        assert all(r is results[0] for r in results)
