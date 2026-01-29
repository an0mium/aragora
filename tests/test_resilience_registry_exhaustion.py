"""Tests for circuit breaker registry exhaustion handling.

Tests verify that the circuit breaker registry properly handles exhaustion
scenarios, including pruning of stale breakers and logging warnings when
the registry remains full after pruning.
"""

import time
import threading
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from aragora.resilience import (
    get_circuit_breaker,
    reset_all_circuit_breakers,
    prune_circuit_breakers,
    get_circuit_breakers,
)
import aragora.resilience as resilience_module

# Access internal registry for testing
_circuit_breakers = resilience_module._circuit_breakers
_circuit_breakers_lock = resilience_module._circuit_breakers_lock
MAX_CIRCUIT_BREAKERS = resilience_module.MAX_CIRCUIT_BREAKERS
STALE_THRESHOLD_SECONDS = resilience_module.STALE_THRESHOLD_SECONDS


def clear_circuit_breakers():
    """Clear all circuit breakers from registry (for testing)."""
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear circuit breaker registry before and after each test."""
    clear_circuit_breakers()
    yield
    clear_circuit_breakers()


class TestRegistryExhaustion:
    """Tests for circuit breaker registry exhaustion scenarios."""

    def test_registry_exhaustion_logs_warning_after_pruning(self, caplog):
        """Test that warning is logged when registry remains full after pruning."""
        import logging

        # Use a smaller limit for faster testing
        test_max = 10

        with patch.object(resilience_module, "MAX_CIRCUIT_BREAKERS", test_max):
            # Fill the registry to the max
            for i in range(test_max):
                get_circuit_breaker(f"breaker_{i}")

            # Clear the log
            caplog.clear()

            # Add one more to trigger pruning (but no stale entries exist)
            with caplog.at_level(logging.WARNING, logger="aragora.resilience"):
                get_circuit_breaker("breaker_overflow")

            # Since no entries are stale (all recently created), warning should be logged
            assert "Circuit breaker registry still large after pruning" in caplog.text

    def test_registry_prunes_stale_breakers_on_overflow(self):
        """Test that stale circuit breakers are pruned when registry overflows."""
        # Create some circuit breakers
        for i in range(5):
            get_circuit_breaker(f"breaker_{i}")

        # Make some of them stale by setting _last_accessed in the past
        with _circuit_breakers_lock:
            for i in range(3):
                _circuit_breakers[f"breaker_{i}"]._last_accessed = (
                    time.time() - STALE_THRESHOLD_SECONDS - 100
                )

        # Verify stale breakers exist
        assert len(_circuit_breakers) == 5

        # Prune stale breakers
        pruned = prune_circuit_breakers()

        # Should have pruned the 3 stale ones
        assert pruned == 3
        assert len(_circuit_breakers) == 2

        # Verify the remaining ones are the non-stale ones
        with _circuit_breakers_lock:
            remaining = list(_circuit_breakers.keys())
        assert "breaker_3" in remaining
        assert "breaker_4" in remaining

    def test_registry_continues_working_after_exhaustion(self):
        """Test that new circuit breakers can still be created after exhaustion."""
        # Fill the registry
        for i in range(MAX_CIRCUIT_BREAKERS - 1):
            get_circuit_breaker(f"breaker_{i}")

        # Make half of them stale
        stale_count = MAX_CIRCUIT_BREAKERS // 2
        with _circuit_breakers_lock:
            for i in range(stale_count):
                _circuit_breakers[f"breaker_{i}"]._last_accessed = (
                    time.time() - STALE_THRESHOLD_SECONDS - 100
                )

        # Add a new one - should trigger pruning and succeed
        cb = get_circuit_breaker("new_breaker")
        assert cb is not None
        assert cb.get_status() == "closed"

        # The new breaker should exist
        with _circuit_breakers_lock:
            assert "new_breaker" in _circuit_breakers

    def test_concurrent_access_during_exhaustion(self):
        """Test that concurrent access during exhaustion is thread-safe."""
        errors = []
        breakers_created = []

        def create_breaker(name):
            try:
                cb = get_circuit_breaker(name)
                breakers_created.append(name)
                # Do some operations
                cb.record_failure()
                cb.record_success()
            except Exception as e:
                errors.append((name, str(e)))

        # Create many threads that all try to get circuit breakers
        threads = []
        for i in range(100):
            t = threading.Thread(target=create_breaker, args=(f"concurrent_{i}",))
            threads.append(t)

        # Start all threads simultaneously
        for t in threads:
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join(timeout=10)

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

        # Verify all breakers were created
        assert len(breakers_created) == 100

    def test_prune_respects_access_time(self):
        """Test that pruning only removes breakers not accessed within threshold."""
        # Create breakers with different access times
        get_circuit_breaker("recent_1")
        get_circuit_breaker("recent_2")
        get_circuit_breaker("stale_1")
        get_circuit_breaker("stale_2")

        # Make some stale
        with _circuit_breakers_lock:
            _circuit_breakers["stale_1"]._last_accessed = (
                time.time() - STALE_THRESHOLD_SECONDS - 1
            )
            _circuit_breakers["stale_2"]._last_accessed = (
                time.time() - STALE_THRESHOLD_SECONDS - 1
            )

        # Prune
        pruned = prune_circuit_breakers()

        assert pruned == 2
        with _circuit_breakers_lock:
            assert "recent_1" in _circuit_breakers
            assert "recent_2" in _circuit_breakers
            assert "stale_1" not in _circuit_breakers
            assert "stale_2" not in _circuit_breakers

    def test_accessing_breaker_updates_timestamp(self):
        """Test that accessing a circuit breaker updates its last_accessed time."""
        cb = get_circuit_breaker("timestamp_test")

        with _circuit_breakers_lock:
            initial_time = _circuit_breakers["timestamp_test"]._last_accessed

        # Wait a bit and access again
        time.sleep(0.1)
        get_circuit_breaker("timestamp_test")

        with _circuit_breakers_lock:
            new_time = _circuit_breakers["timestamp_test"]._last_accessed

        assert new_time > initial_time

    def test_max_circuit_breakers_constant_is_reasonable(self):
        """Test that MAX_CIRCUIT_BREAKERS is set to a reasonable value."""
        # Should be at least 100 to handle typical use cases
        assert MAX_CIRCUIT_BREAKERS >= 100
        # Should be at most 10000 to avoid memory issues
        assert MAX_CIRCUIT_BREAKERS <= 10000

    def test_stale_threshold_is_reasonable(self):
        """Test that STALE_THRESHOLD_SECONDS is set to a reasonable value."""
        # Should be at least 1 hour
        assert STALE_THRESHOLD_SECONDS >= 3600
        # Should be at most 1 week
        assert STALE_THRESHOLD_SECONDS <= 7 * 24 * 3600
