"""Tests for exponential backoff utilities.

Tests cover:
- Exponential delay calculation
- Jitter application
- Failure tracking
- Thread safety
- Reset behavior
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from aragora.shared.rate_limiting.backoff import ExponentialBackoff


class TestExponentialBackoffInit:
    """Tests for ExponentialBackoff initialization."""

    def test_default_values(self):
        """Should initialize with sensible defaults."""
        backoff = ExponentialBackoff()
        assert backoff.base_delay == 1.0
        assert backoff.max_delay == 60.0
        assert backoff.jitter == 0.1
        assert backoff.failure_count == 0

    def test_custom_values(self):
        """Should accept custom configuration."""
        backoff = ExponentialBackoff(base_delay=2.0, max_delay=120.0, jitter=0.2)
        assert backoff.base_delay == 2.0
        assert backoff.max_delay == 120.0
        assert backoff.jitter == 0.2


class TestGetDelay:
    """Tests for get_delay method."""

    def test_first_delay_is_base(self):
        """First delay should be approximately base_delay."""
        backoff = ExponentialBackoff(base_delay=1.0, jitter=0.0)
        delay = backoff.get_delay()
        assert delay == 1.0

    def test_delay_with_jitter(self):
        """Delay should include jitter variance."""
        backoff = ExponentialBackoff(base_delay=1.0, jitter=0.1)
        delays = [backoff.get_delay() for _ in range(100)]

        # All delays should be between base and base + jitter
        assert all(1.0 <= d <= 1.1 for d in delays)
        # There should be some variance (not all identical)
        assert len(set(delays)) > 1

    def test_delay_increases_exponentially_with_failures(self):
        """Delay should double with each failure."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=100.0, jitter=0.0)

        # Simulate failures to increase failure_count
        backoff.failure_count = 0
        delay0 = backoff.get_delay()  # 1.0

        backoff.failure_count = 1
        delay1 = backoff.get_delay()  # 2.0

        backoff.failure_count = 2
        delay2 = backoff.get_delay()  # 4.0

        backoff.failure_count = 3
        delay3 = backoff.get_delay()  # 8.0

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0
        assert delay3 == 8.0

    def test_delay_capped_at_max(self):
        """Delay should not exceed max_delay."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=10.0, jitter=0.0)
        backoff.failure_count = 10  # Would be 1024 without cap

        delay = backoff.get_delay()
        assert delay == 10.0


class TestRecordFailure:
    """Tests for record_failure method."""

    def test_increments_failure_count(self):
        """Should increment failure count on each call."""
        backoff = ExponentialBackoff()
        assert backoff.failure_count == 0

        backoff.record_failure()
        assert backoff.failure_count == 1

        backoff.record_failure()
        assert backoff.failure_count == 2

    def test_returns_delay(self):
        """Should return a positive delay value."""
        backoff = ExponentialBackoff(jitter=0.0)
        delay = backoff.record_failure()
        assert delay > 0

    def test_delays_increase_with_failures(self):
        """Consecutive failures should increase delay."""
        backoff = ExponentialBackoff(base_delay=1.0, jitter=0.0)

        delay1 = backoff.record_failure()  # count=1, delay=2
        delay2 = backoff.record_failure()  # count=2, delay=4
        delay3 = backoff.record_failure()  # count=3, delay=8

        assert delay2 > delay1
        assert delay3 > delay2

    def test_delay_respects_max(self):
        """Delay should not exceed max even after many failures."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=5.0, jitter=0.0)

        for _ in range(10):
            delay = backoff.record_failure()

        assert delay <= 5.0


class TestReset:
    """Tests for reset method."""

    def test_resets_failure_count(self):
        """Reset should set failure count to zero."""
        backoff = ExponentialBackoff()
        backoff.record_failure()
        backoff.record_failure()
        assert backoff.failure_count == 2

        backoff.reset()
        assert backoff.failure_count == 0

    def test_reset_idempotent(self):
        """Reset on zero count should be safe."""
        backoff = ExponentialBackoff()
        assert backoff.failure_count == 0

        backoff.reset()  # Should not error
        assert backoff.failure_count == 0


class TestIsBackingOff:
    """Tests for is_backing_off property."""

    def test_false_initially(self):
        """Should be False when no failures recorded."""
        backoff = ExponentialBackoff()
        assert backoff.is_backing_off is False

    def test_true_after_failure(self):
        """Should be True after recording failure."""
        backoff = ExponentialBackoff()
        backoff.record_failure()
        assert backoff.is_backing_off is True

    def test_false_after_reset(self):
        """Should be False after reset."""
        backoff = ExponentialBackoff()
        backoff.record_failure()
        backoff.reset()
        assert backoff.is_backing_off is False


class TestCurrentFailureCount:
    """Tests for current_failure_count property."""

    def test_returns_current_count(self):
        """Should return the current failure count."""
        backoff = ExponentialBackoff()
        assert backoff.current_failure_count == 0

        backoff.record_failure()
        assert backoff.current_failure_count == 1

        backoff.record_failure()
        assert backoff.current_failure_count == 2


class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_concurrent_record_failure(self):
        """Multiple threads can record failures safely."""
        backoff = ExponentialBackoff()
        num_threads = 10
        iterations = 100

        def record_failures():
            for _ in range(iterations):
                backoff.record_failure()

        threads = [threading.Thread(target=record_failures) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All failures should be counted
        assert backoff.failure_count == num_threads * iterations

    def test_concurrent_reset_and_failure(self):
        """Concurrent reset and failure calls should not corrupt state."""
        backoff = ExponentialBackoff()
        stop_event = threading.Event()
        errors = []

        def failure_thread():
            for _ in range(50):  # Limited iterations
                if stop_event.is_set():
                    break
                try:
                    backoff.record_failure()
                except Exception as e:
                    errors.append(e)

        def reset_thread():
            for _ in range(50):  # Limited iterations
                if stop_event.is_set():
                    break
                try:
                    backoff.reset()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=failure_thread),
            threading.Thread(target=failure_thread),
            threading.Thread(target=reset_thread),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5.0)

        stop_event.set()

        # No errors should have occurred
        assert len(errors) == 0
        # State should be valid (either 0 or positive)
        assert backoff.failure_count >= 0

    def test_thread_pool_usage(self):
        """Should work correctly with thread pool executor."""
        backoff = ExponentialBackoff()

        def worker():
            backoff.record_failure()
            return backoff.current_failure_count

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(100)]
            results = [f.result() for f in futures]

        # All calls should have succeeded
        assert len(results) == 100
        # Final count should be 100
        assert backoff.failure_count == 100


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_base_delay(self):
        """Zero base delay should still work."""
        backoff = ExponentialBackoff(base_delay=0.0, jitter=0.0)
        delay = backoff.get_delay()
        assert delay == 0.0

    def test_zero_jitter(self):
        """Zero jitter should give deterministic delays."""
        backoff = ExponentialBackoff(base_delay=5.0, jitter=0.0)
        delays = [backoff.get_delay() for _ in range(10)]
        assert all(d == 5.0 for d in delays)

    def test_large_failure_count(self):
        """Large failure counts should be handled correctly."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=100.0, jitter=0.0)
        backoff.failure_count = 1000  # Would overflow without cap

        delay = backoff.get_delay()
        assert delay == 100.0  # Should be capped at max

    def test_very_small_jitter(self):
        """Very small jitter should still add variance."""
        backoff = ExponentialBackoff(base_delay=1.0, jitter=0.001)
        delays = [backoff.get_delay() for _ in range(100)]

        # All delays should be very close to 1.0
        assert all(1.0 <= d <= 1.001 for d in delays)

    def test_max_equals_base(self):
        """When max equals base, delay should always be base (+jitter)."""
        backoff = ExponentialBackoff(base_delay=5.0, max_delay=5.0, jitter=0.0)

        for _ in range(10):
            backoff.record_failure()

        delay = backoff.get_delay()
        assert delay == 5.0


class TestLogging:
    """Tests for logging behavior."""

    def test_failure_logs_info(self, caplog):
        """Record failure should log at INFO level."""
        import logging

        backoff = ExponentialBackoff()

        with caplog.at_level(logging.INFO):
            backoff.record_failure()

        assert "backoff_failure" in caplog.text
        assert "count=1" in caplog.text

    def test_reset_logs_debug(self, caplog):
        """Reset should log at DEBUG level."""
        import logging

        backoff = ExponentialBackoff()
        backoff.record_failure()

        with caplog.at_level(logging.DEBUG):
            backoff.reset()

        assert "backoff_reset" in caplog.text
