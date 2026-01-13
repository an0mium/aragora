"""Extended tests for resilience patterns.

Covers:
- Global circuit breaker registry
- Metrics collection
- SQLite persistence
- Thread safety
- Protected call context managers
- CircuitOpenError exception
- Pruning logic
"""

import asyncio
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    MAX_CIRCUIT_BREAKERS,
    STALE_THRESHOLD_SECONDS,
    _circuit_breakers,
    _circuit_breakers_lock,
    _prune_stale_circuit_breakers,
    cleanup_stale_persisted,
    get_circuit_breaker,
    get_circuit_breaker_metrics,
    get_circuit_breaker_status,
    init_circuit_breaker_persistence,
    load_circuit_breakers,
    persist_all_circuit_breakers,
    persist_circuit_breaker,
    prune_circuit_breakers,
    reset_all_circuit_breakers,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_global_registry():
    """Clear global circuit breaker registry before and after each test."""
    with _circuit_breakers_lock:
        _circuit_breakers.clear()
    yield
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


@pytest.fixture
def temp_db(tmp_path):
    """Temporary SQLite database for persistence tests."""
    db_path = str(tmp_path / "circuit_breakers.db")
    init_circuit_breaker_persistence(db_path)
    yield db_path


# =============================================================================
# CircuitOpenError Tests
# =============================================================================


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_error_has_circuit_name(self):
        """CircuitOpenError includes circuit name."""
        error = CircuitOpenError("my-circuit", 30.0)
        assert error.circuit_name == "my-circuit"

    def test_error_has_cooldown_remaining(self):
        """CircuitOpenError includes cooldown remaining."""
        error = CircuitOpenError("test", 45.5)
        assert error.cooldown_remaining == 45.5

    def test_error_message_format(self):
        """Error message includes circuit name and cooldown."""
        error = CircuitOpenError("api-service", 12.3)
        msg = str(error)
        assert "api-service" in msg
        assert "12.3" in msg
        assert "open" in msg.lower()

    def test_error_is_exception(self):
        """CircuitOpenError is a proper Exception."""
        error = CircuitOpenError("test", 10.0)
        assert isinstance(error, Exception)

    def test_error_can_be_raised_and_caught(self):
        """Can raise and catch CircuitOpenError."""
        with pytest.raises(CircuitOpenError) as exc_info:
            raise CircuitOpenError("test-circuit", 5.0)

        assert exc_info.value.circuit_name == "test-circuit"
        assert exc_info.value.cooldown_remaining == 5.0


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestGlobalRegistry:
    """Tests for global circuit breaker registry functions."""

    def test_get_circuit_breaker_creates_new(self):
        """get_circuit_breaker creates new circuit breaker."""
        cb = get_circuit_breaker("test-service")
        assert cb is not None
        assert isinstance(cb, CircuitBreaker)

    def test_get_circuit_breaker_returns_same_instance(self):
        """get_circuit_breaker returns same instance for same name."""
        cb1 = get_circuit_breaker("shared-service")
        cb2 = get_circuit_breaker("shared-service")
        assert cb1 is cb2

    def test_get_circuit_breaker_different_names(self):
        """Different names create different circuit breakers."""
        cb1 = get_circuit_breaker("service-a")
        cb2 = get_circuit_breaker("service-b")
        assert cb1 is not cb2

    def test_get_circuit_breaker_custom_threshold(self):
        """Can specify custom failure threshold."""
        cb = get_circuit_breaker("custom", failure_threshold=10)
        assert cb.failure_threshold == 10

    def test_get_circuit_breaker_custom_cooldown(self):
        """Can specify custom cooldown."""
        cb = get_circuit_breaker("custom-cooldown", cooldown_seconds=120.0)
        assert cb.cooldown_seconds == 120.0

    def test_get_circuit_breaker_updates_last_accessed(self):
        """Accessing updates _last_accessed timestamp."""
        cb = get_circuit_breaker("tracked")
        first_access = cb._last_accessed
        time.sleep(0.01)
        get_circuit_breaker("tracked")
        second_access = cb._last_accessed
        assert second_access > first_access

    def test_reset_all_circuit_breakers(self):
        """reset_all_circuit_breakers resets all registered breakers."""
        cb1 = get_circuit_breaker("service-1", failure_threshold=2)
        cb2 = get_circuit_breaker("service-2", failure_threshold=2)

        cb1.record_failure()
        cb1.record_failure()
        cb2.record_failure()

        assert cb1.is_open is True
        assert cb2.failures == 1

        reset_all_circuit_breakers()

        assert cb1.is_open is False
        assert cb1.failures == 0
        assert cb2.failures == 0

    def test_get_circuit_breaker_status(self):
        """get_circuit_breaker_status returns registry info."""
        get_circuit_breaker("status-test-1")
        get_circuit_breaker("status-test-2")

        status = get_circuit_breaker_status()

        assert "_registry_size" in status
        assert status["_registry_size"] == 2
        assert "status-test-1" in status
        assert "status-test-2" in status

    def test_status_includes_circuit_details(self):
        """Status includes details for each circuit."""
        cb = get_circuit_breaker("detail-test", failure_threshold=2)
        cb.record_failure()

        status = get_circuit_breaker_status()

        assert status["detail-test"]["failures"] == 1
        assert "status" in status["detail-test"]
        assert "last_accessed" in status["detail-test"]


# =============================================================================
# Metrics Tests
# =============================================================================


class TestCircuitBreakerMetrics:
    """Tests for get_circuit_breaker_metrics function."""

    def test_metrics_empty_registry(self):
        """Metrics for empty registry."""
        metrics = get_circuit_breaker_metrics()

        assert metrics["registry_size"] == 0
        assert metrics["summary"]["total"] == 0
        assert metrics["summary"]["open"] == 0
        assert metrics["summary"]["closed"] == 0
        assert metrics["health"]["status"] == "healthy"

    def test_metrics_with_closed_circuits(self):
        """Metrics with all circuits closed."""
        get_circuit_breaker("closed-1")
        get_circuit_breaker("closed-2")

        metrics = get_circuit_breaker_metrics()

        assert metrics["summary"]["total"] == 2
        assert metrics["summary"]["closed"] == 2
        assert metrics["summary"]["open"] == 0
        assert metrics["health"]["status"] == "healthy"

    def test_metrics_with_open_circuit(self):
        """Metrics with open circuit shows degraded health."""
        cb = get_circuit_breaker("open-circuit", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        metrics = get_circuit_breaker_metrics()

        assert metrics["summary"]["open"] == 1
        assert metrics["health"]["status"] == "degraded"
        assert "open-circuit" in metrics["health"]["open_circuits"]

    def test_metrics_critical_with_multiple_open(self):
        """Three or more open circuits shows critical health."""
        for i in range(3):
            cb = get_circuit_breaker(f"critical-{i}", failure_threshold=1)
            cb.record_failure()

        metrics = get_circuit_breaker_metrics()

        assert metrics["summary"]["open"] == 3
        assert metrics["health"]["status"] == "critical"

    def test_metrics_tracks_total_failures(self):
        """Metrics tracks total failures across circuits."""
        cb1 = get_circuit_breaker("fail-1", failure_threshold=10)
        cb2 = get_circuit_breaker("fail-2", failure_threshold=10)

        for _ in range(5):
            cb1.record_failure()
        for _ in range(3):
            cb2.record_failure()

        metrics = get_circuit_breaker_metrics()

        assert metrics["summary"]["total_failures"] == 8
        assert metrics["summary"]["circuits_with_failures"] == 2

    def test_metrics_high_failure_circuits(self):
        """Circuits at 50%+ threshold are flagged."""
        cb = get_circuit_breaker("high-fail", failure_threshold=4)
        cb.record_failure()
        cb.record_failure()  # 50% of threshold

        metrics = get_circuit_breaker_metrics()

        high_fail = metrics["health"]["high_failure_circuits"]
        assert len(high_fail) == 1
        assert high_fail[0]["name"] == "high-fail"
        assert high_fail[0]["percentage"] == 50.0

    def test_metrics_includes_timestamp(self):
        """Metrics include timestamp."""
        before = time.time()
        metrics = get_circuit_breaker_metrics()
        after = time.time()

        assert before <= metrics["timestamp"] <= after

    def test_metrics_per_circuit_details(self):
        """Metrics include per-circuit details."""
        cb = get_circuit_breaker("detailed", failure_threshold=5, cooldown_seconds=120.0)
        cb.record_failure()

        metrics = get_circuit_breaker_metrics()
        details = metrics["circuit_breakers"]["detailed"]

        assert details["status"] == "closed"
        assert details["failures"] == 1
        assert details["failure_threshold"] == 5
        assert details["cooldown_seconds"] == 120.0


# =============================================================================
# Pruning Tests
# =============================================================================


class TestCircuitBreakerPruning:
    """Tests for circuit breaker pruning functionality."""

    def test_prune_removes_stale_breakers(self):
        """Prune removes circuit breakers not accessed recently."""
        cb = get_circuit_breaker("stale-circuit")

        # Make it stale by backdating _last_accessed
        cb._last_accessed = time.time() - STALE_THRESHOLD_SECONDS - 1

        pruned = prune_circuit_breakers()

        assert pruned == 1
        with _circuit_breakers_lock:
            assert "stale-circuit" not in _circuit_breakers

    def test_prune_keeps_recent_breakers(self):
        """Prune keeps recently accessed circuit breakers."""
        get_circuit_breaker("recent-circuit")

        pruned = prune_circuit_breakers()

        assert pruned == 0
        with _circuit_breakers_lock:
            assert "recent-circuit" in _circuit_breakers

    def test_prune_mixed_stale_and_recent(self):
        """Prune only removes stale breakers."""
        fresh = get_circuit_breaker("fresh")
        stale = get_circuit_breaker("stale")

        stale._last_accessed = time.time() - STALE_THRESHOLD_SECONDS - 100

        pruned = prune_circuit_breakers()

        assert pruned == 1
        with _circuit_breakers_lock:
            assert "fresh" in _circuit_breakers
            assert "stale" not in _circuit_breakers

    def test_auto_prune_on_max_registry_size(self):
        """Auto-prune triggers when registry exceeds max size."""
        # Fill registry to max
        for i in range(MAX_CIRCUIT_BREAKERS):
            cb = get_circuit_breaker(f"cb-{i}")
            # Make half of them stale
            if i % 2 == 0:
                cb._last_accessed = time.time() - STALE_THRESHOLD_SECONDS - 1

        # This should trigger auto-prune when adding one more
        get_circuit_breaker("trigger")

        with _circuit_breakers_lock:
            # Stale ones should be removed - we pruned ~500 stale + added 1
            # So we should have ~501 entries (non-stale half + trigger)
            assert len(_circuit_breakers) <= MAX_CIRCUIT_BREAKERS

    def test_prune_empty_registry(self):
        """Prune on empty registry returns 0."""
        pruned = prune_circuit_breakers()
        assert pruned == 0

    def test_prune_all_stale(self):
        """Can prune all circuit breakers if all are stale."""
        for i in range(5):
            cb = get_circuit_breaker(f"all-stale-{i}")
            cb._last_accessed = time.time() - STALE_THRESHOLD_SECONDS - 1

        pruned = prune_circuit_breakers()

        assert pruned == 5
        with _circuit_breakers_lock:
            assert len(_circuit_breakers) == 0

    def test_prune_without_last_accessed(self):
        """Circuit breakers without _last_accessed are not pruned."""
        cb = get_circuit_breaker("no-timestamp")
        delattr(cb, "_last_accessed")

        pruned = prune_circuit_breakers()

        assert pruned == 0


# =============================================================================
# SQLite Persistence Tests
# =============================================================================


class TestSqlitePersistence:
    """Tests for SQLite-based circuit breaker persistence."""

    def test_init_creates_database(self, tmp_path):
        """init_circuit_breaker_persistence creates database file."""
        db_path = str(tmp_path / "new_db.db")
        init_circuit_breaker_persistence(db_path)

        assert Path(db_path).exists()

    def test_init_creates_table(self, temp_db):
        """Init creates circuit_breakers table."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='circuit_breakers'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_persist_single_circuit_breaker(self, temp_db):
        """Can persist a single circuit breaker."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()

        persist_circuit_breaker("test-cb", cb)

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT * FROM circuit_breakers WHERE name = ?", ("test-cb",))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "test-cb"

    def test_persist_all_circuit_breakers(self, temp_db):
        """persist_all_circuit_breakers persists entire registry."""
        get_circuit_breaker("persist-1")
        get_circuit_breaker("persist-2")
        get_circuit_breaker("persist-3")

        count = persist_all_circuit_breakers()

        assert count == 3

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT COUNT(*) FROM circuit_breakers")
        db_count = cursor.fetchone()[0]
        conn.close()

        assert db_count == 3

    def test_load_circuit_breakers(self, temp_db):
        """load_circuit_breakers restores from database."""
        # Create and persist
        cb1 = get_circuit_breaker("load-1", failure_threshold=5)
        cb2 = get_circuit_breaker("load-2", failure_threshold=7)
        cb1.record_failure()
        cb1.record_failure()
        persist_all_circuit_breakers()

        # Clear registry
        with _circuit_breakers_lock:
            _circuit_breakers.clear()

        # Load
        count = load_circuit_breakers()

        assert count == 2
        with _circuit_breakers_lock:
            assert "load-1" in _circuit_breakers
            assert "load-2" in _circuit_breakers

    def test_persist_updates_existing(self, temp_db):
        """Persisting same name updates existing record."""
        cb = get_circuit_breaker("update-test")
        persist_circuit_breaker("update-test", cb)

        # Update state
        cb.record_failure()
        cb.record_failure()
        persist_circuit_breaker("update-test", cb)

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM circuit_breakers WHERE name = ?", ("update-test",)
        )
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_cleanup_stale_persisted(self, temp_db):
        """cleanup_stale_persisted removes old entries."""
        cb = get_circuit_breaker("old-entry")
        persist_circuit_breaker("old-entry", cb)

        # Manually backdate the entry
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "UPDATE circuit_breakers SET updated_at = ? WHERE name = ?",
            ("2020-01-01T00:00:00", "old-entry"),
        )
        conn.commit()
        conn.close()

        deleted = cleanup_stale_persisted(max_age_hours=1.0)

        assert deleted == 1

    def test_cleanup_keeps_recent(self, temp_db):
        """cleanup_stale_persisted keeps recent entries."""
        cb = get_circuit_breaker("recent-entry")
        persist_circuit_breaker("recent-entry", cb)

        deleted = cleanup_stale_persisted(max_age_hours=1.0)

        assert deleted == 0

    def test_persist_without_init_is_noop(self):
        """Persisting without init does not raise."""
        # Reset _DB_PATH
        import aragora.resilience as r

        original = r._DB_PATH
        r._DB_PATH = None

        try:
            cb = CircuitBreaker()
            persist_circuit_breaker("test", cb)  # Should not raise
        finally:
            r._DB_PATH = original

    def test_load_without_init_returns_zero(self):
        """Loading without init returns 0."""
        import aragora.resilience as r

        original = r._DB_PATH
        r._DB_PATH = None

        try:
            count = load_circuit_breakers()
            assert count == 0
        finally:
            r._DB_PATH = original

    def test_load_handles_malformed_json(self, temp_db):
        """Loading handles malformed JSON gracefully."""
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO circuit_breakers (name, state_json, failure_threshold, cooldown_seconds, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("malformed", "{invalid json", 3, 60.0, "2024-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        # Should not raise, just skip malformed
        count = load_circuit_breakers()
        assert count == 0


# =============================================================================
# Protected Call Context Manager Tests
# =============================================================================


class TestProtectedCallAsync:
    """Tests for async protected_call context manager."""

    @pytest.mark.asyncio
    async def test_protected_call_success(self):
        """Successful call records success."""
        cb = CircuitBreaker(failure_threshold=3)

        async with cb.protected_call():
            pass  # Success

        assert cb.failures == 0

    @pytest.mark.asyncio
    async def test_protected_call_failure(self):
        """Exception records failure."""
        cb = CircuitBreaker(failure_threshold=3)

        with pytest.raises(ValueError):
            async with cb.protected_call():
                raise ValueError("test error")

        assert cb.failures == 1

    @pytest.mark.asyncio
    async def test_protected_call_open_circuit_raises(self):
        """Open circuit raises CircuitOpenError."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        cb.record_failure()

        with pytest.raises(CircuitOpenError) as exc_info:
            async with cb.protected_call(circuit_name="test-circuit"):
                pass

        assert exc_info.value.circuit_name == "test-circuit"
        assert exc_info.value.cooldown_remaining > 0

    @pytest.mark.asyncio
    async def test_protected_call_with_entity(self):
        """protected_call works with entity parameter."""
        cb = CircuitBreaker(failure_threshold=2)

        # Fail entity-1
        with pytest.raises(RuntimeError):
            async with cb.protected_call("entity-1"):
                raise RuntimeError("fail")

        assert cb._failures["entity-1"] == 1

        # entity-2 should be unaffected
        async with cb.protected_call("entity-2"):
            pass
        assert "entity-2" not in cb._failures or cb._failures["entity-2"] == 0

    @pytest.mark.asyncio
    async def test_protected_call_cancelled_not_recorded(self):
        """asyncio.CancelledError is not recorded as failure."""
        cb = CircuitBreaker(failure_threshold=3)

        async def cancellable():
            async with cb.protected_call():
                raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await cancellable()

        # Cancellation should not count as failure
        assert cb.failures == 0

    @pytest.mark.asyncio
    async def test_protected_call_cooldown_remaining_calculation(self):
        """CircuitOpenError has correct cooldown remaining."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10.0)
        cb.record_failure()

        time.sleep(0.1)  # Wait a bit

        with pytest.raises(CircuitOpenError) as exc_info:
            async with cb.protected_call():
                pass

        # Should be close to 10.0 - 0.1 = 9.9
        assert 9.0 < exc_info.value.cooldown_remaining < 10.0


class TestProtectedCallSync:
    """Tests for sync protected_call_sync context manager."""

    def test_protected_call_sync_success(self):
        """Successful sync call records success."""
        cb = CircuitBreaker(failure_threshold=3)

        with cb.protected_call_sync():
            pass

        assert cb.failures == 0

    def test_protected_call_sync_failure(self):
        """Exception records failure."""
        cb = CircuitBreaker(failure_threshold=3)

        with pytest.raises(ValueError):
            with cb.protected_call_sync():
                raise ValueError("sync error")

        assert cb.failures == 1

    def test_protected_call_sync_open_circuit_raises(self):
        """Open circuit raises CircuitOpenError."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        cb.record_failure()

        with pytest.raises(CircuitOpenError):
            with cb.protected_call_sync():
                pass

    def test_protected_call_sync_with_entity(self):
        """Sync context manager works with entity."""
        cb = CircuitBreaker(failure_threshold=2)

        with pytest.raises(RuntimeError):
            with cb.protected_call_sync("my-entity"):
                raise RuntimeError("error")

        assert cb._failures["my-entity"] == 1

    def test_protected_call_sync_circuit_name(self):
        """circuit_name appears in error."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        cb.record_failure()

        with pytest.raises(CircuitOpenError) as exc_info:
            with cb.protected_call_sync(circuit_name="my-api"):
                pass

        assert exc_info.value.circuit_name == "my-api"


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_get_circuit_breaker(self):
        """Concurrent get_circuit_breaker is safe."""
        results = []

        def get_cb(name):
            for _ in range(100):
                cb = get_circuit_breaker(name)
                results.append(cb)

        threads = [threading.Thread(target=get_cb, args=(f"thread-cb-{i % 3}",)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All accesses should succeed
        assert len(results) == 1000

    def test_concurrent_failure_recording(self):
        """Concurrent failure recording is safe."""
        cb = get_circuit_breaker("concurrent-fail", failure_threshold=1000)

        def record_failures():
            for _ in range(100):
                cb.record_failure()

        threads = [threading.Thread(target=record_failures) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded all failures
        assert cb.failures == 1000

    def test_concurrent_status_check(self):
        """Concurrent status checks are safe."""
        cb = get_circuit_breaker("status-check")
        errors = []

        def check_status():
            try:
                for _ in range(100):
                    status = cb.get_status()
                    assert status in ["closed", "open", "half-open"]
            except Exception as e:
                errors.append(e)

        def mutate():
            for _ in range(100):
                cb.record_failure()

        threads = [
            threading.Thread(target=check_status),
            threading.Thread(target=mutate),
            threading.Thread(target=check_status),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_metrics_collection(self):
        """Concurrent metrics collection is safe."""
        # Create some circuit breakers
        for i in range(10):
            get_circuit_breaker(f"metrics-{i}")

        errors = []

        def collect_metrics():
            try:
                for _ in range(50):
                    metrics = get_circuit_breaker_metrics()
                    assert "summary" in metrics
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=collect_metrics) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_prune(self):
        """Concurrent pruning is safe."""
        # Create many stale circuit breakers
        for i in range(50):
            cb = get_circuit_breaker(f"prune-{i}")
            cb._last_accessed = time.time() - STALE_THRESHOLD_SECONDS - 1

        results = []

        def prune():
            result = prune_circuit_breakers()
            results.append(result)

        threads = [threading.Thread(target=prune) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total pruned should be 50
        assert sum(results) == 50

    def test_thread_pool_executor_usage(self):
        """Works correctly with ThreadPoolExecutor."""
        cb = get_circuit_breaker("executor-test", failure_threshold=1000)

        def work(n):
            for _ in range(100):
                cb.record_failure()
            return n

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(work, i) for i in range(10)]
            results = [f.result() for f in futures]

        assert len(results) == 10
        assert cb.failures == 1000


# =============================================================================
# Half-Open Behavior Extended Tests
# =============================================================================


class TestHalfOpenBehaviorExtended:
    """Extended tests for half-open state behavior."""

    def test_half_open_allows_trial_request(self):
        """Half-open state allows trial requests."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()

        assert cb.can_proceed() is False

        time.sleep(0.02)

        assert cb.can_proceed() is True
        assert cb.get_status() == "closed"  # Auto-closes on can_proceed

    def test_entity_half_open_success_threshold(self):
        """Entity mode requires multiple successes to close."""
        cb = CircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.01,
            half_open_success_threshold=3,
        )
        cb.record_failure("entity")

        time.sleep(0.02)
        assert cb.get_status("entity") == "half-open"

        cb.record_success("entity")
        assert cb.get_status("entity") == "half-open"

        cb.record_success("entity")
        assert cb.get_status("entity") == "half-open"

        cb.record_success("entity")
        assert cb.get_status("entity") == "closed"

    def test_half_open_failure_increments_count(self):
        """Failure in half-open state increments failure count."""
        cb = CircuitBreaker(
            failure_threshold=2,
            cooldown_seconds=0.01,
        )
        cb.record_failure("entity")
        cb.record_failure("entity")  # Opens circuit

        time.sleep(0.02)
        assert cb.get_status("entity") == "half-open"

        # Failure in half-open doesn't reset timer (cooldown already elapsed)
        # but it does increment the failure count
        cb.record_failure("entity")
        assert cb._failures["entity"] == 3

    def test_single_mode_immediate_close(self):
        """Single-entity mode closes immediately on success."""
        cb = CircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.01,
            half_open_success_threshold=3,  # Ignored in single mode
        )
        cb.record_failure()

        time.sleep(0.02)
        cb.record_success()

        assert cb.get_status() == "closed"
        assert cb.failures == 0


# =============================================================================
# From Dict Extended Tests
# =============================================================================


class TestFromDictExtended:
    """Extended tests for from_dict state restoration."""

    def test_from_dict_restores_single_mode(self):
        """from_dict restores single mode state."""
        data = {
            "single_mode": {
                "failures": 3,
                "is_open": True,
                "open_for_seconds": 10,
            }
        }

        cb = CircuitBreaker.from_dict(data, cooldown_seconds=60.0)

        assert cb._single_failures == 3
        assert cb.is_open is True

    def test_from_dict_expired_single_mode(self):
        """Expired single mode cooldown is not restored."""
        data = {
            "single_mode": {
                "failures": 3,
                "is_open": True,
                "open_for_seconds": 100,  # Exceeds cooldown
            }
        }

        cb = CircuitBreaker.from_dict(data, cooldown_seconds=60.0)

        assert cb._single_failures == 3
        assert cb.is_open is False

    def test_from_dict_legacy_format(self):
        """from_dict handles legacy format without single_mode."""
        data = {
            "failures": {"agent-1": 2},
            "cooldowns": {"agent-1": 5.0},
        }

        cb = CircuitBreaker.from_dict(data, cooldown_seconds=60.0)

        assert cb._failures["agent-1"] == 2
        assert not cb.is_available("agent-1")

    def test_from_dict_preserves_kwargs(self):
        """from_dict uses provided kwargs."""
        data = {}

        cb = CircuitBreaker.from_dict(
            data,
            failure_threshold=10,
            cooldown_seconds=120.0,
            half_open_success_threshold=5,
        )

        assert cb.failure_threshold == 10
        assert cb.cooldown_seconds == 120.0
        assert cb.half_open_success_threshold == 5

    def test_roundtrip_to_from_dict(self):
        """State survives to_dict/from_dict roundtrip."""
        original = CircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)
        original.record_failure()
        original.record_failure()
        original.record_failure("entity-1")
        original.record_failure("entity-1")

        data = original.to_dict()
        restored = CircuitBreaker.from_dict(
            data,
            failure_threshold=original.failure_threshold,
            cooldown_seconds=original.cooldown_seconds,
        )

        assert restored._single_failures == original._single_failures
        assert restored._failures.get("entity-1") == original._failures.get("entity-1")

    def test_from_dict_empty_data(self):
        """from_dict handles empty data."""
        cb = CircuitBreaker.from_dict({})

        assert cb.failures == 0
        assert cb.is_open is False
        assert len(cb._failures) == 0


# =============================================================================
# State Transitions Tests
# =============================================================================


class TestStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_closed_to_open_on_threshold(self):
        """Circuit transitions closed -> open at threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        assert cb.get_status() == "closed"

        cb.record_failure()
        assert cb.get_status() == "closed"

        cb.record_failure()
        assert cb.get_status() == "closed"

        cb.record_failure()
        assert cb.get_status() == "open"

    def test_open_to_half_open_on_cooldown(self):
        """Circuit transitions open -> half-open after cooldown."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()

        assert cb.get_status() == "open"

        time.sleep(0.02)

        assert cb.get_status() == "half-open"

    def test_half_open_to_closed_on_success(self):
        """Circuit transitions half-open -> closed on success."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()

        time.sleep(0.02)
        assert cb.get_status() == "half-open"

        cb.record_success()
        assert cb.get_status() == "closed"

    def test_success_resets_failure_count(self):
        """Success in closed state resets failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 2

        cb.record_success()
        assert cb.failures == 0

    def test_record_failure_returns_true_on_open(self):
        """record_failure returns True when circuit opens."""
        cb = CircuitBreaker(failure_threshold=2)

        assert cb.record_failure() is False
        assert cb.record_failure() is True  # Circuit just opened

    def test_record_failure_returns_false_if_already_open(self):
        """record_failure returns False if circuit already open."""
        cb = CircuitBreaker(failure_threshold=2)

        cb.record_failure()
        cb.record_failure()  # Opens

        assert cb.record_failure() is False  # Already open

    def test_is_open_setter_resets_state(self):
        """Setting is_open = False resets state."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        assert cb.is_open is True
        assert cb.failures == 2

        cb.is_open = False

        assert cb.is_open is False
        assert cb.failures == 0

    def test_is_open_setter_opens_circuit(self):
        """Setting is_open = True opens circuit."""
        cb = CircuitBreaker()

        cb.is_open = True

        assert cb.is_open is True
        assert cb.can_proceed() is False
