"""
Tests for the health monitoring module.

Tests cover:
- HealthStatus creation, to_dict(), from_dict() round-trip
- HealthStatus.from_dict with string datetime and datetime object
- HealthReport creation, to_dict(), overall_healthy logic
- HealthChecker initial state, record_success, record_failure
- HealthChecker state transitions: healthy -> unhealthy -> healthy
- HealthChecker latency window averaging and pruning
- HealthChecker set_metadata and reset
- HealthRegistry register, get, get_or_create, unregister
- HealthRegistry get_report with various component states
- HealthRegistry registered_components property
- get_global_health_registry singleton behavior
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from aragora.resilience.health import (
    HealthChecker,
    HealthRegistry,
    HealthReport,
    HealthStatus,
    get_global_health_registry,
)


# ============================================================================
# HealthStatus Tests
# ============================================================================


class TestHealthStatus:
    """Tests for the HealthStatus dataclass."""

    def test_creation_minimal(self):
        """Test creating a HealthStatus with only required fields."""
        now = datetime.now(timezone.utc)
        status = HealthStatus(healthy=True, last_check=now)

        assert status.healthy is True
        assert status.last_check == now
        assert status.consecutive_failures == 0
        assert status.last_error is None
        assert status.latency_ms is None
        assert status.metadata == {}

    def test_creation_full(self):
        """Test creating a HealthStatus with all fields populated."""
        now = datetime.now(timezone.utc)
        meta = {"version": "1.2.3", "region": "us-east"}
        status = HealthStatus(
            healthy=False,
            last_check=now,
            consecutive_failures=5,
            last_error="Connection refused",
            latency_ms=42.5,
            metadata=meta,
        )

        assert status.healthy is False
        assert status.last_check == now
        assert status.consecutive_failures == 5
        assert status.last_error == "Connection refused"
        assert status.latency_ms == 42.5
        assert status.metadata == meta

    def test_to_dict(self):
        """Test converting HealthStatus to a dictionary."""
        now = datetime.now(timezone.utc)
        status = HealthStatus(
            healthy=True,
            last_check=now,
            consecutive_failures=2,
            last_error="timeout",
            latency_ms=10.5,
            metadata={"key": "value"},
        )
        d = status.to_dict()

        assert d["healthy"] is True
        assert d["last_check"] == now.isoformat()
        assert d["consecutive_failures"] == 2
        assert d["last_error"] == "timeout"
        assert d["latency_ms"] == 10.5
        assert d["metadata"] == {"key": "value"}

    def test_to_dict_none_fields(self):
        """Test to_dict when optional fields are None."""
        now = datetime.now(timezone.utc)
        status = HealthStatus(healthy=True, last_check=now)
        d = status.to_dict()

        assert d["last_error"] is None
        assert d["latency_ms"] is None
        assert d["metadata"] == {}

    def test_from_dict_with_string_datetime(self):
        """Test from_dict when last_check is an ISO format string."""
        now = datetime.now(timezone.utc)
        data = {
            "healthy": False,
            "last_check": now.isoformat(),
            "consecutive_failures": 3,
            "last_error": "disk full",
            "latency_ms": 100.0,
            "metadata": {"disk": "sda1"},
        }
        status = HealthStatus.from_dict(data)

        assert status.healthy is False
        assert status.last_check == now
        assert status.consecutive_failures == 3
        assert status.last_error == "disk full"
        assert status.latency_ms == 100.0
        assert status.metadata == {"disk": "sda1"}

    def test_from_dict_with_datetime_object(self):
        """Test from_dict when last_check is already a datetime object."""
        now = datetime.now(timezone.utc)
        data = {
            "healthy": True,
            "last_check": now,
            "consecutive_failures": 0,
        }
        status = HealthStatus.from_dict(data)

        assert status.last_check == now
        assert status.healthy is True

    def test_from_dict_optional_fields_missing(self):
        """Test from_dict when optional fields are absent uses defaults."""
        now = datetime.now(timezone.utc)
        data = {
            "healthy": True,
            "last_check": now.isoformat(),
        }
        status = HealthStatus.from_dict(data)

        assert status.consecutive_failures == 0
        assert status.last_error is None
        assert status.latency_ms is None
        assert status.metadata == {}

    def test_round_trip_to_dict_from_dict(self):
        """Test that to_dict and from_dict form a lossless round-trip."""
        now = datetime.now(timezone.utc)
        original = HealthStatus(
            healthy=False,
            last_check=now,
            consecutive_failures=7,
            last_error="OOM killed",
            latency_ms=250.75,
            metadata={"pid": 1234, "host": "node-3"},
        )
        d = original.to_dict()
        restored = HealthStatus.from_dict(d)

        assert restored.healthy == original.healthy
        assert restored.last_check == original.last_check
        assert restored.consecutive_failures == original.consecutive_failures
        assert restored.last_error == original.last_error
        assert restored.latency_ms == original.latency_ms
        assert restored.metadata == original.metadata

    def test_metadata_default_factory_independence(self):
        """Test that each HealthStatus gets its own metadata dict."""
        s1 = HealthStatus(healthy=True, last_check=datetime.now(timezone.utc))
        s2 = HealthStatus(healthy=True, last_check=datetime.now(timezone.utc))
        s1.metadata["key"] = "value"

        assert "key" not in s2.metadata


# ============================================================================
# HealthReport Tests
# ============================================================================


class TestHealthReport:
    """Tests for the HealthReport dataclass."""

    def test_creation(self):
        """Test creating a HealthReport."""
        now = datetime.now(timezone.utc)
        components = {
            "db": HealthStatus(healthy=True, last_check=now),
            "cache": HealthStatus(healthy=False, last_check=now, last_error="down"),
        }
        report = HealthReport(
            overall_healthy=False,
            components=components,
            checked_at=now,
            summary="1/2 components unhealthy: cache",
        )

        assert report.overall_healthy is False
        assert len(report.components) == 2
        assert report.checked_at == now
        assert "cache" in report.summary

    def test_creation_default_summary(self):
        """Test that summary defaults to empty string."""
        now = datetime.now(timezone.utc)
        report = HealthReport(
            overall_healthy=True,
            components={},
            checked_at=now,
        )
        assert report.summary == ""

    def test_to_dict(self):
        """Test converting HealthReport to a dictionary."""
        now = datetime.now(timezone.utc)
        components = {
            "api": HealthStatus(healthy=True, last_check=now, latency_ms=5.0),
        }
        report = HealthReport(
            overall_healthy=True,
            components=components,
            checked_at=now,
            summary="All 1 components healthy",
        )
        d = report.to_dict()

        assert d["overall_healthy"] is True
        assert d["checked_at"] == now.isoformat()
        assert d["summary"] == "All 1 components healthy"
        assert "api" in d["components"]
        assert d["components"]["api"]["healthy"] is True
        assert d["components"]["api"]["latency_ms"] == 5.0

    def test_to_dict_empty_components(self):
        """Test to_dict with no components."""
        now = datetime.now(timezone.utc)
        report = HealthReport(
            overall_healthy=True,
            components={},
            checked_at=now,
            summary="No components registered",
        )
        d = report.to_dict()

        assert d["components"] == {}
        assert d["overall_healthy"] is True

    def test_to_dict_multiple_components(self):
        """Test to_dict serializes all component statuses."""
        now = datetime.now(timezone.utc)
        components = {
            "db": HealthStatus(healthy=True, last_check=now),
            "cache": HealthStatus(healthy=False, last_check=now, consecutive_failures=5),
            "queue": HealthStatus(healthy=True, last_check=now, latency_ms=2.0),
        }
        report = HealthReport(
            overall_healthy=False,
            components=components,
            checked_at=now,
        )
        d = report.to_dict()

        assert len(d["components"]) == 3
        assert d["components"]["db"]["healthy"] is True
        assert d["components"]["cache"]["healthy"] is False
        assert d["components"]["cache"]["consecutive_failures"] == 5
        assert d["components"]["queue"]["latency_ms"] == 2.0


# ============================================================================
# HealthChecker Tests
# ============================================================================


class TestHealthCheckerInit:
    """Tests for HealthChecker initialization."""

    def test_default_parameters(self):
        """Test HealthChecker is created with correct defaults."""
        checker = HealthChecker("test-service")

        assert checker.name == "test-service"
        assert checker.failure_threshold == 3
        assert checker.recovery_threshold == 2
        assert checker.latency_window == 10

    def test_custom_parameters(self):
        """Test HealthChecker with custom thresholds."""
        checker = HealthChecker(
            "custom",
            failure_threshold=5,
            recovery_threshold=3,
            latency_window=20,
        )

        assert checker.failure_threshold == 5
        assert checker.recovery_threshold == 3
        assert checker.latency_window == 20

    def test_initial_state_is_healthy(self):
        """Test that a new HealthChecker starts healthy."""
        checker = HealthChecker("svc")
        status = checker.get_status()

        assert status.healthy is True
        assert status.consecutive_failures == 0
        assert status.last_error is None
        assert status.latency_ms is None
        assert status.metadata == {}

    def test_initial_last_check_is_set(self):
        """Test that last_check is set at creation time."""
        before = datetime.now(timezone.utc)
        checker = HealthChecker("svc")
        after = datetime.now(timezone.utc)

        status = checker.get_status()
        assert before <= status.last_check <= after


class TestHealthCheckerRecordSuccess:
    """Tests for HealthChecker.record_success()."""

    def test_record_success_without_latency(self):
        """Test recording success without providing latency."""
        checker = HealthChecker("svc")
        checker.record_success()

        status = checker.get_status()
        assert status.healthy is True
        assert status.latency_ms is None

    def test_record_success_with_latency(self):
        """Test recording success with latency."""
        checker = HealthChecker("svc")
        checker.record_success(latency_ms=15.5)

        status = checker.get_status()
        assert status.latency_ms == 15.5

    def test_record_success_resets_consecutive_failures(self):
        """Test that a success resets the consecutive failure counter."""
        checker = HealthChecker("svc", failure_threshold=5)
        checker.record_failure("err1")
        checker.record_failure("err2")

        status = checker.get_status()
        assert status.consecutive_failures == 2

        checker.record_success()
        status = checker.get_status()
        assert status.consecutive_failures == 0

    def test_record_success_updates_last_check(self):
        """Test that record_success updates the last_check timestamp."""
        checker = HealthChecker("svc")
        before = datetime.now(timezone.utc)
        checker.record_success()
        after = datetime.now(timezone.utc)

        status = checker.get_status()
        assert before <= status.last_check <= after

    def test_record_success_averages_latency(self):
        """Test that multiple latency recordings produce an average."""
        checker = HealthChecker("svc")
        checker.record_success(latency_ms=10.0)
        checker.record_success(latency_ms=20.0)
        checker.record_success(latency_ms=30.0)

        status = checker.get_status()
        assert status.latency_ms == pytest.approx(20.0)


class TestHealthCheckerRecordFailure:
    """Tests for HealthChecker.record_failure()."""

    def test_record_failure_increments_counter(self):
        """Test that each failure increments consecutive_failures."""
        checker = HealthChecker("svc")
        checker.record_failure("err")

        status = checker.get_status()
        assert status.consecutive_failures == 1

        checker.record_failure("err2")
        status = checker.get_status()
        assert status.consecutive_failures == 2

    def test_record_failure_sets_last_error(self):
        """Test that failure records the error message."""
        checker = HealthChecker("svc")
        checker.record_failure("Connection timeout")

        status = checker.get_status()
        assert status.last_error == "Connection timeout"

    def test_record_failure_updates_last_error(self):
        """Test that last_error is updated to the most recent error."""
        checker = HealthChecker("svc")
        checker.record_failure("first error")
        checker.record_failure("second error")

        status = checker.get_status()
        assert status.last_error == "second error"

    def test_record_failure_none_error(self):
        """Test recording failure with no error message."""
        checker = HealthChecker("svc")
        checker.record_failure()

        status = checker.get_status()
        assert status.consecutive_failures == 1
        assert status.last_error is None

    def test_record_failure_resets_consecutive_successes(self):
        """Test that a failure resets the success counter, preventing premature recovery."""
        checker = HealthChecker("svc", failure_threshold=2, recovery_threshold=3)
        # Make it unhealthy
        checker.record_failure("e1")
        checker.record_failure("e2")
        assert checker.get_status().healthy is False

        # Record 2 successes (not enough for recovery_threshold=3)
        checker.record_success()
        checker.record_success()
        assert checker.get_status().healthy is False

        # A failure resets the success counter
        checker.record_failure("e3")

        # Now need 3 fresh successes to recover
        checker.record_success()
        checker.record_success()
        assert checker.get_status().healthy is False  # Still 2, need 3

        checker.record_success()
        assert checker.get_status().healthy is True  # Now 3 consecutive

    def test_record_failure_updates_last_check(self):
        """Test that record_failure updates the last_check timestamp."""
        checker = HealthChecker("svc")
        before = datetime.now(timezone.utc)
        checker.record_failure("err")
        after = datetime.now(timezone.utc)

        status = checker.get_status()
        assert before <= status.last_check <= after


class TestHealthCheckerStateTransitions:
    """Tests for HealthChecker state transition logic."""

    def test_healthy_to_unhealthy_at_threshold(self):
        """Test that checker becomes unhealthy exactly at failure_threshold."""
        checker = HealthChecker("svc", failure_threshold=3)

        checker.record_failure("e1")
        assert checker.get_status().healthy is True  # 1 < 3

        checker.record_failure("e2")
        assert checker.get_status().healthy is True  # 2 < 3

        checker.record_failure("e3")
        assert checker.get_status().healthy is False  # 3 >= 3

    def test_stays_unhealthy_after_more_failures(self):
        """Test that additional failures after threshold keep it unhealthy."""
        checker = HealthChecker("svc", failure_threshold=2)
        checker.record_failure("e1")
        checker.record_failure("e2")
        assert checker.get_status().healthy is False

        checker.record_failure("e3")
        checker.record_failure("e4")
        assert checker.get_status().healthy is False
        assert checker.get_status().consecutive_failures == 4

    def test_unhealthy_to_healthy_at_recovery_threshold(self):
        """Test recovery after recovery_threshold consecutive successes."""
        checker = HealthChecker("svc", failure_threshold=2, recovery_threshold=3)

        # Become unhealthy
        checker.record_failure("e1")
        checker.record_failure("e2")
        assert checker.get_status().healthy is False

        # Not enough successes yet
        checker.record_success()
        assert checker.get_status().healthy is False  # 1 < 3

        checker.record_success()
        assert checker.get_status().healthy is False  # 2 < 3

        checker.record_success()
        assert checker.get_status().healthy is True  # 3 >= 3

    def test_recovery_clears_last_error(self):
        """Test that recovery clears the last_error field."""
        checker = HealthChecker("svc", failure_threshold=1, recovery_threshold=1)
        checker.record_failure("bad thing happened")
        assert checker.get_status().healthy is False
        assert checker.get_status().last_error == "bad thing happened"

        checker.record_success()
        assert checker.get_status().healthy is True
        assert checker.get_status().last_error is None

    def test_single_success_insufficient_for_recovery(self):
        """Test that one success does not recover when recovery_threshold > 1."""
        checker = HealthChecker("svc", failure_threshold=1, recovery_threshold=2)
        checker.record_failure("err")
        assert checker.get_status().healthy is False

        checker.record_success()
        assert checker.get_status().healthy is False

    def test_failure_threshold_one(self):
        """Test that a single failure marks unhealthy when threshold is 1."""
        checker = HealthChecker("svc", failure_threshold=1)
        checker.record_failure("boom")
        assert checker.get_status().healthy is False

    def test_recovery_threshold_one(self):
        """Test that a single success recovers when threshold is 1."""
        checker = HealthChecker("svc", failure_threshold=1, recovery_threshold=1)
        checker.record_failure("err")
        assert checker.get_status().healthy is False

        checker.record_success()
        assert checker.get_status().healthy is True

    def test_interleaved_failures_do_not_accumulate(self):
        """Test that a success between failures resets the failure count."""
        checker = HealthChecker("svc", failure_threshold=3)

        checker.record_failure("e1")
        checker.record_failure("e2")
        checker.record_success()  # Resets consecutive failures
        checker.record_failure("e3")
        checker.record_failure("e4")

        # Only 2 consecutive failures, not 4
        assert checker.get_status().healthy is True
        assert checker.get_status().consecutive_failures == 2

    def test_full_cycle_healthy_unhealthy_healthy(self):
        """Test a complete health cycle: healthy -> unhealthy -> healthy."""
        checker = HealthChecker("svc", failure_threshold=2, recovery_threshold=2)

        # Start healthy
        assert checker.get_status().healthy is True

        # Become unhealthy
        checker.record_failure("err1")
        checker.record_failure("err2")
        assert checker.get_status().healthy is False

        # Recover
        checker.record_success()
        checker.record_success()
        assert checker.get_status().healthy is True

        # Become unhealthy again
        checker.record_failure("err3")
        checker.record_failure("err4")
        assert checker.get_status().healthy is False


class TestHealthCheckerLatency:
    """Tests for HealthChecker latency tracking."""

    def test_no_latency_when_none_recorded(self):
        """Test that latency is None when no latency values are recorded."""
        checker = HealthChecker("svc")
        checker.record_success()

        assert checker.get_status().latency_ms is None

    def test_single_latency_value(self):
        """Test latency with a single value."""
        checker = HealthChecker("svc")
        checker.record_success(latency_ms=42.0)

        assert checker.get_status().latency_ms == 42.0

    def test_average_latency(self):
        """Test latency is averaged over recorded values."""
        checker = HealthChecker("svc")
        checker.record_success(latency_ms=10.0)
        checker.record_success(latency_ms=20.0)
        checker.record_success(latency_ms=30.0)

        assert checker.get_status().latency_ms == pytest.approx(20.0)

    def test_latency_window_pruning(self):
        """Test that latency values beyond the window are pruned."""
        checker = HealthChecker("svc", latency_window=3)

        # Fill window with low values
        checker.record_success(latency_ms=10.0)
        checker.record_success(latency_ms=10.0)
        checker.record_success(latency_ms=10.0)

        # These should push out the old values
        checker.record_success(latency_ms=100.0)
        checker.record_success(latency_ms=100.0)
        checker.record_success(latency_ms=100.0)

        # Window should now contain only [100, 100, 100]
        assert checker.get_status().latency_ms == pytest.approx(100.0)

    def test_latency_window_partial_fill(self):
        """Test latency when window is not full."""
        checker = HealthChecker("svc", latency_window=10)
        checker.record_success(latency_ms=5.0)
        checker.record_success(latency_ms=15.0)

        # Only 2 values, average is 10.0
        assert checker.get_status().latency_ms == pytest.approx(10.0)

    def test_success_without_latency_does_not_affect_average(self):
        """Test that successes without latency don't pollute the latency window."""
        checker = HealthChecker("svc")
        checker.record_success(latency_ms=20.0)
        checker.record_success()  # No latency
        checker.record_success(latency_ms=40.0)

        # Should average only the two recorded values: (20 + 40) / 2 = 30
        assert checker.get_status().latency_ms == pytest.approx(30.0)

    def test_latency_window_exactly_at_boundary(self):
        """Test behavior when exactly latency_window entries are recorded."""
        checker = HealthChecker("svc", latency_window=3)
        checker.record_success(latency_ms=10.0)
        checker.record_success(latency_ms=20.0)
        checker.record_success(latency_ms=30.0)

        assert checker.get_status().latency_ms == pytest.approx(20.0)

    def test_latency_window_one_beyond_boundary(self):
        """Test that the oldest entry is pruned when exceeding the window."""
        checker = HealthChecker("svc", latency_window=3)
        checker.record_success(latency_ms=10.0)
        checker.record_success(latency_ms=20.0)
        checker.record_success(latency_ms=30.0)
        checker.record_success(latency_ms=40.0)

        # Window should now be [20, 30, 40], average 30
        assert checker.get_status().latency_ms == pytest.approx(30.0)


class TestHealthCheckerMetadataAndReset:
    """Tests for HealthChecker.set_metadata() and reset()."""

    def test_set_metadata(self):
        """Test setting a metadata key-value pair."""
        checker = HealthChecker("svc")
        checker.set_metadata("version", "2.1.0")

        status = checker.get_status()
        assert status.metadata == {"version": "2.1.0"}

    def test_set_multiple_metadata_keys(self):
        """Test setting multiple metadata keys."""
        checker = HealthChecker("svc")
        checker.set_metadata("version", "2.1.0")
        checker.set_metadata("region", "eu-west-1")

        status = checker.get_status()
        assert status.metadata == {"version": "2.1.0", "region": "eu-west-1"}

    def test_set_metadata_overwrites_existing_key(self):
        """Test that setting a key that already exists overwrites the value."""
        checker = HealthChecker("svc")
        checker.set_metadata("version", "1.0")
        checker.set_metadata("version", "2.0")

        status = checker.get_status()
        assert status.metadata["version"] == "2.0"

    def test_get_status_returns_metadata_copy(self):
        """Test that get_status returns a copy of metadata, not a reference."""
        checker = HealthChecker("svc")
        checker.set_metadata("key", "value")

        status = checker.get_status()
        status.metadata["key"] = "modified"

        # Original should be unchanged
        assert checker.get_status().metadata["key"] == "value"

    def test_reset_restores_healthy_state(self):
        """Test that reset restores the checker to healthy."""
        checker = HealthChecker("svc", failure_threshold=1)
        checker.record_failure("err")
        assert checker.get_status().healthy is False

        checker.reset()
        assert checker.get_status().healthy is True

    def test_reset_clears_consecutive_failures(self):
        """Test that reset clears the failure counter."""
        checker = HealthChecker("svc")
        checker.record_failure("e1")
        checker.record_failure("e2")

        checker.reset()
        assert checker.get_status().consecutive_failures == 0

    def test_reset_clears_last_error(self):
        """Test that reset clears the last error."""
        checker = HealthChecker("svc")
        checker.record_failure("something bad")

        checker.reset()
        assert checker.get_status().last_error is None

    def test_reset_clears_latencies(self):
        """Test that reset clears all recorded latencies."""
        checker = HealthChecker("svc")
        checker.record_success(latency_ms=50.0)
        checker.record_success(latency_ms=60.0)

        checker.reset()
        assert checker.get_status().latency_ms is None

    def test_reset_clears_metadata(self):
        """Test that reset clears all metadata."""
        checker = HealthChecker("svc")
        checker.set_metadata("key", "value")

        checker.reset()
        assert checker.get_status().metadata == {}

    def test_reset_updates_last_check(self):
        """Test that reset updates the last_check timestamp."""
        checker = HealthChecker("svc")
        before = datetime.now(timezone.utc)
        checker.reset()
        after = datetime.now(timezone.utc)

        status = checker.get_status()
        assert before <= status.last_check <= after


class TestHealthCheckerGetStatus:
    """Tests for HealthChecker.get_status() return values."""

    def test_get_status_returns_health_status(self):
        """Test that get_status returns a HealthStatus instance."""
        checker = HealthChecker("svc")
        status = checker.get_status()

        assert isinstance(status, HealthStatus)

    def test_get_status_reflects_current_state(self):
        """Test that get_status reflects all recorded events."""
        checker = HealthChecker("svc", failure_threshold=5)
        checker.record_failure("err1")
        checker.record_failure("err2")
        checker.record_success(latency_ms=25.0)
        checker.set_metadata("env", "test")

        status = checker.get_status()
        assert status.healthy is True
        assert status.consecutive_failures == 0
        # last_error is only cleared on recovery (unhealthy -> healthy transition),
        # not on a simple success while already healthy
        assert status.last_error == "err2"
        assert status.latency_ms == 25.0
        assert status.metadata == {"env": "test"}

    def test_get_status_consecutive_calls_are_independent(self):
        """Test that modifying one status does not affect subsequent calls."""
        checker = HealthChecker("svc")
        checker.record_success(latency_ms=10.0)

        status1 = checker.get_status()
        status1.latency_ms = 999.0

        status2 = checker.get_status()
        assert status2.latency_ms == 10.0


# ============================================================================
# HealthRegistry Tests
# ============================================================================


class TestHealthRegistryRegister:
    """Tests for HealthRegistry.register()."""

    def test_register_creates_checker(self):
        """Test that register creates and returns a HealthChecker."""
        registry = HealthRegistry()
        checker = registry.register("database")

        assert isinstance(checker, HealthChecker)
        assert checker.name == "database"

    def test_register_returns_existing_checker(self):
        """Test that registering the same name returns the existing checker."""
        registry = HealthRegistry()
        checker1 = registry.register("cache")
        checker2 = registry.register("cache")

        assert checker1 is checker2

    def test_register_with_custom_thresholds(self):
        """Test that custom thresholds are applied to the created checker."""
        registry = HealthRegistry()
        checker = registry.register(
            "api",
            failure_threshold=5,
            recovery_threshold=4,
            latency_window=20,
        )

        assert checker.failure_threshold == 5
        assert checker.recovery_threshold == 4
        assert checker.latency_window == 20

    def test_register_existing_ignores_new_thresholds(self):
        """Test that re-registering with different thresholds returns the original."""
        registry = HealthRegistry()
        checker1 = registry.register("svc", failure_threshold=3)
        checker2 = registry.register("svc", failure_threshold=10)

        assert checker2 is checker1
        assert checker2.failure_threshold == 3  # Original threshold preserved

    def test_register_multiple_components(self):
        """Test registering multiple different components."""
        registry = HealthRegistry()
        db = registry.register("database")
        cache = registry.register("cache")
        api = registry.register("api")

        assert db.name == "database"
        assert cache.name == "cache"
        assert api.name == "api"
        assert len(registry.registered_components) == 3


class TestHealthRegistryGet:
    """Tests for HealthRegistry.get()."""

    def test_get_existing_checker(self):
        """Test getting a registered checker by name."""
        registry = HealthRegistry()
        registered = registry.register("db")
        retrieved = registry.get("db")

        assert retrieved is registered

    def test_get_nonexistent_returns_none(self):
        """Test getting an unregistered name returns None."""
        registry = HealthRegistry()
        result = registry.get("nonexistent")

        assert result is None

    def test_get_after_unregister_returns_none(self):
        """Test that get returns None after the component is unregistered."""
        registry = HealthRegistry()
        registry.register("temp")
        registry.unregister("temp")

        assert registry.get("temp") is None


class TestHealthRegistryGetOrCreate:
    """Tests for HealthRegistry.get_or_create()."""

    def test_creates_new_checker(self):
        """Test that get_or_create creates a new checker when not registered."""
        registry = HealthRegistry()
        checker = registry.get_or_create("new-service")

        assert isinstance(checker, HealthChecker)
        assert checker.name == "new-service"

    def test_returns_existing_checker(self):
        """Test that get_or_create returns the existing checker."""
        registry = HealthRegistry()
        created = registry.get_or_create("svc")
        retrieved = registry.get_or_create("svc")

        assert created is retrieved

    def test_get_or_create_with_custom_thresholds(self):
        """Test that custom thresholds are applied when creating."""
        registry = HealthRegistry()
        checker = registry.get_or_create(
            "svc",
            failure_threshold=7,
            recovery_threshold=5,
        )

        assert checker.failure_threshold == 7
        assert checker.recovery_threshold == 5

    def test_get_or_create_existing_preserves_state(self):
        """Test that get_or_create on existing checker preserves its state."""
        registry = HealthRegistry()
        checker = registry.get_or_create("svc")
        checker.record_failure("err")
        checker.set_metadata("key", "value")

        same_checker = registry.get_or_create("svc")
        status = same_checker.get_status()
        assert status.consecutive_failures == 1
        assert status.metadata == {"key": "value"}

    def test_get_or_create_is_retrievable_via_get(self):
        """Test that a checker created via get_or_create is visible via get."""
        registry = HealthRegistry()
        checker = registry.get_or_create("svc")
        retrieved = registry.get("svc")

        assert retrieved is checker


class TestHealthRegistryUnregister:
    """Tests for HealthRegistry.unregister()."""

    def test_unregister_existing_returns_true(self):
        """Test that unregistering an existing component returns True."""
        registry = HealthRegistry()
        registry.register("temp")

        assert registry.unregister("temp") is True

    def test_unregister_nonexistent_returns_false(self):
        """Test that unregistering a nonexistent component returns False."""
        registry = HealthRegistry()

        assert registry.unregister("nonexistent") is False

    def test_unregister_removes_from_components(self):
        """Test that unregistered component is removed from the registry."""
        registry = HealthRegistry()
        registry.register("svc")
        registry.unregister("svc")

        assert "svc" not in registry.registered_components
        assert registry.get("svc") is None

    def test_unregister_idempotent(self):
        """Test that double unregister returns False on second call."""
        registry = HealthRegistry()
        registry.register("svc")
        assert registry.unregister("svc") is True
        assert registry.unregister("svc") is False

    def test_unregister_does_not_affect_other_components(self):
        """Test that unregistering one component leaves others intact."""
        registry = HealthRegistry()
        registry.register("keep")
        registry.register("remove")
        registry.unregister("remove")

        assert registry.get("keep") is not None
        assert registry.get("remove") is None


class TestHealthRegistryGetReport:
    """Tests for HealthRegistry.get_report()."""

    def test_report_no_components(self):
        """Test report with no registered components."""
        registry = HealthRegistry()
        report = registry.get_report()

        assert isinstance(report, HealthReport)
        assert report.overall_healthy is True  # vacuous truth: all() on empty
        assert report.components == {}
        assert report.summary == "No components registered"

    def test_report_all_healthy(self):
        """Test report when all components are healthy."""
        registry = HealthRegistry()
        registry.register("db")
        registry.register("cache")

        report = registry.get_report()

        assert report.overall_healthy is True
        assert len(report.components) == 2
        assert report.summary == "All 2 components healthy"

    def test_report_some_unhealthy(self):
        """Test report when some components are unhealthy."""
        registry = HealthRegistry()
        db = registry.register("db", failure_threshold=1)
        registry.register("cache")

        db.record_failure("connection lost")

        report = registry.get_report()

        assert report.overall_healthy is False
        assert "db" in report.summary
        assert "1/2" in report.summary

    def test_report_all_unhealthy(self):
        """Test report when all components are unhealthy."""
        registry = HealthRegistry()
        db = registry.register("db", failure_threshold=1)
        cache = registry.register("cache", failure_threshold=1)

        db.record_failure("db down")
        cache.record_failure("cache down")

        report = registry.get_report()

        assert report.overall_healthy is False
        assert "2/2" in report.summary
        assert "db" in report.summary
        assert "cache" in report.summary

    def test_report_checked_at_timestamp(self):
        """Test that report has a valid checked_at timestamp."""
        registry = HealthRegistry()
        before = datetime.now(timezone.utc)
        report = registry.get_report()
        after = datetime.now(timezone.utc)

        assert before <= report.checked_at <= after

    def test_report_contains_component_statuses(self):
        """Test that report includes correct status for each component."""
        registry = HealthRegistry()
        svc = registry.register("svc")
        svc.record_success(latency_ms=15.0)
        svc.set_metadata("version", "3.0")

        report = registry.get_report()
        status = report.components["svc"]

        assert status.healthy is True
        assert status.latency_ms == 15.0
        assert status.metadata == {"version": "3.0"}

    def test_report_single_unhealthy_component(self):
        """Test report summary with a single unhealthy component."""
        registry = HealthRegistry()
        svc = registry.register("failing-service", failure_threshold=1)
        svc.record_failure("timeout")

        report = registry.get_report()

        assert report.overall_healthy is False
        assert "1/1" in report.summary
        assert "failing-service" in report.summary

    def test_report_single_healthy_component(self):
        """Test report summary with a single healthy component."""
        registry = HealthRegistry()
        registry.register("ok-service")

        report = registry.get_report()

        assert report.overall_healthy is True
        assert "All 1 components healthy" in report.summary


class TestHealthRegistryGetAllStatuses:
    """Tests for HealthRegistry.get_all_statuses()."""

    def test_get_all_statuses_empty(self):
        """Test get_all_statuses with no components."""
        registry = HealthRegistry()
        statuses = registry.get_all_statuses()

        assert statuses == {}

    def test_get_all_statuses_returns_dict(self):
        """Test get_all_statuses returns dict of component name to HealthStatus."""
        registry = HealthRegistry()
        registry.register("db")
        registry.register("cache")

        statuses = registry.get_all_statuses()

        assert len(statuses) == 2
        assert "db" in statuses
        assert "cache" in statuses
        assert isinstance(statuses["db"], HealthStatus)
        assert isinstance(statuses["cache"], HealthStatus)

    def test_get_all_statuses_reflects_state(self):
        """Test that returned statuses reflect the actual health state."""
        registry = HealthRegistry()
        db = registry.register("db", failure_threshold=1)
        db.record_failure("down")

        statuses = registry.get_all_statuses()

        assert statuses["db"].healthy is False
        assert statuses["db"].last_error == "down"


class TestHealthRegistryRegisteredComponents:
    """Tests for HealthRegistry.registered_components property."""

    def test_empty_registry(self):
        """Test that empty registry returns empty list."""
        registry = HealthRegistry()
        assert registry.registered_components == []

    def test_returns_component_names(self):
        """Test that registered_components returns all registered names."""
        registry = HealthRegistry()
        registry.register("alpha")
        registry.register("beta")
        registry.register("gamma")

        components = registry.registered_components
        assert set(components) == {"alpha", "beta", "gamma"}

    def test_reflects_unregister(self):
        """Test that unregistered components are excluded."""
        registry = HealthRegistry()
        registry.register("keep")
        registry.register("remove")
        registry.unregister("remove")

        assert registry.registered_components == ["keep"]

    def test_returns_list_type(self):
        """Test that registered_components returns a list."""
        registry = HealthRegistry()
        registry.register("svc")

        result = registry.registered_components
        assert isinstance(result, list)


# ============================================================================
# Global Health Registry Tests
# ============================================================================


class TestGetGlobalHealthRegistry:
    """Tests for get_global_health_registry() singleton."""

    def test_returns_health_registry(self):
        """Test that the function returns a HealthRegistry instance."""
        registry = get_global_health_registry()
        assert isinstance(registry, HealthRegistry)

    def test_returns_singleton(self):
        """Test that multiple calls return the same instance."""
        r1 = get_global_health_registry()
        r2 = get_global_health_registry()

        assert r1 is r2

    def test_singleton_shares_state(self):
        """Test that the singleton shares state across calls."""
        registry = get_global_health_registry()
        name = "test-global-singleton-component"

        # Clean up in case previous test left state
        registry.unregister(name)

        registry.register(name)
        try:
            r2 = get_global_health_registry()
            assert r2.get(name) is not None
        finally:
            # Clean up so we don't affect other tests
            registry.unregister(name)

    def test_singleton_persists_after_reset(self):
        """Test that the singleton uses a module-level variable with lock."""
        # This tests the lazy-initialization pattern
        r1 = get_global_health_registry()
        r2 = get_global_health_registry()
        assert r1 is r2

    def test_singleton_with_patched_global(self):
        """Test singleton initialization when the global is None."""
        with patch("aragora.resilience.health._global_registry", None):
            registry = get_global_health_registry()
            assert isinstance(registry, HealthRegistry)
