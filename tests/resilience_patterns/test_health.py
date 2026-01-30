"""
Tests for Health Monitoring.

Tests the health monitoring implementation including:
- HealthStatus dataclass
- HealthReport dataclass
- HealthChecker class
- HealthRegistry class
- Global registry singleton
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import pytest

from aragora.resilience.health import (
    HealthChecker,
    HealthRegistry,
    HealthReport,
    HealthStatus,
    get_global_health_registry,
)


# =============================================================================
# HealthStatus Tests
# =============================================================================


class TestHealthStatus:
    """Test HealthStatus dataclass."""

    def test_minimal_status(self):
        """Test minimal HealthStatus creation."""
        now = datetime.now(timezone.utc)
        status = HealthStatus(healthy=True, last_check=now)
        assert status.healthy is True
        assert status.last_check == now
        assert status.consecutive_failures == 0
        assert status.last_error is None
        assert status.latency_ms is None
        assert status.metadata == {}

    def test_full_status(self):
        """Test full HealthStatus creation."""
        now = datetime.now(timezone.utc)
        status = HealthStatus(
            healthy=False,
            last_check=now,
            consecutive_failures=5,
            last_error="Connection failed",
            latency_ms=125.5,
            metadata={"region": "us-east-1"},
        )
        assert status.healthy is False
        assert status.consecutive_failures == 5
        assert status.last_error == "Connection failed"
        assert status.latency_ms == 125.5
        assert status.metadata == {"region": "us-east-1"}

    def test_to_dict(self):
        """Test HealthStatus to_dict conversion."""
        now = datetime.now(timezone.utc)
        status = HealthStatus(
            healthy=True,
            last_check=now,
            consecutive_failures=0,
            latency_ms=10.0,
        )
        data = status.to_dict()
        assert data["healthy"] is True
        assert data["last_check"] == now.isoformat()
        assert data["consecutive_failures"] == 0
        assert data["latency_ms"] == 10.0

    def test_from_dict(self):
        """Test HealthStatus from_dict creation."""
        now = datetime.now(timezone.utc)
        data = {
            "healthy": False,
            "last_check": now.isoformat(),
            "consecutive_failures": 3,
            "last_error": "Timeout",
            "latency_ms": 50.0,
            "metadata": {"version": "1.0"},
        }
        status = HealthStatus.from_dict(data)
        assert status.healthy is False
        assert status.consecutive_failures == 3
        assert status.last_error == "Timeout"
        assert status.latency_ms == 50.0
        assert status.metadata == {"version": "1.0"}

    def test_from_dict_with_datetime_object(self):
        """Test from_dict when last_check is already a datetime."""
        now = datetime.now(timezone.utc)
        data = {
            "healthy": True,
            "last_check": now,  # datetime object, not string
        }
        status = HealthStatus.from_dict(data)
        assert status.last_check == now

    def test_roundtrip_serialization(self):
        """Test serialization roundtrip."""
        now = datetime.now(timezone.utc)
        original = HealthStatus(
            healthy=True,
            last_check=now,
            consecutive_failures=2,
            latency_ms=15.0,
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = HealthStatus.from_dict(data)
        assert restored.healthy == original.healthy
        assert restored.consecutive_failures == original.consecutive_failures
        assert restored.latency_ms == original.latency_ms
        assert restored.metadata == original.metadata


# =============================================================================
# HealthReport Tests
# =============================================================================


class TestHealthReport:
    """Test HealthReport dataclass."""

    def test_empty_report(self):
        """Test empty health report."""
        now = datetime.now(timezone.utc)
        report = HealthReport(
            overall_healthy=True,
            components={},
            checked_at=now,
            summary="No components",
        )
        assert report.overall_healthy is True
        assert report.components == {}
        assert report.summary == "No components"

    def test_report_with_components(self):
        """Test report with multiple components."""
        now = datetime.now(timezone.utc)
        components = {
            "db": HealthStatus(healthy=True, last_check=now),
            "cache": HealthStatus(healthy=False, last_check=now, last_error="Down"),
        }
        report = HealthReport(
            overall_healthy=False,
            components=components,
            checked_at=now,
            summary="1/2 unhealthy",
        )
        assert report.overall_healthy is False
        assert len(report.components) == 2
        assert report.components["db"].healthy is True
        assert report.components["cache"].healthy is False

    def test_to_dict(self):
        """Test HealthReport to_dict conversion."""
        now = datetime.now(timezone.utc)
        components = {
            "service": HealthStatus(healthy=True, last_check=now),
        }
        report = HealthReport(
            overall_healthy=True,
            components=components,
            checked_at=now,
            summary="All healthy",
        )
        data = report.to_dict()
        assert data["overall_healthy"] is True
        assert data["checked_at"] == now.isoformat()
        assert "service" in data["components"]
        assert data["summary"] == "All healthy"


# =============================================================================
# HealthChecker Tests
# =============================================================================


class TestHealthChecker:
    """Test HealthChecker class."""

    def test_initial_state(self):
        """Test initial health checker state."""
        checker = HealthChecker("test")
        status = checker.get_status()
        assert status.healthy is True
        assert status.consecutive_failures == 0
        assert status.last_error is None

    def test_record_success(self):
        """Test recording successful checks."""
        checker = HealthChecker("test")
        checker.record_success(latency_ms=10.0)
        status = checker.get_status()
        assert status.healthy is True
        assert status.latency_ms == 10.0

    def test_record_success_without_latency(self):
        """Test recording success without latency."""
        checker = HealthChecker("test")
        checker.record_success()
        status = checker.get_status()
        assert status.healthy is True
        assert status.latency_ms is None

    def test_record_failure(self):
        """Test recording failed checks."""
        checker = HealthChecker("test")
        checker.record_failure("Connection timeout")
        status = checker.get_status()
        assert status.consecutive_failures == 1
        assert status.last_error == "Connection timeout"

    def test_failure_threshold(self):
        """Test health degrades after failure threshold."""
        checker = HealthChecker("test", failure_threshold=3)

        # First two failures - still healthy
        checker.record_failure("Error 1")
        assert checker.get_status().healthy is True

        checker.record_failure("Error 2")
        assert checker.get_status().healthy is True

        # Third failure - becomes unhealthy
        checker.record_failure("Error 3")
        status = checker.get_status()
        assert status.healthy is False
        assert status.consecutive_failures == 3

    def test_recovery_threshold(self):
        """Test recovery after success threshold."""
        checker = HealthChecker("test", failure_threshold=2, recovery_threshold=2)

        # Degrade to unhealthy
        checker.record_failure("Error 1")
        checker.record_failure("Error 2")
        assert checker.get_status().healthy is False

        # One success - not enough to recover
        checker.record_success()
        assert checker.get_status().healthy is False

        # Second success - recovers
        checker.record_success()
        status = checker.get_status()
        assert status.healthy is True
        assert status.last_error is None

    def test_success_resets_failures(self):
        """Test that success resets consecutive failures."""
        checker = HealthChecker("test", failure_threshold=3)

        checker.record_failure("Error 1")
        checker.record_failure("Error 2")
        assert checker.get_status().consecutive_failures == 2

        checker.record_success()
        assert checker.get_status().consecutive_failures == 0

    def test_latency_window(self):
        """Test latency averaging over window."""
        checker = HealthChecker("test", latency_window=3)

        checker.record_success(latency_ms=10.0)
        checker.record_success(latency_ms=20.0)
        checker.record_success(latency_ms=30.0)
        assert checker.get_status().latency_ms == 20.0  # (10+20+30)/3

        # Add fourth sample - oldest drops off
        checker.record_success(latency_ms=40.0)
        assert checker.get_status().latency_ms == 30.0  # (20+30+40)/3

    def test_set_metadata(self):
        """Test setting metadata."""
        checker = HealthChecker("test")
        checker.set_metadata("version", "1.0")
        checker.set_metadata("region", "us-west-2")

        status = checker.get_status()
        assert status.metadata["version"] == "1.0"
        assert status.metadata["region"] == "us-west-2"

    def test_reset(self):
        """Test resetting health checker."""
        checker = HealthChecker("test")

        # Add some state
        checker.record_failure("Error")
        checker.record_failure("Error")
        checker.record_failure("Error")
        checker.set_metadata("key", "value")
        assert checker.get_status().healthy is False

        # Reset
        checker.reset()
        status = checker.get_status()
        assert status.healthy is True
        assert status.consecutive_failures == 0
        assert status.last_error is None
        assert status.metadata == {}

    def test_thread_safety(self):
        """Test thread-safe operations."""
        checker = HealthChecker("test", failure_threshold=100)
        iterations = 100
        errors = []

        def record_operations():
            try:
                for _ in range(iterations):
                    checker.record_success(latency_ms=1.0)
                    checker.record_failure("test")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_operations) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Check we can still get status
        status = checker.get_status()
        assert status is not None


# =============================================================================
# HealthRegistry Tests
# =============================================================================


class TestHealthRegistry:
    """Test HealthRegistry class."""

    def test_register(self):
        """Test registering health checkers."""
        registry = HealthRegistry()
        checker = registry.register("database")
        assert checker is not None
        assert checker.name == "database"

    def test_register_returns_existing(self):
        """Test register returns existing checker."""
        registry = HealthRegistry()
        checker1 = registry.register("cache")
        checker2 = registry.register("cache")
        assert checker1 is checker2

    def test_register_with_options(self):
        """Test register with custom thresholds."""
        registry = HealthRegistry()
        checker = registry.register(
            "service",
            failure_threshold=5,
            recovery_threshold=3,
            latency_window=20,
        )
        assert checker.failure_threshold == 5
        assert checker.recovery_threshold == 3
        assert checker.latency_window == 20

    def test_get(self):
        """Test getting registered checker."""
        registry = HealthRegistry()
        registry.register("test")
        checker = registry.get("test")
        assert checker is not None
        assert checker.name == "test"

    def test_get_unregistered(self):
        """Test getting unregistered checker returns None."""
        registry = HealthRegistry()
        checker = registry.get("nonexistent")
        assert checker is None

    def test_get_or_create(self):
        """Test get_or_create."""
        registry = HealthRegistry()

        # Creates new
        checker1 = registry.get_or_create("new_service")
        assert checker1.name == "new_service"

        # Returns existing
        checker2 = registry.get_or_create("new_service")
        assert checker1 is checker2

    def test_unregister(self):
        """Test unregistering checker."""
        registry = HealthRegistry()
        registry.register("temp")
        assert registry.get("temp") is not None

        result = registry.unregister("temp")
        assert result is True
        assert registry.get("temp") is None

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent returns False."""
        registry = HealthRegistry()
        result = registry.unregister("nonexistent")
        assert result is False

    def test_registered_components(self):
        """Test getting list of registered components."""
        registry = HealthRegistry()
        registry.register("db")
        registry.register("cache")
        registry.register("api")

        components = registry.registered_components
        assert len(components) == 3
        assert "db" in components
        assert "cache" in components
        assert "api" in components

    def test_get_all_statuses(self):
        """Test getting all statuses."""
        registry = HealthRegistry()
        registry.register("service1")
        registry.register("service2")

        registry.get("service1").record_success()
        registry.get("service2").record_failure("Down")

        statuses = registry.get_all_statuses()
        assert len(statuses) == 2
        assert statuses["service1"].healthy is True
        assert statuses["service2"].consecutive_failures == 1


class TestHealthRegistryReport:
    """Test HealthRegistry report generation."""

    def test_empty_report(self):
        """Test report with no components."""
        registry = HealthRegistry()
        report = registry.get_report()
        assert report.overall_healthy is True  # Vacuously true
        assert report.components == {}
        assert report.summary == "No components registered"

    def test_all_healthy_report(self):
        """Test report when all components healthy."""
        registry = HealthRegistry()
        registry.register("db")
        registry.register("cache")

        registry.get("db").record_success()
        registry.get("cache").record_success()

        report = registry.get_report()
        assert report.overall_healthy is True
        assert "All 2 components healthy" in report.summary

    def test_partial_unhealthy_report(self):
        """Test report with some unhealthy components."""
        registry = HealthRegistry()
        db = registry.register("database", failure_threshold=1)
        cache = registry.register("cache", failure_threshold=1)

        db.record_success()
        cache.record_failure("Connection refused")

        report = registry.get_report()
        assert report.overall_healthy is False
        assert "1/2 components unhealthy" in report.summary
        assert "cache" in report.summary

    def test_multiple_unhealthy_report(self):
        """Test report with multiple unhealthy components."""
        registry = HealthRegistry()
        registry.register("db", failure_threshold=1).record_failure("Error")
        registry.register("cache", failure_threshold=1).record_failure("Error")
        registry.register("api", failure_threshold=1).record_success()

        report = registry.get_report()
        assert report.overall_healthy is False
        assert "2/3 components unhealthy" in report.summary

    def test_report_to_dict(self):
        """Test report serialization."""
        registry = HealthRegistry()
        registry.register("test").record_success()

        report = registry.get_report()
        data = report.to_dict()

        assert "overall_healthy" in data
        assert "components" in data
        assert "checked_at" in data
        assert "summary" in data


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestGlobalRegistry:
    """Test global health registry singleton."""

    def test_singleton_pattern(self):
        """Test global registry is singleton."""
        registry1 = get_global_health_registry()
        registry2 = get_global_health_registry()
        assert registry1 is registry2

    def test_global_registry_is_functional(self):
        """Test global registry works normally."""
        registry = get_global_health_registry()

        # Use unique name to avoid conflicts with other tests
        name = f"global_test_{time.time()}"
        checker = registry.register(name)
        checker.record_success(latency_ms=5.0)

        status = checker.get_status()
        assert status.healthy is True
        assert status.latency_ms == 5.0

        # Cleanup
        registry.unregister(name)

    def test_thread_safe_initialization(self):
        """Test thread-safe global registry initialization."""
        registries = []
        errors = []

        def get_registry():
            try:
                reg = get_global_health_registry()
                registries.append(reg)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_registry) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All should be the same instance
        assert all(r is registries[0] for r in registries)
