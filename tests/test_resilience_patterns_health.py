"""
Tests for resilience_patterns.health module.

Tests cover:
- HealthChecker component tracking
- Health status transitions
- Aggregate health reports
- Failure/recovery thresholds
- Latency tracking
"""

from datetime import datetime, timezone
import pytest

from aragora.resilience_patterns import (
    HealthStatus,
    HealthChecker,
    HealthReport,
)


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_healthy_status(self):
        """Test creating healthy status."""
        status = HealthStatus(
            healthy=True,
            last_check=datetime.now(timezone.utc),
        )
        assert status.healthy is True
        assert status.consecutive_failures == 0

    def test_unhealthy_status(self):
        """Test creating unhealthy status."""
        status = HealthStatus(
            healthy=False,
            last_check=datetime.now(timezone.utc),
            consecutive_failures=3,
            last_error="Connection refused",
        )
        assert status.healthy is False
        assert status.consecutive_failures == 3
        assert status.last_error == "Connection refused"

    def test_to_dict(self):
        """Test serialization to dict."""
        now = datetime.now(timezone.utc)
        status = HealthStatus(
            healthy=True,
            last_check=now,
            latency_ms=15.5,
            metadata={"version": "1.0"},
        )
        data = status.to_dict()
        assert data["healthy"] is True
        assert data["latency_ms"] == 15.5
        assert data["metadata"]["version"] == "1.0"


class TestHealthChecker:
    """Tests for HealthChecker class."""

    def test_initial_state(self):
        """Test initial healthy state."""
        checker = HealthChecker("test_component")
        status = checker.get_status()
        assert status.healthy is True
        assert status.consecutive_failures == 0

    def test_record_success(self):
        """Test recording successful checks."""
        checker = HealthChecker("test_component")
        checker.record_success(latency_ms=50.0)

        status = checker.get_status()
        assert status.healthy is True
        assert status.latency_ms == 50.0

    def test_record_failure_below_threshold(self):
        """Test recording failures below threshold."""
        checker = HealthChecker("test_component", failure_threshold=3)
        checker.record_failure("Error 1")

        status = checker.get_status()
        assert status.healthy is True  # Still healthy
        assert status.consecutive_failures == 1

        checker.record_failure("Error 2")
        status = checker.get_status()
        assert status.healthy is True
        assert status.consecutive_failures == 2

    def test_transition_to_unhealthy(self):
        """Test transition to unhealthy after threshold."""
        checker = HealthChecker("test_component", failure_threshold=3)

        for i in range(3):
            checker.record_failure(f"Error {i}")

        status = checker.get_status()
        assert status.healthy is False
        assert status.consecutive_failures == 3

    def test_recovery_after_success(self):
        """Test recovery after successful checks."""
        checker = HealthChecker("test_component", failure_threshold=2, recovery_threshold=2)

        # Make unhealthy
        checker.record_failure("Error 1")
        checker.record_failure("Error 2")
        assert checker.get_status().healthy is False

        # Start recovery
        checker.record_success()
        assert checker.get_status().healthy is False  # Not yet recovered

        checker.record_success()
        assert checker.get_status().healthy is True  # Recovered

    def test_failure_resets_recovery(self):
        """Test that failure resets recovery progress."""
        checker = HealthChecker("test_component", failure_threshold=2, recovery_threshold=3)

        # Make unhealthy
        checker.record_failure("Error 1")
        checker.record_failure("Error 2")
        assert checker.get_status().healthy is False

        # Start recovery
        checker.record_success()
        checker.record_success()

        # Failure during recovery
        checker.record_failure("Error 3")

        # Should need full recovery again
        assert checker.get_status().healthy is False

    def test_latency_tracking(self):
        """Test latency tracking."""
        checker = HealthChecker("test_component")

        checker.record_success(latency_ms=100.0)
        checker.record_success(latency_ms=200.0)
        checker.record_success(latency_ms=150.0)

        status = checker.get_status()
        assert status.latency_ms == 150.0  # (100 + 200 + 150) / 3

    def test_latency_window(self):
        """Test latency sliding window."""
        checker = HealthChecker("test_component", latency_window=3)

        checker.record_success(latency_ms=100.0)
        checker.record_success(latency_ms=200.0)
        checker.record_success(latency_ms=300.0)
        checker.record_success(latency_ms=400.0)  # Pushes out 100

        status = checker.get_status()
        assert status.latency_ms == 300.0  # (200 + 300 + 400) / 3

    def test_get_status(self):
        """Test getting full status."""
        checker = HealthChecker("test_component")
        checker.record_success(latency_ms=50.0)
        checker.record_failure("Test error")

        status = checker.get_status()
        assert status.consecutive_failures == 1
        assert status.last_error == "Test error"
        assert status.last_check is not None

    def test_custom_metadata(self):
        """Test custom metadata storage."""
        checker = HealthChecker("test_component")
        checker.set_metadata("version", "1.0.0")
        checker.set_metadata("region", "us-east-1")

        status = checker.get_status()
        assert status.metadata.get("version") == "1.0.0"
        assert status.metadata.get("region") == "us-east-1"

    def test_reset(self):
        """Test reset functionality."""
        checker = HealthChecker("test_component", failure_threshold=2)

        checker.record_failure("Error 1")
        checker.record_failure("Error 2")
        assert checker.get_status().healthy is False

        checker.reset()

        status = checker.get_status()
        assert status.healthy is True
        assert status.consecutive_failures == 0


class TestHealthReport:
    """Tests for HealthReport aggregate class."""

    def test_empty_report(self):
        """Test empty health report."""
        report = HealthReport(
            components={},
            overall_healthy=True,
            checked_at=datetime.now(timezone.utc),
        )
        assert report.overall_healthy is True
        assert len(report.components) == 0

    def test_all_healthy(self):
        """Test report with all healthy components."""
        components = {
            "comp1": HealthStatus(healthy=True, last_check=datetime.now(timezone.utc)),
            "comp2": HealthStatus(healthy=True, last_check=datetime.now(timezone.utc)),
        }
        report = HealthReport(
            components=components,
            overall_healthy=True,
            checked_at=datetime.now(timezone.utc),
        )

        assert report.overall_healthy is True
        assert len(report.components) == 2

    def test_one_unhealthy(self):
        """Test report with one unhealthy component."""
        components = {
            "comp1": HealthStatus(healthy=True, last_check=datetime.now(timezone.utc)),
            "comp2": HealthStatus(
                healthy=False,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=5,
            ),
        }
        report = HealthReport(
            components=components,
            overall_healthy=False,
            checked_at=datetime.now(timezone.utc),
        )

        assert report.overall_healthy is False
        unhealthy = [name for name, status in report.components.items() if not status.healthy]
        assert len(unhealthy) == 1
        assert "comp2" in unhealthy

    def test_to_dict(self):
        """Test serialization to dict."""
        now = datetime.now(timezone.utc)
        components = {
            "comp1": HealthStatus(healthy=True, last_check=now),
        }
        report = HealthReport(
            components=components,
            overall_healthy=True,
            checked_at=now,
        )

        data = report.to_dict()
        assert data["overall_healthy"] is True
        assert len(data["components"]) == 1
        assert "comp1" in data["components"]


class TestHealthCheckerIntegration:
    """Integration tests for health checking scenarios."""

    def test_flapping_detection(self):
        """Test detection of flapping health status."""
        checker = HealthChecker("flaky_component", failure_threshold=2, recovery_threshold=2)

        # Simulate flapping
        states = []
        for _ in range(10):
            checker.record_success()
            states.append(checker.get_status().healthy)
            checker.record_failure("Flap")
            states.append(checker.get_status().healthy)
            checker.record_failure("Flap")
            states.append(checker.get_status().healthy)

        # Should have transitioned multiple times
        transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
        assert transitions > 0

    def test_gradual_degradation(self):
        """Test gradual degradation scenario."""
        checker = HealthChecker("degrading_service", failure_threshold=5)

        # Gradually increase failures
        for i in range(10):
            # Success ratio decreases
            if i < 5:
                checker.record_success()
            checker.record_failure(f"Error {i}")

        # Should become unhealthy after threshold
        status = checker.get_status()
        assert status.consecutive_failures >= 5

    def test_multiple_checkers(self):
        """Test multiple independent health checkers."""
        checker1 = HealthChecker("service1", failure_threshold=2)
        checker2 = HealthChecker("service2", failure_threshold=3)

        # Service 1 becomes unhealthy
        checker1.record_failure("Error")
        checker1.record_failure("Error")

        # Service 2 stays healthy
        checker2.record_failure("Error")

        assert checker1.get_status().healthy is False
        assert checker2.get_status().healthy is True

        # Get combined report
        statuses = {
            "service1": checker1.get_status(),
            "service2": checker2.get_status(),
        }
        report = HealthReport(
            components=statuses,
            overall_healthy=all(s.healthy for s in statuses.values()),
            checked_at=datetime.now(timezone.utc),
        )

        assert report.overall_healthy is False
        unhealthy = [name for name, status in report.components.items() if not status.healthy]
        assert len(unhealthy) == 1


class TestHealthRegistry:
    """Tests for HealthRegistry class."""

    def test_register_component(self):
        """Test registering a health checker."""
        from aragora.resilience_patterns.health import HealthRegistry

        registry = HealthRegistry()
        checker = registry.register("database")

        assert checker.name == "database"
        assert registry.get("database") is checker

    def test_get_registered(self):
        """Test getting a registered checker."""
        from aragora.resilience_patterns.health import HealthRegistry

        registry = HealthRegistry()
        registry.register("cache")

        checker = registry.get("cache")
        assert checker is not None
        assert checker.name == "cache"

    def test_get_unregistered(self):
        """Test getting an unregistered checker."""
        from aragora.resilience_patterns.health import HealthRegistry

        registry = HealthRegistry()
        assert registry.get("nonexistent") is None

    def test_get_or_create(self):
        """Test get_or_create functionality."""
        from aragora.resilience_patterns.health import HealthRegistry

        registry = HealthRegistry()

        checker1 = registry.get_or_create("service")
        checker2 = registry.get_or_create("service")

        assert checker1 is checker2

    def test_unregister(self):
        """Test unregistering a component."""
        from aragora.resilience_patterns.health import HealthRegistry

        registry = HealthRegistry()
        registry.register("temp")

        assert registry.unregister("temp") is True
        assert registry.get("temp") is None
        assert registry.unregister("temp") is False

    def test_get_report(self):
        """Test aggregate health report."""
        from aragora.resilience_patterns.health import HealthRegistry

        registry = HealthRegistry()

        db = registry.register("database")
        cache = registry.register("cache", failure_threshold=3)

        db.record_success(latency_ms=5.0)
        cache.record_failure("Connection refused")
        cache.record_failure("Connection refused")
        cache.record_failure("Connection refused")

        report = registry.get_report()

        assert report.overall_healthy is False
        assert len(report.components) == 2
        assert "database" in report.components
        assert "cache" in report.components

    def test_register_same_name_returns_existing(self):
        """Test that registering the same name returns existing checker."""
        from aragora.resilience_patterns.health import HealthRegistry

        registry = HealthRegistry()
        checker1 = registry.register("service")
        checker2 = registry.register("service")

        assert checker1 is checker2


class TestGlobalRegistry:
    """Tests for global health registry singleton."""

    def test_singleton(self):
        """Test global registry is a singleton."""
        from aragora.resilience_patterns.health import get_global_health_registry

        reg1 = get_global_health_registry()
        reg2 = get_global_health_registry()
        assert reg1 is reg2

    def test_thread_safety(self):
        """Test thread-safe initialization."""
        import concurrent.futures
        from aragora.resilience_patterns.health import get_global_health_registry

        registries = []

        def get_registry():
            return get_global_health_registry()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_registry) for _ in range(20)]
            registries = [f.result() for f in futures]

        assert all(r is registries[0] for r in registries)
