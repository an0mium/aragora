"""
Tests for resilience_patterns.health module.

Tests cover:
- HealthChecker component tracking
- Health status transitions
- Aggregate health reports
- Failure/recovery thresholds
- Latency tracking
"""

import time
import pytest
from unittest.mock import MagicMock

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
            component="test_component",
            healthy=True,
            last_check=time.time(),
        )
        assert status.component == "test_component"
        assert status.healthy is True
        assert status.consecutive_failures == 0

    def test_unhealthy_status(self):
        """Test creating unhealthy status."""
        status = HealthStatus(
            component="test_component",
            healthy=False,
            last_check=time.time(),
            consecutive_failures=3,
            last_error="Connection refused",
        )
        assert status.healthy is False
        assert status.consecutive_failures == 3
        assert status.last_error == "Connection refused"


class TestHealthChecker:
    """Tests for HealthChecker class."""

    def test_initial_state(self):
        """Test initial healthy state."""
        checker = HealthChecker("test_component")
        assert checker.is_healthy() is True
        assert checker.get_consecutive_failures() == 0

    def test_record_success(self):
        """Test recording successful checks."""
        checker = HealthChecker("test_component")
        checker.record_success(latency_ms=50.0)

        assert checker.is_healthy() is True
        status = checker.get_status()
        assert status.healthy is True

    def test_record_failure_below_threshold(self):
        """Test recording failures below threshold."""
        checker = HealthChecker("test_component", failure_threshold=3)
        checker.record_failure("Error 1")

        assert checker.is_healthy() is True  # Still healthy
        assert checker.get_consecutive_failures() == 1

        checker.record_failure("Error 2")
        assert checker.is_healthy() is True
        assert checker.get_consecutive_failures() == 2

    def test_transition_to_unhealthy(self):
        """Test transition to unhealthy after threshold."""
        checker = HealthChecker("test_component", failure_threshold=3)

        for i in range(3):
            checker.record_failure(f"Error {i}")

        assert checker.is_healthy() is False
        assert checker.get_consecutive_failures() == 3

    def test_recovery_after_success(self):
        """Test recovery after successful checks."""
        checker = HealthChecker("test_component", failure_threshold=2, recovery_threshold=2)

        # Make unhealthy
        checker.record_failure("Error 1")
        checker.record_failure("Error 2")
        assert checker.is_healthy() is False

        # Start recovery
        checker.record_success()
        assert checker.is_healthy() is False  # Not yet recovered

        checker.record_success()
        assert checker.is_healthy() is True  # Recovered

    def test_failure_resets_recovery(self):
        """Test that failure resets recovery progress."""
        checker = HealthChecker("test_component", failure_threshold=2, recovery_threshold=3)

        # Make unhealthy
        checker.record_failure("Error 1")
        checker.record_failure("Error 2")
        assert checker.is_healthy() is False

        # Start recovery
        checker.record_success()
        checker.record_success()

        # Failure during recovery
        checker.record_failure("Error 3")

        # Should need full recovery again
        assert checker.is_healthy() is False

    def test_latency_tracking(self):
        """Test latency tracking."""
        checker = HealthChecker("test_component")

        checker.record_success(latency_ms=100.0)
        checker.record_success(latency_ms=200.0)
        checker.record_success(latency_ms=150.0)

        status = checker.get_status()
        assert status.avg_latency_ms == 150.0  # (100 + 200 + 150) / 3

    def test_latency_window(self):
        """Test latency sliding window."""
        checker = HealthChecker("test_component", latency_window=3)

        checker.record_success(latency_ms=100.0)
        checker.record_success(latency_ms=200.0)
        checker.record_success(latency_ms=300.0)
        checker.record_success(latency_ms=400.0)  # Pushes out 100

        status = checker.get_status()
        assert status.avg_latency_ms == 300.0  # (200 + 300 + 400) / 3

    def test_get_status(self):
        """Test getting full status."""
        checker = HealthChecker("test_component")
        checker.record_success(latency_ms=50.0)
        checker.record_failure("Test error")

        status = checker.get_status()
        assert status.component == "test_component"
        assert status.consecutive_failures == 1
        assert status.last_error == "Test error"
        assert status.last_check > 0

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
        assert checker.is_healthy() is False

        checker.reset()

        assert checker.is_healthy() is True
        assert checker.get_consecutive_failures() == 0


class TestHealthReport:
    """Tests for HealthReport aggregate class."""

    def test_empty_report(self):
        """Test empty health report."""
        report = HealthReport(components=[], overall_healthy=True)
        assert report.overall_healthy is True
        assert len(report.components) == 0

    def test_all_healthy(self):
        """Test report with all healthy components."""
        components = [
            HealthStatus(component="comp1", healthy=True, last_check=time.time()),
            HealthStatus(component="comp2", healthy=True, last_check=time.time()),
        ]
        report = HealthReport(components=components, overall_healthy=True)

        assert report.overall_healthy is True
        assert len(report.components) == 2

    def test_one_unhealthy(self):
        """Test report with one unhealthy component."""
        components = [
            HealthStatus(component="comp1", healthy=True, last_check=time.time()),
            HealthStatus(
                component="comp2",
                healthy=False,
                last_check=time.time(),
                consecutive_failures=5,
            ),
        ]
        report = HealthReport(components=components, overall_healthy=False)

        assert report.overall_healthy is False
        unhealthy = [c for c in report.components if not c.healthy]
        assert len(unhealthy) == 1

    def test_to_dict(self):
        """Test serialization to dict."""
        components = [
            HealthStatus(component="comp1", healthy=True, last_check=1234567890.0),
        ]
        report = HealthReport(components=components, overall_healthy=True)

        data = report.to_dict()
        assert data["overall_healthy"] is True
        assert len(data["components"]) == 1
        assert data["components"][0]["component"] == "comp1"


class TestHealthCheckerIntegration:
    """Integration tests for health checking scenarios."""

    def test_flapping_detection(self):
        """Test detection of flapping health status."""
        checker = HealthChecker("flaky_component", failure_threshold=2, recovery_threshold=2)

        # Simulate flapping
        states = []
        for _ in range(10):
            checker.record_success()
            states.append(checker.is_healthy())
            checker.record_failure("Flap")
            states.append(checker.is_healthy())
            checker.record_failure("Flap")
            states.append(checker.is_healthy())

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

        assert checker1.is_healthy() is False
        assert checker2.is_healthy() is True

        # Get combined report
        statuses = [checker1.get_status(), checker2.get_status()]
        report = HealthReport(components=statuses, overall_healthy=all(s.healthy for s in statuses))

        assert report.overall_healthy is False
        assert len([c for c in report.components if not c.healthy]) == 1
