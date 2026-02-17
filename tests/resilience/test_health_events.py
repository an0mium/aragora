"""Tests for health state transition event emissions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.resilience.health import (
    HealthChecker,
    HealthRegistry,
    get_global_health_registry,
)


class TestHealthStateEvents:
    """Tests for health state change event emissions."""

    def test_emits_degraded_event_on_failure_threshold(self) -> None:
        checker = HealthChecker("test_db", failure_threshold=2)

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            checker.record_failure("conn refused")
            mock_dispatch.assert_not_called()  # Below threshold

            checker.record_failure("conn refused")
            mock_dispatch.assert_called_once()
            data = mock_dispatch.call_args[0][1]
            assert data["component"] == "test_db"
            assert data["new_status"] == "degraded"
            assert data["consecutive_failures"] == 2

    def test_emits_recovered_event_on_recovery(self) -> None:
        checker = HealthChecker("test_cache", failure_threshold=1, recovery_threshold=2)

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            checker.record_failure("timeout")  # Degrades
            mock_dispatch.reset_mock()

            checker.record_success()  # 1st success - not yet recovered
            mock_dispatch.assert_not_called()

            checker.record_success()  # 2nd success - recovery
            mock_dispatch.assert_called_once()
            data = mock_dispatch.call_args[0][1]
            assert data["new_status"] == "recovered"
            assert data["consecutive_failures"] == 0

    def test_no_event_on_normal_success(self) -> None:
        checker = HealthChecker("test_api")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            checker.record_success(latency_ms=5.0)
            mock_dispatch.assert_not_called()

    def test_no_event_on_failure_below_threshold(self) -> None:
        checker = HealthChecker("test_svc", failure_threshold=5)

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            checker.record_failure("error1")
            checker.record_failure("error2")
            mock_dispatch.assert_not_called()

    def test_degraded_event_includes_error_message(self) -> None:
        checker = HealthChecker("test_redis", failure_threshold=1)

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            checker.record_failure("Connection reset by peer")
            data = mock_dispatch.call_args[0][1]
            assert data["last_error"] == "Connection reset by peer"

    def test_import_error_handled_gracefully(self) -> None:
        checker = HealthChecker("test_svc", failure_threshold=1)

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            checker.record_failure("error")

        assert not checker.get_status().healthy

    def test_multiple_degradation_cycles(self) -> None:
        checker = HealthChecker("test_svc", failure_threshold=1, recovery_threshold=1)
        events: list[dict] = []

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=lambda name, data: events.append(data),
        ):
            checker.record_failure("err1")  # degraded
            checker.record_success()  # recovered
            checker.record_failure("err2")  # degraded again

        assert len(events) == 3
        assert events[0]["new_status"] == "degraded"
        assert events[1]["new_status"] == "recovered"
        assert events[2]["new_status"] == "degraded"

    def test_event_dispatched_as_health_state_change(self) -> None:
        checker = HealthChecker("test_svc", failure_threshold=1)

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            checker.record_failure("err")
            assert mock_dispatch.call_args[0][0] == "health_state_change"


class TestHealthRegistryEvents:
    """Tests for events via HealthRegistry."""

    def test_registry_checker_emits_events(self) -> None:
        registry = HealthRegistry()
        checker = registry.register("database", failure_threshold=1)

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            checker.record_failure("conn lost")
            mock_dispatch.assert_called_once()
            data = mock_dispatch.call_args[0][1]
            assert data["component"] == "database"
