"""Comprehensive tests for agent_health module.

Tests all public functions in
aragora/server/handlers/admin/health/agent_health.py:

  TestClassifyAgentStatus       - _classify_agent_status() classification logic
  TestAgentHealthSummary        - agent_health_summary() with various scenarios
  TestAgentHealthDetail         - agent_health_detail() lookup and error handling
  TestAgentAvailabilityStatus   - agent_availability_status() availability checks
  TestEdgeCases                 - Empty data, None values, boundary conditions

60+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.health.agent_health import (
    _classify_agent_status,
    agent_availability_status,
    agent_health_detail,
    agent_health_summary,
)

_MOD = "aragora.server.handlers.admin.health.agent_health"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


@dataclass
class FakeIssue:
    severity: Any = "WARNING"
    message: str = "test issue"
    category: Any = "heartbeat_missing"


@dataclass
class FakeAgentHealth:
    agent_name: str = "claude-opus"
    last_heartbeat: datetime | None = None
    consecutive_failures: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    circuit_breaker_state: str = "closed"
    active_issues: list = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def average_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests


class FakeSeverity:
    name = "WARNING"


class FakeCategory:
    value = "heartbeat_missing"


@dataclass
class FakeIssueWithEnums:
    severity: Any = field(default_factory=FakeSeverity)
    message: str = "test issue"
    category: Any = field(default_factory=FakeCategory)


def _make_handler() -> MagicMock:
    return MagicMock()


# ===========================================================================
# TestClassifyAgentStatus
# ===========================================================================


class TestClassifyAgentStatus:
    """Tests for _classify_agent_status()."""

    def test_healthy_agent(self):
        health = FakeAgentHealth(total_requests=100, failed_requests=0)
        assert _classify_agent_status(health) == "healthy"

    def test_unhealthy_circuit_breaker_open(self):
        health = FakeAgentHealth(circuit_breaker_state="open")
        assert _classify_agent_status(health) == "unhealthy"

    def test_unhealthy_high_consecutive_failures(self):
        health = FakeAgentHealth(consecutive_failures=5)
        assert _classify_agent_status(health) == "unhealthy"

    def test_unhealthy_very_high_error_rate(self):
        health = FakeAgentHealth(total_requests=100, failed_requests=60)
        assert _classify_agent_status(health) == "unhealthy"

    def test_degraded_moderate_error_rate(self):
        health = FakeAgentHealth(total_requests=100, failed_requests=15)
        assert _classify_agent_status(health) == "degraded"

    def test_degraded_some_consecutive_failures(self):
        health = FakeAgentHealth(consecutive_failures=2)
        assert _classify_agent_status(health) == "degraded"

    def test_unhealthy_takes_precedence_over_degraded(self):
        health = FakeAgentHealth(
            circuit_breaker_state="open",
            consecutive_failures=2,
            total_requests=100,
            failed_requests=15,
        )
        assert _classify_agent_status(health) == "unhealthy"

    def test_boundary_error_rate_exactly_0_1(self):
        health = FakeAgentHealth(total_requests=100, failed_requests=10)
        assert _classify_agent_status(health) == "healthy"

    def test_boundary_error_rate_just_over_0_1(self):
        health = FakeAgentHealth(total_requests=100, failed_requests=11)
        assert _classify_agent_status(health) == "degraded"

    def test_boundary_error_rate_exactly_0_5(self):
        health = FakeAgentHealth(total_requests=100, failed_requests=50)
        assert _classify_agent_status(health) == "degraded"

    def test_boundary_error_rate_just_over_0_5(self):
        health = FakeAgentHealth(total_requests=100, failed_requests=51)
        assert _classify_agent_status(health) == "unhealthy"

    def test_boundary_consecutive_failures_1(self):
        health = FakeAgentHealth(consecutive_failures=1)
        assert _classify_agent_status(health) == "healthy"

    def test_boundary_consecutive_failures_4(self):
        health = FakeAgentHealth(consecutive_failures=4)
        assert _classify_agent_status(health) == "degraded"

    def test_zero_requests_healthy(self):
        health = FakeAgentHealth(total_requests=0)
        assert _classify_agent_status(health) == "healthy"

    def test_half_open_circuit_breaker(self):
        health = FakeAgentHealth(circuit_breaker_state="half_open")
        assert _classify_agent_status(health) == "healthy"


# ===========================================================================
# TestAgentHealthSummary
# ===========================================================================


class TestAgentHealthSummary:
    """Tests for agent_health_summary()."""

    def test_no_watchdog_no_registry(self):
        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            with patch(
                "aragora.control_plane.registry.get_default_registry",
                side_effect=ImportError,
            ):
                result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["status"] == "unknown"
        assert body["summary"]["total_agents"] == 0
        assert "watchdog module not available" in body["errors"]

    def test_watchdog_returns_agents(self):
        healthy_agent = FakeAgentHealth(
            agent_name="claude-opus",
            total_requests=100,
            failed_requests=0,
            last_heartbeat=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        degraded_agent = FakeAgentHealth(
            agent_name="gpt-4",
            total_requests=100,
            failed_requests=20,
        )

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {
            "claude-opus": healthy_agent,
            "gpt-4": degraded_agent,
        }

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["status"] == "degraded"
        assert body["summary"]["total_agents"] == 2
        assert body["summary"]["healthy"] == 1
        assert body["summary"]["degraded"] == 1
        assert body["errors"] is None
        assert body["agents"][0]["agent_name"] == "claude-opus"
        assert body["agents"][0]["status"] == "healthy"
        assert body["agents"][1]["status"] == "degraded"

    def test_watchdog_import_error_falls_back_to_registry(self):
        mock_registry = MagicMock()
        mock_registry._agents = {"agent-a": MagicMock(), "agent-b": MagicMock()}

        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            with patch(
                "aragora.control_plane.registry.get_default_registry",
                return_value=mock_registry,
            ):
                result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["summary"]["total_agents"] == 2
        assert all(a["status"] == "unknown" for a in body["agents"])

    def test_watchdog_returns_none(self):
        mock_registry = MagicMock()
        mock_registry._agents = {"agent-x": MagicMock()}

        with patch(f"{_MOD}._get_watchdog", return_value=None):
            with patch(
                "aragora.control_plane.registry.get_default_registry",
                return_value=mock_registry,
            ):
                result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["summary"]["total_agents"] == 1
        assert body["agents"][0]["agent_name"] == "agent-x"

    def test_watchdog_runtime_error(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.side_effect = RuntimeError("boom")

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert "watchdog error: RuntimeError" in body["errors"]

    def test_all_unhealthy_agents(self):
        agent1 = FakeAgentHealth(agent_name="a1", circuit_breaker_state="open")
        agent2 = FakeAgentHealth(agent_name="a2", consecutive_failures=10)

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"a1": agent1, "a2": agent2}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["status"] == "unhealthy"
        assert body["summary"]["unhealthy"] == 2

    def test_all_healthy_agents(self):
        agent1 = FakeAgentHealth(agent_name="a1", total_requests=10, failed_requests=0)
        agent2 = FakeAgentHealth(agent_name="a2", total_requests=5, failed_requests=0)

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"a1": agent1, "a2": agent2}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["status"] == "healthy"

    def test_timestamp_present(self):
        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            with patch(
                "aragora.control_plane.registry.get_default_registry",
                side_effect=ImportError,
            ):
                result = agent_health_summary(_make_handler())

        body = _body(result)
        assert "timestamp" in body
        assert body["timestamp"].endswith("Z")

    def test_registry_returns_none(self):
        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            with patch(
                "aragora.control_plane.registry.get_default_registry",
                return_value=None,
            ):
                result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["status"] == "unknown"
        assert body["summary"]["total_agents"] == 0

    def test_registry_runtime_error(self):
        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            with patch(
                "aragora.control_plane.registry.get_default_registry",
                side_effect=RuntimeError("fail"),
            ):
                result = agent_health_summary(_make_handler())

        body = _body(result)
        assert any("registry error" in e for e in body["errors"])

    def test_agent_with_heartbeat_serialized(self):
        ts = datetime(2026, 4, 12, 10, 30, 0, tzinfo=timezone.utc)
        agent = FakeAgentHealth(agent_name="test", last_heartbeat=ts, total_requests=1)

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"test": agent}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["agents"][0]["last_heartbeat"] == "2026-04-12T10:30:00+00:00Z"

    def test_agent_without_heartbeat(self):
        agent = FakeAgentHealth(agent_name="test", last_heartbeat=None)

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"test": agent}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["agents"][0]["last_heartbeat"] is None

    def test_watchdog_with_empty_agents(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["status"] == "unknown"
        assert body["summary"]["total_agents"] == 0

    def test_memory_usage_rounded(self):
        agent = FakeAgentHealth(agent_name="test", memory_usage_mb=123.456789)

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"test": agent}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["agents"][0]["memory_usage_mb"] == 123.46

    def test_active_issues_count(self):
        agent = FakeAgentHealth(
            agent_name="test",
            active_issues=[FakeIssue(), FakeIssue(), FakeIssue()],
        )

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"test": agent}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["agents"][0]["active_issues"] == 3


# ===========================================================================
# TestAgentHealthDetail
# ===========================================================================


class TestAgentHealthDetail:
    """Tests for agent_health_detail()."""

    def test_agent_found(self):
        agent = FakeAgentHealth(
            agent_name="claude-opus",
            total_requests=50,
            failed_requests=2,
            total_latency_ms=500.0,
            memory_usage_mb=64.5,
            consecutive_failures=1,
            last_heartbeat=datetime(2026, 4, 12, tzinfo=timezone.utc),
        )

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"claude-opus": agent}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "claude-opus")

        assert _status(result) == 200
        body = _body(result)
        assert body["agent_name"] == "claude-opus"
        assert body["total_requests"] == 50
        assert body["consecutive_failures"] == 1
        assert body["total_latency_ms"] == 500.0
        assert body["active_issues"] == []

    def test_agent_not_found(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "nonexistent")

        assert _status(result) == 404
        body = _body(result)
        assert "Agent not found" in body["error"]

    def test_watchdog_import_error(self):
        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            result = agent_health_detail(_make_handler(), "any")

        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"]

    def test_watchdog_runtime_error(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.side_effect = RuntimeError("boom")

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "test")

        assert _status(result) == 500
        body = _body(result)
        assert body["error"] == "Health check failed"

    def test_watchdog_returns_none(self):
        with patch(f"{_MOD}._get_watchdog", return_value=None):
            result = agent_health_detail(_make_handler(), "test")

        assert _status(result) == 404

    def test_active_issues_serialized_with_enums(self):
        issue = FakeIssueWithEnums()
        agent = FakeAgentHealth(
            agent_name="test",
            active_issues=[issue],
        )

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"test": agent}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "test")

        body = _body(result)
        assert len(body["active_issues"]) == 1
        assert body["active_issues"][0]["severity"] == "WARNING"
        assert body["active_issues"][0]["category"] == "heartbeat_missing"
        assert body["active_issues"][0]["message"] == "test issue"

    def test_active_issues_serialized_with_strings(self):
        issue = FakeIssue(severity="CRITICAL", message="bad", category="memory_exceeded")
        agent = FakeAgentHealth(agent_name="test", active_issues=[issue])

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"test": agent}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "test")

        body = _body(result)
        assert body["active_issues"][0]["severity"] == "CRITICAL"

    def test_error_rate_rounded(self):
        agent = FakeAgentHealth(
            agent_name="test",
            total_requests=3,
            failed_requests=1,
        )

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"test": agent}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "test")

        body = _body(result)
        assert body["error_rate"] == round(1 / 3, 4)

    def test_timestamp_in_detail_response(self):
        agent = FakeAgentHealth(agent_name="test")

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"test": agent}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "test")

        body = _body(result)
        assert "timestamp" in body

    def test_attribute_error_returns_500(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.side_effect = AttributeError("no attr")

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "test")

        assert _status(result) == 500

    def test_type_error_returns_500(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.side_effect = TypeError("bad type")

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "test")

        assert _status(result) == 500


# ===========================================================================
# TestAgentAvailabilityStatus
# ===========================================================================


class TestAgentAvailabilityStatus:
    """Tests for agent_availability_status()."""

    def test_all_available(self):
        a1 = FakeAgentHealth(agent_name="a1", circuit_breaker_state="closed")
        a2 = FakeAgentHealth(agent_name="a2", circuit_breaker_state="closed")

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"a1": a1, "a2": a2}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert body["status"] == "all_available"
        assert body["available_count"] == 2
        assert body["unavailable_count"] == 0

    def test_none_available_circuit_breaker(self):
        a1 = FakeAgentHealth(agent_name="a1", circuit_breaker_state="open")

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"a1": a1}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert body["status"] == "none_available"
        assert body["unavailable_count"] == 1
        assert body["unavailable"][0]["reason"] == "circuit_breaker_open"

    def test_none_available_consecutive_failures(self):
        a1 = FakeAgentHealth(agent_name="a1", consecutive_failures=5)

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"a1": a1}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert body["status"] == "none_available"
        assert body["unavailable"][0]["reason"] == "consecutive_failures"

    def test_partial_availability(self):
        a1 = FakeAgentHealth(agent_name="a1", circuit_breaker_state="closed")
        a2 = FakeAgentHealth(agent_name="a2", circuit_breaker_state="open")

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"a1": a1, "a2": a2}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert body["status"] == "partial"
        assert body["available_count"] == 1
        assert body["unavailable_count"] == 1

    def test_unknown_when_no_agents(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert body["status"] == "unknown"

    def test_watchdog_import_error(self):
        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert body["status"] == "unknown"
        assert "watchdog module not available" in body["errors"]

    def test_watchdog_runtime_error(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.side_effect = RuntimeError("boom")

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert "watchdog error: RuntimeError" in body["errors"]

    def test_watchdog_returns_none(self):
        with patch(f"{_MOD}._get_watchdog", return_value=None):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert body["status"] == "unknown"
        assert body["available_count"] == 0

    def test_errors_null_when_no_errors(self):
        a1 = FakeAgentHealth(agent_name="a1")

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"a1": a1}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert body["errors"] is None

    def test_timestamp_present(self):
        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert body["timestamp"].endswith("Z")

    def test_consecutive_failures_boundary_4(self):
        a1 = FakeAgentHealth(agent_name="a1", consecutive_failures=4)

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"a1": a1}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert body["available_count"] == 1

    def test_consecutive_failures_boundary_5(self):
        a1 = FakeAgentHealth(agent_name="a1", consecutive_failures=5)

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"a1": a1}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert body["unavailable_count"] == 1

    def test_os_error_handled(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.side_effect = OSError("disk")

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert "watchdog error: OSError" in body["errors"]


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_summary_watchdog_value_error(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.side_effect = ValueError("bad")

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert any("ValueError" in e for e in body["errors"])

    def test_summary_watchdog_os_error(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.side_effect = OSError("disk")

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert any("OSError" in e for e in body["errors"])

    def test_detail_value_error(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.side_effect = ValueError("v")

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "x")

        assert _status(result) == 500

    def test_detail_os_error(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.side_effect = OSError("o")

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "x")

        assert _status(result) == 500

    def test_summary_registry_attribute_error(self):
        """When watchdog import fails, registry fallback is tried and may also error."""
        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            with patch(
                "aragora.control_plane.registry.get_default_registry",
                side_effect=AttributeError("a"),
            ):
                result = agent_health_summary(_make_handler())

        body = _body(result)
        assert any("registry error" in e for e in body["errors"])

    def test_agent_with_zero_latency(self):
        agent = FakeAgentHealth(agent_name="test", total_latency_ms=0.0, total_requests=5)

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"test": agent}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "test")

        body = _body(result)
        assert body["average_latency_ms"] == 0.0

    def test_summary_mixed_statuses(self):
        agents = {
            "a": FakeAgentHealth(agent_name="a", total_requests=10, failed_requests=0),
            "b": FakeAgentHealth(agent_name="b", total_requests=10, failed_requests=2),
            "c": FakeAgentHealth(agent_name="c", circuit_breaker_state="open"),
        }

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = agents

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["status"] == "unhealthy"
        assert body["summary"]["healthy"] == 1
        assert body["summary"]["degraded"] == 1
        assert body["summary"]["unhealthy"] == 1

    def test_detail_multiple_issues(self):
        issues = [
            FakeIssueWithEnums(),
            FakeIssue(severity="CRITICAL", message="mem exceeded", category="memory_exceeded"),
        ]
        agent = FakeAgentHealth(agent_name="test", active_issues=issues)

        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {"test": agent}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "test")

        body = _body(result)
        assert len(body["active_issues"]) == 2

    def test_availability_value_error(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.side_effect = ValueError("val")

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_availability_status(_make_handler())

        body = _body(result)
        assert any("ValueError" in e for e in body["errors"])

    def test_summary_http_status_is_200(self):
        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            with patch(
                "aragora.control_plane.registry.get_default_registry",
                side_effect=ImportError,
            ):
                result = agent_health_summary(_make_handler())

        assert _status(result) == 200

    def test_availability_http_status_is_200(self):
        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            result = agent_availability_status(_make_handler())

        assert _status(result) == 200

    def test_detail_not_found_includes_agent_id(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_detail(_make_handler(), "my-special-agent")

        body = _body(result)
        assert "my-special-agent" in body["error"]

    def test_registry_import_error(self):
        with patch(f"{_MOD}._get_watchdog", side_effect=ImportError):
            with patch(
                "aragora.control_plane.registry.get_default_registry",
                side_effect=ImportError,
            ):
                result = agent_health_summary(_make_handler())

        body = _body(result)
        assert "registry module not available" in body["errors"]

    def test_watchdog_empty_but_no_error_skips_registry(self):
        mock_watchdog = MagicMock()
        mock_watchdog.get_all_health.return_value = {}

        with patch(f"{_MOD}._get_watchdog", return_value=mock_watchdog):
            result = agent_health_summary(_make_handler())

        body = _body(result)
        assert body["status"] == "unknown"
        assert body["errors"] is None
