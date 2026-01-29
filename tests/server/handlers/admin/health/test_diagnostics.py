"""Tests for deployment diagnostics implementations."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
import concurrent.futures
from typing import Any, Dict
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from enum import Enum

import pytest


class MockHandler:
    """Mock handler for testing diagnostics functions."""

    def __init__(self, ctx: Dict[str, Any] | None = None):
        self.ctx = ctx or {}


class MockSeverity(Enum):
    """Mock severity enum."""

    critical = "critical"
    warning = "warning"
    info = "info"


class MockStatus(Enum):
    """Mock status enum."""

    healthy = "healthy"
    degraded = "degraded"
    unhealthy = "unhealthy"


@dataclass
class MockIssue:
    """Mock issue dataclass."""

    severity: MockSeverity
    component: str
    message: str


@dataclass
class MockComponent:
    """Mock component dataclass."""

    name: str
    status: MockStatus


@dataclass
class MockValidationResult:
    """Mock validation result."""

    ready: bool
    live: bool
    issues: list
    components: list

    def to_dict(self):
        return {
            "ready": self.ready,
            "live": self.live,
            "issues": [
                {"severity": i.severity.value, "component": i.component, "message": i.message}
                for i in self.issues
            ],
            "components": [{"name": c.name, "status": c.status.value} for c in self.components],
        }


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


class TestDeploymentDiagnostics:
    """Tests for deployment_diagnostics function."""

    def test_diagnostics_ready(self):
        """Test diagnostics returns ready status."""
        from aragora.server.handlers.admin.health.diagnostics import deployment_diagnostics

        handler = MockHandler()

        mock_result = MockValidationResult(
            ready=True,
            live=True,
            issues=[],
            components=[
                MockComponent("database", MockStatus.healthy),
                MockComponent("redis", MockStatus.healthy),
            ],
        )

        with patch(
            "aragora.server.handlers.admin.health.diagnostics.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = deployment_diagnostics(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["ready"] is True
        assert body["live"] is True
        assert "summary" in body

    def test_diagnostics_not_ready(self):
        """Test diagnostics returns 503 when not ready."""
        from aragora.server.handlers.admin.health.diagnostics import deployment_diagnostics

        handler = MockHandler()

        mock_result = MockValidationResult(
            ready=False,
            live=True,
            issues=[
                MockIssue(MockSeverity.critical, "jwt_secret", "JWT secret too short"),
            ],
            components=[
                MockComponent("jwt_secret", MockStatus.unhealthy),
            ],
        )

        with patch(
            "aragora.server.handlers.admin.health.diagnostics.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = deployment_diagnostics(handler)

        assert result.status_code == 503
        body = json.loads(result.body.decode("utf-8"))
        assert body["ready"] is False

    def test_diagnostics_with_warnings(self):
        """Test diagnostics returns 200 with warnings."""
        from aragora.server.handlers.admin.health.diagnostics import deployment_diagnostics

        handler = MockHandler()

        mock_result = MockValidationResult(
            ready=True,
            live=True,
            issues=[
                MockIssue(MockSeverity.warning, "cors", "CORS allows all origins"),
            ],
            components=[
                MockComponent("cors", MockStatus.degraded),
            ],
        )

        with patch(
            "aragora.server.handlers.admin.health.diagnostics.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = deployment_diagnostics(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["summary"]["issues"]["warning"] == 1

    def test_diagnostics_module_not_available(self):
        """Test diagnostics returns 500 when validator not available."""
        from aragora.server.handlers.admin.health.diagnostics import deployment_diagnostics

        handler = MockHandler()

        with patch(
            "aragora.server.handlers.admin.health.diagnostics.validate_deployment",
            side_effect=ImportError("Module not found"),
        ):
            result = deployment_diagnostics(handler)

        assert result.status_code == 500
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "error"
        assert "not available" in body["error"]

    def test_diagnostics_timeout(self):
        """Test diagnostics returns 504 on timeout."""
        from aragora.server.handlers.admin.health.diagnostics import deployment_diagnostics

        handler = MockHandler()

        with patch(
            "aragora.server.handlers.admin.health.diagnostics.validate_deployment",
            new_callable=AsyncMock,
            side_effect=concurrent.futures.TimeoutError(),
        ):
            result = deployment_diagnostics(handler)

        assert result.status_code == 504
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "error"
        assert "timeout" in body["error"].lower()

    def test_diagnostics_unexpected_error(self):
        """Test diagnostics returns 500 on unexpected error."""
        from aragora.server.handlers.admin.health.diagnostics import deployment_diagnostics

        handler = MockHandler()

        with patch(
            "aragora.server.handlers.admin.health.diagnostics.validate_deployment",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = deployment_diagnostics(handler)

        assert result.status_code == 500
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "error"

    def test_diagnostics_includes_timestamp(self):
        """Test diagnostics includes timestamp."""
        from aragora.server.handlers.admin.health.diagnostics import deployment_diagnostics

        handler = MockHandler()

        mock_result = MockValidationResult(
            ready=True,
            live=True,
            issues=[],
            components=[],
        )

        with patch(
            "aragora.server.handlers.admin.health.diagnostics.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = deployment_diagnostics(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "timestamp" in body
        assert body["timestamp"].endswith("Z")

    def test_diagnostics_includes_response_time(self):
        """Test diagnostics includes response time."""
        from aragora.server.handlers.admin.health.diagnostics import deployment_diagnostics

        handler = MockHandler()

        mock_result = MockValidationResult(
            ready=True,
            live=True,
            issues=[],
            components=[],
        )

        with patch(
            "aragora.server.handlers.admin.health.diagnostics.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = deployment_diagnostics(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0


class TestGenerateChecklist:
    """Tests for _generate_checklist function."""

    def test_checklist_structure(self):
        """Test checklist has expected structure."""
        from aragora.server.handlers.admin.health.diagnostics import _generate_checklist

        mock_result = MockValidationResult(
            ready=True,
            live=True,
            issues=[],
            components=[
                MockComponent("jwt_secret", MockStatus.healthy),
                MockComponent("encryption", MockStatus.healthy),
                MockComponent("cors", MockStatus.healthy),
                MockComponent("database", MockStatus.healthy),
            ],
        )

        checklist = _generate_checklist(mock_result)

        assert "security" in checklist
        assert "infrastructure" in checklist
        assert "api" in checklist
        assert "environment" in checklist

    def test_checklist_security_items(self):
        """Test checklist includes security items."""
        from aragora.server.handlers.admin.health.diagnostics import _generate_checklist

        mock_result = MockValidationResult(
            ready=True,
            live=True,
            issues=[],
            components=[
                MockComponent("jwt_secret", MockStatus.healthy),
                MockComponent("encryption", MockStatus.healthy),
                MockComponent("cors", MockStatus.healthy),
                MockComponent("tls", MockStatus.healthy),
            ],
        )

        checklist = _generate_checklist(mock_result)

        assert "jwt_secret" in checklist["security"]
        assert "encryption_key" in checklist["security"]
        assert "cors" in checklist["security"]
        assert "tls" in checklist["security"]

    def test_checklist_infrastructure_items(self):
        """Test checklist includes infrastructure items."""
        from aragora.server.handlers.admin.health.diagnostics import _generate_checklist

        mock_result = MockValidationResult(
            ready=True,
            live=True,
            issues=[],
            components=[
                MockComponent("database", MockStatus.healthy),
                MockComponent("redis", MockStatus.healthy),
                MockComponent("storage", MockStatus.healthy),
                MockComponent("supabase", MockStatus.healthy),
            ],
        )

        checklist = _generate_checklist(mock_result)

        assert "database" in checklist["infrastructure"]
        assert "redis" in checklist["infrastructure"]
        assert "storage" in checklist["infrastructure"]
        assert "supabase" in checklist["infrastructure"]

    def test_checklist_pass_status(self):
        """Test checklist shows pass status for healthy components."""
        from aragora.server.handlers.admin.health.diagnostics import _generate_checklist

        mock_result = MockValidationResult(
            ready=True,
            live=True,
            issues=[],
            components=[
                MockComponent("jwt_secret", MockStatus.healthy),
            ],
        )

        checklist = _generate_checklist(mock_result)

        assert checklist["security"]["jwt_secret"]["status"] == "pass"

    def test_checklist_warning_status(self):
        """Test checklist shows warning status for degraded components."""
        from aragora.server.handlers.admin.health.diagnostics import _generate_checklist

        mock_result = MockValidationResult(
            ready=True,
            live=True,
            issues=[],
            components=[
                MockComponent("cors", MockStatus.degraded),
            ],
        )

        checklist = _generate_checklist(mock_result)

        assert checklist["security"]["cors"]["status"] == "warning"

    def test_checklist_fail_status(self):
        """Test checklist shows fail status for unhealthy components."""
        from aragora.server.handlers.admin.health.diagnostics import _generate_checklist

        mock_result = MockValidationResult(
            ready=False,
            live=True,
            issues=[],
            components=[
                MockComponent("database", MockStatus.unhealthy),
            ],
        )

        checklist = _generate_checklist(mock_result)

        assert checklist["infrastructure"]["database"]["status"] == "fail"

    def test_checklist_critical_issues(self):
        """Test checklist shows critical issues."""
        from aragora.server.handlers.admin.health.diagnostics import _generate_checklist

        mock_result = MockValidationResult(
            ready=False,
            live=True,
            issues=[
                MockIssue(MockSeverity.critical, "jwt_secret", "Too short"),
            ],
            components=[
                MockComponent("jwt_secret", MockStatus.unhealthy),
            ],
        )

        checklist = _generate_checklist(mock_result)

        assert checklist["security"]["jwt_secret"]["critical"] is True

    def test_checklist_not_checked_status(self):
        """Test checklist shows not_checked for missing components."""
        from aragora.server.handlers.admin.health.diagnostics import _generate_checklist

        mock_result = MockValidationResult(
            ready=True,
            live=True,
            issues=[],
            components=[],  # No components checked
        )

        checklist = _generate_checklist(mock_result)

        # All items should be not_checked
        assert checklist["security"]["jwt_secret"]["status"] == "not_checked"
