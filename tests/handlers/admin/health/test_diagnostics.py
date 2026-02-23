"""Comprehensive tests for deployment diagnostics handler.

Tests the public functions in aragora/server/handlers/admin/health/diagnostics.py:

  TestCheckDiagnosticsPermission  - _check_diagnostics_permission() RBAC checks
  TestDeploymentDiagnostics       - deployment_diagnostics() full endpoint tests
  TestGenerateChecklist           - _generate_checklist() production readiness
  TestDiagnosticsViaHealthHandler - Integration via HealthHandler routing

90+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.server.handlers.admin.health import HealthHandler
from aragora.server.handlers.admin.health.diagnostics import (
    _check_diagnostics_permission,
    _generate_checklist,
    deployment_diagnostics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: dict[str, Any] | None = None):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
            }
        else:
            self.rfile.read.return_value = b""
            self.headers = {
                "Content-Length": "0",
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
            }
        self.client_address = ("127.0.0.1", 12345)


# ---------------------------------------------------------------------------
# Mock deployment validator types
# ---------------------------------------------------------------------------


class MockSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class MockComponentStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MockValidationIssue:
    component: str
    message: str
    severity: MockSeverity
    suggestion: str | None = None


@dataclass
class MockComponentHealth:
    name: str
    status: MockComponentStatus
    latency_ms: float | None = None
    message: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MockValidationResult:
    ready: bool
    live: bool
    issues: list[MockValidationIssue] = field(default_factory=list)
    components: list[MockComponentHealth] = field(default_factory=list)
    validated_at: float = field(default_factory=time.time)
    validation_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "live": self.live,
            "issues": [
                {
                    "component": i.component,
                    "message": i.message,
                    "severity": i.severity.value,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": c.latency_ms,
                    "message": c.message,
                    "metadata": c.metadata,
                }
                for c in self.components
            ],
            "validated_at": self.validated_at,
            "validation_duration_ms": self.validation_duration_ms,
        }


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

# Patch target for the source module where validate_deployment is defined
_VALIDATOR_MOD = "aragora.ops.deployment_validator"
# Patch base for diagnostics module-level imports
_MOD = "aragora.server.handlers.admin.health.diagnostics"


def _make_result(
    *,
    ready: bool = True,
    live: bool = True,
    issues: list[MockValidationIssue] | None = None,
    components: list[MockComponentHealth] | None = None,
) -> MockValidationResult:
    """Build a MockValidationResult with sensible defaults."""
    return MockValidationResult(
        ready=ready,
        live=live,
        issues=issues or [],
        components=components or [],
    )


def _make_handler(ctx: dict[str, Any] | None = None) -> HealthHandler:
    """Create a HealthHandler with the given context."""
    return HealthHandler(ctx=ctx or {})


def _run_diagnostics(result_obj, *, use_loop=False):
    """Run deployment_diagnostics with a mocked validator and asyncio.

    Args:
        result_obj: The MockValidationResult to return from validation.
        use_loop: If True, simulate being inside a running event loop.
    """
    mock_validate = AsyncMock(return_value=result_obj)

    with (
        patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
        patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
        patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
    ):
        if use_loop:
            # Simulate running event loop -> uses ThreadPoolExecutor
            mock_asyncio_mod.get_running_loop.return_value = MagicMock()
            # When the thread pool calls asyncio.run(coroutine), return result_obj
            mock_asyncio_mod.run.return_value = result_obj
            # Set up the ThreadPoolExecutor mock
            with patch(f"{_MOD}.concurrent.futures.ThreadPoolExecutor") as mock_pool:
                mock_executor = MagicMock()
                mock_pool.return_value.__enter__ = MagicMock(return_value=mock_executor)
                mock_pool.return_value.__exit__ = MagicMock(return_value=False)
                mock_future = MagicMock()
                mock_future.result.return_value = result_obj
                mock_executor.submit.return_value = mock_future
                return deployment_diagnostics(MockHTTPHandler())
        else:
            # No running loop -> uses asyncio.run()
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.return_value = result_obj
            return deployment_diagnostics(MockHTTPHandler())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Default handler with empty context."""
    return _make_handler()


@pytest.fixture
def mock_http():
    """Default mock HTTP handler."""
    return MockHTTPHandler()


# ===========================================================================
# TestCheckDiagnosticsPermission
# ===========================================================================


class TestCheckDiagnosticsPermission:
    """Tests for _check_diagnostics_permission.

    Note: auth_config, extract_user_from_request, and get_permission_checker
    are imported locally inside _check_diagnostics_permission, so we must
    patch them at their source module, not on the diagnostics module.
    """

    # Source-level patch targets for local imports
    _AUTH_CONFIG = "aragora.server.auth.auth_config"
    _EXTRACT = "aragora.billing.jwt_auth.extract_user_from_request"
    _GET_CHECKER = "aragora.rbac.checker.get_permission_checker"

    def test_auth_disabled_allows_access(self):
        """When auth is disabled globally, permission check returns None (allow)."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = False
        with patch(self._AUTH_CONFIG, mock_auth_config), patch(self._EXTRACT):
            result = _check_diagnostics_permission(MockHTTPHandler())
        assert result is None

    def test_unauthenticated_returns_401(self):
        """Unauthenticated user gets 401."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = True
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = False
        with (
            patch(self._AUTH_CONFIG, mock_auth_config),
            patch(self._EXTRACT, return_value=mock_ctx),
        ):
            result = _check_diagnostics_permission(MockHTTPHandler())
        assert _status(result) == 401
        assert "Authentication required" in _body(result).get("error", "")

    def test_authenticated_with_permission_allows(self):
        """Authenticated user with admin:diagnostics permission is allowed."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = True
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = True
        mock_ctx.user_id = "user-1"
        mock_ctx.roles = ["admin"]
        mock_ctx.org_id = "org-1"

        mock_checker = MagicMock()
        perm_result = MagicMock()
        perm_result.allowed = True
        mock_checker.check_permission.return_value = perm_result

        with (
            patch(self._AUTH_CONFIG, mock_auth_config),
            patch(self._EXTRACT, return_value=mock_ctx),
            patch(self._GET_CHECKER, return_value=mock_checker),
        ):
            result = _check_diagnostics_permission(MockHTTPHandler())
        assert result is None

    def test_authenticated_without_permission_returns_403(self):
        """Authenticated user without permission gets 403."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = True
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = True
        mock_ctx.user_id = "user-2"
        mock_ctx.roles = ["viewer"]
        mock_ctx.org_id = "org-1"

        mock_checker = MagicMock()
        perm_result = MagicMock()
        perm_result.allowed = False
        mock_checker.check_permission.return_value = perm_result

        with (
            patch(self._AUTH_CONFIG, mock_auth_config),
            patch(self._EXTRACT, return_value=mock_ctx),
            patch(self._GET_CHECKER, return_value=mock_checker),
        ):
            result = _check_diagnostics_permission(MockHTTPHandler())
        assert _status(result) == 403
        assert "Permission denied" in _body(result).get("error", "")

    def test_rbac_import_error_allows_authenticated(self):
        """When RBAC module cannot be imported, authenticated user is allowed."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = True
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = True
        mock_ctx.user_id = "user-3"

        with (
            patch(self._AUTH_CONFIG, mock_auth_config),
            patch(self._EXTRACT, return_value=mock_ctx),
            patch(self._GET_CHECKER, side_effect=ImportError("no rbac")),
        ):
            result = _check_diagnostics_permission(MockHTTPHandler())
        assert result is None

    def test_checker_is_none_allows_authenticated(self):
        """When permission checker returns None, authenticated user is allowed."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = True
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = True
        mock_ctx.user_id = "user-4"

        with (
            patch(self._AUTH_CONFIG, mock_auth_config),
            patch(self._EXTRACT, return_value=mock_ctx),
            patch(self._GET_CHECKER, return_value=None),
        ):
            result = _check_diagnostics_permission(MockHTTPHandler())
        assert result is None

    def test_user_without_user_id_attr_allows(self):
        """User without user_id attribute skips permission check (allowed)."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = True
        mock_ctx = MagicMock(spec=[])  # empty spec, no user_id
        mock_ctx.is_authenticated = True

        mock_checker = MagicMock()

        with (
            patch(self._AUTH_CONFIG, mock_auth_config),
            patch(self._EXTRACT, return_value=mock_ctx),
            patch(self._GET_CHECKER, return_value=mock_checker),
        ):
            result = _check_diagnostics_permission(MockHTTPHandler())
        assert result is None

    def test_user_with_none_roles(self):
        """User with None roles gets empty roles set in auth context."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = True
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = True
        mock_ctx.user_id = "user-5"
        mock_ctx.roles = None
        mock_ctx.org_id = None

        mock_checker = MagicMock()
        perm_result = MagicMock()
        perm_result.allowed = True
        mock_checker.check_permission.return_value = perm_result

        with (
            patch(self._AUTH_CONFIG, mock_auth_config),
            patch(self._EXTRACT, return_value=mock_ctx),
            patch(self._GET_CHECKER, return_value=mock_checker),
        ):
            result = _check_diagnostics_permission(MockHTTPHandler())
        assert result is None

        # Verify empty roles set was passed
        call_args = mock_checker.check_permission.call_args
        auth_ctx_arg = call_args[0][0]
        assert auth_ctx_arg.roles == set()

    def test_user_with_empty_list_roles(self):
        """User with empty list roles gets empty roles set."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = True
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = True
        mock_ctx.user_id = "user-6"
        mock_ctx.roles = []
        mock_ctx.org_id = "org-2"

        mock_checker = MagicMock()
        perm_result = MagicMock()
        perm_result.allowed = True
        mock_checker.check_permission.return_value = perm_result

        with (
            patch(self._AUTH_CONFIG, mock_auth_config),
            patch(self._EXTRACT, return_value=mock_ctx),
            patch(self._GET_CHECKER, return_value=mock_checker),
        ):
            result = _check_diagnostics_permission(MockHTTPHandler())
        assert result is None

        call_args = mock_checker.check_permission.call_args
        auth_ctx_arg = call_args[0][0]
        assert auth_ctx_arg.roles == set()

    def test_permission_check_uses_correct_permission_string(self):
        """Verify checker is called with 'admin:diagnostics' permission."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = True
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = True
        mock_ctx.user_id = "user-7"
        mock_ctx.roles = ["admin"]
        mock_ctx.org_id = "org-1"

        mock_checker = MagicMock()
        perm_result = MagicMock()
        perm_result.allowed = True
        mock_checker.check_permission.return_value = perm_result

        with (
            patch(self._AUTH_CONFIG, mock_auth_config),
            patch(self._EXTRACT, return_value=mock_ctx),
            patch(self._GET_CHECKER, return_value=mock_checker),
        ):
            _check_diagnostics_permission(MockHTTPHandler())

        mock_checker.check_permission.assert_called_once()
        call_args = mock_checker.check_permission.call_args
        assert call_args[0][1] == "admin:diagnostics"

    def test_auth_context_has_correct_user_id(self):
        """Verify AuthorizationContext is built with the correct user_id."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = True
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = True
        mock_ctx.user_id = "my-user-42"
        mock_ctx.roles = ["editor"]
        mock_ctx.org_id = "my-org-99"

        mock_checker = MagicMock()
        perm_result = MagicMock()
        perm_result.allowed = True
        mock_checker.check_permission.return_value = perm_result

        with (
            patch(self._AUTH_CONFIG, mock_auth_config),
            patch(self._EXTRACT, return_value=mock_ctx),
            patch(self._GET_CHECKER, return_value=mock_checker),
        ):
            _check_diagnostics_permission(MockHTTPHandler())

        call_args = mock_checker.check_permission.call_args
        auth_ctx_arg = call_args[0][0]
        assert auth_ctx_arg.user_id == "my-user-42"
        assert auth_ctx_arg.org_id == "my-org-99"
        assert auth_ctx_arg.roles == {"editor"}

    def test_multiple_roles_converted_to_set(self):
        """Multiple roles in list are converted to a set."""
        mock_auth_config = MagicMock()
        mock_auth_config.enabled = True
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = True
        mock_ctx.user_id = "user-multi"
        mock_ctx.roles = ["admin", "editor", "viewer"]
        mock_ctx.org_id = "org-1"

        mock_checker = MagicMock()
        perm_result = MagicMock()
        perm_result.allowed = True
        mock_checker.check_permission.return_value = perm_result

        with (
            patch(self._AUTH_CONFIG, mock_auth_config),
            patch(self._EXTRACT, return_value=mock_ctx),
            patch(self._GET_CHECKER, return_value=mock_checker),
        ):
            _check_diagnostics_permission(MockHTTPHandler())

        call_args = mock_checker.check_permission.call_args
        auth_ctx_arg = call_args[0][0]
        assert auth_ctx_arg.roles == {"admin", "editor", "viewer"}


# ===========================================================================
# TestDeploymentDiagnostics
# ===========================================================================


class TestDeploymentDiagnostics:
    """Tests for deployment_diagnostics endpoint."""

    # --- Success cases ---

    def test_ready_no_issues(self):
        """Deployment ready with no issues returns 200."""
        result_obj = _make_result(ready=True, live=True)
        result = _run_diagnostics(result_obj)

        assert _status(result) == 200
        body = _body(result)
        assert body["ready"] is True
        assert body["live"] is True
        assert "summary" in body
        assert "checklist" in body
        assert body["summary"]["issues"]["total"] == 0

    def test_ready_with_warnings(self):
        """Deployment ready with warnings returns 200."""
        issues = [
            MockValidationIssue("cors", "Wildcard CORS", MockSeverity.WARNING),
        ]
        result_obj = _make_result(ready=True, live=True, issues=issues)
        result = _run_diagnostics(result_obj)

        assert _status(result) == 200
        body = _body(result)
        assert body["summary"]["issues"]["warning"] == 1
        assert body["summary"]["issues"]["critical"] == 0

    def test_not_ready_returns_503(self):
        """Deployment not ready returns 503."""
        issues = [
            MockValidationIssue("jwt_secret", "Weak secret", MockSeverity.CRITICAL),
        ]
        result_obj = _make_result(ready=False, live=True, issues=issues)
        result = _run_diagnostics(result_obj)

        assert _status(result) == 503
        body = _body(result)
        assert body["ready"] is False
        assert body["summary"]["issues"]["critical"] == 1

    def test_response_includes_timestamp(self):
        """Response includes ISO timestamp."""
        result_obj = _make_result()
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert "timestamp" in body
        assert body["timestamp"].endswith("Z")

    def test_response_includes_response_time(self):
        """Response includes response_time_ms."""
        result_obj = _make_result()
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert "response_time_ms" in body
        assert isinstance(body["response_time_ms"], float)

    # --- Component summary ---

    def test_component_summary_all_healthy(self):
        """Component summary counts all healthy components."""
        components = [
            MockComponentHealth("jwt_secret", MockComponentStatus.HEALTHY),
            MockComponentHealth("database", MockComponentStatus.HEALTHY),
            MockComponentHealth("redis", MockComponentStatus.HEALTHY),
        ]
        result_obj = _make_result(components=components)
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert body["summary"]["components"]["healthy"] == 3
        assert body["summary"]["components"]["degraded"] == 0
        assert body["summary"]["components"]["unhealthy"] == 0
        assert body["summary"]["components"]["total"] == 3

    def test_component_summary_mixed_status(self):
        """Component summary counts mixed component states."""
        components = [
            MockComponentHealth("jwt_secret", MockComponentStatus.HEALTHY),
            MockComponentHealth("database", MockComponentStatus.DEGRADED),
            MockComponentHealth("redis", MockComponentStatus.UNHEALTHY),
        ]
        result_obj = _make_result(components=components)
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert body["summary"]["components"]["healthy"] == 1
        assert body["summary"]["components"]["degraded"] == 1
        assert body["summary"]["components"]["unhealthy"] == 1
        assert body["summary"]["components"]["total"] == 3

    def test_issue_summary_all_severities(self):
        """Issue summary counts all severity levels."""
        issues = [
            MockValidationIssue("a", "Critical issue", MockSeverity.CRITICAL),
            MockValidationIssue("b", "Warning 1", MockSeverity.WARNING),
            MockValidationIssue("c", "Warning 2", MockSeverity.WARNING),
            MockValidationIssue("d", "Info", MockSeverity.INFO),
        ]
        result_obj = _make_result(ready=False, issues=issues)
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert body["summary"]["issues"]["critical"] == 1
        assert body["summary"]["issues"]["warning"] == 2
        assert body["summary"]["issues"]["info"] == 1
        assert body["summary"]["issues"]["total"] == 4

    def test_empty_issues_and_components(self):
        """No issues and no components results in zero counts."""
        result_obj = _make_result()
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert body["summary"]["issues"]["total"] == 0
        assert body["summary"]["components"]["total"] == 0

    # --- Permission delegation ---

    def test_permission_error_returned(self):
        """When permission check returns error, it is returned directly."""
        from aragora.server.handlers.utils.responses import error_response as _err

        perm_error = _err("Forbidden", 403)
        with patch(
            f"{_MOD}._check_diagnostics_permission",
            return_value=perm_error,
        ):
            result = deployment_diagnostics(MockHTTPHandler())

        assert _status(result) == 403

    def test_401_from_permission_check(self):
        """401 from permission check is propagated."""
        from aragora.server.handlers.utils.responses import error_response as _err

        perm_error = _err("Not authenticated", 401)
        with patch(
            f"{_MOD}._check_diagnostics_permission",
            return_value=perm_error,
        ):
            result = deployment_diagnostics(MockHTTPHandler())

        assert _status(result) == 401

    # --- Error handling ---

    def test_import_error_returns_500(self):
        """ImportError for deployment validator returns 500."""
        import builtins

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if "deployment_validator" in name:
                raise ImportError("Deployment validator not available")
            return original_import(name, *args, **kwargs)

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch("builtins.__import__", side_effect=_mock_import),
        ):
            result = deployment_diagnostics(MockHTTPHandler())

        assert _status(result) == 500
        body = _body(result)
        assert body["status"] == "error"
        assert "not available" in body["error"]
        assert "timestamp" in body
        assert "response_time_ms" in body

    def test_timeout_error_returns_504(self):
        """ThreadPoolExecutor timeout returns 504."""
        result_obj = _make_result()
        mock_validate = AsyncMock(return_value=result_obj)

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
            patch(f"{_MOD}.concurrent.futures.ThreadPoolExecutor") as mock_pool,
        ):
            mock_asyncio_mod.get_running_loop.return_value = MagicMock()
            mock_executor = MagicMock()
            mock_pool.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)
            mock_future = MagicMock()
            mock_future.result.side_effect = concurrent.futures.TimeoutError()
            mock_executor.submit.return_value = mock_future
            result = deployment_diagnostics(MockHTTPHandler())

        assert _status(result) == 504
        body = _body(result)
        assert body["status"] == "error"
        assert "timed out" in body["error"]
        assert "30 seconds" in body["error"]

    def test_runtime_error_returns_500(self):
        """RuntimeError during validation returns 500."""
        mock_validate = AsyncMock(return_value=_make_result())

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.side_effect = RuntimeError("Broken")
            result = deployment_diagnostics(MockHTTPHandler())

        assert _status(result) == 500
        body = _body(result)
        assert body["status"] == "error"
        assert "check failed" in body["error"]

    def test_value_error_returns_500(self):
        """ValueError during validation returns 500."""
        mock_validate = AsyncMock(return_value=_make_result())

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.side_effect = ValueError("bad value")
            result = deployment_diagnostics(MockHTTPHandler())

        assert _status(result) == 500
        body = _body(result)
        assert body["status"] == "error"

    def test_type_error_returns_500(self):
        """TypeError during validation returns 500."""
        mock_validate = AsyncMock(return_value=_make_result())

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.side_effect = TypeError("wrong type")
            result = deployment_diagnostics(MockHTTPHandler())

        assert _status(result) == 500

    def test_key_error_returns_500(self):
        """KeyError during validation returns 500."""
        mock_validate = AsyncMock(return_value=_make_result())

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.side_effect = KeyError("missing key")
            result = deployment_diagnostics(MockHTTPHandler())

        assert _status(result) == 500

    def test_attribute_error_returns_500(self):
        """AttributeError during validation returns 500."""
        mock_validate = AsyncMock(return_value=_make_result())

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.side_effect = AttributeError("no attr")
            result = deployment_diagnostics(MockHTTPHandler())

        assert _status(result) == 500

    def test_os_error_returns_500(self):
        """OSError during validation returns 500."""
        mock_validate = AsyncMock(return_value=_make_result())

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.side_effect = OSError("disk error")
            result = deployment_diagnostics(MockHTTPHandler())

        assert _status(result) == 500

    def test_error_response_includes_timestamp(self):
        """Error responses still include timestamp."""
        mock_validate = AsyncMock(return_value=_make_result())

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.side_effect = RuntimeError("Broken")
            result = deployment_diagnostics(MockHTTPHandler())

        body = _body(result)
        assert "timestamp" in body
        assert body["timestamp"].endswith("Z")

    def test_error_response_includes_response_time(self):
        """Error responses still include response_time_ms."""
        mock_validate = AsyncMock(return_value=_make_result())

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.side_effect = RuntimeError("Broken")
            result = deployment_diagnostics(MockHTTPHandler())

        body = _body(result)
        assert "response_time_ms" in body
        assert isinstance(body["response_time_ms"], float)

    # --- Async context handling ---

    def test_no_running_loop_uses_asyncio_run(self):
        """When no event loop is running, asyncio.run() is used directly."""
        result_obj = _make_result()
        mock_validate = AsyncMock(return_value=result_obj)

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.return_value = result_obj
            deployment_diagnostics(MockHTTPHandler())

        mock_asyncio_mod.run.assert_called_once()

    def test_running_loop_uses_thread_pool(self):
        """When event loop is running, ThreadPoolExecutor is used."""
        result_obj = _make_result()
        result = _run_diagnostics(result_obj, use_loop=True)

        assert _status(result) == 200

    # --- Checklist inclusion ---

    def test_response_includes_checklist(self):
        """Response data includes a checklist section."""
        result_obj = _make_result()
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert "checklist" in body
        assert "security" in body["checklist"]
        assert "infrastructure" in body["checklist"]
        assert "api" in body["checklist"]
        assert "environment" in body["checklist"]

    # --- Multiple critical issues ---

    def test_multiple_critical_issues_counted(self):
        """Multiple critical issues are all counted in summary."""
        issues = [
            MockValidationIssue("jwt_secret", "Weak", MockSeverity.CRITICAL),
            MockValidationIssue("encryption", "Missing", MockSeverity.CRITICAL),
            MockValidationIssue("api_keys", "None configured", MockSeverity.CRITICAL),
        ]
        result_obj = _make_result(ready=False, issues=issues)
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert body["summary"]["issues"]["critical"] == 3

    # --- Info-only issues ---

    def test_info_only_issues(self):
        """Info-only issues counted correctly, deployment stays ready."""
        issues = [
            MockValidationIssue("env", "Using dev mode", MockSeverity.INFO),
            MockValidationIssue("tls", "Not configured", MockSeverity.INFO),
        ]
        result_obj = _make_result(ready=True, issues=issues)
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert _status(result) == 200
        assert body["summary"]["issues"]["info"] == 2
        assert body["summary"]["issues"]["total"] == 2

    # --- Thread pool path ---

    def test_thread_pool_timeout_value(self):
        """Thread pool future.result() is called with timeout=30.0."""
        result_obj = _make_result()
        mock_validate = AsyncMock(return_value=result_obj)

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
            patch(f"{_MOD}.concurrent.futures.ThreadPoolExecutor") as mock_pool,
        ):
            mock_asyncio_mod.get_running_loop.return_value = MagicMock()
            mock_executor = MagicMock()
            mock_pool.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_pool.return_value.__exit__ = MagicMock(return_value=False)
            mock_future = MagicMock()
            mock_future.result.return_value = result_obj
            mock_executor.submit.return_value = mock_future
            deployment_diagnostics(MockHTTPHandler())

        mock_future.result.assert_called_once_with(timeout=30.0)


# ===========================================================================
# TestGenerateChecklist
# ===========================================================================


class TestGenerateChecklist:
    """Tests for _generate_checklist."""

    def test_empty_result(self):
        """Empty validation result generates all-not_checked checklist."""
        result = _make_result()
        checklist = _generate_checklist(result)

        assert checklist["security"]["jwt_secret"]["status"] == "not_checked"
        assert checklist["security"]["encryption_key"]["status"] == "not_checked"
        assert checklist["security"]["cors"]["status"] == "not_checked"
        assert checklist["security"]["tls"]["status"] == "not_checked"
        assert checklist["infrastructure"]["database"]["status"] == "not_checked"
        assert checklist["infrastructure"]["redis"]["status"] == "not_checked"
        assert checklist["infrastructure"]["storage"]["status"] == "not_checked"
        assert checklist["infrastructure"]["supabase"]["status"] == "not_checked"
        assert checklist["api"]["api_keys"]["status"] == "not_checked"
        assert checklist["api"]["rate_limiting"]["status"] == "not_checked"
        assert checklist["environment"]["env_mode"]["status"] == "not_checked"

    def test_all_healthy_components(self):
        """All healthy components produce 'pass' status."""
        components = [
            MockComponentHealth("jwt_secret", MockComponentStatus.HEALTHY),
            MockComponentHealth("encryption", MockComponentStatus.HEALTHY),
            MockComponentHealth("cors", MockComponentStatus.HEALTHY),
            MockComponentHealth("tls", MockComponentStatus.HEALTHY),
            MockComponentHealth("database", MockComponentStatus.HEALTHY),
            MockComponentHealth("redis", MockComponentStatus.HEALTHY),
            MockComponentHealth("storage", MockComponentStatus.HEALTHY),
            MockComponentHealth("supabase", MockComponentStatus.HEALTHY),
            MockComponentHealth("api_keys", MockComponentStatus.HEALTHY),
            MockComponentHealth("rate_limiting", MockComponentStatus.HEALTHY),
            MockComponentHealth("environment", MockComponentStatus.HEALTHY),
        ]
        result = _make_result(components=components)
        checklist = _generate_checklist(result)

        assert checklist["security"]["jwt_secret"]["status"] == "pass"
        assert checklist["security"]["encryption_key"]["status"] == "pass"
        assert checklist["security"]["cors"]["status"] == "pass"
        assert checklist["security"]["tls"]["status"] == "pass"
        assert checklist["infrastructure"]["database"]["status"] == "pass"
        assert checklist["infrastructure"]["redis"]["status"] == "pass"
        assert checklist["infrastructure"]["storage"]["status"] == "pass"
        assert checklist["infrastructure"]["supabase"]["status"] == "pass"
        assert checklist["api"]["api_keys"]["status"] == "pass"
        assert checklist["api"]["rate_limiting"]["status"] == "pass"
        assert checklist["environment"]["env_mode"]["status"] == "pass"

    def test_degraded_component_shows_warning(self):
        """Degraded component results in 'warning' status."""
        components = [
            MockComponentHealth("database", MockComponentStatus.DEGRADED),
        ]
        result = _make_result(components=components)
        checklist = _generate_checklist(result)

        assert checklist["infrastructure"]["database"]["status"] == "warning"

    def test_unhealthy_component_shows_fail(self):
        """Unhealthy component results in 'fail' status."""
        components = [
            MockComponentHealth("redis", MockComponentStatus.UNHEALTHY),
        ]
        result = _make_result(components=components)
        checklist = _generate_checklist(result)

        assert checklist["infrastructure"]["redis"]["status"] == "fail"

    def test_unknown_component_status(self):
        """Unknown component status results in 'unknown' status."""
        components = [
            MockComponentHealth("database", MockComponentStatus.UNKNOWN),
        ]
        result = _make_result(components=components)
        checklist = _generate_checklist(result)

        assert checklist["infrastructure"]["database"]["status"] == "unknown"

    def test_critical_issue_flagged(self):
        """Critical issue sets critical=True on the checklist item."""
        issues = [
            MockValidationIssue("jwt_secret", "Weak JWT secret", MockSeverity.CRITICAL),
        ]
        components = [
            MockComponentHealth("jwt_secret", MockComponentStatus.UNHEALTHY),
        ]
        result = _make_result(issues=issues, components=components)
        checklist = _generate_checklist(result)

        assert checklist["security"]["jwt_secret"]["critical"] is True

    def test_no_critical_issue(self):
        """Without critical issue, critical=False on the checklist item."""
        issues = [
            MockValidationIssue("jwt_secret", "Minor issue", MockSeverity.WARNING),
        ]
        components = [
            MockComponentHealth("jwt_secret", MockComponentStatus.DEGRADED),
        ]
        result = _make_result(issues=issues, components=components)
        checklist = _generate_checklist(result)

        assert checklist["security"]["jwt_secret"]["critical"] is False

    def test_checklist_has_descriptions(self):
        """Each checklist item has a description field."""
        result = _make_result()
        checklist = _generate_checklist(result)

        for category in checklist.values():
            for item_name, item in category.items():
                assert "description" in item, f"Missing description for {item_name}"
                assert isinstance(item["description"], str)
                assert len(item["description"]) > 0

    def test_checklist_structure_complete(self):
        """Checklist has all expected categories and items."""
        result = _make_result()
        checklist = _generate_checklist(result)

        # Security
        assert "jwt_secret" in checklist["security"]
        assert "encryption_key" in checklist["security"]
        assert "cors" in checklist["security"]
        assert "tls" in checklist["security"]

        # Infrastructure
        assert "database" in checklist["infrastructure"]
        assert "redis" in checklist["infrastructure"]
        assert "storage" in checklist["infrastructure"]
        assert "supabase" in checklist["infrastructure"]

        # API
        assert "api_keys" in checklist["api"]
        assert "rate_limiting" in checklist["api"]

        # Environment
        assert "env_mode" in checklist["environment"]

    def test_checklist_items_have_status_critical_description(self):
        """Each checklist item has status, critical, and description keys."""
        result = _make_result()
        checklist = _generate_checklist(result)

        for category_name, category in checklist.items():
            for item_name, item in category.items():
                assert "status" in item, f"{category_name}.{item_name} missing status"
                assert "critical" in item, f"{category_name}.{item_name} missing critical"
                assert "description" in item, f"{category_name}.{item_name} missing description"

    def test_multiple_issues_same_component(self):
        """Multiple issues on same component, at least one critical."""
        issues = [
            MockValidationIssue("database", "Slow", MockSeverity.WARNING),
            MockValidationIssue("database", "Unreachable", MockSeverity.CRITICAL),
        ]
        components = [
            MockComponentHealth("database", MockComponentStatus.UNHEALTHY),
        ]
        result = _make_result(issues=issues, components=components)
        checklist = _generate_checklist(result)

        assert checklist["infrastructure"]["database"]["critical"] is True
        assert checklist["infrastructure"]["database"]["status"] == "fail"

    def test_issues_for_nonexistent_component_ignored(self):
        """Issues for components not in the checklist do not cause errors."""
        issues = [
            MockValidationIssue("unknown_component", "Some issue", MockSeverity.WARNING),
        ]
        result = _make_result(issues=issues)
        checklist = _generate_checklist(result)

        # Should not crash and still return valid structure
        assert "security" in checklist

    def test_encryption_key_maps_to_encryption_component(self):
        """The encryption_key checklist item maps to the 'encryption' component name."""
        components = [
            MockComponentHealth("encryption", MockComponentStatus.DEGRADED),
        ]
        result = _make_result(components=components)
        checklist = _generate_checklist(result)

        assert checklist["security"]["encryption_key"]["status"] == "warning"

    def test_env_mode_maps_to_environment_component(self):
        """The env_mode checklist item maps to the 'environment' component name."""
        components = [
            MockComponentHealth("environment", MockComponentStatus.HEALTHY),
        ]
        result = _make_result(components=components)
        checklist = _generate_checklist(result)

        assert checklist["environment"]["env_mode"]["status"] == "pass"

    def test_mixed_components_and_issues(self):
        """Comprehensive test with mix of statuses and issues."""
        components = [
            MockComponentHealth("jwt_secret", MockComponentStatus.HEALTHY),
            MockComponentHealth("encryption", MockComponentStatus.HEALTHY),
            MockComponentHealth("cors", MockComponentStatus.DEGRADED),
            MockComponentHealth("tls", MockComponentStatus.UNHEALTHY),
            MockComponentHealth("database", MockComponentStatus.HEALTHY),
            MockComponentHealth("redis", MockComponentStatus.UNHEALTHY),
            MockComponentHealth("storage", MockComponentStatus.HEALTHY),
            MockComponentHealth("api_keys", MockComponentStatus.DEGRADED),
            MockComponentHealth("rate_limiting", MockComponentStatus.HEALTHY),
            MockComponentHealth("environment", MockComponentStatus.HEALTHY),
        ]
        issues = [
            MockValidationIssue("tls", "Not configured", MockSeverity.CRITICAL),
            MockValidationIssue("redis", "Connection refused", MockSeverity.WARNING),
            MockValidationIssue("cors", "Wildcard origin", MockSeverity.WARNING),
            MockValidationIssue("api_keys", "Only 1 provider", MockSeverity.INFO),
        ]
        result = _make_result(components=components, issues=issues)
        checklist = _generate_checklist(result)

        assert checklist["security"]["jwt_secret"]["status"] == "pass"
        assert checklist["security"]["jwt_secret"]["critical"] is False
        assert checklist["security"]["tls"]["status"] == "fail"
        assert checklist["security"]["tls"]["critical"] is True
        assert checklist["security"]["cors"]["status"] == "warning"
        assert checklist["security"]["cors"]["critical"] is False
        assert checklist["infrastructure"]["redis"]["status"] == "fail"
        assert checklist["infrastructure"]["redis"]["critical"] is False
        assert checklist["api"]["api_keys"]["status"] == "warning"
        assert checklist["api"]["api_keys"]["critical"] is False
        # supabase not in components -> not_checked
        assert checklist["infrastructure"]["supabase"]["status"] == "not_checked"

    def test_no_issues_all_not_critical(self):
        """With no issues at all, all critical flags are False."""
        components = [
            MockComponentHealth("jwt_secret", MockComponentStatus.HEALTHY),
            MockComponentHealth("database", MockComponentStatus.HEALTHY),
        ]
        result = _make_result(components=components)
        checklist = _generate_checklist(result)

        for category in checklist.values():
            for item in category.values():
                assert item["critical"] is False


# ===========================================================================
# TestDiagnosticsViaHealthHandler
# ===========================================================================


class TestDiagnosticsViaHealthHandler:
    """Test diagnostics endpoint through HealthHandler routing."""

    @pytest.mark.asyncio
    async def test_diagnostics_route(self, handler, mock_http):
        """HealthHandler routes /api/diagnostics to deployment_diagnostics."""
        result_obj = _make_result()
        mock_validate = AsyncMock(return_value=result_obj)

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.return_value = result_obj
            result = await handler.handle("/api/diagnostics", {}, mock_http)

        assert _status(result) == 200
        body = _body(result)
        assert "summary" in body

    @pytest.mark.asyncio
    async def test_diagnostics_deployment_route(self, handler, mock_http):
        """HealthHandler routes /api/diagnostics/deployment."""
        result_obj = _make_result()
        mock_validate = AsyncMock(return_value=result_obj)

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.return_value = result_obj
            result = await handler.handle("/api/diagnostics/deployment", {}, mock_http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_diagnostics_route(self, handler, mock_http):
        """HealthHandler routes /api/v1/diagnostics."""
        result_obj = _make_result()
        mock_validate = AsyncMock(return_value=result_obj)

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.return_value = result_obj
            result = await handler.handle("/api/v1/diagnostics", {}, mock_http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_diagnostics_deployment_route(self, handler, mock_http):
        """HealthHandler routes /api/v1/diagnostics/deployment."""
        result_obj = _make_result()
        mock_validate = AsyncMock(return_value=result_obj)

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.return_value = result_obj
            result = await handler.handle("/api/v1/diagnostics/deployment", {}, mock_http)

        assert _status(result) == 200

    def test_can_handle_diagnostics(self, handler):
        """HealthHandler.can_handle returns True for diagnostics routes."""
        assert handler.can_handle("/api/diagnostics") is True
        assert handler.can_handle("/api/diagnostics/deployment") is True
        assert handler.can_handle("/api/v1/diagnostics") is True
        assert handler.can_handle("/api/v1/diagnostics/deployment") is True

    def test_can_handle_non_diagnostics(self, handler):
        """HealthHandler.can_handle returns False for non-existent routes."""
        assert handler.can_handle("/api/diagnostics/unknown") is False
        assert handler.can_handle("/api/v2/diagnostics") is False

    @pytest.mark.asyncio
    async def test_diagnostics_error_via_handler(self, handler, mock_http):
        """Error from diagnostics endpoint propagated through handler."""
        mock_validate = AsyncMock(return_value=_make_result())

        with (
            patch(f"{_MOD}._check_diagnostics_permission", return_value=None),
            patch(f"{_VALIDATOR_MOD}.validate_deployment", mock_validate),
            patch(f"{_MOD}.asyncio") as mock_asyncio_mod,
        ):
            mock_asyncio_mod.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio_mod.run.side_effect = RuntimeError("Broken")

            result = await handler.handle("/api/diagnostics", {}, mock_http)

        assert _status(result) == 500


# ===========================================================================
# TestDeploymentDiagnosticsEdgeCases
# ===========================================================================


class TestDeploymentDiagnosticsEdgeCases:
    """Edge cases and additional coverage for deployment_diagnostics."""

    def test_ready_true_with_zero_warnings_returns_200(self):
        """Ready deployment with zero warnings returns 200."""
        result_obj = _make_result(ready=True)
        result = _run_diagnostics(result_obj)

        assert _status(result) == 200

    def test_ready_true_with_warnings_returns_200(self):
        """Ready deployment with warnings still returns 200 (not 503)."""
        issues = [
            MockValidationIssue("cors", "Wildcard CORS", MockSeverity.WARNING),
            MockValidationIssue("redis", "Not configured", MockSeverity.WARNING),
        ]
        result_obj = _make_result(ready=True, issues=issues)
        result = _run_diagnostics(result_obj)

        assert _status(result) == 200

    def test_response_preserves_to_dict_fields(self):
        """Response includes all fields from result.to_dict()."""
        components = [
            MockComponentHealth(
                "jwt_secret",
                MockComponentStatus.HEALTHY,
                latency_ms=1.5,
                message="OK",
            ),
        ]
        result_obj = _make_result(
            ready=True,
            live=True,
            components=components,
        )
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert "components" in body
        assert len(body["components"]) == 1
        assert body["components"][0]["name"] == "jwt_secret"
        assert body["components"][0]["status"] == "healthy"

    def test_large_number_of_issues(self):
        """Handler works with many issues."""
        issues = []
        for i in range(50):
            severity = [MockSeverity.CRITICAL, MockSeverity.WARNING, MockSeverity.INFO][i % 3]
            issues.append(MockValidationIssue(f"comp_{i}", f"Issue {i}", severity))
        result_obj = _make_result(ready=False, issues=issues)
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert body["summary"]["issues"]["total"] == 50
        # 0, 3, 6, ... are critical: ceil(50/3) = 17
        assert body["summary"]["issues"]["critical"] == 17

    def test_large_number_of_components(self):
        """Handler works with many components."""
        components = []
        for i in range(20):
            status = [
                MockComponentStatus.HEALTHY,
                MockComponentStatus.DEGRADED,
                MockComponentStatus.UNHEALTHY,
            ][i % 3]
            components.append(MockComponentHealth(f"comp_{i}", status))
        result_obj = _make_result(components=components)
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert body["summary"]["components"]["total"] == 20
        # 0,3,6,9,12,15,18 -> 7 healthy
        assert body["summary"]["components"]["healthy"] == 7

    def test_response_time_is_positive(self):
        """response_time_ms is a non-negative number."""
        result_obj = _make_result()
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert body["response_time_ms"] >= 0

    def test_not_live_and_not_ready(self):
        """Both live=False and ready=False."""
        result_obj = _make_result(ready=False, live=False)
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert body["ready"] is False
        assert body["live"] is False
        assert _status(result) == 503

    def test_live_but_not_ready(self):
        """live=True but ready=False returns 503."""
        result_obj = _make_result(ready=False, live=True)
        result = _run_diagnostics(result_obj)

        assert _status(result) == 503

    def test_ready_but_not_live(self):
        """ready=True but live=False returns 200 (ready drives status code)."""
        result_obj = _make_result(ready=True, live=False)
        result = _run_diagnostics(result_obj)

        assert _status(result) == 200
        body = _body(result)
        assert body["live"] is False

    def test_validated_at_preserved(self):
        """validated_at from the result is preserved in response."""
        result_obj = _make_result()
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert "validated_at" in body

    def test_validation_duration_preserved(self):
        """validation_duration_ms from the result is preserved in response."""
        result_obj = MockValidationResult(ready=True, live=True, validation_duration_ms=42.5)
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert body["validation_duration_ms"] == 42.5

    def test_issues_list_preserved(self):
        """Issues list from result.to_dict() is preserved in response."""
        issues = [
            MockValidationIssue("cors", "Wildcard", MockSeverity.WARNING, suggestion="Restrict"),
        ]
        result_obj = _make_result(issues=issues)
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert len(body["issues"]) == 1
        assert body["issues"][0]["component"] == "cors"
        assert body["issues"][0]["severity"] == "warning"
        assert body["issues"][0]["suggestion"] == "Restrict"

    def test_component_metadata_preserved(self):
        """Component metadata from to_dict() is preserved."""
        components = [
            MockComponentHealth(
                "database",
                MockComponentStatus.HEALTHY,
                latency_ms=5.2,
                message="Connected",
                metadata={"version": "15.0"},
            ),
        ]
        result_obj = _make_result(components=components)
        result = _run_diagnostics(result_obj)

        body = _body(result)
        assert body["components"][0]["metadata"] == {"version": "15.0"}
        assert body["components"][0]["latency_ms"] == 5.2
        assert body["components"][0]["message"] == "Connected"


# ===========================================================================
# TestGenerateChecklistEdgeCases
# ===========================================================================


class TestGenerateChecklistEdgeCases:
    """Additional edge cases for _generate_checklist."""

    def test_duplicate_component_names_last_wins(self):
        """If duplicate component names exist, last one in list is used for lookup."""
        components = [
            MockComponentHealth("database", MockComponentStatus.HEALTHY),
            MockComponentHealth("database", MockComponentStatus.UNHEALTHY),
        ]
        result = _make_result(components=components)
        checklist = _generate_checklist(result)

        # dict comprehension: last entry wins
        assert checklist["infrastructure"]["database"]["status"] == "fail"

    def test_issue_component_not_in_checklist_items(self):
        """Issues for component names not mapped in checklist are harmless."""
        issues = [
            MockValidationIssue("nonexistent", "Some problem", MockSeverity.CRITICAL),
        ]
        result = _make_result(issues=issues)
        checklist = _generate_checklist(result)

        # No crash, normal structure
        assert "security" in checklist

    def test_security_descriptions(self):
        """Security items have meaningful descriptions."""
        result = _make_result()
        checklist = _generate_checklist(result)

        assert "JWT" in checklist["security"]["jwt_secret"]["description"]
        assert "ncryption" in checklist["security"]["encryption_key"]["description"]
        assert "CORS" in checklist["security"]["cors"]["description"]
        assert "TLS" in checklist["security"]["tls"]["description"]

    def test_infrastructure_descriptions(self):
        """Infrastructure items have meaningful descriptions."""
        result = _make_result()
        checklist = _generate_checklist(result)

        assert "atabase" in checklist["infrastructure"]["database"]["description"]
        assert "Redis" in checklist["infrastructure"]["redis"]["description"]
        assert "ata" in checklist["infrastructure"]["storage"]["description"]
        assert "Supabase" in checklist["infrastructure"]["supabase"]["description"]

    def test_api_descriptions(self):
        """API items have meaningful descriptions."""
        result = _make_result()
        checklist = _generate_checklist(result)

        assert (
            "provider" in checklist["api"]["api_keys"]["description"].lower()
            or "AI" in checklist["api"]["api_keys"]["description"]
        )
        assert "ate" in checklist["api"]["rate_limiting"]["description"]

    def test_environment_descriptions(self):
        """Environment items have meaningful descriptions."""
        result = _make_result()
        checklist = _generate_checklist(result)

        assert "nvironment" in checklist["environment"]["env_mode"]["description"]

    def test_all_component_statuses_mapped(self):
        """All four ComponentStatus values produce expected checklist statuses."""
        status_map = {
            MockComponentStatus.HEALTHY: "pass",
            MockComponentStatus.DEGRADED: "warning",
            MockComponentStatus.UNHEALTHY: "fail",
            MockComponentStatus.UNKNOWN: "unknown",
        }
        for comp_status, expected_checklist_status in status_map.items():
            components = [MockComponentHealth("database", comp_status)]
            result = _make_result(components=components)
            checklist = _generate_checklist(result)
            assert checklist["infrastructure"]["database"]["status"] == expected_checklist_status, (
                f"Expected {expected_checklist_status} for {comp_status}"
            )

    def test_critical_false_for_warning_issues(self):
        """Warning-severity issues produce critical=False."""
        issues = [
            MockValidationIssue("database", "Slow queries", MockSeverity.WARNING),
        ]
        result = _make_result(issues=issues)
        checklist = _generate_checklist(result)

        assert checklist["infrastructure"]["database"]["critical"] is False

    def test_critical_false_for_info_issues(self):
        """Info-severity issues produce critical=False."""
        issues = [
            MockValidationIssue("redis", "Using defaults", MockSeverity.INFO),
        ]
        result = _make_result(issues=issues)
        checklist = _generate_checklist(result)

        assert checklist["infrastructure"]["redis"]["critical"] is False

    def test_checklist_four_top_level_categories(self):
        """Checklist has exactly 4 top-level categories."""
        result = _make_result()
        checklist = _generate_checklist(result)

        assert set(checklist.keys()) == {"security", "infrastructure", "api", "environment"}

    def test_security_has_four_items(self):
        """Security category has exactly 4 items."""
        result = _make_result()
        checklist = _generate_checklist(result)

        assert len(checklist["security"]) == 4

    def test_infrastructure_has_four_items(self):
        """Infrastructure category has exactly 4 items."""
        result = _make_result()
        checklist = _generate_checklist(result)

        assert len(checklist["infrastructure"]) == 4

    def test_api_has_two_items(self):
        """API category has exactly 2 items."""
        result = _make_result()
        checklist = _generate_checklist(result)

        assert len(checklist["api"]) == 2

    def test_environment_has_one_item(self):
        """Environment category has exactly 1 item."""
        result = _make_result()
        checklist = _generate_checklist(result)

        assert len(checklist["environment"]) == 1

    def test_only_critical_issues_set_critical_flag(self):
        """Only critical severity sets critical=True; warning and info do not."""
        issues = [
            MockValidationIssue("jwt_secret", "Critical!", MockSeverity.CRITICAL),
            MockValidationIssue("cors", "Warn!", MockSeverity.WARNING),
            MockValidationIssue("tls", "Info!", MockSeverity.INFO),
        ]
        result = _make_result(issues=issues)
        checklist = _generate_checklist(result)

        assert checklist["security"]["jwt_secret"]["critical"] is True
        assert checklist["security"]["cors"]["critical"] is False
        assert checklist["security"]["tls"]["critical"] is False
