"""
Tests for OpenClaw security audit findings F06, F07, F08.

F06: RBAC gap on OpenClaw routes - verify @require_permission on all HTTP methods
F07: NavigateAction SSRF risk - verify URL validation blocks internal IPs
F08: Health endpoint info leak - verify minimal response for unauthenticated requests
"""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# =============================================================================
# F06: RBAC gap on OpenClaw routes
# =============================================================================


class TestOpenClawRBACDecorators:
    """Verify all OpenClaw gateway HTTP methods have @require_permission."""

    def test_get_handler_has_require_permission(self):
        """GET handler should have openclaw:read permission decorator."""
        from aragora.server.handlers.openclaw.gateway import OpenClawGatewayHandler

        handle_method = OpenClawGatewayHandler.handle
        # The @require_permission decorator wraps the function;
        # check that the wrapper attributes exist
        assert hasattr(handle_method, "__wrapped__") or hasattr(
            handle_method, "_permission"
        ), "GET handle() missing @require_permission decorator"

    def test_post_handler_has_require_permission(self):
        """POST handler should have openclaw:write permission decorator."""
        from aragora.server.handlers.openclaw.gateway import OpenClawGatewayHandler

        handle_post = OpenClawGatewayHandler.handle_post
        assert hasattr(handle_post, "__wrapped__") or hasattr(
            handle_post, "_permission"
        ), "POST handle_post() missing @require_permission decorator"

    def test_delete_handler_has_require_permission(self):
        """DELETE handler should have openclaw:delete permission decorator."""
        from aragora.server.handlers.openclaw.gateway import OpenClawGatewayHandler

        handle_delete = OpenClawGatewayHandler.handle_delete
        assert hasattr(handle_delete, "__wrapped__") or hasattr(
            handle_delete, "_permission"
        ), "DELETE handle_delete() missing @require_permission decorator"

    def test_post_handler_uses_openclaw_permission_not_debates(self):
        """POST handler should use openclaw:write, not debates:write."""
        from aragora.server.handlers.openclaw import gateway
        import inspect

        source = inspect.getsource(gateway.OpenClawGatewayHandler.handle_post)
        # The permission should not reference 'debates'
        assert "debates:write" not in source, (
            "POST handler still uses debates:write instead of openclaw:write"
        )

    def test_delete_handler_uses_openclaw_permission_not_debates(self):
        """DELETE handler should use openclaw:delete, not debates:delete."""
        from aragora.server.handlers.openclaw import gateway
        import inspect

        source = inspect.getsource(gateway.OpenClawGatewayHandler.handle_delete)
        assert "debates:delete" not in source, (
            "DELETE handler still uses debates:delete instead of openclaw:delete"
        )


# =============================================================================
# F07: NavigateAction SSRF risk
# =============================================================================


class TestNavigateActionSSRF:
    """Verify NavigateAction validates URLs against SSRF attacks.

    Note: The global test conftest sets ARAGORA_SSRF_ALLOW_LOCALHOST=true
    for integration tests. We must unset it for these security tests to
    verify localhost blocking works in production environments.
    """

    @pytest.fixture(autouse=True)
    def _disable_ssrf_localhost_override(self, monkeypatch):
        """Ensure SSRF localhost protection is active for these tests.

        The global test_environment fixture sets ARAGORA_SSRF_ALLOW_LOCALHOST=true
        (session-scoped), so we must explicitly set it to "false" to override.
        Using monkeypatch.setenv to override (not delenv) ensures this takes
        precedence regardless of session-scoped fixture state.
        """
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "false")

    def test_valid_https_url_allowed(self):
        """A valid HTTPS URL should be accepted."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction

        nav = NavigateAction(url="https://example.com")
        assert nav.url == "https://example.com"

    def test_valid_http_url_allowed(self):
        """A valid HTTP URL should be accepted."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction

        nav = NavigateAction(url="http://example.com")
        assert nav.url == "http://example.com"

    def test_empty_url_allowed(self):
        """An empty URL should be accepted (default value)."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction

        nav = NavigateAction(url="")
        assert nav.url == ""

    def test_localhost_blocked(self):
        """URLs targeting localhost should be blocked."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction
        from aragora.security.ssrf_protection import SSRFValidationError

        with pytest.raises(SSRFValidationError):
            NavigateAction(url="http://localhost/admin")

    def test_127_0_0_1_blocked(self):
        """URLs targeting 127.0.0.1 should be blocked."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction
        from aragora.security.ssrf_protection import SSRFValidationError

        with pytest.raises(SSRFValidationError):
            NavigateAction(url="http://127.0.0.1:8080/internal")

    def test_private_10_network_blocked(self):
        """URLs targeting 10.x.x.x should be blocked."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction
        from aragora.security.ssrf_protection import SSRFValidationError

        with pytest.raises(SSRFValidationError):
            NavigateAction(url="http://10.0.0.1/secret")

    def test_private_172_16_network_blocked(self):
        """URLs targeting 172.16.x.x should be blocked."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction
        from aragora.security.ssrf_protection import SSRFValidationError

        with pytest.raises(SSRFValidationError):
            NavigateAction(url="http://172.16.0.1/internal")

    def test_private_192_168_network_blocked(self):
        """URLs targeting 192.168.x.x should be blocked."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction
        from aragora.security.ssrf_protection import SSRFValidationError

        with pytest.raises(SSRFValidationError):
            NavigateAction(url="http://192.168.1.1/router")

    def test_cloud_metadata_blocked(self):
        """URLs targeting cloud metadata endpoint should be blocked."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction
        from aragora.security.ssrf_protection import SSRFValidationError

        with pytest.raises(SSRFValidationError):
            NavigateAction(url="http://169.254.169.254/latest/meta-data/")

    def test_file_protocol_blocked(self):
        """file:// protocol should be blocked."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction
        from aragora.security.ssrf_protection import SSRFValidationError

        with pytest.raises(SSRFValidationError):
            NavigateAction(url="file:///etc/passwd")

    def test_from_openclaw_navigate_validates_url(self):
        """ComputerUseBridge.from_openclaw should validate navigate URLs."""
        from aragora.compat.openclaw.computer_use_bridge import ComputerUseBridge
        from aragora.security.ssrf_protection import SSRFValidationError

        with pytest.raises(SSRFValidationError):
            ComputerUseBridge.from_openclaw(
                "navigate", {"url": "http://127.0.0.1:9090/admin"}
            )

    def test_from_openclaw_navigate_allows_valid_url(self):
        """ComputerUseBridge.from_openclaw should accept valid public URLs."""
        from aragora.compat.openclaw.computer_use_bridge import (
            ComputerUseBridge,
            NavigateAction,
        )

        action = ComputerUseBridge.from_openclaw(
            "navigate", {"url": "https://example.com/page"}
        )
        assert isinstance(action, NavigateAction)
        assert action.url == "https://example.com/page"

    def test_zero_address_blocked(self):
        """URLs targeting 0.0.0.0 should be blocked."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction
        from aragora.security.ssrf_protection import SSRFValidationError

        with pytest.raises(SSRFValidationError):
            NavigateAction(url="http://0.0.0.0:8080/")

    def test_gopher_protocol_blocked(self):
        """gopher:// protocol should be blocked."""
        from aragora.compat.openclaw.computer_use_bridge import NavigateAction
        from aragora.security.ssrf_protection import SSRFValidationError

        with pytest.raises(SSRFValidationError):
            NavigateAction(url="gopher://evil.com/")


# =============================================================================
# F08: Health endpoint info leak
# =============================================================================


class TestHealthEndpointInfoLeak:
    """Verify health endpoints do not leak internal info to unauthenticated users."""

    def test_openclaw_health_does_not_expose_session_counts(self):
        """OpenClaw health endpoint should not expose session/action counts."""
        from aragora.server.handlers.openclaw.policies import PolicyHandlerMixin

        mixin = PolicyHandlerMixin.__new__(PolicyHandlerMixin)

        # Mock the store
        mock_store = MagicMock()
        mock_store.get_metrics.return_value = {
            "sessions": {"active": 42, "total": 100},
            "actions": {"pending": 5, "running": 3, "completed": 200, "failed": 10},
        }

        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
            return_value=mock_store,
        ):
            result = mixin._handle_health(None)

        # Result should have status and timestamp but NOT session/action counts
        response_body = result.body if hasattr(result, "body") else result[0]
        import json

        if isinstance(response_body, str):
            data = json.loads(response_body)
        elif isinstance(response_body, dict):
            data = response_body
        else:
            data = json.loads(response_body.decode() if isinstance(response_body, bytes) else str(response_body))

        assert "status" in data
        assert "timestamp" in data
        assert "active_sessions" not in data, "Health endpoint leaks active_sessions count"
        assert "pending_actions" not in data, "Health endpoint leaks pending_actions count"
        assert "running_actions" not in data, "Health endpoint leaks running_actions count"

    def test_openclaw_health_error_does_not_expose_details(self):
        """OpenClaw health error response should not expose error details."""
        from aragora.server.handlers.openclaw.policies import PolicyHandlerMixin

        mixin = PolicyHandlerMixin.__new__(PolicyHandlerMixin)

        mock_store = MagicMock()
        mock_store.get_metrics.side_effect = RuntimeError("DB connection pool exhausted")

        with patch(
            "aragora.server.handlers.openclaw.policies._get_store",
            return_value=mock_store,
        ):
            result = mixin._handle_health(None)

        response_body = result.body if hasattr(result, "body") else result[0]
        import json

        if isinstance(response_body, str):
            data = json.loads(response_body)
        elif isinstance(response_body, dict):
            data = response_body
        else:
            data = json.loads(response_body.decode() if isinstance(response_body, bytes) else str(response_body))

        assert data["status"] == "error"
        assert data["healthy"] is False
        assert "error" not in data, "Health error response leaks error details"

    def test_main_health_handler_has_public_routes(self):
        """Verify /api/health is listed as a public route."""
        from aragora.server.handlers.admin.health import HealthHandler

        assert "/api/health" in HealthHandler.PUBLIC_ROUTES
        assert "/api/v1/health" in HealthHandler.PUBLIC_ROUTES

    def test_minimal_health_check_returns_only_status_and_timestamp(self):
        """Unauthenticated /api/health should return only status + timestamp."""
        from aragora.server.handlers.admin.health import HealthHandler

        handler = HealthHandler(ctx={})
        result = handler._minimal_health_check()

        response_body = result.body if hasattr(result, "body") else result[0]
        import json

        if isinstance(response_body, str):
            data = json.loads(response_body)
        elif isinstance(response_body, dict):
            data = response_body
        else:
            data = json.loads(response_body.decode() if isinstance(response_body, bytes) else str(response_body))

        # Should only contain status and timestamp
        assert "status" in data
        assert "timestamp" in data
        # Must NOT contain sensitive fields
        assert "version" not in data, "Minimal health leaks version"
        assert "checks" not in data, "Minimal health leaks dependency checks"
        assert "uptime_seconds" not in data, "Minimal health leaks uptime"
        assert "response_time_ms" not in data, "Minimal health leaks response time"
        assert "ai_providers" not in data, "Minimal health leaks AI provider info"

    def test_minimal_health_check_status_values(self):
        """Minimal health check should report 'healthy' or 'degraded'."""
        from aragora.server.handlers.admin.health import HealthHandler

        handler = HealthHandler(ctx={})
        result = handler._minimal_health_check()

        response_body = result.body if hasattr(result, "body") else result[0]
        import json

        if isinstance(response_body, str):
            data = json.loads(response_body)
        elif isinstance(response_body, dict):
            data = response_body
        else:
            data = json.loads(response_body.decode() if isinstance(response_body, bytes) else str(response_body))

        assert data["status"] in ("healthy", "degraded")
