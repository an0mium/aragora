"""
Tests for OpenClaw Gateway Handler.

Tests the main HTTP handler routing and circuit breaker functionality.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.openclaw.gateway import (
    OpenClawGatewayHandler,
    get_openclaw_circuit_breaker,
    get_openclaw_circuit_breaker_status,
    get_openclaw_gateway_handler,
)


class TestOpenClawGatewayHandler:
    """Tests for OpenClawGatewayHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.server_context = {"config": {}, "stores": {}}
        self.handler = OpenClawGatewayHandler(self.server_context)

    def test_can_handle_gateway_path(self):
        """Test can_handle returns True for gateway paths."""
        assert self.handler.can_handle("/api/gateway/openclaw/sessions") is True
        assert self.handler.can_handle("/api/gateway/openclaw/actions") is True
        assert self.handler.can_handle("/api/gateway/openclaw/credentials") is True
        assert self.handler.can_handle("/api/gateway/openclaw/health") is True

    def test_can_handle_v1_path(self):
        """Test can_handle returns True for v1 paths."""
        assert self.handler.can_handle("/api/v1/gateway/openclaw/sessions") is True
        assert self.handler.can_handle("/api/v1/openclaw/sessions") is True

    def test_can_handle_non_gateway_path(self):
        """Test can_handle returns False for non-gateway paths."""
        assert self.handler.can_handle("/api/v1/debates") is False
        assert self.handler.can_handle("/api/health") is False
        assert self.handler.can_handle("/api/gateway/other") is False

    def test_normalize_path_v1_gateway(self):
        """Test path normalization for v1/gateway paths."""
        path = self.handler._normalize_path("/api/v1/gateway/openclaw/sessions")
        assert path == "/api/gateway/openclaw/sessions"

    def test_normalize_path_v1_openclaw(self):
        """Test path normalization for v1/openclaw paths."""
        path = self.handler._normalize_path("/api/v1/openclaw/sessions")
        assert path == "/api/gateway/openclaw/sessions"

    def test_normalize_path_no_change(self):
        """Test path normalization with standard path."""
        path = self.handler._normalize_path("/api/gateway/openclaw/sessions")
        assert path == "/api/gateway/openclaw/sessions"

    def test_get_user_id_with_user(self):
        """Test _get_user_id returns user ID when user exists."""
        mock_handler = MagicMock()
        mock_user = MagicMock()
        mock_user.user_id = "user_123"

        with patch.object(self.handler, "get_current_user", return_value=mock_user):
            user_id = self.handler._get_user_id(mock_handler)
            assert user_id == "user_123"

    def test_get_user_id_anonymous(self):
        """Test _get_user_id returns anonymous when no user."""
        mock_handler = MagicMock()

        with patch.object(self.handler, "get_current_user", return_value=None):
            user_id = self.handler._get_user_id(mock_handler)
            assert user_id == "anonymous"

    def test_get_tenant_id_with_org(self):
        """Test _get_tenant_id returns org_id when available."""
        mock_handler = MagicMock()
        mock_user = MagicMock()
        mock_user.org_id = "org_456"

        with patch.object(self.handler, "get_current_user", return_value=mock_user):
            tenant_id = self.handler._get_tenant_id(mock_handler)
            assert tenant_id == "org_456"

    def test_get_tenant_id_none(self):
        """Test _get_tenant_id returns None when no org."""
        mock_handler = MagicMock()

        with patch.object(self.handler, "get_current_user", return_value=None):
            tenant_id = self.handler._get_tenant_id(mock_handler)
            assert tenant_id is None


class TestCircuitBreaker:
    """Tests for circuit breaker functions."""

    def test_get_circuit_breaker(self):
        """Test getting the circuit breaker instance."""
        cb = get_openclaw_circuit_breaker()
        assert cb is not None
        assert cb.name == "openclaw_gateway_handler"

    def test_get_circuit_breaker_status(self):
        """Test getting circuit breaker status."""
        status = get_openclaw_circuit_breaker_status()
        assert isinstance(status, dict)
        # New circuit breaker status structure with nested keys
        assert "config" in status or "single_mode" in status or "entity_mode" in status

    def test_circuit_breaker_singleton(self):
        """Test that circuit breaker is a singleton."""
        cb1 = get_openclaw_circuit_breaker()
        cb2 = get_openclaw_circuit_breaker()
        assert cb1 is cb2


class TestHandlerFactory:
    """Tests for handler factory function."""

    def test_get_handler(self):
        """Test factory function creates handler."""
        context = {"config": {}}
        handler = get_openclaw_gateway_handler(context)
        assert isinstance(handler, OpenClawGatewayHandler)

    def test_handler_has_mixins(self):
        """Test handler inherits from mixins."""
        context = {"config": {}}
        handler = get_openclaw_gateway_handler(context)

        # Check mixin methods are available
        assert hasattr(handler, "_handle_list_sessions")
        assert hasattr(handler, "_handle_create_session")
        assert hasattr(handler, "_handle_store_credential")
        assert hasattr(handler, "_handle_get_policy_rules")


class TestHandlerRouting:
    """Tests for handler HTTP method routing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.server_context = {"config": {}, "stores": {}}
        self.handler = OpenClawGatewayHandler(self.server_context)

    @patch("aragora.server.handlers.openclaw.gateway.OpenClawGatewayHandler._handle_list_sessions")
    def test_get_sessions_routing(self, mock_list):
        """Test GET /sessions routes to _handle_list_sessions."""
        mock_list.return_value = MagicMock(status_code=200)
        mock_http_handler = MagicMock()

        result = self.handler.handle("/api/gateway/openclaw/sessions", {}, mock_http_handler)

        mock_list.assert_called_once()

    @patch("aragora.server.handlers.openclaw.gateway.OpenClawGatewayHandler._handle_get_session")
    def test_get_session_by_id_routing(self, mock_get):
        """Test GET /sessions/:id routes to _handle_get_session."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_http_handler = MagicMock()

        result = self.handler.handle(
            "/api/gateway/openclaw/sessions/session_123", {}, mock_http_handler
        )

        mock_get.assert_called_once_with("session_123", mock_http_handler)

    @patch("aragora.server.handlers.openclaw.gateway.OpenClawGatewayHandler._handle_health")
    def test_health_routing(self, mock_health):
        """Test GET /health routes to _handle_health."""
        mock_health.return_value = MagicMock(status_code=200)
        mock_http_handler = MagicMock()

        result = self.handler.handle("/api/gateway/openclaw/health", {}, mock_http_handler)

        mock_health.assert_called_once()

    @patch("aragora.server.handlers.openclaw.gateway.OpenClawGatewayHandler._handle_metrics")
    def test_metrics_routing(self, mock_metrics):
        """Test GET /metrics routes to _handle_metrics."""
        mock_metrics.return_value = MagicMock(status_code=200)
        mock_http_handler = MagicMock()

        result = self.handler.handle("/api/gateway/openclaw/metrics", {}, mock_http_handler)

        mock_metrics.assert_called_once()

    def test_unknown_path_returns_none(self):
        """Test unknown paths return None."""
        mock_http_handler = MagicMock()

        result = self.handler.handle("/api/gateway/openclaw/unknown", {}, mock_http_handler)

        assert result is None


class TestPostRouting:
    """Tests for POST request routing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.server_context = {"config": {}, "stores": {}}
        self.handler = OpenClawGatewayHandler(self.server_context)

    @patch.object(OpenClawGatewayHandler, "read_json_body_validated")
    @patch("aragora.server.handlers.openclaw.gateway.OpenClawGatewayHandler._handle_create_session")
    def test_post_sessions_routing(self, mock_create, mock_body):
        """Test POST /sessions routes to _handle_create_session."""
        mock_body.return_value = ({"config": {}}, None)
        mock_create.return_value = MagicMock(status_code=201)
        mock_http_handler = MagicMock()

        result = self.handler.handle_post("/api/gateway/openclaw/sessions", {}, mock_http_handler)

        mock_create.assert_called_once()

    @patch.object(OpenClawGatewayHandler, "read_json_body_validated")
    @patch("aragora.server.handlers.openclaw.gateway.OpenClawGatewayHandler._handle_execute_action")
    def test_post_actions_routing(self, mock_execute, mock_body):
        """Test POST /actions routes to _handle_execute_action."""
        mock_body.return_value = ({"session_id": "s1", "action_type": "click"}, None)
        mock_execute.return_value = MagicMock(status_code=202)
        mock_http_handler = MagicMock()

        result = self.handler.handle_post("/api/gateway/openclaw/actions", {}, mock_http_handler)

        mock_execute.assert_called_once()

    @patch("aragora.server.handlers.openclaw.gateway.OpenClawGatewayHandler._handle_cancel_action")
    def test_post_cancel_routing(self, mock_cancel):
        """Test POST /actions/:id/cancel routes to _handle_cancel_action."""
        mock_cancel.return_value = MagicMock(status_code=200)
        mock_http_handler = MagicMock()

        result = self.handler.handle_post(
            "/api/gateway/openclaw/actions/action_123/cancel", {}, mock_http_handler
        )

        mock_cancel.assert_called_once_with("action_123", mock_http_handler)


class TestDeleteRouting:
    """Tests for DELETE request routing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.server_context = {"config": {}, "stores": {}}
        self.handler = OpenClawGatewayHandler(self.server_context)

    @patch("aragora.server.handlers.openclaw.gateway.OpenClawGatewayHandler._handle_close_session")
    def test_delete_session_routing(self, mock_close):
        """Test DELETE /sessions/:id routes to _handle_close_session."""
        mock_close.return_value = MagicMock(status_code=200)
        mock_http_handler = MagicMock()

        result = self.handler.handle_delete(
            "/api/gateway/openclaw/sessions/session_123", {}, mock_http_handler
        )

        mock_close.assert_called_once_with("session_123", mock_http_handler)

    @patch(
        "aragora.server.handlers.openclaw.gateway.OpenClawGatewayHandler._handle_delete_credential"
    )
    def test_delete_credential_routing(self, mock_delete):
        """Test DELETE /credentials/:id routes to _handle_delete_credential."""
        mock_delete.return_value = MagicMock(status_code=200)
        mock_http_handler = MagicMock()

        result = self.handler.handle_delete(
            "/api/gateway/openclaw/credentials/cred_123", {}, mock_http_handler
        )

        mock_delete.assert_called_once_with("cred_123", mock_http_handler)

    def test_delete_unknown_returns_none(self):
        """Test DELETE unknown path returns None."""
        mock_http_handler = MagicMock()

        result = self.handler.handle_delete(
            "/api/gateway/openclaw/unknown/123", {}, mock_http_handler
        )

        assert result is None
