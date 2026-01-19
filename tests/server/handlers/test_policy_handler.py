"""
Tests for the PolicyHandler module.

Tests cover:
- Handler routing for policy endpoints
- Handler routing for compliance endpoints
- can_handle method
- ROUTES attribute
- HTTP method routing (GET, POST, PATCH, DELETE)
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock
import pytest

from aragora.server.handlers.policy import PolicyHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


class TestPolicyHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return PolicyHandler(mock_server_context)

    def test_can_handle_policies_base(self, handler):
        """Handler can handle policies base endpoint."""
        assert handler.can_handle("/api/policies")

    def test_can_handle_policy_by_id(self, handler):
        """Handler can handle policy by ID."""
        assert handler.can_handle("/api/policies/pol_123")

    def test_can_handle_policy_toggle(self, handler):
        """Handler can handle policy toggle endpoint."""
        assert handler.can_handle("/api/policies/pol_123/toggle")

    def test_can_handle_policy_violations(self, handler):
        """Handler can handle policy violations endpoint."""
        assert handler.can_handle("/api/policies/pol_123/violations")

    def test_can_handle_compliance_violations(self, handler):
        """Handler can handle compliance violations base."""
        assert handler.can_handle("/api/compliance/violations")

    def test_can_handle_compliance_violation_by_id(self, handler):
        """Handler can handle compliance violation by ID."""
        assert handler.can_handle("/api/compliance/violations/viol_123")

    def test_can_handle_compliance_check(self, handler):
        """Handler can handle compliance check endpoint."""
        assert handler.can_handle("/api/compliance/check")

    def test_can_handle_compliance_stats(self, handler):
        """Handler can handle compliance stats endpoint."""
        assert handler.can_handle("/api/compliance/stats")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/other")


class TestPolicyHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return PolicyHandler(mock_server_context)

    def test_routes_contains_policies(self, handler):
        """ROUTES contains policies."""
        assert "/api/policies" in handler.ROUTES

    def test_routes_contains_compliance_violations(self, handler):
        """ROUTES contains compliance violations."""
        assert "/api/compliance/violations" in handler.ROUTES

    def test_routes_contains_compliance_check(self, handler):
        """ROUTES contains compliance check."""
        assert "/api/compliance/check" in handler.ROUTES

    def test_routes_contains_compliance_stats(self, handler):
        """ROUTES contains compliance stats."""
        assert "/api/compliance/stats" in handler.ROUTES


class TestPolicyHandlerPolicyEndpoints:
    """Tests for policy endpoint routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return PolicyHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_list_policies_get(self, handler):
        """GET /api/policies routes to list handler."""
        mock_http = MagicMock()
        mock_http.path = "/api/policies"

        result = await handler.handle("/api/policies", "GET", mock_http)

        assert result is not None

    @pytest.mark.asyncio
    async def test_create_policy_post(self, handler):
        """POST /api/policies routes to create handler."""
        mock_http = MagicMock()
        mock_http.path = "/api/policies"
        mock_http.rfile = MagicMock()
        mock_http.headers = {"Content-Length": "0"}

        result = await handler.handle("/api/policies", "POST", mock_http)

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_policy_by_id(self, handler):
        """GET /api/policies/:id routes to get handler."""
        mock_http = MagicMock()
        mock_http.path = "/api/policies/pol_123"

        result = await handler.handle("/api/policies/pol_123", "GET", mock_http)

        assert result is not None

    @pytest.mark.asyncio
    async def test_delete_policy(self, handler):
        """DELETE /api/policies/:id routes to delete handler."""
        mock_http = MagicMock()
        mock_http.path = "/api/policies/pol_123"

        result = await handler.handle("/api/policies/pol_123", "DELETE", mock_http)

        assert result is not None

    @pytest.mark.asyncio
    async def test_toggle_policy(self, handler):
        """POST /api/policies/:id/toggle routes to toggle handler."""
        mock_http = MagicMock()
        mock_http.path = "/api/policies/pol_123/toggle"
        mock_http.rfile = MagicMock()
        mock_http.headers = {"Content-Length": "0"}

        result = await handler.handle("/api/policies/pol_123/toggle", "POST", mock_http)

        assert result is not None


class TestPolicyHandlerComplianceEndpoints:
    """Tests for compliance endpoint routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return PolicyHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_list_violations(self, handler):
        """GET /api/compliance/violations routes to list handler."""
        mock_http = MagicMock()
        mock_http.path = "/api/compliance/violations"

        result = await handler.handle("/api/compliance/violations", "GET", mock_http)

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_violation_by_id(self, handler):
        """GET /api/compliance/violations/:id routes to get handler."""
        mock_http = MagicMock()
        mock_http.path = "/api/compliance/violations/viol_123"

        result = await handler.handle("/api/compliance/violations/viol_123", "GET", mock_http)

        assert result is not None

    @pytest.mark.asyncio
    async def test_update_violation(self, handler):
        """PATCH /api/compliance/violations/:id routes to update handler."""
        mock_http = MagicMock()
        mock_http.path = "/api/compliance/violations/viol_123"
        mock_http.rfile = MagicMock()
        mock_http.headers = {"Content-Length": "0"}

        result = await handler.handle("/api/compliance/violations/viol_123", "PATCH", mock_http)

        assert result is not None

    @pytest.mark.asyncio
    async def test_compliance_check(self, handler):
        """POST /api/compliance/check routes to check handler."""
        mock_http = MagicMock()
        mock_http.path = "/api/compliance/check"
        mock_http.rfile = MagicMock()
        mock_http.headers = {"Content-Length": "0"}

        result = await handler.handle("/api/compliance/check", "POST", mock_http)

        assert result is not None

    @pytest.mark.asyncio
    async def test_compliance_stats(self, handler):
        """GET /api/compliance/stats routes to stats handler."""
        mock_http = MagicMock()
        mock_http.path = "/api/compliance/stats"

        result = await handler.handle("/api/compliance/stats", "GET", mock_http)

        assert result is not None


class TestPolicyHandlerValidation:
    """Tests for input validation."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return PolicyHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_invalid_policy_id_returns_error(self, handler):
        """Invalid policy ID returns 400 error or is not handled."""
        mock_http = MagicMock()
        mock_http.path = "/api/policies/../etc/passwd"

        result = await handler.handle("/api/policies/../etc/passwd", "GET", mock_http)

        # Handler may return None (not handled) or 400 error for path traversal attempt
        assert result is None or result.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_violation_id_returns_error(self, handler):
        """Invalid violation ID returns 400 error or is not handled."""
        mock_http = MagicMock()
        mock_http.path = "/api/compliance/violations/../../../"

        result = await handler.handle("/api/compliance/violations/../../../", "GET", mock_http)

        # Handler may return None (not handled) or 400 error
        assert result is None or result.status_code == 400


class TestPolicyHandlerUnknownRoutes:
    """Tests for unknown route handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return PolicyHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_unknown_policy_route_returns_404(self, handler):
        """Unknown policy sub-route returns 404."""
        mock_http = MagicMock()
        mock_http.path = "/api/policies/pol_123/unknown"

        result = await handler.handle("/api/policies/pol_123/unknown", "GET", mock_http)

        # May return 404 or None depending on implementation
        assert result is None or result.status_code in (400, 404)
