"""
Tests for SecurityDebateHandler.

Tests cover:
- Route registration
- Handler import and class hierarchy
- POST endpoint validation (missing JSON, empty findings, non-array findings)
- GET endpoint behavior (returns not_found status)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from tests.server.handlers.conftest import parse_handler_response


# ===========================================================================
# Route Registration Tests
# ===========================================================================


class TestSecurityDebateHandlerRoutes:
    """Verify that SecurityDebateHandler declares the expected routes."""

    def test_has_routes(self):
        from aragora.server.handlers.security_debate import SecurityDebateHandler

        assert hasattr(SecurityDebateHandler, "ROUTES")
        assert len(SecurityDebateHandler.ROUTES) == 2

    def test_debate_route(self):
        from aragora.server.handlers.security_debate import SecurityDebateHandler

        assert "/api/v1/audit/security/debate" in SecurityDebateHandler.ROUTES

    def test_debate_id_route(self):
        from aragora.server.handlers.security_debate import SecurityDebateHandler

        assert "/api/v1/audit/security/debate/:id" in SecurityDebateHandler.ROUTES


# ===========================================================================
# Import and Class Hierarchy Tests
# ===========================================================================


class TestSecurityDebateHandlerImport:
    """Verify the handler can be imported and has the correct class hierarchy."""

    def test_importable(self):
        from aragora.server.handlers.security_debate import SecurityDebateHandler

        assert SecurityDebateHandler is not None

    def test_is_secure_handler(self):
        from aragora.server.handlers.security_debate import SecurityDebateHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(SecurityDebateHandler, SecureHandler)

    def test_has_post_method(self):
        from aragora.server.handlers.security_debate import SecurityDebateHandler

        assert hasattr(SecurityDebateHandler, "post_api_v1_audit_security_debate")

    def test_has_get_method(self):
        from aragora.server.handlers.security_debate import SecurityDebateHandler

        assert hasattr(SecurityDebateHandler, "get_api_v1_audit_security_debate_id")


# ===========================================================================
# Package Export Tests
# ===========================================================================


class TestSecurityDebateHandlerExport:
    """Verify the handler is exported from the handlers package."""

    def test_exported_from_package(self):
        from aragora.server.handlers import SecurityDebateHandler

        assert SecurityDebateHandler is not None


# ===========================================================================
# POST Endpoint Tests
# ===========================================================================


class TestSecurityDebatePost:
    """Test POST /api/v1/audit/security/debate validation and behaviour."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.security_debate import SecurityDebateHandler

        mock_context = MagicMock()
        handler = SecurityDebateHandler.__new__(SecurityDebateHandler)
        handler.server_context = mock_context
        return handler

    @pytest.fixture
    def mock_debate_result(self):
        result = MagicMock()
        result.debate_id = "debate-123"
        result.consensus_reached = True
        result.confidence = 0.85
        result.final_answer = "Apply input validation"
        result.rounds_used = 3
        result.votes = []
        return result

    def test_post_missing_json_returns_400(self, handler):
        handler.get_json_body = MagicMock(return_value=None)
        result = handler.post_api_v1_audit_security_debate()
        body = parse_handler_response(result)
        assert result.status_code == 400
        assert "invalid json" in body.get("error", "").lower()

    def test_post_empty_findings_returns_400(self, handler):
        handler.get_json_body = MagicMock(return_value={"findings": []})
        result = handler.post_api_v1_audit_security_debate()
        body = parse_handler_response(result)
        assert result.status_code == 400
        assert "no findings" in body.get("error", "").lower()

    def test_post_findings_not_array_returns_400(self, handler):
        handler.get_json_body = MagicMock(return_value={"findings": "string"})
        result = handler.post_api_v1_audit_security_debate()
        body = parse_handler_response(result)
        assert result.status_code == 400
        assert "must be an array" in body.get("error", "").lower()


# ===========================================================================
# GET Endpoint Tests
# ===========================================================================


class TestSecurityDebateGet:
    """Test GET /api/v1/audit/security/debate/:id behaviour."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.security_debate import SecurityDebateHandler

        mock_context = MagicMock()
        handler = SecurityDebateHandler.__new__(SecurityDebateHandler)
        handler.server_context = mock_context
        return handler

    def test_get_returns_not_found(self, handler):
        result = handler.get_api_v1_audit_security_debate_id("some-id")
        body = parse_handler_response(result)
        assert body.get("status") == "not_found"
        assert body.get("debate_id") == "some-id"
