"""
Tests for ComplianceHandler routing and integration.

Tests cover:
- Route matching (can_handle)
- Request routing to appropriate methods
- Error handling (PermissionDeniedError, general exceptions)
- Rate limiting
- Handler initialization
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.compliance.handler import (
    ComplianceHandler,
    create_compliance_handler,
)
from aragora.server.handlers.base import HandlerResult
from aragora.rbac.decorators import PermissionDeniedError


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def compliance_handler():
    """Create a compliance handler instance."""
    return ComplianceHandler(server_context={})


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = "GET"
    handler.headers = {}
    return handler


# ============================================================================
# Handler Initialization Tests
# ============================================================================


class TestHandlerInitialization:
    """Tests for handler initialization."""

    def test_handler_creates_with_context(self):
        """Handler initializes with server context."""
        context = {"key": "value"}
        handler = ComplianceHandler(server_context=context)
        assert handler is not None

    def test_factory_function(self):
        """Factory function creates handler."""
        context = {"key": "value"}
        handler = create_compliance_handler(context)
        assert isinstance(handler, ComplianceHandler)

    def test_handler_routes_defined(self):
        """Handler has ROUTES class attribute."""
        assert hasattr(ComplianceHandler, "ROUTES")
        assert "/api/v2/compliance" in ComplianceHandler.ROUTES
        assert "/api/v2/compliance/*" in ComplianceHandler.ROUTES


# ============================================================================
# Route Matching Tests (can_handle)
# ============================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_can_handle_compliance_status(self, compliance_handler):
        """Handles /api/v2/compliance/status."""
        assert compliance_handler.can_handle("/api/v2/compliance/status", "GET") is True

    def test_can_handle_soc2_report(self, compliance_handler):
        """Handles /api/v2/compliance/soc2-report."""
        assert compliance_handler.can_handle("/api/v2/compliance/soc2-report", "GET") is True

    def test_can_handle_gdpr_export(self, compliance_handler):
        """Handles /api/v2/compliance/gdpr-export."""
        assert compliance_handler.can_handle("/api/v2/compliance/gdpr-export", "GET") is True

    def test_can_handle_rtbf(self, compliance_handler):
        """Handles /api/v2/compliance/gdpr/right-to-be-forgotten."""
        assert (
            compliance_handler.can_handle("/api/v2/compliance/gdpr/right-to-be-forgotten", "POST")
            is True
        )

    def test_can_handle_legal_holds(self, compliance_handler):
        """Handles legal hold endpoints."""
        assert compliance_handler.can_handle("/api/v2/compliance/gdpr/legal-holds", "GET") is True
        assert compliance_handler.can_handle("/api/v2/compliance/gdpr/legal-holds", "POST") is True
        assert (
            compliance_handler.can_handle("/api/v2/compliance/gdpr/legal-holds/hold-001", "DELETE")
            is True
        )

    def test_can_handle_deletions(self, compliance_handler):
        """Handles deletion endpoints."""
        assert compliance_handler.can_handle("/api/v2/compliance/gdpr/deletions", "GET") is True
        assert (
            compliance_handler.can_handle("/api/v2/compliance/gdpr/deletions/del-001", "GET")
            is True
        )
        assert (
            compliance_handler.can_handle(
                "/api/v2/compliance/gdpr/deletions/del-001/cancel", "POST"
            )
            is True
        )

    def test_can_handle_audit_endpoints(self, compliance_handler):
        """Handles audit endpoints."""
        assert compliance_handler.can_handle("/api/v2/compliance/audit-verify", "POST") is True
        assert compliance_handler.can_handle("/api/v2/compliance/audit-events", "GET") is True

    def test_can_handle_rejects_unsupported_methods(self, compliance_handler):
        """Rejects unsupported HTTP methods."""
        assert compliance_handler.can_handle("/api/v2/compliance/status", "PUT") is False
        assert compliance_handler.can_handle("/api/v2/compliance/status", "PATCH") is False

    def test_can_handle_rejects_other_paths(self, compliance_handler):
        """Rejects non-compliance paths."""
        assert compliance_handler.can_handle("/api/v1/debates", "GET") is False
        assert compliance_handler.can_handle("/api/v2/billing", "GET") is False
        assert compliance_handler.can_handle("/health", "GET") is False


# ============================================================================
# Request Routing Tests
# ============================================================================


class TestRequestRouting:
    """Tests for request routing to appropriate methods."""

    @pytest.mark.asyncio
    async def test_routes_to_status(self, compliance_handler, mock_handler):
        """Routes /api/v2/compliance/status to _get_status."""
        mock_handler.command = "GET"

        with patch.object(compliance_handler, "_get_status", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle("/api/v2/compliance/status", {}, mock_handler)

        mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_soc2_report(self, compliance_handler, mock_handler):
        """Routes /api/v2/compliance/soc2-report to _get_soc2_report."""
        mock_handler.command = "GET"

        with patch.object(
            compliance_handler, "_get_soc2_report", new_callable=AsyncMock
        ) as mock_method:
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle(
                "/api/v2/compliance/soc2-report", {"format": "json"}, mock_handler
            )

        mock_method.assert_called_once_with({"format": "json"})

    @pytest.mark.asyncio
    async def test_routes_to_gdpr_export(self, compliance_handler, mock_handler):
        """Routes /api/v2/compliance/gdpr-export to _gdpr_export."""
        mock_handler.command = "GET"

        with patch.object(
            compliance_handler, "_gdpr_export", new_callable=AsyncMock
        ) as mock_method:
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle(
                "/api/v2/compliance/gdpr-export", {"user_id": "123"}, mock_handler
            )

        mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_rtbf(self, compliance_handler, mock_handler):
        """Routes /api/v2/compliance/gdpr/right-to-be-forgotten to _right_to_be_forgotten."""
        mock_handler.command = "POST"

        with (
            patch.object(
                compliance_handler, "_right_to_be_forgotten", new_callable=AsyncMock
            ) as mock_method,
            patch.object(compliance_handler, "read_json_body", return_value={"user_id": "123"}),
        ):
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle(
                "/api/v2/compliance/gdpr/right-to-be-forgotten", {}, mock_handler
            )

        mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_list_legal_holds(self, compliance_handler, mock_handler):
        """Routes /api/v2/compliance/gdpr/legal-holds GET to _list_legal_holds."""
        mock_handler.command = "GET"

        with patch.object(
            compliance_handler, "_list_legal_holds", new_callable=AsyncMock
        ) as mock_method:
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle("/api/v2/compliance/gdpr/legal-holds", {}, mock_handler)

        mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_create_legal_hold(self, compliance_handler, mock_handler):
        """Routes /api/v2/compliance/gdpr/legal-holds POST to _create_legal_hold."""
        mock_handler.command = "POST"

        with (
            patch.object(
                compliance_handler, "_create_legal_hold", new_callable=AsyncMock
            ) as mock_method,
            patch.object(
                compliance_handler,
                "read_json_body",
                return_value={"user_ids": ["123"], "reason": "test"},
            ),
        ):
            mock_method.return_value = HandlerResult(
                status_code=201, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle("/api/v2/compliance/gdpr/legal-holds", {}, mock_handler)

        mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_release_legal_hold(self, compliance_handler, mock_handler):
        """Routes /api/v2/compliance/gdpr/legal-holds/:id DELETE to _release_legal_hold."""
        mock_handler.command = "DELETE"

        with (
            patch.object(
                compliance_handler, "_release_legal_hold", new_callable=AsyncMock
            ) as mock_method,
            patch.object(compliance_handler, "read_json_body", return_value={}),
        ):
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle(
                "/api/v2/compliance/gdpr/legal-holds/hold-001", {}, mock_handler
            )

        mock_method.assert_called_once_with("hold-001", {})

    @pytest.mark.asyncio
    async def test_routes_to_cancel_deletion(self, compliance_handler, mock_handler):
        """Routes /api/v2/compliance/gdpr/deletions/:id/cancel to _cancel_deletion."""
        mock_handler.command = "POST"

        with (
            patch.object(
                compliance_handler, "_cancel_deletion", new_callable=AsyncMock
            ) as mock_method,
            patch.object(compliance_handler, "read_json_body", return_value={}),
        ):
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle(
                "/api/v2/compliance/gdpr/deletions/del-001/cancel", {}, mock_handler
            )

        mock_method.assert_called_once_with("del-001", {})

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_path(self, compliance_handler, mock_handler):
        """Returns 404 for unknown paths."""
        mock_handler.command = "GET"

        result = await compliance_handler.handle(
            "/api/v2/compliance/unknown-endpoint", {}, mock_handler
        )

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "not found" in body.get("error", "").lower()


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in request routing."""

    @pytest.mark.asyncio
    async def test_handles_permission_denied(self, compliance_handler, mock_handler):
        """Permission denied errors return 403."""
        mock_handler.command = "GET"

        with patch.object(
            compliance_handler,
            "_get_status",
            new_callable=AsyncMock,
            side_effect=PermissionDeniedError("Insufficient permissions"),
        ):
            result = await compliance_handler.handle("/api/v2/compliance/status", {}, mock_handler)

        assert result.status_code == 403
        body = json.loads(result.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_handles_general_exception(self, compliance_handler, mock_handler):
        """General exceptions return 500."""
        mock_handler.command = "GET"

        with patch.object(
            compliance_handler,
            "_get_status",
            new_callable=AsyncMock,
            side_effect=ValueError("Unexpected error"),
        ):
            result = await compliance_handler.handle("/api/v2/compliance/status", {}, mock_handler)

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "error" in body.get("error", "").lower()


# ============================================================================
# Handler Method Extraction Tests
# ============================================================================


class TestMethodExtraction:
    """Tests for HTTP method extraction from handler."""

    @pytest.mark.asyncio
    async def test_extracts_method_from_handler(self, compliance_handler):
        """Extracts HTTP method from handler.command."""
        mock_handler = MagicMock()
        mock_handler.command = "POST"
        mock_handler.headers = {}

        with (
            patch.object(
                compliance_handler, "_right_to_be_forgotten", new_callable=AsyncMock
            ) as mock_method,
            patch.object(compliance_handler, "read_json_body", return_value={"user_id": "123"}),
        ):
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle(
                "/api/v2/compliance/gdpr/right-to-be-forgotten", {}, mock_handler
            )

        mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_defaults_to_get_when_no_handler(self, compliance_handler):
        """Defaults to GET when handler is None."""
        with patch.object(compliance_handler, "_get_status", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle("/api/v2/compliance/status", {}, None)

        mock_method.assert_called_once()


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting decorator."""

    def test_handle_has_rate_limit_decorator(self):
        """Handle method has rate limit decorator."""
        import inspect

        source = inspect.getsource(ComplianceHandler.handle)
        assert "rate_limit" in source
        assert "requests_per_minute" in source


# ============================================================================
# Handler Tracking Tests
# ============================================================================


class TestHandlerTracking:
    """Tests for handler metrics tracking."""

    def test_handle_has_track_handler_decorator(self):
        """Handle method has metrics tracking."""
        import inspect

        source = inspect.getsource(ComplianceHandler.handle)
        assert "track_handler" in source
        assert "compliance/main" in source


# ============================================================================
# Body Reading Tests
# ============================================================================


class TestBodyReading:
    """Tests for request body reading."""

    @pytest.mark.asyncio
    async def test_reads_json_body_for_post(self, compliance_handler, mock_handler):
        """Reads JSON body for POST requests."""
        mock_handler.command = "POST"

        with (
            patch.object(
                compliance_handler, "_right_to_be_forgotten", new_callable=AsyncMock
            ) as mock_method,
            patch.object(
                compliance_handler, "read_json_body", return_value={"user_id": "test"}
            ) as mock_read,
        ):
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle(
                "/api/v2/compliance/gdpr/right-to-be-forgotten", {}, mock_handler
            )

        mock_read.assert_called_once_with(mock_handler)

    @pytest.mark.asyncio
    async def test_handles_null_body(self, compliance_handler, mock_handler):
        """Handles null body from read_json_body."""
        mock_handler.command = "POST"

        with (
            patch.object(
                compliance_handler, "_coordinated_deletion", new_callable=AsyncMock
            ) as mock_method,
            patch.object(compliance_handler, "read_json_body", return_value=None),
        ):
            mock_method.return_value = HandlerResult(
                status_code=400, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle(
                "/api/v2/compliance/gdpr/coordinated-deletion", {}, mock_handler
            )

        # Should be called with empty dict when body is None
        mock_method.assert_called_once()


# ============================================================================
# Headers Extraction Tests
# ============================================================================


class TestHeadersExtraction:
    """Tests for headers extraction from handler."""

    @pytest.mark.asyncio
    async def test_extracts_headers_for_legal_hold(self, compliance_handler, mock_handler):
        """Extracts headers for legal hold creation."""
        mock_handler.command = "POST"
        mock_handler.headers = {"Authorization": "Bearer token123"}

        with (
            patch.object(
                compliance_handler, "_create_legal_hold", new_callable=AsyncMock
            ) as mock_method,
            patch.object(
                compliance_handler,
                "read_json_body",
                return_value={"user_ids": ["123"], "reason": "test"},
            ),
        ):
            mock_method.return_value = HandlerResult(
                status_code=201, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle("/api/v2/compliance/gdpr/legal-holds", {}, mock_handler)

        # Headers should be passed to _create_legal_hold
        call_args = mock_method.call_args
        # Headers can be passed as positional arg (index 1) or keyword arg
        headers_arg = call_args.kwargs.get("headers") if call_args.kwargs else None
        if headers_arg is None and len(call_args.args) > 1:
            headers_arg = call_args.args[1]
        assert headers_arg is not None

    @pytest.mark.asyncio
    async def test_handles_empty_headers(self, compliance_handler, mock_handler):
        """Handles handler with empty headers dict."""
        mock_handler.command = "GET"
        mock_handler.headers = {}

        with patch.object(compliance_handler, "_get_status", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle("/api/v2/compliance/status", {}, mock_handler)

        mock_method.assert_called_once()


# ============================================================================
# Query Params Tests
# ============================================================================


class TestQueryParams:
    """Tests for query parameter handling."""

    @pytest.mark.asyncio
    async def test_passes_query_params_to_handler(self, compliance_handler, mock_handler):
        """Query params are passed to handler methods."""
        mock_handler.command = "GET"

        with patch.object(
            compliance_handler, "_get_soc2_report", new_callable=AsyncMock
        ) as mock_method:
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle(
                "/api/v2/compliance/soc2-report",
                {"format": "html", "period_start": "2025-01-01"},
                mock_handler,
            )

        mock_method.assert_called_once_with({"format": "html", "period_start": "2025-01-01"})

    @pytest.mark.asyncio
    async def test_handles_none_query_params(self, compliance_handler, mock_handler):
        """Handles None query_params."""
        mock_handler.command = "GET"

        with patch.object(compliance_handler, "_get_status", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            await compliance_handler.handle("/api/v2/compliance/status", None, mock_handler)

        mock_method.assert_called_once()
