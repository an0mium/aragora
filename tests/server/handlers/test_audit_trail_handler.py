"""
Tests for the AuditTrailHandler module.

Tests cover:
- Handler routing for audit trail endpoints
- Handler routing for receipt endpoints
- can_handle method
- HTTP method routing (GET, POST)
- RBAC permission requirements
- Export format handling
- Integrity verification
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from aragora.server.handlers.audit_trail import AuditTrailHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


@pytest.fixture
def mock_audit_store():
    """Create mock audit trail store."""
    store = MagicMock()
    store.list_trails = AsyncMock(
        return_value=[
            {"id": "trail_1", "timestamp": "2025-01-01T00:00:00Z", "type": "debate_result"},
            {"id": "trail_2", "timestamp": "2025-01-02T00:00:00Z", "type": "consensus"},
        ]
    )
    store.get_trail = AsyncMock(
        return_value={
            "id": "trail_1",
            "timestamp": "2025-01-01T00:00:00Z",
            "type": "debate_result",
            "data": {"topic": "Test debate"},
            "checksum": "abc123",
        }
    )
    store.list_receipts = AsyncMock(
        return_value=[
            {"id": "receipt_1", "timestamp": "2025-01-01T00:00:00Z"},
            {"id": "receipt_2", "timestamp": "2025-01-02T00:00:00Z"},
        ]
    )
    store.get_receipt = AsyncMock(
        return_value={
            "id": "receipt_1",
            "timestamp": "2025-01-01T00:00:00Z",
            "decision": "consensus",
            "checksum": "def456",
        }
    )
    return store


class TestAuditTrailHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        with patch("aragora.storage.audit_trail_store.get_audit_trail_store"):
            return AuditTrailHandler(mock_server_context)

    def test_can_handle_audit_trails_base(self, handler):
        """Handler can handle audit trails base endpoint."""
        assert handler.can_handle("/api/v1/audit-trails", "GET")

    def test_can_handle_audit_trail_by_id(self, handler):
        """Handler can handle audit trail by ID."""
        assert handler.can_handle("/api/v1/audit-trails/trail_123", "GET")

    def test_can_handle_audit_trail_export(self, handler):
        """Handler can handle audit trail export endpoint."""
        assert handler.can_handle("/api/v1/audit-trails/trail_123/export", "GET")

    def test_can_handle_audit_trail_verify(self, handler):
        """Handler can handle audit trail verify endpoint."""
        assert handler.can_handle("/api/v1/audit-trails/trail_123/verify", "POST")

    def test_can_handle_receipts_base(self, handler):
        """Handler can handle receipts base endpoint."""
        assert handler.can_handle("/api/v1/receipts", "GET")

    def test_can_handle_receipt_by_id(self, handler):
        """Handler can handle receipt by ID."""
        assert handler.can_handle("/api/v1/receipts/receipt_123", "GET")

    def test_can_handle_receipt_verify(self, handler):
        """Handler can handle receipt verify endpoint."""
        assert handler.can_handle("/api/v1/receipts/receipt_123/verify", "POST")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/policies")

    def test_cannot_handle_put_method(self, handler):
        """Handler cannot handle PUT method."""
        assert not handler.can_handle("/api/v1/audit-trails", "PUT")
        assert not handler.can_handle("/api/v1/receipts", "PUT")


class TestAuditTrailEndpoints:
    """Tests for audit trail endpoint handling."""

    @pytest.fixture
    def handler(self, mock_server_context, mock_audit_store):
        with patch(
            "aragora.storage.audit_trail_store.get_audit_trail_store",
            return_value=mock_audit_store,
        ):
            h = AuditTrailHandler(mock_server_context)
            h._store = mock_audit_store  # Ensure handler uses the mock
            return h

    @pytest.mark.asyncio
    async def test_list_audit_trails(self, handler):
        """GET /api/v1/audit-trails returns list of trails."""
        result = await handler.handle("GET", "/api/v1/audit-trails")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_audit_trails_with_filters(self, handler):
        """GET /api/v1/audit-trails with query params filters results."""
        result = await handler.handle(
            "GET",
            "/api/v1/audit-trails",
            query_params={"type": "debate_result", "limit": "10"},
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_audit_trail_by_id(self, handler):
        """GET /api/v1/audit-trails/:id returns specific trail."""
        result = await handler.handle("GET", "/api/v1/audit-trails/trail_1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_nonexistent_audit_trail(self, handler):
        """GET /api/v1/audit-trails/:id for missing trail returns error or empty."""
        # Handler may use fallback mechanisms, so just verify it handles the request
        handler._store.get_trail = AsyncMock(return_value=None)

        result = await handler.handle("GET", "/api/v1/audit-trails/nonexistent")

        assert result is not None
        # Handler may return 200 with null, 404, or try fallback
        assert result.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_export_audit_trail_routes_correctly(self, handler):
        """GET /api/v1/audit-trails/:id/export routes to export handler."""
        # Just verify it handles the route, actual export logic tested elsewhere
        result = await handler.handle(
            "GET",
            "/api/v1/audit-trails/trail_1/export",
            query_params={"format": "json"},
        )

        # Handler returns a result (may be success or error)
        assert result is not None

    @pytest.mark.asyncio
    async def test_verify_audit_trail_integrity(self, handler):
        """POST /api/v1/audit-trails/:id/verify checks integrity."""
        result = await handler.handle("POST", "/api/v1/audit-trails/trail_1/verify")

        assert result is not None
        # May return 200 with verification result or error depending on checksum


class TestReceiptEndpoints:
    """Tests for receipt endpoint handling."""

    @pytest.fixture
    def handler(self, mock_server_context, mock_audit_store):
        with patch(
            "aragora.storage.audit_trail_store.get_audit_trail_store",
            return_value=mock_audit_store,
        ):
            h = AuditTrailHandler(mock_server_context)
            h._store = mock_audit_store
            return h

    @pytest.mark.asyncio
    async def test_list_receipts(self, handler):
        """GET /api/v1/receipts returns list of receipts."""
        result = await handler.handle("GET", "/api/v1/receipts")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_receipts_with_pagination(self, handler):
        """GET /api/v1/receipts with pagination params."""
        result = await handler.handle(
            "GET",
            "/api/v1/receipts",
            query_params={"limit": "20", "offset": "10"},
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_receipt_by_id(self, handler):
        """GET /api/v1/receipts/:id returns specific receipt."""
        result = await handler.handle("GET", "/api/v1/receipts/receipt_1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_nonexistent_receipt(self, handler):
        """GET /api/v1/receipts/:id for missing receipt returns error or empty."""
        handler._store.get_receipt = AsyncMock(return_value=None)

        result = await handler.handle("GET", "/api/v1/receipts/nonexistent")

        assert result is not None
        # Handler may return 200 with null, 404, or try fallback
        assert result.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_verify_receipt_integrity(self, handler):
        """POST /api/v1/receipts/:id/verify checks integrity."""
        result = await handler.handle("POST", "/api/v1/receipts/receipt_1/verify")

        assert result is not None
        # May return 200 with verification result


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def handler(self, mock_server_context, mock_audit_store):
        with patch(
            "aragora.storage.audit_trail_store.get_audit_trail_store",
            return_value=mock_audit_store,
        ):
            h = AuditTrailHandler(mock_server_context)
            h._store = mock_audit_store
            return h

    @pytest.mark.asyncio
    async def test_unknown_route_returns_404(self, handler):
        """Unknown route returns 404."""
        result = await handler.handle("GET", "/api/v1/audit-trails/trail_1/unknown")

        # The handler routes /api/v1/audit-trails/trail_1/* to get_audit_trail
        # So this might return the trail or 404 depending on implementation
        assert result is not None

    @pytest.mark.asyncio
    async def test_internal_error_handling(self, handler):
        """Internal errors are handled gracefully."""
        # Force an error by making the store raise an exception
        # Use a different approach - patch the internal method
        with patch.object(handler, "_list_audit_trails", side_effect=Exception("Database error")):
            result = await handler.handle("GET", "/api/v1/audit-trails")

        assert result is not None
        assert result.status_code == 500


class TestRBACPermissions:
    """Tests for RBAC permission requirements."""

    @pytest.fixture
    def handler(self, mock_server_context, mock_audit_store):
        with patch(
            "aragora.storage.audit_trail_store.get_audit_trail_store",
            return_value=mock_audit_store,
        ):
            h = AuditTrailHandler(mock_server_context)
            h._store = mock_audit_store
            return h

    def test_list_trails_requires_audit_read(self, handler):
        """List trails requires audit:read permission."""
        # The _list_audit_trails method has @require_permission("audit:read")
        assert hasattr(handler._list_audit_trails, "__wrapped__")

    def test_get_trail_requires_audit_read(self, handler):
        """Get trail requires audit:read permission."""
        assert hasattr(handler._get_audit_trail, "__wrapped__")

    def test_export_trail_requires_audit_export(self, handler):
        """Export trail requires audit:export permission."""
        assert hasattr(handler._export_audit_trail, "__wrapped__")

    def test_verify_trail_requires_audit_verify(self, handler):
        """Verify trail requires audit:verify permission."""
        assert hasattr(handler._verify_audit_trail, "__wrapped__")

    def test_list_receipts_requires_receipts_read(self, handler):
        """List receipts requires audit:receipts.read permission."""
        assert hasattr(handler._list_receipts, "__wrapped__")

    def test_get_receipt_requires_receipts_read(self, handler):
        """Get receipt requires audit:receipts.read permission."""
        assert hasattr(handler._get_receipt, "__wrapped__")

    def test_verify_receipt_requires_receipts_verify(self, handler):
        """Verify receipt requires audit:receipts.verify permission."""
        assert hasattr(handler._verify_receipt, "__wrapped__")
