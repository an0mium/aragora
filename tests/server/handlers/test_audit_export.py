"""
Tests for audit export handler.

Tests:
- register_handlers function
- Handler function signatures
- Module exports
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from aragora.server.handlers.audit_export import (
    handle_audit_events,
    handle_audit_export,
    handle_audit_stats,
    handle_audit_verify,
    register_handlers,
    get_audit_log,
)


class TestRegisterHandlers:
    """Tests for register_handlers function."""

    def test_registers_audit_events_route(self):
        """Should register /api/audit/events GET route."""
        app = MagicMock()
        register_handlers(app)
        # Check add_get was called with correct path
        calls = app.router.add_get.call_args_list
        paths = [call[0][0] for call in calls]
        assert "/api/audit/events" in paths

    def test_registers_audit_stats_route(self):
        """Should register /api/audit/stats GET route."""
        app = MagicMock()
        register_handlers(app)
        calls = app.router.add_get.call_args_list
        paths = [call[0][0] for call in calls]
        assert "/api/audit/stats" in paths

    def test_registers_audit_export_route(self):
        """Should register /api/audit/export POST route."""
        app = MagicMock()
        register_handlers(app)
        calls = app.router.add_post.call_args_list
        paths = [call[0][0] for call in calls]
        assert "/api/audit/export" in paths

    def test_registers_audit_verify_route(self):
        """Should register /api/audit/verify POST route."""
        app = MagicMock()
        register_handlers(app)
        calls = app.router.add_post.call_args_list
        paths = [call[0][0] for call in calls]
        assert "/api/audit/verify" in paths


class TestGetAuditLog:
    """Tests for get_audit_log function."""

    def test_returns_audit_log_instance(self):
        """Should return an AuditLog instance or raise ImportError."""
        try:
            result = get_audit_log()
            # If we get here, audit module is available
            assert result is not None
        except (ImportError, ModuleNotFoundError):
            # Audit module not available, that's OK
            pytest.skip("Audit module not available")


class TestHandleAuditEventsSignature:
    """Tests for handle_audit_events function signature."""

    def test_is_async_function(self):
        """handle_audit_events should be an async function."""
        import asyncio
        assert asyncio.iscoroutinefunction(handle_audit_events)


class TestHandleAuditStatsSignature:
    """Tests for handle_audit_stats function signature."""

    def test_is_async_function(self):
        """handle_audit_stats should be an async function."""
        import asyncio
        assert asyncio.iscoroutinefunction(handle_audit_stats)


class TestHandleAuditExportSignature:
    """Tests for handle_audit_export function signature."""

    def test_is_async_function(self):
        """handle_audit_export should be an async function."""
        import asyncio
        assert asyncio.iscoroutinefunction(handle_audit_export)


class TestHandleAuditVerifySignature:
    """Tests for handle_audit_verify function signature."""

    def test_is_async_function(self):
        """handle_audit_verify should be an async function."""
        import asyncio
        assert asyncio.iscoroutinefunction(handle_audit_verify)


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_exports_handle_audit_events(self):
        """Should export handle_audit_events."""
        from aragora.server.handlers import audit_export
        assert "handle_audit_events" in audit_export.__all__

    def test_exports_handle_audit_export(self):
        """Should export handle_audit_export."""
        from aragora.server.handlers import audit_export
        assert "handle_audit_export" in audit_export.__all__

    def test_exports_handle_audit_stats(self):
        """Should export handle_audit_stats."""
        from aragora.server.handlers import audit_export
        assert "handle_audit_stats" in audit_export.__all__

    def test_exports_handle_audit_verify(self):
        """Should export handle_audit_verify."""
        from aragora.server.handlers import audit_export
        assert "handle_audit_verify" in audit_export.__all__

    def test_exports_register_handlers(self):
        """Should export register_handlers."""
        from aragora.server.handlers import audit_export
        assert "register_handlers" in audit_export.__all__


@pytest.mark.asyncio
class TestHandleAuditEventsValidation:
    """Tests for handle_audit_events parameter validation."""

    async def test_invalid_start_date_returns_400(self):
        """Invalid start_date should return 400."""
        request = MagicMock()
        request.query = {"start_date": "not-a-date"}

        response = await handle_audit_events(request)
        assert response.status == 400

    async def test_invalid_end_date_returns_400(self):
        """Invalid end_date should return 400."""
        request = MagicMock()
        request.query = {"end_date": "not-a-date"}

        response = await handle_audit_events(request)
        assert response.status == 400

    async def test_invalid_category_returns_400(self):
        """Invalid category should return 400."""
        request = MagicMock()
        request.query = {"category": "invalid_category_value"}

        response = await handle_audit_events(request)
        assert response.status == 400

    async def test_invalid_outcome_returns_400(self):
        """Invalid outcome should return 400."""
        request = MagicMock()
        request.query = {"outcome": "invalid_outcome_value"}

        response = await handle_audit_events(request)
        assert response.status == 400


@pytest.mark.asyncio
class TestHandleAuditExportValidation:
    """Tests for handle_audit_export parameter validation."""

    async def test_missing_start_date_returns_400(self):
        """Missing start_date should return 400."""
        request = MagicMock()
        request.json = AsyncMock(return_value={"end_date": "2024-01-01"})

        response = await handle_audit_export(request)
        assert response.status == 400

    async def test_missing_end_date_returns_400(self):
        """Missing end_date should return 400."""
        request = MagicMock()
        request.json = AsyncMock(return_value={"start_date": "2024-01-01"})

        response = await handle_audit_export(request)
        assert response.status == 400

    async def test_invalid_json_returns_400(self):
        """Invalid JSON body should return 400."""
        import json
        request = MagicMock()
        request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "test", 0))

        response = await handle_audit_export(request)
        assert response.status == 400

    async def test_invalid_date_format_returns_400(self):
        """Invalid date format should return 400."""
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "start_date": "not-a-date",
            "end_date": "2024-01-01"
        })

        response = await handle_audit_export(request)
        assert response.status == 400

    async def test_invalid_format_returns_400(self):
        """Invalid export format should return 400."""
        # Mock the audit log
        import aragora.server.handlers.audit_export as audit_module
        mock_audit = MagicMock()
        original_get_audit_log = audit_module.get_audit_log
        audit_module.get_audit_log = lambda: mock_audit

        try:
            request = MagicMock()
            request.json = AsyncMock(return_value={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "format": "invalid_format"
            })

            response = await handle_audit_export(request)
            assert response.status == 400
        finally:
            audit_module.get_audit_log = original_get_audit_log


@pytest.mark.asyncio
class TestHandleAuditVerifyValidation:
    """Tests for handle_audit_verify parameter validation."""

    async def test_invalid_start_date_returns_400(self):
        """Invalid start_date should return 400."""
        request = MagicMock()
        request.json = AsyncMock(return_value={"start_date": "not-a-date"})

        response = await handle_audit_verify(request)
        assert response.status == 400

    async def test_invalid_end_date_returns_400(self):
        """Invalid end_date should return 400."""
        request = MagicMock()
        request.json = AsyncMock(return_value={"end_date": "not-a-date"})

        response = await handle_audit_verify(request)
        assert response.status == 400

    async def test_empty_body_is_valid(self):
        """Empty body should be valid (verify all)."""
        import json
        import aragora.server.handlers.audit_export as audit_module
        mock_audit = MagicMock()
        mock_audit.verify_integrity = MagicMock(return_value=(True, []))
        original_get_audit_log = audit_module.get_audit_log
        audit_module.get_audit_log = lambda: mock_audit

        try:
            request = MagicMock()
            request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "test", 0))

            response = await handle_audit_verify(request)
            assert response.status == 200
        finally:
            audit_module.get_audit_log = original_get_audit_log
