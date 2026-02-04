"""Tests for the DeliberationsHandler."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.deliberations import (
    DeliberationsHandler,
    RBAC_AVAILABLE,
    _active_deliberations,
    _stats,
)


class TestDeliberationsHandler:
    """Tests for DeliberationsHandler."""

    def _make_handler(self, ctx: dict | None = None) -> DeliberationsHandler:
        return DeliberationsHandler(server_context=ctx or {})

    def _make_request(
        self,
        path: str = "/api/v1/deliberations/active",
        method: str = "GET",
    ) -> MagicMock:
        mock = MagicMock()
        mock.path = path
        mock.method = method
        mock.headers = {}
        return mock

    # -------------------------------------------------------------------------
    # ROUTES tests
    # -------------------------------------------------------------------------

    def test_routes_include_active(self):
        handler = self._make_handler()
        assert "/api/v1/deliberations/active" in handler.ROUTES

    def test_routes_include_stats(self):
        handler = self._make_handler()
        assert "/api/v1/deliberations/stats" in handler.ROUTES

    def test_routes_include_stream(self):
        handler = self._make_handler()
        assert "/api/v1/deliberations/stream" in handler.ROUTES

    def test_routes_include_deliberation_by_id(self):
        handler = self._make_handler()
        assert "/api/v1/deliberations/{deliberation_id}" in handler.ROUTES

    # -------------------------------------------------------------------------
    # Authorization context tests
    # -------------------------------------------------------------------------

    def test_get_auth_context_no_rbac(self):
        handler = self._make_handler()
        request = self._make_request()

        # With RBAC disabled, should return None
        with patch("aragora.server.handlers.deliberations.RBAC_AVAILABLE", False):
            ctx = handler._get_auth_context(request)

        assert ctx is None

    def test_check_rbac_permission_no_rbac_allows(self):
        handler = self._make_handler()
        request = self._make_request()

        with patch("aragora.server.handlers.deliberations.RBAC_AVAILABLE", False):
            result = handler._check_rbac_permission(request, "analytics.read")

        assert result is None

    # -------------------------------------------------------------------------
    # Request handling tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_handle_request_active_route(self):
        handler = self._make_handler()
        request = self._make_request(path="/api/v1/deliberations/active")

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(
                handler,
                "_get_active_deliberations",
                new_callable=AsyncMock,
                return_value=({"deliberations": [], "count": 0}, 200),
            ) as mock_get:
                result = await handler.handle_request(request)

                mock_get.assert_called_once_with(request)
                assert result[1] == 200

    @pytest.mark.asyncio
    async def test_handle_request_stats_route(self):
        handler = self._make_handler()
        request = self._make_request(path="/api/v1/deliberations/stats")

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(
                handler,
                "_get_stats",
                new_callable=AsyncMock,
                return_value=({"stats": {}}, 200),
            ) as mock_get:
                result = await handler.handle_request(request)

                mock_get.assert_called_once_with(request)
                assert result[1] == 200

    @pytest.mark.asyncio
    async def test_handle_request_single_deliberation(self):
        handler = self._make_handler()
        request = self._make_request(path="/api/v1/deliberations/del_123")

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(
                handler,
                "_get_deliberation",
                new_callable=AsyncMock,
                return_value=({"deliberation": {}}, 200),
            ) as mock_get:
                result = await handler.handle_request(request)

                mock_get.assert_called_once_with(request, "del_123")
                assert result[1] == 200

    @pytest.mark.asyncio
    async def test_handle_request_not_found(self):
        handler = self._make_handler()
        request = self._make_request(path="/api/v1/unknown", method="POST")

        result = await handler.handle_request(request)

        assert result[1] == 404

    @pytest.mark.asyncio
    async def test_handle_request_rbac_denied(self):
        handler = self._make_handler()
        request = self._make_request(path="/api/v1/deliberations/active")

        error_response = ({"error": "Permission denied"}, 403)
        with patch.object(handler, "_check_rbac_permission", return_value=error_response):
            result = await handler.handle_request(request)

            assert result[1] == 403

    # -------------------------------------------------------------------------
    # Active deliberations tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_active_deliberations_empty(self):
        handler = self._make_handler()
        request = self._make_request()

        with patch.object(
            handler, "_fetch_active_from_store", new_callable=AsyncMock, return_value=[]
        ):
            result, status = await handler._get_active_deliberations(request)

        assert status == 200
        assert result["count"] == 0
        assert result["deliberations"] == []
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_get_active_deliberations_with_data(self):
        handler = self._make_handler()
        request = self._make_request()

        mock_deliberations = [
            {"id": "d1", "topic": "Test 1"},
            {"id": "d2", "topic": "Test 2"},
        ]

        with patch.object(
            handler,
            "_fetch_active_from_store",
            new_callable=AsyncMock,
            return_value=mock_deliberations,
        ):
            result, status = await handler._get_active_deliberations(request)

        assert status == 200
        assert result["count"] == 2
        assert len(result["deliberations"]) == 2

    @pytest.mark.asyncio
    async def test_get_active_deliberations_error(self):
        handler = self._make_handler()
        request = self._make_request()

        with patch.object(
            handler,
            "_fetch_active_from_store",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB error"),
        ):
            result, status = await handler._get_active_deliberations(request)

        assert status == 500
        assert "error" in result

    # -------------------------------------------------------------------------
    # Module variables tests
    # -------------------------------------------------------------------------

    def test_stats_structure(self):
        assert "active_count" in _stats
        assert "completed_today" in _stats
        assert "average_consensus_time" in _stats
        assert "average_rounds" in _stats
        assert "top_agents" in _stats

    def test_active_deliberations_is_dict(self):
        assert isinstance(_active_deliberations, dict)


class TestDeliberationsHandlerRBACAvailability:
    """Tests for RBAC availability flag."""

    def test_rbac_available_is_boolean(self):
        assert isinstance(RBAC_AVAILABLE, bool)
