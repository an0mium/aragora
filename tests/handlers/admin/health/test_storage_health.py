"""Comprehensive tests for StorageHealthHandler.

Tests all public methods and routes in
aragora/server/handlers/admin/health/storage_health.py (113 lines):

  TestStorageHealthHandlerInit        - __init__ context handling
  TestCanHandle                       - can_handle() route matching
  TestHandle                          - handle() routing and dispatch
  TestHandleAuth                      - Authentication and RBAC enforcement
  TestDatabaseStoresHealthDelegate    - _database_stores_health() delegation
  TestDatabaseSchemaHealthDelegate    - _database_schema_health() delegation
  TestGetStorage                      - get_storage() context accessor
  TestGetEloSystem                    - get_elo_system() context accessor + class attr
  TestGetNomicDir                     - get_nomic_dir() context accessor
  TestClassAttributes                 - ROUTES, HEALTH_PERMISSION, RESOURCE_TYPE
  TestPathNormalization               - v1 and non-v1 path normalization
  TestEdgeCases                       - None returns, unknown paths, empty params

45+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.admin.health.storage_health import StorageHealthHandler
from aragora.server.handlers.utils.auth import ForbiddenError, UnauthorizedError
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


_STORAGE_MOD = "aragora.server.handlers.admin.health.storage_health"
_DATABASE_MOD = "aragora.server.handlers.admin.health.database"


class MockHTTPHandler:
    """Mock HTTP handler for passing to handle()."""

    def __init__(self):
        self.command = "GET"
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"{}"
        self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Default StorageHealthHandler with empty context."""
    return StorageHealthHandler(ctx={})


@pytest.fixture
def handler_with_ctx():
    """StorageHealthHandler with populated context."""
    return StorageHealthHandler(ctx={
        "storage": MagicMock(name="mock_storage"),
        "elo_system": MagicMock(name="mock_elo"),
        "nomic_dir": "/tmp/nomic",
    })


@pytest.fixture
def http_handler():
    """Mock HTTP handler."""
    return MockHTTPHandler()


# ===========================================================================
# TestStorageHealthHandlerInit
# ===========================================================================


class TestStorageHealthHandlerInit:
    """Tests for __init__ context handling."""

    def test_init_with_empty_ctx(self):
        """Handler initializes with empty dict when ctx is empty."""
        h = StorageHealthHandler(ctx={})
        assert h.ctx == {}

    def test_init_with_none_ctx(self):
        """Handler initializes with empty dict when ctx is None."""
        h = StorageHealthHandler(ctx=None)
        assert h.ctx == {}

    def test_init_with_no_args(self):
        """Handler initializes with empty dict when no args passed."""
        h = StorageHealthHandler()
        assert h.ctx == {}

    def test_init_preserves_ctx(self):
        """Handler preserves the provided context dict."""
        ctx = {"storage": "test", "extra": 42}
        h = StorageHealthHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_init_with_populated_ctx(self):
        """Handler stores references from context."""
        mock_storage = MagicMock()
        h = StorageHealthHandler(ctx={"storage": mock_storage})
        assert h.ctx["storage"] is mock_storage


# ===========================================================================
# TestCanHandle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_can_handle_stores_v1(self, handler):
        """Matches /api/v1/health/stores."""
        assert handler.can_handle("/api/v1/health/stores") is True

    def test_can_handle_database_v1(self, handler):
        """Matches /api/v1/health/database."""
        assert handler.can_handle("/api/v1/health/database") is True

    def test_can_handle_stores_non_v1(self, handler):
        """Matches /api/health/stores (backward compat)."""
        assert handler.can_handle("/api/health/stores") is True

    def test_cannot_handle_unknown_path(self, handler):
        """Rejects unknown paths."""
        assert handler.can_handle("/api/v1/health/unknown") is False

    def test_cannot_handle_empty_path(self, handler):
        """Rejects empty string."""
        assert handler.can_handle("") is False

    def test_cannot_handle_partial_match(self, handler):
        """Rejects partial path matches."""
        assert handler.can_handle("/api/v1/health") is False

    def test_cannot_handle_extra_suffix(self, handler):
        """Rejects paths with extra trailing segments."""
        assert handler.can_handle("/api/v1/health/stores/extra") is False

    def test_cannot_handle_database_non_v1(self, handler):
        """Non-v1 database path is NOT in ROUTES."""
        assert handler.can_handle("/api/health/database") is False

    def test_cannot_handle_different_prefix(self, handler):
        """Rejects paths with wrong prefix."""
        assert handler.can_handle("/v1/health/stores") is False

    def test_cannot_handle_trailing_slash(self, handler):
        """Rejects paths with trailing slash."""
        assert handler.can_handle("/api/v1/health/stores/") is False


# ===========================================================================
# TestHandle
# ===========================================================================


class TestHandle:
    """Tests for handle() routing and dispatch."""

    @pytest.mark.asyncio
    async def test_handle_stores_v1_dispatches(self, handler, http_handler):
        """Route /api/v1/health/stores dispatches to _database_stores_health."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{}')
        with patch.object(handler, "_database_stores_health", return_value=mock_result) as m:
            result = await handler.handle("/api/v1/health/stores", {}, http_handler)
        m.assert_called_once()
        assert result is mock_result

    @pytest.mark.asyncio
    async def test_handle_stores_non_v1_dispatches(self, handler, http_handler):
        """Route /api/health/stores dispatches to _database_stores_health."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{}')
        with patch.object(handler, "_database_stores_health", return_value=mock_result) as m:
            result = await handler.handle("/api/health/stores", {}, http_handler)
        m.assert_called_once()
        assert result is mock_result

    @pytest.mark.asyncio
    async def test_handle_database_v1_dispatches(self, handler, http_handler):
        """Route /api/v1/health/database dispatches to _database_schema_health."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{}')
        with patch.object(handler, "_database_schema_health", return_value=mock_result) as m:
            result = await handler.handle("/api/v1/health/database", {}, http_handler)
        m.assert_called_once()
        assert result is mock_result

    @pytest.mark.asyncio
    async def test_handle_unknown_path_returns_none(self, handler, http_handler):
        """Unknown path returns None."""
        result = await handler.handle("/api/v1/health/unknown", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_passes_query_params_ignored(self, handler, http_handler):
        """Query params are accepted but not used by this handler."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{}')
        with patch.object(handler, "_database_stores_health", return_value=mock_result):
            result = await handler.handle("/api/v1/health/stores", {"verbose": "true"}, http_handler)
        assert result is mock_result

    @pytest.mark.asyncio
    async def test_handle_returns_handler_result(self, handler, http_handler):
        """handle() returns a HandlerResult for valid routes."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{"status":"ok"}')
        with patch.object(handler, "_database_stores_health", return_value=mock_result):
            result = await handler.handle("/api/v1/health/stores", {}, http_handler)
        assert isinstance(result, HandlerResult)


# ===========================================================================
# TestHandleAuth
# ===========================================================================


class TestHandleAuth:
    """Tests for authentication and RBAC enforcement."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self, http_handler):
        """UnauthorizedError from get_auth_context returns 401."""
        h = StorageHealthHandler(ctx={})
        with patch.object(
            h, "get_auth_context", new_callable=AsyncMock, side_effect=UnauthorizedError("Not authenticated")
        ):
            result = await h.handle("/api/v1/health/stores", {}, http_handler)
        assert _status(result) == 401
        assert "Authentication required" in _body(result).get("error", "")

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self, http_handler):
        """ForbiddenError from check_permission returns 403."""
        h = StorageHealthHandler(ctx={})
        mock_auth_ctx = MagicMock()
        with patch.object(
            h, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_ctx
        ), patch.object(
            h, "check_permission", side_effect=ForbiddenError("No access", permission="system.health.read")
        ):
            result = await h.handle("/api/v1/health/stores", {}, http_handler)
        assert _status(result) == 403
        assert "Permission denied" in _body(result).get("error", "")

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_auth_required_before_routing(self, http_handler):
        """Auth check happens before path routing -- even unknown paths get 401."""
        h = StorageHealthHandler(ctx={})
        with patch.object(
            h, "get_auth_context", new_callable=AsyncMock, side_effect=UnauthorizedError("Missing token")
        ):
            result = await h.handle("/api/v1/health/stores", {}, http_handler)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_permission_check_uses_health_permission(self, http_handler):
        """check_permission is called with HEALTH_PERMISSION."""
        h = StorageHealthHandler(ctx={})
        mock_auth_ctx = MagicMock()
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{}')
        with patch.object(
            h, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_ctx
        ), patch.object(
            h, "check_permission"
        ) as mock_perm, patch.object(
            h, "_database_stores_health", return_value=mock_result
        ):
            await h.handle("/api/v1/health/stores", {}, http_handler)
        mock_perm.assert_called_once_with(mock_auth_ctx, "system.health.read")

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_auth_401_on_database_endpoint(self, http_handler):
        """Auth failure on /api/v1/health/database also returns 401."""
        h = StorageHealthHandler(ctx={})
        with patch.object(
            h, "get_auth_context", new_callable=AsyncMock, side_effect=UnauthorizedError()
        ):
            result = await h.handle("/api/v1/health/database", {}, http_handler)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_auth_403_on_stores_non_v1(self, http_handler):
        """Auth forbidden on non-v1 /api/health/stores returns 403."""
        h = StorageHealthHandler(ctx={})
        mock_auth_ctx = MagicMock()
        with patch.object(
            h, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_ctx
        ), patch.object(
            h, "check_permission", side_effect=ForbiddenError("Denied")
        ):
            result = await h.handle("/api/health/stores", {}, http_handler)
        assert _status(result) == 403


# ===========================================================================
# TestDatabaseStoresHealthDelegate
# ===========================================================================


class TestDatabaseStoresHealthDelegate:
    """Tests for _database_stores_health() delegation."""

    def test_delegates_to_database_stores_health(self, handler):
        """_database_stores_health calls database.database_stores_health."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{"status":"ok"}')
        with patch(f"{_STORAGE_MOD}.database_stores_health", return_value=mock_result) as m:
            result = handler._database_stores_health()
        m.assert_called_once()
        assert result is mock_result

    def test_passes_self_as_handler(self, handler):
        """The handler itself is passed to the database function."""
        with patch(f"{_STORAGE_MOD}.database_stores_health") as m:
            handler._database_stores_health()
        # The argument is the handler cast to protocol type
        args = m.call_args[0]
        assert args[0] is handler

    def test_returns_handler_result(self, handler):
        """Return type is HandlerResult from the delegate."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{"stores":{}}')
        with patch(f"{_STORAGE_MOD}.database_stores_health", return_value=mock_result):
            result = handler._database_stores_health()
        assert isinstance(result, HandlerResult)


# ===========================================================================
# TestDatabaseSchemaHealthDelegate
# ===========================================================================


class TestDatabaseSchemaHealthDelegate:
    """Tests for _database_schema_health() delegation."""

    def test_delegates_to_database_schema_health(self, handler):
        """_database_schema_health calls database.database_schema_health."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{"status":"ok"}')
        with patch(f"{_STORAGE_MOD}.database_schema_health", return_value=mock_result) as m:
            result = handler._database_schema_health()
        m.assert_called_once()
        assert result is mock_result

    def test_passes_self_as_handler(self, handler):
        """The handler itself is passed to the database function."""
        with patch(f"{_STORAGE_MOD}.database_schema_health") as m:
            handler._database_schema_health()
        args = m.call_args[0]
        assert args[0] is handler

    def test_returns_handler_result(self, handler):
        """Return type is HandlerResult from the delegate."""
        mock_result = HandlerResult(status_code=503, content_type="application/json", body=b'{"status":"unavailable"}')
        with patch(f"{_STORAGE_MOD}.database_schema_health", return_value=mock_result):
            result = handler._database_schema_health()
        assert isinstance(result, HandlerResult)


# ===========================================================================
# TestGetStorage
# ===========================================================================


class TestGetStorage:
    """Tests for get_storage() context accessor."""

    def test_returns_storage_from_ctx(self, handler_with_ctx):
        """Returns storage instance from context."""
        result = handler_with_ctx.get_storage()
        assert result is not None
        assert result._mock_name == "mock_storage"

    def test_returns_none_when_not_in_ctx(self, handler):
        """Returns None when storage is not in context."""
        assert handler.get_storage() is None

    def test_returns_none_for_empty_ctx(self):
        """Returns None for handler with empty context."""
        h = StorageHealthHandler(ctx={})
        assert h.get_storage() is None


# ===========================================================================
# TestGetEloSystem
# ===========================================================================


class TestGetEloSystem:
    """Tests for get_elo_system() context accessor with class attr fallback."""

    def test_returns_elo_from_ctx(self, handler_with_ctx):
        """Returns ELO system from context."""
        result = handler_with_ctx.get_elo_system()
        assert result is not None
        assert result._mock_name == "mock_elo"

    def test_returns_none_when_not_in_ctx(self, handler):
        """Returns None when elo_system is not in context."""
        assert handler.get_elo_system() is None

    def test_class_attribute_takes_precedence(self):
        """Class-level elo_system attribute takes precedence over ctx."""
        mock_class_elo = MagicMock(name="class_elo")
        h = StorageHealthHandler(ctx={"elo_system": MagicMock(name="ctx_elo")})
        try:
            StorageHealthHandler.elo_system = mock_class_elo
            result = h.get_elo_system()
            assert result is mock_class_elo
        finally:
            del StorageHealthHandler.elo_system

    def test_class_attribute_none_falls_back_to_ctx(self):
        """When class elo_system is None, falls back to ctx."""
        mock_ctx_elo = MagicMock(name="ctx_elo")
        h = StorageHealthHandler(ctx={"elo_system": mock_ctx_elo})
        try:
            StorageHealthHandler.elo_system = None
            result = h.get_elo_system()
            assert result is mock_ctx_elo
        finally:
            del StorageHealthHandler.elo_system

    def test_no_class_attr_no_ctx_returns_none(self):
        """Returns None when neither class attr nor ctx have elo_system."""
        h = StorageHealthHandler(ctx={})
        assert h.get_elo_system() is None


# ===========================================================================
# TestGetNomicDir
# ===========================================================================


class TestGetNomicDir:
    """Tests for get_nomic_dir() context accessor."""

    def test_returns_nomic_dir_from_ctx(self, handler_with_ctx):
        """Returns nomic_dir from context."""
        assert handler_with_ctx.get_nomic_dir() == "/tmp/nomic"

    def test_returns_none_when_not_in_ctx(self, handler):
        """Returns None when nomic_dir is not in context."""
        assert handler.get_nomic_dir() is None


# ===========================================================================
# TestClassAttributes
# ===========================================================================


class TestClassAttributes:
    """Tests for ROUTES, HEALTH_PERMISSION, and RESOURCE_TYPE class attrs."""

    def test_routes_contains_three_paths(self):
        """ROUTES has exactly 3 paths."""
        assert len(StorageHealthHandler.ROUTES) == 3

    def test_routes_v1_stores(self):
        """ROUTES contains /api/v1/health/stores."""
        assert "/api/v1/health/stores" in StorageHealthHandler.ROUTES

    def test_routes_v1_database(self):
        """ROUTES contains /api/v1/health/database."""
        assert "/api/v1/health/database" in StorageHealthHandler.ROUTES

    def test_routes_non_v1_stores(self):
        """ROUTES contains /api/health/stores (backward compat)."""
        assert "/api/health/stores" in StorageHealthHandler.ROUTES

    def test_health_permission(self):
        """HEALTH_PERMISSION is system.health.read."""
        assert StorageHealthHandler.HEALTH_PERMISSION == "system.health.read"

    def test_resource_type(self):
        """RESOURCE_TYPE is health."""
        assert StorageHealthHandler.RESOURCE_TYPE == "health"


# ===========================================================================
# TestPathNormalization
# ===========================================================================


class TestPathNormalization:
    """Tests for v1/non-v1 path normalization in handle()."""

    @pytest.mark.asyncio
    async def test_v1_stores_normalized_to_api_stores(self, handler, http_handler):
        """v1 path /api/v1/health/stores normalizes to /api/health/stores."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{}')
        with patch.object(handler, "_database_stores_health", return_value=mock_result) as m:
            await handler.handle("/api/v1/health/stores", {}, http_handler)
        m.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_v1_stores_already_normalized(self, handler, http_handler):
        """Non-v1 /api/health/stores matches without needing normalization."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{}')
        with patch.object(handler, "_database_stores_health", return_value=mock_result) as m:
            await handler.handle("/api/health/stores", {}, http_handler)
        m.assert_called_once()

    @pytest.mark.asyncio
    async def test_v1_database_normalized_to_api_database(self, handler, http_handler):
        """v1 path /api/v1/health/database normalizes to /api/health/database."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{}')
        with patch.object(handler, "_database_schema_health", return_value=mock_result) as m:
            await handler.handle("/api/v1/health/database", {}, http_handler)
        m.assert_called_once()


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for StorageHealthHandler."""

    @pytest.mark.asyncio
    async def test_handle_empty_query_params(self, handler, http_handler):
        """Empty query params dict works fine."""
        mock_result = HandlerResult(status_code=200, content_type="application/json", body=b'{}')
        with patch.object(handler, "_database_stores_health", return_value=mock_result):
            result = await handler.handle("/api/v1/health/stores", {}, http_handler)
        assert result is not None

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_non_matching_after_auth(self, handler, http_handler):
        """Paths that pass auth but don't match any normalized route return None."""
        # /api/v1/health/other normalizes to /api/health/other which matches neither
        result = await handler.handle("/api/v1/health/other", {}, http_handler)
        assert result is None

    def test_handler_is_instance_of_secure_handler(self, handler):
        """StorageHealthHandler extends SecureHandler."""
        from aragora.server.handlers.secure import SecureHandler
        assert isinstance(handler, SecureHandler)

    def test_multiple_handlers_independent_ctx(self):
        """Multiple handler instances have independent contexts."""
        h1 = StorageHealthHandler(ctx={"storage": "one"})
        h2 = StorageHealthHandler(ctx={"storage": "two"})
        assert h1.ctx["storage"] != h2.ctx["storage"]

    def test_ctx_mutation_reflected(self):
        """Mutating ctx after init is reflected in accessors."""
        h = StorageHealthHandler(ctx={})
        assert h.get_storage() is None
        h.ctx["storage"] = MagicMock()
        assert h.get_storage() is not None

    @pytest.mark.asyncio
    async def test_handle_stores_and_database_independent(self, handler, http_handler):
        """Both routes can be called independently on the same handler."""
        stores_result = HandlerResult(status_code=200, content_type="application/json", body=b'{"type":"stores"}')
        db_result = HandlerResult(status_code=200, content_type="application/json", body=b'{"type":"database"}')
        with patch.object(handler, "_database_stores_health", return_value=stores_result):
            r1 = await handler.handle("/api/v1/health/stores", {}, http_handler)
        with patch.object(handler, "_database_schema_health", return_value=db_result):
            r2 = await handler.handle("/api/v1/health/database", {}, http_handler)
        assert _body(r1)["type"] == "stores"
        assert _body(r2)["type"] == "database"

    def test_all_in_dunder_all(self):
        """__all__ exports StorageHealthHandler."""
        from aragora.server.handlers.admin.health import storage_health
        assert "StorageHealthHandler" in storage_health.__all__
