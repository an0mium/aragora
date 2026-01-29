"""Tests for health implementation handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    import aragora.server.handlers.admin.health as health_mod

    health_mod._HEALTH_CACHE.clear()
    health_mod._HEALTH_CACHE_TIMESTAMPS.clear()
    yield


class TestHealthImplExports:
    """Tests for _health_impl module exports and backward compatibility."""

    def test_health_handler_exported(self):
        """Test HealthHandler is exported from _health_impl."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        assert HealthHandler is not None

    def test_cache_functions_exported(self):
        """Test cache functions are exported from _health_impl."""
        from aragora.server.handlers.admin._health_impl import (
            _get_cached_health,
            _set_cached_health,
        )

        assert _get_cached_health is not None
        assert _set_cached_health is not None

    def test_server_start_time_exported(self):
        """Test server start time is exported."""
        from aragora.server.handlers.admin._health_impl import _SERVER_START_TIME

        assert isinstance(_SERVER_START_TIME, float)
        assert _SERVER_START_TIME > 0

    def test_health_cache_exported(self):
        """Test health cache is exported."""
        from aragora.server.handlers.admin._health_impl import (
            _HEALTH_CACHE,
            _HEALTH_CACHE_TTL,
            _HEALTH_CACHE_TIMESTAMPS,
        )

        assert isinstance(_HEALTH_CACHE, dict)
        assert isinstance(_HEALTH_CACHE_TTL, float)
        assert isinstance(_HEALTH_CACHE_TIMESTAMPS, dict)


class TestHealthCacheFunctions:
    """Tests for health cache utility functions."""

    def test_set_and_get_cached_health(self):
        """Test setting and getting cached health."""
        from aragora.server.handlers.admin._health_impl import (
            _get_cached_health,
            _set_cached_health,
        )

        # Set a cached value
        test_data = {"status": "healthy", "checks": {}}
        _set_cached_health("test_key", test_data)

        # Get the cached value
        result = _get_cached_health("test_key")

        assert result == test_data

    def test_get_cached_health_returns_none_when_expired(self):
        """Test get_cached_health returns None when cache is expired."""
        import time
        from aragora.server.handlers.admin._health_impl import (
            _get_cached_health,
            _set_cached_health,
            _HEALTH_CACHE_TIMESTAMPS,
        )

        # Set a cached value with an old timestamp
        test_data = {"status": "healthy"}
        _set_cached_health("expired_key", test_data)
        _HEALTH_CACHE_TIMESTAMPS["expired_key"] = time.time() - 100  # 100 seconds ago

        # Get should return None (expired)
        result = _get_cached_health("expired_key")

        assert result is None

    def test_get_cached_health_returns_none_for_missing_key(self):
        """Test get_cached_health returns None for missing key."""
        from aragora.server.handlers.admin._health_impl import _get_cached_health

        result = _get_cached_health("nonexistent_key")

        assert result is None


class TestHealthHandlerRoutes:
    """Tests for HealthHandler route configuration."""

    def test_health_handler_routes_defined(self):
        """Test HealthHandler has expected routes."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        routes = HealthHandler.ROUTES

        assert "/healthz" in routes
        assert "/readyz" in routes
        assert "/api/health" in routes or "/api/v1/health" in routes

    def test_health_handler_public_routes_defined(self):
        """Test HealthHandler has public routes defined."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        public_routes = HealthHandler.PUBLIC_ROUTES

        assert "/healthz" in public_routes
        assert "/readyz" in public_routes

    def test_can_handle_health_routes(self):
        """Test can_handle returns True for health routes."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        handler = HealthHandler({"storage": None, "elo_system": None})

        assert handler.can_handle("/healthz") is True
        assert handler.can_handle("/readyz") is True
        assert handler.can_handle("/api/v1/health") is True

    def test_can_handle_non_health_routes(self):
        """Test can_handle returns False for non-health routes."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        handler = HealthHandler({"storage": None, "elo_system": None})

        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agents") is False


class TestHealthHandlerPermissions:
    """Tests for HealthHandler permission requirements."""

    def test_health_permission_defined(self):
        """Test health permission is defined."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        assert hasattr(HealthHandler, "HEALTH_PERMISSION")
        assert HealthHandler.HEALTH_PERMISSION == "system.health.read"

    def test_resource_type_defined(self):
        """Test resource type is defined."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        assert hasattr(HealthHandler, "RESOURCE_TYPE")
        assert HealthHandler.RESOURCE_TYPE == "health"


class TestHealthHandlerHandle:
    """Tests for HealthHandler.handle method."""

    @pytest.mark.asyncio
    async def test_handle_healthz_public(self):
        """Test /healthz is handled without auth."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        handler = HealthHandler({"storage": None, "elo_system": None})

        mock_http_handler = MagicMock()

        with patch.object(handler, "_liveness_probe") as mock_probe:
            mock_probe.return_value = MagicMock(
                status_code=200,
                body=b'{"status": "ok"}',
            )
            result = await handler.handle("/healthz", {}, mock_http_handler)

        assert result.status_code == 200
        mock_probe.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_protected_route_requires_auth(self):
        """Test protected routes require authentication."""
        from aragora.server.handlers.admin._health_impl import HealthHandler
        from aragora.server.handlers.secure import UnauthorizedError

        handler = HealthHandler({"storage": None, "elo_system": None})

        mock_http_handler = MagicMock()

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("Not authenticated")
            result = await handler.handle("/api/v1/health/detailed", {}, mock_http_handler)

        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_handle_protected_route_checks_permission(self):
        """Test protected routes check permission."""
        from aragora.server.handlers.admin._health_impl import HealthHandler
        from aragora.server.handlers.secure import ForbiddenError

        handler = HealthHandler({"storage": None, "elo_system": None})

        mock_http_handler = MagicMock()
        mock_auth_context = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.side_effect = ForbiddenError("Permission denied")
            result = await handler.handle("/api/v1/health/detailed", {}, mock_http_handler)

        assert result.status_code == 403


class TestHealthHandlerDelegation:
    """Tests for HealthHandler method delegation."""

    def test_liveness_probe_returns_result(self):
        """Test _liveness_probe returns a result."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        handler = HealthHandler({"storage": None, "elo_system": None})

        with patch("aragora.server.handlers.admin.health.liveness_probe") as mock_func:
            mock_func.return_value = MagicMock(status_code=200)
            result = handler._liveness_probe()

        assert result.status_code == 200

    def test_readiness_probe_fast_returns_result(self):
        """Test _readiness_probe_fast returns a result."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        handler = HealthHandler({"storage": None, "elo_system": None})

        with patch("aragora.server.handlers.admin.health.readiness_probe_fast") as mock_func:
            mock_func.return_value = MagicMock(status_code=200)
            result = handler._readiness_probe_fast()

        assert result.status_code == 200

    def test_health_check_returns_result(self):
        """Test _health_check returns a result."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        handler = HealthHandler({"storage": None, "elo_system": None})

        with patch("aragora.server.handlers.admin.health.health_check") as mock_func:
            mock_func.return_value = MagicMock(status_code=200)
            result = handler._health_check()

        assert result.status_code == 200

    def test_detailed_health_check_returns_result(self):
        """Test _detailed_health_check returns a result."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        handler = HealthHandler({"storage": None, "elo_system": None})

        with patch("aragora.server.handlers.admin.health.detailed_health_check") as mock_func:
            mock_func.return_value = MagicMock(status_code=200)
            result = handler._detailed_health_check()

        assert result.status_code == 200

    def test_deep_health_check_returns_result(self):
        """Test _deep_health_check returns a result."""
        from aragora.server.handlers.admin._health_impl import HealthHandler

        handler = HealthHandler({"storage": None, "elo_system": None})

        with patch("aragora.server.handlers.admin.health.deep_health_check") as mock_func:
            mock_func.return_value = MagicMock(status_code=200)
            result = handler._deep_health_check()

        assert result.status_code == 200
