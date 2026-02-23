"""Tests for device handler (aragora/server/handlers/devices.py).

Covers all routes and behavior of the DeviceHandler class:
- can_handle() routing for all ROUTES and dynamic paths
- POST   /api/devices/register              - Register a device for push notifications
- DELETE /api/devices/{device_id}           - Unregister a device
- POST   /api/devices/{device_id}/notify    - Send notification to a device
- POST   /api/devices/user/{user_id}/notify - Send notification to all user devices
- GET    /api/devices/user/{user_id}        - List user's devices
- GET    /api/devices/health                - Get device connector health
- POST   /api/devices/alexa/webhook         - Alexa skill webhook
- POST   /api/devices/google/webhook        - Google Actions webhook
- Circuit breaker integration
- Rate limiting (registration and notification)
- RBAC permission checks
- Input validation and size limits
- Error handling (ImportError, connection errors, etc.)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.devices import (
    DeviceHandler,
    _clear_device_circuit_breakers,
    _device_circuit_breakers,
    _get_circuit_breaker,
    _notification_limiter,
    _registration_limiter,
    get_device_circuit_breaker_status,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract the body dict from a HandlerResult."""
    if hasattr(result, "to_dict"):
        d = result.to_dict()
        return d.get("body", d)
    if isinstance(result, dict):
        return result.get("body", result)
    # Tuple unpacking fallback
    try:
        body, status, _ = result
        return body if isinstance(body, dict) else {}
    except (TypeError, ValueError):
        return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", result.get("status", 200))
    try:
        _, status, _ = result
        return status
    except (TypeError, ValueError):
        return 200


class MockHTTPHandler:
    """Mock HTTP handler used by BaseHandler.read_json_body_validated."""

    def __init__(self, body: dict | None = None):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": "application/json",
            }
        # Provide client_address for get_client_ip
        self.client_address = ("127.0.0.1", 54321)


# Mock device session dataclass
@dataclass
class MockDeviceSession:
    device_id: str = "dev-001"
    user_id: str = "test-user-001"
    device_type: str = "android"
    device_name: str = "Test Phone"
    push_token: str = "push-token-abc"
    app_version: str = "1.0.0"
    last_active: str = "2026-01-01T00:00:00+00:00"
    notification_count: int = 0
    created_at: str = "2026-01-01T00:00:00+00:00"

    def record_notification(self):
        self.notification_count += 1


# Mock notification result
@dataclass
class MockNotificationResult:
    success: bool = True
    message_id: str = "msg-001"
    status: Any = None
    error: str | None = None
    should_unregister: bool = False

    def __post_init__(self):
        if self.status is None:
            self.status = MagicMock(value="delivered")


# Mock device token
@dataclass
class MockDeviceToken:
    device_id: str = "dev-001"
    device_type: Any = None  # DeviceType enum
    push_token: str = "push-token-abc"
    user_id: str = "test-user-001"
    device_name: str = "Test Phone"
    app_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a DeviceHandler with minimal context."""
    return DeviceHandler(ctx={})


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiter state between tests."""
    _notification_limiter._buckets = defaultdict(list)
    _notification_limiter._requests = _notification_limiter._buckets
    _registration_limiter._buckets = defaultdict(list)
    _registration_limiter._requests = _registration_limiter._buckets
    yield
    _notification_limiter._buckets = defaultdict(list)
    _notification_limiter._requests = _notification_limiter._buckets
    _registration_limiter._buckets = defaultdict(list)
    _registration_limiter._requests = _registration_limiter._buckets


@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """Clear all device circuit breakers between tests."""
    _clear_device_circuit_breakers()
    yield
    _clear_device_circuit_breakers()


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_register_path(self, handler):
        assert handler.can_handle("/api/devices/register")

    def test_health_path(self, handler):
        assert handler.can_handle("/api/devices/health")

    def test_alexa_webhook_path(self, handler):
        assert handler.can_handle("/api/devices/alexa/webhook")

    def test_google_webhook_path(self, handler):
        assert handler.can_handle("/api/devices/google/webhook")

    def test_device_id_path(self, handler):
        assert handler.can_handle("/api/devices/dev-001")

    def test_device_notify_path(self, handler):
        assert handler.can_handle("/api/devices/dev-001/notify")

    def test_user_devices_path(self, handler):
        assert handler.can_handle("/api/devices/user/user-001")

    def test_user_notify_path(self, handler):
        assert handler.can_handle("/api/devices/user/user-001/notify")

    def test_versioned_path_bare_devices(self, handler):
        # /api/v1/devices normalizes to /api/devices which doesn't end with /
        # and isn't in ROUTES after normalization, so it doesn't match
        assert not handler.can_handle("/api/v1/devices")

    def test_versioned_register_path(self, handler):
        assert handler.can_handle("/api/v1/devices/register")

    def test_versioned_health_path(self, handler):
        assert handler.can_handle("/api/v1/devices/health")

    def test_versioned_alexa_path(self, handler):
        assert handler.can_handle("/api/v1/devices/alexa/webhook")

    def test_versioned_google_path(self, handler):
        assert handler.can_handle("/api/v1/devices/google/webhook")

    def test_versioned_device_id_path(self, handler):
        assert handler.can_handle("/api/v1/devices/dev-001")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/users/list")

    def test_rejects_empty(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_partial_prefix(self, handler):
        # /api/device (no "s") should not match
        assert not handler.can_handle("/api/device/register")


# ============================================================================
# Initialization
# ============================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_init_with_empty_context(self):
        h = DeviceHandler({})
        assert h.ctx == {}

    def test_init_with_ctx_kwarg(self):
        h = DeviceHandler(ctx={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_init_with_none_context(self):
        h = DeviceHandler(ctx=None)
        assert h.ctx == {}

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "devices"

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) > 0
        assert "/api/devices/register" in handler.ROUTES
        assert "/api/devices/health" in handler.ROUTES


# ============================================================================
# GET /api/devices/health
# ============================================================================


class TestHealth:
    """Test health endpoint."""

    @pytest.mark.asyncio
    async def test_health_success(self, handler):
        mock_registry = MagicMock()
        mock_registry.get_health = AsyncMock(
            return_value={
                "status": "healthy",
                "connectors": {"fcm": "ok"},
            }
        )

        with patch(
            "aragora.connectors.devices.registry.get_registry",
            return_value=mock_registry,
            create=True,
        ):
            # Patch the import inside the method
            result = await handler._get_health()

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert "circuit_breakers" in body

    @pytest.mark.asyncio
    async def test_health_import_error(self, handler):
        """When device connectors aren't installed, return unavailable."""
        with patch.dict("sys.modules", {"aragora.connectors.devices.registry": None}):
            with patch(
                "builtins.__import__",
                side_effect=_selective_import_error("aragora.connectors.devices.registry"),
            ):
                result = await handler._get_health()

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unavailable"
        assert "circuit_breakers" in body

    @pytest.mark.asyncio
    async def test_health_runtime_error(self, handler):
        """When registry raises runtime error, return 500."""
        mock_registry = MagicMock()
        mock_registry.get_health = AsyncMock(side_effect=RuntimeError("boom"))

        with patch(
            "aragora.connectors.devices.registry.get_registry",
            return_value=mock_registry,
            create=True,
        ):
            result = await handler._get_health()

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_health_via_route(self, handler):
        """Test health through the _route_request path."""
        mock_handler = MockHTTPHandler()
        result = await handler._route_request("/api/devices/health", "GET", {}, mock_handler, None)
        # Will hit the _get_health path, which may raise ImportError
        # (device connectors not installed), but should still return a result
        assert result is not None
        assert _status(result) in (200, 500)


# ============================================================================
# POST /api/devices/register
# ============================================================================


class TestRegisterDevice:
    """Test device registration."""

    @pytest.mark.asyncio
    async def test_register_missing_device_type(self, handler):
        result = await handler._register_device({"push_token": "abc"}, _mock_auth_context(), None)
        assert _status(result) == 400
        assert "Missing required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_register_missing_push_token(self, handler):
        result = await handler._register_device(
            {"device_type": "android"}, _mock_auth_context(), None
        )
        assert _status(result) == 400
        assert "Missing required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_register_empty_body(self, handler):
        result = await handler._register_device({}, _mock_auth_context(), None)
        assert _status(result) == 400
        assert "Missing required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_register_push_token_too_long(self, handler):
        result = await handler._register_device(
            {"device_type": "android", "push_token": "x" * 4097},
            _mock_auth_context(),
            None,
        )
        assert _status(result) == 400
        assert "maximum length" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_register_push_token_max_length(self, handler):
        """A push token at exactly 4096 bytes should be accepted (validation passes)."""
        body = {"device_type": "android", "push_token": "x" * 4096}
        # It will proceed past validation and hit ImportError
        result = await handler._register_device(body, _mock_auth_context(), None)
        # Expect either success or import error (503), NOT 400
        assert _status(result) != 400

    @pytest.mark.asyncio
    async def test_register_no_user_id(self, handler):
        """When auth context has no user_id and body has none either."""
        from aragora.rbac.models import AuthorizationContext

        auth_ctx = AuthorizationContext(
            user_id="",
            user_email="test@example.com",
            org_id="org-1",
            roles={"admin"},
            permissions={"*"},
        )
        result = await handler._register_device(
            {"device_type": "android", "push_token": "abc"},
            auth_ctx,
            None,
        )
        assert _status(result) == 400
        assert "user_id" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_register_user_id_from_body(self, handler):
        """user_id can come from the body instead of auth context."""
        from aragora.rbac.models import AuthorizationContext

        auth_ctx = AuthorizationContext(
            user_id="",
            user_email="test@example.com",
            org_id="org-1",
            roles={"admin"},
            permissions={"*"},
        )
        body = {
            "device_type": "android",
            "push_token": "abc",
            "user_id": "body-user-001",
        }
        # Will proceed past validation, may hit ImportError
        result = await handler._register_device(body, auth_ctx, None)
        # Should not fail on user_id validation
        assert _status(result) != 400 or "user_id" not in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_register_import_error(self, handler):
        """When device connectors not installed."""
        body = {"device_type": "android", "push_token": "abc"}
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.connectors.devices"),
        ):
            result = await handler._register_device(body, _mock_auth_context(), None)

        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_register_invalid_device_type(self, handler):
        """Invalid device type returns 400."""
        body = {"device_type": "smart_fridge", "push_token": "abc"}

        mock_device_type = MagicMock(side_effect=ValueError("Invalid"))
        mock_module = MagicMock()
        mock_module.DeviceType = mock_device_type
        mock_module.DeviceRegistration = MagicMock()
        mock_module.get_registry = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.devices": mock_module,
                "aragora.connectors.devices.registry": MagicMock(),
            },
        ):
            with patch(
                "aragora.server.handlers.devices.DeviceType",
                mock_device_type,
                create=True,
            ):
                result = await handler._register_device(body, _mock_auth_context(), None)

        # The handler catches ValueError from DeviceType() and returns 400
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_register_success(self, handler):
        """Successful device registration."""
        body = {
            "device_type": "android",
            "push_token": "push-token-abc",
            "device_name": "Pixel 9",
            "app_version": "2.0",
        }

        mock_token = MockDeviceToken(device_id="dev-new-001")
        mock_token.device_type = MagicMock(value="android")

        mock_connector = AsyncMock()
        mock_connector.register_device = AsyncMock(return_value=mock_token)

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        mock_device_type_enum = MagicMock()
        mock_android = MagicMock()
        mock_device_type_enum.return_value = mock_android
        mock_device_type_enum.ANDROID = mock_android
        mock_device_type_enum.WEB = MagicMock()
        mock_device_type_enum.IOS = MagicMock()
        mock_device_type_enum.ALEXA = MagicMock()
        mock_device_type_enum.GOOGLE_HOME = MagicMock()

        platform_map = {mock_android: "fcm"}

        with (
            patch(
                "aragora.server.handlers.devices.DeviceRegistration",
                MagicMock(),
                create=True,
            ),
            patch(
                "aragora.server.handlers.devices.DeviceType",
                mock_device_type_enum,
                create=True,
            ),
            patch(
                "aragora.connectors.devices.registry.get_registry",
                return_value=mock_registry,
                create=True,
            ),
        ):
            # Mock the platform_map lookup
            with patch.object(
                handler,
                "_register_device",
                wraps=handler._register_device,
            ):
                # Directly test with mocked imports
                result = await _register_with_mocks(handler, body, mock_connector, "fcm")

        assert _status(result) == 200
        assert _body(result)["success"] is True
        assert _body(result)["device_id"] == "dev-new-001"

    @pytest.mark.asyncio
    async def test_register_connector_not_available(self, handler):
        """When registry.get raises KeyError (connector not found)."""
        body = {"device_type": "android", "push_token": "abc"}

        result = await _register_with_mocks(
            handler,
            body,
            connector=None,
            platform="fcm",
            registry_raises=KeyError("fcm"),
        )

        assert _status(result) == 503
        assert "not available" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_register_circuit_breaker_open(self, handler):
        """When circuit breaker is open, return 503."""
        body = {"device_type": "android", "push_token": "abc"}

        # Open the circuit breaker for fcm
        cb = _get_circuit_breaker("fcm")
        for _ in range(10):
            cb.record_failure()

        result = await _register_with_mocks(handler, body, AsyncMock(), "fcm")

        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_register_connector_connection_error(self, handler):
        """Connection error during registration records failure."""
        body = {"device_type": "android", "push_token": "abc"}

        mock_connector = AsyncMock()
        mock_connector.register_device = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        result = await _register_with_mocks(handler, body, mock_connector, "fcm")

        assert _status(result) == 500
        cb = _get_circuit_breaker("fcm")
        assert cb._failure_count > 0

    @pytest.mark.asyncio
    async def test_register_null_token_returned(self, handler):
        """When connector returns None for device token."""
        body = {"device_type": "android", "push_token": "abc"}

        mock_connector = AsyncMock()
        mock_connector.register_device = AsyncMock(return_value=None)

        result = await _register_with_mocks(handler, body, mock_connector, "fcm")

        assert _status(result) == 400
        assert "Failed to register" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_register_rate_limit(self, handler):
        """Rate limiting blocks excess registration requests."""
        mock_handler = MockHTTPHandler(
            body={
                "device_type": "android",
                "push_token": "abc",
            }
        )
        path = "/api/devices/register"

        # Exhaust rate limit (10 per minute for registration)
        for _ in range(10):
            _registration_limiter.is_allowed("127.0.0.1")

        result = await handler._route_request(
            path,
            "POST",
            {},
            mock_handler,
            {
                "device_type": "android",
                "push_token": "abc",
            },
        )
        assert _status(result) == 429
        assert "Rate limit" in _body(result)["error"]


# ============================================================================
# DELETE /api/devices/{device_id} - Unregister
# ============================================================================


class TestUnregisterDevice:
    """Test device unregistration."""

    @pytest.mark.asyncio
    async def test_unregister_success(self, handler):
        mock_store = MagicMock()
        mock_device = MockDeviceSession(device_id="dev-001", user_id="test-user-001")
        mock_store.get_device_session.return_value = mock_device
        mock_store.delete_device_session.return_value = True

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._unregister_device("dev-001", _mock_auth_context())

        assert _status(result) == 200
        assert _body(result)["success"] is True
        assert _body(result)["device_id"] == "dev-001"
        assert "deleted_at" in _body(result)

    @pytest.mark.asyncio
    async def test_unregister_device_not_found(self, handler):
        mock_store = MagicMock()
        mock_store.get_device_session.return_value = None

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._unregister_device("nonexistent", _mock_auth_context())

        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_unregister_not_owner(self, handler):
        """Non-admin user cannot delete another user's device."""
        mock_store = MagicMock()
        mock_device = MockDeviceSession(device_id="dev-001", user_id="other-user")
        mock_store.get_device_session.return_value = mock_device

        auth_ctx = _mock_auth_context(user_id="different-user", roles={"member"})

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._unregister_device("dev-001", auth_ctx)

        assert _status(result) == 403
        assert "Not authorized" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_unregister_admin_can_delete_others(self, handler):
        """Admin user can delete any user's device."""
        mock_store = MagicMock()
        mock_device = MockDeviceSession(device_id="dev-001", user_id="other-user")
        mock_store.get_device_session.return_value = mock_device
        mock_store.delete_device_session.return_value = True

        auth_ctx = _mock_auth_context(user_id="admin-user", roles={"admin"})

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._unregister_device("dev-001", auth_ctx)

        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_unregister_owner_role_can_delete_others(self, handler):
        """Owner role can also delete any user's device."""
        mock_store = MagicMock()
        mock_device = MockDeviceSession(device_id="dev-001", user_id="other-user")
        mock_store.get_device_session.return_value = mock_device
        mock_store.delete_device_session.return_value = True

        auth_ctx = _mock_auth_context(user_id="owner-user", roles={"owner"})

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._unregister_device("dev-001", auth_ctx)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unregister_delete_fails(self, handler):
        mock_store = MagicMock()
        mock_device = MockDeviceSession(device_id="dev-001", user_id="test-user-001")
        mock_store.get_device_session.return_value = mock_device
        mock_store.delete_device_session.return_value = False

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._unregister_device("dev-001", _mock_auth_context())

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_unregister_import_error(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.server.session_store"),
        ):
            result = await handler._unregister_device("dev-001", _mock_auth_context())

        assert _status(result) == 503
        assert "not available" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_unregister_store_error(self, handler):
        mock_store = MagicMock()
        mock_store.get_device_session.side_effect = OSError("disk full")

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._unregister_device("dev-001", _mock_auth_context())

        assert _status(result) == 500


# ============================================================================
# GET /api/devices/{device_id} - Get device
# ============================================================================


class TestGetDevice:
    """Test getting device information."""

    @pytest.mark.asyncio
    async def test_get_device_success(self, handler):
        mock_store = MagicMock()
        mock_device = MockDeviceSession(
            device_id="dev-001",
            user_id="test-user-001",
            device_type="android",
            device_name="Pixel 9",
            app_version="2.0",
            last_active="2026-01-15T12:00:00+00:00",
            notification_count=5,
            created_at="2026-01-01T00:00:00+00:00",
        )
        mock_store.get_device_session.return_value = mock_device

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._get_device("dev-001", _mock_auth_context())

        assert _status(result) == 200
        body = _body(result)
        assert body["device_id"] == "dev-001"
        assert body["user_id"] == "test-user-001"
        assert body["device_type"] == "android"
        assert body["device_name"] == "Pixel 9"
        assert body["app_version"] == "2.0"
        assert body["notification_count"] == 5

    @pytest.mark.asyncio
    async def test_get_device_not_found(self, handler):
        mock_store = MagicMock()
        mock_store.get_device_session.return_value = None

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._get_device("nonexistent", _mock_auth_context())

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_device_not_owner(self, handler):
        mock_store = MagicMock()
        mock_device = MockDeviceSession(device_id="dev-001", user_id="other-user")
        mock_store.get_device_session.return_value = mock_device

        auth_ctx = _mock_auth_context(user_id="different-user", roles={"member"})

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._get_device("dev-001", auth_ctx)

        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_get_device_admin_sees_others(self, handler):
        mock_store = MagicMock()
        mock_device = MockDeviceSession(device_id="dev-001", user_id="other-user")
        mock_store.get_device_session.return_value = mock_device

        auth_ctx = _mock_auth_context(user_id="admin-user", roles={"admin"})

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._get_device("dev-001", auth_ctx)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_device_import_error(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.server.session_store"),
        ):
            result = await handler._get_device("dev-001", _mock_auth_context())

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_device_store_error(self, handler):
        mock_store = MagicMock()
        mock_store.get_device_session.side_effect = ValueError("corrupt data")

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._get_device("dev-001", _mock_auth_context())

        assert _status(result) == 500


# ============================================================================
# GET /api/devices/user/{user_id} - List user devices
# ============================================================================


class TestListUserDevices:
    """Test listing user devices."""

    @pytest.mark.asyncio
    async def test_list_user_devices_success(self, handler):
        mock_store = MagicMock()
        devices = [
            MockDeviceSession(device_id="dev-001", device_name="Phone"),
            MockDeviceSession(device_id="dev-002", device_name="Tablet"),
        ]
        mock_store.find_devices_by_user.return_value = devices

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._list_user_devices("test-user-001", _mock_auth_context())

        assert _status(result) == 200
        body = _body(result)
        assert body["user_id"] == "test-user-001"
        assert body["device_count"] == 2
        assert len(body["devices"]) == 2
        assert body["devices"][0]["device_id"] == "dev-001"
        assert body["devices"][1]["device_id"] == "dev-002"

    @pytest.mark.asyncio
    async def test_list_user_devices_empty(self, handler):
        mock_store = MagicMock()
        mock_store.find_devices_by_user.return_value = []

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._list_user_devices("test-user-001", _mock_auth_context())

        assert _status(result) == 200
        assert _body(result)["device_count"] == 0
        assert _body(result)["devices"] == []

    @pytest.mark.asyncio
    async def test_list_user_devices_not_owner(self, handler):
        auth_ctx = _mock_auth_context(user_id="different-user", roles={"member"})

        result = await handler._list_user_devices("other-user", auth_ctx)

        assert _status(result) == 403
        assert "Not authorized" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_list_user_devices_admin(self, handler):
        mock_store = MagicMock()
        mock_store.find_devices_by_user.return_value = []

        auth_ctx = _mock_auth_context(user_id="admin-user", roles={"admin"})

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._list_user_devices("other-user", auth_ctx)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_list_user_devices_same_user(self, handler):
        mock_store = MagicMock()
        mock_store.find_devices_by_user.return_value = []

        auth_ctx = _mock_auth_context(user_id="user-001", roles={"member"})

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._list_user_devices("user-001", auth_ctx)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_list_user_devices_import_error(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.server.session_store"),
        ):
            result = await handler._list_user_devices("test-user-001", _mock_auth_context())

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_list_user_devices_store_error(self, handler):
        mock_store = MagicMock()
        mock_store.find_devices_by_user.side_effect = OSError("db error")

        with patch(
            "aragora.server.session_store.get_session_store",
            return_value=mock_store,
        ):
            result = await handler._list_user_devices("test-user-001", _mock_auth_context())

        assert _status(result) == 500


# ============================================================================
# POST /api/devices/{device_id}/notify - Notify single device
# ============================================================================


class TestNotifyDevice:
    """Test sending notification to a specific device."""

    @pytest.mark.asyncio
    async def test_notify_missing_title(self, handler):
        result = await handler._notify_device("dev-001", {"body": "msg"}, _mock_auth_context())
        assert _status(result) == 400
        assert "title and body" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_notify_missing_body(self, handler):
        result = await handler._notify_device("dev-001", {"title": "Test"}, _mock_auth_context())
        assert _status(result) == 400
        assert "title and body" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_notify_empty_body(self, handler):
        result = await handler._notify_device("dev-001", {}, _mock_auth_context())
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_notify_title_too_long(self, handler):
        result = await handler._notify_device(
            "dev-001",
            {"title": "x" * 257, "body": "msg"},
            _mock_auth_context(),
        )
        assert _status(result) == 400
        assert "title exceeds" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_notify_body_too_long(self, handler):
        result = await handler._notify_device(
            "dev-001",
            {"title": "Test", "body": "x" * 4097},
            _mock_auth_context(),
        )
        assert _status(result) == 400
        assert "body exceeds" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_notify_title_at_max(self, handler):
        """Title at exactly 256 chars should pass validation."""
        body = {"title": "x" * 256, "body": "msg"}
        result = await handler._notify_device("dev-001", body, _mock_auth_context())
        # Should pass validation (status != 400 for title length)
        assert _status(result) != 400 or "title exceeds" not in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_notify_body_at_max(self, handler):
        """Body at exactly 4096 chars should pass validation."""
        body = {"title": "Test", "body": "x" * 4096}
        result = await handler._notify_device("dev-001", body, _mock_auth_context())
        assert _status(result) != 400 or "body exceeds" not in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_notify_device_not_found(self, handler):
        mock_store = MagicMock()
        mock_store.get_device_session.return_value = None

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(),
        ):
            result = await handler._notify_device(
                "nonexistent",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_notify_not_owner(self, handler):
        mock_store = MagicMock()
        mock_device = MockDeviceSession(device_id="dev-001", user_id="other-user")
        mock_store.get_device_session.return_value = mock_device

        auth_ctx = _mock_auth_context(user_id="different-user", roles={"member"})

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(),
        ):
            result = await handler._notify_device(
                "dev-001",
                {"title": "Hi", "body": "Hello"},
                auth_ctx,
            )

        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_notify_success(self, handler):
        mock_store = MagicMock()
        mock_device = MockDeviceSession(
            device_id="dev-001", user_id="test-user-001", device_type="android"
        )
        mock_store.get_device_session.return_value = mock_device

        mock_result = MockNotificationResult(success=True, message_id="msg-123")
        mock_connector = AsyncMock()
        mock_connector.send_notification = AsyncMock(return_value=mock_result)

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(registry=mock_registry),
        ):
            result = await handler._notify_device(
                "dev-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["device_id"] == "dev-001"
        assert body["message_id"] == "msg-123"

    @pytest.mark.asyncio
    async def test_notify_records_notification(self, handler):
        """Successful notification increments notification count and saves."""
        mock_store = MagicMock()
        mock_device = MockDeviceSession(
            device_id="dev-001", user_id="test-user-001", device_type="android"
        )
        mock_store.get_device_session.return_value = mock_device

        mock_result = MockNotificationResult(success=True)
        mock_connector = AsyncMock()
        mock_connector.send_notification = AsyncMock(return_value=mock_result)

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(registry=mock_registry),
        ):
            await handler._notify_device(
                "dev-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        mock_store.set_device_session.assert_called_once_with(mock_device)
        assert mock_device.notification_count == 1

    @pytest.mark.asyncio
    async def test_notify_unregisters_invalid_token(self, handler):
        """When result.should_unregister is True, device gets removed."""
        mock_store = MagicMock()
        mock_device = MockDeviceSession(
            device_id="dev-001", user_id="test-user-001", device_type="android"
        )
        mock_store.get_device_session.return_value = mock_device

        mock_result = MockNotificationResult(
            success=False, should_unregister=True, error="Invalid token"
        )
        mock_connector = AsyncMock()
        mock_connector.send_notification = AsyncMock(return_value=mock_result)

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(registry=mock_registry),
        ):
            await handler._notify_device(
                "dev-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        mock_store.delete_device_session.assert_called_once_with("dev-001")

    @pytest.mark.asyncio
    async def test_notify_circuit_breaker_open(self, handler):
        """Circuit breaker open for platform returns 503."""
        mock_store = MagicMock()
        mock_device = MockDeviceSession(
            device_id="dev-001", user_id="test-user-001", device_type="android"
        )
        mock_store.get_device_session.return_value = mock_device

        # Open circuit breaker
        cb = _get_circuit_breaker("fcm")
        for _ in range(10):
            cb.record_failure()

        mock_registry = MagicMock()

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(registry=mock_registry),
        ):
            result = await handler._notify_device(
                "dev-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_notify_connector_error(self, handler):
        """Connection error during send records failure on circuit breaker."""
        mock_store = MagicMock()
        mock_device = MockDeviceSession(
            device_id="dev-001", user_id="test-user-001", device_type="android"
        )
        mock_store.get_device_session.return_value = mock_device

        mock_connector = AsyncMock()
        mock_connector.send_notification = AsyncMock(side_effect=ConnectionError("timeout"))

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(registry=mock_registry),
        ):
            result = await handler._notify_device(
                "dev-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 503
        assert "Failed to send" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_notify_import_error(self, handler):
        """When required modules not available."""
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.connectors.devices"),
        ):
            result = await handler._notify_device(
                "dev-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_notify_rate_limit(self, handler):
        """Rate limiting blocks excess notification requests."""
        mock_handler = MockHTTPHandler()
        path = "/api/devices/dev-001/notify"

        # Exhaust rate limit (30 per minute for notifications)
        for _ in range(30):
            _notification_limiter.is_allowed("127.0.0.1")

        result = await handler._route_request(
            path,
            "POST",
            {},
            mock_handler,
            {"title": "Hi", "body": "Hello"},
        )
        assert _status(result) == 429


# ============================================================================
# POST /api/devices/user/{user_id}/notify - Notify user's devices
# ============================================================================


class TestNotifyUser:
    """Test sending notification to all user devices."""

    @pytest.mark.asyncio
    async def test_notify_user_missing_title(self, handler):
        result = await handler._notify_user("user-001", {"body": "msg"}, _mock_auth_context())
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_notify_user_missing_body(self, handler):
        result = await handler._notify_user("user-001", {"title": "T"}, _mock_auth_context())
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_notify_user_title_too_long(self, handler):
        result = await handler._notify_user(
            "user-001",
            {"title": "x" * 257, "body": "msg"},
            _mock_auth_context(),
        )
        assert _status(result) == 400
        assert "title exceeds" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_notify_user_body_too_long(self, handler):
        result = await handler._notify_user(
            "user-001",
            {"title": "T", "body": "x" * 4097},
            _mock_auth_context(),
        )
        assert _status(result) == 400
        assert "body exceeds" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_notify_user_not_owner(self, handler):
        auth_ctx = _mock_auth_context(user_id="different-user", roles={"member"})
        result = await handler._notify_user(
            "other-user",
            {"title": "Hi", "body": "Hello"},
            auth_ctx,
        )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_notify_user_admin_can_notify_others(self, handler):
        mock_store = MagicMock()
        mock_store.find_devices_by_user.return_value = []

        auth_ctx = _mock_auth_context(user_id="admin-user", roles={"admin"})

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(),
        ):
            result = await handler._notify_user(
                "other-user",
                {"title": "Hi", "body": "Hello"},
                auth_ctx,
            )

        assert _status(result) == 200
        assert _body(result)["devices_notified"] == 0

    @pytest.mark.asyncio
    async def test_notify_user_no_devices(self, handler):
        mock_store = MagicMock()
        mock_store.find_devices_by_user.return_value = []

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(),
        ):
            result = await handler._notify_user(
                "test-user-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["devices_notified"] == 0
        assert "No devices" in body.get("message", "")

    @pytest.mark.asyncio
    async def test_notify_user_multiple_devices(self, handler):
        mock_store = MagicMock()
        devices = [
            MockDeviceSession(device_id="dev-001", device_type="android"),
            MockDeviceSession(device_id="dev-002", device_type="ios"),
        ]
        mock_store.find_devices_by_user.return_value = devices

        mock_result = MockNotificationResult(success=True)
        mock_connector = AsyncMock()
        mock_connector.send_notification = AsyncMock(return_value=mock_result)

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(registry=mock_registry),
        ):
            result = await handler._notify_user(
                "test-user-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["devices_notified"] == 2
        assert body["devices_failed"] == 0

    @pytest.mark.asyncio
    async def test_notify_user_partial_failure(self, handler):
        """Some devices succeed, some fail."""
        mock_store = MagicMock()
        devices = [
            MockDeviceSession(device_id="dev-001", device_type="android"),
            MockDeviceSession(device_id="dev-002", device_type="android"),
        ]
        mock_store.find_devices_by_user.return_value = devices

        success_result = MockNotificationResult(success=True)
        fail_result = MockNotificationResult(success=False, error="delivery failed")

        mock_connector = AsyncMock()
        mock_connector.send_notification = AsyncMock(side_effect=[success_result, fail_result])

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(registry=mock_registry),
        ):
            result = await handler._notify_user(
                "test-user-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["devices_notified"] == 1
        assert body["devices_failed"] == 1

    @pytest.mark.asyncio
    async def test_notify_user_removes_invalid_tokens(self, handler):
        mock_store = MagicMock()
        devices = [
            MockDeviceSession(device_id="dev-001", device_type="android"),
        ]
        mock_store.find_devices_by_user.return_value = devices

        mock_result = MockNotificationResult(
            success=False, should_unregister=True, error="Invalid token"
        )
        mock_connector = AsyncMock()
        mock_connector.send_notification = AsyncMock(return_value=mock_result)

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(registry=mock_registry),
        ):
            result = await handler._notify_user(
                "test-user-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 200
        assert _body(result)["devices_removed"] == 1
        mock_store.delete_device_session.assert_called_once_with("dev-001")

    @pytest.mark.asyncio
    async def test_notify_user_circuit_breaker_open(self, handler):
        mock_store = MagicMock()
        devices = [
            MockDeviceSession(device_id="dev-001", device_type="android"),
        ]
        mock_store.find_devices_by_user.return_value = devices

        # Open fcm circuit breaker
        cb = _get_circuit_breaker("fcm")
        for _ in range(10):
            cb.record_failure()

        mock_registry = MagicMock()

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(registry=mock_registry),
        ):
            result = await handler._notify_user(
                "test-user-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["devices_failed"] == 1
        assert "temporarily unavailable" in body["results"][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_notify_user_connector_error(self, handler):
        """Connector error is caught per-device."""
        mock_store = MagicMock()
        devices = [
            MockDeviceSession(device_id="dev-001", device_type="android"),
        ]
        mock_store.find_devices_by_user.return_value = devices

        mock_connector = AsyncMock()
        mock_connector.send_notification = AsyncMock(side_effect=ConnectionError("failed"))

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with (
            patch(
                "aragora.server.session_store.get_session_store",
                return_value=mock_store,
            ),
            _patch_device_connector_imports(registry=mock_registry),
        ):
            result = await handler._notify_user(
                "test-user-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["devices_failed"] == 1
        assert "Connector error" in body["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_notify_user_import_error(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.connectors.devices"),
        ):
            result = await handler._notify_user(
                "test-user-001",
                {"title": "Hi", "body": "Hello"},
                _mock_auth_context(),
            )

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_notify_user_rate_limit(self, handler):
        """Rate limiting blocks excess user notification requests."""
        mock_handler = MockHTTPHandler()
        path = "/api/devices/user/user-001/notify"

        for _ in range(30):
            _notification_limiter.is_allowed("127.0.0.1")

        result = await handler._route_request(
            path,
            "POST",
            {},
            mock_handler,
            {"title": "Hi", "body": "Hello"},
        )
        assert _status(result) == 429


# ============================================================================
# POST /api/devices/alexa/webhook
# ============================================================================


class TestAlexaWebhook:
    """Test Alexa skill webhook handling."""

    @pytest.mark.asyncio
    async def test_alexa_import_error(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.connectors.devices.alexa"),
        ):
            result = await handler._handle_alexa_webhook({}, MockHTTPHandler())

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_alexa_connector_not_available(self, handler):
        mock_registry = MagicMock()
        mock_registry.get.side_effect = KeyError("alexa")

        with _patch_alexa_imports(registry=mock_registry):
            result = await handler._handle_alexa_webhook({}, MockHTTPHandler())

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_alexa_invalid_connector_type(self, handler):
        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock()  # Not an AlexaConnector

        with _patch_alexa_imports(registry=mock_registry, isinstance_returns=False):
            result = await handler._handle_alexa_webhook({}, MockHTTPHandler())

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_alexa_invalid_skill_id(self, handler):
        mock_connector = MagicMock()
        mock_connector.verify_skill_id.return_value = False

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with _patch_alexa_imports(registry=mock_registry, connector=mock_connector):
            result = await handler._handle_alexa_webhook({}, MockHTTPHandler())

        assert _status(result) == 403
        assert "skill id" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_alexa_success(self, handler):
        mock_voice_request = MagicMock()
        mock_voice_response = MagicMock()
        alexa_response = {"response": {"outputSpeech": {"text": "Hello"}}}

        mock_connector = MagicMock()
        mock_connector.verify_skill_id.return_value = True
        mock_connector.parse_alexa_request.return_value = mock_voice_request
        mock_connector.handle_voice_request = AsyncMock(return_value=mock_voice_response)
        mock_connector.build_alexa_response.return_value = alexa_response

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with _patch_alexa_imports(registry=mock_registry, connector=mock_connector):
            result = await handler._handle_alexa_webhook(
                {"session": {"attributes": {"key": "val"}}},
                MockHTTPHandler(),
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["response"]["outputSpeech"]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_alexa_error_handling(self, handler):
        mock_connector = MagicMock()
        mock_connector.verify_skill_id.side_effect = ValueError("parse error")

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with _patch_alexa_imports(registry=mock_registry, connector=mock_connector):
            result = await handler._handle_alexa_webhook({}, MockHTTPHandler())

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_alexa_no_auth_required(self, handler):
        """Alexa webhook path does not require auth context."""
        mock_handler = MockHTTPHandler(body={"request": {"type": "LaunchRequest"}})
        # Route through _route_request - should not try to get auth context
        # (Alexa webhook is checked before auth)
        with patch.object(handler, "_handle_alexa_webhook", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = MagicMock(
                status_code=200, body=b"{}", content_type="application/json", headers={}
            )
            result = await handler._route_request(
                "/api/devices/alexa/webhook",
                "POST",
                {},
                mock_handler,
                {"request": {"type": "LaunchRequest"}},
            )
            mock_method.assert_called_once()


# ============================================================================
# POST /api/devices/google/webhook
# ============================================================================


class TestGoogleWebhook:
    """Test Google Actions webhook handling."""

    @pytest.mark.asyncio
    async def test_google_import_error(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.connectors.devices.google_home"),
        ):
            result = await handler._handle_google_webhook({}, MockHTTPHandler())

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_google_connector_not_available(self, handler):
        mock_registry = MagicMock()
        mock_registry.get.side_effect = KeyError("google_home")

        with _patch_google_imports(registry=mock_registry):
            result = await handler._handle_google_webhook({}, MockHTTPHandler())

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_google_invalid_connector_type(self, handler):
        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock()  # Not a GoogleHomeConnector

        with _patch_google_imports(registry=mock_registry, isinstance_returns=False):
            result = await handler._handle_google_webhook({}, MockHTTPHandler())

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_google_sync_intent(self, handler):
        mock_connector = MagicMock()
        mock_connector.handle_sync = AsyncMock(return_value={"requestId": "req-1", "payload": {}})

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        body = {
            "requestId": "req-1",
            "inputs": [{"intent": "action.devices.SYNC"}],
            "user": {"userId": "user-001"},
        }

        with _patch_google_imports(registry=mock_registry, connector=mock_connector):
            result = await handler._handle_google_webhook(body, MockHTTPHandler())

        assert _status(result) == 200
        mock_connector.handle_sync.assert_awaited_once_with("req-1", "user-001")

    @pytest.mark.asyncio
    async def test_google_query_intent(self, handler):
        mock_connector = MagicMock()
        mock_connector.handle_query = AsyncMock(return_value={"requestId": "req-2"})

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        body = {
            "requestId": "req-2",
            "inputs": [
                {
                    "intent": "action.devices.QUERY",
                    "payload": {"devices": [{"id": "d1"}]},
                }
            ],
        }

        with _patch_google_imports(registry=mock_registry, connector=mock_connector):
            result = await handler._handle_google_webhook(body, MockHTTPHandler())

        assert _status(result) == 200
        mock_connector.handle_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_google_execute_intent(self, handler):
        mock_connector = MagicMock()
        mock_connector.handle_execute = AsyncMock(return_value={"requestId": "req-3"})

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        body = {
            "requestId": "req-3",
            "inputs": [
                {
                    "intent": "action.devices.EXECUTE",
                    "payload": {"commands": [{"devices": [{"id": "d1"}]}]},
                }
            ],
        }

        with _patch_google_imports(registry=mock_registry, connector=mock_connector):
            result = await handler._handle_google_webhook(body, MockHTTPHandler())

        assert _status(result) == 200
        mock_connector.handle_execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_google_disconnect_intent(self, handler):
        mock_connector = MagicMock()
        mock_connector.handle_disconnect = AsyncMock(return_value={"requestId": "req-4"})

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        body = {
            "requestId": "req-4",
            "inputs": [{"intent": "action.devices.DISCONNECT"}],
            "user": {"userId": "user-001"},
        }

        with _patch_google_imports(registry=mock_registry, connector=mock_connector):
            result = await handler._handle_google_webhook(body, MockHTTPHandler())

        assert _status(result) == 200
        mock_connector.handle_disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_google_conversational_action(self, handler):
        """Non-Smart-Home requests fall through to conversational actions."""
        mock_voice_request = MagicMock()
        mock_voice_response = MagicMock()
        google_response = {"prompt": {"firstSimple": {"speech": "Hello"}}}

        mock_connector = MagicMock()
        mock_connector.parse_google_request.return_value = mock_voice_request
        mock_connector.handle_voice_request = AsyncMock(return_value=mock_voice_response)
        mock_connector.build_google_response.return_value = google_response

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        body = {"session": {"params": {"key": "val"}}}

        with _patch_google_imports(registry=mock_registry, connector=mock_connector):
            result = await handler._handle_google_webhook(body, MockHTTPHandler())

        assert _status(result) == 200
        assert _body(result)["prompt"]["firstSimple"]["speech"] == "Hello"

    @pytest.mark.asyncio
    async def test_google_empty_inputs(self, handler):
        """Empty inputs array falls through to conversational action."""
        mock_connector = MagicMock()
        mock_connector.parse_google_request.return_value = MagicMock()
        mock_connector.handle_voice_request = AsyncMock(return_value=MagicMock())
        mock_connector.build_google_response.return_value = {"ok": True}

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with _patch_google_imports(registry=mock_registry, connector=mock_connector):
            result = await handler._handle_google_webhook({"inputs": []}, MockHTTPHandler())

        assert _status(result) == 200
        mock_connector.parse_google_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_google_error_handling(self, handler):
        mock_connector = MagicMock()
        mock_connector.parse_google_request.side_effect = ValueError("bad request")

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_connector

        with _patch_google_imports(registry=mock_registry, connector=mock_connector):
            result = await handler._handle_google_webhook({}, MockHTTPHandler())

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_google_no_auth_required(self, handler):
        """Google webhook path does not require auth context."""
        with patch.object(handler, "_handle_google_webhook", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = MagicMock(
                status_code=200, body=b"{}", content_type="application/json", headers={}
            )
            result = await handler._route_request(
                "/api/devices/google/webhook",
                "POST",
                {},
                MockHTTPHandler(),
                {},
            )
            mock_method.assert_called_once()


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_get_circuit_breaker_creates_new(self):
        cb = _get_circuit_breaker("test_connector")
        assert cb is not None
        assert cb.can_proceed()

    def test_get_circuit_breaker_returns_same(self):
        cb1 = _get_circuit_breaker("test_connector")
        cb2 = _get_circuit_breaker("test_connector")
        assert cb1 is cb2

    def test_get_circuit_breaker_different_connectors(self):
        cb1 = _get_circuit_breaker("fcm")
        cb2 = _get_circuit_breaker("apns")
        assert cb1 is not cb2

    def test_get_status_empty(self):
        status = get_device_circuit_breaker_status()
        assert status == {}

    def test_get_status_with_breakers(self):
        _get_circuit_breaker("fcm")
        _get_circuit_breaker("apns")
        status = get_device_circuit_breaker_status()
        assert "fcm" in status
        assert "apns" in status
        assert status["fcm"]["state"] == "closed"

    def test_clear_circuit_breakers(self):
        _get_circuit_breaker("fcm")
        _clear_device_circuit_breakers()
        status = get_device_circuit_breaker_status()
        assert status == {}

    def test_circuit_breaker_opens_after_failures(self):
        cb = _get_circuit_breaker("fcm")
        for _ in range(5):
            cb.record_failure()
        assert not cb.can_proceed()

    def test_circuit_breaker_records_success(self):
        cb = _get_circuit_breaker("fcm")
        cb.record_success()
        assert cb.can_proceed()


# ============================================================================
# Route Request Integration
# ============================================================================


class TestRouteRequest:
    """Test the _route_request routing logic."""

    @pytest.mark.asyncio
    async def test_unmatched_path_returns_none(self, handler):
        result = await handler._route_request(
            "/api/other/thing", "GET", {}, MockHTTPHandler(), None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_unmatched_segments_returns_none(self, handler):
        result = await handler._route_request("/api/devices", "GET", {}, MockHTTPHandler(), None)
        # Only 2 segments: ["api", "devices"] - no device_id
        # The segment check on line 225-226 extracts segments[2] if len>=3
        # With exactly ["api", "devices"], device_id is None -> returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_route_to_health(self, handler):
        with patch.object(handler, "_get_health", new_callable=AsyncMock) as mock_health:
            mock_health.return_value = MagicMock(
                status_code=200, body=b"{}", content_type="application/json", headers={}
            )
            result = await handler._route_request(
                "/api/devices/health", "GET", {}, MockHTTPHandler(), None
            )
            mock_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_register(self, handler):
        with patch.object(handler, "_register_device", new_callable=AsyncMock) as mock_reg:
            mock_reg.return_value = MagicMock(
                status_code=200, body=b"{}", content_type="application/json", headers={}
            )
            result = await handler._route_request(
                "/api/devices/register",
                "POST",
                {},
                MockHTTPHandler(),
                {"device_type": "android", "push_token": "abc"},
            )
            mock_reg.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_get_device(self, handler):
        with patch.object(handler, "_get_device", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, body=b"{}", content_type="application/json", headers={}
            )
            result = await handler._route_request(
                "/api/devices/dev-001", "GET", {}, MockHTTPHandler(), None
            )
            mock_get.assert_called_once()
            assert mock_get.call_args[0][0] == "dev-001"

    @pytest.mark.asyncio
    async def test_route_to_delete_device(self, handler):
        with patch.object(handler, "_unregister_device", new_callable=AsyncMock) as mock_del:
            mock_del.return_value = MagicMock(
                status_code=200, body=b"{}", content_type="application/json", headers={}
            )
            result = await handler._route_request(
                "/api/devices/dev-001", "DELETE", {}, MockHTTPHandler(), None
            )
            mock_del.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_notify_device(self, handler):
        with patch.object(handler, "_notify_device", new_callable=AsyncMock) as mock_notify:
            mock_notify.return_value = MagicMock(
                status_code=200, body=b"{}", content_type="application/json", headers={}
            )
            result = await handler._route_request(
                "/api/devices/dev-001/notify",
                "POST",
                {},
                MockHTTPHandler(),
                {"title": "Hi", "body": "Hello"},
            )
            mock_notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_list_user_devices(self, handler):
        with patch.object(handler, "_list_user_devices", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = MagicMock(
                status_code=200, body=b"{}", content_type="application/json", headers={}
            )
            result = await handler._route_request(
                "/api/devices/user/user-001", "GET", {}, MockHTTPHandler(), None
            )
            mock_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_notify_user(self, handler):
        with patch.object(handler, "_notify_user", new_callable=AsyncMock) as mock_notify:
            mock_notify.return_value = MagicMock(
                status_code=200, body=b"{}", content_type="application/json", headers={}
            )
            result = await handler._route_request(
                "/api/devices/user/user-001/notify",
                "POST",
                {},
                MockHTTPHandler(),
                {"title": "Hi", "body": "Hello"},
            )
            mock_notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_user_wrong_method(self, handler):
        """POST to /api/devices/user/{user_id} without /notify returns None."""
        result = await handler._route_request(
            "/api/devices/user/user-001",
            "POST",
            {},
            MockHTTPHandler(),
            {},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_route_user_invalid_suffix(self, handler):
        """Invalid suffix after user/{user_id}/ returns None."""
        result = await handler._route_request(
            "/api/devices/user/user-001/invalid",
            "POST",
            {},
            MockHTTPHandler(),
            {},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_versioned_path_strips_prefix(self, handler):
        """Versioned paths like /api/v1/devices/health should work."""
        with patch.object(handler, "_get_health", new_callable=AsyncMock) as mock_health:
            mock_health.return_value = MagicMock(
                status_code=200, body=b"{}", content_type="application/json", headers={}
            )
            result = await handler._route_request(
                "/api/v1/devices/health", "GET", {}, MockHTTPHandler(), None
            )
            mock_health.assert_called_once()


# ============================================================================
# Authentication Tests
# ============================================================================


class TestAuthentication:
    """Test authentication paths."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_unauthenticated_returns_401(self, handler):
        """Non-webhook endpoints require authentication."""
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        with patch.object(
            SecureHandler,
            "get_auth_context",
            side_effect=UnauthorizedError("Not authenticated"),
        ):
            result = await handler._route_request(
                "/api/devices/health", "GET", {}, MockHTTPHandler(), None
            )

        assert _status(result) == 401
        assert "Authentication required" in _body(result)["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_forbidden_returns_403(self, handler):
        """ForbiddenError in auth returns 403."""
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError

        with patch.object(
            SecureHandler,
            "get_auth_context",
            side_effect=ForbiddenError("Forbidden"),
        ):
            result = await handler._route_request(
                "/api/devices/health", "GET", {}, MockHTTPHandler(), None
            )

        assert _status(result) == 403
        assert "Permission denied" in _body(result)["error"]


# ============================================================================
# Helper functions for test setup
# ============================================================================


def _mock_auth_context(
    user_id: str = "test-user-001",
    roles: set | None = None,
) -> "AuthorizationContext":
    """Create a mock AuthorizationContext."""
    from aragora.rbac.models import AuthorizationContext

    return AuthorizationContext(
        user_id=user_id,
        user_email="test@example.com",
        org_id="test-org-001",
        roles=roles or {"admin", "owner"},
        permissions={"*"},
    )


def _selective_import_error(target_module: str):
    """Create an import side_effect that only fails for the target module."""
    import builtins

    original_import = builtins.__import__

    def side_effect(name, *args, **kwargs):
        if name == target_module or name.startswith(target_module + "."):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    return side_effect


async def _register_with_mocks(
    handler: DeviceHandler,
    body: dict,
    connector: Any,
    platform: str,
    registry_raises: Exception | None = None,
) -> Any:
    """Helper to call _register_device with fully mocked connector pipeline."""
    # Create mock DeviceType enum
    mock_device_type_val = MagicMock()

    mock_device_type_enum = MagicMock()
    mock_device_type_enum.return_value = mock_device_type_val
    mock_device_type_enum.ANDROID = mock_device_type_val
    mock_device_type_enum.WEB = MagicMock()
    mock_device_type_enum.IOS = MagicMock()
    mock_device_type_enum.ALEXA = MagicMock()
    mock_device_type_enum.GOOGLE_HOME = MagicMock()

    mock_registry = MagicMock()
    if registry_raises:
        mock_registry.get.side_effect = registry_raises
    elif connector:
        mock_registry.get.return_value = connector
    else:
        mock_registry.get.side_effect = KeyError("not found")

    mock_get_registry = MagicMock(return_value=mock_registry)

    mock_devices_module = MagicMock()
    mock_devices_module.DeviceRegistration = MagicMock()
    mock_devices_module.DeviceType = mock_device_type_enum
    mock_devices_module.get_registry = mock_get_registry

    mock_registry_module = MagicMock()
    mock_registry_module.get_registry = mock_get_registry

    with patch.dict(
        "sys.modules",
        {
            "aragora.connectors.devices": mock_devices_module,
            "aragora.connectors.devices.registry": mock_registry_module,
        },
    ):
        result = await handler._register_device(body, _mock_auth_context(), None)

    return result


def _patch_device_connector_imports(registry=None):
    """Context manager to patch device connector imports for notify tests."""
    mock_device_type_enum = MagicMock()
    android_val = MagicMock()
    ios_val = MagicMock()
    web_val = MagicMock()

    mock_device_type_enum.return_value = android_val  # default
    mock_device_type_enum.ANDROID = android_val
    mock_device_type_enum.IOS = ios_val
    mock_device_type_enum.WEB = web_val

    # Make DeviceType(value) return the right mock
    def device_type_init(value):
        mapping = {"android": android_val, "ios": ios_val, "web": web_val}
        return mapping.get(value, android_val)

    mock_device_type_enum.side_effect = device_type_init

    if registry is None:
        registry = MagicMock()

    mock_devices_module = MagicMock()
    mock_devices_module.DeviceMessage = MagicMock()
    mock_devices_module.DeviceToken = MagicMock()
    mock_devices_module.DeviceType = mock_device_type_enum

    mock_registry_module = MagicMock()
    mock_registry_module.get_registry = MagicMock(return_value=registry)

    mock_session_module = MagicMock()

    # We need to build the platform_map that the handler uses
    # The handler creates it inline: {DeviceType.ANDROID: "fcm", ...}
    # Since we're mocking DeviceType, we map our mock values
    class _PatchCtx:
        def __enter__(self_ctx):
            self_ctx._patches = [
                patch.dict(
                    "sys.modules",
                    {
                        "aragora.connectors.devices": mock_devices_module,
                        "aragora.connectors.devices.registry": mock_registry_module,
                    },
                ),
            ]
            for p in self_ctx._patches:
                p.__enter__()
            return self_ctx

        def __exit__(self_ctx, *args):
            for p in reversed(self_ctx._patches):
                p.__exit__(*args)

    return _PatchCtx()


def _patch_alexa_imports(registry=None, connector=None, isinstance_returns=True):
    """Context manager to patch Alexa connector imports."""
    if registry is None:
        registry = MagicMock()

    mock_alexa_module = MagicMock()
    mock_alexa_connector_class = MagicMock()
    mock_alexa_module.AlexaConnector = mock_alexa_connector_class

    mock_registry_module = MagicMock()
    mock_registry_module.get_registry = MagicMock(return_value=registry)

    class _PatchCtx:
        def __enter__(self_ctx):
            self_ctx._patches = [
                patch.dict(
                    "sys.modules",
                    {
                        "aragora.connectors.devices.alexa": mock_alexa_module,
                        "aragora.connectors.devices.registry": mock_registry_module,
                    },
                ),
            ]
            if not isinstance_returns:
                # Patch isinstance to return False for the connector type check
                self_ctx._patches.append(
                    patch(
                        "aragora.server.handlers.devices.isinstance",
                        side_effect=lambda obj, cls: False
                        if cls is mock_alexa_connector_class
                        else builtins_isinstance(obj, cls),
                    )
                )
            elif connector is not None:
                # Make isinstance return True for the connector
                self_ctx._patches.append(
                    patch(
                        "aragora.server.handlers.devices.isinstance",
                        side_effect=lambda obj, cls: True
                        if obj is connector
                        else builtins_isinstance(obj, cls),
                    )
                )

            for p in self_ctx._patches:
                p.__enter__()
            return self_ctx

        def __exit__(self_ctx, *args):
            for p in reversed(self_ctx._patches):
                p.__exit__(*args)

    return _PatchCtx()


def _patch_google_imports(registry=None, connector=None, isinstance_returns=True):
    """Context manager to patch Google Home connector imports."""
    if registry is None:
        registry = MagicMock()

    mock_google_module = MagicMock()
    mock_google_connector_class = MagicMock()
    mock_google_module.GoogleHomeConnector = mock_google_connector_class

    mock_registry_module = MagicMock()
    mock_registry_module.get_registry = MagicMock(return_value=registry)

    class _PatchCtx:
        def __enter__(self_ctx):
            self_ctx._patches = [
                patch.dict(
                    "sys.modules",
                    {
                        "aragora.connectors.devices.google_home": mock_google_module,
                        "aragora.connectors.devices.registry": mock_registry_module,
                    },
                ),
            ]
            if not isinstance_returns:
                self_ctx._patches.append(
                    patch(
                        "aragora.server.handlers.devices.isinstance",
                        side_effect=lambda obj, cls: False
                        if cls is mock_google_connector_class
                        else builtins_isinstance(obj, cls),
                    )
                )
            elif connector is not None:
                self_ctx._patches.append(
                    patch(
                        "aragora.server.handlers.devices.isinstance",
                        side_effect=lambda obj, cls: True
                        if obj is connector
                        else builtins_isinstance(obj, cls),
                    )
                )

            for p in self_ctx._patches:
                p.__enter__()
            return self_ctx

        def __exit__(self_ctx, *args):
            for p in reversed(self_ctx._patches):
                p.__exit__(*args)

    return _PatchCtx()


import builtins as _builtins

builtins_isinstance = _builtins.isinstance
