"""
Tests for DeviceHandler - device registration and notification API.

Covers:
- POST /api/devices/register - Register device for push notifications
- DELETE /api/devices/{device_id} - Unregister device
- GET /api/devices/{device_id} - Get device info
- POST /api/devices/{device_id}/notify - Send notification to device
- POST /api/devices/user/{user_id}/notify - Send to all user devices
- GET /api/devices/user/{user_id} - List user's devices
- GET /api/devices/health - Get device connector health
- POST /api/devices/alexa/webhook - Alexa skill webhook
- POST /api/devices/google/webhook - Google Actions webhook
- RBAC permission enforcement
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.devices import DeviceHandler
from aragora.server.handlers.base import json_response, error_response
from aragora.rbac.models import AuthorizationContext


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def server_context():
    """Create a mock server context."""
    return {"config": {"debug": True}}


@pytest.fixture
def auth_context():
    """Create a test authorization context with device permissions."""
    return AuthorizationContext(
        user_id="user-123",
        org_id="org-456",
        roles={"admin"},
        permissions={"devices.read", "devices.write", "devices.notify"},
    )


@pytest.fixture
def user_auth_context():
    """Create a regular user authorization context."""
    return AuthorizationContext(
        user_id="user-123",
        org_id="org-456",
        roles={"user"},
        permissions={"devices.read", "devices.write", "devices.notify"},
    )


@pytest.fixture
def read_only_context():
    """Create a read-only authorization context."""
    return AuthorizationContext(
        user_id="user-readonly",
        org_id="org-456",
        roles={"viewer"},
        permissions={"devices.read"},
    )


@pytest.fixture
def no_permission_context():
    """Create an authorization context with no device permissions."""
    return AuthorizationContext(
        user_id="user-none",
        org_id="org-456",
        roles={"guest"},
        permissions=set(),
    )


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = "GET"
    handler.headers = {"Authorization": "Bearer test-token"}
    return handler


@pytest.fixture
def device_handler(server_context):
    """Create a DeviceHandler instance."""
    return DeviceHandler(server_context)


@pytest.fixture
def mock_device_session():
    """Create a mock device session."""
    session = MagicMock()
    session.device_id = "device-123"
    session.user_id = "user-123"
    session.device_type = "android"
    session.device_name = "Test Device"
    session.app_version = "1.0.0"
    session.push_token = "test-push-token"
    session.last_active = datetime.now(timezone.utc).isoformat()
    session.notification_count = 5
    session.created_at = datetime.now(timezone.utc).isoformat()
    session.record_notification = MagicMock()
    return session


# -----------------------------------------------------------------------------
# Initialization Tests
# -----------------------------------------------------------------------------


class TestDeviceHandlerInit:
    """Tests for DeviceHandler initialization."""

    def test_init_with_server_context(self, server_context):
        """Handler initializes with server context."""
        handler = DeviceHandler(server_context)
        assert handler.ctx == server_context

    def test_resource_type(self, device_handler):
        """Handler has correct resource type."""
        assert device_handler.RESOURCE_TYPE == "devices"

    def test_routes(self, device_handler):
        """Handler has correct routes."""
        assert "/api/devices/register" in device_handler.ROUTES
        assert "/api/devices/health" in device_handler.ROUTES
        assert "/api/devices/alexa/webhook" in device_handler.ROUTES
        assert "/api/devices/google/webhook" in device_handler.ROUTES

    def test_can_handle_device_paths(self, device_handler):
        """can_handle returns True for device paths."""
        assert device_handler.can_handle("/api/devices/register") is True
        assert device_handler.can_handle("/api/devices/health") is True
        assert device_handler.can_handle("/api/devices/device-123") is True
        assert device_handler.can_handle("/api/devices/user/user-123") is True
        assert device_handler.can_handle("/api/devices/alexa/webhook") is True

    def test_can_handle_non_device_paths(self, device_handler):
        """can_handle returns False for non-device paths."""
        assert device_handler.can_handle("/api/users") is False
        assert device_handler.can_handle("/api/debates") is False


# -----------------------------------------------------------------------------
# RBAC Permission Tests
# -----------------------------------------------------------------------------


class TestRBACPermissions:
    """Tests for RBAC permission enforcement."""

    @pytest.mark.asyncio
    async def test_unauthenticated_request_returns_401(self, device_handler, mock_handler):
        """Unauthenticated requests return 401 for protected endpoints."""
        from aragora.server.handlers.secure import UnauthorizedError

        with patch.object(device_handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("Token required")

            result = await device_handler.handle(
                "/api/devices/health",
                {},
                mock_handler,
            )

            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_no_permission_returns_403(
        self, device_handler, mock_handler, no_permission_context
    ):
        """Requests without required permission return 403."""
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(
            device_handler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=no_permission_context,
        ):
            with patch.object(
                device_handler, "check_permission", side_effect=ForbiddenError("Permission denied")
            ):
                result = await device_handler.handle(
                    "/api/devices/health",
                    {},
                    mock_handler,
                )

                assert result.status_code == 403


# -----------------------------------------------------------------------------
# GET /api/devices/health Tests
# -----------------------------------------------------------------------------


class TestGetHealth:
    """Tests for GET /api/devices/health endpoint."""

    @pytest.mark.asyncio
    async def test_get_health_success(self, device_handler, mock_handler, auth_context):
        """Get health returns connector status."""
        mock_handler.command = "GET"

        mock_health = {
            "status": "healthy",
            "connectors": {
                "fcm": {"status": "connected", "latency_ms": 50},
                "apns": {"status": "connected", "latency_ms": 45},
            },
        }

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch("aragora.connectors.devices.registry.get_registry") as mock_registry:
                    mock_registry.return_value.get_health = AsyncMock(return_value=mock_health)

                    result = await device_handler.handle(
                        "/api/devices/health",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_health_no_connectors(self, device_handler, mock_handler, auth_context):
        """Get health handles missing connectors gracefully."""
        mock_handler.command = "GET"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch(
                    "aragora.connectors.devices.registry.get_registry",
                    side_effect=ImportError("Module not found"),
                ):
                    result = await device_handler.handle(
                        "/api/devices/health",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["status"] == "unavailable"


# -----------------------------------------------------------------------------
# POST /api/devices/register Tests
# -----------------------------------------------------------------------------


class TestRegisterDevice:
    """Tests for POST /api/devices/register endpoint."""

    @pytest.mark.asyncio
    async def test_register_missing_fields_returns_400(
        self, device_handler, mock_handler, auth_context
    ):
        """Register without required fields returns 400."""
        mock_handler.command = "POST"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=({}, None)
                ):
                    result = await device_handler.handle_post(
                        "/api/devices/register",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 400
                    data = json.loads(result.body)
                    assert "Missing required fields" in data["error"]

    @pytest.mark.asyncio
    async def test_register_invalid_device_type_returns_400(
        self, device_handler, mock_handler, auth_context
    ):
        """Register with invalid device type returns 400."""
        mock_handler.command = "POST"

        body = {
            "device_type": "invalid_type",
            "push_token": "test-token-123",
        }

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch("aragora.connectors.devices.DeviceType") as mock_device_type:
                        mock_device_type.side_effect = ValueError("Invalid type")

                        result = await device_handler.handle_post(
                            "/api/devices/register",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_register_device_success(self, device_handler, mock_handler, auth_context):
        """Register device successfully routes to _register_device."""
        mock_handler.command = "POST"

        body = {
            "device_type": "android",
            "push_token": "test-token-123",
            "device_name": "Test Phone",
        }

        mock_response = json_response(
            {"success": True, "device_id": "device-new-123", "device_type": "android"}
        )

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch.object(
                        device_handler,
                        "_register_device",
                        new_callable=AsyncMock,
                        return_value=mock_response,
                    ):
                        result = await device_handler.handle_post(
                            "/api/devices/register",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert data["success"] is True
                        assert data["device_id"] == "device-new-123"

    @pytest.mark.asyncio
    async def test_register_requires_write_permission(
        self, device_handler, mock_handler, read_only_context
    ):
        """Register requires devices.write permission."""
        mock_handler.command = "POST"
        from aragora.server.handlers.secure import ForbiddenError

        body = {"device_type": "android", "push_token": "test-token"}

        with patch.object(
            device_handler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=read_only_context,
        ):
            with patch.object(
                device_handler, "check_permission", side_effect=ForbiddenError("devices.write")
            ):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    result = await device_handler.handle_post(
                        "/api/devices/register",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 403


# -----------------------------------------------------------------------------
# DELETE /api/devices/{device_id} Tests
# -----------------------------------------------------------------------------


class TestUnregisterDevice:
    """Tests for DELETE /api/devices/{device_id} endpoint."""

    @pytest.mark.asyncio
    async def test_unregister_device_not_found(self, device_handler, mock_handler, auth_context):
        """Unregister non-existent device returns 404."""
        mock_handler.command = "DELETE"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch("aragora.server.session_store.get_session_store") as mock_store:
                    mock_store.return_value.get_device_session.return_value = None

                    result = await device_handler.handle_delete(
                        "/api/devices/device-123",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_unregister_device_not_owner_returns_403(
        self, device_handler, mock_handler, user_auth_context, mock_device_session
    ):
        """Non-owner cannot unregister device."""
        mock_handler.command = "DELETE"
        mock_device_session.user_id = "other-user"

        with patch.object(
            device_handler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=user_auth_context,
        ):
            with patch.object(device_handler, "check_permission"):
                with patch("aragora.server.session_store.get_session_store") as mock_store:
                    mock_store.return_value.get_device_session.return_value = mock_device_session

                    result = await device_handler.handle_delete(
                        "/api/devices/device-123",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_unregister_device_success(
        self, device_handler, mock_handler, auth_context, mock_device_session
    ):
        """Owner can unregister their device."""
        mock_handler.command = "DELETE"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch("aragora.server.session_store.get_session_store") as mock_store:
                    mock_store.return_value.get_device_session.return_value = mock_device_session
                    mock_store.return_value.delete_device_session.return_value = True

                    result = await device_handler.handle_delete(
                        "/api/devices/device-123",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["success"] is True
                    assert data["device_id"] == "device-123"


# -----------------------------------------------------------------------------
# GET /api/devices/{device_id} Tests
# -----------------------------------------------------------------------------


class TestGetDevice:
    """Tests for GET /api/devices/{device_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_device_not_found(self, device_handler, mock_handler, auth_context):
        """Get non-existent device returns 404."""
        mock_handler.command = "GET"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch("aragora.server.session_store.get_session_store") as mock_store:
                    mock_store.return_value.get_device_session.return_value = None

                    result = await device_handler.handle(
                        "/api/devices/device-123",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_device_success(
        self, device_handler, mock_handler, auth_context, mock_device_session
    ):
        """Get device returns device info."""
        mock_handler.command = "GET"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch("aragora.server.session_store.get_session_store") as mock_store:
                    mock_store.return_value.get_device_session.return_value = mock_device_session

                    result = await device_handler.handle(
                        "/api/devices/device-123",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["device_id"] == "device-123"
                    assert data["user_id"] == "user-123"


# -----------------------------------------------------------------------------
# GET /api/devices/user/{user_id} Tests
# -----------------------------------------------------------------------------


class TestListUserDevices:
    """Tests for GET /api/devices/user/{user_id} endpoint."""

    @pytest.mark.asyncio
    async def test_list_user_devices_success(
        self, device_handler, mock_handler, auth_context, mock_device_session
    ):
        """List user devices returns device list."""
        mock_handler.command = "GET"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch("aragora.server.session_store.get_session_store") as mock_store:
                    mock_store.return_value.find_devices_by_user.return_value = [
                        mock_device_session
                    ]

                    result = await device_handler.handle(
                        "/api/devices/user/user-123",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["user_id"] == "user-123"
                    assert data["device_count"] == 1

    @pytest.mark.asyncio
    async def test_list_other_user_devices_returns_403(
        self, device_handler, mock_handler, user_auth_context
    ):
        """Non-admin cannot list other user's devices."""
        mock_handler.command = "GET"

        with patch.object(
            device_handler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=user_auth_context,
        ):
            with patch.object(device_handler, "check_permission"):
                result = await device_handler.handle(
                    "/api/devices/user/other-user",
                    {},
                    mock_handler,
                )

                assert result.status_code == 403


# -----------------------------------------------------------------------------
# POST /api/devices/{device_id}/notify Tests
# -----------------------------------------------------------------------------


class TestNotifyDevice:
    """Tests for POST /api/devices/{device_id}/notify endpoint."""

    @pytest.mark.asyncio
    async def test_notify_missing_fields_returns_400(
        self, device_handler, mock_handler, auth_context
    ):
        """Notify without title/body returns 400."""
        mock_handler.command = "POST"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=({}, None)
                ):
                    result = await device_handler.handle_post(
                        "/api/devices/device-123/notify",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 400
                    data = json.loads(result.body)
                    assert "title and body are required" in data["error"]

    @pytest.mark.asyncio
    async def test_notify_device_not_found(self, device_handler, mock_handler, auth_context):
        """Notify non-existent device returns 404."""
        mock_handler.command = "POST"

        body = {"title": "Test", "body": "Message"}

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch("aragora.server.session_store.get_session_store") as mock_store:
                        mock_store.return_value.get_device_session.return_value = None

                        result = await device_handler.handle_post(
                            "/api/devices/device-123/notify",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_notify_device_success(self, device_handler, mock_handler, auth_context):
        """Notify device successfully routes to _notify_device."""
        mock_handler.command = "POST"

        body = {"title": "Test", "body": "Message"}

        mock_response = json_response(
            {"success": True, "device_id": "device-123", "message_id": "msg-123", "status": "sent"}
        )

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch.object(
                        device_handler,
                        "_notify_device",
                        new_callable=AsyncMock,
                        return_value=mock_response,
                    ):
                        result = await device_handler.handle_post(
                            "/api/devices/device-123/notify",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert data["success"] is True
                        assert data["message_id"] == "msg-123"


# -----------------------------------------------------------------------------
# POST /api/devices/user/{user_id}/notify Tests
# -----------------------------------------------------------------------------


class TestNotifyUser:
    """Tests for POST /api/devices/user/{user_id}/notify endpoint."""

    @pytest.mark.asyncio
    async def test_notify_user_no_devices(self, device_handler, mock_handler, auth_context):
        """Notify user with no devices returns success with 0 count."""
        mock_handler.command = "POST"

        body = {"title": "Test", "body": "Message"}

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch("aragora.server.session_store.get_session_store") as mock_store:
                        mock_store.return_value.find_devices_by_user.return_value = []

                        result = await device_handler.handle_post(
                            "/api/devices/user/user-123/notify",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert data["devices_notified"] == 0

    @pytest.mark.asyncio
    async def test_notify_user_success(self, device_handler, mock_handler, auth_context):
        """Notify user successfully routes to _notify_user."""
        mock_handler.command = "POST"

        body = {"title": "Test", "body": "Message"}

        mock_response = json_response(
            {"success": True, "user_id": "user-123", "devices_notified": 1}
        )

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch.object(
                        device_handler,
                        "_notify_user",
                        new_callable=AsyncMock,
                        return_value=mock_response,
                    ):
                        result = await device_handler.handle_post(
                            "/api/devices/user/user-123/notify",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert data["success"] is True
                        assert data["devices_notified"] == 1


# -----------------------------------------------------------------------------
# POST /api/devices/alexa/webhook Tests
# -----------------------------------------------------------------------------


class TestAlexaWebhook:
    """Tests for POST /api/devices/alexa/webhook endpoint."""

    @pytest.mark.asyncio
    async def test_alexa_webhook_no_auth_required(self, device_handler, mock_handler):
        """Alexa webhook does not require auth token."""
        mock_handler.command = "POST"

        # Mock the connector imports
        with patch.object(device_handler, "read_json_body_validated", return_value=({}, None)):
            with patch("aragora.connectors.devices.registry.get_registry") as mock_registry:
                mock_registry.side_effect = ImportError("Module not found")

                result = await device_handler.handle_post(
                    "/api/devices/alexa/webhook",
                    {},
                    mock_handler,
                )

                # Should return 503 (connector unavailable), not 401
                assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_alexa_webhook_invalid_skill_id(self, device_handler, mock_handler):
        """Alexa webhook with invalid skill ID returns 403."""
        mock_handler.command = "POST"

        body = {"session": {"application": {"applicationId": "wrong-skill"}}}

        mock_response = error_response("Invalid skill ID", 403)

        with patch.object(device_handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(
                device_handler,
                "_handle_alexa_webhook",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):
                result = await device_handler.handle_post(
                    "/api/devices/alexa/webhook",
                    {},
                    mock_handler,
                )

                assert result.status_code == 403


# -----------------------------------------------------------------------------
# POST /api/devices/google/webhook Tests
# -----------------------------------------------------------------------------


class TestGoogleWebhook:
    """Tests for POST /api/devices/google/webhook endpoint."""

    @pytest.mark.asyncio
    async def test_google_webhook_no_auth_required(self, device_handler, mock_handler):
        """Google webhook does not require auth token."""
        mock_handler.command = "POST"

        with patch.object(device_handler, "read_json_body_validated", return_value=({}, None)):
            with patch("aragora.connectors.devices.registry.get_registry") as mock_registry:
                mock_registry.side_effect = ImportError("Module not found")

                result = await device_handler.handle_post(
                    "/api/devices/google/webhook",
                    {},
                    mock_handler,
                )

                # Should return 503 (connector unavailable), not 401
                assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_google_webhook_sync_intent(self, device_handler, mock_handler):
        """Google webhook handles SYNC intent."""
        mock_handler.command = "POST"

        body = {
            "requestId": "req-123",
            "inputs": [{"intent": "action.devices.SYNC"}],
            "user": {"userId": "user-123"},
        }

        mock_response = json_response({"requestId": "req-123", "payload": {"devices": []}})

        with patch.object(device_handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(
                device_handler,
                "_handle_google_webhook",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):
                result = await device_handler.handle_post(
                    "/api/devices/google/webhook",
                    {},
                    mock_handler,
                )

                assert result.status_code == 200
                data = json.loads(result.body)
                assert data["requestId"] == "req-123"


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_none(self, device_handler, mock_handler, auth_context):
        """Unknown path returns None."""
        mock_handler.command = "GET"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                result = await device_handler.handle(
                    "/api/devices",
                    {},
                    mock_handler,
                )

                # Returns None for unknown paths
                assert result is None

    @pytest.mark.asyncio
    async def test_internal_error_returns_500(self, device_handler, mock_handler, auth_context):
        """Internal error returns 500."""
        mock_handler.command = "GET"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch("aragora.connectors.devices.registry.get_registry") as mock_registry:
                    mock_registry.return_value.get_health = AsyncMock(
                        side_effect=ValueError("Unexpected error")
                    )

                    result = await device_handler.handle(
                        "/api/devices/health",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 500


# -----------------------------------------------------------------------------
# Circuit Breaker Tests
# -----------------------------------------------------------------------------


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_breaker_init(self):
        """Circuit breaker initializes with correct defaults."""
        from aragora.server.handlers.devices import DeviceCircuitBreaker

        cb = DeviceCircuitBreaker()
        assert cb.state == "closed"
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 30.0

    def test_circuit_breaker_opens_after_threshold(self):
        """Circuit breaker opens after failure threshold is reached."""
        from aragora.server.handlers.devices import DeviceCircuitBreaker

        cb = DeviceCircuitBreaker(failure_threshold=3)

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == "open"
        assert cb.can_proceed() is False

    def test_circuit_breaker_closes_on_success(self):
        """Circuit breaker closes after successful recovery."""
        from aragora.server.handlers.devices import DeviceCircuitBreaker

        cb = DeviceCircuitBreaker(failure_threshold=3, cooldown_seconds=0.01)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

        # Wait for cooldown
        import time

        time.sleep(0.02)

        # Should transition to half-open
        assert cb.can_proceed() is True
        assert cb.state == "half_open"

        # Record successes
        for _ in range(3):
            cb.record_success()

        assert cb.state == "closed"

    def test_circuit_breaker_reopens_on_failure_in_half_open(self):
        """Circuit breaker reopens on failure in half-open state."""
        from aragora.server.handlers.devices import DeviceCircuitBreaker

        cb = DeviceCircuitBreaker(failure_threshold=3, cooldown_seconds=0.01)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()

        # Wait for cooldown
        import time

        time.sleep(0.02)

        # Transition to half-open
        assert cb.can_proceed() is True
        assert cb.state == "half_open"

        # Fail in half-open
        cb.record_failure()
        assert cb.state == "open"

    def test_circuit_breaker_get_status(self):
        """Circuit breaker returns correct status."""
        from aragora.server.handlers.devices import DeviceCircuitBreaker

        cb = DeviceCircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)
        status = cb.get_status()

        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 5
        assert status["cooldown_seconds"] == 30.0

    def test_circuit_breaker_reset(self):
        """Circuit breaker can be reset."""
        from aragora.server.handlers.devices import DeviceCircuitBreaker

        cb = DeviceCircuitBreaker(failure_threshold=3)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

        # Reset
        cb.reset()
        assert cb.state == "closed"
        assert cb.can_proceed() is True

    def test_get_circuit_breaker_status(self):
        """get_device_circuit_breaker_status returns all circuit breakers."""
        from aragora.server.handlers.devices import (
            _get_circuit_breaker,
            get_device_circuit_breaker_status,
            _clear_device_circuit_breakers,
        )

        # Clear existing circuit breakers
        _clear_device_circuit_breakers()

        # Create some circuit breakers
        _get_circuit_breaker("fcm")
        _get_circuit_breaker("apns")

        status = get_device_circuit_breaker_status()
        assert "fcm" in status
        assert "apns" in status
        assert status["fcm"]["state"] == "closed"

        # Cleanup
        _clear_device_circuit_breakers()

    @pytest.mark.asyncio
    async def test_register_device_circuit_breaker_open(
        self, device_handler, mock_handler, auth_context
    ):
        """Registration fails when circuit breaker is open."""
        from aragora.server.handlers.devices import (
            _get_circuit_breaker,
            _clear_device_circuit_breakers,
        )

        # Clear and open the FCM circuit breaker
        _clear_device_circuit_breakers()
        cb = _get_circuit_breaker("fcm")
        for _ in range(5):
            cb.record_failure()

        mock_handler.command = "POST"
        mock_handler.client_address = ("127.0.0.1", 12345)

        body = {
            "device_type": "android",
            "push_token": "test-token-123",
        }

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch(
                        "aragora.server.handlers.devices._registration_limiter"
                    ) as mock_limiter:
                        mock_limiter.is_allowed.return_value = True
                        with patch("aragora.connectors.devices.DeviceType") as mock_dt:
                            from enum import Enum

                            class MockDeviceType(Enum):
                                ANDROID = "android"

                            mock_dt.side_effect = lambda x: MockDeviceType.ANDROID
                            mock_dt.ANDROID = MockDeviceType.ANDROID

                            result = await device_handler.handle_post(
                                "/api/devices/register",
                                {},
                                mock_handler,
                            )

                            # Should return 503 due to circuit breaker
                            assert result.status_code == 503
                            data = json.loads(result.body)
                            assert "temporarily unavailable" in data["error"]

        # Cleanup
        _clear_device_circuit_breakers()


# -----------------------------------------------------------------------------
# Rate Limiting Tests
# -----------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_registration_rate_limit_exceeded(
        self, device_handler, mock_handler, auth_context
    ):
        """Registration returns 429 when rate limit exceeded."""
        mock_handler.command = "POST"
        mock_handler.client_address = ("127.0.0.1", 12345)

        body = {
            "device_type": "android",
            "push_token": "test-token-123",
        }

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch(
                        "aragora.server.handlers.devices._registration_limiter"
                    ) as mock_limiter:
                        mock_limiter.is_allowed.return_value = False

                        result = await device_handler.handle_post(
                            "/api/devices/register",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 429
                        data = json.loads(result.body)
                        assert "Rate limit exceeded" in data["error"]

    @pytest.mark.asyncio
    async def test_notification_rate_limit_exceeded(
        self, device_handler, mock_handler, auth_context
    ):
        """Notification returns 429 when rate limit exceeded."""
        mock_handler.command = "POST"
        mock_handler.client_address = ("127.0.0.1", 12345)

        body = {"title": "Test", "body": "Message"}

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch(
                        "aragora.server.handlers.devices._notification_limiter"
                    ) as mock_limiter:
                        mock_limiter.is_allowed.return_value = False

                        result = await device_handler.handle_post(
                            "/api/devices/device-123/notify",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 429
                        data = json.loads(result.body)
                        assert "Rate limit exceeded" in data["error"]

    @pytest.mark.asyncio
    async def test_user_notification_rate_limit_exceeded(
        self, device_handler, mock_handler, auth_context
    ):
        """User notification returns 429 when rate limit exceeded."""
        mock_handler.command = "POST"
        mock_handler.client_address = ("127.0.0.1", 12345)

        body = {"title": "Test", "body": "Message"}

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch(
                        "aragora.server.handlers.devices._notification_limiter"
                    ) as mock_limiter:
                        mock_limiter.is_allowed.return_value = False

                        result = await device_handler.handle_post(
                            "/api/devices/user/user-123/notify",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 429
                        data = json.loads(result.body)
                        assert "Rate limit exceeded" in data["error"]


# -----------------------------------------------------------------------------
# Input Validation Tests
# -----------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_register_push_token_too_long(self, device_handler, mock_handler, auth_context):
        """Registration fails when push token exceeds max length."""
        mock_handler.command = "POST"
        mock_handler.client_address = ("127.0.0.1", 12345)

        # Create a token that's too long (> 4KB)
        long_token = "x" * 5000
        body = {
            "device_type": "android",
            "push_token": long_token,
        }

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch(
                        "aragora.server.handlers.devices._registration_limiter"
                    ) as mock_limiter:
                        mock_limiter.is_allowed.return_value = True

                        result = await device_handler.handle_post(
                            "/api/devices/register",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 400
                        data = json.loads(result.body)
                        assert "push_token exceeds maximum length" in data["error"]

    @pytest.mark.asyncio
    async def test_notify_title_too_long(
        self, device_handler, mock_handler, auth_context, mock_device_session
    ):
        """Notification fails when title exceeds max length."""
        mock_handler.command = "POST"
        mock_handler.client_address = ("127.0.0.1", 12345)

        # Create a title that's too long (> 256 chars)
        long_title = "x" * 300
        body = {"title": long_title, "body": "Message"}

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch(
                        "aragora.server.handlers.devices._notification_limiter"
                    ) as mock_limiter:
                        mock_limiter.is_allowed.return_value = True
                        with patch("aragora.server.session_store.get_session_store") as mock_store:
                            mock_store.return_value.get_device_session.return_value = (
                                mock_device_session
                            )

                            result = await device_handler.handle_post(
                                "/api/devices/device-123/notify",
                                {},
                                mock_handler,
                            )

                            assert result.status_code == 400
                            data = json.loads(result.body)
                            assert "title exceeds maximum length" in data["error"]

    @pytest.mark.asyncio
    async def test_notify_body_too_long(
        self, device_handler, mock_handler, auth_context, mock_device_session
    ):
        """Notification fails when body exceeds max length."""
        mock_handler.command = "POST"
        mock_handler.client_address = ("127.0.0.1", 12345)

        # Create a body that's too long (> 4KB)
        long_body = "x" * 5000
        body = {"title": "Test", "body": long_body}

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch.object(
                    device_handler, "read_json_body_validated", return_value=(body, None)
                ):
                    with patch(
                        "aragora.server.handlers.devices._notification_limiter"
                    ) as mock_limiter:
                        mock_limiter.is_allowed.return_value = True
                        with patch("aragora.server.session_store.get_session_store") as mock_store:
                            mock_store.return_value.get_device_session.return_value = (
                                mock_device_session
                            )

                            result = await device_handler.handle_post(
                                "/api/devices/device-123/notify",
                                {},
                                mock_handler,
                            )

                            assert result.status_code == 400
                            data = json.loads(result.body)
                            assert "body exceeds maximum length" in data["error"]


# -----------------------------------------------------------------------------
# Health Endpoint with Circuit Breaker Status Tests
# -----------------------------------------------------------------------------


class TestHealthWithCircuitBreaker:
    """Tests for health endpoint including circuit breaker status."""

    @pytest.mark.asyncio
    async def test_health_includes_circuit_breaker_status(
        self, device_handler, mock_handler, auth_context
    ):
        """Health endpoint includes circuit breaker status."""
        from aragora.server.handlers.devices import (
            _get_circuit_breaker,
            _clear_device_circuit_breakers,
        )

        # Clear and create a circuit breaker
        _clear_device_circuit_breakers()
        _get_circuit_breaker("fcm")

        mock_handler.command = "GET"

        mock_health = {
            "status": "healthy",
            "connectors": {"fcm": {"status": "connected"}},
        }

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch("aragora.connectors.devices.registry.get_registry") as mock_registry:
                    mock_registry.return_value.get_health = AsyncMock(return_value=mock_health)

                    result = await device_handler.handle(
                        "/api/devices/health",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "circuit_breakers" in data
                    assert "fcm" in data["circuit_breakers"]

        # Cleanup
        _clear_device_circuit_breakers()

    @pytest.mark.asyncio
    async def test_health_unavailable_includes_circuit_breaker_status(
        self, device_handler, mock_handler, auth_context
    ):
        """Health endpoint returns circuit breaker status even when connectors unavailable."""
        from aragora.server.handlers.devices import (
            _get_circuit_breaker,
            _clear_device_circuit_breakers,
        )

        # Clear and create a circuit breaker
        _clear_device_circuit_breakers()
        cb = _get_circuit_breaker("fcm")
        for _ in range(5):
            cb.record_failure()

        mock_handler.command = "GET"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch(
                    "aragora.connectors.devices.registry.get_registry",
                    side_effect=ImportError("Module not found"),
                ):
                    result = await device_handler.handle(
                        "/api/devices/health",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["status"] == "unavailable"
                    assert "circuit_breakers" in data
                    assert data["circuit_breakers"]["fcm"]["state"] == "open"

        # Cleanup
        _clear_device_circuit_breakers()


# -----------------------------------------------------------------------------
# Versioned Path Tests
# -----------------------------------------------------------------------------


class TestVersionedPaths:
    """Tests for API versioning support."""

    def test_can_handle_versioned_paths(self, device_handler):
        """Handler can handle versioned API paths."""
        assert device_handler.can_handle("/api/v1/devices/register") is True
        assert device_handler.can_handle("/api/v2/devices/health") is True
        assert device_handler.can_handle("/api/v1/devices/device-123") is True

    def test_can_handle_unversioned_paths(self, device_handler):
        """Handler can handle unversioned API paths."""
        assert device_handler.can_handle("/api/devices/register") is True
        assert device_handler.can_handle("/api/devices/health") is True


# -----------------------------------------------------------------------------
# Admin Override Tests
# -----------------------------------------------------------------------------


class TestAdminOverrides:
    """Tests for admin permission overrides."""

    @pytest.mark.asyncio
    async def test_admin_can_view_other_user_devices(
        self, device_handler, mock_handler, auth_context, mock_device_session
    ):
        """Admin can view other user's devices."""
        mock_handler.command = "GET"
        mock_device_session.user_id = "other-user"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch("aragora.server.session_store.get_session_store") as mock_store:
                    mock_store.return_value.get_device_session.return_value = mock_device_session

                    result = await device_handler.handle(
                        "/api/devices/device-123",
                        {},
                        mock_handler,
                    )

                    # Admin should be able to view (auth_context has admin role)
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_admin_can_delete_other_user_device(
        self, device_handler, mock_handler, auth_context, mock_device_session
    ):
        """Admin can delete other user's device."""
        mock_handler.command = "DELETE"
        mock_device_session.user_id = "other-user"

        with patch.object(
            device_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(device_handler, "check_permission"):
                with patch("aragora.server.session_store.get_session_store") as mock_store:
                    mock_store.return_value.get_device_session.return_value = mock_device_session
                    mock_store.return_value.delete_device_session.return_value = True

                    result = await device_handler.handle_delete(
                        "/api/devices/device-123",
                        {},
                        mock_handler,
                    )

                    # Admin should be able to delete
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["success"] is True
