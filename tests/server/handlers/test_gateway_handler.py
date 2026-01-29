"""
Tests for GatewayHandler - Device gateway management HTTP endpoints.

Tests cover:
- Device registration and management
- Device listing with filters
- Channel listing
- Routing statistics and rules
- Message routing
- RBAC protection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest

from aragora.server.handlers.gateway_handler import GatewayHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockDeviceNode:
    """Mock device node for testing."""

    device_id: str = "device-001"
    name: str = "Test Device"
    device_type: str = "laptop"
    capabilities: list[str] = field(default_factory=lambda: ["browser", "terminal"])
    status: MagicMock = field(default_factory=lambda: MagicMock(value="online"))
    paired_at: str = "2025-01-01T00:00:00Z"
    last_seen: str = "2025-01-29T12:00:00Z"
    allowed_channels: list[str] = field(default_factory=lambda: ["slack", "email"])
    metadata: dict[str, Any] = field(default_factory=dict)


class MockDeviceRegistry:
    """Mock device registry for testing."""

    def __init__(self):
        self._devices: dict[str, MockDeviceNode] = {}

    async def list_devices(
        self,
        status: Optional[Any] = None,
        device_type: Optional[str] = None,
    ) -> list[MockDeviceNode]:
        devices = list(self._devices.values())
        if device_type:
            devices = [d for d in devices if d.device_type == device_type]
        return devices

    async def get(self, device_id: str) -> Optional[MockDeviceNode]:
        return self._devices.get(device_id)

    async def register(self, device: MockDeviceNode) -> str:
        device_id = device.device_id or f"device-{len(self._devices) + 1}"
        self._devices[device_id] = device
        return device_id

    async def unregister(self, device_id: str) -> bool:
        if device_id in self._devices:
            del self._devices[device_id]
            return True
        return False

    async def heartbeat(self, device_id: str) -> bool:
        return device_id in self._devices


class MockAgentRouter:
    """Mock agent router for testing."""

    def __init__(self):
        self._rules: list[Any] = []

    def list_rules(self) -> list[Any]:
        return self._rules

    async def route(self, channel: str, content: str) -> MagicMock:
        result = MagicMock()
        result.agent_id = "agent-001"
        result.rule_id = "rule-001"
        return result


class MockRequestHandler:
    """Mock HTTP request handler."""

    def __init__(self, body: Optional[dict] = None, headers: Optional[dict] = None):
        self._body = body
        self.headers = headers or {}

    def read_body(self):
        return json.dumps(self._body).encode() if self._body else b"{}"


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def mock_device_registry():
    """Create mock device registry with sample data."""
    registry = MockDeviceRegistry()
    registry._devices["device-001"] = MockDeviceNode()
    registry._devices["device-002"] = MockDeviceNode(
        device_id="device-002",
        name="Mobile Device",
        device_type="phone",
    )
    return registry


@pytest.fixture
def mock_agent_router():
    """Create mock agent router."""
    return MockAgentRouter()


@pytest.fixture
def handler(mock_server_context, mock_device_registry, mock_agent_router):
    """Create handler with mocked dependencies."""
    h = GatewayHandler(mock_server_context)
    h._device_registry = mock_device_registry
    h._agent_router = mock_agent_router
    return h


# ===========================================================================
# Handler Tests
# ===========================================================================


class TestGatewayHandlerRouting:
    """Test request routing."""

    def test_can_handle_gateway_paths(self, handler):
        """Test that handler recognizes gateway paths."""
        assert handler.can_handle("/api/v1/gateway/devices")
        assert handler.can_handle("/api/v1/gateway/devices/device-001")
        assert handler.can_handle("/api/v1/gateway/channels")
        assert handler.can_handle("/api/v1/gateway/routing/stats")
        assert handler.can_handle("/api/v1/gateway/routing/rules")
        assert handler.can_handle("/api/v1/gateway/messages/route")

    def test_cannot_handle_other_paths(self, handler):
        """Test that handler rejects non-gateway paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/computer-use/tasks")
        assert not handler.can_handle("/api/gateway")  # Missing v1


class TestListDevices:
    """Test list devices endpoint."""

    def test_list_devices_success(self, handler, mock_device_registry):
        """Test listing devices returns correct format."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle(
                "/api/v1/gateway/devices",
                {},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "devices" in body
        assert "total" in body
        assert body["total"] == 2

    def test_list_devices_with_type_filter(self, handler, mock_device_registry):
        """Test listing devices with type filter."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle(
                "/api/v1/gateway/devices",
                {"type": "laptop"},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 200


class TestGetDevice:
    """Test get single device endpoint."""

    def test_get_device_success(self, handler, mock_device_registry):
        """Test getting a specific device."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle(
                "/api/v1/gateway/devices/device-001",
                {},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "device" in body
        assert body["device"]["device_id"] == "device-001"

    def test_get_device_not_found(self, handler):
        """Test getting non-existent device returns 404."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle(
                "/api/v1/gateway/devices/nonexistent",
                {},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 404


class TestRegisterDevice:
    """Test register device endpoint."""

    def test_register_device_success(self, handler, mock_device_registry):
        """Test registering a new device."""
        mock_handler = MockRequestHandler(
            body={
                "name": "New Device",
                "device_type": "tablet",
            }
        )

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(
                handler,
                "read_json_body",
                return_value={
                    "name": "New Device",
                    "device_type": "tablet",
                },
            ):
                result = handler.handle_post(
                    "/api/v1/gateway/devices",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 201

        body = json.loads(result.body)
        assert "device_id" in body
        assert "message" in body

    def test_register_device_missing_name(self, handler):
        """Test registering device without name fails."""
        mock_handler = MockRequestHandler(body={})

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(handler, "read_json_body", return_value={}):
                result = handler.handle_post(
                    "/api/v1/gateway/devices",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 400


class TestUnregisterDevice:
    """Test unregister device endpoint."""

    def test_unregister_device_success(self, handler, mock_device_registry):
        """Test unregistering a device."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle_delete(
                "/api/v1/gateway/devices/device-001",
                {},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 200

    def test_unregister_device_not_found(self, handler):
        """Test unregistering non-existent device returns 404."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle_delete(
                "/api/v1/gateway/devices/nonexistent",
                {},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 404


class TestDeviceHeartbeat:
    """Test device heartbeat endpoint."""

    def test_heartbeat_success(self, handler, mock_device_registry):
        """Test device heartbeat."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle_post(
                "/api/v1/gateway/devices/device-001/heartbeat",
                {},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 200

    def test_heartbeat_device_not_found(self, handler):
        """Test heartbeat for non-existent device returns 404."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle_post(
                "/api/v1/gateway/devices/nonexistent/heartbeat",
                {},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 404


class TestListChannels:
    """Test list channels endpoint."""

    def test_list_channels_success(self, handler):
        """Test listing available channels."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle(
                "/api/v1/gateway/channels",
                {},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "channels" in body
        assert "total" in body


class TestRoutingStats:
    """Test routing statistics endpoint."""

    def test_routing_stats_success(self, handler):
        """Test getting routing statistics."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle(
                "/api/v1/gateway/routing/stats",
                {},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "stats" in body


class TestListRoutingRules:
    """Test list routing rules endpoint."""

    def test_list_rules_success(self, handler):
        """Test listing routing rules."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle(
                "/api/v1/gateway/routing/rules",
                {},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "rules" in body
        assert "total" in body


class TestRouteMessage:
    """Test message routing endpoint."""

    def test_route_message_success(self, handler):
        """Test routing a message."""
        mock_handler = MockRequestHandler(
            body={
                "channel": "slack",
                "content": "Test message",
            }
        )

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(
                handler,
                "read_json_body",
                return_value={
                    "channel": "slack",
                    "content": "Test message",
                },
            ):
                result = handler.handle_post(
                    "/api/v1/gateway/messages/route",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert body["routed"] is True

    def test_route_message_missing_channel(self, handler):
        """Test routing message without channel fails."""
        mock_handler = MockRequestHandler(body={"content": "Test"})

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(handler, "read_json_body", return_value={"content": "Test"}):
                result = handler.handle_post(
                    "/api/v1/gateway/messages/route",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 400

    def test_route_message_missing_content(self, handler):
        """Test routing message without content fails."""
        mock_handler = MockRequestHandler(body={"channel": "slack"})

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(handler, "read_json_body", return_value={"channel": "slack"}):
                result = handler.handle_post(
                    "/api/v1/gateway/messages/route",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 400


class TestModuleNotAvailable:
    """Test behavior when gateway module is not available."""

    def test_returns_503_when_module_unavailable(self, mock_server_context):
        """Test that 503 is returned when gateway module unavailable."""
        with patch("aragora.server.handlers.gateway_handler.GATEWAY_AVAILABLE", False):
            h = GatewayHandler(mock_server_context)
            mock_handler = MockRequestHandler()

            result = h.handle("/api/v1/gateway/devices", {}, mock_handler)

            assert result is not None
            assert result.status_code == 503


class TestRBACProtection:
    """Test RBAC permission enforcement."""

    def test_unauthenticated_returns_401(self, handler):
        """Test that unauthenticated requests return 401."""
        mock_handler = MockRequestHandler()

        from aragora.server.handlers.base import error_response

        with patch.object(
            handler, "_check_rbac_permission", return_value=error_response("Not authenticated", 401)
        ):
            result = handler.handle("/api/v1/gateway/devices", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401

    def test_unauthorized_returns_403(self, handler):
        """Test that unauthorized requests return 403."""
        mock_handler = MockRequestHandler()

        from aragora.server.handlers.base import error_response

        with patch.object(
            handler, "_check_rbac_permission", return_value=error_response("Permission denied", 403)
        ):
            result = handler.handle("/api/v1/gateway/devices", {}, mock_handler)

        assert result is not None
        assert result.status_code == 403
