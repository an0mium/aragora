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


# ===========================================================================
# Circuit Breaker Tests
# ===========================================================================


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_exists(self):
        """Test that circuit breaker can be retrieved."""
        from aragora.server.handlers.gateway_handler import get_gateway_circuit_breaker

        cb = get_gateway_circuit_breaker()
        assert cb is not None
        assert cb.name == "gateway_handler"

    def test_circuit_breaker_status(self):
        """Test that circuit breaker status can be retrieved."""
        from aragora.server.handlers.gateway_handler import (
            get_gateway_circuit_breaker_status,
        )

        status = get_gateway_circuit_breaker_status()
        assert isinstance(status, dict)
        # Status dict has config, entity_mode, and single_mode keys
        assert "config" in status or "single_mode" in status

    def test_circuit_breaker_reset(self):
        """Test that circuit breaker can be reset."""
        from aragora.server.handlers.gateway_handler import (
            get_gateway_circuit_breaker,
            reset_gateway_circuit_breaker,
        )

        cb = get_gateway_circuit_breaker()
        reset_gateway_circuit_breaker()
        # After reset, internal state should be cleared
        assert cb._single_failures == 0

    def test_circuit_breaker_threshold(self):
        """Test circuit breaker has correct failure threshold."""
        from aragora.server.handlers.gateway_handler import get_gateway_circuit_breaker

        cb = get_gateway_circuit_breaker()
        assert cb.failure_threshold == 5

    def test_circuit_breaker_cooldown(self):
        """Test circuit breaker has correct cooldown."""
        from aragora.server.handlers.gateway_handler import get_gateway_circuit_breaker

        cb = get_gateway_circuit_breaker()
        assert cb.cooldown_seconds == 30.0


# ===========================================================================
# Device Validation Tests
# ===========================================================================


class TestDeviceValidation:
    """Test device input validation."""

    def test_register_device_empty_name(self, handler):
        """Test registering device with empty name fails."""
        mock_handler = MockRequestHandler(body={"name": ""})

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(handler, "read_json_body", return_value={"name": ""}):
                result = handler.handle_post(
                    "/api/v1/gateway/devices",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 400

    def test_register_device_with_capabilities(self, handler, mock_device_registry):
        """Test registering device with capabilities."""
        mock_handler = MockRequestHandler(
            body={
                "name": "Capable Device",
                "device_type": "laptop",
                "capabilities": ["browser", "terminal", "voice"],
            }
        )

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(
                handler,
                "read_json_body",
                return_value={
                    "name": "Capable Device",
                    "device_type": "laptop",
                    "capabilities": ["browser", "terminal", "voice"],
                },
            ):
                result = handler.handle_post(
                    "/api/v1/gateway/devices",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 201

    def test_register_device_with_metadata(self, handler, mock_device_registry):
        """Test registering device with metadata."""
        mock_handler = MockRequestHandler(
            body={
                "name": "Metadata Device",
                "device_type": "tablet",
                "metadata": {"os": "android", "version": "14"},
            }
        )

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(
                handler,
                "read_json_body",
                return_value={
                    "name": "Metadata Device",
                    "device_type": "tablet",
                    "metadata": {"os": "android", "version": "14"},
                },
            ):
                result = handler.handle_post(
                    "/api/v1/gateway/devices",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 201

    def test_register_device_with_allowed_channels(self, handler, mock_device_registry):
        """Test registering device with allowed channels."""
        mock_handler = MockRequestHandler(
            body={
                "name": "Channel Device",
                "allowed_channels": ["slack", "email", "telegram"],
            }
        )

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(
                handler,
                "read_json_body",
                return_value={
                    "name": "Channel Device",
                    "allowed_channels": ["slack", "email", "telegram"],
                },
            ):
                result = handler.handle_post(
                    "/api/v1/gateway/devices",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 201

    def test_register_device_invalid_json_body(self, handler):
        """Test registering device with invalid JSON returns 400."""
        mock_handler = MockRequestHandler(body={})

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(handler, "read_json_body", return_value=None):
                result = handler.handle_post(
                    "/api/v1/gateway/devices",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Device Status Filter Tests
# ===========================================================================


class TestDeviceStatusFilter:
    """Test device listing with status filters."""

    def test_list_devices_with_status_filter(self, handler, mock_device_registry):
        """Test listing devices with status filter."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle(
                "/api/v1/gateway/devices",
                {"status": "online"},
                mock_handler,
            )

        assert result is not None
        assert result.status_code == 200

    def test_list_devices_invalid_status(self, handler, mock_device_registry):
        """Test listing devices with invalid status still works."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle(
                "/api/v1/gateway/devices",
                {"status": "invalid_status"},
                mock_handler,
            )

        # Invalid status is ignored, returns all devices
        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Routing Rules Tests
# ===========================================================================


class TestRoutingRulesExtended:
    """Extended tests for routing rules endpoint."""

    def test_list_rules_with_rules(self, handler, mock_agent_router):
        """Test listing routing rules when rules exist."""
        # Add some mock rules
        mock_rule = MagicMock()
        mock_rule.id = "rule-1"
        mock_rule.channel = "slack"
        mock_rule.pattern = ".*"
        mock_rule.agent_id = "agent-1"
        mock_agent_router._rules = [mock_rule]

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
        assert body["total"] >= 0

    def test_routing_stats_format(self, handler):
        """Test routing stats returns expected format."""
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
        assert "total_rules" in body["stats"]
        assert "messages_routed" in body["stats"]
        assert "routing_errors" in body["stats"]


# ===========================================================================
# Message Routing Validation Tests
# ===========================================================================


class TestMessageRoutingValidation:
    """Test message routing input validation."""

    def test_route_message_empty_channel(self, handler):
        """Test routing message with empty channel fails."""
        mock_handler = MockRequestHandler(body={"channel": "", "content": "Test"})

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"channel": "", "content": "Test"},
            ):
                result = handler.handle_post(
                    "/api/v1/gateway/messages/route",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 400

    def test_route_message_empty_content(self, handler):
        """Test routing message with empty content fails."""
        mock_handler = MockRequestHandler(body={"channel": "slack", "content": ""})

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"channel": "slack", "content": ""},
            ):
                result = handler.handle_post(
                    "/api/v1/gateway/messages/route",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 400

    def test_route_message_invalid_json(self, handler):
        """Test routing message with invalid JSON fails."""
        mock_handler = MockRequestHandler(body={})

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            with patch.object(handler, "read_json_body", return_value=None):
                result = handler.handle_post(
                    "/api/v1/gateway/messages/route",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Channel Listing Tests
# ===========================================================================


class TestChannelListingExtended:
    """Extended tests for channel listing."""

    def test_list_channels_returns_expected_channels(self, handler):
        """Test that channel list contains expected channels."""
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
        channel_names = [c["name"] for c in body["channels"]]
        assert "slack" in channel_names
        assert "email" in channel_names

    def test_list_channels_has_status(self, handler):
        """Test that each channel has a status field."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle(
                "/api/v1/gateway/channels",
                {},
                mock_handler,
            )

        assert result is not None
        body = json.loads(result.body)

        for channel in body["channels"]:
            assert "status" in channel


# ===========================================================================
# Registry Unavailable Tests
# ===========================================================================


class TestRegistryUnavailable:
    """Test behavior when device registry is unavailable."""

    def test_list_devices_registry_unavailable(self, mock_server_context):
        """Test listing devices when registry unavailable."""
        h = GatewayHandler(mock_server_context)
        h._device_registry = None  # Simulate unavailable registry

        mock_handler = MockRequestHandler()

        with patch.object(h, "_check_rbac_permission", return_value=None):
            with patch.object(h, "_get_device_registry", return_value=None):
                result = h.handle("/api/v1/gateway/devices", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    def test_get_device_registry_unavailable(self, mock_server_context):
        """Test getting device when registry unavailable."""
        h = GatewayHandler(mock_server_context)

        mock_handler = MockRequestHandler()

        with patch.object(h, "_check_rbac_permission", return_value=None):
            with patch.object(h, "_get_device_registry", return_value=None):
                result = h.handle("/api/v1/gateway/devices/device-001", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# Router Unavailable Tests
# ===========================================================================


class TestRouterUnavailable:
    """Test behavior when agent router is unavailable."""

    def test_routing_stats_router_unavailable(self, mock_server_context):
        """Test routing stats when router unavailable."""
        h = GatewayHandler(mock_server_context)

        mock_handler = MockRequestHandler()

        with patch.object(h, "_check_rbac_permission", return_value=None):
            with patch.object(h, "_get_agent_router", return_value=None):
                result = h.handle("/api/v1/gateway/routing/stats", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    def test_list_rules_router_unavailable(self, mock_server_context):
        """Test listing rules when router unavailable."""
        h = GatewayHandler(mock_server_context)

        mock_handler = MockRequestHandler()

        with patch.object(h, "_check_rbac_permission", return_value=None):
            with patch.object(h, "_get_agent_router", return_value=None):
                result = h.handle("/api/v1/gateway/routing/rules", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    def test_route_message_router_unavailable(self, mock_server_context):
        """Test routing message when router unavailable."""
        h = GatewayHandler(mock_server_context)

        mock_handler = MockRequestHandler(body={"channel": "slack", "content": "Test"})

        with patch.object(h, "_check_rbac_permission", return_value=None):
            with patch.object(
                h,
                "read_json_body",
                return_value={"channel": "slack", "content": "Test"},
            ):
                with patch.object(h, "_get_agent_router", return_value=None):
                    result = h.handle_post("/api/v1/gateway/messages/route", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# Handler Method Routing Tests
# ===========================================================================


class TestHandlerMethodRouting:
    """Test that handlers correctly route to sub-methods."""

    def test_handle_post_returns_none_for_invalid_path(self, handler):
        """Test handle_post returns None for non-gateway paths."""
        mock_handler = MockRequestHandler()
        result = handler.handle_post("/api/v1/debates", {}, mock_handler)
        assert result is None

    def test_handle_delete_returns_none_for_invalid_path(self, handler):
        """Test handle_delete returns None for non-gateway paths."""
        mock_handler = MockRequestHandler()
        result = handler.handle_delete("/api/v1/debates", {}, mock_handler)
        assert result is None

    def test_handle_returns_none_for_unknown_gateway_path(self, handler):
        """Test handle returns None for unknown gateway sub-path."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/gateway/unknown", {}, mock_handler)

        assert result is None

    def test_handle_post_unknown_gateway_path(self, handler):
        """Test handle_post returns None for unknown gateway sub-path."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle_post("/api/v1/gateway/unknown", {}, mock_handler)

        assert result is None
