"""Tests for gateway handler (aragora/server/handlers/gateway_handler.py).

Covers all routes and behavior of the GatewayHandler class:
- can_handle() routing for all ROUTES and non-matching paths
- GET    /api/v1/gateway/devices          - List devices
- GET    /api/v1/gateway/devices/{id}     - Get device details
- POST   /api/v1/gateway/devices          - Register a device
- DELETE /api/v1/gateway/devices/{id}     - Unregister a device
- POST   /api/v1/gateway/devices/{id}/heartbeat - Device heartbeat
- GET    /api/v1/gateway/channels         - List channels
- GET    /api/v1/gateway/routing/stats    - Routing stats
- GET    /api/v1/gateway/routing/rules    - Routing rules
- POST   /api/v1/gateway/messages/route   - Route a message
- Gateway unavailable (503) responses
- Device registry unavailable (503)
- Agent router unavailable (503)
- RBAC permission checks
- Input validation (missing fields, invalid JSON)
- Not-found device error paths
- Circuit breaker helper functions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.gateway_handler import (
    GatewayHandler,
    get_gateway_circuit_breaker,
    get_gateway_circuit_breaker_status,
    reset_gateway_circuit_breaker,
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
    """Mock HTTP handler used by BaseHandler.read_json_body."""

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
        self.client_address = ("127.0.0.1", 54321)


class MockHTTPHandlerInvalidJSON:
    """Mock HTTP handler returning invalid JSON."""

    def __init__(self):
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"NOT-JSON"
        self.headers = {
            "Content-Length": "8",
            "Content-Type": "application/json",
        }
        self.client_address = ("127.0.0.1", 54321)


class MockHTTPHandlerNoBody:
    """Mock HTTP handler with no body content."""

    def __init__(self):
        self.rfile = MagicMock()
        self.rfile.read.return_value = b""
        self.headers = {
            "Content-Length": "0",
            "Content-Type": "application/json",
        }
        self.client_address = ("127.0.0.1", 54321)


# ---------------------------------------------------------------------------
# Mock gateway classes
# ---------------------------------------------------------------------------


class MockDeviceStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    PAIRED = "paired"
    BLOCKED = "blocked"


@dataclass
class MockDeviceNode:
    device_id: str = ""
    name: str = ""
    device_type: str = ""
    capabilities: list[str] = field(default_factory=list)
    status: MockDeviceStatus = MockDeviceStatus.OFFLINE
    paired_at: float | None = None
    last_seen: float | None = None
    allowed_channels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockRoutingRule:
    id: str = "rule-1"
    channel: str = "slack"
    pattern: str = "*"
    agent_id: str = "claude"


@dataclass
class MockRouteResult:
    agent_id: str = "claude"
    rule_id: str = "rule-1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_gateway_available():
    """Ensure GATEWAY_AVAILABLE is True by default for most tests."""
    with patch(
        "aragora.server.handlers.gateway_handler.GATEWAY_AVAILABLE", True
    ):
        yield


# Save the original method before any patching
_original_check_rbac_permission = GatewayHandler._check_rbac_permission


@pytest.fixture(autouse=True)
def _patch_rbac_check():
    """Bypass the internal _check_rbac_permission for most tests."""
    with patch.object(GatewayHandler, "_check_rbac_permission", return_value=None):
        yield


@pytest.fixture
def handler():
    """Create a GatewayHandler instance with empty server context."""
    return GatewayHandler(server_context={})


@pytest.fixture
def mock_registry():
    """Create a mock DeviceRegistry."""
    registry = MagicMock()
    registry.list_devices = AsyncMock(return_value=[])
    registry.get = AsyncMock(return_value=None)
    registry.register = AsyncMock(return_value="dev-abc12345")
    registry.unregister = AsyncMock(return_value=True)
    registry.heartbeat = AsyncMock(return_value=True)
    return registry


@pytest.fixture
def mock_router():
    """Create a mock AgentRouter."""
    router = MagicMock()
    router.list_rules = AsyncMock(return_value=[])
    router.route = AsyncMock(return_value=MockRouteResult())
    return router


@pytest.fixture
def handler_with_registry(handler, mock_registry):
    """Handler with a pre-configured device registry."""
    handler._device_registry = mock_registry
    return handler


@pytest.fixture
def handler_with_router(handler, mock_router):
    """Handler with a pre-configured agent router."""
    handler._agent_router = mock_router
    return handler


@pytest.fixture
def handler_full(handler, mock_registry, mock_router):
    """Handler with both registry and router configured."""
    handler._device_registry = mock_registry
    handler._agent_router = mock_router
    return handler


# ===========================================================================
# can_handle routing tests
# ===========================================================================


class TestCanHandle:
    """Test the can_handle path routing."""

    def test_handles_gateway_devices(self, handler):
        assert handler.can_handle("/api/v1/gateway/devices") is True

    def test_handles_gateway_devices_with_id(self, handler):
        assert handler.can_handle("/api/v1/gateway/devices/dev-123") is True

    def test_handles_gateway_channels(self, handler):
        assert handler.can_handle("/api/v1/gateway/channels") is True

    def test_handles_gateway_routing_stats(self, handler):
        assert handler.can_handle("/api/v1/gateway/routing/stats") is True

    def test_handles_gateway_routing_rules(self, handler):
        assert handler.can_handle("/api/v1/gateway/routing/rules") is True

    def test_handles_gateway_messages_route(self, handler):
        assert handler.can_handle("/api/v1/gateway/messages/route") is True

    def test_rejects_non_gateway_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/gate") is False

    def test_handles_heartbeat_path(self, handler):
        assert handler.can_handle("/api/v1/gateway/devices/dev-1/heartbeat") is True


# ===========================================================================
# GET /api/v1/gateway/devices
# ===========================================================================


class TestListDevices:
    """Test GET /api/v1/gateway/devices."""

    def test_list_devices_empty(self, handler_with_registry):
        http = MockHTTPHandler()
        with patch("aragora.server.handlers.gateway_handler.run_async", side_effect=lambda c: []):
            result = handler_with_registry.handle(
                "/api/v1/gateway/devices", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["devices"] == []
        assert body["total"] == 0

    def test_list_devices_returns_devices(self, handler_with_registry, mock_registry):
        device = MockDeviceNode(
            device_id="dev-1",
            name="Laptop",
            device_type="laptop",
            capabilities=["browser", "shell"],
            status=MockDeviceStatus.PAIRED,
            paired_at=1700000000.0,
            last_seen=1700000100.0,
        )
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: [device],
        ):
            result = handler_with_registry.handle(
                "/api/v1/gateway/devices", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["devices"][0]["device_id"] == "dev-1"
        assert body["devices"][0]["name"] == "Laptop"
        assert body["devices"][0]["device_type"] == "laptop"
        assert body["devices"][0]["capabilities"] == ["browser", "shell"]
        assert body["devices"][0]["status"] == "paired"

    def test_list_devices_with_status_filter(self, handler_with_registry):
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: [],
        ), patch(
            "aragora.server.handlers.gateway_handler.DeviceStatus",
            MockDeviceStatus,
        ):
            result = handler_with_registry.handle(
                "/api/v1/gateway/devices", {"status": "online"}, http
            )
        assert _status(result) == 200

    def test_list_devices_with_type_filter(self, handler_with_registry):
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: [],
        ):
            result = handler_with_registry.handle(
                "/api/v1/gateway/devices", {"type": "laptop"}, http
            )
        assert _status(result) == 200

    def test_list_devices_with_invalid_status_filter(self, handler_with_registry):
        """Invalid status value should be silently ignored (no crash)."""
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: [],
        ), patch(
            "aragora.server.handlers.gateway_handler.DeviceStatus",
            MockDeviceStatus,
        ):
            result = handler_with_registry.handle(
                "/api/v1/gateway/devices", {"status": "invalid_status"}, http
            )
        assert _status(result) == 200

    def test_list_devices_registry_unavailable(self, handler):
        """Returns 503 when device registry is not available."""
        http = MockHTTPHandler()
        with patch.object(handler, "_get_device_registry", return_value=None):
            result = handler.handle("/api/v1/gateway/devices", {}, http)
        assert _status(result) == 503


# ===========================================================================
# GET /api/v1/gateway/devices/{id}
# ===========================================================================


class TestGetDevice:
    """Test GET /api/v1/gateway/devices/{id}."""

    def test_get_device_found(self, handler_with_registry):
        device = MockDeviceNode(
            device_id="dev-1",
            name="Phone",
            device_type="phone",
            capabilities=["camera"],
            status=MockDeviceStatus.ONLINE,
            paired_at=1700000000.0,
            last_seen=1700000100.0,
            allowed_channels=["slack"],
            metadata={"os": "android"},
        )
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: device,
        ):
            result = handler_with_registry.handle(
                "/api/v1/gateway/devices/dev-1", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["device"]["device_id"] == "dev-1"
        assert body["device"]["name"] == "Phone"
        assert body["device"]["allowed_channels"] == ["slack"]
        assert body["device"]["metadata"] == {"os": "android"}

    def test_get_device_not_found(self, handler_with_registry):
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: None,
        ):
            result = handler_with_registry.handle(
                "/api/v1/gateway/devices/dev-nonexist", {}, http
            )
        assert _status(result) == 404

    def test_get_device_registry_unavailable(self, handler):
        http = MockHTTPHandler()
        with patch.object(handler, "_get_device_registry", return_value=None):
            result = handler.handle("/api/v1/gateway/devices/dev-1", {}, http)
        assert _status(result) == 503

    def test_get_device_path_devices_only_returns_none(self, handler_with_registry):
        """Path ending with 'devices' and no id segment should list, not get."""
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: [],
        ):
            result = handler_with_registry.handle(
                "/api/v1/gateway/devices", {}, http
            )
        # Should return list response (200), not a get-device response
        assert _status(result) == 200


# ===========================================================================
# POST /api/v1/gateway/devices (register)
# ===========================================================================


class TestRegisterDevice:
    """Test POST /api/v1/gateway/devices."""

    def test_register_device_success(self, handler_with_registry):
        http = MockHTTPHandler(body={"name": "My Laptop", "device_type": "laptop"})
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: "dev-abc123",
        ), patch(
            "aragora.server.handlers.gateway_handler.DeviceNode",
        ) as mock_dn:
            mock_dn.return_value = MagicMock()
            result = handler_with_registry.handle_post(
                "/api/v1/gateway/devices", {}, http
            )
        assert _status(result) == 201
        body = _body(result)
        assert body["device_id"] == "dev-abc123"
        assert "registered" in body["message"].lower()

    def test_register_device_missing_name(self, handler_with_registry):
        http = MockHTTPHandler(body={"device_type": "laptop"})
        result = handler_with_registry.handle_post(
            "/api/v1/gateway/devices", {}, http
        )
        assert _status(result) == 400
        assert "name" in _body(result).get("error", "").lower()

    def test_register_device_invalid_json(self, handler_with_registry):
        http = MockHTTPHandlerInvalidJSON()
        result = handler_with_registry.handle_post(
            "/api/v1/gateway/devices", {}, http
        )
        assert _status(result) == 400

    def test_register_device_empty_body(self, handler_with_registry):
        http = MockHTTPHandlerNoBody()
        result = handler_with_registry.handle_post(
            "/api/v1/gateway/devices", {}, http
        )
        # Empty body => name is missing
        assert _status(result) == 400

    def test_register_device_with_optional_fields(self, handler_with_registry):
        http = MockHTTPHandler(body={
            "name": "Server",
            "device_type": "server",
            "device_id": "custom-id",
            "capabilities": ["shell"],
            "allowed_channels": ["slack"],
            "metadata": {"region": "us-east-1"},
        })
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: "custom-id",
        ), patch(
            "aragora.server.handlers.gateway_handler.DeviceNode",
        ) as mock_dn:
            mock_dn.return_value = MagicMock()
            result = handler_with_registry.handle_post(
                "/api/v1/gateway/devices", {}, http
            )
        assert _status(result) == 201

    def test_register_device_registry_unavailable(self, handler):
        http = MockHTTPHandler(body={"name": "Test"})
        with patch.object(handler, "_get_device_registry", return_value=None):
            result = handler.handle_post("/api/v1/gateway/devices", {}, http)
        assert _status(result) == 503

    def test_register_device_name_is_empty_string(self, handler_with_registry):
        """Empty name string should be rejected."""
        http = MockHTTPHandler(body={"name": "", "device_type": "laptop"})
        result = handler_with_registry.handle_post(
            "/api/v1/gateway/devices", {}, http
        )
        assert _status(result) == 400


# ===========================================================================
# DELETE /api/v1/gateway/devices/{id}
# ===========================================================================


class TestUnregisterDevice:
    """Test DELETE /api/v1/gateway/devices/{id}."""

    def test_unregister_success(self, handler_with_registry):
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: True,
        ):
            result = handler_with_registry.handle_delete(
                "/api/v1/gateway/devices/dev-1", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert "unregistered" in body.get("message", "").lower()

    def test_unregister_not_found(self, handler_with_registry):
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: False,
        ):
            result = handler_with_registry.handle_delete(
                "/api/v1/gateway/devices/dev-missing", {}, http
            )
        assert _status(result) == 404

    def test_unregister_registry_unavailable(self, handler):
        http = MockHTTPHandler()
        with patch.object(handler, "_get_device_registry", return_value=None):
            result = handler.handle_delete("/api/v1/gateway/devices/dev-1", {}, http)
        assert _status(result) == 503

    def test_unregister_path_without_id_returns_none(self, handler_with_registry):
        """DELETE on /api/v1/gateway/devices (no id) returns None (not handled)."""
        http = MockHTTPHandler()
        result = handler_with_registry.handle_delete(
            "/api/v1/gateway/devices", {}, http
        )
        # Path doesn't start with /api/v1/gateway/devices/ so handle_delete
        # returns None (no match for delete)
        assert result is None


# ===========================================================================
# POST /api/v1/gateway/devices/{id}/heartbeat
# ===========================================================================


class TestHeartbeat:
    """Test POST /api/v1/gateway/devices/{id}/heartbeat."""

    def test_heartbeat_success(self, handler_with_registry):
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: True,
        ):
            result = handler_with_registry.handle_post(
                "/api/v1/gateway/devices/dev-1/heartbeat", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "ok"

    def test_heartbeat_device_not_found(self, handler_with_registry):
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: False,
        ):
            result = handler_with_registry.handle_post(
                "/api/v1/gateway/devices/dev-unknown/heartbeat", {}, http
            )
        assert _status(result) == 404

    def test_heartbeat_registry_unavailable(self, handler):
        http = MockHTTPHandler()
        with patch.object(handler, "_get_device_registry", return_value=None):
            result = handler.handle_post(
                "/api/v1/gateway/devices/dev-1/heartbeat", {}, http
            )
        assert _status(result) == 503


# ===========================================================================
# GET /api/v1/gateway/channels
# ===========================================================================


class TestListChannels:
    """Test GET /api/v1/gateway/channels."""

    def test_list_channels(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/channels", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert "channels" in body
        assert body["total"] == 4
        channel_names = [c["name"] for c in body["channels"]]
        assert "slack" in channel_names
        assert "email" in channel_names
        assert "telegram" in channel_names
        assert "whatsapp" in channel_names

    def test_channels_all_available(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/channels", {}, http)
        body = _body(result)
        for ch in body["channels"]:
            assert ch["status"] == "available"


# ===========================================================================
# GET /api/v1/gateway/routing/stats
# ===========================================================================


class TestRoutingStats:
    """Test GET /api/v1/gateway/routing/stats."""

    def test_routing_stats_success(self, handler_with_router):
        http = MockHTTPHandler()
        result = handler_with_router.handle(
            "/api/v1/gateway/routing/stats", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert "stats" in body
        stats = body["stats"]
        assert "total_rules" in stats
        assert "messages_routed" in stats
        assert "routing_errors" in stats

    def test_routing_stats_router_unavailable(self, handler):
        http = MockHTTPHandler()
        with patch.object(handler, "_get_agent_router", return_value=None):
            result = handler.handle("/api/v1/gateway/routing/stats", {}, http)
        assert _status(result) == 503


# ===========================================================================
# GET /api/v1/gateway/routing/rules
# ===========================================================================


class TestListRules:
    """Test GET /api/v1/gateway/routing/rules."""

    def test_list_rules_empty(self, handler_with_router):
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: [],
        ):
            result = handler_with_router.handle(
                "/api/v1/gateway/routing/rules", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["rules"] == []
        assert body["total"] == 0

    def test_list_rules_with_rules(self, handler_with_router, mock_router):
        rules = [
            MockRoutingRule(id="r-1", channel="slack", pattern="*", agent_id="claude"),
            MockRoutingRule(id="r-2", channel="email", pattern="*.com", agent_id="gpt"),
        ]
        mock_router.list_rules = AsyncMock(return_value=rules)
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: rules,
        ):
            result = handler_with_router.handle(
                "/api/v1/gateway/routing/rules", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert body["rules"][0]["id"] == "r-1"
        assert body["rules"][1]["agent_id"] == "gpt"

    def test_list_rules_router_unavailable(self, handler):
        http = MockHTTPHandler()
        with patch.object(handler, "_get_agent_router", return_value=None):
            result = handler.handle("/api/v1/gateway/routing/rules", {}, http)
        assert _status(result) == 503

    def test_list_rules_no_list_rules_method(self, handler_with_router, mock_router):
        """If router does not have list_rules, return empty list."""
        del mock_router.list_rules
        http = MockHTTPHandler()
        result = handler_with_router.handle(
            "/api/v1/gateway/routing/rules", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0

    def test_list_rules_with_sync_list_rules(self, handler_with_router, mock_router):
        """Router.list_rules may return a sync result."""
        rules = [MockRoutingRule(id="sync-1")]
        mock_router.list_rules = MagicMock(return_value=rules)
        http = MockHTTPHandler()
        result = handler_with_router.handle(
            "/api/v1/gateway/routing/rules", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1

    def test_list_rules_rule_without_attributes(self, handler_with_router, mock_router):
        """Rules missing attributes fall back to getattr defaults."""
        bare_rule = MagicMock(spec=[])  # no attributes at all
        mock_router.list_rules = MagicMock(return_value=[bare_rule])
        http = MockHTTPHandler()
        result = handler_with_router.handle(
            "/api/v1/gateway/routing/rules", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        # id falls back to str(0) i.e. "0"
        assert body["rules"][0]["id"] == "0"


# ===========================================================================
# POST /api/v1/gateway/messages/route
# ===========================================================================


class TestRouteMessage:
    """Test POST /api/v1/gateway/messages/route."""

    def test_route_message_success(self, handler_with_router):
        route_result = MockRouteResult(agent_id="claude", rule_id="rule-1")
        http = MockHTTPHandler(body={"channel": "slack", "content": "Hello"})
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: route_result,
        ):
            result = handler_with_router.handle_post(
                "/api/v1/gateway/messages/route", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["routed"] is True
        assert body["agent_id"] == "claude"
        assert body["rule_id"] == "rule-1"

    def test_route_message_missing_channel(self, handler_with_router):
        http = MockHTTPHandler(body={"content": "Hello"})
        result = handler_with_router.handle_post(
            "/api/v1/gateway/messages/route", {}, http
        )
        assert _status(result) == 400
        assert "channel" in _body(result).get("error", "").lower()

    def test_route_message_missing_content(self, handler_with_router):
        http = MockHTTPHandler(body={"channel": "slack"})
        result = handler_with_router.handle_post(
            "/api/v1/gateway/messages/route", {}, http
        )
        assert _status(result) == 400
        assert "content" in _body(result).get("error", "").lower()

    def test_route_message_missing_both(self, handler_with_router):
        http = MockHTTPHandler(body={})
        result = handler_with_router.handle_post(
            "/api/v1/gateway/messages/route", {}, http
        )
        assert _status(result) == 400

    def test_route_message_invalid_json(self, handler_with_router):
        http = MockHTTPHandlerInvalidJSON()
        result = handler_with_router.handle_post(
            "/api/v1/gateway/messages/route", {}, http
        )
        assert _status(result) == 400

    def test_route_message_router_unavailable(self, handler):
        http = MockHTTPHandler(body={"channel": "slack", "content": "test"})
        with patch.object(handler, "_get_agent_router", return_value=None):
            result = handler.handle_post(
                "/api/v1/gateway/messages/route", {}, http
            )
        assert _status(result) == 503

    def test_route_message_result_without_attributes(self, handler_with_router):
        """Route result missing agent_id/rule_id returns None for those fields."""
        bare_result = MagicMock(spec=[])
        http = MockHTTPHandler(body={"channel": "slack", "content": "Hello"})
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: bare_result,
        ):
            result = handler_with_router.handle_post(
                "/api/v1/gateway/messages/route", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["routed"] is True
        assert body["agent_id"] is None
        assert body["rule_id"] is None


# ===========================================================================
# Gateway unavailable (503) for all methods
# ===========================================================================


class TestGatewayUnavailable:
    """Test 503 responses when GATEWAY_AVAILABLE is False."""

    @pytest.fixture(autouse=True)
    def _disable_gateway(self):
        with patch(
            "aragora.server.handlers.gateway_handler.GATEWAY_AVAILABLE", False
        ):
            yield

    def test_handle_get_503(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/devices", {}, http)
        assert _status(result) == 503

    def test_handle_post_503(self, handler):
        http = MockHTTPHandler(body={"name": "Test"})
        result = handler.handle_post("/api/v1/gateway/devices", {}, http)
        assert _status(result) == 503

    def test_handle_delete_503(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/gateway/devices/dev-1", {}, http)
        assert _status(result) == 503


# ===========================================================================
# Unknown route returns None
# ===========================================================================


class TestUnknownRoutes:
    """Test that unknown routes return None."""

    def test_handle_unknown_path(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/other/endpoint", {}, http)
        assert result is None

    def test_handle_post_unknown_path(self, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/other/endpoint", {}, http)
        assert result is None

    def test_handle_delete_unknown_path(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/other/endpoint", {}, http)
        assert result is None

    def test_handle_post_unmatched_gateway_path(self, handler_full):
        """POST to a gateway path that doesn't match any specific endpoint."""
        http = MockHTTPHandler(body={})
        result = handler_full.handle_post(
            "/api/v1/gateway/unknown/thing", {}, http
        )
        assert result is None

    def test_handle_get_routing_base(self, handler_full):
        """GET /api/v1/gateway/routing (without /stats or /rules) returns None."""
        http = MockHTTPHandler()
        result = handler_full.handle("/api/v1/gateway/routing", {}, http)
        assert result is None

    def test_handle_get_messages_base(self, handler_full):
        """GET /api/v1/gateway/messages returns None."""
        http = MockHTTPHandler()
        result = handler_full.handle("/api/v1/gateway/messages", {}, http)
        assert result is None


# ===========================================================================
# Circuit breaker helper functions
# ===========================================================================


class TestCircuitBreaker:
    """Test the module-level circuit breaker functions."""

    def test_get_circuit_breaker_returns_instance(self):
        cb = get_gateway_circuit_breaker()
        assert cb is not None
        assert cb.name == "gateway_handler"

    def test_get_circuit_breaker_status_returns_dict(self):
        status = get_gateway_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_reset_circuit_breaker(self):
        cb = get_gateway_circuit_breaker()
        # Simulate some failures
        cb._single_failures = 3
        reset_gateway_circuit_breaker()
        assert cb._single_failures == 0
        assert cb._single_open_at == 0.0
        assert cb._single_successes == 0
        assert cb._single_half_open_calls == 0


# ===========================================================================
# RBAC permission checks
# ===========================================================================


class TestRBACPermissionChecks:
    """Test that internal RBAC permission checking works correctly.

    Uses the saved _original_check_rbac_permission to call the real
    method, since the autouse _patch_rbac_check fixture patches it.
    """

    def _call_real_check(self, handler, http_handler, permission):
        """Call the real _check_rbac_permission bypassing the autouse mock."""
        return _original_check_rbac_permission(handler, http_handler, permission)

    def test_rbac_not_available_dev_mode(self):
        """When RBAC is not available in dev mode, _check_rbac_permission returns None."""
        handler = GatewayHandler(server_context={})
        with patch(
            "aragora.server.handlers.gateway_handler.RBAC_AVAILABLE", False
        ), patch(
            "aragora.server.handlers.gateway_handler.rbac_fail_closed",
            return_value=False,
        ):
            result = self._call_real_check(handler, MagicMock(), "gateway:read")
        assert result is None

    def test_rbac_not_available_production_mode(self):
        """When RBAC is not available in production, return 503."""
        handler = GatewayHandler(server_context={})
        with patch(
            "aragora.server.handlers.gateway_handler.RBAC_AVAILABLE", False
        ), patch(
            "aragora.server.handlers.gateway_handler.rbac_fail_closed",
            return_value=True,
        ):
            result = self._call_real_check(handler, MagicMock(), "gateway:read")
        assert _status(result) == 503

    def test_rbac_no_auth_context(self):
        """When user is not authenticated, return 401."""
        handler = GatewayHandler(server_context={})
        with patch(
            "aragora.server.handlers.gateway_handler.RBAC_AVAILABLE", True
        ), patch.object(handler, "_get_auth_context", return_value=None):
            result = self._call_real_check(handler, MagicMock(), "gateway:read")
        assert _status(result) == 401

    def test_rbac_permission_denied(self):
        """When permission check fails, return 403."""
        handler = GatewayHandler(server_context={})
        mock_ctx = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False
        with patch(
            "aragora.server.handlers.gateway_handler.RBAC_AVAILABLE", True
        ), patch.object(
            handler, "_get_auth_context", return_value=mock_ctx
        ), patch(
            "aragora.server.handlers.gateway_handler.check_permission",
            return_value=mock_decision,
        ):
            result = self._call_real_check(handler, MagicMock(), "gateway:write")
        assert _status(result) == 403

    def test_rbac_permission_allowed(self):
        """When permission check passes, return None."""
        handler = GatewayHandler(server_context={})
        mock_ctx = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = True
        with patch(
            "aragora.server.handlers.gateway_handler.RBAC_AVAILABLE", True
        ), patch.object(
            handler, "_get_auth_context", return_value=mock_ctx
        ), patch(
            "aragora.server.handlers.gateway_handler.check_permission",
            return_value=mock_decision,
        ):
            result = self._call_real_check(handler, MagicMock(), "gateway:read")
        assert result is None


# ===========================================================================
# Gateway stores lazy initialization
# ===========================================================================


class TestGatewayStores:
    """Test lazy initialization of gateway stores, registry, and router."""

    def test_get_gateway_stores_not_available(self, handler):
        """Returns None when GATEWAY_AVAILABLE is False."""
        with patch(
            "aragora.server.handlers.gateway_handler.GATEWAY_AVAILABLE", False
        ):
            assert handler._get_gateway_stores() is None

    def test_get_gateway_stores_no_function(self, handler):
        """Returns None when get_canonical_gateway_stores is None."""
        with patch(
            "aragora.server.handlers.gateway_handler.get_canonical_gateway_stores",
            None,
        ):
            assert handler._get_gateway_stores() is None

    def test_get_gateway_stores_caches(self, handler):
        """Stores are cached after first call."""
        mock_stores = MagicMock()
        with patch(
            "aragora.server.handlers.gateway_handler.get_canonical_gateway_stores",
            return_value=mock_stores,
        ):
            result1 = handler._get_gateway_stores()
            result2 = handler._get_gateway_stores()
        assert result1 is result2
        assert result1 is mock_stores

    def test_get_device_registry_not_available(self, handler):
        with patch(
            "aragora.server.handlers.gateway_handler.GATEWAY_AVAILABLE", False
        ):
            assert handler._get_device_registry() is None

    def test_get_device_registry_creates_registry(self, handler):
        mock_stores = MagicMock()
        mock_store = MagicMock()
        mock_stores.gateway_store.return_value = mock_store
        with patch(
            "aragora.server.handlers.gateway_handler.get_canonical_gateway_stores",
            return_value=mock_stores,
        ), patch(
            "aragora.server.handlers.gateway_handler.DeviceRegistry",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            result = handler._get_device_registry()
        assert result is not None
        mock_cls.assert_called_once_with(store=mock_store)

    def test_get_device_registry_caches(self, handler):
        handler._device_registry = MagicMock()
        result = handler._get_device_registry()
        assert result is handler._device_registry

    def test_get_agent_router_not_available(self, handler):
        with patch(
            "aragora.server.handlers.gateway_handler.GATEWAY_AVAILABLE", False
        ):
            assert handler._get_agent_router() is None

    def test_get_agent_router_creates_router(self, handler):
        mock_stores = MagicMock()
        mock_store = MagicMock()
        mock_stores.gateway_store.return_value = mock_store
        with patch(
            "aragora.server.handlers.gateway_handler.get_canonical_gateway_stores",
            return_value=mock_stores,
        ), patch(
            "aragora.server.handlers.gateway_handler.AgentRouter",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            result = handler._get_agent_router()
        assert result is not None
        mock_cls.assert_called_once_with(store=mock_store)

    def test_get_agent_router_caches(self, handler):
        handler._agent_router = MagicMock()
        result = handler._get_agent_router()
        assert result is handler._agent_router

    def test_get_user_store_from_context(self):
        mock_store = MagicMock()
        h = GatewayHandler(server_context={"user_store": mock_store})
        assert h._get_user_store() is mock_store

    def test_get_user_store_missing(self, handler):
        assert handler._get_user_store() is None

    def test_get_gateway_stores_no_stores_returns_none_store(self, handler):
        """When stores return None, registry gets store=None."""
        with patch(
            "aragora.server.handlers.gateway_handler.get_canonical_gateway_stores",
            return_value=None,
        ), patch(
            "aragora.server.handlers.gateway_handler.DeviceRegistry",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            result = handler._get_device_registry()
        mock_cls.assert_called_once_with(store=None)


# ===========================================================================
# Heartbeat path parsing edge cases
# ===========================================================================


class TestHeartbeatPathParsing:
    """Test heartbeat path parsing for various path formats."""

    def test_heartbeat_with_extra_slashes(self, handler_with_registry):
        """Heartbeat path with leading/trailing slashes parsed correctly."""
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: True,
        ):
            result = handler_with_registry.handle_post(
                "/api/v1/gateway/devices/dev-1/heartbeat", {}, http
            )
        assert _status(result) == 200

    def test_heartbeat_short_path_not_matched(self, handler_with_registry):
        """A path with 'heartbeat' but too few segments returns None."""
        http = MockHTTPHandler()
        # Path: /api/v1/gateway/heartbeat -> 4 parts, not 6
        result = handler_with_registry.handle_post(
            "/api/v1/gateway/heartbeat", {}, http
        )
        assert result is None


# ===========================================================================
# ROUTES class attribute
# ===========================================================================


class TestRoutesAttribute:
    """Verify the ROUTES class attribute lists all expected patterns."""

    def test_routes_contains_devices(self):
        assert "/api/v1/gateway/devices" in GatewayHandler.ROUTES

    def test_routes_contains_devices_wildcard(self):
        assert "/api/v1/gateway/devices/*" in GatewayHandler.ROUTES

    def test_routes_contains_channels(self):
        assert "/api/v1/gateway/channels" in GatewayHandler.ROUTES

    def test_routes_contains_routing_stats(self):
        assert "/api/v1/gateway/routing/stats" in GatewayHandler.ROUTES

    def test_routes_contains_routing_rules(self):
        assert "/api/v1/gateway/routing/rules" in GatewayHandler.ROUTES

    def test_routes_contains_messages_route(self):
        assert "/api/v1/gateway/messages/route" in GatewayHandler.ROUTES

    def test_routes_count(self):
        assert len(GatewayHandler.ROUTES) >= 8


# ===========================================================================
# Multiple devices in list
# ===========================================================================


class TestListDevicesMultiple:
    """Test listing multiple devices with various statuses."""

    def test_list_multiple_devices(self, handler_with_registry):
        devices = [
            MockDeviceNode(
                device_id=f"dev-{i}",
                name=f"Device {i}",
                device_type="laptop" if i % 2 == 0 else "phone",
                capabilities=["browser"],
                status=MockDeviceStatus.PAIRED,
                paired_at=1700000000.0 + i,
                last_seen=1700000100.0 + i,
            )
            for i in range(5)
        ]
        http = MockHTTPHandler()
        with patch(
            "aragora.server.handlers.gateway_handler.run_async",
            side_effect=lambda c: devices,
        ):
            result = handler_with_registry.handle(
                "/api/v1/gateway/devices", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 5
        assert len(body["devices"]) == 5
        ids = [d["device_id"] for d in body["devices"]]
        assert "dev-0" in ids
        assert "dev-4" in ids


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    """Test that __all__ exports are correct."""

    def test_all_exports(self):
        from aragora.server.handlers import gateway_handler

        assert "GatewayHandler" in gateway_handler.__all__
        assert "get_gateway_circuit_breaker" in gateway_handler.__all__
        assert "get_gateway_circuit_breaker_status" in gateway_handler.__all__
        assert "reset_gateway_circuit_breaker" in gateway_handler.__all__
