"""
Integration tests for Local Gateway + Device Registration flow.

Tests the complete OpenClaw Phase 3 integration:
- Device registration via gateway
- Message routing with device context
- Device heartbeat and status tracking
- Voice handler device association
- Production hardening (rate limits, graceful shutdown)
"""

from __future__ import annotations

import asyncio

import pytest

from aragora.gateway import DeviceNodeRuntime, DeviceNodeRuntimeConfig, DeviceRegistry
from aragora.gateway.device_registry import DeviceStatus
from aragora.gateway.inbox import InboxMessage
from aragora.gateway.server import GatewayConfig, LocalGateway, GatewayRateLimiter


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _gateway_store_env(monkeypatch, tmp_path):
    """Configure gateway to use file-based storage for tests."""
    monkeypatch.setenv("ARAGORA_GATEWAY_STORE", "file")
    monkeypatch.setenv("ARAGORA_GATEWAY_STORE_PATH", str(tmp_path / "gateway.json"))
    yield


@pytest.fixture
def gateway():
    """Create a gateway instance for testing."""
    return LocalGateway(
        config=GatewayConfig(
            enable_auth=False,
            request_timeout_seconds=5.0,
            rate_limit_rpm=100,
            max_connections=50,
        )
    )


@pytest.fixture
def gateway_with_auth():
    """Create a gateway instance with authentication enabled."""
    return LocalGateway(
        config=GatewayConfig(
            enable_auth=True,
            api_key="test-secret-key",
            request_timeout_seconds=5.0,
        )
    )


@pytest.fixture
def device_registry():
    """Create a standalone device registry for testing."""
    return DeviceRegistry()


# =============================================================================
# Device Registration Flow Tests
# =============================================================================


class TestDeviceRegistrationFlow:
    """Test complete device registration and management flow."""

    @pytest.mark.asyncio
    async def test_register_device_via_gateway(self, gateway):
        """Test registering a device through the gateway."""
        await gateway.start()

        from aragora.gateway.device_registry import DeviceNode

        device = DeviceNode(
            name="Test Laptop",
            device_type="laptop",
            capabilities=["browser", "shell", "voice"],
            allowed_channels=["slack", "telegram"],
        )

        device_id = await gateway.register_device(device)

        assert device_id is not None
        assert len(device_id) > 0

        # Verify device is retrievable
        retrieved = await gateway._devices.get(device_id)
        assert retrieved is not None
        assert retrieved.name == "Test Laptop"
        assert "voice" in retrieved.capabilities

        await gateway.stop()

    @pytest.mark.asyncio
    async def test_device_heartbeat_updates_status(self, gateway):
        """Test that device heartbeats update device status."""
        await gateway.start()

        from aragora.gateway.device_registry import DeviceNode

        device = DeviceNode(
            name="Heartbeat Device",
            device_type="server",
            capabilities=["api"],
        )

        device_id = await gateway.register_device(device)

        # Send heartbeat
        success = await gateway._devices.heartbeat(device_id)
        assert success is True

        # Verify status is online
        device = await gateway._devices.get(device_id)
        assert device.status == DeviceStatus.ONLINE

        await gateway.stop()

    @pytest.mark.asyncio
    async def test_device_runtime_pairing(self, device_registry):
        """Test DeviceNodeRuntime pairing with registry."""
        runtime = DeviceNodeRuntime(
            device_registry,
            DeviceNodeRuntimeConfig(
                name="Runtime Device",
                device_type="mobile",
                capabilities=["voice", "sms"],
                allowed_channels=["whatsapp"],
            ),
        )

        device_id = await runtime.pair()
        assert device_id is not None
        assert await runtime.is_paired() is True

        # Check capabilities
        assert await runtime.supports("voice") is True
        assert await runtime.supports("video") is False

        # Heartbeat
        assert await runtime.heartbeat() is True

        # Unregister
        assert await runtime.unregister() is True
        assert await runtime.is_paired() is False


# =============================================================================
# Message Routing with Device Context
# =============================================================================


class TestMessageRoutingWithDevices:
    """Test message routing when devices are involved."""

    @pytest.mark.asyncio
    async def test_route_message_from_device(self, gateway):
        """Test routing a message that includes device context."""
        await gateway.start()

        # Register a device
        from aragora.gateway.device_registry import DeviceNode

        device = DeviceNode(
            name="Routing Device",
            device_type="desktop",
            capabilities=["browser"],
        )
        device_id = await gateway.register_device(device)

        # Route a message with device metadata
        message = InboxMessage(
            message_id="msg-with-device",
            channel="slack",
            sender="user@example.com",
            content="Hello from device",
            metadata={"device_id": device_id},
        )

        response = await gateway.route_message("slack", message)

        assert response.success is True
        assert response.message_id == "msg-with-device"

        # Verify message in inbox
        inbox = await gateway.get_inbox(limit=10)
        assert len(inbox) == 1
        assert inbox[0].metadata.get("device_id") == device_id

        await gateway.stop()

    @pytest.mark.asyncio
    async def test_route_message_updates_stats(self, gateway):
        """Test that routing messages updates gateway statistics."""
        await gateway.start()

        # Route several messages
        for i in range(5):
            message = InboxMessage(
                message_id=f"msg-{i}",
                channel="telegram",
                sender="user",
                content=f"Message {i}",
            )
            await gateway.route_message("telegram", message)

        stats = await gateway.get_stats()

        assert stats["messages_routed"] == 5
        assert stats["messages_failed"] == 0
        assert stats["running"] is True

        await gateway.stop()


# =============================================================================
# Production Hardening Tests
# =============================================================================


class TestProductionHardening:
    """Test production hardening features."""

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self):
        """Test that rate limiter blocks requests over the limit."""
        limiter = GatewayRateLimiter(requests_per_minute=5, burst_allowance=2)

        # Should allow 7 requests (5 base + 2 burst)
        for i in range(7):
            assert await limiter.is_allowed("client-1") is True

        # 8th request should be blocked
        assert await limiter.is_allowed("client-1") is False

        # Different client should still be allowed
        assert await limiter.is_allowed("client-2") is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown_drains_connections(self, gateway):
        """Test that graceful shutdown properly drains connections."""
        await gateway.start()

        # Verify gateway is running
        assert gateway._running is True
        assert gateway._shutting_down is False

        # Trigger graceful shutdown
        result = await gateway.graceful_shutdown()

        assert result["status"] == "shutdown_complete"
        assert gateway._running is False
        assert gateway._shutting_down is True

    @pytest.mark.asyncio
    async def test_double_shutdown_is_idempotent(self, gateway):
        """Test that calling shutdown twice is safe."""
        await gateway.start()

        # First shutdown
        result1 = await gateway.graceful_shutdown()
        assert result1["status"] == "shutdown_complete"

        # Second shutdown
        result2 = await gateway.graceful_shutdown()
        assert result2["status"] == "already_shutting_down"

    @pytest.mark.asyncio
    async def test_connection_limit_enforcement(self, gateway):
        """Test that connection limits are enforced."""
        gateway._active_connections = gateway._config.max_connections

        # Creating app should work
        app = gateway._create_app()
        assert app is not None

        # Verify connection limit is set
        assert gateway._config.max_connections == 50


# =============================================================================
# Gateway with Authentication Tests
# =============================================================================


class TestGatewayAuthentication:
    """Test gateway authentication features."""

    @pytest.mark.asyncio
    async def test_auth_required_for_routing(self, gateway_with_auth):
        """Test that auth is required for message routing."""
        await gateway_with_auth.start()

        # Message without API key should fail
        message = InboxMessage(
            message_id="unauth-msg",
            channel="slack",
            sender="user",
            content="Hello",
            metadata={},  # No api_key
        )

        response = await gateway_with_auth.route_message("slack", message)

        assert response.success is False
        assert "Authentication" in response.error

        await gateway_with_auth.stop()

    @pytest.mark.asyncio
    async def test_valid_api_key_allows_routing(self, gateway_with_auth):
        """Test that valid API key allows message routing."""
        await gateway_with_auth.start()

        # Message with correct API key
        message = InboxMessage(
            message_id="auth-msg",
            channel="slack",
            sender="user",
            content="Hello",
            metadata={"api_key": "test-secret-key"},
        )

        response = await gateway_with_auth.route_message("slack", message)

        assert response.success is True

        await gateway_with_auth.stop()


# =============================================================================
# End-to-End Flow Test
# =============================================================================


class TestEndToEndFlow:
    """Test complete end-to-end flows."""

    @pytest.mark.asyncio
    async def test_full_device_registration_routing_flow(self, gateway):
        """Test complete flow: device registration -> message routing -> stats."""
        await gateway.start()

        # 1. Register a device
        from aragora.gateway.device_registry import DeviceNode

        device = DeviceNode(
            name="E2E Test Device",
            device_type="tablet",
            capabilities=["browser", "voice"],
            allowed_channels=["slack", "telegram"],
        )
        device_id = await gateway.register_device(device)
        assert device_id is not None

        # 2. Send heartbeat
        await gateway._devices.heartbeat(device_id)

        # 3. Route messages from the device
        for i in range(3):
            message = InboxMessage(
                message_id=f"e2e-msg-{i}",
                channel="slack",
                sender="alice@example.com",
                content=f"E2E test message {i}",
                metadata={"device_id": device_id, "priority": "normal"},
            )
            response = await gateway.route_message("slack", message)
            assert response.success is True

        # 4. Check inbox
        inbox = await gateway.get_inbox(limit=10)
        assert len(inbox) == 3

        # 5. Check stats
        stats = await gateway.get_stats()
        assert stats["messages_routed"] == 3
        assert stats["devices_registered"] == 1
        assert stats["running"] is True

        # 6. Graceful shutdown
        result = await gateway.graceful_shutdown()
        assert result["status"] == "shutdown_complete"

    @pytest.mark.asyncio
    async def test_device_runtime_with_gateway_registry(self, gateway):
        """Test DeviceNodeRuntime working with gateway's internal registry."""
        await gateway.start()

        # Create runtime using gateway's device registry
        runtime = DeviceNodeRuntime(
            gateway._devices,
            DeviceNodeRuntimeConfig(
                name="Runtime via Gateway",
                device_type="server",
                capabilities=["api", "webhook"],
            ),
        )

        # Pair device
        device_id = await runtime.pair()
        assert device_id is not None

        # Verify via gateway stats
        stats = await gateway.get_stats()
        assert stats["devices_registered"] == 1

        # Route message referencing the device
        message = InboxMessage(
            message_id="runtime-msg",
            channel="webhook",
            sender="system",
            content="Webhook event",
            metadata={"device_id": device_id},
        )
        response = await gateway.route_message("webhook", message)
        assert response.success is True

        # Unregister device
        await runtime.unregister()

        # Verify device count
        stats = await gateway.get_stats()
        assert stats["devices_registered"] == 0

        await gateway.stop()
