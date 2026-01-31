"""
Tests for MoltbotGatewayAdapter - Compatibility Layer for Canonical Gateway APIs.

Tests translation between Moltbot models and canonical gateway primitives.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.extensions.moltbot.adapter import MoltbotGatewayAdapter
from aragora.extensions.moltbot.models import (
    DeviceNodeConfig,
    InboxMessage,
    InboxMessageStatus,
)


class TestAdapterInitialization:
    """Tests for adapter initialization."""

    def test_create_adapter_default_gateway(self):
        """Test creating adapter with default gateway."""
        adapter = MoltbotGatewayAdapter()

        assert adapter is not None
        assert adapter._gateway is not None
        assert adapter._gateway_id == "local-gateway"

    def test_create_adapter_custom_gateway_id(self):
        """Test creating adapter with custom gateway ID."""
        adapter = MoltbotGatewayAdapter(gateway_id="custom-gateway")

        assert adapter._gateway_id == "custom-gateway"

    def test_create_adapter_with_gateway(self):
        """Test creating adapter with provided gateway."""
        mock_gateway = MagicMock()
        adapter = MoltbotGatewayAdapter(gateway=mock_gateway)

        assert adapter._gateway is mock_gateway


class TestDeviceRegistration:
    """Tests for device registration through adapter."""

    @pytest.fixture
    def mock_gateway(self):
        """Create a mock gateway."""
        gateway = AsyncMock()
        return gateway

    @pytest.fixture
    def adapter(self, mock_gateway):
        """Create adapter with mock gateway."""
        return MoltbotGatewayAdapter(gateway=mock_gateway, gateway_id="test-gateway")

    @pytest.mark.asyncio
    async def test_register_device(self, adapter, mock_gateway):
        """Test registering a device through adapter."""
        # Setup mock return
        mock_device = MagicMock()
        mock_device.device_id = "device-123"
        mock_gateway.register_device.return_value = mock_device

        config = DeviceNodeConfig(
            name="Test Device",
            device_type="iot",
            capabilities=["temperature", "humidity"],
        )

        device = await adapter.register_device(
            config=config,
            user_id="user-1",
        )

        # Verify gateway was called
        mock_gateway.register_device.assert_called_once()

        # Verify returned device
        assert device is not None
        assert device.id == "device-123"
        assert device.config.name == "Test Device"
        assert device.user_id == "user-1"
        assert device.gateway_id == "test-gateway"

    @pytest.mark.asyncio
    async def test_register_device_with_tenant(self, adapter, mock_gateway):
        """Test registering device with tenant."""
        mock_device = MagicMock()
        mock_device.device_id = "device-456"
        mock_gateway.register_device.return_value = mock_device

        config = DeviceNodeConfig(name="Tenant Device", device_type="mobile")

        device = await adapter.register_device(
            config=config,
            user_id="user-1",
            tenant_id="tenant-1",
        )

        assert device.tenant_id == "tenant-1"

        # Verify tenant was passed to gateway
        call_args = mock_gateway.register_device.call_args
        registered_device = call_args[0][0]
        assert registered_device.metadata["moltbot_tenant_id"] == "tenant-1"

    @pytest.mark.asyncio
    async def test_register_device_passes_metadata(self, adapter, mock_gateway):
        """Test device metadata is passed to gateway."""
        mock_device = MagicMock()
        mock_device.device_id = "device-789"
        mock_gateway.register_device.return_value = mock_device

        config = DeviceNodeConfig(
            name="Metadata Device",
            device_type="embedded",
            connection_type="websocket",
            heartbeat_interval=30,
            metadata={"firmware": "1.0.0"},
        )

        await adapter.register_device(config=config, user_id="user-1")

        call_args = mock_gateway.register_device.call_args
        registered_device = call_args[0][0]

        assert registered_device.metadata["connection_type"] == "websocket"
        assert registered_device.metadata["heartbeat_interval"] == 30
        assert registered_device.metadata["moltbot_metadata"]["firmware"] == "1.0.0"


class TestMessageRouting:
    """Tests for message routing through adapter."""

    @pytest.fixture
    def mock_gateway(self):
        """Create a mock gateway."""
        gateway = AsyncMock()
        return gateway

    @pytest.fixture
    def adapter(self, mock_gateway):
        """Create adapter with mock gateway."""
        return MoltbotGatewayAdapter(gateway=mock_gateway, gateway_id="test-gateway")

    @pytest.mark.asyncio
    async def test_route_message(self, adapter, mock_gateway):
        """Test routing a message through adapter."""
        message = InboxMessage(
            id="msg-1",
            channel_id="channel-1",
            user_id="user-1",
            direction="inbound",
            content="Hello, world!",
        )

        await adapter.route_message(message)

        mock_gateway.route_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_message_with_channel_override(self, adapter, mock_gateway):
        """Test routing message with channel override."""
        message = InboxMessage(
            id="msg-2",
            channel_id="original-channel",
            user_id="user-1",
            direction="outbound",
            content="Test message",
        )

        await adapter.route_message(message, channel="override-channel")

        call_args = mock_gateway.route_message.call_args
        assert call_args[1]["channel"] == "override-channel"

    @pytest.mark.asyncio
    async def test_route_message_with_device(self, adapter, mock_gateway):
        """Test routing message to specific device."""
        message = InboxMessage(
            id="msg-3",
            channel_id="channel-1",
            user_id="user-1",
            direction="outbound",
            content="Device message",
        )

        await adapter.route_message(message, device_id="device-123")

        call_args = mock_gateway.route_message.call_args
        assert call_args[1]["device_id"] == "device-123"

    @pytest.mark.asyncio
    async def test_route_message_preserves_metadata(self, adapter, mock_gateway):
        """Test message metadata is preserved in routing."""
        message = InboxMessage(
            id="msg-4",
            channel_id="channel-1",
            user_id="user-1",
            direction="inbound",
            content="Content",
            content_type="image",
            thread_id="thread-1",
            metadata={"custom": "data"},
        )

        await adapter.route_message(message)

        call_args = mock_gateway.route_message.call_args
        gateway_message = call_args[1]["message"]

        assert gateway_message.metadata["moltbot_message_id"] == "msg-4"
        assert gateway_message.metadata["moltbot_channel_id"] == "channel-1"
        assert gateway_message.metadata["moltbot_direction"] == "inbound"
        assert gateway_message.metadata["moltbot_content_type"] == "image"


class TestAdapterIntegration:
    """Integration tests for the adapter with real gateway."""

    @pytest.mark.asyncio
    async def test_full_device_lifecycle(self):
        """Test full device registration lifecycle."""
        # This test uses the real gateway implementation
        adapter = MoltbotGatewayAdapter(gateway_id="integration-test")

        config = DeviceNodeConfig(
            name="Integration Test Device",
            device_type="iot",
            capabilities=["sensor"],
        )

        # Register device
        device = await adapter.register_device(
            config=config,
            user_id="test-user",
            tenant_id="test-tenant",
        )

        assert device is not None
        assert device.id is not None
        assert device.config.name == "Integration Test Device"
        assert device.user_id == "test-user"
        assert device.tenant_id == "test-tenant"
        assert device.gateway_id == "integration-test"

    @pytest.mark.asyncio
    async def test_message_routing_integration(self):
        """Test message routing with real gateway."""
        adapter = MoltbotGatewayAdapter(gateway_id="integration-test")

        message = InboxMessage(
            id="integration-msg-1",
            channel_id="test-channel",
            user_id="test-user",
            direction="outbound",
            content="Integration test message",
        )

        # This should not raise an exception
        result = await adapter.route_message(message, channel="test")

        # The real gateway returns a result
        assert result is not None


class TestAdapterEdgeCases:
    """Tests for adapter edge cases."""

    @pytest.fixture
    def mock_gateway(self):
        """Create a mock gateway."""
        gateway = AsyncMock()
        return gateway

    @pytest.fixture
    def adapter(self, mock_gateway):
        """Create adapter with mock gateway."""
        return MoltbotGatewayAdapter(gateway=mock_gateway)

    @pytest.mark.asyncio
    async def test_register_device_empty_capabilities(self, adapter, mock_gateway):
        """Test registering device with no capabilities."""
        mock_device = MagicMock()
        mock_device.device_id = "device-empty"
        mock_gateway.register_device.return_value = mock_device

        config = DeviceNodeConfig(
            name="Empty Caps Device",
            device_type="minimal",
            capabilities=[],
        )

        device = await adapter.register_device(config=config, user_id="user-1")

        assert device is not None
        assert device.config.capabilities == []

    @pytest.mark.asyncio
    async def test_route_message_empty_content(self, adapter, mock_gateway):
        """Test routing message with empty content."""
        message = InboxMessage(
            id="msg-empty",
            channel_id="channel-1",
            user_id="user-1",
            direction="outbound",
            content="",
        )

        await adapter.route_message(message)

        mock_gateway.route_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_message_with_all_optional_fields(self, adapter, mock_gateway):
        """Test routing message with all optional fields populated."""
        message = InboxMessage(
            id="msg-full",
            channel_id="channel-1",
            user_id="user-1",
            direction="inbound",
            content="Full message",
            content_type="text",
            status=InboxMessageStatus.DELIVERED,
            thread_id="thread-123",
            reply_to="msg-previous",
            intent="question",
            entities={"topic": "billing"},
            sentiment=0.5,
            response_id="response-1",
            external_id="external-123",
            metadata={"source": "mobile", "version": "2.0"},
        )

        await adapter.route_message(message)

        call_args = mock_gateway.route_message.call_args
        gateway_message = call_args[1]["message"]

        assert gateway_message.thread_id == "thread-123"
        assert gateway_message.metadata["moltbot_content_type"] == "text"
