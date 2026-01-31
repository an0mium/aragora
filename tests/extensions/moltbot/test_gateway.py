"""
Tests for Moltbot LocalGateway - Edge Orchestration for Device Networks.

Tests device registration, heartbeats, state management, and command routing.
"""

import asyncio
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from aragora.extensions.moltbot import LocalGateway, DeviceNodeConfig


class TestLocalGatewayBasic:
    """Tests for basic LocalGateway operations."""

    @pytest.fixture
    def gateway(self, tmp_path: Path) -> LocalGateway:
        """Create a gateway for testing."""
        return LocalGateway(
            gateway_id="test-gateway",
            storage_path=tmp_path / "gateway",
            heartbeat_timeout=60.0,
        )

    @pytest.mark.asyncio
    async def test_gateway_id(self, gateway: LocalGateway):
        """Test gateway has correct ID."""
        assert gateway.gateway_id == "test-gateway"

    @pytest.mark.asyncio
    async def test_start_stop(self, gateway: LocalGateway):
        """Test starting and stopping the gateway."""
        await gateway.start()
        stats = await gateway.get_stats()
        assert stats["gateway_id"] == "test-gateway"

        await gateway.stop()
        # Gateway should stop cleanly

    @pytest.mark.asyncio
    async def test_auto_generated_gateway_id(self, tmp_path: Path):
        """Test gateway auto-generates ID if not provided."""
        gateway = LocalGateway(storage_path=tmp_path / "gateway")
        assert gateway.gateway_id is not None
        assert len(gateway.gateway_id) > 0


class TestDeviceRegistration:
    """Tests for device registration."""

    @pytest.fixture
    def gateway(self, tmp_path: Path) -> LocalGateway:
        """Create a gateway for testing."""
        return LocalGateway(
            gateway_id="test-gateway",
            storage_path=tmp_path / "gateway",
        )

    @pytest.mark.asyncio
    async def test_register_device(self, gateway: LocalGateway):
        """Test registering a device."""
        config = DeviceNodeConfig(
            name="Sensor 1",
            device_type="iot",
            capabilities=["temperature", "humidity"],
        )

        device = await gateway.register_device(
            config=config,
            user_id="user-1",
        )

        assert device is not None
        assert device.id is not None
        assert device.config.name == "Sensor 1"
        assert device.config.device_type == "iot"
        assert device.user_id == "user-1"
        assert device.gateway_id == "test-gateway"
        assert device.status == "offline"  # Initially offline until heartbeat

    @pytest.mark.asyncio
    async def test_register_device_with_tenant(self, gateway: LocalGateway):
        """Test registering a device with tenant."""
        config = DeviceNodeConfig(name="Tenant Device", device_type="mobile")

        device = await gateway.register_device(
            config=config,
            user_id="user-1",
            tenant_id="tenant-1",
        )

        assert device.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_get_device(self, gateway: LocalGateway):
        """Test getting a device by ID."""
        config = DeviceNodeConfig(name="Test Device", device_type="iot")
        registered = await gateway.register_device(config=config, user_id="user-1")

        device = await gateway.get_device(registered.id)
        assert device is not None
        assert device.id == registered.id
        assert device.config.name == "Test Device"

    @pytest.mark.asyncio
    async def test_get_nonexistent_device(self, gateway: LocalGateway):
        """Test getting a nonexistent device."""
        device = await gateway.get_device("nonexistent")
        assert device is None

    @pytest.mark.asyncio
    async def test_list_devices(self, gateway: LocalGateway):
        """Test listing devices."""
        # Register multiple devices
        for i in range(3):
            config = DeviceNodeConfig(name=f"Device {i}", device_type="iot")
            await gateway.register_device(config=config, user_id="user-1")

        devices = await gateway.list_devices()
        assert len(devices) == 3

    @pytest.mark.asyncio
    async def test_list_devices_by_user(self, gateway: LocalGateway):
        """Test listing devices filtered by user."""
        config1 = DeviceNodeConfig(name="User1 Device", device_type="iot")
        config2 = DeviceNodeConfig(name="User2 Device", device_type="iot")

        await gateway.register_device(config=config1, user_id="user-1")
        await gateway.register_device(config=config2, user_id="user-2")

        user1_devices = await gateway.list_devices(user_id="user-1")
        assert len(user1_devices) == 1
        assert user1_devices[0].user_id == "user-1"

    @pytest.mark.asyncio
    async def test_list_devices_by_type(self, gateway: LocalGateway):
        """Test listing devices filtered by type."""
        iot_config = DeviceNodeConfig(name="IoT", device_type="iot")
        mobile_config = DeviceNodeConfig(name="Mobile", device_type="mobile")

        await gateway.register_device(config=iot_config, user_id="user-1")
        await gateway.register_device(config=mobile_config, user_id="user-1")

        iot_devices = await gateway.list_devices(device_type="iot")
        assert len(iot_devices) == 1
        assert iot_devices[0].config.device_type == "iot"

    @pytest.mark.asyncio
    async def test_unregister_device(self, gateway: LocalGateway):
        """Test unregistering a device."""
        config = DeviceNodeConfig(name="To Remove", device_type="iot")
        device = await gateway.register_device(config=config, user_id="user-1")

        result = await gateway.unregister_device(device.id)
        assert result is True

        # Device should no longer exist
        retrieved = await gateway.get_device(device.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_device(self, gateway: LocalGateway):
        """Test unregistering a nonexistent device."""
        result = await gateway.unregister_device("nonexistent")
        assert result is False


class TestDeviceHeartbeats:
    """Tests for device heartbeat handling."""

    @pytest.fixture
    def gateway(self, tmp_path: Path) -> LocalGateway:
        """Create a gateway for testing."""
        return LocalGateway(
            gateway_id="test-gateway",
            storage_path=tmp_path / "gateway",
            heartbeat_timeout=60.0,
        )

    @pytest.mark.asyncio
    async def test_heartbeat_sets_online(self, gateway: LocalGateway):
        """Test heartbeat sets device online."""
        config = DeviceNodeConfig(name="Heartbeat Device", device_type="iot")
        device = await gateway.register_device(config=config, user_id="user-1")

        assert device.status == "offline"

        updated = await gateway.heartbeat(device.id)
        assert updated is not None
        assert updated.status == "online"
        assert updated.last_heartbeat is not None

    @pytest.mark.asyncio
    async def test_heartbeat_with_state(self, gateway: LocalGateway):
        """Test heartbeat with state update."""
        config = DeviceNodeConfig(name="State Device", device_type="iot")
        device = await gateway.register_device(config=config, user_id="user-1")

        updated = await gateway.heartbeat(
            device.id,
            state={"temperature": 22.5, "humidity": 45},
        )

        assert updated.state["temperature"] == 22.5
        assert updated.state["humidity"] == 45

    @pytest.mark.asyncio
    async def test_heartbeat_with_metrics(self, gateway: LocalGateway):
        """Test heartbeat with metrics update."""
        config = DeviceNodeConfig(name="Metrics Device", device_type="mobile")
        device = await gateway.register_device(config=config, user_id="user-1")

        updated = await gateway.heartbeat(
            device.id,
            metrics={
                "battery_level": 0.85,
                "signal_strength": 0.7,
                "firmware_version": "2.0.1",
            },
        )

        assert updated.battery_level == 0.85
        assert updated.signal_strength == 0.7
        assert updated.firmware_version == "2.0.1"

    @pytest.mark.asyncio
    async def test_heartbeat_increments_messages(self, gateway: LocalGateway):
        """Test heartbeat increments message counter."""
        config = DeviceNodeConfig(name="Counter Device", device_type="iot")
        device = await gateway.register_device(config=config, user_id="user-1")

        initial_count = device.messages_received

        await gateway.heartbeat(device.id)
        await gateway.heartbeat(device.id)
        await gateway.heartbeat(device.id)

        updated = await gateway.get_device(device.id)
        assert updated.messages_received == initial_count + 3

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent_device(self, gateway: LocalGateway):
        """Test heartbeat for nonexistent device."""
        result = await gateway.heartbeat("nonexistent")
        assert result is None


class TestDeviceState:
    """Tests for device state management."""

    @pytest.fixture
    def gateway(self, tmp_path: Path) -> LocalGateway:
        """Create a gateway for testing."""
        return LocalGateway(
            gateway_id="test-gateway",
            storage_path=tmp_path / "gateway",
        )

    @pytest.mark.asyncio
    async def test_update_state_merge(self, gateway: LocalGateway):
        """Test updating device state with merge."""
        config = DeviceNodeConfig(name="State Device", device_type="iot")
        device = await gateway.register_device(config=config, user_id="user-1")

        # Set initial state
        await gateway.update_state(device.id, {"a": 1, "b": 2})

        # Merge new state
        updated = await gateway.update_state(device.id, {"b": 3, "c": 4}, merge=True)

        assert updated.state["a"] == 1  # Preserved
        assert updated.state["b"] == 3  # Updated
        assert updated.state["c"] == 4  # Added

    @pytest.mark.asyncio
    async def test_update_state_replace(self, gateway: LocalGateway):
        """Test updating device state with replace."""
        config = DeviceNodeConfig(name="State Device", device_type="iot")
        device = await gateway.register_device(config=config, user_id="user-1")

        # Set initial state
        await gateway.update_state(device.id, {"a": 1, "b": 2})

        # Replace state
        updated = await gateway.update_state(device.id, {"c": 3}, merge=False)

        assert "a" not in updated.state
        assert "b" not in updated.state
        assert updated.state["c"] == 3

    @pytest.mark.asyncio
    async def test_get_state(self, gateway: LocalGateway):
        """Test getting device state."""
        config = DeviceNodeConfig(name="State Device", device_type="iot")
        device = await gateway.register_device(config=config, user_id="user-1")

        await gateway.update_state(device.id, {"temperature": 22.5})

        state = await gateway.get_state(device.id)
        assert state["temperature"] == 22.5

    @pytest.mark.asyncio
    async def test_get_state_nonexistent_device(self, gateway: LocalGateway):
        """Test getting state for nonexistent device."""
        state = await gateway.get_state("nonexistent")
        assert state is None


class TestDeviceCommands:
    """Tests for device command handling."""

    @pytest.fixture
    def gateway(self, tmp_path: Path) -> LocalGateway:
        """Create a gateway for testing."""
        return LocalGateway(
            gateway_id="test-gateway",
            storage_path=tmp_path / "gateway",
        )

    @pytest.mark.asyncio
    async def test_send_command_device_not_found(self, gateway: LocalGateway):
        """Test sending command to nonexistent device."""
        result = await gateway.send_command("nonexistent", "test_command")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_send_command_device_offline(self, gateway: LocalGateway):
        """Test sending command to offline device."""
        config = DeviceNodeConfig(
            name="Offline Device",
            device_type="iot",
            capabilities=["test_command"],
        )
        device = await gateway.register_device(config=config, user_id="user-1")
        # Device is offline by default

        result = await gateway.send_command(device.id, "test_command")

        assert result["success"] is False
        assert "offline" in result["error"]

    @pytest.mark.asyncio
    async def test_send_command_unsupported(self, gateway: LocalGateway):
        """Test sending unsupported command."""
        config = DeviceNodeConfig(
            name="Limited Device",
            device_type="iot",
            capabilities=["read_temp"],
        )
        device = await gateway.register_device(config=config, user_id="user-1")
        await gateway.heartbeat(device.id)  # Set online

        result = await gateway.send_command(device.id, "unsupported_command")

        assert result["success"] is False
        assert "not support" in result["error"]

    @pytest.mark.asyncio
    async def test_send_command_success(self, gateway: LocalGateway):
        """Test successful command send (default handler)."""
        config = DeviceNodeConfig(
            name="Command Device",
            device_type="iot",
            capabilities=["set_led"],
        )
        device = await gateway.register_device(config=config, user_id="user-1")
        await gateway.heartbeat(device.id)  # Set online

        result = await gateway.send_command(
            device.id,
            "set_led",
            payload={"color": "red"},
        )

        assert result["success"] is True
        assert result["result"]["command"] == "set_led"

    @pytest.mark.asyncio
    async def test_register_command_handler(self, gateway: LocalGateway):
        """Test registering a custom command handler."""
        config = DeviceNodeConfig(
            name="Handler Device",
            device_type="iot",
            capabilities=["custom_cmd"],
        )
        device = await gateway.register_device(config=config, user_id="user-1")
        await gateway.heartbeat(device.id)

        # Register custom handler
        async def custom_handler(device, payload):
            return {"handled": True, "value": payload.get("value", 0) * 2}

        gateway.register_command_handler("custom_cmd", custom_handler)

        result = await gateway.send_command(
            device.id,
            "custom_cmd",
            payload={"value": 21},
        )

        assert result["success"] is True
        assert result["result"]["handled"] is True
        assert result["result"]["value"] == 42

    @pytest.mark.asyncio
    async def test_broadcast_command(self, gateway: LocalGateway):
        """Test broadcasting command to multiple devices."""
        # Create multiple devices with same capability
        for i in range(3):
            config = DeviceNodeConfig(
                name=f"Device {i}",
                device_type="iot",
                capabilities=["broadcast_cmd"],
            )
            device = await gateway.register_device(config=config, user_id="user-1")
            await gateway.heartbeat(device.id)

        result = await gateway.broadcast_command("broadcast_cmd")

        assert result["total"] == 3
        assert result["success"] == 3
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_broadcast_command_with_filter(self, gateway: LocalGateway):
        """Test broadcasting with device filter."""
        # Create IoT device
        iot_config = DeviceNodeConfig(
            name="IoT Device",
            device_type="iot",
            capabilities=["cmd"],
        )
        iot_device = await gateway.register_device(config=iot_config, user_id="user-1")
        await gateway.heartbeat(iot_device.id)

        # Create mobile device
        mobile_config = DeviceNodeConfig(
            name="Mobile Device",
            device_type="mobile",
            capabilities=["cmd"],
        )
        mobile_device = await gateway.register_device(config=mobile_config, user_id="user-1")
        await gateway.heartbeat(mobile_device.id)

        # Broadcast only to IoT
        result = await gateway.broadcast_command(
            "cmd",
            device_filter={"device_type": "iot"},
        )

        assert result["total"] == 1
        assert iot_device.id in result["results"]
        assert mobile_device.id not in result["results"]


class TestGatewayEvents:
    """Tests for gateway event handling."""

    @pytest.fixture
    def gateway(self, tmp_path: Path) -> LocalGateway:
        """Create a gateway for testing."""
        return LocalGateway(
            gateway_id="test-gateway",
            storage_path=tmp_path / "gateway",
        )

    @pytest.mark.asyncio
    async def test_subscribe_to_events(self, gateway: LocalGateway):
        """Test subscribing to gateway events."""
        events = []

        def callback(event):
            events.append(event)

        gateway.subscribe(callback)

        config = DeviceNodeConfig(name="Event Device", device_type="iot")
        await gateway.register_device(config=config, user_id="user-1")

        # Should have received device_registered event
        assert len(events) == 1
        assert events[0]["type"] == "device_registered"

    @pytest.mark.asyncio
    async def test_async_event_callback(self, gateway: LocalGateway):
        """Test async event callback."""
        events = []

        async def async_callback(event):
            events.append(event)

        gateway.subscribe(async_callback)

        config = DeviceNodeConfig(name="Async Event Device", device_type="iot")
        await gateway.register_device(config=config, user_id="user-1")

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, gateway: LocalGateway):
        """Test unsubscribing from events."""
        events = []

        def callback(event):
            events.append(event)

        gateway.subscribe(callback)
        gateway.unsubscribe(callback)

        config = DeviceNodeConfig(name="Test Device", device_type="iot")
        await gateway.register_device(config=config, user_id="user-1")

        # Should not have received event
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_device_online_event(self, gateway: LocalGateway):
        """Test device online event."""
        events = []

        def callback(event):
            events.append(event)

        gateway.subscribe(callback)

        config = DeviceNodeConfig(name="Online Device", device_type="iot")
        device = await gateway.register_device(config=config, user_id="user-1")

        # First heartbeat should trigger online event
        await gateway.heartbeat(device.id)

        online_events = [e for e in events if e["type"] == "device_online"]
        assert len(online_events) == 1

    @pytest.mark.asyncio
    async def test_device_unregistered_event(self, gateway: LocalGateway):
        """Test device unregistered event."""
        events = []

        def callback(event):
            events.append(event)

        gateway.subscribe(callback)

        config = DeviceNodeConfig(name="To Unregister", device_type="iot")
        device = await gateway.register_device(config=config, user_id="user-1")
        await gateway.unregister_device(device.id)

        unregister_events = [e for e in events if e["type"] == "device_unregistered"]
        assert len(unregister_events) == 1


class TestGatewayStats:
    """Tests for gateway statistics."""

    @pytest.fixture
    def gateway(self, tmp_path: Path) -> LocalGateway:
        """Create a gateway for testing."""
        return LocalGateway(
            gateway_id="test-gateway",
            storage_path=tmp_path / "gateway",
        )

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, gateway: LocalGateway):
        """Test getting stats with no devices."""
        stats = await gateway.get_stats()

        assert stats["gateway_id"] == "test-gateway"
        assert stats["devices_total"] == 0
        assert stats["devices_online"] == 0
        assert stats["devices_offline"] == 0
        assert stats["total_messages"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_devices(self, gateway: LocalGateway):
        """Test getting stats with devices."""
        # Create and register devices
        for i in range(3):
            config = DeviceNodeConfig(name=f"Device {i}", device_type="iot")
            device = await gateway.register_device(config=config, user_id="user-1")
            if i < 2:  # Make 2 online
                await gateway.heartbeat(device.id)

        stats = await gateway.get_stats()

        assert stats["devices_total"] == 3
        assert stats["devices_online"] == 2
        assert stats["devices_offline"] == 1
        assert stats["devices_by_type"]["iot"] == 3

    @pytest.mark.asyncio
    async def test_stats_include_message_counts(self, gateway: LocalGateway):
        """Test stats include message counts."""
        config = DeviceNodeConfig(
            name="Message Device",
            device_type="iot",
            capabilities=["cmd"],
        )
        device = await gateway.register_device(config=config, user_id="user-1")
        await gateway.heartbeat(device.id)
        await gateway.heartbeat(device.id)
        await gateway.send_command(device.id, "cmd")

        stats = await gateway.get_stats()

        assert stats["total_messages"] > 0
