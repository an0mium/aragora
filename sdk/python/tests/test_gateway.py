"""Tests for Gateway namespace API.

Tests for both synchronous (GatewayAPI) and asynchronous (AsyncGatewayAPI) classes.
Covers device management, channel listing, routing, and message routing endpoints.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# =============================================================================
# Device Tests (Sync)
# =============================================================================

class TestGatewayListDevices:
    """Tests for listing devices."""

    def test_list_devices_default(self) -> None:
        """List devices with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "devices": [{"device_id": "dev_1", "name": "Smart Speaker"}],
                "total": 1,
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gateway.list_devices()
            mock_request.assert_called_once_with("GET", "/api/v1/gateway/devices", params={})
            assert result["total"] == 1
            assert result["devices"][0]["device_id"] == "dev_1"
            client.close()

    def test_list_devices_filtered_by_status(self) -> None:
        """List devices filtered by status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"devices": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gateway.list_devices(status="online")
            mock_request.assert_called_once_with(
                "GET", "/api/v1/gateway/devices", params={"status": "online"}
            )
            client.close()

    def test_list_devices_filtered_by_type(self) -> None:
        """List devices filtered by device type."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"devices": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gateway.list_devices(device_type="alexa")
            mock_request.assert_called_once_with(
                "GET", "/api/v1/gateway/devices", params={"type": "alexa"}
            )
            client.close()

    def test_list_devices_filtered_by_status_and_type(self) -> None:
        """List devices filtered by both status and type."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"devices": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gateway.list_devices(status="offline", device_type="google_home")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gateway/devices",
                params={"status": "offline", "type": "google_home"},
            )
            client.close()

class TestGatewayGetDevice:
    """Tests for getting device details."""

    def test_get_device(self) -> None:
        """Get device details by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "device_id": "dev_123",
                "name": "Kitchen Speaker",
                "device_type": "alexa",
                "status": "online",
                "capabilities": ["voice", "tts"],
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gateway.get_device("dev_123")
            mock_request.assert_called_once_with("GET", "/api/v1/gateway/devices/dev_123")
            assert result["device_id"] == "dev_123"
            assert result["name"] == "Kitchen Speaker"
            assert result["status"] == "online"
            client.close()

class TestGatewayRegisterDevice:
    """Tests for device registration."""

    def test_register_device_minimal(self) -> None:
        """Register a device with minimal parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "device_id": "dev_new",
                "message": "Device registered successfully",
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gateway.register_device(name="Living Room Speaker")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gateway/devices",
                json={"name": "Living Room Speaker", "device_type": "unknown"},
            )
            assert result["device_id"] == "dev_new"
            client.close()

    def test_register_device_with_type(self) -> None:
        """Register a device with specific type."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"device_id": "dev_alexa"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gateway.register_device(name="Echo Dot", device_type="alexa")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gateway/devices",
                json={"name": "Echo Dot", "device_type": "alexa"},
            )
            client.close()

    def test_register_device_with_custom_id(self) -> None:
        """Register a device with a custom device ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"device_id": "custom_id_123"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gateway.register_device(name="Custom Device", device_id="custom_id_123")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gateway/devices",
                json={
                    "name": "Custom Device",
                    "device_type": "unknown",
                    "device_id": "custom_id_123",
                },
            )
            client.close()

    def test_register_device_with_capabilities(self) -> None:
        """Register a device with capabilities."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"device_id": "dev_capable"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gateway.register_device(
                name="Smart Display",
                device_type="google_home",
                capabilities=["voice", "tts", "display"],
            )
            call_json = mock_request.call_args[1]["json"]
            assert call_json["capabilities"] == ["voice", "tts", "display"]
            client.close()

    def test_register_device_with_allowed_channels(self) -> None:
        """Register a device with allowed channels."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"device_id": "dev_channels"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gateway.register_device(
                name="Channel Device",
                allowed_channels=["slack", "telegram"],
            )
            call_json = mock_request.call_args[1]["json"]
            assert call_json["allowed_channels"] == ["slack", "telegram"]
            client.close()

    def test_register_device_with_metadata(self) -> None:
        """Register a device with metadata."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"device_id": "dev_meta"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gateway.register_device(
                name="Metadata Device",
                metadata={"location": "office", "floor": 3},
            )
            call_json = mock_request.call_args[1]["json"]
            assert call_json["metadata"] == {"location": "office", "floor": 3}
            client.close()

    def test_register_device_full_options(self) -> None:
        """Register a device with all options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"device_id": "dev_full"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gateway.register_device(
                name="Full Feature Device",
                device_type="alexa",
                device_id="custom_full_123",
                capabilities=["voice", "tts"],
                allowed_channels=["slack"],
                metadata={"room": "conference"},
            )
            call_json = mock_request.call_args[1]["json"]
            assert call_json["name"] == "Full Feature Device"
            assert call_json["device_type"] == "alexa"
            assert call_json["device_id"] == "custom_full_123"
            assert call_json["capabilities"] == ["voice", "tts"]
            assert call_json["allowed_channels"] == ["slack"]
            assert call_json["metadata"] == {"room": "conference"}
            client.close()

class TestGatewayUnregisterDevice:
    """Tests for device unregistration."""

    def test_unregister_device(self) -> None:
        """Unregister a device."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"message": "Device unregistered successfully"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gateway.unregister_device("dev_123")
            mock_request.assert_called_once_with("DELETE", "/api/v1/gateway/devices/dev_123")
            assert "message" in result
            client.close()

class TestGatewayListChannels:
    """Tests for listing channels."""

    def test_list_channels(self) -> None:
        """List active channels."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "channels": [
                    {"channel_id": "slack", "name": "Slack", "status": "active"},
                    {"channel_id": "telegram", "name": "Telegram", "status": "active"},
                ],
                "total": 2,
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gateway.list_channels()
            mock_request.assert_called_once_with("GET", "/api/v1/gateway/channels")
            assert result["total"] == 2
            assert result["channels"][0]["channel_id"] == "slack"
            client.close()

# =============================================================================
# Routing Tests (Sync)
# =============================================================================

class TestAsyncGatewayDevices:
    """Tests for async device management."""

    @pytest.mark.asyncio
    async def test_list_devices(self) -> None:
        """List devices asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "devices": [{"device_id": "dev_async"}],
                "total": 1,
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.gateway.list_devices()
            mock_request.assert_called_once_with("GET", "/api/v1/gateway/devices", params={})
            assert result["total"] == 1
            await client.close()

    @pytest.mark.asyncio
    async def test_list_devices_filtered(self) -> None:
        """List devices with filters asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"devices": [], "total": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            await client.gateway.list_devices(status="online", device_type="alexa")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gateway/devices",
                params={"status": "online", "type": "alexa"},
            )
            await client.close()

    @pytest.mark.asyncio
    async def test_get_device(self) -> None:
        """Get device details asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"device_id": "dev_async", "name": "Async Device"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.gateway.get_device("dev_async")
            mock_request.assert_called_once_with("GET", "/api/v1/gateway/devices/dev_async")
            assert result["name"] == "Async Device"
            await client.close()

    @pytest.mark.asyncio
    async def test_register_device(self) -> None:
        """Register a device asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"device_id": "dev_new_async"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.gateway.register_device(
                name="Async Speaker",
                device_type="google_home",
                capabilities=["voice"],
            )
            call_json = mock_request.call_args[1]["json"]
            assert call_json["name"] == "Async Speaker"
            assert call_json["device_type"] == "google_home"
            assert call_json["capabilities"] == ["voice"]
            assert result["device_id"] == "dev_new_async"
            await client.close()

    @pytest.mark.asyncio
    async def test_register_device_full_options(self) -> None:
        """Register a device with all options asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"device_id": "dev_full_async"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            await client.gateway.register_device(
                name="Full Async Device",
                device_type="alexa",
                device_id="custom_async_id",
                capabilities=["voice", "tts"],
                allowed_channels=["slack", "teams"],
                metadata={"location": "lobby"},
            )
            call_json = mock_request.call_args[1]["json"]
            assert call_json["name"] == "Full Async Device"
            assert call_json["device_type"] == "alexa"
            assert call_json["device_id"] == "custom_async_id"
            assert call_json["capabilities"] == ["voice", "tts"]
            assert call_json["allowed_channels"] == ["slack", "teams"]
            assert call_json["metadata"] == {"location": "lobby"}
            await client.close()

    @pytest.mark.asyncio
    async def test_unregister_device(self) -> None:
        """Unregister a device asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"message": "Unregistered"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.gateway.unregister_device("dev_async")
            mock_request.assert_called_once_with("DELETE", "/api/v1/gateway/devices/dev_async")
            assert result["message"] == "Unregistered"
            await client.close()

class TestAsyncGatewayChannels:
    """Tests for async channel management."""

    @pytest.mark.asyncio
    async def test_list_channels(self) -> None:
        """List channels asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "channels": [{"channel_id": "slack", "status": "active"}],
                "total": 1,
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.gateway.list_channels()
            mock_request.assert_called_once_with("GET", "/api/v1/gateway/channels")
            assert result["total"] == 1
            await client.close()

# =============================================================================
# Async Routing Tests
# =============================================================================

