"""Tests for device security and secure pairing."""

from __future__ import annotations

import asyncio
import time

import pytest

from aragora.gateway.device_registry import DeviceRegistry, DeviceStatus
from aragora.gateway.device_security import (
    PairingRequest,
    PairingStatus,
    SecureDeviceRegistry,
)


@pytest.fixture
def registry():
    return DeviceRegistry()


@pytest.fixture
def secure_registry(registry):
    return SecureDeviceRegistry(
        registry=registry,
        pairing_timeout=300.0,
        max_requests_per_minute=5,
        heartbeat_interval=1.0,
        offline_threshold=3.0,
    )


class TestPairingRequest:
    def test_default_values(self):
        request = PairingRequest(
            request_id="req-1",
            device_id="dev-1",
            device_name="My Laptop",
            device_type="laptop",
            verification_code="123456",
            challenge="abc123",
            requested_at=time.time(),
        )
        assert request.status == PairingStatus.REQUESTED
        assert request.approved_by is None
        assert request.capabilities == []

    def test_with_capabilities(self):
        request = PairingRequest(
            request_id="req-1",
            device_id="dev-1",
            device_name="My Phone",
            device_type="phone",
            verification_code="654321",
            challenge="xyz789",
            requested_at=time.time(),
            capabilities=["camera", "mic"],
            metadata={"os": "iOS"},
        )
        assert request.capabilities == ["camera", "mic"]
        assert request.metadata == {"os": "iOS"}


class TestSecurePairing:
    @pytest.mark.asyncio
    async def test_request_pairing(self, secure_registry):
        request = await secure_registry.request_pairing(
            device_name="Test Laptop",
            device_type="laptop",
            capabilities=["browser", "shell"],
        )

        assert request.device_name == "Test Laptop"
        assert request.device_type == "laptop"
        assert request.capabilities == ["browser", "shell"]
        assert request.status == PairingStatus.REQUESTED
        assert len(request.verification_code) == 6
        assert request.verification_code.isdigit()

    @pytest.mark.asyncio
    async def test_approve_pairing(self, secure_registry):
        request = await secure_registry.request_pairing(
            device_name="Test Laptop",
            device_type="laptop",
        )

        result = await secure_registry.approve_pairing(request.request_id, approved_by="admin")
        assert result is True
        assert request.status == PairingStatus.APPROVED
        assert request.approved_by == "admin"

    @pytest.mark.asyncio
    async def test_confirm_pairing(self, secure_registry):
        request = await secure_registry.request_pairing(
            device_name="Test Laptop",
            device_type="laptop",
            capabilities=["browser"],
        )

        await secure_registry.approve_pairing(request.request_id)

        device = await secure_registry.confirm_pairing(
            request.request_id,
            request.verification_code,
        )

        assert device is not None
        assert device.name == "Test Laptop"
        assert device.device_type == "laptop"
        assert device.capabilities == ["browser"]

    @pytest.mark.asyncio
    async def test_confirm_without_approval_fails(self, secure_registry):
        request = await secure_registry.request_pairing(
            device_name="Test Laptop",
            device_type="laptop",
        )

        # Try to confirm without approving
        device = await secure_registry.confirm_pairing(
            request.request_id,
            request.verification_code,
        )

        assert device is None

    @pytest.mark.asyncio
    async def test_confirm_with_wrong_code_fails(self, secure_registry):
        request = await secure_registry.request_pairing(
            device_name="Test Laptop",
            device_type="laptop",
        )

        await secure_registry.approve_pairing(request.request_id)

        device = await secure_registry.confirm_pairing(
            request.request_id,
            "000000",  # Wrong code
        )

        assert device is None

    @pytest.mark.asyncio
    async def test_reject_pairing(self, secure_registry):
        request = await secure_registry.request_pairing(
            device_name="Suspicious Device",
            device_type="unknown",
        )

        result = await secure_registry.reject_pairing(request.request_id)
        assert result is True

        # Should not be in pending requests
        pending = await secure_registry.get_pending_requests()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_get_pending_requests(self, secure_registry):
        req1 = await secure_registry.request_pairing(
            device_name="Laptop",
            device_type="laptop",
        )
        req2 = await secure_registry.request_pairing(
            device_name="Phone",
            device_type="phone",
        )

        pending = await secure_registry.get_pending_requests()
        assert len(pending) == 2

        # Approve one
        await secure_registry.approve_pairing(req1.request_id)

        pending = await secure_registry.get_pending_requests()
        assert len(pending) == 2  # Still pending until confirmed


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, registry):
        secure_registry = SecureDeviceRegistry(
            registry=registry,
            max_requests_per_minute=3,
        )

        # Make 3 requests (should succeed)
        for i in range(3):
            await secure_registry.request_pairing(
                device_name=f"Device {i}",
                device_type="laptop",
            )

        # 4th request should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            await secure_registry.request_pairing(
                device_name="One more",
                device_type="laptop",
            )


class TestPresenceMonitoring:
    @pytest.mark.asyncio
    async def test_heartbeat_updates_status(self, secure_registry):
        # Pair a device
        request = await secure_registry.request_pairing(
            device_name="Test Device",
            device_type="laptop",
        )
        await secure_registry.approve_pairing(request.request_id)
        device = await secure_registry.confirm_pairing(
            request.request_id,
            request.verification_code,
        )

        # Send heartbeat
        result = await secure_registry.heartbeat(device.device_id)
        assert result is True

        # Device should be online
        updated = await secure_registry.get(device.device_id)
        assert updated.status == DeviceStatus.ONLINE

    @pytest.mark.asyncio
    async def test_offline_detection(self, registry):
        offline_devices = []

        def on_offline(device_id: str):
            offline_devices.append(device_id)

        secure_registry = SecureDeviceRegistry(
            registry=registry,
            heartbeat_interval=0.1,
            offline_threshold=0.3,
            on_device_offline=on_offline,
        )

        # Pair a device
        request = await secure_registry.request_pairing(
            device_name="Test Device",
            device_type="laptop",
        )
        await secure_registry.approve_pairing(request.request_id)
        device = await secure_registry.confirm_pairing(
            request.request_id,
            request.verification_code,
        )

        # Set to online
        await secure_registry.heartbeat(device.device_id)

        # Start monitoring
        await secure_registry.start()

        try:
            # Wait for offline detection
            await asyncio.sleep(0.5)

            # Device should be offline
            updated = await secure_registry.get(device.device_id)
            assert updated.status == DeviceStatus.OFFLINE
            assert device.device_id in offline_devices
        finally:
            await secure_registry.stop()

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, secure_registry):
        await secure_registry.start()
        assert secure_registry._running is True
        assert secure_registry._monitor_task is not None

        await secure_registry.stop()
        assert secure_registry._running is False


class TestDelegatedOperations:
    @pytest.mark.asyncio
    async def test_list_devices(self, secure_registry):
        # Pair some devices
        for name in ["Laptop", "Phone"]:
            request = await secure_registry.request_pairing(
                device_name=name,
                device_type=name.lower(),
            )
            await secure_registry.approve_pairing(request.request_id)
            await secure_registry.confirm_pairing(
                request.request_id,
                request.verification_code,
            )

        devices = await secure_registry.list_devices()
        assert len(devices) == 2

    @pytest.mark.asyncio
    async def test_unregister_device(self, secure_registry):
        request = await secure_registry.request_pairing(
            device_name="Test Device",
            device_type="laptop",
        )
        await secure_registry.approve_pairing(request.request_id)
        device = await secure_registry.confirm_pairing(
            request.request_id,
            request.verification_code,
        )

        result = await secure_registry.unregister(device.device_id)
        assert result is True

        found = await secure_registry.get(device.device_id)
        assert found is None

    @pytest.mark.asyncio
    async def test_block_device(self, secure_registry):
        request = await secure_registry.request_pairing(
            device_name="Bad Device",
            device_type="laptop",
        )
        await secure_registry.approve_pairing(request.request_id)
        device = await secure_registry.confirm_pairing(
            request.request_id,
            request.verification_code,
        )

        result = await secure_registry.block(device.device_id)
        assert result is True

        blocked = await secure_registry.get(device.device_id)
        assert blocked.status == DeviceStatus.BLOCKED
