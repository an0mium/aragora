"""
Acceptance tests for device node runtime (Moltbot parity scaffold).
"""

import pytest

from aragora.gateway import DeviceNodeRuntime, DeviceNodeRuntimeConfig, DeviceRegistry
from aragora.gateway.device_registry import DeviceStatus


@pytest.mark.asyncio
async def test_device_pair_and_heartbeat():
    registry = DeviceRegistry()
    runtime = DeviceNodeRuntime(
        registry,
        DeviceNodeRuntimeConfig(
            name="Test Device",
            device_type="laptop",
            capabilities=["browser", "shell"],
            allowed_channels=["slack"],
        ),
    )

    device_id = await runtime.pair()
    assert device_id
    assert await runtime.is_paired() is True
    assert await runtime.supports("browser") is True

    heartbeat_ok = await runtime.heartbeat()
    assert heartbeat_ok is True

    device = await registry.get(device_id)
    assert device is not None
    assert device.status == DeviceStatus.ONLINE


@pytest.mark.asyncio
async def test_device_unregister():
    registry = DeviceRegistry()
    runtime = DeviceNodeRuntime(
        registry,
        DeviceNodeRuntimeConfig(
            name="Test Device",
            device_type="laptop",
        ),
    )

    device_id = await runtime.pair()
    assert device_id
    removed = await runtime.unregister()
    assert removed is True
    assert await runtime.is_paired() is False
