"""Unit tests for DeviceRegistry operations."""

from __future__ import annotations

import pytest

from aragora.gateway.device_registry import DeviceNode, DeviceRegistry, DeviceStatus


@pytest.fixture
def registry() -> DeviceRegistry:
    return DeviceRegistry(store=None)


def _make_device(**kwargs) -> DeviceNode:
    defaults = {
        "name": "test-device",
        "device_type": "laptop",
        "capabilities": ["browser", "shell"],
    }
    defaults.update(kwargs)
    return DeviceNode(**defaults)


@pytest.mark.asyncio
async def test_register_stores_device(registry: DeviceRegistry):
    device = _make_device()
    device_id = await registry.register(device)
    assert device_id.startswith("dev-")
    stored = await registry.get(device_id)
    assert stored is not None
    assert stored.name == "test-device"
    assert stored.device_type == "laptop"
    assert stored.status == DeviceStatus.PAIRED
    assert stored.paired_at is not None
    assert stored.last_seen is not None


@pytest.mark.asyncio
async def test_register_with_explicit_id(registry: DeviceRegistry):
    device = _make_device(device_id="my-device-1")
    device_id = await registry.register(device)
    assert device_id == "my-device-1"
    assert await registry.get("my-device-1") is not None


@pytest.mark.asyncio
async def test_get_unknown_device(registry: DeviceRegistry):
    result = await registry.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_unregister(registry: DeviceRegistry):
    device = _make_device(device_id="to-remove")
    await registry.register(device)
    assert await registry.unregister("to-remove") is True
    assert await registry.get("to-remove") is None
    assert await registry.count() == 0


@pytest.mark.asyncio
async def test_unregister_unknown(registry: DeviceRegistry):
    assert await registry.unregister("nonexistent") is False


@pytest.mark.asyncio
async def test_list_devices_filter_by_status(registry: DeviceRegistry):
    await registry.register(_make_device(device_id="d1"))
    await registry.register(_make_device(device_id="d2"))
    await registry.block("d1")
    paired = await registry.list_devices(status=DeviceStatus.PAIRED)
    blocked = await registry.list_devices(status=DeviceStatus.BLOCKED)
    assert len(paired) == 1
    assert len(blocked) == 1
    assert blocked[0].device_id == "d1"


@pytest.mark.asyncio
async def test_list_devices_filter_by_type(registry: DeviceRegistry):
    await registry.register(_make_device(device_id="phone1", device_type="phone"))
    await registry.register(_make_device(device_id="laptop1", device_type="laptop"))
    phones = await registry.list_devices(device_type="phone")
    assert len(phones) == 1
    assert phones[0].device_id == "phone1"


@pytest.mark.asyncio
async def test_heartbeat_updates_status(registry: DeviceRegistry):
    await registry.register(_make_device(device_id="hb"))
    assert await registry.heartbeat("hb") is True
    device = await registry.get("hb")
    assert device is not None
    assert device.status == DeviceStatus.ONLINE


@pytest.mark.asyncio
async def test_heartbeat_blocked_stays_blocked(registry: DeviceRegistry):
    await registry.register(_make_device(device_id="blk"))
    await registry.block("blk")
    await registry.heartbeat("blk")
    device = await registry.get("blk")
    assert device is not None
    assert device.status == DeviceStatus.BLOCKED


@pytest.mark.asyncio
async def test_heartbeat_unknown(registry: DeviceRegistry):
    assert await registry.heartbeat("ghost") is False


@pytest.mark.asyncio
async def test_has_capability(registry: DeviceRegistry):
    await registry.register(_make_device(device_id="cap", capabilities=["browser", "mic"]))
    assert await registry.has_capability("cap", "browser") is True
    assert await registry.has_capability("cap", "camera") is False
    assert await registry.has_capability("missing", "browser") is False


@pytest.mark.asyncio
async def test_count(registry: DeviceRegistry):
    assert await registry.count() == 0
    await registry.register(_make_device(device_id="a"))
    await registry.register(_make_device(device_id="b"))
    assert await registry.count() == 2
