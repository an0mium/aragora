"""
Device Registry - Registry of device capabilities and permissions.

Tracks devices (laptops, phones, servers) that connect to the gateway,
their capabilities, and access permissions.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.gateway.persistence import GatewayStore


class DeviceStatus(Enum):
    """Device status."""

    ONLINE = "online"
    OFFLINE = "offline"
    PAIRED = "paired"
    BLOCKED = "blocked"


@dataclass
class DeviceNode:
    """A device registered with the gateway."""

    device_id: str = ""
    name: str = ""
    device_type: str = ""  # laptop, phone, server, tablet
    capabilities: list[str] = field(default_factory=list)  # browser, shell, mic, camera
    status: DeviceStatus = DeviceStatus.OFFLINE
    paired_at: float | None = None
    last_seen: float | None = None
    allowed_channels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DeviceRegistry:
    """
    Registry for device capabilities and permissions.

    Features:
    - Register and track devices
    - Manage device capabilities (browser, shell, mic, etc.)
    - Device pairing and allowlisting
    - Online/offline status tracking
    """

    def __init__(self, store: "GatewayStore | None" = None) -> None:
        self._devices: dict[str, DeviceNode] = {}
        self._store = store

    async def hydrate(self) -> None:
        """Load persisted devices into memory."""
        if not self._store:
            return
        devices = await self._store.load_devices()
        self._devices = {device.device_id: device for device in devices if device.device_id}

    async def register(self, device: DeviceNode) -> str:
        """
        Register a device. Returns the device ID.

        If device_id is empty, auto-generates one.
        """
        if not device.device_id:
            device.device_id = (
                f"dev-{hashlib.sha256(f'{device.name}-{time.time()}'.encode()).hexdigest()[:8]}"
            )

        device.status = DeviceStatus.PAIRED
        device.paired_at = time.time()
        device.last_seen = time.time()
        self._devices[device.device_id] = device
        if self._store:
            await self._store.save_device(device)
        return device.device_id

    async def unregister(self, device_id: str) -> bool:
        """Unregister a device."""
        if device_id in self._devices:
            del self._devices[device_id]
            if self._store:
                await self._store.delete_device(device_id)
            return True
        return False

    async def get(self, device_id: str) -> DeviceNode | None:
        """Get a device by ID."""
        return self._devices.get(device_id)

    async def list_devices(
        self,
        status: DeviceStatus | None = None,
        device_type: str | None = None,
    ) -> list[DeviceNode]:
        """List devices with optional filters."""
        results = []
        for device in self._devices.values():
            if status and device.status != status:
                continue
            if device_type and device.device_type != device_type:
                continue
            results.append(device)
        return results

    async def heartbeat(self, device_id: str) -> bool:
        """Update device last_seen timestamp."""
        device = self._devices.get(device_id)
        if device:
            device.last_seen = time.time()
            device.status = DeviceStatus.ONLINE
            if self._store:
                await self._store.save_device(device)
            return True
        return False

    async def block(self, device_id: str) -> bool:
        """Block a device."""
        device = self._devices.get(device_id)
        if device:
            device.status = DeviceStatus.BLOCKED
            if self._store:
                await self._store.save_device(device)
            return True
        return False

    async def count(self) -> int:
        """Get the number of registered devices."""
        return len(self._devices)

    async def has_capability(self, device_id: str, capability: str) -> bool:
        """Check if a device has a specific capability."""
        device = self._devices.get(device_id)
        if device:
            return capability in device.capabilities
        return False
