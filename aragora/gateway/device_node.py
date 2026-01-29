"""
Device Node Runtime.

Provides a lightweight device-side runtime for pairing and heartbeat
flows against the gateway DeviceRegistry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from aragora.gateway.device_registry import DeviceNode, DeviceRegistry

@dataclass
class DeviceNodeRuntimeConfig:
    """Configuration for a device node runtime."""

    name: str
    device_type: str
    capabilities: list[str] = field(default_factory=list)
    allowed_channels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

class DeviceNodeRuntime:
    """
    Device-side runtime wrapper around DeviceRegistry.

    This provides a simple pairing + heartbeat flow that can be used
    in tests and extended into a full device client.
    """

    def __init__(self, registry: DeviceRegistry, config: DeviceNodeRuntimeConfig):
        self._registry = registry
        self._config = config
        self._device_id: str | None = None
        self._paired_at: float | None = None

    @property
    def device_id(self) -> str | None:
        return self._device_id

    async def pair(self) -> str:
        """Register the device and return its device_id."""
        device = DeviceNode(
            name=self._config.name,
            device_type=self._config.device_type,
            capabilities=self._config.capabilities,
            allowed_channels=self._config.allowed_channels,
            metadata=self._config.metadata,
        )
        self._device_id = await self._registry.register(device)
        self._paired_at = time.time()
        return self._device_id

    async def heartbeat(self) -> bool:
        """Send a heartbeat for this device."""
        if not self._device_id:
            return False
        return await self._registry.heartbeat(self._device_id)

    async def is_paired(self) -> bool:
        """Check if the device is paired and registered."""
        if not self._device_id:
            return False
        device = await self._registry.get(self._device_id)
        return device is not None

    async def supports(self, capability: str) -> bool:
        """Check if the device supports a capability."""
        if not self._device_id:
            return False
        return await self._registry.has_capability(self._device_id, capability)

    async def unregister(self) -> bool:
        """Unregister this device."""
        if not self._device_id:
            return False
        result = await self._registry.unregister(self._device_id)
        if result:
            self._device_id = None
        return result
