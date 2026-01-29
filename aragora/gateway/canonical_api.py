"""
Canonical Gateway API for device/inbox routing.

Adapters (e.g., Moltbot compatibility layers) should call into this
API instead of reimplementing gateway primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from aragora.gateway.capability_router import CapabilityRouter, RoutingResult
from aragora.gateway.device_registry import DeviceNode, DeviceRegistry, DeviceStatus
from aragora.gateway.inbox import InboxAggregator, InboxMessage, MessagePriority


@runtime_checkable
class GatewayAPI(Protocol):
    async def register_device(self, device: DeviceNode) -> DeviceNode: ...
    async def get_device(self, device_id: str) -> DeviceNode | None: ...
    async def list_devices(
        self,
        *,
        status: DeviceStatus | None = None,
        device_type: str | None = None,
    ) -> list[DeviceNode]: ...

    async def enqueue_message(self, message: InboxMessage) -> None: ...
    async def get_messages(
        self,
        *,
        channel: str | None = None,
        is_read: bool | None = None,
        priority: MessagePriority | None = None,
        limit: int = 50,
    ) -> list[InboxMessage]: ...

    async def route_message(
        self,
        channel: str,
        message: InboxMessage,
        device_id: str | None = None,
    ) -> RoutingResult: ...


@dataclass
class GatewayRuntime(GatewayAPI):
    """In-process gateway runtime for canonical routing and inbox management."""

    default_agent: str = "default"
    max_inbox_size: int = 10000
    registry: DeviceRegistry = field(default_factory=DeviceRegistry)
    inbox: InboxAggregator = field(init=False)
    router: CapabilityRouter = field(init=False)

    def __post_init__(self) -> None:
        self.inbox = InboxAggregator(max_size=self.max_inbox_size)
        self.router = CapabilityRouter(
            default_agent=self.default_agent, device_registry=self.registry
        )

    async def register_device(self, device: DeviceNode) -> DeviceNode:
        await self.registry.register(device)
        return device

    async def get_device(self, device_id: str) -> DeviceNode | None:
        return await self.registry.get(device_id)

    async def list_devices(
        self,
        *,
        status: DeviceStatus | None = None,
        device_type: str | None = None,
    ) -> list[DeviceNode]:
        return await self.registry.list_devices(status=status, device_type=device_type)

    async def enqueue_message(self, message: InboxMessage) -> None:
        await self.inbox.add_message(message)

    async def get_messages(
        self,
        *,
        channel: str | None = None,
        is_read: bool | None = None,
        priority: MessagePriority | None = None,
        limit: int = 50,
    ) -> list[InboxMessage]:
        return await self.inbox.get_messages(
            channel=channel,
            is_read=is_read,
            priority=priority,
            limit=limit,
        )

    async def route_message(
        self,
        channel: str,
        message: InboxMessage,
        device_id: str | None = None,
    ) -> RoutingResult:
        await self.inbox.add_message(message)
        return await self.router.route_with_details(channel, message, device_id=device_id)
