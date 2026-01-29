"""
Moltbot compatibility adapter for canonical Gateway APIs.

Provides a thin translation layer between Moltbot's models and the
canonical gateway runtime.
"""

from __future__ import annotations

from dataclasses import replace

from aragora.gateway.canonical_api import GatewayRuntime
from aragora.gateway.device_registry import DeviceNode as GatewayDeviceNode
from aragora.gateway.device_registry import DeviceStatus as GatewayDeviceStatus
from aragora.gateway.inbox import InboxMessage as GatewayInboxMessage
from aragora.gateway.inbox import MessagePriority

from .models import DeviceNode, DeviceNodeConfig, InboxMessage as MoltbotInboxMessage


class MoltbotGatewayAdapter:
    """Adapter that maps Moltbot models onto canonical gateway primitives."""

    def __init__(
        self, gateway: GatewayRuntime | None = None, gateway_id: str = "local-gateway"
    ) -> None:
        self._gateway = gateway or GatewayRuntime()
        self._gateway_id = gateway_id

    async def register_device(
        self,
        config: DeviceNodeConfig,
        user_id: str,
        tenant_id: str | None = None,
    ) -> DeviceNode:
        gateway_device = GatewayDeviceNode(
            name=config.name,
            device_type=config.device_type,
            capabilities=list(config.capabilities),
            status=GatewayDeviceStatus.PAIRED,
            metadata={
                "moltbot_user_id": user_id,
                "moltbot_tenant_id": tenant_id,
                "connection_type": config.connection_type,
                "heartbeat_interval": config.heartbeat_interval,
                "moltbot_metadata": dict(config.metadata),
            },
        )
        gateway_device = await self._gateway.register_device(gateway_device)
        return DeviceNode(
            id=gateway_device.device_id,
            config=replace(config),
            user_id=user_id,
            gateway_id=self._gateway_id,
            tenant_id=tenant_id,
        )

    async def route_message(
        self,
        message: MoltbotInboxMessage,
        *,
        channel: str | None = None,
        device_id: str | None = None,
    ):
        gateway_message = GatewayInboxMessage(
            message_id=message.id,
            channel=channel or message.channel_id,
            sender=message.user_id,
            content=message.content,
            thread_id=message.thread_id,
            priority=MessagePriority.NORMAL,
            metadata={
                "moltbot_message_id": message.id,
                "moltbot_channel_id": message.channel_id,
                "moltbot_direction": message.direction,
                "moltbot_content_type": message.content_type,
                "moltbot_metadata": dict(message.metadata),
            },
        )
        return await self._gateway.route_message(
            channel=gateway_message.channel,
            message=gateway_message,
            device_id=device_id,
        )
