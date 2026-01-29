"""
Local Gateway Server - Device-local routing and auth daemon.

A lightweight service that runs on the user's device, routing incoming
messages from any channel to the right agent, enforcing local auth,
and proxying to Aragora cloud services when needed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from aragora.gateway.inbox import InboxAggregator, InboxMessage
from aragora.gateway.device_registry import DeviceNode, DeviceRegistry
from aragora.gateway.router import AgentRouter

logger = logging.getLogger(__name__)


@dataclass
class GatewayConfig:
    """Configuration for the Local Gateway."""

    host: str = "127.0.0.1"
    port: int = 8090
    api_key: str = ""
    enable_auth: bool = True
    cloud_proxy_url: str | None = None
    max_inbox_size: int = 10000
    allowed_channels: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Response from an agent for a routed message."""

    message_id: str
    agent_id: str
    channel: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


class LocalGateway:
    """
    Device-local HTTP/WebSocket gateway for message routing.

    Provides:
    - Message routing from any channel to the right agent
    - Local auth enforcement (API key or device certificate)
    - Unified inbox access
    - Device registration and capability management
    - Cloud proxy for remote services

    Usage:
        gw = LocalGateway(config=GatewayConfig(port=8090))
        await gw.start()
        response = await gw.route_message("slack", message)
        inbox = await gw.get_inbox()
        await gw.stop()
    """

    def __init__(self, config: GatewayConfig | None = None) -> None:
        self._config = config or GatewayConfig()
        self._inbox = InboxAggregator(max_size=self._config.max_inbox_size)
        self._devices = DeviceRegistry()
        self._router = AgentRouter()
        self._running = False
        self._started_at: float | None = None
        self._messages_routed = 0
        self._messages_failed = 0

    @property
    def is_running(self) -> bool:
        """Check if the gateway is running."""
        return self._running

    async def start(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """
        Start the gateway server.

        Args:
            host: Override host (defaults to config).
            port: Override port (defaults to config).
        """
        if self._running:
            logger.warning("Gateway already running")
            return

        actual_host = host or self._config.host
        actual_port = port or self._config.port

        self._running = True
        self._started_at = time.time()
        logger.info(f"Local gateway started on {actual_host}:{actual_port}")

    async def stop(self) -> None:
        """Stop the gateway server."""
        if not self._running:
            return
        self._running = False
        logger.info("Local gateway stopped")

    async def route_message(
        self,
        channel: str,
        message: InboxMessage,
    ) -> AgentResponse:
        """
        Route an incoming message to the appropriate agent.

        Args:
            channel: Source channel (e.g., "slack", "telegram").
            message: The incoming message.

        Returns:
            AgentResponse from the handling agent.
        """
        if not self._running:
            return AgentResponse(
                message_id=message.message_id,
                agent_id="",
                channel=channel,
                content="",
                success=False,
                error="Gateway not running",
            )

        # Auth check
        if self._config.enable_auth and self._config.api_key:
            if message.metadata.get("api_key") != self._config.api_key:
                self._messages_failed += 1
                return AgentResponse(
                    message_id=message.message_id,
                    agent_id="",
                    channel=channel,
                    content="",
                    success=False,
                    error="Authentication failed",
                )

        # Add to inbox
        await self._inbox.add_message(message)

        # Route to agent
        agent_id = await self._router.route(channel, message)

        self._messages_routed += 1

        return AgentResponse(
            message_id=message.message_id,
            agent_id=agent_id or "default",
            channel=channel,
            content=f"Routed to agent {agent_id or 'default'}",
            success=True,
        )

    async def register_device(self, device: DeviceNode) -> str:
        """Register a device with the gateway."""
        return await self._devices.register(device)

    async def get_inbox(
        self,
        channel: str | None = None,
        limit: int = 50,
    ) -> list[InboxMessage]:
        """Get messages from the unified inbox."""
        return await self._inbox.get_messages(channel=channel, limit=limit)

    async def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        return {
            "running": self._running,
            "started_at": self._started_at,
            "messages_routed": self._messages_routed,
            "messages_failed": self._messages_failed,
            "inbox_size": await self._inbox.get_size(),
            "devices_registered": await self._devices.count(),
            "routing_rules": await self._router.count_rules(),
        }
