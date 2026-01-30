"""
Local Gateway Server - Device-local routing and auth daemon.

Pattern: Local-first Gateway
Inspired by: Moltbot (https://github.com/moltbot)
Aragora adaptation: Unified inbox routing with Agent Fabric integration

A lightweight service that runs on the user's device, routing incoming
messages from any channel to the right agent, enforcing local auth,
and proxying to Aragora cloud services when needed.

Endpoints:
- GET  /health              - Health check
- GET  /stats               - Gateway statistics
- GET  /inbox               - Get inbox messages
- POST /route               - Route a message to an agent
- POST /device              - Register a device
- GET  /device/{device_id}  - Get device info
- WS   /ws                  - Real-time inbox updates
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web, WSMsgType

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

    # =========================================================================
    # HTTP Server Infrastructure
    # =========================================================================

    def _create_app(self) -> web.Application:
        """Create the aiohttp web application with routes."""
        app = web.Application(middlewares=[self._auth_middleware])
        app["gateway"] = self

        # Health and stats
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/stats", self._handle_stats)

        # Inbox operations
        app.router.add_get("/inbox", self._handle_get_inbox)

        # Message routing
        app.router.add_post("/route", self._handle_route)

        # Device management
        app.router.add_post("/device", self._handle_register_device)
        app.router.add_get("/device/{device_id}", self._handle_get_device)

        # WebSocket for real-time updates
        app.router.add_get("/ws", self._handle_websocket)

        return app

    @web.middleware
    async def _auth_middleware(
        self,
        request: web.Request,
        handler: Any,
    ) -> web.Response:
        """Middleware to enforce API key authentication."""
        # Skip auth for health endpoint
        if request.path == "/health":
            return await handler(request)

        if self._config.enable_auth and self._config.api_key:
            auth_header = request.headers.get("Authorization", "")
            api_key = request.headers.get("X-API-Key", "")

            # Check Bearer token or X-API-Key header
            if auth_header.startswith("Bearer "):
                provided_key = auth_header[7:]
            else:
                provided_key = api_key

            if provided_key != self._config.api_key:
                return web.json_response(
                    {"error": "Authentication failed", "code": "AUTH_FAILED"},
                    status=401,
                )

        return await handler(request)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response(
            {
                "status": "healthy" if self._running else "unhealthy",
                "service": "aragora-gateway",
                "uptime_seconds": time.time() - self._started_at if self._started_at else 0,
            }
        )

    async def _handle_stats(self, request: web.Request) -> web.Response:
        """Get gateway statistics."""
        stats = await self.get_stats()
        return web.json_response(stats)

    async def _handle_get_inbox(self, request: web.Request) -> web.Response:
        """Get inbox messages."""
        channel = request.query.get("channel")
        limit = int(request.query.get("limit", "50"))
        unread_only = request.query.get("unread_only", "false").lower() == "true"

        messages = await self._inbox.get_messages(
            channel=channel,
            limit=limit,
            is_read=False if unread_only else None,
        )

        return web.json_response(
            {
                "messages": [
                    {
                        "message_id": m.message_id,
                        "channel": m.channel,
                        "sender": m.sender,
                        "content": m.content,
                        "timestamp": m.timestamp,
                        "is_read": m.is_read,
                        "is_replied": m.is_replied,
                        "priority": m.priority.value
                        if hasattr(m.priority, "value")
                        else m.priority,
                        "thread_id": m.thread_id,
                    }
                    for m in messages
                ],
                "total": len(messages),
            }
        )

    async def _handle_route(self, request: web.Request) -> web.Response:
        """Route a message to an agent."""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON", "code": "INVALID_JSON"},
                status=400,
            )

        # Validate required fields
        required = ["channel", "sender", "content"]
        missing = [f for f in required if f not in data]
        if missing:
            return web.json_response(
                {"error": f"Missing fields: {missing}", "code": "MISSING_FIELDS"},
                status=400,
            )

        # Create inbox message
        message = InboxMessage(
            message_id=data.get("message_id", f"msg-{uuid.uuid4().hex[:12]}"),
            channel=data["channel"],
            sender=data["sender"],
            content=data["content"],
            thread_id=data.get("thread_id"),
            metadata=data.get("metadata", {}),
        )

        # Route the message
        response = await self.route_message(data["channel"], message)

        # Notify WebSocket subscribers
        await self._notify_subscribers(message)

        return web.json_response(
            {
                "message_id": response.message_id,
                "agent_id": response.agent_id,
                "channel": response.channel,
                "success": response.success,
                "error": response.error,
            }
        )

    async def _handle_register_device(self, request: web.Request) -> web.Response:
        """Register a new device."""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON", "code": "INVALID_JSON"},
                status=400,
            )

        if "name" not in data:
            return web.json_response(
                {"error": "Missing 'name' field", "code": "MISSING_NAME"},
                status=400,
            )

        device = DeviceNode(
            device_id=data.get("device_id"),
            name=data["name"],
            device_type=data.get("device_type", "unknown"),
            capabilities=data.get("capabilities", []),
        )

        device_id = await self.register_device(device)

        return web.json_response(
            {
                "device_id": device_id,
                "status": "registered",
            },
            status=201,
        )

    async def _handle_get_device(self, request: web.Request) -> web.Response:
        """Get device information."""
        device_id = request.match_info["device_id"]
        device = await self._devices.get(device_id)

        if not device:
            return web.json_response(
                {"error": "Device not found", "code": "NOT_FOUND"},
                status=404,
            )

        return web.json_response(
            {
                "device_id": device.device_id,
                "name": device.name,
                "device_type": device.device_type,
                "status": device.status.value
                if hasattr(device.status, "value")
                else str(device.status),
                "capabilities": device.capabilities,
                "last_seen": device.last_seen,
            }
        )

    # =========================================================================
    # WebSocket Support
    # =========================================================================

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket endpoint for real-time inbox updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        from aragora.gateway.protocol import GatewayProtocolAdapter, GatewayWebSocketProtocol

        protocol = GatewayWebSocketProtocol(GatewayProtocolAdapter(self))

        # Add to subscribers
        if not hasattr(self, "_ws_subscribers"):
            self._ws_subscribers: set[web.WebSocketResponse] = set()
        self._ws_subscribers.add(ws)

        logger.debug("WebSocket client connected")

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json(
                            {
                                "type": "error",
                                "error": {"code": "invalid_json", "message": "invalid JSON"},
                            }
                        )
                        continue

                    response = await protocol.handle_message(data)
                    if response is not None:
                        await ws.send_json(response)
                elif msg.type == WSMsgType.ERROR:
                    logger.warning(f"WebSocket error: {ws.exception()}")
        finally:
            self._ws_subscribers.discard(ws)
            logger.debug("WebSocket client disconnected")

        return ws

    async def _notify_subscribers(self, message: InboxMessage) -> None:
        """Notify WebSocket subscribers of new message."""
        if not hasattr(self, "_ws_subscribers"):
            return

        event = {
            "type": "new_message",
            "message": {
                "message_id": message.message_id,
                "channel": message.channel,
                "sender": message.sender,
                "content": message.content[:100],  # Truncate for notification
                "timestamp": message.timestamp,
            },
        }

        # Send to all subscribers (fire and forget)
        for ws in list(self._ws_subscribers):
            if not ws.closed:
                try:
                    await ws.send_json(event)
                except Exception:
                    self._ws_subscribers.discard(ws)

    # =========================================================================
    # Server Lifecycle with HTTP
    # =========================================================================

    async def start_http(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> web.AppRunner:
        """
        Start the gateway as an HTTP server.

        Args:
            host: Override host (defaults to config).
            port: Override port (defaults to config).

        Returns:
            AppRunner for managing the server lifecycle.
        """
        actual_host = host or self._config.host
        actual_port = port or self._config.port

        app = self._create_app()
        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, actual_host, actual_port)
        await site.start()

        self._running = True
        self._started_at = time.time()
        self._runner = runner

        logger.info(f"Local gateway HTTP server started on {actual_host}:{actual_port}")
        return runner

    async def stop_http(self) -> None:
        """Stop the HTTP server."""
        if hasattr(self, "_runner") and self._runner:
            await self._runner.cleanup()
            self._runner = None

        self._running = False
        logger.info("Local gateway HTTP server stopped")
