"""
Local Gateway - Moltbot parity extension for consumer/device interface.

Provides a device-local routing and auth service that aggregates messages
from multiple channels into a unified inbox, routes them to appropriate
agents, and manages device registrations.

This module implements the Moltbot consumer interface model on top of
the existing Aragora connector infrastructure.

Key concepts:
- LocalGateway: Device-local HTTP service for message routing and auth.
- InboxAggregator: Unified inbox across all connected channels.
- DeviceRegistry: Registry of device capabilities and permissions.
- AgentRouter: Per-channel/account agent assignment rules.

Usage:
    from aragora.gateway import LocalGateway

    gw = LocalGateway()
    await gw.start(host="127.0.0.1", port=8090)
"""

from aragora.gateway.server import LocalGateway, GatewayConfig
from aragora.gateway.inbox import InboxAggregator, InboxMessage, InboxThread
from aragora.gateway.device_registry import DeviceNode, DeviceRegistry
from aragora.gateway.router import AgentRouter, RoutingRule

__all__ = [
    "LocalGateway",
    "GatewayConfig",
    "InboxAggregator",
    "InboxMessage",
    "InboxThread",
    "DeviceNode",
    "DeviceRegistry",
    "AgentRouter",
    "RoutingRule",
]
