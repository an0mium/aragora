"""
Protocol Bridge.

Unified interface for MCP and A2A protocols.
Allows seamless interaction with external tools and agents
regardless of the underlying protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

from aragora.protocols.a2a import (
    A2AClient,
    A2AServer,
    AgentCard,
    AgentCapability,
    ContextItem,
    TaskRequest,
    TaskResult,
    TaskStatus,
)

if TYPE_CHECKING:
    from aragora.agents.base import BaseDebateAgent as BaseAgent

logger = logging.getLogger(__name__)


class Protocol(str, Enum):
    """Supported protocols."""

    MCP = "mcp"
    A2A = "a2a"


@dataclass
class ExternalResource:
    """An external resource accessible via protocol."""

    protocol: Protocol
    uri: str
    name: str
    description: str = ""
    mime_type: str = "application/json"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BridgeConfig:
    """Configuration for the protocol bridge."""

    # MCP settings
    enable_mcp: bool = True
    mcp_timeout: float = 60.0

    # A2A settings
    enable_a2a: bool = True
    a2a_timeout: float = 300.0
    a2a_registries: list[str] = field(default_factory=list)

    # General settings
    default_protocol: Protocol = Protocol.A2A
    cache_agent_cards: bool = True


class ProtocolBridge:
    """
    Unified interface for MCP and A2A protocols.

    Provides:
    - Automatic protocol detection and routing
    - Unified tool/agent invocation
    - Resource access across protocols
    - Aragora agent wrapping for external exposure
    """

    def __init__(self, config: BridgeConfig | None = None):
        """
        Initialize the protocol bridge.

        Args:
            config: Bridge configuration
        """
        self.config = config or BridgeConfig()

        # Protocol clients
        self._a2a_client: A2AClient | None = None
        self._a2a_server: A2AServer | None = None

        # Cached resources and agents
        self._external_agents: dict[str, AgentCard] = {}
        self._external_resources: dict[str, ExternalResource] = {}

    async def initialize(self) -> None:
        """Initialize protocol clients."""
        if self.config.enable_a2a:
            self._a2a_client = A2AClient(timeout=self.config.a2a_timeout)
            self._a2a_server = A2AServer()

            # Discover agents from registries
            for registry in self.config.a2a_registries:
                try:
                    agents = await self._a2a_client.discover_agents(registry)
                    for agent in agents:
                        self._external_agents[agent.name] = agent
                except Exception as e:
                    logger.warning(f"Failed to discover agents from {registry}: {e}")

        logger.info("Protocol bridge initialized")

    async def invoke_external(
        self,
        target: str,
        task: str,
        context: Optional[list[dict[str, Any]]] = None,
        protocol: Protocol | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Invoke an external tool or agent.

        Args:
            target: Target URL or agent name
            task: Task description/instruction
            context: Optional context items
            protocol: Protocol to use (auto-detected if not specified)
            **kwargs: Additional arguments

        Returns:
            Result from the external invocation
        """
        # Detect protocol
        if protocol is None:
            protocol = self._detect_protocol(target)

        if protocol == Protocol.MCP:
            return await self._invoke_mcp(target, task, context, **kwargs)
        elif protocol == Protocol.A2A:
            return await self._invoke_a2a(target, task, context, **kwargs)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

    def _detect_protocol(self, target: str) -> Protocol:
        """Detect the appropriate protocol for a target."""
        # Check if it's a known A2A agent
        if target in self._external_agents:
            return Protocol.A2A

        # Check URL schemes
        if target.startswith("mcp://"):
            return Protocol.MCP
        if target.startswith("a2a://"):
            return Protocol.A2A

        # Default
        return self.config.default_protocol

    async def _invoke_mcp(
        self,
        target: str,
        task: str,
        context: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke a tool via MCP protocol.

        Connects to an MCP server and calls a named tool. The ``target``
        parameter is treated as the tool name. If a ``server_url`` keyword
        argument is provided, the bridge connects to that server via SSE;
        otherwise it falls back to the local ``AragoraMCPServer`` instance.

        Args:
            target: MCP tool name (e.g. ``"debate"``). ``mcp://`` prefix
                is stripped automatically.
            task: Task description, passed as the ``"query"`` argument
                to the tool.
            context: Optional context items forwarded as a ``"context"``
                tool argument.
            **kwargs: ``server_url`` for remote MCP servers; any other
                keyword arguments are forwarded to the tool call.
        """
        # Strip mcp:// prefix if present
        tool_name = target.removeprefix("mcp://").strip("/")
        logger.info(f"MCP invocation: tool={tool_name}")

        tool_arguments: dict[str, Any] = {"query": task}
        if context:
            tool_arguments["context"] = context
        # Forward extra kwargs as tool arguments (except internal ones)
        for k, v in kwargs.items():
            if k not in ("server_url",):
                tool_arguments[k] = v

        server_url = kwargs.get("server_url")

        # Try remote MCP server via SSE transport if a URL is given
        if server_url:
            try:
                from mcp import ClientSession
                from mcp.client.sse import sse_client

                async with sse_client(server_url) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, tool_arguments)
                        # Extract text content from MCP response
                        output_parts = []
                        for content_item in result.content:
                            if hasattr(content_item, "text"):
                                output_parts.append(content_item.text)
                        return {
                            "protocol": "mcp",
                            "target": target,
                            "status": "success",
                            "result": "\n".join(output_parts) if output_parts else str(result),
                        }
            except ImportError:
                logger.warning("mcp package not installed, cannot connect to remote MCP server")
            except Exception as e:
                logger.warning(f"Remote MCP invocation failed: {e}")
                return {
                    "protocol": "mcp",
                    "target": target,
                    "status": "error",
                    "error": str(e),
                }

        # Fallback: invoke on the local AragoraMCPServer
        try:
            from aragora.mcp.server import AragoraMCPServer

            server = AragoraMCPServer()
            result = await server.call_tool(tool_name, tool_arguments)
            return {
                "protocol": "mcp",
                "target": target,
                "status": "success",
                "result": result,
            }
        except ImportError:
            logger.warning("AragoraMCPServer not available")
        except Exception as e:
            logger.warning(f"Local MCP invocation failed: {e}")
            return {
                "protocol": "mcp",
                "target": target,
                "status": "error",
                "error": str(e),
            }

        return {
            "protocol": "mcp",
            "target": target,
            "status": "error",
            "error": "No MCP server available (install mcp package or start local server)",
        }

    async def _invoke_a2a(
        self,
        target: str,
        task: str,
        context: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke via A2A protocol."""
        if not self._a2a_client:
            raise RuntimeError("A2A client not initialized")

        # Convert context to A2A format
        a2a_context = []
        if context:
            for ctx in context:
                a2a_context.append(
                    ContextItem(
                        type=ctx.get("type", "text"),
                        content=ctx.get("content", ""),
                        metadata=ctx.get("metadata", {}),
                    )
                )

        # Get capability from kwargs
        capability = None
        if "capability" in kwargs:
            capability = AgentCapability(kwargs["capability"])

        # Invoke agent
        result = await self._a2a_client.invoke(
            agent_name=target,
            instruction=task,
            context=a2a_context,
            capability=capability,
        )

        return result.to_dict()

    async def stream_invoke(
        self,
        target: str,
        task: str,
        context: Optional[list[dict[str, Any]]] = None,
        protocol: Protocol | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Invoke with streaming output.

        Args:
            target: Target URL or agent name
            task: Task description
            context: Optional context
            protocol: Protocol to use

        Yields:
            Stream events
        """
        if protocol is None:
            protocol = self._detect_protocol(target)

        if protocol == Protocol.A2A and self._a2a_client:
            a2a_context = []
            if context:
                for ctx in context:
                    a2a_context.append(
                        ContextItem(
                            type=ctx.get("type", "text"),
                            content=ctx.get("content", ""),
                        )
                    )

            async for event in self._a2a_client.stream_invoke(
                agent_name=target,
                instruction=task,
                context=a2a_context,
            ):
                yield event
        else:
            yield {
                "type": "error",
                "message": "Streaming not supported for this protocol/target",
            }

    def wrap_aragora_agent(
        self,
        agent: "BaseAgent",
        capabilities: Optional[list[AgentCapability]] = None,
    ) -> AgentCard:
        """
        Wrap an Aragora agent as an A2A agent card.

        Args:
            agent: Aragora agent to wrap
            capabilities: Capabilities to advertise

        Returns:
            AgentCard for the wrapped agent
        """
        return AgentCard(
            name=f"aragora-{agent.name}",
            description=f"Aragora agent: {agent.role}",
            capabilities=capabilities or [AgentCapability.REASONING],
            input_modes=["text"],
            output_modes=["text"],
            organization="aragora",
            tags=["aragora", agent.role],
        )

    def register_external_agent(self, agent: AgentCard) -> None:
        """Register an external agent for invocation."""
        self._external_agents[agent.name] = agent
        if self._a2a_client:
            self._a2a_client.register_agent(agent)

    def list_external_agents(
        self,
        capability: AgentCapability | None = None,
    ) -> list[AgentCard]:
        """List available external agents."""
        agents = list(self._external_agents.values())

        if capability:
            agents = [a for a in agents if a.supports_capability(capability)]

        return agents

    def get_a2a_server(self) -> A2AServer | None:
        """Get the A2A server for handling incoming requests."""
        return self._a2a_server

    async def handle_incoming_task(
        self,
        request: TaskRequest,
    ) -> TaskResult:
        """
        Handle an incoming A2A task request.

        Routes to the A2A server for processing.

        Args:
            request: Incoming task request

        Returns:
            Task result
        """
        if not self._a2a_server:
            return TaskResult(
                task_id=request.task_id,
                agent_name="aragora",
                status=TaskStatus.FAILED,
                error_message="A2A server not initialized",
            )

        return await self._a2a_server.handle_task(request)


# Global bridge instance
_bridge: ProtocolBridge | None = None


def get_protocol_bridge(config: BridgeConfig | None = None) -> ProtocolBridge:
    """Get or create the global protocol bridge."""
    global _bridge
    if _bridge is None:
        _bridge = ProtocolBridge(config)
    return _bridge


__all__ = [
    "Protocol",
    "ExternalResource",
    "BridgeConfig",
    "ProtocolBridge",
    "get_protocol_bridge",
]
