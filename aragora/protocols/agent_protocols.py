"""Agent protocol definitions.

Provides Protocol classes for agent implementations including
basic agents, streaming agents, and tool-using agents.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from collections.abc import AsyncIterator


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for agent implementations.

    All agents must have a name and be able to respond to prompts.
    Streaming is optional but recommended for long responses.

    Example:
        class MyAgent:
            name = "my-agent"

            async def respond(self, prompt: str, context: str | None = None) -> str:
                return "Response"

        agent: AgentProtocol = MyAgent()  # Type checks correctly
    """

    name: str

    async def respond(self, prompt: str, context: str | None = None) -> str:
        """Generate a response to the given prompt."""
        ...


@runtime_checkable
class StreamingAgentProtocol(AgentProtocol, Protocol):
    """Protocol for agents that support streaming responses."""

    async def stream(self, prompt: str, context: str | None = None) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...


@runtime_checkable
class ToolUsingAgentProtocol(AgentProtocol, Protocol):
    """Protocol for agents that can use tools."""

    available_tools: list[str]

    async def respond_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        context: str | None = None,
    ) -> str:
        """Generate response with tool use."""
        ...


__all__ = [
    "AgentProtocol",
    "StreamingAgentProtocol",
    "ToolUsingAgentProtocol",
]
