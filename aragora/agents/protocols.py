"""
Protocol definitions for agent interfaces.

This module provides Protocol classes that define the expected interfaces
for agents and mixins, enabling proper type checking across the codebase
without circular imports.

Usage:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from aragora.agents.protocols import CritiqueCapable, GenerativeAgent

Protocols defined:
- GenerativeAgent: Base interface for agents that can generate responses
- CritiqueCapable: Interface for agents that can provide critiques
- StreamingAgent: Interface for agents supporting streaming responses
- CircuitBreakerAware: Interface for agents with circuit breaker integration
- FallbackCapable: Interface for agents supporting fallback providers
- TokenTrackingAgent: Interface for agents tracking token usage
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any, Optional, Protocol, Union, runtime_checkable

from aragora.core import Critique, Message


@runtime_checkable
class GenerativeAgent(Protocol):
    """
    Protocol for agents that can generate text responses.

    This is the minimal interface required for an agent to participate
    in debates and generate content.
    """

    name: str
    agent_type: str

    async def generate(
        self,
        prompt: str,
        context: Optional[list[Message]] = None,
    ) -> str:
        """Generate a response to the given prompt.

        Args:
            prompt: The input prompt/question
            context: Optional conversation history

        Returns:
            Generated text response
        """
        ...


@runtime_checkable
class CritiqueCapable(Protocol):
    """
    Protocol for agents that can provide critiques.

    Agents implementing this interface can analyze proposals from
    other agents and provide structured feedback.
    """

    name: str

    def _build_context_prompt(
        self,
        context: Optional[list[Message]] = None,
        truncate: bool = False,
        sanitize_fn: Optional[Callable[[str], str]] = None,
    ) -> str:
        """Build formatted context from previous messages.

        Args:
            context: List of previous messages
            truncate: Whether to truncate long messages
            sanitize_fn: Optional sanitization function

        Returns:
            Formatted context string
        """
        ...

    def _parse_critique(
        self,
        response: str,
        target_agent: str,
        target_content: str,
    ) -> Critique:
        """Parse a critique response into structured format.

        Args:
            response: Raw critique response text
            target_agent: Name of agent being critiqued
            target_content: Content being critiqued

        Returns:
            Structured Critique object
        """
        ...

    async def critique(
        self,
        proposal: str,
        target_agent: Optional[str] = None,
        context: Optional[list[Message]] = None,
    ) -> Critique:
        """Critique another agent's proposal.

        Args:
            proposal: The proposal to critique
            target_agent: Name of agent who made the proposal
            context: Conversation context

        Returns:
            Structured critique
        """
        ...


@runtime_checkable
class StreamingAgent(Protocol):
    """
    Protocol for agents that support streaming responses.

    Streaming allows real-time output of generated text,
    useful for long responses and better UX.
    """

    name: str

    async def generate_stream(
        self,
        prompt: str,
        context: Optional[list[Message]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the agent.

        Args:
            prompt: The input prompt
            context: Optional conversation history

        Yields:
            Text chunks as they are generated
        """
        ...


@runtime_checkable
class CircuitBreakerAware(Protocol):
    """
    Protocol for agents integrated with circuit breaker pattern.

    Circuit breakers prevent cascading failures by temporarily
    stopping requests to failing services.
    """

    name: str
    agent_type: str

    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is open (blocking requests).

        Returns:
            True if circuit is open and requests are blocked
        """
        ...

    def record_success(self) -> None:
        """Record a successful operation."""
        ...

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed operation.

        Args:
            error: The exception that caused the failure
        """
        ...


@runtime_checkable
class FallbackCapable(Protocol):
    """
    Protocol for agents supporting fallback to alternative providers.

    When primary provider fails (e.g., rate limit), agents can
    fall back to OpenRouter or other providers.
    """

    name: str

    def is_quota_error(self, status_code: int, error_text: str) -> bool:
        """Check if error indicates a quota/billing issue.

        Args:
            status_code: HTTP status code
            error_text: Error message text

        Returns:
            True if this is a quota/billing error
        """
        ...

    async def fallback_generate(
        self,
        prompt: str,
        context: Optional[list[Message]],
        original_status: int,
    ) -> Optional[str]:
        """Attempt to generate using fallback provider.

        Args:
            prompt: The original prompt
            context: Conversation context
            original_status: Status code from original failure

        Returns:
            Generated text if fallback succeeds, None otherwise
        """
        ...


@runtime_checkable
class TokenTrackingAgent(Protocol):
    """
    Protocol for agents that track token usage.

    Token tracking is essential for billing, cost management,
    and optimizing context windows.
    """

    name: str

    def _record_token_usage(self, tokens_in: int, tokens_out: int) -> None:
        """Record token usage for an API call.

        Args:
            tokens_in: Number of input/prompt tokens
            tokens_out: Number of output/completion tokens
        """
        ...

    def get_token_usage(self) -> dict[str, int]:
        """Get cumulative token usage statistics.

        Returns:
            Dict with 'total_in', 'total_out', 'last_in', 'last_out'
        """
        ...


class OpenAICompatibleBase(Protocol):
    """
    Protocol for the base class expected by OpenAICompatibleMixin.

    This defines the interface that must be provided by the base
    class (typically APIAgent) when using OpenAICompatibleMixin.
    """

    name: str
    agent_type: str
    api_key: Optional[str]
    base_url: Optional[str]
    model: str
    timeout: int

    def _build_context_prompt(
        self,
        context: Optional[list[Message]] = None,
        truncate: bool = False,
        sanitize_fn: Optional[Callable[[str], str]] = None,
    ) -> str:
        """Build formatted context from previous messages."""
        ...

    def _parse_critique(
        self,
        response: str,
        target_agent: str,
        target_content: str,
    ) -> Critique:
        """Parse a critique response into structured format."""
        ...

    def _record_token_usage(self, tokens_in: int, tokens_out: int) -> None:
        """Record token usage for billing."""
        ...


# Type alias for any agent
AnyAgent = Union[GenerativeAgent, CritiqueCapable, StreamingAgent]


__all__ = [
    "GenerativeAgent",
    "CritiqueCapable",
    "StreamingAgent",
    "CircuitBreakerAware",
    "FallbackCapable",
    "TokenTrackingAgent",
    "OpenAICompatibleBase",
    "AnyAgent",
]
