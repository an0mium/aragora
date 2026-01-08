"""
Base class for API-based agents.
"""

from __future__ import annotations

from aragora.agents.base import CritiqueMixin
from aragora.core import Agent, Message
from aragora.resilience import CircuitBreaker


class APIAgent(CritiqueMixin, Agent):
    """Base class for API-based agents.

    Includes circuit breaker protection for graceful failure handling.
    Supports dependency injection of circuit breaker for better testability.
    """

    def __init__(
        self,
        name: str,
        model: str,
        role: str = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
        base_url: str | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        enable_circuit_breaker: bool = True,
    ):
        super().__init__(name, model, role)
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url
        self.agent_type = "api"  # Default for API agents
        self.enable_circuit_breaker = enable_circuit_breaker

        # Use provided circuit breaker or create a new local instance
        # This avoids global state and improves testability
        if circuit_breaker is not None:
            self._circuit_breaker = circuit_breaker
        elif enable_circuit_breaker:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=3,
                cooldown_seconds=60.0,
            )
        else:
            self._circuit_breaker = None

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        """Get the circuit breaker for this agent."""
        return self._circuit_breaker

    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is open (blocking requests)."""
        if self._circuit_breaker is None:
            return False
        return not self._circuit_breaker.can_proceed()

    def _build_context_prompt(self, context: list[Message] | None = None) -> str:
        """Build context from previous messages.

        Delegates to CritiqueMixin (no truncation for API agents).
        """
        return CritiqueMixin._build_context_prompt(self, context, truncate=False)

    # _parse_critique is inherited from CritiqueMixin


__all__ = ["APIAgent"]
