"""
Base class for API-based agents.
"""

from __future__ import annotations

from aragora.agents.base import CritiqueMixin
from aragora.core import Agent, Message
from aragora.resilience import CircuitBreaker, get_circuit_breaker


class APIAgent(CritiqueMixin, Agent):
    """Base class for API-based agents.

    Includes circuit breaker protection for graceful failure handling.
    Supports dependency injection of circuit breaker for better testability.
    Supports persona-based generation parameters for diversity.
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
        # Circuit breaker configuration (allows per-agent tuning)
        circuit_breaker_threshold: int = 8,
        circuit_breaker_cooldown: float = 90.0,
        # Generation parameters (can be set from Persona)
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
    ):
        super().__init__(name, model, role)  # type: ignore[arg-type]
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url
        self.agent_type = "api"  # Default for API agents
        self.enable_circuit_breaker = enable_circuit_breaker

        # Generation parameters (from persona or explicit)
        self.temperature = temperature  # None means use provider default
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty

        # Use provided circuit breaker, global registry, or disable
        # Global registry ensures consistent state across agent instances
        if circuit_breaker is not None:
            self._circuit_breaker = circuit_breaker
        elif enable_circuit_breaker:
            # Use global registry with agent name for shared state
            self._circuit_breaker = get_circuit_breaker(
                f"agent_{name}",
                failure_threshold=circuit_breaker_threshold,
                cooldown_seconds=circuit_breaker_cooldown,
            )
        else:
            self._circuit_breaker = None

        # Token usage tracking for billing
        self._last_tokens_in: int = 0
        self._last_tokens_out: int = 0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0

    def set_generation_params(
        self,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
    ) -> None:
        """Set generation parameters, typically from a Persona."""
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if frequency_penalty is not None:
            self.frequency_penalty = frequency_penalty

    def get_generation_params(self) -> dict[str, float]:
        """Get generation parameters as a dict (excludes None values)."""
        params = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        return params

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        """Get the circuit breaker for this agent."""
        return self._circuit_breaker

    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is open (blocking requests)."""
        if self._circuit_breaker is None:
            return False
        return not self._circuit_breaker.can_proceed()

    def _record_token_usage(self, tokens_in: int, tokens_out: int) -> None:
        """Record token usage from an API call.

        Called by subclasses after successful API calls to track usage.

        Args:
            tokens_in: Input/prompt tokens used
            tokens_out: Output/completion tokens used
        """
        self._last_tokens_in = tokens_in
        self._last_tokens_out = tokens_out
        self._total_tokens_in += tokens_in
        self._total_tokens_out += tokens_out

    @property
    def last_tokens_in(self) -> int:
        """Get input tokens from last API call."""
        return self._last_tokens_in

    @property
    def last_tokens_out(self) -> int:
        """Get output tokens from last API call."""
        return self._last_tokens_out

    @property
    def total_tokens_in(self) -> int:
        """Get total input tokens across all API calls."""
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        """Get total output tokens across all API calls."""
        return self._total_tokens_out

    def get_token_usage(self) -> dict[str, int]:
        """Get token usage summary for billing.

        Returns:
            Dict with tokens_in, tokens_out, total_tokens_in, total_tokens_out
        """
        return {
            "tokens_in": self._last_tokens_in,
            "tokens_out": self._last_tokens_out,
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
        }

    def reset_token_usage(self) -> None:
        """Reset token usage counters (e.g., at start of new debate)."""
        self._last_tokens_in = 0
        self._last_tokens_out = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    def _build_context_prompt(
        self,
        context: list[Message] | None = None,
        truncate: bool = False,
        sanitize_fn: object = None,
    ) -> str:
        """Build context from previous messages.

        Delegates to CritiqueMixin. API agents typically don't truncate.

        Args:
            context: List of previous messages
            truncate: Whether to truncate (default False for API agents)
            sanitize_fn: Optional sanitization function (unused by API agents)
        """
        # API agents typically don't need truncation, but respect the parameter
        return CritiqueMixin._build_context_prompt(self, context, truncate=truncate)

    # _parse_critique is inherited from CritiqueMixin


__all__ = ["APIAgent"]
