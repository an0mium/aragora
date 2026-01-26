"""
Circuit breaker configuration module.

Provides configurable thresholds for circuit breakers, with support for
per-provider defaults and environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for a circuit breaker instance.

    All thresholds can be customized per-agent or per-provider to match
    the reliability characteristics of different services.

    Attributes:
        failure_threshold: Number of consecutive failures before opening circuit.
        success_threshold: Number of successes in half-open state before closing.
        timeout_seconds: Time in seconds the circuit stays open before half-open.
        half_open_max_calls: Maximum concurrent calls allowed in half-open state.

    Example:
        # Create a strict config for unreliable services
        strict_config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=120.0,
        )

        # Create a lenient config for stable services
        lenient_config = CircuitBreakerConfig(
            failure_threshold=10,
            timeout_seconds=30.0,
        )
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")

    def with_overrides(
        self,
        failure_threshold: Optional[int] = None,
        success_threshold: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        half_open_max_calls: Optional[int] = None,
    ) -> CircuitBreakerConfig:
        """Create a new config with specified overrides.

        Args:
            failure_threshold: Override for failure_threshold
            success_threshold: Override for success_threshold
            timeout_seconds: Override for timeout_seconds
            half_open_max_calls: Override for half_open_max_calls

        Returns:
            New CircuitBreakerConfig with overrides applied
        """
        return CircuitBreakerConfig(
            failure_threshold=(
                failure_threshold if failure_threshold is not None else self.failure_threshold
            ),
            success_threshold=(
                success_threshold if success_threshold is not None else self.success_threshold
            ),
            timeout_seconds=(
                timeout_seconds if timeout_seconds is not None else self.timeout_seconds
            ),
            half_open_max_calls=(
                half_open_max_calls if half_open_max_calls is not None else self.half_open_max_calls
            ),
        )


# Default configurations per provider
# These reflect the reliability characteristics of each provider
PROVIDER_CONFIGS: dict[str, CircuitBreakerConfig] = {
    # Anthropic: Generally reliable but can hit rate limits
    "anthropic": CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=30.0,
        half_open_max_calls=2,
    ),
    # OpenAI: High reliability, more lenient thresholds
    "openai": CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout_seconds=60.0,
        half_open_max_calls=3,
    ),
    # Mistral: Moderate reliability
    "mistral": CircuitBreakerConfig(
        failure_threshold=4,
        success_threshold=2,
        timeout_seconds=45.0,
        half_open_max_calls=2,
    ),
    # OpenRouter: Aggregator with multiple backends, can be variable
    "openrouter": CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=3,
        timeout_seconds=90.0,
        half_open_max_calls=2,
    ),
    # xAI/Grok: Newer service, more conservative
    "xai": CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=60.0,
        half_open_max_calls=2,
    ),
    "grok": CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=60.0,
        half_open_max_calls=2,
    ),
    # Gemini/Google: Generally reliable
    "gemini": CircuitBreakerConfig(
        failure_threshold=4,
        success_threshold=2,
        timeout_seconds=45.0,
        half_open_max_calls=3,
    ),
    "google": CircuitBreakerConfig(
        failure_threshold=4,
        success_threshold=2,
        timeout_seconds=45.0,
        half_open_max_calls=3,
    ),
    # Default for unknown providers
    "default": CircuitBreakerConfig(),
}


def _get_env_int(name: str) -> Optional[int]:
    """Get an integer from environment variable, or None if not set/invalid."""
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _get_env_float(name: str) -> Optional[float]:
    """Get a float from environment variable, or None if not set/invalid."""
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def get_circuit_breaker_config(
    provider: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> CircuitBreakerConfig:
    """Get circuit breaker configuration for a provider or agent.

    Resolution order:
    1. Environment variable overrides (applied on top of base config)
    2. Agent-specific config (if agent_name provided and registered)
    3. Provider-specific config (if provider provided and registered)
    4. Default config

    Environment variables:
        ARAGORA_CB_FAILURE_THRESHOLD: Override failure threshold
        ARAGORA_CB_SUCCESS_THRESHOLD: Override success threshold
        ARAGORA_CB_TIMEOUT_SECONDS: Override timeout
        ARAGORA_CB_HALF_OPEN_MAX_CALLS: Override half-open max calls

    Args:
        provider: Provider name (e.g., "anthropic", "openai")
        agent_name: Specific agent name (for agent-level overrides)

    Returns:
        CircuitBreakerConfig with appropriate settings

    Example:
        # Get config for Anthropic provider
        config = get_circuit_breaker_config(provider="anthropic")

        # Get config with environment overrides
        os.environ["ARAGORA_CB_FAILURE_THRESHOLD"] = "2"
        config = get_circuit_breaker_config(provider="openai")
        # config.failure_threshold == 2 (from env)
    """
    # Start with base config
    base_config: CircuitBreakerConfig

    # Check for agent-specific config first
    if agent_name and agent_name in _AGENT_CONFIGS:
        base_config = _AGENT_CONFIGS[agent_name]
    # Then provider config
    elif provider and provider.lower() in PROVIDER_CONFIGS:
        base_config = PROVIDER_CONFIGS[provider.lower()]
    else:
        base_config = PROVIDER_CONFIGS["default"]

    # Apply environment variable overrides
    env_failure = _get_env_int("ARAGORA_CB_FAILURE_THRESHOLD")
    env_success = _get_env_int("ARAGORA_CB_SUCCESS_THRESHOLD")
    env_timeout = _get_env_float("ARAGORA_CB_TIMEOUT_SECONDS")
    env_half_open = _get_env_int("ARAGORA_CB_HALF_OPEN_MAX_CALLS")

    # If any env vars are set, create a new config with overrides
    if any(v is not None for v in [env_failure, env_success, env_timeout, env_half_open]):
        return base_config.with_overrides(
            failure_threshold=env_failure,
            success_threshold=env_success,
            timeout_seconds=env_timeout,
            half_open_max_calls=env_half_open,
        )

    return base_config


# Agent-specific configurations (can be extended at runtime)
_AGENT_CONFIGS: dict[str, CircuitBreakerConfig] = {}


def register_agent_config(agent_name: str, config: CircuitBreakerConfig) -> None:
    """Register a circuit breaker configuration for a specific agent.

    This allows fine-grained control over circuit breaker behavior
    for individual agents beyond provider-level defaults.

    Args:
        agent_name: Unique agent identifier
        config: CircuitBreakerConfig to use for this agent

    Example:
        # Make a specific agent more resilient to failures
        register_agent_config(
            "claude-sonnet",
            CircuitBreakerConfig(failure_threshold=10, timeout_seconds=120)
        )
    """
    _AGENT_CONFIGS[agent_name] = config


def unregister_agent_config(agent_name: str) -> bool:
    """Remove an agent-specific configuration.

    Args:
        agent_name: Agent identifier to remove

    Returns:
        True if the agent was registered and removed, False otherwise
    """
    if agent_name in _AGENT_CONFIGS:
        del _AGENT_CONFIGS[agent_name]
        return True
    return False


def get_registered_agent_configs() -> dict[str, CircuitBreakerConfig]:
    """Get all registered agent-specific configurations.

    Returns:
        Copy of the agent configurations dictionary
    """
    return dict(_AGENT_CONFIGS)


def clear_agent_configs() -> None:
    """Clear all agent-specific configurations. Useful for testing."""
    _AGENT_CONFIGS.clear()
