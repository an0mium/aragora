"""
Quota detection and fallback utilities for API agents.

Provides shared logic for detecting quota/rate limit errors and falling back
to OpenRouter when the primary provider is unavailable.

Also provides AgentFallbackChain for multi-provider sequencing with
CircuitBreaker integration.
"""

__all__ = [
    "QUOTA_ERROR_KEYWORDS",
    "QuotaFallbackMixin",
    "FallbackMetrics",
    "AllProvidersExhaustedError",
    "FallbackTimeoutError",
    "AgentFallbackChain",
    "get_local_fallback_providers",
    "build_fallback_chain_with_local",
    "is_local_llm_available",
    "get_default_fallback_enabled",
]

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

try:
    from aragora.agents.registry import AgentRegistry
except (ImportError, ModuleNotFoundError):
    AgentRegistry = None  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from aragora.resilience import CircuitBreaker

    from .api_agents import OpenRouterAgent

logger = logging.getLogger(__name__)

# Common keywords indicating quota/rate limit errors across providers
QUOTA_ERROR_KEYWORDS = frozenset(
    [
        # Rate limiting
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        # Quota exceeded
        "quota",
        "exceeded",
        "limit exceeded",
        "resource exhausted",
        "resource_exhausted",
        # Billing/credits
        "billing",
        "credit balance",
        "insufficient",
        "insufficient_quota",
        "purchase credits",
    ]
)


class QuotaFallbackMixin:
    """Mixin providing shared quota detection and OpenRouter fallback logic.

    This mixin extracts the common quota error detection and fallback pattern
    used by Gemini, Anthropic, OpenAI, and Grok agents.

    The mixin expects the following attributes on the class:
        - name: str - Agent name for logging
        - enable_fallback: bool - Whether fallback is enabled
        - model: str - Current model name
        - timeout: int - Timeout setting
        - role: str - Agent role (optional, defaults to "proposer")
        - system_prompt: str - System prompt (optional)

    Class attributes that can be overridden:
        - OPENROUTER_MODEL_MAP: dict[str, str] - Maps provider models to OpenRouter models
        - DEFAULT_FALLBACK_MODEL: str - Default model if no mapping found

    Usage:
        class MyAgent(APIAgent, QuotaFallbackMixin):
            OPENROUTER_MODEL_MAP = {
                "gpt-4o": "openai/gpt-4o",
                "gpt-4": "openai/gpt-4",
            }
            DEFAULT_FALLBACK_MODEL = "openai/gpt-4o"

            async def generate(self, prompt, context):
                # ... make API call ...
                if self.is_quota_error(status, error_text):
                    result = await self.fallback_generate(prompt, context)
                    if result is not None:
                        return result
                    raise RuntimeError("Quota exceeded and fallback unavailable")
    """

    # Override these in subclasses for provider-specific model mappings
    OPENROUTER_MODEL_MAP: dict[str, str] = {}
    DEFAULT_FALLBACK_MODEL: str = "anthropic/claude-sonnet-4"

    # Instance-level cached fallback agent (set by _get_cached_fallback_agent)
    _fallback_agent: Optional["OpenRouterAgent"] = None

    def _get_cached_fallback_agent(self) -> Optional["OpenRouterAgent"]:
        """Get or create a cached OpenRouter fallback agent.

        Unlike _get_openrouter_fallback(), this caches the agent for reuse.
        """
        if self._fallback_agent is None:
            self._fallback_agent = self._get_openrouter_fallback()
            if self._fallback_agent:
                name = getattr(self, "name", "unknown")
                logger.info(
                    f"[{name}] Created OpenRouter fallback agent with model {self._fallback_agent.model}"
                )
        return self._fallback_agent

    def get_fallback_model(self) -> str:
        """Get the OpenRouter model for fallback based on current model.

        Uses the class's OPENROUTER_MODEL_MAP to find a matching model,
        falling back to DEFAULT_FALLBACK_MODEL if no match is found.
        """
        model = getattr(self, "model", "")
        return self.OPENROUTER_MODEL_MAP.get(model, self.DEFAULT_FALLBACK_MODEL)

    def is_quota_error(self, status_code: int, error_text: str) -> bool:
        """Check if an error indicates quota/rate limit issues or timeouts.

        This is a unified check that works across providers:
        - 429: Rate limit (all providers)
        - 403: Can indicate quota exceeded (Gemini)
        - 408, 504, 524: Timeout errors (should trigger fallback)

        Args:
            status_code: HTTP status code from response
            error_text: Error message text from response body

        Returns:
            True if this appears to be a quota/rate limit/timeout error
        """
        # 429 is universally rate limit
        if status_code == 429:
            return True

        # Timeout status codes - treat as quota to trigger fallback
        # 408: Request Timeout, 504: Gateway Timeout, 524: Cloudflare timeout
        if status_code in (408, 504, 524):
            return True

        # 403 can indicate quota exceeded (especially for Gemini)
        if status_code == 403:
            # Only treat as quota if error text contains quota keywords
            error_lower = error_text.lower()
            if any(kw in error_lower for kw in ["quota", "exceeded", "billing"]):
                return True

        # Check for timeout keywords in error text
        error_lower = error_text.lower()
        if "timeout" in error_lower or "timed out" in error_lower:
            return True

        # Check for quota-related keywords in any error
        return any(kw in error_lower for kw in QUOTA_ERROR_KEYWORDS)

    def _get_openrouter_fallback(self) -> Optional["OpenRouterAgent"]:
        """Get an OpenRouter fallback agent if available.

        Returns:
            OpenRouterAgent instance if OPENROUTER_API_KEY is set, None otherwise
        """
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            return None

        # Lazy import to avoid circular dependency
        from .api_agents import OpenRouterAgent

        # Use the class's model mapping to get the fallback model
        fallback_model = self.get_fallback_model()

        # Get agent attributes with sensible defaults
        name = getattr(self, "name", "fallback")
        role = getattr(self, "role", "proposer")
        timeout = getattr(self, "timeout", 120)
        system_prompt = getattr(self, "system_prompt", None)

        agent = OpenRouterAgent(
            name=f"{name}_fallback",
            model=fallback_model,
            role=role,
            timeout=timeout,
        )
        if system_prompt:
            agent.system_prompt = system_prompt

        return agent

    async def fallback_generate(
        self,
        prompt: str,
        context: Optional[list] = None,
        status_code: Optional[int] = None,
    ) -> Optional[str]:
        """Attempt to generate using OpenRouter fallback.

        Args:
            prompt: The prompt to send
            context: Optional conversation context
            status_code: Optional HTTP status code that triggered the fallback

        Returns:
            Generated response string if fallback succeeded, None otherwise
        """
        if not getattr(self, "enable_fallback", True):
            return None

        fallback = self._get_cached_fallback_agent()
        if not fallback:
            name = getattr(self, "name", "unknown")
            logger.warning(
                f"{name} quota exceeded but OPENROUTER_API_KEY not set - cannot fallback"
            )
            return None

        name = getattr(self, "name", "unknown")
        status_info = f" (status {status_code})" if status_code else ""
        logger.warning(
            f"API quota/rate limit error{status_info} for {name}, falling back to OpenRouter"
        )
        return await fallback.generate(prompt, context)

    async def fallback_generate_stream(
        self,
        prompt: str,
        context: Optional[list] = None,
        status_code: Optional[int] = None,
    ):
        """Attempt to stream using OpenRouter fallback.

        Args:
            prompt: The prompt to send
            context: Optional conversation context
            status_code: Optional HTTP status code that triggered the fallback

        Yields:
            Content tokens from fallback stream, or nothing if fallback unavailable
        """
        if not getattr(self, "enable_fallback", True):
            return

        fallback = self._get_cached_fallback_agent()
        if not fallback:
            name = getattr(self, "name", "unknown")
            logger.warning(
                f"{name} quota exceeded but OPENROUTER_API_KEY not set - cannot fallback"
            )
            return

        name = getattr(self, "name", "unknown")
        status_info = f" (status {status_code})" if status_code else ""
        logger.warning(
            f"API quota/rate limit error{status_info} for {name}, falling back to OpenRouter streaming"
        )
        async for token in fallback.generate_stream(prompt, context):
            yield token


@dataclass
class FallbackMetrics:
    """Metrics for tracking fallback chain behavior."""

    primary_attempts: int = 0
    primary_successes: int = 0
    fallback_attempts: int = 0
    fallback_successes: int = 0
    total_failures: int = 0
    last_fallback_time: float = 0.0
    fallback_providers_used: dict[str, int] = field(default_factory=dict)

    def record_primary_attempt(self, success: bool) -> None:
        """Record a primary provider attempt."""
        self.primary_attempts += 1
        if success:
            self.primary_successes += 1
        else:
            self.total_failures += 1

    def record_fallback_attempt(self, provider: str, success: bool) -> None:
        """Record a fallback provider attempt."""
        self.fallback_attempts += 1
        self.fallback_providers_used[provider] = self.fallback_providers_used.get(provider, 0) + 1
        self.last_fallback_time = time.time()
        if success:
            self.fallback_successes += 1
        else:
            self.total_failures += 1

    @property
    def fallback_rate(self) -> float:
        """Percentage of requests that needed fallback."""
        total = self.primary_attempts + self.fallback_attempts
        if total == 0:
            return 0.0
        return self.fallback_attempts / total

    @property
    def success_rate(self) -> float:
        """Overall success rate including fallbacks."""
        total = self.primary_successes + self.fallback_successes
        attempts = self.primary_attempts + self.fallback_attempts
        if attempts == 0:
            return 0.0
        return total / attempts


class AllProvidersExhaustedError(RuntimeError):
    """Raised when all providers in a fallback chain have failed."""

    def __init__(self, providers: list[str], last_error: Optional[Exception] = None):
        self.providers = providers
        self.last_error = last_error
        super().__init__(
            f"All providers exhausted: {', '.join(providers)}. Last error: {last_error}"
        )


class FallbackTimeoutError(Exception):
    """Raised when fallback chain exceeds time limit."""

    def __init__(self, elapsed: float, limit: float, tried: list[str]):
        self.elapsed = elapsed
        self.limit = limit
        self.tried_providers = tried
        super().__init__(
            f"Fallback chain timeout after {elapsed:.1f}s (limit {limit}s). "
            f"Tried: {', '.join(tried)}"
        )


class AgentFallbackChain:
    """Sequences agent providers with automatic fallback and CircuitBreaker integration.

    This class manages a chain of providers (e.g., OpenAI -> OpenRouter -> Anthropic -> CLI)
    and automatically falls back to the next provider when one fails. It integrates with
    CircuitBreaker to track provider health and avoid repeatedly calling failing providers.

    Usage:
        from aragora.resilience import get_circuit_breaker

        chain = AgentFallbackChain(
            providers=["openai", "openrouter", "anthropic"],
            circuit_breaker=get_circuit_breaker("fallback_chain", failure_threshold=3, cooldown_seconds=60),
            max_retries=3,  # Only try 3 providers before giving up
            max_fallback_time=30.0,  # Give up after 30 seconds total
        )

        # Register provider factories
        chain.register_provider("openai", lambda: OpenAIAPIAgent())
        chain.register_provider("openrouter", lambda: OpenRouterAgent())
        chain.register_provider("anthropic", lambda: AnthropicAPIAgent())

        # Generate with automatic fallback
        result = await chain.generate(prompt, context)

        # Check metrics
        print(f"Fallback rate: {chain.metrics.fallback_rate:.1%}")
    """

    # Default limits
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_MAX_FALLBACK_TIME = 120.0  # 2 minutes

    def __init__(
        self,
        providers: list[Any],
        circuit_breaker: Optional["CircuitBreaker"] = None,
        max_retries: Optional[int] = None,
        max_fallback_time: Optional[float] = None,
    ):
        """Initialize the fallback chain.

        Args:
            providers: Ordered list of provider names or agent instances (first is primary)
            circuit_breaker: CircuitBreaker instance for tracking provider health
            max_retries: Maximum number of providers to try (default: 5)
            max_fallback_time: Maximum time in seconds for the entire fallback chain (default: 120)
        """
        self.providers = providers
        self.circuit_breaker = circuit_breaker
        self.max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        self.max_fallback_time = (
            max_fallback_time if max_fallback_time is not None else self.DEFAULT_MAX_FALLBACK_TIME
        )
        self.metrics = FallbackMetrics()
        self._provider_factories: dict[str, Callable[[], Any]] = {}
        self._cached_agents: dict[str, Any] = {}

    def _provider_key(self, provider: Any) -> str:
        """Return the name used for metrics/circuit breaker tracking."""
        if isinstance(provider, str):
            return provider
        return getattr(provider, "name", str(provider))

    def register_provider(
        self,
        name: str,
        factory: Callable[[], Any],
    ) -> None:
        """Register a factory function for creating a provider agent.

        Args:
            name: Provider name (must be in self.providers)
            factory: Callable that returns an agent instance
        """
        if name not in self.providers:
            logger.warning(f"Registering provider '{name}' not in chain: {self.providers}")
        self._provider_factories[name] = factory

    def _get_agent(self, provider: Any) -> Optional[Any]:
        """Get or create an agent for the given provider."""
        if not isinstance(provider, str):
            return provider

        if provider in self._cached_agents:
            return self._cached_agents[provider]

        factory = self._provider_factories.get(provider)
        if not factory:
            logger.debug(f"No factory registered for provider '{provider}'")
            return None

        try:
            agent = factory()
            self._cached_agents[provider] = agent
            return agent
        except Exception as e:
            logger.warning(f"Failed to create agent for provider '{provider}': {e}")
            return None

    def _is_available(self, provider: str) -> bool:
        """Check if a provider is available (not tripped in circuit breaker)."""
        if not self.circuit_breaker:
            return True
        return self.circuit_breaker.is_available(provider)

    def _record_success(self, provider: str) -> None:
        """Record a successful call to a provider."""
        if self.circuit_breaker:
            self.circuit_breaker.record_success(provider)

    def _record_failure(self, provider: str) -> None:
        """Record a failed call to a provider."""
        if self.circuit_breaker:
            self.circuit_breaker.record_failure(provider)

    def get_available_providers(self) -> list[str]:
        """Get list of providers currently available (not circuit-broken)."""
        available: list[str] = []
        for provider in self.providers:
            provider_key = self._provider_key(provider)
            if self._is_available(provider_key):
                available.append(provider_key)
        return available

    async def generate(
        self,
        prompt: str,
        context: Optional[list] = None,
    ) -> str:
        """Generate a response using the fallback chain.

        Tries each provider in order, skipping those that are circuit-broken,
        until one succeeds or all fail. Respects max_retries and max_fallback_time limits.

        Args:
            prompt: The prompt to send
            context: Optional conversation context

        Returns:
            Generated response string

        Raises:
            AllProvidersExhaustedError: If all providers fail
            FallbackTimeoutError: If max_fallback_time is exceeded
        """
        last_error: Optional[Exception] = None
        tried_providers: list[str] = []
        start_time = time.time()
        retry_count = 0

        for i, provider in enumerate(self.providers):
            provider_key = self._provider_key(provider)
            # Check retry limit
            if retry_count >= self.max_retries:
                logger.warning(f"Max retries ({self.max_retries}) reached, stopping fallback chain")
                break

            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > self.max_fallback_time:
                raise FallbackTimeoutError(elapsed, self.max_fallback_time, tried_providers)

            # Skip if circuit breaker has this provider tripped
            if not self._is_available(provider_key):
                logger.debug(f"Skipping circuit-broken provider: {provider_key}")
                continue

            agent = self._get_agent(provider)
            if not agent:
                continue

            tried_providers.append(provider_key)
            retry_count += 1
            is_primary = i == 0

            try:
                result = await agent.generate(prompt, context)

                # Record success
                self._record_success(provider_key)
                if is_primary:
                    self.metrics.record_primary_attempt(success=True)
                else:
                    self.metrics.record_fallback_attempt(provider_key, success=True)
                    logger.info(
                        f"fallback_success provider={provider_key} "
                        f"fallback_rate={self.metrics.fallback_rate:.1%}"
                    )

                return result

            except Exception as e:
                last_error = e
                self._record_failure(provider_key)

                if is_primary:
                    self.metrics.record_primary_attempt(success=False)
                    logger.warning(
                        f"Primary provider '{provider_key}' failed: {e}, trying fallback"
                    )
                else:
                    self.metrics.record_fallback_attempt(provider_key, success=False)
                    logger.warning(f"Fallback provider '{provider_key}' failed: {e}")

                # Check if this looks like a rate limit error
                if self._is_rate_limit_error(e):
                    logger.info(f"Rate limit detected for {provider_key}, moving to next")

                continue

        raise AllProvidersExhaustedError(tried_providers, last_error)

    async def generate_stream(
        self,
        prompt: str,
        context: Optional[list] = None,
    ):
        """Stream a response using the fallback chain.

        Tries each provider in order until one succeeds or all fail.
        Respects max_retries and max_fallback_time limits.

        Args:
            prompt: The prompt to send
            context: Optional conversation context

        Yields:
            Content tokens from the successful provider

        Raises:
            AllProvidersExhaustedError: If all providers fail
            FallbackTimeoutError: If max_fallback_time is exceeded
        """
        last_error: Optional[Exception] = None
        tried_providers: list[str] = []
        start_time = time.time()
        retry_count = 0

        for i, provider in enumerate(self.providers):
            provider_key = self._provider_key(provider)
            # Check retry limit
            if retry_count >= self.max_retries:
                logger.warning(f"Max retries ({self.max_retries}) reached for stream, stopping")
                break

            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > self.max_fallback_time:
                raise FallbackTimeoutError(elapsed, self.max_fallback_time, tried_providers)

            if not self._is_available(provider_key):
                logger.debug(f"Skipping circuit-broken provider: {provider_key}")
                continue

            agent = self._get_agent(provider)
            if not agent:
                continue

            # Check if agent supports streaming
            if not hasattr(agent, "generate_stream"):
                logger.debug(f"Provider {provider_key} doesn't support streaming, skipping")
                continue

            tried_providers.append(provider_key)
            retry_count += 1
            is_primary = i == 0

            try:
                # Try to get first token to verify stream works
                first_token = None
                async for token in agent.generate_stream(prompt, context):
                    if first_token is None:
                        first_token = token
                        # Stream started successfully
                        self._record_success(provider_key)
                        if is_primary:
                            self.metrics.record_primary_attempt(success=True)
                        else:
                            self.metrics.record_fallback_attempt(provider_key, success=True)
                            logger.info(f"fallback_stream_success provider={provider_key}")
                    yield token

                # If we got here, stream completed successfully
                return

            except Exception as e:
                last_error = e
                self._record_failure(provider_key)

                if is_primary:
                    self.metrics.record_primary_attempt(success=False)
                    logger.warning(f"Primary provider '{provider_key}' stream failed: {e}")
                else:
                    self.metrics.record_fallback_attempt(provider_key, success=False)
                    logger.warning(f"Fallback provider '{provider_key}' stream failed: {e}")

                continue

        raise AllProvidersExhaustedError(tried_providers, last_error)

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception indicates a rate limit error."""
        error_str = str(error).lower()
        return any(kw in error_str for kw in QUOTA_ERROR_KEYWORDS)

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self.metrics = FallbackMetrics()

    def get_status(self) -> dict:
        """Get current status of the fallback chain."""
        return {
            "providers": [self._provider_key(provider) for provider in self.providers],
            "available_providers": self.get_available_providers(),
            "limits": {
                "max_retries": self.max_retries,
                "max_fallback_time": self.max_fallback_time,
            },
            "metrics": {
                "primary_attempts": self.metrics.primary_attempts,
                "primary_successes": self.metrics.primary_successes,
                "fallback_attempts": self.metrics.fallback_attempts,
                "fallback_successes": self.metrics.fallback_successes,
                "fallback_rate": f"{self.metrics.fallback_rate:.1%}",
                "success_rate": f"{self.metrics.success_rate:.1%}",
                "providers_used": self.metrics.fallback_providers_used,
            },
        }


def get_local_fallback_providers() -> list[str]:
    """Get list of available local LLM providers for fallback.

    Checks for running Ollama or LM Studio instances and returns
    their provider names if available.

    Returns:
        List of provider names (e.g., ["ollama", "lm-studio"])
    """
    if AgentRegistry is None:
        return []
    try:
        local_agents = AgentRegistry.detect_local_agents()
        return [agent["name"] for agent in local_agents if agent.get("available", False)]
    except Exception as e:
        logger.debug(f"Could not detect local LLMs: {e}")
        return []


def build_fallback_chain_with_local(
    primary_providers: list[str],
    include_local: bool = True,
    local_priority: bool = False,
) -> list[str]:
    """Build a fallback chain that includes local LLMs.

    Args:
        primary_providers: Primary cloud providers to use
        include_local: Whether to include local LLMs in the chain
        local_priority: If True, local LLMs come before OpenRouter

    Returns:
        Ordered list of providers for fallback chain

    Example:
        # Default: OpenAI -> OpenRouter -> Local -> Anthropic
        chain = build_fallback_chain_with_local(
            ["openai", "openrouter", "anthropic"],
            include_local=True,
        )

        # Priority: OpenAI -> Local -> OpenRouter -> Anthropic
        chain = build_fallback_chain_with_local(
            ["openai", "openrouter", "anthropic"],
            include_local=True,
            local_priority=True,
        )
    """
    if not include_local:
        return primary_providers

    local_providers = get_local_fallback_providers()
    if not local_providers:
        return primary_providers

    result = []
    openrouter_idx = -1

    for i, provider in enumerate(primary_providers):
        if provider == "openrouter":
            openrouter_idx = i
            if local_priority:
                # Insert local before OpenRouter
                result.extend(local_providers)
            result.append(provider)
            if not local_priority:
                # Insert local after OpenRouter
                result.extend(local_providers)
        else:
            result.append(provider)

    # If no OpenRouter in chain, append local at the end
    if openrouter_idx == -1:
        result.extend(local_providers)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_result: list[str] = []
    for p in result:
        if p not in seen:
            seen.add(p)
            unique_result.append(p)
    return unique_result


def is_local_llm_available() -> bool:
    """Check if any local LLM server is available.

    Returns:
        True if Ollama, LM Studio, or compatible server is running
    """
    if AgentRegistry is None:
        logger.debug("AgentRegistry not available for local LLM check")
        return False
    try:
        status = AgentRegistry.get_local_status()
        return status.get("any_available", False)
    except Exception as e:
        logger.warning(f"Failed to check local LLM availability: {e}")
        return False


def get_default_fallback_enabled() -> bool:
    """Get the default value for enable_fallback from config.

    Returns False by default (opt-in) to prevent silent billing and
    unexpected model behavior when OpenRouter fallback activates.

    Set ARAGORA_OPENROUTER_FALLBACK_ENABLED=true to enable.

    Returns:
        True if fallback is enabled in settings, False otherwise
    """
    try:
        from aragora.config.settings import get_settings

        settings = get_settings()
        return settings.agent.openrouter_fallback_enabled
    except Exception:
        # If settings can't be loaded, default to False (opt-in)
        return False
