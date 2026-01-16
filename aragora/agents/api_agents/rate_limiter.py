"""
API provider rate limiting infrastructure.

Provides token bucket rate limiting for API calls with:
- Per-provider rate limiters (Anthropic, OpenAI, OpenRouter, etc.)
- Configurable tiers with different RPM limits
- Thread-safe operation with per-provider locks (no global lock contention)
- Exponential backoff for rate limit recovery
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from aragora.shared.rate_limiting import ExponentialBackoff

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterTier:
    """Rate limit configuration for an OpenRouter pricing tier."""

    name: str
    requests_per_minute: int
    tokens_per_minute: int = 0  # 0 = unlimited
    burst_size: int = 10  # Allow short bursts


# OpenRouter tier configurations (based on their pricing)
OPENROUTER_TIERS = {
    "free": OpenRouterTier(name="free", requests_per_minute=20, burst_size=5),
    "basic": OpenRouterTier(name="basic", requests_per_minute=60, burst_size=15),
    "standard": OpenRouterTier(name="standard", requests_per_minute=200, burst_size=30),
    "premium": OpenRouterTier(name="premium", requests_per_minute=500, burst_size=50),
    "unlimited": OpenRouterTier(name="unlimited", requests_per_minute=1000, burst_size=100),
}


@dataclass
class ProviderTier:
    """Rate limit configuration for an API provider."""

    name: str
    requests_per_minute: int
    tokens_per_minute: int = 0  # 0 = unlimited
    burst_size: int = 10  # Allow short bursts


# Provider-specific default tiers (based on typical API limits)
# These can be overridden via environment variables
PROVIDER_DEFAULT_TIERS: Dict[str, ProviderTier] = {
    # Anthropic: 1000 RPM for paid, 60 for free tier
    "anthropic": ProviderTier(name="anthropic", requests_per_minute=1000, burst_size=50),
    # OpenAI: Varies by tier, using reasonable default
    "openai": ProviderTier(name="openai", requests_per_minute=500, burst_size=30),
    # Mistral: 5 requests/sec = 300 RPM
    "mistral": ProviderTier(name="mistral", requests_per_minute=300, burst_size=20),
    # Gemini: 60 RPM free, 1000 RPM paid
    "gemini": ProviderTier(name="gemini", requests_per_minute=60, burst_size=15),
    # Grok (xAI): Similar to Anthropic
    "grok": ProviderTier(name="grok", requests_per_minute=500, burst_size=30),
    # OpenRouter: Use standard tier as default
    "openrouter": ProviderTier(name="openrouter", requests_per_minute=200, burst_size=30),
    # Ollama (local): Higher limits since it's local
    "ollama": ProviderTier(name="ollama", requests_per_minute=1000, burst_size=100),
    # LM Studio (local): Higher limits since it's local
    "lm_studio": ProviderTier(name="lm_studio", requests_per_minute=1000, burst_size=100),
}


class OpenRouterRateLimiter:
    """Rate limiter for OpenRouter API calls.

    Uses token bucket algorithm with configurable tiers.
    Thread-safe for use across multiple agent instances.
    """

    def __init__(self, tier: str = "standard"):
        """
        Initialize rate limiter with specified tier.

        Tier can be set via OPENROUTER_TIER environment variable.
        """
        tier_name = os.environ.get("OPENROUTER_TIER", tier).lower()
        self.tier = OPENROUTER_TIERS.get(tier_name, OPENROUTER_TIERS["standard"])

        self._tokens = float(self.tier.burst_size)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

        # Track rate limit headers from API
        self._api_limit: Optional[int] = None
        self._api_remaining: Optional[int] = None
        self._api_reset: Optional[float] = None

        # Exponential backoff for quota exhaustion recovery
        self._backoff = ExponentialBackoff(base_delay=2.0, max_delay=60.0, jitter=0.15)

        logger.debug(
            f"OpenRouter rate limiter initialized: tier={self.tier.name}, rpm={self.tier.requests_per_minute}"
        )

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed_minutes = (now - self._last_refill) / 60.0
        refill_amount = elapsed_minutes * self.tier.requests_per_minute
        self._tokens = min(self.tier.burst_size, self._tokens + refill_amount)
        self._last_refill = now

    async def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire permission to make an API request.

        Blocks until a token is available or timeout is reached.
        Uses exponential backoff when recovering from rate limit errors.
        Returns True if acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout

        # If in backoff state, wait before trying
        if self._backoff.is_backing_off:
            backoff_delay = self._backoff.get_delay()
            remaining = deadline - time.monotonic()
            if backoff_delay > remaining:
                logger.warning(
                    f"Backoff delay {backoff_delay:.1f}s exceeds timeout {remaining:.1f}s"
                )
                return False
            logger.info(f"rate_limiter_backoff_wait delay={backoff_delay:.1f}s")
            await asyncio.sleep(backoff_delay)

        while True:
            with self._lock:
                self._refill()

                # Check API-reported limits if available
                if self._api_remaining is not None and self._api_remaining <= 0:
                    wait_time = (self._api_reset or 60) - time.time()
                    if wait_time > 0 and wait_time < timeout:
                        logger.debug(f"OpenRouter API limit reached, waiting {wait_time:.1f}s")
                        await asyncio.sleep(min(wait_time, 1.0))
                        continue

                if self._tokens >= 1:
                    self._tokens -= 1
                    # Add inter-request delay to prevent burst requests
                    from aragora.config import OPENROUTER_INTER_REQUEST_DELAY

                    if OPENROUTER_INTER_REQUEST_DELAY > 0:
                        await asyncio.sleep(OPENROUTER_INTER_REQUEST_DELAY)
                    return True

            # Wait and retry
            if time.monotonic() >= deadline:
                logger.warning("OpenRouter rate limit timeout")
                return False

            # Use backoff delay if in backoff state, otherwise use token refill time
            if self._backoff.is_backing_off:
                wait_time = min(self._backoff.get_delay(), deadline - time.monotonic())
            else:
                wait_time = 60.0 / self.tier.requests_per_minute  # Time for 1 token
            await asyncio.sleep(min(wait_time, 1.0))

    def update_from_headers(self, headers: dict) -> None:
        """Update rate limit state from API response headers.

        OpenRouter returns standard rate limit headers:
        - X-RateLimit-Limit: Total requests allowed
        - X-RateLimit-Remaining: Requests remaining
        - X-RateLimit-Reset: Unix timestamp when limit resets
        """
        with self._lock:
            if "X-RateLimit-Limit" in headers:
                try:
                    self._api_limit = int(headers["X-RateLimit-Limit"])
                except ValueError as e:
                    logger.warning(
                        f"Failed to parse X-RateLimit-Limit header: {headers.get('X-RateLimit-Limit')!r} - {e}"
                    )

            if "X-RateLimit-Remaining" in headers:
                try:
                    self._api_remaining = int(headers["X-RateLimit-Remaining"])
                except ValueError as e:
                    logger.warning(
                        f"Failed to parse X-RateLimit-Remaining header: {headers.get('X-RateLimit-Remaining')!r} - {e}"
                    )

            if "X-RateLimit-Reset" in headers:
                try:
                    self._api_reset = float(headers["X-RateLimit-Reset"])
                except ValueError as e:
                    logger.warning(
                        f"Failed to parse X-RateLimit-Reset header: {headers.get('X-RateLimit-Reset')!r} - {e}"
                    )

    def release_on_error(self) -> None:
        """Release a token back on request error (optional, for retries)."""
        with self._lock:
            self._tokens = min(self.tier.burst_size, self._tokens + 1.0)

    def record_rate_limit_error(self, status_code: int = 429) -> float:
        """Record a rate limit error (429/403) and return backoff delay.

        Call this when the API returns a rate limit error. The limiter will
        enter backoff state and subsequent acquire() calls will wait accordingly.

        Args:
            status_code: HTTP status code (429=rate limited, 403=quota exceeded)

        Returns:
            The recommended delay before retrying (in seconds)
        """
        logger.warning(f"rate_limit_error status={status_code}")
        delay = self._backoff.record_failure()
        # Also release the token back since request failed
        self.release_on_error()
        return delay

    def record_success(self) -> None:
        """Record a successful API request.

        Call this after a request succeeds to reset backoff state.
        This allows normal rate limiting to resume after recovery.
        """
        self._backoff.reset()

    @property
    def is_backing_off(self) -> bool:
        """Check if currently in exponential backoff due to rate limit errors."""
        return self._backoff.is_backing_off

    @property
    def stats(self) -> dict:
        """Get current rate limiter statistics."""
        with self._lock:
            return {
                "tier": self.tier.name,
                "rpm_limit": self.tier.requests_per_minute,
                "tokens_available": int(self._tokens),
                "burst_size": self.tier.burst_size,
                "api_limit": self._api_limit,
                "api_remaining": self._api_remaining,
                "backoff_failures": self._backoff.failure_count,
                "is_backing_off": self._backoff.is_backing_off,
            }

    def request(self, timeout: float = 30.0) -> "RateLimitContext":
        """Context manager for rate-limited API requests.

        Provides cleaner async with syntax for acquiring and optionally
        releasing rate limit tokens.

        Usage:
            async with limiter.request() as acquired:
                if acquired:
                    response = await make_api_call()
                else:
                    raise TimeoutError("Rate limit timeout")

            # Or with auto-release on error:
            async with limiter.request() as ctx:
                if ctx:
                    try:
                        response = await make_api_call()
                    except Exception:
                        ctx.release_on_error()
                        raise
        """
        return RateLimitContext(self, timeout)


class RateLimitContext:
    """Async context manager for rate limit acquisition.

    Acquires a rate limit token on entry and optionally releases on error.
    """

    def __init__(self, limiter: OpenRouterRateLimiter, timeout: float):
        self._limiter = limiter
        self._timeout = timeout
        self._acquired = False

    async def __aenter__(self) -> "RateLimitContext":
        """Acquire rate limit on context entry."""
        self._acquired = await self._limiter.acquire(self._timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context (no automatic release)."""
        pass

    def __bool__(self) -> bool:
        """Check if rate limit was acquired."""
        return self._acquired

    def release_on_error(self) -> None:
        """Release the token back on request error."""
        if self._acquired:
            self._limiter.release_on_error()


class ProviderRateLimiter:
    """Generic rate limiter for any API provider.

    Uses token bucket algorithm with provider-specific configurations.
    Thread-safe with per-instance locks (no global lock contention).
    """

    def __init__(self, provider: str, rpm: Optional[int] = None, burst: Optional[int] = None):
        """
        Initialize rate limiter for a specific provider.

        Args:
            provider: Provider name (e.g., 'anthropic', 'openai', 'gemini')
            rpm: Override requests per minute (uses provider default if None)
            burst: Override burst size (uses provider default if None)

        Rate limits can be overridden via environment variables:
            ARAGORA_{PROVIDER}_RPM - Requests per minute
            ARAGORA_{PROVIDER}_BURST - Burst size
        """
        self.provider = provider.lower()

        # Get default tier for provider
        default_tier = PROVIDER_DEFAULT_TIERS.get(
            self.provider, ProviderTier(name=self.provider, requests_per_minute=100, burst_size=10)
        )

        # Allow environment variable overrides
        env_prefix = f"ARAGORA_{self.provider.upper()}"
        self.requests_per_minute = (
            rpm or int(os.environ.get(f"{env_prefix}_RPM", 0)) or default_tier.requests_per_minute
        )
        self.burst_size = (
            burst or int(os.environ.get(f"{env_prefix}_BURST", 0)) or default_tier.burst_size
        )

        self._tokens = float(self.burst_size)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()  # Per-instance lock, no global contention

        # Track rate limit headers from API
        self._api_limit: Optional[int] = None
        self._api_remaining: Optional[int] = None
        self._api_reset: Optional[float] = None

        # Exponential backoff for quota exhaustion recovery
        self._backoff = ExponentialBackoff(base_delay=2.0, max_delay=60.0, jitter=0.15)

        logger.debug(
            f"Provider rate limiter initialized: provider={self.provider}, "
            f"rpm={self.requests_per_minute}, burst={self.burst_size}"
        )

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed_minutes = (now - self._last_refill) / 60.0
        refill_amount = elapsed_minutes * self.requests_per_minute
        self._tokens = min(self.burst_size, self._tokens + refill_amount)
        self._last_refill = now

    async def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire permission to make an API request.

        Blocks until a token is available or timeout is reached.
        Uses exponential backoff when recovering from rate limit errors.
        Returns True if acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout

        # If in backoff state, wait before trying
        if self._backoff.is_backing_off:
            backoff_delay = self._backoff.get_delay()
            remaining = deadline - time.monotonic()
            if backoff_delay > remaining:
                logger.warning(
                    f"[{self.provider}] Backoff delay {backoff_delay:.1f}s "
                    f"exceeds timeout {remaining:.1f}s"
                )
                return False
            logger.info(f"[{self.provider}] rate_limiter_backoff_wait delay={backoff_delay:.1f}s")
            await asyncio.sleep(backoff_delay)

        while True:
            with self._lock:
                self._refill()

                # Check API-reported limits if available
                if self._api_remaining is not None and self._api_remaining <= 0:
                    wait_time = (self._api_reset or 60) - time.time()
                    if wait_time > 0 and wait_time < timeout:
                        logger.debug(
                            f"[{self.provider}] API limit reached, waiting {wait_time:.1f}s"
                        )
                        await asyncio.sleep(min(wait_time, 1.0))
                        continue

                if self._tokens >= 1:
                    self._tokens -= 1
                    # Add inter-request delay to prevent burst requests
                    from aragora.config import INTER_REQUEST_DELAY_SECONDS

                    if INTER_REQUEST_DELAY_SECONDS > 0:
                        await asyncio.sleep(INTER_REQUEST_DELAY_SECONDS)
                    return True

            # Wait and retry
            if time.monotonic() >= deadline:
                logger.warning(f"[{self.provider}] rate limit timeout")
                return False

            # Use backoff delay if in backoff state, otherwise use token refill time
            if self._backoff.is_backing_off:
                wait_time = min(self._backoff.get_delay(), deadline - time.monotonic())
            else:
                wait_time = 60.0 / self.requests_per_minute  # Time for 1 token
            await asyncio.sleep(min(wait_time, 1.0))

    def update_from_headers(self, headers: dict) -> None:
        """Update rate limit state from API response headers."""
        with self._lock:
            # Try common header formats
            for limit_header in ["X-RateLimit-Limit", "x-ratelimit-limit", "RateLimit-Limit"]:
                if limit_header in headers:
                    try:
                        self._api_limit = int(headers[limit_header])
                        break
                    except ValueError:
                        pass

            for remaining_header in [
                "X-RateLimit-Remaining",
                "x-ratelimit-remaining",
                "RateLimit-Remaining",
            ]:
                if remaining_header in headers:
                    try:
                        self._api_remaining = int(headers[remaining_header])
                        break
                    except ValueError:
                        pass

            for reset_header in ["X-RateLimit-Reset", "x-ratelimit-reset", "RateLimit-Reset"]:
                if reset_header in headers:
                    try:
                        self._api_reset = float(headers[reset_header])
                        break
                    except ValueError:
                        pass

    def release_on_error(self) -> None:
        """Release a token back on request error (optional, for retries)."""
        with self._lock:
            self._tokens = min(self.burst_size, self._tokens + 1.0)

    def record_rate_limit_error(self, status_code: int = 429) -> float:
        """Record a rate limit error (429/403) and return backoff delay."""
        logger.warning(f"[{self.provider}] rate_limit_error status={status_code}")
        delay = self._backoff.record_failure()
        self.release_on_error()
        return delay

    def record_success(self) -> None:
        """Record a successful API request to reset backoff state."""
        self._backoff.reset()

    @property
    def is_backing_off(self) -> bool:
        """Check if currently in exponential backoff due to rate limit errors."""
        return self._backoff.is_backing_off

    @property
    def stats(self) -> dict:
        """Get current rate limiter statistics."""
        with self._lock:
            return {
                "provider": self.provider,
                "rpm_limit": self.requests_per_minute,
                "tokens_available": int(self._tokens),
                "burst_size": self.burst_size,
                "api_limit": self._api_limit,
                "api_remaining": self._api_remaining,
                "backoff_failures": self._backoff.failure_count,
                "is_backing_off": self._backoff.is_backing_off,
            }

    def request(self, timeout: float = 30.0) -> "ProviderRateLimitContext":
        """Context manager for rate-limited API requests."""
        return ProviderRateLimitContext(self, timeout)


class ProviderRateLimitContext:
    """Async context manager for provider rate limit acquisition."""

    def __init__(self, limiter: ProviderRateLimiter, timeout: float):
        self._limiter = limiter
        self._timeout = timeout
        self._acquired = False

    async def __aenter__(self) -> "ProviderRateLimitContext":
        """Acquire rate limit on context entry."""
        self._acquired = await self._limiter.acquire(self._timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context (no automatic release)."""
        pass

    def __bool__(self) -> bool:
        """Check if rate limit was acquired."""
        return self._acquired

    def release_on_error(self) -> None:
        """Release the token back on request error."""
        if self._acquired:
            self._limiter.release_on_error()


class ProviderRateLimiterRegistry:
    """Registry for per-provider rate limiters.

    Provides isolated rate limiters for each API provider to avoid
    global lock contention. Each provider gets its own rate limiter
    instance with its own lock.

    Thread-safe access to rate limiters with lazy initialization.
    """

    def __init__(self) -> None:
        self._limiters: Dict[str, ProviderRateLimiter] = {}
        self._lock = threading.Lock()

    def get(
        self, provider: str, rpm: Optional[int] = None, burst: Optional[int] = None
    ) -> ProviderRateLimiter:
        """Get or create a rate limiter for a provider.

        Args:
            provider: Provider name (e.g., 'anthropic', 'openai')
            rpm: Override requests per minute (only used on first access)
            burst: Override burst size (only used on first access)

        Returns:
            ProviderRateLimiter instance for the provider
        """
        provider = provider.lower()

        # Fast path: check without lock
        if provider in self._limiters:
            return self._limiters[provider]

        # Slow path: create new limiter with lock
        with self._lock:
            # Double-check after acquiring lock
            if provider not in self._limiters:
                self._limiters[provider] = ProviderRateLimiter(
                    provider=provider, rpm=rpm, burst=burst
                )
                logger.debug(f"Created rate limiter for provider: {provider}")
            return self._limiters[provider]

    def reset(self, provider: Optional[str] = None) -> None:
        """Reset rate limiter(s).

        Args:
            provider: Provider to reset, or None to reset all
        """
        with self._lock:
            if provider:
                if provider.lower() in self._limiters:
                    del self._limiters[provider.lower()]
                    logger.debug(f"Reset rate limiter for provider: {provider}")
            else:
                self._limiters.clear()
                logger.debug("Reset all provider rate limiters")

    def stats(self) -> Dict[str, dict]:
        """Get statistics for all registered rate limiters."""
        with self._lock:
            return {provider: limiter.stats for provider, limiter in self._limiters.items()}

    def providers(self) -> list:
        """Get list of registered provider names."""
        with self._lock:
            return list(self._limiters.keys())


# Global registry for per-provider rate limiters
_provider_registry: Optional[ProviderRateLimiterRegistry] = None
_provider_registry_lock = threading.Lock()


def get_provider_limiter(
    provider: str, rpm: Optional[int] = None, burst: Optional[int] = None
) -> ProviderRateLimiter:
    """Get a rate limiter for a specific API provider.

    This is the primary interface for getting rate limiters.
    Each provider gets its own rate limiter instance with its own lock,
    eliminating global lock contention.

    Args:
        provider: Provider name (e.g., 'anthropic', 'openai', 'gemini', 'grok')
        rpm: Override requests per minute (only used on first access)
        burst: Override burst size (only used on first access)

    Returns:
        ProviderRateLimiter instance for the provider

    Example:
        # Get rate limiter for Anthropic
        limiter = get_provider_limiter("anthropic")

        # Use rate limiter
        async with limiter.request() as ctx:
            if ctx:
                response = await make_api_call()
    """
    global _provider_registry

    if _provider_registry is None:
        with _provider_registry_lock:
            if _provider_registry is None:
                _provider_registry = ProviderRateLimiterRegistry()

    return _provider_registry.get(provider, rpm=rpm, burst=burst)


def get_provider_registry() -> ProviderRateLimiterRegistry:
    """Get the global provider rate limiter registry.

    Returns:
        The singleton ProviderRateLimiterRegistry instance
    """
    global _provider_registry

    if _provider_registry is None:
        with _provider_registry_lock:
            if _provider_registry is None:
                _provider_registry = ProviderRateLimiterRegistry()

    return _provider_registry


def reset_provider_limiters(provider: Optional[str] = None) -> None:
    """Reset rate limiter(s) for providers.

    Args:
        provider: Provider to reset, or None to reset all
    """
    registry = get_provider_registry()
    registry.reset(provider)


# Use ServiceRegistry for rate limiter singleton management
_openrouter_limiter_lock = threading.Lock()


def get_openrouter_limiter() -> OpenRouterRateLimiter:
    """Get or create the global OpenRouter rate limiter.

    Uses ServiceRegistry for centralized singleton management.
    """
    from aragora.services import ServiceRegistry

    with _openrouter_limiter_lock:
        registry = ServiceRegistry.get()
        if not registry.has(OpenRouterRateLimiter):
            registry.register(OpenRouterRateLimiter, OpenRouterRateLimiter())
        return registry.resolve(OpenRouterRateLimiter)


def set_openrouter_tier(tier: str) -> None:
    """Set the OpenRouter rate limit tier.

    Valid tiers: free, basic, standard, premium, unlimited
    """
    from aragora.services import ServiceRegistry

    with _openrouter_limiter_lock:
        registry = ServiceRegistry.get()
        registry.register(OpenRouterRateLimiter, OpenRouterRateLimiter(tier=tier))


__all__ = [
    # Exponential backoff
    "ExponentialBackoff",
    # OpenRouter (legacy)
    "OpenRouterTier",
    "OPENROUTER_TIERS",
    "OpenRouterRateLimiter",
    "RateLimitContext",
    "get_openrouter_limiter",
    "set_openrouter_tier",
    # Per-provider rate limiters (new)
    "ProviderTier",
    "PROVIDER_DEFAULT_TIERS",
    "ProviderRateLimiter",
    "ProviderRateLimitContext",
    "ProviderRateLimiterRegistry",
    "get_provider_limiter",
    "get_provider_registry",
    "reset_provider_limiters",
]
