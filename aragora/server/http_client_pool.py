"""
HTTP Client Connection Pooling for External API Calls.

Provides optimized HTTP connection pooling for LLM API providers:
- Per-provider connection pools (Anthropic, OpenAI, Mistral, etc.)
- Connection reuse and keep-alive
- Timeout management
- Retry with exponential backoff
- Health checking and metrics

Usage:
    from aragora.server.http_client_pool import HTTPClientPool

    pool = HTTPClientPool.get_instance()

    # Get a pooled session for a provider
    async with pool.get_session("anthropic") as session:
        response = await session.post(url, json=payload)

    # Or use the synchronous interface
    session = pool.get_sync_session("openai")
    response = session.post(url, json=payload)

Environment Variables:
    ARAGORA_HTTP_POOL_SIZE: Connections per provider (default: 20)
    ARAGORA_HTTP_TIMEOUT: Request timeout in seconds (default: 60)
    ARAGORA_HTTP_CONNECT_TIMEOUT: Connection timeout (default: 10)
    ARAGORA_HTTP_KEEPALIVE: Keep-alive timeout (default: 30)
    ARAGORA_HTTP_MAX_RETRIES: Max retry attempts (default: 3)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class HTTPPoolConfig:
    """Configuration for HTTP connection pool."""

    # Pool sizing
    pool_size: int = 20  # Connections per provider
    max_overflow: int = 10  # Additional connections during spikes

    # Timeouts
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    write_timeout: float = 30.0
    pool_timeout: float = 5.0  # Wait for available connection
    keepalive_timeout: float = 30.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.5
    retry_multiplier: float = 2.0
    retry_max_delay: float = 10.0

    # Rate limit handling
    retry_on_429: bool = True
    retry_429_delay: float = 1.0

    # Health checking
    health_check_interval: float = 60.0


@dataclass
class ProviderMetrics:
    """Metrics for a single provider's connection pool."""

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    requests_retried: int = 0
    rate_limits_hit: int = 0
    timeouts: int = 0
    total_latency_ms: float = 0.0
    connections_reused: int = 0
    connections_created: int = 0
    last_request_time: float = 0.0


@dataclass
class HTTPPoolMetrics:
    """Aggregated metrics for all pools."""

    providers: Dict[str, ProviderMetrics] = field(default_factory=dict)
    pool_exhaustion_count: int = 0
    total_wait_time_ms: float = 0.0

    def get_provider_metrics(self, provider: str) -> ProviderMetrics:
        if provider not in self.providers:
            self.providers[provider] = ProviderMetrics()
        return self.providers[provider]


# Provider-specific configurations
PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "pool_size": 25,  # Higher for primary provider
        "read_timeout": 120.0,  # Claude can have long responses
    },
    "openai": {
        "base_url": "https://api.openai.com",
        "pool_size": 20,
        "read_timeout": 90.0,
    },
    "mistral": {
        "base_url": "https://api.mistral.ai",
        "pool_size": 15,
        "read_timeout": 60.0,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api",
        "pool_size": 20,  # Fallback provider needs capacity
        "read_timeout": 120.0,
    },
    "xai": {
        "base_url": "https://api.x.ai",
        "pool_size": 10,
        "read_timeout": 60.0,
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com",
        "pool_size": 15,
        "read_timeout": 90.0,
    },
}


class HTTPClientPool:
    """
    Manages HTTP connection pools for external API providers.

    Provides connection reuse, health checking, and metrics for
    all LLM API provider connections.
    """

    _instance: Optional["HTTPClientPool"] = None
    _lock = threading.Lock()

    def __init__(self, config: Optional[HTTPPoolConfig] = None):
        """Initialize the HTTP client pool."""
        self.config = config or self._load_config_from_env()
        self.metrics = HTTPPoolMetrics()
        self._sync_sessions: Dict[str, Any] = {}
        self._async_clients: Dict[str, Any] = {}
        self._session_lock = threading.Lock()
        self._initialized_providers: Set[str] = set()
        self._closed = False

    @classmethod
    def get_instance(cls) -> "HTTPClientPool":
        """Get singleton instance of the HTTP client pool."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = HTTPClientPool()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None

    def _load_config_from_env(self) -> HTTPPoolConfig:
        """Load configuration from environment variables."""
        return HTTPPoolConfig(
            pool_size=int(os.getenv("ARAGORA_HTTP_POOL_SIZE", "20")),
            connect_timeout=float(os.getenv("ARAGORA_HTTP_CONNECT_TIMEOUT", "10")),
            read_timeout=float(os.getenv("ARAGORA_HTTP_TIMEOUT", "60")),
            keepalive_timeout=float(os.getenv("ARAGORA_HTTP_KEEPALIVE", "30")),
            max_retries=int(os.getenv("ARAGORA_HTTP_MAX_RETRIES", "3")),
        )

    def _get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get merged configuration for a provider."""
        base_config = {
            "pool_size": self.config.pool_size,
            "connect_timeout": self.config.connect_timeout,
            "read_timeout": self.config.read_timeout,
            "keepalive_timeout": self.config.keepalive_timeout,
        }

        provider_overrides = PROVIDER_CONFIGS.get(provider, {})
        base_config.update(provider_overrides)
        return base_config

    def _create_sync_session(self, provider: str) -> Any:
        """Create a synchronous requests session with connection pooling."""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
        except ImportError:
            logger.warning("requests not installed, falling back to basic urllib")
            return None

        config = self._get_provider_config(provider)
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"],
        )

        # Configure connection pool adapter
        adapter = HTTPAdapter(
            pool_connections=config["pool_size"],
            pool_maxsize=config["pool_size"] + self.config.max_overflow,
            max_retries=retry_strategy,
            pool_block=False,
        )

        # Mount adapter for HTTPS
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Set default headers
        session.headers.update(
            {
                "User-Agent": "Aragora/1.0",
                "Accept": "application/json",
                "Connection": "keep-alive",
            }
        )

        logger.debug(f"Created sync session for {provider} with pool_size={config['pool_size']}")
        self.metrics.get_provider_metrics(provider).connections_created += 1

        return session

    async def _create_async_client(self, provider: str) -> Any:
        """Create an async HTTP client with connection pooling."""
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed, async HTTP pooling unavailable")
            return None

        config = self._get_provider_config(provider)

        # Configure timeouts
        timeout = httpx.Timeout(
            connect=config["connect_timeout"],
            read=config["read_timeout"],
            write=self.config.write_timeout,
            pool=self.config.pool_timeout,
        )

        # Configure connection limits
        limits = httpx.Limits(
            max_connections=config["pool_size"] + self.config.max_overflow,
            max_keepalive_connections=config["pool_size"],
            keepalive_expiry=config["keepalive_timeout"],
        )

        # Create async client
        client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            http2=True,  # Enable HTTP/2 for better connection reuse
            headers={
                "User-Agent": "Aragora/1.0",
                "Accept": "application/json",
            },
        )

        logger.debug(f"Created async client for {provider} with pool_size={config['pool_size']}")
        self.metrics.get_provider_metrics(provider).connections_created += 1

        return client

    def get_sync_session(self, provider: str) -> Any:
        """Get a synchronous session for a provider.

        Returns a requests.Session with connection pooling configured.
        The session should be reused across requests.

        Args:
            provider: Provider name (anthropic, openai, mistral, etc.)

        Returns:
            Configured requests.Session or None if unavailable
        """
        if self._closed:
            raise RuntimeError("HTTPClientPool has been closed")

        with self._session_lock:
            if provider not in self._sync_sessions:
                self._sync_sessions[provider] = self._create_sync_session(provider)
            else:
                self.metrics.get_provider_metrics(provider).connections_reused += 1

        return self._sync_sessions[provider]

    @asynccontextmanager
    async def get_session(self, provider: str) -> AsyncIterator[Any]:
        """Get an async HTTP client for a provider.

        Usage:
            async with pool.get_session("anthropic") as client:
                response = await client.post(url, json=data)

        Args:
            provider: Provider name

        Yields:
            Configured httpx.AsyncClient
        """
        if self._closed:
            raise RuntimeError("HTTPClientPool has been closed")

        start_time = time.time()

        # Get or create client
        with self._session_lock:
            if provider not in self._async_clients:
                self._async_clients[provider] = await self._create_async_client(provider)
            else:
                self.metrics.get_provider_metrics(provider).connections_reused += 1

        client = self._async_clients[provider]
        if client is None:
            raise RuntimeError(f"Failed to create async client for {provider}")

        provider_metrics = self.metrics.get_provider_metrics(provider)

        try:
            yield client
            provider_metrics.requests_success += 1
        except Exception as e:
            provider_metrics.requests_failed += 1
            if "429" in str(e) or "rate" in str(e).lower():
                provider_metrics.rate_limits_hit += 1
            elif "timeout" in str(e).lower():
                provider_metrics.timeouts += 1
            raise
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            provider_metrics.total_latency_ms += elapsed_ms
            provider_metrics.requests_total += 1
            provider_metrics.last_request_time = time.time()

    async def request_with_retry(
        self,
        provider: str,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Any:
        """Make a request with automatic retry and backoff.

        Args:
            provider: Provider name for pool selection
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments for httpx

        Returns:
            httpx.Response on success

        Raises:
            Exception: If all retries fail
        """
        provider_metrics = self.metrics.get_provider_metrics(provider)
        last_exception: Optional[Exception] = None
        delay = self.config.retry_delay

        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.get_session(provider) as client:
                    response = await client.request(method, url, **kwargs)

                    # Handle rate limiting
                    if response.status_code == 429 and self.config.retry_on_429:
                        provider_metrics.rate_limits_hit += 1
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            delay = float(retry_after)
                        else:
                            delay = self.config.retry_429_delay * (attempt + 1)

                        if attempt < self.config.max_retries:
                            provider_metrics.requests_retried += 1
                            await asyncio.sleep(delay)
                            continue

                    return response

            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    provider_metrics.requests_retried += 1
                    await asyncio.sleep(delay)
                    delay = min(delay * self.config.retry_multiplier, self.config.retry_max_delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Request failed after all retries")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current pool metrics."""
        return {
            "providers": {
                name: {
                    "requests_total": m.requests_total,
                    "requests_success": m.requests_success,
                    "requests_failed": m.requests_failed,
                    "requests_retried": m.requests_retried,
                    "rate_limits_hit": m.rate_limits_hit,
                    "timeouts": m.timeouts,
                    "avg_latency_ms": (
                        m.total_latency_ms / m.requests_total if m.requests_total > 0 else 0
                    ),
                    "connections_reused": m.connections_reused,
                    "connections_created": m.connections_created,
                    "last_request_time": m.last_request_time,
                }
                for name, m in self.metrics.providers.items()
            },
            "pool_exhaustion_count": self.metrics.pool_exhaustion_count,
            "total_wait_time_ms": self.metrics.total_wait_time_ms,
            "active_providers": list(self._sync_sessions.keys()) + list(self._async_clients.keys()),
        }

    def close(self) -> None:
        """Close all connections and cleanup resources."""
        self._closed = True

        # Close sync sessions
        for provider, session in self._sync_sessions.items():
            try:
                if session and hasattr(session, "close"):
                    session.close()
                    logger.debug(f"Closed sync session for {provider}")
            except Exception as e:
                logger.warning(f"Error closing sync session for {provider}: {e}")

        self._sync_sessions.clear()

        # Close async clients (must be done in async context)
        # They will be closed when the event loop closes
        self._async_clients.clear()
        self._initialized_providers.clear()

        logger.info("HTTP client pool closed")

    async def aclose(self) -> None:
        """Async close for proper cleanup of async clients."""
        self._closed = True

        # Close async clients
        for provider, client in list(self._async_clients.items()):
            try:
                if client and hasattr(client, "aclose"):
                    await client.aclose()
                    logger.debug(f"Closed async client for {provider}")
            except Exception as e:
                logger.warning(f"Error closing async client for {provider}: {e}")

        self._async_clients.clear()

        # Close sync sessions
        self.close()


# Convenience function for getting the global pool
def get_http_pool() -> HTTPClientPool:
    """Get the global HTTP client pool instance."""
    return HTTPClientPool.get_instance()


__all__ = [
    "HTTPClientPool",
    "HTTPPoolConfig",
    "HTTPPoolMetrics",
    "ProviderMetrics",
    "PROVIDER_CONFIGS",
    "get_http_pool",
]
