"""
Redis connection pool configuration.

Provides centralized Redis connection management with:
- Lazy initialization on first use
- Connection pooling for efficiency
- Automatic fallback when Redis unavailable
- Health checking with ping validation

Usage:
    from aragora.server.redis_config import get_redis_pool, is_redis_available

    if is_redis_available():
        pool = get_redis_pool()
        client = redis.Redis(connection_pool=pool)
        client.set("key", "value")

Environment variables:
    ARAGORA_REDIS_URL: Redis connection URL (e.g., redis://localhost:6379)
    ARAGORA_REDIS_MAX_CONNECTIONS: Max pool connections (default: 50)
    ARAGORA_REDIS_SOCKET_TIMEOUT: Socket timeout in seconds (default: 5.0)
"""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Module-level connection pool (lazy initialized)
_redis_pool: Optional[Any] = None
_redis_available: Optional[bool] = None


def get_redis_url() -> Optional[str]:
    """Get the Redis URL from environment.

    Returns:
        Redis URL if configured, None otherwise
    """
    return os.getenv("ARAGORA_REDIS_URL")


def get_redis_pool() -> Optional[Any]:
    """Get shared Redis connection pool (lazy initialization).

    Thread-safe lazy initialization of the Redis connection pool.
    Returns None if Redis is not configured or unavailable.

    The pool is configured with:
    - Max connections from ARAGORA_REDIS_MAX_CONNECTIONS (default 50)
    - Socket timeout of 5 seconds
    - Retry on timeout enabled
    - Automatic reconnection

    Returns:
        redis.ConnectionPool if Redis available, None otherwise
    """
    global _redis_pool, _redis_available

    # Return cached pool if already initialized
    if _redis_pool is not None:
        return _redis_pool

    # Return None if we already know Redis is unavailable
    if _redis_available is False:
        return None

    url = get_redis_url()
    if not url:
        _redis_available = False
        return None

    try:
        import redis

        max_connections = int(os.getenv("ARAGORA_REDIS_MAX_CONNECTIONS", "50"))
        socket_timeout = float(os.getenv("ARAGORA_REDIS_SOCKET_TIMEOUT", "5.0"))

        _redis_pool = redis.ConnectionPool.from_url(
            url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            retry_on_timeout=True,
            decode_responses=True,  # Return strings instead of bytes
        )

        # Test connection with ping
        test_client = redis.Redis(connection_pool=_redis_pool)
        test_client.ping()

        _redis_available = True
        # Mask password in URL for logging
        safe_url = url.split("@")[-1] if "@" in url else url
        logger.info(f"Redis connected: {safe_url}")
        return _redis_pool

    except ImportError:
        logger.debug("redis package not installed, Redis caching disabled")
        _redis_available = False
        return None
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        _redis_available = False
        return None


def is_redis_available() -> bool:
    """Check if Redis is available.

    Triggers pool initialization if not already done.

    Returns:
        True if Redis is configured and responding, False otherwise
    """
    global _redis_available

    if _redis_available is not None:
        return _redis_available

    # Try to initialize the pool
    get_redis_pool()
    return _redis_available or False


def get_redis_client() -> Optional[Any]:
    """Get a Redis client using the shared pool.

    Convenience function that returns a ready-to-use Redis client.

    Returns:
        redis.Redis instance if available, None otherwise
    """
    pool = get_redis_pool()
    if pool is None:
        return None

    try:
        import redis
        return redis.Redis(connection_pool=pool)
    except ImportError:
        return None


def close_redis_pool() -> None:
    """Close the Redis connection pool.

    Call during graceful shutdown to release connections.
    Safe to call even if pool was never initialized.
    """
    global _redis_pool, _redis_available

    if _redis_pool is not None:
        try:
            _redis_pool.disconnect()
            logger.debug("Redis connection pool closed")
        except Exception as e:
            logger.warning(f"Error closing Redis pool: {e}")
        finally:
            _redis_pool = None

    _redis_available = None


def reset_redis_state() -> None:
    """Reset Redis state for testing.

    Clears cached pool and availability flag to allow re-initialization.
    """
    global _redis_pool, _redis_available
    _redis_pool = None
    _redis_available = None


__all__ = [
    "get_redis_url",
    "get_redis_pool",
    "get_redis_client",
    "is_redis_available",
    "close_redis_pool",
    "reset_redis_state",
]
