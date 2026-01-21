"""
Standardized HTTP client configuration with proper timeouts.

Provides consistent timeout and session management across all aiohttp usage.
All aiohttp sessions should use these utilities instead of creating sessions directly.

Usage:
    from aragora.http_client import get_default_timeout, create_client_session

    # For context manager usage:
    async with create_client_session() as session:
        await session.get(url)

    # For persistent sessions:
    session = await create_client_session()
    try:
        await session.get(url)
    finally:
        await session.close()
"""

from __future__ import annotations

import aiohttp
from aiohttp import ClientTimeout

__all__ = [
    "DEFAULT_TIMEOUT",
    "WEBHOOK_TIMEOUT",
    "LONG_TIMEOUT",
    "get_default_timeout",
    "create_client_session",
]

# Default timeout for most HTTP requests (30 seconds total)
DEFAULT_TIMEOUT = ClientTimeout(
    total=30,  # Total time for the entire request
    connect=10,  # Time to establish connection
    sock_read=20,  # Time to read response
)

# Shorter timeout for webhooks to fail fast
WEBHOOK_TIMEOUT = ClientTimeout(
    total=15,
    connect=5,
    sock_read=10,
)

# Longer timeout for slow operations (uploads, AI inference)
LONG_TIMEOUT = ClientTimeout(
    total=120,
    connect=10,
    sock_read=110,
)


def get_default_timeout() -> ClientTimeout:
    """Get the default timeout configuration.

    Returns:
        ClientTimeout with sensible defaults for most operations.
    """
    return DEFAULT_TIMEOUT


def create_client_session(
    timeout: ClientTimeout | None = None,
    **kwargs,
) -> aiohttp.ClientSession:
    """Create an aiohttp ClientSession with proper timeout configuration.

    Args:
        timeout: Optional custom timeout. Uses DEFAULT_TIMEOUT if not specified.
        **kwargs: Additional arguments passed to ClientSession.

    Returns:
        Configured aiohttp.ClientSession.

    Example:
        async with create_client_session() as session:
            async with session.get("https://api.example.com") as resp:
                data = await resp.json()
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    return aiohttp.ClientSession(timeout=timeout, **kwargs)
