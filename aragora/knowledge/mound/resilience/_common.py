"""Shared utilities for the resilience package."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, TypeVar

# Python 3.11+ has asyncio.timeout, earlier versions need async-timeout
if sys.version_info >= (3, 11):
    asyncio_timeout = asyncio.timeout
else:
    try:
        from async_timeout import timeout as asyncio_timeout
    except ImportError:
        # Fallback: create a simple context manager that doesn't timeout
        @asynccontextmanager
        async def _fallback_timeout(delay: float) -> AsyncIterator[None]:
            """Fallback timeout context manager (no-op)."""
            _ = delay  # unused, but matches signature
            yield

        asyncio_timeout = _fallback_timeout


T = TypeVar("T")
