"""Deprecated import location for the Redis-backed TTL cache.

The implementation now lives at :mod:`aragora.caching.redis`. Importing from
``aragora.utils.redis_cache`` still works but is deprecated; import from
``aragora.caching.redis`` instead.
"""

from __future__ import annotations

import warnings

from aragora.caching.redis import HybridTTLCache, RedisTTLCache

warnings.warn(
    "aragora.utils.redis_cache is deprecated; import from aragora.caching.redis instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "RedisTTLCache",
    "HybridTTLCache",
]
