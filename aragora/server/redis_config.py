"""Deprecated import location for the Redis connection-pool helpers.

The shared surface moved down to :mod:`aragora.utils.redis_config` during the
P4a layering work so that foundation-layer modules (e.g.
``aragora.utils.redis_cache``) can reach it without importing
``aragora.server``. Importing from ``aragora.server.redis_config`` still works
but is deprecated; import from ``aragora.utils.redis_config`` instead.
"""

from __future__ import annotations

import warnings

from aragora.utils.redis_config import (
    close_redis_pool,
    get_async_redis_client,
    get_redis_client,
    get_redis_pool,
    get_redis_url,
    is_redis_available,
    reset_redis_state,
)

warnings.warn(
    "aragora.server.redis_config is deprecated; import from aragora.utils.redis_config instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "get_redis_url",
    "get_redis_pool",
    "get_redis_client",
    "get_async_redis_client",
    "is_redis_available",
    "close_redis_pool",
    "reset_redis_state",
]
