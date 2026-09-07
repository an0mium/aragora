"""Deprecated import location for the adaptive TTL cache.

The implementation now lives at :mod:`aragora.caching.adaptive`. Importing from
``aragora.performance.adaptive_cache`` still works but is deprecated; import
from ``aragora.caching.adaptive`` instead.
"""

from __future__ import annotations

import warnings

from aragora.caching.adaptive import (
    AccessPattern,
    AdaptiveTTLCache,
    CacheEntry,
    CacheOptimizer,
    CacheStats,
)

warnings.warn(
    "aragora.performance.adaptive_cache is deprecated; "
    "import from aragora.caching.adaptive instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AccessPattern",
    "CacheEntry",
    "CacheStats",
    "AdaptiveTTLCache",
    "CacheOptimizer",
]
