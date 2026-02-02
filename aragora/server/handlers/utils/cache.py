"""Compatibility shim for legacy handler cache imports.

Historically, handlers imported cache helpers from
`aragora.server.handlers.utils.cache`. Those helpers were moved to
`aragora.server.handlers.admin.cache`, but some code paths still
reference the old module path. Re-export from the new location to
preserve backward compatibility.
"""

from __future__ import annotations

from aragora.server.handlers.admin.cache import (
    CACHE_INVALIDATION_MAP,
    BoundedTTLCache,
    _cache,
    async_ttl_cache,
    clear_cache,
    get_cache_stats,
    get_handler_cache,
    invalidate_agent_cache,
    invalidate_cache,
    invalidate_debate_cache,
    invalidate_leaderboard_cache,
    invalidate_on_event,
    ttl_cache,
)

__all__ = [
    "CACHE_INVALIDATION_MAP",
    "BoundedTTLCache",
    "_cache",
    "async_ttl_cache",
    "clear_cache",
    "get_cache_stats",
    "get_handler_cache",
    "invalidate_agent_cache",
    "invalidate_cache",
    "invalidate_debate_cache",
    "invalidate_leaderboard_cache",
    "invalidate_on_event",
    "ttl_cache",
]
