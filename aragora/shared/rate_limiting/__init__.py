"""
Shared rate limiting utilities.

This module provides common rate limiting components used across the codebase:

- TokenBucket: Core token bucket rate limiter (sync and async)
- KeyedTokenBucket: Per-key rate limiting (per-user, per-endpoint, etc.)
- ExponentialBackoff: Backoff with jitter for rate limit recovery

For server-side HTTP rate limiting, see:
    aragora.server.middleware.rate_limit

For client-side API rate limiting, see:
    aragora.agents.api_agents.rate_limiter

Example:
    from aragora.shared.rate_limiting import TokenBucket

    # Create a rate limiter: 100 requests/minute, burst of 20
    bucket = TokenBucket(rate_per_minute=100, burst=20)

    # Async acquisition (recommended for API agents)
    if await bucket.acquire_async(timeout=5.0):
        await make_request()

    # Sync non-blocking check (for events)
    if bucket.try_acquire():
        process_event()
"""

from .backoff import ExponentialBackoff
from .token_bucket import KeyedTokenBucket, TokenBucket, TokenBucketStats

__all__ = [
    "TokenBucket",
    "KeyedTokenBucket",
    "TokenBucketStats",
    "ExponentialBackoff",
]
