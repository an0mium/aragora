"""
Shared rate limiting utilities.

This module provides common rate limiting components used across the codebase:

- ExponentialBackoff: Backoff with jitter for rate limit recovery

For server-side HTTP rate limiting, see:
    aragora.server.middleware.rate_limit

For client-side API rate limiting, see:
    aragora.agents.api_agents.rate_limiter
"""

from .backoff import ExponentialBackoff

__all__ = ["ExponentialBackoff"]
