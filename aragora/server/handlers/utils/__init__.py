"""Handler utilities module.

Provides reusable utilities for HTTP handlers including:
- Rate limiting (token bucket algorithm)
- Response formatting (to be extracted from base.py)
- Authentication decorators (to be extracted from base.py)
"""

from .rate_limit import RateLimiter, rate_limit, get_client_ip

__all__ = [
    "RateLimiter",
    "rate_limit",
    "get_client_ip",
]
