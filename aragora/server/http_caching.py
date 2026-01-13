"""
HTTP Caching utilities for ETag and Cache-Control headers.

Provides decorators and utilities for enabling HTTP caching on API endpoints,
supporting:
- ETag generation and validation (304 Not Modified)
- Cache-Control headers (max-age, public/private)
- Last-Modified headers

Usage:
    from aragora.server.http_caching import cache_control, with_etag

    @cache_control(max_age=60, public=True)
    def handle_leaderboard(handler):
        ...

    # Or apply ETag to response data
    response_data, headers = with_etag(data, request_etag)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

# Type for response data
T = TypeVar("T")


# =============================================================================
# Cache Configuration
# =============================================================================

# Default cache durations by endpoint pattern
CACHE_DURATIONS: Dict[str, int] = {
    "/api/leaderboard": 60,  # 1 minute
    "/api/agents/": 120,  # 2 minutes for agent profiles
    "/api/analytics/": 300,  # 5 minutes for analytics
    "/api/consensus/stats": 180,  # 3 minutes
    "/api/pulse/trends": 120,  # 2 minutes
}

# Endpoints that should never be cached
NO_CACHE_PATTERNS = [
    "/api/debates/",  # Active debates change frequently
    "/api/auth/",  # Auth endpoints
    "/api/user/",  # User-specific data
]


def get_cache_duration(path: str) -> Optional[int]:
    """Get cache duration for a path, or None if not cacheable.

    Args:
        path: The request path

    Returns:
        Cache duration in seconds, or None if path should not be cached
    """
    # Check no-cache patterns first
    for pattern in NO_CACHE_PATTERNS:
        if pattern in path:
            return None

    # Check for specific duration
    for pattern, duration in CACHE_DURATIONS.items():
        if path.startswith(pattern):
            return duration

    return None


# =============================================================================
# ETag Generation
# =============================================================================


def generate_etag(data: Any) -> str:
    """Generate an ETag for response data.

    Uses MD5 hash of JSON-serialized data for efficient ETag generation.
    The hash is truncated and quoted per HTTP spec.

    Args:
        data: Response data (must be JSON serializable)

    Returns:
        ETag string (quoted, e.g., '"abc123"')
    """
    try:
        # Serialize with sorted keys for consistent hashing
        serialized = json.dumps(data, sort_keys=True, default=str)
        hash_value = hashlib.md5(serialized.encode()).hexdigest()[:16]
        return f'"{hash_value}"'
    except (TypeError, ValueError) as e:
        logger.debug(f"ETag generation failed: {e}")
        return f'"{hashlib.md5(str(data).encode()).hexdigest()[:16]}"'


def generate_weak_etag(data: Any) -> str:
    """Generate a weak ETag for response data.

    Weak ETags indicate semantic equivalence, not byte-for-byte identity.
    Useful when data may have minor formatting differences.

    Args:
        data: Response data

    Returns:
        Weak ETag string (e.g., 'W/"abc123"')
    """
    strong_etag = generate_etag(data)
    return f"W/{strong_etag}"


def etag_matches(request_etag: Optional[str], response_etag: str) -> bool:
    """Check if request ETag matches response ETag.

    Handles both strong and weak ETag comparison per RFC 7232.

    Args:
        request_etag: The If-None-Match header value
        response_etag: The generated ETag

    Returns:
        True if ETags match (304 should be returned)
    """
    if not request_etag:
        return False

    # Strip weak prefix for comparison
    def normalize(etag: str) -> str:
        if etag.startswith("W/"):
            return etag[2:]
        return etag

    # Handle comma-separated list of ETags
    request_etags = [e.strip() for e in request_etag.split(",")]

    for req_etag in request_etags:
        # Handle wildcard
        if req_etag == "*":
            return True
        if normalize(req_etag) == normalize(response_etag):
            return True

    return False


# =============================================================================
# Cache Headers
# =============================================================================


def cache_headers(
    max_age: int = 60,
    public: bool = True,
    must_revalidate: bool = False,
    etag: Optional[str] = None,
    last_modified: Optional[datetime] = None,
) -> Dict[str, str]:
    """Generate cache headers for a response.

    Args:
        max_age: Cache duration in seconds
        public: Whether response can be cached by shared caches (CDN)
        must_revalidate: Whether cache must revalidate before serving stale
        etag: Optional ETag value
        last_modified: Optional last modification timestamp

    Returns:
        Dictionary of headers to add to response
    """
    headers: Dict[str, str] = {}

    # Build Cache-Control
    directives = []
    directives.append("public" if public else "private")
    directives.append(f"max-age={max_age}")
    if must_revalidate:
        directives.append("must-revalidate")

    headers["Cache-Control"] = ", ".join(directives)

    # Add ETag
    if etag:
        headers["ETag"] = etag

    # Add Last-Modified
    if last_modified:
        # Format per HTTP spec (RFC 7231)
        headers["Last-Modified"] = last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT")

    return headers


def no_cache_headers() -> Dict[str, str]:
    """Generate headers to prevent caching.

    Returns:
        Dictionary of headers that prevent caching
    """
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }


# =============================================================================
# High-Level Utilities
# =============================================================================


def with_etag(
    data: T,
    request_etag: Optional[str] = None,
    max_age: int = 60,
    public: bool = True,
) -> Tuple[Optional[T], Dict[str, str], bool]:
    """Apply ETag caching to response data.

    Generates ETag for data and checks if it matches request ETag.
    If matched, returns None for data (caller should return 304).

    Args:
        data: Response data
        request_etag: The If-None-Match header from request
        max_age: Cache duration in seconds
        public: Whether response can be publicly cached

    Returns:
        Tuple of (data_or_none, headers, is_not_modified)
        - If ETags match: (None, headers, True) - return 304
        - Otherwise: (data, headers, False) - return 200 with data
    """
    etag = generate_etag(data)

    if etag_matches(request_etag, etag):
        # 304 Not Modified
        return None, {"ETag": etag}, True

    # Return full response with caching headers
    headers = cache_headers(max_age=max_age, public=public, etag=etag)
    return data, headers, False


def apply_cache_headers_to_response(
    response: Any,
    path: str,
    data: Any = None,
    request_etag: Optional[str] = None,
) -> Optional[bool]:
    """Apply appropriate cache headers to a response object.

    Determines cache duration based on path and adds headers.
    Returns True if 304 should be returned (ETag match).

    Args:
        response: Response object with headers dict
        path: Request path for duration lookup
        data: Response data for ETag generation
        request_etag: If-None-Match header from request

    Returns:
        True if 304 should be returned, False otherwise, None if not cacheable
    """
    duration = get_cache_duration(path)

    if duration is None:
        # Apply no-cache headers
        for key, value in no_cache_headers().items():
            response.headers[key] = value
        return None

    # Generate headers with ETag
    if data is not None:
        etag = generate_etag(data)

        # Check for 304
        if etag_matches(request_etag, etag):
            response.headers["ETag"] = etag
            return True

        headers = cache_headers(max_age=duration, etag=etag)
    else:
        headers = cache_headers(max_age=duration)

    for key, value in headers.items():
        response.headers[key] = value

    return False


__all__ = [
    # Configuration
    "CACHE_DURATIONS",
    "NO_CACHE_PATTERNS",
    "get_cache_duration",
    # ETag utilities
    "generate_etag",
    "generate_weak_etag",
    "etag_matches",
    # Headers
    "cache_headers",
    "no_cache_headers",
    # High-level
    "with_etag",
    "apply_cache_headers_to_response",
]
