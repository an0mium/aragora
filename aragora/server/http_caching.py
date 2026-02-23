"""
HTTP Caching utilities for ETag and Cache-Control headers.

Provides decorators and utilities for enabling HTTP caching on API endpoints,
supporting:
- ETag generation and validation (304 Not Modified)
- Cache-Control headers (max-age, public/private)
- Last-Modified headers
- Serialization cache with TTL to avoid re-serializing identical data

Usage:
    from aragora.server.http_caching import cache_control, with_etag

    @cache_control(max_age=60, public=True)
    def handle_leaderboard(handler):
        ...

    # Or apply ETag to response data
    response_data, headers = with_etag(data, request_etag)

    # For high-throughput endpoints, use the serialization cache
    from aragora.server.http_caching import get_cached_serialization
    cached = get_cached_serialization(cache_key, data, ttl=60)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime
from typing import Any, NamedTuple, TypeVar

logger = logging.getLogger(__name__)

# Type for response data
T = TypeVar("T")


# =============================================================================
# Serialization Cache (TTL-based)
# =============================================================================


class CacheEntry(NamedTuple):
    """A cached serialization entry."""

    serialized: str
    etag: str
    timestamp: float


# Global serialization cache
_serialization_cache: dict[str, CacheEntry] = {}
_cache_lock = threading.Lock()

# Cache configuration
SERIALIZATION_CACHE_MAX_SIZE = 1000  # Max entries
SERIALIZATION_CACHE_DEFAULT_TTL = 60  # Default TTL in seconds


def _evict_expired_cache_entries() -> None:
    """Remove expired entries from the serialization cache.

    Called under _cache_lock.
    """
    now = time.time()
    expired_keys = [
        key
        for key, entry in _serialization_cache.items()
        if now - entry.timestamp > SERIALIZATION_CACHE_DEFAULT_TTL * 2  # 2x TTL for cleanup
    ]
    for key in expired_keys:
        del _serialization_cache[key]


def _evict_oldest_entries(count: int) -> None:
    """Evict oldest entries to make room.

    Called under _cache_lock.
    """
    if count <= 0:
        return
    # Sort by timestamp and remove oldest
    sorted_keys = sorted(
        _serialization_cache.keys(),
        key=lambda k: _serialization_cache[k].timestamp,
    )
    for key in sorted_keys[:count]:
        del _serialization_cache[key]


def get_cached_serialization(
    cache_key: str,
    data: Any,
    ttl: int = SERIALIZATION_CACHE_DEFAULT_TTL,
) -> tuple[str, str]:
    """Get or create cached serialization and ETag for data.

    Avoids re-serializing identical data within TTL window, which provides
    significant performance improvement for high-throughput analytics endpoints.

    Args:
        cache_key: Unique key for this data (e.g., endpoint + params hash)
        data: The data to serialize
        ttl: Time-to-live in seconds (default 60)

    Returns:
        Tuple of (serialized_json, etag)
    """
    now = time.time()

    with _cache_lock:
        # Check cache first
        if cache_key in _serialization_cache:
            entry = _serialization_cache[cache_key]
            if now - entry.timestamp <= ttl:
                return entry.serialized, entry.etag

        # Evict expired entries periodically (every ~10% of max size)
        if len(_serialization_cache) > SERIALIZATION_CACHE_MAX_SIZE * 0.9:
            _evict_expired_cache_entries()

        # If still at capacity, evict oldest 10%
        if len(_serialization_cache) >= SERIALIZATION_CACHE_MAX_SIZE:
            _evict_oldest_entries(int(SERIALIZATION_CACHE_MAX_SIZE * 0.1))

    # Serialize outside lock
    try:
        serialized = json.dumps(data, sort_keys=True, default=str)
        hash_value = hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()[:16]
        etag = f'"{hash_value}"'
    except (TypeError, ValueError) as e:
        logger.debug("Cached serialization failed: %s", e)
        serialized = json.dumps(str(data))
        etag = f'"{hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()[:16]}"'

    # Store in cache
    entry = CacheEntry(serialized=serialized, etag=etag, timestamp=now)
    with _cache_lock:
        _serialization_cache[cache_key] = entry

    return serialized, etag


def clear_serialization_cache() -> int:
    """Clear the serialization cache.

    Returns:
        Number of entries cleared
    """
    with _cache_lock:
        count = len(_serialization_cache)
        _serialization_cache.clear()
        return count


def get_serialization_cache_stats() -> dict[str, Any]:
    """Get statistics about the serialization cache.

    Returns:
        Dictionary with cache statistics
    """
    with _cache_lock:
        now = time.time()
        valid_count = sum(
            1
            for entry in _serialization_cache.values()
            if now - entry.timestamp <= SERIALIZATION_CACHE_DEFAULT_TTL
        )
        return {
            "total_entries": len(_serialization_cache),
            "valid_entries": valid_count,
            "expired_entries": len(_serialization_cache) - valid_count,
            "max_size": SERIALIZATION_CACHE_MAX_SIZE,
            "default_ttl": SERIALIZATION_CACHE_DEFAULT_TTL,
        }


# =============================================================================
# Cache Configuration
# =============================================================================

# Default cache durations by endpoint pattern
CACHE_DURATIONS: dict[str, int] = {
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


def get_cache_duration(path: str) -> int | None:
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
        hash_value = hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()[:16]
        return f'"{hash_value}"'
    except (TypeError, ValueError) as e:
        logger.debug("ETag generation failed: %s", e)
        return f'"{hashlib.md5(str(data).encode(), usedforsecurity=False).hexdigest()[:16]}"'


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


def etag_matches(request_etag: str | None, response_etag: str) -> bool:
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
    etag: str | None = None,
    last_modified: datetime | None = None,
) -> dict[str, str]:
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
    headers: dict[str, str] = {}

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


def no_cache_headers() -> dict[str, str]:
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
    request_etag: str | None = None,
    max_age: int = 60,
    public: bool = True,
) -> tuple[T | None, dict[str, str], bool]:
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
    request_etag: str | None = None,
) -> bool | None:
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
    "SERIALIZATION_CACHE_MAX_SIZE",
    "SERIALIZATION_CACHE_DEFAULT_TTL",
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
    # Serialization cache
    "get_cached_serialization",
    "clear_serialization_cache",
    "get_serialization_cache_stats",
]
