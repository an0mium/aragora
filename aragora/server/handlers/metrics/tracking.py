"""
Thread-safe request and verification tracking for metrics.

This module provides tracking functions for:
- Request counts and error rates per endpoint
- Formal verification (Z3) outcomes and timing
"""

from __future__ import annotations

import threading
import time
from typing import Any

# Request tracking for metrics (thread-safe)
_request_counts: dict[str, int] = {}
_error_counts: dict[str, int] = {}
_metrics_lock = threading.Lock()
_start_time = time.time()
MAX_TRACKED_ENDPOINTS = 1000  # Prevent unbounded dict growth
_VALID_VERIFICATION_STATUSES = frozenset(
    {
        "z3_verified",
        "z3_disproved",
        "z3_timeout",
        "z3_translation_failed",
        "confidence_fallback",
    }
)

# Verification metrics tracking (thread-safe)
_verification_stats: dict[str, int | float] = {
    "total_claims_processed": 0,
    "z3_verified": 0,
    "z3_disproved": 0,
    "z3_timeout": 0,
    "z3_translation_failed": 0,
    "confidence_fallback": 0,
    "total_verification_time_ms": 0.0,
}
_verification_lock = threading.Lock()


def get_start_time() -> float:
    """Get the server start time."""
    return _start_time


def track_verification(
    status: str,
    verification_time_ms: float = 0.0,
) -> None:
    """Track verification outcome (thread-safe).

    Args:
        status: One of 'z3_verified', 'z3_disproved', 'z3_timeout',
                'z3_translation_failed', 'confidence_fallback'
        verification_time_ms: Time taken for verification in milliseconds
    """
    if not isinstance(status, str) or not status.strip():
        raise ValueError("status is required and must be a non-empty string")
    if status not in _VALID_VERIFICATION_STATUSES:
        raise ValueError(
            f"status must be one of: {', '.join(sorted(_VALID_VERIFICATION_STATUSES))}"
        )
    if isinstance(verification_time_ms, bool) or not isinstance(verification_time_ms, int | float):
        raise TypeError("verification_time_ms must be a number")
    if verification_time_ms < 0:
        raise ValueError("verification_time_ms must be non-negative")

    with _verification_lock:
        _verification_stats["total_claims_processed"] += 1
        _verification_stats[status] += 1
        _verification_stats["total_verification_time_ms"] += verification_time_ms


def get_verification_stats() -> dict[str, Any]:
    """Get verification statistics (thread-safe snapshot)."""
    with _verification_lock:
        stats = dict(_verification_stats)

    # Calculate derived metrics
    total = stats["total_claims_processed"]
    if total > 0:
        stats["avg_verification_time_ms"] = round(stats["total_verification_time_ms"] / total, 2)
        stats["z3_success_rate"] = round(stats["z3_verified"] / total, 4)
    else:
        stats["avg_verification_time_ms"] = 0.0
        stats["z3_success_rate"] = 0.0

    return stats


def track_request(endpoint: str, is_error: bool = False) -> None:
    """Track a request for metrics (thread-safe)."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint is required and must be a non-empty string")
    if not isinstance(is_error, bool):
        raise TypeError("is_error must be a boolean")

    with _metrics_lock:
        # Enforce max size - remove oldest entries if at capacity
        if endpoint not in _request_counts and len(_request_counts) >= MAX_TRACKED_ENDPOINTS:
            # Remove first 10% of entries (approximate LRU eviction)
            keys = list(_request_counts.keys())
            remove_count = max(1, len(keys) // 10)
            for k in keys[:remove_count]:
                del _request_counts[k]
                _error_counts.pop(k, None)
        _request_counts[endpoint] = _request_counts.get(endpoint, 0) + 1
        if is_error:
            _error_counts[endpoint] = _error_counts.get(endpoint, 0) + 1


def get_request_stats() -> dict[str, Any]:
    """Get request tracking statistics (thread-safe snapshot).

    Returns:
        Dictionary with total_requests, total_errors, and counts_snapshot
    """
    with _metrics_lock:
        total_requests = sum(_request_counts.values())
        total_errors = sum(_error_counts.values())
        counts_snapshot = list(_request_counts.items())

    return {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "counts_snapshot": counts_snapshot,
    }
