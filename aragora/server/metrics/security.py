"""
Security Metrics for Aragora.

Tracks authentication failures, rate limits, and security violations.
"""

from __future__ import annotations

from .types import Counter

# =============================================================================
# Security Metrics
# =============================================================================

AUTH_FAILURES = Counter(
    name="aragora_auth_failures_total",
    help="Authentication failures by reason and endpoint",
    label_names=["reason", "endpoint"],
)

RATE_LIMIT_HITS = Counter(
    name="aragora_rate_limit_hits_total",
    help="Rate limit hits by endpoint and limit type",
    label_names=["endpoint", "limit_type"],
)

SECURITY_VIOLATIONS = Counter(
    name="aragora_security_violations_total",
    help="Security violations by type (path_traversal, xss_attempt, etc)",
    label_names=["type"],
)


# =============================================================================
# Helpers
# =============================================================================


def track_auth_failure(reason: str, endpoint: str = "unknown") -> None:
    """Track an authentication failure.

    Args:
        reason: Why authentication failed (invalid_token, expired, wrong_password, etc)
        endpoint: The endpoint where the failure occurred
    """
    AUTH_FAILURES.inc(reason=reason, endpoint=endpoint)


def track_rate_limit_hit(endpoint: str, limit_type: str = "request") -> None:
    """Track when a rate limit is hit.

    Args:
        endpoint: The rate-limited endpoint
        limit_type: Type of limit (request, upload, etc)
    """
    RATE_LIMIT_HITS.inc(endpoint=endpoint, limit_type=limit_type)


def track_security_violation(violation_type: str) -> None:
    """Track a security violation attempt.

    Args:
        violation_type: Type of violation (path_traversal, xss_attempt, sql_injection, etc)
    """
    SECURITY_VIOLATIONS.inc(type=violation_type)


__all__ = [
    "AUTH_FAILURES",
    "RATE_LIMIT_HITS",
    "SECURITY_VIOLATIONS",
    "track_auth_failure",
    "track_rate_limit_hit",
    "track_security_violation",
]
