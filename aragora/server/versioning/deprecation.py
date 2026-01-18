"""
API Deprecation Middleware.

Provides:
- Deprecation headers (RFC 8594)
- Sunset headers for end-of-life dates
- Logging of deprecated endpoint usage
- Metrics for deprecation monitoring

Usage:
    middleware = DeprecationMiddleware()

    @deprecated(
        sunset_date=date(2025, 6, 1),
        replacement="/api/v2/users",
    )
    def old_handler(request):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# Deprecation Types
# ============================================================================


class DeprecationLevel(Enum):
    """Deprecation severity levels."""

    WARNING = "warning"  # Still functional, will be removed
    CRITICAL = "critical"  # Imminent removal
    SUNSET = "sunset"  # Past sunset date, may fail


@dataclass
class DeprecationWarning:
    """Deprecation warning for an endpoint."""

    path: str
    method: str
    level: DeprecationLevel
    message: str
    sunset_date: Optional[date] = None
    replacement: Optional[str] = None
    migration_guide: Optional[str] = None
    deprecated_since: Optional[date] = None

    def to_header_value(self) -> str:
        """Format as Deprecation header value (RFC 8594)."""
        if self.sunset_date:
            return f'@{int(datetime.combine(self.sunset_date, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp())}'
        return "true"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON responses."""
        return {
            "path": self.path,
            "method": self.method,
            "level": self.level.value,
            "message": self.message,
            "sunset_date": self.sunset_date.isoformat() if self.sunset_date else None,
            "replacement": self.replacement,
            "migration_guide": self.migration_guide,
        }


# ============================================================================
# Deprecation Registry
# ============================================================================


@dataclass
class DeprecationRegistry:
    """Registry of deprecated endpoints."""

    entries: Dict[str, DeprecationWarning] = field(default_factory=dict)
    usage_counts: Dict[str, int] = field(default_factory=dict)

    def _key(self, path: str, method: str) -> str:
        return f"{method.upper()}:{path}"

    def register(
        self,
        path: str,
        method: str,
        sunset_date: Optional[date] = None,
        replacement: Optional[str] = None,
        message: Optional[str] = None,
        migration_guide: Optional[str] = None,
    ) -> None:
        """Register a deprecated endpoint."""
        key = self._key(path, method)

        # Determine deprecation level
        level = DeprecationLevel.WARNING
        if sunset_date:
            days_left = (sunset_date - date.today()).days
            if days_left < 0:
                level = DeprecationLevel.SUNSET
            elif days_left < 30:
                level = DeprecationLevel.CRITICAL

        self.entries[key] = DeprecationWarning(
            path=path,
            method=method,
            level=level,
            message=message or f"Endpoint {method} {path} is deprecated",
            sunset_date=sunset_date,
            replacement=replacement,
            migration_guide=migration_guide,
            deprecated_since=date.today(),
        )

    def is_deprecated(self, path: str, method: str) -> bool:
        """Check if endpoint is deprecated."""
        key = self._key(path, method)
        return key in self.entries

    def get_warning(self, path: str, method: str) -> Optional[DeprecationWarning]:
        """Get deprecation warning for endpoint."""
        key = self._key(path, method)
        return self.entries.get(key)

    def record_usage(self, path: str, method: str) -> None:
        """Record usage of deprecated endpoint."""
        key = self._key(path, method)
        self.usage_counts[key] = self.usage_counts.get(key, 0) + 1

    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for deprecated endpoints."""
        return dict(self.usage_counts)

    def get_all_deprecated(self) -> List[DeprecationWarning]:
        """Get all deprecated endpoints."""
        return list(self.entries.values())

    def get_critical_deprecations(self) -> List[DeprecationWarning]:
        """Get endpoints near sunset."""
        return [
            w for w in self.entries.values()
            if w.level in (DeprecationLevel.CRITICAL, DeprecationLevel.SUNSET)
        ]


# Global registry
_registry = DeprecationRegistry()


def get_deprecation_registry() -> DeprecationRegistry:
    """Get global deprecation registry."""
    return _registry


# ============================================================================
# Deprecation Middleware
# ============================================================================


@dataclass
class DeprecationMiddleware:
    """
    Middleware for handling deprecated endpoints.

    Adds deprecation headers and logs usage.
    """

    registry: DeprecationRegistry = field(default_factory=get_deprecation_registry)
    log_usage: bool = True
    add_headers: bool = True
    block_sunset: bool = False  # Block requests to sunset endpoints

    def process_request(
        self,
        path: str,
        method: str,
        headers: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """
        Process request for deprecation.

        Returns error response if sunset endpoint should be blocked.
        """
        warning = self.registry.get_warning(path, method)

        if not warning:
            return None

        # Record usage
        if self.log_usage:
            self.registry.record_usage(path, method)
            logger.warning(
                f"Deprecated endpoint accessed: {method} {path}",
                extra={
                    "deprecation_level": warning.level.value,
                    "sunset_date": warning.sunset_date.isoformat() if warning.sunset_date else None,
                    "replacement": warning.replacement,
                },
            )

        # Block sunset endpoints
        if self.block_sunset and warning.level == DeprecationLevel.SUNSET:
            return {
                "error": "endpoint_sunset",
                "message": f"This endpoint was removed on {warning.sunset_date}",
                "replacement": warning.replacement,
                "status": 410,  # Gone
            }

        return None

    def add_response_headers(
        self,
        path: str,
        method: str,
        headers: Dict[str, str],
    ) -> Dict[str, str]:
        """Add deprecation headers to response."""
        if not self.add_headers:
            return headers

        warning = self.registry.get_warning(path, method)
        if not warning:
            return headers

        # RFC 8594 Deprecation header
        headers["Deprecation"] = warning.to_header_value()

        # Sunset header
        if warning.sunset_date:
            headers["Sunset"] = warning.sunset_date.isoformat()

        # Link to replacement
        if warning.replacement:
            headers["Link"] = f'<{warning.replacement}>; rel="successor-version"'

        # Custom deprecation info header
        headers["X-Deprecation-Level"] = warning.level.value
        if warning.migration_guide:
            headers["X-Migration-Guide"] = warning.migration_guide

        return headers


# ============================================================================
# Decorators
# ============================================================================


def deprecated(
    sunset_date: Optional[date] = None,
    replacement: Optional[str] = None,
    message: Optional[str] = None,
    migration_guide: Optional[str] = None,
) -> Callable:
    """
    Mark a handler as deprecated.

    Args:
        sunset_date: When this endpoint will be removed
        replacement: URL of replacement endpoint
        message: Custom deprecation message
        migration_guide: URL to migration documentation

    Example:
        @deprecated(
            sunset_date=date(2025, 6, 1),
            replacement="/api/v2/users",
        )
        def get_users(request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        # Store deprecation info on function
        func._deprecated = True
        func._sunset_date = sunset_date
        func._replacement = replacement
        func._deprecation_message = message

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log at call time
            logger.warning(
                f"Deprecated function called: {func.__name__}",
                extra={
                    "function": func.__name__,
                    "sunset_date": sunset_date.isoformat() if sunset_date else None,
                    "replacement": replacement,
                },
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def sunset_date(d: date) -> Callable:
    """
    Set sunset date for a deprecated handler.

    Shorthand for @deprecated(sunset_date=date(...))

    Example:
        @sunset_date(date(2025, 6, 1))
        def old_handler(request):
            ...
    """
    return deprecated(sunset_date=d)


# ============================================================================
# Metrics Integration
# ============================================================================


def get_deprecation_metrics() -> Dict[str, Any]:
    """Get metrics for deprecated endpoint usage."""
    registry = get_deprecation_registry()

    all_deprecated = registry.get_all_deprecated()
    critical = registry.get_critical_deprecations()
    usage = registry.get_usage_stats()

    return {
        "total_deprecated_endpoints": len(all_deprecated),
        "critical_endpoints": len(critical),
        "sunset_endpoints": len([w for w in all_deprecated if w.level == DeprecationLevel.SUNSET]),
        "total_deprecated_calls": sum(usage.values()),
        "endpoints": [
            {
                "path": w.path,
                "method": w.method,
                "level": w.level.value,
                "sunset_date": w.sunset_date.isoformat() if w.sunset_date else None,
                "calls": usage.get(f"{w.method}:{w.path}", 0),
            }
            for w in all_deprecated
        ],
    }


# ============================================================================
# Startup Registration
# ============================================================================


def register_deprecations(deprecations: List[Dict[str, Any]]) -> None:
    """
    Register multiple deprecations at startup.

    Args:
        deprecations: List of deprecation configs with path, method, sunset_date, etc.
    """
    registry = get_deprecation_registry()

    for dep in deprecations:
        sunset = dep.get("sunset_date")
        if isinstance(sunset, str):
            sunset = date.fromisoformat(sunset)

        registry.register(
            path=dep["path"],
            method=dep.get("method", "GET"),
            sunset_date=sunset,
            replacement=dep.get("replacement"),
            message=dep.get("message"),
            migration_guide=dep.get("migration_guide"),
        )

    logger.info(f"Registered {len(deprecations)} deprecated endpoints")
