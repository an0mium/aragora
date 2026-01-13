"""
API Deprecation utilities for Aragora.

Provides decorators and utilities for marking endpoints, functions, and
features as deprecated with sunset dates and migration guidance.

Usage:
    from aragora.server.deprecation import deprecated, add_deprecation_headers

    @deprecated(sunset="2026-06-01", replacement="/api/v2/debates")
    def old_endpoint():
        ...

    # Add headers manually
    headers = add_deprecation_headers({}, sunset="2026-06-01")

See docs/DEPRECATION_POLICY.md for the full deprecation policy.
"""

from __future__ import annotations

import functools
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Deprecation Registry
# =============================================================================


@dataclass
class DeprecationInfo:
    """Information about a deprecated item."""

    name: str
    sunset: Optional[str] = None  # ISO date format
    replacement: Optional[str] = None
    message: Optional[str] = None
    deprecated_since: Optional[str] = None
    category: str = "api"  # api, function, feature

    def format_warning(self) -> str:
        """Format deprecation warning message."""
        parts = [f"{self.name} is deprecated."]
        if self.replacement:
            parts.append(f"Use {self.replacement} instead.")
        if self.sunset:
            parts.append(f"Will be removed after {self.sunset}.")
        if self.message:
            parts.append(self.message)
        return " ".join(parts)


class DeprecationRegistry:
    """Registry of deprecated items for tracking and reporting.

    Singleton pattern for global deprecation tracking.
    """

    _instance: Optional["DeprecationRegistry"] = None
    _deprecations: Dict[str, DeprecationInfo]
    _call_counts: Dict[str, int]

    def __new__(cls) -> "DeprecationRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._deprecations = {}
            cls._instance._call_counts = {}
        return cls._instance

    def register(self, info: DeprecationInfo) -> None:
        """Register a deprecation."""
        self._deprecations[info.name] = info
        self._call_counts[info.name] = 0

    def record_call(self, name: str) -> None:
        """Record a call to a deprecated item."""
        if name in self._call_counts:
            self._call_counts[name] += 1

    def get_all(self) -> List[DeprecationInfo]:
        """Get all registered deprecations."""
        return list(self._deprecations.values())

    def get_by_category(self, category: str) -> List[DeprecationInfo]:
        """Get deprecations by category."""
        return [d for d in self._deprecations.values() if d.category == category]

    def get_call_count(self, name: str) -> int:
        """Get call count for a deprecated item."""
        return self._call_counts.get(name, 0)

    def get_report(self) -> Dict[str, Any]:
        """Get a report of all deprecations and their usage."""
        return {
            "total_deprecations": len(self._deprecations),
            "deprecations": [
                {
                    "name": info.name,
                    "sunset": info.sunset,
                    "replacement": info.replacement,
                    "category": info.category,
                    "call_count": self._call_counts.get(info.name, 0),
                }
                for info in self._deprecations.values()
            ],
        }

    def clear(self) -> None:
        """Clear all registrations (for testing)."""
        self._deprecations.clear()
        self._call_counts.clear()


# Global registry instance
_registry: Optional[DeprecationRegistry] = None


def get_deprecation_registry() -> DeprecationRegistry:
    """Get the global deprecation registry."""
    global _registry
    if _registry is None:
        _registry = DeprecationRegistry()
    return _registry


# =============================================================================
# Deprecation Decorator
# =============================================================================


def deprecated(
    sunset: Optional[str] = None,
    replacement: Optional[str] = None,
    message: Optional[str] = None,
    deprecated_since: Optional[str] = None,
    category: str = "function",
    warn_once: bool = True,
) -> Callable[[F], F]:
    """
    Mark a function or method as deprecated.

    Args:
        sunset: Date when the item will be removed (ISO format: YYYY-MM-DD)
        replacement: Suggested replacement (e.g., "/api/v2/debates")
        message: Additional deprecation message
        deprecated_since: Version when deprecated
        category: Category (api, function, feature)
        warn_once: Only warn once per function (default True)

    Returns:
        Decorated function that emits deprecation warning

    Example:
        @deprecated(sunset="2026-06-01", replacement="new_function")
        def old_function():
            ...

        @deprecated(
            sunset="2026-06-01",
            replacement="/api/v2/debates",
            category="api",
            message="The v1 debates endpoint uses a different response format."
        )
        def _handle_v1_debates(self, handler):
            ...
    """
    warned: set = set()

    def decorator(func: F) -> F:
        # Register deprecation
        info = DeprecationInfo(
            name=func.__qualname__,
            sunset=sunset,
            replacement=replacement,
            message=message,
            deprecated_since=deprecated_since,
            category=category,
        )
        get_deprecation_registry().register(info)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Record the call
            registry = get_deprecation_registry()
            registry.record_call(func.__qualname__)

            # Emit warning
            warning_msg = info.format_warning()
            func_id = id(func)

            if not warn_once or func_id not in warned:
                logger.warning(f"deprecated_call: {warning_msg}")
                warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
                if warn_once:
                    warned.add(func_id)

            return func(*args, **kwargs)

        # Store deprecation info on the wrapper
        wrapper._deprecated = True  # type: ignore[attr-defined]
        wrapper._deprecation_info = info  # type: ignore[attr-defined]
        wrapper._sunset = sunset  # type: ignore[attr-defined]
        wrapper._replacement = replacement  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# HTTP Header Utilities
# =============================================================================


def add_deprecation_headers(
    headers: Dict[str, str],
    sunset: Optional[str] = None,
    replacement: Optional[str] = None,
    deprecated_since: Optional[str] = None,
) -> Dict[str, str]:
    """
    Add deprecation headers to HTTP response.

    Follows RFC 8594 (Sunset Header) and draft-ietf-httpapi-deprecation-header.

    Args:
        headers: Existing headers dict (modified in place)
        sunset: Sunset date (ISO format or HTTP date)
        replacement: Replacement URL or endpoint
        deprecated_since: When deprecation started

    Returns:
        Updated headers dict

    Headers added:
        - Deprecation: true (or date)
        - Sunset: <date> (if provided)
        - Link: <replacement>; rel="successor-version" (if provided)

    Example:
        headers = add_deprecation_headers(
            {},
            sunset="2026-06-01",
            replacement="/api/v2/debates"
        )
        # Returns: {
        #   "Deprecation": "true",
        #   "Sunset": "Sat, 01 Jun 2026 00:00:00 GMT",
        #   "Link": "</api/v2/debates>; rel=\"successor-version\""
        # }
    """
    # Deprecation header
    if deprecated_since:
        headers["Deprecation"] = deprecated_since
    else:
        headers["Deprecation"] = "true"

    # Sunset header (convert ISO date to HTTP date format)
    if sunset:
        try:
            dt = datetime.fromisoformat(sunset)
            http_date = dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
            headers["Sunset"] = http_date
        except ValueError:
            # Already in HTTP date format or invalid
            headers["Sunset"] = sunset

    # Link header for replacement
    if replacement:
        # Ensure replacement URL is properly formatted
        if not replacement.startswith("<"):
            replacement = f"<{replacement}>"
        headers["Link"] = f'{replacement}; rel="successor-version"'

    return headers


def parse_deprecation_headers(headers: Dict[str, str]) -> Optional[DeprecationInfo]:
    """
    Parse deprecation info from HTTP response headers.

    Args:
        headers: Response headers dict

    Returns:
        DeprecationInfo if deprecated, None otherwise

    Example:
        info = parse_deprecation_headers(response.headers)
        if info:
            print(f"Warning: {info.format_warning()}")
    """
    deprecation = headers.get("Deprecation")
    if not deprecation:
        return None

    sunset = headers.get("Sunset")
    link = headers.get("Link")

    replacement = None
    if link and 'rel="successor-version"' in link:
        # Extract URL from Link header
        import re

        match = re.search(r"<([^>]+)>", link)
        if match:
            replacement = match.group(1)

    return DeprecationInfo(
        name="endpoint",
        sunset=sunset,
        replacement=replacement,
        deprecated_since=deprecation if deprecation != "true" else None,
    )


# =============================================================================
# Deprecation Checking Utilities
# =============================================================================


def is_deprecated(func: Callable[..., Any]) -> bool:
    """Check if a function is marked as deprecated."""
    return getattr(func, "_deprecated", False)


def get_deprecation_info(func: Callable[..., Any]) -> Optional[DeprecationInfo]:
    """Get deprecation info for a function."""
    return getattr(func, "_deprecation_info", None)


def is_past_sunset(sunset: Optional[str]) -> bool:
    """Check if a sunset date has passed."""
    if not sunset:
        return False
    try:
        sunset_dt = datetime.fromisoformat(sunset)
        return datetime.now() > sunset_dt
    except ValueError:
        return False


def days_until_sunset(sunset: Optional[str]) -> Optional[int]:
    """Calculate days until sunset date."""
    if not sunset:
        return None
    try:
        sunset_dt = datetime.fromisoformat(sunset)
        delta = sunset_dt - datetime.now()
        return max(0, delta.days)
    except ValueError:
        return None


# =============================================================================
# Sunset Response
# =============================================================================


def sunset_response(
    replacement: Optional[str] = None,
    message: str = "This endpoint has been removed.",
) -> Dict[str, Any]:
    """
    Create a standard 410 Gone response for sunset endpoints.

    Returns a dict suitable for json_response().

    Example:
        @deprecated(sunset="2026-01-01")
        def old_endpoint():
            if is_past_sunset("2026-01-01"):
                return error_response(**sunset_response("/api/v2/endpoint"))
            ...
    """
    response = {
        "error": {
            "code": "ENDPOINT_SUNSET",
            "message": message,
        }
    }
    if replacement:
        response["error"]["replacement"] = replacement
        response["error"]["suggestion"] = f"Use {replacement} instead"
    return response


__all__ = [
    # Decorator
    "deprecated",
    # Header utilities
    "add_deprecation_headers",
    "parse_deprecation_headers",
    # Data classes
    "DeprecationInfo",
    "DeprecationRegistry",
    # Registry
    "get_deprecation_registry",
    # Utilities
    "is_deprecated",
    "get_deprecation_info",
    "is_past_sunset",
    "days_until_sunset",
    # Response helpers
    "sunset_response",
]
