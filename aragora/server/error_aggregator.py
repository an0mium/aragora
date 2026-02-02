"""
Error Aggregation and Deduplication Service.

Provides local error aggregation and deduplication that works alongside Sentry:
- Groups similar errors by signature (type + message pattern + location)
- Tracks error counts and rates over time windows
- Deduplicates high-frequency errors to prevent alert fatigue
- Enables local error analysis without external dependencies

This module complements error_monitoring.py (Sentry) by providing:
- Local error visibility for self-hosted deployments
- Rate limiting for high-frequency errors before Sentry
- Error budget tracking per component/handler
- Trend detection for error spikes

Usage:
    from aragora.server.error_aggregator import (
        ErrorAggregator,
        get_error_aggregator,
        record_error,
    )

    # Record an error
    record_error(
        error=exc,
        component="debate.orchestrator",
        context={"debate_id": "d-123"},
    )

    # Get aggregated errors
    agg = get_error_aggregator()
    recent = agg.get_recent_errors(minutes=15)
    top_errors = agg.get_top_errors(limit=10)
    stats = agg.get_stats()
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aragora.config.env_helpers import env_int

logger = logging.getLogger(__name__)

# Configuration (using safe int parsing)
DEFAULT_WINDOW_MINUTES = env_int("ARAGORA_ERROR_WINDOW_MINUTES", 60)
DEFAULT_MAX_UNIQUE_ERRORS = env_int("ARAGORA_ERROR_MAX_UNIQUE", 1000)
DEFAULT_DEDUP_WINDOW_SECONDS = env_int("ARAGORA_ERROR_DEDUP_SECONDS", 60)


def _normalize_message(message: str) -> str:
    """Normalize error message for grouping.

    Replaces variable parts (IDs, numbers, paths) with placeholders.
    """
    # Replace UUIDs
    message = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "<UUID>",
        message,
        flags=re.IGNORECASE,
    )
    # Replace hex IDs (8+ chars)
    message = re.sub(r"\b[0-9a-f]{8,}\b", "<HEX_ID>", message, flags=re.IGNORECASE)
    # Replace numbers
    message = re.sub(r"\b\d+\b", "<N>", message)
    # Replace file paths
    message = re.sub(r"(/[^\s:]+)+", "<PATH>", message)
    # Replace quoted strings
    message = re.sub(r"['\"][^'\"]+['\"]", "<STR>", message)
    # Normalize whitespace
    message = re.sub(r"\s+", " ", message).strip()
    return message


def _extract_location(tb: str | None) -> str:
    """Extract the error location from traceback."""
    if not tb:
        return "unknown"
    # Get the last file:line from traceback
    lines = tb.strip().split("\n")
    for line in reversed(lines):
        if 'File "' in line:
            match = re.search(r'File "([^"]+)", line (\d+)', line)
            if match:
                filepath, lineno = match.groups()
                # Simplify path
                if "aragora" in filepath:
                    filepath = filepath.split("aragora", 1)[1]
                return f"aragora{filepath}:{lineno}"
    return "unknown"


@dataclass
class ErrorSignature:
    """Unique signature for grouping similar errors."""

    error_type: str
    normalized_message: str
    location: str
    component: str

    def __hash__(self) -> int:
        return hash((self.error_type, self.normalized_message, self.location, self.component))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ErrorSignature):
            return False
        return (
            self.error_type == other.error_type
            and self.normalized_message == other.normalized_message
            and self.location == other.location
            and self.component == other.component
        )

    @property
    def fingerprint(self) -> str:
        """Generate a short fingerprint for this signature."""
        data = f"{self.error_type}:{self.normalized_message}:{self.location}:{self.component}"
        return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()[:12]


@dataclass
class ErrorOccurrence:
    """A single occurrence of an error."""

    timestamp: float
    message: str
    context: dict[str, Any]
    trace_id: str | None = None


@dataclass
class AggregatedError:
    """An aggregated error with occurrence counts."""

    signature: ErrorSignature
    first_seen: float
    last_seen: float
    count: int
    occurrences: deque  # deque[ErrorOccurrence] - recent samples
    contexts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "fingerprint": self.signature.fingerprint,
            "error_type": self.signature.error_type,
            "message_pattern": self.signature.normalized_message,
            "location": self.signature.location,
            "component": self.signature.component,
            "first_seen": datetime.fromtimestamp(self.first_seen).isoformat(),
            "last_seen": datetime.fromtimestamp(self.last_seen).isoformat(),
            "count": self.count,
            "rate_per_minute": self._calculate_rate(),
            "sample_contexts": dict(self.contexts),
            "recent_occurrences": [
                {
                    "timestamp": datetime.fromtimestamp(o.timestamp).isoformat(),
                    "message": o.message[:200],
                    "trace_id": o.trace_id,
                }
                for o in list(self.occurrences)[-5:]
            ],
        }

    def _calculate_rate(self) -> float:
        """Calculate errors per minute."""
        if self.count <= 1:
            return 0.0
        duration = self.last_seen - self.first_seen
        if duration < 1:
            return float(self.count)
        return self.count / (duration / 60.0)


@dataclass
class ErrorAggregatorStats:
    """Statistics for the error aggregator."""

    unique_errors: int = 0
    total_occurrences: int = 0
    errors_last_minute: int = 0
    errors_last_5_minutes: int = 0
    errors_last_hour: int = 0
    top_components: dict[str, int] = field(default_factory=dict)
    top_error_types: dict[str, int] = field(default_factory=dict)
    dedup_ratio: float = 0.0  # How much deduplication saved

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "unique_errors": self.unique_errors,
            "total_occurrences": self.total_occurrences,
            "errors_last_minute": self.errors_last_minute,
            "errors_last_5_minutes": self.errors_last_5_minutes,
            "errors_last_hour": self.errors_last_hour,
            "top_components": self.top_components,
            "top_error_types": self.top_error_types,
            "dedup_ratio": self.dedup_ratio,
        }


class ErrorAggregator:
    """Aggregates and deduplicates errors for local analysis.

    Tracks errors in time windows, groups by signature, and provides
    statistics for monitoring and alerting.
    """

    def __init__(
        self,
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
        max_unique_errors: int = DEFAULT_MAX_UNIQUE_ERRORS,
        dedup_window_seconds: int = DEFAULT_DEDUP_WINDOW_SECONDS,
        max_samples_per_error: int = 10,
    ):
        """Initialize the error aggregator.

        Args:
            window_minutes: How long to keep error data (default: 60)
            max_unique_errors: Maximum unique error signatures to track
            dedup_window_seconds: Window for deduplication (same error)
            max_samples_per_error: Max recent occurrences to keep per error
        """
        self._window_minutes = window_minutes
        self._max_unique_errors = max_unique_errors
        self._dedup_window_seconds = dedup_window_seconds
        self._max_samples = max_samples_per_error

        self._errors: dict[ErrorSignature, AggregatedError] = {}
        self._timeline: deque = deque(
            maxlen=100000
        )  # (timestamp, signature) for time-based queries
        self._lock = threading.RLock()

        # Stats tracking
        self._total_recorded = 0
        self._total_deduplicated = 0

    def record(
        self,
        error: Exception | str,
        component: str = "unknown",
        context: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> tuple[ErrorSignature, bool]:
        """Record an error occurrence.

        Args:
            error: The exception or error message
            component: Component/module where error occurred
            context: Additional context (debate_id, user_id, etc.)
            trace_id: Distributed trace ID

        Returns:
            Tuple of (signature, is_new) - signature and whether this is a new error
        """
        now = time.time()

        # Extract error info
        if isinstance(error, Exception):
            error_type = type(error).__name__
            message = str(error)
            tb = traceback.format_exc() if error.__traceback__ else None
        else:
            error_type = "Error"
            message = str(error)
            tb = None

        # Create signature
        location = _extract_location(tb)
        normalized = _normalize_message(message)
        signature = ErrorSignature(
            error_type=error_type,
            normalized_message=normalized[:200],  # Limit length
            location=location,
            component=component,
        )

        occurrence = ErrorOccurrence(
            timestamp=now,
            message=message[:500],  # Limit length
            context=context or {},
            trace_id=trace_id,
        )

        with self._lock:
            self._total_recorded += 1

            # Cleanup old entries first
            self._cleanup_old_entries(now)

            # Check for deduplication
            is_new = signature not in self._errors
            if not is_new:
                agg = self._errors[signature]
                # Check if within dedup window
                if now - agg.last_seen < self._dedup_window_seconds:
                    self._total_deduplicated += 1

            # Update or create aggregated error
            if is_new:
                # Check capacity
                if len(self._errors) >= self._max_unique_errors:
                    self._evict_oldest()

                self._errors[signature] = AggregatedError(
                    signature=signature,
                    first_seen=now,
                    last_seen=now,
                    count=1,
                    occurrences=deque([occurrence], maxlen=self._max_samples),
                )
            else:
                agg = self._errors[signature]
                agg.last_seen = now
                agg.count += 1
                agg.occurrences.append(occurrence)

                # Track context values for analysis (cap at 1000 unique keys)
                if context:
                    for key, value in context.items():
                        if isinstance(value, (str, int, bool)):
                            ctx_key = f"{key}:{value}"
                            # Allow incrementing existing keys even at cap
                            if ctx_key in agg.contexts or len(agg.contexts) < 1000:
                                agg.contexts[ctx_key] += 1

            # Update timeline
            self._timeline.append((now, signature))

            return signature, is_new

    def get_error(self, fingerprint: str) -> AggregatedError | None:
        """Get an aggregated error by fingerprint."""
        with self._lock:
            for sig, agg in self._errors.items():
                if sig.fingerprint == fingerprint:
                    return agg
        return None

    def get_recent_errors(
        self, minutes: int = 15, component: str | None = None
    ) -> list[AggregatedError]:
        """Get errors from the last N minutes.

        Args:
            minutes: Time window in minutes
            component: Optional filter by component

        Returns:
            List of aggregated errors, sorted by last_seen desc
        """
        cutoff = time.time() - (minutes * 60)

        with self._lock:
            errors = [
                agg
                for agg in self._errors.values()
                if agg.last_seen >= cutoff
                and (component is None or agg.signature.component == component)
            ]
            return sorted(errors, key=lambda e: e.last_seen, reverse=True)

    def get_top_errors(self, limit: int = 10, minutes: int | None = None) -> list[AggregatedError]:
        """Get top errors by occurrence count.

        Args:
            limit: Maximum errors to return
            minutes: Optional time window (None = all time in window)

        Returns:
            List of aggregated errors, sorted by count desc
        """
        if minutes:
            errors = self.get_recent_errors(minutes)
        else:
            with self._lock:
                errors = list(self._errors.values())

        return sorted(errors, key=lambda e: e.count, reverse=True)[:limit]

    def get_errors_by_component(self, component: str) -> list[AggregatedError]:
        """Get all errors for a specific component."""
        with self._lock:
            return [agg for agg in self._errors.values() if agg.signature.component == component]

    def get_error_rate(self, minutes: int = 5) -> float:
        """Get errors per minute over the specified window."""
        cutoff = time.time() - (minutes * 60)
        count = 0

        with self._lock:
            for ts, _ in reversed(self._timeline):
                if ts < cutoff:
                    break
                count += 1

        return count / minutes if minutes > 0 else 0.0

    def get_stats(self) -> ErrorAggregatorStats:
        """Get aggregator statistics."""
        now = time.time()
        stats = ErrorAggregatorStats()

        with self._lock:
            stats.unique_errors = len(self._errors)
            stats.total_occurrences = sum(e.count for e in self._errors.values())

            # Count by time window
            for ts, sig in reversed(self._timeline):
                age = now - ts
                if age <= 60:
                    stats.errors_last_minute += 1
                if age <= 300:
                    stats.errors_last_5_minutes += 1
                if age <= 3600:
                    stats.errors_last_hour += 1
                if age > 3600:
                    break

            # Top components
            component_counts: dict[str, int] = defaultdict(int)
            type_counts: dict[str, int] = defaultdict(int)
            for agg in self._errors.values():
                component_counts[agg.signature.component] += agg.count
                type_counts[agg.signature.error_type] += agg.count

            stats.top_components = dict(
                sorted(component_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            stats.top_error_types = dict(
                sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )

            # Dedup ratio
            if self._total_recorded > 0:
                stats.dedup_ratio = self._total_deduplicated / self._total_recorded

        return stats

    def clear(self) -> None:
        """Clear all error data."""
        with self._lock:
            self._errors.clear()
            self._timeline.clear()
            self._total_recorded = 0
            self._total_deduplicated = 0

    def _cleanup_old_entries(self, now: float) -> None:
        """Remove entries older than the window."""
        cutoff = now - (self._window_minutes * 60)

        # Remove from timeline
        while self._timeline and self._timeline[0][0] < cutoff:
            self._timeline.popleft()

        # Remove old errors (those not seen recently)
        to_remove = [sig for sig, agg in self._errors.items() if agg.last_seen < cutoff]
        for sig in to_remove:
            del self._errors[sig]

    def _evict_oldest(self) -> None:
        """Evict the oldest error when at capacity."""
        if not self._errors:
            return

        oldest_sig = min(self._errors.keys(), key=lambda s: self._errors[s].last_seen)
        del self._errors[oldest_sig]


# ---------------------------------------------------------------------------
# Global instance and convenience functions
# ---------------------------------------------------------------------------

_aggregator: ErrorAggregator | None = None
_lock = threading.Lock()


def get_error_aggregator() -> ErrorAggregator:
    """Get or create the global error aggregator."""
    global _aggregator
    if _aggregator is None:
        with _lock:
            if _aggregator is None:
                _aggregator = ErrorAggregator()
    return _aggregator


def reset_error_aggregator() -> None:
    """Reset the global error aggregator (for testing)."""
    global _aggregator
    with _lock:
        _aggregator = None


def record_error(
    error: Exception | str,
    component: str = "unknown",
    context: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> ErrorSignature:
    """Record an error in the global aggregator.

    Convenience function for recording errors.

    Args:
        error: The exception or error message
        component: Component/module where error occurred
        context: Additional context (debate_id, user_id, etc.)
        trace_id: Distributed trace ID

    Returns:
        The error signature
    """
    sig, _ = get_error_aggregator().record(error, component, context, trace_id)
    return sig


__all__ = [
    "AggregatedError",
    "ErrorAggregator",
    "ErrorAggregatorStats",
    "ErrorOccurrence",
    "ErrorSignature",
    "get_error_aggregator",
    "record_error",
    "reset_error_aggregator",
]
