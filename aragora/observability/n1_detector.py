"""
N+1 Query Detection Utility.

Provides runtime detection of N+1 query patterns to identify
performance issues before they reach production.

Usage:
    from aragora.observability.n1_detector import N1QueryDetector, detect_n1

    # As context manager
    with N1QueryDetector(threshold=5) as detector:
        for item in items:
            db.query(f"SELECT * FROM table WHERE id = {item.id}")
    # Warning: "N+1 detected: table queried 10 times in 0.5s"

    # As decorator
    @detect_n1(threshold=3)
    async def handler(request):
        ...

Configuration:
    ARAGORA_N1_DETECTION: warn|error|off (default: off)
    ARAGORA_N1_THRESHOLD: int (default: 5)
"""

from __future__ import annotations

import contextvars
import logging
import os
import re
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)


# Environment configuration
def _parse_detection_mode(raw: str) -> str:
    value = raw.lower()
    if value not in {"off", "warn", "error"}:
        logger.warning("Invalid ARAGORA_N1_DETECTION='%s', using 'off'", raw)
        return "off"
    return value


def _parse_threshold(raw: str) -> int:
    try:
        value = int(raw)
        if value < 1:
            logger.warning("ARAGORA_N1_THRESHOLD=%s too low, using 1", value)
            return 1
        return value
    except ValueError:
        logger.warning("Invalid ARAGORA_N1_THRESHOLD='%s', using default 5", raw)
        return 5


N1_DETECTION_MODE = _parse_detection_mode(os.environ.get("ARAGORA_N1_DETECTION", "off"))
N1_THRESHOLD = _parse_threshold(os.environ.get("ARAGORA_N1_THRESHOLD", "5"))

# Context variable for tracking queries across async boundaries
_current_detector: contextvars.ContextVar[Optional["N1QueryDetector"]] = contextvars.ContextVar(
    "n1_detector", default=None
)


class N1QueryError(Exception):
    """Raised when N+1 pattern detected and mode is 'error'."""

    def __init__(self, table: str, count: int, threshold: int):
        self.table = table
        self.count = count
        self.threshold = threshold
        super().__init__(
            f"N+1 query pattern detected: {table} queried {count} times (threshold: {threshold})"
        )


@dataclass
class QueryRecord:
    """Record of a single query execution."""

    table: str
    query_pattern: str  # Normalized query pattern
    timestamp: float
    duration_ms: float = 0.0


@dataclass
class N1Detection:
    """Detection result for a single table."""

    table: str
    query_count: int
    unique_patterns: int
    total_duration_ms: float
    queries: list[QueryRecord] = field(default_factory=list)

    @property
    def is_likely_n1(self) -> bool:
        """Check if this is likely an N+1 pattern (many similar queries)."""
        # N+1 typically has many queries with few unique patterns
        return self.query_count > 1 and self.unique_patterns <= 2


class N1QueryDetector:
    """
    Context manager for detecting N+1 query patterns.

    Tracks queries executed within a code block and warns/errors
    if the same table is queried too many times.

    Example:
        with N1QueryDetector(threshold=5) as detector:
            # Your database operations
            ...
        # Automatically checks for N+1 patterns on exit
    """

    def __init__(
        self,
        threshold: int | None = None,
        mode: str | None = None,
        name: str = "unnamed",
        include_patterns: bool = False,
    ):
        """
        Initialize the detector.

        Args:
            threshold: Number of queries to same table before warning (default from env)
            mode: Detection mode: 'warn', 'error', or 'off' (default from env)
            name: Name for this detection scope (for logging)
            include_patterns: Include query patterns in detection results
        """
        self.threshold = threshold or N1_THRESHOLD
        self.mode = mode or N1_DETECTION_MODE
        self.name = name
        self.include_patterns = include_patterns

        self.queries: list[QueryRecord] = []
        self.start_time: float = 0.0
        self._token: Optional[contextvars.Token] = None

    def record_query(
        self,
        table: str,
        query: str,
        duration_ms: float = 0.0,
    ) -> None:
        """
        Record a query execution.

        Called by database utilities to track queries.

        Args:
            table: Table name being queried
            query: The SQL query (will be normalized)
            duration_ms: Query execution time in milliseconds
        """
        if self.mode == "off":
            return

        # Normalize query to detect similar patterns
        pattern = self._normalize_query(query)

        self.queries.append(
            QueryRecord(
                table=table,
                query_pattern=pattern,
                timestamp=time.time(),
                duration_ms=duration_ms,
            )
        )

    def _normalize_query(self, query: str) -> str:
        """
        Normalize a query to identify similar patterns.

        Replaces literal values with placeholders to group similar queries.
        """
        # Remove extra whitespace
        normalized = " ".join(query.split())

        # Replace string literals
        normalized = re.sub(r"'[^']*'", "'?'", normalized)

        # Replace numeric literals
        normalized = re.sub(r"\b\d+\b", "?", normalized)

        # Replace UUIDs
        normalized = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "?",
            normalized,
            flags=re.IGNORECASE,
        )

        # Replace IN lists
        normalized = re.sub(r"IN\s*\([^)]+\)", "IN (?)", normalized, flags=re.IGNORECASE)

        return normalized

    def analyze(self) -> dict[str, N1Detection]:
        """
        Analyze recorded queries for N+1 patterns.

        Returns:
            Dict mapping table names to detection results
        """
        if not self.queries:
            return {}

        # Group by table
        by_table: dict[str, list[QueryRecord]] = {}
        for record in self.queries:
            if record.table not in by_table:
                by_table[record.table] = []
            by_table[record.table].append(record)

        # Build detection results
        results: dict[str, N1Detection] = {}
        for table, records in by_table.items():
            patterns = Counter(r.query_pattern for r in records)
            total_duration = sum(r.duration_ms for r in records)

            results[table] = N1Detection(
                table=table,
                query_count=len(records),
                unique_patterns=len(patterns),
                total_duration_ms=total_duration,
                queries=records if self.include_patterns else [],
            )

        return results

    def get_violations(self) -> list[N1Detection]:
        """Get all tables that exceed the threshold."""
        results = self.analyze()
        return [d for d in results.values() if d.query_count >= self.threshold]

    def __enter__(self) -> "N1QueryDetector":
        """Enter the detection context."""
        self.start_time = time.time()
        self._token = _current_detector.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the detection context and check for violations."""
        if self._token is not None:
            _current_detector.reset(self._token)
            self._token = None

        if self.mode == "off":
            return False

        elapsed_ms = (time.time() - self.start_time) * 1000
        violations = self.get_violations()

        for violation in violations:
            msg = (
                f"N+1 detected in '{self.name}': {violation.table} queried "
                f"{violation.query_count} times in {elapsed_ms:.1f}ms "
                f"(threshold: {self.threshold}, patterns: {violation.unique_patterns})"
            )

            if self.mode == "error":
                raise N1QueryError(violation.table, violation.query_count, self.threshold)
            elif self.mode == "warn":
                logger.warning(msg)

        return False


def get_current_detector() -> Optional[N1QueryDetector]:
    """Get the current N+1 detector from context, if any."""
    return _current_detector.get()


def record_query(table: str, query: str, duration_ms: float = 0.0) -> None:
    """
    Record a query to the current detector, if one is active.

    Called by database utilities to track queries.
    """
    detector = get_current_detector()
    if detector is not None:
        detector.record_query(table, query, duration_ms)


# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def detect_n1(
    threshold: int | None = None,
    mode: str | None = None,
    name: str | None = None,
) -> Callable[[F], F]:
    """
    Decorator to detect N+1 query patterns in a function.

    Example:
        @detect_n1(threshold=3)
        async def get_users_with_posts(user_ids: list[str]):
            ...

        @detect_n1(mode="error")  # Raise exception on detection
        def handler(request):
            ...
    """

    def decorator(func: F) -> F:
        func_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with N1QueryDetector(threshold=threshold, mode=mode, name=func_name):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with N1QueryDetector(threshold=threshold, mode=mode, name=func_name):
                return func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def n1_detection_scope(
    name: str = "scope",
    threshold: int | None = None,
    mode: str | None = None,
):
    """
    Context manager for N+1 detection.

    Example:
        with n1_detection_scope("user_fetch", threshold=5):
            for user_id in user_ids:
                db.get_user(user_id)
    """
    with N1QueryDetector(threshold=threshold, mode=mode, name=name) as detector:
        yield detector


__all__ = [
    "N1QueryDetector",
    "N1QueryError",
    "N1Detection",
    "QueryRecord",
    "detect_n1",
    "get_current_detector",
    "record_query",
    "n1_detection_scope",
]
