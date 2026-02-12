"""Callback and result type definitions.

Provides type aliases for common callback patterns and a generic
Result type for operations that can fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
AgentT = TypeVar("AgentT")  # Bound to AgentProtocol at usage sites
MemoryT = TypeVar("MemoryT")  # Bound to MemoryProtocol at usage sites

# =============================================================================
# Callback Types
# =============================================================================

# Event callback type
EventCallback = Callable[[str, dict[str, Any]], None]

# Async event callback type
AsyncEventCallback = Callable[[str, dict[str, Any]], Any]

# Response filter type
ResponseFilter = Callable[[str], str]

# Vote callback type
VoteCallback = Callable[[Any], None]

# =============================================================================
# Result Types
# =============================================================================


@dataclass
class Result(Generic[T]):
    """Generic result type for operations that can fail.

    Example:
        def get_user(id: str) -> Result[User]:
            user = db.get(id)
            if user:
                return Result(success=True, value=user)
            return Result(success=False, error="User not found")
    """

    success: bool
    value: T | None = None
    error: str | None = None

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        """Create successful result."""
        return cls(success=True, value=value)

    @classmethod
    def fail(cls, error: str) -> "Result[T]":
        """Create failed result."""
        return cls(success=False, error=error)


__all__ = [
    "T",
    "AgentT",
    "MemoryT",
    "EventCallback",
    "AsyncEventCallback",
    "ResponseFilter",
    "VoteCallback",
    "Result",
]
