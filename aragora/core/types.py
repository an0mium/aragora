"""
Core shared types for Aragora.

This module provides canonical definitions for commonly used types across the codebase.
Import these types from here to avoid duplication and ensure consistency.

Usage:
    from aragora.core.types import HealthLevel, ValidationResult, SyncResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# =============================================================================
# Health Types
# =============================================================================


class HealthLevel(str, Enum):
    """Health level for a component.

    This is the canonical health level enum. Use this instead of defining
    local HealthStatus enums in your module.

    Values:
        HEALTHY: Component is fully operational
        DEGRADED: Component is operational but with reduced performance or capacity
        UNHEALTHY: Component is not operational
        UNKNOWN: Health status cannot be determined
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

    def is_operational(self) -> bool:
        """Check if the component is operational (healthy or degraded)."""
        return self in (HealthLevel.HEALTHY, HealthLevel.DEGRADED)


@dataclass
class HealthReport:
    """Detailed health report for a component.

    This is the canonical health report structure. For simple health checks,
    use HealthLevel directly. For detailed health monitoring with metrics,
    use this class.

    Attributes:
        level: Health level (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)
        component: Name of the component being monitored
        last_check: Timestamp of last health check
        consecutive_failures: Number of consecutive failures
        last_error: Last error message (if any)
        latency_ms: Average latency in milliseconds
        metadata: Additional health metadata
    """

    level: HealthLevel
    component: str = ""
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    consecutive_failures: int = 0
    last_error: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        """Check if the component is healthy."""
        return self.level == HealthLevel.HEALTHY

    @property
    def is_operational(self) -> bool:
        """Check if the component is operational."""
        return self.level.is_operational()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "level": self.level.value,
            "component": self.component,
            "healthy": self.healthy,
            "last_check": self.last_check.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


# =============================================================================
# Validation Types
# =============================================================================


@dataclass
class ValidationResult:
    """Result of a validation operation.

    This is the canonical validation result structure. Use this for JSON/schema
    validation, request validation, and general data validation.

    Attributes:
        is_valid: Whether validation passed
        error: Error message if validation failed
        data: Validated/transformed data (optional)
        errors: List of individual errors (for multi-field validation)
        warnings: List of warnings (validation passed but with concerns)
    """

    is_valid: bool
    error: str | None = None
    data: Any = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def success(cls, data: Any = None) -> ValidationResult:
        """Create a successful validation result."""
        return cls(is_valid=True, data=data)

    @classmethod
    def failure(cls, error: str, errors: list[str] | None = None) -> ValidationResult:
        """Create a failed validation result."""
        return cls(is_valid=False, error=error, errors=errors or [])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "error": self.error,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# =============================================================================
# Sync Types
# =============================================================================


@dataclass
class SyncResult:
    """Result of a synchronization operation.

    This is the canonical sync result structure. Use this for tracking
    the outcome of data synchronization, migration, or batch operations.

    Attributes:
        records_synced: Number of records successfully synced
        records_skipped: Number of records skipped
        records_failed: Number of records that failed
        errors: List of error messages from failed records
        duration_ms: Duration of the sync operation in milliseconds
        metadata: Additional sync metadata
    """

    records_synced: int = 0
    records_skipped: int = 0
    records_failed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_processed(self) -> int:
        """Total number of records processed."""
        return self.records_synced + self.records_skipped + self.records_failed

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage (0-100)."""
        total = self.total_processed
        if total == 0:
            return 100.0
        return (self.records_synced / total) * 100

    @property
    def has_errors(self) -> bool:
        """Check if the sync had any errors."""
        return self.records_failed > 0 or len(self.errors) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "records_synced": self.records_synced,
            "records_skipped": self.records_skipped,
            "records_failed": self.records_failed,
            "total_processed": self.total_processed,
            "success_rate": self.success_rate,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# These aliases help with gradual migration from local definitions
HealthStatus = HealthLevel  # Many modules use HealthStatus as the enum name


__all__ = [
    # Health types
    "HealthLevel",
    "HealthReport",
    "HealthStatus",  # Alias for backward compatibility
    # Validation types
    "ValidationResult",
    # Sync types
    "SyncResult",
]
