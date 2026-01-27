"""
Health Monitoring Utilities for Aragora.

Provides health check infrastructure for services, storage backends,
and other components.

Usage:
    from aragora.resilience_patterns import HealthChecker, HealthStatus

    checker = HealthChecker("database")

    # Manual health recording
    checker.record_success(latency_ms=15.0)
    checker.record_failure("Connection timeout")

    # Get health status
    status = checker.get_status()
    print(f"Healthy: {status.healthy}, Latency: {status.latency_ms}ms")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status for a component.

    Attributes:
        healthy: Whether the component is healthy
        last_check: Timestamp of last health check
        consecutive_failures: Number of consecutive failures
        last_error: Last error message (if any)
        latency_ms: Average latency in milliseconds
        metadata: Additional health metadata
    """

    healthy: bool
    last_check: datetime
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "healthy": self.healthy,
            "last_check": self.last_check.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthStatus":
        """Create from dictionary."""
        return cls(
            healthy=data["healthy"],
            last_check=(
                datetime.fromisoformat(data["last_check"])
                if isinstance(data["last_check"], str)
                else data["last_check"]
            ),
            consecutive_failures=data.get("consecutive_failures", 0),
            last_error=data.get("last_error"),
            latency_ms=data.get("latency_ms"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class HealthReport:
    """Aggregated health report for multiple components.

    Attributes:
        overall_healthy: Whether all components are healthy
        components: Dictionary of component name to HealthStatus
        checked_at: When the report was generated
        summary: Human-readable summary
    """

    overall_healthy: bool
    components: Dict[str, HealthStatus]
    checked_at: datetime
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overall_healthy": self.overall_healthy,
            "components": {name: status.to_dict() for name, status in self.components.items()},
            "checked_at": self.checked_at.isoformat(),
            "summary": self.summary,
        }


class HealthChecker:
    """Health checker for a single component.

    Tracks health status, latency, and consecutive failures.

    Args:
        name: Component name
        failure_threshold: Failures before marking unhealthy
        recovery_threshold: Successes before marking healthy
        latency_window: Number of latency samples to average
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
        latency_window: int = 10,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self.latency_window = latency_window

        self._lock = threading.Lock()
        self._healthy = True
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_error: Optional[str] = None
        self._last_check: datetime = datetime.now(timezone.utc)
        self._latencies: List[float] = []
        self._metadata: Dict[str, Any] = {}

    def record_success(self, latency_ms: Optional[float] = None) -> None:
        """Record a successful health check.

        Args:
            latency_ms: Optional latency in milliseconds
        """
        with self._lock:
            self._last_check = datetime.now(timezone.utc)
            self._consecutive_successes += 1
            self._consecutive_failures = 0

            if latency_ms is not None:
                self._latencies.append(latency_ms)
                if len(self._latencies) > self.latency_window:
                    self._latencies = self._latencies[-self.latency_window :]

            # Check for recovery
            if not self._healthy and self._consecutive_successes >= self.recovery_threshold:
                self._healthy = True
                self._last_error = None
                logger.info(
                    f"[{self.name}] Health recovered after {self._consecutive_successes} successes"
                )

    def record_failure(self, error: Optional[str] = None) -> None:
        """Record a failed health check.

        Args:
            error: Optional error message
        """
        with self._lock:
            self._last_check = datetime.now(timezone.utc)
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._last_error = error

            # Check for degradation
            if self._healthy and self._consecutive_failures >= self.failure_threshold:
                self._healthy = False
                logger.warning(
                    f"[{self.name}] Health degraded after {self._consecutive_failures} failures: {error}"
                )

    def get_status(self) -> HealthStatus:
        """Get current health status."""
        with self._lock:
            avg_latency = None
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)

            return HealthStatus(
                healthy=self._healthy,
                last_check=self._last_check,
                consecutive_failures=self._consecutive_failures,
                last_error=self._last_error,
                latency_ms=avg_latency,
                metadata=self._metadata.copy(),
            )

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        with self._lock:
            self._metadata[key] = value

    def reset(self) -> None:
        """Reset health checker to initial state."""
        with self._lock:
            self._healthy = True
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._last_error = None
            self._last_check = datetime.now(timezone.utc)
            self._latencies.clear()
            self._metadata.clear()


class HealthRegistry:
    """Registry for managing multiple health checkers.

    Provides aggregated health reporting for all registered components.

    Example:
        registry = HealthRegistry()
        registry.register("database")
        registry.register("cache")

        registry.get("database").record_success(latency_ms=5.0)
        registry.get("cache").record_failure("Connection refused")

        report = registry.get_report()
        print(f"Overall healthy: {report.overall_healthy}")
    """

    def __init__(self):
        self._checkers: Dict[str, HealthChecker] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
        latency_window: int = 10,
    ) -> HealthChecker:
        """Register a new health checker.

        Args:
            name: Component name
            failure_threshold: Failures before marking unhealthy
            recovery_threshold: Successes before marking healthy
            latency_window: Number of latency samples to average

        Returns:
            The created HealthChecker
        """
        with self._lock:
            if name in self._checkers:
                return self._checkers[name]

            checker = HealthChecker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_threshold=recovery_threshold,
                latency_window=latency_window,
            )
            self._checkers[name] = checker
            return checker

    def get(self, name: str) -> Optional[HealthChecker]:
        """Get a health checker by name."""
        with self._lock:
            return self._checkers.get(name)

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
    ) -> HealthChecker:
        """Get existing or create new health checker."""
        with self._lock:
            if name not in self._checkers:
                self._checkers[name] = HealthChecker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_threshold=recovery_threshold,
                )
            return self._checkers[name]

    def unregister(self, name: str) -> bool:
        """Unregister a health checker."""
        with self._lock:
            if name in self._checkers:
                del self._checkers[name]
                return True
            return False

    def get_report(self) -> HealthReport:
        """Get aggregated health report for all components."""
        with self._lock:
            components = {name: checker.get_status() for name, checker in self._checkers.items()}

        overall_healthy = all(status.healthy for status in components.values())

        # Build summary
        unhealthy = [name for name, status in components.items() if not status.healthy]
        if not components:
            summary = "No components registered"
        elif overall_healthy:
            summary = f"All {len(components)} components healthy"
        else:
            summary = (
                f"{len(unhealthy)}/{len(components)} components unhealthy: {', '.join(unhealthy)}"
            )

        return HealthReport(
            overall_healthy=overall_healthy,
            components=components,
            checked_at=datetime.now(timezone.utc),
            summary=summary,
        )

    def get_all_statuses(self) -> Dict[str, HealthStatus]:
        """Get status for all registered components."""
        with self._lock:
            return {name: checker.get_status() for name, checker in self._checkers.items()}

    @property
    def registered_components(self) -> List[str]:
        """List of registered component names."""
        with self._lock:
            return list(self._checkers.keys())


# Global health registry for convenience
_global_registry: Optional[HealthRegistry] = None
_global_registry_lock = threading.Lock()


def get_global_health_registry() -> HealthRegistry:
    """Get the global health registry."""
    global _global_registry
    with _global_registry_lock:
        if _global_registry is None:
            _global_registry = HealthRegistry()
        return _global_registry
