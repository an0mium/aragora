"""
Public Status Page and SLA Instrumentation for Aragora.

Provides a public-facing status page that exposes service health,
component statuses, active incidents, and uptime metrics without
requiring authentication.

Integrates with the existing HealthRegistry from aragora.resilience.health
to derive component health from actual runtime checks.

Usage:
    from aragora.ops.status_page import StatusPage, ServiceStatus

    page = StatusPage()

    # Get overall service status
    overall = page.get_overall_status()
    print(f"Status: {overall.value}")

    # Get per-component health
    components = page.get_component_statuses()
    for c in components:
        print(f"{c.name}: {c.status.value} ({c.response_time_ms}ms)")

    # Manage incidents
    incident = page.create_incident("API Latency Spike", "warning", "p99 above SLO target")
    page.update_incident(incident.id, "Investigating root cause", "investigating")
    page.resolve_incident(incident.id, "Scaled up API pods, latency normalized")

    # Check uptime
    uptime = page.get_uptime("api", days=30)
    print(f"API uptime: {uptime:.2f}%")
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Models
# =============================================================================


class ServiceStatus(str, Enum):
    """Overall service status for the public status page.

    Values ordered by severity (operational is best, maintenance is neutral).
    """

    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    PARTIAL_OUTAGE = "partial_outage"
    MAJOR_OUTAGE = "major_outage"
    MAINTENANCE = "maintenance"


@dataclass
class ComponentStatus:
    """Health status for an individual service component.

    Attributes:
        name: Human-readable component name (e.g. "API", "Database")
        status: Current operational status
        description: Brief description of the component's role
        last_checked: Timestamp of most recent health check
        response_time_ms: Average response time in milliseconds (None if unavailable)
    """

    name: str
    status: ServiceStatus
    description: str = ""
    last_checked: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "description": self.description,
            "last_checked": self.last_checked.isoformat(),
            "response_time_ms": self.response_time_ms,
        }


@dataclass
class IncidentUpdate:
    """A single update posted to an incident timeline.

    Attributes:
        timestamp: When the update was posted
        message: Update message describing the current state
        status: Incident status at time of update
    """

    timestamp: datetime
    message: str
    status: str  # investigating, identified, monitoring, resolved


@dataclass
class IncidentRecord:
    """A service incident with a timeline of updates.

    Attributes:
        id: Unique incident identifier
        title: Short incident title
        status: Current incident status
        severity: Incident severity (critical, warning, info)
        started_at: When the incident began
        resolved_at: When the incident was resolved (None if ongoing)
        updates: Chronological list of incident updates
    """

    id: str
    title: str
    status: str  # investigating, identified, monitoring, resolved
    severity: str  # critical, warning, info
    started_at: datetime
    resolved_at: datetime | None = None
    updates: list[IncidentUpdate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "severity": self.severity,
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "updates": [
                {
                    "timestamp": u.timestamp.isoformat(),
                    "message": u.message,
                    "status": u.status,
                }
                for u in self.updates
            ],
        }


# =============================================================================
# Default Components
# =============================================================================

DEFAULT_COMPONENTS = {
    "api": "REST API endpoints",
    "websocket": "WebSocket real-time streaming",
    "database": "Primary data store",
    "queue": "Background task processing",
    "search": "Full-text and semantic search",
}


# =============================================================================
# Status Page
# =============================================================================


class StatusPage:
    """Public status page aggregating component health and incidents.

    Integrates with the HealthRegistry from aragora.resilience.health
    to derive real-time component status from existing health checks.

    Thread-safe: all mutable state is protected by a lock.

    Args:
        health_registry: Optional HealthRegistry instance. If not provided,
            uses the global health registry.
        components: Optional dict mapping component ID to description.
            Defaults to the standard Aragora components.
    """

    def __init__(
        self,
        health_registry: Any | None = None,
        components: dict[str, str] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._health_registry = health_registry
        self._components = components or dict(DEFAULT_COMPONENTS)
        self._incidents: list[IncidentRecord] = []
        self._uptime_records: dict[str, list[tuple[datetime, bool]]] = {}

    def _get_health_registry(self) -> Any:
        """Get the health registry, lazily importing if needed."""
        if self._health_registry is not None:
            return self._health_registry
        try:
            from aragora.resilience.health import get_global_health_registry

            return get_global_health_registry()
        except ImportError:
            logger.debug("Health registry not available")
            return None

    @staticmethod
    def _health_to_service_status(healthy: bool, consecutive_failures: int) -> ServiceStatus:
        """Map health checker state to a public ServiceStatus.

        Args:
            healthy: Whether the health checker reports healthy
            consecutive_failures: Number of consecutive failures

        Returns:
            Appropriate ServiceStatus value
        """
        if healthy:
            return ServiceStatus.OPERATIONAL
        if consecutive_failures >= 5:
            return ServiceStatus.MAJOR_OUTAGE
        if consecutive_failures >= 3:
            return ServiceStatus.PARTIAL_OUTAGE
        return ServiceStatus.DEGRADED

    def get_component_statuses(self) -> list[ComponentStatus]:
        """Get current status for all tracked components.

        Queries the HealthRegistry for components that have registered
        health checkers, and returns OPERATIONAL for components without
        active checkers (assumed healthy by default).

        Returns:
            List of ComponentStatus for each tracked component
        """
        registry = self._get_health_registry()
        now = datetime.now(timezone.utc)
        results: list[ComponentStatus] = []

        for comp_id, description in self._components.items():
            checker_status = None
            if registry is not None:
                checker = registry.get(comp_id)
                if checker is not None:
                    checker_status = checker.get_status()

            if checker_status is not None:
                status = self._health_to_service_status(
                    checker_status.healthy,
                    checker_status.consecutive_failures,
                )
                results.append(
                    ComponentStatus(
                        name=comp_id,
                        status=status,
                        description=description,
                        last_checked=checker_status.last_check,
                        response_time_ms=checker_status.latency_ms,
                    )
                )
            else:
                # No health checker registered -- assume operational
                results.append(
                    ComponentStatus(
                        name=comp_id,
                        status=ServiceStatus.OPERATIONAL,
                        description=description,
                        last_checked=now,
                        response_time_ms=None,
                    )
                )

            # Record uptime sample
            self._record_uptime(comp_id, results[-1].status == ServiceStatus.OPERATIONAL)

        return results

    def get_overall_status(self) -> ServiceStatus:
        """Compute the aggregate service status from all components.

        The overall status is the worst status among all components.
        If any active incidents exist, their severity also influences
        the result.

        Returns:
            Aggregate ServiceStatus
        """
        statuses = self.get_component_statuses()

        # Check for active maintenance incidents
        with self._lock:
            active = [i for i in self._incidents if i.status != "resolved"]
            has_maintenance = any(i.severity == "maintenance" for i in active)

        if has_maintenance:
            return ServiceStatus.MAINTENANCE

        if not statuses:
            return ServiceStatus.OPERATIONAL

        # Severity ordering for comparison
        severity_order = {
            ServiceStatus.OPERATIONAL: 0,
            ServiceStatus.DEGRADED: 1,
            ServiceStatus.PARTIAL_OUTAGE: 2,
            ServiceStatus.MAJOR_OUTAGE: 3,
            ServiceStatus.MAINTENANCE: 4,
        }

        worst = max(statuses, key=lambda s: severity_order.get(s.status, 0))
        return worst.status

    def get_active_incidents(self) -> list[IncidentRecord]:
        """Get all currently unresolved incidents.

        Returns:
            List of incidents where status is not 'resolved',
            sorted by start time (newest first).
        """
        with self._lock:
            active = [i for i in self._incidents if i.status != "resolved"]
        return sorted(active, key=lambda i: i.started_at, reverse=True)

    def get_incident_history(self, days: int = 30) -> list[IncidentRecord]:
        """Get resolved incidents within the specified time window.

        Args:
            days: Number of days to look back (default 30)

        Returns:
            List of resolved incidents within the window, newest first
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._lock:
            history = [
                i for i in self._incidents if i.status == "resolved" and i.started_at >= cutoff
            ]
        return sorted(history, key=lambda i: i.started_at, reverse=True)

    def get_uptime(self, component: str, days: int = 30) -> float:
        """Calculate uptime percentage for a component over a time window.

        Uses recorded health check samples. If no samples exist for the
        component, returns 100.0 (assumed fully operational).

        Args:
            component: Component ID (e.g. "api", "database")
            days: Number of days to compute uptime for

        Returns:
            Uptime percentage (0.0 to 100.0)
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._lock:
            records = self._uptime_records.get(component, [])
            in_window = [(ts, ok) for ts, ok in records if ts >= cutoff]

        if not in_window:
            return 100.0

        healthy_count = sum(1 for _, ok in in_window if ok)
        return (healthy_count / len(in_window)) * 100.0

    def create_incident(
        self,
        title: str,
        severity: str,
        message: str,
    ) -> IncidentRecord:
        """Create a new incident.

        Args:
            title: Short incident title
            severity: Severity level (critical, warning, info)
            message: Initial update message

        Returns:
            The created IncidentRecord
        """
        now = datetime.now(timezone.utc)
        incident = IncidentRecord(
            id=str(uuid.uuid4()),
            title=title,
            status="investigating",
            severity=severity,
            started_at=now,
            updates=[
                IncidentUpdate(
                    timestamp=now,
                    message=message,
                    status="investigating",
                )
            ],
        )

        with self._lock:
            self._incidents.append(incident)

        logger.info("Incident created: %s [%s] %s", incident.id, severity, title)
        return incident

    def update_incident(self, incident_id: str, message: str, status: str) -> bool:
        """Post an update to an existing incident.

        Args:
            incident_id: ID of the incident to update
            message: Update message
            status: New status (investigating, identified, monitoring, resolved)

        Returns:
            True if the incident was found and updated, False otherwise
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            for incident in self._incidents:
                if incident.id == incident_id:
                    incident.status = status
                    incident.updates.append(
                        IncidentUpdate(timestamp=now, message=message, status=status)
                    )
                    if status == "resolved":
                        incident.resolved_at = now
                    logger.info("Incident updated: %s -> %s", incident_id, status)
                    return True
        return False

    def resolve_incident(self, incident_id: str, message: str) -> bool:
        """Resolve an incident with a closing message.

        Args:
            incident_id: ID of the incident to resolve
            message: Resolution message

        Returns:
            True if the incident was found and resolved, False otherwise
        """
        return self.update_incident(incident_id, message, "resolved")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full status page to a JSON-friendly dictionary.

        Returns:
            Dictionary with overall status, components, incidents, and uptime
        """
        components = self.get_component_statuses()
        overall = self.get_overall_status()
        active_incidents = self.get_active_incidents()
        history = self.get_incident_history()

        return {
            "status": overall.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "components": [c.to_dict() for c in components],
            "active_incidents": [i.to_dict() for i in active_incidents],
            "incident_history": [i.to_dict() for i in history],
            "uptime": {comp_id: self.get_uptime(comp_id) for comp_id in self._components},
        }

    def _record_uptime(self, component: str, healthy: bool) -> None:
        """Record a single uptime sample for a component.

        Automatically trims old records beyond 90 days.

        Args:
            component: Component ID
            healthy: Whether the component is healthy
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=90)

        with self._lock:
            if component not in self._uptime_records:
                self._uptime_records[component] = []

            records = self._uptime_records[component]
            records.append((now, healthy))

            # Trim old records
            self._uptime_records[component] = [(ts, ok) for ts, ok in records if ts >= cutoff]


# =============================================================================
# Module-level singleton
# =============================================================================

_global_status_page: StatusPage | None = None
_global_status_page_lock = threading.Lock()


def get_status_page() -> StatusPage:
    """Get or create the global StatusPage singleton."""
    global _global_status_page
    with _global_status_page_lock:
        if _global_status_page is None:
            _global_status_page = StatusPage()
        return _global_status_page


def reset_status_page() -> None:
    """Reset the global StatusPage (for testing)."""
    global _global_status_page
    with _global_status_page_lock:
        _global_status_page = None


__all__ = [
    "ServiceStatus",
    "ComponentStatus",
    "IncidentRecord",
    "IncidentUpdate",
    "StatusPage",
    "get_status_page",
    "reset_status_page",
    "DEFAULT_COMPONENTS",
]
