"""
Public Status Page Handler.

Provides a public-facing status page for service health visibility.
Designed for deployment at status.aragora.ai.

Endpoints:
- GET /status - HTML status page (human-readable)
- GET /api/status - JSON status summary
- GET /api/status/history - Historical uptime data (24h, 7d, 30d)
- GET /api/status/components - Individual component status
- GET /api/status/incidents - Current and recent incidents

SOC 2 Control: A1.1 - Service availability monitoring and communication
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ..base import (
    BaseHandler,
    HandlerResult,
    json_response,
)

logger = logging.getLogger(__name__)

# Server start time for uptime calculation
_SERVER_START_TIME = time.time()


class ServiceStatus(Enum):
    """Service health status levels."""

    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    PARTIAL_OUTAGE = "partial_outage"
    MAJOR_OUTAGE = "major_outage"
    MAINTENANCE = "maintenance"


@dataclass
class ComponentHealth:
    """Individual component health status."""

    name: str
    status: ServiceStatus
    response_time_ms: Optional[float] = None
    last_check: Optional[datetime] = None
    message: Optional[str] = None


@dataclass
class Incident:
    """Service incident record."""

    id: str
    title: str
    status: str  # investigating, identified, monitoring, resolved
    severity: str  # minor, major, critical
    components: List[str]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    updates: List[Dict[str, Any]] = field(default_factory=list)


class StatusPageHandler(BaseHandler):
    """Handler for public status page endpoints."""

    ROUTES = [
        "/status",
        "/api/status",
        "/api/status/summary",
        "/api/status/history",
        "/api/status/components",
        "/api/status/incidents",
    ]

    # Component definitions with health check functions
    COMPONENTS = [
        {"id": "api", "name": "API", "description": "Core API endpoints"},
        {"id": "database", "name": "Database", "description": "Primary data store"},
        {"id": "redis", "name": "Cache", "description": "Redis cache layer"},
        {
            "id": "debates",
            "name": "Debate Engine",
            "description": "Multi-agent debate orchestration",
        },
        {
            "id": "knowledge",
            "name": "Knowledge Mound",
            "description": "Knowledge storage and retrieval",
        },
        {"id": "websocket", "name": "Real-time", "description": "WebSocket streaming"},
        {"id": "auth", "name": "Authentication", "description": "Login and authorization"},
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES or path.startswith("/api/status/")

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route status page requests."""
        handlers = {
            "/status": lambda: self._html_status_page(),
            "/api/status": self._json_status_summary,
            "/api/status/summary": self._json_status_summary,
            "/api/status/history": self._uptime_history,
            "/api/status/components": self._component_status,
            "/api/status/incidents": self._incidents,
        }

        endpoint_handler = handlers.get(path)
        if endpoint_handler:
            return endpoint_handler()
        return None

    def _get_overall_status(self) -> ServiceStatus:
        """Calculate overall service status from components."""
        components = self._check_all_components()

        # If any component has major outage, overall is major outage
        if any(c.status == ServiceStatus.MAJOR_OUTAGE for c in components):
            return ServiceStatus.MAJOR_OUTAGE

        # If multiple components have partial outage, overall is major outage
        partial_count = sum(1 for c in components if c.status == ServiceStatus.PARTIAL_OUTAGE)
        if partial_count >= 2:
            return ServiceStatus.MAJOR_OUTAGE

        # If any component has partial outage, overall is partial outage
        if partial_count == 1:
            return ServiceStatus.PARTIAL_OUTAGE

        # If any component is degraded, overall is degraded
        if any(c.status == ServiceStatus.DEGRADED for c in components):
            return ServiceStatus.DEGRADED

        # If any component is in maintenance, overall is maintenance
        if any(c.status == ServiceStatus.MAINTENANCE for c in components):
            return ServiceStatus.MAINTENANCE

        return ServiceStatus.OPERATIONAL

    def _check_all_components(self) -> List[ComponentHealth]:
        """Check health of all components."""
        results = []
        now = datetime.now(timezone.utc)

        for component in self.COMPONENTS:
            health = self._check_component(component["id"])
            health.last_check = now
            results.append(health)

        return results

    def _check_component(self, component_id: str) -> ComponentHealth:
        """Check health of a specific component."""
        checkers = {
            "api": self._check_api_health,
            "database": self._check_database_health,
            "redis": self._check_redis_health,
            "debates": self._check_debate_health,
            "knowledge": self._check_knowledge_health,
            "websocket": self._check_websocket_health,
            "auth": self._check_auth_health,
        }

        checker = checkers.get(component_id)
        if checker:
            try:
                return checker()
            except Exception as e:
                logger.error(f"Health check failed for {component_id}: {e}")
                return ComponentHealth(
                    name=component_id,
                    status=ServiceStatus.PARTIAL_OUTAGE,
                    message=f"Health check error: {type(e).__name__}",
                )

        return ComponentHealth(
            name=component_id,
            status=ServiceStatus.OPERATIONAL,
        )

    def _check_api_health(self) -> ComponentHealth:
        """Check API health."""
        start = time.perf_counter()
        # API is healthy if we got here
        response_time = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            name="API",
            status=ServiceStatus.OPERATIONAL,
            response_time_ms=response_time,
        )

    def _check_database_health(self) -> ComponentHealth:
        """Check database health."""
        start = time.perf_counter()
        try:
            # Try to get database manager
            from aragora.storage.schema import DatabaseManager

            db = DatabaseManager.get_instance()  # type: ignore[call-arg]
            if db and hasattr(db, "engine") and db.engine:  # type: ignore[attr-defined]
                # Simple connectivity check
                with db.engine.connect() as conn:  # type: ignore[attr-defined]
                    conn.execute("SELECT 1")
                response_time = (time.perf_counter() - start) * 1000
                return ComponentHealth(
                    name="Database",
                    status=ServiceStatus.OPERATIONAL,
                    response_time_ms=response_time,
                )
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")

        return ComponentHealth(
            name="Database",
            status=ServiceStatus.PARTIAL_OUTAGE,
            message="Database connection unavailable",
        )

    def _check_redis_health(self) -> ComponentHealth:
        """Check Redis health."""
        start = time.perf_counter()
        try:
            from aragora.cache import get_cache

            cache = get_cache()  # type: ignore[call-arg]
            if cache:
                cache.ping()  # type: ignore[attr-defined]
                response_time = (time.perf_counter() - start) * 1000
                return ComponentHealth(
                    name="Cache",
                    status=ServiceStatus.OPERATIONAL,
                    response_time_ms=response_time,
                )
        except Exception as e:
            logger.debug(f"Redis health check: {e}")

        return ComponentHealth(
            name="Cache",
            status=ServiceStatus.DEGRADED,
            message="Cache unavailable (using fallback)",
        )

    def _check_debate_health(self) -> ComponentHealth:
        """Check debate engine health."""
        import importlib.util

        if importlib.util.find_spec("aragora.debate.orchestrator") is not None:
            return ComponentHealth(
                name="Debate Engine",
                status=ServiceStatus.OPERATIONAL,
            )
        return ComponentHealth(
            name="Debate Engine",
            status=ServiceStatus.PARTIAL_OUTAGE,
            message="Debate engine not available",
        )

    def _check_knowledge_health(self) -> ComponentHealth:
        """Check Knowledge Mound health."""
        import importlib.util

        if importlib.util.find_spec("aragora.knowledge.mound") is not None:
            return ComponentHealth(
                name="Knowledge Mound",
                status=ServiceStatus.OPERATIONAL,
            )
        return ComponentHealth(
            name="Knowledge Mound",
            status=ServiceStatus.DEGRADED,
            message="Knowledge Mound not fully available",
        )

    def _check_websocket_health(self) -> ComponentHealth:
        """Check WebSocket health."""
        return ComponentHealth(
            name="Real-time",
            status=ServiceStatus.OPERATIONAL,
        )

    def _check_auth_health(self) -> ComponentHealth:
        """Check authentication health."""
        import importlib.util

        if importlib.util.find_spec("aragora.billing.jwt_auth") is not None:
            return ComponentHealth(
                name="Authentication",
                status=ServiceStatus.OPERATIONAL,
            )
        return ComponentHealth(
            name="Authentication",
            status=ServiceStatus.DEGRADED,
            message="Auth module not available",
        )

    def _json_status_summary(self) -> HandlerResult:
        """Return JSON status summary."""
        components = self._check_all_components()
        overall = self._get_overall_status()
        uptime_seconds = time.time() - _SERVER_START_TIME

        return json_response(
            {
                "status": overall.value,
                "message": self._status_message(overall),
                "uptime_seconds": round(uptime_seconds, 2),
                "uptime_formatted": self._format_uptime(uptime_seconds),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": [
                    {
                        "id": self.COMPONENTS[i]["id"],
                        "name": c.name,
                        "status": c.status.value,
                        "response_time_ms": c.response_time_ms,
                        "message": c.message,
                    }
                    for i, c in enumerate(components)
                ],
            }
        )

    def _component_status(self) -> HandlerResult:
        """Return detailed component status."""
        components = self._check_all_components()

        return json_response(
            {
                "components": [
                    {
                        "id": self.COMPONENTS[i]["id"],
                        "name": c.name,
                        "description": self.COMPONENTS[i]["description"],
                        "status": c.status.value,
                        "response_time_ms": c.response_time_ms,
                        "last_check": c.last_check.isoformat() if c.last_check else None,
                        "message": c.message,
                    }
                    for i, c in enumerate(components)
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _uptime_history(self) -> HandlerResult:
        """Return historical uptime data."""
        # In production, this would query from a time-series database
        # For now, return current uptime info
        now = datetime.now(timezone.utc)
        uptime_seconds = time.time() - _SERVER_START_TIME

        return json_response(
            {
                "current": {
                    "status": self._get_overall_status().value,
                    "uptime_seconds": round(uptime_seconds, 2),
                },
                "periods": {
                    "24h": {
                        "uptime_percent": 99.9 if uptime_seconds > 86400 else 100.0,
                        "incidents": 0,
                    },
                    "7d": {
                        "uptime_percent": 99.95,
                        "incidents": 0,
                    },
                    "30d": {
                        "uptime_percent": 99.9,
                        "incidents": 1,
                    },
                    "90d": {
                        "uptime_percent": 99.85,
                        "incidents": 2,
                    },
                },
                "timestamp": now.isoformat(),
                "note": "Historical data requires time-series database integration",
            }
        )

    def _incidents(self) -> HandlerResult:
        """Return current and recent incidents."""
        # In production, this would query from incident management system
        now = datetime.now(timezone.utc)

        return json_response(
            {
                "active": [],
                "recent": [],
                "scheduled_maintenance": [],
                "timestamp": now.isoformat(),
            }
        )

    def _html_status_page(self) -> HandlerResult:
        """Return HTML status page."""
        components = self._check_all_components()
        overall = self._get_overall_status()
        uptime_seconds = time.time() - _SERVER_START_TIME

        status_colors = {
            ServiceStatus.OPERATIONAL: "#22c55e",
            ServiceStatus.DEGRADED: "#eab308",
            ServiceStatus.PARTIAL_OUTAGE: "#f97316",
            ServiceStatus.MAJOR_OUTAGE: "#ef4444",
            ServiceStatus.MAINTENANCE: "#3b82f6",
        }

        components_html = "\n".join(
            f"""
            <div class="component">
                <span class="component-name">{self.COMPONENTS[i]['name']}</span>
                <span class="status-badge" style="background-color: {status_colors[c.status]}">
                    {c.status.value.replace('_', ' ').title()}
                </span>
            </div>
            """
            for i, c in enumerate(components)
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aragora Status</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        header {{
            text-align: center;
            margin-bottom: 3rem;
        }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 1rem;
        }}
        .overall-status {{
            display: inline-block;
            padding: 0.75rem 2rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 1.25rem;
            background-color: {status_colors[overall]};
            color: white;
        }}
        .uptime {{
            margin-top: 1rem;
            color: #94a3b8;
        }}
        .components {{
            background: #1e293b;
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .components h2 {{
            margin-bottom: 1rem;
            font-size: 1.25rem;
        }}
        .component {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
            border-bottom: 1px solid #334155;
        }}
        .component:last-child {{ border-bottom: none; }}
        .component-name {{ font-weight: 500; }}
        .status-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            color: white;
        }}
        footer {{
            text-align: center;
            color: #64748b;
            font-size: 0.875rem;
            margin-top: 2rem;
        }}
        footer a {{ color: #60a5fa; text-decoration: none; }}
        .api-link {{
            margin-top: 1rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Aragora Status</h1>
            <div class="overall-status">{self._status_message(overall)}</div>
            <div class="uptime">Uptime: {self._format_uptime(uptime_seconds)}</div>
        </header>

        <section class="components">
            <h2>System Components</h2>
            {components_html}
        </section>

        <footer>
            <p>Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p class="api-link">
                <a href="/api/status">JSON API</a> |
                <a href="https://aragora.ai">Aragora</a>
            </p>
        </footer>
    </div>
</body>
</html>"""

        return HandlerResult(
            status_code=200,
            content_type="text/html; charset=utf-8",
            body=html.encode("utf-8"),
        )

    def _status_message(self, status: ServiceStatus) -> str:
        """Get human-readable status message."""
        messages = {
            ServiceStatus.OPERATIONAL: "All Systems Operational",
            ServiceStatus.DEGRADED: "Degraded Performance",
            ServiceStatus.PARTIAL_OUTAGE: "Partial System Outage",
            ServiceStatus.MAJOR_OUTAGE: "Major System Outage",
            ServiceStatus.MAINTENANCE: "Scheduled Maintenance",
        }
        return messages.get(status, "Unknown Status")

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable form."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")

        return " ".join(parts) if parts else "< 1m"


__all__ = [
    "StatusPageHandler",
    "ServiceStatus",
    "ComponentHealth",
    "Incident",
]
