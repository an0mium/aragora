"""
Health monitoring handlers for Control Plane.

Provides REST API endpoints for:
- System and agent health status
- Detailed health with component status
- Circuit breaker states
- Control plane statistics and metrics
- Notifications
- Audit logs
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, cast

from aragora.server.http_utils import run_async as _run_async
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from aragora.server.handlers.openapi_decorator import api_endpoint
from aragora.server.handlers.utils.decorators import has_permission as _has_permission
from aragora.server.handlers.utils.decorators import require_permission
from aragora.server.validation.query_params import safe_query_int

logger = logging.getLogger(__name__)

# Cached permission function and timestamp
_cached_has_permission: Any = None
_cache_timestamp: float = 0
_CACHE_TTL_SECONDS = 60.0


def _get_has_permission():
    """Get the has_permission function with caching to avoid repeated module lookups."""
    global _cached_has_permission, _cache_timestamp

    now = time.time()
    if _cached_has_permission is not None and (now - _cache_timestamp) < _CACHE_TTL_SECONDS:
        return _cached_has_permission

    control_plane = sys.modules.get("aragora.server.handlers.control_plane")
    if control_plane is not None:
        candidate = getattr(control_plane, "has_permission", None)
        if callable(candidate):
            _cached_has_permission = candidate
            _cache_timestamp = now
            return candidate

    _cached_has_permission = _has_permission
    _cache_timestamp = now
    return _has_permission


class HealthHandlerMixin:
    """
    Mixin class providing health monitoring handlers.

    This mixin provides methods for:
    - System health status
    - Agent health status
    - Detailed health with components
    - Circuit breaker states
    - Control plane statistics
    - Dashboard metrics
    - Notifications
    - Audit logs
    """

    # These methods are expected from the base class
    def _get_coordinator(self) -> Any | None:
        """Get the control plane coordinator."""
        raise NotImplementedError

    def _require_coordinator(self) -> tuple[Any | None, HandlerResult | None]:
        """Return coordinator and None, or None and error response if not initialized."""
        raise NotImplementedError

    def _handle_coordinator_error(self, error: Exception, operation: str) -> HandlerResult:
        """Unified error handler for coordinator operations."""
        raise NotImplementedError

    def require_auth_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]:
        """Require authentication and return user or error."""
        # Cast super() to Any - mixin expects base class to provide this method
        return cast(Any, super()).require_auth_or_error(handler)

    # Attribute declaration - provided by BaseHandler
    ctx: dict[str, Any]

    # =========================================================================
    # Health Handlers
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/control-plane/health",
        summary="Get system health status",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:health.read")
    def _handle_system_health(self) -> HandlerResult:
        """Get system health status."""
        coordinator, err = self._require_coordinator()
        if err:
            return err

        try:
            health_status = coordinator.get_system_health()
            all_health = coordinator._health_monitor.get_all_health()

            return json_response(
                {
                    "status": health_status.value,
                    "agents": {agent_id: hc.to_dict() for agent_id, hc in all_health.items()},
                }
            )
        except Exception as e:
            return self._handle_coordinator_error(e, "system_health")

    @api_endpoint(
        method="GET",
        path="/api/control-plane/health/{agent_id}",
        summary="Get agent health status",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:health.read")
    def _handle_agent_health(self, agent_id: str) -> HandlerResult:
        """Get health status for specific agent."""
        coordinator, err = self._require_coordinator()
        if err:
            return err

        try:
            health = coordinator.get_agent_health(agent_id)

            if not health:
                return error_response(f"No health data for agent: {agent_id}", 404)

            return json_response(health.to_dict())
        except Exception as e:
            return self._handle_coordinator_error(e, f"agent_health:{agent_id}")

    @api_endpoint(
        method="GET",
        path="/api/control-plane/health/detailed",
        summary="Get detailed system health",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:health.read")
    def _handle_detailed_health(self) -> HandlerResult:
        """Get detailed system health with component status."""
        coordinator = self._get_coordinator()

        try:
            import time

            start_time = getattr(self, "_start_time", time.time())
            uptime = int(time.time() - start_time)

            components = []

            # Check coordinator/scheduler health
            if coordinator:
                components.append(
                    {
                        "name": "Coordinator",
                        "status": "healthy",
                        "latency_ms": 0,
                    }
                )

                if hasattr(coordinator, "_scheduler"):
                    components.append(
                        {
                            "name": "Scheduler",
                            "status": "healthy",
                            "latency_ms": 0,
                        }
                    )

            # Check Redis if available
            try:
                from aragora.control_plane.shared_state import get_shared_state_sync

                state = get_shared_state_sync()
                if state and hasattr(state, "redis"):
                    start = time.time()
                    _run_async(state.redis.ping())
                    latency = int((time.time() - start) * 1000)
                    components.append(
                        {
                            "name": "Redis",
                            "status": "healthy",
                            "latency_ms": latency,
                        }
                    )
            except Exception as e:
                logger.debug(f"Redis health check failed: {e}")
                components.append(
                    {
                        "name": "Redis",
                        "status": "unhealthy",
                        "error": f"Not connected: {type(e).__name__}",
                    }
                )

            # Check database if available
            try:
                from aragora.db.database import Database

                db = Database()
                if db.is_connected():
                    components.append(
                        {
                            "name": "Database",
                            "status": "healthy",
                            "latency_ms": 10,
                        }
                    )
            except Exception as e:
                logger.debug(f"Database health check skipped: {e}")

            # Overall status
            unhealthy = any(c["status"] == "unhealthy" for c in components)
            degraded = any(c["status"] == "degraded" for c in components)
            status = "unhealthy" if unhealthy else ("degraded" if degraded else "healthy")

            return json_response(
                {
                    "status": status,
                    "uptime_seconds": uptime,
                    "version": "2.1.0",
                    "components": components,
                }
            )
        except Exception as e:
            logger.error(f"Error getting detailed health: {e}")
            return error_response(safe_error_message(e, "control plane"), 500)

    @api_endpoint(
        method="GET",
        path="/api/control-plane/breakers",
        summary="Get circuit breaker states",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:health.read")
    def _handle_circuit_breakers(self) -> HandlerResult:
        """Get circuit breaker states."""
        try:
            breakers = []

            # Try to get circuit breakers from resilience module
            try:
                from aragora.resilience import get_circuit_breakers

                for name, breaker in get_circuit_breakers().items():
                    breakers.append(
                        {
                            "name": name,
                            "state": (
                                breaker.state.value
                                if hasattr(breaker.state, "value")
                                else str(breaker.state)
                            ),
                            "failure_count": getattr(breaker, "failure_count", 0),
                            "success_count": getattr(breaker, "success_count", 0),
                            "last_failure": getattr(breaker, "last_failure_time", None),
                            "reset_timeout_ms": getattr(breaker, "reset_timeout", 30) * 1000,
                        }
                    )
            except (ImportError, AttributeError):
                # Resilience module not available or no breakers
                pass

            return json_response({"breakers": breakers})
        except Exception as e:
            logger.error(f"Error getting circuit breakers: {e}")
            return error_response(safe_error_message(e, "control plane"), 500)

    @api_endpoint(
        method="GET",
        path="/api/control-plane/stats",
        summary="Get control plane statistics",
        tags=["Control Plane"],
        operation_id="get_control_plane_stats",
    )
    @require_permission("controlplane:read")
    def _handle_stats(self) -> HandlerResult:
        """Get control plane statistics."""
        coordinator, err = self._require_coordinator()
        if err:
            return err

        try:
            stats = _run_async(coordinator.get_stats())

            return json_response(stats)
        except Exception as e:
            return self._handle_coordinator_error(e, "stats")

    @api_endpoint(
        method="GET",
        path="/api/control-plane/metrics",
        summary="Get dashboard metrics",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:metrics.read")
    def _handle_get_metrics(self) -> HandlerResult:
        """Get control plane metrics for dashboard."""
        coordinator, err = self._require_coordinator()
        if err:
            return err

        try:
            # Get comprehensive stats
            stats = _run_async(coordinator.get_stats())

            scheduler_stats = stats.get("scheduler", {})
            registry_stats = stats.get("registry", {})

            by_status = scheduler_stats.get("by_status", {})
            agent_by_status = registry_stats.get("by_status", {})

            # Calculate metrics for dashboard
            active_jobs = by_status.get("running", 0)
            queued_jobs = by_status.get("pending", 0)
            completed_jobs = by_status.get("completed", 0)

            agents_available = registry_stats.get("available_agents", 0)
            agents_busy = agent_by_status.get("busy", 0)
            total_agents = registry_stats.get("total_agents", 0)

            return json_response(
                {
                    "active_jobs": active_jobs,
                    "queued_jobs": queued_jobs,
                    "completed_jobs": completed_jobs,
                    "agents_available": agents_available,
                    "agents_busy": agents_busy,
                    "total_agents": total_agents,
                    # These could come from a metrics store if available
                    "documents_processed_today": scheduler_stats.get("by_type", {}).get(
                        "document_processing", 0
                    ),
                    "audits_completed_today": scheduler_stats.get("by_type", {}).get("audit", 0),
                    "tokens_used_today": 0,  # Would need token tracking integration
                }
            )
        except Exception as e:
            return self._handle_coordinator_error(e, "metrics")

    # =========================================================================
    # Notification Handlers
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/control-plane/notifications",
        summary="Get recent notifications",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:notifications.read")
    def _handle_get_notifications(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get recent notification history."""
        try:
            # Import check for availability
            from aragora.control_plane.channels import NotificationManager  # noqa: F401

            manager = self.ctx.get("notification_manager")
            if not manager:
                # Return empty if not configured
                return json_response(
                    {
                        "notifications": [],
                        "total": 0,
                        "message": "Notification manager not configured",
                    }
                )

            stats_fn = getattr(manager, "get_stats", None)
            stats = stats_fn() if stats_fn else {}
            return json_response(
                {
                    "notifications": [],  # History is internal; could expose if needed
                    "stats": stats,
                }
            )
        except ImportError:
            return json_response(
                {
                    "notifications": [],
                    "total": 0,
                    "message": "Notification module not available",
                }
            )
        except Exception as e:
            logger.error(f"Error getting notifications: {e}")
            return error_response(safe_error_message(e, "notifications"), 500)

    @api_endpoint(
        method="GET",
        path="/api/control-plane/notifications/stats",
        summary="Get notification statistics",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:notifications.read")
    def _handle_get_notification_stats(self) -> HandlerResult:
        """Get notification statistics."""
        try:
            manager = self.ctx.get("notification_manager")
            if not manager:
                return json_response(
                    {
                        "total_sent": 0,
                        "successful": 0,
                        "failed": 0,
                        "success_rate": 0,
                        "by_channel": {},
                        "channels_configured": 0,
                    }
                )

            stats_fn = getattr(manager, "get_stats", None)
            return json_response(stats_fn() if stats_fn else {})
        except Exception as e:
            logger.error(f"Error getting notification stats: {e}")
            return error_response(safe_error_message(e, "notifications"), 500)

    # =========================================================================
    # Audit Log Handlers
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/control-plane/audit",
        summary="Query audit logs",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:audit.read")
    def _handle_get_audit_logs(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """Query audit logs with filtering."""
        # Require authentication for audit access
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Check permission for audit access
        if not _get_has_permission()(
            user.role if hasattr(user, "role") else None, "controlplane:audit"
        ):
            return error_response("Permission denied: controlplane:audit required", 403)

        try:
            from aragora.control_plane.audit import AuditQuery, AuditAction, ActorType
            from datetime import datetime

            audit_log = self.ctx.get("audit_log")
            if not audit_log:
                return json_response(
                    {
                        "entries": [],
                        "total": 0,
                        "message": "Audit log not configured",
                    }
                )

            # Parse query parameters
            start_time = None
            end_time = None
            if query_params.get("start_time"):
                start_time = datetime.fromisoformat(query_params["start_time"])
            if query_params.get("end_time"):
                end_time = datetime.fromisoformat(query_params["end_time"])

            actions = None
            if query_params.get("actions"):
                action_strs = query_params["actions"].split(",")
                actions = [AuditAction(a.strip()) for a in action_strs if a.strip()]

            actor_types = None
            if query_params.get("actor_types"):
                type_strs = query_params["actor_types"].split(",")
                actor_types = [ActorType(t.strip()) for t in type_strs if t.strip()]

            query = AuditQuery(
                start_time=start_time,
                end_time=end_time,
                actions=actions,
                actor_types=actor_types,
                actor_ids=(
                    query_params.get("actor_ids", "").split(",")
                    if query_params.get("actor_ids")
                    else None
                ),
                resource_types=(
                    query_params.get("resource_types", "").split(",")
                    if query_params.get("resource_types")
                    else None
                ),
                workspace_ids=(
                    query_params.get("workspace_ids", "").split(",")
                    if query_params.get("workspace_ids")
                    else None
                ),
                limit=safe_query_int(query_params, "limit", default=100, max_val=1000),
                offset=safe_query_int(
                    query_params, "offset", default=0, min_val=0, max_val=1000000
                ),
            )

            query_fn = getattr(audit_log, "query", None)
            if query_fn is None:
                entries = []
            else:
                entries = _run_async(query_fn(query))

            return json_response(
                {
                    "entries": [e.to_dict() for e in entries],
                    "total": len(entries),
                    "query": {
                        "limit": query.limit,
                        "offset": query.offset,
                    },
                }
            )
        except ImportError:
            return json_response(
                {
                    "entries": [],
                    "total": 0,
                    "message": "Audit module not available",
                }
            )
        except Exception as e:
            logger.error(f"Error querying audit logs: {e}")
            return error_response(safe_error_message(e, "audit"), 500)

    @api_endpoint(
        method="GET",
        path="/api/control-plane/audit/stats",
        summary="Get audit log statistics",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:audit.read")
    def _handle_get_audit_stats(self) -> HandlerResult:
        """Get audit log statistics."""
        try:
            audit_log = self.ctx.get("audit_log")
            if not audit_log:
                return json_response(
                    {
                        "total_entries": 0,
                        "storage_backend": "none",
                        "message": "Audit log not configured",
                    }
                )

            stats_fn = getattr(audit_log, "get_stats", None)
            return json_response(stats_fn() if stats_fn else {})
        except Exception as e:
            logger.error(f"Error getting audit stats: {e}")
            return error_response(safe_error_message(e, "audit"), 500)

    @api_endpoint(
        method="GET",
        path="/api/control-plane/audit/verify",
        summary="Verify audit log integrity",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:audit.verify")
    def _handle_verify_audit_integrity(
        self, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Verify audit log integrity."""
        # Require authentication for integrity verification
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Check permission for audit access
        if not _get_has_permission()(
            user.role if hasattr(user, "role") else None, "controlplane:audit"
        ):
            return error_response("Permission denied: controlplane:audit required", 403)

        try:
            audit_log = self.ctx.get("audit_log")
            if not audit_log:
                return error_response("Audit log not configured", 503)

            start_seq = safe_query_int(
                query_params, "start_seq", default=0, min_val=0, max_val=9223372036854775807
            )
            end_seq = (
                safe_query_int(
                    query_params, "end_seq", default=0, min_val=0, max_val=9223372036854775807
                )
                if query_params.get("end_seq")
                else None
            )

            verify_fn = getattr(audit_log, "verify_integrity", None)
            if verify_fn is None:
                is_valid = False
            else:
                is_valid = _run_async(verify_fn(start_seq, end_seq))

            return json_response(
                {
                    "valid": is_valid,
                    "start_seq": start_seq,
                    "end_seq": end_seq,
                    "message": (
                        "Integrity verified"
                        if is_valid
                        else "Integrity check failed - possible tampering detected"
                    ),
                }
            )
        except Exception as e:
            logger.error(f"Error verifying audit integrity: {e}")
            return error_response(safe_error_message(e, "audit"), 500)
