"""
Agent health monitoring endpoints.

Provides:
- /api/health/agents - Summary of all registered agent health statuses
- /api/health/agents/{agent_id} - Detailed health for a specific agent
- /api/health/agents/availability - Agent availability overview
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from aragora.rbac.checker import get_permission_checker
from aragora.rbac.models import AuthorizationContext

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)
HEALTH_PERMISSION = "system.health.read"


def _require_health_permission(handler: Any) -> HandlerResult | None:
    """Honor an attached auth_context when these helpers are called directly.

    The main health handler enforces `system.health.read` before routing into
    this module. This secondary guard covers direct helper entrypoints that are
    invoked with an AuthorizationContext-bearing handler object.
    """
    auth_context = getattr(handler, "_auth_context", None)
    if not isinstance(auth_context, AuthorizationContext):
        return None

    decision = get_permission_checker().check_permission(auth_context, HEALTH_PERMISSION)
    if decision.allowed:
        return None

    logger.warning(
        "Permission denied for agent health helper: %s user=%s",
        HEALTH_PERMISSION,
        auth_context.user_id,
    )
    return json_response(
        {
            "error": "Permission denied",
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        },
        status=403,
    )


def _get_watchdog() -> Any | None:
    """Load the watchdog singleton. Returns None on ImportError."""
    from aragora.control_plane.watchdog import get_watchdog

    return get_watchdog()


def agent_health_summary(handler: Any) -> HandlerResult:
    """Get health summary for all registered agents.

    Returns aggregate health status including:
    - total_agents: Number of registered agents
    - healthy: Agents with no issues
    - degraded: Agents with elevated error rates or latency
    - unhealthy: Agents with open circuit breakers or missing heartbeats
    - agents: Per-agent health details

    Returns:
        JSON response with agent health summary
    """
    if permission_error := _require_health_permission(handler):
        return permission_error

    agents: list[dict[str, Any]] = []
    errors: list[str] = []
    watchdog_available = False

    try:
        watchdog = _get_watchdog()
        if watchdog:
            watchdog_available = True
            for agent_name, health in watchdog.get_all_health().items():
                agents.append(
                    {
                        "agent_name": agent_name,
                        "status": _classify_agent_status(health),
                        "last_heartbeat": (
                            health.last_heartbeat.isoformat() + "Z"
                            if health.last_heartbeat
                            else None
                        ),
                        "error_rate": round(health.error_rate, 4),
                        "average_latency_ms": round(health.average_latency_ms, 2),
                        "total_requests": health.total_requests,
                        "failed_requests": health.failed_requests,
                        "circuit_breaker_state": health.circuit_breaker_state,
                        "memory_usage_mb": round(health.memory_usage_mb, 2),
                        "active_issues": len(health.active_issues),
                    }
                )
    except ImportError:
        errors.append("watchdog module not available")
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as e:
        logger.warning("Agent health check failed: %s", e)
        errors.append(f"watchdog error: {type(e).__name__}")

    if not agents and not watchdog_available:
        try:
            from aragora.control_plane.registry import get_default_registry

            registry = get_default_registry()
            if registry:
                for agent_id, _info in (registry._agents or {}).items():
                    agents.append(
                        {
                            "agent_name": agent_id,
                            "status": "unknown",
                            "last_heartbeat": None,
                            "error_rate": 0.0,
                            "average_latency_ms": 0.0,
                            "total_requests": 0,
                            "failed_requests": 0,
                            "circuit_breaker_state": "closed",
                            "memory_usage_mb": 0.0,
                            "active_issues": 0,
                        }
                    )
        except ImportError:
            errors.append("registry module not available")
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as e:
            errors.append(f"registry error: {type(e).__name__}")

    healthy = sum(1 for a in agents if a.get("status") == "healthy")
    degraded = sum(1 for a in agents if a.get("status") == "degraded")
    unhealthy = sum(1 for a in agents if a.get("status") == "unhealthy")

    if unhealthy > 0:
        overall_status = "unhealthy"
    elif degraded > 0:
        overall_status = "degraded"
    elif agents:
        overall_status = "healthy"
    else:
        overall_status = "unknown"

    return json_response(
        {
            "status": overall_status,
            "summary": {
                "total_agents": len(agents),
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
            },
            "agents": agents,
            "errors": errors if errors else None,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }
    )


def agent_health_detail(handler: Any, agent_id: str) -> HandlerResult:
    """Get detailed health status for a specific agent.

    Args:
        handler: HTTP handler instance
        agent_id: Agent identifier to look up

    Returns:
        JSON response with detailed agent health or 404 if not found
    """
    if permission_error := _require_health_permission(handler):
        return permission_error

    try:
        watchdog = _get_watchdog()
        if watchdog:
            all_health = watchdog.get_all_health()
            health = all_health.get(agent_id)
            if health:
                return json_response(
                    {
                        "agent_name": health.agent_name,
                        "status": _classify_agent_status(health),
                        "last_heartbeat": (
                            health.last_heartbeat.isoformat() + "Z"
                            if health.last_heartbeat
                            else None
                        ),
                        "error_rate": round(health.error_rate, 4),
                        "average_latency_ms": round(health.average_latency_ms, 2),
                        "total_requests": health.total_requests,
                        "failed_requests": health.failed_requests,
                        "consecutive_failures": health.consecutive_failures,
                        "total_latency_ms": round(health.total_latency_ms, 2),
                        "circuit_breaker_state": health.circuit_breaker_state,
                        "memory_usage_mb": round(health.memory_usage_mb, 2),
                        "active_issues": [
                            {
                                "severity": issue.severity.name
                                if hasattr(issue.severity, "name")
                                else str(issue.severity),
                                "message": issue.message,
                                "category": issue.category.value
                                if hasattr(issue.category, "value")
                                else str(issue.category),
                            }
                            for issue in health.active_issues
                        ],
                        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    }
                )

        return json_response(
            {
                "error": f"Agent not found: {agent_id}",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            },
            status=404,
        )
    except ImportError:
        return json_response(
            {
                "error": "watchdog module not available",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            },
            status=503,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as e:
        logger.warning("Agent health detail failed for %s: %s", agent_id, e)
        return json_response(
            {
                "error": "Health check failed",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            },
            status=500,
        )


def agent_availability_status(handler: Any) -> HandlerResult:
    """Get agent availability overview.

    Returns a lightweight availability check for all agents
    focused on whether each agent can accept new requests.

    Returns:
        JSON response with availability data
    """
    if permission_error := _require_health_permission(handler):
        return permission_error

    available: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        watchdog = _get_watchdog()
        if watchdog:
            for agent_name, health in watchdog.get_all_health().items():
                entry = {
                    "agent_name": agent_name,
                    "circuit_breaker_state": health.circuit_breaker_state,
                }
                if health.circuit_breaker_state == "open":
                    entry["reason"] = "circuit_breaker_open"
                    unavailable.append(entry)
                elif health.consecutive_failures >= 5:
                    entry["reason"] = "consecutive_failures"
                    unavailable.append(entry)
                else:
                    available.append(entry)
    except ImportError:
        errors.append("watchdog module not available")
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as e:
        logger.warning("Agent availability check failed: %s", e)
        errors.append(f"watchdog error: {type(e).__name__}")

    total = len(available) + len(unavailable)
    if total == 0:
        status = "unknown"
    elif len(unavailable) == 0:
        status = "all_available"
    elif len(available) == 0:
        status = "none_available"
    else:
        status = "partial"

    return json_response(
        {
            "status": status,
            "available_count": len(available),
            "unavailable_count": len(unavailable),
            "available": available,
            "unavailable": unavailable,
            "errors": errors if errors else None,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }
    )


def _classify_agent_status(health: Any) -> str:
    """Classify agent health into healthy/degraded/unhealthy."""
    if health.circuit_breaker_state == "open":
        return "unhealthy"
    if health.consecutive_failures >= 5:
        return "unhealthy"
    if health.error_rate > 0.5:
        return "unhealthy"
    if health.error_rate > 0.1:
        return "degraded"
    if health.consecutive_failures >= 2:
        return "degraded"
    return "healthy"


__all__ = [
    "agent_health_summary",
    "agent_health_detail",
    "agent_availability_status",
]
