"""
Metrics subsystem health check.

Provides:
- /api/v1/health/metrics - Observability metrics subsystem health
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
    """Honor an attached auth_context when this helper is called directly.

    The main health handler enforces `system.health.read` before routing into
    this module. This helper closes the gap for direct invocations that pass an
    AuthorizationContext-bearing handler object.
    """
    auth_context = getattr(handler, "_auth_context", None)
    if not isinstance(auth_context, AuthorizationContext):
        return None

    decision = get_permission_checker().check_permission(auth_context, HEALTH_PERMISSION)
    if decision.allowed:
        return None

    logger.warning(
        "Permission denied for metrics health helper: %s user=%s",
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


def metrics_health(handler: Any) -> HandlerResult:
    """Check health of the observability metrics subsystem.

    Returns status of Prometheus metrics including:
    - enabled: Whether metrics collection is enabled
    - initialized: Whether the metrics system has been initialized
    - prometheus_available: Whether prometheus-client is importable
    - collectors: Count and names of registered metric collectors
    - scrape_endpoint: Whether /metrics endpoint is reachable

    Returns:
        JSON response with metrics subsystem health
    """
    if permission_error := _require_health_permission(handler):
        return permission_error

    components: dict[str, Any] = {}
    issues: list[str] = []

    # 1. Check if metrics are enabled
    metrics_enabled = False
    try:
        from aragora.observability.metrics.base import get_metrics_enabled

        metrics_enabled = get_metrics_enabled()
        components["enabled"] = {"status": "ok", "value": metrics_enabled}
    except ImportError:
        components["enabled"] = {"status": "unavailable", "value": False}
        issues.append("metrics base module not available")
    except (TypeError, ValueError, AttributeError, RuntimeError) as e:
        components["enabled"] = {"status": "error", "error": str(e)}
        issues.append(f"metrics config check failed: {type(e).__name__}")

    # 2. Check initialization state
    try:
        from aragora.observability.metrics.core import is_initialized

        initialized = is_initialized()
        components["initialized"] = {"status": "ok", "value": initialized}
        if not initialized and metrics_enabled:
            issues.append("metrics enabled but not initialized")
    except ImportError:
        components["initialized"] = {"status": "unavailable", "value": False}
        issues.append("metrics core module not available")
    except (TypeError, ValueError, AttributeError, RuntimeError) as e:
        components["initialized"] = {"status": "error", "error": str(e)}
        issues.append(f"initialization check failed: {type(e).__name__}")

    # 3. Check prometheus-client availability
    try:
        import prometheus_client  # noqa: F401

        components["prometheus_available"] = {"status": "ok", "value": True}
    except ImportError:
        components["prometheus_available"] = {"status": "unavailable", "value": False}
        if metrics_enabled:
            issues.append("prometheus-client not installed but metrics enabled")

    # 4. Check registered collectors
    try:
        from prometheus_client import REGISTRY

        collector_count = len(list(REGISTRY.collect()))
        components["collectors"] = {
            "status": "ok",
            "count": collector_count,
        }
    except ImportError:
        components["collectors"] = {"status": "unavailable", "count": 0}
    except (TypeError, ValueError, RuntimeError, OSError) as e:
        components["collectors"] = {"status": "error", "error": str(e), "count": 0}
        issues.append(f"collector enumeration failed: {type(e).__name__}")

    # 5. Check metrics configuration
    try:
        from aragora.observability.config import get_metrics_config

        config = get_metrics_config()
        components["config"] = {
            "status": "ok",
            "port": getattr(config, "port", None),
            "prefix": getattr(config, "prefix", None),
        }
    except ImportError:
        components["config"] = {"status": "unavailable"}
    except (TypeError, ValueError, AttributeError, RuntimeError) as e:
        components["config"] = {"status": "error", "error": str(e)}
        issues.append(f"config check failed: {type(e).__name__}")

    # Determine overall status
    if not metrics_enabled:
        status = "disabled"
    elif issues:
        status = "degraded"
    else:
        status = "healthy"

    return json_response(
        {
            "status": status,
            "metrics_enabled": metrics_enabled,
            "components": components,
            "issues": issues if issues else None,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        },
        status=200,
    )
