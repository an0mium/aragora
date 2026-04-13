"""
Metrics subsystem health check.

Provides:
- /api/metrics/health - Observability metrics subsystem health
- /api/v1/metrics/health - Observability metrics subsystem health
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)

_CHECK_STATUS_MAP = {
    "ok": "healthy",
    "error": "unhealthy",
    "unavailable": "unavailable",
}


def _component_check(component: dict[str, Any]) -> dict[str, Any]:
    """Adapt component details to the legacy dashboard check shape."""
    check: dict[str, Any] = {
        "status": _CHECK_STATUS_MAP.get(str(component.get("status", "")), "unavailable")
    }
    if "error" in component:
        check["error"] = component["error"]
    return check


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

    checks = {
        component_name: _component_check(component_data)
        for component_name, component_data in components.items()
    }

    return json_response(
        {
            "status": status,
            "metrics_enabled": metrics_enabled,
            "components": components,
            "checks": checks,
            "issues": issues if issues else None,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        status=200,
    )
