"""
Health Check Endpoints.

Provides Kubernetes-compatible health endpoints:
- /healthz - Liveness probe
- /readyz - Readiness probe
- /api/v2/health - Detailed health status
"""

from __future__ import annotations

import os
import platform
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])

# Track server start time
_start_time = time.time()


def _get_uptime() -> str:
    """Get server uptime as human-readable string."""
    uptime_seconds = int(time.time() - _start_time)
    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")

    return " ".join(parts)


@router.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """
    Kubernetes liveness probe.

    Returns 200 if the server is running.
    """
    return {"status": "ok"}


@router.get("/readyz", include_in_schema=False)
async def readyz(request: Request) -> dict[str, str]:
    """
    Kubernetes readiness probe.

    Returns 200 if the server is ready to accept traffic.
    Checks that essential subsystems are initialized.
    """
    ctx = getattr(request.app.state, "context", None)

    if not ctx:
        return {"status": "initializing"}

    # Check essential subsystems
    storage_ready = ctx.get("storage") is not None

    if storage_ready:
        return {"status": "ready"}
    else:
        return {"status": "degraded", "reason": "storage not ready"}


@router.get("/api/v2/health")
async def health_detailed(request: Request) -> dict[str, Any]:
    """
    Detailed health status.

    Returns comprehensive health information including:
    - Server status
    - Subsystem health
    - Uptime and version info
    """
    ctx = getattr(request.app.state, "context", {})

    subsystems: dict[str, dict[str, Any]] = {}

    # Check storage
    storage = ctx.get("storage")
    if storage:
        try:
            count = storage.count_debates() if hasattr(storage, "count_debates") else 0
            subsystems["storage"] = {
                "status": "healthy",
                "debates_count": count,
            }
        except Exception as e:
            subsystems["storage"] = {"status": "unhealthy", "error": str(e)}
    else:
        subsystems["storage"] = {"status": "not_initialized"}

    # Check ELO system
    elo = ctx.get("elo_system")
    if elo:
        subsystems["elo_system"] = {"status": "healthy"}
    else:
        subsystems["elo_system"] = {"status": "not_initialized"}

    # Check RBAC
    rbac = ctx.get("rbac_checker")
    if rbac:
        subsystems["rbac"] = {"status": "healthy"}
    else:
        subsystems["rbac"] = {"status": "not_initialized"}

    # Overall status
    all_healthy = all(
        s.get("status") in ("healthy", "not_initialized")
        for s in subsystems.values()
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": _get_uptime(),
        "version": {
            "api": "2.0.0",
            "server": "fastapi",
            "python": platform.python_version(),
        },
        "environment": os.environ.get("ARAGORA_ENV", "development"),
        "subsystems": subsystems,
    }


@router.get("/api/v2/metrics/summary")
async def metrics_summary(request: Request) -> dict[str, Any]:
    """
    Basic metrics summary.

    For full metrics, use the Prometheus /metrics endpoint.
    """
    ctx = getattr(request.app.state, "context", {})

    metrics: dict[str, Any] = {
        "uptime_seconds": int(time.time() - _start_time),
    }

    # Get debate count if available
    storage = ctx.get("storage")
    if storage and hasattr(storage, "count_debates"):
        try:
            metrics["debates_total"] = storage.count_debates()
        except Exception:
            pass

    return metrics
