"""
Kubernetes liveness and readiness probe implementations.

Provides lightweight probes suitable for K8s deployments:
- /healthz - Liveness probe (is the server alive?)
- /readyz - Fast readiness probe (<10ms, in-memory only)
- /readyz/dependencies - Full dependency validation (slow, 2-5s)

These endpoints are public (no auth required) and designed to be fast.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time
from typing import Any

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)


def liveness_probe(handler: Any) -> HandlerResult:
    """Kubernetes liveness probe - lightweight check that server is alive.

    Returns 200 if the server process is running and can respond.
    This should be very fast and not check external dependencies.
    Used by k8s to determine if the container should be restarted.

    IMPORTANT: Returns 200 even in degraded mode. The container is alive
    and shouldn't be restarted just because of misconfiguration. Use
    /readyz for routing decisions.

    Returns:
        {"status": "ok"} with 200 status
    """
    try:
        from aragora.server.degraded_mode import is_degraded, get_degraded_reason

        if is_degraded():
            return json_response(
                {
                    "status": "ok",
                    "degraded": True,
                    "degraded_reason": get_degraded_reason()[:100],
                    "note": "Server alive but degraded. Check /api/health for details.",
                }
            )
    except ImportError:
        pass

    return json_response({"status": "ok"})


def readiness_probe_fast(handler: Any) -> HandlerResult:
    """Fast Kubernetes readiness probe - check if ready to serve traffic.

    Optimized for K8s probes (<10ms latency requirement).
    Only checks in-memory state, no network calls.

    Returns 200 if critical services are initialized and ready.
    Returns 503 if the service is not ready to accept traffic.

    For full dependency validation, use /readyz/dependencies instead.
    """
    from . import _get_cached_health, _set_cached_health

    # Return cached result if available (5 second cache for K8s probes)
    cached = _get_cached_health("readiness_fast")
    if cached is not None:
        status_code = 200 if cached.get("status") == "ready" else 503
        return json_response(cached, status=status_code)

    start_time = time.time()
    ready = True
    checks: dict[str, Any] = {}

    # Check for degraded mode first - return 503 immediately
    try:
        from aragora.server.degraded_mode import get_degraded_state, is_degraded

        if is_degraded():
            state = get_degraded_state()
            return json_response(
                {
                    "status": "not_ready",
                    "reason": "Server in degraded mode",
                    "degraded": {
                        "error_code": state.error_code.value,
                        "reason": state.reason,
                        "recovery_hint": state.recovery_hint,
                    },
                    "checks": {"degraded_mode": False},
                },
                status=503,
            )
        checks["degraded_mode"] = True
    except ImportError:
        checks["degraded_mode"] = True  # Module not available = not degraded

    # Check server startup completed (in-memory, no I/O)
    try:
        from aragora.server.unified_server import is_server_ready

        startup_complete = is_server_ready()
        checks["startup_complete"] = startup_complete
        if not startup_complete:
            ready = False
    except ImportError:
        checks["startup_complete"] = True  # Module not available = skip check

    # Check handler route index has been populated (in-memory, no I/O)
    try:
        from aragora.server.handler_registry.core import get_route_index

        route_index = get_route_index()
        has_routes = bool(route_index._exact_routes)
        checks["handlers_initialized"] = has_routes
        if not has_routes:
            ready = False
    except ImportError:
        checks["handlers_initialized"] = True  # Module not available = skip check

    # Check storage initialization (fast - no DB queries)
    try:
        storage = handler.get_storage()
        checks["storage_initialized"] = storage is not None
        if not storage:
            # Storage not configured is OK for readiness
            checks["storage_initialized"] = True
    except (OSError, RuntimeError, ValueError):
        checks["storage_initialized"] = False
        ready = False

    # Check ELO system initialization (fast - no DB queries)
    try:
        elo = handler.get_elo_system()
        checks["elo_initialized"] = elo is not None
        if not elo:
            checks["elo_initialized"] = True
    except (OSError, RuntimeError, ValueError):
        checks["elo_initialized"] = False
        ready = False

    # Quick Redis pool check (no network call - just check if pool exists)
    redis_url = os.environ.get("REDIS_URL") or os.environ.get("ARAGORA_REDIS_URL")
    if redis_url:
        try:
            from aragora.cache.redis_cache import get_redis_pool

            pool = get_redis_pool()
            checks["redis_pool"] = pool is not None
        except (ImportError, RuntimeError):
            checks["redis_pool"] = "not_configured"
    else:
        checks["redis_pool"] = "not_configured"

    # Quick database pool check (no network call - just check if pool exists)
    database_url = os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_POSTGRES_DSN")
    if database_url:
        try:
            from aragora.storage.postgres_pool import get_pool

            pool = get_pool()
            checks["db_pool"] = pool is not None
        except (ImportError, RuntimeError):
            checks["db_pool"] = "not_configured"
    else:
        checks["db_pool"] = "not_configured"

    status_code = 200 if ready else 503
    latency_ms = (time.time() - start_time) * 1000
    result = {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
        "latency_ms": round(latency_ms, 2),
        "fast_probe": True,
        "full_validation": "/readyz/dependencies",
    }

    # Cache result
    _set_cached_health("readiness_fast", result)

    return json_response(result, status=status_code)


def readiness_dependencies(handler: Any) -> HandlerResult:
    """Full dependency validation probe - checks all external connections.

    SLOW: May take 2-5 seconds due to network validation.
    Use /readyz for K8s probes instead.

    Returns 200 if all configured dependencies are reachable.
    Returns 503 if any required dependency is unreachable.

    Checks:
    - Degraded mode (server misconfiguration)
    - Storage initialized (if configured)
    - ELO system available (if configured)
    - Redis connectivity (if distributed state required)
    - PostgreSQL connectivity (if required)
    """
    from . import _get_cached_health, _set_cached_health

    # Return cached result if available (1 second cache)
    cached = _get_cached_health("readiness")
    if cached is not None:
        status_code = 200 if cached.get("status") == "ready" else 503
        return json_response(cached, status=status_code)

    start_time = time.time()
    ready = True
    checks: dict[str, Any] = {}

    # Check for degraded mode first - return 503 immediately
    try:
        from aragora.server.degraded_mode import is_degraded, get_degraded_state

        if is_degraded():
            state = get_degraded_state()
            return json_response(
                {
                    "status": "not_ready",
                    "reason": "Server in degraded mode",
                    "degraded": {
                        "error_code": state.error_code.value,
                        "reason": state.reason,
                        "recovery_hint": state.recovery_hint,
                    },
                    "checks": {"degraded_mode": False},
                },
                status=503,
            )
    except ImportError:
        pass

    # Check storage readiness
    try:
        storage = handler.get_storage()
        checks["storage"] = storage is not None
        if not storage:
            # Storage not configured is OK for readiness
            checks["storage"] = True
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("Storage readiness check failed: %s: %s", type(e).__name__, e)
        checks["storage"] = False
        ready = False

    # Check ELO system readiness
    try:
        elo = handler.get_elo_system()
        checks["elo_system"] = elo is not None
        if not elo:
            # ELO not configured is OK for readiness
            checks["elo_system"] = True
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("ELO system readiness check failed: %s: %s", type(e).__name__, e)
        checks["elo_system"] = False
        ready = False

    # Check Redis connectivity (if distributed state required)
    try:
        from aragora.control_plane.leader import is_distributed_state_required
        from aragora.server.startup import validate_redis_connectivity

        distributed_required = is_distributed_state_required()
        redis_url = os.environ.get("REDIS_URL") or os.environ.get("ARAGORA_REDIS_URL")

        if distributed_required and redis_url:
            # Run async validation in sync context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop:
                # Already in async context - schedule task
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, validate_redis_connectivity(timeout_seconds=2.0)
                    )
                    redis_ok, redis_msg = future.result(timeout=3.0)
            else:
                redis_ok, redis_msg = asyncio.run(validate_redis_connectivity(timeout_seconds=2.0))

            checks["redis"] = {"connected": redis_ok, "message": redis_msg}
            if not redis_ok:
                ready = False
        elif redis_url:
            # Redis configured but not required - check but don't fail
            checks["redis"] = {"configured": True, "required": False}
        else:
            checks["redis"] = {"configured": False}

    except ImportError:
        # Modules not available - skip check
        checks["redis"] = {"status": "check_skipped"}
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.warning("Redis connectivity failed: %s: %s", type(e).__name__, e)
        checks["redis"] = {"error": "Redis connectivity failed", "error_type": "connectivity"}
        # Fail readiness for connectivity errors when distributed required
        try:
            if is_distributed_state_required():
                ready = False
        except (ImportError, RuntimeError):
            pass
    except (asyncio.TimeoutError, concurrent.futures.TimeoutError) as e:
        logger.warning("Redis check timed out: %s: %s", type(e).__name__, e)
        checks["redis"] = {"error": "timeout", "error_type": "timeout"}
        try:
            if is_distributed_state_required():
                ready = False
        except (ImportError, RuntimeError):
            pass
    except (RuntimeError, ValueError, TypeError, AttributeError) as e:  # broad catch: last-resort handler
        logger.warning("Redis readiness check failed: %s: %s", type(e).__name__, e)
        checks["redis"] = {"error": "Redis check failed"}
        # Don't fail readiness for Redis errors unless distributed required
        try:
            if is_distributed_state_required():
                ready = False
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug("Error checking distributed state requirement: %s", e)

    # Check PostgreSQL connectivity (if required)
    try:
        from aragora.server.startup import validate_database_connectivity

        database_url = os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_POSTGRES_DSN")
        require_database = os.environ.get("ARAGORA_REQUIRE_DATABASE", "").lower() in (
            "true",
            "1",
            "yes",
        )

        if require_database and database_url:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, validate_database_connectivity(timeout_seconds=2.0)
                    )
                    db_ok, db_msg = future.result(timeout=3.0)
            else:
                db_ok, db_msg = asyncio.run(validate_database_connectivity(timeout_seconds=2.0))

            checks["postgresql"] = {"connected": db_ok, "message": db_msg}
            if not db_ok:
                ready = False
        elif database_url:
            checks["postgresql"] = {"configured": True, "required": False}
        else:
            checks["postgresql"] = {"configured": False}

    except ImportError:
        checks["postgresql"] = {"status": "check_skipped"}
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.warning("PostgreSQL connectivity failed: %s: %s", type(e).__name__, e)
        checks["postgresql"] = {"error": "PostgreSQL connectivity failed", "error_type": "connectivity"}
        if require_database:
            ready = False
    except (asyncio.TimeoutError, concurrent.futures.TimeoutError) as e:
        logger.warning("PostgreSQL check timed out: %s: %s", type(e).__name__, e)
        checks["postgresql"] = {"error": "timeout", "error_type": "timeout"}
        if require_database:
            ready = False
    except (RuntimeError, ValueError, TypeError, AttributeError) as e:  # broad catch: last-resort handler
        logger.warning("PostgreSQL readiness check failed: %s: %s", type(e).__name__, e)
        checks["postgresql"] = {"error": "PostgreSQL check failed"}

    status_code = 200 if ready else 503
    latency_ms = (time.time() - start_time) * 1000
    result = {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
        "latency_ms": round(latency_ms, 2),
    }

    # Cache result for subsequent requests
    _set_cached_health("readiness", result)

    return json_response(result, status=status_code)
