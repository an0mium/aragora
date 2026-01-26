"""
Health and readiness endpoint handlers.

Endpoints:
- GET /healthz - Kubernetes liveness probe (lightweight, public)
- GET /readyz - Kubernetes readiness probe (checks dependencies, public)
- GET /api/health - Comprehensive health check (requires auth)
- GET /api/health/detailed - Detailed health with observer metrics (requires auth)
- GET /api/health/deep - Deep health check with all external dependencies (requires auth)

Extracted from system.py for better modularity.
RBAC: K8s probes are public; detailed endpoints require system.health.read permission.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..base import (
    HandlerResult,
    json_response,
    error_response,
)
from ..secure import SecureHandler, ForbiddenError, UnauthorizedError
from .health_utils import (
    check_ai_providers_health,
    check_filesystem_health,
    check_redis_health,
    check_security_services,
)

logger = logging.getLogger(__name__)

# Server start time for uptime tracking
_SERVER_START_TIME = time.time()

# Health check cache for performance
# K8s probes need 5s TTL to ensure fast responses; detailed checks use 2s
_HEALTH_CACHE: Dict[str, Any] = {}
_HEALTH_CACHE_TTL = 5.0  # seconds for K8s probes (liveness/readiness)
_HEALTH_CACHE_TTL_DETAILED = 2.0  # seconds for detailed health checks
_HEALTH_CACHE_TIMESTAMPS: Dict[str, float] = {}


def _get_cached_health(key: str) -> Optional[Dict[str, Any]]:
    """Get cached health result if still valid."""
    if key in _HEALTH_CACHE:
        cached_time = _HEALTH_CACHE_TIMESTAMPS.get(key, 0)
        if time.time() - cached_time < _HEALTH_CACHE_TTL:
            return _HEALTH_CACHE[key]
    return None


def _set_cached_health(key: str, value: Dict[str, Any]) -> None:
    """Cache health check result."""
    _HEALTH_CACHE[key] = value
    _HEALTH_CACHE_TIMESTAMPS[key] = time.time()


class HealthHandler(SecureHandler):
    """Handler for health and readiness endpoints.

    RBAC Policy:
    - /healthz, /readyz: Public (K8s probes, no auth required)
    - All other endpoints: Require authentication and system.health.read permission
    """

    ROUTES = [
        "/healthz",
        "/readyz",
        "/readyz/dependencies",  # Full dependency validation (slow)
        # v1 routes
        "/api/v1/health",
        "/api/v1/health/detailed",
        "/api/v1/health/deep",
        "/api/v1/health/stores",
        "/api/v1/health/sync",
        "/api/v1/health/circuits",
        "/api/v1/health/slow-debates",
        "/api/v1/health/cross-pollination",
        "/api/v1/health/knowledge-mound",
        "/api/v1/health/decay",  # Confidence decay scheduler status
        "/api/v1/health/encryption",
        "/api/v1/health/database",
        "/api/v1/health/platform",
        "/api/v1/platform/health",
        "/api/v1/diagnostics",
        "/api/v1/diagnostics/deployment",
        # Non-v1 routes (for backward compatibility)
        "/api/health",
        "/api/health/detailed",
        "/api/health/deep",
        "/api/health/stores",
        "/api/diagnostics",
        "/api/diagnostics/deployment",
    ]

    # K8s probe routes that remain public (no auth required)
    PUBLIC_ROUTES = {"/healthz", "/readyz", "/readyz/dependencies"}

    # Permission required for protected health endpoints
    HEALTH_PERMISSION = "system.health.read"
    RESOURCE_TYPE = "health"

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    async def handle(  # type: ignore[override]
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route health endpoint requests with RBAC for non-public routes."""
        # K8s probes are public - no auth required
        if path not in self.PUBLIC_ROUTES:
            # All other health endpoints require authentication and permission
            try:
                auth_context = await self.get_auth_context(handler, require_auth=True)
                self.check_permission(auth_context, self.HEALTH_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                logger.warning(f"Health endpoint access denied: {e}")
                return error_response(str(e), 403)

        # Normalize path for routing (support both v1 and non-v1)
        normalized = path.replace("/api/v1/", "/api/")

        handlers = {
            "/healthz": self._liveness_probe,
            "/readyz": self._readiness_probe_fast,  # Fast check for K8s (<10ms)
            "/readyz/dependencies": self._readiness_dependencies,  # Full validation (slow)
            "/api/health": self._health_check,
            "/api/health/detailed": self._detailed_health_check,
            "/api/health/deep": self._deep_health_check,
            "/api/health/stores": self._database_stores_health,
            "/api/health/sync": self._sync_status,
            "/api/health/circuits": self._circuit_breakers_status,
            "/api/health/slow-debates": self._slow_debates_status,
            "/api/health/cross-pollination": self._cross_pollination_health,
            "/api/health/knowledge-mound": self._knowledge_mound_health,
            "/api/health/database": self._database_schema_health,
            "/api/health/platform": self._platform_health,
            "/api/platform/health": self._platform_health,
            "/api/diagnostics": self._deployment_diagnostics,
            "/api/diagnostics/deployment": self._deployment_diagnostics,
        }

        endpoint_handler = handlers.get(normalized)
        if endpoint_handler:
            return endpoint_handler()
        return None

    def _liveness_probe(self) -> HandlerResult:
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

    def _readiness_probe_fast(self) -> HandlerResult:
        """Fast Kubernetes readiness probe - check if ready to serve traffic.

        Optimized for K8s probes (<10ms latency requirement).
        Only checks in-memory state, no network calls.

        Returns 200 if critical services are initialized and ready.
        Returns 503 if the service is not ready to accept traffic.

        For full dependency validation, use /readyz/dependencies instead.
        """
        import os

        # Return cached result if available (5 second cache for K8s probes)
        cached = _get_cached_health("readiness_fast")
        if cached is not None:
            status_code = 200 if cached.get("status") == "ready" else 503
            return json_response(cached, status=status_code)

        start_time = time.time()
        ready = True
        checks: Dict[str, Any] = {}

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

        # Check storage initialization (fast - no DB queries)
        try:
            storage = self.get_storage()
            checks["storage_initialized"] = storage is not None
            if not storage:
                # Storage not configured is OK for readiness
                checks["storage_initialized"] = True
        except (OSError, RuntimeError, ValueError):
            checks["storage_initialized"] = False
            ready = False

        # Check ELO system initialization (fast - no DB queries)
        try:
            elo = self.get_elo_system()
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

    def _readiness_dependencies(self) -> HandlerResult:
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
        import os

        # Return cached result if available (1 second cache)
        cached = _get_cached_health("readiness")
        if cached is not None:
            status_code = 200 if cached.get("status") == "ready" else 503
            return json_response(cached, status=status_code)

        start_time = time.time()
        ready = True
        checks: Dict[str, Any] = {}

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
            storage = self.get_storage()
            checks["storage"] = storage is not None
            if not storage:
                # Storage not configured is OK for readiness
                checks["storage"] = True
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"Storage readiness check failed: {type(e).__name__}: {e}")
            checks["storage"] = False
            ready = False

        # Check ELO system readiness
        try:
            elo = self.get_elo_system()
            checks["elo_system"] = elo is not None
            if not elo:
                # ELO not configured is OK for readiness
                checks["elo_system"] = True
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"ELO system readiness check failed: {type(e).__name__}: {e}")
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
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, validate_redis_connectivity(timeout_seconds=2.0)
                        )
                        redis_ok, redis_msg = future.result(timeout=3.0)
                else:
                    redis_ok, redis_msg = asyncio.run(
                        validate_redis_connectivity(timeout_seconds=2.0)
                    )

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
            logger.warning(f"Redis connectivity failed: {type(e).__name__}: {e}")
            checks["redis"] = {"error": str(e)[:80], "error_type": "connectivity"}
            # Fail readiness for connectivity errors when distributed required
            try:
                if is_distributed_state_required():
                    ready = False
            except (ImportError, RuntimeError):
                pass
        except (asyncio.TimeoutError, concurrent.futures.TimeoutError) as e:
            logger.warning(f"Redis check timed out: {type(e).__name__}: {e}")
            checks["redis"] = {"error": "timeout", "error_type": "timeout"}
            try:
                if is_distributed_state_required():
                    ready = False
            except (ImportError, RuntimeError):
                pass
        except Exception as e:
            logger.warning(f"Redis readiness check failed: {type(e).__name__}: {e}")
            checks["redis"] = {"error": str(e)[:80]}
            # Don't fail readiness for Redis errors unless distributed required
            try:
                if is_distributed_state_required():
                    ready = False
            except (ImportError, RuntimeError, AttributeError) as e:
                logger.debug(f"Error checking distributed state requirement: {e}")

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
                    import concurrent.futures

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
            logger.warning(f"PostgreSQL connectivity failed: {type(e).__name__}: {e}")
            checks["postgresql"] = {"error": str(e)[:80], "error_type": "connectivity"}
            if require_database:
                ready = False
        except (asyncio.TimeoutError, concurrent.futures.TimeoutError) as e:
            logger.warning(f"PostgreSQL check timed out: {type(e).__name__}: {e}")
            checks["postgresql"] = {"error": "timeout", "error_type": "timeout"}
            if require_database:
                ready = False
        except Exception as e:
            logger.warning(f"PostgreSQL readiness check failed: {type(e).__name__}: {e}")
            checks["postgresql"] = {"error": str(e)[:80]}

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

    def _health_check(self) -> HandlerResult:
        """Comprehensive health check for k8s/docker deployments.

        Returns 200 when all critical services are healthy, 503 when degraded.
        Suitable for kubernetes liveness/readiness probes.

        Checks:
        - degraded_mode: Server configuration and startup status
        - database: Can execute a query
        - storage: Debate storage is initialized
        - elo_system: ELO ranking system is available
        - nomic_dir: Nomic state directory exists
        - filesystem: Can write to data directory
        - redis: Cache connectivity (if configured)
        - ai_providers: API key availability
        """
        checks: Dict[str, Dict[str, Any]] = {}
        all_healthy = True
        start_time = time.time()

        # Check for degraded mode first - this is the most critical check
        try:
            from aragora.server.degraded_mode import is_degraded, get_degraded_state

            if is_degraded():
                state = get_degraded_state()
                checks["degraded_mode"] = {
                    "healthy": False,
                    "status": "degraded",
                    "reason": state.reason,
                    "error_code": state.error_code.value,
                    "recovery_hint": state.recovery_hint,
                    "degraded_since": state.timestamp,
                }
                all_healthy = False
            else:
                checks["degraded_mode"] = {"healthy": True, "status": "normal"}
        except ImportError:
            checks["degraded_mode"] = {"healthy": True, "status": "module_not_available"}

        # Check database connectivity
        db_start = time.time()
        try:
            storage = self.get_storage()
            if storage is not None:
                # Try to execute a simple query to verify DB is responsive
                storage.list_recent(limit=1)
                db_latency = round((time.time() - db_start) * 1000, 2)
                checks["database"] = {"healthy": True, "latency_ms": db_latency}
            else:
                # Storage is optional - server functions without it (degraded mode)
                checks["database"] = {
                    "healthy": True,  # Downgrade to warning, not failure
                    "warning": "Storage not initialized (degraded mode)",
                    "initialized": False,
                }
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as e:
            # Database errors are non-critical - server still functions for debates
            checks["database"] = {
                "healthy": True,  # Downgrade to warning
                "warning": f"{type(e).__name__}: {str(e)[:80]}",
                "initialized": False,
            }

        # Check ELO system
        try:
            elo = self.get_elo_system()
            if elo is not None:
                # Verify ELO system is functional
                elo.get_leaderboard(limit=1)
                checks["elo_system"] = {"healthy": True}
            else:
                # ELO is optional - service functions without it
                checks["elo_system"] = {
                    "healthy": True,  # Downgrade to warning, not failure
                    "warning": "ELO system not initialized",
                    "initialized": False,
                }
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            # ELO errors are non-critical - service still functions
            checks["elo_system"] = {
                "healthy": True,  # Downgrade to warning
                "warning": f"{type(e).__name__}: {str(e)[:80]}",
                "initialized": False,
            }

        # Check nomic directory
        nomic_dir = self.get_nomic_dir()
        if nomic_dir is not None and nomic_dir.exists():
            checks["nomic_dir"] = {"healthy": True, "path": str(nomic_dir)}
        else:
            checks["nomic_dir"] = {"healthy": False, "error": "Directory not found"}
            # Non-critical - don't fail health check for this
            checks["nomic_dir"]["healthy"] = True  # Downgrade to warning
            checks["nomic_dir"]["warning"] = "Nomic directory not configured"

        # Check filesystem write access
        checks["filesystem"] = self._check_filesystem_health()
        if not checks["filesystem"]["healthy"]:
            all_healthy = False

        # Check Redis connectivity (if configured)
        checks["redis"] = self._check_redis_health()
        # Redis is optional - don't fail if not configured
        if checks["redis"].get("error") and checks["redis"].get("configured", False):
            all_healthy = False

        # Check AI provider availability
        checks["ai_providers"] = self._check_ai_providers_health()
        # At least one provider should be available
        if not checks["ai_providers"].get("any_available", False):
            checks["ai_providers"]["warning"] = "No AI providers configured"
            # Don't fail health check - just warn

        # Check WebSocket manager (if available in context)
        ws_manager = self.ctx.get("ws_manager")
        if ws_manager is not None:
            try:
                client_count = len(getattr(ws_manager, "clients", []))
                checks["websocket"] = {"healthy": True, "active_clients": client_count}
            except (AttributeError, TypeError) as e:
                checks["websocket"] = {
                    "healthy": False,
                    "error": f"{type(e).__name__}: {str(e)[:80]}",
                }
        else:
            # WebSocket runs as separate aiohttp service
            checks["websocket"] = {"healthy": True, "note": "Managed by separate aiohttp server"}

        # Check circuit breaker status
        try:
            from aragora.resilience import get_circuit_breaker_metrics

            cb_metrics = get_circuit_breaker_metrics()
            checks["circuit_breakers"] = {
                "healthy": cb_metrics.get("summary", {}).get("open_count", 0) < 3,
                "open": cb_metrics.get("summary", {}).get("open_count", 0),
                "half_open": cb_metrics.get("summary", {}).get("half_open_count", 0),
                "closed": cb_metrics.get("summary", {}).get("closed_count", 0),
            }
            if checks["circuit_breakers"]["open"] >= 3:
                all_healthy = False
        except ImportError:
            checks["circuit_breakers"] = {"healthy": True, "status": "module_not_available"}
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Circuit breaker metrics access error: {type(e).__name__}: {e}")
            checks["circuit_breakers"] = {"healthy": True, "error": str(e)[:80]}
        except Exception as e:
            checks["circuit_breakers"] = {"healthy": True, "error": str(e)[:80]}

        # Check rate limiter status
        try:
            from aragora.server.auth import auth_config

            rl_stats = auth_config.get_rate_limit_stats()
            checks["rate_limiters"] = {
                "healthy": True,
                "active_ips": rl_stats.get("ip_entries", 0),
                "active_tokens": rl_stats.get("token_entries", 0),
                "revoked_tokens": rl_stats.get("revoked_tokens", 0),
            }
        except ImportError:
            checks["rate_limiters"] = {"healthy": True, "status": "module_not_available"}
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Rate limiter stats access error: {type(e).__name__}: {e}")
            checks["rate_limiters"] = {"healthy": True, "error": str(e)[:80]}
        except Exception as e:
            checks["rate_limiters"] = {"healthy": True, "error": str(e)[:80]}

        # Check security services
        checks["security_services"] = self._check_security_services()
        # Security service unavailability in production is a warning
        import os as _os

        if _os.environ.get("ARAGORA_ENV") == "production":
            if not checks["security_services"].get("encryption_configured"):
                all_healthy = False

        # Calculate response time and uptime
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        uptime_seconds = int(time.time() - _SERVER_START_TIME)

        status_code = 200 if all_healthy else 503

        # Get version from package
        try:
            from aragora import __version__

            version = __version__
        except (ImportError, AttributeError):
            version = "unknown"

        health = {
            "status": "healthy" if all_healthy else "degraded",
            "version": version,
            "uptime_seconds": uptime_seconds,
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "response_time_ms": response_time_ms,
        }

        return json_response(health, status=status_code)

    def _check_filesystem_health(self) -> Dict[str, Any]:
        """Check filesystem write access to data directory."""
        nomic_dir = self.get_nomic_dir()
        return check_filesystem_health(nomic_dir)

    def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity if configured."""
        return check_redis_health()

    def _check_ai_providers_health(self) -> Dict[str, Any]:
        """Check AI provider API key availability."""
        return check_ai_providers_health()

    def _check_security_services(self) -> Dict[str, Any]:
        """Check security services health."""
        return check_security_services()

    def _detailed_health_check(self) -> HandlerResult:
        """Return detailed health status with system observer metrics.

        Includes:
        - Basic health components
        - Agent success/failure rates (from SimpleObserver)
        - Null byte incidents
        - Timeout incidents
        - Memory stats
        - Maintenance status
        """
        nomic_dir = self.get_nomic_dir()
        storage = self.get_storage()
        elo = self.get_elo_system()

        health: dict[str, object] = {
            "status": "healthy",
            "components": {
                "storage": storage is not None,
                "elo_system": elo is not None,
                "nomic_dir": nomic_dir is not None and nomic_dir.exists() if nomic_dir else False,
            },
            "version": "1.0.0",
            "warnings": [],
        }

        # Add observer metrics if available
        try:
            from aragora.monitoring.simple_observer import SimpleObserver

            log_path = str(nomic_dir / "system_health.log") if nomic_dir else "system_health.log"
            observer = SimpleObserver(log_file=log_path)
            observer_report = observer.get_report()

            if "error" not in observer_report:
                health["observer"] = observer_report

                # Set status based on failure rate thresholds
                failure_rate = observer_report.get("failure_rate", 0)
                if failure_rate > 0.5:
                    health["status"] = "degraded"
                    if "warnings" not in health:
                        health["warnings"] = []
                    warnings_list = health["warnings"]
                    if isinstance(warnings_list, list):
                        warnings_list.append(f"High failure rate: {failure_rate:.1%}")
                elif failure_rate > 0.3:
                    if "warnings" not in health:
                        health["warnings"] = []
                    warnings_list = health["warnings"]
                    if isinstance(warnings_list, list):
                        warnings_list.append(f"Elevated failure rate: {failure_rate:.1%}")

        except ImportError:
            health["observer"] = {"status": "unavailable", "reason": "module not found"}
        except (OSError, ValueError, KeyError, AttributeError) as e:
            health["observer"] = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:80]}"}

        # Add maintenance stats if available
        try:
            if nomic_dir:
                from aragora.maintenance import DatabaseMaintenance

                maintenance = DatabaseMaintenance(nomic_dir)
                health["maintenance"] = maintenance.get_stats()
        except ImportError:
            pass
        except (OSError, RuntimeError) as e:
            health["maintenance"] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}

        # Check for SQLite in production (scalability warning)
        import os

        env_mode = os.environ.get("ARAGORA_ENV", os.environ.get("NODE_ENV", "development"))
        database_url = os.environ.get("DATABASE_URL", "")
        is_production = env_mode.lower() == "production"
        is_sqlite = (
            not database_url
            or "sqlite" in database_url.lower()
            or not any(db in database_url.lower() for db in ["postgres", "mysql", "mariadb"])
        )

        if is_production and is_sqlite:
            warnings_list = health.get("warnings", [])
            if isinstance(warnings_list, list):
                warnings_list.append(
                    "SQLite detected in production. For multi-replica deployments, migrate to PostgreSQL."
                )
                health["warnings"] = warnings_list
            health["database"] = {
                "type": "sqlite",
                "production_ready": False,
                "recommendation": "Set DATABASE_URL to a PostgreSQL connection string",
            }
        else:
            health["database"] = {
                "type": "postgresql" if "postgres" in database_url.lower() else "unknown",
                "production_ready": True,
            }

        # Add memory stats if available
        try:
            import psutil

            process = psutil.Process()
            health["memory"] = {
                "rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                "percent": round(process.memory_percent(), 2),
            }
        except ImportError:
            pass
        except OSError as e:
            logger.debug(f"Could not get memory stats: {type(e).__name__}: {e}")

        # Add HTTP connector status (for API agent calls)
        try:
            from aragora.agents.api_agents.common import get_shared_connector

            connector = get_shared_connector()
            health["http_connector"] = {
                "status": "healthy" if not connector.closed else "closed",
                "closed": connector.closed,
            }
            if connector.closed:
                health["status"] = "degraded"
                warnings_list = health.get("warnings", [])
                if isinstance(warnings_list, list):
                    warnings_list.append("HTTP connector is closed")
                    health["warnings"] = warnings_list
        except ImportError:
            health["http_connector"] = {"status": "unavailable", "reason": "module not found"}
        except (AttributeError, RuntimeError) as e:
            health["http_connector"] = {
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # Add export cache status
        try:
            from aragora.visualization.exporter import _export_cache, _export_cache_lock

            with _export_cache_lock:
                cache_size = len(_export_cache)
            health["export_cache"] = {
                "status": "healthy",
                "entries": cache_size,
            }
        except ImportError:
            health["export_cache"] = {"status": "unavailable"}
        except (RuntimeError, AttributeError) as e:
            health["export_cache"] = {
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # Add handler cache status
        try:
            from aragora.server.handlers.admin.cache import get_cache_stats

            cache_stats = get_cache_stats()
            health["handler_cache"] = {
                "status": "healthy",
                **cache_stats,
            }
        except ImportError:
            health["handler_cache"] = {"status": "unavailable"}
        except (RuntimeError, AttributeError, KeyError) as e:
            health["handler_cache"] = {
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        return json_response(health)

    def _database_schema_health(self) -> HandlerResult:
        """Check health of consolidated database schema.

        Validates that all required tables exist in consolidated databases:
        - core.db: debates, traces, tournaments, embeddings, positions
        - memory.db: continuum_memory, consensus, critiques, patterns
        - analytics.db: ratings, matches, insights, predictions
        - agents.db: personas, genomes, populations, genesis_events

        This endpoint helps diagnose schema issues after migration.

        Returns:
            JSON with database health status, missing tables, and validation errors.
        """
        try:
            from aragora.persistence.validator import get_database_health

            health = get_database_health()
            status_code = 200 if health["status"] == "healthy" else 503
            return json_response(health, status=status_code)
        except ImportError:
            return json_response(
                {
                    "status": "unavailable",
                    "error": "Database validator not available",
                },
                status=503,
            )
        except Exception as e:
            logger.exception(f"Database schema health check failed: {e}")
            return json_response(
                {
                    "status": "error",
                    "error": str(e),
                },
                status=500,
            )

    def _database_stores_health(self) -> HandlerResult:
        """Check health of all database stores.

        Returns detailed status for each database store:
        - debate_storage: Main debate persistence (SQLite/Supabase)
        - elo_system: Agent rankings database
        - insight_store: Debate insights database
        - flip_detector: Flip detection database
        - consensus_memory: Consensus patterns database
        - user_store: User and organization data

        This endpoint helps diagnose which specific stores are
        initialized, connected, and functioning.
        """
        stores: Dict[str, Dict[str, Any]] = {}
        all_healthy = True
        start_time = time.time()

        # 1. Debate Storage
        try:
            import sqlite3

            storage = self.get_storage()
            if storage is not None:
                count = len(storage.list_recent(limit=1))
                stores["debate_storage"] = {
                    "healthy": True,
                    "status": "connected",
                    "type": type(storage).__name__,
                }
            else:
                stores["debate_storage"] = {
                    "healthy": True,
                    "status": "not_initialized",
                    "hint": "Will auto-create on first debate",
                }
        except (sqlite3.Error, OSError, IOError) as e:
            logger.warning(f"Debate storage database error: {type(e).__name__}: {e}")
            stores["debate_storage"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
                "error_type": "database",
            }
            all_healthy = False
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Debate storage data access error: {type(e).__name__}: {e}")
            stores["debate_storage"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
                "error_type": "data_access",
            }
            all_healthy = False
        except Exception as e:
            stores["debate_storage"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }
            all_healthy = False

        # 2. ELO System
        try:
            elo = self.get_elo_system()
            if elo is not None:
                leaderboard = elo.get_leaderboard(limit=5)
                stores["elo_system"] = {
                    "healthy": True,
                    "status": "connected",
                    "agent_count": len(leaderboard),
                }
            else:
                stores["elo_system"] = {
                    "healthy": True,
                    "status": "not_initialized",
                    "hint": "Run: python scripts/seed_agents.py",
                }
        except (sqlite3.Error, OSError, IOError) as e:
            logger.warning(f"ELO system database error: {type(e).__name__}: {e}")
            stores["elo_system"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
                "error_type": "database",
            }
            all_healthy = False
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"ELO system data access error: {type(e).__name__}: {e}")
            stores["elo_system"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
                "error_type": "data_access",
            }
            all_healthy = False
        except Exception as e:
            stores["elo_system"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }
            all_healthy = False

        # 3. Insight Store
        try:
            insight_store = self.ctx.get("insight_store")
            if insight_store is not None:
                stores["insight_store"] = {
                    "healthy": True,
                    "status": "connected",
                    "type": type(insight_store).__name__,
                }
            else:
                stores["insight_store"] = {
                    "healthy": True,
                    "status": "not_initialized",
                    "hint": "Will auto-create on first insight",
                }
        except Exception as e:
            stores["insight_store"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

        # 4. Flip Detector
        try:
            flip_detector = self.ctx.get("flip_detector")
            if flip_detector is not None:
                stores["flip_detector"] = {
                    "healthy": True,
                    "status": "connected",
                    "type": type(flip_detector).__name__,
                }
            else:
                stores["flip_detector"] = {
                    "healthy": True,
                    "status": "not_initialized",
                }
        except Exception as e:
            stores["flip_detector"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

        # 5. User Store
        try:
            user_store = self.ctx.get("user_store")
            if user_store is not None:
                stores["user_store"] = {
                    "healthy": True,
                    "status": "connected",
                    "type": type(user_store).__name__,
                }
            else:
                stores["user_store"] = {
                    "healthy": True,
                    "status": "not_initialized",
                }
        except Exception as e:
            stores["user_store"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

        # 6. Consensus Memory
        try:
            from aragora.memory.consensus import ConsensusMemory  # noqa: F401

            nomic_dir = self.get_nomic_dir()
            if nomic_dir is not None:
                consensus_path = nomic_dir / "consensus_memory.db"
                if consensus_path.exists():
                    stores["consensus_memory"] = {
                        "healthy": True,
                        "status": "exists",
                        "path": str(consensus_path),
                    }
                else:
                    stores["consensus_memory"] = {
                        "healthy": True,
                        "status": "not_initialized",
                        "hint": "Run: python scripts/seed_consensus.py",
                    }
            else:
                stores["consensus_memory"] = {
                    "healthy": True,
                    "status": "nomic_dir_not_set",
                }
        except ImportError:
            stores["consensus_memory"] = {
                "healthy": True,
                "status": "module_not_available",
            }
        except Exception as e:
            stores["consensus_memory"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

        # 7. Agent Metadata (from seed script)
        try:
            nomic_dir = self.get_nomic_dir()
            if nomic_dir is not None:
                elo_path = nomic_dir / "elo.db"
                if elo_path.exists():
                    import sqlite3

                    conn = sqlite3.connect(elo_path)
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_metadata'"
                    )
                    has_metadata = cursor.fetchone() is not None
                    if has_metadata:
                        cursor = conn.execute("SELECT COUNT(*) FROM agent_metadata")
                        count = cursor.fetchone()[0]
                        stores["agent_metadata"] = {
                            "healthy": True,
                            "status": "connected",
                            "agent_count": count,
                        }
                    else:
                        stores["agent_metadata"] = {
                            "healthy": True,
                            "status": "table_not_exists",
                            "hint": "Run: python scripts/seed_agents.py --with-metadata",
                        }
                    conn.close()
                else:
                    stores["agent_metadata"] = {
                        "healthy": True,
                        "status": "database_not_exists",
                    }
            else:
                stores["agent_metadata"] = {
                    "healthy": True,
                    "status": "nomic_dir_not_set",
                }
        except Exception as e:
            stores["agent_metadata"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return json_response(
            {
                "status": "healthy" if all_healthy else "degraded",
                "stores": stores,
                "elapsed_ms": elapsed_ms,
                "summary": {
                    "total": len(stores),
                    "healthy": sum(1 for s in stores.values() if s.get("healthy", False)),
                    "connected": sum(1 for s in stores.values() if s.get("status") == "connected"),
                    "not_initialized": sum(
                        1 for s in stores.values() if s.get("status") == "not_initialized"
                    ),
                },
            }
        )

    def _deep_health_check(self) -> HandlerResult:
        """Deep health check - verifies all external dependencies.

        This is the most comprehensive health check, suitable for:
        - Pre-deployment validation
        - Debugging connectivity issues
        - Monitoring dashboards

        Checks:
        - All basic health checks
        - Supabase connectivity
        - User store database
        - Billing system
        - Disk space
        - Memory usage
        - CPU load
        """
        all_healthy = True
        checks: Dict[str, Dict[str, Any]] = {}
        start_time = time.time()
        warnings: list[str] = []

        # 1. Database/Storage
        try:
            storage = self.get_storage()
            if storage is not None:
                storage.list_recent(limit=1)
                checks["storage"] = {"healthy": True, "status": "connected"}
            else:
                checks["storage"] = {"healthy": True, "status": "not_configured"}
        except Exception as e:
            checks["storage"] = {"healthy": False, "error": f"{type(e).__name__}: {str(e)[:80]}"}
            all_healthy = False

        # 2. ELO System
        try:
            elo = self.get_elo_system()
            if elo is not None:
                elo.get_leaderboard(limit=1)
                checks["elo_system"] = {"healthy": True, "status": "connected"}
            else:
                checks["elo_system"] = {"healthy": True, "status": "not_configured"}
        except Exception as e:
            checks["elo_system"] = {"healthy": False, "error": f"{type(e).__name__}: {str(e)[:80]}"}
            all_healthy = False

        # 3. Supabase
        try:
            from aragora.persistence.supabase_client import SupabaseClient

            supabase_wrapper = SupabaseClient()
            if supabase_wrapper.is_configured and supabase_wrapper.client is not None:
                ping_start = time.time()
                supabase_wrapper.client.table("debates").select("id").limit(1).execute()
                ping_latency = round((time.time() - ping_start) * 1000, 2)
                checks["supabase"] = {
                    "healthy": True,
                    "status": "connected",
                    "latency_ms": ping_latency,
                }
            else:
                checks["supabase"] = {"healthy": True, "status": "not_configured"}
        except ImportError:
            checks["supabase"] = {"healthy": True, "status": "module_not_available"}
        except Exception as e:
            error_msg = str(e)[:80]
            checks["supabase"] = {"healthy": True, "status": "error", "warning": error_msg}
            warnings.append(f"Supabase: {error_msg}")

        # 4. User Store
        try:
            user_store = getattr(self.__class__, "user_store", None) or self.ctx.get("user_store")
            if user_store is not None:
                from aragora.storage.user_store import UserStore

                if isinstance(user_store, UserStore):
                    user_store.get_user_by_email("__health_check_nonexistent__")
                    checks["user_store"] = {"healthy": True, "status": "connected"}
            else:
                checks["user_store"] = {"healthy": True, "status": "not_configured"}
        except Exception as e:
            checks["user_store"] = {"healthy": False, "error": f"{type(e).__name__}: {str(e)[:80]}"}
            all_healthy = False

        # 5. Billing System
        try:
            from aragora.billing.stripe_client import StripeClient

            stripe_client = StripeClient()
            if stripe_client._is_configured():
                checks["billing"] = {"healthy": True, "status": "configured"}
            else:
                checks["billing"] = {"healthy": True, "status": "not_configured"}
        except ImportError:
            checks["billing"] = {"healthy": True, "status": "module_not_available"}
        except Exception as e:
            checks["billing"] = {"healthy": True, "status": "error", "warning": str(e)[:80]}

        # 6. Redis
        checks["redis"] = self._check_redis_health()
        if checks["redis"].get("configured") and not checks["redis"].get("healthy", True):
            all_healthy = False

        # 7. AI Providers
        checks["ai_providers"] = self._check_ai_providers_health()
        if not checks["ai_providers"].get("any_available", False):
            warnings.append("No AI providers configured")

        # 8. Filesystem
        checks["filesystem"] = self._check_filesystem_health()
        if not checks["filesystem"]["healthy"]:
            all_healthy = False

        # 9. System Resources
        try:
            import psutil

            # Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            checks["memory"] = {
                "healthy": memory_percent < 90,
                "used_percent": round(memory_percent, 1),
                "available_gb": round(memory.available / (1024**3), 2),
            }
            if memory_percent >= 90:
                all_healthy = False
                warnings.append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent >= 80:
                warnings.append(f"Elevated memory usage: {memory_percent:.1f}%")

            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            checks["cpu"] = {
                "healthy": cpu_percent < 90,
                "used_percent": round(cpu_percent, 1),
                "core_count": psutil.cpu_count(),
            }
            if cpu_percent >= 90:
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")

            # Disk
            nomic_dir = self.get_nomic_dir()
            disk_path = str(nomic_dir) if nomic_dir else "/"
            disk = psutil.disk_usage(disk_path)
            disk_percent = disk.percent
            checks["disk"] = {
                "healthy": disk_percent < 90,
                "used_percent": round(disk_percent, 1),
                "free_gb": round(disk.free / (1024**3), 2),
                "path": disk_path,
            }
            if disk_percent >= 90:
                all_healthy = False
                warnings.append(f"Low disk space: {disk_percent:.1f}% used")
            elif disk_percent >= 80:
                warnings.append(f"Disk space warning: {disk_percent:.1f}% used")

        except ImportError:
            checks["system_resources"] = {"healthy": True, "status": "psutil_not_available"}
        except Exception as e:
            checks["system_resources"] = {"healthy": True, "warning": str(e)[:80]}

        # 10. Email Services (follow-up tracker, snooze recommender)
        try:
            from aragora.services.followup_tracker import FollowUpTracker
            from aragora.services.snooze_recommender import SnoozeRecommender

            # Check follow-up tracker - instantiate to verify module works
            _tracker = FollowUpTracker()  # noqa: F841
            checks["email_followup_tracker"] = {"healthy": True, "status": "available"}

            # Check snooze recommender - instantiate to verify module works
            _recommender = SnoozeRecommender()  # noqa: F841
            checks["email_snooze_recommender"] = {"healthy": True, "status": "available"}

        except ImportError as e:
            checks["email_services"] = {"healthy": True, "status": f"not_available: {e}"}
        except Exception as e:
            checks["email_services"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }
            all_healthy = False

        # 11. Dependency Analyzer
        try:
            from aragora.audit.dependency_analyzer import DependencyAnalyzer

            # Instantiate to verify module works
            _analyzer = DependencyAnalyzer()  # noqa: F841
            checks["dependency_analyzer"] = {"healthy": True, "status": "available"}

        except ImportError as e:
            checks["dependency_analyzer"] = {"healthy": True, "status": f"not_available: {e}"}
        except Exception as e:
            checks["dependency_analyzer"] = {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }
            all_healthy = False

        # 12. Stripe API connectivity
        from .health_utils import check_stripe_health

        checks["stripe"] = check_stripe_health()
        if checks["stripe"].get("configured") and not checks["stripe"].get("healthy", True):
            all_healthy = False
            warnings.append("Stripe API connectivity issue")

        # 13. Slack API connectivity
        from .health_utils import check_slack_health

        checks["slack"] = check_slack_health()
        if checks["slack"].get("configured") and not checks["slack"].get("healthy", True):
            all_healthy = False
            warnings.append("Slack API connectivity issue")

        # Calculate response time
        response_time_ms = round((time.time() - start_time) * 1000, 2)

        status = "healthy" if all_healthy else "degraded"
        if not all_healthy:
            status = "degraded"
        elif warnings:
            status = "healthy_with_warnings"

        # Get version
        try:
            from aragora import __version__

            version = __version__
        except (ImportError, AttributeError):
            version = "unknown"

        return json_response(
            {
                "status": status,
                "version": version,
                "checks": checks,
                "warnings": warnings if warnings else None,
                "response_time_ms": response_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )

    def _sync_status(self) -> HandlerResult:
        """Get Supabase sync service status.

        Returns status of the background sync service including:
        - enabled: Whether sync is enabled via SUPABASE_SYNC_ENABLED
        - running: Whether the background thread is running
        - queue_size: Number of items pending sync
        - synced_count: Total items successfully synced
        - failed_count: Total items that failed after max retries
        - last_sync_at: Timestamp of last sync attempt
        - last_error: Most recent error message (if any)

        Returns:
            JSON response with sync service status
        """
        try:
            from aragora.persistence.sync_service import get_sync_service

            sync = get_sync_service()
            status = sync.get_status()

            return json_response(
                {
                    "enabled": status.enabled,
                    "running": status.running,
                    "queue_size": status.queue_size,
                    "synced_count": status.synced_count,
                    "failed_count": status.failed_count,
                    "last_sync_at": (
                        status.last_sync_at.isoformat() + "Z" if status.last_sync_at else None
                    ),
                    "last_error": status.last_error,
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                }
            )

        except ImportError:
            return json_response(
                {
                    "enabled": False,
                    "error": "sync_service module not available",
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                }
            )
        except Exception as e:
            logger.warning(f"Sync status check failed: {e}")
            return json_response(
                {
                    "enabled": False,
                    "error": f"{type(e).__name__}: {str(e)[:80]}",
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                }
            )

    def _slow_debates_status(self) -> HandlerResult:
        """Get status of slow-running debates.

        Returns debates that have been running longer than the slow threshold
        (default 30 seconds per round). Useful for identifying performance
        bottlenecks and stuck debates.

        Includes:
        - current_slow: Currently active debates running slow
        - recent_slow: Historical slow debates from performance monitor

        Returns:
            JSON response with slow debate information
        """
        # Slow threshold: 30 seconds total, or configurable via env
        import os

        slow_threshold = float(os.getenv("ARAGORA_SLOW_DEBATE_THRESHOLD", "30"))

        current_slow = []
        recent_slow = []
        errors = []

        # Get currently active slow debates
        try:
            from aragora.server.stream.state_manager import (
                get_active_debates,
                get_active_debates_lock,
            )

            now = time.time()

            with get_active_debates_lock():
                for debate_id, debate_info in get_active_debates().items():
                    start_time = debate_info.get("start_time", now)
                    duration = now - start_time
                    if duration > slow_threshold:
                        current_slow.append(
                            {
                                "debate_id": debate_id,
                                "duration_seconds": round(duration, 2),
                                "task": debate_info.get("task", "")[:100],
                                "agents": debate_info.get("agents", []),
                                "current_round": debate_info.get("current_round", 0),
                                "total_rounds": debate_info.get("total_rounds", 0),
                                "started_at": datetime.fromtimestamp(start_time).isoformat() + "Z",
                            }
                        )

            # Sort by duration descending
            current_slow.sort(key=lambda x: x["duration_seconds"], reverse=True)

        except ImportError:
            errors.append("state_manager not available")
        except (OSError, RuntimeError, ValueError) as e:
            errors.append(f"state_manager error: {type(e).__name__}")

        # Get historical slow debates from performance monitor
        try:
            from aragora.debate.performance_monitor import get_debate_monitor

            monitor = get_debate_monitor()

            # Get historical slow debates
            recent_slow = monitor.get_slow_debates(limit=20)

            # Also get currently monitored slow debates
            monitored_current = monitor.get_current_slow_debates()
            for debate in monitored_current:
                # Avoid duplicates with state manager
                if not any(d["debate_id"] == debate["debate_id"] for d in current_slow):
                    current_slow.append(debate)

        except ImportError:
            errors.append("performance_monitor not available")
        except (OSError, RuntimeError, ValueError) as e:
            errors.append(f"performance_monitor error: {type(e).__name__}")

        # Determine overall status
        total_slow = len(current_slow) + len(recent_slow)
        status = "healthy" if total_slow == 0 else "degraded"
        if errors and total_slow == 0:
            status = "partial"

        return json_response(
            {
                "status": status,
                "slow_threshold_seconds": slow_threshold,
                "current_slow_count": len(current_slow),
                "recent_slow_count": len(recent_slow),
                "current_slow": current_slow[:20],
                "recent_slow": recent_slow[:20],
                "errors": errors if errors else None,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )

    def _circuit_breakers_status(self) -> HandlerResult:
        """Get detailed circuit breaker status for all registered breakers.

        Returns comprehensive metrics including:
        - summary: Counts of open, closed, half-open circuits
        - circuit_breakers: Per-breaker details with failures, cooldowns, entity tracking
        - health: Overall health status and high-failure circuit warnings

        This endpoint is useful for:
        - Monitoring dashboards
        - Debugging cascading failures
        - Tuning circuit breaker thresholds

        Returns:
            JSON response with detailed circuit breaker metrics
        """
        try:
            from aragora.resilience import get_circuit_breaker_metrics

            metrics = get_circuit_breaker_metrics()

            return json_response(
                {
                    "status": metrics.get("health", {}).get("status", "unknown"),
                    "summary": metrics.get("summary", {}),
                    "circuit_breakers": metrics.get("circuit_breakers", {}),
                    "health": metrics.get("health", {}),
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                }
            )

        except ImportError:
            return json_response(
                {
                    "status": "unavailable",
                    "error": "resilience module not available",
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                }
            )
        except Exception as e:
            logger.warning(f"Circuit breaker status check failed: {e}")
            return json_response(
                {
                    "status": "error",
                    "error": f"{type(e).__name__}: {str(e)[:80]}",
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                }
            )

    def _cross_pollination_health(self) -> HandlerResult:
        """Check health of cross-pollination feature integrations.

        Returns status of:
        - ELO skill weighting integration
        - Calibration tracking
        - Evidence quality scoring
        - RLM hierarchy caching
        - Knowledge Mound operations

        This endpoint is useful for:
        - Verifying cross-pollination features are operational
        - Debugging feature integration issues
        - Monitoring feature-level health

        Returns:
            JSON response with cross-pollination health metrics
        """
        features: Dict[str, Dict[str, Any]] = {}
        all_healthy = True

        # Check ELO system for skill weighting
        try:
            elo = self.get_elo_system()
            if elo is not None:
                # Check if ELO has domain ratings
                leaderboard = elo.get_leaderboard(limit=1)
                features["elo_weighting"] = {
                    "healthy": True,
                    "status": "active",
                    "agents_tracked": len(leaderboard) if leaderboard else 0,
                }
            else:
                features["elo_weighting"] = {
                    "healthy": True,
                    "status": "not_configured",
                    "note": "ELO system not initialized",
                }
        except Exception as e:
            features["elo_weighting"] = {
                "healthy": False,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }
            all_healthy = False

        # Check calibration tracker
        try:
            calibration = self.ctx.get("calibration_tracker")
            if calibration is not None:
                features["calibration"] = {
                    "healthy": True,
                    "status": "active",
                }
                # Try to get calibration stats
                try:
                    if hasattr(calibration, "get_calibration_stats"):
                        stats = calibration.get_calibration_stats()
                        features["calibration"]["tracked_agents"] = stats.get("tracked_agents", 0)
                except (AttributeError, KeyError):
                    pass
            else:
                features["calibration"] = {
                    "healthy": True,
                    "status": "not_configured",
                    "note": "Calibration tracker not initialized",
                }
        except Exception as e:
            features["calibration"] = {
                "healthy": False,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }
            all_healthy = False

        # Check evidence quality scoring
        try:
            evidence_store = self.ctx.get("evidence_store")
            if evidence_store is not None:
                features["evidence_quality"] = {
                    "healthy": True,
                    "status": "active",
                }
            else:
                features["evidence_quality"] = {
                    "healthy": True,
                    "status": "not_configured",
                    "note": "Evidence store not initialized",
                }
        except Exception as e:
            features["evidence_quality"] = {
                "healthy": False,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }
            all_healthy = False

        # Check RLM hierarchy caching
        try:
            from aragora.rlm.cache import get_rlm_cache_stats

            cache_stats = get_rlm_cache_stats()
            hits = cache_stats.get("hits", 0)
            misses = cache_stats.get("misses", 0)
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0

            features["rlm_caching"] = {
                "healthy": True,
                "status": "active",
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate": round(hit_rate, 3),
            }
        except ImportError:
            features["rlm_caching"] = {
                "healthy": True,
                "status": "not_available",
                "note": "RLM cache module not available",
            }
        except Exception as e:
            features["rlm_caching"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # Check Knowledge Mound
        try:
            from aragora.debate.knowledge_mound_ops import get_knowledge_mound_stats  # type: ignore[attr-defined]

            km_stats = get_knowledge_mound_stats()
            features["knowledge_mound"] = {
                "healthy": True,
                "status": "active",
                "facts_count": km_stats.get("facts_count", 0),
                "consensus_stored": km_stats.get("consensus_stored", 0),
            }
        except ImportError:
            features["knowledge_mound"] = {
                "healthy": True,
                "status": "not_available",
                "note": "Knowledge Mound module not available",
            }
        except Exception as e:
            features["knowledge_mound"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # Check trending topics (Pulse)
        try:
            from aragora.pulse import get_pulse_stats  # type: ignore[attr-defined]

            pulse_stats = get_pulse_stats()
            features["trending_topics"] = {
                "healthy": True,
                "status": "active",
                "topics_tracked": pulse_stats.get("topics_count", 0),
            }
        except ImportError:
            features["trending_topics"] = {
                "healthy": True,
                "status": "not_available",
                "note": "Pulse module not available",
            }
        except Exception as e:
            features["trending_topics"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # Count active features
        active_count = sum(1 for f in features.values() if f.get("status") == "active")
        total_features = len(features)

        status = "healthy" if all_healthy else "degraded"
        if active_count == 0:
            status = "not_configured"

        return json_response(
            {
                "status": status,
                "active_features": active_count,
                "total_features": total_features,
                "features": features,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )

    def _knowledge_mound_health(self) -> HandlerResult:
        """Comprehensive health check for Knowledge Mound subsystem.

        Returns detailed status of:
        - Core mound: Storage layer, fact storage, workspace config
        - Culture accumulator: Organizational pattern tracking
        - Staleness tracking: Fact age and refresh status
        - Query performance: Retrieval latency stats
        - Storage backends: PostgreSQL/SQLite status
        - RLM integration: Context compression caching

        This endpoint is useful for:
        - Monitoring Knowledge Mound operational health
        - Debugging knowledge retrieval issues
        - Verifying debate-knowledge integration
        - Tracking storage capacity and performance

        Returns:
            JSON response with comprehensive KM health metrics
        """
        components: Dict[str, Dict[str, Any]] = {}
        all_healthy = True
        warnings: list[str] = []
        start_time = time.time()

        # 1. Check if Knowledge Mound module is available
        try:
            from aragora.knowledge.mound import KnowledgeMound  # noqa: F401
            from aragora.knowledge.mound.types import MoundConfig  # noqa: F401

            components["module"] = {
                "healthy": True,
                "status": "available",
            }
        except ImportError as e:
            components["module"] = {
                "healthy": False,
                "status": "not_available",
                "error": str(e)[:100],
            }
            all_healthy = False
            return json_response(
                {
                    "status": "unavailable",
                    "error": "Knowledge Mound module not installed",
                    "components": components,
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                }
            )

        # 2. Check core mound initialization
        try:
            mound = KnowledgeMound(workspace_id="health_check")  # type: ignore[abstract]

            components["core"] = {
                "healthy": True,
                "status": "initialized",
                "workspace_id": mound.workspace_id,
            }

            # Check config
            if hasattr(mound, "config") and mound.config:
                components["core"]["config"] = {
                    "enable_staleness_tracking": mound.config.enable_staleness_tracking,  # type: ignore[attr-defined]
                    "enable_culture_accumulator": mound.config.enable_culture_accumulator,  # type: ignore[attr-defined]
                    "enable_rlm_summaries": mound.config.enable_rlm_summaries,  # type: ignore[attr-defined]
                    "default_staleness_hours": mound.config.default_staleness_hours,  # type: ignore[attr-defined]
                }

        except Exception as e:
            components["core"] = {
                "healthy": False,
                "status": "initialization_failed",
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }
            all_healthy = False

        # 3. Check storage backend
        try:
            import os

            database_url = os.environ.get("KNOWLEDGE_MOUND_DATABASE_URL", "")

            if "postgres" in database_url.lower():
                components["storage"] = {
                    "healthy": True,
                    "backend": "postgresql",
                    "status": "configured",
                }
            else:
                components["storage"] = {
                    "healthy": True,
                    "backend": "sqlite",
                    "status": "configured",
                    "note": "Using local SQLite storage",
                }

            # Try to get storage stats if mound initialized
            if "mound" in locals() and hasattr(mound, "_store"):
                try:
                    # Check if store is accessible
                    if mound._store is not None:
                        components["storage"]["store_type"] = type(mound._store).__name__
                except AttributeError:
                    pass

        except Exception as e:
            components["storage"] = {
                "healthy": True,
                "status": "unknown",
                "warning": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 4. Check culture accumulator
        try:
            if "mound" in locals():
                if hasattr(mound, "_culture_accumulator") and mound._culture_accumulator:
                    accumulator = mound._culture_accumulator
                    components["culture_accumulator"] = {
                        "healthy": True,
                        "status": "active",
                        "type": type(accumulator).__name__,
                    }

                    # Try to get pattern counts
                    try:
                        if hasattr(accumulator, "_patterns"):
                            workspace_count = len(accumulator._patterns)
                            components["culture_accumulator"]["workspaces_tracked"] = (
                                workspace_count
                            )
                    except (AttributeError, TypeError):
                        pass
                else:
                    components["culture_accumulator"] = {
                        "healthy": True,
                        "status": "not_initialized",
                        "note": "Culture accumulator disabled or not yet created",
                    }
        except Exception as e:
            components["culture_accumulator"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 5. Check staleness tracker
        try:
            if "mound" in locals():
                if hasattr(mound, "_staleness_tracker") and mound._staleness_tracker:
                    components["staleness_tracker"] = {
                        "healthy": True,
                        "status": "active",
                    }
                else:
                    components["staleness_tracker"] = {
                        "healthy": True,
                        "status": "not_initialized",
                        "note": "Staleness tracking disabled",
                    }
        except Exception as e:
            components["staleness_tracker"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 6. Check RLM integration
        try:
            from aragora.rlm import HAS_OFFICIAL_RLM

            if HAS_OFFICIAL_RLM:
                components["rlm_integration"] = {
                    "healthy": True,
                    "status": "active",
                    "type": "official_rlm",
                }
            else:
                components["rlm_integration"] = {
                    "healthy": True,
                    "status": "fallback",
                    "type": "compression_only",
                    "note": "Using compression fallback (official RLM not installed)",
                }
        except ImportError:
            components["rlm_integration"] = {
                "healthy": True,
                "status": "not_available",
                "note": "RLM module not installed",
            }
        except Exception as e:
            components["rlm_integration"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 7. Check debate integration via knowledge_mound_ops
        try:
            from aragora.debate.knowledge_mound_ops import get_knowledge_mound_stats  # type: ignore[attr-defined]

            km_stats = get_knowledge_mound_stats()
            components["debate_integration"] = {
                "healthy": True,
                "status": "active",
                "facts_count": km_stats.get("facts_count", 0),
                "consensus_stored": km_stats.get("consensus_stored", 0),
                "retrievals_count": km_stats.get("retrievals_count", 0),
            }
        except ImportError:
            components["debate_integration"] = {
                "healthy": True,
                "status": "not_available",
                "note": "knowledge_mound_ops module not available",
            }
        except Exception as e:
            components["debate_integration"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 8. Check Redis cache for mound (if configured)
        try:
            import os

            redis_url = os.environ.get("KNOWLEDGE_MOUND_REDIS_URL") or os.environ.get("REDIS_URL")
            if redis_url:
                from aragora.knowledge.mound.redis_cache import KnowledgeMoundCache  # type: ignore[attr-defined]

                cache = KnowledgeMoundCache(redis_url=redis_url)
                components["redis_cache"] = {
                    "healthy": True,
                    "status": "configured",
                }
            else:
                components["redis_cache"] = {
                    "healthy": True,
                    "status": "not_configured",
                    "note": "Knowledge Mound Redis cache not configured",
                }
        except ImportError:
            components["redis_cache"] = {
                "healthy": True,
                "status": "not_available",
                "note": "Redis cache module not installed",
            }
        except Exception as e:
            components["redis_cache"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 9. Check bidirectional adapters
        try:
            from aragora.knowledge.mound.adapters import (
                ContinuumAdapter,
                ConsensusAdapter,
                CritiqueAdapter,
                EvidenceAdapter,
                BeliefAdapter,
                InsightsAdapter,
                EloAdapter,
                PulseAdapter,
                CostAdapter,
                RankingAdapter,
                CultureAdapter,
            )

            adapter_classes = [
                ("continuum", ContinuumAdapter),
                ("consensus", ConsensusAdapter),
                ("critique", CritiqueAdapter),
                ("evidence", EvidenceAdapter),
                ("belief", BeliefAdapter),
                ("insights", InsightsAdapter),
                ("elo", EloAdapter),
                ("pulse", PulseAdapter),
                ("cost", CostAdapter),
                ("ranking", RankingAdapter),
                ("culture", CultureAdapter),
            ]

            components["bidirectional_adapters"] = {
                "healthy": True,
                "status": "available",
                "adapters_available": len(adapter_classes),
                "adapter_list": [name for name, _ in adapter_classes],
            }
        except ImportError as e:
            components["bidirectional_adapters"] = {
                "healthy": True,
                "status": "partial",
                "error": f"Some adapters not available: {str(e)[:80]}",
            }
        except Exception as e:
            components["bidirectional_adapters"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 10. Check Control Plane adapter
        try:
            from aragora.knowledge.mound.adapters.control_plane_adapter import (  # noqa: F401
                ControlPlaneAdapter,
                TaskOutcome,
                AgentCapabilityRecord,
                CrossWorkspaceInsight,
            )

            components["control_plane_adapter"] = {
                "healthy": True,
                "status": "available",
                "features": [
                    "task_outcome_storage",
                    "capability_records",
                    "cross_workspace_insights",
                    "agent_recommendations",
                ],
            }
        except ImportError as e:
            components["control_plane_adapter"] = {
                "healthy": True,
                "status": "not_available",
                "error": str(e)[:80],
            }
        except Exception as e:
            components["control_plane_adapter"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 11. Check KM metrics availability
        try:
            from aragora.observability.metrics.km import (  # noqa: F401
                init_km_metrics,
                record_km_operation,
                record_cp_task_outcome,
            )

            components["km_metrics"] = {
                "healthy": True,
                "status": "available",
                "prometheus_integration": True,
            }
        except ImportError:
            components["km_metrics"] = {
                "healthy": True,
                "status": "not_available",
                "prometheus_integration": False,
            }
        except Exception as e:
            components["km_metrics"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 12. Check Confidence Decay Scheduler
        try:
            from aragora.knowledge.mound.confidence_decay_scheduler import (
                get_decay_scheduler,
            )

            scheduler = get_decay_scheduler()
            if scheduler:
                stats = scheduler.get_stats()
                components["confidence_decay"] = {
                    "healthy": True,
                    "status": "active" if scheduler.is_running else "stopped",
                    "running": scheduler.is_running,
                    "decay_interval_hours": stats.get("decay_interval_hours", 24),
                    "total_cycles": stats.get("total_decay_cycles", 0),
                    "total_items_processed": stats.get("total_items_processed", 0),
                    "last_run": stats.get("last_run", {}),
                    "workspaces_monitored": stats.get("workspaces"),
                }
                # Add alerting info
                if scheduler.is_running:
                    last_runs = stats.get("last_run", {})
                    if last_runs:
                        # Check if any workspace hasn't been processed in >48 hours
                        now = datetime.now()
                        stale_workspaces = []
                        for ws_id, run_time_str in last_runs.items():
                            try:
                                run_time = datetime.fromisoformat(run_time_str)
                                hours_since = (now - run_time).total_seconds() / 3600
                                if hours_since > 48:
                                    stale_workspaces.append(ws_id)
                            except (ValueError, TypeError):
                                pass
                        if stale_workspaces:
                            components["confidence_decay"]["alert"] = {
                                "level": "warning",
                                "message": f"Decay not run in >48h for: {', '.join(stale_workspaces)}",
                            }
                            warnings.append(
                                f"Confidence decay stale for workspaces: {stale_workspaces}"
                            )
            else:
                components["confidence_decay"] = {
                    "healthy": True,
                    "status": "not_configured",
                    "note": "Confidence decay scheduler not initialized",
                }
        except ImportError:
            components["confidence_decay"] = {
                "healthy": True,
                "status": "not_available",
                "note": "Confidence decay module not installed",
            }
        except Exception as e:
            components["confidence_decay"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # Calculate response time
        response_time_ms = round((time.time() - start_time) * 1000, 2)

        # Determine overall status
        healthy_count = sum(1 for c in components.values() if c.get("healthy", False))
        active_count = sum(1 for c in components.values() if c.get("status") == "active")
        total_components = len(components)

        status = "healthy" if all_healthy else "degraded"
        if active_count == 0:
            status = "not_configured"

        return json_response(
            {
                "status": status,
                "summary": {
                    "total_components": total_components,
                    "healthy": healthy_count,
                    "active": active_count,
                },
                "components": components,
                "warnings": warnings if warnings else None,
                "response_time_ms": response_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )

    def _encryption_health(self) -> HandlerResult:
        """Encryption health check - verifies encryption service status.

        Checks:
        - Cryptography library availability
        - Encryption service initialization
        - Active encryption key status
        - Key age and rotation recommendations
        - Encrypt/decrypt round-trip verification
        """
        start_time = time.time()
        issues: list[str] = []
        warnings: list[str] = []
        health: Dict[str, Any] = {}

        # Check 1: Crypto library availability
        try:
            from aragora.security.encryption import get_encryption_service, CRYPTO_AVAILABLE

            if CRYPTO_AVAILABLE:
                health["cryptography_library"] = {"healthy": True, "status": "installed"}
            else:
                health["cryptography_library"] = {"healthy": False, "status": "not_installed"}
                issues.append("Cryptography library not installed")
        except ImportError:
            health["cryptography_library"] = {"healthy": False, "status": "import_error"}
            issues.append("Cannot import encryption module")
            return json_response(
                {
                    "status": "error",
                    "issues": issues,
                    "health": health,
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                },
                status=503,
            )

        # Check 2: Encryption service initialization
        try:
            service = get_encryption_service()
            health["encryption_service"] = {"healthy": True, "status": "initialized"}
        except Exception as e:
            health["encryption_service"] = {
                "healthy": False,
                "status": "error",
                "error": str(e)[:100],
            }
            issues.append(f"Encryption service error: {str(e)[:50]}")
            return json_response(
                {
                    "status": "error",
                    "issues": issues,
                    "health": health,
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                },
                status=503,
            )

        # Check 3: Active key status
        active_key = service.get_active_key()
        if active_key:
            age_days = (datetime.now(timezone.utc) - active_key.created_at).days
            health["active_key"] = {
                "healthy": True,
                "key_id": service.get_active_key_id(),
                "version": active_key.version,
                "age_days": age_days,
                "created_at": active_key.created_at.isoformat(),
            }

            # Key age warnings
            if age_days > 90:
                warnings.append(f"Key is {age_days} days old (>90 days). Rotation recommended.")
                health["active_key"]["rotation_recommended"] = True
            elif age_days > 60:
                health["active_key"]["days_until_rotation"] = 90 - age_days
        else:
            health["active_key"] = {"healthy": False, "status": "no_active_key"}
            issues.append("No active encryption key")

        # Check 4: Encrypt/decrypt round-trip
        try:
            test_data = b"encryption_health_check"
            encrypted = service.encrypt(test_data)
            decrypted = service.decrypt(encrypted)

            if decrypted == test_data:
                health["roundtrip_test"] = {"healthy": True, "status": "passed"}
            else:
                health["roundtrip_test"] = {"healthy": False, "status": "data_mismatch"}
                issues.append("Encrypt/decrypt round-trip failed")
        except Exception as e:
            health["roundtrip_test"] = {
                "healthy": False,
                "status": "error",
                "error": str(e)[:100],
            }
            issues.append(f"Encrypt/decrypt error: {str(e)[:50]}")

        # Calculate overall status
        response_time_ms = round((time.time() - start_time) * 1000, 2)

        if issues:
            status = "error"
            http_status = 503
        elif warnings:
            status = "warning"
            http_status = 200
        else:
            status = "healthy"
            http_status = 200

        return json_response(
            {
                "status": status,
                "health": health,
                "issues": issues if issues else None,
                "warnings": warnings if warnings else None,
                "response_time_ms": response_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            },
            status=http_status,
        )

    def _platform_health(self) -> HandlerResult:
        """Platform resilience health check for chat integrations.

        Checks:
        - Platform circuit breakers status (Slack, Discord, Teams, etc.)
        - Platform-specific rate limiters
        - Dead letter queue status
        - Platform delivery metrics
        - Webhook health

        Returns:
            JSON response with comprehensive platform health metrics
        """
        start_time = time.time()
        components: Dict[str, Dict[str, Any]] = {}
        all_healthy = True
        warnings: list[str] = []

        # 1. Check platform rate limiters
        try:
            from aragora.server.middleware.rate_limit.platform_limiter import (
                PLATFORM_RATE_LIMITS,
                get_platform_rate_limiter,
            )

            platform_limiters = {}
            for platform in PLATFORM_RATE_LIMITS.keys():
                limiter = get_platform_rate_limiter(platform)
                platform_limiters[platform] = {
                    "rpm": limiter.rpm,
                    "burst_size": limiter.burst_size,
                    "daily_limit": limiter.daily_limit,
                }

            components["rate_limiters"] = {
                "healthy": True,
                "status": "active",
                "platforms": list(PLATFORM_RATE_LIMITS.keys()),
                "config": platform_limiters,
            }
        except ImportError:
            components["rate_limiters"] = {
                "healthy": True,
                "status": "not_available",
                "note": "Platform rate limiter module not installed",
            }
        except Exception as e:
            components["rate_limiters"] = {
                "healthy": False,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }
            all_healthy = False

        # 2. Check platform resilience module
        try:
            from aragora.integrations.platform_resilience import (
                get_platform_resilience,
                DLQ_ENABLED,
            )

            resilience = get_platform_resilience()
            stats = resilience.get_stats()

            components["resilience"] = {
                "healthy": True,
                "status": "active",
                "dlq_enabled": DLQ_ENABLED,
                "platforms_tracked": stats.get("platforms_tracked", 0),
                "circuit_breakers": stats.get("circuit_breakers", {}),
            }
        except ImportError:
            components["resilience"] = {
                "healthy": True,
                "status": "not_available",
                "note": "Platform resilience module not installed",
            }
        except Exception as e:
            components["resilience"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 3. Check dead letter queue
        try:
            from aragora.integrations.platform_resilience import get_dlq

            dlq = get_dlq()
            dlq_stats = dlq.get_stats()

            components["dead_letter_queue"] = {
                "healthy": True,
                "status": "active",
                "pending_count": dlq_stats.get("pending", 0),
                "failed_count": dlq_stats.get("failed", 0),
                "processed_count": dlq_stats.get("processed", 0),
            }

            # Warn if DLQ is backing up
            pending = dlq_stats.get("pending", 0)
            if pending > 100:
                warnings.append(f"DLQ has {pending} pending messages")
            elif pending > 50:
                warnings.append(f"DLQ has {pending} pending messages (elevated)")

        except ImportError:
            components["dead_letter_queue"] = {
                "healthy": True,
                "status": "not_available",
                "note": "DLQ module not installed",
            }
        except Exception as e:
            components["dead_letter_queue"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 4. Check platform metrics
        try:
            from aragora.observability.metrics.platform import (
                get_platform_metrics_summary,
            )

            metrics = get_platform_metrics_summary()
            components["metrics"] = {
                "healthy": True,
                "status": "active",
                "prometheus_enabled": True,
                "summary": metrics,
            }
        except ImportError:
            components["metrics"] = {
                "healthy": True,
                "status": "not_available",
                "prometheus_enabled": False,
            }
        except Exception as e:
            components["metrics"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # 5. Check individual platform circuit breakers
        try:
            from aragora.resilience import get_circuit_breaker

            platform_circuits = {}
            platforms = ["slack", "discord", "teams", "telegram", "whatsapp", "matrix"]

            for platform in platforms:
                try:
                    cb = get_circuit_breaker(f"platform_{platform}")
                    if cb:
                        platform_circuits[platform] = {
                            "state": (
                                cb.state.value if hasattr(cb.state, "value") else str(cb.state)
                            ),
                            "failure_count": cb.failure_count,  # type: ignore[attr-defined]
                            "success_count": getattr(cb, "success_count", 0),
                        }
                except Exception as e:
                    logger.debug(f"Error getting circuit breaker for {platform}: {e}")
                    platform_circuits[platform] = {"state": "not_configured"}

            components["platform_circuits"] = {
                "healthy": True,
                "status": "active",
                "circuits": platform_circuits,
            }

            # Check for open circuits
            open_circuits = [p for p, c in platform_circuits.items() if c.get("state") == "open"]
            if open_circuits:
                all_healthy = False
                warnings.append(f"Open circuit breakers: {', '.join(open_circuits)}")

        except ImportError:
            components["platform_circuits"] = {
                "healthy": True,
                "status": "not_available",
                "note": "Circuit breaker module not available",
            }
        except Exception as e:
            components["platform_circuits"] = {
                "healthy": True,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

        # Calculate response time
        response_time_ms = round((time.time() - start_time) * 1000, 2)

        # Determine overall status
        healthy_count = sum(1 for c in components.values() if c.get("healthy", False))
        active_count = sum(1 for c in components.values() if c.get("status") == "active")

        status = "healthy" if all_healthy else "degraded"
        if active_count == 0:
            status = "not_configured"
        elif warnings and all_healthy:
            status = "healthy_with_warnings"

        return json_response(
            {
                "status": status,
                "summary": {
                    "total_components": len(components),
                    "healthy": healthy_count,
                    "active": active_count,
                },
                "components": components,
                "warnings": warnings if warnings else None,
                "response_time_ms": response_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )

    def _deployment_diagnostics(self) -> HandlerResult:
        """Comprehensive deployment diagnostics endpoint.

        Runs the full deployment validator and returns detailed results
        including all production readiness checks:
        - JWT secret strength and configuration
        - AI provider API key availability
        - Database connectivity (Supabase/PostgreSQL)
        - Redis configuration for distributed state
        - CORS and security settings
        - Rate limiting configuration
        - TLS/HTTPS settings
        - Encryption key configuration
        - Storage accessibility

        This endpoint is useful for:
        - Pre-deployment validation
        - Production readiness verification
        - Debugging configuration issues
        - CI/CD deployment checks

        Returns:
            JSON response with comprehensive deployment validation results
        """

        start_time = time.time()

        try:
            from aragora.ops.deployment_validator import validate_deployment

            # Run async validation in sync context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop:
                # Already in async context - use thread pool
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, validate_deployment())
                    result = future.result(timeout=30.0)
            else:
                result = asyncio.run(validate_deployment())

            # Convert to response format
            response_data = result.to_dict()

            # Add summary information
            critical_issues = [i for i in result.issues if i.severity.value == "critical"]
            warning_issues = [i for i in result.issues if i.severity.value == "warning"]
            info_issues = [i for i in result.issues if i.severity.value == "info"]

            # Add component summary
            healthy_components = [c for c in result.components if c.status.value == "healthy"]
            degraded_components = [c for c in result.components if c.status.value == "degraded"]
            unhealthy_components = [c for c in result.components if c.status.value == "unhealthy"]

            response_data["summary"] = {
                "ready": result.ready,
                "live": result.live,
                "issues": {
                    "critical": len(critical_issues),
                    "warning": len(warning_issues),
                    "info": len(info_issues),
                    "total": len(result.issues),
                },
                "components": {
                    "healthy": len(healthy_components),
                    "degraded": len(degraded_components),
                    "unhealthy": len(unhealthy_components),
                    "total": len(result.components),
                },
            }

            # Add production readiness checklist
            response_data["checklist"] = self._generate_checklist(result)

            response_data["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
            response_data["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z"

            # Return appropriate status code
            if not result.ready:
                return json_response(response_data, status=503)
            elif len(warning_issues) > 0:
                return json_response(response_data, status=200)
            else:
                return json_response(response_data, status=200)

        except ImportError as e:
            return json_response(
                {
                    "status": "error",
                    "error": f"Deployment validator not available: {e}",
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                },
                status=500,
            )
        except concurrent.futures.TimeoutError:
            return json_response(
                {
                    "status": "error",
                    "error": "Deployment validation timed out after 30 seconds",
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                },
                status=504,
            )
        except Exception as e:
            logger.warning(f"Deployment diagnostics failed: {e}")
            return json_response(
                {
                    "status": "error",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                },
                status=500,
            )

    def _generate_checklist(self, result) -> Dict[str, Any]:
        """Generate a production readiness checklist from validation results.

        Args:
            result: ValidationResult from deployment validator

        Returns:
            Dictionary with checklist items and their status
        """
        # Build component lookup
        components = {c.name: c for c in result.components}
        issues_by_component: dict[str, list] = {}
        for issue in result.issues:
            if issue.component not in issues_by_component:
                issues_by_component[issue.component] = []
            issues_by_component[issue.component].append(issue)

        def get_status(component_name: str) -> str:
            comp = components.get(component_name)
            if not comp:
                return "not_checked"
            if comp.status.value == "healthy":
                return "pass"
            elif comp.status.value == "degraded":
                return "warning"
            elif comp.status.value == "unhealthy":
                return "fail"
            return "unknown"

        def has_critical_issue(component_name: str) -> bool:
            issues = issues_by_component.get(component_name, [])
            return any(i.severity.value == "critical" for i in issues)

        return {
            "security": {
                "jwt_secret": {
                    "status": get_status("jwt_secret"),
                    "critical": has_critical_issue("jwt_secret"),
                    "description": "JWT secret configured with 32+ characters",
                },
                "encryption_key": {
                    "status": get_status("encryption"),
                    "critical": has_critical_issue("encryption"),
                    "description": "Encryption key configured (32-byte hex)",
                },
                "cors": {
                    "status": get_status("cors"),
                    "critical": has_critical_issue("cors"),
                    "description": "CORS origins properly restricted",
                },
                "tls": {
                    "status": get_status("tls"),
                    "critical": has_critical_issue("tls"),
                    "description": "TLS/HTTPS configured or behind proxy",
                },
            },
            "infrastructure": {
                "database": {
                    "status": get_status("database"),
                    "critical": has_critical_issue("database"),
                    "description": "Database connectivity verified",
                },
                "redis": {
                    "status": get_status("redis"),
                    "critical": has_critical_issue("redis"),
                    "description": "Redis configured for distributed state",
                },
                "storage": {
                    "status": get_status("storage"),
                    "critical": has_critical_issue("storage"),
                    "description": "Data directory writable",
                },
                "supabase": {
                    "status": get_status("supabase"),
                    "critical": has_critical_issue("supabase"),
                    "description": "Supabase configured (if used)",
                },
            },
            "api": {
                "api_keys": {
                    "status": get_status("api_keys"),
                    "critical": has_critical_issue("api_keys"),
                    "description": "At least one AI provider configured",
                },
                "rate_limiting": {
                    "status": get_status("rate_limiting"),
                    "critical": has_critical_issue("rate_limiting"),
                    "description": "Rate limiting enabled and configured",
                },
            },
            "environment": {
                "env_mode": {
                    "status": get_status("environment"),
                    "critical": has_critical_issue("environment"),
                    "description": "Environment mode set correctly",
                },
            },
        }
