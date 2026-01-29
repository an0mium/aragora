# mypy: ignore-errors
"""
Detailed and deep health check implementations.

Provides comprehensive health checks:
- /api/health - Comprehensive health check (basic)
- /api/health/detailed - Detailed with observer metrics
- /api/health/deep - Deep health with all external dependencies
- WebSocket health check
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from ...base import HandlerResult, json_response
from ..health_utils import (
    check_ai_providers_health,
    check_filesystem_health,
    check_redis_health,
    check_security_services,
)

logger = logging.getLogger(__name__)


def health_check(handler) -> HandlerResult:
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
    from . import _SERVER_START_TIME

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
        storage = handler.get_storage()
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
        elo = handler.get_elo_system()
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
    nomic_dir = handler.get_nomic_dir()
    if nomic_dir is not None and nomic_dir.exists():
        checks["nomic_dir"] = {"healthy": True, "path": str(nomic_dir)}
    else:
        checks["nomic_dir"] = {"healthy": False, "error": "Directory not found"}
        # Non-critical - don't fail health check for this
        checks["nomic_dir"]["healthy"] = True  # Downgrade to warning
        checks["nomic_dir"]["warning"] = "Nomic directory not configured"

    # Check filesystem write access
    checks["filesystem"] = check_filesystem_health(nomic_dir)
    if not checks["filesystem"]["healthy"]:
        all_healthy = False

    # Check Redis connectivity (if configured)
    checks["redis"] = check_redis_health()
    # Redis is optional - don't fail if not configured
    if checks["redis"].get("error") and checks["redis"].get("configured", False):
        all_healthy = False

    # Check AI provider availability
    checks["ai_providers"] = check_ai_providers_health()
    # At least one provider should be available
    if not checks["ai_providers"].get("any_available", False):
        checks["ai_providers"]["warning"] = "No AI providers configured"
        # Don't fail health check - just warn

    # Check WebSocket manager (if available in context)
    ws_manager = handler.ctx.get("ws_manager")
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
    checks["security_services"] = check_security_services()
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


def websocket_health(handler) -> HandlerResult:
    """Basic WebSocket health check for availability and client count."""
    ws_manager = handler.ctx.get("ws_manager")
    if ws_manager is None:
        return json_response(
            {
                "status": "unavailable",
                "clients": 0,
                "message": "WebSocket manager not configured",
            },
            status=200,
        )

    try:
        client_count = len(getattr(ws_manager, "clients", []))
        return json_response({"status": "healthy", "clients": client_count}, status=200)
    except Exception as e:
        return json_response(
            {"status": "error", "clients": 0, "message": str(e)[:120]},
            status=503,
        )


def detailed_health_check(handler) -> HandlerResult:
    """Return detailed health status with system observer metrics.

    Includes:
    - Basic health components
    - Agent success/failure rates (from SimpleObserver)
    - Null byte incidents
    - Timeout incidents
    - Memory stats
    - Maintenance status
    """
    nomic_dir = handler.get_nomic_dir()
    storage = handler.get_storage()
    elo = handler.get_elo_system()

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


def deep_health_check(handler) -> HandlerResult:
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
        storage = handler.get_storage()
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
        elo = handler.get_elo_system()
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
        user_store = getattr(handler.__class__, "user_store", None) or handler.ctx.get("user_store")
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
    checks["redis"] = check_redis_health()
    if checks["redis"].get("configured") and not checks["redis"].get("healthy", True):
        all_healthy = False

    # 7. AI Providers
    checks["ai_providers"] = check_ai_providers_health()
    if not checks["ai_providers"].get("any_available", False):
        warnings.append("No AI providers configured")

    # 8. Filesystem
    nomic_dir = handler.get_nomic_dir()
    checks["filesystem"] = check_filesystem_health(nomic_dir)
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
    from ..health_utils import check_stripe_health

    checks["stripe"] = check_stripe_health()
    if checks["stripe"].get("configured") and not checks["stripe"].get("healthy", True):
        all_healthy = False
        warnings.append("Stripe API connectivity issue")

    # 13. Slack API connectivity
    from ..health_utils import check_slack_health

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
