"""
Health and readiness endpoint handlers.

Endpoints:
- GET /healthz - Kubernetes liveness probe (lightweight)
- GET /readyz - Kubernetes readiness probe (checks dependencies)
- GET /api/health - Comprehensive health check
- GET /api/health/detailed - Detailed health with observer metrics
- GET /api/health/deep - Deep health check with all external dependencies

Extracted from system.py for better modularity.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..base import (
    BaseHandler,
    HandlerResult,
    json_response,
)

logger = logging.getLogger(__name__)

# Server start time for uptime tracking
_SERVER_START_TIME = time.time()


class HealthHandler(BaseHandler):
    """Handler for health and readiness endpoints."""

    ROUTES = [
        "/healthz",
        "/readyz",
        "/api/health",
        "/api/health/detailed",
        "/api/health/deep",
        "/api/health/stores",
        "/api/health/sync",
        "/api/health/circuits",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route health endpoint requests."""
        handlers = {
            "/healthz": self._liveness_probe,
            "/readyz": self._readiness_probe,
            "/api/health": self._health_check,
            "/api/health/detailed": self._detailed_health_check,
            "/api/health/deep": self._deep_health_check,
            "/api/health/stores": self._database_stores_health,
            "/api/health/sync": self._sync_status,
            "/api/health/circuits": self._circuit_breakers_status,
        }

        endpoint_handler = handlers.get(path)
        if endpoint_handler:
            return endpoint_handler()
        return None

    def _liveness_probe(self) -> HandlerResult:
        """Kubernetes liveness probe - lightweight check that server is alive.

        Returns 200 if the server process is running and can respond.
        This should be very fast and not check external dependencies.
        Used by k8s to determine if the container should be restarted.

        Returns:
            {"status": "ok"} with 200 status
        """
        return json_response({"status": "ok"})

    def _readiness_probe(self) -> HandlerResult:
        """Kubernetes readiness probe - check if ready to serve traffic.

        Returns 200 if critical services are initialized and ready.
        Returns 503 if the service is not ready to accept traffic.
        Used by k8s to determine if traffic should be routed to this pod.

        Checks:
        - Storage initialized (if configured)
        - ELO system available (if configured)
        """
        ready = True
        checks: Dict[str, bool] = {}

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

        status_code = 200 if ready else 503
        return json_response(
            {"status": "ready" if ready else "not_ready", "checks": checks},
            status=status_code,
        )

    def _health_check(self) -> HandlerResult:
        """Comprehensive health check for k8s/docker deployments.

        Returns 200 when all critical services are healthy, 503 when degraded.
        Suitable for kubernetes liveness/readiness probes.

        Checks:
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
        except Exception as e:
            checks["rate_limiters"] = {"healthy": True, "error": str(e)[:80]}

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
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "response_time_ms": response_time_ms,
        }

        return json_response(health, status=status_code)

    def _check_filesystem_health(self) -> Dict[str, Any]:
        """Check filesystem write access to data directory."""
        import os
        import tempfile

        try:
            # Try to use nomic_dir first, then fall back to temp dir
            nomic_dir = self.get_nomic_dir()
            test_dir = (
                nomic_dir if nomic_dir and nomic_dir.exists() else Path(tempfile.gettempdir())
            )

            test_file = test_dir / f".health_check_{os.getpid()}"
            try:
                # Write test
                test_file.write_text("health_check")
                # Read verify
                content = test_file.read_text()
                if content != "health_check":
                    return {"healthy": False, "error": "Read verification failed"}
                return {"healthy": True, "path": str(test_dir)}
            finally:
                # Cleanup
                if test_file.exists():
                    test_file.unlink()

        except PermissionError as e:
            return {"healthy": False, "error": f"Permission denied: {e}"}
        except OSError as e:
            return {"healthy": False, "error": f"Filesystem error: {e}"}

    def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity if configured."""
        import os

        redis_url = os.environ.get("REDIS_URL") or os.environ.get("CACHE_REDIS_URL")

        if not redis_url:
            return {"healthy": True, "configured": False, "note": "Redis not configured"}

        try:
            import redis

            client = redis.from_url(redis_url, socket_timeout=2.0)
            ping_start = time.time()
            pong = client.ping()
            ping_latency = round((time.time() - ping_start) * 1000, 2)

            if pong:
                return {"healthy": True, "configured": True, "latency_ms": ping_latency}
            else:
                return {"healthy": False, "configured": True, "error": "Ping returned False"}

        except ImportError:
            return {"healthy": True, "configured": True, "warning": "redis package not installed"}
        except Exception as e:
            return {
                "healthy": False,
                "configured": True,
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }

    def _check_ai_providers_health(self) -> Dict[str, Any]:
        """Check AI provider API key availability."""
        import os

        providers = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "grok": "GROK_API_KEY",
            "xai": "XAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        available = {}
        for name, env_var in providers.items():
            key = os.environ.get(env_var)
            available[name] = key is not None and len(key) > 10

        any_available = any(available.values())
        available_count = sum(1 for v in available.values() if v)

        return {
            "healthy": True,
            "any_available": any_available,
            "available_count": available_count,
            "providers": available,
        }

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
            from aragora.memory.consensus import ConsensusMemory

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

        return json_response({
            "status": "healthy" if all_healthy else "degraded",
            "stores": stores,
            "elapsed_ms": elapsed_ms,
            "summary": {
                "total": len(stores),
                "healthy": sum(1 for s in stores.values() if s.get("healthy", False)),
                "connected": sum(1 for s in stores.values() if s.get("status") == "connected"),
                "not_initialized": sum(1 for s in stores.values() if s.get("status") == "not_initialized"),
            },
        })

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
                "timestamp": datetime.utcnow().isoformat() + "Z",
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
                        status.last_sync_at.isoformat() + "Z"
                        if status.last_sync_at
                        else None
                    ),
                    "last_error": status.last_error,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )

        except ImportError:
            return json_response(
                {
                    "enabled": False,
                    "error": "sync_service module not available",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )
        except Exception as e:
            logger.warning(f"Sync status check failed: {e}")
            return json_response(
                {
                    "enabled": False,
                    "error": f"{type(e).__name__}: {str(e)[:80]}",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
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
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )

        except ImportError:
            return json_response(
                {
                    "status": "unavailable",
                    "error": "resilience module not available",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )
        except Exception as e:
            logger.warning(f"Circuit breaker status check failed: {e}")
            return json_response(
                {
                    "status": "error",
                    "error": f"{type(e).__name__}: {str(e)[:80]}",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )
