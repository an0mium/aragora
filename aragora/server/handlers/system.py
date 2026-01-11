"""
System and utility endpoint handlers.

Endpoints:
- GET /healthz - Kubernetes liveness probe (lightweight)
- GET /readyz - Kubernetes readiness probe (checks dependencies)
- GET /api/health - Health check
- GET /api/health/detailed - Detailed health check with component status
- GET /api/nomic/state - Get nomic loop state
- GET /api/nomic/health - Get nomic loop health with stall detection
- GET /api/nomic/log - Get nomic loop logs
- GET /api/nomic/risk-register - Get risk register entries
- GET /api/modes - Get available operational modes
- GET /api/history/cycles - Get cycle history
- GET /api/history/events - Get event history
- GET /api/history/debates - Get debate history
- GET /api/history/summary - Get history summary
- GET /api/system/maintenance?task=<task> - Run database maintenance (status|vacuum|analyze|checkpoint|full)
- GET /api/openapi - OpenAPI 3.0 JSON specification
- GET /api/openapi.yaml - OpenAPI 3.0 YAML specification
- GET /api/docs - Swagger UI interactive documentation
- GET /api/auth/stats - Get authentication statistics
- POST /api/auth/revoke - Revoke a token to invalidate it
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
from .base import (
    BaseHandler, HandlerResult, json_response, error_response,
    get_int_param, get_string_param, validate_path_segment, SAFE_ID_PATTERN,
    ttl_cache, safe_error_message, handle_errors,
)

# Cache TTLs for system endpoints (in seconds)
CACHE_TTL_NOMIC_STATE = 10  # Short TTL for state (changes frequently)
CACHE_TTL_HISTORY = 60  # History queries
CACHE_TTL_OPENAPI = 3600  # OpenAPI spec (rarely changes)


class SystemHandler(BaseHandler):
    """Handler for system-related endpoints."""

    ROUTES = [
        # Kubernetes-standard health probes
        "/healthz",
        "/readyz",
        # Debug endpoint
        "/api/debug/test",
        # API health endpoints
        "/api/health",
        "/api/health/detailed",
        "/api/nomic/state",
        "/api/nomic/health",
        "/api/nomic/log",
        "/api/nomic/risk-register",
        "/api/modes",
        "/api/history/cycles",
        "/api/history/events",
        "/api/history/debates",
        "/api/history/summary",
        "/api/system/maintenance",
        "/api/openapi",
        "/api/openapi.json",
        "/api/openapi.yaml",
        # Swagger UI documentation
        "/api/docs",
        "/api/docs/",
        "/api/auth/stats",
        "/api/auth/revoke",
        # Prometheus metrics
        "/metrics",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route system requests to appropriate methods."""
        # Debug endpoint for testing
        if path == "/api/debug/test":
            method = getattr(handler, 'command', 'GET')
            return json_response({
                "status": "ok",
                "method": method,
                "message": "Modular handler works"
            })

        # Kubernetes-standard health probes
        if path == "/healthz":
            return self._liveness_probe()

        if path == "/readyz":
            return self._readiness_probe()

        if path == "/api/health":
            return self._health_check()

        if path == "/api/health/detailed":
            return self._detailed_health_check()

        if path == "/api/nomic/state":
            return self._get_nomic_state()

        if path == "/api/nomic/health":
            return self._get_nomic_health()

        if path == "/api/nomic/log":
            lines = get_int_param(query_params, 'lines', 100)
            return self._get_nomic_log(max(1, min(lines, 1000)))

        if path == "/api/nomic/risk-register":
            limit = get_int_param(query_params, 'limit', 50)
            return self._get_risk_register(max(1, min(limit, 200)))

        if path == "/api/modes":
            return self._get_modes()

        if path == "/api/history/cycles":
            loop_id = get_string_param(query_params, 'loop_id')
            if loop_id:
                is_valid, err = validate_path_segment(loop_id, "loop_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            limit = get_int_param(query_params, 'limit', 50)
            return self._get_history_cycles(loop_id, max(1, min(limit, 200)))

        if path == "/api/history/events":
            loop_id = get_string_param(query_params, 'loop_id')
            if loop_id:
                is_valid, err = validate_path_segment(loop_id, "loop_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            limit = get_int_param(query_params, 'limit', 100)
            return self._get_history_events(loop_id, max(1, min(limit, 500)))

        if path == "/api/history/debates":
            loop_id = get_string_param(query_params, 'loop_id')
            if loop_id:
                is_valid, err = validate_path_segment(loop_id, "loop_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            limit = get_int_param(query_params, 'limit', 50)
            return self._get_history_debates(loop_id, max(1, min(limit, 200)))

        if path == "/api/history/summary":
            loop_id = get_string_param(query_params, 'loop_id')
            if loop_id:
                is_valid, err = validate_path_segment(loop_id, "loop_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            return self._get_history_summary(loop_id)

        if path == "/api/system/maintenance":
            task = get_string_param(query_params, 'task', 'status')
            if task not in ('status', 'vacuum', 'analyze', 'checkpoint', 'full'):
                return error_response("Invalid task. Use: status, vacuum, analyze, checkpoint, full", 400)
            return self._run_maintenance(task)

        if path in ("/api/openapi", "/api/openapi.json"):
            return self._get_openapi_spec("json")

        if path == "/api/openapi.yaml":
            return self._get_openapi_spec("yaml")

        if path in ("/api/docs", "/api/docs/"):
            return self._get_swagger_ui()

        if path == "/api/auth/stats":
            return self._get_auth_stats()

        if path == "/metrics":
            return self._get_prometheus_metrics()

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle POST requests for auth endpoints."""
        if path == "/api/auth/revoke":
            return self._revoke_token(handler)
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
        """
        checks: Dict[str, Dict[str, Any]] = {}
        all_healthy = True
        start_time = time.time()

        # Check database connectivity
        try:
            storage = self.get_storage()
            if storage is not None:
                # Try to execute a simple query to verify DB is responsive
                storage.list_recent(limit=1)
                checks["database"] = {"healthy": True, "latency_ms": 0}
            else:
                checks["database"] = {"healthy": False, "error": "Storage not initialized"}
                all_healthy = False
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as e:
            checks["database"] = {"healthy": False, "error": f"{type(e).__name__}: {str(e)[:80]}"}
            all_healthy = False

        # Check ELO system
        try:
            elo = self.get_elo_system()
            if elo is not None:
                # Verify ELO system is functional
                elo.get_leaderboard(limit=1)
                checks["elo_system"] = {"healthy": True}
            else:
                checks["elo_system"] = {"healthy": False, "error": "ELO system not initialized"}
                all_healthy = False
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            checks["elo_system"] = {"healthy": False, "error": f"{type(e).__name__}: {str(e)[:80]}"}
            all_healthy = False

        # Check nomic directory
        nomic_dir = self.get_nomic_dir()
        if nomic_dir is not None and nomic_dir.exists():
            checks["nomic_dir"] = {"healthy": True, "path": str(nomic_dir)}
        else:
            checks["nomic_dir"] = {"healthy": False, "error": "Directory not found"}
            # Non-critical - don't fail health check for this
            checks["nomic_dir"]["healthy"] = True  # Downgrade to warning
            checks["nomic_dir"]["warning"] = "Nomic directory not configured"

        # Check WebSocket manager (if available in context)
        # Note: ws_manager is only available when running via AiohttpUnifiedServer.
        # When using standard HTTP handlers, WebSocket runs as a separate service.
        ws_manager = self.ctx.get("ws_manager")
        if ws_manager is not None:
            try:
                client_count = len(getattr(ws_manager, 'clients', []))
                checks["websocket"] = {"healthy": True, "active_clients": client_count}
            except (AttributeError, TypeError) as e:
                checks["websocket"] = {"healthy": False, "error": f"{type(e).__name__}: {str(e)[:80]}"}
        else:
            # WebSocket runs as separate aiohttp service - check via different mechanism
            checks["websocket"] = {"healthy": True, "note": "Managed by separate aiohttp server"}

        # Calculate response time
        response_time_ms = round((time.time() - start_time) * 1000, 2)

        status_code = 200 if all_healthy else 503
        health = {
            "status": "healthy" if all_healthy else "degraded",
            "checks": checks,
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "response_time_ms": response_time_ms,
        }

        return json_response(health, status=status_code)

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
            health["http_connector"] = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:80]}"}

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
            health["export_cache"] = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:80]}"}

        # Add handler cache status
        try:
            from aragora.server.handlers.cache import get_cache_stats
            cache_stats = get_cache_stats()
            health["handler_cache"] = {
                "status": "healthy",
                **cache_stats,
            }
        except ImportError:
            health["handler_cache"] = {"status": "unavailable"}
        except (RuntimeError, AttributeError, KeyError) as e:
            health["handler_cache"] = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:80]}"}

        return json_response(health)

    def _get_nomic_state(self) -> HandlerResult:
        """Get current nomic loop state."""
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        state_file = nomic_dir / "nomic_state.json"
        if not state_file.exists():
            return json_response({"state": "not_running", "cycle": 0})

        try:
            with open(state_file) as f:
                state = json.load(f)
            return json_response(state)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in nomic state file: %s", e)
            return error_response(f"Invalid state file format: {e}", 500)
        except (OSError, PermissionError) as e:
            logger.error("Failed to read nomic state: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "read state"), 500)

    def _get_nomic_log(self, lines: int) -> HandlerResult:
        """Get recent nomic loop log lines."""
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        log_file = nomic_dir / "nomic_loop.log"
        if not log_file.exists():
            return json_response({"lines": [], "total": 0})

        try:
            with open(log_file) as f:
                all_lines = f.readlines()

            recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return json_response({
                "lines": [line.rstrip() for line in recent],
                "total": len(all_lines),
                "showing": len(recent),
            })
        except (OSError, PermissionError, UnicodeDecodeError) as e:
            logger.error("Failed to read nomic log: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "read log"), 500)

    def _get_nomic_health(self) -> HandlerResult:
        """Get nomic loop health with stall detection.

        Returns health status including:
        - status: healthy | stalled | not_running | error
        - cycle: current cycle number
        - phase: current phase (debate, design, implement, verify)
        - last_activity: ISO timestamp of last activity
        - stall_duration_seconds: seconds since last activity if stalled
        - warnings: any active warnings
        """
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        state_file = nomic_dir / "nomic_state.json"
        if not state_file.exists():
            return json_response({
                "status": "not_running",
                "cycle": 0,
                "phase": None,
                "last_activity": None,
                "stall_duration_seconds": None,
                "warnings": [],
            })

        try:
            with open(state_file) as f:
                state = json.load(f)

            # Check for stall (no activity in 30 min)
            last_update = state.get("last_update") or state.get("updated_at")
            stall_threshold = 1800  # 30 minutes
            stalled = False
            stall_duration = None

            if last_update:
                try:
                    # Handle various ISO format variations
                    last_update_clean = last_update.replace("Z", "+00:00")
                    last_dt = datetime.fromisoformat(last_update_clean)
                    # Make comparison timezone-naive if needed
                    if last_dt.tzinfo is not None:
                        elapsed = (datetime.now(last_dt.tzinfo) - last_dt).total_seconds()
                    else:
                        elapsed = (datetime.now() - last_dt).total_seconds()

                    if elapsed > stall_threshold:
                        stalled = True
                        stall_duration = int(elapsed)
                except (ValueError, TypeError):
                    pass  # Invalid date format, can't determine stall

            # Collect warnings
            warnings = state.get("warnings", [])
            if stalled:
                warnings.append(f"No activity for {stall_duration // 60} minutes")

            return json_response({
                "status": "stalled" if stalled else "healthy",
                "cycle": state.get("cycle", 0),
                "phase": state.get("phase", "unknown"),
                "last_activity": last_update,
                "stall_duration_seconds": stall_duration,
                "warnings": warnings,
            })
        except json.JSONDecodeError as e:
            return json_response({
                "status": "error",
                "error": f"Invalid state file: {e}",
                "cycle": 0,
            })
        except (OSError, PermissionError) as e:
            return json_response({
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:80]}",
                "cycle": 0,
            })

    def _get_risk_register(self, limit: int) -> HandlerResult:
        """Get risk register entries.

        The risk register tracks identified issues, blockers, and concerns
        from the nomic loop execution.

        Returns:
            risks: List of recent risk entries
            total: Total number of entries
            critical_count: Number of critical severity risks
            high_count: Number of high severity risks
        """
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        risk_file = nomic_dir / "risk_register.jsonl"
        if not risk_file.exists():
            return json_response({
                "risks": [],
                "total": 0,
                "critical_count": 0,
                "high_count": 0,
            })

        try:
            risks = []
            with open(risk_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            risks.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines

            # Count by severity
            critical = sum(1 for r in risks if r.get("severity") == "critical")
            high = sum(1 for r in risks if r.get("severity") == "high")

            # Return most recent entries (last N)
            recent_risks = risks[-limit:] if len(risks) > limit else risks

            return json_response({
                "risks": list(reversed(recent_risks)),  # Most recent first
                "total": len(risks),
                "critical_count": critical,
                "high_count": high,
            })
        except (OSError, PermissionError, UnicodeDecodeError) as e:
            logger.error("Failed to read risk register: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "read risk register"), 500)

    def _get_modes(self) -> HandlerResult:
        """Get available operational modes (builtin + custom)."""
        modes = []

        # Add builtin modes
        builtin_modes = [
            {"name": "architect", "type": "builtin", "description": "Architecture planning mode"},
            {"name": "coder", "type": "builtin", "description": "Code implementation mode"},
            {"name": "debugger", "type": "builtin", "description": "Debugging and error analysis"},
            {"name": "orchestrator", "type": "builtin", "description": "Multi-agent orchestration"},
            {"name": "reviewer", "type": "builtin", "description": "Code review mode"},
        ]
        modes.extend(builtin_modes)

        # Add custom modes from nomic directory
        try:
            from aragora.modes.custom import CustomModeLoader
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                loader = CustomModeLoader([str(nomic_dir / "modes")])
                custom_modes = loader.load_all()
                for mode in custom_modes:
                    modes.append({
                        "slug": getattr(mode, 'name', '').lower().replace(" ", "-"),
                        "name": getattr(mode, 'name', ''),
                        "type": "custom",
                        "description": getattr(mode, 'description', ''),
                    })
        except (ImportError, OSError, AttributeError, ValueError) as e:
            logger.debug(f"Could not load custom modes: {type(e).__name__}: {e}")

        return json_response({"modes": modes, "total": len(modes)})

    def _load_filtered_json(
        self,
        file_path: Path,
        loop_id: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """Load JSON file with optional filtering and early termination.

        Filters during load to avoid loading entire large datasets when
        only a subset is needed.

        Args:
            file_path: Path to JSON file
            loop_id: Optional loop_id filter
            limit: Maximum items to return

        Returns:
            List of matching items, limited to `limit` entries

        Raises:
            json.JSONDecodeError: If file contains invalid JSON
            OSError: If file cannot be read
        """
        if not file_path.exists():
            return []

        with open(file_path) as f:
            data = json.load(f)

        if loop_id:
            # Filter with early termination
            filtered = []
            for item in data:
                if item.get("loop_id") == loop_id:
                    filtered.append(item)
                    if len(filtered) >= limit:
                        break
            return filtered
        else:
            return data[:limit]

    @ttl_cache(ttl_seconds=CACHE_TTL_HISTORY, key_prefix="history_cycles", skip_first=True)
    @handle_errors("get cycles")
    def _get_history_cycles(self, loop_id: Optional[str], limit: int) -> HandlerResult:
        """Get cycle history from Supabase or local storage."""
        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            cycles_file = nomic_dir / "cycles.json"
            cycles = self._load_filtered_json(cycles_file, loop_id, limit)
            return json_response({"cycles": cycles})

        return json_response({"cycles": []})

    @ttl_cache(ttl_seconds=CACHE_TTL_HISTORY, key_prefix="history_events", skip_first=True)
    @handle_errors("get events")
    def _get_history_events(self, loop_id: Optional[str], limit: int) -> HandlerResult:
        """Get event history."""
        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            events_file = nomic_dir / "events.json"
            events = self._load_filtered_json(events_file, loop_id, limit)
            return json_response({"events": events})

        return json_response({"events": []})

    @ttl_cache(ttl_seconds=CACHE_TTL_HISTORY, key_prefix="history_debates", skip_first=True)
    @handle_errors("get debates")
    def _get_history_debates(self, loop_id: Optional[str], limit: int) -> HandlerResult:
        """Get debate history."""
        storage = self.get_storage()
        if not storage:
            return json_response({"debates": []})

        # When filtering, fetch more to account for non-matching items
        fetch_limit = limit * 3 if loop_id else limit
        debate_metadata = storage.list_recent(limit=fetch_limit)

        if loop_id:
            # Filter with early termination
            debates = []
            for d in debate_metadata:
                item = d.__dict__ if hasattr(d, '__dict__') else d
                if item.get("loop_id") == loop_id:
                    debates.append(item)
                    if len(debates) >= limit:
                        break
        else:
            debates = [d.__dict__ if hasattr(d, '__dict__') else d for d in debate_metadata[:limit]]

        return json_response({"debates": debates})

    @ttl_cache(ttl_seconds=CACHE_TTL_HISTORY, key_prefix="history_summary", skip_first=True)
    def _get_history_summary(self, loop_id: Optional[str]) -> HandlerResult:
        """Get history summary statistics."""
        storage = self.get_storage()
        elo = self.get_elo_system()

        summary = {
            "total_debates": 0,
            "total_agents": 0,
            "total_matches": 0,
        }

        try:
            if storage:
                debates = storage.list_recent(limit=1000)
                summary["total_debates"] = len(debates)

            if elo:
                rankings = elo.get_leaderboard(limit=100)
                summary["total_agents"] = len(rankings)

            return json_response(summary)
        except Exception as e:
            logger.error("Failed to get history summary: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "get summary"), 500)

    def _run_maintenance(self, task: str) -> HandlerResult:
        """Run database maintenance tasks.

        Args:
            task: One of 'status', 'vacuum', 'analyze', 'checkpoint', 'full'

        Returns:
            Maintenance results including affected databases and stats.
        """
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            from aragora.maintenance import DatabaseMaintenance

            maintenance = DatabaseMaintenance(nomic_dir)
            result: dict[str, Any] = {"task": task}

            if task == "status":
                result["stats"] = maintenance.get_stats()

            elif task == "checkpoint":
                result["checkpoint"] = maintenance.checkpoint_all_wal()
                result["stats"] = maintenance.get_stats()

            elif task == "analyze":
                result["analyze"] = maintenance.analyze_all()
                result["stats"] = maintenance.get_stats()

            elif task == "vacuum":
                result["vacuum"] = maintenance.vacuum_all()
                result["stats"] = maintenance.get_stats()

            elif task == "full":
                # Run all maintenance tasks
                result["checkpoint"] = maintenance.checkpoint_all_wal()
                result["analyze"] = maintenance.analyze_all()
                result["vacuum"] = maintenance.vacuum_all()
                result["stats"] = maintenance.get_stats()

            return json_response(result)

        except ImportError:
            return error_response("Maintenance module not available", 503)
        except Exception as e:
            logger.exception(f"Maintenance task '{task}' failed: {e}")
            return error_response(safe_error_message(e, "maintenance"), 500)

    @ttl_cache(ttl_seconds=CACHE_TTL_OPENAPI, key_prefix="openapi_spec", skip_first=True)
    def _get_openapi_spec(self, format: str = "json") -> HandlerResult:
        """Get OpenAPI specification.

        Args:
            format: Output format - 'json' or 'yaml'

        Returns:
            OpenAPI 3.0 schema in requested format.
        """
        try:
            from aragora.server.openapi import handle_openapi_request

            content, content_type = handle_openapi_request(format=format)
            return HandlerResult(
                status_code=200,
                content_type=content_type,
                body=content.encode('utf-8') if isinstance(content, str) else content,
            )
        except ImportError:
            return error_response("OpenAPI module not available", 503)
        except Exception as e:
            logger.exception(f"OpenAPI generation failed: {e}")
            return error_response(safe_error_message(e, "OpenAPI generation"), 500)

    def _get_swagger_ui(self) -> HandlerResult:
        """Serve Swagger UI for interactive API documentation.

        Returns an HTML page that loads Swagger UI from CDN and points it
        to the /api/openapi.json endpoint.

        Returns:
            HTML page with embedded Swagger UI
        """
        swagger_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aragora API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>
        html { box-sizing: border-box; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin: 0; background: #fafafa; }
        .swagger-ui .topbar { display: none; }
        .swagger-ui .info { margin: 20px 0; }
        .swagger-ui .info .title { font-size: 2em; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: "/api/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                validatorUrl: null,
                docExpansion: "list",
                defaultModelsExpandDepth: 1,
                displayRequestDuration: true,
                filter: true,
                showExtensions: true,
                showCommonExtensions: true,
                persistAuthorization: true
            });
        };
    </script>
</body>
</html>"""
        return HandlerResult(
            status_code=200,
            content_type="text/html; charset=utf-8",
            body=swagger_html.encode("utf-8"),
        )

    def _get_auth_stats(self) -> HandlerResult:
        """Get authentication and rate limiting statistics.

        Returns:
            enabled: Whether auth is enabled
            rate_limit_stats: Current rate limiting counters
            revoked_tokens: Number of revoked tokens being tracked
        """
        from aragora.server.auth import auth_config

        stats = auth_config.get_rate_limit_stats()

        return json_response({
            "enabled": auth_config.enabled,
            "rate_limit_per_minute": auth_config.rate_limit_per_minute,
            "ip_rate_limit_per_minute": auth_config.ip_rate_limit_per_minute,
            "token_ttl_seconds": auth_config.token_ttl,
            "stats": stats,
        })

    def _revoke_token(self, handler) -> HandlerResult:
        """Revoke a token to invalidate it immediately.

        POST body:
            token: The token to revoke (required)
            reason: Optional reason for revocation

        Returns:
            success: Whether revocation succeeded
            revoked_count: Total revoked tokens being tracked
        """
        from aragora.server.auth import auth_config

        # Read request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        token = body.get("token")
        if not token:
            return error_response("Token is required", 400)

        reason = body.get("reason", "")

        # Revoke the token using both in-memory and persistent backends
        success = auth_config.revoke_token(token, reason)

        # Also persist revocation for multi-instance consistency
        try:
            from aragora.billing.jwt_auth import revoke_token_persistent
            persistent_ok = revoke_token_persistent(token)
            if not persistent_ok:
                logger.warning("Token revoked in-memory but persistent revocation failed")
        except ImportError:
            logger.debug("Persistent token blacklist not available")

        if success:
            logger.info("Token revoked: reason=%s", reason or "not specified")

        return json_response({
            "success": success,
            "revoked_count": auth_config.get_revocation_count(),
        })

    def _get_prometheus_metrics(self) -> HandlerResult:
        """Get Prometheus-format metrics.

        Exposes metrics for:
        - Subscription events and active subscriptions by tier
        - Usage (debates, tokens) by tier
        - API request counts and latency
        - Agent performance metrics

        Returns:
            Prometheus text format metrics
        """
        try:
            from aragora.server.metrics import generate_metrics

            metrics_text = generate_metrics()
            return HandlerResult(
                status_code=200,
                content_type="text/plain; version=0.0.4; charset=utf-8",
                body=metrics_text.encode("utf-8"),
            )
        except ImportError:
            return error_response("Metrics module not available", 503)
        except Exception as e:
            logger.exception(f"Metrics generation failed: {e}")
            return error_response(safe_error_message(e, "metrics"), 500)
