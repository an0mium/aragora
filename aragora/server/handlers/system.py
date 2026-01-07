"""
System and utility endpoint handlers.

Endpoints:
- GET /api/health - Health check
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
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
from .base import (
    BaseHandler, HandlerResult, json_response, error_response,
    get_int_param, get_string_param, validate_path_segment, SAFE_ID_PATTERN,
)


class SystemHandler(BaseHandler):
    """Handler for system-related endpoints."""

    ROUTES = [
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
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route system requests to appropriate methods."""
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
            return self._get_nomic_log(min(lines, 1000))

        if path == "/api/nomic/risk-register":
            limit = get_int_param(query_params, 'limit', 50)
            return self._get_risk_register(min(limit, 200))

        if path == "/api/modes":
            return self._get_modes()

        if path == "/api/history/cycles":
            loop_id = get_string_param(query_params, 'loop_id')
            if loop_id:
                is_valid, err = validate_path_segment(loop_id, "loop_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            limit = get_int_param(query_params, 'limit', 50)
            return self._get_history_cycles(loop_id, min(limit, 200))

        if path == "/api/history/events":
            loop_id = get_string_param(query_params, 'loop_id')
            if loop_id:
                is_valid, err = validate_path_segment(loop_id, "loop_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            limit = get_int_param(query_params, 'limit', 100)
            return self._get_history_events(loop_id, min(limit, 500))

        if path == "/api/history/debates":
            loop_id = get_string_param(query_params, 'loop_id')
            if loop_id:
                is_valid, err = validate_path_segment(loop_id, "loop_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            limit = get_int_param(query_params, 'limit', 50)
            return self._get_history_debates(loop_id, min(limit, 200))

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

        return None

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
        checks = {}
        all_healthy = True
        start_time = time.time()

        # Check database connectivity
        try:
            storage = self.get_storage()
            if storage is not None:
                # Try to execute a simple query to verify DB is responsive
                storage.list_debates(limit=1)
                checks["database"] = {"healthy": True, "latency_ms": 0}
            else:
                checks["database"] = {"healthy": False, "error": "Storage not initialized"}
                all_healthy = False
        except Exception as e:
            checks["database"] = {"healthy": False, "error": str(e)[:100]}
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
        except Exception as e:
            checks["elo_system"] = {"healthy": False, "error": str(e)[:100]}
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
        ws_manager = self.ctx.get("ws_manager")
        if ws_manager is not None:
            try:
                client_count = len(getattr(ws_manager, 'clients', []))
                checks["websocket"] = {"healthy": True, "active_clients": client_count}
            except Exception as e:
                checks["websocket"] = {"healthy": False, "error": str(e)[:100]}
        else:
            checks["websocket"] = {"healthy": True, "warning": "WebSocket manager not available"}

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

        health = {
            "status": "healthy",
            "components": {
                "storage": storage is not None,
                "elo_system": elo is not None,
                "nomic_dir": nomic_dir is not None and nomic_dir.exists() if nomic_dir else False,
            },
            "version": "1.0.0",
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
                    health["warnings"] = health.get("warnings", [])
                    health["warnings"].append(f"High failure rate: {failure_rate:.1%}")
                elif failure_rate > 0.3:
                    health["warnings"] = health.get("warnings", [])
                    health["warnings"].append(f"Elevated failure rate: {failure_rate:.1%}")

        except ImportError:
            health["observer"] = {"status": "unavailable", "reason": "module not found"}
        except Exception as e:
            health["observer"] = {"status": "error", "error": str(e)}

        # Add maintenance stats if available
        try:
            if nomic_dir:
                from aragora.maintenance import DatabaseMaintenance
                maintenance = DatabaseMaintenance(nomic_dir)
                health["maintenance"] = maintenance.get_stats()
        except ImportError:
            pass
        except Exception as e:
            health["maintenance"] = {"error": str(e)}

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
        except Exception as e:
            logger.debug(f"Could not get memory stats: {type(e).__name__}: {e}")

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
        except Exception as e:
            return error_response(f"Failed to read state: {e}", 500)

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
        except Exception as e:
            return error_response(f"Failed to read log: {e}", 500)

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
        except Exception as e:
            return json_response({
                "status": "error",
                "error": str(e),
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
        except Exception as e:
            return error_response(f"Failed to read risk register: {e}", 500)

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
                loader = CustomModeLoader(str(nomic_dir / "modes"))
                custom = loader.list_modes()
                for m in custom:
                    m["type"] = "custom"
                modes.extend(custom)
        except Exception as e:
            logger.debug(f"Could not load custom modes: {e}")

        return json_response({"modes": modes, "total": len(modes)})

    def _get_history_cycles(self, loop_id: Optional[str], limit: int) -> HandlerResult:
        """Get cycle history from Supabase or local storage."""
        try:
            # Try local storage first
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                cycles_file = nomic_dir / "cycles.json"
                if cycles_file.exists():
                    with open(cycles_file) as f:
                        cycles = json.load(f)
                    if loop_id:
                        cycles = [c for c in cycles if c.get("loop_id") == loop_id]
                    return json_response({"cycles": cycles[:limit]})

            return json_response({"cycles": []})
        except Exception as e:
            return error_response(f"Failed to get cycles: {e}", 500)

    def _get_history_events(self, loop_id: Optional[str], limit: int) -> HandlerResult:
        """Get event history."""
        try:
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                events_file = nomic_dir / "events.json"
                if events_file.exists():
                    with open(events_file) as f:
                        events = json.load(f)
                    if loop_id:
                        events = [e for e in events if e.get("loop_id") == loop_id]
                    return json_response({"events": events[:limit]})

            return json_response({"events": []})
        except Exception as e:
            return error_response(f"Failed to get events: {e}", 500)

    def _get_history_debates(self, loop_id: Optional[str], limit: int) -> HandlerResult:
        """Get debate history."""
        storage = self.get_storage()
        if not storage:
            return json_response({"debates": []})

        try:
            debates = storage.list_debates(limit=limit)
            if loop_id:
                debates = [d for d in debates if d.get("loop_id") == loop_id]
            return json_response({"debates": debates})
        except Exception as e:
            return error_response(f"Failed to get debates: {e}", 500)

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
                debates = storage.list_debates(limit=1000)
                summary["total_debates"] = len(debates)

            if elo:
                rankings = elo.get_leaderboard(limit=100)
                summary["total_agents"] = len(rankings)

            return json_response(summary)
        except Exception as e:
            return error_response(f"Failed to get summary: {e}", 500)

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
            result = {"task": task}

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
            return error_response(f"Maintenance failed: {e}", 500)
