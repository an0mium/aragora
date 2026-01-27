"""
Nomic loop state and monitoring endpoint handlers.

Endpoints:
- GET /api/nomic/state - Get nomic loop state
- GET /api/nomic/health - Get nomic loop health with stall detection
- GET /api/nomic/metrics - Get nomic loop Prometheus metrics summary
- GET /api/nomic/log - Get nomic loop logs
- GET /api/nomic/risk-register - Get risk register entries
- GET /api/nomic/witness/status - Get Gas Town witness patrol status
- GET /api/modes - Get available operational modes
- WS /api/nomic/stream - Real-time WebSocket event stream

Control endpoints (POST):
- POST /api/nomic/control/start - Start nomic loop
- POST /api/nomic/control/stop - Stop nomic loop
- POST /api/nomic/control/pause - Pause nomic loop
- POST /api/nomic/control/resume - Resume nomic loop
- POST /api/nomic/control/skip-phase - Skip current phase
- POST /api/nomic/proposals/approve - Approve proposal
- POST /api/nomic/proposals/reject - Reject proposal

Extracted from system.py for better modularity.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.server.stream.nomic_loop_stream import NomicLoopStreamServer

from aragora.server.http_utils import run_async as _run_async
from aragora.server.versioning.compat import strip_version_prefix

from .base import (
    HandlerResult,
    error_response,
    get_int_param,
    json_response,
    safe_error_message,
)
from .secure import SecureHandler, ForbiddenError, UnauthorizedError
from .utils.rate_limit import rate_limit

from aragora.audit.unified import audit_admin, audit_security

logger = logging.getLogger(__name__)


class NomicHandler(SecureHandler):
    """Handler for nomic loop state, monitoring, and control endpoints.

    Supports real-time WebSocket event streaming when a stream server is configured.

    RBAC Permissions:
    - nomic:read - View state, health, metrics, logs (GET operations)
    - nomic:admin - Control operations (start, stop, pause, resume, skip-phase, approve, reject)
    """

    RESOURCE_TYPE = "nomic"

    ROUTES = [
        "/api/nomic/state",
        "/api/nomic/health",
        "/api/nomic/metrics",
        "/api/nomic/log",
        "/api/nomic/risk-register",
        "/api/nomic/witness/status",
        "/api/nomic/control/start",
        "/api/nomic/control/stop",
        "/api/nomic/control/pause",
        "/api/nomic/control/resume",
        "/api/nomic/control/skip-phase",
        "/api/nomic/proposals",
        "/api/nomic/proposals/approve",
        "/api/nomic/proposals/reject",
        "/api/modes",
    ]

    def __init__(self, server_context: dict):
        """Initialize nomic handler.

        Args:
            server_context: Server context with shared resources
        """
        super().__init__(server_context)  # type: ignore[arg-type]
        self._stream: Optional["NomicLoopStreamServer"] = None

    def set_stream_server(self, stream: "NomicLoopStreamServer") -> None:
        """Set the WebSocket stream server for event emission.

        Args:
            stream: The Nomic Loop stream server instance
        """
        self._stream = stream

    def _get_stream(self) -> Optional["NomicLoopStreamServer"]:
        """Get the stream server from context or instance."""
        if self._stream:
            return self._stream
        return self.ctx.get("nomic_loop_stream")  # type: ignore[return-value]

    def _emit_event(
        self,
        emit_method: str,
        *args,
        max_retries: int = 3,
        base_delay: float = 0.1,
        **kwargs,
    ) -> None:
        """Emit an event to the Nomic Loop stream with retry logic.

        Args:
            emit_method: Name of the stream method to call (e.g., "emit_loop_started")
            *args: Positional arguments for the emit method
            max_retries: Maximum retry attempts on failure
            base_delay: Base delay between retries (exponential backoff)
            **kwargs: Keyword arguments for the emit method
        """
        stream = self._get_stream()
        if not stream:
            return

        method = getattr(stream, emit_method, None)
        if not method:
            return

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                _run_async(method(*args, **kwargs))
                return  # Success
            except (ConnectionError, OSError, TimeoutError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.debug(
                        f"Nomic stream emission attempt {attempt + 1} failed with connection error, "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
            except Exception as e:
                logger.warning(f"Nomic stream emission unexpected error: {e}")
                last_error = e
                break  # Don't retry unexpected errors

        logger.warning(
            f"Nomic stream emission failed after {max_retries} attempts "
            f"for {emit_method}: {last_error}"
        )

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        path = strip_version_prefix(path)
        return path in self.ROUTES or path.startswith("/api/nomic/")

    @rate_limit(rpm=30)
    async def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:  # type: ignore[override]
        """Route nomic endpoint requests."""
        # Require authentication for all nomic operations
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

        # Check read permission for GET endpoints
        try:
            self.check_permission(auth_context, "nomic:read")
        except ForbiddenError:
            logger.warning(f"Nomic read permission denied for user {auth_context.user_id}")
            return error_response("Permission denied: nomic:read", 403)

        path = strip_version_prefix(path)
        handlers = {
            "/api/nomic/state": self._get_nomic_state,
            "/api/nomic/health": self._get_nomic_health,
            "/api/nomic/metrics": self._get_nomic_metrics,
            "/api/nomic/proposals": self._get_proposals,
            "/api/nomic/witness/status": self._get_witness_status,
            "/api/modes": self._get_modes,
        }

        endpoint_handler = handlers.get(path)
        if endpoint_handler:
            return endpoint_handler()

        # Endpoints with parameters
        if path == "/api/nomic/log":
            lines = get_int_param(query_params, "lines", 100)
            lines = max(1, min(lines, 1000))  # Clamp to valid range
            return self._get_nomic_log(lines)

        if path == "/api/nomic/risk-register":
            limit = get_int_param(query_params, "limit", 50)
            limit = max(1, min(limit, 200))  # Clamp to valid range
            return self._get_risk_register(limit)

        return None

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
            return json_response(
                {
                    "lines": [line.rstrip() for line in recent],
                    "total": len(all_lines),
                    "showing": len(recent),
                }
            )
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
            return json_response(
                {
                    "status": "not_running",
                    "cycle": 0,
                    "phase": None,
                    "last_activity": None,
                    "stall_duration_seconds": None,
                    "warnings": [],
                }
            )

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

            return json_response(
                {
                    "status": "stalled" if stalled else "healthy",
                    "cycle": state.get("cycle", 0),
                    "phase": state.get("phase", "unknown"),
                    "last_activity": last_update,
                    "stall_duration_seconds": stall_duration,
                    "warnings": warnings,
                }
            )
        except json.JSONDecodeError as e:
            return json_response(
                {
                    "status": "error",
                    "error": f"Invalid state file: {e}",
                    "cycle": 0,
                }
            )
        except (OSError, PermissionError) as e:
            return json_response(
                {
                    "status": "error",
                    "error": f"{type(e).__name__}: {str(e)[:80]}",
                    "cycle": 0,
                }
            )

    def _get_nomic_metrics(self) -> HandlerResult:
        """Get nomic loop Prometheus metrics summary.

        Returns aggregated metrics for the nomic loop including:
        - Phase transitions counts
        - Cycle outcomes (success/failure)
        - Current phase
        - Circuit breaker states
        - Stuck phase detection

        This is useful for monitoring dashboards and alerting.
        """
        try:
            from aragora.nomic.metrics import (
                check_stuck_phases,
                get_nomic_metrics_summary,
            )

            summary = get_nomic_metrics_summary()
            stuck_info = check_stuck_phases(max_idle_seconds=1800)  # 30 min threshold

            return json_response(
                {
                    "summary": summary,
                    "stuck_detection": stuck_info,
                    "status": "stuck" if stuck_info.get("is_stuck") else "healthy",
                }
            )
        except ImportError:
            return json_response(
                {
                    "summary": {},
                    "stuck_detection": {"is_stuck": False},
                    "status": "metrics_unavailable",
                    "message": "Nomic metrics module not available",
                }
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("Invalid data in nomic metrics: %s", e)
            return error_response(f"Invalid metrics data: {e}", 500)
        except Exception as e:
            logger.exception("Unexpected error getting nomic metrics: %s", e)
            return error_response(f"Failed to get nomic metrics: {e}", 500)

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
            return json_response(
                {
                    "risks": [],
                    "total": 0,
                    "critical_count": 0,
                    "high_count": 0,
                }
            )

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

            return json_response(
                {
                    "risks": list(reversed(recent_risks)),  # Most recent first
                    "total": len(risks),
                    "critical_count": critical,
                    "high_count": high,
                }
            )
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
                    modes.append(
                        {
                            "slug": getattr(mode, "name", "").lower().replace(" ", "-"),
                            "name": getattr(mode, "name", ""),
                            "type": "custom",
                            "description": getattr(mode, "description", ""),
                        }
                    )
        except (ImportError, OSError, AttributeError, ValueError) as e:
            logger.debug(f"Could not load custom modes: {type(e).__name__}: {e}")

        return json_response({"modes": modes, "total": len(modes)})

    def _get_witness_status(self) -> HandlerResult:
        """Get witness patrol status and health report.

        Returns the current state of the Gas Town witness patrol, including:
        - patrolling: whether the patrol loop is active
        - statistics: patrol cycle stats (checks, alerts, escalations)
        - health_report: latest agent and convoy health summary
        - alerts: active alerts from the witness
        """
        try:
            from aragora.server.startup import get_witness_behavior

            witness = get_witness_behavior()
            if not witness:
                return json_response(
                    {
                        "patrolling": False,
                        "initialized": False,
                        "message": "Witness patrol not initialized",
                    }
                )

            # Build response from witness state
            response: dict[str, Any] = {
                "patrolling": witness._running,
                "initialized": True,
                "config": {
                    "patrol_interval_seconds": witness.config.patrol_interval_seconds,
                    "heartbeat_timeout_seconds": witness.config.heartbeat_timeout_seconds,
                    "stuck_threshold_minutes": witness.config.stuck_threshold_minutes,
                    "notify_mayor_on_critical": witness.config.notify_mayor_on_critical,
                },
                "statistics": {
                    "total_patrol_cycles": getattr(witness, "_patrol_cycles", 0),
                    "total_alerts_generated": len(witness._alerts)
                    if hasattr(witness, "_alerts")
                    else 0,
                    "agents_monitored": len(witness.hierarchy._assignments)
                    if witness.hierarchy
                    else 0,
                },
            }

            # Include active alerts
            if hasattr(witness, "_alerts") and witness._alerts:
                response["alerts"] = [
                    {
                        "id": alert.id,
                        "severity": alert.severity.value,
                        "target": alert.target,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "acknowledged": alert.acknowledged,
                    }
                    for alert in list(witness._alerts.values())[:10]  # Latest 10
                ]

            # Try to get health report if available
            try:
                report = _run_async(witness.generate_health_report())
                if report:
                    response["health_report"] = {
                        "report_id": report.report_id,
                        "overall_status": report.overall_status.value,
                        "agent_count": len(report.agent_checks),
                        "convoy_count": len(report.convoy_checks),
                        "recommendations": report.recommendations[:5],  # Top 5
                    }
            except Exception as e:
                logger.debug(f"Could not generate health report: {e}")

            return json_response(response)

        except ImportError as e:
            logger.debug(f"Witness behavior not available: {e}")
            return json_response(
                {
                    "patrolling": False,
                    "initialized": False,
                    "error": "Witness module not available",
                }
            )
        except Exception as e:
            logger.error(f"Failed to get witness status: {e}")
            return error_response(f"Failed to get witness status: {e}", 500)

    # =========================================================================
    # POST Handlers - Control Operations
    # =========================================================================

    @rate_limit(rpm=30)
    async def handle_post(  # type: ignore[override]
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests for control operations."""
        # Require authentication and admin permission for control operations
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

        # Control operations require admin permission
        try:
            self.check_permission(auth_context, "nomic:admin")
        except ForbiddenError:
            logger.warning(f"Nomic admin permission denied for user {auth_context.user_id}")
            return error_response("Permission denied: nomic:admin", 403)

        if path == "/api/v1/nomic/control/start":
            body = self.read_json_body(handler) or {}
            return self._start_nomic_loop(body)

        if path == "/api/v1/nomic/control/stop":
            body = self.read_json_body(handler) or {}
            return self._stop_nomic_loop(body)

        if path == "/api/v1/nomic/control/pause":
            return self._pause_nomic_loop()

        if path == "/api/v1/nomic/control/resume":
            return self._resume_nomic_loop()

        if path == "/api/v1/nomic/control/skip-phase":
            return self._skip_phase()

        if path == "/api/v1/nomic/proposals/approve":
            body = self.read_json_body(handler) or {}
            return self._approve_proposal(body)

        if path == "/api/v1/nomic/proposals/reject":
            body = self.read_json_body(handler) or {}
            return self._reject_proposal(body)

        return None

    def _start_nomic_loop(self, body: dict) -> HandlerResult:
        """Start the nomic loop with optional configuration."""
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            import subprocess

            # Check if already running
            state_file = nomic_dir / "nomic_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                if state.get("running"):
                    return error_response("Nomic loop is already running", 409)

            # Extract and validate configuration
            try:
                cycles = int(body.get("cycles", 1))
                max_cycles = int(body.get("max_cycles", 10))
            except (ValueError, TypeError):
                return error_response("cycles and max_cycles must be integers", 400)

            # Ensure positive bounds
            cycles = max(1, min(cycles, 100))  # Cap at 100 cycles
            max_cycles = max(1, min(max_cycles, 100))
            auto_approve = bool(body.get("auto_approve", False))

            # Start nomic loop as subprocess
            script_path = nomic_dir.parent.parent / "scripts" / "nomic_loop.py"
            if not script_path.exists():
                # Try alternate path
                script_path = nomic_dir.parent / "scripts" / "nomic_loop.py"

            if not script_path.exists():
                return error_response("Nomic loop script not found", 500)

            cmd = [
                "python",
                str(script_path),
                "--cycles",
                str(min(cycles, max_cycles)),
            ]
            if auto_approve:
                cmd.append("--auto-approve")

            # Start in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(nomic_dir.parent.parent),
                start_new_session=True,
            )

            # Update state file
            state = {
                "running": True,
                "pid": process.pid,
                "started_at": datetime.now().isoformat(),
                "cycle": 0,
                "phase": "starting",
                "target_cycles": cycles,
                "auto_approve": auto_approve,
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            # Emit loop started event
            self._emit_event(
                "emit_loop_started",
                cycles=cycles,
                auto_approve=auto_approve,
                dry_run=body.get("dry_run", False),
            )

            # Audit log
            audit_admin(
                admin_id="system",
                action="nomic_loop_started",
                target_type="nomic_loop",
                target_id=str(process.pid),
                target_cycles=cycles,
                auto_approve=auto_approve,
            )

            return json_response(
                {
                    "status": "started",
                    "pid": process.pid,
                    "target_cycles": cycles,
                },
                status=202,
            )

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in nomic state file: %s", e)
            return error_response(f"Invalid state file: {e}", 500)
        except (OSError, PermissionError) as e:
            logger.error("Failed to start nomic loop due to file/process error: %s", e)
            return error_response(f"Failed to start nomic loop: {e}", 500)
        except Exception as e:
            logger.exception("Unexpected error starting nomic loop: %s", e)
            return error_response(f"Failed to start nomic loop: {e}", 500)

    def _stop_nomic_loop(self, body: dict) -> HandlerResult:
        """Stop the running nomic loop."""
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            import signal
            import os

            state_file = nomic_dir / "nomic_state.json"
            if not state_file.exists():
                return error_response("Nomic loop is not running", 404)

            with open(state_file) as f:
                state = json.load(f)

            pid = state.get("pid")
            if not pid:
                return error_response("No PID found in state", 404)

            # Check if process exists
            try:
                os.kill(pid, 0)
            except OSError:
                # Process doesn't exist, update state
                state["running"] = False
                state["stopped_at"] = datetime.now().isoformat()
                with open(state_file, "w") as f:
                    json.dump(state, f, indent=2)
                return json_response({"status": "already_stopped"})

            # Send graceful shutdown signal
            graceful = body.get("graceful", True)
            sig = signal.SIGTERM if graceful else signal.SIGKILL
            os.kill(pid, sig)

            # Update state
            state["running"] = False
            state["stopped_at"] = datetime.now().isoformat()
            state["stopped_reason"] = "user_requested"
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            # Emit loop stopped event
            self._emit_event(
                "emit_loop_stopped",
                forced=not graceful,
                reason="user_requested",
            )

            # Audit log
            audit_admin(
                admin_id="system",
                action="nomic_loop_stopped",
                target_type="nomic_loop",
                target_id=str(pid),
                graceful=graceful,
                reason="user_requested",
            )

            return json_response(
                {
                    "status": "stopping" if graceful else "killed",
                    "pid": pid,
                }
            )

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in nomic state file: %s", e)
            return error_response(f"Invalid state file: {e}", 500)
        except (OSError, PermissionError) as e:
            logger.error("Failed to stop nomic loop due to file/process error: %s", e)
            return error_response(f"Failed to stop nomic loop: {e}", 500)
        except Exception as e:
            logger.exception("Unexpected error stopping nomic loop: %s", e)
            return error_response(f"Failed to stop nomic loop: {e}", 500)

    def _pause_nomic_loop(self) -> HandlerResult:
        """Pause the nomic loop at current phase."""
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            state_file = nomic_dir / "nomic_state.json"
            if not state_file.exists():
                return error_response("Nomic loop is not running", 404)

            with open(state_file) as f:
                state = json.load(f)

            if not state.get("running"):
                return error_response("Nomic loop is not running", 404)

            if state.get("paused"):
                return error_response("Nomic loop is already paused", 409)

            state["paused"] = True
            state["paused_at"] = datetime.now().isoformat()

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            # Emit loop paused event
            self._emit_event(
                "emit_loop_paused",
                current_phase=state.get("phase", "unknown"),
                current_cycle=state.get("cycle", 0),
            )

            # Audit log
            audit_admin(
                admin_id="system",
                action="nomic_loop_paused",
                target_type="nomic_loop",
                target_id=str(state.get("pid", "unknown")),
                current_phase=state.get("phase", "unknown"),
                current_cycle=state.get("cycle", 0),
            )

            return json_response(
                {
                    "status": "paused",
                    "cycle": state.get("cycle", 0),
                    "phase": state.get("phase", "unknown"),
                }
            )

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in nomic state file: %s", e)
            return error_response(f"Invalid state file: {e}", 500)
        except (OSError, PermissionError) as e:
            logger.error("Failed to pause nomic loop due to file error: %s", e)
            return error_response(f"Failed to pause nomic loop: {e}", 500)
        except Exception as e:
            logger.exception("Unexpected error pausing nomic loop: %s", e)
            return error_response(f"Failed to pause nomic loop: {e}", 500)

    def _resume_nomic_loop(self) -> HandlerResult:
        """Resume a paused nomic loop."""
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            state_file = nomic_dir / "nomic_state.json"
            if not state_file.exists():
                return error_response("Nomic loop is not running", 404)

            with open(state_file) as f:
                state = json.load(f)

            if not state.get("paused"):
                return error_response("Nomic loop is not paused", 409)

            state["paused"] = False
            state["resumed_at"] = datetime.now().isoformat()

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            # Emit loop resumed event
            self._emit_event(
                "emit_loop_resumed",
                current_phase=state.get("phase", "unknown"),
                current_cycle=state.get("cycle", 0),
            )

            # Audit log
            audit_admin(
                admin_id="system",
                action="nomic_loop_resumed",
                target_type="nomic_loop",
                target_id=str(state.get("pid", "unknown")),
                current_phase=state.get("phase", "unknown"),
                current_cycle=state.get("cycle", 0),
            )

            return json_response(
                {
                    "status": "resumed",
                    "cycle": state.get("cycle", 0),
                    "phase": state.get("phase", "unknown"),
                }
            )

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in nomic state file: %s", e)
            return error_response(f"Invalid state file: {e}", 500)
        except (OSError, PermissionError) as e:
            logger.error("Failed to resume nomic loop due to file error: %s", e)
            return error_response(f"Failed to resume nomic loop: {e}", 500)
        except Exception as e:
            logger.exception("Unexpected error resuming nomic loop: %s", e)
            return error_response(f"Failed to resume nomic loop: {e}", 500)

    def _skip_phase(self) -> HandlerResult:
        """Skip the current phase and move to the next."""
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            state_file = nomic_dir / "nomic_state.json"
            if not state_file.exists():
                return error_response("Nomic loop is not running", 404)

            with open(state_file) as f:
                state = json.load(f)

            current_phase = state.get("phase", "unknown")
            phases = ["context", "debate", "design", "implement", "verify"]

            if current_phase in phases:
                current_idx = phases.index(current_phase)
                next_idx = (current_idx + 1) % len(phases)
                next_phase = phases[next_idx]

                # If we're wrapping to context, increment cycle
                if next_phase == "context":
                    state["cycle"] = state.get("cycle", 0) + 1

                state["phase"] = next_phase
                state["skip_requested"] = True
                state["skipped_at"] = datetime.now().isoformat()

                with open(state_file, "w") as f:
                    json.dump(state, f, indent=2)

                # Emit phase skipped event
                self._emit_event(
                    "emit_phase_skipped",
                    phase=current_phase,
                    cycle=state.get("cycle", 0),
                    reason="user_requested",
                )

                # Audit log
                audit_admin(
                    admin_id="system",
                    action="nomic_phase_skipped",
                    target_type="nomic_phase",
                    target_id=current_phase,
                    previous_phase=current_phase,
                    next_phase=next_phase,
                    current_cycle=state.get("cycle", 0),
                )

                return json_response(
                    {
                        "status": "skip_requested",
                        "previous_phase": current_phase,
                        "next_phase": next_phase,
                        "cycle": state.get("cycle", 0),
                    }
                )
            else:
                return error_response(f"Unknown phase: {current_phase}", 400)

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in nomic state file: %s", e)
            return error_response(f"Invalid state file: {e}", 500)
        except (OSError, PermissionError) as e:
            logger.error("Failed to skip phase due to file error: %s", e)
            return error_response(f"Failed to skip phase: {e}", 500)
        except Exception as e:
            logger.exception("Unexpected error skipping phase: %s", e)
            return error_response(f"Failed to skip phase: {e}", 500)

    def _get_proposals(self) -> HandlerResult:
        """Get pending improvement proposals."""
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            proposals_file = nomic_dir / "proposals.json"
            if not proposals_file.exists():
                return json_response({"proposals": [], "total": 0})

            with open(proposals_file) as f:
                data = json.load(f)

            proposals = data.get("proposals", [])
            pending = [p for p in proposals if p.get("status") == "pending"]

            return json_response(
                {
                    "proposals": pending,
                    "total": len(pending),
                    "all_proposals": len(proposals),
                }
            )

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in proposals file: %s", e)
            return error_response(f"Invalid proposals file: {e}", 500)
        except (OSError, PermissionError) as e:
            logger.error("Failed to read proposals file: %s", e)
            return error_response(f"Failed to get proposals: {e}", 500)
        except Exception as e:
            logger.exception("Unexpected error getting proposals: %s", e)
            return error_response(f"Failed to get proposals: {e}", 500)

    def _approve_proposal(self, body: dict) -> HandlerResult:
        """Approve a pending proposal."""
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        proposal_id = body.get("proposal_id")
        if not proposal_id:
            return error_response("proposal_id is required", 400)

        try:
            proposals_file = nomic_dir / "proposals.json"
            if not proposals_file.exists():
                return error_response("No proposals found", 404)

            with open(proposals_file) as f:
                data = json.load(f)

            proposals = data.get("proposals", [])
            found = False
            for p in proposals:
                if p.get("id") == proposal_id:
                    p["status"] = "approved"
                    p["approved_at"] = datetime.now().isoformat()
                    p["approved_by"] = body.get("approved_by", "user")
                    found = True
                    break

            if not found:
                return error_response(f"Proposal not found: {proposal_id}", 404)

            data["proposals"] = proposals
            with open(proposals_file, "w") as f:
                json.dump(data, f, indent=2)

            # Emit proposal approved event
            self._emit_event(
                "emit_proposal_approved",
                proposal_id=proposal_id,
                approved_by=body.get("approved_by", "user"),
            )

            # Audit log (security-sensitive: approves code changes)
            audit_security(
                event_type="nomic_proposal_approved",
                actor_id=body.get("approved_by", "user"),
                resource_type="nomic_proposal",
                resource_id=proposal_id,
            )

            return json_response(
                {
                    "status": "approved",
                    "proposal_id": proposal_id,
                }
            )

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in proposals file: %s", e)
            return error_response(f"Invalid proposals file: {e}", 500)
        except (OSError, PermissionError) as e:
            logger.error("Failed to write proposals file: %s", e)
            return error_response(f"Failed to approve proposal: {e}", 500)
        except Exception as e:
            logger.exception("Unexpected error approving proposal: %s", e)
            return error_response(f"Failed to approve proposal: {e}", 500)

    def _reject_proposal(self, body: dict) -> HandlerResult:
        """Reject a pending proposal."""
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        proposal_id = body.get("proposal_id")
        if not proposal_id:
            return error_response("proposal_id is required", 400)

        try:
            proposals_file = nomic_dir / "proposals.json"
            if not proposals_file.exists():
                return error_response("No proposals found", 404)

            with open(proposals_file) as f:
                data = json.load(f)

            proposals = data.get("proposals", [])
            found = False
            for p in proposals:
                if p.get("id") == proposal_id:
                    p["status"] = "rejected"
                    p["rejected_at"] = datetime.now().isoformat()
                    p["rejected_by"] = body.get("rejected_by", "user")
                    p["rejection_reason"] = body.get("reason", "")
                    found = True
                    break

            if not found:
                return error_response(f"Proposal not found: {proposal_id}", 404)

            data["proposals"] = proposals
            with open(proposals_file, "w") as f:
                json.dump(data, f, indent=2)

            # Emit proposal rejected event
            self._emit_event(
                "emit_proposal_rejected",
                proposal_id=proposal_id,
                rejected_by=body.get("rejected_by", "user"),
                reason=body.get("reason", ""),
            )

            # Audit log (security-sensitive: rejects code changes)
            audit_security(
                event_type="nomic_proposal_rejected",
                actor_id=body.get("rejected_by", "user"),
                resource_type="nomic_proposal",
                resource_id=proposal_id,
                rejection_reason=body.get("reason", ""),
            )

            return json_response(
                {
                    "status": "rejected",
                    "proposal_id": proposal_id,
                }
            )

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in proposals file: %s", e)
            return error_response(f"Invalid proposals file: {e}", 500)
        except (OSError, PermissionError) as e:
            logger.error("Failed to write proposals file: %s", e)
            return error_response(f"Failed to reject proposal: {e}", 500)
        except Exception as e:
            logger.exception("Unexpected error rejecting proposal: %s", e)
            return error_response(f"Failed to reject proposal: {e}", 500)
