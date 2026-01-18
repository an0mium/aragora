"""
Nomic loop state and monitoring endpoint handlers.

Endpoints:
- GET /api/nomic/state - Get nomic loop state
- GET /api/nomic/health - Get nomic loop health with stall detection
- GET /api/nomic/metrics - Get nomic loop Prometheus metrics summary
- GET /api/nomic/log - Get nomic loop logs
- GET /api/nomic/risk-register - Get risk register entries
- GET /api/modes - Get available operational modes

Extracted from system.py for better modularity.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_int_param,
    json_response,
    safe_error_message,
)

logger = logging.getLogger(__name__)


class NomicHandler(BaseHandler):
    """Handler for nomic loop state, monitoring, and control endpoints."""

    ROUTES = [
        "/api/nomic/state",
        "/api/nomic/health",
        "/api/nomic/metrics",
        "/api/nomic/log",
        "/api/nomic/risk-register",
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

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES or path.startswith("/api/nomic/")

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route nomic endpoint requests."""
        handlers = {
            "/api/nomic/state": self._get_nomic_state,
            "/api/nomic/health": self._get_nomic_health,
            "/api/nomic/metrics": self._get_nomic_metrics,
            "/api/nomic/proposals": self._get_proposals,
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
        except Exception as e:
            logger.error("Error getting nomic metrics: %s", e, exc_info=True)
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

    # =========================================================================
    # POST Handlers - Control Operations
    # =========================================================================

    def handle_post(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests for control operations."""
        if path == "/api/nomic/control/start":
            body = self.read_json_body(handler) or {}
            return self._start_nomic_loop(body)

        if path == "/api/nomic/control/stop":
            body = self.read_json_body(handler) or {}
            return self._stop_nomic_loop(body)

        if path == "/api/nomic/control/pause":
            return self._pause_nomic_loop()

        if path == "/api/nomic/control/resume":
            return self._resume_nomic_loop()

        if path == "/api/nomic/control/skip-phase":
            return self._skip_phase()

        if path == "/api/nomic/proposals/approve":
            body = self.read_json_body(handler) or {}
            return self._approve_proposal(body)

        if path == "/api/nomic/proposals/reject":
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

            # Extract configuration
            cycles = body.get("cycles", 1)
            max_cycles = body.get("max_cycles", 10)
            auto_approve = body.get("auto_approve", False)

            # Start nomic loop as subprocess
            script_path = nomic_dir.parent.parent / "scripts" / "nomic_loop.py"
            if not script_path.exists():
                # Try alternate path
                script_path = nomic_dir.parent / "scripts" / "nomic_loop.py"

            if not script_path.exists():
                return error_response("Nomic loop script not found", 500)

            cmd = [
                "python", str(script_path),
                "--cycles", str(min(cycles, max_cycles)),
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

            return json_response({
                "status": "started",
                "pid": process.pid,
                "target_cycles": cycles,
            }, status=202)

        except Exception as e:
            logger.error("Failed to start nomic loop: %s", e, exc_info=True)
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

            return json_response({
                "status": "stopping" if graceful else "killed",
                "pid": pid,
            })

        except Exception as e:
            logger.error("Failed to stop nomic loop: %s", e, exc_info=True)
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

            return json_response({
                "status": "paused",
                "cycle": state.get("cycle", 0),
                "phase": state.get("phase", "unknown"),
            })

        except Exception as e:
            logger.error("Failed to pause nomic loop: %s", e, exc_info=True)
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

            return json_response({
                "status": "resumed",
                "cycle": state.get("cycle", 0),
                "phase": state.get("phase", "unknown"),
            })

        except Exception as e:
            logger.error("Failed to resume nomic loop: %s", e, exc_info=True)
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

                return json_response({
                    "status": "skip_requested",
                    "previous_phase": current_phase,
                    "next_phase": next_phase,
                    "cycle": state.get("cycle", 0),
                })
            else:
                return error_response(f"Unknown phase: {current_phase}", 400)

        except Exception as e:
            logger.error("Failed to skip phase: %s", e, exc_info=True)
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

            return json_response({
                "proposals": pending,
                "total": len(pending),
                "all_proposals": len(proposals),
            })

        except Exception as e:
            logger.error("Failed to get proposals: %s", e, exc_info=True)
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

            return json_response({
                "status": "approved",
                "proposal_id": proposal_id,
            })

        except Exception as e:
            logger.error("Failed to approve proposal: %s", e, exc_info=True)
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

            return json_response({
                "status": "rejected",
                "proposal_id": proposal_id,
            })

        except Exception as e:
            logger.error("Failed to reject proposal: %s", e, exc_info=True)
            return error_response(f"Failed to reject proposal: {e}", 500)
