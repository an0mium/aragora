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
    """Handler for nomic loop state and monitoring endpoints."""

    ROUTES = [
        "/api/nomic/state",
        "/api/nomic/health",
        "/api/nomic/metrics",
        "/api/nomic/log",
        "/api/nomic/risk-register",
        "/api/modes",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route nomic endpoint requests."""
        handlers = {
            "/api/nomic/state": self._get_nomic_state,
            "/api/nomic/health": self._get_nomic_health,
            "/api/nomic/metrics": self._get_nomic_metrics,
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
