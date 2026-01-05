"""
System and utility endpoint handlers.

Endpoints:
- GET /api/health - Health check
- GET /api/nomic/state - Get nomic loop state
- GET /api/nomic/log - Get nomic loop logs
- GET /api/modes - Get available operational modes
- GET /api/history/cycles - Get cycle history
- GET /api/history/events - Get event history
- GET /api/history/debates - Get debate history
- GET /api/history/summary - Get history summary
"""

import json
from pathlib import Path
from typing import Optional
from .base import BaseHandler, HandlerResult, json_response, error_response, get_int_param


class SystemHandler(BaseHandler):
    """Handler for system-related endpoints."""

    ROUTES = [
        "/api/health",
        "/api/nomic/state",
        "/api/nomic/log",
        "/api/modes",
        "/api/history/cycles",
        "/api/history/events",
        "/api/history/debates",
        "/api/history/summary",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route system requests to appropriate methods."""
        if path == "/api/health":
            return self._health_check()

        if path == "/api/nomic/state":
            return self._get_nomic_state()

        if path == "/api/nomic/log":
            lines = get_int_param(query_params, 'lines', 100)
            return self._get_nomic_log(min(lines, 1000))

        if path == "/api/modes":
            return self._get_modes()

        if path == "/api/history/cycles":
            loop_id = query_params.get('loop_id')
            limit = get_int_param(query_params, 'limit', 50)
            return self._get_history_cycles(loop_id, min(limit, 200))

        if path == "/api/history/events":
            loop_id = query_params.get('loop_id')
            limit = get_int_param(query_params, 'limit', 100)
            return self._get_history_events(loop_id, min(limit, 500))

        if path == "/api/history/debates":
            loop_id = query_params.get('loop_id')
            limit = get_int_param(query_params, 'limit', 50)
            return self._get_history_debates(loop_id, min(limit, 200))

        if path == "/api/history/summary":
            loop_id = query_params.get('loop_id')
            return self._get_history_summary(loop_id)

        return None

    def _health_check(self) -> HandlerResult:
        """Return health check status."""
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

    def _get_modes(self) -> HandlerResult:
        """Get available operational modes."""
        try:
            from aragora.modes.custom import CustomModeLoader
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                loader = CustomModeLoader(str(nomic_dir / "modes"))
                modes = loader.list_modes()
                return json_response({"modes": modes})
            return json_response({"modes": []})
        except Exception as e:
            return error_response(f"Failed to get modes: {e}", 500)

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
