"""
Agent SelectionFeedbackLoop state API handler.

Endpoints:
- GET /api/v1/agents/feedback/metrics    - Loop metrics summary
- GET /api/v1/agents/feedback/states     - Per-agent win rate, timeout rate, calibration
- GET /api/v1/agents/{name}/feedback/domains - Domain-specific weights
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.versioning.compat import strip_version_prefix

from ..base import (
    SAFE_AGENT_PATTERN,
    BaseHandler,
    HandlerResult,
    get_string_param,
    json_response,
)

logger = logging.getLogger(__name__)


def _get_feedback_loop() -> Any:
    """Get the global SelectionFeedbackLoop if available."""
    try:
        from aragora.debate.selection_feedback import SelectionFeedbackLoop

        return SelectionFeedbackLoop()
    except (ImportError, AttributeError):
        pass
    return None


class FeedbackHandler(BaseHandler):
    """Handler for SelectionFeedbackLoop state endpoints."""

    ROUTES = [
        "/api/v1/agents/feedback/metrics",
        "/api/v1/agents/feedback/states",
        "/api/v1/agents/*/feedback/domains",
    ]

    def __init__(self, ctx: dict[str, Any] | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        cleaned = strip_version_prefix(path)
        if cleaned in ("/api/agents/feedback/metrics", "/api/agents/feedback/states"):
            return True
        # /api/agents/{name}/feedback/domains
        if cleaned.startswith("/api/agents/") and cleaned.endswith("/feedback/domains"):
            return True
        return False

    @require_permission("agents:read")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route GET requests."""
        cleaned = strip_version_prefix(path)

        if cleaned == "/api/agents/feedback/metrics":
            return self._get_metrics()

        if cleaned == "/api/agents/feedback/states":
            return self._get_states()

        # /api/agents/{name}/feedback/domains
        if cleaned.startswith("/api/agents/") and cleaned.endswith("/feedback/domains"):
            # Extract agent name: /api/agents/{name}/feedback/domains
            # parts: ["", "api", "agents", "{name}", "feedback", "domains"]
            agent_name, err = self.extract_path_param(cleaned, 3, "agent_name", SAFE_AGENT_PATTERN)
            if err:
                return err
            domain = get_string_param(query_params, "domain")
            return self._get_domain_weights(agent_name, domain)

        return None

    def _get_metrics(self) -> HandlerResult:
        """Return feedback loop metrics summary."""
        loop = _get_feedback_loop()
        if not loop:
            return json_response(
                {
                    "debates_processed": 0,
                    "adjustments_computed": 0,
                    "agents_tracked": 0,
                    "average_adjustment": 0.0,
                    "last_processed": None,
                }
            )

        return json_response(loop.get_metrics())

    def _get_states(self) -> HandlerResult:
        """Return per-agent feedback states."""
        loop = _get_feedback_loop()
        if not loop:
            return json_response({"agents": {}, "count": 0})

        all_states = loop.get_all_states()
        agents: dict[str, dict[str, Any]] = {}
        for name, state in all_states.items():
            agents[name] = {
                "total_debates": state.total_debates,
                "wins": state.wins,
                "losses": state.losses,
                "timeouts": state.timeouts,
                "win_rate": round(state.win_rate, 4),
                "timeout_rate": round(state.timeout_rate, 4),
                "calibration_score": round(state.calibration_score, 4),
                "avg_confidence": round(state.avg_confidence, 4),
                "avg_response_time_ms": round(state.avg_response_time_ms, 2),
            }

        return json_response({"agents": agents, "count": len(agents)})

    def _get_domain_weights(self, agent_name: str, domain: str | None) -> HandlerResult:
        """Return domain-specific weights for an agent."""
        loop = _get_feedback_loop()
        if not loop:
            return json_response(
                {
                    "agent": agent_name,
                    "domains": {},
                }
            )

        state = loop.get_agent_state(agent_name)
        if not state:
            return json_response(
                {
                    "agent": agent_name,
                    "domains": {},
                }
            )

        # If a specific domain is requested, return just that
        if domain:
            weight = loop.get_domain_adjustment(agent_name, domain)
            return json_response(
                {
                    "agent": agent_name,
                    "domains": {domain: round(weight, 4)},
                }
            )

        # Return all domains this agent has participated in
        domains: dict[str, float] = {}
        all_domains = set(state.domain_wins.keys()) | set(state.domain_losses.keys())
        for d in sorted(all_domains):
            domains[d] = round(loop.get_domain_adjustment(agent_name, d), 4)

        return json_response(
            {
                "agent": agent_name,
                "domains": domains,
            }
        )
