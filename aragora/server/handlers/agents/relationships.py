"""
Agent Relationship Summary API handler.

Endpoints:
- GET /api/v1/agents/{name}/relationships          - Rivals + allies for an agent
- GET /api/v1/agents/{name}/relationships/{other}   - Pairwise metrics between two agents
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
    get_int_param,
    json_response,
)

logger = logging.getLogger(__name__)


def _get_relationship_tracker() -> Any:
    """Get RelationshipTracker from the ELO system database."""
    try:
        from aragora.persistence.db_config import get_default_data_dir
        from aragora.ranking.relationships import RelationshipTracker

        db_path = get_default_data_dir() / "elo_rankings.db"
        return RelationshipTracker(db_path)
    except (ImportError, OSError):
        return None


class RelationshipHandler(BaseHandler):
    """Handler for agent relationship summary endpoints."""

    ROUTES = [
        "/api/v1/agents/*/relationships",
        "/api/v1/agents/*/relationships/*",
    ]

    def __init__(self, ctx: dict[str, Any] | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        cleaned = strip_version_prefix(path)
        # /api/agents/{name}/relationships or /api/agents/{name}/relationships/{other}
        if not cleaned.startswith("/api/agents/"):
            return False
        parts = cleaned.split("/")
        # ["", "api", "agents", "{name}", "relationships"] = 5 parts
        # ["", "api", "agents", "{name}", "relationships", "{other}"] = 6 parts
        if len(parts) >= 5 and parts[4] == "relationships":
            return True
        return False

    @require_permission("agents:read")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route GET requests."""
        cleaned = strip_version_prefix(path)
        parts = cleaned.split("/")
        # ["", "api", "agents", "{name}", "relationships", ...]

        if len(parts) < 5 or parts[4] != "relationships":
            return None

        agent_name, err = self.extract_path_param(cleaned, 3, "agent_name", SAFE_AGENT_PATTERN)
        if err:
            return err

        # Pairwise: /api/agents/{name}/relationships/{other}
        if len(parts) == 6 and parts[5]:
            other_name, err = self.extract_path_param(cleaned, 5, "other_agent", SAFE_AGENT_PATTERN)
            if err:
                return err
            return self._get_pairwise(agent_name, other_name)

        # Summary: /api/agents/{name}/relationships
        limit = get_int_param(query_params, "limit", 5)
        limit = max(1, min(limit, 20))
        return self._get_summary(agent_name, limit)

    def _get_summary(self, agent_name: str, limit: int) -> HandlerResult:
        """Return rivals and allies for an agent."""
        tracker = _get_relationship_tracker()
        if not tracker:
            return json_response(
                {
                    "agent": agent_name,
                    "rivals": [],
                    "allies": [],
                }
            )

        rivals_raw = tracker.get_rivals(agent_name, limit=limit)
        allies_raw = tracker.get_allies(agent_name, limit=limit)

        rivals = [
            {
                "agent": r.agent_b if r.agent_a == agent_name else r.agent_a,
                "rivalry_score": r.rivalry_score,
                "debate_count": r.debate_count,
                "relationship": r.relationship,
            }
            for r in rivals_raw
        ]

        allies = [
            {
                "agent": a.agent_b if a.agent_a == agent_name else a.agent_a,
                "alliance_score": a.alliance_score,
                "debate_count": a.debate_count,
                "relationship": a.relationship,
            }
            for a in allies_raw
        ]

        return json_response(
            {
                "agent": agent_name,
                "rivals": rivals,
                "allies": allies,
            }
        )

    def _get_pairwise(self, agent_a: str, agent_b: str) -> HandlerResult:
        """Return pairwise metrics between two agents."""
        tracker = _get_relationship_tracker()
        if not tracker:
            return json_response(
                {
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "debate_count": 0,
                    "relationship": "unknown",
                }
            )

        metrics = tracker.compute_metrics(agent_a, agent_b)

        return json_response(
            {
                "agent_a": metrics.agent_a,
                "agent_b": metrics.agent_b,
                "debate_count": metrics.debate_count,
                "rivalry_score": metrics.rivalry_score,
                "alliance_score": metrics.alliance_score,
                "relationship": metrics.relationship,
                "agreement_rate": metrics.agreement_rate,
                "head_to_head": metrics.head_to_head,
            }
        )
