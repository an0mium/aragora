"""
Critique pattern and reputation endpoint handlers.

Endpoints:
- GET /api/critiques/patterns - Get high-impact critique patterns
- GET /api/critiques/archive - Get archive statistics
- GET /api/reputation/all - Get all agent reputations
- GET /api/agent/:name/reputation - Get specific agent reputation
"""

import logging
import re
from pathlib import Path
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_clamped_int_param,
    get_bounded_float_param,
)
from aragora.server.validation import SAFE_ID_PATTERN_WITH_DOTS as SAFE_ID_PATTERN
from aragora.utils.optional_imports import try_import_class

logger = logging.getLogger(__name__)

# Lazy import for optional dependency using centralized utility
CritiqueStore, CRITIQUE_STORE_AVAILABLE = try_import_class(
    "aragora.memory.store", "CritiqueStore"
)

from aragora.server.error_utils import safe_error_message as _safe_error_message


class CritiqueHandler(BaseHandler):
    """Handler for critique pattern and reputation endpoints."""

    ROUTES = [
        "/api/critiques/patterns",
        "/api/critiques/archive",
        "/api/reputation/all",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Dynamic route for agent reputation
        if path.startswith("/api/agent/") and path.endswith("/reputation"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route critique requests to appropriate methods."""
        nomic_dir = self.ctx.get("nomic_dir")

        if path == "/api/critiques/patterns":
            limit = get_clamped_int_param(query_params, 'limit', 10, min_val=1, max_val=50)
            min_success = get_bounded_float_param(query_params, 'min_success', 0.5, min_val=0.0, max_val=1.0)
            return self._get_critique_patterns(nomic_dir, limit, min_success)

        if path == "/api/critiques/archive":
            return self._get_archive_stats(nomic_dir)

        if path == "/api/reputation/all":
            return self._get_all_reputations(nomic_dir)

        if path.startswith("/api/agent/") and path.endswith("/reputation"):
            agent = self._extract_agent_name(path)
            if agent is None:
                return error_response("Invalid agent name", 400)
            return self._get_agent_reputation(nomic_dir, agent)

        return None

    def _extract_agent_name(self, path: str) -> Optional[str]:
        """Extract and validate agent name from path."""
        # Pattern: /api/agent/{name}/reputation
        # Block path traversal attempts
        if '..' in path:
            return None
        parts = path.split('/')
        if len(parts) >= 4:
            agent = parts[3]
            if re.match(SAFE_ID_PATTERN, agent):
                return agent
        return None

    def _get_critique_patterns(
        self, nomic_dir: Optional[Path], limit: int, min_success: float
    ) -> HandlerResult:
        """Get high-impact critique patterns for learning."""
        if not CRITIQUE_STORE_AVAILABLE:
            return error_response("Critique store not available", 503)

        try:
            db_path = nomic_dir / "debates.db" if nomic_dir else None
            if not db_path or not db_path.exists():
                return json_response({"patterns": [], "count": 0})

            store = CritiqueStore(str(db_path))
            patterns = store.retrieve_patterns(min_success_rate=min_success, limit=limit)
            stats = store.get_stats()

            return json_response({
                "patterns": [
                    {
                        "issue_type": p.issue_type,
                        "pattern": p.pattern_text,
                        "success_rate": p.success_rate,
                        "usage_count": p.usage_count,
                    }
                    for p in patterns
                ],
                "count": len(patterns),
                "stats": stats,
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "critique_patterns"), 500)

    def _get_archive_stats(self, nomic_dir: Optional[Path]) -> HandlerResult:
        """Get archive statistics from critique store."""
        if not CRITIQUE_STORE_AVAILABLE:
            return error_response("Critique store not available", 503)

        try:
            db_path = nomic_dir / "debates.db" if nomic_dir else None
            if not db_path or not db_path.exists():
                return json_response({"archived": 0, "by_type": {}})

            store = CritiqueStore(str(db_path))
            stats = store.get_archive_stats()
            return json_response(stats)
        except Exception as e:
            return error_response(_safe_error_message(e, "archive_stats"), 500)

    def _get_all_reputations(self, nomic_dir: Optional[Path]) -> HandlerResult:
        """Get all agent reputations ranked by score."""
        if not CRITIQUE_STORE_AVAILABLE:
            return error_response("Critique store not available", 503)

        try:
            db_path = nomic_dir / "debates.db" if nomic_dir else None
            if not db_path or not db_path.exists():
                return json_response({"reputations": [], "count": 0})

            store = CritiqueStore(str(db_path))
            reputations = store.get_all_reputations()
            return json_response({
                "reputations": [
                    {
                        "agent": r.agent_name,
                        "score": r.reputation_score,
                        "vote_weight": r.vote_weight,
                        "proposal_acceptance_rate": r.proposal_acceptance_rate,
                        "critique_value": r.critique_value,
                        "debates_participated": r.debates_participated,
                    }
                    for r in reputations
                ],
                "count": len(reputations),
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "reputations"), 500)

    def _get_agent_reputation(
        self, nomic_dir: Optional[Path], agent: str
    ) -> HandlerResult:
        """Get reputation for a specific agent."""
        if not CRITIQUE_STORE_AVAILABLE:
            return error_response("Critique store not available", 503)

        try:
            db_path = nomic_dir / "debates.db" if nomic_dir else None
            if not db_path or not db_path.exists():
                return json_response({
                    "agent": agent,
                    "reputation": None,
                    "message": "No reputation data available"
                })

            store = CritiqueStore(str(db_path))
            rep = store.get_reputation(agent)

            if rep:
                return json_response({
                    "agent": agent,
                    "reputation": {
                        "score": rep.reputation_score,
                        "vote_weight": rep.vote_weight,
                        "proposal_acceptance_rate": rep.proposal_acceptance_rate,
                        "critique_value": rep.critique_value,
                        "debates_participated": rep.debates_participated,
                    }
                })
            else:
                return json_response({
                    "agent": agent,
                    "reputation": None,
                    "message": "Agent not found"
                })
        except Exception as e:
            return error_response(_safe_error_message(e, "agent_reputation"), 500)
