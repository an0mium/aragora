"""
Critique pattern and reputation endpoint handlers.

Endpoints:
- GET /api/critiques/patterns - Get high-impact critique patterns
- GET /api/critiques/archive - Get archive statistics
- GET /api/reputation/all - Get all agent reputations
- GET /api/agent/:name/reputation - Get specific agent reputation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

from aragora.server.validation import validate_agent_name_with_version
from aragora.utils.optional_imports import try_import_class

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_bounded_float_param,
    get_clamped_int_param,
    json_response,
)
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for critique endpoints (60 requests per minute - read-heavy)
_critique_limiter = RateLimiter(requests_per_minute=60)

# Lazy import for optional dependency using centralized utility
CritiqueStore, CRITIQUE_STORE_AVAILABLE = try_import_class("aragora.memory.store", "CritiqueStore")

from aragora.server.errors import safe_error_message as _safe_error_message


class CritiqueHandler(BaseHandler):
    """Handler for critique pattern and reputation endpoints."""

    ROUTES = [
        "/api/v1/critiques/patterns",
        "/api/v1/critiques/archive",
        "/api/v1/reputation/all",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Dynamic route for agent reputation
        if path.startswith("/api/v1/agent/") and path.endswith("/reputation"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route critique requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _critique_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for critique endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        nomic_dir = self.ctx.get("nomic_dir")

        if path == "/api/v1/critiques/patterns":
            limit = get_clamped_int_param(query_params, "limit", 10, min_val=1, max_val=50)
            min_success = get_bounded_float_param(
                query_params, "min_success", 0.5, min_val=0.0, max_val=1.0
            )
            return self._get_critique_patterns(nomic_dir, limit, min_success)

        if path == "/api/v1/critiques/archive":
            return self._get_archive_stats(nomic_dir)

        if path == "/api/v1/reputation/all":
            return self._get_all_reputations(nomic_dir)

        if path.startswith("/api/v1/agent/") and path.endswith("/reputation"):
            agent = self._extract_agent_name(path)
            if agent is None:
                return error_response("Invalid agent name", 400)
            return self._get_agent_reputation(nomic_dir, agent)

        return None

    def _extract_agent_name(self, path: str) -> Optional[str]:
        """Extract and validate agent name from path."""
        # Pattern: /api/agent/{name}/reputation
        # Block path traversal attempts
        if ".." in path:
            return None
        parts = path.split("/")
        if len(parts) >= 4:
            agent = parts[3]
            is_valid, _ = validate_agent_name_with_version(agent)
            if is_valid:
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

            return json_response(
                {
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
                }
            )
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
            return json_response(
                {
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
                }
            )
        except Exception as e:
            return error_response(_safe_error_message(e, "reputations"), 500)

    def _get_agent_reputation(self, nomic_dir: Optional[Path], agent: str) -> HandlerResult:
        """Get reputation for a specific agent."""
        if not CRITIQUE_STORE_AVAILABLE:
            return error_response("Critique store not available", 503)

        try:
            db_path = nomic_dir / "debates.db" if nomic_dir else None
            if not db_path or not db_path.exists():
                return json_response(
                    {"agent": agent, "reputation": None, "message": "No reputation data available"}
                )

            store = CritiqueStore(str(db_path))
            rep = store.get_reputation(agent)

            if rep:
                return json_response(
                    {
                        "agent": agent,
                        "reputation": {
                            "score": rep.reputation_score,
                            "vote_weight": rep.vote_weight,
                            "proposal_acceptance_rate": rep.proposal_acceptance_rate,
                            "critique_value": rep.critique_value,
                            "debates_participated": rep.debates_participated,
                        },
                    }
                )
            else:
                return json_response(
                    {"agent": agent, "reputation": None, "message": "Agent not found"}
                )
        except Exception as e:
            return error_response(_safe_error_message(e, "agent_reputation"), 500)
