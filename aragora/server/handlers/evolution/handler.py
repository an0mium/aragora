"""
Prompt evolution endpoint handlers.

Endpoints:
- GET /api/evolution/patterns - Get top patterns across all agents
- GET /api/evolution/summary - Get evolution summary statistics
- GET /api/evolution/{agent}/history - Get prompt evolution history for an agent
- GET /api/evolution/{agent}/prompt - Get current/specific prompt version for an agent
"""

from __future__ import annotations

import logging
from typing import Optional

from aragora.persistence.db_config import DatabaseType, get_db_path

from ..base import (
    SAFE_AGENT_PATTERN,
    BaseHandler,
    HandlerResult,
    error_response,
    get_int_param,
    get_string_param,
    json_response,
)
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for evolution endpoints (10 requests per minute - mutation ops are expensive)
_evolution_limiter = RateLimiter(requests_per_minute=10)

# Lazy imports for optional dependencies
EVOLUTION_AVAILABLE = False
PromptEvolver = None

try:
    from aragora.evolution.evolver import PromptEvolver as _PE

    PromptEvolver = _PE
    EVOLUTION_AVAILABLE = True
except ImportError:
    pass


class EvolutionHandler(BaseHandler):
    """Handler for prompt evolution endpoints."""

    ROUTES = [
        "/api/v1/evolution/patterns",
        "/api/v1/evolution/summary",
        "/api/v1/evolution/*/history",
        "/api/v1/evolution/*/prompt",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path == "/api/v1/evolution/patterns":
            return True
        if path == "/api/v1/evolution/summary":
            return True
        if path.startswith("/api/v1/evolution/") and path.endswith("/history"):
            return True
        if path.startswith("/api/v1/evolution/") and path.endswith("/prompt"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route evolution requests to appropriate methods."""
        if not path.startswith("/api/v1/evolution/"):
            return None

        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _evolution_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for evolution endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Global patterns endpoint
        if path == "/api/v1/evolution/patterns":
            pattern_type = get_string_param(query_params, "type")
            limit = get_int_param(query_params, "limit", 10)
            limit = min(max(limit, 1), 50)
            return self._get_patterns(pattern_type, limit)

        # Summary endpoint
        if path == "/api/v1/evolution/summary":
            return self._get_summary()

        # Agent-specific history endpoint
        if path.endswith("/history"):
            agent, err = self.extract_path_param(path, 3, "agent", SAFE_AGENT_PATTERN)
            if err:
                return err
            limit = get_int_param(query_params, "limit", 10)
            limit = min(max(limit, 1), 50)
            return self._get_evolution_history(agent, limit)

        # Agent-specific prompt endpoint
        if path.endswith("/prompt"):
            agent, err = self.extract_path_param(path, 3, "agent", SAFE_AGENT_PATTERN)
            if err:
                return err
            version = get_int_param(query_params, "version")
            return self._get_prompt_version(agent, version)

        return None

    def _get_evolution_history(self, agent: str, limit: int) -> HandlerResult:
        """Get prompt evolution history for an agent."""
        if not EVOLUTION_AVAILABLE or not PromptEvolver:
            return error_response("Prompt evolution not available", 503)

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            evolver = PromptEvolver(
                db_path=str(get_db_path(DatabaseType.PROMPT_EVOLUTION, nomic_dir))
            )
            history = evolver.get_evolution_history(agent, limit=limit)

            return json_response(
                {
                    "agent": agent,
                    "history": history,
                    "count": len(history),
                }
            )
        except Exception as e:
            logger.error(f"Error getting evolution history for {agent}: {e}", exc_info=True)
            return error_response("Failed to get evolution history", 500)

    def _get_patterns(self, pattern_type: Optional[str], limit: int) -> HandlerResult:
        """Get top evolution patterns across all agents."""
        if not EVOLUTION_AVAILABLE or not PromptEvolver:
            return error_response("Prompt evolution not available", 503)

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            evolver = PromptEvolver(
                db_path=str(get_db_path(DatabaseType.PROMPT_EVOLUTION, nomic_dir))
            )
            patterns = evolver.get_top_patterns(pattern_type=pattern_type, limit=limit)

            return json_response(
                {
                    "patterns": patterns,
                    "count": len(patterns),
                    "filter": pattern_type,
                }
            )
        except Exception as e:
            logger.error(f"Error getting evolution patterns: {e}", exc_info=True)
            return error_response("Failed to get evolution patterns", 500)

    def _get_prompt_version(self, agent: str, version: Optional[int]) -> HandlerResult:
        """Get a specific prompt version for an agent."""
        if not EVOLUTION_AVAILABLE or not PromptEvolver:
            return error_response("Prompt evolution not available", 503)

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            evolver = PromptEvolver(
                db_path=str(get_db_path(DatabaseType.PROMPT_EVOLUTION, nomic_dir))
            )
            prompt_version = evolver.get_prompt_version(agent, version)

            if not prompt_version:
                return error_response(f"No prompt version found for agent {agent}", 404)

            return json_response(
                {
                    "agent": agent,
                    "version": prompt_version.version,
                    "prompt": prompt_version.prompt,
                    "performance_score": prompt_version.performance_score,
                    "debates_count": prompt_version.debates_count,
                    "consensus_rate": prompt_version.consensus_rate,
                    "metadata": prompt_version.metadata,
                    "created_at": prompt_version.created_at,
                }
            )
        except Exception as e:
            logger.error(f"Error getting prompt version for {agent}: {e}", exc_info=True)
            return error_response("Failed to get prompt version", 500)

    def _get_summary(self) -> HandlerResult:
        """Get evolution summary statistics."""
        if not EVOLUTION_AVAILABLE or not PromptEvolver:
            return error_response("Prompt evolution not available", 503)

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            evolver = PromptEvolver(
                db_path=str(get_db_path(DatabaseType.PROMPT_EVOLUTION, nomic_dir))
            )

            # Get summary statistics from the database
            with evolver.connection() as conn:
                cursor = conn.cursor()

                # Count total prompt versions
                cursor.execute("SELECT COUNT(*) FROM prompt_versions")
                total_versions = cursor.fetchone()[0]

                # Count unique agents with evolutions
                cursor.execute("SELECT COUNT(DISTINCT agent_name) FROM prompt_versions")
                total_agents = cursor.fetchone()[0]

                # Count total patterns
                cursor.execute("SELECT COUNT(*) FROM extracted_patterns")
                total_patterns = cursor.fetchone()[0]

                # Get pattern type distribution
                cursor.execute(
                    """
                    SELECT pattern_type, COUNT(*) as count
                    FROM extracted_patterns
                    GROUP BY pattern_type
                    ORDER BY count DESC
                """
                )
                pattern_distribution = {row[0]: row[1] for row in cursor.fetchall()}

                # Get top performing agents
                cursor.execute(
                    """
                    SELECT agent_name, MAX(performance_score) as best_score,
                           MAX(version) as latest_version
                    FROM prompt_versions
                    GROUP BY agent_name
                    ORDER BY best_score DESC
                    LIMIT 10
                """
                )
                top_agents = [
                    {
                        "agent": row[0],
                        "best_score": row[1],
                        "latest_version": row[2],
                    }
                    for row in cursor.fetchall()
                ]

                # Get recent evolution activity
                cursor.execute(
                    """
                    SELECT agent_name, strategy, created_at
                    FROM evolution_history
                    ORDER BY created_at DESC
                    LIMIT 5
                """
                )
                recent_activity = [
                    {
                        "agent": row[0],
                        "strategy": row[1],
                        "created_at": row[2],
                    }
                    for row in cursor.fetchall()
                ]

            return json_response(
                {
                    "total_prompt_versions": total_versions,
                    "total_agents": total_agents,
                    "total_patterns": total_patterns,
                    "pattern_distribution": pattern_distribution,
                    "top_agents": top_agents,
                    "recent_activity": recent_activity,
                }
            )
        except Exception as e:
            logger.error(f"Error getting evolution summary: {e}", exc_info=True)
            return error_response("Failed to get evolution summary", 500)
