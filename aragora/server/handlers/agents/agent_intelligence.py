"""Agent intelligence endpoint methods (AgentIntelligenceMixin).

Metadata, introspection, head-to-head, and opponent briefing endpoints.
Extracted from agents.py to reduce file size.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aragora.config import CACHE_TTL_AGENT_H2H

if TYPE_CHECKING:
    from aragora.ranking.elo import EloSystem
    from aragora.storage.postgres_store import PostgresStore
from aragora.persistence.db_config import DatabaseType, get_db_path

from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    ttl_cache,
)
from ..openapi_decorator import api_endpoint

logger = logging.getLogger(__name__)


class AgentIntelligenceMixin:
    """Mixin providing agent intelligence and analysis endpoints.

    Expects the composing class to provide:
    - get_nomic_dir() -> Path | None
    - get_elo_system() -> EloSystem | None
    - get_storage() -> PostgresStore | None

    These are provided by BaseHandler.
    """

    if TYPE_CHECKING:

        def get_nomic_dir(self) -> Path | None: ...
        def get_elo_system(self) -> EloSystem | None: ...
        def get_storage(self) -> PostgresStore | None: ...

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/metadata",
        summary="Get agent metadata and capabilities",
        tags=["Agents"],
    )
    @handle_errors("agent metadata")
    def _get_metadata(self, agent: str) -> HandlerResult:
        """Get rich metadata about an agent.

        Returns model information, capabilities, and provider details
        from the agent_metadata table (populated by seed script).

        Args:
            agent: Agent name to look up

        Returns:
            JSON with agent metadata including:
            - provider: LLM provider (anthropic, openai, google, etc.)
            - model_id: Full model identifier
            - context_window: Maximum context window size
            - specialties: Areas of expertise
            - strengths: Key capabilities
            - release_date: When the model was released
        """
        import json
        import sqlite3

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return json_response(
                {
                    "agent": agent,
                    "metadata": None,
                    "message": "Database not available",
                }
            )

        elo_path = get_db_path(DatabaseType.ELO, nomic_dir)
        if not elo_path.exists():
            return json_response(
                {
                    "agent": agent,
                    "metadata": None,
                    "message": "ELO database not found",
                }
            )

        try:
            conn = sqlite3.connect(elo_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT agent_name, provider, model_id, context_window,
                       specialties, strengths, release_date, updated_at
                FROM agent_metadata
                WHERE agent_name = ?
                """,
                (agent,),
            )
            row = cursor.fetchone()
            conn.close()

            if not row:
                return json_response(
                    {
                        "agent": agent,
                        "metadata": None,
                        "message": "Agent metadata not found",
                    }
                )

            # Parse JSON fields
            specialties = []
            strengths = []
            try:
                if row["specialties"]:
                    specialties = json.loads(row["specialties"])
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                if row["strengths"]:
                    strengths = json.loads(row["strengths"])
            except (json.JSONDecodeError, TypeError):
                pass

            return json_response(
                {
                    "agent": agent,
                    "metadata": {
                        "provider": row["provider"],
                        "model_id": row["model_id"],
                        "context_window": row["context_window"],
                        "specialties": specialties,
                        "strengths": strengths,
                        "release_date": row["release_date"],
                        "updated_at": row["updated_at"],
                    },
                }
            )
        except sqlite3.OperationalError as e:
            # Table may not exist yet
            if "no such table" in str(e):
                return json_response(
                    {
                        "agent": agent,
                        "metadata": None,
                        "message": "Agent metadata table not initialized. Run seed_agents.py to populate.",
                    }
                )
            raise

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/introspect",
        summary="Get agent introspection data",
        tags=["Agents"],
    )
    @handle_errors("agent introspect")
    def _get_agent_introspect(self, agent: str, debate_id: str | None = None) -> HandlerResult:
        """Get agent introspection data for self-awareness and debugging.

        This endpoint provides comprehensive internal state information that
        agents can query to understand their own cognitive state, useful for
        debugging, self-improvement, and mid-debate introspection.

        Args:
            agent: Agent name to introspect
            debate_id: Optional debate ID for debate-specific state

        Returns:
            JSON with agent's internal state including:
            - identity: Basic agent info and persona
            - calibration: Prediction accuracy metrics
            - positions: Recent stance history
            - performance: Win/loss and rating data
            - memory_summary: Memory tier statistics
            - fatigue_indicators: Signs of cognitive fatigue (if available)
        """
        elo = self.get_elo_system()

        introspection: dict[str, Any] = {
            "agent_id": agent,
            "timestamp": self._get_timestamp(),
            "identity": {"name": agent},
            "calibration": {},
            "positions": [],
            "performance": {},
            "memory_summary": {},
            "fatigue_indicators": None,  # Placeholder for fatigue detection
            "debate_context": None,
        }

        # Get basic rating/performance data
        if elo:
            try:
                rating = elo.get_rating(agent)
                total_games = rating.wins + rating.losses + rating.draws
                introspection["performance"] = {
                    "elo": rating.elo,
                    "total_games": total_games,
                    "wins": rating.wins,
                    "losses": rating.losses,
                    "win_rate": rating.wins / total_games if total_games > 0 else 0.0,
                }
                introspection["calibration"] = {
                    "accuracy": round(rating.calibration_accuracy, 3),
                    "brier_score": round(rating.calibration_brier_score, 3),
                    "prediction_count": rating.calibration_total,
                    "confidence_level": self._compute_confidence(rating),
                }
            except (KeyError, ValueError, AttributeError, TypeError) as e:
                logger.debug("Could not get ELO data for %s: %s", agent, e)

        # Get position history
        try:
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                from aragora.ranking.position_tracker import PositionTracker

                tracker_path = nomic_dir / "position_tracker.json"
                if tracker_path.exists():
                    tracker = PositionTracker(str(tracker_path))
                    positions = tracker.get_agent_positions(agent, limit=5)
                    introspection["positions"] = [
                        {
                            "topic": p.get("topic", ""),
                            "stance": p.get("stance", ""),
                            "confidence": p.get("confidence", 0.5),
                            "timestamp": p.get("timestamp", ""),
                        }
                        for p in positions
                    ]
        except (ImportError, OSError, KeyError, ValueError) as e:
            logger.debug("Could not get position data for %s: %s", agent, e)

        # Get memory tier summary
        try:
            from aragora.memory.continuum import ContinuumMemory

            memory = ContinuumMemory()
            stats = memory.get_stats()
            by_tier = stats.get("by_tier", {})
            tier_counts: dict[str, int] = {
                tier: data.get("count", 0) for tier, data in by_tier.items()
            }
            introspection["memory_summary"] = {
                "tier_counts": tier_counts,
                "total_memories": sum(tier_counts.values()),
                "red_line_count": len(memory.get_red_line_memories()),
            }
        except (ImportError, KeyError, ValueError, TypeError) as e:
            logger.debug("Could not get memory data: %s", e)

        # Get persona info if available
        try:
            from aragora.agents.personas import PersonaManager

            persona_mgr = PersonaManager()
            persona = persona_mgr.get_persona(agent)
            if persona:
                introspection["identity"]["persona"] = {
                    "style": getattr(persona, "style", None),
                    "temperature": getattr(persona, "temperature", None),
                    "system_prompt_preview": (
                        getattr(persona, "system_prompt", "")[:200]
                        if getattr(persona, "system_prompt", None)
                        else None
                    ),
                }
        except (ImportError, KeyError, ValueError, AttributeError) as e:
            logger.debug("Could not get persona data for %s: %s", agent, e)

        # Add debate-specific context if debate_id provided
        if debate_id:
            try:
                storage = self.get_storage()
                if storage:
                    debate = storage.get_debate(debate_id)  # type: ignore[attr-defined]
                    if debate:
                        # Find agent's messages in this debate
                        agent_msgs = [
                            m for m in debate.get("messages", []) if m.get("agent") == agent
                        ]
                        introspection["debate_context"] = {
                            "debate_id": debate_id,
                            "messages_sent": len(agent_msgs),
                            "current_round": debate.get("current_round", 0),
                            "debate_status": debate.get("status", "unknown"),
                        }
            except (KeyError, ValueError, AttributeError, TypeError) as e:
                logger.debug("Could not get debate context: %s", e)

        return json_response(introspection)

    def _compute_confidence(self, rating: Any) -> str:
        """Compute confidence level from calibration data."""
        accuracy = rating.calibration_accuracy
        count = rating.calibration_total
        if count < 5:
            return "insufficient_data"
        if accuracy >= 0.8:
            return "high"
        if accuracy >= 0.6:
            return "medium"
        return "low"

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/head-to-head/{opponent}",
        summary="Get head-to-head stats between two agents",
        tags=["Agents"],
    )
    @ttl_cache(ttl_seconds=CACHE_TTL_AGENT_H2H, key_prefix="agent_h2h", skip_first=True)
    @handle_errors("head-to-head stats")
    def _get_head_to_head(self, agent: str, opponent: str) -> HandlerResult:
        """Get head-to-head stats between two agents."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        if hasattr(elo, "get_head_to_head"):
            stats = elo.get_head_to_head(agent, opponent)
        else:
            stats = {"matches": 0, "agent1_wins": 0, "agent2_wins": 0}
        return json_response(
            {
                "agent1": agent,
                "agent2": opponent,
                **stats,
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/opponent-briefing/{opponent}",
        summary="Get strategic briefing about an opponent",
        tags=["Agents"],
    )
    @handle_errors("opponent briefing")
    def _get_opponent_briefing(self, agent: str, opponent: str) -> HandlerResult:
        """Get strategic briefing about an opponent for an agent."""
        elo = self.get_elo_system()
        nomic_dir = self.get_nomic_dir()

        from aragora.agents.grounded import PersonaSynthesizer

        # Get position ledger if available
        position_ledger = None
        if nomic_dir:
            try:
                from aragora.agents.grounded import PositionLedger

                db_path = get_db_path(DatabaseType.POSITIONS, nomic_dir)
                if db_path.exists():
                    position_ledger = PositionLedger(str(db_path))
            except ImportError:
                pass

        synthesizer = PersonaSynthesizer(
            elo_system=elo,
            position_ledger=position_ledger,
        )
        briefing = synthesizer.get_opponent_briefing(agent, opponent)

        if briefing:
            return json_response(
                {
                    "agent": agent,
                    "opponent": opponent,
                    "briefing": briefing,
                }
            )
        else:
            return json_response(
                {
                    "agent": agent,
                    "opponent": opponent,
                    "briefing": None,
                    "message": "No opponent data available",
                }
            )
