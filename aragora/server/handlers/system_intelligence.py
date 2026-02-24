"""System Intelligence Dashboard Handler.

Aggregates data from multiple subsystems to provide a unified view
of the system's learning, agent performance, institutional memory,
and improvement queue state.

Endpoints:
- GET /api/v1/system-intelligence/overview          - High-level system stats
- GET /api/v1/system-intelligence/agent-performance  - ELO, calibration, win rates
- GET /api/v1/system-intelligence/institutional-memory - Cross-debate injection stats
- GET /api/v1/system-intelligence/improvement-queue   - Queue contents + breakdown
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from .base import (
    HandlerResult,
    json_response,
)
from .secure import SecureHandler
from .utils.auth_mixins import SecureEndpointMixin
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


async def _maybe_await(value: Any) -> Any:
    """Await value when it is a coroutine, otherwise return as-is."""
    if asyncio.iscoroutine(value):
        return await value
    return value


class SystemIntelligenceHandler(SecureEndpointMixin, SecureHandler):  # type: ignore[misc]
    """Handler for the system intelligence dashboard.

    Aggregates ELO rankings, calibration data, Nomic cycle stats,
    selection feedback, and Knowledge Mound counts into a single
    dashboard view.

    RBAC Permissions:
    - system_intelligence:read - View all intelligence endpoints
    """

    RESOURCE_TYPE = "system_intelligence"

    ROUTES = [
        "/api/system-intelligence/overview",
        "/api/system-intelligence/agent-performance",
        "/api/system-intelligence/institutional-memory",
        "/api/system-intelligence/improvement-queue",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        path = strip_version_prefix(path)
        return path in self.ROUTES

    @rate_limit(requests_per_minute=30)
    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route GET requests to the appropriate endpoint."""
        path = strip_version_prefix(path)

        handlers = {
            "/api/system-intelligence/overview": self._get_overview,
            "/api/system-intelligence/agent-performance": self._get_agent_performance,
            "/api/system-intelligence/institutional-memory": self._get_institutional_memory,
            "/api/system-intelligence/improvement-queue": self._get_improvement_queue,
        }

        endpoint_handler = handlers.get(path)
        if endpoint_handler:
            return await endpoint_handler()

        return None

    # ------------------------------------------------------------------
    # GET /api/v1/system-intelligence/overview
    # ------------------------------------------------------------------

    async def _get_overview(self) -> HandlerResult:
        """High-level system intelligence overview.

        Returns:
            {"data": {
                "totalCycles": int,
                "successRate": float,
                "activeAgents": int,
                "knowledgeItems": int,
                "topAgents": [...],
                "recentImprovements": [...]
            }}
        """
        total_cycles = 0
        success_rate = 0.0
        active_agents = 0
        knowledge_items = 0
        top_agents: list[dict[str, Any]] = []
        recent_improvements: list[dict[str, Any]] = []

        # Nomic cycle stats
        try:
            from aragora.nomic.cycle_store import get_cycle_store

            store = get_cycle_store()
            recent_cycles: list[Any] = []
            recent_cycles_getter = getattr(store, "get_recent_cycles", None)
            if callable(recent_cycles_getter):
                recent_candidate = recent_cycles_getter(50)
                if isinstance(recent_candidate, list):
                    recent_cycles = recent_candidate

            if not recent_cycles:
                recent_getter = getattr(store, "get_recent", None)
                if callable(recent_getter):
                    recent_candidate = recent_getter(limit=50)
                    if isinstance(recent_candidate, list):
                        recent_cycles = recent_candidate

            total_cycles = len(recent_cycles)
            successes = 0
            for cycle in recent_cycles:
                if isinstance(cycle, dict):
                    is_success = bool(cycle.get("success", False))
                else:
                    is_success = bool(getattr(cycle, "success", False))
                if is_success:
                    successes += 1
            if total_cycles > 0:
                success_rate = round(successes / total_cycles, 4)
        except (ImportError, RuntimeError, ValueError, OSError, AttributeError):
            logger.debug("CycleStore not available for overview")

        # ELO leaderboard for top agents
        try:
            from aragora.ranking.elo import EloSystem

            elo = EloSystem()
            leaderboard = elo.get_leaderboard(limit=10)
            for rating in leaderboard:
                if isinstance(rating, dict):
                    agent_id = rating.get("agent_name", "")
                    elo_score = rating.get("elo", rating.get("rating", 1500))
                    wins = rating.get("wins", 0)
                else:
                    agent_id = getattr(rating, "agent_name", "")
                    elo_score = getattr(rating, "elo", getattr(rating, "rating", 1500))
                    wins = getattr(rating, "wins", 0)
                top_agents.append({
                    "id": agent_id,
                    "elo": elo_score,
                    "wins": wins,
                })
            active_agents = len(leaderboard)
        except (ImportError, RuntimeError, ValueError, OSError, AttributeError):
            logger.debug("EloSystem not available for overview")

        # Knowledge Mound item count
        try:
            from aragora.knowledge.mound.core import KnowledgeMoundCore

            km = KnowledgeMoundCore()
            get_stats = getattr(km, "get_stats", None)
            stats: Any = await _maybe_await(get_stats()) if callable(get_stats) else {}
            if isinstance(stats, dict):
                knowledge_items = int(stats.get("total_items", stats.get("total_nodes", 0)))
            else:
                knowledge_items = int(getattr(stats, "total_nodes", 0))
        except (ImportError, RuntimeError, ValueError, OSError, AttributeError):
            logger.debug("KnowledgeMound not available for overview")

        # Recent improvement queue items
        try:
            from aragora.nomic.improvement_queue import get_improvement_queue

            queue = get_improvement_queue()
            for s in queue.peek(5):
                recent_improvements.append(
                    {
                        "id": s.debate_id,
                        "goal": s.task[:200],
                        "status": s.category,
                    }
                )
        except (ImportError, RuntimeError):
            pass

        return json_response(
            {
                "data": {
                    "totalCycles": total_cycles,
                    "successRate": success_rate,
                    "activeAgents": active_agents,
                    "knowledgeItems": knowledge_items,
                    "topAgents": top_agents,
                    "recentImprovements": recent_improvements,
                }
            }
        )

    # ------------------------------------------------------------------
    # GET /api/v1/system-intelligence/agent-performance
    # ------------------------------------------------------------------

    async def _get_agent_performance(self) -> HandlerResult:
        """Agent performance details: ELO history, calibration, win rates.

        Returns:
            {"data": {"agents": [...]}}
        """
        agents: list[dict[str, Any]] = []

        # ELO data
        try:
            from aragora.ranking.elo import EloSystem

            elo = EloSystem()
            leaderboard = elo.get_leaderboard(limit=50)

            for entry in leaderboard:
                if isinstance(entry, dict):
                    name = entry.get("agent_name", "")
                    rating = entry.get("elo", entry.get("rating", 1500))
                    wins = entry.get("wins", 0)
                    losses = entry.get("losses", 0)
                else:
                    name = getattr(entry, "agent_name", "")
                    rating = getattr(entry, "elo", getattr(entry, "rating", 1500))
                    wins = getattr(entry, "wins", 0)
                    losses = getattr(entry, "losses", 0)

                total = wins + losses
                win_rate = round(wins / total, 4) if total > 0 else 0.0

                # ELO history
                elo_history: list[dict[str, Any]] = []
                try:
                    get_agent_history = getattr(elo, "get_agent_history", None)
                    if callable(get_agent_history):
                        history = get_agent_history(name, limit=20)
                        for history_entry in history:
                            if isinstance(history_entry, dict):
                                elo_history.append({
                                    "date": history_entry.get("timestamp", ""),
                                    "elo": history_entry.get("rating", rating),
                                })
                            else:
                                elo_history.append({
                                    "date": getattr(history_entry, "timestamp", ""),
                                    "elo": getattr(history_entry, "rating", rating),
                                })
                    else:
                        history = elo.get_elo_history(name, limit=20)
                        for ts, elo_value in history:
                            elo_history.append({
                                "date": ts,
                                "elo": elo_value,
                            })
                except (AttributeError, TypeError, ValueError):
                    pass

                # Calibration score
                calibration = 0.0
                try:
                    get_calibration_score = getattr(elo, "get_calibration_score", None)
                    if callable(get_calibration_score):
                        cal_data = get_calibration_score(name)
                        if isinstance(cal_data, (int, float)):
                            calibration = float(cal_data)
                        elif isinstance(cal_data, dict):
                            calibration = float(cal_data.get("score", 0.0))
                    else:
                        calibration = float(elo.get_rating(name).calibration_score)
                except (AttributeError, TypeError, ValueError):
                    pass

                # Domain performance from SelectionFeedbackLoop
                domains: list[str] = []
                try:
                    from aragora.debate.selection_feedback import SelectionFeedbackLoop

                    feedback = SelectionFeedbackLoop()
                    state = feedback.get_agent_state(name)
                    if state:
                        domains = list(state.domain_wins.keys())
                except (ImportError, AttributeError, TypeError):
                    pass

                agents.append(
                    {
                        "id": name,
                        "name": name,
                        "elo": rating,
                        "eloHistory": elo_history,
                        "calibration": calibration,
                        "winRate": win_rate,
                        "domains": domains,
                    }
                )

        except (ImportError, RuntimeError, ValueError, OSError, AttributeError):
            logger.debug("EloSystem not available for agent performance")

        return json_response({"data": {"agents": agents}})

    # ------------------------------------------------------------------
    # GET /api/v1/system-intelligence/institutional-memory
    # ------------------------------------------------------------------

    async def _get_institutional_memory(self) -> HandlerResult:
        """Cross-debate injection stats and knowledge patterns.

        Returns:
            {"data": {
                "totalInjections": int,
                "retrievalCount": int,
                "topPatterns": [...],
                "confidenceChanges": [...]
            }}
        """
        total_injections = 0
        retrieval_count = 0
        top_patterns: list[dict[str, Any]] = []
        confidence_changes: list[dict[str, Any]] = []

        # NomicCycleAdapter for pattern data
        try:
            from aragora.knowledge.mound.adapters.nomic_cycle_adapter import (
                get_nomic_cycle_adapter,
            )

            adapter = get_nomic_cycle_adapter()

            # High-ROI patterns double as "top patterns"
            try:
                roi_data = await adapter.find_high_roi_goal_types(limit=10)
                for entry in roi_data:
                    top_patterns.append(
                        {
                            "pattern": entry.get("pattern", ""),
                            "frequency": entry.get("cycle_count", 0),
                            "confidence": entry.get("avg_improvement_score", 0.0),
                        }
                    )
            except (RuntimeError, ValueError, OSError, AttributeError):
                pass

        except (ImportError, RuntimeError, ValueError, OSError):
            logger.debug("NomicCycleAdapter not available for institutional memory")

        # Cross-debate memory stats
        try:
            from aragora.memory.cross_debate_rlm import CrossDebateMemory

            cdm = CrossDebateMemory()
            stats_getter = getattr(cdm, "get_statistics", None)
            stats: Any = {}
            if callable(stats_getter):
                stats = stats_getter()
            if not isinstance(stats, dict):
                legacy_stats_getter = getattr(cdm, "get_stats", None)
                stats = legacy_stats_getter() if callable(legacy_stats_getter) else {}
            if isinstance(stats, dict):
                total_injections = int(stats.get("total_injections", stats.get("total_entries", 0)))
                retrieval_count = int(stats.get("retrieval_count", stats.get("total_tokens", 0)))
        except (ImportError, RuntimeError, AttributeError):
            logger.debug("CrossDebateMemory not available")

        # Confidence changes from Knowledge Mound
        try:
            from aragora.knowledge.mound.core import KnowledgeMoundCore

            km = KnowledgeMoundCore()
            decay_getter = getattr(km, "get_confidence_decay_stats", None)
            if callable(decay_getter):
                decay_stats = await _maybe_await(decay_getter())
                if isinstance(decay_stats, list):
                    for entry in decay_stats[:10]:
                        if isinstance(entry, dict):
                            confidence_changes.append({
                                "topic": entry.get("topic", ""),
                                "before": entry.get("initial_confidence", 0.0),
                                "after": entry.get("current_confidence", 0.0),
                            })
            else:
                get_stats = getattr(km, "get_stats", None)
                mound_stats: Any = await _maybe_await(get_stats()) if callable(get_stats) else None
                if mound_stats is not None and hasattr(mound_stats, "average_confidence"):
                    confidence_changes.append({
                        "topic": "global",
                        "before": float(getattr(mound_stats, "average_confidence", 0.0)),
                        "after": float(getattr(mound_stats, "average_confidence", 0.0)),
                    })
        except (ImportError, RuntimeError, AttributeError):
            logger.debug("KM confidence decay stats not available")

        return json_response(
            {
                "data": {
                    "totalInjections": total_injections,
                    "retrievalCount": retrieval_count,
                    "topPatterns": top_patterns,
                    "confidenceChanges": confidence_changes,
                }
            }
        )

    # ------------------------------------------------------------------
    # GET /api/v1/system-intelligence/improvement-queue
    # ------------------------------------------------------------------

    async def _get_improvement_queue(self) -> HandlerResult:
        """Improvement queue contents with source breakdown.

        Returns:
            {"data": {
                "items": [...],
                "totalSize": int,
                "sourceBreakdown": {"debate": N, "user": N, ...}
            }}
        """
        items: list[dict[str, Any]] = []
        total_size = 0
        source_breakdown: dict[str, int] = {}

        try:
            from aragora.nomic.improvement_queue import get_improvement_queue

            queue = get_improvement_queue()
            total_size = len(queue)

            for s in queue.peek(50):
                items.append(
                    {
                        "id": s.debate_id,
                        "goal": s.task[:200],
                        "priority": int(s.confidence * 100),
                        "source": s.category,
                        "status": "pending",
                        "createdAt": str(s.created_at),
                    }
                )
                source_breakdown[s.category] = source_breakdown.get(s.category, 0) + 1

        except (ImportError, RuntimeError):
            logger.debug("ImprovementQueue not available")

        return json_response(
            {
                "data": {
                    "items": items,
                    "totalSize": total_size,
                    "sourceBreakdown": source_breakdown,
                }
            }
        )
