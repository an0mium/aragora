"""Continuous learning HTTP handlers."""

from __future__ import annotations

import logging

from aiohttp import web

from aragora.autonomous import ContinuousLearner
from aragora.server.handlers.utils.auth import (
    get_auth_context,
    UnauthorizedError,
    ForbiddenError,
)
from aragora.server.handlers.utils import parse_json_body
from aragora.rbac.checker import get_permission_checker
from aragora.rbac.decorators import require_permission

logger = logging.getLogger(__name__)

# RBAC permission keys for autonomous operations
AUTONOMOUS_READ_PERMISSION = "autonomous:read"
AUTONOMOUS_WRITE_PERMISSION = "autonomous:write"

# Global continuous learner instance
_continuous_learner: ContinuousLearner | None = None


def get_continuous_learner() -> ContinuousLearner:
    """Get or create the global continuous learner instance."""
    global _continuous_learner
    if _continuous_learner is None:
        _continuous_learner = ContinuousLearner()
    return _continuous_learner


def set_continuous_learner(learner: ContinuousLearner) -> None:
    """Set the global continuous learner instance."""
    global _continuous_learner
    _continuous_learner = learner


class LearningHandler:
    """HTTP handlers for continuous learning operations."""

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    @staticmethod
    @require_permission("autonomous:learning:read")
    async def get_agent_ratings(request: web.Request) -> web.Response:
        """
        Get ELO ratings for all agents.

        GET /api/autonomous/learning/ratings

        Requires authentication and 'autonomous:read' permission.

        Returns:
            Dict of agent_id -> rating
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_READ_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            learner = get_continuous_learner()
            ratings = learner.elo_updater.get_all_ratings()

            return web.json_response(
                {
                    "success": True,
                    "ratings": ratings,
                    "count": len(ratings),
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized getting ratings: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden getting ratings: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error getting ratings: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to retrieve ratings"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:learning:read")
    async def get_agent_calibration(request: web.Request) -> web.Response:
        """
        Get calibration data for an agent.

        GET /api/autonomous/learning/calibration/{agent_id}

        Requires authentication and 'autonomous:read' permission.

        Returns:
            Agent calibration data
        """
        agent_id = request.match_info.get("agent_id")

        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_READ_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            learner = get_continuous_learner()
            calibration = learner.get_calibration(agent_id)

            if not calibration:
                return web.json_response(
                    {
                        "success": True,
                        "calibration": None,
                        "message": "No calibration data for this agent",
                    }
                )

            return web.json_response(
                {
                    "success": True,
                    "calibration": {
                        "agent_id": calibration.agent_id,
                        "elo_rating": calibration.elo_rating,
                        "confidence_accuracy": calibration.confidence_accuracy,
                        "topic_strengths": calibration.topic_strengths,
                        "topic_weaknesses": calibration.topic_weaknesses,
                        "last_updated": (
                            calibration.last_updated.isoformat()
                            if calibration.last_updated
                            else None
                        ),
                        "total_debates": calibration.total_debates,
                        "win_rate": calibration.win_rate,
                    },
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized getting calibration: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden getting calibration: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error getting calibration: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to retrieve calibration data"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:learning:read")
    async def get_all_calibrations(request: web.Request) -> web.Response:
        """
        Get calibration data for all agents.

        GET /api/autonomous/learning/calibrations

        Requires authentication and 'autonomous:read' permission.

        Returns:
            Dict of agent_id -> calibration data
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_READ_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            learner = get_continuous_learner()
            calibrations = learner.get_all_calibrations()

            return web.json_response(
                {
                    "success": True,
                    "calibrations": {
                        agent_id: {
                            "elo_rating": cal.elo_rating,
                            "total_debates": cal.total_debates,
                            "win_rate": cal.win_rate,
                            "last_updated": (
                                cal.last_updated.isoformat() if cal.last_updated else None
                            ),
                        }
                        for agent_id, cal in calibrations.items()
                    },
                    "count": len(calibrations),
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized getting calibrations: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden getting calibrations: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error getting calibrations: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to retrieve calibrations"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:learning:write")
    async def record_debate_outcome(request: web.Request) -> web.Response:
        """
        Record a debate outcome for learning.

        POST /api/autonomous/learning/debate

        Requires authentication and 'autonomous:write' permission.

        Body:
            debate_id: str - ID of the debate
            agents: list[str] - Agents that participated
            winner: str (optional) - Winning agent
            votes: dict[str, int] - Votes per agent
            consensus_reached: bool - Whether consensus was reached
            topics: list[str] - Topics discussed

        Returns:
            Learning event created
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_WRITE_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            data, err = await parse_json_body(request, context="record_debate_outcome")
            if err:
                return err
            debate_id = data.get("debate_id")
            agents = data.get("agents", [])
            winner = data.get("winner")
            votes = data.get("votes", {})
            consensus_reached = data.get("consensus_reached", False)
            topics = data.get("topics", [])

            if not debate_id or not agents:
                return web.json_response(
                    {"success": False, "error": "debate_id and agents are required"},
                    status=400,
                )

            learner = get_continuous_learner()
            event = await learner.on_debate_completed(
                debate_id=debate_id,
                agents=agents,
                winner=winner,
                votes=votes,
                consensus_reached=consensus_reached,
                topics=topics,
                metadata=data.get("metadata"),
            )

            return web.json_response(
                {
                    "success": True,
                    "event": {
                        "id": event.id,
                        "event_type": event.event_type.value,
                        "applied": event.applied,
                    },
                    "updated_ratings": {
                        agent: learner.elo_updater.get_rating(agent) for agent in agents
                    },
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized recording debate outcome: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden recording debate outcome: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error recording debate outcome: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to record debate outcome"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:learning:write")
    async def record_user_feedback(request: web.Request) -> web.Response:
        """
        Record user feedback for learning.

        POST /api/autonomous/learning/feedback

        Requires authentication and 'autonomous:write' permission.

        Body:
            debate_id: str - Related debate ID
            agent_id: str - Agent receiving feedback
            feedback_type: str - Type (helpful, unhelpful, accurate, inaccurate)
            score: float - Feedback score (-1 to 1)

        Returns:
            Learning event created
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_WRITE_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            data, err = await parse_json_body(request, context="record_user_feedback")
            if err:
                return err
            debate_id = data.get("debate_id")
            agent_id = data.get("agent_id")
            feedback_type = data.get("feedback_type")
            score = data.get("score", 0.0)

            if not debate_id or not agent_id or not feedback_type:
                return web.json_response(
                    {
                        "success": False,
                        "error": "debate_id, agent_id, and feedback_type are required",
                    },
                    status=400,
                )

            learner = get_continuous_learner()
            event = await learner.on_user_feedback(
                debate_id=debate_id,
                agent_id=agent_id,
                feedback_type=feedback_type,
                score=float(score),
                metadata=data.get("metadata"),
            )

            return web.json_response(
                {
                    "success": True,
                    "event": {
                        "id": event.id,
                        "event_type": event.event_type.value,
                        "applied": event.applied,
                    },
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized recording feedback: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden recording feedback: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error recording feedback: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to record feedback"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:learning:read")
    async def get_patterns(request: web.Request) -> web.Response:
        """
        Get extracted patterns.

        GET /api/autonomous/learning/patterns

        Requires authentication and 'autonomous:read' permission.

        Query params:
            pattern_type: str (optional) - Filter by pattern type

        Returns:
            List of extracted patterns
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_READ_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            pattern_type = request.query.get("pattern_type")

            learner = get_continuous_learner()
            patterns = learner.pattern_extractor.get_patterns(pattern_type)

            return web.json_response(
                {
                    "success": True,
                    "patterns": [
                        {
                            "id": p.id,
                            "pattern_type": p.pattern_type,
                            "description": p.description,
                            "confidence": p.confidence,
                            "evidence_count": p.evidence_count,
                            "first_seen": p.first_seen.isoformat(),
                            "last_seen": p.last_seen.isoformat(),
                            "agents_involved": p.agents_involved,
                            "topics": p.topics,
                        }
                        for p in patterns
                    ],
                    "count": len(patterns),
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized getting patterns: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden getting patterns: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error getting patterns: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to retrieve patterns"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:learning:write")
    async def run_periodic_learning(request: web.Request) -> web.Response:
        """
        Manually trigger periodic learning tasks.

        POST /api/autonomous/learning/run

        Requires authentication and 'autonomous:write' permission.

        Returns:
            Summary of actions taken
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_WRITE_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            learner = get_continuous_learner()
            summary = await learner.run_periodic_learning()

            return web.json_response(
                {
                    "success": True,
                    "summary": summary,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized running periodic learning: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden running periodic learning: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error running periodic learning: %s", e)
            return web.json_response(
                {"success": False, "error": "Periodic learning run failed"},
                status=500,
            )

    @staticmethod
    def register_routes(app: web.Application, prefix: str = "/api/v1/autonomous") -> None:
        """Register learning routes with the application."""
        app.router.add_get(
            f"{prefix}/learning/ratings",
            LearningHandler.get_agent_ratings,
        )
        app.router.add_get(
            f"{prefix}/learning/calibration/{{agent_id}}",
            LearningHandler.get_agent_calibration,
        )
        app.router.add_get(
            f"{prefix}/learning/calibrations",
            LearningHandler.get_all_calibrations,
        )
        app.router.add_post(
            f"{prefix}/learning/debate",
            LearningHandler.record_debate_outcome,
        )
        app.router.add_post(
            f"{prefix}/learning/feedback",
            LearningHandler.record_user_feedback,
        )
        app.router.add_get(
            f"{prefix}/learning/patterns",
            LearningHandler.get_patterns,
        )
        app.router.add_post(
            f"{prefix}/learning/run",
            LearningHandler.run_periodic_learning,
        )
