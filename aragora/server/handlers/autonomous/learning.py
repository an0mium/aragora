"""Continuous learning HTTP handlers."""

import logging
from typing import Optional

from aiohttp import web

from aragora.autonomous import ContinuousLearner

logger = logging.getLogger(__name__)

# Global continuous learner instance
_continuous_learner: Optional[ContinuousLearner] = None


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

    @staticmethod
    async def get_agent_ratings(request: web.Request) -> web.Response:
        """
        Get ELO ratings for all agents.

        GET /api/autonomous/learning/ratings

        Returns:
            Dict of agent_id -> rating
        """
        try:
            learner = get_continuous_learner()
            ratings = learner.elo_updater.get_all_ratings()

            return web.json_response({
                "success": True,
                "ratings": ratings,
                "count": len(ratings),
            })

        except Exception as e:
            logger.error(f"Error getting ratings: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def get_agent_calibration(request: web.Request) -> web.Response:
        """
        Get calibration data for an agent.

        GET /api/autonomous/learning/calibration/{agent_id}

        Returns:
            Agent calibration data
        """
        agent_id = request.match_info.get("agent_id")

        try:
            learner = get_continuous_learner()
            calibration = learner.get_calibration(agent_id)

            if not calibration:
                return web.json_response({
                    "success": True,
                    "calibration": None,
                    "message": "No calibration data for this agent",
                })

            return web.json_response({
                "success": True,
                "calibration": {
                    "agent_id": calibration.agent_id,
                    "elo_rating": calibration.elo_rating,
                    "confidence_accuracy": calibration.confidence_accuracy,
                    "topic_strengths": calibration.topic_strengths,
                    "topic_weaknesses": calibration.topic_weaknesses,
                    "last_updated": calibration.last_updated.isoformat() if calibration.last_updated else None,
                    "total_debates": calibration.total_debates,
                    "win_rate": calibration.win_rate,
                },
            })

        except Exception as e:
            logger.error(f"Error getting calibration: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def get_all_calibrations(request: web.Request) -> web.Response:
        """
        Get calibration data for all agents.

        GET /api/autonomous/learning/calibrations

        Returns:
            Dict of agent_id -> calibration data
        """
        try:
            learner = get_continuous_learner()
            calibrations = learner.get_all_calibrations()

            return web.json_response({
                "success": True,
                "calibrations": {
                    agent_id: {
                        "elo_rating": cal.elo_rating,
                        "total_debates": cal.total_debates,
                        "win_rate": cal.win_rate,
                        "last_updated": cal.last_updated.isoformat() if cal.last_updated else None,
                    }
                    for agent_id, cal in calibrations.items()
                },
                "count": len(calibrations),
            })

        except Exception as e:
            logger.error(f"Error getting calibrations: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def record_debate_outcome(request: web.Request) -> web.Response:
        """
        Record a debate outcome for learning.

        POST /api/autonomous/learning/debate

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
            data = await request.json()
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

            return web.json_response({
                "success": True,
                "event": {
                    "id": event.id,
                    "event_type": event.event_type.value,
                    "applied": event.applied,
                },
                "updated_ratings": {
                    agent: learner.elo_updater.get_rating(agent)
                    for agent in agents
                },
            })

        except Exception as e:
            logger.error(f"Error recording debate outcome: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def record_user_feedback(request: web.Request) -> web.Response:
        """
        Record user feedback for learning.

        POST /api/autonomous/learning/feedback

        Body:
            debate_id: str - Related debate ID
            agent_id: str - Agent receiving feedback
            feedback_type: str - Type (helpful, unhelpful, accurate, inaccurate)
            score: float - Feedback score (-1 to 1)

        Returns:
            Learning event created
        """
        try:
            data = await request.json()
            debate_id = data.get("debate_id")
            agent_id = data.get("agent_id")
            feedback_type = data.get("feedback_type")
            score = data.get("score", 0.0)

            if not debate_id or not agent_id or not feedback_type:
                return web.json_response(
                    {"success": False, "error": "debate_id, agent_id, and feedback_type are required"},
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

            return web.json_response({
                "success": True,
                "event": {
                    "id": event.id,
                    "event_type": event.event_type.value,
                    "applied": event.applied,
                },
            })

        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def get_patterns(request: web.Request) -> web.Response:
        """
        Get extracted patterns.

        GET /api/autonomous/learning/patterns

        Query params:
            pattern_type: str (optional) - Filter by pattern type

        Returns:
            List of extracted patterns
        """
        try:
            pattern_type = request.query.get("pattern_type")

            learner = get_continuous_learner()
            patterns = learner.pattern_extractor.get_patterns(pattern_type)

            return web.json_response({
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
            })

        except Exception as e:
            logger.error(f"Error getting patterns: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def run_periodic_learning(request: web.Request) -> web.Response:
        """
        Manually trigger periodic learning tasks.

        POST /api/autonomous/learning/run

        Returns:
            Summary of actions taken
        """
        try:
            learner = get_continuous_learner()
            summary = await learner.run_periodic_learning()

            return web.json_response({
                "success": True,
                "summary": summary,
            })

        except Exception as e:
            logger.error(f"Error running periodic learning: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    def register_routes(app: web.Application, prefix: str = "/api/autonomous") -> None:
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
