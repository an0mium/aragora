"""
Uncertainty estimation endpoint handlers.

Exposes the uncertainty quantification system for confidence calibration
and disagreement analysis.

Endpoints:
- POST /api/uncertainty/estimate - Estimate uncertainty for a debate/response
- GET /api/uncertainty/debate/:id - Get debate uncertainty metrics
- GET /api/uncertainty/agent/:id - Get agent calibration profile
- POST /api/uncertainty/followups - Generate follow-up suggestions from cruxes
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from aragora.server.validation import validate_path_segment, SAFE_ID_PATTERN

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


class UncertaintyHandler(BaseHandler):
    """Handler for uncertainty estimation endpoints."""

    ROUTES = [
        "/api/uncertainty/estimate",
        "/api/uncertainty/followups",
        "/api/uncertainty/debate/*",
        "/api/uncertainty/agent/*",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the request."""
        if path.startswith("/api/uncertainty/"):
            return True
        return False

    @rate_limit(rpm=60)
    async def handle(  # type: ignore[override]
        self, path: str, method: str, handler: Any = None
    ) -> Optional[HandlerResult]:
        """Route request to appropriate handler method."""
        if handler:
            query_str = handler.path.split("?", 1)[1] if "?" in handler.path else ""
            from urllib.parse import parse_qs
            parse_qs(query_str)

        # POST /api/uncertainty/estimate
        if path == "/api/uncertainty/estimate" and method == "POST":
            return await self._estimate_uncertainty(handler)

        # POST /api/uncertainty/followups
        if path == "/api/uncertainty/followups" and method == "POST":
            return await self._generate_followups(handler)

        # GET /api/uncertainty/debate/:id
        if path.startswith("/api/uncertainty/debate/") and method == "GET":
            parts = path.split("/")
            if len(parts) == 5:
                debate_id = parts[4]
                is_valid, err = validate_path_segment(debate_id, "debate_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
                return await self._get_debate_uncertainty(debate_id)

        # GET /api/uncertainty/agent/:id
        if path.startswith("/api/uncertainty/agent/") and method == "GET":
            parts = path.split("/")
            if len(parts) == 5:
                agent_id = parts[4]
                is_valid, err = validate_path_segment(agent_id, "agent_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
                return self._get_agent_calibration(agent_id)

        return None

    def _get_estimator(self) -> Optional[Any]:
        """Get the ConfidenceEstimator instance."""
        try:
            from aragora.uncertainty.estimator import ConfidenceEstimator
            # Use a shared instance from context if available
            if hasattr(self, "_ctx") and self._ctx and "confidence_estimator" in self._ctx:
                return self._ctx["confidence_estimator"]
            # Otherwise create a new instance
            return ConfidenceEstimator()
        except ImportError:
            logger.warning("Uncertainty module not available")
            return None

    def _get_analyzer(self) -> Optional[Any]:
        """Get the DisagreementAnalyzer instance."""
        try:
            from aragora.uncertainty.estimator import DisagreementAnalyzer
            return DisagreementAnalyzer()
        except ImportError:
            logger.warning("Uncertainty module not available")
            return None

    async def _estimate_uncertainty(self, handler: Any) -> HandlerResult:
        """Estimate uncertainty for provided debate data.

        Request body:
        {
            "messages": [...],  # List of debate messages
            "votes": [...],     # List of votes
            "proposals": {}     # Agent proposals
        }
        """
        estimator = self._get_estimator()
        if estimator is None:
            return error_response("Uncertainty module not available", 503)

        data = self.read_json_body(handler)
        if data is None:
            return error_response("Invalid or too large request body", 400)

        try:
            from aragora.core import Message, Vote

            # Parse messages
            messages = []
            for msg_data in data.get("messages", []):
                if isinstance(msg_data, dict):
                    messages.append(Message(
                        content=msg_data.get("content", ""),
                        agent=msg_data.get("agent", "unknown"),
                        role=msg_data.get("role", "agent"),
                        round=msg_data.get("round", 0),
                    ))

            # Parse votes
            votes = []
            for vote_data in data.get("votes", []):
                if isinstance(vote_data, dict):
                    votes.append(Vote(
                        agent=vote_data.get("agent", "unknown"),
                        choice=vote_data.get("choice", ""),
                        reasoning=vote_data.get("reasoning", ""),
                        confidence=vote_data.get("confidence", 0.5),
                    ))

            proposals = data.get("proposals", {})

            # Analyze uncertainty
            metrics = estimator.analyze_disagreement(messages, votes, proposals)

            return json_response({
                "metrics": metrics.to_dict(),
                "message": "Uncertainty estimated successfully",
            })

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid data for uncertainty estimation: {e}")
            return error_response(f"Invalid request data: {e}", 400)
        except Exception as e:
            logger.exception(f"Unexpected error estimating uncertainty: {e}")
            return error_response(f"Failed to estimate uncertainty: {e}", 500)

    async def _generate_followups(self, handler: Any) -> HandlerResult:
        """Generate follow-up debate suggestions from cruxes.

        Request body:
        {
            "cruxes": [...],           # List of disagreement cruxes
            "parent_debate_id": "...", # Optional parent debate ID
            "available_agents": [...]  # Optional list of available agents
        }
        """
        analyzer = self._get_analyzer()
        if analyzer is None:
            return error_response("Uncertainty module not available", 503)

        data = self.read_json_body(handler)
        if data is None:
            return error_response("Invalid or too large request body", 400)

        try:
            from aragora.uncertainty.estimator import DisagreementCrux

            # Parse cruxes
            cruxes = []
            for crux_data in data.get("cruxes", []):
                if isinstance(crux_data, dict):
                    cruxes.append(DisagreementCrux(
                        description=crux_data.get("description", ""),
                        divergent_agents=crux_data.get("divergent_agents", []),
                        evidence_needed=crux_data.get("evidence_needed", ""),
                        severity=crux_data.get("severity", 0.5),
                        crux_id=crux_data.get("id", ""),
                    ))

            if not cruxes:
                return error_response("No cruxes provided", 400)

            parent_debate_id = data.get("parent_debate_id")
            available_agents = data.get("available_agents")

            # Generate follow-up suggestions
            suggestions = analyzer.suggest_followups(
                cruxes=cruxes,
                parent_debate_id=parent_debate_id,
                available_agents=available_agents,
            )

            return json_response({
                "followups": [s.to_dict() for s in suggestions],
                "total": len(suggestions),
            })

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid data for follow-up generation: {e}")
            return error_response(f"Invalid request data: {e}", 400)
        except Exception as e:
            logger.exception(f"Unexpected error generating follow-ups: {e}")
            return error_response(f"Failed to generate follow-ups: {e}", 500)

    async def _get_debate_uncertainty(self, debate_id: str) -> HandlerResult:
        """Get uncertainty metrics for a specific debate."""
        try:
            # Try to get debate from storage
            storage = self._ctx.get("storage") if hasattr(self, "_ctx") and self._ctx else None
            if storage is None:
                return error_response("Storage not available", 503)

            # Look up debate
            debate = None
            if hasattr(storage, "get_debate"):
                debate = await storage.get_debate(debate_id)
            elif hasattr(storage, "get"):
                debate = storage.get(debate_id)

            if debate is None:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Get messages and votes from debate
            messages = getattr(debate, "messages", [])
            votes = getattr(debate, "votes", [])
            proposals = getattr(debate, "proposals", {})

            # If no uncertainty metrics stored, compute them
            estimator = self._get_estimator()
            if estimator is None:
                return error_response("Uncertainty module not available", 503)

            metrics = estimator.analyze_disagreement(messages, votes, proposals)

            return json_response({
                "debate_id": debate_id,
                "metrics": metrics.to_dict(),
            })

        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error getting debate uncertainty: {e}")
            return error_response(f"Invalid debate data: {e}", 400)
        except Exception as e:
            logger.exception(f"Unexpected error getting debate uncertainty: {e}")
            return error_response(f"Failed to get debate uncertainty: {e}", 500)

    def _get_agent_calibration(self, agent_id: str) -> HandlerResult:
        """Get calibration profile for a specific agent."""
        estimator = self._get_estimator()
        if estimator is None:
            return error_response("Uncertainty module not available", 503)

        try:
            # Get calibration quality
            calibration_quality = estimator.get_agent_calibration_quality(agent_id)

            # Get confidence history if available
            confidence_history: List[Dict[str, Any]] = []
            if agent_id in estimator.agent_confidences:
                for score in estimator.agent_confidences[agent_id][-10:]:  # Last 10
                    confidence_history.append(score.to_dict())

            # Get calibration history if available
            calibration_history: List[Dict[str, Any]] = []
            if agent_id in estimator.calibration_history:
                for confidence, was_correct in estimator.calibration_history[agent_id][-10:]:
                    calibration_history.append({
                        "confidence": confidence,
                        "was_correct": was_correct,
                    })

            return json_response({
                "agent_id": agent_id,
                "calibration_quality": calibration_quality,
                "confidence_history": confidence_history,
                "calibration_history": calibration_history,
                "brier_score": estimator.brier_scores.get(agent_id),
            })

        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error getting agent calibration: {e}")
            return error_response(f"Invalid agent data: {e}", 400)
        except Exception as e:
            logger.exception(f"Unexpected error getting agent calibration: {e}")
            return error_response(f"Failed to get agent calibration: {e}", 500)
