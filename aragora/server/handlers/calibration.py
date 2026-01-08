"""
Calibration endpoint handlers.

Provides calibration curve and summary data for agents.

Endpoints:
- GET /api/agent/{name}/calibration-curve - Get calibration curve (confidence vs accuracy)
- GET /api/agent/{name}/calibration-summary - Get calibration summary metrics
- GET /api/calibration/leaderboard - Get top agents by calibration score
"""

import logging
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_clamped_int_param,
    get_string_param,
    SAFE_AGENT_PATTERN,
)
from aragora.utils.optional_imports import try_import_class

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies using centralized utility
EloSystem, ELO_AVAILABLE = try_import_class("aragora.ranking.elo", "EloSystem")
CalibrationTracker, CALIBRATION_AVAILABLE = try_import_class(
    "aragora.agents.calibration", "CalibrationTracker"
)


class CalibrationHandler(BaseHandler):
    """Handler for calibration-related endpoints."""

    ROUTES = [
        "/api/agent/*/calibration-curve",
        "/api/agent/*/calibration-summary",
        "/api/calibration/leaderboard",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path.startswith("/api/agent/") and (
            path.endswith("/calibration-curve") or path.endswith("/calibration-summary")
        ):
            return True
        if path == "/api/calibration/leaderboard":
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route calibration requests to appropriate methods."""
        # Handle leaderboard endpoint
        if path == "/api/calibration/leaderboard":
            limit = get_clamped_int_param(query_params, 'limit', 20, min_val=1, max_val=100)
            metric = get_string_param(query_params, 'metric') or 'brier'
            min_predictions = get_clamped_int_param(query_params, 'min_predictions', 5, min_val=1, max_val=1000)
            return self._get_calibration_leaderboard(limit, metric, min_predictions)

        if not path.startswith("/api/agent/"):
            return None

        # Extract agent name: /api/agent/{name}/calibration-*
        agent, err = self.extract_path_param(path, 2, "agent", SAFE_AGENT_PATTERN)
        if err:
            return err

        if path.endswith("/calibration-curve"):
            buckets = get_clamped_int_param(query_params, 'buckets', 10, min_val=5, max_val=20)
            domain = get_string_param(query_params, 'domain')
            return self._get_calibration_curve(agent, buckets, domain)
        elif path.endswith("/calibration-summary"):
            domain = get_string_param(query_params, 'domain')
            return self._get_calibration_summary(agent, domain)

        return None

    def _get_calibration_curve(
        self, agent: str, buckets: int, domain: Optional[str]
    ) -> HandlerResult:
        """Get calibration curve (expected vs actual accuracy per bucket)."""
        if not CALIBRATION_AVAILABLE or not CalibrationTracker:
            return error_response("Calibration tracker not available", 503)

        try:
            tracker = CalibrationTracker()
            curve = tracker.get_calibration_curve(agent, num_buckets=buckets, domain=domain)
            return json_response({
                "agent": agent,
                "domain": domain,
                "buckets": [
                    {
                        "range_start": b.range_start,
                        "range_end": b.range_end,
                        "total_predictions": b.total_predictions,
                        "correct_predictions": b.correct_predictions,
                        "accuracy": b.accuracy,
                        "expected_accuracy": (b.range_start + b.range_end) / 2,
                        "brier_score": b.brier_score,
                    }
                    for b in curve
                ],
                "count": len(curve),
            })
        except Exception as e:
            logger.error(f"Error getting calibration curve for {agent}: {e}", exc_info=True)
            return error_response("Failed to get calibration curve", 500)

    def _get_calibration_summary(
        self, agent: str, domain: Optional[str]
    ) -> HandlerResult:
        """Get comprehensive calibration summary for an agent."""
        if not CALIBRATION_AVAILABLE or not CalibrationTracker:
            return error_response("Calibration tracker not available", 503)

        try:
            tracker = CalibrationTracker()
            summary = tracker.get_calibration_summary(agent, domain=domain)

            return json_response({
                "agent": summary.agent,
                "domain": domain,
                "total_predictions": summary.total_predictions,
                "total_correct": summary.total_correct,
                "accuracy": summary.accuracy,
                "brier_score": summary.brier_score,
                "ece": summary.ece,
                "is_overconfident": summary.is_overconfident,
                "is_underconfident": summary.is_underconfident,
            })
        except Exception as e:
            logger.error(f"Error getting calibration summary for {agent}: {e}", exc_info=True)
            return error_response("Failed to get calibration summary", 500)

    def _get_calibration_leaderboard(
        self, limit: int, metric: str, min_predictions: int
    ) -> HandlerResult:
        """Get leaderboard of agents ranked by calibration score.

        Args:
            limit: Maximum number of agents to return
            metric: Sorting metric - 'brier' (default), 'ece', 'accuracy', or 'composite'
            min_predictions: Minimum predictions required to appear on leaderboard

        Returns:
            JSON response with ranked agent calibration data
        """
        if not ELO_AVAILABLE or not EloSystem:
            return error_response("ELO system not available", 503)

        try:
            elo = EloSystem()

            # Get all agents with enough calibration data
            # EloSystem.get_leaderboard returns basic data; we need to query each agent
            leaderboard_data = elo.get_leaderboard(limit=100)  # Get more to filter

            calibration_entries = []
            for entry in leaderboard_data:
                agent_name = entry.get("agent", "")
                if not agent_name:
                    continue

                try:
                    rating = elo.get_rating(agent_name)

                    # Skip agents with insufficient predictions
                    if rating.calibration_total < min_predictions:
                        continue

                    calibration_entries.append({
                        "agent": agent_name,
                        "calibration_score": rating.calibration_score,
                        "brier_score": rating.calibration_brier_score,
                        "accuracy": rating.calibration_accuracy,
                        "ece": 1.0 - rating.calibration_score if rating.calibration_score > 0 else 1.0,
                        "predictions_count": rating.calibration_total,
                        "correct_count": rating.calibration_correct,
                        "elo": rating.elo,
                    })
                except Exception as e:
                    logger.debug(f"Skipping agent {agent_name}: {e}")
                    continue

            # Sort by the requested metric
            if metric == "brier":
                # Lower Brier is better
                calibration_entries.sort(key=lambda x: x["brier_score"])
            elif metric == "ece":
                # Lower ECE is better
                calibration_entries.sort(key=lambda x: x["ece"])
            elif metric == "accuracy":
                # Higher accuracy is better
                calibration_entries.sort(key=lambda x: x["accuracy"], reverse=True)
            else:  # composite (default)
                # Higher calibration_score is better
                calibration_entries.sort(key=lambda x: x["calibration_score"], reverse=True)

            # Limit results
            calibration_entries = calibration_entries[:limit]

            return json_response({
                "metric": metric,
                "min_predictions": min_predictions,
                "agents": calibration_entries,
                "count": len(calibration_entries),
            })

        except Exception as e:
            logger.error(f"Error getting calibration leaderboard: {e}", exc_info=True)
            return error_response("Failed to get calibration leaderboard", 500)
