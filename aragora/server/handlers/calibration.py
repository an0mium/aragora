"""
Calibration endpoint handlers.

Provides calibration curve and summary data for agents.

Endpoints:
- GET /api/agent/{name}/calibration-curve - Get calibration curve (confidence vs accuracy)
- GET /api/agent/{name}/calibration-summary - Get calibration summary metrics
- GET /api/calibration/leaderboard - Get top agents by calibration score
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_clamped_int_param,
    get_string_param,
    handle_errors,
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
        "/api/calibration/visualization",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path.startswith("/api/agent/") and (
            path.endswith("/calibration-curve") or path.endswith("/calibration-summary")
        ):
            return True
        if path in ("/api/calibration/leaderboard", "/api/calibration/visualization"):
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

        # Handle visualization endpoint
        if path == "/api/calibration/visualization":
            limit = get_clamped_int_param(query_params, 'limit', 5, min_val=1, max_val=10)
            return self._get_calibration_visualization(limit)

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

    @handle_errors("calibration curve retrieval")
    def _get_calibration_curve(
        self, agent: str, buckets: int, domain: Optional[str]
    ) -> HandlerResult:
        """Get calibration curve (expected vs actual accuracy per bucket)."""
        if not CALIBRATION_AVAILABLE or not CalibrationTracker:
            return error_response("Calibration tracker not available", 503)

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

    @handle_errors("calibration summary retrieval")
    def _get_calibration_summary(
        self, agent: str, domain: Optional[str]
    ) -> HandlerResult:
        """Get comprehensive calibration summary for an agent."""
        if not CALIBRATION_AVAILABLE or not CalibrationTracker:
            return error_response("Calibration tracker not available", 503)

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

    @handle_errors("calibration leaderboard retrieval")
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

    @handle_errors("calibration visualization retrieval")
    def _get_calibration_visualization(self, limit: int) -> HandlerResult:
        """Get comprehensive calibration visualization data.

        Returns data optimized for chart rendering:
        - Calibration curves for each agent (perfect calibration line comparison)
        - Agent comparison scatter plot data (confidence vs accuracy)
        - Confidence distribution histogram
        - Domain performance heatmap data

        Args:
            limit: Maximum number of agents to include

        Returns:
            JSON response with visualization-ready data
        """
        if not CALIBRATION_AVAILABLE or not CalibrationTracker:
            return error_response("Calibration tracker not available", 503)

        tracker = CalibrationTracker()
        result: dict = {
            "calibration_curves": {},
            "scatter_data": [],
            "confidence_histogram": [],
            "domain_heatmap": {},
            "summary": {
                "total_agents": 0,
                "avg_brier": 0.0,
                "avg_ece": 0.0,
                "best_calibrated": None,
                "worst_calibrated": None,
            },
        }

        try:
            all_agents = tracker.get_all_agents()
            result["summary"]["total_agents"] = len(all_agents)

            if not all_agents:
                return json_response(result)

            # Collect data for each agent
            agent_summaries = []
            for agent in all_agents[:limit]:
                try:
                    summary = tracker.get_calibration_summary(agent)
                    if summary and summary.total_predictions >= 5:
                        agent_summaries.append({
                            "agent": agent,
                            "summary": summary,
                        })
                except Exception as e:
                    logger.debug(f"Error getting summary for {agent}: {e}")
                    continue

            if not agent_summaries:
                return json_response(result)

            # 1. Calibration curves for each agent
            for entry in agent_summaries:
                agent = entry["agent"]
                try:
                    curve = tracker.get_calibration_curve(agent, num_buckets=10)
                    if curve:
                        result["calibration_curves"][agent] = {
                            "buckets": [
                                {
                                    "x": (b.range_start + b.range_end) / 2,  # Midpoint
                                    "expected": (b.range_start + b.range_end) / 2,
                                    "actual": b.accuracy,
                                    "count": b.total_predictions,
                                }
                                for b in curve
                            ],
                            "perfect_line": [
                                {"x": i / 10, "y": i / 10}
                                for i in range(11)
                            ],
                        }
                except Exception as e:
                    logger.debug(f"Error getting curve for {agent}: {e}")

            # 2. Scatter plot data (agent confidence vs accuracy)
            for entry in agent_summaries:
                agent = entry["agent"]
                summary = entry["summary"]
                result["scatter_data"].append({
                    "agent": agent,
                    "accuracy": summary.accuracy,
                    "brier_score": summary.brier_score,
                    "ece": summary.ece,
                    "predictions": summary.total_predictions,
                    "is_overconfident": summary.is_overconfident,
                    "is_underconfident": summary.is_underconfident,
                })

            # 3. Confidence distribution histogram (aggregate)
            confidence_buckets = {i: 0 for i in range(10)}
            for entry in agent_summaries:
                agent = entry["agent"]
                try:
                    curve = tracker.get_calibration_curve(agent, num_buckets=10)
                    if curve:
                        for i, b in enumerate(curve):
                            if i < 10:
                                confidence_buckets[i] += b.total_predictions
                except Exception as e:
                    logger.debug(f"Failed to get calibration curve for {agent}: {e}")

            result["confidence_histogram"] = [
                {
                    "range": f"{i * 10}-{(i + 1) * 10}%",
                    "count": count,
                }
                for i, count in confidence_buckets.items()
            ]

            # 4. Domain heatmap (agents x domains)
            for entry in agent_summaries:
                agent = entry["agent"]
                try:
                    domain_data = tracker.get_domain_breakdown(agent)
                    if domain_data:
                        result["domain_heatmap"][agent] = {
                            domain: {
                                "accuracy": s.accuracy,
                                "brier": s.brier_score,
                                "count": s.total_predictions,
                            }
                            for domain, s in domain_data.items()
                        }
                except Exception as e:
                    logger.debug(f"Error getting domains for {agent}: {e}")

            # 5. Summary statistics
            brier_scores = [e["summary"].brier_score for e in agent_summaries]
            ece_scores = [e["summary"].ece for e in agent_summaries]

            if brier_scores:
                result["summary"]["avg_brier"] = round(sum(brier_scores) / len(brier_scores), 4)
                best_idx = brier_scores.index(min(brier_scores))
                worst_idx = brier_scores.index(max(brier_scores))
                result["summary"]["best_calibrated"] = agent_summaries[best_idx]["agent"]
                result["summary"]["worst_calibrated"] = agent_summaries[worst_idx]["agent"]

            if ece_scores:
                result["summary"]["avg_ece"] = round(sum(ece_scores) / len(ece_scores), 4)

        except Exception as e:
            logger.warning(f"Calibration visualization error: {e}")

        return json_response(result)
