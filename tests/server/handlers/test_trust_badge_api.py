"""
Tests for Calibration Trust Scores API integration.

Tests cover:
- _compute_trust_tier boundary values
- agent_to_dict enrichment with calibration data
- Graceful degradation when CalibrationTracker unavailable
- Trust tier semantics
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Trust tier computation
# ---------------------------------------------------------------------------

class TestComputeTrustTier:
    """Tests for _compute_trust_tier helper."""

    def test_unrated_below_threshold(self):
        """Agents with fewer than 20 predictions are unrated."""
        from aragora.server.handlers.base import _compute_trust_tier

        assert _compute_trust_tier(0.05, 19) == "unrated"
        assert _compute_trust_tier(0.05, 0) == "unrated"
        assert _compute_trust_tier(0.05, 1) == "unrated"

    def test_excellent_tier(self):
        """Brier < 0.1 with sufficient predictions = excellent."""
        from aragora.server.handlers.base import _compute_trust_tier

        assert _compute_trust_tier(0.05, 50) == "excellent"
        assert _compute_trust_tier(0.0, 20) == "excellent"
        assert _compute_trust_tier(0.099, 100) == "excellent"

    def test_good_tier(self):
        """0.1 <= Brier < 0.2 = good."""
        from aragora.server.handlers.base import _compute_trust_tier

        assert _compute_trust_tier(0.1, 50) == "good"
        assert _compute_trust_tier(0.15, 50) == "good"
        assert _compute_trust_tier(0.199, 50) == "good"

    def test_moderate_tier(self):
        """0.2 <= Brier < 0.35 = moderate."""
        from aragora.server.handlers.base import _compute_trust_tier

        assert _compute_trust_tier(0.2, 50) == "moderate"
        assert _compute_trust_tier(0.25, 50) == "moderate"
        assert _compute_trust_tier(0.349, 50) == "moderate"

    def test_poor_tier(self):
        """Brier >= 0.35 = poor."""
        from aragora.server.handlers.base import _compute_trust_tier

        assert _compute_trust_tier(0.35, 50) == "poor"
        assert _compute_trust_tier(0.5, 50) == "poor"
        assert _compute_trust_tier(1.0, 50) == "poor"

    def test_boundary_excellent_good(self):
        """Exact boundary at 0.1: should be good, not excellent."""
        from aragora.server.handlers.base import _compute_trust_tier

        assert _compute_trust_tier(0.1, 50) == "good"

    def test_boundary_good_moderate(self):
        """Exact boundary at 0.2: should be moderate, not good."""
        from aragora.server.handlers.base import _compute_trust_tier

        assert _compute_trust_tier(0.2, 50) == "moderate"

    def test_boundary_moderate_poor(self):
        """Exact boundary at 0.35: should be poor, not moderate."""
        from aragora.server.handlers.base import _compute_trust_tier

        assert _compute_trust_tier(0.35, 50) == "poor"

    def test_unrated_takes_priority_over_brier(self):
        """Even with high Brier, <20 predictions → unrated."""
        from aragora.server.handlers.base import _compute_trust_tier

        assert _compute_trust_tier(0.9, 10) == "unrated"

    def test_exactly_20_predictions_not_unrated(self):
        """Exactly 20 predictions → uses Brier score, not unrated."""
        from aragora.server.handlers.base import _compute_trust_tier

        assert _compute_trust_tier(0.05, 20) == "excellent"


# ---------------------------------------------------------------------------
# agent_to_dict enrichment
# ---------------------------------------------------------------------------

@dataclass
class _MockSummary:
    brier_score: float = 0.15
    ece: float = 0.08
    total_predictions: int = 50


class TestAgentToDictCalibration:
    """Tests for agent_to_dict with calibration enrichment."""

    @pytest.fixture
    def tracker(self):
        """Create a mock CalibrationTracker."""
        mock = MagicMock()
        mock.get_calibration_summary.return_value = _MockSummary(
            brier_score=0.15, ece=0.08, total_predictions=50
        )
        return mock

    @pytest.fixture
    def tracker_no_data(self):
        """Tracker that has no data for the agent."""
        mock = MagicMock()
        mock.get_calibration_summary.return_value = _MockSummary(
            brier_score=0.0, ece=0.0, total_predictions=0
        )
        return mock

    def test_no_tracker_no_calibration(self):
        """Without tracker, no calibration field added."""
        from aragora.server.handlers.base import agent_to_dict

        agent = MagicMock()
        agent.name = "claude"
        agent.agent_name = "claude"
        agent.elo = 1600
        agent.wins = 10
        agent.losses = 5
        agent.draws = 0
        agent.win_rate = 0.67
        agent.games_played = 15
        agent.matches = 15
        result = agent_to_dict(agent)
        assert "calibration" not in result

    def test_tracker_enriches_agent(self, tracker):
        """With tracker, calibration sub-dict is added."""
        from aragora.server.handlers.base import agent_to_dict

        agent = MagicMock()
        agent.name = "claude"
        agent.agent_name = "claude"
        agent.elo = 1600
        agent.wins = 10
        agent.losses = 5
        agent.draws = 0
        agent.win_rate = 0.67
        agent.games_played = 15
        agent.matches = 15
        result = agent_to_dict(agent, calibration_tracker=tracker)
        assert "calibration" in result
        cal = result["calibration"]
        assert cal["brier_score"] == 0.15
        assert cal["ece"] == 0.08
        assert cal["trust_tier"] == "good"
        assert cal["prediction_count"] == 50

    def test_tracker_no_predictions_skips(self, tracker_no_data):
        """Tracker with zero predictions doesn't add calibration."""
        from aragora.server.handlers.base import agent_to_dict

        agent = MagicMock()
        agent.name = "new_agent"
        agent.agent_name = "new_agent"
        agent.elo = 1500
        agent.wins = 0
        agent.losses = 0
        agent.draws = 0
        agent.win_rate = 0.0
        agent.games_played = 0
        agent.matches = 0
        result = agent_to_dict(agent, calibration_tracker=tracker_no_data)
        assert "calibration" not in result

    def test_dict_agent_also_enriched(self, tracker):
        """Dict-style agents are also enriched when tracker is provided."""
        from aragora.server.handlers.base import agent_to_dict

        agent_dict = {"name": "claude", "agent_name": "claude", "elo": 1600}
        result = agent_to_dict(agent_dict, calibration_tracker=tracker)
        assert "calibration" in result
        assert result["calibration"]["trust_tier"] == "good"

    def test_dict_agent_without_name_not_enriched(self, tracker):
        """Dict without name key is not enriched."""
        from aragora.server.handlers.base import agent_to_dict

        agent_dict = {"elo": 1600}
        result = agent_to_dict(agent_dict, calibration_tracker=tracker)
        assert "calibration" not in result

    def test_none_agent_returns_empty(self, tracker):
        """None agent returns empty dict regardless of tracker."""
        from aragora.server.handlers.base import agent_to_dict

        result = agent_to_dict(None, calibration_tracker=tracker)
        assert result == {}

    def test_tracker_exception_graceful(self):
        """Tracker that raises exceptions is handled gracefully."""
        from aragora.server.handlers.base import agent_to_dict

        tracker = MagicMock()
        tracker.get_calibration_summary.side_effect = AttributeError("no such attr")

        agent = MagicMock()
        agent.name = "claude"
        agent.agent_name = "claude"
        agent.elo = 1600
        agent.wins = 10
        agent.losses = 5
        agent.draws = 0
        agent.win_rate = 0.67
        agent.games_played = 15
        agent.matches = 15
        result = agent_to_dict(agent, calibration_tracker=tracker)
        # Should still return valid agent dict, just without calibration
        assert result["name"] == "claude"
        assert "calibration" not in result

    def test_brier_score_rounded(self, tracker):
        """Brier score is rounded to 4 decimal places."""
        from aragora.server.handlers.base import agent_to_dict

        tracker.get_calibration_summary.return_value = _MockSummary(
            brier_score=0.123456789, ece=0.087654321, total_predictions=50
        )
        agent = MagicMock()
        agent.name = "claude"
        agent.agent_name = "claude"
        agent.elo = 1600
        agent.wins = 10
        agent.losses = 5
        agent.draws = 0
        agent.win_rate = 0.67
        agent.games_played = 15
        agent.matches = 15
        result = agent_to_dict(agent, calibration_tracker=tracker)
        assert result["calibration"]["brier_score"] == 0.1235
        assert result["calibration"]["ece"] == 0.0877

    def test_backward_compat_without_tracker(self):
        """Existing callers without calibration_tracker still work."""
        from aragora.server.handlers.base import agent_to_dict

        agent = MagicMock()
        agent.name = "gpt-4"
        agent.agent_name = "gpt-4"
        agent.elo = 1550
        agent.wins = 20
        agent.losses = 10
        agent.draws = 2
        agent.win_rate = 0.65
        agent.games_played = 32
        agent.matches = 32
        result = agent_to_dict(agent)
        assert result["name"] == "gpt-4"
        assert result["elo"] == 1550
        assert "calibration" not in result

    def test_excellent_tier_in_enrichment(self):
        """Excellent calibration (Brier < 0.1) is reflected in enriched dict."""
        from aragora.server.handlers.base import agent_to_dict

        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = _MockSummary(
            brier_score=0.05, ece=0.03, total_predictions=100
        )
        agent = MagicMock()
        agent.name = "claude"
        agent.agent_name = "claude"
        agent.elo = 1700
        agent.wins = 50
        agent.losses = 10
        agent.draws = 0
        agent.win_rate = 0.83
        agent.games_played = 60
        agent.matches = 60
        result = agent_to_dict(agent, calibration_tracker=tracker)
        assert result["calibration"]["trust_tier"] == "excellent"
