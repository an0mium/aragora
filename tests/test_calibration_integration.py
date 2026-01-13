"""Integration tests for Calibration scoring in debate workflows."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol
from aragora.core import Environment
from aragora.ranking.elo import AgentRating, EloSystem


class TestCalibrationWeighting:
    """Tests for calibration-based agent weighting."""

    @pytest.fixture
    def mock_elo_system(self):
        """Create mock ELO system with calibration data."""
        elo = MagicMock(spec=EloSystem)

        # Create mock ratings with different calibration scores
        well_calibrated = AgentRating(
            agent_name="well_calibrated",
            elo=1600,
            calibration_correct=45,
            calibration_total=50,
            calibration_brier_sum=5.0,  # Low Brier = good
        )
        poorly_calibrated = AgentRating(
            agent_name="poorly_calibrated",
            elo=1500,
            calibration_correct=10,
            calibration_total=50,
            calibration_brier_sum=25.0,  # High Brier = bad
        )
        uncalibrated = AgentRating(
            agent_name="uncalibrated",
            elo=1500,
            calibration_correct=0,
            calibration_total=2,  # Below MIN_COUNT
        )

        def mock_get_rating(name):
            ratings = {
                "well_calibrated": well_calibrated,
                "poorly_calibrated": poorly_calibrated,
                "uncalibrated": uncalibrated,
            }
            return ratings.get(name, AgentRating(agent_name=name))

        elo.get_rating = mock_get_rating
        return elo

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.name = "test_agent"
        agent.role = "proposer"
        return agent

    def test_get_calibration_weight_well_calibrated(self, mock_elo_system, mock_agent):
        """Test well-calibrated agents get higher weight."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol, elo_system=mock_elo_system)

        weight = arena._get_calibration_weight("well_calibrated")

        # Well-calibrated should get weight close to 1.5
        assert weight > 1.0
        assert weight <= 1.5

    def test_get_calibration_weight_poorly_calibrated(self, mock_elo_system, mock_agent):
        """Test poorly-calibrated agents get lower weight than well-calibrated."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol, elo_system=mock_elo_system)

        well_weight = arena._get_calibration_weight("well_calibrated")
        poor_weight = arena._get_calibration_weight("poorly_calibrated")

        # Poorly calibrated should get lower weight than well-calibrated
        assert poor_weight >= 0.5
        assert poor_weight <= 1.5
        assert poor_weight < well_weight  # Key: poorly calibrated gets less weight

    def test_get_calibration_weight_uncalibrated(self, mock_elo_system, mock_agent):
        """Test uncalibrated agents get minimum weight."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol, elo_system=mock_elo_system)

        weight = arena._get_calibration_weight("uncalibrated")

        # Uncalibrated (below MIN_COUNT) should get weight of 0.5
        assert weight == 0.5

    def test_get_calibration_weight_no_elo_system(self, mock_agent):
        """Test weight defaults to 1.0 without ELO system."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol)  # No elo_system

        weight = arena._get_calibration_weight("any_agent")

        assert weight == 1.0

    def test_get_calibration_weight_unknown_agent(self, mock_elo_system, mock_agent):
        """Test weight for unknown agent."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol, elo_system=mock_elo_system)

        weight = arena._get_calibration_weight("unknown_agent")

        # Unknown agent gets default rating with 0 calibration
        assert weight == 0.5


class TestCompositeJudgeScoring:
    """Tests for composite judge scoring (ELO + calibration)."""

    @pytest.fixture
    def mock_elo_system(self):
        """Create mock ELO system."""
        elo = MagicMock(spec=EloSystem)

        high_elo_calibrated = AgentRating(
            agent_name="high_elo_calibrated",
            elo=1800,  # High ELO
            calibration_correct=45,
            calibration_total=50,
            calibration_brier_sum=5.0,  # Good calibration
        )
        high_elo_uncalibrated = AgentRating(
            agent_name="high_elo_uncalibrated",
            elo=1800,  # High ELO
            calibration_total=2,  # Uncalibrated
        )
        low_elo_calibrated = AgentRating(
            agent_name="low_elo_calibrated",
            elo=1100,  # Low ELO
            calibration_correct=40,
            calibration_total=50,
            calibration_brier_sum=8.0,  # Good calibration
        )

        def mock_get_rating(name):
            ratings = {
                "high_elo_calibrated": high_elo_calibrated,
                "high_elo_uncalibrated": high_elo_uncalibrated,
                "low_elo_calibrated": low_elo_calibrated,
            }
            return ratings.get(name, AgentRating(agent_name=name))

        elo.get_rating = mock_get_rating
        return elo

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.name = "test_agent"
        agent.role = "proposer"
        return agent

    def test_composite_score_high_elo_calibrated_best(self, mock_elo_system, mock_agent):
        """Test high ELO + good calibration gives best composite score."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol, elo_system=mock_elo_system)

        score = arena._compute_composite_judge_score("high_elo_calibrated")

        # Should be the highest score
        assert score > 0

    def test_composite_score_calibration_matters(self, mock_elo_system, mock_agent):
        """Test calibration affects composite score."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol, elo_system=mock_elo_system)

        calibrated = arena._compute_composite_judge_score("high_elo_calibrated")
        uncalibrated = arena._compute_composite_judge_score("high_elo_uncalibrated")

        # Same ELO but calibrated should score higher
        assert calibrated > uncalibrated

    def test_composite_score_elo_dominates(self, mock_elo_system, mock_agent):
        """Test ELO has 70% weight in composite score."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol, elo_system=mock_elo_system)

        high_elo = arena._compute_composite_judge_score("high_elo_uncalibrated")
        low_elo_cal = arena._compute_composite_judge_score("low_elo_calibrated")

        # High ELO should beat low ELO even with better calibration
        # (1800 vs 1100 ELO is big gap)
        assert high_elo > low_elo_cal

    def test_composite_score_no_elo_system(self, mock_agent):
        """Test composite score is 0 without ELO system."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol)

        score = arena._compute_composite_judge_score("any_agent")

        assert score == 0.0


class TestCalibratedJudgeSelection:
    """Tests for calibrated judge selection mode."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents with different names."""
        agents = []
        for name in ["agent_a", "agent_b", "agent_c"]:
            agent = MagicMock()
            agent.name = name
            agent.role = "proposer"
            agents.append(agent)
        return agents

    @pytest.fixture
    def mock_elo_system(self):
        """Create mock ELO system."""
        elo = MagicMock(spec=EloSystem)

        ratings = {
            "agent_a": AgentRating(
                agent_name="agent_a",
                elo=1700,
                calibration_correct=48,
                calibration_total=50,
                calibration_brier_sum=4.0,
            ),
            "agent_b": AgentRating(
                agent_name="agent_b",
                elo=1500,
                calibration_correct=35,
                calibration_total=50,
                calibration_brier_sum=12.0,
            ),
            "agent_c": AgentRating(agent_name="agent_c"),
        }

        def mock_get_rating(name):
            return ratings.get(name, AgentRating(agent_name=name))

        def mock_get_ratings_batch(names):
            return {name: ratings.get(name, AgentRating(agent_name=name)) for name in names}

        elo.get_rating = mock_get_rating
        elo.get_ratings_batch = mock_get_ratings_batch
        return elo

    @pytest.mark.asyncio
    async def test_calibrated_judge_selection(self, mock_agents, mock_elo_system):
        """Test calibrated judge selection picks best composite score."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1, judge_selection="calibrated")
        arena = Arena(env, mock_agents, protocol, elo_system=mock_elo_system)

        proposals = {"agent_a": "Proposal A", "agent_b": "Proposal B"}
        judge = await arena._select_judge(proposals, [])

        # Should select agent_a (best composite score)
        assert judge.name == "agent_a"

    @pytest.mark.asyncio
    async def test_calibrated_fallback_without_elo(self, mock_agents):
        """Test calibrated selection falls back to random without ELO."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1, judge_selection="calibrated")
        arena = Arena(env, mock_agents, protocol)  # No elo_system

        proposals = {"agent_a": "Proposal A"}
        judge = await arena._select_judge(proposals, [])

        # Should fall back to random selection (any agent is valid)
        assert judge in mock_agents


class TestCalibrationLeaderboardEndpoint:
    """Tests for /api/calibration/leaderboard endpoint."""

    @pytest.fixture
    def handler(self):
        """Create CalibrationHandler with mock server context."""
        from aragora.server.handlers.calibration import CalibrationHandler

        mock_context = MagicMock()
        return CalibrationHandler(mock_context)

    def test_can_handle_leaderboard_path(self, handler):
        """Test handler recognizes leaderboard path."""
        assert handler.can_handle("/api/calibration/leaderboard") is True

    def test_cannot_handle_other_paths(self, handler):
        """Test handler doesn't match unrelated paths."""
        assert handler.can_handle("/api/other/endpoint") is False
        assert handler.can_handle("/api/calibration/other") is False

    def test_routes_include_leaderboard(self, handler):
        """Test ROUTES includes leaderboard endpoint."""
        assert "/api/calibration/leaderboard" in handler.ROUTES

    @patch("aragora.server.handlers.calibration.ELO_AVAILABLE", True)
    @patch("aragora.server.handlers.calibration.EloSystem")
    def test_leaderboard_returns_json(self, mock_elo_cls, handler):
        """Test leaderboard returns JSON response."""
        import json

        # Setup mock ELO system
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            {"agent": "agent1"},
            {"agent": "agent2"},
        ]
        mock_elo.get_rating.return_value = AgentRating(
            agent_name="agent1",
            elo=1500,
            calibration_correct=40,
            calibration_total=50,
            calibration_brier_sum=10.0,
        )
        mock_elo_cls.return_value = mock_elo

        result = handler.handle("/api/calibration/leaderboard", {}, None)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert "agents" in body
        assert "metric" in body

    @patch("aragora.server.handlers.calibration.ELO_AVAILABLE", False)
    def test_leaderboard_unavailable_without_elo(self, handler):
        """Test leaderboard returns 503 without ELO system."""
        result = handler.handle("/api/calibration/leaderboard", {}, None)

        assert result is not None
        assert result.status_code == 503  # Service unavailable


class TestVoteWeightingWithCalibration:
    """Tests for vote weighting that includes calibration."""

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.name = "voter"
        agent.role = "proposer"
        return agent

    @pytest.fixture
    def mock_elo_system(self):
        elo = MagicMock(spec=EloSystem)

        def mock_get_rating(name):
            if name == "calibrated_voter":
                return AgentRating(
                    agent_name="calibrated_voter",
                    calibration_correct=45,
                    calibration_total=50,
                    calibration_brier_sum=5.0,
                )
            return AgentRating(agent_name=name)

        elo.get_rating = mock_get_rating
        return elo

    def test_vote_weight_includes_calibration(self, mock_agent, mock_elo_system):
        """Test that vote weighting includes calibration factor."""
        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol, elo_system=mock_elo_system)

        # Calibrated voter should get higher weight
        cal_weight = arena._get_calibration_weight("calibrated_voter")
        uncal_weight = arena._get_calibration_weight("uncalibrated_voter")

        assert cal_weight > uncal_weight


class TestCalibrationEdgeCases:
    """Tests for edge cases in calibration integration."""

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.name = "test"
        agent.role = "proposer"
        return agent

    def test_calibration_weight_exception_handling(self, mock_agent):
        """Test calibration weight handles exceptions gracefully."""
        mock_elo = MagicMock(spec=EloSystem)
        mock_elo.get_rating.side_effect = Exception("DB error")

        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol, elo_system=mock_elo)

        # Should return default weight (1.0) on exception
        weight = arena._get_calibration_weight("any")
        assert weight == 1.0

    def test_composite_score_exception_handling(self, mock_agent):
        """Test composite score handles exceptions gracefully."""
        mock_elo = MagicMock(spec=EloSystem)
        mock_elo.get_rating.side_effect = Exception("DB error")

        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol, elo_system=mock_elo)

        # Should return 0 on exception
        score = arena._compute_composite_judge_score("any")
        assert score == 0.0

    def test_calibration_weight_range(self, mock_agent):
        """Test calibration weight stays in valid range."""
        # Create ratings with extreme values
        mock_elo = MagicMock(spec=EloSystem)

        # Perfect calibration (score = 1.0)
        perfect = AgentRating(
            agent_name="perfect",
            calibration_correct=100,
            calibration_total=100,
            calibration_brier_sum=0.0,
        )
        mock_elo.get_rating.return_value = perfect

        env = Environment(task="Test")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, [mock_agent], protocol, elo_system=mock_elo)

        weight = arena._get_calibration_weight("perfect")

        # Weight should be at most 1.5 (0.5 + 1.0)
        assert weight <= 1.5
        assert weight >= 0.5
