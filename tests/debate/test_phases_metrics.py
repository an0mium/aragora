"""
Tests for the MetricsHelper class and related utilities.

Tests calibration weights, composite scores, and domain extraction.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


@dataclass
class MockRating:
    """Mock agent rating for testing."""

    elo: float = 1200.0
    calibration_score: float = 0.7


class TestMetricsHelper:
    """Tests for MetricsHelper class."""

    def test_init_defaults(self):
        """Should initialize with default weights."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        assert helper._elo_weight == 0.7
        assert helper._calibration_weight == 0.3
        assert helper._min_cal_weight == 0.5
        assert helper._max_cal_weight == 1.5

    def test_init_custom_weights(self):
        """Should accept custom weight parameters."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper(
            elo_weight=0.6,
            calibration_weight=0.4,
            min_calibration_weight=0.3,
            max_calibration_weight=2.0,
        )

        assert helper._elo_weight == 0.6
        assert helper._calibration_weight == 0.4
        assert helper._min_cal_weight == 0.3
        assert helper._max_cal_weight == 2.0


class TestCalibrationWeight:
    """Tests for calibration weight calculation."""

    def test_returns_one_without_elo_system(self):
        """Should return 1.0 when no ELO system."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper(elo_system=None)

        weight = helper.get_calibration_weight("alice")

        assert weight == 1.0

    def test_maps_zero_score_to_min(self):
        """Zero calibration score should map to min weight."""
        from aragora.debate.phases.metrics import MetricsHelper

        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = MockRating(calibration_score=0.0)

        helper = MetricsHelper(
            elo_system=mock_elo,
            min_calibration_weight=0.5,
            max_calibration_weight=1.5,
        )

        weight = helper.get_calibration_weight("alice")

        assert weight == 0.5

    def test_maps_one_score_to_max(self):
        """Perfect calibration score should map to max weight."""
        from aragora.debate.phases.metrics import MetricsHelper

        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = MockRating(calibration_score=1.0)

        helper = MetricsHelper(
            elo_system=mock_elo,
            min_calibration_weight=0.5,
            max_calibration_weight=1.5,
        )

        weight = helper.get_calibration_weight("alice")

        assert weight == 1.5

    def test_maps_mid_score_correctly(self):
        """Mid calibration score should map to middle of range."""
        from aragora.debate.phases.metrics import MetricsHelper

        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = MockRating(calibration_score=0.5)

        helper = MetricsHelper(
            elo_system=mock_elo,
            min_calibration_weight=0.5,
            max_calibration_weight=1.5,
        )

        weight = helper.get_calibration_weight("alice")

        assert weight == 1.0  # Midpoint of 0.5-1.5 range

    def test_handles_lookup_error_gracefully(self):
        """Should return 1.0 on lookup error."""
        from aragora.debate.phases.metrics import MetricsHelper

        mock_elo = MagicMock()
        mock_elo.get_rating.side_effect = KeyError("Unknown agent")

        helper = MetricsHelper(elo_system=mock_elo)

        weight = helper.get_calibration_weight("unknown_agent")

        assert weight == 1.0


class TestCompositeJudgeScore:
    """Tests for composite judge score calculation."""

    def test_returns_zero_without_elo_system(self):
        """Should return 0.0 when no ELO system."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper(elo_system=None)

        score = helper.get_composite_judge_score("alice")

        assert score == 0.0

    def test_combines_elo_and_calibration(self):
        """Should combine ELO and calibration with configured weights."""
        from aragora.debate.phases.metrics import MetricsHelper

        mock_elo = MagicMock()
        # ELO 1500 -> normalized (1500-1000)/500 = 1.0
        # Calibration 0.8
        mock_elo.get_rating.return_value = MockRating(elo=1500, calibration_score=0.8)

        helper = MetricsHelper(
            elo_system=mock_elo,
            elo_weight=0.7,
            calibration_weight=0.3,
        )

        score = helper.get_composite_judge_score("alice")

        # Expected: (1.0 * 0.7) + (0.8 * 0.3) = 0.7 + 0.24 = 0.94
        assert score == pytest.approx(0.94, abs=0.01)

    def test_floors_negative_elo_at_zero(self):
        """ELO below 1000 should floor at 0 for scoring."""
        from aragora.debate.phases.metrics import MetricsHelper

        mock_elo = MagicMock()
        # ELO 800 -> normalized (800-1000)/500 = -0.4 -> floored to 0
        mock_elo.get_rating.return_value = MockRating(elo=800, calibration_score=1.0)

        helper = MetricsHelper(
            elo_system=mock_elo,
            elo_weight=0.7,
            calibration_weight=0.3,
        )

        score = helper.get_composite_judge_score("alice")

        # Expected: (0 * 0.7) + (1.0 * 0.3) = 0.3
        assert score == pytest.approx(0.3, abs=0.01)

    def test_handles_error_gracefully(self):
        """Should return 0.0 on error."""
        from aragora.debate.phases.metrics import MetricsHelper

        mock_elo = MagicMock()
        mock_elo.get_rating.side_effect = Exception("Database error")

        helper = MetricsHelper(elo_system=mock_elo)

        score = helper.get_composite_judge_score("alice")

        assert score == 0.0


class TestDomainExtraction:
    """Tests for domain extraction from tasks."""

    def test_extracts_security_domain(self):
        """Should detect security domain."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        assert helper.extract_domain("Fix the authentication vulnerability") == "security"
        assert helper.extract_domain("Review encryption implementation") == "security"
        assert helper.extract_domain("Audit for security issues") == "security"

    def test_extracts_performance_domain(self):
        """Should detect performance domain."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        assert helper.extract_domain("Reduce latency issues") == "performance"
        assert helper.extract_domain("Add a cache layer") == "performance"
        assert helper.extract_domain("Improve speed") == "performance"
        assert helper.extract_domain("Measure performance metrics") == "performance"

    def test_extracts_testing_domain(self):
        """Should detect testing domain."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        assert helper.extract_domain("Write unit tests for module") == "testing"
        assert helper.extract_domain("Improve test coverage") == "testing"

    def test_extracts_architecture_domain(self):
        """Should detect architecture domain."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        assert helper.extract_domain("Design system architecture") == "architecture"
        assert helper.extract_domain("Review design patterns") == "architecture"

    def test_extracts_debugging_domain(self):
        """Should detect debugging domain."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        assert helper.extract_domain("Fix the null pointer bug") == "debugging"
        assert helper.extract_domain("Handle the crash on startup") == "debugging"

    def test_extracts_api_domain(self):
        """Should detect API domain."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        assert helper.extract_domain("Create REST endpoint") == "api"
        assert helper.extract_domain("Implement GraphQL schema") == "api"

    def test_extracts_database_domain(self):
        """Should detect database domain."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        assert helper.extract_domain("Write SQL migration") == "database"
        assert helper.extract_domain("Create database table") == "database"
        assert helper.extract_domain("Run complex query") == "database"

    def test_extracts_frontend_domain(self):
        """Should detect frontend domain."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        assert helper.extract_domain("Build React component") == "frontend"
        assert helper.extract_domain("Update CSS styles") == "frontend"
        assert helper.extract_domain("Improve UI layout") == "frontend"

    def test_defaults_to_general(self):
        """Unknown tasks should default to 'general'."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        assert helper.extract_domain("Implement new feature") == "general"
        assert helper.extract_domain("Random task description") == "general"

    def test_caches_results(self):
        """Should cache domain extraction results."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        task = "Fix security vulnerability"
        helper.extract_domain(task)

        assert task in helper._domain_cache
        assert helper._domain_cache[task] == "security"

    def test_clear_cache(self):
        """Should clear domain cache."""
        from aragora.debate.phases.metrics import MetricsHelper

        helper = MetricsHelper()

        helper.extract_domain("Test task")
        assert len(helper._domain_cache) == 1

        helper.clear_cache()
        assert len(helper._domain_cache) == 0


class TestBuildRelationshipUpdates:
    """Tests for the build_relationship_updates utility function."""

    def test_generates_all_pairs(self):
        """Should generate updates for all agent pairs."""
        from aragora.debate.phases.metrics import build_relationship_updates

        updates = build_relationship_updates(
            participants=["alice", "bob", "charlie"],
            vote_choices={},
            winner=None,
        )

        # 3 agents -> 3 pairs: (alice,bob), (alice,charlie), (bob,charlie)
        assert len(updates) == 3

    def test_tracks_agreement(self):
        """Should track when agents agree on votes."""
        from aragora.debate.phases.metrics import build_relationship_updates

        updates = build_relationship_updates(
            participants=["alice", "bob", "charlie"],
            vote_choices={"alice": "proposal_1", "bob": "proposal_1", "charlie": "proposal_2"},
            winner=None,
        )

        # alice and bob agreed
        alice_bob = next(u for u in updates if u["agent_a"] == "alice" and u["agent_b"] == "bob")
        assert alice_bob["agreement_increment"] == 1

        # alice and charlie disagreed
        alice_charlie = next(
            u for u in updates if u["agent_a"] == "alice" and u["agent_b"] == "charlie"
        )
        assert alice_charlie["agreement_increment"] == 0

    def test_tracks_winner(self):
        """Should track wins for the winning agent."""
        from aragora.debate.phases.metrics import build_relationship_updates

        updates = build_relationship_updates(
            participants=["alice", "bob", "charlie"],
            vote_choices={},
            winner="bob",
        )

        for update in updates:
            if update["agent_a"] == "bob":
                assert update["a_win"] == 1
                assert update["b_win"] == 0
            elif update["agent_b"] == "bob":
                assert update["a_win"] == 0
                assert update["b_win"] == 1
            else:
                assert update["a_win"] == 0
                assert update["b_win"] == 0

    def test_increments_debate_count(self):
        """Should increment debate count for all pairs."""
        from aragora.debate.phases.metrics import build_relationship_updates

        updates = build_relationship_updates(
            participants=["alice", "bob"],
            vote_choices={},
            winner=None,
        )

        assert all(u["debate_increment"] == 1 for u in updates)

    def test_empty_participants(self):
        """Empty participants should return empty updates."""
        from aragora.debate.phases.metrics import build_relationship_updates

        updates = build_relationship_updates(
            participants=[],
            vote_choices={},
            winner=None,
        )

        assert updates == []

    def test_single_participant(self):
        """Single participant should return empty updates."""
        from aragora.debate.phases.metrics import build_relationship_updates

        updates = build_relationship_updates(
            participants=["alice"],
            vote_choices={},
            winner=None,
        )

        assert updates == []
