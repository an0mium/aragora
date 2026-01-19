"""
Tests for metrics helper module.

Tests cover:
- MetricsHelper class
- DOMAIN_KEYWORDS mapping
- build_relationship_updates function
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from aragora.debate.phases.metrics import (
    DOMAIN_KEYWORDS,
    MetricsHelper,
    build_relationship_updates,
)


@dataclass
class MockRating:
    """Mock rating for testing."""

    elo: float = 1000.0
    calibration_score: float = 0.5


class TestDomainKeywords:
    """Tests for DOMAIN_KEYWORDS mapping."""

    def test_security_keywords(self):
        """Security domain has expected keywords."""
        assert "security" in DOMAIN_KEYWORDS
        assert "hack" in DOMAIN_KEYWORDS["security"]
        assert "vulnerability" in DOMAIN_KEYWORDS["security"]

    def test_performance_keywords(self):
        """Performance domain has expected keywords."""
        assert "performance" in DOMAIN_KEYWORDS
        assert "optimize" in DOMAIN_KEYWORDS["performance"]

    def test_testing_keywords(self):
        """Testing domain has expected keywords."""
        assert "testing" in DOMAIN_KEYWORDS
        assert "test" in DOMAIN_KEYWORDS["testing"]

    def test_all_domains_exist(self):
        """All expected domains are present."""
        expected = [
            "security",
            "performance",
            "testing",
            "architecture",
            "debugging",
            "api",
            "database",
            "frontend",
        ]
        for domain in expected:
            assert domain in DOMAIN_KEYWORDS


class TestMetricsHelper:
    """Tests for MetricsHelper class."""

    def test_init_default(self):
        """Default initialization stores values."""
        helper = MetricsHelper()

        assert helper.elo_system is None
        assert helper._elo_weight == 0.7
        assert helper._calibration_weight == 0.3

    def test_init_custom(self):
        """Custom initialization overrides defaults."""
        elo = MagicMock()
        helper = MetricsHelper(
            elo_system=elo,
            elo_weight=0.8,
            calibration_weight=0.2,
        )

        assert helper.elo_system is elo
        assert helper._elo_weight == 0.8
        assert helper._calibration_weight == 0.2

    def test_get_calibration_weight_no_elo(self):
        """Calibration weight is 1.0 without ELO system."""
        helper = MetricsHelper()

        weight = helper.get_calibration_weight("agent1")

        assert weight == 1.0

    def test_get_calibration_weight_with_elo(self):
        """Calibration weight is calculated from ELO."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(calibration_score=0.8)

        helper = MetricsHelper(elo_system=elo)
        weight = helper.get_calibration_weight("agent1")

        elo.get_rating.assert_called_with("agent1")
        # 0.5 + (0.8 * 1.0) = 1.3
        assert weight == 1.3

    def test_get_calibration_weight_bounds(self):
        """Calibration weight respects custom bounds."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(calibration_score=1.0)

        helper = MetricsHelper(
            elo_system=elo,
            min_calibration_weight=0.0,
            max_calibration_weight=2.0,
        )
        weight = helper.get_calibration_weight("agent1")

        # 0.0 + (1.0 * 2.0) = 2.0
        assert weight == 2.0

    def test_get_calibration_weight_error(self):
        """Calibration weight is 1.0 on error."""
        elo = MagicMock()
        elo.get_rating.side_effect = Exception("Rating error")

        helper = MetricsHelper(elo_system=elo)
        weight = helper.get_calibration_weight("agent1")

        assert weight == 1.0

    def test_get_composite_judge_score_no_elo(self):
        """Composite score is 0.0 without ELO system."""
        helper = MetricsHelper()

        score = helper.get_composite_judge_score("agent1")

        assert score == 0.0

    def test_get_composite_judge_score(self):
        """Composite score combines ELO and calibration."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(
            elo=1500.0,  # Normalized: (1500-1000)/500 = 1.0
            calibration_score=0.8,
        )

        helper = MetricsHelper(elo_system=elo)
        score = helper.get_composite_judge_score("agent1")

        # (1.0 * 0.7) + (0.8 * 0.3) = 0.7 + 0.24 = 0.94
        assert score == pytest.approx(0.94)

    def test_get_composite_judge_score_low_elo(self):
        """Composite score floors ELO at 0."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(
            elo=800.0,  # Normalized: (800-1000)/500 = -0.4 -> 0
            calibration_score=0.5,
        )

        helper = MetricsHelper(elo_system=elo)
        score = helper.get_composite_judge_score("agent1")

        # (0.0 * 0.7) + (0.5 * 0.3) = 0 + 0.15 = 0.15
        assert score == pytest.approx(0.15)

    def test_get_composite_judge_score_error(self):
        """Composite score is 0.0 on error."""
        elo = MagicMock()
        elo.get_rating.side_effect = Exception("Rating error")

        helper = MetricsHelper(elo_system=elo)
        score = helper.get_composite_judge_score("agent1")

        assert score == 0.0

    def test_extract_domain_security(self):
        """Extract security domain from task."""
        helper = MetricsHelper()

        domain = helper.extract_domain("Fix the authentication vulnerability")

        assert domain == "security"

    def test_extract_domain_performance(self):
        """Extract performance domain from task."""
        helper = MetricsHelper()

        domain = helper.extract_domain("Optimize the database queries for speed")

        assert domain == "performance"

    def test_extract_domain_testing(self):
        """Extract testing domain from task."""
        helper = MetricsHelper()

        domain = helper.extract_domain("Increase test coverage")

        assert domain == "testing"

    def test_extract_domain_architecture(self):
        """Extract architecture domain from task."""
        helper = MetricsHelper()

        domain = helper.extract_domain("Design a new microservices architecture")

        assert domain == "architecture"

    def test_extract_domain_api(self):
        """Extract API domain from task."""
        helper = MetricsHelper()

        domain = helper.extract_domain("Create a new REST endpoint")

        assert domain == "api"

    def test_extract_domain_database(self):
        """Extract database domain from task."""
        helper = MetricsHelper()

        domain = helper.extract_domain("Write a SQL query for user data")

        assert domain == "database"

    def test_extract_domain_frontend(self):
        """Extract frontend domain from task."""
        helper = MetricsHelper()

        domain = helper.extract_domain("Build a React component for the UI")

        assert domain == "frontend"

    def test_extract_domain_general(self):
        """Extract general domain when no match."""
        helper = MetricsHelper()

        domain = helper.extract_domain("Implement a new feature")

        assert domain == "general"

    def test_extract_domain_caching(self):
        """Domain extraction is cached."""
        helper = MetricsHelper()
        task = "Fix the security issue"

        domain1 = helper.extract_domain(task)
        domain2 = helper.extract_domain(task)

        assert domain1 == domain2
        assert task in helper._domain_cache

    def test_clear_cache(self):
        """clear_cache empties domain cache."""
        helper = MetricsHelper()
        helper.extract_domain("test task")

        assert len(helper._domain_cache) == 1

        helper.clear_cache()

        assert helper._domain_cache == {}

    def test_get_agent_rating_no_elo(self):
        """get_agent_rating returns None without ELO."""
        helper = MetricsHelper()

        rating = helper.get_agent_rating("agent1")

        assert rating is None

    def test_get_agent_rating(self):
        """get_agent_rating returns rating from ELO."""
        elo = MagicMock()
        mock_rating = MockRating()
        elo.get_rating.return_value = mock_rating

        helper = MetricsHelper(elo_system=elo)
        rating = helper.get_agent_rating("agent1")

        assert rating is mock_rating

    def test_get_ratings_batch(self):
        """get_ratings_batch returns multiple ratings."""
        elo = MagicMock()
        elo.get_ratings_batch.return_value = {
            "agent1": MockRating(),
            "agent2": MockRating(),
        }

        helper = MetricsHelper(elo_system=elo)
        ratings = helper.get_ratings_batch(["agent1", "agent2"])

        assert "agent1" in ratings
        assert "agent2" in ratings

    def test_get_ratings_batch_no_elo(self):
        """get_ratings_batch returns empty dict without ELO."""
        helper = MetricsHelper()

        ratings = helper.get_ratings_batch(["agent1"])

        assert ratings == {}


class TestBuildRelationshipUpdates:
    """Tests for build_relationship_updates function."""

    def test_two_participants(self):
        """Two participants create one update."""
        participants = ["agent1", "agent2"]
        vote_choices = {"agent1": "A", "agent2": "A"}

        updates = build_relationship_updates(participants, vote_choices)

        assert len(updates) == 1
        assert updates[0]["agent_a"] == "agent1"
        assert updates[0]["agent_b"] == "agent2"
        assert updates[0]["debate_increment"] == 1
        assert updates[0]["agreement_increment"] == 1  # Both voted A

    def test_three_participants(self):
        """Three participants create three updates."""
        participants = ["agent1", "agent2", "agent3"]
        vote_choices = {}

        updates = build_relationship_updates(participants, vote_choices)

        assert len(updates) == 3  # 3 pairs

    def test_agreement_tracking(self):
        """Agreement is tracked when votes match."""
        participants = ["agent1", "agent2", "agent3"]
        vote_choices = {
            "agent1": "A",
            "agent2": "A",
            "agent3": "B",
        }

        updates = build_relationship_updates(participants, vote_choices)

        # agent1-agent2: agreed (both A)
        # agent1-agent3: disagreed
        # agent2-agent3: disagreed
        agreements = [u["agreement_increment"] for u in updates]
        assert agreements.count(1) == 1
        assert agreements.count(0) == 2

    def test_winner_tracking(self):
        """Winner is tracked correctly."""
        participants = ["agent1", "agent2"]
        vote_choices = {}

        updates = build_relationship_updates(
            participants, vote_choices, winner="agent1"
        )

        assert updates[0]["a_win"] == 1
        assert updates[0]["b_win"] == 0

    def test_no_winner(self):
        """No winner sets both wins to 0."""
        participants = ["agent1", "agent2"]
        vote_choices = {}

        updates = build_relationship_updates(participants, vote_choices)

        assert updates[0]["a_win"] == 0
        assert updates[0]["b_win"] == 0

    def test_empty_participants(self):
        """Empty participants returns empty list."""
        updates = build_relationship_updates([], {})

        assert updates == []

    def test_single_participant(self):
        """Single participant returns empty list."""
        updates = build_relationship_updates(["agent1"], {})

        assert updates == []
