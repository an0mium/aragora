"""Tests for the EloAdapter."""

import pytest
from unittest.mock import Mock
from datetime import datetime

from aragora.knowledge.mound.adapters.elo_adapter import (
    EloAdapter,
    RatingSearchResult,
)
from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource


class TestRatingSearchResult:
    """Tests for RatingSearchResult dataclass."""

    def test_basic_creation(self):
        """Create a basic search result."""
        result = RatingSearchResult(
            rating={"id": "el_1", "elo": 1500},
            relevance_score=0.8,
        )
        assert result.rating["id"] == "el_1"
        assert result.relevance_score == 0.8


class TestEloAdapterInit:
    """Tests for EloAdapter initialization."""

    def test_init_without_system(self):
        """Initialize without ELO system."""
        adapter = EloAdapter()
        assert adapter.elo_system is None

    def test_init_with_system(self):
        """Initialize with ELO system."""
        mock_elo = Mock()
        adapter = EloAdapter(elo_system=mock_elo)
        assert adapter.elo_system is mock_elo

    def test_set_elo_system(self):
        """Set ELO system after init."""
        adapter = EloAdapter()
        mock_elo = Mock()
        adapter.set_elo_system(mock_elo)
        assert adapter.elo_system is mock_elo

    def test_constants(self):
        """Verify adapter constants."""
        assert EloAdapter.ID_PREFIX == "el_"
        assert EloAdapter.MIN_DEBATES_FOR_RELATIONSHIP == 5


class TestEloAdapterStoreRating:
    """Tests for store_rating method."""

    def test_store_rating(self):
        """Store an agent rating."""
        adapter = EloAdapter()

        mock_rating = Mock()
        mock_rating.agent_name = "claude"
        mock_rating.elo = 1650.5
        mock_rating.domain_elos = {"legal": 1700, "tech": 1600}
        mock_rating.wins = 10
        mock_rating.losses = 5
        mock_rating.draws = 2
        mock_rating.debates_count = 17
        mock_rating.win_rate = 0.59
        mock_rating.games_played = 17
        mock_rating.critiques_accepted = 8
        mock_rating.critiques_total = 12
        mock_rating.critique_acceptance_rate = 0.67
        mock_rating.calibration_correct = 15
        mock_rating.calibration_total = 20
        mock_rating.calibration_accuracy = 0.75
        mock_rating.updated_at = "2024-01-01T00:00:00Z"

        rating_id = adapter.store_rating(mock_rating, debate_id="debate_1")

        assert rating_id is not None
        assert rating_id.startswith("el_claude_")
        assert "claude" in adapter._agent_ratings

    def test_store_updates_domain_index(self):
        """Verify domain index is updated."""
        adapter = EloAdapter()

        mock_rating = Mock()
        mock_rating.agent_name = "claude"
        mock_rating.elo = 1500
        mock_rating.domain_elos = {"legal": 1600}
        mock_rating.wins = 5
        mock_rating.losses = 3
        mock_rating.draws = 1
        mock_rating.debates_count = 9
        mock_rating.win_rate = 0.56
        mock_rating.games_played = 9
        mock_rating.critiques_accepted = 5
        mock_rating.critiques_total = 8
        mock_rating.critique_acceptance_rate = 0.63
        mock_rating.calibration_correct = 7
        mock_rating.calibration_total = 10
        mock_rating.calibration_accuracy = 0.7
        mock_rating.updated_at = "2024-01-01T00:00:00Z"

        adapter.store_rating(mock_rating)

        assert "legal" in adapter._domain_ratings


class TestEloAdapterStoreMatch:
    """Tests for store_match method."""

    def test_store_match(self):
        """Store a match result."""
        adapter = EloAdapter()

        mock_match = Mock()
        mock_match.debate_id = "debate_123"
        mock_match.winner = "claude"
        mock_match.participants = ["claude", "gpt4"]
        mock_match.domain = "legal"
        mock_match.scores = {"claude": 0.7, "gpt4": 0.3}
        mock_match.created_at = "2024-01-01T00:00:00Z"

        match_id = adapter.store_match(mock_match)

        assert match_id is not None
        assert match_id.startswith("el_match_")
        assert "claude" in adapter._agent_matches
        assert "gpt4" in adapter._agent_matches


class TestEloAdapterStoreCalibration:
    """Tests for store_calibration method."""

    def test_store_calibration(self):
        """Store a calibration prediction."""
        adapter = EloAdapter()

        cal_id = adapter.store_calibration(
            agent_name="claude",
            debate_id="debate_123",
            predicted_winner="gpt4",
            predicted_confidence=0.7,
            actual_winner="gpt4",
            was_correct=True,
            brier_score=0.09,
        )

        assert cal_id is not None
        assert cal_id.startswith("el_cal_")


class TestEloAdapterStoreRelationship:
    """Tests for store_relationship method."""

    def test_store_relationship_above_threshold(self):
        """Store relationship with sufficient debates."""
        adapter = EloAdapter()

        mock_metrics = Mock()
        mock_metrics.agent_a = "claude"
        mock_metrics.agent_b = "gpt4"
        mock_metrics.debates_together = 10
        mock_metrics.a_wins_vs_b = 6
        mock_metrics.b_wins_vs_a = 3
        mock_metrics.draws = 1
        mock_metrics.avg_elo_diff = 50.5
        mock_metrics.synergy_score = 0.7

        rel_id = adapter.store_relationship(mock_metrics)

        assert rel_id is not None
        assert rel_id.startswith("el_rel_")

    def test_skip_relationship_below_threshold(self):
        """Don't store relationships with few debates."""
        adapter = EloAdapter()

        mock_metrics = Mock()
        mock_metrics.agent_a = "claude"
        mock_metrics.agent_b = "gpt4"
        mock_metrics.debates_together = 3  # Below 5

        rel_id = adapter.store_relationship(mock_metrics)
        assert rel_id is None


class TestEloAdapterGetAgentSkillHistory:
    """Tests for get_agent_skill_history method."""

    def test_get_skill_history(self):
        """Get skill progression for an agent."""
        adapter = EloAdapter()

        adapter._ratings["el_1"] = {
            "id": "el_1",
            "agent_name": "claude",
            "elo": 1500,
            "created_at": "2024-01-01T00:00:00Z",
        }
        adapter._ratings["el_2"] = {
            "id": "el_2",
            "agent_name": "claude",
            "elo": 1550,
            "created_at": "2024-01-02T00:00:00Z",
        }
        adapter._agent_ratings["claude"] = ["el_1", "el_2"]

        results = adapter.get_agent_skill_history("claude")

        assert len(results) == 2
        # Should be sorted newest first
        assert results[0]["elo"] == 1550


class TestEloAdapterGetDomainExpertise:
    """Tests for get_domain_expertise method."""

    def test_get_domain_experts(self):
        """Get agents with domain expertise."""
        adapter = EloAdapter()

        adapter._ratings["el_1"] = {
            "id": "el_1",
            "agent_name": "claude",
            "domain_elos": {"legal": 1700},
            "created_at": "2024-01-01T00:00:00Z",
        }
        adapter._ratings["el_2"] = {
            "id": "el_2",
            "agent_name": "gpt4",
            "domain_elos": {"legal": 1500},
            "created_at": "2024-01-01T00:00:00Z",
        }
        adapter._domain_ratings["legal"] = ["el_1", "el_2"]

        results = adapter.get_domain_expertise("legal", min_elo=1600)

        assert len(results) == 1
        assert results[0]["agent_name"] == "claude"


class TestEloAdapterGetRelationship:
    """Tests for get_relationship method."""

    def test_get_relationship_either_order(self):
        """Get relationship regardless of agent order."""
        adapter = EloAdapter()

        adapter._relationships["el_rel_claude_gpt4"] = {
            "id": "el_rel_claude_gpt4",
            "agent_a": "claude",
            "agent_b": "gpt4",
        }

        # Should work either way
        result1 = adapter.get_relationship("claude", "gpt4")
        result2 = adapter.get_relationship("gpt4", "claude")

        assert result1 is not None
        assert result2 is not None


class TestEloAdapterToKnowledgeItem:
    """Tests for to_knowledge_item method."""

    def test_convert_high_games_rating(self):
        """Convert rating with many games."""
        adapter = EloAdapter()

        rating = {
            "id": "el_claude_2024",
            "agent_name": "claude",
            "elo": 1650,
            "wins": 30,
            "losses": 15,
            "draws": 5,
            "games_played": 50,
            "win_rate": 0.6,
            "domain_elos": {"legal": 1700},
            "calibration_accuracy": 0.75,
            "created_at": "2024-01-01T00:00:00Z",
        }

        item = adapter.to_knowledge_item(rating)

        assert item.id == "el_claude_2024"
        assert "claude" in item.content
        assert "1650" in item.content
        assert item.source == KnowledgeSource.ELO
        assert item.confidence == ConfidenceLevel.VERIFIED  # 50+ games

    def test_convert_low_games_rating(self):
        """Convert rating with few games."""
        adapter = EloAdapter()

        rating = {
            "id": "el_new_agent",
            "agent_name": "new_agent",
            "elo": 1000,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "games_played": 2,
            "created_at": "2024-01-01T00:00:00Z",
        }

        item = adapter.to_knowledge_item(rating)

        assert item.confidence == ConfidenceLevel.UNVERIFIED  # <3 games

    def test_importance_normalized(self):
        """Verify importance is normalized from ELO."""
        adapter = EloAdapter()

        # High ELO
        high_rating = {
            "id": "el_1",
            "agent_name": "expert",
            "elo": 2000,
            "games_played": 100,
            "created_at": "2024-01-01T00:00:00Z",
        }
        high_item = adapter.to_knowledge_item(high_rating)
        assert high_item.importance == 1.0  # Capped at 1.0

        # Low ELO
        low_rating = {
            "id": "el_2",
            "agent_name": "newbie",
            "elo": 800,
            "games_played": 5,
            "created_at": "2024-01-01T00:00:00Z",
        }
        low_item = adapter.to_knowledge_item(low_rating)
        assert low_item.importance == 0.2  # (800-500)/1500


class TestEloAdapterGetStats:
    """Tests for get_stats method."""

    def test_get_stats(self):
        """Get adapter statistics."""
        adapter = EloAdapter()

        adapter._ratings["el_1"] = {"agent_name": "claude"}
        adapter._matches["el_m1"] = {}
        adapter._calibrations["el_c1"] = {}
        adapter._relationships["el_r1"] = {}
        adapter._agent_ratings["claude"] = ["el_1"]
        adapter._domain_ratings["legal"] = ["el_1"]

        stats = adapter.get_stats()

        assert stats["total_ratings"] == 1
        assert stats["total_matches"] == 1
        assert stats["total_calibrations"] == 1
        assert stats["total_relationships"] == 1
        assert stats["agents_tracked"] == 1
        assert stats["domains_tracked"] == 1
