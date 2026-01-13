"""
Tests for FlipDetector batch methods.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from aragora.insights.flip_detector import FlipDetector, AgentConsistencyScore


class TestFlipDetectorBatch:
    """Tests for batch consistency fetching."""

    @pytest.fixture
    def detector(self, tmp_path):
        """Create a FlipDetector with test database."""
        db_path = tmp_path / "test_positions.db"
        detector = FlipDetector(str(db_path))
        return detector

    @pytest.fixture
    def populated_detector(self, detector):
        """Create a FlipDetector with test data."""
        with sqlite3.connect(detector.db_path) as conn:
            # Add some positions (round_num is required)
            conn.execute(
                "INSERT INTO positions (id, agent_name, claim, confidence, debate_id, round_num, domain) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("pos1", "agent_a", "Claim 1", 0.9, "debate1", 1, "science"),
            )
            conn.execute(
                "INSERT INTO positions (id, agent_name, claim, confidence, debate_id, round_num, domain) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("pos2", "agent_a", "Claim 2", 0.8, "debate2", 1, "science"),
            )
            conn.execute(
                "INSERT INTO positions (id, agent_name, claim, confidence, debate_id, round_num, domain) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("pos3", "agent_b", "Claim 3", 0.7, "debate1", 1, "tech"),
            )

            # Add some flips
            conn.execute(
                """INSERT INTO detected_flips
                   (id, agent_name, original_claim, new_claim, original_confidence, new_confidence,
                    original_debate_id, new_debate_id, original_position_id, new_position_id,
                    similarity_score, flip_type, domain)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "flip1",
                    "agent_a",
                    "Claim 1",
                    "Claim 1 changed",
                    0.9,
                    0.3,
                    "debate1",
                    "debate2",
                    "pos1",
                    "pos2",
                    0.8,
                    "contradiction",
                    "science",
                ),
            )
            conn.commit()
        return detector

    def test_batch_empty_list(self, detector):
        """Empty list should return empty dict."""
        result = detector.get_agents_consistency_batch([])
        assert result == {}

    def test_batch_single_agent(self, populated_detector):
        """Should return consistency for single agent."""
        result = populated_detector.get_agents_consistency_batch(["agent_a"])

        assert "agent_a" in result
        score = result["agent_a"]
        assert isinstance(score, AgentConsistencyScore)
        assert score.total_positions == 2
        assert score.total_flips == 1
        assert score.contradictions == 1

    def test_batch_multiple_agents(self, populated_detector):
        """Should return consistency for multiple agents."""
        result = populated_detector.get_agents_consistency_batch(["agent_a", "agent_b"])

        assert len(result) == 2
        assert "agent_a" in result
        assert "agent_b" in result

        # agent_a has flips
        assert result["agent_a"].total_flips == 1
        # agent_b has no flips
        assert result["agent_b"].total_flips == 0
        assert result["agent_b"].total_positions == 1

    def test_batch_nonexistent_agent(self, populated_detector):
        """Nonexistent agents should get default values."""
        result = populated_detector.get_agents_consistency_batch(["nonexistent"])

        assert "nonexistent" in result
        score = result["nonexistent"]
        assert score.total_positions == 0
        assert score.total_flips == 0

    def test_batch_matches_individual(self, populated_detector):
        """Batch results should match individual queries."""
        batch_result = populated_detector.get_agents_consistency_batch(["agent_a"])
        individual_result = populated_detector.get_agent_consistency("agent_a")

        batch_score = batch_result["agent_a"]
        assert batch_score.total_positions == individual_result.total_positions
        assert batch_score.total_flips == individual_result.total_flips
        assert batch_score.contradictions == individual_result.contradictions

    def test_batch_domains_collected(self, populated_detector):
        """Domains with flips should be collected."""
        result = populated_detector.get_agents_consistency_batch(["agent_a"])

        assert "science" in result["agent_a"].domains_with_flips
