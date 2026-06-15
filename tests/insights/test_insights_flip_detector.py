"""
Tests for aragora.insights.flip_detector module.

Tests FlipDetector, FlipEvent, AgentConsistencyScore, and helper functions.
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path

from aragora.insights.flip_detector import (
    FlipEvent,
    AgentConsistencyScore,
    FlipDetector,
    format_flip_for_ui,
    format_consistency_for_ui,
)


# ============================================================================
# FlipEvent Tests
# ============================================================================


class TestFlipEvent:
    """Tests for FlipEvent dataclass."""

    def test_to_dict(self):
        """Should serialize all fields."""
        flip = FlipEvent(
            id="flip-001",
            agent_name="claude",
            original_claim="X is good",
            new_claim="X is bad",
            original_confidence=0.8,
            new_confidence=0.7,
            original_debate_id="d1",
            new_debate_id="d2",
            original_position_id="p1",
            new_position_id="p2",
            similarity_score=0.3,
            flip_type="contradiction",
            domain="security",
        )
        d = flip.to_dict()

        assert d["id"] == "flip-001"
        assert d["agent_name"] == "claude"
        assert d["flip_type"] == "contradiction"
        assert d["domain"] == "security"
        assert d["similarity_score"] == 0.3

    def test_defaults(self):
        """Should have sensible defaults."""
        flip = FlipEvent(
            id="f1",
            agent_name="a",
            original_claim="c1",
            new_claim="c2",
            original_confidence=0.5,
            new_confidence=0.5,
            original_debate_id="d1",
            new_debate_id="d2",
            original_position_id="p1",
            new_position_id="p2",
            similarity_score=0.5,
            flip_type="refinement",
        )

        assert flip.domain is None
        assert flip.detected_at is not None


# ============================================================================
# AgentConsistencyScore Tests
# ============================================================================


class TestAgentConsistencyScore:
    """Tests for AgentConsistencyScore dataclass."""

    def test_consistency_score_perfect(self):
        """Should return 1.0 for perfectly consistent agent."""
        score = AgentConsistencyScore(
            agent_name="consistent_agent",
            total_positions=100,
            total_flips=0,
        )

        assert score.consistency_score == 1.0

    def test_consistency_score_with_contradictions(self):
        """Should penalize contradictions heavily."""
        score = AgentConsistencyScore(
            agent_name="a",
            total_positions=10,
            contradictions=2,
        )

        # 2 contradictions / 10 positions = 0.2 reduction
        assert score.consistency_score < 1.0
        assert score.consistency_score >= 0.0

    def test_consistency_score_with_refinements(self):
        """Should penalize refinements lightly."""
        score = AgentConsistencyScore(
            agent_name="a",
            total_positions=10,
            refinements=5,
        )

        # Refinements have 0.1 weight
        assert score.consistency_score > 0.9

    def test_consistency_score_no_positions(self):
        """Should return 1.0 when no positions."""
        score = AgentConsistencyScore(agent_name="new_agent")

        assert score.consistency_score == 1.0

    def test_flip_rate(self):
        """Should calculate flip rate correctly."""
        score = AgentConsistencyScore(
            agent_name="a",
            total_positions=20,
            total_flips=5,
        )

        assert score.flip_rate == 0.25

    def test_flip_rate_no_positions(self):
        """Should return 0 flip rate when no positions."""
        score = AgentConsistencyScore(agent_name="a")

        assert score.flip_rate == 0.0

    def test_to_dict(self):
        """Should serialize all fields including computed."""
        score = AgentConsistencyScore(
            agent_name="claude",
            total_positions=50,
            total_flips=5,
            contradictions=2,
            refinements=3,
            domains_with_flips=["security", "performance"],
        )
        d = score.to_dict()

        assert d["agent_name"] == "claude"
        assert d["total_positions"] == 50
        assert d["consistency_score"] > 0
        assert d["flip_rate"] == 0.1
        assert d["domains_with_flips"] == ["security", "performance"]


# ============================================================================
# FlipDetector Tests
# ============================================================================


class TestFlipDetector:
    """Tests for FlipDetector class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def detector(self, temp_db):
        """Create a FlipDetector with temp database."""
        return FlipDetector(db_path=temp_db)

    def test_init_creates_tables(self, detector, temp_db):
        """Should create required tables on init."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Check for detected_flips table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='detected_flips'"
        )
        assert cursor.fetchone() is not None

        # Check for positions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions'")
        assert cursor.fetchone() is not None

        conn.close()

    def test_compute_similarity_identical(self, detector):
        """Should return 1.0 for identical strings."""
        sim = detector._compute_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_compute_similarity_different(self, detector):
        """Should return low score for different strings."""
        sim = detector._compute_similarity("hello world", "xyz abc")
        assert sim < 0.5

    def test_compute_similarity_similar(self, detector):
        """Should return high score for similar strings."""
        sim = detector._compute_similarity("hello world", "hello there world")
        assert sim > 0.5

    def test_compute_similarity_case_insensitive(self, detector):
        """Should be case insensitive."""
        sim = detector._compute_similarity("Hello World", "hello world")
        assert sim == 1.0

    def test_classify_flip_type_contradiction(self, detector):
        """Should detect contradictions."""
        flip_type = detector._classify_flip_type(
            "X is good",
            "X is bad",
            0.8,
            0.7,
        )
        assert flip_type == "contradiction"

    def test_classify_flip_type_retraction(self, detector):
        """Should detect retractions."""
        flip_type = detector._classify_flip_type(
            "X is true",
            "I was wrong about X",
            0.9,
            0.3,
        )
        assert flip_type == "retraction"

    def test_classify_flip_type_qualification(self, detector):
        """Should detect qualifications."""
        flip_type = detector._classify_flip_type(
            "X is always true",
            "X is sometimes true in some cases",
            0.9,
            0.7,
        )
        assert flip_type == "qualification"

    def test_classify_flip_type_refinement(self, detector):
        """Should classify high similarity as refinement."""
        flip_type = detector._classify_flip_type(
            "The approach is mostly correct and efficient",
            "The approach is mostly correct and efficient overall",
            0.8,
            0.8,
        )
        assert flip_type == "refinement"

    def test_store_and_retrieve_flip(self, detector, temp_db):
        """Should store and retrieve flips."""
        flip = FlipEvent(
            id="flip-test",
            agent_name="claude",
            original_claim="A",
            new_claim="B",
            original_confidence=0.8,
            new_confidence=0.6,
            original_debate_id="d1",
            new_debate_id="d2",
            original_position_id="p1",
            new_position_id="p2",
            similarity_score=0.3,
            flip_type="contradiction",
            domain="test",
        )

        detector._store_flip(flip)
        recent = detector.get_recent_flips(limit=10)

        assert len(recent) == 1
        assert recent[0].id == "flip-test"
        assert recent[0].agent_name == "claude"

    def test_get_agent_consistency_empty(self, detector):
        """Should return default consistency for agent with no data."""
        score = detector.get_agent_consistency("unknown_agent")

        assert score.agent_name == "unknown_agent"
        assert score.total_positions == 0
        assert score.consistency_score == 1.0

    def test_get_agent_consistency_with_data(self, detector, temp_db):
        """Should compute consistency from stored data."""
        # Insert some positions
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO positions (id, agent_name, claim, confidence, debate_id, round_num) VALUES (?, ?, ?, ?, ?, ?)",
            ("p1", "claude", "claim1", 0.8, "d1", 1),
        )
        conn.execute(
            "INSERT INTO positions (id, agent_name, claim, confidence, debate_id, round_num) VALUES (?, ?, ?, ?, ?, ?)",
            ("p2", "claude", "claim2", 0.7, "d2", 1),
        )
        conn.commit()
        conn.close()

        score = detector.get_agent_consistency("claude")

        assert score.agent_name == "claude"
        assert score.total_positions == 2

    def test_get_agents_consistency_batch_empty(self, detector):
        """Should return empty dict for empty input."""
        result = detector.get_agents_consistency_batch([])
        assert result == {}

    def test_get_agents_consistency_batch_with_agents(self, detector, temp_db):
        """Should return consistency for multiple agents."""
        # Insert positions for multiple agents
        conn = sqlite3.connect(temp_db)
        for agent in ["claude", "gpt4"]:
            conn.execute(
                "INSERT INTO positions (id, agent_name, claim, confidence, debate_id, round_num) VALUES (?, ?, ?, ?, ?, ?)",
                (f"p_{agent}", agent, "claim", 0.8, "d1", 1),
            )
        conn.commit()
        conn.close()

        result = detector.get_agents_consistency_batch(["claude", "gpt4", "unknown"])

        assert "claude" in result
        assert "gpt4" in result
        assert "unknown" in result
        assert result["claude"].total_positions == 1
        assert result["unknown"].total_positions == 0

    def test_get_recent_flips_limit(self, detector, temp_db):
        """Should respect limit parameter."""
        # Insert multiple flips
        conn = sqlite3.connect(temp_db)
        for i in range(5):
            conn.execute(
                """INSERT INTO detected_flips
                   (id, agent_name, original_claim, new_claim, original_confidence,
                    new_confidence, similarity_score, flip_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (f"flip-{i}", "agent", "A", "B", 0.8, 0.6, 0.3, "refinement"),
            )
        conn.commit()
        conn.close()

        recent = detector.get_recent_flips(limit=3)

        assert len(recent) == 3

    def test_get_flip_summary_empty(self, detector):
        """Should return zeroes for empty database."""
        summary = detector.get_flip_summary()

        assert summary["total_flips"] == 0
        assert summary["by_type"] == {}
        assert summary["by_agent"] == {}
        assert summary["recent_24h"] == 0

    def test_get_flip_summary_with_data(self, detector, temp_db):
        """Should compute summary from stored data."""
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """INSERT INTO detected_flips
               (id, agent_name, original_claim, new_claim, original_confidence,
                new_confidence, similarity_score, flip_type, detected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            ("f1", "claude", "A", "B", 0.8, 0.6, 0.3, "contradiction"),
        )
        conn.execute(
            """INSERT INTO detected_flips
               (id, agent_name, original_claim, new_claim, original_confidence,
                new_confidence, similarity_score, flip_type, detected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            ("f2", "claude", "C", "D", 0.7, 0.6, 0.5, "refinement"),
        )
        conn.commit()
        conn.close()

        summary = detector.get_flip_summary()

        assert summary["total_flips"] == 2
        assert summary["by_type"]["contradiction"] == 1
        assert summary["by_type"]["refinement"] == 1
        assert summary["by_agent"]["claude"] == 2
        assert summary["recent_24h"] == 2


# ============================================================================
# UI Formatting Tests
# ============================================================================


class TestUIFormatting:
    """Tests for UI formatting functions."""

    def test_format_flip_for_ui(self):
        """Should format flip event for display."""
        flip = FlipEvent(
            id="f1",
            agent_name="claude",
            original_claim="This is a long original claim that should be truncated " * 3,
            new_claim="New short claim",
            original_confidence=0.85,
            new_confidence=0.65,
            original_debate_id="d1",
            new_debate_id="d2",
            original_position_id="p1",
            new_position_id="p2",
            similarity_score=0.42,
            flip_type="contradiction",
            domain="security",
        )

        formatted = format_flip_for_ui(flip)

        assert formatted["id"] == "f1"
        assert formatted["agent"] == "claude"
        assert formatted["type"] == "contradiction"
        assert formatted["type_emoji"] == "\U0001f504"  # 🔄
        assert formatted["before"]["confidence"] == "85%"
        assert formatted["after"]["confidence"] == "65%"
        assert formatted["similarity"] == "42%"
        assert formatted["domain"] == "security"
        # Long claim should be truncated
        assert len(formatted["before"]["claim"]) <= 103  # 100 + "..."

    def test_format_flip_emojis(self):
        """Should use correct emojis for each type."""
        for flip_type, emoji in [
            ("contradiction", "\U0001f504"),  # 🔄
            ("retraction", "↩️"),
            ("qualification", "\U0001f4dd"),  # 📝
            ("refinement", "\U0001f527"),  # 🔧
        ]:
            flip = FlipEvent(
                id="f",
                agent_name="a",
                original_claim="c1",
                new_claim="c2",
                original_confidence=0.5,
                new_confidence=0.5,
                original_debate_id="d1",
                new_debate_id="d2",
                original_position_id="p1",
                new_position_id="p2",
                similarity_score=0.5,
                flip_type=flip_type,
            )
            formatted = format_flip_for_ui(flip)
            assert formatted["type_emoji"] == emoji

    def test_format_consistency_for_ui_high(self):
        """Should format high consistency correctly."""
        score = AgentConsistencyScore(
            agent_name="reliable_agent",
            total_positions=100,
            total_flips=5,
            contradictions=1,
            refinements=4,
        )

        formatted = format_consistency_for_ui(score)

        assert formatted["agent"] == "reliable_agent"
        assert formatted["consistency_class"] == "high"
        assert formatted["total_positions"] == 100
        assert "%" in formatted["flip_rate"]

    def test_format_consistency_for_ui_low(self):
        """Should format low consistency correctly."""
        score = AgentConsistencyScore(
            agent_name="flip_flopper",
            total_positions=10,
            total_flips=8,
            contradictions=6,
            retractions=2,
        )

        formatted = format_consistency_for_ui(score)

        assert formatted["consistency_class"] == "low"

    def test_format_consistency_breakdown(self):
        """Should include flip type breakdown."""
        score = AgentConsistencyScore(
            agent_name="a",
            contradictions=2,
            retractions=3,
            qualifications=1,
            refinements=4,
        )

        formatted = format_consistency_for_ui(score)

        assert formatted["breakdown"]["contradictions"] == 2
        assert formatted["breakdown"]["retractions"] == 3
        assert formatted["breakdown"]["qualifications"] == 1
        assert formatted["breakdown"]["refinements"] == 4

    def test_format_consistency_limits_domains(self):
        """Should limit domains list to 3."""
        score = AgentConsistencyScore(
            agent_name="a",
            domains_with_flips=["security", "performance", "clarity", "testing", "architecture"],
        )

        formatted = format_consistency_for_ui(score)

        assert len(formatted["problem_domains"]) == 3
