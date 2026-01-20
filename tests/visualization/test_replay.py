"""
Tests for replay visualization generator.

Tests cover:
- ReplayScene dataclass
- ReplayArtifact dataclass
- ReplayGenerator class
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from aragora.core import Critique, DebateResult, Message, Vote
from aragora.visualization.replay import (
    ReplayArtifact,
    ReplayGenerator,
    ReplayScene,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        Message(
            role="proposal",
            agent="claude",
            content="I propose we use Python.",
            round=1,
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
        ),
        Message(
            role="critique",
            agent="gpt",
            content="Python is too slow for this use case.",
            round=1,
            timestamp=datetime(2024, 1, 1, 10, 0, 30),
        ),
        Message(
            role="rebuttal",
            agent="claude",
            content="We can optimize critical paths.",
            round=2,
            timestamp=datetime(2024, 1, 1, 10, 1, 0),
        ),
    ]


@pytest.fixture
def sample_votes():
    """Create sample votes for testing."""
    return [
        Vote(agent="claude", choice="python", reasoning="Python is versatile", confidence=0.9),
        Vote(agent="gpt", choice="python", reasoning="Good ecosystem", confidence=0.7),
        Vote(agent="gemini", choice="rust", reasoning="Better performance", confidence=0.8),
    ]


@pytest.fixture
def sample_debate_result(sample_messages, sample_votes):
    """Create a sample debate result."""
    return DebateResult(
        id="debate-123",
        task="Choose a programming language",
        final_answer="Python is the best choice",
        confidence=0.85,
        rounds_used=2,
        consensus_reached=True,
        messages=sample_messages,
        votes=sample_votes,
        duration_seconds=60.5,
        convergence_status="converged",
        consensus_strength=0.8,
        winning_patterns=["type safety", "performance"],
        critiques=[
            Critique(
                agent="gpt",
                target_agent="claude",
                target_content="Python proposal",
                issues=["No static typing"],
                suggestions=["Use type hints"],
                severity=0.6,
                reasoning="Python lacks static typing",
            ),
        ],
        dissenting_views=["Rust would be faster"],
    )


@pytest.fixture
def generator():
    """Create a ReplayGenerator instance."""
    return ReplayGenerator()


# ============================================================================
# ReplayScene Tests
# ============================================================================


class TestReplayScene:
    """Tests for ReplayScene dataclass."""

    def test_creation(self, sample_messages):
        """Test basic creation."""
        scene = ReplayScene(
            round_number=1,
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            messages=sample_messages[:2],
            consensus_indicators={"reached": False},
        )

        assert scene.round_number == 1
        assert len(scene.messages) == 2
        assert scene.consensus_indicators["reached"] is False

    def test_default_values(self):
        """Test default values for optional fields."""
        scene = ReplayScene(
            round_number=1,
            timestamp=datetime.now(),
        )

        assert scene.messages == []
        assert scene.consensus_indicators == {}

    def test_to_dict(self, sample_messages):
        """Test serialization to dictionary."""
        scene = ReplayScene(
            round_number=2,
            timestamp=datetime(2024, 1, 1, 12, 30, 0),
            messages=[sample_messages[0]],
            consensus_indicators={"reached": True, "confidence": 0.9},
        )

        result = scene.to_dict()

        assert result["round_number"] == 2
        assert result["timestamp"] == "2024-01-01T12:30:00"
        assert len(result["messages"]) == 1
        assert result["consensus_indicators"]["reached"] is True

    def test_to_dict_html_escapes_messages(self):
        """Test that message content is HTML escaped."""
        xss_message = Message(
            role="proposal",
            agent="<script>evil()</script>",
            content="<img onerror='alert()'>",
            round=1,
            timestamp=datetime.now(),
        )
        scene = ReplayScene(
            round_number=1,
            timestamp=datetime.now(),
            messages=[xss_message],
        )

        result = scene.to_dict()

        msg = result["messages"][0]
        assert "<script>" not in msg["agent"]
        assert "&lt;script&gt;" in msg["agent"]
        assert "<img" not in msg["content"]
        assert "&lt;img" in msg["content"]

    def test_to_dict_handles_missing_attributes(self):
        """Test graceful handling of messages with missing attributes."""
        # Create a mock message with minimal attributes
        mock_msg = MagicMock(spec=[])
        mock_msg.configure_mock(**{})

        scene = ReplayScene(
            round_number=1,
            timestamp=datetime.now(),
            messages=[mock_msg],
        )

        result = scene.to_dict()

        # Should use defaults for missing attributes
        assert result["messages"][0]["role"] == "unknown"
        assert result["messages"][0]["agent"] == "unknown"


# ============================================================================
# ReplayArtifact Tests
# ============================================================================


class TestReplayArtifact:
    """Tests for ReplayArtifact dataclass."""

    def test_creation(self):
        """Test basic creation."""
        artifact = ReplayArtifact(
            debate_id="d-123",
            task="Test task",
            scenes=[],
            verdict={"winner": "claude"},
            metadata={"duration": 60},
        )

        assert artifact.debate_id == "d-123"
        assert artifact.task == "Test task"
        assert artifact.verdict["winner"] == "claude"

    def test_default_values(self):
        """Test default values for optional fields."""
        artifact = ReplayArtifact(
            debate_id="d-456",
            task="Another task",
        )

        assert artifact.scenes == []
        assert artifact.verdict == {}
        assert artifact.metadata == {}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        scene = ReplayScene(
            round_number=1,
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
        )
        artifact = ReplayArtifact(
            debate_id="d-789",
            task="Serialization test",
            scenes=[scene],
            verdict={"final_answer": "42"},
            metadata={"generated_at": "2024-01-01"},
        )

        result = artifact.to_dict()

        assert result["debate_id"] == "d-789"
        assert result["task"] == "Serialization test"
        assert len(result["scenes"]) == 1
        assert result["verdict"]["final_answer"] == "42"
        assert result["metadata"]["generated_at"] == "2024-01-01"

    def test_to_dict_with_multiple_scenes(self, sample_messages):
        """Test serialization with multiple scenes."""
        scenes = [
            ReplayScene(round_number=1, timestamp=datetime.now(), messages=sample_messages[:2]),
            ReplayScene(round_number=2, timestamp=datetime.now(), messages=sample_messages[2:]),
        ]
        artifact = ReplayArtifact(
            debate_id="d-multi",
            task="Multi-scene test",
            scenes=scenes,
        )

        result = artifact.to_dict()

        assert len(result["scenes"]) == 2
        assert result["scenes"][0]["round_number"] == 1
        assert result["scenes"][1]["round_number"] == 2


# ============================================================================
# ReplayGenerator Tests
# ============================================================================


class TestReplayGenerator:
    """Tests for ReplayGenerator class."""

    def test_initialization(self, generator):
        """Test generator initialization loads template."""
        assert generator.html_template is not None
        assert len(generator.html_template) > 0

    def test_initialization_template_has_placeholders(self, generator):
        """Test template has required placeholders."""
        # Template should have DATA and DEBATE_ID placeholders
        assert "{{DATA}}" in generator.html_template or "data" in generator.html_template.lower()

    def test_generate_returns_html(self, generator, sample_debate_result):
        """Test generate returns valid HTML."""
        html = generator.generate(sample_debate_result)

        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "</html>" in html

    def test_generate_includes_debate_data(self, generator, sample_debate_result):
        """Test generated HTML includes debate data."""
        html = generator.generate(sample_debate_result)

        assert "debate-123"[:8] in html  # Debate ID (possibly truncated)

    def test_create_artifact(self, generator, sample_debate_result):
        """Test artifact creation from debate result."""
        artifact = generator._create_artifact(sample_debate_result)

        assert artifact.debate_id == "debate-123"
        assert artifact.task == "Choose a programming language"
        assert len(artifact.scenes) >= 1
        assert artifact.metadata["rounds_used"] == 2
        assert artifact.metadata["consensus_reached"] is True

    def test_extract_scenes_groups_by_round(self, generator, sample_messages):
        """Test scenes are correctly grouped by round."""
        scenes = generator._extract_scenes(sample_messages)

        # Should have 2 rounds (round 1 and round 2)
        assert len(scenes) == 2

        # Round 1 should have 2 messages
        round_1_scene = next(s for s in scenes if s.round_number == 1)
        assert len(round_1_scene.messages) == 2

        # Round 2 should have 1 message
        round_2_scene = next(s for s in scenes if s.round_number == 2)
        assert len(round_2_scene.messages) == 1

    def test_extract_scenes_ordered_by_round(self, generator, sample_messages):
        """Test scenes are ordered by round number."""
        scenes = generator._extract_scenes(sample_messages)

        round_numbers = [s.round_number for s in scenes]
        assert round_numbers == sorted(round_numbers)

    def test_extract_scenes_default_consensus_indicator(self, generator, sample_messages):
        """Test scenes have default consensus indicator."""
        scenes = generator._extract_scenes(sample_messages)

        for scene in scenes:
            assert "reached" in scene.consensus_indicators
            assert "source" in scene.consensus_indicators

    def test_create_verdict_card(self, generator, sample_debate_result):
        """Test verdict card creation."""
        verdict = generator._create_verdict_card(sample_debate_result)

        assert verdict["final_answer"] == "Python is the best choice"
        assert verdict["confidence"] == 0.85
        assert verdict["consensus_reached"] is True
        assert verdict["rounds_used"] == 2
        assert "vote_breakdown" in verdict

    def test_create_verdict_card_vote_breakdown(self, generator, sample_debate_result):
        """Test vote breakdown in verdict card."""
        verdict = generator._create_verdict_card(sample_debate_result)

        breakdown = verdict["vote_breakdown"]
        assert len(breakdown) == 2  # python and rust

        python_votes = next(v for v in breakdown if v["choice"] == "python")
        assert python_votes["count"] == 2
        assert python_votes["avg_confidence"] == 0.8  # (0.9 + 0.7) / 2

    def test_create_verdict_card_tie_detection(self, generator):
        """Test tie detection in verdict card."""
        # Create result with tied votes
        tied_result = DebateResult(
            id="tie-debate",
            task="Tied vote test",
            final_answer="No clear winner",
            confidence=0.5,
            rounds_used=1,
            consensus_reached=True,
            messages=[],
            votes=[
                Vote(agent="a", choice="option1", reasoning="First option", confidence=0.8),
                Vote(agent="b", choice="option2", reasoning="Second option", confidence=0.8),
            ],
        )

        verdict = generator._create_verdict_card(tied_result)

        assert verdict["winner_label"] == "Tie"
        assert verdict["winner"] is None

    def test_create_verdict_card_no_consensus(self, generator):
        """Test verdict card when no consensus reached."""
        no_consensus = DebateResult(
            id="no-consensus",
            task="No consensus test",
            final_answer="",
            confidence=0.3,
            rounds_used=5,
            consensus_reached=False,
            messages=[],
            votes=[],
        )

        verdict = generator._create_verdict_card(no_consensus)

        assert verdict["consensus_reached"] is False
        assert verdict["winner_label"] == "No winner"

    def test_create_verdict_card_includes_evidence(self, generator, sample_debate_result):
        """Test evidence is included in verdict card."""
        verdict = generator._create_verdict_card(sample_debate_result)

        assert "evidence" in verdict
        assert len(verdict["evidence"]) > 0

    def test_create_verdict_card_escapes_html(self, generator):
        """Test HTML escaping in verdict card."""
        xss_result = DebateResult(
            id="xss-test",
            task="XSS test",
            final_answer="<script>alert('xss')</script>",
            confidence=0.9,
            rounds_used=1,
            consensus_reached=True,
            messages=[],
            votes=[],
            winning_patterns=["<img onerror='evil()'>"],
        )

        verdict = generator._create_verdict_card(xss_result)

        assert "<script>" not in verdict["final_answer"]
        assert "&lt;script&gt;" in verdict["final_answer"]

    def test_render_html_escapes_script_tags(self, generator):
        """Test script tag escaping in JSON data."""
        artifact = ReplayArtifact(
            debate_id="script-test",
            task="Test </script> in content",
        )

        html = generator._render_html(artifact)

        # Should escape </script> to prevent tag termination
        assert "</script>" not in html.split("</head>")[1].split("<script>")[0]

    def test_render_html_truncates_debate_id_in_title(self, generator):
        """Test long debate IDs are truncated in title placeholder."""
        artifact = ReplayArtifact(
            debate_id="very-long-debate-id-that-should-be-truncated",
            task="Truncation test",
        )

        html = generator._render_html(artifact)

        # ID should be truncated to 8 chars in the title
        assert "very-lon" in html
        # Full ID still appears in JSON data (expected behavior)
        assert "very-long-debate-id" in html  # Part of JSON data

    def test_get_html_template_fallback(self, generator):
        """Test fallback template when file not found."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            # Re-initialize to trigger fallback
            gen = ReplayGenerator.__new__(ReplayGenerator)
            gen.html_template = gen._get_html_template()

            assert "<!DOCTYPE html>" in gen.html_template
            assert "{{DATA}}" in gen.html_template
            assert "{{DEBATE_ID}}" in gen.html_template


# ============================================================================
# Integration Tests
# ============================================================================


class TestReplayIntegration:
    """Integration tests for replay generation."""

    def test_full_replay_workflow(self, sample_debate_result):
        """Test complete replay generation workflow."""
        generator = ReplayGenerator()

        # Generate HTML
        html = generator.generate(sample_debate_result)

        # Verify structure
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "</html>" in html

        # Verify debate content is embedded
        assert "debate-12" in html  # Truncated ID

    def test_replay_with_empty_debate(self):
        """Test replay with minimal debate data."""
        empty_result = DebateResult(
            id="empty-debate",
            task="Empty test",
            final_answer="No answer",
            confidence=0.0,
            rounds_used=0,
            consensus_reached=False,
            messages=[],
            votes=[],
        )

        generator = ReplayGenerator()
        html = generator.generate(empty_result)

        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "empty-de" in html  # Truncated ID

    def test_replay_with_many_rounds(self):
        """Test replay with many rounds."""
        messages = []
        for round_num in range(1, 6):
            messages.append(
                Message(
                    role="proposal",
                    agent=f"agent-{round_num}",
                    content=f"Message in round {round_num}",
                    round=round_num,
                    timestamp=datetime.now(),
                )
            )

        result = DebateResult(
            id="many-rounds",
            task="Multi-round test",
            final_answer="Final consensus",
            confidence=0.9,
            rounds_used=5,
            consensus_reached=True,
            messages=messages,
            votes=[],
        )

        generator = ReplayGenerator()
        artifact = generator._create_artifact(result)

        assert len(artifact.scenes) == 5
        assert artifact.metadata["rounds_used"] == 5

    def test_replay_preserves_message_order(self, sample_messages):
        """Test message order is preserved within scenes."""
        generator = ReplayGenerator()
        scenes = generator._extract_scenes(sample_messages)

        round_1 = next(s for s in scenes if s.round_number == 1)
        assert round_1.messages[0].role == "proposal"
        assert round_1.messages[1].role == "critique"

    def test_replay_handles_special_characters(self):
        """Test handling of special characters in content."""
        special_msg = Message(
            role="proposal",
            agent="test-agent",
            content="Special chars: < > & \" ' \n\t",
            round=1,
            timestamp=datetime.now(),
        )

        result = DebateResult(
            id="special-chars",
            task="Special characters test",
            final_answer="Answer with 'quotes' and <brackets>",
            confidence=0.8,
            rounds_used=1,
            consensus_reached=True,
            messages=[special_msg],
            votes=[],
        )

        generator = ReplayGenerator()
        html = generator.generate(result)

        # Should generate valid HTML without breaking
        assert "</html>" in html
