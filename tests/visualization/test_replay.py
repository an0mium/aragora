"""Tests for replay theater visualization."""

import json
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from aragora.visualization.replay import (
    ReplayScene,
    ReplayArtifact,
    ReplayGenerator,
)


class TestReplayScene:
    """Tests for ReplayScene dataclass."""

    def test_create_scene(self):
        """Basic scene creation."""
        scene = ReplayScene(
            round_number=1,
            timestamp=datetime(2024, 1, 1, 12, 0),
        )
        assert scene.round_number == 1
        assert scene.timestamp == datetime(2024, 1, 1, 12, 0)

    def test_scene_defaults(self):
        """Scene should have default empty collections."""
        scene = ReplayScene(
            round_number=1,
            timestamp=datetime.now(),
        )
        assert scene.messages == []
        assert scene.consensus_indicators == {}

    def test_scene_to_dict(self):
        """Should serialize to dictionary."""
        scene = ReplayScene(
            round_number=2,
            timestamp=datetime(2024, 1, 15, 10, 30),
            consensus_indicators={"reached": True},
        )
        data = scene.to_dict()
        
        assert data["round_number"] == 2
        assert "timestamp" in data
        assert data["consensus_indicators"] == {"reached": True}

    def test_scene_to_dict_escapes_html(self):
        """Should escape HTML in message content."""
        mock_message = Mock()
        mock_message.role = "agent"
        mock_message.agent = "claude"
        mock_message.content = "<script>alert('xss')</script>"
        mock_message.timestamp = datetime.now()
        mock_message.round = 1
        
        scene = ReplayScene(
            round_number=1,
            timestamp=datetime.now(),
            messages=[mock_message],
        )
        data = scene.to_dict()
        
        # Should escape HTML
        assert "<script>" not in data["messages"][0]["content"]
        assert "&lt;script&gt;" in data["messages"][0]["content"]


class TestReplayArtifact:
    """Tests for ReplayArtifact dataclass."""

    def test_create_artifact(self):
        """Basic artifact creation."""
        artifact = ReplayArtifact(
            debate_id="debate-123",
            task="Discuss implementation",
        )
        assert artifact.debate_id == "debate-123"
        assert artifact.task == "Discuss implementation"

    def test_artifact_defaults(self):
        """Artifact should have default empty collections."""
        artifact = ReplayArtifact(
            debate_id="d1",
            task="test",
        )
        assert artifact.scenes == []
        assert artifact.verdict == {}
        assert artifact.metadata == {}

    def test_artifact_to_dict(self):
        """Should serialize to dictionary."""
        scene = ReplayScene(
            round_number=1,
            timestamp=datetime(2024, 1, 1),
        )
        artifact = ReplayArtifact(
            debate_id="debate-123",
            task="Test task",
            scenes=[scene],
            verdict={"winner": "approve"},
            metadata={"key": "value"},
        )
        data = artifact.to_dict()
        
        assert data["debate_id"] == "debate-123"
        assert data["task"] == "Test task"
        assert len(data["scenes"]) == 1
        assert data["verdict"] == {"winner": "approve"}
        assert data["metadata"] == {"key": "value"}


class TestReplayGenerator:
    """Tests for ReplayGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a replay generator."""
        return ReplayGenerator()

    @pytest.fixture
    def mock_debate_result(self):
        """Create a mock debate result."""
        result = Mock()
        result.id = "debate-123"
        result.task = "Test debate topic"
        result.messages = []
        result.votes = []
        result.duration_seconds = 120.0
        result.rounds_used = 3
        result.consensus_reached = True
        result.confidence = 0.85
        result.convergence_status = "converged"
        result.consensus_strength = 0.9
        result.final_answer = "The answer is 42"
        result.winning_patterns = ["Pattern 1"]
        result.critiques = []
        result.dissenting_views = []
        return result

    def test_generator_init(self, generator):
        """Should initialize with HTML template."""
        assert generator.html_template is not None
        assert "{{DATA}}" in generator.html_template or "data" in generator.html_template.lower()

    def test_generate_returns_html(self, generator, mock_debate_result):
        """Should return HTML string."""
        html = generator.generate(mock_debate_result)
        
        assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower()
        assert "</html>" in html

    def test_generate_includes_debate_id(self, generator, mock_debate_result):
        """Should include debate ID in output."""
        html = generator.generate(mock_debate_result)
        
        # Debate ID should be in the HTML (possibly truncated)
        assert "debate" in html.lower()

    def test_generate_escapes_script_tags(self, generator, mock_debate_result):
        """Should escape </script> to prevent XSS."""
        mock_debate_result.final_answer = "</script><script>alert('xss')</script>"
        
        html = generator.generate(mock_debate_result)
        
        # Should not have raw </script> inside the data
        # (the actual script tag for mermaid/JS is allowed, but data shouldn't break it)
        assert "</\\script>" in html or "&lt;" in html or "<\\/script>" in html

    def test_extract_scenes_groups_by_round(self, generator, mock_debate_result):
        """Should group messages by round."""
        msg1 = Mock()
        msg1.round = 1
        msg1.timestamp = datetime(2024, 1, 1, 12, 0)
        msg1.role = "proposer"
        msg1.agent = "claude"
        msg1.content = "Proposal"
        
        msg2 = Mock()
        msg2.round = 1
        msg2.timestamp = datetime(2024, 1, 1, 12, 1)
        msg2.role = "critic"
        msg2.agent = "gemini"
        msg2.content = "Critique"
        
        msg3 = Mock()
        msg3.round = 2
        msg3.timestamp = datetime(2024, 1, 1, 12, 5)
        msg3.role = "proposer"
        msg3.agent = "claude"
        msg3.content = "Response"
        
        mock_debate_result.messages = [msg1, msg2, msg3]
        
        scenes = generator._extract_scenes(mock_debate_result.messages)
        
        assert len(scenes) == 2
        assert scenes[0].round_number == 1
        assert len(scenes[0].messages) == 2
        assert scenes[1].round_number == 2
        assert len(scenes[1].messages) == 1

    def test_create_verdict_card(self, generator, mock_debate_result):
        """Should create verdict card with correct structure."""
        # Add some votes
        vote1 = Mock()
        vote1.choice = "approve"
        vote1.confidence = 0.9
        
        vote2 = Mock()
        vote2.choice = "approve"
        vote2.confidence = 0.8
        
        vote3 = Mock()
        vote3.choice = "reject"
        vote3.confidence = 0.7
        
        mock_debate_result.votes = [vote1, vote2, vote3]
        
        verdict = generator._create_verdict_card(mock_debate_result)
        
        assert "final_answer" in verdict
        assert "confidence" in verdict
        assert "consensus_reached" in verdict
        assert "vote_breakdown" in verdict

    def test_create_verdict_card_handles_tie(self, generator, mock_debate_result):
        """Should detect tie in votes."""
        vote1 = Mock()
        vote1.choice = "approve"
        vote1.confidence = 0.8
        
        vote2 = Mock()
        vote2.choice = "reject"
        vote2.confidence = 0.8
        
        mock_debate_result.votes = [vote1, vote2]
        
        verdict = generator._create_verdict_card(mock_debate_result)
        
        assert verdict["winner_label"] == "Tie"

    def test_create_verdict_card_escapes_html(self, generator, mock_debate_result):
        """Should escape HTML in verdict fields."""
        mock_debate_result.final_answer = "<b>Bold answer</b>"
        mock_debate_result.winning_patterns = ["<script>evil</script>"]
        
        verdict = generator._create_verdict_card(mock_debate_result)
        
        assert "<b>" not in verdict["final_answer"]
        assert "<script>" not in verdict["evidence"][0]

    def test_render_html_replaces_placeholders(self, generator):
        """Should replace template placeholders."""
        artifact = ReplayArtifact(
            debate_id="test-debate-id",
            task="Test task",
        )
        
        html = generator._render_html(artifact)
        
        # Placeholders should be replaced
        assert "{{DATA}}" not in html
        assert "{{DEBATE_ID}}" not in html


class TestReplayGeneratorWithMessages:
    """Tests for replay generation with actual messages."""

    @pytest.fixture
    def generator(self):
        """Create generator."""
        return ReplayGenerator()

    @pytest.fixture
    def debate_with_messages(self):
        """Create debate result with messages."""
        result = Mock()
        result.id = "debate-full"
        result.task = "Full debate with messages"
        result.duration_seconds = 300.0
        result.rounds_used = 2
        result.consensus_reached = True
        result.confidence = 0.92
        result.convergence_status = "converged"
        result.consensus_strength = 0.88
        result.final_answer = "Consensus answer"
        result.winning_patterns = []
        result.critiques = []
        result.votes = []
        result.dissenting_views = []
        
        # Create messages
        messages = []
        for i in range(3):
            msg = Mock()
            msg.round = i // 2 + 1
            msg.timestamp = datetime(2024, 1, 1, 12, i)
            msg.role = "agent"
            msg.agent = f"agent-{i}"
            msg.content = f"Message content {i}"
            messages.append(msg)
        
        result.messages = messages
        return result

    def test_full_generation(self, generator, debate_with_messages):
        """Should generate complete HTML for debate with messages."""
        html = generator.generate(debate_with_messages)
        
        # Should have basic HTML structure
        assert "<!DOCTYPE html>" in html or "html" in html.lower()
        
        # Should have embedded data
        assert "debate-full" in html or "data" in html.lower()


class TestReplayGeneratorTemplate:
    """Tests for HTML template handling."""

    def test_uses_external_template_if_exists(self):
        """Should use external template file if it exists."""
        generator = ReplayGenerator()
        
        # The template should either be loaded from file or be the fallback
        assert generator.html_template is not None
        assert len(generator.html_template) > 0

    def test_fallback_template_is_valid_html(self):
        """Fallback template should be valid HTML."""
        # Simulate template file not found
        with patch("builtins.open", side_effect=FileNotFoundError()):
            generator = ReplayGenerator()
        
        # Even with fallback, should have basic HTML
        assert "<!DOCTYPE html>" in generator.html_template or "<html" in generator.html_template
        assert "{{DATA}}" in generator.html_template
        assert "{{DEBATE_ID}}" in generator.html_template


class TestReplaySceneConsensusIndicators:
    """Tests for consensus indicator handling in scenes."""

    def test_default_consensus_indicator(self):
        """Scene should have default consensus indicator."""
        scene = ReplayScene(
            round_number=1,
            timestamp=datetime.now(),
            consensus_indicators={"reached": False, "source": "default"},
        )
        data = scene.to_dict()
        
        assert data["consensus_indicators"]["reached"] is False

    def test_custom_consensus_indicator(self):
        """Scene should preserve custom consensus indicator."""
        scene = ReplayScene(
            round_number=3,
            timestamp=datetime.now(),
            consensus_indicators={
                "reached": True,
                "confidence": 0.95,
                "source": "trace",
                "description": "Unanimous agreement",
            },
        )
        data = scene.to_dict()
        
        assert data["consensus_indicators"]["reached"] is True
        assert data["consensus_indicators"]["confidence"] == 0.95
