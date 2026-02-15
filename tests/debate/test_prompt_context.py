"""
Tests for PromptContextBuilder - debate prompt context building.

Tests cover:
- Persona context building
- Vertical context building
- Flip detection context
- Audience suggestion context preparation
- Proposal and revision prompt building
- Edge cases with missing or incomplete data
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.prompt_context import PromptContextBuilder


# --- Mock Classes ---


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "claude_proposer", role: str = "proposer"):
        self.name = name
        self.role = role
        self.model = "claude-3-opus"


class MockPersona:
    """Mock persona with to_prompt_context method."""

    def __init__(self, description: str = "", traits: list = None, expertise: dict = None):
        self.description = description
        self.traits = traits or []
        self.expertise = expertise or {}
        self.agent_name = "mock_agent"

    def to_prompt_context(self) -> str:
        """Generate prompt context from persona."""
        parts = []
        if self.description:
            parts.append(f"Your role: {self.description}")
        if self.traits:
            parts.append(f"Your approach: {', '.join(self.traits)}")
        if self.expertise:
            exp_str = ", ".join([f"{k} ({v:.0%})" for k, v in self.expertise.items()])
            parts.append(f"Your expertise areas: {exp_str}")
        return "\n".join(parts) if parts else ""


class MockAgentConsistencyScore:
    """Mock consistency score for flip detector."""

    def __init__(
        self,
        total_positions: int = 10,
        total_flips: int = 2,
        contradictions: int = 1,
        retractions: int = 1,
        consistency_score: float = 0.8,
        domains_with_flips: list = None,
    ):
        self.total_positions = total_positions
        self.total_flips = total_flips
        self.contradictions = contradictions
        self.retractions = retractions
        self.consistency_score = consistency_score
        self.domains_with_flips = domains_with_flips or []


class MockProtocol:
    """Mock debate protocol."""

    def __init__(self, audience_injection: str = None):
        self.audience_injection = audience_injection
        self.rounds = 3


class MockSuggestionCluster:
    """Mock suggestion cluster."""

    def __init__(self, representative: str, count: int):
        self.representative = representative
        self.count = count
        self.user_ids = ["user1", "user2"]


class MockCritique:
    """Mock critique for revision prompts."""

    def __init__(self, agent: str = "critic", issues: list = None):
        self.agent = agent
        self.issues = issues or ["Issue 1"]

    def to_prompt(self) -> str:
        return f"[{self.agent}]: {', '.join(self.issues)}"


# --- Fixtures ---


@pytest.fixture
def mock_agent():
    """Create mock agent."""
    return MockAgent(name="claude_proposer")


@pytest.fixture
def mock_persona_manager():
    """Create mock persona manager."""
    manager = MagicMock()
    persona = MockPersona(
        description="Security expert",
        traits=["thorough", "conservative"],
        expertise={"security": 0.9, "performance": 0.7},
    )
    manager.get_persona.return_value = persona
    return manager


@pytest.fixture
def mock_flip_detector():
    """Create mock flip detector."""
    detector = MagicMock()
    consistency = MockAgentConsistencyScore(
        total_positions=10,
        total_flips=2,
        contradictions=1,
        retractions=1,
        consistency_score=0.8,
    )
    detector.get_agent_consistency.return_value = consistency
    return detector


@pytest.fixture
def mock_protocol():
    """Create mock protocol."""
    return MockProtocol()


@pytest.fixture
def mock_audience_manager():
    """Create mock audience manager."""
    manager = MagicMock()
    manager._suggestions = deque(
        [
            {"suggestion": "Consider security implications", "user_id": "user1"},
            {"suggestion": "Add error handling", "user_id": "user2"},
        ]
    )
    manager.drain_events = MagicMock()
    return manager


@pytest.fixture
def mock_prompt_builder():
    """Create mock prompt builder."""
    builder = MagicMock()
    builder.build_proposal_prompt.return_value = "Proposal prompt"
    builder.build_revision_prompt.return_value = "Revision prompt"
    return builder


@pytest.fixture
def mock_spectator():
    """Create mock spectator stream."""
    spectator = MagicMock()
    spectator.emit = MagicMock()
    return spectator


@pytest.fixture
def basic_builder():
    """Create basic PromptContextBuilder instance."""
    return PromptContextBuilder()


@pytest.fixture
def full_builder(
    mock_persona_manager,
    mock_flip_detector,
    mock_protocol,
    mock_prompt_builder,
    mock_audience_manager,
    mock_spectator,
):
    """Create PromptContextBuilder with all dependencies."""
    return PromptContextBuilder(
        persona_manager=mock_persona_manager,
        flip_detector=mock_flip_detector,
        protocol=mock_protocol,
        prompt_builder=mock_prompt_builder,
        audience_manager=mock_audience_manager,
        spectator=mock_spectator,
    )


# --- Test Classes ---


class TestPromptContextBuilderInitialization:
    """Tests for PromptContextBuilder initialization."""

    def test_init_no_dependencies(self):
        """Test initialization without any dependencies."""
        builder = PromptContextBuilder()

        assert builder.persona_manager is None
        assert builder.flip_detector is None
        assert builder.protocol is None
        assert builder.prompt_builder is None
        assert builder.audience_manager is None
        assert builder.spectator is None
        assert builder._notify_callback is None
        assert builder.vertical is None
        assert builder.vertical_persona_manager is None

    def test_init_with_all_dependencies(
        self,
        mock_persona_manager,
        mock_flip_detector,
        mock_protocol,
        mock_prompt_builder,
        mock_audience_manager,
        mock_spectator,
    ):
        """Test initialization with all dependencies."""
        callback = MagicMock()

        builder = PromptContextBuilder(
            persona_manager=mock_persona_manager,
            flip_detector=mock_flip_detector,
            protocol=mock_protocol,
            prompt_builder=mock_prompt_builder,
            audience_manager=mock_audience_manager,
            spectator=mock_spectator,
            notify_callback=callback,
        )

        assert builder.persona_manager is mock_persona_manager
        assert builder.flip_detector is mock_flip_detector
        assert builder.protocol is mock_protocol
        assert builder.prompt_builder is mock_prompt_builder
        assert builder.audience_manager is mock_audience_manager
        assert builder.spectator is mock_spectator
        assert builder._notify_callback is callback

    def test_init_with_vertical(self):
        """Test initialization with vertical settings."""
        mock_vertical = MagicMock()
        mock_vertical_manager = MagicMock()

        builder = PromptContextBuilder(
            vertical=mock_vertical,
            vertical_persona_manager=mock_vertical_manager,
        )

        assert builder.vertical is mock_vertical
        assert builder.vertical_persona_manager is mock_vertical_manager


class TestGetPersonaContext:
    """Tests for get_persona_context method."""

    def test_no_persona_manager(self, basic_builder, mock_agent):
        """Test returns empty string when no persona manager."""
        result = basic_builder.get_persona_context(mock_agent)

        assert result == ""

    def test_persona_found_in_manager(self, mock_persona_manager, mock_agent):
        """Test returns persona context when persona found in manager."""
        builder = PromptContextBuilder(persona_manager=mock_persona_manager)

        result = builder.get_persona_context(mock_agent)

        assert "Your role: Security expert" in result
        assert "thorough" in result
        assert "security" in result
        mock_persona_manager.get_persona.assert_called_once_with(mock_agent.name)

    def test_persona_not_found_fallback_to_default(self):
        """Test fallback to default persona when not found in manager."""
        manager = MagicMock()
        manager.get_persona.return_value = None

        builder = PromptContextBuilder(persona_manager=manager)
        agent = MockAgent(name="claude_proposer")

        # Patch DEFAULT_PERSONAS in the personas module (imported inside function)
        with patch(
            "aragora.agents.personas.DEFAULT_PERSONAS",
            {"claude": MockPersona(description="Default Claude persona")},
        ):
            result = builder.get_persona_context(agent)

        assert "Default Claude persona" in result

    def test_persona_not_found_no_default(self):
        """Test returns empty string when persona not found and no default."""
        manager = MagicMock()
        manager.get_persona.return_value = None

        builder = PromptContextBuilder(persona_manager=manager)
        agent = MockAgent(name="unknown_agent")

        with patch("aragora.agents.personas.DEFAULT_PERSONAS", {}):
            result = builder.get_persona_context(agent)

        assert result == ""

    def test_agent_type_extraction(self):
        """Test agent type is extracted correctly from agent name."""
        manager = MagicMock()
        manager.get_persona.return_value = None

        builder = PromptContextBuilder(persona_manager=manager)

        # Test various agent name patterns
        test_cases = [
            ("claude_proposer_1", "claude"),
            ("gpt4_critic", "gpt4"),
            ("gemini_judge_v2", "gemini"),
        ]

        for name, expected_type in test_cases:
            agent = MockAgent(name=name)

            with patch(
                "aragora.agents.personas.DEFAULT_PERSONAS",
                {expected_type: MockPersona(description=f"{expected_type} default")},
            ):
                result = builder.get_persona_context(agent)

            assert expected_type in result.lower() or f"{expected_type} default" in result


class TestGetVerticalContext:
    """Tests for get_vertical_context method."""

    def test_no_vertical_or_manager(self, basic_builder):
        """Test returns empty string when no vertical or manager."""
        result = basic_builder.get_vertical_context()

        assert result == ""

    def test_with_vertical_config(self):
        """Test returns vertical context when vertical is set."""
        # Create mock vertical and config
        mock_vertical = MagicMock()
        mock_vertical.value = "software"
        mock_vertical.__eq__ = lambda self, other: False  # Not GENERAL

        mock_config = MagicMock()
        mock_config.vertical = mock_vertical
        mock_config.description = "Software development focus"
        mock_config.compliance_frameworks = ["owasp", "soc2", "iso_27001"]
        mock_config.expertise_domains = ["security", "performance", "architecture"]
        mock_config.requires_high_accuracy = True
        mock_config.max_temperature = 0.4

        mock_manager = MagicMock()
        mock_manager.get_vertical_config.return_value = mock_config

        builder = PromptContextBuilder(
            vertical=mock_vertical, vertical_persona_manager=mock_manager
        )

        result = builder.get_vertical_context()

        assert "Industry Vertical Context" in result
        assert "Software" in result
        assert "Software development focus" in result
        assert "owasp" in result
        assert "security" in result
        assert "high accuracy" in result
        assert "formal" in result  # Due to max_temperature < 0.5

    def test_auto_detect_vertical_from_task(self):
        """Test vertical auto-detection from task description."""
        from aragora.agents.vertical_personas import Vertical, VerticalPersonaManager

        mock_manager = MagicMock(spec=VerticalPersonaManager)

        # Mock the detect method to return LEGAL vertical
        mock_manager.detect_vertical_from_task.return_value = Vertical.LEGAL

        # Mock config
        mock_config = MagicMock()
        mock_config.vertical = Vertical.LEGAL
        mock_config.description = "Legal analysis"
        mock_config.compliance_frameworks = ["aba_ethics", "gdpr"]
        mock_config.expertise_domains = ["legal", "compliance"]
        mock_config.requires_high_accuracy = True
        mock_config.max_temperature = 0.5
        mock_manager.get_vertical_config.return_value = mock_config

        builder = PromptContextBuilder(vertical_persona_manager=mock_manager)
        result = builder.get_vertical_context(task="Review this contract")

        # Should attempt auto-detection
        mock_manager.detect_vertical_from_task.assert_called_once_with("Review this contract")
        assert "Legal" in result

    def test_general_vertical_returns_empty(self):
        """Test returns empty string for GENERAL vertical."""
        from aragora.agents.vertical_personas import Vertical, VerticalPersonaManager

        # Create builder with GENERAL vertical
        builder = PromptContextBuilder(vertical=Vertical.GENERAL)
        result = builder.get_vertical_context()

        assert result == ""

    def test_import_error_returns_empty(self, basic_builder):
        """Test returns empty string on import error."""
        # This is handled by the try/except in get_vertical_context
        # The method should return "" on any ImportError
        result = basic_builder.get_vertical_context()
        assert result == ""


class TestGetFlipContext:
    """Tests for get_flip_context method."""

    def test_no_flip_detector(self, basic_builder, mock_agent):
        """Test returns empty string when no flip detector."""
        result = basic_builder.get_flip_context(mock_agent)

        assert result == ""

    def test_no_position_history(self, mock_agent):
        """Test returns empty string when agent has no position history."""
        detector = MagicMock()
        detector.get_agent_consistency.return_value = MockAgentConsistencyScore(
            total_positions=0, total_flips=0
        )

        builder = PromptContextBuilder(flip_detector=detector)
        result = builder.get_flip_context(mock_agent)

        assert result == ""

    def test_no_flips(self, mock_agent):
        """Test returns empty string when no flips detected."""
        detector = MagicMock()
        detector.get_agent_consistency.return_value = MockAgentConsistencyScore(
            total_positions=10, total_flips=0, contradictions=0, retractions=0
        )

        builder = PromptContextBuilder(flip_detector=detector)
        result = builder.get_flip_context(mock_agent)

        assert result == ""

    def test_with_contradictions(self, mock_flip_detector, mock_agent):
        """Test context includes contradiction warning."""
        builder = PromptContextBuilder(flip_detector=mock_flip_detector)
        result = builder.get_flip_context(mock_agent)

        assert "Position Consistency Note" in result
        assert "contradiction" in result.lower()

    def test_with_retractions(self, mock_agent):
        """Test context includes retraction note."""
        detector = MagicMock()
        detector.get_agent_consistency.return_value = MockAgentConsistencyScore(
            total_positions=10,
            total_flips=2,
            contradictions=0,
            retractions=2,
            consistency_score=0.8,
        )

        builder = PromptContextBuilder(flip_detector=detector)
        result = builder.get_flip_context(mock_agent)

        assert "retracted" in result.lower()

    def test_low_consistency_score(self, mock_agent):
        """Test context includes warning for low consistency score."""
        detector = MagicMock()
        detector.get_agent_consistency.return_value = MockAgentConsistencyScore(
            total_positions=10,
            total_flips=5,
            contradictions=3,
            retractions=2,
            consistency_score=0.5,  # Below 0.7 threshold
        )

        builder = PromptContextBuilder(flip_detector=detector)
        result = builder.get_flip_context(mock_agent)

        assert "consistency score" in result.lower()
        assert "50%" in result

    def test_domains_with_flips(self, mock_agent):
        """Test context includes domains with position changes."""
        detector = MagicMock()
        detector.get_agent_consistency.return_value = MockAgentConsistencyScore(
            total_positions=10,
            total_flips=2,
            contradictions=1,
            retractions=1,
            consistency_score=0.8,
            domains_with_flips=["security", "performance", "architecture"],
        )

        builder = PromptContextBuilder(flip_detector=detector)
        result = builder.get_flip_context(mock_agent)

        assert "security" in result
        assert "performance" in result
        assert "architecture" in result

    def test_detector_error_returns_empty(self, mock_agent):
        """Test returns empty string on detector error."""
        detector = MagicMock()
        detector.get_agent_consistency.side_effect = ValueError("Test error")

        builder = PromptContextBuilder(flip_detector=detector)
        result = builder.get_flip_context(mock_agent)

        assert result == ""

    def test_detector_attribute_error(self, mock_agent):
        """Test handles AttributeError gracefully."""
        detector = MagicMock()
        detector.get_agent_consistency.side_effect = AttributeError("Missing attribute")

        builder = PromptContextBuilder(flip_detector=detector)
        result = builder.get_flip_context(mock_agent)

        assert result == ""


class TestNotifySpectator:
    """Tests for _notify_spectator method."""

    def test_notify_with_callback(self):
        """Test notification uses callback when available."""
        callback = MagicMock()
        builder = PromptContextBuilder(notify_callback=callback)

        builder._notify_spectator("test_event", key="value")

        callback.assert_called_once_with("test_event", key="value")

    def test_notify_with_spectator(self, mock_spectator):
        """Test notification uses spectator when no callback."""
        builder = PromptContextBuilder(spectator=mock_spectator)

        builder._notify_spectator("test_event", key="value")

        mock_spectator.emit.assert_called_once_with("test_event", key="value")

    def test_notify_prefers_callback_over_spectator(self, mock_spectator):
        """Test callback is preferred over spectator."""
        callback = MagicMock()
        builder = PromptContextBuilder(spectator=mock_spectator, notify_callback=callback)

        builder._notify_spectator("test_event")

        callback.assert_called_once()
        mock_spectator.emit.assert_not_called()

    def test_notify_spectator_error_handled(self, mock_spectator):
        """Test spectator errors are handled gracefully."""
        mock_spectator.emit.side_effect = TypeError("Test error")
        builder = PromptContextBuilder(spectator=mock_spectator)

        # Should not raise
        builder._notify_spectator("test_event")

    def test_notify_no_spectator_or_callback(self, basic_builder):
        """Test no-op when neither spectator nor callback."""
        # Should not raise
        basic_builder._notify_spectator("test_event")


class TestPrepareAudienceContext:
    """Tests for prepare_audience_context method."""

    def test_no_audience_manager(self, basic_builder):
        """Test returns empty when no audience manager."""
        result = basic_builder.prepare_audience_context()

        assert result == ""

    def test_no_protocol(self, mock_audience_manager):
        """Test returns empty when no protocol."""
        builder = PromptContextBuilder(audience_manager=mock_audience_manager)

        result = builder.prepare_audience_context()

        assert result == ""

    def test_audience_injection_disabled(self, mock_audience_manager):
        """Test returns empty when audience injection disabled."""
        protocol = MockProtocol(audience_injection=None)
        builder = PromptContextBuilder(audience_manager=mock_audience_manager, protocol=protocol)

        result = builder.prepare_audience_context()

        assert result == ""

    def test_no_suggestions(self):
        """Test returns empty when no suggestions."""
        manager = MagicMock()
        manager._suggestions = deque()
        manager.drain_events = MagicMock()

        protocol = MockProtocol(audience_injection="summary")
        builder = PromptContextBuilder(audience_manager=manager, protocol=protocol)

        result = builder.prepare_audience_context()

        assert result == ""

    def test_audience_injection_summary(self, mock_audience_manager):
        """Test audience context with summary injection."""
        protocol = MockProtocol(audience_injection="summary")
        builder = PromptContextBuilder(audience_manager=mock_audience_manager, protocol=protocol)

        with patch("aragora.debate.prompt_context.cluster_suggestions") as mock_cluster:
            with patch("aragora.debate.prompt_context.format_for_prompt") as mock_format:
                mock_cluster.return_value = [MockSuggestionCluster("Consider security", 2)]
                mock_format.return_value = (
                    "## AUDIENCE SUGGESTIONS\n- [2 similar]: Consider security"
                )

                result = builder.prepare_audience_context()

        assert "AUDIENCE SUGGESTIONS" in result
        mock_audience_manager.drain_events.assert_called_once()

    def test_audience_injection_inject(self, mock_audience_manager):
        """Test audience context with inject mode."""
        protocol = MockProtocol(audience_injection="inject")
        builder = PromptContextBuilder(audience_manager=mock_audience_manager, protocol=protocol)

        with patch("aragora.debate.prompt_context.cluster_suggestions") as mock_cluster:
            with patch("aragora.debate.prompt_context.format_for_prompt") as mock_format:
                mock_cluster.return_value = [MockSuggestionCluster("Add tests", 3)]
                mock_format.return_value = "## AUDIENCE SUGGESTIONS\n- [3 similar]: Add tests"

                result = builder.prepare_audience_context()

        assert "AUDIENCE SUGGESTIONS" in result

    def test_emit_event_for_dashboard(self, mock_audience_manager, mock_spectator):
        """Test spectator event emission when requested."""
        protocol = MockProtocol(audience_injection="summary")
        builder = PromptContextBuilder(
            audience_manager=mock_audience_manager,
            protocol=protocol,
            spectator=mock_spectator,
        )

        with patch("aragora.debate.prompt_context.cluster_suggestions") as mock_cluster:
            with patch("aragora.debate.prompt_context.format_for_prompt") as mock_format:
                cluster = MockSuggestionCluster("Test suggestion", 2)
                mock_cluster.return_value = [cluster]
                mock_format.return_value = "Formatted"

                builder.prepare_audience_context(emit_event=True)

        mock_spectator.emit.assert_called_once()
        call_args = mock_spectator.emit.call_args
        assert call_args[0][0] == "audience_summary"

    def test_no_emit_event_when_not_requested(self, mock_audience_manager, mock_spectator):
        """Test no spectator event when not requested."""
        protocol = MockProtocol(audience_injection="summary")
        builder = PromptContextBuilder(
            audience_manager=mock_audience_manager,
            protocol=protocol,
            spectator=mock_spectator,
        )

        with patch("aragora.debate.prompt_context.cluster_suggestions") as mock_cluster:
            with patch("aragora.debate.prompt_context.format_for_prompt") as mock_format:
                mock_cluster.return_value = [MockSuggestionCluster("Test", 1)]
                mock_format.return_value = "Formatted"

                builder.prepare_audience_context(emit_event=False)

        mock_spectator.emit.assert_not_called()


class TestBuildProposalPrompt:
    """Tests for build_proposal_prompt method."""

    def test_no_prompt_builder_raises(self, basic_builder, mock_agent):
        """Test raises ValueError when no prompt builder."""
        with pytest.raises(ValueError, match="PromptBuilder is required"):
            basic_builder.build_proposal_prompt(mock_agent)

    def test_builds_proposal_prompt(self, mock_prompt_builder, mock_agent):
        """Test builds proposal prompt with audience context."""
        builder = PromptContextBuilder(prompt_builder=mock_prompt_builder)

        result = builder.build_proposal_prompt(mock_agent)

        assert result == "Proposal prompt"
        mock_prompt_builder.build_proposal_prompt.assert_called_once()
        # Check agent was passed
        call_args = mock_prompt_builder.build_proposal_prompt.call_args
        assert call_args[0][0] is mock_agent

    def test_includes_audience_context(
        self, mock_prompt_builder, mock_audience_manager, mock_agent
    ):
        """Test includes audience context in proposal prompt."""
        protocol = MockProtocol(audience_injection="summary")
        builder = PromptContextBuilder(
            prompt_builder=mock_prompt_builder,
            audience_manager=mock_audience_manager,
            protocol=protocol,
        )

        with patch("aragora.debate.prompt_context.cluster_suggestions") as mock_cluster:
            with patch("aragora.debate.prompt_context.format_for_prompt") as mock_format:
                mock_cluster.return_value = [MockSuggestionCluster("Test", 1)]
                mock_format.return_value = "Audience section"

                builder.build_proposal_prompt(mock_agent)

        # Check audience_section was passed
        call_args = mock_prompt_builder.build_proposal_prompt.call_args
        assert call_args[0][1] == "Audience section"


class TestBuildRevisionPrompt:
    """Tests for build_revision_prompt method."""

    def test_no_prompt_builder_raises(self, basic_builder, mock_agent):
        """Test raises ValueError when no prompt builder."""
        with pytest.raises(ValueError, match="PromptBuilder is required"):
            basic_builder.build_revision_prompt(mock_agent, "Original", [MockCritique()])

    def test_builds_revision_prompt(self, mock_prompt_builder, mock_agent):
        """Test builds revision prompt with all parameters."""
        builder = PromptContextBuilder(prompt_builder=mock_prompt_builder)
        critiques = [MockCritique(agent="critic1"), MockCritique(agent="critic2")]

        result = builder.build_revision_prompt(
            mock_agent, "Original proposal", critiques, round_number=2
        )

        assert result == "Revision prompt"
        mock_prompt_builder.build_revision_prompt.assert_called_once()

        # Check all parameters passed
        call_args = mock_prompt_builder.build_revision_prompt.call_args
        assert call_args[0][0] is mock_agent
        assert call_args[0][1] == "Original proposal"
        assert call_args[0][2] == critiques
        assert call_args[1]["round_number"] == 2

    def test_revision_prompt_with_audience_context(
        self, mock_prompt_builder, mock_audience_manager, mock_agent
    ):
        """Test revision prompt includes audience context."""
        protocol = MockProtocol(audience_injection="inject")
        builder = PromptContextBuilder(
            prompt_builder=mock_prompt_builder,
            audience_manager=mock_audience_manager,
            protocol=protocol,
        )

        with patch("aragora.debate.prompt_context.cluster_suggestions") as mock_cluster:
            with patch("aragora.debate.prompt_context.format_for_prompt") as mock_format:
                mock_cluster.return_value = [MockSuggestionCluster("Test", 1)]
                mock_format.return_value = "Audience context"

                builder.build_revision_prompt(
                    mock_agent, "Original", [MockCritique()], round_number=1
                )

        # Check audience_section passed (4th positional arg)
        call_args = mock_prompt_builder.build_revision_prompt.call_args
        assert call_args[0][3] == "Audience context"

    def test_revision_prompt_no_emit_event(
        self, mock_prompt_builder, mock_audience_manager, mock_spectator, mock_agent
    ):
        """Test revision prompt does not emit spectator event."""
        protocol = MockProtocol(audience_injection="summary")
        builder = PromptContextBuilder(
            prompt_builder=mock_prompt_builder,
            audience_manager=mock_audience_manager,
            protocol=protocol,
            spectator=mock_spectator,
        )

        with patch("aragora.debate.prompt_context.cluster_suggestions") as mock_cluster:
            with patch("aragora.debate.prompt_context.format_for_prompt") as mock_format:
                mock_cluster.return_value = [MockSuggestionCluster("Test", 1)]
                mock_format.return_value = "Audience"

                builder.build_revision_prompt(
                    mock_agent, "Original", [MockCritique()], round_number=1
                )

        # Revision prompt should NOT emit event (emit_event=False)
        mock_spectator.emit.assert_not_called()


class TestContextMerging:
    """Tests for context merging scenarios."""

    def test_multiple_context_sources(self, mock_persona_manager, mock_flip_detector, mock_agent):
        """Test building context from multiple sources."""
        builder = PromptContextBuilder(
            persona_manager=mock_persona_manager, flip_detector=mock_flip_detector
        )

        persona_ctx = builder.get_persona_context(mock_agent)
        flip_ctx = builder.get_flip_context(mock_agent)

        # Both should have content
        assert persona_ctx
        assert flip_ctx

        # Combined context should contain both
        combined = f"{persona_ctx}\n\n{flip_ctx}"
        assert "Security expert" in combined
        assert "contradiction" in combined.lower()

    def test_partial_context_sources(self, mock_persona_manager, mock_agent):
        """Test context with only some sources available."""
        builder = PromptContextBuilder(
            persona_manager=mock_persona_manager,
            flip_detector=None,  # No flip detector
        )

        persona_ctx = builder.get_persona_context(mock_agent)
        flip_ctx = builder.get_flip_context(mock_agent)

        assert persona_ctx  # Should have content
        assert flip_ctx == ""  # Should be empty


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_persona(self, mock_agent):
        """Test handling of empty persona."""
        manager = MagicMock()
        manager.get_persona.return_value = MockPersona()  # Empty persona

        builder = PromptContextBuilder(persona_manager=manager)
        result = builder.get_persona_context(mock_agent)

        assert result == ""

    def test_agent_name_without_underscore(self):
        """Test agent name parsing without underscore."""
        manager = MagicMock()
        manager.get_persona.return_value = None

        builder = PromptContextBuilder(persona_manager=manager)
        agent = MockAgent(name="claude")  # No underscore

        with patch(
            "aragora.agents.personas.DEFAULT_PERSONAS",
            {"claude": MockPersona(description="Claude default")},
        ):
            result = builder.get_persona_context(agent)

        assert "Claude default" in result

    def test_flip_detector_key_error(self, mock_agent):
        """Test handling of KeyError from flip detector."""
        detector = MagicMock()
        detector.get_agent_consistency.side_effect = KeyError("agent_not_found")

        builder = PromptContextBuilder(flip_detector=detector)
        result = builder.get_flip_context(mock_agent)

        assert result == ""

    def test_flip_detector_runtime_error(self, mock_agent):
        """Test handling of RuntimeError from flip detector."""
        detector = MagicMock()
        detector.get_agent_consistency.side_effect = RuntimeError("DB connection failed")

        builder = PromptContextBuilder(flip_detector=detector)
        result = builder.get_flip_context(mock_agent)

        assert result == ""

    def test_audience_manager_suggestions_attribute(self):
        """Test accessing _suggestions attribute."""
        # Test with missing attribute
        manager = MagicMock(spec=[])  # Empty spec, no _suggestions
        manager._suggestions = deque()
        manager.drain_events = MagicMock()

        protocol = MockProtocol(audience_injection="summary")
        builder = PromptContextBuilder(audience_manager=manager, protocol=protocol)

        # Should handle gracefully
        result = builder.prepare_audience_context()
        assert result == ""

    def test_spectator_unexpected_error(self, mock_spectator):
        """Test handling of unexpected spectator error."""
        mock_spectator.emit.side_effect = Exception("Unexpected error")
        builder = PromptContextBuilder(spectator=mock_spectator)

        # Should not raise, but log warning
        builder._notify_spectator("test_event")

    def test_very_low_consistency_score(self, mock_agent):
        """Test flip context with very low consistency score."""
        detector = MagicMock()
        detector.get_agent_consistency.return_value = MockAgentConsistencyScore(
            total_positions=20,
            total_flips=15,
            contradictions=10,
            retractions=5,
            consistency_score=0.2,  # Very low
            domains_with_flips=["security", "performance", "architecture", "testing"],
        )

        builder = PromptContextBuilder(flip_detector=detector)
        result = builder.get_flip_context(mock_agent)

        assert "20%" in result
        assert "Prioritize coherent positions" in result


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ exports."""
        from aragora.debate import prompt_context

        assert hasattr(prompt_context, "__all__")
        assert "PromptContextBuilder" in prompt_context.__all__

    def test_import_prompt_context_builder(self):
        """Test direct import of PromptContextBuilder."""
        from aragora.debate.prompt_context import PromptContextBuilder

        assert PromptContextBuilder is not None
