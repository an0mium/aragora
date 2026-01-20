"""Tests for PromptBuilder module."""

import pytest
from unittest.mock import MagicMock, patch

from aragora.debate.prompt_builder import PromptBuilder
from aragora.debate.protocol import DebateProtocol
from aragora.core import Environment


class TestPromptBuilderCreation:
    """Tests for PromptBuilder instantiation."""

    def test_creates_with_minimal_args(self):
        """PromptBuilder can be created with just protocol and env."""
        protocol = DebateProtocol()
        env = Environment(task="Test task")
        builder = PromptBuilder(protocol=protocol, env=env)
        assert builder.protocol == protocol
        assert builder.env == env

    def test_creates_with_all_optional_dependencies(self):
        """PromptBuilder accepts all optional dependencies."""
        protocol = DebateProtocol()
        env = Environment(task="Test task")
        mock_memory = MagicMock()
        mock_continuum = MagicMock()
        mock_dissent = MagicMock()
        mock_rotator = MagicMock()
        mock_persona = MagicMock()
        mock_flip = MagicMock()

        builder = PromptBuilder(
            protocol=protocol,
            env=env,
            memory=mock_memory,
            continuum_memory=mock_continuum,
            dissent_retriever=mock_dissent,
            role_rotator=mock_rotator,
            persona_manager=mock_persona,
            flip_detector=mock_flip,
        )

        assert builder.memory == mock_memory
        assert builder.continuum_memory == mock_continuum
        assert builder.dissent_retriever == mock_dissent
        assert builder.role_rotator == mock_rotator
        assert builder.persona_manager == mock_persona
        assert builder.flip_detector == mock_flip


class TestFormatPatternsForPrompt:
    """Tests for format_patterns_for_prompt method."""

    def test_returns_empty_for_empty_patterns(self):
        """Empty patterns return empty string."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )
        result = builder.format_patterns_for_prompt([])
        assert result == ""

    def test_formats_single_pattern(self):
        """Single pattern is formatted correctly."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )
        patterns = [
            {
                "category": "logic",
                "pattern": "Avoid circular reasoning",
                "occurrences": 3,
                "avg_severity": 0.5,
            }
        ]
        result = builder.format_patterns_for_prompt(patterns)
        assert "LEARNED PATTERNS" in result
        assert "LOGIC" in result
        assert "Avoid circular reasoning" in result
        assert "3 past debates" in result

    def test_shows_high_severity_label(self):
        """High severity patterns get labeled."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )
        patterns = [
            {"category": "fact", "pattern": "Check sources", "occurrences": 5, "avg_severity": 0.8}
        ]
        result = builder.format_patterns_for_prompt(patterns)
        assert "[HIGH SEVERITY]" in result

    def test_shows_medium_severity_label(self):
        """Medium severity patterns get labeled."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )
        patterns = [
            {"category": "clarity", "pattern": "Be concise", "occurrences": 2, "avg_severity": 0.5}
        ]
        result = builder.format_patterns_for_prompt(patterns)
        assert "[MEDIUM]" in result

    def test_limits_to_five_patterns(self):
        """Only first 5 patterns are included."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )
        patterns = [
            {
                "category": f"cat{i}",
                "pattern": f"Pattern {i}",
                "occurrences": 1,
                "avg_severity": 0.3,
            }
            for i in range(10)
        ]
        result = builder.format_patterns_for_prompt(patterns)
        assert "Pattern 4" in result
        assert "Pattern 5" not in result


class TestGetStanceGuidance:
    """Tests for get_stance_guidance method."""

    def test_returns_empty_when_stances_disabled(self):
        """Returns empty when asymmetric_stances is False."""
        protocol = DebateProtocol(asymmetric_stances=False)
        builder = PromptBuilder(protocol=protocol, env=Environment(task="Test"))
        agent = MagicMock(stance="affirmative")
        result = builder.get_stance_guidance(agent)
        assert result == ""

    def test_returns_affirmative_guidance(self):
        """Returns affirmative guidance for affirmative stance."""
        protocol = DebateProtocol(asymmetric_stances=True)
        builder = PromptBuilder(protocol=protocol, env=Environment(task="Test"))
        agent = MagicMock(stance="affirmative")
        result = builder.get_stance_guidance(agent)
        assert "AFFIRMATIVE" in result
        assert "DEFEND and SUPPORT" in result

    def test_returns_negative_guidance(self):
        """Returns negative guidance for negative stance."""
        protocol = DebateProtocol(asymmetric_stances=True)
        builder = PromptBuilder(protocol=protocol, env=Environment(task="Test"))
        agent = MagicMock(stance="negative")
        result = builder.get_stance_guidance(agent)
        assert "NEGATIVE" in result
        assert "CHALLENGE and CRITIQUE" in result

    def test_returns_neutral_guidance(self):
        """Returns neutral guidance for neutral stance."""
        protocol = DebateProtocol(asymmetric_stances=True)
        builder = PromptBuilder(protocol=protocol, env=Environment(task="Test"))
        agent = MagicMock(stance="neutral")
        result = builder.get_stance_guidance(agent)
        assert "NEUTRAL" in result
        assert "EVALUATE FAIRLY" in result


class TestGetAgreementIntensityGuidance:
    """Tests for get_agreement_intensity_guidance method."""

    def test_returns_empty_when_intensity_not_set(self):
        """Returns empty when agreement_intensity is None."""
        protocol = DebateProtocol(agreement_intensity=None)
        builder = PromptBuilder(protocol=protocol, env=Environment(task="Test"))
        result = builder.get_agreement_intensity_guidance()
        assert result == ""

    def test_returns_adversarial_for_low_intensity(self):
        """Returns adversarial guidance for low intensity."""
        protocol = DebateProtocol(agreement_intensity=1)
        builder = PromptBuilder(protocol=protocol, env=Environment(task="Test"))
        result = builder.get_agreement_intensity_guidance()
        assert "strongly disagree" in result

    def test_returns_skeptical_for_medium_low_intensity(self):
        """Returns skeptical guidance for medium-low intensity."""
        protocol = DebateProtocol(agreement_intensity=3)
        builder = PromptBuilder(protocol=protocol, env=Environment(task="Test"))
        result = builder.get_agreement_intensity_guidance()
        assert "healthy skepticism" in result

    def test_returns_balanced_for_medium_intensity(self):
        """Returns balanced guidance for medium intensity."""
        protocol = DebateProtocol(agreement_intensity=5)
        builder = PromptBuilder(protocol=protocol, env=Environment(task="Test"))
        result = builder.get_agreement_intensity_guidance()
        assert "merits" in result

    def test_returns_collaborative_for_high_intensity(self):
        """Returns collaborative guidance for high intensity."""
        protocol = DebateProtocol(agreement_intensity=8)
        builder = PromptBuilder(protocol=protocol, env=Environment(task="Test"))
        result = builder.get_agreement_intensity_guidance()
        assert "common ground" in result

    def test_returns_synthesis_for_max_intensity(self):
        """Returns synthesis guidance for max intensity."""
        protocol = DebateProtocol(agreement_intensity=10)
        builder = PromptBuilder(protocol=protocol, env=Environment(task="Test"))
        result = builder.get_agreement_intensity_guidance()
        assert "collaborative synthesis" in result


class TestGetRoleContext:
    """Tests for get_role_context method."""

    def test_returns_empty_without_rotator(self):
        """Returns empty when no role_rotator is set."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
            role_rotator=None,
        )
        agent = MagicMock(name="agent1")
        result = builder.get_role_context(agent)
        assert result == ""

    def test_returns_empty_when_agent_not_assigned(self):
        """Returns empty when agent has no role assignment."""
        mock_rotator = MagicMock()
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
            role_rotator=mock_rotator,
        )
        builder.current_role_assignments = {}
        agent = MagicMock()
        agent.name = "agent1"
        result = builder.get_role_context(agent)
        assert result == ""

    def test_delegates_to_rotator_when_assigned(self):
        """Delegates to role_rotator.format_role_context when agent is assigned."""
        mock_rotator = MagicMock()
        mock_rotator.format_role_context.return_value = "Role: Critic"
        mock_assignment = MagicMock()

        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
            role_rotator=mock_rotator,
        )
        builder.current_role_assignments = {"agent1": mock_assignment}

        agent = MagicMock()
        agent.name = "agent1"
        result = builder.get_role_context(agent)

        mock_rotator.format_role_context.assert_called_once_with(mock_assignment)
        assert result == "Role: Critic"


class TestGetFlipContext:
    """Tests for get_flip_context method."""

    def test_returns_empty_without_detector(self):
        """Returns empty when no flip_detector is set."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
            flip_detector=None,
        )
        agent = MagicMock(name="agent1")
        result = builder.get_flip_context(agent)
        assert result == ""

    def test_returns_empty_when_no_positions(self):
        """Returns empty when agent has no position history."""
        mock_detector = MagicMock()
        mock_consistency = MagicMock(total_positions=0)
        mock_detector.get_agent_consistency.return_value = mock_consistency

        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
            flip_detector=mock_detector,
        )
        agent = MagicMock(name="agent1")
        result = builder.get_flip_context(agent)
        assert result == ""

    def test_returns_empty_when_no_flips(self):
        """Returns empty when agent has no position flips."""
        mock_detector = MagicMock()
        mock_consistency = MagicMock(total_positions=5, total_flips=0)
        mock_detector.get_agent_consistency.return_value = mock_consistency

        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
            flip_detector=mock_detector,
        )
        agent = MagicMock(name="agent1")
        result = builder.get_flip_context(agent)
        assert result == ""

    def test_includes_contradiction_warning(self):
        """Includes contradiction warning when agent has contradictions."""
        mock_detector = MagicMock()
        mock_consistency = MagicMock(
            total_positions=10,
            total_flips=3,
            contradictions=2,
            retractions=0,
            consistency_score=0.8,
            domains_with_flips=[],
        )
        mock_detector.get_agent_consistency.return_value = mock_consistency

        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
            flip_detector=mock_detector,
        )
        agent = MagicMock(name="agent1")
        result = builder.get_flip_context(agent)
        assert "2 prior position contradiction" in result


class TestBuildProposalPrompt:
    """Tests for build_proposal_prompt method."""

    def test_includes_task(self):
        """Proposal prompt includes the task."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="What is the best programming language?"),
        )
        agent = MagicMock(role="proposer", name="agent1")
        result = builder.build_proposal_prompt(agent)
        assert "What is the best programming language?" in result

    def test_includes_agent_role(self):
        """Proposal prompt includes agent role."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test task"),
        )
        agent = MagicMock(role="analyst", name="agent1")
        result = builder.build_proposal_prompt(agent)
        assert "analyst" in result

    def test_includes_context_when_provided(self):
        """Proposal prompt includes context when available."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test task", context="Important background info"),
        )
        agent = MagicMock(role="proposer", name="agent1")
        result = builder.build_proposal_prompt(agent)
        assert "Important background info" in result

    def test_includes_audience_section(self):
        """Proposal prompt includes audience suggestions when provided."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test task"),
        )
        agent = MagicMock(role="proposer", name="agent1")
        result = builder.build_proposal_prompt(
            agent, audience_section="User suggests: Consider ethics"
        )
        assert "User suggests: Consider ethics" in result


class TestBuildRevisionPrompt:
    """Tests for build_revision_prompt method."""

    def test_includes_original_proposal(self):
        """Revision prompt includes original proposal."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test task"),
        )
        agent = MagicMock(role="proposer", name="agent1")
        critiques = []
        result = builder.build_revision_prompt(agent, "My original proposal text", critiques)
        assert "My original proposal text" in result

    def test_includes_critiques(self):
        """Revision prompt includes critiques."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test task"),
        )
        agent = MagicMock(role="proposer", name="agent1")

        mock_critique = MagicMock()
        mock_critique.to_prompt.return_value = "Critique: Missing evidence"

        result = builder.build_revision_prompt(agent, "Original", [mock_critique])
        assert "Critique: Missing evidence" in result

    def test_includes_task(self):
        """Revision prompt includes the original task."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Explain quantum computing"),
        )
        agent = MagicMock(role="proposer", name="agent1")
        result = builder.build_revision_prompt(agent, "Original", [])
        assert "Explain quantum computing" in result


class TestBuildJudgePrompt:
    """Tests for build_judge_prompt method."""

    def test_includes_all_proposals(self):
        """Judge prompt includes all proposals."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )
        proposals = {
            "agent1": "First proposal",
            "agent2": "Second proposal",
        }
        result = builder.build_judge_prompt(proposals, "Test task", [])
        assert "First proposal" in result
        assert "Second proposal" in result
        assert "[agent1]" in result
        assert "[agent2]" in result

    def test_includes_task(self):
        """Judge prompt includes the task."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )
        result = builder.build_judge_prompt({}, "Evaluate AI safety approaches", [])
        assert "Evaluate AI safety approaches" in result

    def test_includes_key_critiques(self):
        """Judge prompt includes key critiques."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )
        mock_critique = MagicMock()
        mock_critique.agent = "critic1"
        mock_critique.issues = ["Issue 1", "Issue 2", "Issue 3"]

        result = builder.build_judge_prompt({}, "Task", [mock_critique])
        assert "critic1" in result
        assert "Issue 1" in result


class TestBuildJudgeVotePrompt:
    """Tests for build_judge_vote_prompt method."""

    def test_includes_candidate_names(self):
        """Vote prompt includes candidate names."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )
        alice = MagicMock()
        alice.name = "alice"
        bob = MagicMock()
        bob.name = "bob"
        candidates = [alice, bob]
        proposals = {"alice": "Proposal A", "bob": "Proposal B"}
        result = builder.build_judge_vote_prompt(candidates, proposals)
        assert "alice" in result
        assert "bob" in result

    def test_includes_proposals_summary(self):
        """Vote prompt includes proposal summaries."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )
        agent1 = MagicMock()
        agent1.name = "agent1"
        candidates = [agent1]
        proposals = {"agent1": "This is my detailed proposal about the topic"}
        result = builder.build_judge_vote_prompt(candidates, proposals)
        assert "This is my detailed proposal" in result

    def test_truncates_long_proposals(self):
        """Vote prompt truncates very long proposals."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )
        agent1 = MagicMock()
        agent1.name = "agent1"
        candidates = [agent1]
        long_proposal = "A" * 500
        proposals = {"agent1": long_proposal}
        result = builder.build_judge_vote_prompt(candidates, proposals)
        # Should be truncated to 300 chars + "..."
        assert "A" * 300 in result
        assert "A" * 301 not in result


class TestClassifyQuestionAsync:
    """Tests for async question classification."""

    @pytest.mark.asyncio
    async def test_classify_question_async_calls_async_classifier(self):
        """classify_question_async should call async classify method."""
        from unittest.mock import AsyncMock
        from dataclasses import dataclass

        @dataclass
        class MockClassification:
            category: str = "technical"
            confidence: float = 0.9
            recommended_personas: list = None

            def __post_init__(self):
                if self.recommended_personas is None:
                    self.recommended_personas = ["scientist"]

        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="What is quantum computing?"),
        )

        # Mock the classifier with async classify method
        mock_classifier = MagicMock()
        mock_classification = MockClassification()
        mock_classifier.classify = AsyncMock(return_value=mock_classification)

        builder._question_classifier = mock_classifier

        result = await builder.classify_question_async(use_llm=True)

        # Should have called the async classify method
        mock_classifier.classify.assert_called_once_with(builder.env.task)
        assert result == "technical"

    @pytest.mark.asyncio
    async def test_classify_question_async_returns_cached(self):
        """classify_question_async should return cached classification."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="Test"),
        )

        # Set cached classification
        mock_classification = MagicMock()
        mock_classification.category = "general"
        builder._classification = mock_classification

        result = await builder.classify_question_async()

        # Should return cached value without calling classifier
        assert result == "general"

    @pytest.mark.asyncio
    async def test_classify_question_async_fallback_on_error(self):
        """classify_question_async should fallback to keyword detection on error."""
        builder = PromptBuilder(
            protocol=DebateProtocol(),
            env=Environment(task="How do I fix my Python code?"),
        )

        # Mock classifier to raise an error
        with patch("asyncio.to_thread", side_effect=Exception("API error")):
            result = await builder.classify_question_async(use_llm=True)

            # Should fallback to keyword-based detection
            assert result in ["general", "technical", "code"]  # Some category from fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
