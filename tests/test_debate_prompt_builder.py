"""
Tests for aragora.debate.prompt_builder module.

Tests PromptBuilder class which handles construction of prompts for
proposals, revisions, and judgments with context injection.
"""

import pytest
from unittest.mock import MagicMock

from aragora.debate.prompt_builder import PromptBuilder
from aragora.core import Environment, Critique


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_protocol():
    """Create a mock DebateProtocol."""
    protocol = MagicMock()
    protocol.asymmetric_stances = False
    protocol.agreement_intensity = None
    return protocol


@pytest.fixture
def mock_environment():
    """Create a mock Environment."""
    env = MagicMock()
    env.task = "Design a secure authentication system"
    env.context = "Web application context"
    return env


@pytest.fixture
def mock_agent():
    """Create a mock Agent."""
    agent = MagicMock()
    agent.name = "claude"
    agent.role = "proposer"
    return agent


@pytest.fixture
def mock_memory():
    """Create a mock CritiqueStore."""
    memory = MagicMock()
    memory.retrieve_patterns = MagicMock(return_value=[])
    return memory


@pytest.fixture
def mock_continuum_memory():
    """Create a mock ContinuumMemory."""
    memory = MagicMock()
    memory.retrieve = MagicMock(return_value=[])
    return memory


@pytest.fixture
def mock_role_rotator():
    """Create a mock RoleRotator."""
    rotator = MagicMock()
    rotator.format_role_context = MagicMock(return_value="Role context")
    return rotator


@pytest.fixture
def mock_persona_manager():
    """Create a mock PersonaManager."""
    manager = MagicMock()
    mock_persona = MagicMock()
    mock_persona.to_prompt_context = MagicMock(return_value="Persona context")
    manager.get_persona = MagicMock(return_value=mock_persona)
    return manager


@pytest.fixture
def mock_flip_detector():
    """Create a mock FlipDetector."""
    detector = MagicMock()
    mock_consistency = MagicMock()
    mock_consistency.total_positions = 0
    mock_consistency.total_flips = 0
    mock_consistency.contradictions = 0
    mock_consistency.retractions = 0
    mock_consistency.consistency_score = 1.0
    mock_consistency.domains_with_flips = []
    detector.get_agent_consistency = MagicMock(return_value=mock_consistency)
    return detector


@pytest.fixture
def mock_evidence_pack():
    """Create a mock EvidencePack."""
    pack = MagicMock()
    snippet = MagicMock()
    snippet.title = "Test Evidence"
    snippet.source = "web"
    snippet.snippet = "Evidence content here"
    snippet.url = "https://example.com"
    snippet.reliability_score = 0.8
    pack.snippets = [snippet]
    return pack


@pytest.fixture
def mock_calibration_tracker():
    """Create a mock CalibrationTracker."""
    tracker = MagicMock()
    mock_summary = MagicMock()
    mock_summary.total_predictions = 10
    mock_summary.brier_score = 0.3
    mock_summary.is_overconfident = True
    mock_summary.is_underconfident = False
    tracker.get_calibration_summary = MagicMock(return_value=mock_summary)
    return tracker


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = MagicMock()
    mock_rating = MagicMock()
    mock_rating.elo = 1500
    mock_rating.wins = 10
    mock_rating.losses = 5
    elo.get_ratings_batch = MagicMock(return_value={"claude": mock_rating})
    return elo


@pytest.fixture
def prompt_builder(mock_protocol, mock_environment):
    """Create a basic PromptBuilder."""
    return PromptBuilder(
        protocol=mock_protocol,
        env=mock_environment,
    )


# ============================================================================
# Initialization Tests
# ============================================================================

class TestPromptBuilderInit:
    """Tests for PromptBuilder initialization."""

    def test_basic_init(self, mock_protocol, mock_environment):
        """Test basic initialization."""
        builder = PromptBuilder(protocol=mock_protocol, env=mock_environment)
        assert builder.protocol == mock_protocol
        assert builder.env == mock_environment
        assert builder.domain == "general"

    def test_init_with_memory(self, mock_protocol, mock_environment, mock_memory):
        """Test initialization with memory."""
        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_environment,
            memory=mock_memory,
        )
        assert builder.memory == mock_memory

    def test_init_with_continuum_memory(self, mock_protocol, mock_environment, mock_continuum_memory):
        """Test initialization with continuum memory."""
        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_environment,
            continuum_memory=mock_continuum_memory,
        )
        assert builder.continuum_memory == mock_continuum_memory

    def test_init_with_domain(self, mock_protocol, mock_environment):
        """Test initialization with custom domain."""
        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_environment,
            domain="security",
        )
        assert builder.domain == "security"

    def test_init_state(self, mock_protocol, mock_environment):
        """Test initial state of internal variables."""
        builder = PromptBuilder(protocol=mock_protocol, env=mock_environment)
        assert builder.current_role_assignments == {}
        assert builder._historical_context_cache == ""
        assert builder._continuum_context_cache == ""
        assert builder.user_suggestions == []


# ============================================================================
# Format Patterns Tests
# ============================================================================

class TestFormatPatternsForPrompt:
    """Tests for format_patterns_for_prompt method."""

    def test_empty_patterns(self, prompt_builder):
        """Test formatting empty patterns list."""
        result = prompt_builder.format_patterns_for_prompt([])
        assert result == ""

    def test_single_pattern(self, prompt_builder):
        """Test formatting single pattern."""
        patterns = [
            {"category": "logic", "pattern": "test pattern", "occurrences": 3}
        ]
        result = prompt_builder.format_patterns_for_prompt(patterns)

        assert "LEARNED PATTERNS" in result
        assert "LOGIC" in result
        assert "test pattern" in result

    def test_high_severity_pattern(self, prompt_builder):
        """Test formatting high severity pattern."""
        patterns = [
            {"category": "security", "pattern": "test", "occurrences": 1, "avg_severity": 0.8}
        ]
        result = prompt_builder.format_patterns_for_prompt(patterns)
        assert "[HIGH SEVERITY]" in result

    def test_medium_severity_pattern(self, prompt_builder):
        """Test formatting medium severity pattern."""
        patterns = [
            {"category": "style", "pattern": "test", "occurrences": 1, "avg_severity": 0.5}
        ]
        result = prompt_builder.format_patterns_for_prompt(patterns)
        assert "[MEDIUM]" in result

    def test_limits_to_5_patterns(self, prompt_builder):
        """Test pattern list limits to 5."""
        patterns = [
            {"category": f"cat-{i}", "pattern": f"pattern {i}", "occurrences": i}
            for i in range(10)
        ]
        result = prompt_builder.format_patterns_for_prompt(patterns)
        # Count category entries
        assert result.count("CAT-") <= 5


# ============================================================================
# Stance Guidance Tests
# ============================================================================

class TestGetStanceGuidance:
    """Tests for get_stance_guidance method."""

    def test_no_asymmetric_stances(self, prompt_builder, mock_agent):
        """Test returns empty when asymmetric_stances disabled."""
        prompt_builder.protocol.asymmetric_stances = False
        result = prompt_builder.get_stance_guidance(mock_agent)
        assert result == ""

    def test_affirmative_stance(self, prompt_builder, mock_agent):
        """Test affirmative stance guidance."""
        prompt_builder.protocol.asymmetric_stances = True
        mock_agent.stance = "affirmative"
        result = prompt_builder.get_stance_guidance(mock_agent)

        assert "AFFIRMATIVE" in result
        assert "DEFEND" in result

    def test_negative_stance(self, prompt_builder, mock_agent):
        """Test negative stance guidance."""
        prompt_builder.protocol.asymmetric_stances = True
        mock_agent.stance = "negative"
        result = prompt_builder.get_stance_guidance(mock_agent)

        assert "NEGATIVE" in result
        assert "CHALLENGE" in result

    def test_neutral_stance(self, prompt_builder, mock_agent):
        """Test neutral stance guidance."""
        prompt_builder.protocol.asymmetric_stances = True
        mock_agent.stance = "neutral"
        result = prompt_builder.get_stance_guidance(mock_agent)

        assert "NEUTRAL" in result
        assert "EVALUATE" in result


# ============================================================================
# Agreement Intensity Guidance Tests
# ============================================================================

class TestGetAgreementIntensityGuidance:
    """Tests for get_agreement_intensity_guidance method."""

    def test_no_intensity_set(self, prompt_builder):
        """Test returns empty when intensity is None."""
        prompt_builder.protocol.agreement_intensity = None
        result = prompt_builder.get_agreement_intensity_guidance()
        assert result == ""

    def test_very_low_intensity(self, prompt_builder):
        """Test very low intensity (adversarial)."""
        prompt_builder.protocol.agreement_intensity = 1
        result = prompt_builder.get_agreement_intensity_guidance()
        assert "strongly disagree" in result

    def test_low_intensity(self, prompt_builder):
        """Test low intensity (skeptical)."""
        prompt_builder.protocol.agreement_intensity = 2
        result = prompt_builder.get_agreement_intensity_guidance()
        assert "skepticism" in result

    def test_medium_intensity(self, prompt_builder):
        """Test medium intensity (balanced)."""
        prompt_builder.protocol.agreement_intensity = 5
        result = prompt_builder.get_agreement_intensity_guidance()
        assert "merits" in result

    def test_high_intensity(self, prompt_builder):
        """Test high intensity (cooperative)."""
        prompt_builder.protocol.agreement_intensity = 7
        result = prompt_builder.get_agreement_intensity_guidance()
        assert "common ground" in result

    def test_very_high_intensity(self, prompt_builder):
        """Test very high intensity (collaborative)."""
        prompt_builder.protocol.agreement_intensity = 10
        result = prompt_builder.get_agreement_intensity_guidance()
        assert "collaborative" in result


# ============================================================================
# Role Context Tests
# ============================================================================

class TestGetRoleContext:
    """Tests for get_role_context method."""

    def test_no_role_rotator(self, prompt_builder, mock_agent):
        """Test returns empty without role_rotator."""
        result = prompt_builder.get_role_context(mock_agent)
        assert result == ""

    def test_no_assignment(self, prompt_builder, mock_agent, mock_role_rotator):
        """Test returns empty when agent has no assignment."""
        prompt_builder.role_rotator = mock_role_rotator
        prompt_builder.current_role_assignments = {}
        result = prompt_builder.get_role_context(mock_agent)
        assert result == ""

    def test_with_assignment(self, prompt_builder, mock_agent, mock_role_rotator):
        """Test returns role context when assigned."""
        prompt_builder.role_rotator = mock_role_rotator
        mock_assignment = MagicMock()
        prompt_builder.current_role_assignments = {"claude": mock_assignment}

        result = prompt_builder.get_role_context(mock_agent)

        assert result == "Role context"
        mock_role_rotator.format_role_context.assert_called_with(mock_assignment)


# ============================================================================
# Persona Context Tests
# ============================================================================

class TestGetPersonaContext:
    """Tests for get_persona_context method."""

    def test_no_persona_manager(self, prompt_builder, mock_agent):
        """Test returns empty without persona_manager."""
        result = prompt_builder.get_persona_context(mock_agent)
        assert result == ""

    def test_with_persona(self, prompt_builder, mock_agent, mock_persona_manager):
        """Test returns persona context when available."""
        prompt_builder.persona_manager = mock_persona_manager
        result = prompt_builder.get_persona_context(mock_agent)
        assert result == "Persona context"

    def test_no_persona_returns_empty(self, prompt_builder, mock_agent, mock_persona_manager):
        """Test returns empty when no persona found."""
        mock_persona_manager.get_persona = MagicMock(return_value=None)
        prompt_builder.persona_manager = mock_persona_manager
        result = prompt_builder.get_persona_context(mock_agent)
        # May return default persona or empty
        assert isinstance(result, str)


# ============================================================================
# Flip Context Tests
# ============================================================================

class TestGetFlipContext:
    """Tests for get_flip_context method."""

    def test_no_flip_detector(self, prompt_builder, mock_agent):
        """Test returns empty without flip_detector."""
        result = prompt_builder.get_flip_context(mock_agent)
        assert result == ""

    def test_no_positions_yet(self, prompt_builder, mock_agent, mock_flip_detector):
        """Test returns empty when no position history."""
        prompt_builder.flip_detector = mock_flip_detector
        result = prompt_builder.get_flip_context(mock_agent)
        assert result == ""

    def test_no_flips(self, prompt_builder, mock_agent, mock_flip_detector):
        """Test returns empty when no flips."""
        mock_consistency = MagicMock()
        mock_consistency.total_positions = 5
        mock_consistency.total_flips = 0
        mock_flip_detector.get_agent_consistency = MagicMock(return_value=mock_consistency)

        prompt_builder.flip_detector = mock_flip_detector
        result = prompt_builder.get_flip_context(mock_agent)
        assert result == ""

    def test_with_contradictions(self, prompt_builder, mock_agent, mock_flip_detector):
        """Test returns context when contradictions exist."""
        mock_consistency = MagicMock()
        mock_consistency.total_positions = 5
        mock_consistency.total_flips = 2
        mock_consistency.contradictions = 1
        mock_consistency.retractions = 1
        mock_consistency.consistency_score = 0.6
        mock_consistency.domains_with_flips = ["security"]
        mock_flip_detector.get_agent_consistency = MagicMock(return_value=mock_consistency)

        prompt_builder.flip_detector = mock_flip_detector
        result = prompt_builder.get_flip_context(mock_agent)

        assert "Position Consistency" in result
        assert "contradiction" in result


# ============================================================================
# Continuum Context Tests
# ============================================================================

class TestGetContinuumContext:
    """Tests for get_continuum_context method."""

    def test_empty_cache(self, prompt_builder):
        """Test returns empty when cache is empty."""
        result = prompt_builder.get_continuum_context()
        assert result == ""

    def test_returns_cached_context(self, prompt_builder):
        """Test returns cached context."""
        prompt_builder._continuum_context_cache = "Cached continuum context"
        result = prompt_builder.get_continuum_context()
        assert result == "Cached continuum context"


# ============================================================================
# Belief Context Tests
# ============================================================================

class TestInjectBeliefContext:
    """Tests for _inject_belief_context method."""

    def test_no_continuum_memory(self, prompt_builder):
        """Test returns empty without continuum_memory."""
        result = prompt_builder._inject_belief_context()
        assert result == ""

    def test_no_memories(self, prompt_builder, mock_continuum_memory):
        """Test returns empty when no memories found."""
        mock_continuum_memory.retrieve = MagicMock(return_value=[])
        prompt_builder.continuum_memory = mock_continuum_memory
        result = prompt_builder._inject_belief_context()
        assert result == ""

    def test_with_crux_claims(self, prompt_builder, mock_continuum_memory):
        """Test returns belief context with crux claims."""
        mock_memory = MagicMock()
        mock_memory.metadata = {"crux_claims": ["crux1", "crux2"]}
        mock_continuum_memory.retrieve = MagicMock(return_value=[mock_memory])

        prompt_builder.continuum_memory = mock_continuum_memory
        result = prompt_builder._inject_belief_context()

        assert "Historical Disagreement Points" in result
        assert "crux1" in result


# ============================================================================
# Calibration Context Tests
# ============================================================================

class TestInjectCalibrationContext:
    """Tests for _inject_calibration_context method."""

    def test_no_calibration_tracker(self, prompt_builder, mock_agent):
        """Test returns empty without calibration_tracker."""
        result = prompt_builder._inject_calibration_context(mock_agent)
        assert result == ""

    def test_too_few_predictions(self, prompt_builder, mock_agent, mock_calibration_tracker):
        """Test returns empty with too few predictions."""
        mock_summary = MagicMock()
        mock_summary.total_predictions = 3
        mock_calibration_tracker.get_calibration_summary = MagicMock(return_value=mock_summary)

        prompt_builder.calibration_tracker = mock_calibration_tracker
        result = prompt_builder._inject_calibration_context(mock_agent)
        assert result == ""

    def test_well_calibrated(self, prompt_builder, mock_agent, mock_calibration_tracker):
        """Test returns empty for well-calibrated agents."""
        mock_summary = MagicMock()
        mock_summary.total_predictions = 10
        mock_summary.brier_score = 0.2  # Good calibration
        mock_calibration_tracker.get_calibration_summary = MagicMock(return_value=mock_summary)

        prompt_builder.calibration_tracker = mock_calibration_tracker
        result = prompt_builder._inject_calibration_context(mock_agent)
        assert result == ""

    def test_overconfident(self, prompt_builder, mock_agent, mock_calibration_tracker):
        """Test returns feedback for overconfident agent."""
        prompt_builder.calibration_tracker = mock_calibration_tracker
        result = prompt_builder._inject_calibration_context(mock_agent)

        assert "Calibration Feedback" in result
        assert "OVERCONFIDENT" in result


# ============================================================================
# ELO Context Tests
# ============================================================================

class TestGetEloContext:
    """Tests for get_elo_context method."""

    def test_no_elo_system(self, prompt_builder, mock_agent):
        """Test returns empty without elo_system."""
        result = prompt_builder.get_elo_context(mock_agent, [mock_agent])
        assert result == ""

    def test_no_ratings(self, prompt_builder, mock_agent, mock_elo_system):
        """Test returns empty when no ratings."""
        mock_elo_system.get_ratings_batch = MagicMock(return_value={})
        prompt_builder.elo_system = mock_elo_system

        result = prompt_builder.get_elo_context(mock_agent, [mock_agent])
        assert result == ""

    def test_with_ratings(self, prompt_builder, mock_agent, mock_elo_system):
        """Test returns ELO context with ratings."""
        prompt_builder.elo_system = mock_elo_system
        result = prompt_builder.get_elo_context(mock_agent, [mock_agent])

        assert "Agent Rankings" in result
        assert "claude" in result
        assert "1500" in result


# ============================================================================
# Evidence Format Tests
# ============================================================================

class TestFormatEvidenceForPrompt:
    """Tests for format_evidence_for_prompt method."""

    def test_no_evidence_pack(self, prompt_builder):
        """Test returns empty without evidence_pack."""
        result = prompt_builder.format_evidence_for_prompt()
        assert result == ""

    def test_empty_snippets(self, prompt_builder, mock_evidence_pack):
        """Test returns empty with no snippets."""
        mock_evidence_pack.snippets = []
        prompt_builder.evidence_pack = mock_evidence_pack
        result = prompt_builder.format_evidence_for_prompt()
        assert result == ""

    def test_with_evidence(self, prompt_builder, mock_evidence_pack):
        """Test returns formatted evidence."""
        prompt_builder.evidence_pack = mock_evidence_pack
        result = prompt_builder.format_evidence_for_prompt()

        assert "AVAILABLE EVIDENCE" in result
        assert "[EVID-1]" in result
        assert "Test Evidence" in result
        assert "80%" in result


# ============================================================================
# Set Evidence Pack Tests
# ============================================================================

class TestSetEvidencePack:
    """Tests for set_evidence_pack method."""

    def test_set_evidence_pack(self, prompt_builder, mock_evidence_pack):
        """Test setting evidence pack."""
        prompt_builder.set_evidence_pack(mock_evidence_pack)
        assert prompt_builder.evidence_pack == mock_evidence_pack

    def test_set_none(self, prompt_builder):
        """Test setting None evidence pack."""
        prompt_builder.set_evidence_pack(None)
        assert prompt_builder.evidence_pack is None


# ============================================================================
# Build Proposal Prompt Tests
# ============================================================================

class TestBuildProposalPrompt:
    """Tests for build_proposal_prompt method."""

    def test_basic_proposal(self, prompt_builder, mock_agent):
        """Test basic proposal prompt generation."""
        result = prompt_builder.build_proposal_prompt(mock_agent)

        assert "multi-agent debate" in result
        assert mock_agent.role in result
        assert prompt_builder.env.task in result

    def test_with_context(self, prompt_builder, mock_agent):
        """Test proposal prompt with environment context."""
        result = prompt_builder.build_proposal_prompt(mock_agent)
        assert "Context:" in result

    def test_without_research_context(self, prompt_builder, mock_agent):
        """Test proposal prompt shows research status when no research."""
        prompt_builder.env.context = None
        result = prompt_builder.build_proposal_prompt(mock_agent)
        assert "RESEARCH STATUS" in result

    def test_with_audience_section(self, prompt_builder, mock_agent):
        """Test proposal prompt with audience suggestions."""
        result = prompt_builder.build_proposal_prompt(
            mock_agent,
            audience_section="User suggested: test suggestion",
        )
        assert "User suggested: test suggestion" in result

    def test_with_all_agents(self, prompt_builder, mock_agent, mock_elo_system):
        """Test proposal prompt with all agents for ELO context."""
        prompt_builder.elo_system = mock_elo_system
        result = prompt_builder.build_proposal_prompt(
            mock_agent,
            all_agents=[mock_agent],
        )
        # May include ELO context
        assert isinstance(result, str)


# ============================================================================
# Build Revision Prompt Tests
# ============================================================================

class TestBuildRevisionPrompt:
    """Tests for build_revision_prompt method."""

    def test_basic_revision(self, prompt_builder, mock_agent):
        """Test basic revision prompt generation."""
        critique = Critique(
            agent="critic",
            target_agent="claude",
            target_content="original content",
            issues=["Issue 1"],
            suggestions=["Suggestion 1"],
            severity=0.5,
            reasoning="Test reasoning",
        )

        result = prompt_builder.build_revision_prompt(
            mock_agent,
            original="Original proposal",
            critiques=[critique],
        )

        assert "revising your proposal" in result
        assert "Original proposal" in result
        assert "Critiques Received" in result

    def test_with_audience_section(self, prompt_builder, mock_agent):
        """Test revision prompt with audience suggestions."""
        result = prompt_builder.build_revision_prompt(
            mock_agent,
            original="Original",
            critiques=[],
            audience_section="User feedback: good point",
        )
        assert "User feedback" in result


# ============================================================================
# Build Judge Prompt Tests
# ============================================================================

class TestBuildJudgePrompt:
    """Tests for build_judge_prompt method."""

    def test_basic_judge_prompt(self, prompt_builder):
        """Test basic judge/synthesizer prompt generation."""
        result = prompt_builder.build_judge_prompt(
            proposals={"claude": "Proposal A", "gpt": "Proposal B"},
            task="Test task",
            critiques=[],
        )

        assert "synthesizer" in result
        assert "Test task" in result
        assert "Proposal A" in result
        assert "Proposal B" in result


# ============================================================================
# Build Judge Vote Prompt Tests
# ============================================================================

class TestBuildJudgeVotePrompt:
    """Tests for build_judge_vote_prompt method."""

    def test_basic_vote_prompt(self, prompt_builder, mock_agent):
        """Test basic judge vote prompt generation."""
        result = prompt_builder.build_judge_vote_prompt(
            candidates=[mock_agent],
            proposals={"claude": "Test proposal"},
        )

        assert "vote" in result
        assert "claude" in result
        assert "cannot vote for yourself" in result


# ============================================================================
# Format Successful Patterns Tests
# ============================================================================

class TestFormatSuccessfulPatterns:
    """Tests for format_successful_patterns method."""

    def test_no_memory(self, prompt_builder):
        """Test returns empty without memory."""
        result = prompt_builder.format_successful_patterns()
        assert result == ""

    def test_no_patterns(self, prompt_builder, mock_memory):
        """Test returns empty when no patterns found."""
        mock_memory.retrieve_patterns = MagicMock(return_value=[])
        prompt_builder.memory = mock_memory
        result = prompt_builder.format_successful_patterns()
        assert result == ""

    def test_with_patterns(self, prompt_builder, mock_memory):
        """Test returns formatted patterns."""
        mock_pattern = MagicMock()
        mock_pattern.issue_type = "logic"
        mock_pattern.issue_text = "Test issue"
        mock_pattern.suggestion_text = "Test suggestion"
        mock_pattern.success_count = 5
        mock_memory.retrieve_patterns = MagicMock(return_value=[mock_pattern])

        prompt_builder.memory = mock_memory
        result = prompt_builder.format_successful_patterns()

        assert "SUCCESSFUL PATTERNS" in result
        assert "logic" in result


# ============================================================================
# Integration Tests
# ============================================================================

class TestPromptBuilderIntegration:
    """Integration tests for PromptBuilder."""

    def test_full_proposal_with_all_contexts(
        self,
        mock_protocol,
        mock_environment,
        mock_memory,
        mock_persona_manager,
        mock_flip_detector,
        mock_evidence_pack,
        mock_elo_system,
        mock_agent,
    ):
        """Test proposal prompt with all context types enabled."""
        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_environment,
            memory=mock_memory,
            persona_manager=mock_persona_manager,
            flip_detector=mock_flip_detector,
            evidence_pack=mock_evidence_pack,
            elo_system=mock_elo_system,
        )

        result = builder.build_proposal_prompt(
            mock_agent,
            all_agents=[mock_agent],
        )

        assert "multi-agent debate" in result
        assert len(result) > 200  # Should be substantial

    def test_revision_with_multiple_critiques(self, prompt_builder, mock_agent):
        """Test revision prompt with multiple critiques."""
        critiques = [
            Critique(
                agent=f"critic-{i}",
                target_agent="claude",
                target_content="content",
                issues=[f"Issue {i}"],
                suggestions=[f"Suggestion {i}"],
                severity=0.5,
                reasoning=f"Reasoning {i}",
            )
            for i in range(3)
        ]

        result = prompt_builder.build_revision_prompt(
            mock_agent,
            original="Original",
            critiques=critiques,
        )

        assert "Issue 0" in result
        assert "Issue 1" in result
        assert "Issue 2" in result
