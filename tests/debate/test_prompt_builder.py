"""
Tests for PromptBuilder - debate prompt construction.

Tests cover:
- Initialization with various dependencies
- Proposal prompt building
- Revision prompt building
- Judge prompt building
- Stance and role guidance
- Evidence formatting
- Trending topics injection
- Persona and calibration context
- Question classification
- ELO context
- RLM context integration
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.prompt_builder import PromptBuilder


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "claude", role: str = "proposer", stance: str = None):
        self.name = name
        self.role = role
        self.model = "claude-3-opus"
        if stance:
            self.stance = stance


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, task: str = "Test task", context: str = None):
        self.task = task
        self.context = context


class MockProtocol:
    """Mock debate protocol for testing."""

    def __init__(
        self,
        rounds: int = 3,
        asymmetric_stances: bool = False,
        agreement_intensity: int = None,
        enable_trending_injection: bool = False,
        trending_injection_max_topics: int = 3,
        trending_relevance_filter: bool = True,
    ):
        self.rounds = rounds
        self.asymmetric_stances = asymmetric_stances
        self.agreement_intensity = agreement_intensity
        self.enable_trending_injection = enable_trending_injection
        self.trending_injection_max_topics = trending_injection_max_topics
        self.trending_relevance_filter = trending_relevance_filter
        self.language = None
        self.require_evidence = False
        self.require_uncertainty = False
        self.consensus = "majority"

    def get_round_phase(self, round_number: int):
        """Return None for testing (no structured phases)."""
        return None


class MockCritique:
    """Mock critique for testing."""

    def __init__(self, agent: str = "critic", issues: list = None, suggestions: list = None):
        self.agent = agent
        self.issues = issues or ["Issue 1", "Issue 2"]
        self.suggestions = suggestions or []

    def to_prompt(self) -> str:
        return f"[{self.agent}]: {', '.join(self.issues)}"


@pytest.fixture
def mock_protocol():
    """Create mock debate protocol."""
    return MockProtocol()


@pytest.fixture
def mock_env():
    """Create mock environment."""
    return MockEnvironment(
        task="Discuss the best approach to API design", context="We are building a REST API"
    )


@pytest.fixture
def mock_agent():
    """Create mock agent."""
    return MockAgent(name="claude")


@pytest.fixture
def basic_builder(mock_protocol, mock_env):
    """Create basic PromptBuilder instance."""
    return PromptBuilder(protocol=mock_protocol, env=mock_env)


class TestPromptBuilderInitialization:
    """Tests for PromptBuilder initialization."""

    def test_init_basic(self, mock_protocol, mock_env):
        """Test basic initialization."""
        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)

        assert builder.protocol is mock_protocol
        assert builder.env is mock_env
        assert builder.memory is None
        assert builder.continuum_memory is None

    def test_init_with_memory(self, mock_protocol, mock_env):
        """Test initialization with critique memory."""
        mock_memory = MagicMock()
        builder = PromptBuilder(protocol=mock_protocol, env=mock_env, memory=mock_memory)

        assert builder.memory is mock_memory

    def test_init_with_all_dependencies(self, mock_protocol, mock_env):
        """Test initialization with all dependencies."""
        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            memory=MagicMock(),
            continuum_memory=MagicMock(),
            dissent_retriever=MagicMock(),
            role_rotator=MagicMock(),
            persona_manager=MagicMock(),
            flip_detector=MagicMock(),
            evidence_pack=MagicMock(),
            calibration_tracker=MagicMock(),
            elo_system=MagicMock(),
            domain="security",
        )

        assert builder.domain == "security"
        assert builder.persona_manager is not None
        assert builder.flip_detector is not None


class TestStanceGuidance:
    """Tests for stance guidance generation."""

    def test_get_stance_guidance_no_asymmetric(self, mock_agent):
        """Test stance guidance when asymmetric stances disabled."""
        protocol = MockProtocol(asymmetric_stances=False)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment())

        result = builder.get_stance_guidance(mock_agent)

        assert result == ""

    def test_get_stance_guidance_affirmative(self):
        """Test affirmative stance guidance."""
        protocol = MockProtocol(asymmetric_stances=True)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment())
        agent = MockAgent(stance="affirmative")

        result = builder.get_stance_guidance(agent)

        assert "AFFIRMATIVE" in result
        assert "DEFEND" in result or "SUPPORT" in result

    def test_get_stance_guidance_negative(self):
        """Test negative stance guidance."""
        protocol = MockProtocol(asymmetric_stances=True)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment())
        agent = MockAgent(stance="negative")

        result = builder.get_stance_guidance(agent)

        assert "NEGATIVE" in result
        assert "CHALLENGE" in result or "CRITIQUE" in result

    def test_get_stance_guidance_neutral(self):
        """Test neutral stance guidance."""
        protocol = MockProtocol(asymmetric_stances=True)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment())
        agent = MockAgent(stance="neutral")

        result = builder.get_stance_guidance(agent)

        assert "NEUTRAL" in result


class TestAgreementIntensityGuidance:
    """Tests for agreement intensity guidance."""

    def test_intensity_none(self):
        """Test guidance when intensity not set."""
        protocol = MockProtocol(agreement_intensity=None)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment())

        result = builder.get_agreement_intensity_guidance()

        assert result == ""

    def test_intensity_low(self):
        """Test adversarial intensity guidance."""
        protocol = MockProtocol(agreement_intensity=1)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment())

        result = builder.get_agreement_intensity_guidance()

        assert "disagree" in result.lower() or "adversarial" in result.lower()

    def test_intensity_medium(self):
        """Test balanced intensity guidance."""
        protocol = MockProtocol(agreement_intensity=5)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment())

        result = builder.get_agreement_intensity_guidance()

        assert "merit" in result.lower()

    def test_intensity_high(self):
        """Test collaborative intensity guidance."""
        protocol = MockProtocol(agreement_intensity=9)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment())

        result = builder.get_agreement_intensity_guidance()

        assert "collaborative" in result.lower() or "synthesis" in result.lower()


class TestRoleContext:
    """Tests for cognitive role context."""

    def test_get_role_context_no_rotator(self, basic_builder, mock_agent):
        """Test role context without role rotator."""
        result = basic_builder.get_role_context(mock_agent)

        assert result == ""

    def test_get_role_context_with_rotator_no_assignment(self):
        """Test role context when agent has no assignment."""
        mock_rotator = MagicMock()
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(),
            role_rotator=mock_rotator,
        )
        builder.current_role_assignments = {}
        agent = MockAgent()

        result = builder.get_role_context(agent)

        assert result == ""

    def test_get_role_context_with_assignment(self):
        """Test role context with valid assignment."""
        mock_rotator = MagicMock()
        mock_rotator.format_role_context.return_value = "You are the SKEPTIC"

        assignment = MagicMock()
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(),
            role_rotator=mock_rotator,
        )
        builder.current_role_assignments = {"claude": assignment}
        agent = MockAgent()

        result = builder.get_role_context(agent)

        assert result == "You are the SKEPTIC"


class TestRoundPhaseContext:
    """Tests for round phase context."""

    def test_round_phase_no_phase(self, basic_builder):
        """Test context when no phase defined."""
        result = basic_builder.get_round_phase_context(round_number=1)

        # MockProtocol.get_round_phase returns None
        assert result == ""


class TestQuestionClassification:
    """Tests for question classification."""

    @pytest.mark.asyncio
    async def test_classify_question_async(self, basic_builder):
        """Test async question classification."""
        result = await basic_builder.classify_question_async(use_llm=False)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_detect_domain_philosophical(self):
        """Test philosophical domain detection."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="What is the meaning of life?"),
        )

        result = builder._detect_question_domain_keywords("What is the meaning of life?")

        assert result == "philosophical"

    def test_detect_domain_ethics(self):
        """Test ethics domain detection."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Should we allow AI?"),
        )

        result = builder._detect_question_domain_keywords("Should we allow AI?")

        assert result == "ethics"

    def test_detect_domain_technical(self):
        """Test technical domain detection."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="How to design the API?"),
        )

        result = builder._detect_question_domain_keywords("How to design the API?")

        assert result == "technical"


class TestPersonaContext:
    """Tests for persona context injection."""

    def test_get_persona_context_philosophical(self):
        """Test persona context for philosophical questions."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="What is the meaning of life?"),
        )
        agent = MockAgent()

        result = builder.get_persona_context(agent)

        assert "human condition" in result.lower() or "philosophy" in result.lower()

    def test_get_persona_context_ethics(self):
        """Test persona context for ethics questions."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Should we allow AI in healthcare?"),
        )
        agent = MockAgent()

        result = builder.get_persona_context(agent)

        assert "ethical" in result.lower() or "moral" in result.lower()

    def test_get_persona_context_technical_with_manager(self):
        """Test persona context with manager for technical questions."""
        mock_persona = MagicMock()
        mock_persona.to_prompt_context.return_value = "Security expert context"

        mock_manager = MagicMock()
        mock_manager.get_persona.return_value = mock_persona

        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="How to design the API endpoint?"),
            persona_manager=mock_manager,
        )
        agent = MockAgent()

        result = builder.get_persona_context(agent)

        assert result == "Security expert context"


class TestFlipContext:
    """Tests for position flip detection context."""

    def test_get_flip_context_no_detector(self, basic_builder, mock_agent):
        """Test flip context without detector."""
        result = basic_builder.get_flip_context(mock_agent)

        assert result == ""

    def test_get_flip_context_no_flips(self):
        """Test flip context when no flips detected."""
        mock_detector = MagicMock()
        consistency = MagicMock()
        consistency.total_positions = 5
        consistency.total_flips = 0
        mock_detector.get_agent_consistency.return_value = consistency

        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(),
            flip_detector=mock_detector,
        )
        agent = MockAgent()

        result = builder.get_flip_context(agent)

        assert result == ""

    def test_get_flip_context_with_contradictions(self):
        """Test flip context with contradictions."""
        mock_detector = MagicMock()
        consistency = MagicMock()
        consistency.total_positions = 10
        consistency.total_flips = 3
        consistency.contradictions = 2
        consistency.retractions = 1
        consistency.consistency_score = 0.5
        consistency.domains_with_flips = ["ethics"]
        mock_detector.get_agent_consistency.return_value = consistency

        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(),
            flip_detector=mock_detector,
        )
        agent = MockAgent()

        result = builder.get_flip_context(agent)

        assert "contradiction" in result.lower()


class TestEvidenceFormatting:
    """Tests for evidence formatting."""

    def test_format_evidence_no_pack(self, basic_builder):
        """Test evidence formatting without evidence pack."""
        result = basic_builder.format_evidence_for_prompt(max_snippets=5)

        assert result == ""

    def test_format_evidence_with_pack(self, mock_protocol, mock_env):
        """Test evidence formatting with evidence pack."""
        mock_snippet = MagicMock()
        mock_snippet.title = "Test Source"
        mock_snippet.source = "web"
        mock_snippet.reliability_score = 0.8
        mock_snippet.url = "https://example.com"
        mock_snippet.snippet = "Evidence content"

        mock_pack = MagicMock()
        mock_pack.snippets = [mock_snippet]

        builder = PromptBuilder(protocol=mock_protocol, env=mock_env, evidence_pack=mock_pack)
        result = builder.format_evidence_for_prompt(max_snippets=5)

        assert "EVID-1" in result
        assert "Test Source" in result

    def test_set_evidence_pack(self, basic_builder):
        """Test setting evidence pack."""
        mock_pack = MagicMock()
        basic_builder.set_evidence_pack(mock_pack)

        assert basic_builder.evidence_pack is mock_pack


class TestTrendingTopicsFormatting:
    """Tests for trending topics formatting."""

    def test_format_trending_disabled(self):
        """Test trending formatting when disabled."""
        protocol = MockProtocol(enable_trending_injection=False)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment())

        result = builder.format_trending_for_prompt()

        assert result == ""

    def test_format_trending_empty(self):
        """Test trending formatting with no topics."""
        protocol = MockProtocol(enable_trending_injection=True)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment())

        result = builder.format_trending_for_prompt()

        assert result == ""

    def test_format_trending_with_topics(self):
        """Test trending formatting with topics."""
        protocol = MockProtocol(enable_trending_injection=True)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment())

        mock_topic = MagicMock()
        mock_topic.topic = "AI Safety News"
        mock_topic.platform = "hackernews"
        mock_topic.volume = 5000
        mock_topic.category = "tech"

        builder.set_trending_topics([mock_topic])

        result = builder.format_trending_for_prompt()

        assert "AI Safety News" in result
        assert "TRENDING" in result

    def test_set_trending_topics(self, basic_builder):
        """Test setting trending topics."""
        mock_topics = [MagicMock()]
        basic_builder.set_trending_topics(mock_topics)

        assert len(basic_builder.trending_topics) == 1


class TestCalibrationContext:
    """Tests for calibration context injection."""

    def test_inject_calibration_context_no_tracker(self, basic_builder, mock_agent):
        """Test calibration context without tracker."""
        result = basic_builder._inject_calibration_context(mock_agent)

        assert result == ""

    def test_inject_calibration_context_insufficient_data(self):
        """Test calibration with insufficient predictions."""
        mock_tracker = MagicMock()
        summary = MagicMock()
        summary.total_predictions = 3  # Less than 5
        mock_tracker.get_calibration_summary.return_value = summary

        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(),
            calibration_tracker=mock_tracker,
        )
        agent = MockAgent()

        result = builder._inject_calibration_context(agent)

        assert result == ""

    def test_inject_calibration_context_well_calibrated(self):
        """Test calibration for well-calibrated agent."""
        mock_tracker = MagicMock()
        summary = MagicMock()
        summary.total_predictions = 10
        summary.brier_score = 0.1  # Good score, below 0.25
        mock_tracker.get_calibration_summary.return_value = summary

        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(),
            calibration_tracker=mock_tracker,
        )
        agent = MockAgent()

        result = builder._inject_calibration_context(agent)

        assert result == ""  # No feedback for well-calibrated agents


class TestEloContext:
    """Tests for ELO ranking context."""

    def test_get_elo_context_no_system(self, basic_builder, mock_agent):
        """Test ELO context without system."""
        all_agents = [mock_agent]
        result = basic_builder.get_elo_context(mock_agent, all_agents)

        assert result == ""

    def test_get_elo_context_with_system(self):
        """Test ELO context with system."""
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.elo = 1600
        mock_rating.wins = 10
        mock_rating.losses = 5
        mock_rating.calibration_score = 0.7
        mock_elo.get_ratings_batch.return_value = {"claude": mock_rating}

        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(),
            elo_system=mock_elo,
            domain="general",
        )
        agent = MockAgent()

        result = builder.get_elo_context(agent, [agent])

        assert "Rankings" in result or "ELO" in result


class TestRLMContext:
    """Tests for RLM context integration."""

    def test_set_rlm_context(self, basic_builder):
        """Test setting RLM context."""
        mock_context = MagicMock()
        basic_builder.set_rlm_context(mock_context)

        assert basic_builder._rlm_context is mock_context

    def test_get_rlm_context_hint_no_context(self, basic_builder):
        """Test RLM hint without context."""
        result = basic_builder.get_rlm_context_hint()

        assert result == ""

    def test_get_rlm_abstract_no_context(self, basic_builder):
        """Test RLM abstract without context."""
        result = basic_builder.get_rlm_abstract(max_chars=2000)

        assert result == ""


class TestLanguageConstraint:
    """Tests for language constraint."""

    def test_get_language_constraint_default_english(self):
        """Test language constraint defaults to English when no language set."""
        protocol = MockProtocol()
        protocol.language = None  # No language specified

        builder = PromptBuilder(
            protocol=protocol,
            env=MockEnvironment(),
        )

        result = builder.get_language_constraint()
        # Default behavior returns English constraint
        assert "English" in result or result == ""

    def test_get_language_constraint_with_language(self):
        """Test language constraint with language specified."""
        protocol = MockProtocol()
        protocol.language = "Japanese"

        builder = PromptBuilder(
            protocol=protocol,
            env=MockEnvironment(),
        )

        result = builder.get_language_constraint()
        # Should contain specified language instruction
        assert "Japanese" in result


class TestBuildProposalPrompt:
    """Tests for proposal prompt building."""

    def test_build_proposal_prompt_basic(self, mock_agent):
        """Test basic proposal prompt."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Design a rate limiter"),
        )

        result = builder.build_proposal_prompt(agent=mock_agent)

        assert isinstance(result, str)
        assert "proposer" in result
        assert "Design a rate limiter" in result

    def test_build_proposal_prompt_with_context(self, mock_agent):
        """Test proposal prompt with environment context."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Test task", context="Background info"),
        )

        result = builder.build_proposal_prompt(agent=mock_agent)

        assert "Background info" in result

    def test_build_proposal_prompt_with_stance(self):
        """Test proposal prompt with stance guidance."""
        protocol = MockProtocol(asymmetric_stances=True)
        builder = PromptBuilder(
            protocol=protocol,
            env=MockEnvironment(task="Test task"),
        )
        agent = MockAgent(stance="affirmative")

        result = builder.build_proposal_prompt(agent=agent)

        assert "AFFIRMATIVE" in result

    def test_build_proposal_prompt_with_audience(self, mock_agent):
        """Test proposal prompt with audience section."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Test task"),
        )

        result = builder.build_proposal_prompt(
            agent=mock_agent,
            audience_section="Consider user input",
        )

        assert "Consider user input" in result

    def test_build_proposal_prompt_with_all_agents(self, mock_agent):
        """Test proposal prompt with all agents for ELO context."""
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.elo = 1600
        mock_rating.wins = 10
        mock_rating.losses = 5
        mock_rating.calibration_score = 0.7
        mock_elo.get_ratings_batch.return_value = {"claude": mock_rating}

        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Test task"),
            elo_system=mock_elo,
        )

        result = builder.build_proposal_prompt(
            agent=mock_agent,
            all_agents=[mock_agent],
        )

        assert isinstance(result, str)


class TestBuildRevisionPrompt:
    """Tests for revision prompt building."""

    def test_build_revision_prompt_basic(self, mock_agent):
        """Test basic revision prompt."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Test task"),
        )
        critiques = [MockCritique()]

        result = builder.build_revision_prompt(
            agent=mock_agent,
            original="Original proposal text",
            critiques=critiques,
        )

        assert "revising" in result.lower()
        assert "Original proposal text" in result
        assert "Issue 1" in result

    def test_build_revision_prompt_with_intensity(self, mock_agent):
        """Test revision prompt with agreement intensity."""
        protocol = MockProtocol(agreement_intensity=2)
        builder = PromptBuilder(protocol=protocol, env=MockEnvironment(task="Test"))

        result = builder.build_revision_prompt(
            agent=mock_agent,
            original="Original",
            critiques=[MockCritique()],
        )

        assert "disagree" in result.lower() or "skeptic" in result.lower()

    def test_build_revision_prompt_with_round_number(self, mock_agent):
        """Test revision prompt with round number."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Test"),
        )

        result = builder.build_revision_prompt(
            agent=mock_agent,
            original="Original",
            critiques=[MockCritique()],
            round_number=2,
        )

        assert isinstance(result, str)


class TestBuildJudgePrompt:
    """Tests for judge prompt building."""

    def test_build_judge_prompt_basic(self):
        """Test basic judge prompt."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Test task"),
        )
        proposals = {
            "agent1": "Proposal 1 content",
            "agent2": "Proposal 2 content",
        }
        critiques = [MockCritique()]

        result = builder.build_judge_prompt(
            proposals=proposals,
            task="Test task",
            critiques=critiques,
        )

        assert "synthesizer" in result.lower() or "judge" in result.lower()
        assert "agent1" in result
        assert "Proposal 1 content" in result

    def test_build_judge_prompt_with_evidence(self):
        """Test judge prompt with evidence pack."""
        mock_snippet = MagicMock()
        mock_snippet.title = "Source"
        mock_snippet.source = "web"
        mock_snippet.reliability_score = 0.9
        mock_snippet.url = "https://test.com"
        mock_snippet.snippet = "Evidence text"

        mock_pack = MagicMock()
        mock_pack.snippets = [mock_snippet]

        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Test"),
            evidence_pack=mock_pack,
        )

        result = builder.build_judge_prompt(
            proposals={"a": "p"},
            task="Test",
            critiques=[],
        )

        assert "EVID" in result or "Evidence" in result


class TestBuildJudgeVotePrompt:
    """Tests for judge vote prompt building."""

    def test_build_judge_vote_prompt(self, mock_agent):
        """Test judge vote prompt."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Test"),
        )
        candidates = [MockAgent(name="agent1"), MockAgent(name="agent2")]
        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}

        result = builder.build_judge_vote_prompt(candidates, proposals)

        assert "vote" in result.lower()
        assert "agent1" in result
        assert "agent2" in result


class TestFormatPatterns:
    """Tests for pattern formatting."""

    def test_format_patterns_empty(self, basic_builder):
        """Test pattern formatting with empty list."""
        result = basic_builder.format_patterns_for_prompt([])

        assert result == ""

    def test_format_patterns_with_data(self, basic_builder):
        """Test pattern formatting with patterns."""
        patterns = [
            {
                "category": "security",
                "pattern": "Missing input validation",
                "occurrences": 5,
                "avg_severity": 0.8,
            },
            {
                "category": "logic",
                "pattern": "Edge case not handled",
                "occurrences": 3,
                "avg_severity": 0.4,
            },
        ]
        result = basic_builder.format_patterns_for_prompt(patterns)

        assert "LEARNED PATTERNS" in result
        assert "SECURITY" in result

    def test_format_successful_patterns_no_memory(self, basic_builder):
        """Test successful patterns without memory."""
        result = basic_builder.format_successful_patterns(limit=3)

        assert result == ""

    def test_format_successful_patterns_with_memory(self, mock_protocol, mock_env):
        """Test successful patterns with memory."""
        mock_pattern = MagicMock()
        mock_pattern.issue_text = "Test issue"
        mock_pattern.suggestion_text = "Test suggestion"
        mock_pattern.issue_type = "logic"
        mock_pattern.success_count = 5

        mock_memory = MagicMock()
        mock_memory.retrieve_patterns.return_value = [mock_pattern]

        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            memory=mock_memory,
        )

        result = builder.format_successful_patterns(limit=3)

        assert "SUCCESSFUL PATTERNS" in result
        assert "Test issue" in result


class TestContinuumContext:
    """Tests for continuum memory context."""

    def test_get_continuum_context_empty(self, basic_builder):
        """Test continuum context when cache is empty."""
        result = basic_builder.get_continuum_context()

        assert result == ""

    def test_get_continuum_context_cached(self, basic_builder):
        """Test continuum context with cached value."""
        basic_builder._continuum_context_cache = "Cached context"

        result = basic_builder.get_continuum_context()

        assert result == "Cached context"


class TestBeliefContext:
    """Tests for belief/crux context injection."""

    def test_inject_belief_context_no_memory(self, basic_builder):
        """Test belief context without continuum memory."""
        result = basic_builder._inject_belief_context()

        assert result == ""

    def test_inject_belief_context_with_cruxes(self, mock_protocol, mock_env):
        """Test belief context with cruxes in memory."""
        mock_mem = MagicMock()
        mock_mem.metadata = {"crux_claims": ["Claim 1", "Claim 2"]}

        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = [mock_mem]

        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            continuum_memory=mock_memory,
        )

        result = builder._inject_belief_context()

        assert "Claim 1" in result
        assert "Historical Disagreement" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_prompt_with_unicode_task(self, mock_agent):
        """Test prompt building with unicode in task."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Discuss API design with emojis"),
        )

        result = builder.build_proposal_prompt(agent=mock_agent)

        assert isinstance(result, str)

    def test_prompt_with_long_task(self, mock_agent):
        """Test prompt building with very long task."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="A" * 10000),
        )

        result = builder.build_proposal_prompt(agent=mock_agent)

        assert isinstance(result, str)

    def test_prompt_with_empty_context(self, mock_agent):
        """Test prompt building with empty context."""
        builder = PromptBuilder(
            protocol=MockProtocol(),
            env=MockEnvironment(task="Test", context=""),
        )

        result = builder.build_proposal_prompt(agent=mock_agent)

        assert isinstance(result, str)
