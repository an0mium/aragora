"""
Tests for the arena_phases module.

Tests cover:
- _create_verify_claims_callback function
- init_phases function (high-level)
- create_phase_executor function
- Optional feature initialization (trickster, rhetorical observer)
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from aragora.debate.arena_phases import (
    _create_verify_claims_callback,
    init_phases,
    create_phase_executor,
    GENESIS_AVAILABLE,
)


class TestCreateVerifyClaimsCallback:
    """Tests for _create_verify_claims_callback function."""

    def test_creates_callback(self):
        """Creates a callable callback."""
        mock_arena = MagicMock()

        callback = _create_verify_claims_callback(mock_arena)

        assert callable(callback)

    @pytest.mark.asyncio
    async def test_callback_returns_zero_for_empty_text(self):
        """Callback returns zero counts for empty text."""
        mock_arena = MagicMock()
        callback = _create_verify_claims_callback(mock_arena)

        result = await callback("")

        assert result == {"verified": 0, "disproven": 0}

    @pytest.mark.asyncio
    async def test_callback_returns_zero_for_none_text(self):
        """Callback returns zero counts for None text."""
        mock_arena = MagicMock()
        callback = _create_verify_claims_callback(mock_arena)

        result = await callback(None)

        assert result == {"verified": 0, "disproven": 0}

    @pytest.mark.asyncio
    async def test_callback_extracts_claims(self):
        """Callback extracts claims from proposal text."""
        mock_arena = MagicMock()
        callback = _create_verify_claims_callback(mock_arena)

        # Text with high-confidence claim pattern
        text = "According to the evidence, X is clearly true because Y."

        with patch("aragora.debate.arena_phases.fast_extract_claims") as mock_extract:
            mock_extract.return_value = [
                {"text": "X is clearly true", "type": "LOGICAL", "confidence": 0.8}
            ]

            result = await callback(text)

        mock_extract.assert_called_once_with(text, author="proposal")
        assert result["verified"] >= 0

    @pytest.mark.asyncio
    async def test_callback_returns_zero_for_no_claims(self):
        """Callback returns zero when no claims extracted."""
        mock_arena = MagicMock()
        callback = _create_verify_claims_callback(mock_arena)

        with patch("aragora.debate.arena_phases.fast_extract_claims") as mock_extract:
            mock_extract.return_value = []

            result = await callback("Some text without claims")

        assert result == {"verified": 0, "disproven": 0}

    @pytest.mark.asyncio
    async def test_callback_respects_limit(self):
        """Callback respects the limit parameter."""
        mock_arena = MagicMock()
        callback = _create_verify_claims_callback(mock_arena)

        claims = [
            {"text": f"claim{i}", "type": "LOGICAL", "confidence": 0.8}
            for i in range(10)
        ]

        with patch("aragora.debate.arena_phases.fast_extract_claims") as mock_extract:
            mock_extract.return_value = claims

            # Request only 2 claims to be verified
            result = await callback("text", limit=2)

        # Should only process 2 claims max
        assert result["verified"] <= 2

    @pytest.mark.asyncio
    async def test_callback_confidence_fallback(self):
        """Callback uses confidence fallback when Z3 unavailable."""
        mock_arena = MagicMock()
        callback = _create_verify_claims_callback(mock_arena)

        with patch("aragora.debate.arena_phases.fast_extract_claims") as mock_extract:
            # High confidence claim without Z3
            mock_extract.return_value = [
                {"text": "claim1", "type": "FACTUAL", "confidence": 0.7}
            ]

            result = await callback("text")

        # Should count as verified via confidence fallback
        assert result["verified"] == 1


class TestInitPhases:
    """Tests for init_phases function."""

    def test_init_phases_sets_voting_phase(self):
        """init_phases sets voting_phase attribute."""
        mock_arena = self._create_mock_arena()

        init_phases(mock_arena)

        assert mock_arena.voting_phase is not None

    def test_init_phases_sets_prompt_builder(self):
        """init_phases sets prompt_builder attribute."""
        mock_arena = self._create_mock_arena()

        init_phases(mock_arena)

        assert mock_arena.prompt_builder is not None

    def test_init_phases_sets_memory_manager(self):
        """init_phases sets memory_manager attribute."""
        mock_arena = self._create_mock_arena()

        init_phases(mock_arena)

        assert mock_arena.memory_manager is not None

    def test_init_phases_sets_context_gatherer(self):
        """init_phases sets context_gatherer attribute."""
        mock_arena = self._create_mock_arena()

        init_phases(mock_arena)

        assert mock_arena.context_gatherer is not None

    def test_init_phases_sets_context_initializer(self):
        """init_phases sets context_initializer attribute."""
        mock_arena = self._create_mock_arena()

        init_phases(mock_arena)

        assert mock_arena.context_initializer is not None

    def test_init_phases_sets_proposal_phase(self):
        """init_phases sets proposal_phase attribute."""
        mock_arena = self._create_mock_arena()

        init_phases(mock_arena)

        assert mock_arena.proposal_phase is not None

    def test_init_phases_sets_debate_rounds_phase(self):
        """init_phases sets debate_rounds_phase attribute."""
        mock_arena = self._create_mock_arena()

        init_phases(mock_arena)

        assert mock_arena.debate_rounds_phase is not None

    def test_init_phases_sets_consensus_phase(self):
        """init_phases sets consensus_phase attribute."""
        mock_arena = self._create_mock_arena()

        init_phases(mock_arena)

        assert mock_arena.consensus_phase is not None

    def test_init_phases_sets_analytics_phase(self):
        """init_phases sets analytics_phase attribute."""
        mock_arena = self._create_mock_arena()

        init_phases(mock_arena)

        assert mock_arena.analytics_phase is not None

    def test_init_phases_sets_feedback_phase(self):
        """init_phases sets feedback_phase attribute."""
        mock_arena = self._create_mock_arena()

        init_phases(mock_arena)

        assert mock_arena.feedback_phase is not None

    def test_init_phases_handles_trickster_protocol_flag(self):
        """init_phases respects trickster protocol flag."""
        mock_arena = self._create_mock_arena()
        mock_arena.protocol.enable_trickster = True
        mock_arena.protocol.trickster_sensitivity = 0.7

        # Should not raise even if trickster module unavailable
        init_phases(mock_arena)

        # Verify debate_rounds_phase is set
        assert mock_arena.debate_rounds_phase is not None

    def test_init_phases_handles_rhetorical_observer_protocol_flag(self):
        """init_phases respects rhetorical observer protocol flag."""
        mock_arena = self._create_mock_arena()
        mock_arena.protocol.enable_rhetorical_observer = True

        # Should not raise even if rhetorical_observer module unavailable
        init_phases(mock_arena)

        # Verify debate_rounds_phase is set
        assert mock_arena.debate_rounds_phase is not None

    def test_init_phases_handles_missing_optional_features(self):
        """init_phases handles missing optional features gracefully."""
        mock_arena = self._create_mock_arena()
        mock_arena.protocol.enable_trickster = True
        mock_arena.protocol.enable_rhetorical_observer = True

        # Mock imports to fail
        with patch.dict("sys.modules", {
            "aragora.debate.trickster": None,
            "aragora.debate.rhetorical_observer": None,
        }):
            # Should not raise
            init_phases(mock_arena)

    def _create_mock_arena(self):
        """Create a mock Arena with required attributes."""
        mock_arena = MagicMock()

        # Protocol with defaults
        mock_protocol = MagicMock()
        mock_protocol.enable_trickster = False
        mock_protocol.enable_rhetorical_observer = False
        mock_protocol.rounds = 3
        mock_arena.protocol = mock_protocol

        # Required attributes
        mock_arena.env = MagicMock()
        mock_arena.memory = MagicMock()
        mock_arena.continuum_memory = MagicMock()
        mock_arena.dissent_retriever = MagicMock()
        mock_arena.role_rotator = MagicMock()
        mock_arena.persona_manager = MagicMock()
        mock_arena.flip_detector = MagicMock()
        mock_arena.calibration_tracker = MagicMock()
        mock_arena.debate_embeddings = MagicMock()
        mock_arena.event_emitter = MagicMock()
        mock_arena.spectator = MagicMock()
        mock_arena.loop_id = "test-loop"
        mock_arena.circuit_breaker = MagicMock()
        mock_arena.position_tracker = MagicMock()
        mock_arena.position_ledger = MagicMock()
        mock_arena.recorder = MagicMock()
        mock_arena.hooks = {}
        mock_arena.convergence_detector = MagicMock()
        mock_arena.elo_system = MagicMock()
        mock_arena.agent_weights = {}
        mock_arena.user_votes = []
        mock_arena.insight_store = MagicMock()
        mock_arena.relationship_tracker = MagicMock()
        mock_arena.moment_detector = MagicMock()
        mock_arena.autonomic = MagicMock()
        mock_arena.autonomic.generate = AsyncMock()
        mock_arena.autonomic.critique = AsyncMock()
        mock_arena.autonomic.vote = AsyncMock()
        mock_arena.autonomic.with_timeout = AsyncMock()
        mock_arena.checkpoint_manager = None
        mock_arena.initial_messages = []
        mock_arena.trending_topic = None
        mock_arena.evidence_collector = MagicMock()
        mock_arena.pulse_manager = MagicMock()
        mock_arena.auto_fetch_trending = False
        mock_arena.auto_evolve = False
        mock_arena.population_manager = None
        mock_arena.breeding_threshold = 0.8
        mock_arena.prompt_evolver = MagicMock()
        mock_arena.consensus_memory = MagicMock()
        mock_arena.extensions = MagicMock()
        mock_arena.extensions.broadcast_pipeline = None
        mock_arena.extensions.auto_broadcast = False
        mock_arena.extensions.broadcast_min_confidence = 0.8

        # Methods
        mock_arena._extract_debate_domain = MagicMock(return_value="general")
        mock_arena._store_evidence_in_memory = MagicMock()
        mock_arena._fetch_historical_context = AsyncMock()
        mock_arena._format_patterns_for_prompt = MagicMock()
        mock_arena._get_successful_patterns_from_memory = MagicMock()
        mock_arena._perform_research = AsyncMock()
        mock_arena._fetch_knowledge_context = AsyncMock()
        mock_arena._build_proposal_prompt = MagicMock()
        mock_arena._notify_spectator = MagicMock()
        mock_arena._update_role_assignments = MagicMock()
        mock_arena._record_grounded_position = MagicMock()
        mock_arena._extract_citation_needs = MagicMock()
        mock_arena._assign_stances = MagicMock()
        mock_arena._select_critics_for_proposal = MagicMock()
        mock_arena._build_revision_prompt = MagicMock()
        mock_arena._check_judge_termination = MagicMock()
        mock_arena._check_early_stopping = MagicMock()
        mock_arena._refresh_evidence_for_round = AsyncMock()
        mock_arena._create_checkpoint = MagicMock()
        mock_arena._select_judge = MagicMock()
        mock_arena._group_similar_votes = MagicMock()
        mock_arena._get_calibration_weight = MagicMock()
        mock_arena._drain_user_events = MagicMock()
        mock_arena._update_agent_relationships = MagicMock()
        mock_arena._create_grounded_verdict = AsyncMock()
        mock_arena._verify_claims_formally = AsyncMock()
        mock_arena._format_conclusion = MagicMock()
        mock_arena._emit_moment_event = MagicMock()
        mock_arena._store_debate_outcome_as_memory = MagicMock()
        mock_arena._update_continuum_memory_outcomes = MagicMock()
        mock_arena._index_debate_async = AsyncMock()
        mock_arena._ingest_debate_outcome = AsyncMock()

        return mock_arena


class TestCreatePhaseExecutor:
    """Tests for create_phase_executor function."""

    def test_creates_phase_executor(self):
        """create_phase_executor creates a PhaseExecutor instance."""
        mock_arena = self._create_mock_arena()

        executor = create_phase_executor(mock_arena)

        assert executor is not None

    def test_executor_is_phase_executor_type(self):
        """create_phase_executor returns PhaseExecutor instance."""
        from aragora.debate.phase_executor import PhaseExecutor

        mock_arena = self._create_mock_arena()

        executor = create_phase_executor(mock_arena)

        assert isinstance(executor, PhaseExecutor)

    def test_handles_many_agents(self):
        """create_phase_executor handles many agents."""
        mock_arena = self._create_mock_arena()
        mock_arena.agents = [MagicMock() for _ in range(8)]  # 8 agents
        mock_arena.protocol.rounds = 5

        executor = create_phase_executor(mock_arena)

        # Should create executor without error
        assert executor is not None

    def test_handles_protocol_timeout(self):
        """create_phase_executor handles protocol timeout configuration."""
        mock_arena = self._create_mock_arena()
        mock_arena.protocol.timeout = 3600  # 1 hour

        executor = create_phase_executor(mock_arena)

        # Should create executor without error
        assert executor is not None

    def test_handles_empty_agents(self):
        """create_phase_executor handles empty agents list."""
        mock_arena = self._create_mock_arena()
        mock_arena.agents = []

        executor = create_phase_executor(mock_arena)

        # Should create executor without error (uses default agent count)
        assert executor is not None

    def _create_mock_arena(self):
        """Create a mock Arena with required attributes for create_phase_executor."""
        mock_arena = MagicMock()

        # Protocol
        mock_protocol = MagicMock()
        mock_protocol.rounds = 3
        mock_protocol.timeout = 300
        mock_arena.protocol = mock_protocol

        # Agents
        mock_arena.agents = [MagicMock() for _ in range(4)]

        # Phases (need to be set by init_phases first, but mock them)
        mock_arena.context_initializer = MagicMock()
        mock_arena.context_initializer.initialize = AsyncMock()
        mock_arena.proposal_phase = MagicMock()
        mock_arena.debate_rounds_phase = MagicMock()
        mock_arena.consensus_phase = MagicMock()
        mock_arena.analytics_phase = MagicMock()
        mock_arena.feedback_phase = MagicMock()

        return mock_arena


class TestGenesisAvailability:
    """Tests for GENESIS_AVAILABLE flag."""

    def test_genesis_available_is_boolean(self):
        """GENESIS_AVAILABLE is a boolean."""
        assert isinstance(GENESIS_AVAILABLE, bool)


class TestInitPhasesWithAutoEvolve:
    """Tests for auto-evolution initialization."""

    def test_auto_evolve_initializes_population_manager(self):
        """init_phases initializes PopulationManager when auto_evolve enabled."""
        mock_arena = MagicMock()

        # Setup protocol
        mock_arena.protocol = MagicMock()
        mock_arena.protocol.enable_trickster = False
        mock_arena.protocol.enable_rhetorical_observer = False

        # Enable auto evolve
        mock_arena.auto_evolve = True
        mock_arena.population_manager = None

        # Setup other required attributes
        mock_arena.env = MagicMock()
        mock_arena.memory = MagicMock()
        mock_arena.continuum_memory = MagicMock()
        mock_arena.dissent_retriever = MagicMock()
        mock_arena.role_rotator = MagicMock()
        mock_arena.persona_manager = MagicMock()
        mock_arena.flip_detector = MagicMock()
        mock_arena.calibration_tracker = MagicMock()
        mock_arena.debate_embeddings = MagicMock()
        mock_arena.event_emitter = MagicMock()
        mock_arena.spectator = MagicMock()
        mock_arena.loop_id = "test"
        mock_arena.circuit_breaker = MagicMock()
        mock_arena.position_tracker = MagicMock()
        mock_arena.position_ledger = MagicMock()
        mock_arena.recorder = MagicMock()
        mock_arena.hooks = {}
        mock_arena.convergence_detector = MagicMock()
        mock_arena.elo_system = MagicMock()
        mock_arena.agent_weights = {}
        mock_arena.user_votes = []
        mock_arena.insight_store = MagicMock()
        mock_arena.relationship_tracker = MagicMock()
        mock_arena.moment_detector = MagicMock()
        mock_arena.autonomic = MagicMock()
        mock_arena.checkpoint_manager = None
        mock_arena.initial_messages = []
        mock_arena.trending_topic = None
        mock_arena.evidence_collector = MagicMock()
        mock_arena.pulse_manager = MagicMock()
        mock_arena.auto_fetch_trending = False
        mock_arena.breeding_threshold = 0.8
        mock_arena.prompt_evolver = MagicMock()
        mock_arena.consensus_memory = MagicMock()
        mock_arena.extensions = MagicMock()
        mock_arena.extensions.broadcast_pipeline = None
        mock_arena.extensions.auto_broadcast = False
        mock_arena.extensions.broadcast_min_confidence = 0.8

        # Methods
        mock_arena._extract_debate_domain = MagicMock(return_value="general")
        mock_arena._store_evidence_in_memory = MagicMock()

        if GENESIS_AVAILABLE:
            with patch("aragora.debate.arena_phases.PopulationManager") as mock_pm:
                mock_pm.return_value = MagicMock()
                init_phases(mock_arena)

            mock_pm.assert_called_once()
