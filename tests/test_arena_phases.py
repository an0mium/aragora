"""
Tests for arena_phases.py - Phase initialization for debates.

Tests cover:
- Phase initialization via init_phases
- Verify claims callback creation
- Component wiring and error handling
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass


# =============================================================================
# Test the verify_claims callback
# =============================================================================


class TestVerifyClaimsCallback:
    """Tests for _create_verify_claims_callback function."""

    @pytest.fixture
    def mock_arena(self):
        """Create a mock Arena for testing."""
        arena = MagicMock()
        return arena

    @pytest.mark.asyncio
    async def test_verify_claims_empty_text(self, mock_arena):
        """Should return dict with zeros for empty proposal text."""
        from aragora.debate.arena_phases import _create_verify_claims_callback

        callback = _create_verify_claims_callback(mock_arena)
        result = await callback("", limit=2)

        # Phase 11A: Now returns dict with verified/disproven counts
        assert result == {"verified": 0, "disproven": 0}

    @pytest.mark.asyncio
    async def test_verify_claims_no_claims(self, mock_arena):
        """Should return dict with zeros when no claims are extracted."""
        from aragora.debate.arena_phases import _create_verify_claims_callback

        with patch("aragora.debate.arena_phases.fast_extract_claims") as mock_extract:
            mock_extract.return_value = []

            callback = _create_verify_claims_callback(mock_arena)
            result = await callback("Some proposal text with no claims", limit=2)

            assert result == {"verified": 0, "disproven": 0}

    @pytest.mark.asyncio
    async def test_verify_claims_high_confidence(self, mock_arena):
        """Should count high-confidence claims as verified."""
        from aragora.debate.arena_phases import _create_verify_claims_callback

        mock_claims = [
            {"type": "FACTUAL", "text": "Python is interpreted", "confidence": 0.8},
            {"type": "FACTUAL", "text": "2+2=4", "confidence": 0.9},
        ]

        with patch("aragora.debate.arena_phases.fast_extract_claims") as mock_extract:
            mock_extract.return_value = mock_claims

            callback = _create_verify_claims_callback(mock_arena)
            result = await callback("Proposal with high confidence claims", limit=5)

            # Both claims should be counted (confidence >= 0.5)
            assert result["verified"] == 2
            assert result["disproven"] == 0

    @pytest.mark.asyncio
    async def test_verify_claims_low_confidence(self, mock_arena):
        """Should not count low-confidence claims."""
        from aragora.debate.arena_phases import _create_verify_claims_callback

        mock_claims = [
            {"type": "FACTUAL", "text": "Maybe true", "confidence": 0.3},
            {"type": "FACTUAL", "text": "Uncertain claim", "confidence": 0.4},
        ]

        with patch("aragora.debate.arena_phases.fast_extract_claims") as mock_extract:
            mock_extract.return_value = mock_claims

            callback = _create_verify_claims_callback(mock_arena)
            result = await callback("Proposal with low confidence", limit=5)

            # Neither claim should be counted (confidence < 0.5)
            assert result["verified"] == 0
            assert result["disproven"] == 0

    @pytest.mark.asyncio
    async def test_verify_claims_respects_limit(self, mock_arena):
        """Should only verify up to the limit."""
        from aragora.debate.arena_phases import _create_verify_claims_callback

        mock_claims = [
            {"type": "FACTUAL", "text": f"Claim {i}", "confidence": 0.8} for i in range(10)
        ]

        with patch("aragora.debate.arena_phases.fast_extract_claims") as mock_extract:
            mock_extract.return_value = mock_claims

            callback = _create_verify_claims_callback(mock_arena)
            result = await callback("Proposal with many claims", limit=2)

            # Should only check first 2 claims
            assert result["verified"] == 2

    @pytest.mark.asyncio
    async def test_verify_claims_mixed_confidence(self, mock_arena):
        """Should correctly count only high-confidence claims."""
        from aragora.debate.arena_phases import _create_verify_claims_callback

        mock_claims = [
            {"type": "FACTUAL", "text": "High confidence", "confidence": 0.7},
            {"type": "FACTUAL", "text": "Low confidence", "confidence": 0.3},
            {"type": "FACTUAL", "text": "Borderline", "confidence": 0.5},
        ]

        with patch("aragora.debate.arena_phases.fast_extract_claims") as mock_extract:
            mock_extract.return_value = mock_claims

            callback = _create_verify_claims_callback(mock_arena)
            result = await callback("Mixed claims", limit=10)

            # Claims with confidence >= 0.5 should count
            assert result["verified"] == 2
            assert result["disproven"] == 0

    @pytest.mark.asyncio
    async def test_verify_claims_z3_disproven(self, mock_arena):
        """Should track disproven claims from Z3 verification (Phase 11A)."""
        from aragora.debate.arena_phases import _create_verify_claims_callback
        from aragora.verification.formal import FormalProofStatus, FormalProofResult, FormalLanguage

        mock_claims = [
            {"type": "LOGICAL", "text": "1 > 2", "confidence": 0.8},
            {"type": "LOGICAL", "text": "true = true", "confidence": 0.8},
        ]

        # Mock Z3 returning disproven for first claim, verified for second
        mock_manager = MagicMock()
        mock_manager.get_available_backends.return_value = [
            MagicMock(language=MagicMock(value="z3_smt"))
        ]

        async def mock_verify(claim_text, claim_type, timeout_seconds):
            if "1 > 2" in claim_text:
                return FormalProofResult(
                    status=FormalProofStatus.PROOF_FAILED,
                    language=FormalLanguage.Z3_SMT,
                    proof_search_time_ms=10.0,
                    error_message="Counterexample found",
                )
            return FormalProofResult(
                status=FormalProofStatus.PROOF_FOUND,
                language=FormalLanguage.Z3_SMT,
                proof_search_time_ms=5.0,
            )

        mock_manager.attempt_formal_verification = mock_verify

        with patch("aragora.debate.arena_phases.fast_extract_claims") as mock_extract:
            with patch(
                "aragora.verification.formal.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                mock_extract.return_value = mock_claims

                callback = _create_verify_claims_callback(mock_arena)
                result = await callback("Logical claims", limit=10)

                # One verified, one disproven
                assert result["verified"] == 1
                assert result["disproven"] == 1


# =============================================================================
# Test init_phases function
# =============================================================================


class TestInitPhases:
    """Tests for init_phases function."""

    @pytest.fixture
    def minimal_arena(self):
        """Create a minimal mock Arena with required attributes."""
        arena = MagicMock()

        # Required attributes for phase initialization
        arena.protocol = MagicMock()
        arena.protocol.enable_rhetorical_observer = False
        arena.protocol.enable_trickster = False

        arena.env = MagicMock()
        arena.memory = MagicMock()
        arena.continuum_memory = MagicMock()
        arena.dissent_retriever = MagicMock()
        arena.role_rotator = MagicMock()
        arena.persona_manager = MagicMock()
        arena.flip_detector = MagicMock()
        arena.calibration_tracker = MagicMock()
        arena.debate_embeddings = MagicMock()
        arena.event_emitter = MagicMock()
        arena.spectator = MagicMock()
        arena.loop_id = "test-loop"

        # Phase dependencies
        arena.initial_messages = []
        arena.trending_topic = None
        arena.recorder = MagicMock()
        arena.insight_store = MagicMock()
        arena.evidence_collector = MagicMock()
        arena.pulse_manager = MagicMock()
        arena.auto_fetch_trending = False

        arena.circuit_breaker = MagicMock()
        arena.position_tracker = MagicMock()
        arena.position_ledger = MagicMock()
        arena.hooks = MagicMock()
        arena.autonomic = MagicMock()

        arena.convergence_detector = MagicMock()
        arena.elo_system = MagicMock()
        arena.agent_weights = {}
        arena.user_votes = {}

        arena.relationship_tracker = MagicMock()
        arena.moment_detector = MagicMock()
        arena.consensus_memory = MagicMock()

        arena.auto_evolve = False
        arena.breeding_threshold = 0.8
        arena.prompt_evolver = None
        arena.population_manager = None

        # Callbacks
        arena._fetch_historical_context = MagicMock()
        arena._format_patterns_for_prompt = MagicMock()
        arena._get_successful_patterns_from_memory = MagicMock()
        arena._perform_research = MagicMock()
        arena._build_proposal_prompt = MagicMock()
        arena._notify_spectator = MagicMock()
        arena._update_role_assignments = MagicMock()
        arena._record_grounded_position = MagicMock()
        arena._extract_citation_needs = MagicMock()
        arena._assign_stances = MagicMock()
        arena._select_critics_for_proposal = MagicMock()
        arena._build_revision_prompt = MagicMock()
        arena._check_judge_termination = MagicMock()
        arena._check_early_stopping = MagicMock()
        arena._refresh_evidence_for_round = MagicMock()
        arena._select_judge = MagicMock()
        arena._build_judge_prompt = MagicMock()
        arena._group_similar_votes = MagicMock()
        arena._get_calibration_weight = MagicMock()
        arena._drain_user_events = MagicMock()
        arena._extract_debate_domain = MagicMock()
        arena._update_agent_relationships = MagicMock()
        arena._generate_disagreement_report = MagicMock()
        arena._create_grounded_verdict = MagicMock()
        arena._verify_claims_formally = MagicMock()
        arena._format_conclusion = MagicMock()
        arena._emit_moment_event = MagicMock()
        arena._store_debate_outcome_as_memory = MagicMock()
        arena._update_continuum_memory_outcomes = MagicMock()
        arena._index_debate_async = MagicMock()
        arena._store_evidence_in_memory = MagicMock()

        # Nomic dir getter
        arena.get_nomic_dir = MagicMock(return_value=None)

        return arena

    def test_init_phases_creates_voting_phase(self, minimal_arena):
        """Should create VotingPhase."""
        from aragora.debate.arena_phases import init_phases

        init_phases(minimal_arena)

        assert minimal_arena.voting_phase is not None

    def test_init_phases_creates_prompt_builder(self, minimal_arena):
        """Should create PromptBuilder."""
        from aragora.debate.arena_phases import init_phases

        init_phases(minimal_arena)

        assert minimal_arena.prompt_builder is not None

    def test_init_phases_creates_memory_manager(self, minimal_arena):
        """Should create MemoryManager."""
        from aragora.debate.arena_phases import init_phases

        init_phases(minimal_arena)

        assert minimal_arena.memory_manager is not None

    def test_init_phases_creates_context_gatherer(self, minimal_arena):
        """Should create ContextGatherer."""
        from aragora.debate.arena_phases import init_phases

        init_phases(minimal_arena)

        assert minimal_arena.context_gatherer is not None

    def test_init_phases_creates_proposal_phase(self, minimal_arena):
        """Should create ProposalPhase."""
        from aragora.debate.arena_phases import init_phases

        init_phases(minimal_arena)

        assert minimal_arena.proposal_phase is not None

    def test_init_phases_creates_debate_rounds_phase(self, minimal_arena):
        """Should create DebateRoundsPhase."""
        from aragora.debate.arena_phases import init_phases

        init_phases(minimal_arena)

        assert minimal_arena.debate_rounds_phase is not None

    def test_init_phases_creates_consensus_phase(self, minimal_arena):
        """Should create ConsensusPhase."""
        from aragora.debate.arena_phases import init_phases

        init_phases(minimal_arena)

        assert minimal_arena.consensus_phase is not None

    def test_init_phases_creates_analytics_phase(self, minimal_arena):
        """Should create AnalyticsPhase."""
        from aragora.debate.arena_phases import init_phases

        init_phases(minimal_arena)

        assert minimal_arena.analytics_phase is not None

    def test_init_phases_creates_feedback_phase(self, minimal_arena):
        """Should create FeedbackPhase."""
        from aragora.debate.arena_phases import init_phases

        init_phases(minimal_arena)

        assert minimal_arena.feedback_phase is not None

    def test_init_phases_creates_context_initializer(self, minimal_arena):
        """Should create ContextInitializer."""
        from aragora.debate.arena_phases import init_phases

        init_phases(minimal_arena)

        assert minimal_arena.context_initializer is not None

    def test_init_phases_with_rhetorical_observer(self, minimal_arena):
        """Should initialize rhetorical observer when enabled."""
        minimal_arena.protocol.enable_rhetorical_observer = True

        from aragora.debate.arena_phases import init_phases

        # Should not raise even if observer import fails
        init_phases(minimal_arena)

    def test_init_phases_with_trickster(self, minimal_arena):
        """Should initialize trickster when enabled."""
        minimal_arena.protocol.enable_trickster = True
        minimal_arena.protocol.trickster_sensitivity = 0.7
        minimal_arena.protocol.novelty_threshold = 0.15

        from aragora.debate.arena_phases import init_phases

        # Should not raise even if trickster import fails
        init_phases(minimal_arena)

    @pytest.mark.slow
    def test_init_phases_with_auto_evolve(self, minimal_arena):
        """Should initialize population manager when auto_evolve is enabled.

        Note: PopulationManager import takes ~1.3s due to scipy/numpy dependencies.
        """
        minimal_arena.auto_evolve = True

        from aragora.debate.arena_phases import init_phases

        # Should not raise even if PopulationManager import fails
        init_phases(minimal_arena)


# =============================================================================
# Integration-style tests
# =============================================================================


class TestArenaPhaseIntegration:
    """Integration-style tests for arena phase setup."""

    @pytest.fixture
    def fully_mocked_arena(self):
        """Create a more complete mock Arena."""
        arena = MagicMock()

        # All required attributes
        arena.protocol = MagicMock()
        arena.protocol.enable_rhetorical_observer = False
        arena.protocol.enable_trickster = False

        arena.env = MagicMock()
        arena.memory = MagicMock()
        arena.continuum_memory = MagicMock()
        arena.dissent_retriever = MagicMock()
        arena.role_rotator = MagicMock()
        arena.persona_manager = MagicMock()
        arena.flip_detector = MagicMock()
        arena.calibration_tracker = MagicMock()
        arena.debate_embeddings = MagicMock()
        arena.event_emitter = MagicMock()
        arena.spectator = MagicMock()
        arena.loop_id = "integration-test"

        arena.initial_messages = []
        arena.trending_topic = None
        arena.recorder = MagicMock()
        arena.insight_store = MagicMock()
        arena.evidence_collector = MagicMock()
        arena.pulse_manager = MagicMock()
        arena.auto_fetch_trending = False

        arena.circuit_breaker = MagicMock()
        arena.position_tracker = MagicMock()
        arena.position_ledger = MagicMock()
        arena.hooks = MagicMock()
        arena.autonomic = MagicMock()
        arena.autonomic.generate = AsyncMock()
        arena.autonomic.critique = AsyncMock()
        arena.autonomic.vote = AsyncMock()
        arena.autonomic.with_timeout = MagicMock(side_effect=lambda x: x)

        arena.convergence_detector = MagicMock()
        arena.elo_system = MagicMock()
        arena.agent_weights = {}
        arena.user_votes = {}

        arena.relationship_tracker = MagicMock()
        arena.moment_detector = MagicMock()
        arena.consensus_memory = MagicMock()

        arena.auto_evolve = False
        arena.breeding_threshold = 0.8
        arena.prompt_evolver = None
        arena.population_manager = None

        # All callbacks
        arena._fetch_historical_context = AsyncMock()
        arena._format_patterns_for_prompt = MagicMock(return_value="")
        arena._get_successful_patterns_from_memory = MagicMock(return_value=[])
        arena._perform_research = AsyncMock()
        arena._build_proposal_prompt = MagicMock(return_value="")
        arena._notify_spectator = MagicMock()
        arena._update_role_assignments = MagicMock()
        arena._record_grounded_position = MagicMock()
        arena._extract_citation_needs = MagicMock(return_value=[])
        arena._assign_stances = MagicMock()
        arena._select_critics_for_proposal = MagicMock(return_value=[])
        arena._build_revision_prompt = MagicMock(return_value="")
        arena._check_judge_termination = MagicMock(return_value=False)
        arena._check_early_stopping = MagicMock(return_value=False)
        arena._refresh_evidence_for_round = AsyncMock()
        arena._select_judge = MagicMock(return_value=None)
        arena._build_judge_prompt = MagicMock(return_value="")
        arena._group_similar_votes = MagicMock(return_value=[])
        arena._get_calibration_weight = MagicMock(return_value=1.0)
        arena._drain_user_events = MagicMock(return_value=([], []))
        arena._extract_debate_domain = MagicMock(return_value="general")
        arena._update_agent_relationships = MagicMock()
        arena._generate_disagreement_report = MagicMock(return_value="")
        arena._create_grounded_verdict = MagicMock(return_value=None)
        arena._verify_claims_formally = AsyncMock()
        arena._format_conclusion = MagicMock(return_value="")
        arena._emit_moment_event = MagicMock()
        arena._store_debate_outcome_as_memory = MagicMock()
        arena._update_continuum_memory_outcomes = MagicMock()
        arena._index_debate_async = AsyncMock()
        arena._store_evidence_in_memory = MagicMock()

        arena.get_nomic_dir = MagicMock(return_value=None)

        return arena

    def test_all_phases_initialized(self, fully_mocked_arena):
        """All phases should be initialized after init_phases."""
        from aragora.debate.arena_phases import init_phases

        init_phases(fully_mocked_arena)

        # Check all expected phases exist
        assert hasattr(fully_mocked_arena, "voting_phase")
        assert hasattr(fully_mocked_arena, "prompt_builder")
        assert hasattr(fully_mocked_arena, "memory_manager")
        assert hasattr(fully_mocked_arena, "context_gatherer")
        assert hasattr(fully_mocked_arena, "context_initializer")
        assert hasattr(fully_mocked_arena, "proposal_phase")
        assert hasattr(fully_mocked_arena, "debate_rounds_phase")
        assert hasattr(fully_mocked_arena, "consensus_phase")
        assert hasattr(fully_mocked_arena, "analytics_phase")
        assert hasattr(fully_mocked_arena, "feedback_phase")

    def test_phases_have_correct_types(self, fully_mocked_arena):
        """Phases should have correct types."""
        from aragora.debate.arena_phases import init_phases
        from aragora.debate.phases import (
            VotingPhase,
            ContextInitializer,
            ProposalPhase,
            DebateRoundsPhase,
            ConsensusPhase,
            AnalyticsPhase,
            FeedbackPhase,
        )
        from aragora.debate.prompt_builder import PromptBuilder
        from aragora.debate.memory_manager import MemoryManager
        from aragora.debate.context_gatherer import ContextGatherer

        init_phases(fully_mocked_arena)

        assert isinstance(fully_mocked_arena.voting_phase, VotingPhase)
        assert isinstance(fully_mocked_arena.prompt_builder, PromptBuilder)
        assert isinstance(fully_mocked_arena.memory_manager, MemoryManager)
        assert isinstance(fully_mocked_arena.context_gatherer, ContextGatherer)
        assert isinstance(fully_mocked_arena.context_initializer, ContextInitializer)
        assert isinstance(fully_mocked_arena.proposal_phase, ProposalPhase)
        assert isinstance(fully_mocked_arena.debate_rounds_phase, DebateRoundsPhase)
        assert isinstance(fully_mocked_arena.consensus_phase, ConsensusPhase)
        assert isinstance(fully_mocked_arena.analytics_phase, AnalyticsPhase)
        assert isinstance(fully_mocked_arena.feedback_phase, FeedbackPhase)
