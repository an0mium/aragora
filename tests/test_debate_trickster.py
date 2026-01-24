"""
Tests for the Evidence-Powered Trickster system.

Tests hollow consensus detection, intervention creation,
cross-proposal analysis, and the overall trickster lifecycle.
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.debate.trickster import (
    EvidencePoweredTrickster,
    TricksterConfig,
    TricksterState,
    TricksterIntervention,
    InterventionType,
    create_trickster_for_debate,
)
from aragora.debate.evidence_quality import (
    EvidenceQualityScore,
    HollowConsensusAlert,
)
from aragora.debate.cross_proposal_analyzer import (
    CrossProposalAnalysis,
    EvidenceGap,
)
from aragora.debate.roles import CognitiveRole


# =============================================================================
# TricksterConfig Tests
# =============================================================================


class TestTricksterConfig:
    """Tests for TricksterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TricksterConfig()
        assert config.sensitivity == 0.5
        assert config.min_quality_threshold == 0.65
        assert config.hollow_detection_threshold == 0.5
        assert config.intervention_cooldown_rounds == 1
        assert config.enable_challenge_prompts is True
        assert config.enable_role_assignment is True
        assert config.enable_extended_rounds is True
        assert config.enable_breakpoints is True
        assert config.max_challenges_per_round == 3
        assert config.max_interventions_total == 5

    def test_sensitivity_adjusts_threshold_high(self):
        """Test that high sensitivity lowers the detection threshold."""
        config = TricksterConfig(sensitivity=0.9)
        # sensitivity 0.9 -> threshold = 0.8 - (0.9 * 0.6) = 0.26
        assert config.hollow_detection_threshold == pytest.approx(0.26, abs=0.01)

    def test_sensitivity_adjusts_threshold_low(self):
        """Test that low sensitivity raises the detection threshold."""
        config = TricksterConfig(sensitivity=0.1)
        # sensitivity 0.1 -> threshold = 0.8 - (0.1 * 0.6) = 0.74
        assert config.hollow_detection_threshold == pytest.approx(0.74, abs=0.01)

    def test_default_sensitivity_no_adjustment(self):
        """Test that default sensitivity doesn't adjust threshold."""
        config = TricksterConfig(sensitivity=0.5)
        # Default sensitivity should not change from specified threshold
        assert config.hollow_detection_threshold == 0.5

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = TricksterConfig(
            min_quality_threshold=0.8,
            hollow_detection_threshold=0.4,
            intervention_cooldown_rounds=3,
        )
        assert config.min_quality_threshold == 0.8
        assert config.hollow_detection_threshold == 0.4
        assert config.intervention_cooldown_rounds == 3

    def test_disabled_features(self):
        """Test disabling various features."""
        config = TricksterConfig(
            enable_challenge_prompts=False,
            enable_role_assignment=False,
            enable_extended_rounds=False,
            enable_breakpoints=False,
        )
        assert config.enable_challenge_prompts is False
        assert config.enable_role_assignment is False
        assert config.enable_extended_rounds is False
        assert config.enable_breakpoints is False


# =============================================================================
# TricksterState Tests
# =============================================================================


class TestTricksterState:
    """Tests for TricksterState."""

    def test_initial_state(self):
        """Test initial state values."""
        state = TricksterState()
        assert state.interventions == []
        assert state.quality_history == []
        assert state.last_intervention_round == -10
        assert state.hollow_alerts == []
        assert state.total_interventions == 0

    def test_state_tracks_interventions(self):
        """Test that state tracks interventions."""
        state = TricksterState()
        intervention = TricksterIntervention(
            intervention_type=InterventionType.CHALLENGE_PROMPT,
            round_num=1,
            target_agents=["agent1"],
            challenge_text="Test challenge",
            evidence_gaps={},
            priority=0.7,
        )
        state.interventions.append(intervention)
        state.total_interventions += 1
        state.last_intervention_round = 1

        assert len(state.interventions) == 1
        assert state.total_interventions == 1
        assert state.last_intervention_round == 1


# =============================================================================
# TricksterIntervention Tests
# =============================================================================


class TestTricksterIntervention:
    """Tests for TricksterIntervention dataclass."""

    def test_intervention_creation(self):
        """Test creating an intervention."""
        intervention = TricksterIntervention(
            intervention_type=InterventionType.EVIDENCE_GAP,
            round_num=2,
            target_agents=["agent1", "agent2"],
            challenge_text="Address evidence gaps",
            evidence_gaps={"agent1": ["citations", "specificity"]},
            priority=0.85,
            metadata={"total_gaps": 3},
        )
        assert intervention.intervention_type == InterventionType.EVIDENCE_GAP
        assert intervention.round_num == 2
        assert len(intervention.target_agents) == 2
        assert "agent1" in intervention.evidence_gaps
        assert intervention.priority == 0.85
        assert intervention.metadata["total_gaps"] == 3

    def test_intervention_default_metadata(self):
        """Test intervention with default metadata."""
        intervention = TricksterIntervention(
            intervention_type=InterventionType.CHALLENGE_PROMPT,
            round_num=1,
            target_agents=["agent1"],
            challenge_text="Test",
            evidence_gaps={},
            priority=0.5,
        )
        assert intervention.metadata == {}


# =============================================================================
# InterventionType Tests
# =============================================================================


class TestInterventionType:
    """Tests for InterventionType enum."""

    def test_all_intervention_types(self):
        """Test all intervention types exist."""
        assert InterventionType.CHALLENGE_PROMPT.value == "challenge_prompt"
        assert InterventionType.QUALITY_ROLE.value == "quality_role"
        assert InterventionType.EXTENDED_ROUND.value == "extended_round"
        assert InterventionType.BREAKPOINT.value == "breakpoint"
        assert InterventionType.NOVELTY_CHALLENGE.value == "novelty_challenge"
        assert InterventionType.EVIDENCE_GAP.value == "evidence_gap"
        assert InterventionType.ECHO_CHAMBER.value == "echo_chamber"


# =============================================================================
# EvidencePoweredTrickster Initialization Tests
# =============================================================================


class TestTricksterInitialization:
    """Tests for EvidencePoweredTrickster initialization."""

    def test_default_initialization(self):
        """Test default trickster initialization."""
        # Pass a mock linker to avoid slow lazy loading
        mock_linker = MagicMock()
        trickster = EvidencePoweredTrickster(linker=mock_linker)
        assert trickster.config.sensitivity == 0.5
        assert trickster.on_intervention is None
        assert trickster.on_alert is None
        assert trickster._state.total_interventions == 0

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = TricksterConfig(sensitivity=0.8, max_interventions_total=10)
        mock_linker = MagicMock()
        trickster = EvidencePoweredTrickster(config=config, linker=mock_linker)
        assert trickster.config.sensitivity == 0.8
        assert trickster.config.max_interventions_total == 10

    def test_callbacks(self):
        """Test initialization with callbacks."""
        on_intervention = MagicMock()
        on_alert = MagicMock()
        mock_linker = MagicMock()
        trickster = EvidencePoweredTrickster(
            on_intervention=on_intervention, on_alert=on_alert, linker=mock_linker
        )
        assert trickster.on_intervention is on_intervention
        assert trickster.on_alert is on_alert

    def test_custom_linker(self):
        """Test initialization with custom linker."""
        mock_linker = MagicMock()
        trickster = EvidencePoweredTrickster(linker=mock_linker)
        assert trickster._linker is mock_linker


# =============================================================================
# EvidencePoweredTrickster.check_and_intervene Tests
# =============================================================================


class TestCheckAndIntervene:
    """Tests for check_and_intervene method."""

    def test_no_intervention_quality_acceptable(self):
        """Test no intervention when quality is acceptable."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        # Mock analyzer to return good quality
        trickster._analyzer.analyze_batch = MagicMock(
            return_value={
                "agent1": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.8),
                "agent2": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.9),
            }
        )

        # Mock detector to not detect hollow consensus
        trickster._detector.check = MagicMock(
            return_value=HollowConsensusAlert(
                detected=False, severity=0.2, round_num=1, avg_quality=0.85
            )
        )

        result = trickster.check_and_intervene(
            responses={"agent1": "Good response", "agent2": "Another good response"},
            convergence_similarity=0.5,
            round_num=1,
        )

        assert result is None

    def test_intervention_triggered_on_hollow_consensus(self):
        """Test intervention triggered when hollow consensus detected."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        # Mock analyzer to return poor quality
        trickster._analyzer.analyze_batch = MagicMock(
            return_value={
                "agent1": EvidenceQualityScore(
                    agent="agent1",
                    round_num=1,
                    overall_quality=0.3,
                    citation_density=0.1,
                    specificity_score=0.2,
                    logical_chain_score=0.2,
                    evidence_diversity=0.1,
                ),
                "agent2": EvidenceQualityScore(
                    agent="agent2",
                    round_num=1,
                    overall_quality=0.3,
                    citation_density=0.1,
                    specificity_score=0.2,
                    logical_chain_score=0.2,
                    evidence_diversity=0.1,
                ),
            }
        )

        # Mock detector to detect hollow consensus
        trickster._detector.check = MagicMock(
            return_value=HollowConsensusAlert(
                detected=True,
                severity=0.7,
                round_num=1,
                avg_quality=0.3,
                min_quality=0.3,
                quality_variance=0.0,
                agent_scores={"agent1": 0.3, "agent2": 0.3},
                reason="Low evidence quality",
                recommended_challenges=["Provide citations"],
            )
        )

        result = trickster.check_and_intervene(
            responses={"agent1": "Vague claim", "agent2": "Vague claim"},
            convergence_similarity=0.9,
            round_num=1,
        )

        assert result is not None
        assert isinstance(result, TricksterIntervention)
        assert len(result.target_agents) > 0

    def test_cooldown_prevents_intervention(self):
        """Test that cooldown prevents immediate re-intervention."""
        config = TricksterConfig(intervention_cooldown_rounds=2)
        trickster = EvidencePoweredTrickster(config=config, linker=MagicMock())

        # Set state as if we just intervened
        trickster._state.last_intervention_round = 1
        trickster._state.total_interventions = 1

        # Mock detection of hollow consensus
        trickster._analyzer.analyze_batch = MagicMock(return_value={})
        trickster._detector.check = MagicMock(
            return_value=HollowConsensusAlert(
                detected=True, severity=0.8, round_num=2, avg_quality=0.3
            )
        )

        result = trickster.check_and_intervene(
            responses={"agent1": "Response"},
            convergence_similarity=0.9,
            round_num=2,  # Only 1 round since last intervention
        )

        assert result is None  # Cooldown should prevent intervention

    def test_max_interventions_limit(self):
        """Test that max interventions limit is respected."""
        config = TricksterConfig(max_interventions_total=2)
        trickster = EvidencePoweredTrickster(config=config, linker=MagicMock())

        # Set state at max interventions
        trickster._state.total_interventions = 2

        # Mock detection
        trickster._analyzer.analyze_batch = MagicMock(return_value={})
        trickster._detector.check = MagicMock(
            return_value=HollowConsensusAlert(
                detected=True, severity=0.9, round_num=5, avg_quality=0.2
            )
        )

        result = trickster.check_and_intervene(
            responses={"agent1": "Response"},
            convergence_similarity=0.95,
            round_num=5,
        )

        assert result is None  # Max interventions reached

    def test_alert_callback_called(self):
        """Test that on_alert callback is called."""
        on_alert = MagicMock()
        trickster = EvidencePoweredTrickster(on_alert=on_alert, linker=MagicMock())

        alert = HollowConsensusAlert(detected=True, severity=0.6, round_num=1, avg_quality=0.4)
        trickster._analyzer.analyze_batch = MagicMock(return_value={})
        trickster._detector.check = MagicMock(return_value=alert)

        trickster.check_and_intervene(
            responses={"agent1": "Response"},
            convergence_similarity=0.8,
            round_num=1,
        )

        on_alert.assert_called_once_with(alert)


# =============================================================================
# Evidence Gap Intervention Tests
# =============================================================================


class TestEvidenceGapIntervention:
    """Tests for evidence gap detection and intervention."""

    def test_evidence_gap_intervention_created(self):
        """Test evidence gap intervention is created."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        # Mock cross-analyzer to return evidence gaps
        gap = EvidenceGap(
            claim="AI will replace all jobs",
            agents_making_claim=["agent1", "agent2"],
            gap_severity=0.9,
        )
        cross_analysis = CrossProposalAnalysis(
            evidence_gaps=[gap],
            redundancy_score=0.3,
            unique_evidence_sources=5,
            total_evidence_sources=10,
            agent_coverage={},
            evidence_corroboration_score=0.5,
        )
        trickster._cross_analyzer.analyze = MagicMock(return_value=cross_analysis)

        # Mock other components
        trickster._analyzer.analyze_batch = MagicMock(return_value={})
        trickster._detector.check = MagicMock(
            return_value=HollowConsensusAlert(
                detected=False, severity=0.2, round_num=1, avg_quality=0.7
            )
        )

        result = trickster.check_and_intervene(
            responses={"agent1": "Claims", "agent2": "Also claims"},
            convergence_similarity=0.7,  # Above 0.6 to trigger cross-analysis
            round_num=1,
        )

        assert result is not None
        assert result.intervention_type == InterventionType.EVIDENCE_GAP
        assert "agent1" in result.target_agents or "agent2" in result.target_agents

    def test_evidence_gap_challenge_text(self):
        """Test evidence gap challenge text content."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        gap = EvidenceGap(
            claim="This claim has no evidence",
            agents_making_claim=["agent1"],
            gap_severity=0.8,
        )
        cross_analysis = CrossProposalAnalysis(
            evidence_gaps=[gap],
            redundancy_score=0.2,
            unique_evidence_sources=3,
            total_evidence_sources=5,
            agent_coverage={"agent1": 0.5},
            evidence_corroboration_score=0.4,
        )

        challenge = trickster._build_evidence_gap_challenge(cross_analysis)

        assert "EVIDENCE GAP DETECTED" in challenge
        assert "without supporting evidence" in challenge
        assert "agent1" in challenge


# =============================================================================
# Echo Chamber Intervention Tests
# =============================================================================


class TestEchoChamberIntervention:
    """Tests for echo chamber detection and intervention."""

    def test_echo_chamber_intervention_created(self):
        """Test echo chamber intervention when redundancy is high."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        # Mock cross-analyzer to return high redundancy
        cross_analysis = CrossProposalAnalysis(
            evidence_gaps=[],
            redundancy_score=0.85,  # High redundancy triggers echo chamber
            unique_evidence_sources=2,
            total_evidence_sources=20,
            agent_coverage={"agent1": 0.8, "agent2": 0.8},
            evidence_corroboration_score=0.9,
        )
        trickster._cross_analyzer.analyze = MagicMock(return_value=cross_analysis)

        trickster._analyzer.analyze_batch = MagicMock(return_value={})
        trickster._detector.check = MagicMock(
            return_value=HollowConsensusAlert(
                detected=False, severity=0.2, round_num=1, avg_quality=0.7
            )
        )

        result = trickster.check_and_intervene(
            responses={"agent1": "Same source", "agent2": "Same source"},
            convergence_similarity=0.8,
            round_num=1,
        )

        assert result is not None
        assert result.intervention_type == InterventionType.ECHO_CHAMBER

    def test_echo_chamber_challenge_text(self):
        """Test echo chamber challenge text content."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        cross_analysis = CrossProposalAnalysis(
            evidence_gaps=[],
            redundancy_score=0.9,
            unique_evidence_sources=2,
            total_evidence_sources=15,
            agent_coverage={"agent1": 0.9, "agent2": 0.9},
            evidence_corroboration_score=0.95,
        )

        challenge = trickster._build_echo_chamber_challenge(cross_analysis)

        assert "ECHO CHAMBER WARNING" in challenge
        assert "90%" in challenge  # redundancy_score
        assert "Unique evidence sources: 2" in challenge


# =============================================================================
# Novelty Challenge Tests
# =============================================================================


class TestNoveltyChallenge:
    """Tests for novelty challenge creation."""

    def test_create_novelty_challenge(self):
        """Test creating a novelty challenge."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        result = trickster.create_novelty_challenge(
            low_novelty_agents=["agent1", "agent2"],
            novelty_scores={"agent1": 0.2, "agent2": 0.3},
            round_num=3,
        )

        assert result is not None
        assert result.intervention_type == InterventionType.NOVELTY_CHALLENGE
        assert "agent1" in result.target_agents
        assert "agent2" in result.target_agents
        assert result.priority == pytest.approx(0.8, abs=0.01)  # 1 - 0.2 = 0.8

    def test_novelty_challenge_empty_agents(self):
        """Test novelty challenge with no low novelty agents."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        result = trickster.create_novelty_challenge(
            low_novelty_agents=[],
            novelty_scores={},
            round_num=3,
        )

        assert result is None

    def test_novelty_challenge_respects_cooldown(self):
        """Test novelty challenge respects cooldown."""
        config = TricksterConfig(intervention_cooldown_rounds=2)
        trickster = EvidencePoweredTrickster(config=config, linker=MagicMock())

        # Set recent intervention
        trickster._state.last_intervention_round = 2

        result = trickster.create_novelty_challenge(
            low_novelty_agents=["agent1"],
            novelty_scores={"agent1": 0.1},
            round_num=3,  # Only 1 round since last
        )

        assert result is None

    def test_novelty_challenge_text(self):
        """Test novelty challenge text content."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        challenge = trickster._build_novelty_challenge(
            low_novelty_agents=["agent1", "agent2"],
            novelty_scores={"agent1": 0.15, "agent2": 0.25},
        )

        assert "NOVELTY CHALLENGE" in challenge
        assert "too similar" in challenge
        assert "agent1" in challenge
        assert "15%" in challenge  # agent1 novelty score


# =============================================================================
# Intervention Type Selection Tests
# =============================================================================


class TestInterventionTypeSelection:
    """Tests for intervention type selection logic."""

    def test_high_severity_triggers_breakpoint(self):
        """Test high severity triggers breakpoint intervention."""
        config = TricksterConfig(enable_breakpoints=True)
        trickster = EvidencePoweredTrickster(config=config, linker=MagicMock())

        alert = HollowConsensusAlert(detected=True, severity=0.85, round_num=1, avg_quality=0.2)

        intervention_type = trickster._select_intervention_type(alert, round_num=1)
        assert intervention_type == InterventionType.BREAKPOINT

    def test_first_intervention_gets_role(self):
        """Test first intervention assigns quality role."""
        config = TricksterConfig(enable_role_assignment=True, enable_breakpoints=False)
        trickster = EvidencePoweredTrickster(config=config, linker=MagicMock())

        alert = HollowConsensusAlert(detected=True, severity=0.6, round_num=1, avg_quality=0.4)

        intervention_type = trickster._select_intervention_type(alert, round_num=1)
        assert intervention_type == InterventionType.QUALITY_ROLE

    def test_subsequent_interventions_use_prompts(self):
        """Test subsequent interventions use challenge prompts."""
        config = TricksterConfig(enable_role_assignment=True)
        trickster = EvidencePoweredTrickster(config=config, linker=MagicMock())
        trickster._state.total_interventions = 1  # Not first intervention

        alert = HollowConsensusAlert(detected=True, severity=0.6, round_num=2, avg_quality=0.4)

        intervention_type = trickster._select_intervention_type(alert, round_num=2)
        assert intervention_type == InterventionType.CHALLENGE_PROMPT

    def test_disabled_breakpoints_uses_prompt(self):
        """Test disabled breakpoints falls back to prompt."""
        config = TricksterConfig(enable_breakpoints=False, enable_role_assignment=False)
        trickster = EvidencePoweredTrickster(config=config, linker=MagicMock())

        alert = HollowConsensusAlert(detected=True, severity=0.9, round_num=1, avg_quality=0.1)

        intervention_type = trickster._select_intervention_type(alert, round_num=1)
        assert intervention_type == InterventionType.CHALLENGE_PROMPT


# =============================================================================
# Role Assignment Tests
# =============================================================================


class TestRoleAssignment:
    """Tests for quality challenger role assignment."""

    def test_get_quality_challenger_assignment(self):
        """Test getting quality challenger role assignment."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        assignment = trickster.get_quality_challenger_assignment(agent_name="agent1", round_num=2)

        assert assignment.agent_name == "agent1"
        assert assignment.role == CognitiveRole.QUALITY_CHALLENGER
        assert assignment.round_num == 2
        assert "QUALITY_CHALLENGER" in str(assignment.role)


# =============================================================================
# Stats and Reset Tests
# =============================================================================


class TestStatsAndReset:
    """Tests for statistics and reset functionality."""

    def test_get_stats_empty(self):
        """Test stats with no interventions."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        stats = trickster.get_stats()

        assert stats["total_interventions"] == 0
        assert stats["hollow_alerts_detected"] == 0
        assert stats["avg_quality_per_round"] == []
        assert stats["interventions"] == []

    def test_get_stats_with_data(self):
        """Test stats with interventions and alerts."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        # Add some state
        trickster._state.total_interventions = 2
        trickster._state.hollow_alerts = [
            HollowConsensusAlert(detected=True, severity=0.7, round_num=1, avg_quality=0.3),
            HollowConsensusAlert(detected=False, severity=0.2, round_num=2, avg_quality=0.8),
        ]
        trickster._state.quality_history = [
            {"agent1": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.4)},
            {"agent1": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.7)},
        ]
        trickster._state.interventions = [
            TricksterIntervention(
                intervention_type=InterventionType.CHALLENGE_PROMPT,
                round_num=1,
                target_agents=["agent1"],
                challenge_text="Challenge",
                evidence_gaps={},
                priority=0.7,
            )
        ]

        stats = trickster.get_stats()

        assert stats["total_interventions"] == 2
        assert stats["hollow_alerts_detected"] == 1
        assert len(stats["avg_quality_per_round"]) == 2
        assert len(stats["interventions"]) == 1

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        # Add some state
        trickster._state.total_interventions = 5
        trickster._state.last_intervention_round = 10
        trickster._state.hollow_alerts.append(
            HollowConsensusAlert(detected=True, severity=0.5, round_num=1, avg_quality=0.4)
        )

        trickster.reset()

        assert trickster._state.total_interventions == 0
        assert trickster._state.last_intervention_round == -10
        assert len(trickster._state.hollow_alerts) == 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateTricksterForDebate:
    """Tests for create_trickster_for_debate factory."""

    @patch("aragora.debate.trickster._get_evidence_linker_class", return_value=None)
    def test_factory_default(self, mock_get_linker):
        """Test factory with default parameters."""
        trickster = create_trickster_for_debate()

        assert trickster.config.min_quality_threshold == 0.4
        assert trickster.config.enable_breakpoints is True

    @patch("aragora.debate.trickster._get_evidence_linker_class", return_value=None)
    def test_factory_custom_quality(self, mock_get_linker):
        """Test factory with custom quality threshold."""
        trickster = create_trickster_for_debate(min_quality=0.7)

        assert trickster.config.min_quality_threshold == 0.7

    @patch("aragora.debate.trickster._get_evidence_linker_class", return_value=None)
    def test_factory_disabled_breakpoints(self, mock_get_linker):
        """Test factory with breakpoints disabled."""
        trickster = create_trickster_for_debate(enable_breakpoints=False)

        assert trickster.config.enable_breakpoints is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestTricksterIntegration:
    """Integration tests for the trickster system."""

    def test_full_lifecycle(self):
        """Test full trickster lifecycle across multiple rounds."""
        intervention_log = []
        alert_log = []

        trickster = EvidencePoweredTrickster(
            config=TricksterConfig(
                min_quality_threshold=0.5,
                hollow_detection_threshold=0.4,
                intervention_cooldown_rounds=1,
                max_interventions_total=3,
            ),
            on_intervention=lambda i: intervention_log.append(i),
            on_alert=lambda a: alert_log.append(a),
            linker=MagicMock(),
        )

        # Round 1: Good quality, no intervention
        trickster._analyzer.analyze_batch = MagicMock(
            return_value={
                "agent1": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.8)
            }
        )
        trickster._detector.check = MagicMock(
            return_value=HollowConsensusAlert(
                detected=False, severity=0.1, round_num=1, avg_quality=0.8
            )
        )

        result1 = trickster.check_and_intervene(
            responses={"agent1": "Good response"},
            convergence_similarity=0.3,
            round_num=1,
        )
        assert result1 is None

        # Round 2: Poor quality, should intervene
        trickster._analyzer.analyze_batch = MagicMock(
            return_value={
                "agent1": EvidenceQualityScore(
                    agent="agent1",
                    round_num=2,
                    overall_quality=0.3,
                    citation_density=0.1,
                    specificity_score=0.2,
                    logical_chain_score=0.1,
                    evidence_diversity=0.1,
                )
            }
        )
        trickster._detector.check = MagicMock(
            return_value=HollowConsensusAlert(
                detected=True,
                severity=0.7,
                round_num=2,
                avg_quality=0.3,
                agent_scores={"agent1": 0.3},
                reason="Low quality",
            )
        )

        result2 = trickster.check_and_intervene(
            responses={"agent1": "Vague response"},
            convergence_similarity=0.8,
            round_num=2,
        )
        assert result2 is not None
        assert len(intervention_log) == 1

        # Check stats
        stats = trickster.get_stats()
        assert stats["total_interventions"] == 1

    def test_quality_history_tracking(self):
        """Test that quality history is tracked across rounds."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        scores_r1 = {
            "agent1": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.6),
            "agent2": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.7),
        }
        scores_r2 = {
            "agent1": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.8),
            "agent2": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.9),
        }

        trickster._analyzer.analyze_batch = MagicMock(side_effect=[scores_r1, scores_r2])
        trickster._detector.check = MagicMock(
            return_value=HollowConsensusAlert(
                detected=False, severity=0.1, round_num=1, avg_quality=0.7
            )
        )

        trickster.check_and_intervene(
            responses={"agent1": "R1", "agent2": "R1"},
            convergence_similarity=0.5,
            round_num=1,
        )
        trickster.check_and_intervene(
            responses={"agent1": "R2", "agent2": "R2"},
            convergence_similarity=0.5,
            round_num=2,
        )

        assert len(trickster._state.quality_history) == 2


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_responses(self):
        """Test handling of empty responses."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        trickster._analyzer.analyze_batch = MagicMock(return_value={})
        trickster._detector.check = MagicMock(
            return_value=HollowConsensusAlert(
                detected=False, severity=0.0, round_num=1, avg_quality=0.0
            )
        )

        result = trickster.check_and_intervene(
            responses={},
            convergence_similarity=0.0,
            round_num=1,
        )

        assert result is None

    def test_single_agent(self):
        """Test with single agent."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        trickster._analyzer.analyze_batch = MagicMock(
            return_value={
                "agent1": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.3)
            }
        )
        trickster._detector.check = MagicMock(
            return_value=HollowConsensusAlert(
                detected=True,
                severity=0.6,
                round_num=1,
                avg_quality=0.3,
                agent_scores={"agent1": 0.3},
            )
        )

        result = trickster.check_and_intervene(
            responses={"agent1": "Solo response"},
            convergence_similarity=1.0,
            round_num=1,
        )

        # Should still be able to intervene with single agent
        assert result is not None

    def test_very_high_convergence_no_hollow(self):
        """Test high convergence with good quality doesn't trigger."""
        trickster = EvidencePoweredTrickster(linker=MagicMock())

        # Create mock cross analysis result (no concerns)
        mock_cross_analysis = MagicMock()
        mock_cross_analysis.has_concerns = False
        mock_cross_analysis.evidence_gaps = []
        mock_cross_analysis.contradictory_evidence = []
        mock_cross_analysis.redundancy_score = 0.3  # Low redundancy

        # Properly mock the internal components
        with (
            patch.object(
                trickster._analyzer,
                "analyze_batch",
                return_value={
                    "agent1": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.9),
                    "agent2": EvidenceQualityScore(agent="agent", round_num=1, overall_quality=0.9),
                },
            ),
            patch.object(
                trickster._detector,
                "check",
                return_value=HollowConsensusAlert(
                    detected=False,
                    severity=0.0,
                    reason="High quality responses",
                    agent_scores={"agent1": 0.9, "agent2": 0.9},
                    recommended_challenges=[],
                    avg_quality=0.9,
                ),
            ),
            patch.object(
                trickster._cross_analyzer,
                "analyze",
                return_value=mock_cross_analysis,
            ),
        ):
            result = trickster.check_and_intervene(
                responses={"agent1": "Well-supported claim", "agent2": "Also well-supported"},
                convergence_similarity=0.99,
                round_num=1,
            )

            assert result is None  # High convergence but good quality = no intervention
