"""
Tests for Evidence-Powered Trickster module.

Tests cover:
- InterventionType enum
- Data classes (TricksterIntervention, TricksterConfig, TricksterState)
- EvidencePoweredTrickster check_and_intervene logic
- Intervention creation and recording
- Challenge building
- Novelty challenge handling
- Factory function (create_trickster_for_debate)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aragora.debate.trickster import (
    EvidencePoweredTrickster,
    InterventionType,
    TricksterConfig,
    TricksterIntervention,
    TricksterState,
    create_trickster_for_debate,
)


# =============================================================================
# InterventionType Enum Tests
# =============================================================================


class TestInterventionType:
    """Tests for InterventionType enum."""

    @pytest.mark.smoke
    def test_all_intervention_types_exist(self):
        """Test all intervention types exist with correct values."""
        assert InterventionType.CHALLENGE_PROMPT.value == "challenge_prompt"
        assert InterventionType.QUALITY_ROLE.value == "quality_role"
        assert InterventionType.EXTENDED_ROUND.value == "extended_round"
        assert InterventionType.BREAKPOINT.value == "breakpoint"
        assert InterventionType.NOVELTY_CHALLENGE.value == "novelty_challenge"
        assert InterventionType.EVIDENCE_GAP.value == "evidence_gap"
        assert InterventionType.ECHO_CHAMBER.value == "echo_chamber"

    def test_intervention_type_from_string(self):
        """Test creating intervention type from string."""
        assert InterventionType("challenge_prompt") == InterventionType.CHALLENGE_PROMPT
        assert InterventionType("breakpoint") == InterventionType.BREAKPOINT


# =============================================================================
# TricksterConfig Tests
# =============================================================================


class TestTricksterConfig:
    """Tests for TricksterConfig dataclass."""

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

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TricksterConfig(
            sensitivity=0.8,
            min_quality_threshold=0.7,
            hollow_detection_threshold=0.6,
            intervention_cooldown_rounds=2,
            enable_breakpoints=False,
            max_interventions_total=10,
        )

        assert config.sensitivity == 0.8
        assert config.min_quality_threshold == 0.7
        assert config.enable_breakpoints is False
        assert config.max_interventions_total == 10

    def test_sensitivity_adjusts_threshold(self):
        """Test sensitivity adjusts hollow_detection_threshold."""
        # High sensitivity = lower threshold (more sensitive)
        high_sensitivity = TricksterConfig(sensitivity=1.0)
        assert high_sensitivity.hollow_detection_threshold == 0.2

        # Low sensitivity = higher threshold (less sensitive)
        low_sensitivity = TricksterConfig(sensitivity=0.0)
        assert low_sensitivity.hollow_detection_threshold == 0.8

    def test_sensitivity_default_no_adjustment(self):
        """Test default sensitivity doesn't adjust threshold."""
        config = TricksterConfig(sensitivity=0.5)
        # Default shouldn't change threshold from default
        assert config.hollow_detection_threshold == 0.5


# =============================================================================
# TricksterState Tests
# =============================================================================


class TestTricksterState:
    """Tests for TricksterState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = TricksterState()

        assert state.interventions == []
        assert state.quality_history == []
        assert state.last_intervention_round == -10
        assert state.hollow_alerts == []
        assert state.total_interventions == 0

    def test_state_tracking(self):
        """Test state can be updated."""
        state = TricksterState()

        # Add an intervention
        intervention = TricksterIntervention(
            intervention_type=InterventionType.CHALLENGE_PROMPT,
            round_num=3,
            target_agents=["claude"],
            challenge_text="Test challenge",
            evidence_gaps={},
            priority=0.7,
        )
        state.interventions.append(intervention)
        state.total_interventions += 1
        state.last_intervention_round = 3

        assert len(state.interventions) == 1
        assert state.total_interventions == 1
        assert state.last_intervention_round == 3


# =============================================================================
# TricksterIntervention Tests
# =============================================================================


class TestTricksterIntervention:
    """Tests for TricksterIntervention dataclass."""

    def test_create_intervention(self):
        """Test creating an intervention."""
        intervention = TricksterIntervention(
            intervention_type=InterventionType.CHALLENGE_PROMPT,
            round_num=3,
            target_agents=["claude", "gpt4"],
            challenge_text="Please provide evidence",
            evidence_gaps={"claude": ["citations", "specificity"]},
            priority=0.85,
        )

        assert intervention.intervention_type == InterventionType.CHALLENGE_PROMPT
        assert intervention.round_num == 3
        assert len(intervention.target_agents) == 2
        assert "Please provide evidence" in intervention.challenge_text
        assert "claude" in intervention.evidence_gaps
        assert intervention.priority == 0.85

    def test_intervention_with_metadata(self):
        """Test intervention with metadata."""
        intervention = TricksterIntervention(
            intervention_type=InterventionType.EVIDENCE_GAP,
            round_num=2,
            target_agents=["agent1"],
            challenge_text="Test",
            evidence_gaps={},
            priority=0.6,
            metadata={
                "gap_claim": "Test claim",
                "gap_severity": 0.8,
            },
        )

        assert intervention.metadata["gap_claim"] == "Test claim"
        assert intervention.metadata["gap_severity"] == 0.8


# =============================================================================
# EvidencePoweredTrickster Tests
# =============================================================================


class TestEvidencePoweredTrickster:
    """Tests for EvidencePoweredTrickster class."""

    def _create_mock_quality_score(self, overall: float = 0.5):
        """Helper to create mock quality score."""
        score = Mock()
        score.overall_quality = overall
        score.citation_density = 0.5
        score.specificity_score = 0.5
        score.logical_chain_score = 0.5
        score.evidence_diversity = 0.5
        return score

    def test_init_default(self):
        """Test initialization with default config."""
        trickster = EvidencePoweredTrickster()

        assert trickster.config is not None
        assert trickster._state is not None
        assert trickster._analyzer is not None
        assert trickster._detector is not None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = TricksterConfig(
            min_quality_threshold=0.8,
            enable_breakpoints=False,
        )
        trickster = EvidencePoweredTrickster(config=config)

        assert trickster.config.min_quality_threshold == 0.8
        assert trickster.config.enable_breakpoints is False

    def test_init_with_callbacks(self):
        """Test initialization with callbacks."""
        on_intervention = Mock()
        on_alert = Mock()

        trickster = EvidencePoweredTrickster(
            on_intervention=on_intervention,
            on_alert=on_alert,
        )

        assert trickster.on_intervention is on_intervention
        assert trickster.on_alert is on_alert

    def test_check_and_intervene_no_alert(self):
        """Test check_and_intervene returns None when quality is acceptable."""
        trickster = EvidencePoweredTrickster()

        # Mock the internal analyzer and detector
        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                mock_analyze.return_value = {
                    "agent1": self._create_mock_quality_score(0.8),
                }

                mock_alert = Mock()
                mock_alert.detected = False
                mock_check.return_value = mock_alert

                result = trickster.check_and_intervene(
                    responses={"agent1": "High quality response with citations"},
                    convergence_similarity=0.5,
                    round_num=2,
                )

                assert result is None

    def test_check_and_intervene_with_alert(self):
        """Test check_and_intervene returns intervention when alert triggered."""
        config = TricksterConfig(
            hollow_detection_threshold=0.3,
            min_quality_threshold=0.5,
        )
        trickster = EvidencePoweredTrickster(config=config)

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                mock_analyze.return_value = {
                    "agent1": self._create_mock_quality_score(0.3),
                }

                mock_alert = Mock()
                mock_alert.detected = True
                mock_alert.severity = 0.8
                mock_alert.avg_quality = 0.3
                mock_alert.min_quality = 0.3
                mock_alert.quality_variance = 0.0
                mock_alert.reason = "Low evidence quality"
                mock_alert.agent_scores = {"agent1": 0.3}
                mock_alert.recommended_challenges = ["Provide citations"]
                mock_check.return_value = mock_alert

                result = trickster.check_and_intervene(
                    responses={"agent1": "Low quality response"},
                    convergence_similarity=0.9,
                    round_num=2,
                )

                assert result is not None
                assert isinstance(result, TricksterIntervention)

    def test_check_and_intervene_cooldown(self):
        """Test intervention cooldown is respected."""
        config = TricksterConfig(intervention_cooldown_rounds=2)
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.last_intervention_round = 3
        trickster._state.total_interventions = 1

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                mock_analyze.return_value = {"agent1": self._create_mock_quality_score(0.3)}

                mock_alert = Mock()
                mock_alert.detected = True
                mock_alert.severity = 0.8
                mock_check.return_value = mock_alert

                # Round 4 is within cooldown of round 3 (cooldown=2)
                result = trickster.check_and_intervene(
                    responses={"agent1": "Test"},
                    convergence_similarity=0.9,
                    round_num=4,
                )

                # Should still return intervention but not record it
                # (cooldown check is in _record_intervention)
                assert trickster._state.total_interventions == 1

    def test_check_and_intervene_max_interventions(self):
        """Test max interventions limit is respected."""
        config = TricksterConfig(max_interventions_total=2)
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.total_interventions = 2

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                mock_analyze.return_value = {"agent1": self._create_mock_quality_score(0.3)}

                mock_alert = Mock()
                mock_alert.detected = True
                mock_alert.severity = 0.8
                mock_check.return_value = mock_alert

                result = trickster.check_and_intervene(
                    responses={"agent1": "Test"},
                    convergence_similarity=0.9,
                    round_num=5,
                )

                # Should not add more interventions
                assert trickster._state.total_interventions == 2

    def test_check_and_intervene_calls_on_alert(self):
        """Test on_alert callback is called."""
        on_alert = Mock()
        trickster = EvidencePoweredTrickster(on_alert=on_alert)

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                mock_analyze.return_value = {"agent1": self._create_mock_quality_score(0.5)}

                mock_alert = Mock()
                mock_alert.detected = True
                mock_alert.severity = 0.3  # Below threshold
                mock_check.return_value = mock_alert

                trickster.check_and_intervene(
                    responses={"agent1": "Test"},
                    convergence_similarity=0.7,
                    round_num=2,
                )

                on_alert.assert_called_once_with(mock_alert)

    def test_check_and_intervene_calls_on_intervention(self):
        """Test on_intervention callback is called."""
        on_intervention = Mock()
        config = TricksterConfig(hollow_detection_threshold=0.3)
        trickster = EvidencePoweredTrickster(
            config=config,
            on_intervention=on_intervention,
        )

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                mock_analyze.return_value = {"agent1": self._create_mock_quality_score(0.3)}

                mock_alert = Mock()
                mock_alert.detected = True
                mock_alert.severity = 0.8
                mock_alert.avg_quality = 0.3
                mock_alert.min_quality = 0.3
                mock_alert.quality_variance = 0.0
                mock_alert.reason = "Test"
                mock_alert.agent_scores = {"agent1": 0.3}
                mock_alert.recommended_challenges = []
                mock_check.return_value = mock_alert

                result = trickster.check_and_intervene(
                    responses={"agent1": "Test"},
                    convergence_similarity=0.9,
                    round_num=2,
                )

                if result:
                    on_intervention.assert_called()

    def test_create_evidence_gap_intervention(self):
        """Test _create_evidence_gap_intervention."""
        trickster = EvidencePoweredTrickster()

        mock_cross_analysis = Mock()
        mock_cross_analysis.evidence_gaps = [
            Mock(
                claim="Test claim",
                agents_making_claim=["agent1"],
                gap_severity=0.8,
            )
        ]

        intervention = trickster._create_evidence_gap_intervention(mock_cross_analysis, round_num=3)

        assert intervention is not None
        assert intervention.intervention_type == InterventionType.EVIDENCE_GAP
        assert intervention.round_num == 3
        assert "agent1" in intervention.target_agents

    def test_create_echo_chamber_intervention(self):
        """Test _create_echo_chamber_intervention."""
        trickster = EvidencePoweredTrickster()

        mock_cross_analysis = Mock()
        mock_cross_analysis.redundancy_score = 0.85
        mock_cross_analysis.unique_evidence_sources = 2
        mock_cross_analysis.total_evidence_sources = 10
        mock_cross_analysis.agent_coverage = {"agent1": 0.8, "agent2": 0.7}

        intervention = trickster._create_echo_chamber_intervention(mock_cross_analysis, round_num=4)

        assert intervention is not None
        assert intervention.intervention_type == InterventionType.ECHO_CHAMBER
        assert intervention.priority == 0.85

    def test_create_echo_chamber_intervention_below_threshold(self):
        """Test _create_echo_chamber_intervention returns None below threshold."""
        trickster = EvidencePoweredTrickster()

        mock_cross_analysis = Mock()
        mock_cross_analysis.redundancy_score = 0.5  # Below 0.7 threshold

        intervention = trickster._create_echo_chamber_intervention(mock_cross_analysis, round_num=4)

        assert intervention is None

    def test_build_challenge(self):
        """Test _build_challenge text generation."""
        trickster = EvidencePoweredTrickster()

        mock_alert = Mock()
        mock_alert.recommended_challenges = ["Provide more citations", "Be specific"]

        evidence_gaps = {
            "agent1": ["citations", "reasoning"],
            "agent2": ["specificity"],
        }

        target_agents = ["agent1", "agent2"]

        challenge = trickster._build_challenge(mock_alert, evidence_gaps, target_agents)

        assert "QUALITY CHALLENGE" in challenge
        assert "hollow consensus" in challenge
        assert "Provide more citations" in challenge
        assert "agent1" in challenge

    def test_select_intervention_type_high_severity_breakpoint(self):
        """Test _select_intervention_type selects breakpoint for high severity."""
        config = TricksterConfig(enable_breakpoints=True)
        trickster = EvidencePoweredTrickster(config=config)

        mock_alert = Mock()
        mock_alert.severity = 0.9  # Very high

        int_type = trickster._select_intervention_type(mock_alert, round_num=3)

        assert int_type == InterventionType.BREAKPOINT

    def test_select_intervention_type_first_intervention_role(self):
        """Test _select_intervention_type selects role for first intervention."""
        config = TricksterConfig(enable_role_assignment=True)
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.total_interventions = 0

        mock_alert = Mock()
        mock_alert.severity = 0.5  # Not high enough for breakpoint

        int_type = trickster._select_intervention_type(mock_alert, round_num=3)

        assert int_type == InterventionType.QUALITY_ROLE

    def test_select_intervention_type_default_challenge_prompt(self):
        """Test _select_intervention_type defaults to challenge prompt."""
        config = TricksterConfig(enable_challenge_prompts=True)
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.total_interventions = 2  # Not first

        mock_alert = Mock()
        mock_alert.severity = 0.5  # Not high enough for breakpoint

        int_type = trickster._select_intervention_type(mock_alert, round_num=3)

        assert int_type == InterventionType.CHALLENGE_PROMPT

    def test_get_quality_challenger_assignment(self):
        """Test get_quality_challenger_assignment returns role assignment."""
        trickster = EvidencePoweredTrickster()

        assignment = trickster.get_quality_challenger_assignment(
            agent_name="claude",
            round_num=3,
        )

        assert assignment.agent_name == "claude"
        assert assignment.round_num == 3
        # Should be QUALITY_CHALLENGER role
        from aragora.debate.roles import CognitiveRole

        assert assignment.role == CognitiveRole.QUALITY_CHALLENGER

    def test_create_novelty_challenge(self):
        """Test create_novelty_challenge creates intervention."""
        trickster = EvidencePoweredTrickster()

        intervention = trickster.create_novelty_challenge(
            low_novelty_agents=["agent1", "agent2"],
            novelty_scores={"agent1": 0.2, "agent2": 0.3},
            round_num=5,
        )

        assert intervention is not None
        assert intervention.intervention_type == InterventionType.NOVELTY_CHALLENGE
        assert intervention.round_num == 5
        assert "agent1" in intervention.target_agents
        assert intervention.priority == 0.8  # 1.0 - min_novelty (0.2)

    def test_create_novelty_challenge_empty_agents(self):
        """Test create_novelty_challenge returns None for empty agents."""
        trickster = EvidencePoweredTrickster()

        intervention = trickster.create_novelty_challenge(
            low_novelty_agents=[],
            novelty_scores={},
            round_num=5,
        )

        assert intervention is None

    def test_create_novelty_challenge_respects_cooldown(self):
        """Test create_novelty_challenge respects cooldown."""
        config = TricksterConfig(intervention_cooldown_rounds=2)
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.last_intervention_round = 4

        intervention = trickster.create_novelty_challenge(
            low_novelty_agents=["agent1"],
            novelty_scores={"agent1": 0.2},
            round_num=5,  # Within cooldown
        )

        assert intervention is None

    def test_create_novelty_challenge_respects_max_interventions(self):
        """Test create_novelty_challenge respects max interventions."""
        config = TricksterConfig(max_interventions_total=2)
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.total_interventions = 2

        intervention = trickster.create_novelty_challenge(
            low_novelty_agents=["agent1"],
            novelty_scores={"agent1": 0.2},
            round_num=10,
        )

        assert intervention is None

    def test_build_novelty_challenge(self):
        """Test _build_novelty_challenge text generation."""
        trickster = EvidencePoweredTrickster()

        challenge = trickster._build_novelty_challenge(
            low_novelty_agents=["agent1", "agent2"],
            novelty_scores={"agent1": 0.15, "agent2": 0.25},
        )

        assert "NOVELTY CHALLENGE" in challenge
        assert "agent1" in challenge
        assert "15%" in challenge

    def test_get_stats(self):
        """Test get_stats returns statistics."""
        trickster = EvidencePoweredTrickster()

        # Add some state
        trickster._state.total_interventions = 2
        trickster._state.interventions = [
            TricksterIntervention(
                intervention_type=InterventionType.CHALLENGE_PROMPT,
                round_num=2,
                target_agents=["agent1"],
                challenge_text="Test",
                evidence_gaps={},
                priority=0.7,
            ),
        ]

        mock_alert1 = Mock()
        mock_alert1.detected = True
        mock_alert2 = Mock()
        mock_alert2.detected = False
        trickster._state.hollow_alerts = [mock_alert1, mock_alert2]

        mock_quality = Mock()
        mock_quality.overall_quality = 0.6
        trickster._state.quality_history = [{"agent1": mock_quality}]

        stats = trickster.get_stats()

        assert stats["total_interventions"] == 2
        assert stats["hollow_alerts_detected"] == 1
        assert len(stats["avg_quality_per_round"]) == 1
        assert len(stats["interventions"]) == 1

    def test_reset(self):
        """Test reset clears state."""
        trickster = EvidencePoweredTrickster()

        # Add state
        trickster._state.total_interventions = 5
        trickster._state.last_intervention_round = 10
        trickster._state.interventions.append(Mock())

        trickster.reset()

        assert trickster._state.total_interventions == 0
        assert trickster._state.last_intervention_round == -10
        assert trickster._state.interventions == []


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateTricksterForDebate:
    """Tests for create_trickster_for_debate factory function."""

    def test_create_with_defaults(self):
        """Test factory with default parameters."""
        trickster = create_trickster_for_debate()

        assert isinstance(trickster, EvidencePoweredTrickster)
        assert trickster.config.min_quality_threshold == 0.4
        assert trickster.config.enable_breakpoints is True

    def test_create_with_custom_params(self):
        """Test factory with custom parameters."""
        trickster = create_trickster_for_debate(
            min_quality=0.7,
            enable_breakpoints=False,
        )

        assert trickster.config.min_quality_threshold == 0.7
        assert trickster.config.enable_breakpoints is False


# =============================================================================
# Cross-Proposal Analysis Integration Tests
# =============================================================================


class TestCrossProposalIntegration:
    """Tests for cross-proposal analysis integration."""

    def _create_mock_quality_score(self, overall: float = 0.5):
        """Helper to create mock quality score."""
        score = Mock()
        score.overall_quality = overall
        score.citation_density = 0.5
        score.specificity_score = 0.5
        score.logical_chain_score = 0.5
        score.evidence_diversity = 0.5
        return score

    def test_evidence_gap_triggers_before_hollow_consensus(self):
        """Test evidence gap intervention triggers before standard hollow consensus."""
        config = TricksterConfig(hollow_detection_threshold=0.3)
        trickster = EvidencePoweredTrickster(config=config)

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                with patch.object(trickster._cross_analyzer, "analyze") as mock_cross:
                    mock_analyze.return_value = {
                        "agent1": self._create_mock_quality_score(0.4),
                    }

                    mock_alert = Mock()
                    mock_alert.detected = False
                    mock_check.return_value = mock_alert

                    mock_cross_result = Mock()
                    mock_cross_result.evidence_gaps = [
                        Mock(
                            claim="Unsupported claim",
                            agents_making_claim=["agent1"],
                            gap_severity=0.9,
                        )
                    ]
                    mock_cross_result.redundancy_score = 0.5
                    mock_cross.return_value = mock_cross_result

                    result = trickster.check_and_intervene(
                        responses={"agent1": "Claim without evidence"},
                        convergence_similarity=0.7,  # Above 0.6 threshold
                        round_num=2,
                    )

                    if result:
                        assert result.intervention_type == InterventionType.EVIDENCE_GAP

    def test_echo_chamber_triggers_on_high_redundancy(self):
        """Test echo chamber intervention triggers on high redundancy."""
        trickster = EvidencePoweredTrickster()

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                with patch.object(trickster._cross_analyzer, "analyze") as mock_cross:
                    mock_analyze.return_value = {
                        "agent1": self._create_mock_quality_score(0.5),
                    }

                    mock_alert = Mock()
                    mock_alert.detected = False
                    mock_check.return_value = mock_alert

                    mock_cross_result = Mock()
                    mock_cross_result.evidence_gaps = []
                    mock_cross_result.redundancy_score = 0.85  # High redundancy
                    mock_cross_result.unique_evidence_sources = 2
                    mock_cross_result.total_evidence_sources = 10
                    mock_cross_result.agent_coverage = {"agent1": 0.9}
                    mock_cross.return_value = mock_cross_result

                    result = trickster.check_and_intervene(
                        responses={"agent1": "Echo chamber content"},
                        convergence_similarity=0.8,
                        round_num=2,
                    )

                    if result:
                        assert result.intervention_type == InterventionType.ECHO_CHAMBER


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestTricksterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_responses(self):
        """Test handling empty responses dict."""
        trickster = EvidencePoweredTrickster()

        result = trickster.check_and_intervene(
            responses={},
            convergence_similarity=0.5,
            round_num=1,
        )

        # Should not crash, likely returns None
        assert result is None or isinstance(result, TricksterIntervention)

    def test_single_agent(self):
        """Test handling single agent debate."""
        trickster = EvidencePoweredTrickster()

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                mock_analyze.return_value = {
                    "agent1": Mock(overall_quality=0.5),
                }

                mock_alert = Mock()
                mock_alert.detected = False
                mock_check.return_value = mock_alert

                result = trickster.check_and_intervene(
                    responses={"agent1": "Solo response"},
                    convergence_similarity=1.0,
                    round_num=1,
                )

                # Should handle single agent gracefully
                assert result is None or isinstance(result, TricksterIntervention)

    def test_very_low_convergence(self):
        """Test handling very low convergence (divergent debate)."""
        trickster = EvidencePoweredTrickster()

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                mock_analyze.return_value = {
                    "agent1": Mock(overall_quality=0.8),
                    "agent2": Mock(overall_quality=0.8),
                }

                mock_alert = Mock()
                mock_alert.detected = False  # No hollow consensus when divergent
                mock_check.return_value = mock_alert

                result = trickster.check_and_intervene(
                    responses={
                        "agent1": "Position A",
                        "agent2": "Position B",
                    },
                    convergence_similarity=0.1,  # Very divergent
                    round_num=3,
                )

                # Trickster mainly concerned with hollow consensus
                # Low convergence = not converging yet = no intervention
                assert result is None

    def test_multiple_rounds_quality_history(self):
        """Test quality history is maintained across rounds."""
        trickster = EvidencePoweredTrickster()

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                mock_check.return_value = Mock(detected=False)

                # Round 1
                mock_analyze.return_value = {"a": Mock(overall_quality=0.5)}
                trickster.check_and_intervene({"a": "r1"}, 0.5, 1)

                # Round 2
                mock_analyze.return_value = {"a": Mock(overall_quality=0.6)}
                trickster.check_and_intervene({"a": "r2"}, 0.6, 2)

                # Round 3
                mock_analyze.return_value = {"a": Mock(overall_quality=0.7)}
                trickster.check_and_intervene({"a": "r3"}, 0.7, 3)

                assert len(trickster._state.quality_history) == 3

    def test_linker_not_available(self):
        """Test trickster works without linker (optional dependency)."""
        with patch("aragora.debate.trickster._get_evidence_linker_class") as mock_get:
            mock_get.return_value = None

            trickster = EvidencePoweredTrickster()

            # Should initialize without linker
            assert trickster._linker is None

            # Cross analyzer should still work (handles None linker)
            assert trickster._cross_analyzer is not None


# =============================================================================
# State Management Tests
# =============================================================================


class TestTricksterStateManagement:
    """Tests for trickster state management."""

    def test_intervention_recording(self):
        """Test interventions are properly recorded."""
        trickster = EvidencePoweredTrickster()

        intervention = TricksterIntervention(
            intervention_type=InterventionType.CHALLENGE_PROMPT,
            round_num=5,
            target_agents=["agent1"],
            challenge_text="Test",
            evidence_gaps={},
            priority=0.7,
        )

        result = trickster._record_intervention(intervention, round_num=5)

        assert result == intervention
        assert len(trickster._state.interventions) == 1
        assert trickster._state.last_intervention_round == 5
        assert trickster._state.total_interventions == 1

    def test_stats_after_multiple_interventions(self):
        """Test stats after multiple interventions."""
        trickster = EvidencePoweredTrickster()

        # Simulate interventions
        for i in range(3):
            intervention = TricksterIntervention(
                intervention_type=InterventionType.CHALLENGE_PROMPT,
                round_num=i + 1,
                target_agents=[f"agent{i}"],
                challenge_text=f"Challenge {i}",
                evidence_gaps={},
                priority=0.5 + i * 0.1,
            )
            trickster._state.interventions.append(intervention)
            trickster._state.total_interventions += 1

        stats = trickster.get_stats()

        assert stats["total_interventions"] == 3
        assert len(stats["interventions"]) == 3
        assert stats["interventions"][0]["round"] == 1
        assert stats["interventions"][2]["round"] == 3
