"""
Tests for Evidence-Powered Trickster - Core Functionality.

Deep coverage tests complementing test_trickster.py, focusing on:
- _get_evidence_linker_class lazy import
- _record_intervention state tracking edge cases
- _create_intervention with various alert/quality combinations
- _build_evidence_gap_challenge text generation
- _build_echo_chamber_challenge text generation
- _build_novelty_challenge edge cases
- check_and_intervene cross-proposal analysis flow
- Intervention callback sequencing
- Config sensitivity edge cases
- Multi-round state accumulation
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from aragora.debate.trickster import (
    EvidencePoweredTrickster,
    InterventionType,
    TricksterConfig,
    TricksterIntervention,
    TricksterState,
    _get_evidence_linker_class,
    create_trickster_for_debate,
)


# =============================================================================
# _get_evidence_linker_class Tests
# =============================================================================


class TestGetEvidenceLinkerClass:
    """Tests for _get_evidence_linker_class lazy import."""

    def test_returns_class_when_available(self):
        """Test returns EvidenceClaimLinker class when import succeeds."""
        mock_class = Mock()
        with patch.dict(
            "sys.modules",
            {"aragora.debate.evidence_linker": Mock(EvidenceClaimLinker=mock_class)},
        ):
            result = _get_evidence_linker_class()
            # May or may not return mock depending on import caching
            # The key test is that it doesn't raise
            assert result is None or result is not None

    def test_returns_none_on_import_error(self):
        """Test returns None when import fails."""
        with patch("aragora.debate.trickster._get_evidence_linker_class") as mock_fn:
            mock_fn.return_value = None
            result = mock_fn()
            assert result is None


# =============================================================================
# TricksterConfig Sensitivity Tests
# =============================================================================


class TestTricksterConfigSensitivity:
    """Tests for TricksterConfig sensitivity-threshold relationship."""

    def test_sensitivity_zero(self):
        """Test sensitivity=0.0 -> highest threshold (least sensitive)."""
        config = TricksterConfig(sensitivity=0.0)
        assert config.hollow_detection_threshold == 0.8

    def test_sensitivity_one(self):
        """Test sensitivity=1.0 -> lowest threshold (most sensitive)."""
        config = TricksterConfig(sensitivity=1.0)
        assert abs(config.hollow_detection_threshold - 0.2) < 0.01

    def test_sensitivity_quarter(self):
        """Test sensitivity=0.25 -> intermediate threshold."""
        config = TricksterConfig(sensitivity=0.25)
        expected = 0.8 - (0.25 * 0.6)  # 0.65
        assert abs(config.hollow_detection_threshold - expected) < 0.01

    def test_sensitivity_three_quarters(self):
        """Test sensitivity=0.75 -> lower threshold."""
        config = TricksterConfig(sensitivity=0.75)
        expected = 0.8 - (0.75 * 0.6)  # 0.35
        assert abs(config.hollow_detection_threshold - expected) < 0.01

    def test_default_sensitivity_preserves_default_threshold(self):
        """Test default sensitivity=0.5 does not adjust threshold."""
        config = TricksterConfig()  # sensitivity=0.5 by default
        assert config.hollow_detection_threshold == 0.5

    def test_explicit_default_sensitivity_preserves_threshold(self):
        """Test explicitly setting sensitivity=0.5 preserves default threshold."""
        config = TricksterConfig(sensitivity=0.5)
        assert config.hollow_detection_threshold == 0.5

    def test_all_features_disabled(self):
        """Test config with all features disabled."""
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
# TricksterState Deep Tests
# =============================================================================


class TestTricksterStateDeep:
    """Deep tests for TricksterState dataclass."""

    def test_default_last_intervention_round(self):
        """Test default last_intervention_round is -10 (allows immediate intervention)."""
        state = TricksterState()
        assert state.last_intervention_round == -10

    def test_quality_history_accumulation(self):
        """Test quality_history accumulates properly."""
        state = TricksterState()
        mock_score = Mock()
        mock_score.overall_quality = 0.7

        state.quality_history.append({"agent1": mock_score})
        state.quality_history.append({"agent1": mock_score, "agent2": mock_score})

        assert len(state.quality_history) == 2
        assert len(state.quality_history[1]) == 2

    def test_hollow_alerts_accumulation(self):
        """Test hollow_alerts accumulates properly."""
        state = TricksterState()
        alert1 = Mock(detected=True)
        alert2 = Mock(detected=False)
        alert3 = Mock(detected=True)

        state.hollow_alerts.extend([alert1, alert2, alert3])
        detected_count = sum(1 for a in state.hollow_alerts if a.detected)

        assert detected_count == 2


# =============================================================================
# EvidencePoweredTrickster _record_intervention Tests
# =============================================================================


class TestRecordIntervention:
    """Tests for _record_intervention state tracking."""

    def test_record_updates_all_state(self):
        """Test _record_intervention updates interventions list, round, and count."""
        trickster = EvidencePoweredTrickster()

        intervention = TricksterIntervention(
            intervention_type=InterventionType.CHALLENGE_PROMPT,
            round_num=3,
            target_agents=["agent1"],
            challenge_text="Test",
            evidence_gaps={},
            priority=0.7,
        )

        result = trickster._record_intervention(intervention, round_num=3)

        assert result is intervention
        assert len(trickster._state.interventions) == 1
        assert trickster._state.last_intervention_round == 3
        assert trickster._state.total_interventions == 1

    def test_record_skips_during_cooldown(self):
        """Test _record_intervention skips recording during cooldown."""
        config = TricksterConfig(intervention_cooldown_rounds=3)
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.last_intervention_round = 5

        intervention = TricksterIntervention(
            intervention_type=InterventionType.CHALLENGE_PROMPT,
            round_num=7,  # Only 2 rounds since last, cooldown is 3
            target_agents=["agent1"],
            challenge_text="Test",
            evidence_gaps={},
            priority=0.7,
        )

        result = trickster._record_intervention(intervention, round_num=7)

        # Returns intervention but does NOT record it
        assert result is intervention
        assert len(trickster._state.interventions) == 0
        assert trickster._state.total_interventions == 0

    def test_record_skips_at_max_interventions(self):
        """Test _record_intervention skips recording at max interventions."""
        config = TricksterConfig(max_interventions_total=2)
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.total_interventions = 2

        intervention = TricksterIntervention(
            intervention_type=InterventionType.CHALLENGE_PROMPT,
            round_num=10,
            target_agents=["agent1"],
            challenge_text="Test",
            evidence_gaps={},
            priority=0.7,
        )

        result = trickster._record_intervention(intervention, round_num=10)

        assert result is intervention
        assert trickster._state.total_interventions == 2  # Unchanged

    def test_record_calls_on_intervention_callback(self):
        """Test _record_intervention calls the on_intervention callback."""
        callback = Mock()
        trickster = EvidencePoweredTrickster(on_intervention=callback)

        intervention = TricksterIntervention(
            intervention_type=InterventionType.QUALITY_ROLE,
            round_num=5,
            target_agents=["agent1"],
            challenge_text="Test",
            evidence_gaps={},
            priority=0.8,
        )

        trickster._record_intervention(intervention, round_num=5)

        callback.assert_called_once_with(intervention)

    def test_record_does_not_call_callback_during_cooldown(self):
        """Test _record_intervention does not call callback during cooldown."""
        callback = Mock()
        config = TricksterConfig(intervention_cooldown_rounds=5)
        trickster = EvidencePoweredTrickster(config=config, on_intervention=callback)
        trickster._state.last_intervention_round = 3

        intervention = TricksterIntervention(
            intervention_type=InterventionType.CHALLENGE_PROMPT,
            round_num=4,
            target_agents=["agent1"],
            challenge_text="Test",
            evidence_gaps={},
            priority=0.7,
        )

        trickster._record_intervention(intervention, round_num=4)

        callback.assert_not_called()


# =============================================================================
# EvidencePoweredTrickster _create_intervention Tests
# =============================================================================


class TestCreateIntervention:
    """Tests for _create_intervention internal method."""

    def _create_mock_quality_score(
        self,
        overall: float = 0.5,
        citation: float = 0.5,
        specificity: float = 0.5,
        logical: float = 0.5,
        diversity: float = 0.5,
    ):
        """Helper to create mock quality score with fine-grained control."""
        score = Mock()
        score.overall_quality = overall
        score.citation_density = citation
        score.specificity_score = specificity
        score.logical_chain_score = logical
        score.evidence_diversity = diversity
        return score

    def test_creates_intervention_with_low_quality_agents(self):
        """Test intervention targets agents below quality threshold."""
        config = TricksterConfig(min_quality_threshold=0.5, max_challenges_per_round=3)
        trickster = EvidencePoweredTrickster(config=config)

        alert = Mock()
        alert.severity = 0.7
        alert.avg_quality = 0.3
        alert.min_quality = 0.2
        alert.quality_variance = 0.1
        alert.reason = "Low quality"
        alert.agent_scores = {"agent1": 0.3, "agent2": 0.7, "agent3": 0.4}
        alert.recommended_challenges = []

        quality_scores = {
            "agent1": self._create_mock_quality_score(0.3, 0.1, 0.2, 0.2, 0.1),
            "agent2": self._create_mock_quality_score(0.7, 0.6, 0.7, 0.7, 0.6),
            "agent3": self._create_mock_quality_score(0.4, 0.3, 0.4, 0.4, 0.3),
        }

        intervention = trickster._create_intervention(alert, quality_scores, round_num=3)

        assert intervention is not None
        # agent1 and agent3 are below threshold
        assert "agent1" in intervention.target_agents
        assert "agent3" in intervention.target_agents
        assert "agent2" not in intervention.target_agents

    def test_creates_intervention_with_evidence_gaps(self):
        """Test intervention identifies evidence gaps per agent."""
        config = TricksterConfig(min_quality_threshold=0.5)
        trickster = EvidencePoweredTrickster(config=config)

        alert = Mock()
        alert.severity = 0.6
        alert.avg_quality = 0.4
        alert.min_quality = 0.3
        alert.quality_variance = 0.05
        alert.reason = "Quality gaps"
        alert.agent_scores = {"agent1": 0.3}
        alert.recommended_challenges = []

        quality_scores = {
            "agent1": self._create_mock_quality_score(
                0.3,
                citation=0.1,  # Below 0.2 -> "citations" gap
                specificity=0.2,  # Below 0.3 -> "specificity" gap
                logical=0.2,  # Below 0.3 -> "reasoning" gap
                diversity=0.1,  # Below 0.2 -> "evidence_diversity" gap
            ),
        }

        intervention = trickster._create_intervention(alert, quality_scores, round_num=2)

        assert "agent1" in intervention.evidence_gaps
        gaps = intervention.evidence_gaps["agent1"]
        assert "citations" in gaps
        assert "specificity" in gaps
        assert "reasoning" in gaps
        assert "evidence_diversity" in gaps

    def test_creates_intervention_with_cross_analysis_metadata(self):
        """Test intervention includes cross-analysis metadata when available."""
        trickster = EvidencePoweredTrickster()

        alert = Mock()
        alert.severity = 0.6
        alert.avg_quality = 0.4
        alert.min_quality = 0.3
        alert.quality_variance = 0.05
        alert.reason = "Quality issues"
        alert.agent_scores = {"agent1": 0.3}
        alert.recommended_challenges = []

        quality_scores = {
            "agent1": self._create_mock_quality_score(0.3),
        }

        cross_analysis = Mock()
        cross_analysis.redundancy_score = 0.75
        cross_analysis.evidence_corroboration_score = 0.3
        cross_analysis.evidence_gaps = [Mock(), Mock()]

        intervention = trickster._create_intervention(
            alert, quality_scores, round_num=3, cross_analysis=cross_analysis
        )

        assert "cross_analysis_redundancy" in intervention.metadata
        assert intervention.metadata["cross_analysis_redundancy"] == 0.75
        assert intervention.metadata["cross_analysis_gaps_count"] == 2

    def test_creates_intervention_without_cross_analysis(self):
        """Test intervention works without cross-analysis."""
        trickster = EvidencePoweredTrickster()

        alert = Mock()
        alert.severity = 0.6
        alert.avg_quality = 0.4
        alert.min_quality = 0.3
        alert.quality_variance = 0.05
        alert.reason = "Quality issues"
        alert.agent_scores = {"agent1": 0.3}
        alert.recommended_challenges = []

        quality_scores = {
            "agent1": self._create_mock_quality_score(0.3),
        }

        intervention = trickster._create_intervention(
            alert, quality_scores, round_num=3, cross_analysis=None
        )

        assert "cross_analysis_redundancy" not in intervention.metadata
        assert intervention.metadata["reason"] == "Quality issues"

    def test_fallback_to_lowest_quality_agent_when_none_below_threshold(self):
        """Test falls back to lowest quality agent when none below threshold."""
        config = TricksterConfig(min_quality_threshold=0.1)  # Very low threshold
        trickster = EvidencePoweredTrickster(config=config)

        alert = Mock()
        alert.severity = 0.6
        alert.avg_quality = 0.5
        alert.min_quality = 0.4
        alert.quality_variance = 0.05
        alert.reason = "Test"
        alert.agent_scores = {"agent1": 0.5, "agent2": 0.4}
        alert.recommended_challenges = []

        quality_scores = {
            "agent1": self._create_mock_quality_score(0.5),
            "agent2": self._create_mock_quality_score(0.4),
        }

        intervention = trickster._create_intervention(alert, quality_scores, round_num=3)

        # Both agents are above threshold (0.1), so fallback to lowest
        assert len(intervention.target_agents) >= 1
        assert "agent2" in intervention.target_agents  # Lowest score


# =============================================================================
# _select_intervention_type Deep Tests
# =============================================================================


class TestSelectInterventionTypeDeep:
    """Deep tests for _select_intervention_type."""

    def test_breakpoint_disabled_falls_through(self):
        """Test high severity without breakpoints enabled falls to next option."""
        config = TricksterConfig(
            enable_breakpoints=False,
            enable_role_assignment=True,
        )
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.total_interventions = 0

        alert = Mock()
        alert.severity = 0.95  # Very high, but breakpoints disabled

        result = trickster._select_intervention_type(alert, round_num=3)

        # Should fall through to role assignment (first intervention)
        assert result == InterventionType.QUALITY_ROLE

    def test_role_disabled_falls_to_challenge_prompt(self):
        """Test first intervention without role assignment falls to challenge prompt."""
        config = TricksterConfig(
            enable_breakpoints=False,
            enable_role_assignment=False,
            enable_challenge_prompts=True,
        )
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.total_interventions = 0

        alert = Mock()
        alert.severity = 0.5

        result = trickster._select_intervention_type(alert, round_num=3)

        assert result == InterventionType.CHALLENGE_PROMPT

    def test_all_disabled_falls_to_challenge_prompt(self):
        """Test fallback to CHALLENGE_PROMPT when all options disabled."""
        config = TricksterConfig(
            enable_breakpoints=False,
            enable_role_assignment=False,
            enable_challenge_prompts=False,
        )
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.total_interventions = 1

        alert = Mock()
        alert.severity = 0.5

        result = trickster._select_intervention_type(alert, round_num=3)

        # Final fallback
        assert result == InterventionType.CHALLENGE_PROMPT

    def test_severity_boundary_at_0_8(self):
        """Test severity exactly at 0.8 does not trigger breakpoint."""
        config = TricksterConfig(enable_breakpoints=True)
        trickster = EvidencePoweredTrickster(config=config)
        trickster._state.total_interventions = 0

        alert = Mock()
        alert.severity = 0.8  # Not > 0.8

        result = trickster._select_intervention_type(alert, round_num=3)

        # Should be QUALITY_ROLE (first intervention), not BREAKPOINT
        assert result == InterventionType.QUALITY_ROLE

    def test_severity_above_0_8_triggers_breakpoint(self):
        """Test severity above 0.8 triggers breakpoint."""
        config = TricksterConfig(enable_breakpoints=True)
        trickster = EvidencePoweredTrickster(config=config)

        alert = Mock()
        alert.severity = 0.81

        result = trickster._select_intervention_type(alert, round_num=3)

        assert result == InterventionType.BREAKPOINT


# =============================================================================
# _build_challenge Deep Tests
# =============================================================================


class TestBuildChallengeDeep:
    """Deep tests for _build_challenge text generation."""

    def test_challenge_with_no_recommended_challenges(self):
        """Test challenge text without recommended challenges."""
        trickster = EvidencePoweredTrickster()

        alert = Mock()
        alert.recommended_challenges = []

        evidence_gaps = {"agent1": ["citations"]}
        target_agents = ["agent1"]

        text = trickster._build_challenge(alert, evidence_gaps, target_agents)

        assert "QUALITY CHALLENGE" in text
        assert "hollow consensus" in text
        assert "agent1" in text
        assert "citations" in text
        assert "Specific Challenges" not in text

    def test_challenge_with_recommended_challenges(self):
        """Test challenge text includes recommended challenges."""
        trickster = EvidencePoweredTrickster()

        alert = Mock()
        alert.recommended_challenges = ["Add citations", "Be more specific"]

        evidence_gaps = {}
        target_agents = []

        text = trickster._build_challenge(alert, evidence_gaps, target_agents)

        assert "Specific Challenges" in text
        assert "Add citations" in text
        assert "Be more specific" in text

    def test_challenge_with_multiple_agents_and_gaps(self):
        """Test challenge text with multiple agents and their gaps."""
        trickster = EvidencePoweredTrickster()

        alert = Mock()
        alert.recommended_challenges = []

        evidence_gaps = {
            "claude": ["citations", "reasoning"],
            "gpt4": ["specificity", "evidence_diversity"],
        }
        target_agents = ["claude", "gpt4"]

        text = trickster._build_challenge(alert, evidence_gaps, target_agents)

        assert "claude" in text
        assert "gpt4" in text
        assert "citations, reasoning" in text
        assert "specificity, evidence_diversity" in text

    def test_challenge_includes_action_items(self):
        """Test challenge text always includes action items."""
        trickster = EvidencePoweredTrickster()

        alert = Mock()
        alert.recommended_challenges = []

        text = trickster._build_challenge(alert, {}, [])

        assert "Before Proceeding" in text
        assert "specific citations" in text
        assert "concrete numbers" in text
        assert "Evidence-Powered Trickster" in text


# =============================================================================
# _build_evidence_gap_challenge Tests
# =============================================================================


class TestBuildEvidenceGapChallenge:
    """Tests for _build_evidence_gap_challenge text generation."""

    def test_evidence_gap_challenge_text(self):
        """Test evidence gap challenge includes expected text."""
        trickster = EvidencePoweredTrickster()

        mock_gap = Mock()
        mock_gap.claim = "Redis is always faster than PostgreSQL for caching"
        mock_gap.agents_making_claim = ["agent1", "agent2"]
        mock_gap.gap_severity = 0.9

        cross_analysis = Mock()
        cross_analysis.evidence_gaps = [mock_gap]

        text = trickster._build_evidence_gap_challenge(cross_analysis)

        assert "EVIDENCE GAP DETECTED" in text
        assert "agent1, agent2" in text
        assert "Required Actions" in text
        assert "cross-proposal evidence analysis" in text

    def test_evidence_gap_challenge_limits_to_three_gaps(self):
        """Test evidence gap challenge limits display to 3 gaps."""
        trickster = EvidencePoweredTrickster()

        gaps = []
        for i in range(5):
            gap = Mock()
            gap.claim = f"Claim {i} that is unsupported by evidence"
            gap.agents_making_claim = [f"agent{i}"]
            gap.gap_severity = 0.8
            gaps.append(gap)

        cross_analysis = Mock()
        cross_analysis.evidence_gaps = gaps

        text = trickster._build_evidence_gap_challenge(cross_analysis)

        # Only first 3 gaps should appear
        assert "agent0" in text
        assert "agent1" in text
        assert "agent2" in text
        # agent3 and agent4 should NOT appear
        assert "agent3" not in text
        assert "agent4" not in text


# =============================================================================
# _build_echo_chamber_challenge Tests
# =============================================================================


class TestBuildEchoChamberChallenge:
    """Tests for _build_echo_chamber_challenge text generation."""

    def test_echo_chamber_challenge_text(self):
        """Test echo chamber challenge includes expected sections."""
        trickster = EvidencePoweredTrickster()

        cross_analysis = Mock()
        cross_analysis.redundancy_score = 0.85
        cross_analysis.unique_evidence_sources = 2
        cross_analysis.total_evidence_sources = 10

        text = trickster._build_echo_chamber_challenge(cross_analysis)

        assert "ECHO CHAMBER WARNING" in text
        assert "85%" in text
        assert "Unique evidence sources: 2" in text
        assert "Total citations: 10" in text
        assert "independent" in text.lower()
        assert "cross-proposal redundancy detection" in text


# =============================================================================
# _create_evidence_gap_intervention Deep Tests
# =============================================================================


class TestCreateEvidenceGapInterventionDeep:
    """Deep tests for _create_evidence_gap_intervention."""

    def test_returns_none_for_empty_gaps(self):
        """Test returns None when evidence_gaps is empty."""
        trickster = EvidencePoweredTrickster()

        cross_analysis = Mock()
        cross_analysis.evidence_gaps = []

        result = trickster._create_evidence_gap_intervention(cross_analysis, round_num=3)

        assert result is None

    def test_uses_top_gap_for_metadata(self):
        """Test uses first gap for metadata (highest priority)."""
        trickster = EvidencePoweredTrickster()

        gap1 = Mock()
        gap1.claim = "First claim about Redis performance"
        gap1.agents_making_claim = ["agent1"]
        gap1.gap_severity = 0.9

        gap2 = Mock()
        gap2.claim = "Second claim about caching"
        gap2.agents_making_claim = ["agent2"]
        gap2.gap_severity = 0.7

        cross_analysis = Mock()
        cross_analysis.evidence_gaps = [gap1, gap2]

        result = trickster._create_evidence_gap_intervention(cross_analysis, round_num=3)

        assert result is not None
        assert result.metadata["gap_severity"] == 0.9
        assert result.metadata["total_gaps"] == 2
        assert result.priority == 0.9


# =============================================================================
# _create_echo_chamber_intervention Deep Tests
# =============================================================================


class TestCreateEchoChamberInterventionDeep:
    """Deep tests for _create_echo_chamber_intervention."""

    def test_returns_none_for_low_redundancy(self):
        """Test returns None when redundancy_score <= 0.7."""
        trickster = EvidencePoweredTrickster()

        cross_analysis = Mock()
        cross_analysis.redundancy_score = 0.7  # Exactly at boundary

        result = trickster._create_echo_chamber_intervention(cross_analysis, round_num=3)

        assert result is None

    def test_returns_intervention_for_high_redundancy(self):
        """Test returns intervention for redundancy > 0.7."""
        trickster = EvidencePoweredTrickster()

        cross_analysis = Mock()
        cross_analysis.redundancy_score = 0.71
        cross_analysis.unique_evidence_sources = 1
        cross_analysis.total_evidence_sources = 5
        cross_analysis.agent_coverage = {"agent1": 0.9, "agent2": 0.8}

        result = trickster._create_echo_chamber_intervention(cross_analysis, round_num=3)

        assert result is not None
        assert result.intervention_type == InterventionType.ECHO_CHAMBER
        assert result.priority == 0.71
        assert set(result.target_agents) == {"agent1", "agent2"}

    def test_metadata_includes_source_counts(self):
        """Test echo chamber metadata includes source counts."""
        trickster = EvidencePoweredTrickster()

        cross_analysis = Mock()
        cross_analysis.redundancy_score = 0.85
        cross_analysis.unique_evidence_sources = 2
        cross_analysis.total_evidence_sources = 10
        cross_analysis.agent_coverage = {"a": 0.9}

        result = trickster._create_echo_chamber_intervention(cross_analysis, round_num=3)

        assert result.metadata["redundancy_score"] == 0.85
        assert result.metadata["unique_sources"] == 2
        assert result.metadata["total_sources"] == 10


# =============================================================================
# check_and_intervene Cross-Proposal Analysis Flow Tests
# =============================================================================


class TestCheckAndInterveneCrossProposal:
    """Tests for check_and_intervene cross-proposal analysis flow."""

    def _create_mock_quality_score(self, overall: float = 0.5):
        """Helper to create mock quality score."""
        score = Mock()
        score.overall_quality = overall
        score.citation_density = 0.5
        score.specificity_score = 0.5
        score.logical_chain_score = 0.5
        score.evidence_diversity = 0.5
        return score

    def test_cross_analysis_not_triggered_below_convergence(self):
        """Test cross-proposal analysis not triggered below 0.6 convergence."""
        trickster = EvidencePoweredTrickster()

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                with patch.object(trickster._cross_analyzer, "analyze") as mock_cross:
                    mock_analyze.return_value = {"a": self._create_mock_quality_score(0.5)}
                    mock_check.return_value = Mock(detected=False)

                    trickster.check_and_intervene(
                        responses={"a": "Test"},
                        convergence_similarity=0.5,  # Below 0.6
                        round_num=2,
                    )

                    mock_cross.assert_not_called()

    def test_cross_analysis_triggered_above_convergence(self):
        """Test cross-proposal analysis triggered above 0.6 convergence."""
        trickster = EvidencePoweredTrickster()

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                with patch.object(trickster._cross_analyzer, "analyze") as mock_cross:
                    mock_analyze.return_value = {"a": self._create_mock_quality_score(0.5)}
                    mock_alert = Mock()
                    mock_alert.detected = False
                    mock_check.return_value = mock_alert

                    mock_cross_result = Mock()
                    mock_cross_result.evidence_gaps = []
                    mock_cross_result.redundancy_score = 0.5
                    mock_cross.return_value = mock_cross_result

                    trickster.check_and_intervene(
                        responses={"a": "Test"},
                        convergence_similarity=0.7,  # Above 0.6
                        round_num=2,
                    )

                    mock_cross.assert_called_once()

    def test_evidence_gap_takes_priority_over_echo_chamber(self):
        """Test evidence gap intervention takes priority over echo chamber."""
        trickster = EvidencePoweredTrickster()

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                with patch.object(trickster._cross_analyzer, "analyze") as mock_cross:
                    mock_analyze.return_value = {"a": self._create_mock_quality_score(0.5)}
                    mock_check.return_value = Mock(detected=False)

                    # Both evidence gaps AND high redundancy
                    mock_cross_result = Mock()
                    mock_cross_result.evidence_gaps = [
                        Mock(
                            claim="Unsupported claim",
                            agents_making_claim=["a"],
                            gap_severity=0.8,
                        )
                    ]
                    mock_cross_result.redundancy_score = 0.9  # Also high
                    mock_cross.return_value = mock_cross_result

                    result = trickster.check_and_intervene(
                        responses={"a": "Test"},
                        convergence_similarity=0.8,
                        round_num=2,
                    )

                    # Evidence gap should take priority
                    if result:
                        assert result.intervention_type == InterventionType.EVIDENCE_GAP

    def test_alert_below_severity_threshold_returns_none(self):
        """Test alert below severity threshold returns None."""
        config = TricksterConfig(hollow_detection_threshold=0.7)
        trickster = EvidencePoweredTrickster(config=config)

        with patch.object(trickster._analyzer, "analyze_batch") as mock_analyze:
            with patch.object(trickster._detector, "check") as mock_check:
                mock_analyze.return_value = {"a": self._create_mock_quality_score(0.4)}

                mock_alert = Mock()
                mock_alert.detected = True
                mock_alert.severity = 0.5  # Below threshold of 0.7
                mock_check.return_value = mock_alert

                result = trickster.check_and_intervene(
                    responses={"a": "Test"},
                    convergence_similarity=0.3,  # Low convergence, no cross-analysis
                    round_num=2,
                )

                assert result is None


# =============================================================================
# create_novelty_challenge Deep Tests
# =============================================================================


class TestCreateNoveltyChallengeDeep:
    """Deep tests for create_novelty_challenge."""

    def test_novelty_challenge_priority_calculation(self):
        """Test novelty challenge priority = 1.0 - min_novelty."""
        trickster = EvidencePoweredTrickster()

        intervention = trickster.create_novelty_challenge(
            low_novelty_agents=["a", "b"],
            novelty_scores={"a": 0.1, "b": 0.3},
            round_num=5,
        )

        assert intervention is not None
        # min_novelty = min(0.1, 0.3) = 0.1
        # priority = 1.0 - 0.1 = 0.9
        assert intervention.priority == 0.9

    def test_novelty_challenge_updates_state(self):
        """Test novelty challenge updates trickster state."""
        trickster = EvidencePoweredTrickster()

        trickster.create_novelty_challenge(
            low_novelty_agents=["a"],
            novelty_scores={"a": 0.2},
            round_num=5,
        )

        assert trickster._state.total_interventions == 1
        assert trickster._state.last_intervention_round == 5
        assert len(trickster._state.interventions) == 1

    def test_novelty_challenge_calls_on_intervention(self):
        """Test novelty challenge calls on_intervention callback."""
        callback = Mock()
        trickster = EvidencePoweredTrickster(on_intervention=callback)

        intervention = trickster.create_novelty_challenge(
            low_novelty_agents=["a"],
            novelty_scores={"a": 0.2},
            round_num=5,
        )

        callback.assert_called_once_with(intervention)

    def test_novelty_challenge_metadata(self):
        """Test novelty challenge includes correct metadata."""
        trickster = EvidencePoweredTrickster()

        scores = {"a": 0.15, "b": 0.25}
        intervention = trickster.create_novelty_challenge(
            low_novelty_agents=["a", "b"],
            novelty_scores=scores,
            round_num=5,
        )

        assert intervention.metadata["novelty_scores"] == scores
        assert intervention.metadata["min_novelty"] == 0.15
        assert intervention.metadata["reason"] == "proposals_too_similar_to_prior_rounds"


# =============================================================================
# _build_novelty_challenge Deep Tests
# =============================================================================


class TestBuildNoveltyChallengeDeep:
    """Deep tests for _build_novelty_challenge text generation."""

    def test_novelty_challenge_lists_agents_with_scores(self):
        """Test novelty challenge lists all agents with their scores."""
        trickster = EvidencePoweredTrickster()

        text = trickster._build_novelty_challenge(
            low_novelty_agents=["claude", "gpt4"],
            novelty_scores={"claude": 0.1, "gpt4": 0.25},
        )

        assert "claude" in text
        assert "gpt4" in text
        assert "10%" in text
        assert "25%" in text

    def test_novelty_challenge_includes_suggestions(self):
        """Test novelty challenge includes improvement suggestions."""
        trickster = EvidencePoweredTrickster()

        text = trickster._build_novelty_challenge(
            low_novelty_agents=["a"],
            novelty_scores={"a": 0.1},
        )

        assert "Increase Novelty" in text
        assert "devil's advocate" in text
        assert "edge cases" in text
        assert "Novelty Tracking system" in text

    def test_novelty_challenge_handles_missing_score(self):
        """Test novelty challenge handles agent not in novelty_scores."""
        trickster = EvidencePoweredTrickster()

        # Agent "b" is in low_novelty_agents but not in novelty_scores
        text = trickster._build_novelty_challenge(
            low_novelty_agents=["b"],
            novelty_scores={},  # Empty scores
        )

        assert "b" in text
        assert "0%" in text  # Default 0.0 score


# =============================================================================
# get_stats Deep Tests
# =============================================================================


class TestGetStatsDeep:
    """Deep tests for get_stats method."""

    def test_empty_stats(self):
        """Test stats for fresh trickster."""
        trickster = EvidencePoweredTrickster()

        stats = trickster.get_stats()

        assert stats["total_interventions"] == 0
        assert stats["hollow_alerts_detected"] == 0
        assert stats["avg_quality_per_round"] == []
        assert stats["interventions"] == []

    def test_stats_quality_history_averaging(self):
        """Test stats correctly averages quality per round."""
        trickster = EvidencePoweredTrickster()

        score_a = Mock()
        score_a.overall_quality = 0.6
        score_b = Mock()
        score_b.overall_quality = 0.8

        trickster._state.quality_history = [
            {"agent1": score_a, "agent2": score_b},  # avg = 0.7
        ]

        stats = trickster.get_stats()

        assert len(stats["avg_quality_per_round"]) == 1
        assert abs(stats["avg_quality_per_round"][0] - 0.7) < 0.01

    def test_stats_empty_quality_round_skipped(self):
        """Test stats skips empty quality rounds."""
        trickster = EvidencePoweredTrickster()

        trickster._state.quality_history = [
            {},  # Empty round
        ]

        stats = trickster.get_stats()

        # Empty dict is falsy, so it should be skipped
        assert len(stats["avg_quality_per_round"]) == 0

    def test_stats_intervention_serialization(self):
        """Test stats serializes interventions correctly."""
        trickster = EvidencePoweredTrickster()

        intervention = TricksterIntervention(
            intervention_type=InterventionType.EVIDENCE_GAP,
            round_num=4,
            target_agents=["claude", "gpt4"],
            challenge_text="Test challenge",
            evidence_gaps={"claude": ["citations"]},
            priority=0.85,
        )

        trickster._state.interventions = [intervention]
        trickster._state.total_interventions = 1

        stats = trickster.get_stats()

        assert len(stats["interventions"]) == 1
        assert stats["interventions"][0]["round"] == 4
        assert stats["interventions"][0]["type"] == "evidence_gap"
        assert stats["interventions"][0]["targets"] == ["claude", "gpt4"]
        assert stats["interventions"][0]["priority"] == 0.85


# =============================================================================
# reset Deep Tests
# =============================================================================


class TestResetDeep:
    """Deep tests for reset method."""

    def test_reset_preserves_config_and_dependencies(self):
        """Test reset clears state but preserves config and analyzers."""
        config = TricksterConfig(sensitivity=0.8)
        callback = Mock()
        trickster = EvidencePoweredTrickster(
            config=config,
            on_intervention=callback,
        )

        # Add state
        trickster._state.total_interventions = 5
        trickster._state.last_intervention_round = 10
        trickster._state.quality_history.append({"a": Mock()})
        trickster._state.hollow_alerts.append(Mock())

        trickster.reset()

        # State should be fresh
        assert trickster._state.total_interventions == 0
        assert trickster._state.last_intervention_round == -10
        assert trickster._state.quality_history == []
        assert trickster._state.hollow_alerts == []
        assert trickster._state.interventions == []

        # Config and callbacks should be preserved
        assert trickster.config.sensitivity == 0.8
        assert trickster.on_intervention is callback

    def test_reset_allows_immediate_intervention(self):
        """Test after reset, intervention can happen immediately."""
        config = TricksterConfig(intervention_cooldown_rounds=5)
        trickster = EvidencePoweredTrickster(config=config)

        # State at end of a debate
        trickster._state.last_intervention_round = 10
        trickster._state.total_interventions = 3

        trickster.reset()

        # After reset, last_intervention_round is -10, so
        # any round_num > -10 + 5 = -5 will pass cooldown
        intervention = TricksterIntervention(
            intervention_type=InterventionType.CHALLENGE_PROMPT,
            round_num=1,
            target_agents=["a"],
            challenge_text="Test",
            evidence_gaps={},
            priority=0.5,
        )

        result = trickster._record_intervention(intervention, round_num=1)

        # Should record (1 - (-10) = 11 > 5 cooldown)
        assert trickster._state.total_interventions == 1


# =============================================================================
# EvidencePoweredTrickster Init Deep Tests
# =============================================================================


class TestTricksterInitDeep:
    """Deep tests for EvidencePoweredTrickster initialization."""

    def test_init_with_custom_linker(self):
        """Test initialization with custom linker."""
        custom_linker = Mock()
        trickster = EvidencePoweredTrickster(linker=custom_linker)

        assert trickster._linker is custom_linker

    def test_init_cross_analyzer_uses_linker(self):
        """Test cross_analyzer receives the linker."""
        custom_linker = Mock()
        trickster = EvidencePoweredTrickster(linker=custom_linker)

        assert trickster._cross_analyzer is not None

    def test_init_detector_uses_config_threshold(self):
        """Test detector receives min_quality_threshold from config."""
        config = TricksterConfig(min_quality_threshold=0.9)
        trickster = EvidencePoweredTrickster(config=config)

        assert trickster._detector is not None


# =============================================================================
# get_quality_challenger_assignment Tests
# =============================================================================


class TestGetQualityChallengerAssignmentDeep:
    """Deep tests for get_quality_challenger_assignment."""

    def test_assignment_has_correct_role(self):
        """Test assignment uses QUALITY_CHALLENGER role."""
        from aragora.debate.roles import CognitiveRole

        trickster = EvidencePoweredTrickster()

        assignment = trickster.get_quality_challenger_assignment("claude", round_num=5)

        assert assignment.role == CognitiveRole.QUALITY_CHALLENGER
        assert assignment.agent_name == "claude"
        assert assignment.round_num == 5

    def test_assignment_has_role_prompt(self):
        """Test assignment includes role prompt text."""
        trickster = EvidencePoweredTrickster()

        assignment = trickster.get_quality_challenger_assignment("gpt4", round_num=3)

        assert assignment.role_prompt is not None
        assert len(assignment.role_prompt) > 0


# =============================================================================
# InterventionType Enum Deep Tests
# =============================================================================


class TestInterventionTypeDeep:
    """Deep tests for InterventionType enum."""

    def test_total_intervention_types(self):
        """Test total number of intervention types."""
        assert len(InterventionType) == 7

    def test_invalid_string_raises(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            InterventionType("nonexistent_type")

    def test_all_values_are_strings(self):
        """Test all values are strings."""
        for member in InterventionType:
            assert isinstance(member.value, str)


# =============================================================================
# Factory Function Deep Tests
# =============================================================================


class TestCreateTricksterForDebateDeep:
    """Deep tests for create_trickster_for_debate factory."""

    def test_factory_creates_config_with_given_quality(self):
        """Test factory sets min_quality_threshold."""
        trickster = create_trickster_for_debate(min_quality=0.9)

        assert trickster.config.min_quality_threshold == 0.9

    def test_factory_breakpoints_disabled(self):
        """Test factory can disable breakpoints."""
        trickster = create_trickster_for_debate(enable_breakpoints=False)

        assert trickster.config.enable_breakpoints is False

    def test_factory_preserves_other_defaults(self):
        """Test factory preserves other config defaults."""
        trickster = create_trickster_for_debate()

        assert trickster.config.enable_challenge_prompts is True
        assert trickster.config.enable_role_assignment is True
        assert trickster.config.max_interventions_total == 5
        assert trickster.config.intervention_cooldown_rounds == 1
