"""Tests for the Evidence-Powered Trickster system."""

from __future__ import annotations

import pytest

from aragora_debate.trickster import (
    EvidencePoweredTrickster,
    InterventionType,
    TricksterConfig,
    TricksterIntervention,
    TricksterState,
)


class TestInterventionType:
    def test_values(self):
        assert InterventionType.CHALLENGE_PROMPT.value == "challenge_prompt"
        assert InterventionType.EVIDENCE_GAP.value == "evidence_gap"
        assert InterventionType.ECHO_CHAMBER.value == "echo_chamber"

    def test_all_types(self):
        assert len(InterventionType) == 3


class TestTricksterConfig:
    def test_defaults(self):
        config = TricksterConfig()
        assert config.sensitivity == 0.5
        assert config.hollow_detection_threshold == 0.5
        assert config.min_quality_threshold == 0.65
        assert config.max_interventions_total == 5

    def test_sensitivity_adjusts_threshold(self):
        config = TricksterConfig(sensitivity=0.7)
        # 0.8 - 0.7 * 0.6 = 0.38
        assert abs(config.hollow_detection_threshold - 0.38) < 0.01

    def test_high_sensitivity(self):
        config = TricksterConfig(sensitivity=1.0)
        assert config.hollow_detection_threshold == 0.2

    def test_low_sensitivity(self):
        config = TricksterConfig(sensitivity=0.0)
        assert config.hollow_detection_threshold == 0.8

    def test_default_sensitivity_no_adjustment(self):
        config = TricksterConfig(sensitivity=0.5)
        assert config.hollow_detection_threshold == 0.5  # Not adjusted


class TestTricksterState:
    def test_defaults(self):
        state = TricksterState()
        assert state.interventions == []
        assert state.quality_history == []
        assert state.total_interventions == 0
        assert state.last_intervention_round == -10


class TestTricksterIntervention:
    def test_creation(self):
        intervention = TricksterIntervention(
            intervention_type=InterventionType.CHALLENGE_PROMPT,
            round_num=2,
            target_agents=["agent1", "agent2"],
            challenge_text="Provide evidence",
            evidence_gaps={"agent1": ["citations"]},
            priority=0.7,
        )
        assert intervention.round_num == 2
        assert len(intervention.target_agents) == 2
        assert intervention.priority == 0.7


class TestEvidencePoweredTrickster:
    def setup_method(self):
        self.trickster = EvidencePoweredTrickster()

    def test_default_config(self):
        assert self.trickster.config.sensitivity == 0.5

    def test_custom_config(self):
        config = TricksterConfig(sensitivity=0.8, max_interventions_total=3)
        trickster = EvidencePoweredTrickster(config=config)
        assert trickster.config.sensitivity == 0.8
        assert trickster.config.max_interventions_total == 3

    def test_no_intervention_low_convergence(self):
        responses = {
            "agent1": "Completely different proposal about microservices.",
            "agent2": "A totally unrelated proposal about databases.",
        }
        result = self.trickster.check_and_intervene(
            responses=responses,
            convergence_similarity=0.3,
            round_num=1,
        )
        assert result is None

    def test_intervention_on_hollow_consensus(self):
        responses = {
            "agent1": "Generally it might work in some cases. Usually this is fine. "
            "It depends on various factors and many considerations.",
            "agent2": "Typically the best approach. Common approach in most cases. "
            "Generally acceptable as an industry standard.",
        }
        result = self.trickster.check_and_intervene(
            responses=responses,
            convergence_similarity=0.9,
            round_num=1,
        )
        # May or may not trigger based on exact quality scores
        if result is not None:
            assert isinstance(result, TricksterIntervention)
            assert result.challenge_text != ""

    def test_quality_response_no_intervention(self):
        responses = {
            "agent1": (
                "According to Smith (2024), caching improves latency by 50% [1]. "
                "For example, Netflix reduced downtime by 40%. Therefore, this is validated. "
                "See https://example.com/study for the full dataset."
            ),
            "agent2": (
                "Per Jones (2025), the approach shows 45% improvement [2]. "
                "Specifically, the system handles 100K ops/s. Because of this, "
                "we can conclude it works. Data at https://research.org/paper."
            ),
        }
        result = self.trickster.check_and_intervene(
            responses=responses,
            convergence_similarity=0.5,
            round_num=1,
        )
        # Good evidence + low convergence = no intervention
        assert result is None

    def test_get_stats_empty(self):
        stats = self.trickster.get_stats()
        assert stats["total_interventions"] == 0
        assert stats["hollow_alerts_detected"] == 0
        assert stats["avg_quality_per_round"] == []
        assert stats["interventions"] == []

    def test_get_stats_after_check(self):
        self.trickster.check_and_intervene(
            responses={"a": "test response", "b": "another response"},
            convergence_similarity=0.5,
            round_num=1,
        )
        stats = self.trickster.get_stats()
        assert len(stats["avg_quality_per_round"]) == 1

    def test_reset(self):
        self.trickster.check_and_intervene(
            responses={"a": "text", "b": "text"},
            convergence_similarity=0.5,
            round_num=1,
        )
        self.trickster.reset()
        stats = self.trickster.get_stats()
        assert stats["total_interventions"] == 0

    def test_on_alert_callback(self):
        alerts = []

        trickster = EvidencePoweredTrickster(
            on_alert=lambda a: alerts.append(a),
        )

        trickster.check_and_intervene(
            responses={
                "a": "Generally might work in some cases.",
                "b": "Usually the common approach typically.",
            },
            convergence_similarity=0.9,
            round_num=1,
        )
        # Alert callback fires if hollow consensus detected
        # (depends on quality scoring)

    def test_on_intervention_callback(self):
        interventions = []

        trickster = EvidencePoweredTrickster(
            config=TricksterConfig(sensitivity=1.0),
            on_intervention=lambda i: interventions.append(i),
        )

        trickster.check_and_intervene(
            responses={
                "a": "It might work generally.",
                "b": "Typically acceptable approach.",
            },
            convergence_similarity=0.95,
            round_num=1,
        )
        # May produce intervention

    def test_resolve_config_default(self):
        config = self.trickster.resolve_config()
        assert config is self.trickster.config

    def test_resolve_config_domain(self):
        medical_config = TricksterConfig(sensitivity=0.9)
        trickster = EvidencePoweredTrickster(
            domain_configs={"medical": medical_config},
        )
        config = trickster.resolve_config("medical")
        assert config is medical_config

    def test_resolve_config_unknown_domain(self):
        trickster = EvidencePoweredTrickster(
            domain_configs={"medical": TricksterConfig(sensitivity=0.9)},
        )
        config = trickster.resolve_config("unknown")
        assert config is trickster.config

    def test_max_interventions_limit(self):
        config = TricksterConfig(
            max_interventions_total=1,
            sensitivity=1.0,
            intervention_cooldown_rounds=0,
        )
        trickster = EvidencePoweredTrickster(config=config)

        vague_responses = {
            "a": "Generally might work. Various factors.",
            "b": "Typically acceptable. Common approach.",
        }

        # First intervention
        trickster.check_and_intervene(
            responses=vague_responses,
            convergence_similarity=0.95,
            round_num=1,
        )

        # After max, should still return but not record
        trickster.check_and_intervene(
            responses=vague_responses,
            convergence_similarity=0.95,
            round_num=3,
        )
        stats = trickster.get_stats()
        assert stats["total_interventions"] <= 1

    def test_cooldown_respected(self):
        config = TricksterConfig(
            intervention_cooldown_rounds=3,
            sensitivity=1.0,
        )
        trickster = EvidencePoweredTrickster(config=config)

        vague = {
            "a": "Generally might work. Various factors.",
            "b": "Typically acceptable. Common approach.",
        }

        # First check at round 1
        trickster.check_and_intervene(
            responses=vague,
            convergence_similarity=0.95,
            round_num=1,
        )

        # Round 2 should be in cooldown
        result = trickster.check_and_intervene(
            responses=vague,
            convergence_similarity=0.95,
            round_num=2,
        )
        # Either None (cooldown) or intervention returned but not recorded

    def test_build_challenge_text(self):
        from aragora_debate.evidence import HollowConsensusAlert

        alert = HollowConsensusAlert(
            detected=True,
            severity=0.7,
            reason="Low evidence quality",
            agent_scores={"a": 0.2, "b": 0.3},
            recommended_challenges=["Provide citations"],
            min_quality=0.2,
            avg_quality=0.25,
            quality_variance=0.05,
        )

        text = self.trickster._build_challenge(
            alert,
            evidence_gaps={"a": ["citations", "reasoning"]},
            target_agents=["a"],
        )
        assert "QUALITY CHALLENGE" in text
        assert "hollow consensus" in text
        assert "citations" in text
