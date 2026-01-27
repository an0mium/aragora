"""
Tests for explainability Decision dataclasses.

Tests EvidenceLink, VotePivot, BeliefChange, ConfidenceAttribution,
Counterfactual, and Decision dataclasses.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aragora.explainability.decision import (
    BeliefChange,
    ConfidenceAttribution,
    Counterfactual,
    Decision,
    EvidenceLink,
    InfluenceType,
    VotePivot,
)


class TestInfluenceType:
    """Tests for InfluenceType enum."""

    def test_influence_types_exist(self):
        """Test all expected influence types exist."""
        assert InfluenceType.EVIDENCE == "evidence"
        assert InfluenceType.VOTE == "vote"
        assert InfluenceType.ARGUMENT == "argument"
        assert InfluenceType.CALIBRATION == "calibration"
        assert InfluenceType.ELO == "elo"
        assert InfluenceType.CONSENSUS == "consensus"
        assert InfluenceType.USER == "user"

    def test_influence_type_is_string_enum(self):
        """Test InfluenceType values are strings."""
        for member in InfluenceType:
            assert isinstance(member.value, str)


class TestEvidenceLink:
    """Tests for EvidenceLink dataclass."""

    def test_evidence_link_creation(self):
        """Test basic EvidenceLink creation."""
        link = EvidenceLink(
            id="ev-123",
            content="Test evidence content",
            source="claude",
            relevance_score=0.85,
        )
        assert link.id == "ev-123"
        assert link.content == "Test evidence content"
        assert link.source == "claude"
        assert link.relevance_score == 0.85
        assert link.quality_scores == {}
        assert link.cited_by == []
        assert link.grounding_type == "claim"
        assert link.timestamp is None
        assert link.metadata == {}

    def test_evidence_link_with_all_fields(self):
        """Test EvidenceLink with all fields populated."""
        link = EvidenceLink(
            id="ev-456",
            content="Detailed evidence",
            source="gemini",
            relevance_score=0.92,
            quality_scores={"authority": 0.9, "freshness": 0.8},
            cited_by=["claude", "gpt"],
            grounding_type="fact",
            timestamp="2026-01-27T10:00:00Z",
            metadata={"round": 2, "type": "proposal"},
        )
        assert link.quality_scores == {"authority": 0.9, "freshness": 0.8}
        assert link.cited_by == ["claude", "gpt"]
        assert link.grounding_type == "fact"
        assert link.timestamp == "2026-01-27T10:00:00Z"
        assert link.metadata["round"] == 2

    def test_evidence_link_to_dict(self):
        """Test EvidenceLink serialization."""
        link = EvidenceLink(
            id="ev-789",
            content="Evidence for serialization",
            source="gpt",
            relevance_score=0.75,
            quality_scores={"completeness": 0.7},
            cited_by=["mistral"],
            grounding_type="citation",
            timestamp="2026-01-27T11:00:00Z",
            metadata={"key": "value"},
        )
        result = link.to_dict()

        assert result["id"] == "ev-789"
        assert result["content"] == "Evidence for serialization"
        assert result["source"] == "gpt"
        assert result["relevance_score"] == 0.75
        assert result["quality_scores"] == {"completeness": 0.7}
        assert result["cited_by"] == ["mistral"]
        assert result["grounding_type"] == "citation"
        assert result["timestamp"] == "2026-01-27T11:00:00Z"
        assert result["metadata"] == {"key": "value"}


class TestVotePivot:
    """Tests for VotePivot dataclass."""

    def test_vote_pivot_creation(self):
        """Test basic VotePivot creation."""
        pivot = VotePivot(
            agent="claude",
            choice="option_a",
            confidence=0.85,
            weight=1.5,
            reasoning_summary="Strong evidence supports option A",
            influence_score=0.65,
        )
        assert pivot.agent == "claude"
        assert pivot.choice == "option_a"
        assert pivot.confidence == 0.85
        assert pivot.weight == 1.5
        assert pivot.reasoning_summary == "Strong evidence supports option A"
        assert pivot.influence_score == 0.65
        assert pivot.calibration_adjustment is None
        assert pivot.elo_rating is None
        assert pivot.flip_detected is False
        assert pivot.metadata == {}

    def test_vote_pivot_with_calibration_and_elo(self):
        """Test VotePivot with calibration and ELO data."""
        pivot = VotePivot(
            agent="gpt",
            choice="option_b",
            confidence=0.9,
            weight=1.8,
            reasoning_summary="Comprehensive analysis",
            influence_score=0.75,
            calibration_adjustment=0.05,
            elo_rating=1450.0,
            flip_detected=True,
            metadata={"prior_choice": "option_a"},
        )
        assert pivot.calibration_adjustment == 0.05
        assert pivot.elo_rating == 1450.0
        assert pivot.flip_detected is True
        assert pivot.metadata["prior_choice"] == "option_a"

    def test_vote_pivot_to_dict(self):
        """Test VotePivot serialization."""
        pivot = VotePivot(
            agent="gemini",
            choice="option_c",
            confidence=0.7,
            weight=1.2,
            reasoning_summary="Balanced view",
            influence_score=0.4,
            calibration_adjustment=-0.02,
            elo_rating=1320.0,
            flip_detected=False,
            metadata={"round": 3},
        )
        result = pivot.to_dict()

        assert result["agent"] == "gemini"
        assert result["choice"] == "option_c"
        assert result["confidence"] == 0.7
        assert result["weight"] == 1.2
        assert result["reasoning_summary"] == "Balanced view"
        assert result["influence_score"] == 0.4
        assert result["calibration_adjustment"] == -0.02
        assert result["elo_rating"] == 1320.0
        assert result["flip_detected"] is False
        assert result["metadata"] == {"round": 3}


class TestBeliefChange:
    """Tests for BeliefChange dataclass."""

    def test_belief_change_creation(self):
        """Test basic BeliefChange creation."""
        change = BeliefChange(
            agent="claude",
            round=2,
            topic="API design pattern",
            prior_belief="REST is best",
            posterior_belief="GraphQL has merits",
            prior_confidence=0.8,
            posterior_confidence=0.6,
            trigger="critique",
            trigger_source="gpt",
        )
        assert change.agent == "claude"
        assert change.round == 2
        assert change.topic == "API design pattern"
        assert change.prior_belief == "REST is best"
        assert change.posterior_belief == "GraphQL has merits"
        assert change.prior_confidence == 0.8
        assert change.posterior_confidence == 0.6
        assert change.trigger == "critique"
        assert change.trigger_source == "gpt"
        assert change.metadata == {}

    def test_belief_change_confidence_delta_positive(self):
        """Test confidence_delta property with increase."""
        change = BeliefChange(
            agent="gpt",
            round=1,
            topic="Test topic",
            prior_belief="Initial",
            posterior_belief="Updated",
            prior_confidence=0.5,
            posterior_confidence=0.8,
            trigger="evidence",
            trigger_source="user",
        )
        assert change.confidence_delta == pytest.approx(0.3)

    def test_belief_change_confidence_delta_negative(self):
        """Test confidence_delta property with decrease."""
        change = BeliefChange(
            agent="gemini",
            round=3,
            topic="Architecture choice",
            prior_belief="Microservices",
            posterior_belief="Monolith acceptable",
            prior_confidence=0.9,
            posterior_confidence=0.55,
            trigger="argument",
            trigger_source="claude",
        )
        assert change.confidence_delta == pytest.approx(-0.35)

    def test_belief_change_confidence_delta_zero(self):
        """Test confidence_delta when no change."""
        change = BeliefChange(
            agent="mistral",
            round=2,
            topic="Same confidence",
            prior_belief="Position A",
            posterior_belief="Position B",
            prior_confidence=0.7,
            posterior_confidence=0.7,
            trigger="debate",
            trigger_source="consensus",
        )
        assert change.confidence_delta == pytest.approx(0.0)

    def test_belief_change_to_dict(self):
        """Test BeliefChange serialization includes confidence_delta."""
        change = BeliefChange(
            agent="claude",
            round=1,
            topic="Test",
            prior_belief="A",
            posterior_belief="B",
            prior_confidence=0.4,
            posterior_confidence=0.9,
            trigger="evidence",
            trigger_source="external",
            metadata={"important": True},
        )
        result = change.to_dict()

        assert result["agent"] == "claude"
        assert result["round"] == 1
        assert result["topic"] == "Test"
        assert result["prior_belief"] == "A"
        assert result["posterior_belief"] == "B"
        assert result["prior_confidence"] == 0.4
        assert result["posterior_confidence"] == 0.9
        assert result["confidence_delta"] == pytest.approx(0.5)
        assert result["trigger"] == "evidence"
        assert result["trigger_source"] == "external"
        assert result["metadata"] == {"important": True}


class TestConfidenceAttribution:
    """Tests for ConfidenceAttribution dataclass."""

    def test_confidence_attribution_creation(self):
        """Test basic ConfidenceAttribution creation."""
        attr = ConfidenceAttribution(
            factor="consensus_strength",
            contribution=0.4,
            explanation="Agreement level among agents (80% margin)",
        )
        assert attr.factor == "consensus_strength"
        assert attr.contribution == 0.4
        assert attr.explanation == "Agreement level among agents (80% margin)"
        assert attr.raw_value is None
        assert attr.metadata == {}

    def test_confidence_attribution_with_raw_value(self):
        """Test ConfidenceAttribution with raw_value."""
        attr = ConfidenceAttribution(
            factor="evidence_quality",
            contribution=0.3,
            explanation="Quality of supporting evidence (75% average)",
            raw_value=0.75,
            metadata={"source_count": 5},
        )
        assert attr.raw_value == 0.75
        assert attr.metadata["source_count"] == 5

    def test_confidence_attribution_to_dict(self):
        """Test ConfidenceAttribution serialization."""
        attr = ConfidenceAttribution(
            factor="agent_calibration",
            contribution=0.2,
            explanation="Historical accuracy of agents",
            raw_value=0.85,
            metadata={"agents_analyzed": 3},
        )
        result = attr.to_dict()

        assert result["factor"] == "agent_calibration"
        assert result["contribution"] == 0.2
        assert result["explanation"] == "Historical accuracy of agents"
        assert result["raw_value"] == 0.85
        assert result["metadata"] == {"agents_analyzed": 3}


class TestCounterfactual:
    """Tests for Counterfactual dataclass."""

    def test_counterfactual_creation(self):
        """Test basic Counterfactual creation."""
        cf = Counterfactual(
            condition="If claude had voted differently",
            outcome_change="Possible change in consensus",
            likelihood=0.3,
            sensitivity=0.65,
        )
        assert cf.condition == "If claude had voted differently"
        assert cf.outcome_change == "Possible change in consensus"
        assert cf.likelihood == 0.3
        assert cf.sensitivity == 0.65
        assert cf.affected_agents == []
        assert cf.metadata == {}

    def test_counterfactual_with_affected_agents(self):
        """Test Counterfactual with affected agents."""
        cf = Counterfactual(
            condition="Without evidence from external sources",
            outcome_change="Lower confidence",
            likelihood=0.2,
            sensitivity=0.8,
            affected_agents=["claude", "gpt", "gemini"],
            metadata={"evidence_count": 3},
        )
        assert cf.affected_agents == ["claude", "gpt", "gemini"]
        assert cf.metadata["evidence_count"] == 3

    def test_counterfactual_to_dict(self):
        """Test Counterfactual serialization."""
        cf = Counterfactual(
            condition="With fewer participants",
            outcome_change="Potentially lower confidence",
            likelihood=0.5,
            sensitivity=0.3,
            affected_agents=["mistral"],
            metadata={"min_agents": 2},
        )
        result = cf.to_dict()

        assert result["condition"] == "With fewer participants"
        assert result["outcome_change"] == "Potentially lower confidence"
        assert result["likelihood"] == 0.5
        assert result["sensitivity"] == 0.3
        assert result["affected_agents"] == ["mistral"]
        assert result["metadata"] == {"min_agents": 2}


class TestDecision:
    """Tests for Decision dataclass."""

    def test_decision_creation_minimal(self):
        """Test Decision with minimal fields."""
        decision = Decision(
            decision_id="dec-test123",
            debate_id="debate-456",
        )
        assert decision.decision_id == "dec-test123"
        assert decision.debate_id == "debate-456"
        assert decision.conclusion == ""
        assert decision.consensus_reached is False
        assert decision.confidence == 0.0
        assert decision.consensus_type == "majority"
        assert decision.task == ""
        assert decision.domain == "general"
        assert decision.rounds_used == 0
        assert decision.agents_participated == []
        assert decision.evidence_chain == []
        assert decision.vote_pivots == []
        assert decision.belief_changes == []
        assert decision.confidence_attribution == []
        assert decision.counterfactuals == []

    def test_decision_creation_full(self):
        """Test Decision with all fields populated."""
        evidence = [
            EvidenceLink(
                id="ev-1",
                content="Evidence 1",
                source="claude",
                relevance_score=0.9,
            )
        ]
        pivots = [
            VotePivot(
                agent="claude",
                choice="yes",
                confidence=0.85,
                weight=1.5,
                reasoning_summary="Good reasoning",
                influence_score=0.7,
            )
        ]
        belief_changes = [
            BeliefChange(
                agent="gpt",
                round=2,
                topic="Test",
                prior_belief="A",
                posterior_belief="B",
                prior_confidence=0.5,
                posterior_confidence=0.8,
                trigger="critique",
                trigger_source="claude",
            )
        ]
        conf_attrs = [
            ConfidenceAttribution(
                factor="consensus",
                contribution=0.5,
                explanation="Strong agreement",
            )
        ]
        counterfactuals = [
            Counterfactual(
                condition="If X",
                outcome_change="Then Y",
                likelihood=0.3,
                sensitivity=0.6,
            )
        ]

        decision = Decision(
            decision_id="dec-full",
            debate_id="debate-full",
            timestamp="2026-01-27T12:00:00Z",
            conclusion="The answer is yes",
            consensus_reached=True,
            confidence=0.85,
            consensus_type="supermajority",
            task="Evaluate proposal",
            domain="engineering",
            rounds_used=3,
            agents_participated=["claude", "gpt", "gemini"],
            evidence_chain=evidence,
            vote_pivots=pivots,
            belief_changes=belief_changes,
            confidence_attribution=conf_attrs,
            counterfactuals=counterfactuals,
            evidence_quality_score=0.88,
            agent_agreement_score=0.75,
            belief_stability_score=0.9,
            metadata={"version": 1},
        )

        assert decision.conclusion == "The answer is yes"
        assert decision.consensus_reached is True
        assert decision.confidence == 0.85
        assert decision.rounds_used == 3
        assert len(decision.evidence_chain) == 1
        assert len(decision.vote_pivots) == 1
        assert decision.evidence_quality_score == 0.88

    def test_decision_auto_generates_id(self):
        """Test Decision generates ID when not provided."""
        decision = Decision(
            decision_id="",
            debate_id="debate-auto",
            conclusion="Test conclusion",
        )
        assert decision.decision_id.startswith("dec-")
        assert len(decision.decision_id) > 4

    def test_decision_to_dict(self):
        """Test Decision serialization."""
        decision = Decision(
            decision_id="dec-serial",
            debate_id="debate-serial",
            conclusion="Serialized conclusion",
            consensus_reached=True,
            confidence=0.9,
            agents_participated=["claude"],
        )
        result = decision.to_dict()

        assert result["decision_id"] == "dec-serial"
        assert result["debate_id"] == "debate-serial"
        assert result["conclusion"] == "Serialized conclusion"
        assert result["consensus_reached"] is True
        assert result["confidence"] == 0.9
        assert result["agents_participated"] == ["claude"]
        assert "evidence_chain" in result
        assert "vote_pivots" in result

    def test_decision_to_json(self):
        """Test Decision JSON export."""
        decision = Decision(
            decision_id="dec-json",
            debate_id="debate-json",
            conclusion="JSON conclusion",
        )
        json_str = decision.to_json()
        parsed = json.loads(json_str)

        assert parsed["decision_id"] == "dec-json"
        assert parsed["debate_id"] == "debate-json"
        assert parsed["conclusion"] == "JSON conclusion"

    def test_decision_from_dict(self):
        """Test Decision deserialization."""
        data = {
            "decision_id": "dec-from-dict",
            "debate_id": "debate-from-dict",
            "timestamp": "2026-01-27T13:00:00Z",
            "conclusion": "Deserialized conclusion",
            "consensus_reached": True,
            "confidence": 0.88,
            "consensus_type": "unanimous",
            "task": "Test task",
            "domain": "testing",
            "rounds_used": 2,
            "agents_participated": ["claude", "gpt"],
            "evidence_chain": [
                {
                    "id": "ev-1",
                    "content": "Evidence",
                    "source": "claude",
                    "relevance_score": 0.9,
                }
            ],
            "vote_pivots": [
                {
                    "agent": "claude",
                    "choice": "yes",
                    "confidence": 0.9,
                    "weight": 1.5,
                    "reasoning_summary": "Good",
                    "influence_score": 0.8,
                }
            ],
            "belief_changes": [],
            "confidence_attribution": [],
            "counterfactuals": [],
            "evidence_quality_score": 0.85,
            "agent_agreement_score": 0.9,
            "belief_stability_score": 1.0,
            "metadata": {},
        }

        decision = Decision.from_dict(data)

        assert decision.decision_id == "dec-from-dict"
        assert decision.consensus_reached is True
        assert decision.confidence == 0.88
        assert len(decision.evidence_chain) == 1
        assert decision.evidence_chain[0].source == "claude"
        assert len(decision.vote_pivots) == 1
        assert decision.vote_pivots[0].agent == "claude"

    def test_decision_from_dict_with_defaults(self):
        """Test Decision.from_dict handles missing fields."""
        data = {
            "debate_id": "debate-minimal",
        }
        decision = Decision.from_dict(data)

        # Decision auto-generates ID when not provided via __post_init__
        assert decision.decision_id.startswith("dec-")
        assert decision.debate_id == "debate-minimal"
        assert decision.consensus_reached is False
        assert decision.confidence == 0.0
        assert decision.domain == "general"


class TestDecisionQueryMethods:
    """Tests for Decision query methods."""

    @pytest.fixture
    def decision_with_data(self):
        """Create a decision with test data."""
        evidence = [
            EvidenceLink(
                id="ev-1", content="High relevance", source="claude", relevance_score=0.95
            ),
            EvidenceLink(id="ev-2", content="Medium relevance", source="gpt", relevance_score=0.7),
            EvidenceLink(id="ev-3", content="Low relevance", source="gemini", relevance_score=0.4),
            EvidenceLink(id="ev-4", content="Very high", source="mistral", relevance_score=0.98),
        ]
        pivots = [
            VotePivot(
                agent="claude",
                choice="yes",
                confidence=0.9,
                weight=1.5,
                reasoning_summary="R1",
                influence_score=0.8,
            ),
            VotePivot(
                agent="gpt",
                choice="yes",
                confidence=0.7,
                weight=1.2,
                reasoning_summary="R2",
                influence_score=0.5,
            ),
            VotePivot(
                agent="gemini",
                choice="no",
                confidence=0.6,
                weight=1.0,
                reasoning_summary="R3",
                influence_score=0.2,
            ),
        ]
        belief_changes = [
            BeliefChange(
                agent="claude",
                round=1,
                topic="T",
                prior_belief="A",
                posterior_belief="B",
                prior_confidence=0.5,
                posterior_confidence=0.8,
                trigger="t",
                trigger_source="s",
            ),
            BeliefChange(
                agent="gpt",
                round=2,
                topic="T",
                prior_belief="X",
                posterior_belief="Y",
                prior_confidence=0.7,
                posterior_confidence=0.75,
                trigger="t",
                trigger_source="s",
            ),
            BeliefChange(
                agent="gemini",
                round=1,
                topic="T",
                prior_belief="P",
                posterior_belief="Q",
                prior_confidence=0.9,
                posterior_confidence=0.4,
                trigger="t",
                trigger_source="s",
            ),
        ]
        conf_attrs = [
            ConfidenceAttribution(factor="consensus", contribution=0.5, explanation="High"),
            ConfidenceAttribution(factor="evidence", contribution=0.3, explanation="Medium"),
            ConfidenceAttribution(factor="calibration", contribution=0.08, explanation="Low"),
            ConfidenceAttribution(factor="other", contribution=0.12, explanation="Other"),
        ]
        counterfactuals = [
            Counterfactual(
                condition="If A", outcome_change="Then B", likelihood=0.3, sensitivity=0.9
            ),
            Counterfactual(
                condition="If C", outcome_change="Then D", likelihood=0.2, sensitivity=0.4
            ),
            Counterfactual(
                condition="If E", outcome_change="Then F", likelihood=0.1, sensitivity=0.6
            ),
        ]

        return Decision(
            decision_id="dec-query",
            debate_id="debate-query",
            evidence_chain=evidence,
            vote_pivots=pivots,
            belief_changes=belief_changes,
            confidence_attribution=conf_attrs,
            counterfactuals=counterfactuals,
        )

    def test_get_top_evidence(self, decision_with_data):
        """Test get_top_evidence returns highest relevance items."""
        top = decision_with_data.get_top_evidence(2)

        assert len(top) == 2
        assert top[0].relevance_score == 0.98
        assert top[1].relevance_score == 0.95

    def test_get_top_evidence_default(self, decision_with_data):
        """Test get_top_evidence default limit."""
        top = decision_with_data.get_top_evidence()
        assert len(top) == 4  # All evidence (< default limit of 5)

    def test_get_pivotal_votes(self, decision_with_data):
        """Test get_pivotal_votes with threshold."""
        pivotal = decision_with_data.get_pivotal_votes(0.3)

        assert len(pivotal) == 2
        assert all(v.influence_score >= 0.3 for v in pivotal)

    def test_get_pivotal_votes_high_threshold(self, decision_with_data):
        """Test get_pivotal_votes with high threshold."""
        pivotal = decision_with_data.get_pivotal_votes(0.7)

        assert len(pivotal) == 1
        assert pivotal[0].agent == "claude"

    def test_get_significant_belief_changes(self, decision_with_data):
        """Test get_significant_belief_changes with threshold."""
        significant = decision_with_data.get_significant_belief_changes(0.2)

        assert len(significant) == 2  # claude (+0.3) and gemini (-0.5)

    def test_get_significant_belief_changes_low_threshold(self, decision_with_data):
        """Test get_significant_belief_changes includes small deltas."""
        significant = decision_with_data.get_significant_belief_changes(0.05)

        assert len(significant) == 3  # All changes

    def test_get_major_confidence_factors(self, decision_with_data):
        """Test get_major_confidence_factors with threshold."""
        major = decision_with_data.get_major_confidence_factors(0.1)

        assert len(major) == 3  # consensus (0.5), evidence (0.3), other (0.12)
        assert all(f.contribution >= 0.1 for f in major)

    def test_get_high_sensitivity_counterfactuals(self, decision_with_data):
        """Test get_high_sensitivity_counterfactuals with threshold."""
        high_sens = decision_with_data.get_high_sensitivity_counterfactuals(0.5)

        assert len(high_sens) == 2  # sensitivity 0.9 and 0.6
        assert all(c.sensitivity >= 0.5 for c in high_sens)


class TestDecisionEdgeCases:
    """Edge case tests for Decision."""

    def test_decision_empty_evidence_chain(self):
        """Test query methods with empty evidence."""
        decision = Decision(decision_id="dec-empty", debate_id="debate-empty")

        assert decision.get_top_evidence() == []
        assert decision.get_pivotal_votes() == []
        assert decision.get_significant_belief_changes() == []
        assert decision.get_major_confidence_factors() == []
        assert decision.get_high_sensitivity_counterfactuals() == []

    def test_decision_roundtrip_serialization(self):
        """Test decision survives serialization roundtrip."""
        original = Decision(
            decision_id="dec-roundtrip",
            debate_id="debate-roundtrip",
            conclusion="Test conclusion",
            consensus_reached=True,
            confidence=0.87,
            evidence_chain=[
                EvidenceLink(
                    id="ev-1",
                    content="Evidence content",
                    source="claude",
                    relevance_score=0.9,
                    quality_scores={"authority": 0.8},
                )
            ],
            vote_pivots=[
                VotePivot(
                    agent="claude",
                    choice="yes",
                    confidence=0.9,
                    weight=1.5,
                    reasoning_summary="Strong reasoning",
                    influence_score=0.8,
                    elo_rating=1400.0,
                )
            ],
        )

        json_str = original.to_json()
        restored = Decision.from_dict(json.loads(json_str))

        assert restored.decision_id == original.decision_id
        assert restored.conclusion == original.conclusion
        assert restored.confidence == original.confidence
        assert len(restored.evidence_chain) == 1
        assert restored.evidence_chain[0].source == "claude"
        assert restored.evidence_chain[0].quality_scores == {"authority": 0.8}
        assert len(restored.vote_pivots) == 1
        assert restored.vote_pivots[0].elo_rating == 1400.0
