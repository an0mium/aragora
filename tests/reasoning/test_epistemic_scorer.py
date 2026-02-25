"""
Tests for Composite Epistemic Quality Scorer.

Tests each sub-scorer independently with mock data, composite scoring
with all components, graceful degradation, and configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.reasoning.epistemic_scorer import (
    EpistemicScore,
    EpistemicScorer,
    EpistemicScorerConfig,
    _infer_provider,
)


# =====================================================================
# Test helpers / fixtures
# =====================================================================


@dataclass
class MockVote:
    """Minimal vote mock compatible with ConsensusVote."""

    agent: str
    vote: str  # "agree", "disagree", "conditional", "abstain"
    confidence: float = 0.8
    reasoning: str = ""


@dataclass
class MockEvidence:
    """Minimal evidence mock."""

    evidence_id: str = "ev-1"
    evidence_type: str = "citation"
    source: str = "agent-1"
    content: str = "Some data."
    strength: float = 0.7
    supports_claim: bool = True


@dataclass
class MockClaim:
    """Minimal claim mock."""

    claim_id: str = "c-1"
    statement: str = "Test claim"
    author: str = "agent"
    confidence: float = 0.7
    supporting_evidence: list[Any] = field(default_factory=list)
    refuting_evidence: list[Any] = field(default_factory=list)
    parent_claim_id: str | None = None


@dataclass
class MockMessage:
    """Minimal message mock."""

    agent: str = "claude-agent"
    content: str = "This is a test response."
    role: str = "proposer"
    round: int = 1


@dataclass
class MockDebateResult:
    """Minimal debate result mock."""

    messages: list[Any] = field(default_factory=list)
    claims: list[Any] = field(default_factory=list)
    final_answer: str = ""
    consensus_reached: bool = True
    confidence: float = 0.8


@pytest.fixture
def scorer() -> EpistemicScorer:
    return EpistemicScorer()


@pytest.fixture
def diverse_votes() -> list[MockVote]:
    """Votes from agents across three different providers."""
    return [
        MockVote(agent="claude-opus", vote="agree", confidence=0.85),
        MockVote(agent="gpt-4o", vote="agree", confidence=0.8),
        MockVote(agent="gemini-pro", vote="conditional", confidence=0.75),
    ]


@pytest.fixture
def uniform_votes() -> list[MockVote]:
    """Votes from agents under the same provider."""
    return [
        MockVote(agent="claude-opus", vote="agree", confidence=0.9),
        MockVote(agent="claude-sonnet", vote="agree", confidence=0.92),
        MockVote(agent="claude-haiku", vote="agree", confidence=0.91),
    ]


# =====================================================================
# Provider inference
# =====================================================================


class TestProviderInference:
    """Test provider inference from agent names."""

    def test_known_providers(self):
        assert _infer_provider("claude-opus") == "anthropic"
        assert _infer_provider("gpt-4o") == "openai"
        assert _infer_provider("gemini-pro") == "google"
        assert _infer_provider("grok-2") == "xai"
        assert _infer_provider("mistral-large") == "mistral"
        assert _infer_provider("deepseek-v3") == "deepseek"

    def test_unknown_agent_returns_name(self):
        result = _infer_provider("custom-agent-99")
        assert result == "custom-agent-99"

    def test_case_insensitive(self):
        assert _infer_provider("Claude-Opus-4") == "anthropic"
        assert _infer_provider("GPT-4o") == "openai"


# =====================================================================
# Consensus diversity sub-scorer
# =====================================================================


class TestConsensusDiversity:
    """Test the consensus diversity sub-scorer."""

    def test_diverse_providers_high_score(self, scorer: EpistemicScorer, diverse_votes):
        score = scorer._score_consensus_diversity(diverse_votes)
        assert score >= 0.8

    def test_single_provider_low_score(self, scorer: EpistemicScorer, uniform_votes):
        score = scorer._score_consensus_diversity(uniform_votes)
        assert score == 0.3

    def test_two_providers_medium_score(self, scorer: EpistemicScorer):
        votes = [
            MockVote(agent="claude-opus", vote="agree"),
            MockVote(agent="gpt-4o", vote="agree"),
        ]
        score = scorer._score_consensus_diversity(votes)
        assert score == 0.6

    def test_no_votes_neutral(self, scorer: EpistemicScorer):
        score = scorer._score_consensus_diversity([])
        assert score == 0.5

    def test_all_disagree_zero(self, scorer: EpistemicScorer):
        votes = [
            MockVote(agent="claude-opus", vote="disagree"),
            MockVote(agent="gpt-4o", vote="disagree"),
        ]
        score = scorer._score_consensus_diversity(votes)
        assert score == 0.0

    def test_enum_vote_type_handled(self, scorer: EpistemicScorer):
        """Handles VoteType enum objects with .value attribute."""

        class FakeEnum:
            value = "agree"

        @dataclass
        class EnumVote:
            agent: str = "claude-opus"
            vote: Any = field(default_factory=FakeEnum)
            confidence: float = 0.8

        votes = [EnumVote(), EnumVote(agent="gpt-4")]
        score = scorer._score_consensus_diversity(votes)
        assert score >= 0.5


# =====================================================================
# Claim decomposition sub-scorer
# =====================================================================


class TestClaimDecomposition:
    """Test the claim decomposition sub-scorer."""

    def test_all_claims_supported(self, scorer: EpistemicScorer):
        claims = [
            MockClaim(
                claim_id="c1",
                supporting_evidence=[MockEvidence()],
            ),
            MockClaim(
                claim_id="c2",
                supporting_evidence=[MockEvidence(), MockEvidence()],
            ),
        ]
        result = MockDebateResult(claims=claims)
        score = scorer._score_claim_decomposition(result)
        assert score == 1.0

    def test_no_claims_neutral(self, scorer: EpistemicScorer):
        result = MockDebateResult(claims=[])
        score = scorer._score_claim_decomposition(result)
        assert score == 0.5

    def test_half_supported(self, scorer: EpistemicScorer):
        claims = [
            MockClaim(claim_id="c1", supporting_evidence=[MockEvidence()]),
            MockClaim(claim_id="c2", supporting_evidence=[]),
        ]
        result = MockDebateResult(claims=claims)
        score = scorer._score_claim_decomposition(result)
        assert score == 0.5

    def test_parent_claim_counts_as_supported(self, scorer: EpistemicScorer):
        claims = [
            MockClaim(claim_id="c1", parent_claim_id="c0"),
        ]
        result = MockDebateResult(claims=claims)
        score = scorer._score_claim_decomposition(result)
        assert score == 1.0

    def test_missing_claims_attribute_neutral(self, scorer: EpistemicScorer):
        score = scorer._score_claim_decomposition({})
        assert score == 0.5


# =====================================================================
# Calibration quality sub-scorer
# =====================================================================


class TestCalibrationQuality:
    """Test the calibration quality sub-scorer."""

    def test_good_calibration(self, scorer: EpistemicScorer):
        data = {
            "agent-1": {"brier_score": 0.1, "calibration_total": 30},
            "agent-2": {"brier_score": 0.15, "calibration_total": 25},
        }
        score = scorer._score_calibration_quality(data)
        assert score > 0.8

    def test_poor_calibration(self, scorer: EpistemicScorer):
        data = {
            "agent-1": {"brier_score": 0.8, "calibration_total": 30},
        }
        score = scorer._score_calibration_quality(data)
        assert score < 0.3

    def test_no_data_neutral(self, scorer: EpistemicScorer):
        score = scorer._score_calibration_quality(None)
        assert score == 0.5

    def test_top_level_brier_score(self, scorer: EpistemicScorer):
        data = {"brier_score": 0.2}
        score = scorer._score_calibration_quality(data)
        assert abs(score - 0.8) < 0.01

    def test_insufficient_samples_ignored(self, scorer: EpistemicScorer):
        data = {
            "agent-1": {"brier_score": 0.1, "calibration_total": 2},
        }
        score = scorer._score_calibration_quality(data)
        # Too few samples, should return neutral
        assert score == 0.5


# =====================================================================
# Uncertainty acknowledgment sub-scorer
# =====================================================================


class TestUncertaintyAcknowledgment:
    """Test the uncertainty acknowledgment sub-scorer."""

    def test_acknowledges_uncertainty(self, scorer: EpistemicScorer):
        msgs = [
            MockMessage(
                content="I'm not certain about this. There are limitations "
                "and the data is approximately correct."
            ),
        ]
        result = MockDebateResult(messages=msgs)
        score = scorer._score_uncertainty(result)
        assert score > 0.5

    def test_no_uncertainty_low_score(self, scorer: EpistemicScorer):
        msgs = [
            MockMessage(content="The answer is definitely X. This is correct."),
            MockMessage(content="I agree completely. X is the right approach."),
        ]
        result = MockDebateResult(messages=msgs)
        score = scorer._score_uncertainty(result)
        # No uncertainty markers detected
        assert score <= 0.3

    def test_no_messages_neutral(self, scorer: EpistemicScorer):
        result = MockDebateResult(messages=[])
        score = scorer._score_uncertainty(result)
        assert score == 0.5

    def test_dict_with_responses(self, scorer: EpistemicScorer):
        result = {
            "responses": {
                "agent-1": "There is uncertainty about this caveat.",
                "agent-2": "I might be wrong, this is an assumption.",
            }
        }
        score = scorer._score_uncertainty(result)
        assert score > 0.5


# =====================================================================
# Provenance completeness sub-scorer
# =====================================================================


class TestProvenanceCompleteness:
    """Test the provenance completeness sub-scorer."""

    def test_all_claims_have_citations(self, scorer: EpistemicScorer):
        claims = [
            MockClaim(
                claim_id="c1",
                supporting_evidence=[MockEvidence(evidence_type="citation")],
            ),
            MockClaim(
                claim_id="c2",
                supporting_evidence=[MockEvidence(evidence_type="data")],
            ),
        ]
        result = MockDebateResult(claims=claims)
        score = scorer._score_provenance(result, None)
        assert score == 1.0

    def test_no_provenance_neutral(self, scorer: EpistemicScorer):
        result = MockDebateResult()
        score = scorer._score_provenance(result, None)
        assert score == 0.5

    def test_partial_provenance(self, scorer: EpistemicScorer):
        claims = [
            MockClaim(
                claim_id="c1",
                supporting_evidence=[MockEvidence(evidence_type="citation")],
            ),
            MockClaim(claim_id="c2", supporting_evidence=[]),
        ]
        result = MockDebateResult(claims=claims)
        score = scorer._score_provenance(result, None)
        assert score == 0.5

    def test_provenance_chain_without_claims(self, scorer: EpistemicScorer):
        """Having a provenance chain but no claims yields 0.7."""

        @dataclass
        class FakeChain:
            records: list = field(default_factory=lambda: [{"id": "r1"}])

        result = MockDebateResult(claims=[])
        score = scorer._score_provenance(result, FakeChain())
        assert score == 0.7


# =====================================================================
# Hollow consensus risk sub-scorer
# =====================================================================


class TestHollowConsensusRisk:
    """Test the hollow consensus risk sub-scorer."""

    def test_no_trickster_no_votes_neutral(self, scorer: EpistemicScorer):
        score = scorer._score_hollow_consensus(None, [])
        assert score == 0.5

    def test_trickster_clean(self, scorer: EpistemicScorer):
        report = {"hollow_alerts_detected": 0, "total_interventions": 0}
        score = scorer._score_hollow_consensus(report, [])
        assert score == 1.0

    def test_trickster_many_alerts(self, scorer: EpistemicScorer):
        report = {"hollow_alerts_detected": 5, "total_interventions": 3}
        score = scorer._score_hollow_consensus(report, [])
        assert score < 0.5

    def test_uniform_high_confidence_suspicious(self, scorer: EpistemicScorer):
        votes = [
            MockVote(agent="a1", vote="agree", confidence=0.95),
            MockVote(agent="a2", vote="agree", confidence=0.96),
            MockVote(agent="a3", vote="agree", confidence=0.95),
        ]
        score = scorer._score_hollow_consensus(None, votes)
        assert score <= 0.4

    def test_diverse_votes_safe(self, scorer: EpistemicScorer):
        votes = [
            MockVote(agent="a1", vote="agree", confidence=0.8),
            MockVote(agent="a2", vote="disagree", confidence=0.6),
            MockVote(agent="a3", vote="conditional", confidence=0.7),
        ]
        score = scorer._score_hollow_consensus(None, votes)
        assert score >= 0.7


# =====================================================================
# Composite scoring
# =====================================================================


class TestCompositeScoring:
    """Test the full composite scoring pipeline."""

    def test_full_score_all_components(self, scorer: EpistemicScorer):
        claims = [
            MockClaim(
                claim_id="c1",
                supporting_evidence=[MockEvidence(evidence_type="citation")],
            ),
        ]
        msgs = [
            MockMessage(
                content="I'm uncertain about this caveat. There are limitations."
            ),
        ]
        result = MockDebateResult(claims=claims, messages=msgs)
        votes = [
            MockVote(agent="claude-opus", vote="agree", confidence=0.8),
            MockVote(agent="gpt-4o", vote="conditional", confidence=0.7),
            MockVote(agent="gemini-pro", vote="agree", confidence=0.75),
        ]
        calibration = {
            "claude-opus": {"brier_score": 0.15, "calibration_total": 30},
        }
        trickster = {"hollow_alerts_detected": 0, "total_interventions": 0}

        score = scorer.score_debate(
            debate_result=result,
            votes=votes,
            trickster_report=trickster,
            calibration_data=calibration,
        )

        assert 0.0 <= score.overall <= 1.0
        assert score.consensus_diversity >= 0.8
        assert score.hollow_consensus_risk == 1.0
        assert score.calibration_quality > 0.7
        assert len(score.components) == 6

    def test_to_dict(self, scorer: EpistemicScorer):
        result = MockDebateResult()
        votes = [MockVote(agent="claude", vote="agree")]
        score = scorer.score_debate(result, votes)
        d = score.to_dict()
        assert "overall" in d
        assert "components" in d
        assert isinstance(d["components"], dict)

    def test_empty_debate_graceful(self, scorer: EpistemicScorer):
        result = MockDebateResult()
        score = scorer.score_debate(result, [])
        # All components should be neutral or near-neutral
        assert 0.3 <= score.overall <= 0.7

    def test_score_decision_receipt(self, scorer: EpistemicScorer):
        receipt = {
            "votes": [
                {"agent": "claude-opus", "vote": "agree", "confidence": 0.8, "reasoning": "ok"},
                {"agent": "gpt-4", "vote": "agree", "confidence": 0.75, "reasoning": "ok"},
            ],
            "claims": [
                {
                    "claim_id": "c1",
                    "statement": "test",
                    "supporting_evidence": [{"evidence_type": "citation"}],
                },
            ],
            "trickster": {"hollow_alerts_detected": 0, "total_interventions": 0},
            "calibration": {"brier_score": 0.2},
            "messages": [
                {"content": "I'm uncertain about the caveat and limitation."},
            ],
        }
        score = scorer.score_decision(receipt)
        assert 0.0 <= score.overall <= 1.0
        assert score.hollow_consensus_risk == 1.0


# =====================================================================
# Configurable weights
# =====================================================================


class TestConfigurableWeights:
    """Test that weight configuration affects scoring."""

    def test_zero_weight_excludes_component(self):
        config = EpistemicScorerConfig(
            weight_consensus_diversity=0.0,
            weight_claim_decomposition=1.0,
            weight_calibration_quality=0.0,
            weight_uncertainty_acknowledgment=0.0,
            weight_provenance_completeness=0.0,
            weight_hollow_consensus_risk=0.0,
        )
        scorer = EpistemicScorer(config)
        claims = [
            MockClaim(
                claim_id="c1",
                supporting_evidence=[MockEvidence()],
            ),
        ]
        result = MockDebateResult(claims=claims)
        score = scorer.score_debate(result, [])
        # Only claim decomposition matters; it should be 1.0
        assert abs(score.overall - 1.0) < 0.01

    def test_different_weights_change_overall(self):
        config_high_diversity = EpistemicScorerConfig(
            weight_consensus_diversity=5.0,
            weight_claim_decomposition=0.1,
            weight_calibration_quality=0.1,
            weight_uncertainty_acknowledgment=0.1,
            weight_provenance_completeness=0.1,
            weight_hollow_consensus_risk=0.1,
        )
        config_low_diversity = EpistemicScorerConfig(
            weight_consensus_diversity=0.1,
            weight_claim_decomposition=5.0,
            weight_calibration_quality=0.1,
            weight_uncertainty_acknowledgment=0.1,
            weight_provenance_completeness=0.1,
            weight_hollow_consensus_risk=0.1,
        )

        scorer_high = EpistemicScorer(config_high_diversity)
        scorer_low = EpistemicScorer(config_low_diversity)

        # Three diverse providers agreeing, no claims
        votes = [
            MockVote(agent="claude-opus", vote="agree"),
            MockVote(agent="gpt-4", vote="agree"),
            MockVote(agent="gemini", vote="agree"),
        ]
        result = MockDebateResult()

        score_high = scorer_high.score_debate(result, votes)
        score_low = scorer_low.score_debate(result, votes)

        # When diversity weight is high, the high-diversity votes
        # should raise the overall score more
        assert score_high.overall > score_low.overall

    def test_default_config_equal_weights(self):
        config = EpistemicScorerConfig()
        assert config.weight_consensus_diversity == 1.0
        assert config.weight_claim_decomposition == 1.0
        assert config.weight_calibration_quality == 1.0
        assert config.weight_uncertainty_acknowledgment == 1.0
        assert config.weight_provenance_completeness == 1.0
        assert config.weight_hollow_consensus_risk == 1.0


# =====================================================================
# Graceful degradation
# =====================================================================


class TestGracefulDegradation:
    """Test that missing components degrade to neutral 0.5."""

    def test_none_result_safe(self, scorer: EpistemicScorer):
        """Passing None-like objects should not crash."""
        score = scorer.score_debate(
            debate_result={},
            votes=[],
            provenance_chain=None,
            trickster_report=None,
            calibration_data=None,
        )
        assert 0.0 <= score.overall <= 1.0

    def test_dict_result(self, scorer: EpistemicScorer):
        """A plain dict as debate result should work."""
        result = {
            "messages": [],
            "claims": [],
            "final_answer": "Some answer.",
        }
        score = scorer.score_debate(result, [])
        assert 0.0 <= score.overall <= 1.0

    def test_partial_calibration_data(self, scorer: EpistemicScorer):
        """Calibration data with mixed quality entries."""
        data = {
            "good-agent": {"brier_score": 0.1, "calibration_total": 50},
            "bad-entry": "not a dict",
            "small-sample": {"brier_score": 0.05, "calibration_total": 2},
        }
        score = scorer._score_calibration_quality(data)
        # Should use only the good-agent data
        assert score > 0.7
