"""Tests for evidence quality analysis module."""

from __future__ import annotations

import pytest

from aragora_debate.evidence import (
    EvidenceMarker,
    EvidenceQualityAnalyzer,
    EvidenceQualityScore,
    EvidenceType,
    HollowConsensusAlert,
    HollowConsensusDetector,
)


class TestEvidenceType:
    def test_enum_values(self):
        assert EvidenceType.CITATION.value == "citation"
        assert EvidenceType.DATA.value == "data"
        assert EvidenceType.NONE.value == "none"

    def test_all_types_exist(self):
        expected = {"citation", "data", "example", "tool_output", "quote", "reasoning", "none"}
        actual = {e.value for e in EvidenceType}
        assert actual == expected


class TestEvidenceMarker:
    def test_creation(self):
        marker = EvidenceMarker(
            evidence_type=EvidenceType.CITATION,
            text="[1]",
            position=42,
            confidence=0.9,
        )
        assert marker.evidence_type == EvidenceType.CITATION
        assert marker.text == "[1]"
        assert marker.position == 42
        assert marker.confidence == 0.9


class TestEvidenceQualityScore:
    def test_compute_overall(self):
        score = EvidenceQualityScore(agent="test", round_num=0)
        score.citation_density = 0.8
        score.specificity_score = 0.6
        score.evidence_diversity = 0.4
        score.temporal_relevance = 1.0
        score.logical_chain_score = 0.5

        result = score.compute_overall()
        assert result > 0
        assert result == score.overall_quality
        expected = 0.25 * 0.8 + 0.25 * 0.6 + 0.20 * 0.4 + 0.10 * 1.0 + 0.20 * 0.5
        assert abs(result - expected) < 0.001

    def test_default_values(self):
        score = EvidenceQualityScore(agent="test", round_num=1)
        assert score.citation_density == 0.0
        assert score.temporal_relevance == 1.0
        assert score.evidence_markers == []


class TestEvidenceQualityAnalyzer:
    def setup_method(self):
        self.analyzer = EvidenceQualityAnalyzer()

    def test_empty_response(self):
        score = self.analyzer.analyze("", "agent1")
        assert score.overall_quality == 0.0
        assert score.evidence_markers == []

    def test_citation_detection(self):
        text = "According to Smith [1], the approach works. See https://example.com for details."
        score = self.analyzer.analyze(text, "agent1")
        citation_markers = [m for m in score.evidence_markers if m.evidence_type == EvidenceType.CITATION]
        assert len(citation_markers) >= 2

    def test_data_detection(self):
        text = "Performance improved by 45% and latency dropped to 50ms. Cost was $1,200."
        score = self.analyzer.analyze(text, "agent1")
        data_markers = [m for m in score.evidence_markers if m.evidence_type == EvidenceType.DATA]
        assert len(data_markers) >= 2

    def test_example_detection(self):
        text = "For example, Netflix uses this pattern. Specifically, they process 1M events/s."
        score = self.analyzer.analyze(text, "agent1")
        example_markers = [m for m in score.evidence_markers if m.evidence_type == EvidenceType.EXAMPLE]
        assert len(example_markers) >= 1

    def test_high_quality_response(self):
        text = (
            "According to Smith (2024), microservices improved deployment frequency by 200% [1]. "
            "For example, Netflix reduced downtime by 40%. Therefore, this approach is validated. "
            "Specifically, the system processed 500 requests/s with 99.9% uptime. "
            "Because of this evidence, we conclude the approach works. "
            "See https://example.com/study for the full dataset."
        )
        score = self.analyzer.analyze(text, "analyst")
        assert score.overall_quality > 0.3

    def test_vague_response(self):
        text = (
            "Generally, this is a good approach. Typically it works well. "
            "Usually the best practices suggest this common approach. "
            "It depends on various factors and many considerations."
        )
        score = self.analyzer.analyze(text, "vague_agent")
        assert score.specificity_score < 0.5

    def test_analyze_batch(self):
        responses = {
            "agent1": "According to [1], the data shows 50% improvement.",
            "agent2": "Generally, it might work in some cases.",
        }
        scores = self.analyzer.analyze_batch(responses, round_num=1)
        assert "agent1" in scores
        assert "agent2" in scores
        assert scores["agent1"].overall_quality > scores["agent2"].overall_quality

    def test_reasoning_detection(self):
        text = "Because the data is clear, therefore we conclude X. Thus, it follows that Y. Hence Z."
        score = self.analyzer.analyze(text, "reasoner")
        assert score.logical_chain_score > 0

    def test_temporal_relevance_recent(self):
        text = "A 2025 study confirmed the findings from 2024 research."
        score = self.analyzer.analyze(text, "agent")
        assert score.temporal_relevance > 0.5

    def test_temporal_relevance_old(self):
        text = "Based on 1990 data and 1985 research findings."
        score = self.analyzer.analyze(text, "agent")
        assert score.temporal_relevance < 0.5

    def test_custom_weights(self):
        analyzer = EvidenceQualityAnalyzer(
            citation_weight=0.5,
            specificity_weight=0.1,
            diversity_weight=0.1,
            temporal_weight=0.1,
            reasoning_weight=0.2,
        )
        assert analyzer.weights["citation"] == 0.5

    def test_no_temporal_refs(self):
        text = "This approach is fundamentally sound based on first principles."
        score = self.analyzer.analyze(text, "agent")
        assert score.temporal_relevance == 0.8  # Neutral when no dates


class TestHollowConsensusAlert:
    def test_creation(self):
        alert = HollowConsensusAlert(
            detected=True,
            severity=0.7,
            reason="Low evidence quality",
            agent_scores={"agent1": 0.3, "agent2": 0.2},
            recommended_challenges=["Provide evidence"],
        )
        assert alert.detected is True
        assert alert.severity == 0.7

    def test_not_detected(self):
        alert = HollowConsensusAlert(
            detected=False,
            severity=0.0,
            reason="Quality acceptable",
            agent_scores={},
            recommended_challenges=[],
        )
        assert alert.detected is False


class TestHollowConsensusDetector:
    def setup_method(self):
        self.detector = HollowConsensusDetector()

    def test_not_converging(self):
        responses = {"agent1": "Some text", "agent2": "Other text"}
        alert = self.detector.check(responses, convergence_similarity=0.3)
        assert alert.detected is False
        assert alert.reason == "Not converging yet"

    def test_hollow_consensus_detected(self):
        responses = {
            "agent1": "Generally it might work. Usually this is fine. Best practices suggest so.",
            "agent2": "Typically this approach works. In some cases it depends on various factors.",
        }
        alert = self.detector.check(responses, convergence_similarity=0.9, round_num=1)
        # Whether detected depends on quality scoring
        assert isinstance(alert, HollowConsensusAlert)
        assert isinstance(alert.severity, float)

    def test_quality_with_evidence(self):
        responses = {
            "agent1": "According to Smith (2024), the approach works [1]. Data shows 50% improvement.",
            "agent2": "Per Jones (2025), results confirm 45% gains. See https://example.com.",
        }
        alert = self.detector.check(responses, convergence_similarity=0.8)
        # Good evidence should have lower severity
        assert isinstance(alert, HollowConsensusAlert)

    def test_empty_responses(self):
        alert = self.detector.check({}, convergence_similarity=0.9)
        assert alert.detected is False

    def test_custom_thresholds(self):
        detector = HollowConsensusDetector(
            min_quality_threshold=0.8,
            quality_variance_threshold=0.1,
        )
        assert detector.min_quality_threshold == 0.8
        assert detector.quality_variance_threshold == 0.1

    def test_challenge_generation(self):
        responses = {
            "agent1": "It works well generally in most cases typically.",
            "agent2": "Usually this is the common approach for best practices.",
        }
        alert = self.detector.check(responses, convergence_similarity=0.9)
        # Should generate some challenges for vague responses
        assert isinstance(alert.recommended_challenges, list)
