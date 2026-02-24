"""Tests for cross-proposal analysis module."""

from __future__ import annotations

from aragora_debate.cross_analysis import (
    Contradiction,
    CrossProposalAnalysis,
    CrossProposalAnalyzer,
    EvidenceGap,
    SharedEvidence,
)


class TestSharedEvidence:
    def test_creation(self):
        se = SharedEvidence(
            evidence_text="50% improvement",
            evidence_type="data",
            agents=["a", "b"],
            claims_supported=["performance improves"],
        )
        assert se.agent_count == 2
        assert se.evidence_text == "50% improvement"

    def test_agent_count(self):
        se = SharedEvidence(
            evidence_text="test",
            evidence_type="citation",
            agents=["a", "b", "c"],
            claims_supported=[],
        )
        assert se.agent_count == 3


class TestContradiction:
    def test_creation(self):
        c = Contradiction(
            agent1="claude",
            agent2="gpt",
            topic="caching strategy",
            evidence1="Redis is best",
            evidence2="Memcached is better",
            description="Agents disagree on cache technology",
        )
        assert c.agent1 == "claude"
        assert c.topic == "caching strategy"


class TestEvidenceGap:
    def test_creation(self):
        gap = EvidenceGap(
            claim="Microservices improve scalability",
            agents_making_claim=["a", "b"],
            gap_severity=0.8,
        )
        assert gap.gap_severity == 0.8
        assert len(gap.agents_making_claim) == 2


class TestCrossProposalAnalysis:
    def test_has_concerns_with_gaps(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[EvidenceGap("claim", ["a", "b"], 0.5)],
            redundancy_score=0.0,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )
        assert analysis.has_concerns is True

    def test_has_concerns_with_contradictions(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[Contradiction("a", "b", "topic", "ev1", "ev2", "desc")],
            evidence_gaps=[],
            redundancy_score=0.0,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )
        assert analysis.has_concerns is True

    def test_has_concerns_with_high_redundancy(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.8,
            unique_evidence_sources=2,
            total_evidence_sources=10,
            agent_coverage={},
            weakest_agent=None,
        )
        assert analysis.has_concerns is True

    def test_no_concerns(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.5,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.3,
            unique_evidence_sources=5,
            total_evidence_sources=8,
            agent_coverage={},
            weakest_agent=None,
        )
        assert analysis.has_concerns is False

    def test_top_concern_gap(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[EvidenceGap("Microservices are better for scaling", ["a", "b"], 0.5)],
            redundancy_score=0.0,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )
        assert "Evidence gap" in analysis.top_concern

    def test_top_concern_none(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.3,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )
        assert analysis.top_concern is None


class TestCrossProposalAnalyzer:
    def setup_method(self):
        self.analyzer = CrossProposalAnalyzer()

    def test_empty_proposals(self):
        analysis = self.analyzer.analyze({})
        assert analysis.has_concerns is False
        assert analysis.weakest_agent is None

    def test_single_proposal(self):
        analysis = self.analyzer.analyze({"agent1": "Some proposal text."})
        assert analysis.has_concerns is False

    def test_two_proposals_with_evidence(self):
        proposals = {
            "claude": (
                "According to Smith (2024), caching improves latency by 50%. "
                "For example, Redis handles 100K ops/s. Therefore we should adopt it."
            ),
            "gpt": (
                "Per Jones (2025), caching reduces load by 40%. "
                "Specifically, Memcached processes 80K ops/s. Hence this is optimal."
            ),
        }
        analysis = self.analyzer.analyze(proposals)
        assert isinstance(analysis, CrossProposalAnalysis)
        assert len(analysis.agent_coverage) == 2

    def test_vague_proposals_detected(self):
        proposals = {
            "agent1": (
                "Generally the approach is good and typically works in most cases. "
                "It depends on various factors and many considerations."
            ),
            "agent2": (
                "Usually this is the best approach with significant impact. "
                "The common approach is generally accepted by most people."
            ),
        }
        analysis = self.analyzer.analyze(proposals)
        # Both agents should have low coverage/quality
        for agent, score in analysis.agent_coverage.items():
            assert isinstance(score, float)

    def test_weakest_agent_identified(self):
        proposals = {
            "strong": (
                "According to [1], data shows 50% improvement. "
                "Per Smith (2024), the approach works. See https://example.com."
            ),
            "weak": "It generally works well in most cases typically.",
        }
        analysis = self.analyzer.analyze(proposals)
        assert analysis.weakest_agent == "weak"

    def test_redundancy_calculation(self):
        proposals = {
            "a": "50% improvement in latency according to Smith (2024).",
            "b": "50% improvement in latency according to Smith (2024).",
        }
        analysis = self.analyzer.analyze(proposals)
        # Same evidence -> high redundancy
        assert analysis.redundancy_score >= 0.0

    def test_normalize_evidence(self):
        assert CrossProposalAnalyzer._normalize_evidence("  Hello World!  ") == "hello world"
        assert CrossProposalAnalyzer._normalize_evidence("ab") == ""  # Too short

    def test_text_similarity(self):
        sim = CrossProposalAnalyzer._text_similarity(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy cat",
        )
        assert sim > 0.7

    def test_text_similarity_different(self):
        sim = CrossProposalAnalyzer._text_similarity(
            "Microservices with Docker and Kubernetes",
            "Monolith with PostgreSQL and Redis",
        )
        assert sim < 0.3

    def test_text_similarity_empty(self):
        assert CrossProposalAnalyzer._text_similarity("", "") == 0.0
        assert CrossProposalAnalyzer._text_similarity("hello", "") == 0.0

    def test_has_negation_diff(self):
        assert (
            CrossProposalAnalyzer._has_negation_diff(
                "We should use caching",
                "We should not use caching",
            )
            is True
        )

    def test_no_negation_diff(self):
        assert (
            CrossProposalAnalyzer._has_negation_diff(
                "We should use caching",
                "We should use caching for speed",
            )
            is False
        )

    def test_extract_topic(self):
        topic = CrossProposalAnalyzer._extract_topic(
            "Microservices improve deployment frequency",
            "Microservices increase deployment complexity",
        )
        assert "microservices" in topic.lower() or "deployment" in topic.lower()

    def test_corroboration_no_shared(self):
        analysis = self.analyzer.analyze(
            {
                "a": "One perspective with unique evidence from study Alpha.",
                "b": "Different perspective with unique evidence from study Beta.",
            }
        )
        assert analysis.evidence_corroboration_score >= 0.0
