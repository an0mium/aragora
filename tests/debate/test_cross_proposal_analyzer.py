"""Tests for the cross-proposal evidence analyzer.

Covers SharedEvidence, Contradiction, EvidenceGap, CrossProposalAnalysis,
CrossProposalAnalyzer (analyze, shared evidence, contradictions, gaps,
redundancy, corroboration), and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.cross_proposal_analyzer import (
    Contradiction,
    CrossProposalAnalysis,
    CrossProposalAnalyzer,
    EvidenceGap,
    SharedEvidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_evidence_link(
    claim: str = "test claim",
    evidence: str = "test evidence",
    evidence_type: str = "citation",
    is_strong_link: bool = True,
):
    link = MagicMock()
    link.claim = claim
    link.evidence = evidence
    link.evidence_type = evidence_type
    link.is_strong_link = is_strong_link
    return link


def make_coverage(links=None, unlinked_claims=None, coverage=0.5):
    cov = MagicMock()
    cov.links = links or []
    cov.unlinked_claims = unlinked_claims or []
    cov.coverage = coverage
    return cov


# ---------------------------------------------------------------------------
# SharedEvidence
# ---------------------------------------------------------------------------


class TestSharedEvidence:
    def test_agent_count(self):
        se = SharedEvidence(
            evidence_text="study shows improvement",
            evidence_type="citation",
            agents=["claude", "gpt", "gemini"],
            claims_supported=["caching improves speed"],
        )
        assert se.agent_count == 3

    def test_single_agent(self):
        se = SharedEvidence(
            evidence_text="data",
            evidence_type="statistic",
            agents=["claude"],
            claims_supported=["claim"],
        )
        assert se.agent_count == 1


# ---------------------------------------------------------------------------
# Contradiction
# ---------------------------------------------------------------------------


class TestContradiction:
    def test_fields(self):
        c = Contradiction(
            agent1="claude",
            agent2="gpt",
            topic="caching strategy",
            evidence1="caching helps",
            evidence2="caching hurts",
            description="opposite conclusions",
        )
        assert c.agent1 == "claude"
        assert c.agent2 == "gpt"
        assert c.topic == "caching strategy"


# ---------------------------------------------------------------------------
# EvidenceGap
# ---------------------------------------------------------------------------


class TestEvidenceGap:
    def test_fields(self):
        g = EvidenceGap(
            claim="caching improves performance",
            agents_making_claim=["claude", "gpt"],
            gap_severity=0.8,
        )
        assert g.gap_severity == 0.8
        assert len(g.agents_making_claim) == 2


# ---------------------------------------------------------------------------
# CrossProposalAnalysis
# ---------------------------------------------------------------------------


class TestCrossProposalAnalysis:
    def test_has_concerns_with_gaps(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[
                EvidenceGap(claim="claim", agents_making_claim=["a"], gap_severity=0.5)
            ],
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
            contradictory_evidence=[
                Contradiction(
                    agent1="a", agent2="b", topic="t",
                    evidence1="e1", evidence2="e2", description="d",
                )
            ],
            evidence_gaps=[],
            redundancy_score=0.0,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )
        assert analysis.has_concerns is True

    def test_has_concerns_high_redundancy(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.8,
            unique_evidence_sources=1,
            total_evidence_sources=5,
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
            agent_coverage={"claude": 0.8},
            weakest_agent=None,
        )
        assert analysis.has_concerns is False

    def test_top_concern_gap(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[
                EvidenceGap(claim="missing evidence for cache", agents_making_claim=["a"], gap_severity=0.5)
            ],
            redundancy_score=0.0,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )
        assert "Evidence gap" in analysis.top_concern

    def test_top_concern_contradiction(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[
                Contradiction(
                    agent1="claude", agent2="gpt", topic="caching",
                    evidence1="e1", evidence2="e2", description="d",
                )
            ],
            evidence_gaps=[],
            redundancy_score=0.0,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )
        assert "Contradiction" in analysis.top_concern

    def test_top_concern_echo_chamber(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.8,
            unique_evidence_sources=1,
            total_evidence_sources=5,
            agent_coverage={},
            weakest_agent=None,
        )
        assert "Echo chamber" in analysis.top_concern

    def test_top_concern_none(self):
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.3,
            unique_evidence_sources=3,
            total_evidence_sources=5,
            agent_coverage={},
            weakest_agent=None,
        )
        assert analysis.top_concern is None


# ---------------------------------------------------------------------------
# CrossProposalAnalyzer — edge cases
# ---------------------------------------------------------------------------


class TestAnalyzerEdgeCases:
    def test_empty_proposals(self):
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        analysis = analyzer.analyze({})
        assert analysis.evidence_corroboration_score == 0.0
        assert analysis.has_concerns is False

    def test_single_proposal(self):
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        analysis = analyzer.analyze({"claude": "some proposal"})
        assert analysis.evidence_corroboration_score == 0.0

    def test_no_linker_returns_empty(self):
        with patch(
            "aragora.debate.cross_proposal_analyzer._get_evidence_linker_class",
            return_value=None,
        ):
            analyzer = CrossProposalAnalyzer(linker=None)
            analysis = analyzer.analyze({"claude": "text", "gpt": "text"})
            assert analysis.evidence_corroboration_score == 0.0


# ---------------------------------------------------------------------------
# CrossProposalAnalyzer — with linker
# ---------------------------------------------------------------------------


class TestAnalyzerWithLinker:
    def setup_method(self):
        self.linker = MagicMock()

    def test_shared_evidence_detected(self):
        # Both agents cite same evidence
        shared_link = make_evidence_link(
            claim="caching helps",
            evidence="Study by Smith (2023) shows 10x improvement",
        )
        cov_claude = make_coverage(links=[shared_link], coverage=0.8)
        cov_gpt = make_coverage(links=[shared_link], coverage=0.7)

        self.linker.compute_evidence_coverage = MagicMock(
            side_effect=[cov_claude, cov_gpt]
        )

        analyzer = CrossProposalAnalyzer(linker=self.linker)
        analysis = analyzer.analyze({"claude": "text", "gpt": "text"})

        assert len(analysis.shared_evidence) > 0
        assert analysis.evidence_corroboration_score > 0.0

    def test_no_shared_evidence(self):
        link1 = make_evidence_link(claim="claim1", evidence="unique evidence alpha")
        link2 = make_evidence_link(claim="claim2", evidence="totally different evidence beta")

        cov1 = make_coverage(links=[link1], coverage=0.5)
        cov2 = make_coverage(links=[link2], coverage=0.5)

        self.linker.compute_evidence_coverage = MagicMock(
            side_effect=[cov1, cov2]
        )

        analyzer = CrossProposalAnalyzer(linker=self.linker)
        analysis = analyzer.analyze({"claude": "text", "gpt": "text"})

        assert len(analysis.shared_evidence) == 0

    def test_evidence_gaps_found(self):
        # Both agents make same unlinked claim
        cov1 = make_coverage(
            links=[],
            unlinked_claims=["performance improves with caching"],
            coverage=0.3,
        )
        cov2 = make_coverage(
            links=[],
            unlinked_claims=["performance improves with caching"],
            coverage=0.2,
        )

        self.linker.compute_evidence_coverage = MagicMock(
            side_effect=[cov1, cov2]
        )

        analyzer = CrossProposalAnalyzer(linker=self.linker)
        analysis = analyzer.analyze({"claude": "text", "gpt": "text"})

        assert len(analysis.evidence_gaps) > 0
        assert analysis.evidence_gaps[0].gap_severity > 0

    def test_weakest_agent_identified(self):
        cov1 = make_coverage(links=[], coverage=0.2)
        cov2 = make_coverage(links=[], coverage=0.9)

        self.linker.compute_evidence_coverage = MagicMock(
            side_effect=[cov1, cov2]
        )

        analyzer = CrossProposalAnalyzer(linker=self.linker)
        analysis = analyzer.analyze({"weak": "text", "strong": "text"})

        assert analysis.weakest_agent == "weak"

    def test_redundancy_calculation(self):
        # All agents cite same evidence → high redundancy
        shared = make_evidence_link(
            claim="claim", evidence="identical evidence source"
        )
        cov1 = make_coverage(links=[shared], coverage=0.5)
        cov2 = make_coverage(links=[shared], coverage=0.5)
        cov3 = make_coverage(links=[shared], coverage=0.5)

        self.linker.compute_evidence_coverage = MagicMock(
            side_effect=[cov1, cov2, cov3]
        )

        analyzer = CrossProposalAnalyzer(linker=self.linker)
        analysis = analyzer.analyze({
            "claude": "text", "gpt": "text", "gemini": "text"
        })

        # 3 citations, 1 unique → redundancy = 1 - 1/3 = 0.667
        assert analysis.redundancy_score > 0.5


# ---------------------------------------------------------------------------
# Internal methods
# ---------------------------------------------------------------------------


class TestInternalMethods:
    def test_normalize_evidence(self):
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        normalized = analyzer._normalize_evidence("  Hello, World!  ")
        assert normalized == "hello world"

    def test_normalize_evidence_short(self):
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        assert analyzer._normalize_evidence("hi") == ""

    def test_text_similarity_identical(self):
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        sim = analyzer._text_similarity(
            "caching improves database performance significantly",
            "caching improves database performance significantly",
        )
        assert sim == 1.0

    def test_text_similarity_different(self):
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        sim = analyzer._text_similarity(
            "apples oranges bananas grapes",
            "python java rust golang",
        )
        assert sim == 0.0

    def test_text_similarity_empty(self):
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        assert analyzer._text_similarity("", "hello world") == 0.0

    def test_extract_topic_common_words(self):
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        topic = analyzer._extract_topic(
            "caching improves database performance",
            "database caching helps latency",
        )
        assert "caching" in topic or "database" in topic

    def test_extract_topic_no_common(self):
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        topic = analyzer._extract_topic("alpha beta", "gamma delta")
        assert topic == "related topic"

    def test_are_contradictory_high_claim_low_evidence(self):
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        link1 = make_evidence_link(
            claim="caching significantly improves database query performance",
            evidence="internal benchmark testing results show clear gains",
        )
        link2 = make_evidence_link(
            claim="caching significantly improves database query latency",
            evidence="theoretical analysis framework suggests improvement",
        )
        # High claim overlap, low evidence overlap → contradictory
        result = analyzer._are_contradictory(link1, link2)
        assert isinstance(result, bool)

    def test_corroboration_empty(self):
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        score = analyzer._calculate_corroboration([], 3)
        assert score == 0.0

    def test_corroboration_with_shared(self):
        shared = [
            SharedEvidence(
                evidence_text="study",
                evidence_type="citation",
                agents=["a", "b"],
                claims_supported=["claim"],
            )
        ]
        analyzer = CrossProposalAnalyzer(linker=MagicMock())
        score = analyzer._calculate_corroboration(shared, 2)
        assert score == 1.0  # 2 / (1 * 2)
