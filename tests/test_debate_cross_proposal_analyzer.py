"""
Tests for Cross-Proposal Analyzer module.

Tests cover:
- SharedEvidence dataclass
- Contradiction dataclass
- EvidenceGap dataclass
- CrossProposalAnalysis dataclass
- CrossProposalAnalyzer initialization
- Empty analysis handling
- Evidence normalization
- Text similarity calculation
- Topic extraction
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.evidence_quality import EvidenceType


# ============================================================================
# Mock Classes
# ============================================================================

@dataclass
class MockEvidenceLink:
    """Mock evidence link for testing."""
    claim: str
    evidence: str
    evidence_type: EvidenceType = EvidenceType.CITATION
    is_strong_link: bool = True


@dataclass
class MockEvidenceCoverageResult:
    """Mock evidence coverage result for testing."""
    coverage: float = 0.5
    links: list = field(default_factory=list)
    unlinked_claims: list = field(default_factory=list)


class MockEvidenceClaimLinker:
    """Mock linker for testing without ML dependencies."""

    def compute_evidence_coverage(self, text: str) -> MockEvidenceCoverageResult:
        # Simple mock: detect some patterns
        links = []
        if "according to" in text.lower():
            links.append(MockEvidenceLink(
                claim="We should use caching",
                evidence="According to best practices...",
                evidence_type=EvidenceType.CITATION,
            ))
        if "studies show" in text.lower():
            links.append(MockEvidenceLink(
                claim="Caching improves performance",
                evidence="Studies show 50% improvement...",
                evidence_type=EvidenceType.DATA,
            ))

        coverage = len(links) / 3 if links else 0.0
        return MockEvidenceCoverageResult(coverage=coverage, links=links)


# ============================================================================
# Import with mocking
# ============================================================================

@pytest.fixture
def analyzer_module():
    """Import the module with mocked dependencies."""
    from aragora.debate import cross_proposal_analyzer
    return cross_proposal_analyzer


@pytest.fixture
def SharedEvidence(analyzer_module):
    return analyzer_module.SharedEvidence


@pytest.fixture
def Contradiction(analyzer_module):
    return analyzer_module.Contradiction


@pytest.fixture
def EvidenceGap(analyzer_module):
    return analyzer_module.EvidenceGap


@pytest.fixture
def CrossProposalAnalysis(analyzer_module):
    return analyzer_module.CrossProposalAnalysis


@pytest.fixture
def CrossProposalAnalyzer(analyzer_module):
    return analyzer_module.CrossProposalAnalyzer


# ============================================================================
# SharedEvidence Tests
# ============================================================================

class TestSharedEvidence:
    """Tests for SharedEvidence dataclass."""

    def test_basic_creation(self, SharedEvidence):
        """Test basic SharedEvidence creation."""
        evidence = SharedEvidence(
            evidence_text="According to research...",
            evidence_type=EvidenceType.CITATION,
            agents=["claude", "gpt"],
            claims_supported=["Caching helps"],
        )

        assert evidence.evidence_text == "According to research..."
        assert evidence.evidence_type == EvidenceType.CITATION
        assert evidence.agents == ["claude", "gpt"]

    def test_agent_count_property(self, SharedEvidence):
        """Test agent_count property."""
        evidence = SharedEvidence(
            evidence_text="Test",
            evidence_type=EvidenceType.DATA,
            agents=["a", "b", "c"],
            claims_supported=[],
        )

        assert evidence.agent_count == 3

    def test_single_agent(self, SharedEvidence):
        """Test with single agent."""
        evidence = SharedEvidence(
            evidence_text="Test",
            evidence_type=EvidenceType.EXAMPLE,
            agents=["solo"],
            claims_supported=[],
        )

        assert evidence.agent_count == 1


# ============================================================================
# Contradiction Tests
# ============================================================================

class TestContradiction:
    """Tests for Contradiction dataclass."""

    def test_basic_creation(self, Contradiction):
        """Test basic Contradiction creation."""
        contradiction = Contradiction(
            agent1="claude",
            agent2="gpt",
            topic="caching approach",
            evidence1="Caching is always good",
            evidence2="Caching adds complexity",
            description="Agents disagree on caching",
        )

        assert contradiction.agent1 == "claude"
        assert contradiction.agent2 == "gpt"
        assert contradiction.topic == "caching approach"

    def test_description_field(self, Contradiction):
        """Test description field."""
        contradiction = Contradiction(
            agent1="a",
            agent2="b",
            topic="topic",
            evidence1="e1",
            evidence2="e2",
            description="Custom description here",
        )

        assert "Custom description" in contradiction.description


# ============================================================================
# EvidenceGap Tests
# ============================================================================

class TestEvidenceGap:
    """Tests for EvidenceGap dataclass."""

    def test_basic_creation(self, EvidenceGap):
        """Test basic EvidenceGap creation."""
        gap = EvidenceGap(
            claim="Caching is essential",
            agents_making_claim=["claude", "gpt", "gemini"],
            gap_severity=0.8,
        )

        assert gap.claim == "Caching is essential"
        assert len(gap.agents_making_claim) == 3
        assert gap.gap_severity == 0.8

    def test_severity_bounds(self, EvidenceGap):
        """Test gap severity values."""
        gap_low = EvidenceGap(
            claim="Test",
            agents_making_claim=["a"],
            gap_severity=0.1,
        )
        gap_high = EvidenceGap(
            claim="Test",
            agents_making_claim=["a", "b", "c"],
            gap_severity=1.0,
        )

        assert gap_low.gap_severity == 0.1
        assert gap_high.gap_severity == 1.0


# ============================================================================
# CrossProposalAnalysis Tests
# ============================================================================

class TestCrossProposalAnalysis:
    """Tests for CrossProposalAnalysis dataclass."""

    def test_basic_creation(self, CrossProposalAnalysis, SharedEvidence, Contradiction, EvidenceGap):
        """Test basic analysis creation."""
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.5,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.3,
            unique_evidence_sources=5,
            total_evidence_sources=10,
            agent_coverage={"claude": 0.8, "gpt": 0.6},
            weakest_agent="gpt",
        )

        assert analysis.evidence_corroboration_score == 0.5
        assert analysis.redundancy_score == 0.3
        assert analysis.weakest_agent == "gpt"

    def test_has_concerns_with_gaps(self, CrossProposalAnalysis, EvidenceGap):
        """Test has_concerns with evidence gaps."""
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[EvidenceGap(
                claim="Test claim",
                agents_making_claim=["a", "b"],
                gap_severity=0.5,
            )],
            redundancy_score=0.0,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )

        assert analysis.has_concerns is True

    def test_has_concerns_with_contradictions(self, CrossProposalAnalysis, Contradiction):
        """Test has_concerns with contradictions."""
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[Contradiction(
                agent1="a",
                agent2="b",
                topic="test",
                evidence1="e1",
                evidence2="e2",
                description="desc",
            )],
            evidence_gaps=[],
            redundancy_score=0.0,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )

        assert analysis.has_concerns is True

    def test_has_concerns_high_redundancy(self, CrossProposalAnalysis):
        """Test has_concerns with high redundancy (echo chamber)."""
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.8,  # Above 0.7 threshold
            unique_evidence_sources=1,
            total_evidence_sources=10,
            agent_coverage={},
            weakest_agent=None,
        )

        assert analysis.has_concerns is True

    def test_has_concerns_false(self, CrossProposalAnalysis):
        """Test has_concerns is false when no issues."""
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.8,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.3,  # Below threshold
            unique_evidence_sources=5,
            total_evidence_sources=8,
            agent_coverage={},
            weakest_agent=None,
        )

        assert analysis.has_concerns is False

    def test_top_concern_gap(self, CrossProposalAnalysis, EvidenceGap):
        """Test top_concern returns evidence gap."""
        gap = EvidenceGap(
            claim="Important unsupported claim here",
            agents_making_claim=["a", "b"],
            gap_severity=0.8,
        )
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[gap],
            redundancy_score=0.3,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )

        assert "Evidence gap" in analysis.top_concern
        assert "Important" in analysis.top_concern

    def test_top_concern_contradiction(self, CrossProposalAnalysis, Contradiction):
        """Test top_concern returns contradiction."""
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[Contradiction(
                agent1="claude",
                agent2="gpt",
                topic="performance",
                evidence1="e1",
                evidence2="e2",
                description="desc",
            )],
            evidence_gaps=[],
            redundancy_score=0.3,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )

        assert "Contradiction" in analysis.top_concern
        assert "claude" in analysis.top_concern
        assert "gpt" in analysis.top_concern

    def test_top_concern_echo_chamber(self, CrossProposalAnalysis):
        """Test top_concern returns echo chamber warning."""
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.85,
            unique_evidence_sources=1,
            total_evidence_sources=10,
            agent_coverage={},
            weakest_agent=None,
        )

        assert "Echo chamber" in analysis.top_concern
        assert "85%" in analysis.top_concern

    def test_top_concern_none(self, CrossProposalAnalysis):
        """Test top_concern is None when no concerns."""
        analysis = CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.8,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.3,
            unique_evidence_sources=5,
            total_evidence_sources=8,
            agent_coverage={},
            weakest_agent=None,
        )

        assert analysis.top_concern is None


# ============================================================================
# CrossProposalAnalyzer Init Tests
# ============================================================================

class TestAnalyzerInit:
    """Tests for CrossProposalAnalyzer initialization."""

    def test_default_init(self, CrossProposalAnalyzer):
        """Test default initialization."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        assert analyzer.min_redundancy_similarity == 0.7
        assert analyzer.min_claim_overlap == 0.5

    def test_custom_thresholds(self, CrossProposalAnalyzer):
        """Test custom threshold initialization."""
        analyzer = CrossProposalAnalyzer(
            linker=MockEvidenceClaimLinker(),
            min_redundancy_similarity=0.8,
            min_claim_overlap=0.6,
        )

        assert analyzer.min_redundancy_similarity == 0.8
        assert analyzer.min_claim_overlap == 0.6


# ============================================================================
# Empty Analysis Tests
# ============================================================================

class TestEmptyAnalysis:
    """Tests for empty analysis scenarios."""

    def test_empty_proposals(self, CrossProposalAnalyzer):
        """Test analysis with empty proposals."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())
        analysis = analyzer.analyze({})

        assert analysis.shared_evidence == []
        assert analysis.evidence_corroboration_score == 0.0
        assert analysis.redundancy_score == 0.0

    def test_single_proposal(self, CrossProposalAnalyzer):
        """Test analysis with single proposal."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())
        analysis = analyzer.analyze({"claude": "Test proposal"})

        assert analysis.shared_evidence == []
        assert analysis.evidence_corroboration_score == 0.0

    def test_none_linker(self, CrossProposalAnalyzer):
        """Test analysis when linker is None."""
        analyzer = CrossProposalAnalyzer(linker=None)
        analysis = analyzer.analyze({
            "claude": "Test 1",
            "gpt": "Test 2",
        })

        # Should return empty analysis
        assert analysis.shared_evidence == []
        assert analysis.evidence_corroboration_score == 0.0


# ============================================================================
# Evidence Normalization Tests
# ============================================================================

class TestEvidenceNormalization:
    """Tests for evidence normalization."""

    def test_normalize_whitespace(self, CrossProposalAnalyzer):
        """Test whitespace normalization."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        result = analyzer._normalize_evidence("This   has   extra   spaces")
        assert "   " not in result

    def test_normalize_lowercase(self, CrossProposalAnalyzer):
        """Test lowercase normalization."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        result = analyzer._normalize_evidence("ALL CAPS TEXT")
        assert result == result.lower()

    def test_normalize_punctuation(self, CrossProposalAnalyzer):
        """Test punctuation removal."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        result = analyzer._normalize_evidence("Test, with: punctuation!")
        assert "," not in result
        assert ":" not in result
        assert "!" not in result

    def test_normalize_short_string(self, CrossProposalAnalyzer):
        """Test short strings return empty."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        result = analyzer._normalize_evidence("Hi")
        assert result == ""


# ============================================================================
# Text Similarity Tests
# ============================================================================

class TestTextSimilarity:
    """Tests for text similarity calculation."""

    def test_identical_text(self, CrossProposalAnalyzer):
        """Test identical texts have high similarity."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        similarity = analyzer._text_similarity(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        )

        assert similarity == 1.0

    def test_similar_text(self, CrossProposalAnalyzer):
        """Test similar texts have moderate similarity."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        similarity = analyzer._text_similarity(
            "The quick brown fox jumps over the lazy dog",
            "A quick brown fox leaps over a sleeping dog",
        )

        assert similarity > 0.3  # Some overlap

    def test_different_text(self, CrossProposalAnalyzer):
        """Test different texts have low similarity."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        similarity = analyzer._text_similarity(
            "Implementing caching mechanisms",
            "Weather forecast for tomorrow",
        )

        assert similarity < 0.3

    def test_empty_text(self, CrossProposalAnalyzer):
        """Test empty texts return 0."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        similarity = analyzer._text_similarity("", "Test")
        assert similarity == 0.0


# ============================================================================
# Topic Extraction Tests
# ============================================================================

class TestTopicExtraction:
    """Tests for topic extraction from claims."""

    def test_common_words_extracted(self, CrossProposalAnalyzer):
        """Test common words are extracted."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        topic = analyzer._extract_topic(
            "Caching improves performance significantly",
            "Performance benefits from caching strategies",
        )

        # Should find common words
        assert len(topic) > 0

    def test_no_common_words(self, CrossProposalAnalyzer):
        """Test fallback when no common words."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        topic = analyzer._extract_topic(
            "Apple banana cherry",
            "Delta echo foxtrot",
        )

        assert topic == "related topic"

    def test_topic_length_limited(self, CrossProposalAnalyzer):
        """Test topic is limited to 3 words."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        topic = analyzer._extract_topic(
            "Testing common words shared between multiple claims here",
            "Testing common words shared between different statements there",
        )

        # Should be at most 3 words
        assert len(topic.split()) <= 3


# ============================================================================
# Corroboration Calculation Tests
# ============================================================================

class TestCorroboration:
    """Tests for corroboration score calculation."""

    def test_no_shared_evidence(self, CrossProposalAnalyzer):
        """Test corroboration with no shared evidence."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        score = analyzer._calculate_corroboration([], 3)
        assert score == 0.0

    def test_single_agent(self, CrossProposalAnalyzer):
        """Test corroboration with single agent."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        score = analyzer._calculate_corroboration([], 1)
        assert score == 0.0


# ============================================================================
# Redundancy Calculation Tests
# ============================================================================

class TestRedundancy:
    """Tests for redundancy calculation."""

    def test_no_evidence(self, CrossProposalAnalyzer):
        """Test redundancy with no evidence."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        redundancy, unique, total = analyzer._calculate_redundancy({}, {})

        assert redundancy == 0.0
        assert unique == 0
        assert total == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for CrossProposalAnalyzer."""

    def test_analyze_with_mock_linker(self, CrossProposalAnalyzer):
        """Test full analysis with mock linker."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        proposals = {
            "claude": "According to best practices, we should use caching.",
            "gpt": "Studies show that caching improves performance by 50%.",
            "gemini": "I agree that caching is beneficial.",
        }

        analysis = analyzer.analyze(proposals)

        # Should produce valid analysis structure
        assert isinstance(analysis.shared_evidence, list)
        assert isinstance(analysis.contradictory_evidence, list)
        assert isinstance(analysis.evidence_gaps, list)
        assert 0.0 <= analysis.redundancy_score <= 1.0
        assert 0.0 <= analysis.evidence_corroboration_score <= 1.0

    def test_agent_coverage_populated(self, CrossProposalAnalyzer):
        """Test agent coverage is populated."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        proposals = {
            "claude": "According to documentation, this works.",
            "gpt": "The evidence suggests this approach.",
        }

        analysis = analyzer.analyze(proposals)

        # Agent coverage should be populated
        assert "claude" in analysis.agent_coverage
        assert "gpt" in analysis.agent_coverage

    def test_weakest_agent_identified(self, CrossProposalAnalyzer):
        """Test weakest agent is identified."""
        analyzer = CrossProposalAnalyzer(linker=MockEvidenceClaimLinker())

        proposals = {
            "claude": "According to research and studies show improvement.",
            "gpt": "I think this is good.",  # No evidence markers
        }

        analysis = analyzer.analyze(proposals)

        # Should identify a weakest agent
        if analysis.agent_coverage:
            assert analysis.weakest_agent in ["claude", "gpt"]
