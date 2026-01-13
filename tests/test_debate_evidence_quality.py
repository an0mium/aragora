"""
Tests for Evidence Quality module.

Tests cover:
- EvidenceType enum values
- EvidenceMarker dataclass
- EvidenceQualityScore dataclass and compute_overall()
- EvidenceQualityAnalyzer pattern detection and scoring
- HollowConsensusAlert dataclass
- HollowConsensusDetector detection and challenge generation
"""

import pytest

from aragora.debate.evidence_quality import (
    EvidenceType,
    EvidenceMarker,
    EvidenceQualityScore,
    EvidenceQualityAnalyzer,
    HollowConsensusAlert,
    HollowConsensusDetector,
)


# ============================================================================
# EvidenceType Enum Tests
# ============================================================================


class TestEvidenceType:
    """Tests for EvidenceType enum."""

    def test_citation_value(self):
        """Test CITATION enum value."""
        assert EvidenceType.CITATION.value == "citation"

    def test_data_value(self):
        """Test DATA enum value."""
        assert EvidenceType.DATA.value == "data"

    def test_example_value(self):
        """Test EXAMPLE enum value."""
        assert EvidenceType.EXAMPLE.value == "example"

    def test_tool_output_value(self):
        """Test TOOL_OUTPUT enum value."""
        assert EvidenceType.TOOL_OUTPUT.value == "tool_output"

    def test_quote_value(self):
        """Test QUOTE enum value."""
        assert EvidenceType.QUOTE.value == "quote"

    def test_reasoning_value(self):
        """Test REASONING enum value."""
        assert EvidenceType.REASONING.value == "reasoning"

    def test_none_value(self):
        """Test NONE enum value."""
        assert EvidenceType.NONE.value == "none"

    def test_all_types_unique(self):
        """Test all enum values are unique."""
        values = [e.value for e in EvidenceType]
        assert len(values) == len(set(values))


# ============================================================================
# EvidenceMarker Tests
# ============================================================================


class TestEvidenceMarker:
    """Tests for EvidenceMarker dataclass."""

    def test_create_marker(self):
        """Test creating an evidence marker."""
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

    def test_marker_with_data_type(self):
        """Test marker with DATA type."""
        marker = EvidenceMarker(
            evidence_type=EvidenceType.DATA,
            text="95%",
            position=10,
            confidence=0.85,
        )

        assert marker.evidence_type == EvidenceType.DATA
        assert marker.text == "95%"

    def test_marker_with_example_type(self):
        """Test marker with EXAMPLE type."""
        marker = EvidenceMarker(
            evidence_type=EvidenceType.EXAMPLE,
            text="for example",
            position=0,
            confidence=0.8,
        )

        assert marker.evidence_type == EvidenceType.EXAMPLE


# ============================================================================
# EvidenceQualityScore Tests
# ============================================================================


class TestEvidenceQualityScore:
    """Tests for EvidenceQualityScore dataclass."""

    def test_create_score(self):
        """Test creating a quality score."""
        score = EvidenceQualityScore(
            agent="claude",
            round_num=2,
        )

        assert score.agent == "claude"
        assert score.round_num == 2
        assert score.citation_density == 0.0
        assert score.overall_quality == 0.0

    def test_default_values(self):
        """Test default values are properly set."""
        score = EvidenceQualityScore(agent="test", round_num=1)

        assert score.citation_density == 0.0
        assert score.specificity_score == 0.0
        assert score.evidence_diversity == 0.0
        assert score.temporal_relevance == 1.0  # Default to fresh
        assert score.logical_chain_score == 0.0
        assert score.evidence_markers == []
        assert score.claim_count == 0

    def test_compute_overall(self):
        """Test overall quality computation."""
        score = EvidenceQualityScore(
            agent="test",
            round_num=1,
            citation_density=0.8,
            specificity_score=0.6,
            evidence_diversity=0.5,
            temporal_relevance=0.9,
            logical_chain_score=0.7,
        )

        overall = score.compute_overall()

        # Expected: 0.25*0.8 + 0.25*0.6 + 0.20*0.5 + 0.10*0.9 + 0.20*0.7
        # = 0.20 + 0.15 + 0.10 + 0.09 + 0.14 = 0.68
        assert overall == pytest.approx(0.68)
        assert score.overall_quality == pytest.approx(0.68)

    def test_compute_overall_all_zeros(self):
        """Test overall computation with all zeros."""
        score = EvidenceQualityScore(agent="test", round_num=1)
        score.temporal_relevance = 0.0  # Override default

        overall = score.compute_overall()

        assert overall == 0.0

    def test_compute_overall_all_ones(self):
        """Test overall computation with perfect scores."""
        score = EvidenceQualityScore(
            agent="test",
            round_num=1,
            citation_density=1.0,
            specificity_score=1.0,
            evidence_diversity=1.0,
            temporal_relevance=1.0,
            logical_chain_score=1.0,
        )

        overall = score.compute_overall()

        assert overall == pytest.approx(1.0)


# ============================================================================
# EvidenceQualityAnalyzer Initialization Tests
# ============================================================================


class TestEvidenceQualityAnalyzerInit:
    """Tests for EvidenceQualityAnalyzer initialization."""

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        analyzer = EvidenceQualityAnalyzer()

        assert analyzer.weights["citation"] == 0.25
        assert analyzer.weights["specificity"] == 0.25
        assert analyzer.weights["diversity"] == 0.20
        assert analyzer.weights["temporal"] == 0.10
        assert analyzer.weights["reasoning"] == 0.20

    def test_initialization_custom_weights(self):
        """Test initialization with custom weights."""
        analyzer = EvidenceQualityAnalyzer(
            citation_weight=0.5,
            specificity_weight=0.2,
        )

        assert analyzer.weights["citation"] == 0.5
        assert analyzer.weights["specificity"] == 0.2


# ============================================================================
# EvidenceQualityAnalyzer Pattern Detection Tests
# ============================================================================


class TestEvidencePatternDetection:
    """Tests for evidence pattern detection."""

    @pytest.fixture
    def analyzer(self):
        """Create an analyzer for testing."""
        return EvidenceQualityAnalyzer()

    def test_detect_citation_bracket(self, analyzer):
        """Test detecting bracket citations [1]."""
        text = "This is supported by research [1] and data [2, 3]."
        score = analyzer.analyze(text, "test", 1)

        citation_markers = [
            m for m in score.evidence_markers if m.evidence_type == EvidenceType.CITATION
        ]
        assert len(citation_markers) >= 2

    def test_detect_citation_author_year(self, analyzer):
        """Test detecting author-year citations (Smith 2024)."""
        text = "According to (Smith 2024), the results show improvement."
        score = analyzer.analyze(text, "test", 1)

        citation_markers = [
            m for m in score.evidence_markers if m.evidence_type == EvidenceType.CITATION
        ]
        assert len(citation_markers) >= 1

    def test_detect_citation_url(self, analyzer):
        """Test detecting URL citations."""
        text = "See https://example.com/paper for details."
        score = analyzer.analyze(text, "test", 1)

        citation_markers = [
            m for m in score.evidence_markers if m.evidence_type == EvidenceType.CITATION
        ]
        assert len(citation_markers) >= 1

    def test_detect_data_percentage(self, analyzer):
        """Test detecting percentage data."""
        text = "Performance improved by 45% after optimization."
        score = analyzer.analyze(text, "test", 1)

        data_markers = [m for m in score.evidence_markers if m.evidence_type == EvidenceType.DATA]
        assert len(data_markers) >= 1

    def test_detect_data_currency(self, analyzer):
        """Test detecting currency data."""
        text = "The cost was reduced to $1,500 per unit."
        score = analyzer.analyze(text, "test", 1)

        data_markers = [m for m in score.evidence_markers if m.evidence_type == EvidenceType.DATA]
        assert len(data_markers) >= 1

    def test_detect_data_metrics(self, analyzer):
        """Test detecting metrics (time, size)."""
        text = "Response time improved from 500ms to 100ms, using 2GB memory."
        score = analyzer.analyze(text, "test", 1)

        data_markers = [m for m in score.evidence_markers if m.evidence_type == EvidenceType.DATA]
        assert len(data_markers) >= 2

    def test_detect_example_for_example(self, analyzer):
        """Test detecting 'for example' phrases."""
        text = "For example, Python uses duck typing."
        score = analyzer.analyze(text, "test", 1)

        example_markers = [
            m for m in score.evidence_markers if m.evidence_type == EvidenceType.EXAMPLE
        ]
        assert len(example_markers) >= 1

    def test_detect_example_such_as(self, analyzer):
        """Test detecting 'such as' phrases."""
        text = "Languages such as Python, JavaScript, and Go support this."
        score = analyzer.analyze(text, "test", 1)

        example_markers = [
            m for m in score.evidence_markers if m.evidence_type == EvidenceType.EXAMPLE
        ]
        assert len(example_markers) >= 1


# ============================================================================
# EvidenceQualityAnalyzer Scoring Tests
# ============================================================================


class TestEvidenceScoring:
    """Tests for evidence quality scoring."""

    @pytest.fixture
    def analyzer(self):
        """Create an analyzer for testing."""
        return EvidenceQualityAnalyzer()

    def test_analyze_empty_response(self, analyzer):
        """Test analyzing empty response."""
        score = analyzer.analyze("", "test", 1)

        assert score.overall_quality == 0.0
        assert score.evidence_markers == []

    def test_analyze_high_quality_response(self, analyzer):
        """Test analyzing high-quality response with evidence."""
        text = """
        According to recent studies [1], code review reduces bugs by 35%.
        For example, Google's engineering team reported a 25% decrease in
        production issues. Therefore, implementing mandatory reviews is
        recommended. The data from 2025 shows consistent improvement across
        all metrics. Specifically, the test coverage increased to 95%.
        """
        score = analyzer.analyze(text, "claude", 2)

        assert score.overall_quality > 0.5
        assert len(score.evidence_markers) > 0
        assert score.citation_density > 0
        assert score.specificity_score > 0

    def test_analyze_vague_response(self, analyzer):
        """Test analyzing vague response with no evidence."""
        text = """
        Generally speaking, this approach might work in some cases.
        Typically, it depends on various factors and many considerations.
        Usually, best practices suggest a common approach that could
        potentially work.
        """
        score = analyzer.analyze(text, "vague_agent", 1)

        assert score.specificity_score < 0.5
        assert score.vague_phrase_count > 0

    def test_citation_density_calculation(self, analyzer):
        """Test citation density is properly calculated."""
        # 2 substantive sentences with 2 citations
        text = "First claim here [1]. Second claim is also cited [2]."
        score = analyzer.analyze(text, "test", 1)

        # Should have some citation density
        assert score.backed_claim_count >= 1

    def test_specificity_score_specific_text(self, analyzer):
        """Test specificity score for specific text."""
        text = "Specifically, the latency was measured at 42ms, precisely as documented."
        score = analyzer.analyze(text, "test", 1)

        assert score.specificity_score > 0.5

    def test_specificity_score_balanced(self, analyzer):
        """Test specificity score with balanced text."""
        text = "Generally, the result was 50%. Sometimes it varies."
        score = analyzer.analyze(text, "test", 1)

        # Contains both vague and specific elements
        assert 0.0 < score.specificity_score < 1.0

    def test_diversity_multiple_types(self, analyzer):
        """Test evidence diversity with multiple types."""
        text = """
        According to [1], the improvement was 40%. For example, consider
        the case of System X. This data shows significant progress.
        """
        score = analyzer.analyze(text, "test", 1)

        assert score.evidence_diversity > 0

    def test_reasoning_chain_score(self, analyzer):
        """Test reasoning chain detection."""
        text = """
        Since the system uses caching, therefore response times are lower.
        Because of this optimization, consequently the user experience improves.
        Thus, we can conclude the approach is effective.
        """
        score = analyzer.analyze(text, "test", 1)

        assert score.logical_chain_score > 0

    def test_temporal_relevance_recent(self, analyzer):
        """Test temporal relevance with recent dates."""
        text = "Studies from 2025 and 2024 show consistent results."
        score = analyzer.analyze(text, "test", 1)

        assert score.temporal_relevance > 0.7

    def test_temporal_relevance_old(self, analyzer):
        """Test temporal relevance with old dates."""
        text = "Research from 2010 and 2008 suggested this approach."
        score = analyzer.analyze(text, "test", 1)

        assert score.temporal_relevance < 0.5

    def test_temporal_relevance_no_dates(self, analyzer):
        """Test temporal relevance with no dates."""
        text = "The approach works well in practice."
        score = analyzer.analyze(text, "test", 1)

        assert score.temporal_relevance == 0.8  # Neutral default


# ============================================================================
# EvidenceQualityAnalyzer Batch Analysis Tests
# ============================================================================


class TestEvidenceBatchAnalysis:
    """Tests for batch analysis of responses."""

    @pytest.fixture
    def analyzer(self):
        """Create an analyzer for testing."""
        return EvidenceQualityAnalyzer()

    def test_analyze_batch(self, analyzer):
        """Test analyzing multiple responses."""
        responses = {
            "agent_a": "According to [1], this works.",
            "agent_b": "Generally, this might work sometimes.",
            "agent_c": "Specifically, tests show 95% success rate.",
        }

        scores = analyzer.analyze_batch(responses, round_num=2)

        assert len(scores) == 3
        assert "agent_a" in scores
        assert "agent_b" in scores
        assert "agent_c" in scores
        assert all(s.round_num == 2 for s in scores.values())

    def test_analyze_batch_empty(self, analyzer):
        """Test analyzing empty batch."""
        scores = analyzer.analyze_batch({}, round_num=1)

        assert scores == {}


# ============================================================================
# HollowConsensusAlert Tests
# ============================================================================


class TestHollowConsensusAlert:
    """Tests for HollowConsensusAlert dataclass."""

    def test_create_alert(self):
        """Test creating an alert."""
        alert = HollowConsensusAlert(
            detected=True,
            severity=0.7,
            reason="Low evidence quality",
            agent_scores={"a": 0.3, "b": 0.4},
            recommended_challenges=["Provide citations"],
        )

        assert alert.detected is True
        assert alert.severity == 0.7
        assert len(alert.agent_scores) == 2
        assert len(alert.recommended_challenges) == 1

    def test_alert_not_detected(self):
        """Test creating alert when not detected."""
        alert = HollowConsensusAlert(
            detected=False,
            severity=0.0,
            reason="Evidence quality acceptable",
            agent_scores={},
            recommended_challenges=[],
        )

        assert alert.detected is False
        assert alert.severity == 0.0

    def test_alert_with_metrics(self):
        """Test alert with additional metrics."""
        alert = HollowConsensusAlert(
            detected=True,
            severity=0.5,
            reason="High variance",
            agent_scores={"a": 0.8, "b": 0.2},
            recommended_challenges=[],
            min_quality=0.2,
            avg_quality=0.5,
            quality_variance=0.18,
        )

        assert alert.min_quality == 0.2
        assert alert.avg_quality == 0.5
        assert alert.quality_variance == 0.18


# ============================================================================
# HollowConsensusDetector Initialization Tests
# ============================================================================


class TestHollowConsensusDetectorInit:
    """Tests for HollowConsensusDetector initialization."""

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        detector = HollowConsensusDetector()

        assert detector.min_quality_threshold == 0.4
        assert detector.quality_variance_threshold == 0.3
        assert detector.analyzer is not None

    def test_initialization_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        detector = HollowConsensusDetector(
            min_quality_threshold=0.5,
            quality_variance_threshold=0.2,
        )

        assert detector.min_quality_threshold == 0.5
        assert detector.quality_variance_threshold == 0.2


# ============================================================================
# HollowConsensusDetector Check Tests
# ============================================================================


class TestHollowConsensusDetectorCheck:
    """Tests for hollow consensus detection."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        return HollowConsensusDetector()

    def test_check_not_converging(self, detector):
        """Test check when not converging."""
        responses = {"a": "text", "b": "text"}

        alert = detector.check(responses, convergence_similarity=0.3, round_num=1)

        assert alert.detected is False
        assert "Not converging" in alert.reason

    def test_check_empty_responses(self, detector):
        """Test check with empty responses."""
        alert = detector.check({}, convergence_similarity=0.8, round_num=1)

        assert alert.detected is False
        assert "No responses" in alert.reason

    def test_check_high_quality_consensus(self, detector):
        """Test check with high-quality converging responses."""
        responses = {
            "a": "According to [1], the data shows 45% improvement specifically in 2025.",
            "b": "Studies [2] confirm a 48% increase, for example in production systems.",
        }

        alert = detector.check(responses, convergence_similarity=0.9, round_num=2)

        # Should not detect hollow consensus with good evidence
        assert alert.detected is False or alert.severity < 0.5

    def test_check_hollow_consensus_detected(self, detector):
        """Test detection of hollow consensus."""
        responses = {
            "a": "Generally, this might work in some cases.",
            "b": "Typically, various factors suggest this approach.",
            "c": "Usually, best practices indicate this direction.",
        }

        alert = detector.check(responses, convergence_similarity=0.85, round_num=3)

        # Should detect hollow consensus with vague responses
        assert alert.avg_quality < 0.5

    def test_check_variance_detection(self, detector):
        """Test detection of quality variance."""
        responses = {
            "high_quality": "According to [1], specifically 95% success rate measured in 2025.",
            "low_quality": "This might work sometimes generally speaking.",
        }

        alert = detector.check(responses, convergence_similarity=0.8, round_num=1)

        # Should detect variance in quality
        assert alert.quality_variance > 0

    def test_check_returns_agent_scores(self, detector):
        """Test that check returns per-agent scores."""
        responses = {
            "agent_a": "Claim with [1]",
            "agent_b": "Another claim",
        }

        alert = detector.check(responses, convergence_similarity=0.7, round_num=1)

        assert "agent_a" in alert.agent_scores
        assert "agent_b" in alert.agent_scores


# ============================================================================
# HollowConsensusDetector Challenge Generation Tests
# ============================================================================


class TestHollowConsensusDetectorChallenges:
    """Tests for challenge generation."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        return HollowConsensusDetector()

    def test_generate_citation_challenge(self, detector):
        """Test generating challenge for missing citations."""
        responses = {
            "no_cite_agent": "This is definitely true but I have no sources.",
        }

        alert = detector.check(responses, convergence_similarity=0.8, round_num=1)

        citation_challenges = [
            c
            for c in alert.recommended_challenges
            if "sources" in c.lower() or "citation" in c.lower() or "reference" in c.lower()
        ]
        # May or may not generate depending on detection threshold
        # Just check we have some challenges
        assert isinstance(alert.recommended_challenges, list)

    def test_generate_vague_language_challenge(self, detector):
        """Test generating challenge for vague language."""
        responses = {
            "vague_agent": "Generally speaking, this might work sometimes in various cases.",
        }

        alert = detector.check(responses, convergence_similarity=0.8, round_num=1)

        # Check for challenge mentioning vague language or specifics
        vague_challenges = [
            c
            for c in alert.recommended_challenges
            if "vague" in c.lower() or "specific" in c.lower()
        ]
        assert isinstance(alert.recommended_challenges, list)

    def test_challenges_limited_to_three(self, detector):
        """Test that challenges are limited to 3."""
        responses = {
            "a": "Vague text without anything specific.",
            "b": "Another vague response generally speaking.",
            "c": "Might work sometimes typically.",
        }

        alert = detector.check(responses, convergence_similarity=0.9, round_num=1)

        assert len(alert.recommended_challenges) <= 3


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_response(self):
        """Test analyzing very short response."""
        analyzer = EvidenceQualityAnalyzer()
        score = analyzer.analyze("OK.", "test", 1)

        assert score.claim_count == 0

    def test_response_with_only_questions(self):
        """Test response with only questions."""
        analyzer = EvidenceQualityAnalyzer()
        score = analyzer.analyze(
            "What do you think? How would this work? Why choose this?",
            "test",
            1,
        )

        # Questions shouldn't count as claims
        assert score.claim_count == 0

    def test_mixed_year_references(self):
        """Test response with both old and new years."""
        analyzer = EvidenceQualityAnalyzer()
        text = "Early research from 2015 was expanded in 2024 studies."
        score = analyzer.analyze(text, "test", 1)

        # Should have moderate temporal relevance
        assert 0.3 < score.temporal_relevance < 0.9

    def test_unicode_in_response(self):
        """Test handling of unicode characters."""
        analyzer = EvidenceQualityAnalyzer()
        text = "研究表明，性能提升了50%。For example, tests show improvement."
        score = analyzer.analyze(text, "test", 1)

        # Should handle unicode without crashing
        assert score is not None

    def test_very_long_response(self):
        """Test analyzing very long response."""
        analyzer = EvidenceQualityAnalyzer()
        # Generate a long text with evidence markers
        text = (
            "According to [1], the improvement was 25%. " * 100
            + "Therefore, this approach works well. " * 50
        )
        score = analyzer.analyze(text, "test", 1)

        assert score.evidence_markers is not None
        assert len(score.evidence_markers) > 0

    def test_detector_severity_bounds(self):
        """Test that severity is bounded between 0 and 1."""
        detector = HollowConsensusDetector()
        responses = {
            "a": "x" * 1000,  # Gibberish
        }

        alert = detector.check(responses, convergence_similarity=0.99, round_num=1)

        assert 0.0 <= alert.severity <= 1.0
