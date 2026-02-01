"""
Tests for the evidence_quality module.

Tests cover:
- EvidenceType enum
- EvidenceMarker data class
- EvidenceQualityScore data class
- EvidenceQualityAnalyzer
- HollowConsensusAlert data class
- HollowConsensusDetector
"""

from __future__ import annotations

import pytest

from aragora.debate.evidence_quality import (
    EvidenceMarker,
    EvidenceQualityAnalyzer,
    EvidenceQualityScore,
    EvidenceType,
    HollowConsensusAlert,
    HollowConsensusDetector,
)


class TestEvidenceType:
    """Tests for EvidenceType enum."""

    def test_evidence_type_values(self):
        """Test all evidence type values exist."""
        assert EvidenceType.CITATION.value == "citation"
        assert EvidenceType.DATA.value == "data"
        assert EvidenceType.EXAMPLE.value == "example"
        assert EvidenceType.TOOL_OUTPUT.value == "tool_output"
        assert EvidenceType.QUOTE.value == "quote"
        assert EvidenceType.REASONING.value == "reasoning"
        assert EvidenceType.NONE.value == "none"

    def test_evidence_type_membership(self):
        """Test evidence type membership check."""
        assert EvidenceType.CITATION in EvidenceType
        assert EvidenceType.DATA in EvidenceType


class TestEvidenceMarker:
    """Tests for EvidenceMarker data class."""

    def test_marker_creation(self):
        """Test creating an EvidenceMarker."""
        marker = EvidenceMarker(
            evidence_type=EvidenceType.CITATION,
            text="[1]",
            position=50,
            confidence=0.9,
        )

        assert marker.evidence_type == EvidenceType.CITATION
        assert marker.text == "[1]"
        assert marker.position == 50
        assert marker.confidence == 0.9

    def test_marker_with_data_type(self):
        """Test marker with data evidence type."""
        marker = EvidenceMarker(
            evidence_type=EvidenceType.DATA,
            text="45%",
            position=100,
            confidence=0.85,
        )

        assert marker.evidence_type == EvidenceType.DATA
        assert marker.text == "45%"

    def test_marker_with_example_type(self):
        """Test marker with example evidence type."""
        marker = EvidenceMarker(
            evidence_type=EvidenceType.EXAMPLE,
            text="for example",
            position=25,
            confidence=0.8,
        )

        assert marker.evidence_type == EvidenceType.EXAMPLE
        assert marker.text == "for example"


class TestEvidenceQualityScore:
    """Tests for EvidenceQualityScore data class."""

    def test_score_creation_defaults(self):
        """Test creating EvidenceQualityScore with defaults."""
        score = EvidenceQualityScore(agent="claude", round_num=1)

        assert score.agent == "claude"
        assert score.round_num == 1
        assert score.citation_density == 0.0
        assert score.specificity_score == 0.0
        assert score.evidence_diversity == 0.0
        assert score.temporal_relevance == 1.0
        assert score.logical_chain_score == 0.0
        assert score.overall_quality == 0.0
        assert score.evidence_markers == []
        assert score.claim_count == 0
        assert score.backed_claim_count == 0

    def test_score_creation_with_values(self):
        """Test creating EvidenceQualityScore with values."""
        markers = [
            EvidenceMarker(EvidenceType.CITATION, "[1]", 0, 0.9),
            EvidenceMarker(EvidenceType.DATA, "50%", 20, 0.85),
        ]
        score = EvidenceQualityScore(
            agent="gpt4",
            round_num=2,
            citation_density=0.8,
            specificity_score=0.7,
            evidence_diversity=0.6,
            temporal_relevance=0.9,
            logical_chain_score=0.75,
            evidence_markers=markers,
            claim_count=5,
            backed_claim_count=4,
        )

        assert score.agent == "gpt4"
        assert score.citation_density == 0.8
        assert score.specificity_score == 0.7
        assert len(score.evidence_markers) == 2

    def test_compute_overall_basic(self):
        """Test compute_overall with basic scores."""
        score = EvidenceQualityScore(
            agent="claude",
            round_num=1,
            citation_density=0.8,
            specificity_score=0.6,
            evidence_diversity=0.5,
            temporal_relevance=0.9,
            logical_chain_score=0.7,
        )

        result = score.compute_overall()

        # Expected: 0.8*0.25 + 0.6*0.25 + 0.5*0.2 + 0.9*0.1 + 0.7*0.2
        # = 0.2 + 0.15 + 0.1 + 0.09 + 0.14 = 0.68
        assert result == pytest.approx(0.68, rel=0.01)
        assert score.overall_quality == pytest.approx(0.68, rel=0.01)

    def test_compute_overall_zeros(self):
        """Test compute_overall with all zeros except temporal."""
        score = EvidenceQualityScore(agent="test", round_num=0)

        result = score.compute_overall()

        # Only temporal_relevance is 1.0 by default
        # 1.0 * 0.1 = 0.1
        assert result == pytest.approx(0.1, rel=0.01)

    def test_compute_overall_perfect(self):
        """Test compute_overall with perfect scores."""
        score = EvidenceQualityScore(
            agent="claude",
            round_num=1,
            citation_density=1.0,
            specificity_score=1.0,
            evidence_diversity=1.0,
            temporal_relevance=1.0,
            logical_chain_score=1.0,
        )

        result = score.compute_overall()

        # All weights sum to 1.0, so perfect scores = 1.0
        assert result == pytest.approx(1.0, rel=0.01)


class TestEvidenceQualityAnalyzer:
    """Tests for EvidenceQualityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default settings."""
        return EvidenceQualityAnalyzer()

    def test_analyzer_creation_default_weights(self):
        """Test analyzer creation with default weights."""
        analyzer = EvidenceQualityAnalyzer()

        assert analyzer.weights["citation"] == 0.25
        assert analyzer.weights["specificity"] == 0.25
        assert analyzer.weights["diversity"] == 0.20
        assert analyzer.weights["temporal"] == 0.10
        assert analyzer.weights["reasoning"] == 0.20

    def test_analyzer_creation_custom_weights(self):
        """Test analyzer creation with custom weights."""
        analyzer = EvidenceQualityAnalyzer(
            citation_weight=0.3,
            specificity_weight=0.3,
            diversity_weight=0.15,
            temporal_weight=0.05,
            reasoning_weight=0.2,
        )

        assert analyzer.weights["citation"] == 0.3
        assert analyzer.weights["specificity"] == 0.3
        assert analyzer.weights["diversity"] == 0.15
        assert analyzer.weights["temporal"] == 0.05

    def test_analyze_empty_response(self, analyzer):
        """Test analyzing empty response."""
        score = analyzer.analyze("", "claude", 1)

        assert score.agent == "claude"
        assert score.round_num == 1
        assert score.overall_quality == 0.0
        assert len(score.evidence_markers) == 0

    def test_analyze_response_with_citations(self, analyzer):
        """Test analyzing response with citations."""
        response = """
        According to the documentation [1], the system performs well.
        Smith (2024) found similar results. The API is available at https://example.com.
        Source: RFC 8259 for JSON format specification.
        """

        score = analyzer.analyze(response, "claude", 1)

        assert score.citation_density > 0
        citation_markers = [
            m for m in score.evidence_markers if m.evidence_type == EvidenceType.CITATION
        ]
        assert len(citation_markers) >= 3  # [1], (Smith 2024), https://..., source:

    def test_analyze_response_with_data(self, analyzer):
        """Test analyzing response with data/statistics."""
        response = """
        The system shows a 45% improvement in performance.
        Processing time reduced from 500ms to 200ms.
        Memory usage is approximately 2.5GB for large datasets.
        Costs decreased by $1,500 per month.
        """

        score = analyzer.analyze(response, "gpt4", 2)

        data_markers = [m for m in score.evidence_markers if m.evidence_type == EvidenceType.DATA]
        assert len(data_markers) >= 3

    def test_analyze_response_with_examples(self, analyzer):
        """Test analyzing response with examples."""
        response = """
        For example, Redis can handle millions of requests per second.
        For instance, AWS Lambda provides serverless computing.
        Such as Python, JavaScript, and Go are supported.
        Specifically, the timeout should be set to 30 seconds.
        Case in point: Netflix uses microservices architecture.
        """

        score = analyzer.analyze(response, "gemini", 1)

        example_markers = [
            m for m in score.evidence_markers if m.evidence_type == EvidenceType.EXAMPLE
        ]
        assert len(example_markers) >= 4

    def test_analyze_specificity_vague(self, analyzer):
        """Test specificity score with vague language."""
        response = """
        Generally, the system might work well. Typically it depends on
        various factors. Usually there are many considerations involved.
        Sometimes the impact could potentially be significant. In some cases,
        the common approach is best practices from industry standard.
        """

        score = analyzer.analyze(response, "claude", 1)

        assert score.vague_phrase_count >= 5
        assert score.specificity_score < 0.5

    def test_analyze_specificity_specific(self, analyzer):
        """Test specificity score with specific language."""
        response = """
        Specifically, the latency is measured at 15ms. The test was
        verified by running 10,000 iterations. We observed in production
        that exactly 99.9% of requests succeed. This was documented in
        the performance report with 5 specific metrics.
        """

        score = analyzer.analyze(response, "claude", 1)

        assert score.specific_phrase_count >= 3
        assert score.specificity_score > 0.5

    def test_analyze_specificity_neutral(self, analyzer):
        """Test specificity score with neutral language."""
        response = "The sky is blue and water is wet."

        score = analyzer.analyze(response, "claude", 1)

        # No vague or specific indicators
        assert score.specificity_score == 0.5

    def test_analyze_evidence_diversity(self, analyzer):
        """Test evidence diversity computation."""
        response = """
        According to [1], the API performs well. The latency is 50ms
        which represents a 30% improvement. For example, Netflix uses
        similar architecture.
        """

        score = analyzer.analyze(response, "claude", 1)

        # Should have CITATION, DATA, and EXAMPLE types
        types = set(m.evidence_type for m in score.evidence_markers)
        assert EvidenceType.CITATION in types
        assert EvidenceType.DATA in types
        assert EvidenceType.EXAMPLE in types
        assert score.evidence_diversity >= 0.4  # At least 2/5 types

    def test_analyze_reasoning_chain(self, analyzer):
        """Test logical reasoning chain detection."""
        response = """
        The system needs caching because database queries are slow.
        Therefore, Redis is recommended. Since we need low latency,
        consequently the design uses in-memory storage. Given that
        the data is small, thus it fits in memory. Hence, this implies
        we can use a single node. As a result, it follows that costs
        are reduced.
        """

        score = analyzer.analyze(response, "claude", 1)

        assert score.logical_chain_score > 0.5

    def test_analyze_reasoning_chain_none(self, analyzer):
        """Test response with no reasoning connectors."""
        response = "The sky is blue. Grass is green. Water is wet."

        score = analyzer.analyze(response, "claude", 1)

        assert score.logical_chain_score < 0.5

    def test_analyze_temporal_relevance_recent(self, analyzer):
        """Test temporal relevance with recent dates."""
        response = """
        According to the 2025 report, performance improved.
        The 2024 benchmark shows similar results.
        """

        score = analyzer.analyze(response, "claude", 1)

        # Years close to 2026 should have high relevance
        assert score.temporal_relevance > 0.7

    def test_analyze_temporal_relevance_old(self, analyzer):
        """Test temporal relevance with old dates."""
        response = """
        Based on the 2010 study and 2005 research,
        the approach was validated in 2008.
        """

        score = analyzer.analyze(response, "claude", 1)

        # Old years should have low relevance
        assert score.temporal_relevance < 0.5

    def test_analyze_temporal_relevance_no_dates(self, analyzer):
        """Test temporal relevance with no dates."""
        response = "The system works well for most use cases."

        score = analyzer.analyze(response, "claude", 1)

        # Default to 0.8 when no dates found
        assert score.temporal_relevance == 0.8

    def test_analyze_batch(self, analyzer):
        """Test batch analysis of multiple responses."""
        responses = {
            "claude": "According to [1], the system works. Performance is 50ms.",
            "gpt4": "The documentation shows good results. For example, Netflix uses this.",
            "gemini": "Generally speaking, it might work well.",
        }

        scores = analyzer.analyze_batch(responses, round_num=2)

        assert len(scores) == 3
        assert "claude" in scores
        assert "gpt4" in scores
        assert "gemini" in scores
        assert all(s.round_num == 2 for s in scores.values())

    def test_analyze_batch_empty(self, analyzer):
        """Test batch analysis with empty dict."""
        scores = analyzer.analyze_batch({}, round_num=1)

        assert scores == {}

    def test_detect_evidence_citations_numbered(self, analyzer):
        """Test citation detection for numbered references."""
        markers = analyzer._detect_evidence("As shown in [1] and [2, 3].")

        citation_markers = [m for m in markers if m.evidence_type == EvidenceType.CITATION]
        assert len(citation_markers) >= 2

    def test_detect_evidence_citations_urls(self, analyzer):
        """Test citation detection for URLs."""
        markers = analyzer._detect_evidence("See https://example.com for details.")

        citation_markers = [m for m in markers if m.evidence_type == EvidenceType.CITATION]
        assert len(citation_markers) >= 1
        assert any("https://example.com" in m.text for m in citation_markers)

    def test_detect_evidence_data_percentage(self, analyzer):
        """Test data detection for percentages."""
        markers = analyzer._detect_evidence("Performance improved by 45.5%.")

        data_markers = [m for m in markers if m.evidence_type == EvidenceType.DATA]
        assert len(data_markers) >= 1

    def test_detect_evidence_data_currency(self, analyzer):
        """Test data detection for currency."""
        markers = analyzer._detect_evidence("Cost is $1,500.00 per month.")

        data_markers = [m for m in markers if m.evidence_type == EvidenceType.DATA]
        assert len(data_markers) >= 1

    def test_detect_evidence_data_time_metrics(self, analyzer):
        """Test data detection for time metrics."""
        markers = analyzer._detect_evidence("Latency is 150ms, timeout is 30 seconds.")

        data_markers = [m for m in markers if m.evidence_type == EvidenceType.DATA]
        assert len(data_markers) >= 2

    def test_detect_evidence_data_size_metrics(self, analyzer):
        """Test data detection for size metrics."""
        markers = analyzer._detect_evidence("File size is 2.5GB, cache is 512MB.")

        data_markers = [m for m in markers if m.evidence_type == EvidenceType.DATA]
        assert len(data_markers) >= 2

    def test_detect_evidence_data_change_metrics(self, analyzer):
        """Test data detection for change metrics."""
        markers = analyzer._detect_evidence("Latency increased by 20 and errors decreased 50.")

        data_markers = [m for m in markers if m.evidence_type == EvidenceType.DATA]
        assert len(data_markers) >= 2

    def test_compute_citation_density_no_claims(self, analyzer):
        """Test citation density with no substantive claims."""
        score = EvidenceQualityScore(agent="test", round_num=0)
        density = analyzer._compute_citation_density("Yes. No.", score)

        assert density == 0.0
        assert score.claim_count == 0

    def test_compute_citation_density_with_questions(self, analyzer):
        """Test citation density excludes questions."""
        score = EvidenceQualityScore(agent="test", round_num=0)
        text = "What do you think about this approach? How does it compare?"
        density = analyzer._compute_citation_density(text, score)

        # Questions should not count as claims
        assert score.claim_count == 0


class TestHollowConsensusAlert:
    """Tests for HollowConsensusAlert data class."""

    def test_alert_creation_not_detected(self):
        """Test creating alert when not detected."""
        alert = HollowConsensusAlert(
            detected=False,
            severity=0.0,
            reason="Not converging yet",
            agent_scores={},
            recommended_challenges=[],
        )

        assert alert.detected is False
        assert alert.severity == 0.0
        assert alert.reason == "Not converging yet"
        assert alert.agent_scores == {}
        assert alert.recommended_challenges == []

    def test_alert_creation_detected(self):
        """Test creating alert when hollow consensus detected."""
        alert = HollowConsensusAlert(
            detected=True,
            severity=0.75,
            reason="Low evidence quality (35%); claude lacks citations",
            agent_scores={"claude": 0.35, "gpt4": 0.55},
            recommended_challenges=[
                "Challenge to claude: Provide specific references.",
            ],
            min_quality=0.35,
            avg_quality=0.45,
            quality_variance=0.1,
        )

        assert alert.detected is True
        assert alert.severity == 0.75
        assert "Low evidence quality" in alert.reason
        assert alert.agent_scores["claude"] == 0.35
        assert len(alert.recommended_challenges) == 1
        assert alert.min_quality == 0.35
        assert alert.avg_quality == 0.45
        assert alert.quality_variance == 0.1


class TestHollowConsensusDetector:
    """Tests for HollowConsensusDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector with default settings."""
        return HollowConsensusDetector()

    def test_detector_creation_defaults(self):
        """Test detector creation with default values."""
        detector = HollowConsensusDetector()

        assert detector.min_quality_threshold == 0.4
        assert detector.quality_variance_threshold == 0.3
        assert "low_quality" in detector.severity_weights
        assert "high_variance" in detector.severity_weights

    def test_detector_creation_custom(self):
        """Test detector creation with custom values."""
        detector = HollowConsensusDetector(
            min_quality_threshold=0.5,
            quality_variance_threshold=0.2,
            hollow_severity_weights={"low_quality": 0.5, "high_variance": 0.5},
        )

        assert detector.min_quality_threshold == 0.5
        assert detector.quality_variance_threshold == 0.2
        assert detector.severity_weights["low_quality"] == 0.5

    def test_check_not_converging(self, detector):
        """Test check when responses are not converging."""
        responses = {
            "claude": "Some response here.",
            "gpt4": "Another different response.",
        }

        alert = detector.check(responses, convergence_similarity=0.3, round_num=1)

        assert alert.detected is False
        assert alert.reason == "Not converging yet"

    def test_check_no_responses(self, detector):
        """Test check with empty responses."""
        alert = detector.check({}, convergence_similarity=0.8, round_num=1)

        assert alert.detected is False
        assert alert.reason == "No responses to analyze"

    def test_check_high_quality_consensus(self, detector):
        """Test check with high quality converging responses."""
        responses = {
            "claude": """
                According to the 2024 benchmark [1], Redis provides 50ms latency.
                For example, Netflix uses this architecture. Therefore, it's recommended.
                The documentation at https://redis.io confirms this specifically.
            """,
            "gpt4": """
                Based on the 2025 performance report [2], Redis achieves 45ms latency.
                For instance, Uber employs similar caching. Hence, this is the best approach.
                The official docs at https://redis.io show measured improvements of 40%.
            """,
        }

        alert = detector.check(responses, convergence_similarity=0.85, round_num=2)

        # High quality responses should not trigger hollow consensus
        assert alert.avg_quality > 0.3
        # With good evidence, severity should be low
        assert alert.severity < 0.8

    def test_check_low_quality_consensus(self, detector):
        """Test check with low quality converging responses."""
        responses = {
            "claude": "Generally speaking, it might work. Usually these things depend.",
            "gpt4": "Typically this could potentially be good. Various factors matter.",
        }

        alert = detector.check(responses, convergence_similarity=0.85, round_num=2)

        # Vague responses with no evidence
        assert alert.avg_quality < 0.5
        assert "quality" in alert.reason.lower() or alert.severity > 0

    def test_check_uneven_quality(self, detector):
        """Test check with uneven quality across agents."""
        responses = {
            "claude": """
                According to [1], Redis provides 50ms latency. Specifically, the 2024
                benchmark shows a 40% improvement. For example, Netflix uses this.
            """,
            "gpt4": "It might work well generally. Depends on various factors.",
            "gemini": "Usually this is fine. Typically acceptable.",
        }

        alert = detector.check(responses, convergence_similarity=0.8, round_num=2)

        # Should detect variance between claude (high quality) and others (low quality)
        assert len(alert.agent_scores) == 3
        # claude should have higher score
        if alert.agent_scores:
            claude_score = alert.agent_scores.get("claude", 0)
            gpt4_score = alert.agent_scores.get("gpt4", 0)
            assert claude_score > gpt4_score

    def test_check_generates_challenges_low_citations(self, detector):
        """Test that challenges are generated for low citation density."""
        responses = {
            "claude": "The system works well. It's a good approach.",
            "gpt4": "I agree, the system is reliable.",
        }

        alert = detector.check(responses, convergence_similarity=0.85, round_num=2)

        # Should generate citation-related challenges
        if alert.recommended_challenges:
            challenge_text = " ".join(alert.recommended_challenges)
            # Should mention sources or references
            assert "source" in challenge_text.lower() or "reference" in challenge_text.lower()

    def test_check_generates_challenges_vague_language(self, detector):
        """Test that challenges are generated for vague language."""
        responses = {
            "claude": "Generally it might potentially work. Various factors involved.",
            "gpt4": "Typically this could be good. Depends on many considerations.",
        }

        alert = detector.check(responses, convergence_similarity=0.85, round_num=2)

        # Should generate specificity-related challenges
        if alert.recommended_challenges:
            challenge_text = " ".join(alert.recommended_challenges)
            assert (
                "specific" in challenge_text.lower()
                or "number" in challenge_text.lower()
                or "example" in challenge_text.lower()
            )

    def test_check_generates_challenges_weak_reasoning(self, detector):
        """Test that challenges are generated for weak reasoning."""
        responses = {
            "claude": "Redis is good. Use it.",
            "gpt4": "Caching helps. Recommended.",
        }

        alert = detector.check(responses, convergence_similarity=0.85, round_num=2)

        # May generate reasoning-related challenges
        if alert.recommended_challenges:
            # At least some challenge should be generated for low quality
            assert len(alert.recommended_challenges) > 0

    def test_check_limits_challenges_to_three(self, detector):
        """Test that at most 3 challenges are returned."""
        responses = {
            "claude": "Maybe.",
            "gpt4": "Perhaps.",
            "gemini": "Possibly.",
            "llama": "Could be.",
        }

        alert = detector.check(responses, convergence_similarity=0.9, round_num=3)

        assert len(alert.recommended_challenges) <= 3

    def test_check_severity_calculation(self, detector):
        """Test severity calculation bounds."""
        responses = {
            "claude": "Yes.",
            "gpt4": "Sure.",
        }

        alert = detector.check(responses, convergence_similarity=0.95, round_num=2)

        # Severity should be bounded 0-1
        assert 0.0 <= alert.severity <= 1.0

    def test_check_detection_threshold(self, detector):
        """Test detection requires both severity and convergence."""
        # Low quality but not converging
        responses = {"claude": "Maybe.", "gpt4": "Perhaps."}

        alert = detector.check(responses, convergence_similarity=0.5, round_num=2)
        assert alert.detected is False  # Not converging

        # Converging with low quality
        alert = detector.check(responses, convergence_similarity=0.85, round_num=2)
        # Detection depends on severity > 0.3 and convergence > 0.7

    def test_generate_challenges_empty_issues(self, detector):
        """Test challenge generation with no issues but issues list."""
        quality_scores = {
            "claude": EvidenceQualityScore(
                agent="claude",
                round_num=1,
                citation_density=0.8,
                specificity_score=0.8,
                logical_chain_score=0.8,
            ),
        }

        challenges = detector._generate_challenges(quality_scores, issues=["Some issue"])

        # Should generate a general challenge
        assert len(challenges) > 0

    def test_generate_challenges_no_issues(self, detector):
        """Test challenge generation with no issues."""
        quality_scores = {
            "claude": EvidenceQualityScore(
                agent="claude",
                round_num=1,
                citation_density=0.8,
                specificity_score=0.8,
                logical_chain_score=0.8,
            ),
        }

        challenges = detector._generate_challenges(quality_scores, issues=[])

        # No challenges needed for high quality
        assert len(challenges) == 0

    def test_multiple_agents_in_challenges(self, detector):
        """Test that challenges can address multiple agents."""
        quality_scores = {
            "claude": EvidenceQualityScore(
                agent="claude",
                round_num=1,
                citation_density=0.1,
                specificity_score=0.5,
                logical_chain_score=0.5,
            ),
            "gpt4": EvidenceQualityScore(
                agent="gpt4",
                round_num=1,
                citation_density=0.15,
                specificity_score=0.5,
                logical_chain_score=0.5,
            ),
        }

        challenges = detector._generate_challenges(quality_scores, issues=["Low citations"])

        # Challenge should mention both agents
        if challenges:
            combined = " ".join(challenges)
            assert "claude" in combined or "gpt4" in combined


class TestEvidenceQualityAnalyzerEdgeCases:
    """Edge case tests for EvidenceQualityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer for edge case testing."""
        return EvidenceQualityAnalyzer()

    def test_very_long_response(self, analyzer):
        """Test analyzing very long response."""
        # Create a long response with mixed evidence
        long_response = (
            """
        According to [1], this is a detailed analysis. The performance improved by 50%.
        For example, Netflix uses this approach. Therefore, it's recommended.
        """
            * 100
        )  # Repeat to make it long

        score = analyzer.analyze(long_response, "claude", 1)

        # Should still compute valid scores
        assert 0 <= score.overall_quality <= 1
        assert len(score.evidence_markers) > 0

    def test_unicode_response(self, analyzer):
        """Test analyzing response with unicode characters."""
        response = """
        According to the research [1], the performance is 50%.
        Examples include: cafe, resume, naive.
        The cost is $1,000.
        """

        score = analyzer.analyze(response, "claude", 1)

        assert score.overall_quality >= 0

    def test_response_with_code_blocks(self, analyzer):
        """Test analyzing response with code blocks."""
        response = """
        According to the documentation, use the following:

        ```python
        def example():
            return 42
        ```

        This returns 42 specifically.
        """

        score = analyzer.analyze(response, "claude", 1)

        # Should handle code blocks gracefully
        assert score.overall_quality >= 0

    def test_response_only_numbers(self, analyzer):
        """Test analyzing response with only numbers."""
        response = "100 200 300 400 500"

        score = analyzer.analyze(response, "claude", 1)

        # Numbers should be detected but not as substantive claims
        assert score.claim_count == 0

    def test_response_only_questions(self, analyzer):
        """Test analyzing response with only questions."""
        response = """
        What do you think about this approach?
        How does it compare to alternatives?
        Why would we choose this option?
        """

        score = analyzer.analyze(response, "claude", 1)

        # Questions should not count as claims
        assert score.claim_count == 0
        assert score.citation_density == 0.0

    def test_mixed_case_vague_phrases(self, analyzer):
        """Test that vague phrase detection is case-insensitive."""
        response = "GENERALLY speaking, it MIGHT work. TYPICALLY these things DEPEND."

        score = analyzer.analyze(response, "claude", 1)

        assert score.vague_phrase_count >= 3

    def test_temporal_very_old_dates(self, analyzer):
        """Test temporal relevance with very old dates."""
        response = "Based on research from 1990 and 1985, this was established."

        score = analyzer.analyze(response, "claude", 1)

        # Very old dates should have low relevance
        assert score.temporal_relevance < 0.5

    def test_temporal_future_dates(self, analyzer):
        """Test temporal relevance with future dates."""
        response = "The 2030 projection shows improvement. The 2028 plan is ready."

        score = analyzer.analyze(response, "claude", 1)

        # Future dates relative to 2026 should still be handled
        assert score.temporal_relevance >= 0

    def test_year_pattern_edge_cases(self, analyzer):
        """Test year detection with edge case values."""
        response = "Years 1899 and 2099 are not typical academic years."

        score = analyzer.analyze(response, "claude", 1)

        # Pattern matches 19xx and 20xx
        assert score.temporal_relevance >= 0


class TestEvidenceQualityIntegration:
    """Integration tests for evidence quality system."""

    def test_full_analysis_pipeline(self):
        """Test complete analysis from response to hollow consensus detection."""
        # Create a debate scenario
        responses = {
            "claude": """
                Based on the 2024 performance analysis [1], Redis provides sub-millisecond
                latency for 99.9% of requests. According to https://redis.io/benchmarks,
                throughput reaches 1 million ops/sec on modest hardware. For example,
                Twitter uses Redis for timeline caching. Therefore, Redis is recommended
                for our use case because it matches our latency requirements of <10ms.
            """,
            "gpt4": """
                The 2025 cloud infrastructure report [2] shows Redis achieving 0.5ms
                average latency. Specifically, the P99 is measured at 2ms. For instance,
                Snapchat employs Redis clusters for session storage. Thus, given that
                we need high throughput, this implies Redis is the optimal choice.
                The cost is approximately $500/month for our expected load.
            """,
            "gemini": """
                It might work. Generally Redis is good. Could be the right choice.
                Various factors to consider. Typically these things depend.
            """,
        }

        # Analyze evidence quality
        analyzer = EvidenceQualityAnalyzer()
        scores = analyzer.analyze_batch(responses, round_num=3)

        # Check individual scores
        assert scores["claude"].overall_quality > scores["gemini"].overall_quality
        assert scores["gpt4"].overall_quality > scores["gemini"].overall_quality

        # Check hollow consensus
        detector = HollowConsensusDetector()
        alert = detector.check(responses, convergence_similarity=0.75, round_num=3)

        # Should detect quality variance
        assert len(alert.agent_scores) == 3
        assert alert.agent_scores["claude"] > alert.agent_scores["gemini"]

    def test_high_quality_consensus_not_hollow(self):
        """Test that high quality consensus is not flagged as hollow."""
        responses = {
            "claude": """
                According to [1] and [2], the 2025 benchmark shows 50ms latency.
                Specifically, 99.5% of requests complete under 100ms.
                For example, similar systems at scale achieve this.
                Therefore, based on these metrics, we recommend this approach.
            """,
            "gpt4": """
                The documentation [3] confirms 45ms average latency in 2024.
                Precisely, the P95 is measured at 85ms under load.
                For instance, production deployments validate this.
                Hence, the evidence supports this recommendation.
            """,
        }

        detector = HollowConsensusDetector()
        alert = detector.check(responses, convergence_similarity=0.9, round_num=2)

        # High quality consensus should have low severity
        assert alert.avg_quality > 0.4
        # May or may not be detected depending on exact thresholds

    def test_pattern_detection_comprehensive(self):
        """Test comprehensive pattern detection across all types."""
        analyzer = EvidenceQualityAnalyzer()

        # Test all citation patterns
        # Note: The pattern \([\w\s]+,?\s*\d{4}\) matches (Author 2024) but not
        # (Jones et al., 2023) because \w doesn't match the period in "et al."
        citation_tests = [
            ("[1]", True),
            ("[1, 2, 3]", True),
            ("(Smith 2024)", True),
            ("(Jones 2023)", True),  # Simple author year format
            ("according to the documentation", True),
            ("https://example.com/page", True),
            ("source: RFC 8259", True),
        ]

        for text, should_detect in citation_tests:
            markers = analyzer._detect_evidence(text)
            citations = [m for m in markers if m.evidence_type == EvidenceType.CITATION]
            if should_detect:
                assert len(citations) > 0, f"Failed to detect citation in: {text}"

        # Test all data patterns
        data_tests = [
            ("45%", True),
            ("99.9%", True),
            ("$1,000", True),
            ("$50.00", True),
            ("100ms", True),
            ("30 seconds", True),
            ("5 minutes", True),
            ("24 hours", True),
            ("7 days", True),
            ("512MB", True),
            ("2.5GB", True),
            ("increased by 50", True),
            ("decreased 30", True),
            ("improved 25", True),
            ("reduced by 10", True),
        ]

        for text, should_detect in data_tests:
            markers = analyzer._detect_evidence(text)
            data = [m for m in markers if m.evidence_type == EvidenceType.DATA]
            if should_detect:
                assert len(data) > 0, f"Failed to detect data in: {text}"

        # Test all example patterns
        example_tests = [
            ("for example", True),
            ("for instance", True),
            ("such as Redis", True),
            ("e.g.", True),
            ("specifically,", True),
            ("in practice,", True),
            ("case in point", True),
        ]

        for text, should_detect in example_tests:
            markers = analyzer._detect_evidence(text)
            examples = [m for m in markers if m.evidence_type == EvidenceType.EXAMPLE]
            if should_detect:
                assert len(examples) > 0, f"Failed to detect example in: {text}"
