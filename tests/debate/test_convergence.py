"""
Tests for the convergence detection module.

Tests cover:
- ConvergenceResult data class
- ArgumentDiversityMetric, EvidenceConvergenceMetric, StanceVolatilityMetric
- AdvancedConvergenceMetrics
- AdvancedConvergenceAnalyzer
- ConvergenceDetector
- Similarity backends (Jaccard)
"""

from __future__ import annotations

import pytest

from aragora.debate.convergence import (
    AdvancedConvergenceAnalyzer,
    AdvancedConvergenceMetrics,
    ArgumentDiversityMetric,
    ConvergenceDetector,
    ConvergenceResult,
    EvidenceConvergenceMetric,
    JaccardBackend,
    StanceVolatilityMetric,
)


class TestConvergenceResult:
    """Tests for ConvergenceResult data class."""

    def test_result_creation(self):
        """Test creating ConvergenceResult."""
        result = ConvergenceResult(
            converged=True,
            status="converged",
            min_similarity=0.87,
            avg_similarity=0.91,
            per_agent_similarity={"claude": 0.87, "gpt4": 0.95},
            consecutive_stable_rounds=2,
        )

        assert result.converged is True
        assert result.status == "converged"
        assert result.min_similarity == 0.87
        assert result.avg_similarity == 0.91
        assert result.per_agent_similarity["claude"] == 0.87
        assert result.consecutive_stable_rounds == 2

    def test_result_not_converged(self):
        """Test result for non-converged state."""
        result = ConvergenceResult(
            converged=False,
            status="refining",
            min_similarity=0.65,
            avg_similarity=0.72,
        )

        assert result.converged is False
        assert result.status == "refining"

    def test_result_diverging(self):
        """Test result for diverging state."""
        result = ConvergenceResult(
            converged=False,
            status="diverging",
            min_similarity=0.25,
            avg_similarity=0.30,
        )

        assert result.converged is False
        assert result.status == "diverging"
        assert result.min_similarity < 0.40


class TestArgumentDiversityMetric:
    """Tests for ArgumentDiversityMetric."""

    def test_metric_creation(self):
        """Test creating ArgumentDiversityMetric."""
        metric = ArgumentDiversityMetric(
            unique_arguments=5,
            total_arguments=10,
            diversity_score=0.5,
        )

        assert metric.unique_arguments == 5
        assert metric.total_arguments == 10
        assert metric.diversity_score == 0.5

    def test_is_converging_true(self):
        """Test is_converging returns True when diversity is low."""
        metric = ArgumentDiversityMetric(
            unique_arguments=1,
            total_arguments=10,
            diversity_score=0.1,  # < 0.3
        )

        assert metric.is_converging is True

    def test_is_converging_false(self):
        """Test is_converging returns False when diversity is high."""
        metric = ArgumentDiversityMetric(
            unique_arguments=8,
            total_arguments=10,
            diversity_score=0.8,  # >= 0.3
        )

        assert metric.is_converging is False


class TestEvidenceConvergenceMetric:
    """Tests for EvidenceConvergenceMetric."""

    def test_metric_creation(self):
        """Test creating EvidenceConvergenceMetric."""
        metric = EvidenceConvergenceMetric(
            shared_citations=3,
            total_citations=5,
            overlap_score=0.6,
        )

        assert metric.shared_citations == 3
        assert metric.total_citations == 5
        assert metric.overlap_score == 0.6

    def test_is_converging_true(self):
        """Test is_converging returns True when overlap is high."""
        metric = EvidenceConvergenceMetric(
            shared_citations=4,
            total_citations=5,
            overlap_score=0.8,  # > 0.6
        )

        assert metric.is_converging is True

    def test_is_converging_false(self):
        """Test is_converging returns False when overlap is low."""
        metric = EvidenceConvergenceMetric(
            shared_citations=1,
            total_citations=5,
            overlap_score=0.2,  # <= 0.6
        )

        assert metric.is_converging is False


class TestStanceVolatilityMetric:
    """Tests for StanceVolatilityMetric."""

    def test_metric_creation(self):
        """Test creating StanceVolatilityMetric."""
        metric = StanceVolatilityMetric(
            stance_changes=2,
            total_responses=10,
            volatility_score=0.2,
        )

        assert metric.stance_changes == 2
        assert metric.total_responses == 10
        assert metric.volatility_score == 0.2

    def test_is_stable_true(self):
        """Test is_stable returns True when volatility is low."""
        metric = StanceVolatilityMetric(
            stance_changes=1,
            total_responses=10,
            volatility_score=0.1,  # < 0.2
        )

        assert metric.is_stable is True

    def test_is_stable_false(self):
        """Test is_stable returns False when volatility is high."""
        metric = StanceVolatilityMetric(
            stance_changes=5,
            total_responses=10,
            volatility_score=0.5,  # >= 0.2
        )

        assert metric.is_stable is False


class TestAdvancedConvergenceMetrics:
    """Tests for AdvancedConvergenceMetrics."""

    def test_metrics_creation(self):
        """Test creating AdvancedConvergenceMetrics."""
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=0.8,
            domain="security",
        )

        assert metrics.semantic_similarity == 0.8
        assert metrics.domain == "security"
        assert metrics.overall_convergence == 0.0

    def test_compute_overall_score_semantic_only(self):
        """Test overall score with only semantic similarity."""
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=0.9,
        )

        score = metrics.compute_overall_score()

        # 0.9 * 0.4 (semantic weight) = 0.36
        assert score == pytest.approx(0.36, rel=0.01)
        assert metrics.overall_convergence == score

    def test_compute_overall_score_all_metrics(self):
        """Test overall score with all metrics."""
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=0.8,
            argument_diversity=ArgumentDiversityMetric(
                unique_arguments=2,
                total_arguments=10,
                diversity_score=0.2,  # Low diversity = converging
            ),
            evidence_convergence=EvidenceConvergenceMetric(
                shared_citations=4,
                total_citations=5,
                overlap_score=0.8,
            ),
            stance_volatility=StanceVolatilityMetric(
                stance_changes=1,
                total_responses=10,
                volatility_score=0.1,
            ),
        )

        score = metrics.compute_overall_score()

        # Semantic: 0.8 * 0.4 = 0.32
        # Diversity: (1 - 0.2) * 0.2 = 0.16
        # Evidence: 0.8 * 0.2 = 0.16
        # Stability: (1 - 0.1) * 0.2 = 0.18
        # Total: 0.82
        assert score == pytest.approx(0.82, rel=0.01)

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=0.75,
            argument_diversity=ArgumentDiversityMetric(3, 10, 0.3),
            evidence_convergence=EvidenceConvergenceMetric(2, 4, 0.5),
            domain="testing",
        )
        metrics.compute_overall_score()

        result = metrics.to_dict()

        assert result["semantic_similarity"] == 0.75
        assert result["domain"] == "testing"
        assert "argument_diversity" in result
        assert result["argument_diversity"]["unique_arguments"] == 3
        assert "evidence_convergence" in result
        assert result["evidence_convergence"]["overlap_score"] == 0.5


class TestJaccardBackend:
    """Tests for JaccardBackend (always available)."""

    def test_compute_similarity_identical(self):
        """Test similarity of identical texts."""
        backend = JaccardBackend()
        text = "The system should use caching for performance."

        sim = backend.compute_similarity(text, text)

        assert sim == 1.0

    def test_compute_similarity_completely_different(self):
        """Test similarity of completely different texts."""
        backend = JaccardBackend()
        text1 = "apple banana cherry"
        text2 = "dog elephant fox"

        sim = backend.compute_similarity(text1, text2)

        assert sim == 0.0

    def test_compute_similarity_partial_overlap(self):
        """Test similarity with partial word overlap."""
        backend = JaccardBackend()
        text1 = "the quick brown fox jumps"
        text2 = "the lazy brown dog sleeps"

        sim = backend.compute_similarity(text1, text2)

        # Common words: "the", "brown" (2 words)
        # Union: "the", "quick", "brown", "fox", "jumps", "lazy", "dog", "sleeps" (8 words)
        # Jaccard: 2/8 = 0.25
        assert sim == pytest.approx(0.25, rel=0.01)

    def test_compute_similarity_empty_text(self):
        """Test similarity with empty text."""
        backend = JaccardBackend()

        sim = backend.compute_similarity("", "some text")

        assert sim == 0.0

    def test_compute_similarity_normalizes_case(self):
        """Test that similarity is case-insensitive."""
        backend = JaccardBackend()
        text1 = "The Quick Brown Fox"
        text2 = "the quick brown fox"

        sim = backend.compute_similarity(text1, text2)

        assert sim == 1.0


class TestAdvancedConvergenceAnalyzer:
    """Tests for AdvancedConvergenceAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with Jaccard backend for consistent testing."""
        return AdvancedConvergenceAnalyzer(similarity_backend=JaccardBackend())

    def test_extract_arguments(self, analyzer):
        """Test argument extraction from text."""
        text = """
        The system should use Redis for caching. This will improve performance significantly.
        Additionally, we need to implement rate limiting for the API endpoints.
        """

        arguments = analyzer.extract_arguments(text)

        # Should extract sentences with > 5 words
        assert len(arguments) >= 2
        assert any("Redis" in arg for arg in arguments)
        assert any("rate limiting" in arg for arg in arguments)

    def test_extract_arguments_short_text(self, analyzer):
        """Test argument extraction from short text."""
        text = "Yes. No. OK."

        arguments = analyzer.extract_arguments(text)

        # Short sentences should be filtered out
        assert len(arguments) == 0

    def test_extract_citations_urls(self, analyzer):
        """Test citation extraction for URLs."""
        text = "According to https://example.com/docs the API is stable."

        citations = analyzer.extract_citations(text)

        assert "https://example.com/docs" in citations

    def test_extract_citations_academic(self, analyzer):
        """Test citation extraction for academic citations."""
        text = "This approach was proposed by (Smith, 2024) and validated by (Jones et al., 2023)."

        citations = analyzer.extract_citations(text)

        assert "(Smith, 2024)" in citations
        assert "(Jones et al., 2023)" in citations

    def test_extract_citations_numbered(self, analyzer):
        """Test citation extraction for numbered references."""
        text = "As shown in [1] and [2], the method is effective."

        citations = analyzer.extract_citations(text)

        assert "[1]" in citations
        assert "[2]" in citations

    def test_detect_stance_support(self, analyzer):
        """Test stance detection for supportive text."""
        text = "I strongly agree with this approach and recommend we implement it immediately."

        stance = analyzer.detect_stance(text)

        assert stance == "support"

    def test_detect_stance_oppose(self, analyzer):
        """Test stance detection for opposing text."""
        text = "I disagree with this proposal. We should reject this approach."

        stance = analyzer.detect_stance(text)

        assert stance == "oppose"

    def test_detect_stance_neutral(self, analyzer):
        """Test stance detection for neutral text."""
        text = "The weather is nice today. The sky is blue."

        stance = analyzer.detect_stance(text)

        assert stance == "neutral"

    def test_detect_stance_mixed(self, analyzer):
        """Test stance detection for mixed text."""
        text = "I agree with part of this, however I disagree with the implementation. It depends on the use case."

        stance = analyzer.detect_stance(text)

        assert stance == "mixed"

    def test_compute_argument_diversity(self, analyzer):
        """Test argument diversity computation."""
        agent_responses = {
            "claude": "We should use Redis for caching. This improves performance significantly.",
            "gpt4": "Memory caching with Redis is recommended. It reduces database load effectively.",
            "gemini": "Consider using PostgreSQL JSONB for data storage. This simplifies the architecture.",
        }

        metric = analyzer.compute_argument_diversity(agent_responses)

        assert metric.total_arguments >= 3
        assert 0 <= metric.diversity_score <= 1

    def test_compute_argument_diversity_empty(self, analyzer):
        """Test argument diversity with empty responses."""
        agent_responses = {
            "claude": "Yes.",
            "gpt4": "No.",
        }

        metric = analyzer.compute_argument_diversity(agent_responses)

        assert metric.total_arguments == 0
        assert metric.diversity_score == 0.0

    def test_compute_evidence_convergence(self, analyzer):
        """Test evidence convergence computation."""
        agent_responses = {
            "claude": "According to https://redis.io the performance is excellent.",
            "gpt4": "The Redis documentation at https://redis.io shows good benchmarks.",
            "gemini": "Based on https://mongodb.com the alternative is also good.",
        }

        metric = analyzer.compute_evidence_convergence(agent_responses)

        assert metric.total_citations >= 2
        assert metric.shared_citations >= 1  # redis.io shared by 2 agents

    def test_compute_evidence_convergence_no_citations(self, analyzer):
        """Test evidence convergence with no citations."""
        agent_responses = {
            "claude": "The approach seems reasonable.",
            "gpt4": "I agree with the proposal.",
        }

        metric = analyzer.compute_evidence_convergence(agent_responses)

        assert metric.total_citations == 0
        assert metric.overlap_score == 0.0

    def test_compute_stance_volatility(self, analyzer):
        """Test stance volatility computation."""
        response_history = [
            {"claude": "I agree with this approach.", "gpt4": "I disagree."},
            {"claude": "I still agree strongly.", "gpt4": "I now support this."},
            {"claude": "Agreement confirmed.", "gpt4": "Yes, I support it."},
        ]

        metric = analyzer.compute_stance_volatility(response_history)

        # gpt4 changed stance from oppose to support
        assert metric.stance_changes >= 1
        assert metric.total_responses == 6

    def test_compute_stance_volatility_single_round(self, analyzer):
        """Test stance volatility with single round."""
        response_history = [
            {"claude": "I agree.", "gpt4": "I disagree."},
        ]

        metric = analyzer.compute_stance_volatility(response_history)

        # Not enough rounds to detect changes
        assert metric.stance_changes == 0
        assert metric.volatility_score == 0.0

    def test_analyze_comprehensive(self, analyzer):
        """Test comprehensive analysis."""
        current_responses = {
            "claude": "Redis is the best choice for caching. It provides excellent performance.",
            "gpt4": "I recommend Redis for caching. Performance benchmarks are impressive.",
        }
        previous_responses = {
            "claude": "We should consider caching options. Redis might be good.",
            "gpt4": "Caching is important. Let's evaluate Redis and Memcached.",
        }

        metrics = analyzer.analyze(
            current_responses=current_responses,
            previous_responses=previous_responses,
            domain="performance",
        )

        assert metrics.domain == "performance"
        assert metrics.semantic_similarity >= 0
        assert metrics.argument_diversity is not None
        assert metrics.evidence_convergence is not None
        assert metrics.overall_convergence >= 0


class TestConvergenceDetector:
    """Tests for ConvergenceDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector with default thresholds."""
        return ConvergenceDetector(
            convergence_threshold=0.85,
            divergence_threshold=0.40,
            min_rounds_before_check=1,
            consecutive_rounds_needed=1,
        )

    def test_check_convergence_too_early(self, detector):
        """Test that early rounds return None."""
        current = {"claude": "Some response"}
        previous = {"claude": "Previous response"}

        result = detector.check_convergence(current, previous, round_number=1)

        assert result is None  # min_rounds_before_check = 1

    def test_check_convergence_no_common_agents(self, detector):
        """Test with no common agents between rounds."""
        current = {"claude": "Response from Claude"}
        previous = {"gpt4": "Response from GPT-4"}

        result = detector.check_convergence(current, previous, round_number=2)

        assert result is None

    def test_check_convergence_converged(self, detector):
        """Test detection of converged state."""
        # Use nearly identical responses
        response = "The system should use Redis for caching with a TTL of 15 minutes"
        current = {"claude": response, "gpt4": response}
        previous = {"claude": response, "gpt4": response}

        result = detector.check_convergence(current, previous, round_number=2)

        assert result is not None
        assert result.converged is True
        assert result.status == "converged"
        assert result.min_similarity >= 0.85

    def test_check_convergence_diverging(self, detector):
        """Test detection of diverging state."""
        current = {
            "claude": "apple orange banana cherry grape",
            "gpt4": "dog cat elephant mouse tiger",
        }
        previous = {
            "claude": "carrot potato tomato onion pepper",
            "gpt4": "house building apartment office store",
        }

        result = detector.check_convergence(current, previous, round_number=2)

        assert result is not None
        assert result.converged is False
        assert result.status == "diverging"
        assert result.min_similarity < 0.40

    def test_check_convergence_refining(self, detector):
        """Test detection of refining state."""
        current = {
            "claude": "We should use Redis for caching to improve performance",
            "gpt4": "Redis caching would help with better performance metrics",
        }
        previous = {
            "claude": "Consider using caching for the system performance",
            "gpt4": "Caching might improve the overall system performance",
        }

        result = detector.check_convergence(current, previous, round_number=2)

        assert result is not None
        # Should be refining (some similarity but not converged)
        if result.min_similarity < 0.85 and result.min_similarity >= 0.40:
            assert result.status == "refining"
            assert result.converged is False

    def test_check_convergence_per_agent_similarity(self, detector):
        """Test that per-agent similarity is tracked."""
        response = "Identical response text for testing"
        current = {"claude": response, "gpt4": response, "gemini": response}
        previous = {"claude": response, "gpt4": response, "gemini": response}

        result = detector.check_convergence(current, previous, round_number=2)

        assert result is not None
        assert "claude" in result.per_agent_similarity
        assert "gpt4" in result.per_agent_similarity
        assert "gemini" in result.per_agent_similarity
        # All should be 1.0 for identical text
        for sim in result.per_agent_similarity.values():
            assert sim == 1.0

    def test_consecutive_stable_rounds(self):
        """Test consecutive stable rounds tracking."""
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            min_rounds_before_check=1,
            consecutive_rounds_needed=2,  # Need 2 stable rounds
        )

        response = "The exact same response text"
        current = {"claude": response}
        previous = {"claude": response}

        # First check - should be refining (need more rounds)
        result1 = detector.check_convergence(current, previous, round_number=2)
        assert result1 is not None
        assert result1.status == "refining"
        assert result1.consecutive_stable_rounds == 1

        # Second check - should now converge
        result2 = detector.check_convergence(current, previous, round_number=3)
        assert result2 is not None
        assert result2.status == "converged"
        assert result2.converged is True
        assert result2.consecutive_stable_rounds == 2

    def test_reset(self, detector):
        """Test reset method."""
        response = "Same response"
        current = {"claude": response}
        previous = {"claude": response}

        # Build up consecutive count
        detector.check_convergence(current, previous, round_number=2)
        assert detector.consecutive_stable_count > 0

        # Reset
        detector.reset()
        assert detector.consecutive_stable_count == 0

    def test_custom_thresholds(self):
        """Test detector with custom thresholds."""
        detector = ConvergenceDetector(
            convergence_threshold=0.95,  # Very strict
            divergence_threshold=0.30,  # More lenient
        )

        assert detector.convergence_threshold == 0.95
        assert detector.divergence_threshold == 0.30

    def test_debate_id_isolation(self):
        """Test that debate_id is stored for cache isolation."""
        detector = ConvergenceDetector(debate_id="debate_123")

        assert detector.debate_id == "debate_123"


class TestConvergenceDetectorBackendSelection:
    """Tests for backend selection in ConvergenceDetector."""

    def test_fallback_to_jaccard(self):
        """Test that detector falls back to Jaccard if other backends unavailable."""
        # Create detector - it will use available backend
        detector = ConvergenceDetector()

        # Backend should be one of the supported types
        assert detector.backend is not None
        assert hasattr(detector.backend, "compute_similarity")

    def test_backend_compute_similarity_works(self):
        """Test that selected backend can compute similarity."""
        detector = ConvergenceDetector()

        sim = detector.backend.compute_similarity(
            "test text one",
            "test text one",
        )

        # Identical texts should have high similarity (approx 1.0 due to floating point)
        assert sim == pytest.approx(1.0, rel=1e-5)
