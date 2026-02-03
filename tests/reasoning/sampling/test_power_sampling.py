"""
Tests for Power Sampling module.

Tests cover:
- Power-law weight computation
- Diversity calculation
- Sample selection
- Configuration
- Full sampling pipeline
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

# Import the modules we're testing
from aragora.reasoning.sampling.power_sampling import (
    PowerSampler,
    PowerSamplingConfig,
    SamplingResult,
    ScoredSample,
    DefaultScorer,
    power_law_weights,
    compute_diversity,
    select_diverse_samples,
    sample_with_power_law,
)


class TestPowerLawWeights:
    """Tests for power_law_weights function."""

    def test_empty_scores(self):
        """Empty scores return empty weights."""
        assert power_law_weights([]) == []

    def test_single_score(self):
        """Single score returns weight of 1.0."""
        weights = power_law_weights([0.5])
        assert len(weights) == 1
        assert weights[0] == pytest.approx(1.0)

    def test_uniform_scores(self):
        """Uniform scores return uniform weights."""
        weights = power_law_weights([0.5, 0.5, 0.5])
        assert len(weights) == 3
        assert all(w == pytest.approx(1 / 3, rel=0.01) for w in weights)

    def test_weights_sum_to_one(self):
        """Weights always sum to 1.0."""
        weights = power_law_weights([0.1, 0.5, 0.9, 0.3])
        assert sum(weights) == pytest.approx(1.0)

    def test_higher_alpha_concentrates_mass(self):
        """Higher alpha gives more weight to top score."""
        scores = [0.2, 0.8, 0.5]

        weights_low = power_law_weights(scores, alpha=1.0)
        weights_high = power_law_weights(scores, alpha=4.0)

        # Find index of highest score
        best_idx = scores.index(max(scores))

        # Higher alpha should concentrate more mass on best
        assert weights_high[best_idx] > weights_low[best_idx]

    def test_zero_scores_fallback_to_uniform(self):
        """All-zero scores fallback to uniform distribution."""
        weights = power_law_weights([0.0, 0.0, 0.0])
        assert all(w == pytest.approx(1 / 3, rel=0.01) for w in weights)


class TestComputeDiversity:
    """Tests for compute_diversity function."""

    def test_identical_responses(self):
        """Identical responses have zero diversity."""
        text = "This is a test response"
        assert compute_diversity(text, text) == pytest.approx(0.0)

    def test_completely_different_responses(self):
        """Completely different word sets have diversity 1.0."""
        text1 = "apple banana cherry"
        text2 = "dog elephant frog"
        assert compute_diversity(text1, text2) == pytest.approx(1.0)

    def test_partial_overlap(self):
        """Partial overlap results in intermediate diversity."""
        text1 = "the quick brown fox"
        text2 = "the slow brown dog"
        diversity = compute_diversity(text1, text2)
        assert 0.0 < diversity < 1.0

    def test_empty_responses(self):
        """Empty responses have zero diversity."""
        assert compute_diversity("", "") == 0.0

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        text1 = "Hello World"
        text2 = "hello world"
        assert compute_diversity(text1, text2) == pytest.approx(0.0)


class TestSelectDiverseSamples:
    """Tests for select_diverse_samples function."""

    def test_fewer_samples_than_k(self):
        """Returns all samples if fewer than k available."""
        samples = [
            ScoredSample(response="a", score=0.8, temperature=1.0),
            ScoredSample(response="b", score=0.6, temperature=1.0),
        ]
        selected = select_diverse_samples(samples, k=5)
        assert selected == [0, 1]

    def test_selects_highest_first(self):
        """First selection is always highest scoring."""
        samples = [
            ScoredSample(response="low", score=0.2, temperature=1.0),
            ScoredSample(response="high", score=0.9, temperature=1.0),
            ScoredSample(response="mid", score=0.5, temperature=1.0),
        ]
        selected = select_diverse_samples(samples, k=1)
        assert selected == [1]  # Index of highest score

    def test_diversity_influences_selection(self):
        """Diversity affects selection of subsequent samples."""
        samples = [
            ScoredSample(response="unique words here", score=0.7, temperature=1.0),
            ScoredSample(response="unique words here exactly", score=0.75, temperature=1.0),
            ScoredSample(response="completely different text", score=0.72, temperature=1.0),
        ]
        selected = select_diverse_samples(samples, k=2, diversity_weight=0.5)
        # With high diversity weight, should prefer the different text over similar
        assert 2 in selected  # The diverse response should be selected


class TestPowerSamplingConfig:
    """Tests for PowerSamplingConfig dataclass."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = PowerSamplingConfig()
        assert config.n_samples == 8
        assert config.power_alpha == 2.0
        assert config.min_samples == 3
        assert config.timeout_seconds == 30.0

    def test_custom_values(self):
        """Custom values are properly set."""
        config = PowerSamplingConfig(
            n_samples=16,
            power_alpha=3.0,
            temperature_schedule=[0.5, 0.7, 0.9],
        )
        assert config.n_samples == 16
        assert config.power_alpha == 3.0
        assert config.temperature_schedule == [0.5, 0.7, 0.9]


class TestScoredSample:
    """Tests for ScoredSample dataclass."""

    def test_creation(self):
        """Basic creation works."""
        sample = ScoredSample(
            response="test response",
            score=0.85,
            temperature=0.7,
            generation_time=1.5,
        )
        assert sample.response == "test response"
        assert sample.score == 0.85
        assert sample.temperature == 0.7
        assert sample.generation_time == 1.5

    def test_default_metadata(self):
        """Metadata defaults to empty dict."""
        sample = ScoredSample(response="test", score=0.5, temperature=1.0)
        assert sample.metadata == {}


class TestSamplingResult:
    """Tests for SamplingResult dataclass."""

    def test_creation(self):
        """Basic creation with required fields."""
        result = SamplingResult(
            best_response="best",
            best_score=0.9,
            confidence=0.85,
            all_samples=[],
            selected_indices=[0],
        )
        assert result.best_response == "best"
        assert result.best_score == 0.9
        assert result.confidence == 0.85

    def test_optional_fields(self):
        """Optional fields have defaults."""
        result = SamplingResult(
            best_response="",
            best_score=0.0,
            confidence=0.0,
            all_samples=[],
            selected_indices=[],
        )
        assert result.aggregated_response is None
        assert result.early_stopped is False
        assert result.total_time == 0.0


class TestDefaultScorer:
    """Tests for DefaultScorer class."""

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """Empty response scores 0."""
        scorer = DefaultScorer()
        score = await scorer.score("", "test prompt")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_short_response_lower_score(self):
        """Very short responses score lower."""
        scorer = DefaultScorer()
        short_score = await scorer.score("Yes.", "test prompt")
        long_score = await scorer.score(
            "This is a longer response with reasoning because "
            "it explains the context and provides a conclusion.",
            "test prompt",
        )
        assert long_score > short_score

    @pytest.mark.asyncio
    async def test_reasoning_words_boost_score(self):
        """Responses with reasoning words score higher."""
        scorer = DefaultScorer()
        basic = await scorer.score("We should use option A. It is good for the project.", "test")
        with_reasoning = await scorer.score(
            "We should use option A because it offers better scalability. "
            "Therefore, this is the recommended approach. However, we should "
            "also consider the implementation cost.",
            "test",
        )
        assert with_reasoning > basic


class TestPowerSampler:
    """Tests for PowerSampler class."""

    def test_initialization_default_config(self):
        """Sampler initializes with default config."""
        sampler = PowerSampler()
        assert sampler.config.n_samples == 8

    def test_initialization_custom_config(self):
        """Sampler accepts custom config."""
        config = PowerSamplingConfig(n_samples=4)
        sampler = PowerSampler(config=config)
        assert sampler.config.n_samples == 4

    @pytest.mark.asyncio
    async def test_sample_best_reasoning_basic(self):
        """Basic sampling pipeline works."""
        # Mock generator that returns different responses
        responses = [
            "Response one with reasoning because it explains things.",
            "Response two therefore concludes differently.",
            "Response three however takes another approach.",
        ]
        call_count = [0]

        async def mock_generator(prompt: str) -> str:
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            return responses[idx]

        # Use default scorer
        scorer = DefaultScorer()

        config = PowerSamplingConfig(n_samples=3, timeout_seconds=5.0)
        sampler = PowerSampler(config=config)

        result = await sampler.sample_best_reasoning(
            generator=mock_generator,
            prompt="Test prompt",
            scorer=scorer,
        )

        assert result.best_response in responses
        assert 0.0 <= result.best_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.all_samples) <= 3

    @pytest.mark.asyncio
    async def test_handles_generator_timeout(self):
        """Sampler handles generator timeouts gracefully."""

        async def slow_generator(prompt: str) -> str:
            await asyncio.sleep(10)  # Longer than timeout
            return "response"

        scorer = DefaultScorer()
        config = PowerSamplingConfig(n_samples=2, timeout_seconds=0.1)
        sampler = PowerSampler(config=config)

        result = await sampler.sample_best_reasoning(
            generator=slow_generator,
            prompt="test",
            scorer=scorer,
        )

        # Should return empty result, not crash
        assert result.best_response == ""

    @pytest.mark.asyncio
    async def test_handles_generator_exception(self):
        """Sampler handles generator exceptions gracefully."""

        async def failing_generator(prompt: str) -> str:
            raise ValueError("Generator failed")

        scorer = DefaultScorer()
        config = PowerSamplingConfig(n_samples=2, timeout_seconds=5.0)
        sampler = PowerSampler(config=config)

        result = await sampler.sample_best_reasoning(
            generator=failing_generator,
            prompt="test",
            scorer=scorer,
        )

        # Should return empty result, not crash
        assert result.best_response == ""


class TestSampleWithPowerLaw:
    """Tests for sample_with_power_law convenience function."""

    @pytest.mark.asyncio
    async def test_convenience_function_works(self):
        """sample_with_power_law function works correctly."""

        async def simple_generator(prompt: str) -> str:
            return f"Response to: {prompt}"

        scorer = DefaultScorer()
        config = PowerSamplingConfig(n_samples=2, timeout_seconds=5.0)

        result = await sample_with_power_law(
            generator=simple_generator,
            prompt="test prompt",
            scorer=scorer,
            config=config,
        )

        assert isinstance(result, SamplingResult)
        assert "Response to:" in result.best_response


class TestConfidenceComputation:
    """Tests for confidence computation in PowerSampler."""

    def test_high_confidence_when_dominant_weight(self):
        """High confidence when one weight dominates."""
        sampler = PowerSampler()

        # One very high weight, others low
        weights = [0.9, 0.05, 0.05]
        scores = [0.95, 0.3, 0.25]

        confidence = sampler._compute_confidence(weights, scores)
        assert confidence > 0.7

    def test_low_confidence_when_uniform_weights(self):
        """Lower confidence when weights are uniform."""
        sampler = PowerSampler()

        # Uniform weights
        weights = [0.33, 0.33, 0.34]
        scores = [0.5, 0.5, 0.5]

        confidence = sampler._compute_confidence(weights, scores)
        assert confidence < 0.7

    def test_empty_inputs(self):
        """Empty inputs return zero confidence."""
        sampler = PowerSampler()
        assert sampler._compute_confidence([], []) == 0.0
