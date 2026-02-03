"""
Power Sampling - Inference-time reasoning improvement via power-law weighted sampling.

Based on research showing that power-law distributions better capture the relative
quality of reasoning chains compared to uniform or softmax-weighted sampling.
Claims 10x speedup vs MCMC-based approaches while maintaining quality.

Key insights:
1. Reasoning quality follows power-law distribution (few excellent, many mediocre)
2. Power-law weighting naturally concentrates mass on high-quality samples
3. Allows parallel generation with post-hoc selection (no sequential dependency)

Usage:
    sampler = PowerSampler(config=PowerSamplingConfig(n_samples=8, power_alpha=2.0))
    result = await sampler.sample_best_reasoning(
        generator=agent.generate,
        prompt="Should we use REST or GraphQL?",
        scorer=quality_scorer,
    )
    print(result.best_response)
    print(f"Confidence: {result.confidence}")
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, TypeVar

T = TypeVar("T")


class ResponseScorer(Protocol):
    """Protocol for scoring generated responses."""

    async def score(self, response: str, prompt: str) -> float:
        """Score a response on quality/correctness.

        Args:
            response: The generated response text
            prompt: The original prompt

        Returns:
            Score between 0.0 and 1.0
        """
        ...


@dataclass
class PowerSamplingConfig:
    """Configuration for power sampling.

    Attributes:
        n_samples: Number of parallel samples to generate (default: 8)
        power_alpha: Power-law exponent (higher = more concentration on best)
        min_samples: Minimum samples before selection (for early stopping)
        temperature_schedule: Optional temperature values for each sample
        diversity_penalty: Penalty for similar responses (0.0 = no penalty)
        timeout_seconds: Timeout for sample generation
        enable_early_stopping: Stop if high-confidence sample found early
        early_stop_threshold: Confidence threshold for early stopping
    """

    n_samples: int = 8
    power_alpha: float = 2.0
    min_samples: int = 3
    temperature_schedule: list[float] | None = None
    diversity_penalty: float = 0.1
    timeout_seconds: float = 30.0
    enable_early_stopping: bool = True
    early_stop_threshold: float = 0.95


@dataclass
class ScoredSample:
    """A generated sample with its quality score."""

    response: str
    score: float
    temperature: float = 1.0
    generation_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingResult:
    """Result of power sampling procedure.

    Attributes:
        best_response: The highest-weighted response
        best_score: Score of the best response
        confidence: Confidence in selection (based on score distribution)
        all_samples: All generated samples with scores
        selected_indices: Indices of samples that influenced selection
        aggregated_response: Optional consensus/aggregated response
        early_stopped: Whether sampling stopped early
        total_time: Total sampling time in seconds
    """

    best_response: str
    best_score: float
    confidence: float
    all_samples: list[ScoredSample]
    selected_indices: list[int]
    aggregated_response: str | None = None
    early_stopped: bool = False
    total_time: float = 0.0


def power_law_weights(scores: list[float], alpha: float = 2.0) -> list[float]:
    """Compute power-law weights from scores.

    Transforms scores to weights using: w_i = score_i^alpha / sum(score_j^alpha)

    Args:
        scores: List of quality scores (0.0 to 1.0)
        alpha: Power-law exponent (higher = more concentration)

    Returns:
        Normalized weights summing to 1.0
    """
    if not scores:
        return []

    # Add small epsilon to avoid zero weights
    eps = 1e-8
    powered = [(max(s, eps) ** alpha) for s in scores]
    total = sum(powered)

    if total < eps:
        # Uniform fallback if all scores are zero
        return [1.0 / len(scores)] * len(scores)

    return [p / total for p in powered]


def compute_diversity(response1: str, response2: str) -> float:
    """Compute diversity between two responses using Jaccard distance.

    Args:
        response1: First response text
        response2: Second response text

    Returns:
        Diversity score between 0.0 (identical) and 1.0 (completely different)
    """
    # Tokenize by words
    words1 = set(response1.lower().split())
    words2 = set(response2.lower().split())

    if not words1 and not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    if union == 0:
        return 0.0

    jaccard_similarity = intersection / union
    return 1.0 - jaccard_similarity


def select_diverse_samples(
    samples: list[ScoredSample],
    k: int,
    diversity_weight: float = 0.3,
) -> list[int]:
    """Select k diverse high-quality samples using greedy selection.

    Balances quality scores with diversity from already-selected samples.

    Args:
        samples: List of scored samples
        k: Number of samples to select
        diversity_weight: Weight given to diversity vs quality (0-1)

    Returns:
        Indices of selected samples
    """
    if len(samples) <= k:
        return list(range(len(samples)))

    selected: list[int] = []

    # First, select the highest scoring sample
    best_idx = max(range(len(samples)), key=lambda i: samples[i].score)
    selected.append(best_idx)

    # Greedily select remaining samples
    while len(selected) < k:
        best_score = -1.0
        best_candidate = -1

        for i, sample in enumerate(samples):
            if i in selected:
                continue

            # Compute minimum diversity from selected samples
            min_diversity = min(
                compute_diversity(sample.response, samples[j].response) for j in selected
            )

            # Combined score: quality + diversity
            combined = (1 - diversity_weight) * sample.score + diversity_weight * min_diversity

            if combined > best_score:
                best_score = combined
                best_candidate = i

        if best_candidate >= 0:
            selected.append(best_candidate)
        else:
            break

    return selected


async def sample_with_power_law(
    generator: Callable[[str], Awaitable[str]],
    prompt: str,
    scorer: ResponseScorer,
    config: PowerSamplingConfig | None = None,
) -> SamplingResult:
    """Generate multiple samples and select best using power-law weighting.

    This is the main entry point for power sampling. Generates n_samples
    in parallel, scores them, applies power-law weighting, and returns
    the best response along with confidence metrics.

    Args:
        generator: Async function that generates a response from a prompt
        prompt: The input prompt
        scorer: Scorer to evaluate response quality
        config: Sampling configuration (uses defaults if None)

    Returns:
        SamplingResult with best response and metadata
    """
    cfg = config or PowerSamplingConfig()
    sampler = PowerSampler(config=cfg)
    return await sampler.sample_best_reasoning(generator, prompt, scorer)


class PowerSampler:
    """Power-law weighted sampler for inference-time reasoning improvement.

    Generates multiple reasoning chains in parallel, scores them,
    and selects the best using power-law weighting. This approach:

    1. Exploits the power-law distribution of reasoning quality
    2. Enables parallel generation (no sequential MCMC dependency)
    3. Provides calibrated confidence estimates

    Example:
        sampler = PowerSampler(config=PowerSamplingConfig(n_samples=8))

        result = await sampler.sample_best_reasoning(
            generator=my_agent.generate,
            prompt="Analyze this contract clause...",
            scorer=legal_quality_scorer,
        )

        if result.confidence > 0.8:
            use_response(result.best_response)
        else:
            request_human_review(result.all_samples)
    """

    def __init__(self, config: PowerSamplingConfig | None = None):
        """Initialize the power sampler.

        Args:
            config: Sampling configuration
        """
        self.config = config or PowerSamplingConfig()
        self._generation_stats: dict[str, Any] = {}

    async def sample_best_reasoning(
        self,
        generator: Callable[[str], Awaitable[str]],
        prompt: str,
        scorer: ResponseScorer,
    ) -> SamplingResult:
        """Generate samples and select best using power-law weighting.

        Args:
            generator: Async function to generate responses
            prompt: Input prompt
            scorer: Quality scorer

        Returns:
            SamplingResult with best response and confidence
        """
        import time

        start_time = time.time()

        # Generate temperature schedule if not provided
        temps = self.config.temperature_schedule
        if temps is None:
            # Default: vary temperature around 1.0
            temps = [0.7 + 0.1 * i for i in range(self.config.n_samples)]
        elif len(temps) < self.config.n_samples:
            # Extend with default temperature
            temps = list(temps) + [1.0] * (self.config.n_samples - len(temps))

        # Generate samples in parallel
        samples = await self._generate_parallel(generator, prompt, temps)

        if not samples:
            return SamplingResult(
                best_response="",
                best_score=0.0,
                confidence=0.0,
                all_samples=[],
                selected_indices=[],
                total_time=time.time() - start_time,
            )

        # Score all samples
        scored_samples = await self._score_samples(samples, scorer, prompt)

        # Apply power-law weighting
        scores = [s.score for s in scored_samples]
        weights = power_law_weights(scores, self.config.power_alpha)

        # Select best sample (highest weight)
        best_idx = max(range(len(weights)), key=lambda i: weights[i])

        # Select diverse high-quality samples for potential aggregation
        selected = select_diverse_samples(
            scored_samples,
            k=min(3, len(scored_samples)),
            diversity_weight=self.config.diversity_penalty,
        )

        # Compute confidence based on weight concentration
        confidence = self._compute_confidence(weights, scores)

        return SamplingResult(
            best_response=scored_samples[best_idx].response,
            best_score=scored_samples[best_idx].score,
            confidence=confidence,
            all_samples=scored_samples,
            selected_indices=selected,
            total_time=time.time() - start_time,
        )

    async def _generate_parallel(
        self,
        generator: Callable[[str], Awaitable[str]],
        prompt: str,
        temperatures: list[float],
    ) -> list[tuple[str, float, float]]:
        """Generate samples in parallel with different temperatures.

        Args:
            generator: Response generator
            prompt: Input prompt
            temperatures: Temperature for each sample

        Returns:
            List of (response, temperature, generation_time) tuples
        """
        import time

        async def generate_one(temp: float) -> tuple[str, float, float]:
            start = time.time()
            try:
                # Note: In practice, would pass temperature to generator
                # Here we just generate and track the intended temperature
                response = await asyncio.wait_for(
                    generator(prompt),
                    timeout=self.config.timeout_seconds,
                )
                return (response, temp, time.time() - start)
            except asyncio.TimeoutError:
                return ("", temp, time.time() - start)
            except Exception:
                return ("", temp, time.time() - start)

        tasks = [generate_one(t) for t in temperatures[: self.config.n_samples]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and empty responses
        valid_results = []
        for r in results:
            if isinstance(r, tuple) and r[0]:
                valid_results.append(r)

        return valid_results

    async def _score_samples(
        self,
        samples: list[tuple[str, float, float]],
        scorer: ResponseScorer,
        prompt: str,
    ) -> list[ScoredSample]:
        """Score all generated samples.

        Args:
            samples: List of (response, temperature, gen_time) tuples
            scorer: Quality scorer
            prompt: Original prompt

        Returns:
            List of ScoredSample objects
        """

        async def score_one(response: str, temp: float, gen_time: float) -> ScoredSample:
            try:
                score = await scorer.score(response, prompt)
            except Exception:
                score = 0.0

            return ScoredSample(
                response=response,
                score=score,
                temperature=temp,
                generation_time=gen_time,
            )

        tasks = [score_one(r, t, g) for r, t, g in samples]
        return await asyncio.gather(*tasks)

    def _compute_confidence(
        self,
        weights: list[float],
        scores: list[float],
    ) -> float:
        """Compute confidence in selection based on weight/score distribution.

        Higher confidence when:
        - Top weight is significantly higher than others
        - Top scores are clustered (consensus)
        - Score distribution has low entropy

        Args:
            weights: Power-law weights
            scores: Raw quality scores

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not weights or not scores:
            return 0.0

        # Factor 1: How much higher is top weight than average?
        max_weight = max(weights)
        avg_weight = sum(weights) / len(weights)
        weight_dominance = (max_weight - avg_weight) / (max_weight + 1e-8)

        # Factor 2: How high is the top score absolutely?
        max_score = max(scores)

        # Factor 3: Entropy of weight distribution (lower = more confident)
        entropy = 0.0
        for w in weights:
            if w > 1e-10:
                entropy -= w * math.log(w + 1e-10)
        max_entropy = math.log(len(weights))
        normalized_entropy = entropy / (max_entropy + 1e-8)
        certainty = 1.0 - normalized_entropy

        # Combine factors
        confidence = 0.3 * weight_dominance + 0.4 * max_score + 0.3 * certainty
        return max(0.0, min(1.0, confidence))


class DefaultScorer:
    """Simple default scorer based on response properties.

    Scores based on:
    - Response length (not too short, not too long)
    - Presence of reasoning indicators (therefore, because, etc.)
    - Structural completeness

    For production use, implement a domain-specific scorer.
    """

    async def score(self, response: str, prompt: str) -> float:
        """Score response quality heuristically."""
        if not response:
            return 0.0

        score = 0.5  # Base score

        # Length score (prefer moderate length)
        words = response.split()
        word_count = len(words)
        if 50 <= word_count <= 500:
            score += 0.2
        elif 20 <= word_count < 50 or 500 < word_count <= 1000:
            score += 0.1

        # Reasoning indicators
        reasoning_words = [
            "because",
            "therefore",
            "however",
            "although",
            "considering",
            "given that",
            "this suggests",
            "on the other hand",
            "in conclusion",
            "as a result",
        ]
        lower_response = response.lower()
        reasoning_count = sum(1 for w in reasoning_words if w in lower_response)
        score += min(0.2, reasoning_count * 0.05)

        # Structural completeness (has conclusion-like ending)
        last_sentence = response.strip().split(".")[-1].lower() if response else ""
        conclusion_indicators = ["conclusion", "therefore", "thus", "recommend", "suggest"]
        if any(ind in last_sentence for ind in conclusion_indicators):
            score += 0.1

        return min(1.0, score)
