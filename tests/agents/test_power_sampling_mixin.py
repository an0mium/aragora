"""Tests for power sampling mixin."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.agents.power_sampling_mixin import PowerSamplingMixin
from aragora.debate.arena_sub_configs import PowerSamplingConfig


class MockAgent(PowerSamplingMixin):
    """Mock agent for testing power sampling."""

    def __init__(self, name: str = "test"):
        self.name = name
        self._call_count = 0
        self._responses: list[str] = []

    def set_responses(self, responses: list[str]) -> None:
        """Set the responses to return in sequence."""
        self._responses = responses
        self._call_count = 0

    async def generate(self, prompt: str, context=None) -> str:
        """Mock generate that returns pre-set responses."""
        idx = self._call_count % len(self._responses) if self._responses else 0
        self._call_count += 1
        if self._responses:
            return self._responses[idx]
        return f"Response {self._call_count}"

    async def _base_generate(self, prompt: str, context=None) -> str:
        """Override base generate to use mock."""
        return await self.generate(prompt, context)


class TestPowerSamplingConfiguration:
    """Tests for power sampling configuration."""

    def test_disabled_by_default(self):
        """Power sampling is disabled when no config is set."""
        agent = MockAgent()
        assert not agent.power_sampling_enabled

    def test_disabled_when_config_disables_it(self):
        """Power sampling is disabled when config sets enable_power_sampling=False."""
        agent = MockAgent()
        config = PowerSamplingConfig(enable_power_sampling=False)
        agent.configure_power_sampling(config)
        assert not agent.power_sampling_enabled

    def test_enabled_when_config_enables_it(self):
        """Power sampling is enabled when config sets enable_power_sampling=True."""
        agent = MockAgent()
        config = PowerSamplingConfig(enable_power_sampling=True)
        agent.configure_power_sampling(config)
        assert agent.power_sampling_enabled


class TestPowerSamplingGeneration:
    """Tests for power sampling generation."""

    @pytest.mark.asyncio
    async def test_generates_multiple_samples(self):
        """Power sampling generates the configured number of samples."""
        agent = MockAgent()
        agent.set_responses(
            [
                "Short",
                "A longer response with more content",
                "The longest response with lots of reasoning because it considers multiple factors",
            ]
        )

        config = PowerSamplingConfig(
            enable_power_sampling=True,
            n_samples=3,
            alpha=2.0,
            min_quality_threshold=0.0,  # Accept all samples
        )
        agent.configure_power_sampling(config)

        result = await agent.generate_with_power_sampling("test prompt")

        # Should have called generate 3 times
        assert agent._call_count == 3
        # Result should be one of the responses
        assert result in agent._responses

    @pytest.mark.asyncio
    async def test_selects_higher_quality_sample(self):
        """Power sampling tends to select higher quality samples."""
        agent = MockAgent()

        # First response is low quality, second is high quality
        low_quality = "ok"
        high_quality = "A comprehensive response because it considers multiple factors first and second and therefore reaches a solid conclusion"

        agent.set_responses([low_quality, high_quality, low_quality])

        config = PowerSamplingConfig(
            enable_power_sampling=True,
            n_samples=3,
            alpha=5.0,  # High alpha strongly prefers best
            k_diverse=1,
            min_quality_threshold=0.0,
        )
        agent.configure_power_sampling(config)

        # Run multiple times to verify tendency
        selections = []
        for _ in range(10):
            agent._call_count = 0
            result = await agent.generate_with_power_sampling("test")
            selections.append(result)

        # High quality should be selected more often
        high_quality_count = selections.count(high_quality)
        assert high_quality_count >= 5  # At least half

    @pytest.mark.asyncio
    async def test_handles_failed_samples(self):
        """Power sampling handles failed sample generation gracefully."""
        agent = MockAgent()

        async def failing_generate(prompt, context=None):
            agent._call_count += 1
            if agent._call_count == 1:
                raise RuntimeError("Generation failed")
            return "Valid response"

        agent._base_generate = failing_generate

        config = PowerSamplingConfig(
            enable_power_sampling=True,
            n_samples=3,
            sample_timeout=5.0,
        )
        agent.configure_power_sampling(config)

        result = await agent.generate_with_power_sampling("test")

        # Should still return a valid result despite one failure
        assert result == "Valid response"

    @pytest.mark.asyncio
    async def test_min_quality_threshold_filters(self):
        """Samples below min_quality_threshold are filtered."""
        agent = MockAgent()

        # All low quality responses
        agent.set_responses(["a", "b", "c"])

        config = PowerSamplingConfig(
            enable_power_sampling=True,
            n_samples=3,
            min_quality_threshold=0.9,  # Very high threshold
        )
        agent.configure_power_sampling(config)

        # Should still return something (best available even if below threshold)
        result = await agent.generate_with_power_sampling("test")
        assert result in ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_custom_scorer(self):
        """Custom scorer function is used when provided."""
        agent = MockAgent()
        agent.set_responses(["first", "second", "third"])

        # Custom scorer that always prefers "second"
        def custom_scorer(response: str) -> float:
            if response == "second":
                return 1.0
            return 0.0

        config = PowerSamplingConfig(
            enable_power_sampling=True,
            n_samples=3,
            alpha=10.0,  # Very high alpha to strongly prefer best
            min_quality_threshold=0.0,
        )
        agent.configure_power_sampling(config, scorer=custom_scorer)

        # With high alpha and custom scorer, should always select "second"
        for _ in range(5):
            agent._call_count = 0
            result = await agent.generate_with_power_sampling("test")
            assert result == "second"


class TestDefaultScorer:
    """Tests for the default quality scorer."""

    def test_empty_response_zero_score(self):
        """Empty responses get zero score."""
        agent = MockAgent()
        assert agent._default_scorer("") == 0.0
        assert agent._default_scorer("   ") == 0.0

    def test_longer_responses_higher_score(self):
        """Longer responses tend to score higher."""
        agent = MockAgent()
        short = agent._default_scorer("Hi")
        medium = agent._default_scorer("This is a medium length response with some content")
        long = agent._default_scorer("This is a much longer response " * 20)

        assert long > medium > short

    def test_reasoning_markers_boost_score(self):
        """Responses with reasoning markers score higher."""
        agent = MockAgent()
        no_markers = agent._default_scorer("Here is an answer")
        with_markers = agent._default_scorer(
            "Here is an answer because the first reason therefore we conclude"
        )

        assert with_markers > no_markers

    def test_structure_markers_boost_score(self):
        """Responses with structure markers score higher."""
        agent = MockAgent()
        plain = agent._default_scorer("Here is some code")
        structured = agent._default_scorer("Here is some code:\n```python\nprint('hi')\n```")

        assert structured > plain


class TestWeightedSample:
    """Tests for weighted sampling."""

    def test_deterministic_when_single_weight(self):
        """Returns the only option when there's just one."""
        agent = MockAgent()
        assert agent._weighted_sample([1.0]) == 0

    def test_respects_weights_distribution(self):
        """Sampling respects weight distribution."""
        agent = MockAgent()

        # Heavy weight on first option
        weights = [0.9, 0.05, 0.05]
        counts = [0, 0, 0]

        for _ in range(100):
            idx = agent._weighted_sample(weights)
            counts[idx] += 1

        # First should be selected most often
        assert counts[0] > counts[1]
        assert counts[0] > counts[2]


class TestIntegrationWithArenaConfig:
    """Tests for integration with ArenaConfig."""

    def test_power_sampling_config_exists(self):
        """PowerSamplingConfig is importable and usable."""
        config = PowerSamplingConfig(
            enable_power_sampling=True,
            n_samples=16,
            alpha=2.5,
        )

        assert config.enable_power_sampling
        assert config.n_samples == 16
        assert config.alpha == 2.5

    def test_default_config_values(self):
        """PowerSamplingConfig has sensible defaults."""
        config = PowerSamplingConfig()

        assert not config.enable_power_sampling  # Disabled by default
        assert config.n_samples == 8
        assert config.alpha == 2.0
        assert config.k_diverse == 3
        assert config.sampling_temperature == 1.0
        assert config.min_quality_threshold == 0.3
        assert not config.enable_for_critiques
        assert config.custom_scorer is None
        assert config.sample_timeout == 30.0
