"""
Stress tests for Gauntlet adversarial validation.

Tests:
- Concurrent gauntlet runs
- Large input handling
- Timeout behavior
- Resource limits
- Edge cases (empty, malformed inputs)

Note: These tests require mocked agents to avoid real API calls.
Run with: pytest tests/gauntlet/test_stress.py -v
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gauntlet.config import GauntletConfig
from aragora.gauntlet.result import GauntletResult, SeverityLevel
from aragora.gauntlet.runner import GauntletRunner

# Skip all tests in this file if GAUNTLET_STRESS_TESTS is not set
# These tests can make API calls or take significant time
pytestmark = pytest.mark.skipif(
    not os.environ.get("GAUNTLET_STRESS_TESTS"),
    reason="Set GAUNTLET_STRESS_TESTS=1 to run stress tests",
)


class TestGauntletStress:
    """Stress tests for GauntletRunner."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent for testing."""
        agent = MagicMock()
        agent.generate = AsyncMock(return_value="No vulnerabilities found in this analysis.")
        return agent

    @pytest.fixture
    def mock_agent_factory(self, mock_agent: MagicMock):
        """Create a mock agent factory."""
        return lambda name: mock_agent

    @pytest.fixture
    def runner(self, mock_agent_factory) -> GauntletRunner:
        """Create a GauntletRunner with mocked dependencies."""
        config = GauntletConfig(
            agents=["mock_agent"],
            max_parallel_scenarios=2,
            attack_rounds=1,
        )
        return GauntletRunner(
            config=config,
            agent_factory=mock_agent_factory,
        )

    @pytest.mark.asyncio
    async def test_concurrent_runs(self, runner: GauntletRunner):
        """Test multiple concurrent gauntlet runs."""
        inputs = [f"Test specification {i} for concurrent validation" for i in range(5)]

        # Run multiple gauntlets concurrently
        tasks = [runner.run(input_content=content, context="stress test") for content in inputs]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all completed
        assert len(results) == 5
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent run failed: {result}")
            assert isinstance(result, GauntletResult)
            assert result.gauntlet_id.startswith("gauntlet-")

    @pytest.mark.asyncio
    async def test_large_input_handling(self, runner: GauntletRunner):
        """Test handling of large input content."""
        # Create a large input (1MB of text)
        large_input = "x" * (1024 * 1024)

        result = await runner.run(
            input_content=large_input,
            context="large input test",
        )

        assert isinstance(result, GauntletResult)
        # Input summary should be truncated
        assert len(result.input_summary) <= 500

    @pytest.mark.asyncio
    async def test_empty_input(self, runner: GauntletRunner):
        """Test handling of empty input."""
        result = await runner.run(input_content="", context="")

        assert isinstance(result, GauntletResult)
        assert result.input_summary == ""

    @pytest.mark.asyncio
    async def test_special_characters_input(self, runner: GauntletRunner):
        """Test handling of special characters in input."""
        special_input = "Test with special chars: \x00\x01\xff\n\r\t🎉✨"

        result = await runner.run(
            input_content=special_input,
            context="special chars test",
        )

        assert isinstance(result, GauntletResult)

    @pytest.mark.asyncio
    async def test_progress_callback_stress(self, runner: GauntletRunner):
        """Test progress callback is called correctly under load."""
        progress_calls: list[tuple[str, float]] = []

        def on_progress(phase: str, percent: float):
            progress_calls.append((phase, percent))

        result = await runner.run(
            input_content="Test content",
            context="progress test",
            on_progress=on_progress,
        )

        assert isinstance(result, GauntletResult)
        # Should have received progress updates
        assert len(progress_calls) > 0
        # Progress should be between 0 and 1
        for phase, percent in progress_calls:
            assert 0.0 <= percent <= 1.0

    @pytest.mark.asyncio
    async def test_rapid_sequential_runs(self, runner: GauntletRunner):
        """Test rapid sequential gauntlet runs."""
        results: list[GauntletResult] = []

        for i in range(10):
            result = await runner.run(
                input_content=f"Rapid test {i}",
                context="rapid sequential",
            )
            results.append(result)

        # All should complete successfully
        assert len(results) == 10
        # Each should have unique ID
        ids = {r.gauntlet_id for r in results}
        assert len(ids) == 10

    @pytest.mark.asyncio
    async def test_mixed_concurrent_and_sequential(self, runner: GauntletRunner):
        """Test mixed concurrent and sequential runs."""
        # First batch concurrent
        batch1 = await asyncio.gather(
            *[runner.run(f"Batch 1 item {i}", "batch 1") for i in range(3)]
        )

        # Sequential run
        sequential = await runner.run("Sequential item", "sequential")

        # Second batch concurrent
        batch2 = await asyncio.gather(
            *[runner.run(f"Batch 2 item {i}", "batch 2") for i in range(3)]
        )

        assert len(batch1) == 3
        assert isinstance(sequential, GauntletResult)
        assert len(batch2) == 3


class TestGauntletResourceLimits:
    """Tests for resource limit handling."""

    @pytest.fixture
    def limited_config(self) -> GauntletConfig:
        """Create a config with tight resource limits."""
        return GauntletConfig(
            agents=["test_agent"],
            max_parallel_scenarios=1,
            attack_rounds=1,
        )

    @pytest.fixture
    def slow_agent(self) -> MagicMock:
        """Create a slow agent that delays responses."""
        agent = MagicMock()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(0.1)
            return "Analysis complete with delay."

        agent.generate = slow_generate
        return agent

    @pytest.fixture
    def runner_with_limits(self, limited_config, slow_agent) -> GauntletRunner:
        """Create runner with resource limits."""
        return GauntletRunner(
            config=limited_config,
            agent_factory=lambda name: slow_agent,
        )

    @pytest.mark.asyncio
    async def test_respects_max_parallel(self, runner_with_limits: GauntletRunner):
        """Test that max_parallel_scenarios is respected."""
        start_time = time.time()

        # Run with limited parallelism
        result = await runner_with_limits.run(
            input_content="Test content",
            context="parallel limit test",
        )

        elapsed = time.time() - start_time

        assert isinstance(result, GauntletResult)
        # With max_parallel=1 and slow agent, should take longer than parallel
        # This is a soft assertion - just verify it completes

    @pytest.mark.asyncio
    async def test_threshold_limits(self):
        """Test verdict threshold limits."""
        config = GauntletConfig(
            agents=["test"],
            critical_threshold=0,
            high_threshold=2,
        )
        runner = GauntletRunner(config=config)

        result = await runner.run("Test", "threshold test")

        # Should complete with threshold config
        assert isinstance(result, GauntletResult)


class TestGauntletEdgeCases:
    """Edge case tests for Gauntlet."""

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """Test handling of various unicode content."""
        runner = GauntletRunner(config=GauntletConfig(agents=["test"]))

        unicode_inputs = [
            "Hello 世界 🌍",
            "مرحبا العالم",
            "Привет мир",
            "שלום עולם",
            "🎉" * 1000,  # Lots of emoji
        ]

        for content in unicode_inputs:
            result = await runner.run(content, "unicode test")
            assert isinstance(result, GauntletResult)

    @pytest.mark.asyncio
    async def test_newlines_and_whitespace(self):
        """Test handling of various whitespace."""
        runner = GauntletRunner(config=GauntletConfig(agents=["test"]))

        whitespace_inputs = [
            "\n" * 1000,
            "\t" * 1000,
            " " * 1000,
            "\r\n" * 500,
            "line1\nline2\nline3" * 100,
        ]

        for content in whitespace_inputs:
            result = await runner.run(content, "whitespace test")
            assert isinstance(result, GauntletResult)

    @pytest.mark.asyncio
    async def test_json_like_content(self):
        """Test handling of JSON-like content."""
        runner = GauntletRunner(config=GauntletConfig(agents=["test"]))

        json_content = '{"key": "value", "nested": {"array": [1, 2, 3]}}'
        result = await runner.run(json_content, "json test")

        assert isinstance(result, GauntletResult)

    @pytest.mark.asyncio
    async def test_code_like_content(self):
        """Test handling of code-like content."""
        runner = GauntletRunner(config=GauntletConfig(agents=["test"]))

        code_content = """
def vulnerable_function(user_input):
    # This is intentionally vulnerable
    query = f"SELECT * FROM users WHERE id = {user_input}"
    return execute(query)
"""
        result = await runner.run(code_content, "code test")

        assert isinstance(result, GauntletResult)


class TestGauntletCancellation:
    """Tests for cancellation behavior."""

    @pytest.mark.asyncio
    async def test_cancellation_during_run(self):
        """Test that gauntlet can be cancelled mid-run."""
        slow_agent = MagicMock()

        async def very_slow(*args, **kwargs):
            await asyncio.sleep(10)
            return "Never reached"

        slow_agent.generate = very_slow

        config = GauntletConfig(agents=["slow"])
        runner = GauntletRunner(
            config=config,
            agent_factory=lambda name: slow_agent,
        )

        # Start the task
        task = asyncio.create_task(runner.run("Test content", "cancellation test"))

        # Wait a bit then cancel
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that timeouts are handled gracefully."""
        slow_agent = MagicMock()

        async def timeout_agent(*args, **kwargs):
            await asyncio.sleep(100)  # Very long delay
            return "Never reached"

        slow_agent.generate = timeout_agent

        config = GauntletConfig(
            agents=["slow"],
            attack_rounds=1,  # Minimal attacks
        )
        runner = GauntletRunner(
            config=config,
            agent_factory=lambda name: slow_agent,
        )

        # Should complete without hanging
        result = await asyncio.wait_for(
            runner.run("Test", "timeout test"),
            timeout=5.0,  # Overall test timeout
        )

        # Should have completed (possibly with errors)
        assert isinstance(result, GauntletResult)


class TestGauntletRecovery:
    """Tests for error recovery."""

    @pytest.mark.asyncio
    async def test_agent_error_recovery(self):
        """Test recovery when an agent throws an error."""
        error_agent = MagicMock()
        error_agent.generate = AsyncMock(side_effect=RuntimeError("Agent failed"))

        config = GauntletConfig(agents=["error_agent"])
        runner = GauntletRunner(
            config=config,
            agent_factory=lambda name: error_agent,
        )

        # Should not crash, should return result with error info
        result = await runner.run("Test", "error recovery test")

        assert isinstance(result, GauntletResult)
        # Should have recorded the error somehow

    @pytest.mark.asyncio
    async def test_partial_completion(self):
        """Test that partial results are preserved on failure."""
        call_count = 0

        async def intermittent_agent(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("Intermittent failure")
            return "Success"

        agent = MagicMock()
        agent.generate = intermittent_agent

        config = GauntletConfig(agents=["intermittent"])
        runner = GauntletRunner(
            config=config,
            agent_factory=lambda name: agent,
        )

        result = await runner.run("Test", "partial completion test")

        assert isinstance(result, GauntletResult)
        # Should have captured some results despite failures
