"""
Tests for matrix debate parallel scenario execution.

Phase 8: Debate Integration Test Gaps - Parallel execution tests.

Tests:
- test_parallel_scenario_execution - asyncio.gather correctness
- test_scenario_failure_isolation - One scenario fails, others complete
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ============================================================================
# Mock Scenario Executor
# ============================================================================


class ScenarioExecutor:
    """Mock scenario executor for testing parallel execution."""

    def __init__(self, failure_scenarios: set[str] | None = None):
        """Initialize executor with optional failure scenarios."""
        self.failure_scenarios = failure_scenarios or set()
        self.execution_order: list[str] = []
        self.completed_scenarios: list[str] = []

    async def execute_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """Execute a single scenario."""
        scenario_name = scenario["name"]
        self.execution_order.append(scenario_name)

        # Simulate some work
        await asyncio.sleep(scenario.get("duration", 0.01))

        if scenario_name in self.failure_scenarios:
            raise ValueError(f"Scenario {scenario_name} failed")

        self.completed_scenarios.append(scenario_name)
        return {
            "scenario_name": scenario_name,
            "consensus_reached": True,
            "final_answer": f"Answer for {scenario_name}",
            "confidence": 0.8,
            "rounds_used": 3,
        }

    async def execute_scenarios_parallel(
        self, scenarios: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute multiple scenarios in parallel using asyncio.gather."""
        tasks = [self.execute_scenario(s) for s in scenarios]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                continue
            valid_results.append(result)

        return valid_results

    async def execute_scenarios_parallel_with_errors(
        self, scenarios: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[tuple[str, Exception]]]:
        """Execute scenarios and collect both results and errors."""
        tasks = [self.execute_scenario(s) for s in scenarios]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        errors = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append((scenarios[i]["name"], result))
            else:
                valid_results.append(result)

        return valid_results, errors


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def executor():
    """Create a fresh scenario executor."""
    return ScenarioExecutor()


@pytest.fixture
def failing_executor():
    """Create an executor that fails on specific scenarios."""
    return ScenarioExecutor(failure_scenarios={"Scenario B"})


@pytest.fixture
def test_scenarios():
    """Create test scenarios."""
    return [
        {"name": "Scenario A", "parameters": {"param": 1}, "duration": 0.02},
        {"name": "Scenario B", "parameters": {"param": 2}, "duration": 0.01},
        {"name": "Scenario C", "parameters": {"param": 3}, "duration": 0.03},
    ]


# ============================================================================
# Test: Parallel Scenario Execution
# ============================================================================


class TestParallelScenarioExecution:
    """Test parallel scenario execution with asyncio.gather."""

    @pytest.mark.asyncio
    async def test_parallel_scenario_execution(self, executor, test_scenarios):
        """Test that all scenarios execute and complete."""
        results = await executor.execute_scenarios_parallel(test_scenarios)

        # All 3 scenarios should complete
        assert len(results) == 3

        # Check all scenario names are in results
        result_names = {r["scenario_name"] for r in results}
        assert result_names == {"Scenario A", "Scenario B", "Scenario C"}

    @pytest.mark.asyncio
    async def test_parallel_execution_returns_all_results(self, executor, test_scenarios):
        """Test that parallel execution returns results for all scenarios."""
        results = await executor.execute_scenarios_parallel(test_scenarios)

        for result in results:
            assert "scenario_name" in result
            assert "consensus_reached" in result
            assert "final_answer" in result
            assert "confidence" in result

    @pytest.mark.asyncio
    async def test_parallel_execution_faster_than_sequential(self, executor):
        """Test that parallel execution is faster than sequential."""
        import time

        scenarios = [{"name": f"Scenario {i}", "duration": 0.05} for i in range(5)]

        start = time.time()
        await executor.execute_scenarios_parallel(scenarios)
        parallel_time = time.time() - start

        # If sequential, would take 5 * 0.05 = 0.25 seconds
        # Parallel should be significantly faster (close to 0.05 seconds + overhead)
        assert parallel_time < 0.2  # Well under sequential time

    @pytest.mark.asyncio
    async def test_all_scenarios_start_execution(self, executor, test_scenarios):
        """Test that all scenarios start execution."""
        await executor.execute_scenarios_parallel(test_scenarios)

        # All scenarios should have been started
        assert len(executor.execution_order) == 3

    @pytest.mark.asyncio
    async def test_empty_scenarios_returns_empty(self, executor):
        """Test that empty scenario list returns empty results."""
        results = await executor.execute_scenarios_parallel([])
        assert results == []


# ============================================================================
# Test: Scenario Failure Isolation
# ============================================================================


class TestScenarioFailureIsolation:
    """Test that one scenario failure doesn't affect others."""

    @pytest.mark.asyncio
    async def test_scenario_failure_isolation(self, failing_executor, test_scenarios):
        """Test that one scenario failure doesn't prevent others from completing."""
        # Scenario B is configured to fail
        results = await failing_executor.execute_scenarios_parallel(test_scenarios)

        # Only 2 results (A and C completed, B failed)
        assert len(results) == 2

        result_names = {r["scenario_name"] for r in results}
        assert "Scenario A" in result_names
        assert "Scenario C" in result_names
        assert "Scenario B" not in result_names

    @pytest.mark.asyncio
    async def test_failure_error_collection(self, failing_executor, test_scenarios):
        """Test that errors are properly collected."""
        results, errors = await failing_executor.execute_scenarios_parallel_with_errors(
            test_scenarios
        )

        # 2 successes, 1 error
        assert len(results) == 2
        assert len(errors) == 1

        # Error should be for Scenario B
        error_name, error = errors[0]
        assert error_name == "Scenario B"
        assert isinstance(error, ValueError)

    @pytest.mark.asyncio
    async def test_partial_failure_preserves_successful_data(
        self, failing_executor, test_scenarios
    ):
        """Test that successful results have complete data despite partial failure."""
        results = await failing_executor.execute_scenarios_parallel(test_scenarios)

        for result in results:
            # Each successful result should have all expected fields
            assert result["consensus_reached"] is True
            assert "final_answer" in result
            assert result["confidence"] == 0.8
            assert result["rounds_used"] == 3

    @pytest.mark.asyncio
    async def test_multiple_failures_isolated(self):
        """Test that multiple failures don't cascade."""
        executor = ScenarioExecutor(failure_scenarios={"Scenario A", "Scenario C"})

        scenarios = [
            {"name": "Scenario A"},
            {"name": "Scenario B"},
            {"name": "Scenario C"},
            {"name": "Scenario D"},
        ]

        results = await executor.execute_scenarios_parallel(scenarios)

        # Only B and D should succeed
        assert len(results) == 2
        result_names = {r["scenario_name"] for r in results}
        assert result_names == {"Scenario B", "Scenario D"}

    @pytest.mark.asyncio
    async def test_all_failures_returns_empty(self):
        """Test that all failures returns empty list."""
        executor = ScenarioExecutor(failure_scenarios={"A", "B", "C"})

        scenarios = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        results = await executor.execute_scenarios_parallel(scenarios)

        assert results == []

    @pytest.mark.asyncio
    async def test_failure_doesnt_stop_other_tasks(self, failing_executor, test_scenarios):
        """Test that a fast-failing task doesn't stop slower tasks."""
        # Scenario B fails but has shortest duration (0.01)
        # Scenario C has longest duration (0.03) but should still complete

        results = await failing_executor.execute_scenarios_parallel(test_scenarios)

        # Scenario C should be in completed scenarios
        assert "Scenario C" in failing_executor.completed_scenarios


# ============================================================================
# Test: Execution Order and Concurrency
# ============================================================================


class TestExecutionOrderAndConcurrency:
    """Test execution order and concurrent behavior."""

    @pytest.mark.asyncio
    async def test_execution_order_not_sequential(self, executor):
        """Test that execution order may differ from input order due to concurrency."""
        # Give scenarios different durations
        scenarios = [
            {"name": "Slow", "duration": 0.03},
            {"name": "Fast", "duration": 0.01},
            {"name": "Medium", "duration": 0.02},
        ]

        results = await executor.execute_scenarios_parallel(scenarios)

        # All should complete regardless of order
        assert len(results) == 3

        # The completion order likely differs from input order
        # (Fast finishes before Slow and Medium)
        assert "Fast" in executor.completed_scenarios
        assert "Slow" in executor.completed_scenarios
        assert "Medium" in executor.completed_scenarios

    @pytest.mark.asyncio
    async def test_concurrent_execution_overlaps(self, executor):
        """Test that scenarios execute concurrently, not sequentially."""
        import time

        # Three scenarios each taking 0.1 seconds
        scenarios = [{"name": f"Scenario {i}", "duration": 0.1} for i in range(3)]

        start = time.time()
        await executor.execute_scenarios_parallel(scenarios)
        elapsed = time.time() - start

        # If concurrent, should take ~0.1 seconds (not 0.3)
        # Allow some overhead
        assert elapsed < 0.2


__all__ = [
    "TestParallelScenarioExecution",
    "TestScenarioFailureIsolation",
    "TestExecutionOrderAndConcurrency",
]
