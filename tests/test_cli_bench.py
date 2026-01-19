"""
Tests for CLI bench module.

Tests benchmark result dataclass and benchmark functions.
"""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.bench import (
    BENCHMARK_TASKS,
    BenchmarkResult,
    benchmark_agent,
    cmd_bench,
    print_result,
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation_with_defaults(self):
        """BenchmarkResult can be created with minimal args."""
        result = BenchmarkResult(
            agent="test-agent",
            task="Test task",
            iterations=5,
        )
        assert result.agent == "test-agent"
        assert result.task == "Test task"
        assert result.iterations == 5
        assert result.response_times == []
        assert result.token_counts == []
        assert result.errors == 0
        assert result.success_rate == 0.0

    def test_avg_response_time_empty(self):
        """avg_response_time returns 0 for empty list."""
        result = BenchmarkResult(agent="a", task="t", iterations=1)
        assert result.avg_response_time == 0.0

    def test_avg_response_time_calculated(self):
        """avg_response_time calculates mean correctly."""
        result = BenchmarkResult(agent="a", task="t", iterations=3)
        result.response_times = [1.0, 2.0, 3.0]
        assert result.avg_response_time == 2.0

    def test_p50_response_time_empty(self):
        """p50_response_time returns 0 for empty list."""
        result = BenchmarkResult(agent="a", task="t", iterations=1)
        assert result.p50_response_time == 0.0

    def test_p50_response_time_calculated(self):
        """p50_response_time calculates median correctly."""
        result = BenchmarkResult(agent="a", task="t", iterations=5)
        result.response_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert result.p50_response_time == 3.0

    def test_p95_response_time_empty(self):
        """p95_response_time returns avg for small samples."""
        result = BenchmarkResult(agent="a", task="t", iterations=1)
        result.response_times = [1.0]
        assert result.p95_response_time == 1.0

    def test_p95_response_time_calculated(self):
        """p95_response_time returns high percentile value."""
        result = BenchmarkResult(agent="a", task="t", iterations=20)
        result.response_times = list(range(1, 21))  # 1-20
        p95 = result.p95_response_time
        # P95 of [1-20] should be around 19
        assert p95 >= 18

    def test_avg_tokens_empty(self):
        """avg_tokens returns 0 for empty list."""
        result = BenchmarkResult(agent="a", task="t", iterations=1)
        assert result.avg_tokens == 0.0

    def test_avg_tokens_calculated(self):
        """avg_tokens calculates mean correctly."""
        result = BenchmarkResult(agent="a", task="t", iterations=3)
        result.token_counts = [100, 200, 300]
        assert result.avg_tokens == 200.0


class TestBenchmarkTasks:
    """Tests for BENCHMARK_TASKS constant."""

    def test_tasks_not_empty(self):
        """BENCHMARK_TASKS is not empty."""
        assert len(BENCHMARK_TASKS) > 0

    def test_all_tasks_are_strings(self):
        """All benchmark tasks are strings."""
        for task in BENCHMARK_TASKS:
            assert isinstance(task, str)

    def test_all_tasks_are_questions(self):
        """All benchmark tasks look like questions/prompts."""
        question_words = ["what", "explain", "name", "how", "why", "describe"]
        for task in BENCHMARK_TASKS:
            task_lower = task.lower()
            has_question = any(word in task_lower for word in question_words)
            assert has_question, f"Task doesn't look like a question: {task}"


class TestBenchmarkAgent:
    """Tests for benchmark_agent function.

    Note: These tests use a non-existent agent type to test error handling.
    The function handles agent creation failures gracefully.
    """

    @pytest.mark.asyncio
    async def test_benchmark_agent_invalid_type_handled(self):
        """Invalid agent type is handled gracefully."""
        result = await benchmark_agent(
            agent_type="nonexistent-agent-type-xyz",
            task="Test task",
            iterations=2,
        )

        # Agent creation should fail, all iterations as errors
        assert result.errors == 2
        assert len(result.response_times) == 0

    @pytest.mark.asyncio
    async def test_benchmark_result_structure(self):
        """benchmark_agent returns correct result structure."""
        result = await benchmark_agent(
            agent_type="nonexistent-agent",
            task="Test task for structure check",
            iterations=1,
        )

        # Verify result has expected attributes
        assert hasattr(result, "agent")
        assert hasattr(result, "task")
        assert hasattr(result, "iterations")
        assert hasattr(result, "response_times")
        assert hasattr(result, "errors")
        assert hasattr(result, "success_rate")

    @pytest.mark.asyncio
    async def test_benchmark_truncates_task(self):
        """Long tasks are truncated in result."""
        long_task = "A" * 100

        result = await benchmark_agent(
            agent_type="nonexistent-agent",
            task=long_task,
            iterations=1,
        )

        # Task should be truncated to 50 chars
        assert len(result.task) <= 50


class TestPrintResult:
    """Tests for print_result function."""

    def test_prints_agent_name(self, capsys):
        """print_result shows agent name."""
        result = BenchmarkResult(
            agent="test-agent",
            task="Test task",
            iterations=3,
        )
        result.success_rate = 1.0

        print_result(result)

        captured = capsys.readouterr()
        assert "test-agent" in captured.out

    def test_prints_success_rate(self, capsys):
        """print_result shows success rate."""
        result = BenchmarkResult(
            agent="test-agent",
            task="Test task",
            iterations=3,
        )
        result.success_rate = 0.667

        print_result(result)

        captured = capsys.readouterr()
        assert "66" in captured.out or "67" in captured.out  # 66.7%

    def test_prints_response_times_when_available(self, capsys):
        """print_result shows response times when available."""
        result = BenchmarkResult(
            agent="test-agent",
            task="Test task",
            iterations=3,
        )
        result.response_times = [1.0, 2.0, 3.0]
        result.success_rate = 1.0

        print_result(result)

        captured = capsys.readouterr()
        assert "Response Time" in captured.out
        assert "Avg" in captured.out
        assert "P50" in captured.out
        assert "P95" in captured.out

    def test_prints_errors_when_present(self, capsys):
        """print_result shows error count when present."""
        result = BenchmarkResult(
            agent="test-agent",
            task="Test task",
            iterations=5,
        )
        result.errors = 2
        result.success_rate = 0.6

        print_result(result)

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "2" in captured.out


class TestCmdBench:
    """Tests for cmd_bench CLI command."""

    @patch("aragora.cli.bench.asyncio.run")
    def test_cmd_bench_parses_agents(self, mock_run, capsys):
        """cmd_bench parses comma-separated agents."""
        mock_run.return_value = None  # Will fail gracefully

        args = argparse.Namespace(
            agents="agent1,agent2",
            iterations=1,
            task="Test",
            quick=False,
        )

        cmd_bench(args)

        captured = capsys.readouterr()
        assert "agent1" in captured.out
        assert "agent2" in captured.out

    @patch("aragora.cli.bench.asyncio.run")
    def test_cmd_bench_quick_mode(self, mock_run, capsys):
        """cmd_bench quick mode sets iterations to 1."""
        args = argparse.Namespace(
            agents="agent1",
            iterations=5,  # Will be overridden
            task=None,
            quick=True,
        )

        cmd_bench(args)

        captured = capsys.readouterr()
        assert "Iterations: 1" in captured.out

    @patch("aragora.cli.bench.asyncio.run")
    def test_cmd_bench_prints_header(self, mock_run, capsys):
        """cmd_bench prints benchmark header."""
        args = argparse.Namespace(
            agents="test-api",
            iterations=2,
            task="Custom task",
            quick=False,
        )

        cmd_bench(args)

        captured = capsys.readouterr()
        assert "Aragora Agent Benchmark" in captured.out
        assert "Agents:" in captured.out
        assert "Iterations:" in captured.out

    @patch("aragora.cli.bench.asyncio.run")
    def test_cmd_bench_keyboard_interrupt(self, mock_run, capsys):
        """cmd_bench handles keyboard interrupt."""
        mock_run.side_effect = KeyboardInterrupt()

        args = argparse.Namespace(
            agents="test-api",
            iterations=2,
            task=None,
            quick=False,
        )

        cmd_bench(args)  # Should not raise

        captured = capsys.readouterr()
        assert "interrupted" in captured.out.lower()
