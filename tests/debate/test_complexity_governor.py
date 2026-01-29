"""
Tests for the Adaptive Complexity Governor.

Tests cover:
- Task complexity classification (classify_task_complexity)
- StressLevel enum
- GovernorConstraints dataclass
- AgentPerformanceMetrics dataclass properties
- AdaptiveComplexityGovernor class (stress transitions, recording, constraints)
- Singleton pattern (get_complexity_governor, reset_complexity_governor)
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from aragora.core import TaskComplexity
from aragora.debate.complexity_governor import (
    COMPLEXITY_TIMEOUT_FACTORS,
    AdaptiveComplexityGovernor,
    AgentPerformanceMetrics,
    GovernorConstraints,
    StressLevel,
    classify_task_complexity,
    get_complexity_governor,
    reset_complexity_governor,
)


class TestClassifyTaskComplexity:
    """Tests for classify_task_complexity function."""

    def test_empty_task_returns_unknown(self):
        """Empty task should return UNKNOWN."""
        assert classify_task_complexity("") == TaskComplexity.UNKNOWN
        assert classify_task_complexity(None) == TaskComplexity.UNKNOWN  # type: ignore[arg-type]

    def test_complex_keywords_detected(self):
        """Tasks with complex keywords should be classified as COMPLEX."""
        complex_tasks = [
            "Prove that this algorithm is correct",
            "Formally verify the implementation",
            "Design a scalable architecture",
            "Optimize the database performance",
            "Analyze the security implications",
            "Implement a distributed system",
            "Refactor the authentication module",
        ]
        for task in complex_tasks:
            assert classify_task_complexity(task) == TaskComplexity.COMPLEX, f"Failed for: {task}"

    def test_simple_keywords_detected(self):
        """Tasks with simple keywords should be classified as SIMPLE."""
        simple_tasks = [
            "What is Python?",
            "Define a variable",
            "List the available options",
            "Quick question about syntax",
            "Simple example of a loop",
            "Which is the capital of France?",
            "Yes or no question",
            "True or false: Python is typed",
            "How many elements are there?",
            "Who is the author?",
            "When did this release?",
            "Where is the configuration?",
        ]
        for task in simple_tasks:
            assert classify_task_complexity(task) == TaskComplexity.SIMPLE, f"Failed for: {task}"

    def test_short_task_classified_as_simple(self):
        """Very short tasks (< 30 chars) should be SIMPLE."""
        short_task = "Fix the typo"  # 12 chars
        assert len(short_task) < 30
        assert classify_task_complexity(short_task) == TaskComplexity.SIMPLE

    def test_long_task_classified_as_complex(self):
        """Very long tasks (> 500 chars) should be COMPLEX."""
        long_task = "a" * 501
        assert len(long_task) > 500
        assert classify_task_complexity(long_task) == TaskComplexity.COMPLEX

    def test_moderate_task_default(self):
        """Tasks without signals and moderate length should be MODERATE."""
        moderate_task = "Please review this code and provide feedback on the general structure"
        assert 30 <= len(moderate_task) <= 500
        assert classify_task_complexity(moderate_task) == TaskComplexity.MODERATE

    def test_complex_keywords_case_insensitive(self):
        """Keyword matching should be case-insensitive."""
        assert classify_task_complexity("DESIGN A SYSTEM") == TaskComplexity.COMPLEX
        assert classify_task_complexity("Prove This") == TaskComplexity.COMPLEX

    def test_simple_define_with_space(self):
        """'define ' with space matches, 'defined' without space doesn't."""
        assert classify_task_complexity("define a function") == TaskComplexity.SIMPLE
        # 'defined' shouldn't trigger simple - will be moderate by length
        result = classify_task_complexity(
            "This is something that was defined earlier and needs work"
        )
        assert result in (TaskComplexity.MODERATE, TaskComplexity.COMPLEX)


class TestStressLevel:
    """Tests for StressLevel enum."""

    def test_stress_level_values(self):
        """Test StressLevel enum values."""
        assert StressLevel.NOMINAL.value == "nominal"
        assert StressLevel.ELEVATED.value == "elevated"
        assert StressLevel.HIGH.value == "high"
        assert StressLevel.CRITICAL.value == "critical"

    def test_stress_level_ordering(self):
        """Test StressLevel ordering for comparisons."""
        levels = list(StressLevel)
        assert levels == [
            StressLevel.NOMINAL,
            StressLevel.ELEVATED,
            StressLevel.HIGH,
            StressLevel.CRITICAL,
        ]


class TestGovernorConstraints:
    """Tests for GovernorConstraints dataclass."""

    def test_default_values(self):
        """Test default constraint values."""
        constraints = GovernorConstraints()

        assert constraints.max_context_tokens == 8000
        assert constraints.max_history_messages == 20
        assert constraints.max_prompt_length == 4000
        assert constraints.max_agents_per_round == 5
        assert constraints.max_critique_length == 1000
        assert constraints.max_proposal_length == 2000
        assert constraints.enable_deep_analysis is True
        assert constraints.enable_cross_references is True
        assert constraints.enable_formal_verification is True

    def test_post_init_sets_timeouts_from_config(self):
        """Test that post_init sets timeouts from config when None."""
        constraints = GovernorConstraints()
        # Timeouts should be set from AGENT_TIMEOUT_SECONDS
        assert constraints.agent_timeout_seconds is not None
        assert constraints.round_timeout_seconds is not None
        assert constraints.round_timeout_seconds == constraints.agent_timeout_seconds * 2

    def test_explicit_timeout_not_overwritten(self):
        """Explicit timeout values should not be overwritten."""
        constraints = GovernorConstraints(
            agent_timeout_seconds=60.0,
            round_timeout_seconds=180.0,
        )
        assert constraints.agent_timeout_seconds == 60.0
        assert constraints.round_timeout_seconds == 180.0

    def test_to_dict_serialization(self):
        """Test to_dict serialization."""
        constraints = GovernorConstraints(
            max_context_tokens=4000,
            agent_timeout_seconds=30.0,
            enable_deep_analysis=False,
        )
        result = constraints.to_dict()

        assert result["max_context_tokens"] == 4000
        assert result["agent_timeout_seconds"] == 30.0
        assert result["enable_deep_analysis"] is False
        assert "max_history_messages" in result
        assert "round_timeout_seconds" in result


class TestAgentPerformanceMetrics:
    """Tests for AgentPerformanceMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = AgentPerformanceMetrics(name="claude")

        assert metrics.name == "claude"
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.timeout_count == 0
        assert metrics.error_count == 0
        assert metrics.total_latency_ms == 0.0

    def test_success_rate_no_requests(self):
        """Success rate should be 1.0 when no requests."""
        metrics = AgentPerformanceMetrics(name="test")
        assert metrics.success_rate == 1.0

    def test_success_rate_calculation(self):
        """Success rate should be calculated correctly."""
        metrics = AgentPerformanceMetrics(
            name="test",
            total_requests=10,
            successful_requests=7,
        )
        assert metrics.success_rate == 0.7

    def test_timeout_rate_no_requests(self):
        """Timeout rate should be 0.0 when no requests."""
        metrics = AgentPerformanceMetrics(name="test")
        assert metrics.timeout_rate == 0.0

    def test_timeout_rate_calculation(self):
        """Timeout rate should be calculated correctly."""
        metrics = AgentPerformanceMetrics(
            name="test",
            total_requests=10,
            timeout_count=3,
        )
        assert metrics.timeout_rate == 0.3

    def test_avg_latency_no_successful_requests(self):
        """Avg latency should be 0.0 when no successful requests."""
        metrics = AgentPerformanceMetrics(name="test")
        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self):
        """Avg latency should be calculated correctly."""
        metrics = AgentPerformanceMetrics(
            name="test",
            successful_requests=5,
            total_latency_ms=5000.0,
        )
        assert metrics.avg_latency_ms == 1000.0


class TestAdaptiveComplexityGovernor:
    """Tests for AdaptiveComplexityGovernor class."""

    def test_init_default_constraints(self):
        """Governor should use default constraints on init."""
        governor = AdaptiveComplexityGovernor()

        assert governor.stress_level == StressLevel.NOMINAL
        assert isinstance(governor.current_constraints, GovernorConstraints)
        assert governor.consecutive_failures == 0
        assert governor.task_complexity == TaskComplexity.MODERATE

    def test_init_custom_constraints(self):
        """Governor should accept custom initial constraints."""
        custom = GovernorConstraints(max_context_tokens=4000)
        governor = AdaptiveComplexityGovernor(initial_constraints=custom)

        assert governor.current_constraints.max_context_tokens == 4000

    def test_record_agent_response_success(self):
        """Successful response should update metrics correctly."""
        governor = AdaptiveComplexityGovernor()
        governor.record_agent_response("claude", latency_ms=1500, success=True)

        metrics = governor.agent_metrics["claude"]
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.total_latency_ms == 1500
        assert governor.consecutive_failures == 0

    def test_record_agent_response_failure(self):
        """Failed response should update metrics and consecutive failures."""
        governor = AdaptiveComplexityGovernor()
        governor.record_agent_response("claude", latency_ms=0, success=False)

        metrics = governor.agent_metrics["claude"]
        assert metrics.total_requests == 1
        assert metrics.error_count == 1
        assert governor.consecutive_failures == 1

    def test_record_agent_timeout(self):
        """Timeout should update timeout count and consecutive failures."""
        governor = AdaptiveComplexityGovernor()
        governor.record_agent_timeout("deepseek", timeout_seconds=180)

        metrics = governor.agent_metrics["deepseek"]
        assert metrics.total_requests == 1
        assert metrics.timeout_count == 1
        assert governor.consecutive_failures == 1

    def test_record_round_complete(self):
        """Round completion should be recorded in history."""
        governor = AdaptiveComplexityGovernor()
        governor.record_round_complete(
            round_id=1,
            duration_seconds=120,
            agents_participated=3,
            agents_failed=0,
        )

        assert len(governor.round_history) == 1
        assert governor.round_history[0]["round_id"] == 1
        assert governor.round_history[0]["duration_seconds"] == 120

    def test_round_history_bounded(self):
        """Round history should be bounded to prevent memory issues."""
        governor = AdaptiveComplexityGovernor()

        # Add 110 rounds (exceeds 100 limit)
        for i in range(110):
            governor.record_round_complete(i, 60, 3, 0)

        # Should be trimmed to 50
        assert len(governor.round_history) == 50

    def test_stress_level_escalation_on_consecutive_failures(self):
        """Consecutive failures should escalate stress level."""
        governor = AdaptiveComplexityGovernor()
        governor.adjustment_cooldown_seconds = 0  # Disable cooldown for testing

        # Two consecutive failures should escalate to HIGH
        governor.record_agent_response("agent1", latency_ms=0, success=False)
        governor.record_agent_response("agent2", latency_ms=0, success=False)

        assert governor.stress_level == StressLevel.HIGH

    def test_stress_level_from_timeout_rate(self):
        """High timeout rate should escalate stress level."""
        governor = AdaptiveComplexityGovernor()
        governor.adjustment_cooldown_seconds = 0

        # Create high timeout rate (>30% for critical)
        for i in range(10):
            if i < 4:  # 40% timeout rate
                governor.record_agent_timeout(f"agent{i}", timeout_seconds=60)
            else:
                governor.record_agent_response(f"agent{i}", latency_ms=1000, success=True)

        assert governor.stress_level == StressLevel.CRITICAL

    def test_stress_callback_invoked(self):
        """Stress callback should be invoked on level change."""
        callback_levels = []

        def callback(level: StressLevel):
            callback_levels.append(level)

        governor = AdaptiveComplexityGovernor(stress_callback=callback)
        governor.adjustment_cooldown_seconds = 0

        # Trigger stress escalation
        governor.record_agent_response("a", latency_ms=0, success=False)
        governor.record_agent_response("b", latency_ms=0, success=False)

        assert len(callback_levels) > 0
        assert StressLevel.HIGH in callback_levels

    def test_get_constraints_returns_current(self):
        """get_constraints should return current constraints."""
        governor = AdaptiveComplexityGovernor()
        constraints = governor.get_constraints()

        assert constraints == governor.current_constraints

    def test_get_agent_constraints_insufficient_data(self):
        """Agent constraints should use defaults when insufficient data."""
        governor = AdaptiveComplexityGovernor()
        constraints = governor.get_agent_constraints("new_agent")

        assert "timeout_seconds" in constraints
        assert "max_response_tokens" in constraints

    def test_get_agent_constraints_adjusted_for_slow_agent(self):
        """Slow agents should get adjusted constraints."""
        governor = AdaptiveComplexityGovernor()

        # Record some data for the agent
        for _ in range(5):
            governor.record_agent_response("slow_agent", latency_ms=50000, success=True)

        constraints = governor.get_agent_constraints("slow_agent")

        # Should have reliability score
        assert "reliability_score" in constraints
        assert constraints["reliability_score"] == 1.0

    def test_should_reduce_complexity(self):
        """should_reduce_complexity should return True when stress is HIGH or CRITICAL."""
        governor = AdaptiveComplexityGovernor()

        governor.stress_level = StressLevel.NOMINAL
        assert governor.should_reduce_complexity() is False

        governor.stress_level = StressLevel.ELEVATED
        assert governor.should_reduce_complexity() is False

        governor.stress_level = StressLevel.HIGH
        assert governor.should_reduce_complexity() is True

        governor.stress_level = StressLevel.CRITICAL
        assert governor.should_reduce_complexity() is True

    def test_should_skip_agent_insufficient_data(self):
        """should_skip_agent returns False when insufficient data."""
        governor = AdaptiveComplexityGovernor()
        assert governor.should_skip_agent("unknown_agent") is False

    def test_should_skip_agent_high_timeout_rate(self):
        """Agents with >70% timeout rate should be skipped."""
        governor = AdaptiveComplexityGovernor()

        # Create agent with high timeout rate
        for i in range(10):
            if i < 8:  # 80% timeout
                governor.record_agent_timeout("bad_agent", timeout_seconds=60)
            else:
                governor.record_agent_response("bad_agent", latency_ms=1000, success=True)

        assert governor.should_skip_agent("bad_agent") is True

    def test_get_recommended_agent_count(self):
        """Should return max_agents_per_round from current constraints."""
        governor = AdaptiveComplexityGovernor()
        count = governor.get_recommended_agent_count()

        assert count == governor.current_constraints.max_agents_per_round

    def test_set_task_complexity(self):
        """set_task_complexity should update the task complexity."""
        governor = AdaptiveComplexityGovernor()

        governor.set_task_complexity(TaskComplexity.COMPLEX)
        assert governor.task_complexity == TaskComplexity.COMPLEX

        governor.set_task_complexity(TaskComplexity.SIMPLE)
        assert governor.task_complexity == TaskComplexity.SIMPLE

    def test_get_scaled_timeout_simple_task(self):
        """Simple tasks should get shorter timeouts."""
        governor = AdaptiveComplexityGovernor()
        governor.set_task_complexity(TaskComplexity.SIMPLE)

        scaled = governor.get_scaled_timeout(base_timeout=180.0)

        # Simple factor is 0.5
        assert scaled <= 180.0 * 0.5

    def test_get_scaled_timeout_complex_task(self):
        """Complex tasks should get longer timeouts (up to stress limit)."""
        governor = AdaptiveComplexityGovernor()
        governor.set_task_complexity(TaskComplexity.COMPLEX)

        scaled = governor.get_scaled_timeout(base_timeout=180.0)

        # Complex factor is 1.5, but capped by stress constraints
        expected = min(
            180.0 * COMPLEXITY_TIMEOUT_FACTORS[TaskComplexity.COMPLEX],
            governor.current_constraints.agent_timeout_seconds,  # type: ignore[arg-type]
        )
        assert scaled == expected

    def test_get_status(self):
        """get_status should return comprehensive status dict."""
        governor = AdaptiveComplexityGovernor()
        governor.record_agent_response("claude", latency_ms=1500, success=True)
        governor.record_round_complete(1, 60, 3, 0)

        status = governor.get_status()

        assert status["stress_level"] == "nominal"
        assert status["task_complexity"] == "moderate"
        assert "constraints" in status
        assert "agent_metrics" in status
        assert "claude" in status["agent_metrics"]
        assert "recent_rounds" in status

    def test_reset_metrics(self):
        """reset_metrics should clear all state."""
        governor = AdaptiveComplexityGovernor()
        governor.record_agent_response("claude", latency_ms=1500, success=False)
        governor.record_agent_response("gpt", latency_ms=0, success=False)

        governor.reset_metrics()

        assert len(governor.agent_metrics) == 0
        assert len(governor.round_history) == 0
        assert governor.consecutive_failures == 0
        assert governor.stress_level == StressLevel.NOMINAL
        assert governor.task_complexity == TaskComplexity.MODERATE


class TestComplexityTimeoutFactors:
    """Tests for COMPLEXITY_TIMEOUT_FACTORS constant."""

    def test_all_complexities_have_factors(self):
        """All TaskComplexity values should have timeout factors."""
        for complexity in TaskComplexity:
            assert complexity in COMPLEXITY_TIMEOUT_FACTORS

    def test_factor_ordering(self):
        """Factors should increase with complexity."""
        assert (
            COMPLEXITY_TIMEOUT_FACTORS[TaskComplexity.SIMPLE]
            < COMPLEXITY_TIMEOUT_FACTORS[TaskComplexity.MODERATE]
        )
        assert (
            COMPLEXITY_TIMEOUT_FACTORS[TaskComplexity.MODERATE]
            < COMPLEXITY_TIMEOUT_FACTORS[TaskComplexity.COMPLEX]
        )


class TestConstraintPresets:
    """Tests for CONSTRAINT_PRESETS in AdaptiveComplexityGovernor."""

    def test_all_stress_levels_have_presets(self):
        """All stress levels should have constraint presets."""
        for level in StressLevel:
            assert level in AdaptiveComplexityGovernor.CONSTRAINT_PRESETS

    def test_presets_decrease_with_stress(self):
        """Higher stress should have more restrictive constraints."""
        nominal = AdaptiveComplexityGovernor.CONSTRAINT_PRESETS[StressLevel.NOMINAL]
        elevated = AdaptiveComplexityGovernor.CONSTRAINT_PRESETS[StressLevel.ELEVATED]
        high = AdaptiveComplexityGovernor.CONSTRAINT_PRESETS[StressLevel.HIGH]
        critical = AdaptiveComplexityGovernor.CONSTRAINT_PRESETS[StressLevel.CRITICAL]

        # Context tokens should decrease
        assert nominal.max_context_tokens > elevated.max_context_tokens
        assert elevated.max_context_tokens > high.max_context_tokens
        assert high.max_context_tokens > critical.max_context_tokens

        # Agent count should decrease
        assert nominal.max_agents_per_round > critical.max_agents_per_round


class TestSingletonPattern:
    """Tests for singleton pattern functions."""

    def test_get_complexity_governor_returns_same_instance(self):
        """get_complexity_governor should return same instance."""
        reset_complexity_governor()  # Start fresh

        governor1 = get_complexity_governor()
        governor2 = get_complexity_governor()

        assert governor1 is governor2

    def test_reset_complexity_governor_clears_instance(self):
        """reset_complexity_governor should clear the global instance."""
        governor1 = get_complexity_governor()
        reset_complexity_governor()
        governor2 = get_complexity_governor()

        assert governor1 is not governor2

    def test_state_persists_in_singleton(self):
        """State should persist across get_complexity_governor calls."""
        reset_complexity_governor()

        governor1 = get_complexity_governor()
        governor1.record_agent_response("test", latency_ms=100, success=True)

        governor2 = get_complexity_governor()
        assert "test" in governor2.agent_metrics


class TestStressThresholds:
    """Tests for stress threshold constants."""

    def test_threshold_values_are_reasonable(self):
        """Threshold values should be in reasonable ranges."""
        thresholds = AdaptiveComplexityGovernor.STRESS_THRESHOLDS

        # Timeout rates should be between 0 and 1
        assert 0 < thresholds["timeout_rate_elevated"] < 1
        assert 0 < thresholds["timeout_rate_high"] < 1
        assert 0 < thresholds["timeout_rate_critical"] < 1

        # Thresholds should increase
        assert thresholds["timeout_rate_elevated"] < thresholds["timeout_rate_high"]
        assert thresholds["timeout_rate_high"] < thresholds["timeout_rate_critical"]

        # Latency thresholds should be positive
        assert thresholds["latency_elevated_ms"] > 0
        assert thresholds["latency_high_ms"] > 0
        assert thresholds["latency_elevated_ms"] < thresholds["latency_high_ms"]
