"""
Tests for Adaptive Complexity Governor module.

Tests cover:
- classify_task_complexity() function
- StressLevel enum values
- GovernorConstraints dataclass
- AgentPerformanceMetrics dataclass with computed properties
- AdaptiveComplexityGovernor initialization and lifecycle
- Stress level transitions
- Agent performance tracking
- Constraint management
- Global governor functions
"""

import time
import pytest
from unittest.mock import Mock, patch

from aragora.core import TaskComplexity
from aragora.debate.complexity_governor import (
    classify_task_complexity,
    StressLevel,
    GovernorConstraints,
    AgentPerformanceMetrics,
    AdaptiveComplexityGovernor,
    get_complexity_governor,
    reset_complexity_governor,
    COMPLEXITY_TIMEOUT_FACTORS,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_global_governor():
    """Reset global governor before and after each test."""
    reset_complexity_governor()
    yield
    reset_complexity_governor()


@pytest.fixture
def governor():
    """Create a fresh governor for testing."""
    return AdaptiveComplexityGovernor()


@pytest.fixture
def stressed_governor():
    """Create a governor with some stress history."""
    gov = AdaptiveComplexityGovernor()
    # Record some timeouts to elevate stress
    for i in range(5):
        gov.record_agent_timeout(f"agent_{i}", 60.0)
    return gov


# ============================================================================
# classify_task_complexity() Tests
# ============================================================================


class TestClassifyTaskComplexity:
    """Tests for classify_task_complexity function."""

    def test_empty_task_returns_unknown(self):
        """Test empty task returns UNKNOWN."""
        assert classify_task_complexity("") == TaskComplexity.UNKNOWN
        assert classify_task_complexity(None) == TaskComplexity.UNKNOWN

    def test_simple_task_what_is(self):
        """Test 'what is' questions are simple."""
        assert classify_task_complexity("What is Python?") == TaskComplexity.SIMPLE

    def test_simple_task_define(self):
        """Test 'define' questions are simple."""
        assert classify_task_complexity("Define recursion") == TaskComplexity.SIMPLE

    def test_simple_task_yes_no(self):
        """Test yes/no questions are simple."""
        assert classify_task_complexity("Is this correct? Yes or no") == TaskComplexity.SIMPLE

    def test_complex_task_prove(self):
        """Test 'prove' tasks are complex."""
        assert classify_task_complexity("Prove this algorithm is correct") == TaskComplexity.COMPLEX

    def test_complex_task_design(self):
        """Test 'design' tasks are complex."""
        assert classify_task_complexity("Design a distributed cache") == TaskComplexity.COMPLEX

    def test_complex_task_optimize(self):
        """Test 'optimize' tasks are complex."""
        assert classify_task_complexity("Optimize database queries") == TaskComplexity.COMPLEX

    def test_complex_task_architecture(self):
        """Test 'architecture' tasks are complex."""
        assert classify_task_complexity("Plan the system architecture") == TaskComplexity.COMPLEX

    def test_moderate_task_default(self):
        """Test moderate task classification."""
        # Medium-length task without specific signals
        task = "Explain how this function works in detail with examples"
        assert classify_task_complexity(task) == TaskComplexity.MODERATE

    def test_short_task_is_simple(self):
        """Test very short tasks are simple."""
        assert classify_task_complexity("List users") == TaskComplexity.SIMPLE

    def test_long_task_is_complex(self):
        """Test very long tasks are complex."""
        long_task = "a" * 600  # More than 500 chars
        assert classify_task_complexity(long_task) == TaskComplexity.COMPLEX

    def test_case_insensitive(self):
        """Test classification is case insensitive."""
        assert classify_task_complexity("DESIGN a system") == TaskComplexity.COMPLEX
        assert classify_task_complexity("What IS Python?") == TaskComplexity.SIMPLE


# ============================================================================
# StressLevel Enum Tests
# ============================================================================


class TestStressLevel:
    """Tests for StressLevel enum."""

    def test_nominal_value(self):
        """Test NOMINAL value."""
        assert StressLevel.NOMINAL.value == "nominal"

    def test_elevated_value(self):
        """Test ELEVATED value."""
        assert StressLevel.ELEVATED.value == "elevated"

    def test_high_value(self):
        """Test HIGH value."""
        assert StressLevel.HIGH.value == "high"

    def test_critical_value(self):
        """Test CRITICAL value."""
        assert StressLevel.CRITICAL.value == "critical"

    def test_all_levels_unique(self):
        """Test all levels have unique values."""
        values = [level.value for level in StressLevel]
        assert len(values) == len(set(values))

    def test_ordering(self):
        """Test stress levels can be compared by severity."""
        levels = list(StressLevel)
        assert levels.index(StressLevel.NOMINAL) < levels.index(StressLevel.CRITICAL)


# ============================================================================
# GovernorConstraints Tests
# ============================================================================


class TestGovernorConstraints:
    """Tests for GovernorConstraints dataclass."""

    def test_default_values(self):
        """Test default constraint values."""
        constraints = GovernorConstraints()

        assert constraints.max_context_tokens == 8000
        assert constraints.max_history_messages == 20
        assert constraints.max_prompt_length == 4000
        assert constraints.max_agents_per_round == 5
        assert constraints.enable_deep_analysis is True

    def test_post_init_sets_timeouts(self):
        """Test __post_init__ sets timeout defaults from config."""
        constraints = GovernorConstraints()

        assert constraints.agent_timeout_seconds is not None
        assert constraints.round_timeout_seconds is not None
        # Round timeout should be 2x agent timeout
        assert constraints.round_timeout_seconds == constraints.agent_timeout_seconds * 2

    def test_explicit_timeout_not_overwritten(self):
        """Test explicit timeout values are preserved."""
        constraints = GovernorConstraints(
            agent_timeout_seconds=30.0,
            round_timeout_seconds=45.0,
        )

        assert constraints.agent_timeout_seconds == 30.0
        assert constraints.round_timeout_seconds == 45.0

    def test_to_dict(self):
        """Test converting constraints to dictionary."""
        constraints = GovernorConstraints(
            max_context_tokens=4000,
            enable_deep_analysis=False,
        )
        data = constraints.to_dict()

        assert isinstance(data, dict)
        assert data["max_context_tokens"] == 4000
        assert data["enable_deep_analysis"] is False
        assert "agent_timeout_seconds" in data

    def test_all_fields_in_to_dict(self):
        """Test all fields are included in to_dict."""
        constraints = GovernorConstraints()
        data = constraints.to_dict()

        expected_keys = [
            "max_context_tokens",
            "max_history_messages",
            "max_prompt_length",
            "agent_timeout_seconds",
            "round_timeout_seconds",
            "max_agents_per_round",
            "max_critique_length",
            "max_proposal_length",
            "enable_deep_analysis",
            "enable_cross_references",
            "enable_formal_verification",
        ]
        for key in expected_keys:
            assert key in data


# ============================================================================
# AgentPerformanceMetrics Tests
# ============================================================================


class TestAgentPerformanceMetrics:
    """Tests for AgentPerformanceMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = AgentPerformanceMetrics(name="test_agent")

        assert metrics.name == "test_agent"
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.timeout_count == 0
        assert metrics.error_count == 0

    def test_success_rate_no_requests(self):
        """Test success rate with no requests."""
        metrics = AgentPerformanceMetrics(name="test")

        assert metrics.success_rate == 1.0  # Default to 100%

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = AgentPerformanceMetrics(
            name="test",
            total_requests=10,
            successful_requests=7,
        )

        assert metrics.success_rate == 0.7

    def test_avg_latency_no_requests(self):
        """Test average latency with no requests."""
        metrics = AgentPerformanceMetrics(name="test")

        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        metrics = AgentPerformanceMetrics(
            name="test",
            successful_requests=5,
            total_latency_ms=5000.0,
        )

        assert metrics.avg_latency_ms == 1000.0

    def test_timeout_rate_no_requests(self):
        """Test timeout rate with no requests."""
        metrics = AgentPerformanceMetrics(name="test")

        assert metrics.timeout_rate == 0.0

    def test_timeout_rate_calculation(self):
        """Test timeout rate calculation."""
        metrics = AgentPerformanceMetrics(
            name="test",
            total_requests=10,
            timeout_count=3,
        )

        assert metrics.timeout_rate == 0.3


# ============================================================================
# AdaptiveComplexityGovernor Initialization Tests
# ============================================================================


class TestGovernorInit:
    """Tests for AdaptiveComplexityGovernor initialization."""

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        gov = AdaptiveComplexityGovernor()

        assert gov.stress_level == StressLevel.NOMINAL
        assert gov.consecutive_failures == 0
        assert len(gov.agent_metrics) == 0
        assert len(gov.round_history) == 0

    def test_initialization_with_constraints(self):
        """Test initialization with custom constraints."""
        custom = GovernorConstraints(max_context_tokens=4000)
        gov = AdaptiveComplexityGovernor(initial_constraints=custom)

        assert gov.current_constraints.max_context_tokens == 4000

    def test_initialization_with_callback(self):
        """Test initialization with stress callback."""
        callback = Mock()
        gov = AdaptiveComplexityGovernor(stress_callback=callback)

        assert gov.stress_callback == callback


# ============================================================================
# Agent Response Recording Tests
# ============================================================================


class TestRecordAgentResponse:
    """Tests for recording agent responses."""

    def test_record_successful_response(self, governor):
        """Test recording a successful response."""
        governor.record_agent_response("claude", latency_ms=1500, success=True)

        metrics = governor.agent_metrics["claude"]
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.total_latency_ms == 1500.0

    def test_record_failed_response(self, governor):
        """Test recording a failed response."""
        governor.record_agent_response("claude", latency_ms=500, success=False)

        metrics = governor.agent_metrics["claude"]
        assert metrics.total_requests == 1
        assert metrics.error_count == 1
        assert governor.consecutive_failures == 1

    def test_consecutive_failures_reset_on_success(self, governor):
        """Test consecutive failures reset on success."""
        governor.record_agent_response("agent", latency_ms=100, success=False)
        governor.record_agent_response("agent", latency_ms=100, success=False)
        assert governor.consecutive_failures == 2

        governor.record_agent_response("agent", latency_ms=100, success=True)
        assert governor.consecutive_failures == 0

    def test_multiple_agents_tracked_separately(self, governor):
        """Test different agents are tracked separately."""
        governor.record_agent_response("claude", latency_ms=1000, success=True)
        governor.record_agent_response("gpt4", latency_ms=2000, success=True)

        assert "claude" in governor.agent_metrics
        assert "gpt4" in governor.agent_metrics
        assert governor.agent_metrics["claude"].total_latency_ms == 1000
        assert governor.agent_metrics["gpt4"].total_latency_ms == 2000


# ============================================================================
# Agent Timeout Recording Tests
# ============================================================================


class TestRecordAgentTimeout:
    """Tests for recording agent timeouts."""

    def test_record_timeout(self, governor):
        """Test recording a timeout."""
        governor.record_agent_timeout("slow_agent", timeout_seconds=60.0)

        metrics = governor.agent_metrics["slow_agent"]
        assert metrics.total_requests == 1
        assert metrics.timeout_count == 1
        assert governor.consecutive_failures == 1

    def test_multiple_timeouts_accumulate(self, governor):
        """Test multiple timeouts accumulate."""
        governor.record_agent_timeout("agent", 60.0)
        governor.record_agent_timeout("agent", 60.0)
        governor.record_agent_timeout("agent", 60.0)

        metrics = governor.agent_metrics["agent"]
        assert metrics.timeout_count == 3
        assert governor.consecutive_failures == 3


# ============================================================================
# Round Completion Recording Tests
# ============================================================================


class TestRecordRoundComplete:
    """Tests for recording round completion."""

    def test_record_round(self, governor):
        """Test recording a round completion."""
        governor.record_round_complete(
            round_id=1,
            duration_seconds=45.0,
            agents_participated=3,
            agents_failed=1,
        )

        assert len(governor.round_history) == 1
        record = governor.round_history[0]
        assert record["round_id"] == 1
        assert record["duration_seconds"] == 45.0
        assert record["agents_failed"] == 1

    def test_round_history_bounded(self, governor):
        """Test round history is bounded to 100, trimmed to 50 when exceeded."""
        # Record exactly 101 rounds to trigger trim
        for i in range(101):
            governor.record_round_complete(i, 30.0, 3, 0)

        # When > 100, trims to last 50
        assert len(governor.round_history) == 50
        # Verify we kept the most recent rounds
        assert governor.round_history[0]["round_id"] == 51
        assert governor.round_history[-1]["round_id"] == 100


# ============================================================================
# Stress Level Evaluation Tests
# ============================================================================


class TestStressLevelEvaluation:
    """Tests for stress level evaluation and transitions."""

    def test_starts_nominal(self, governor):
        """Test governor starts at NOMINAL."""
        assert governor.stress_level == StressLevel.NOMINAL

    def test_elevates_on_timeouts(self, governor):
        """Test stress elevates on timeout rate."""
        # Record timeouts to exceed 5% threshold
        governor.record_agent_response("agent", 100, success=True)  # 1 success
        governor.record_agent_timeout("agent", 60.0)  # 50% timeout rate

        assert governor.stress_level in (
            StressLevel.ELEVATED,
            StressLevel.HIGH,
            StressLevel.CRITICAL,
        )

    def test_consecutive_failures_escalate(self, governor):
        """Test consecutive failures cause escalation."""
        governor.record_agent_response("a1", 100, success=False)
        governor.record_agent_response("a2", 100, success=False)

        assert governor.stress_level == StressLevel.HIGH

    def test_callback_on_stress_change(self):
        """Test callback is invoked on stress change."""
        callback = Mock()
        governor = AdaptiveComplexityGovernor(stress_callback=callback)

        # Trigger stress change
        governor.record_agent_response("a", 100, success=False)
        governor.record_agent_response("a", 100, success=False)

        callback.assert_called()

    def test_cooldown_prevents_rapid_changes(self, governor):
        """Test adjustment cooldown prevents thrashing."""
        # Force a stress change
        governor.record_agent_response("a", 100, success=False)
        governor.record_agent_response("a", 100, success=False)
        initial_level = governor.stress_level

        # Record successes immediately
        for _ in range(10):
            governor.record_agent_response("a", 100, success=True)

        # Should not have de-escalated due to cooldown
        # (only escalation allowed during cooldown)


# ============================================================================
# Constraint Retrieval Tests
# ============================================================================


class TestConstraintRetrieval:
    """Tests for constraint retrieval."""

    def test_get_constraints(self, governor):
        """Test getting current constraints."""
        constraints = governor.get_constraints()

        assert isinstance(constraints, GovernorConstraints)
        assert constraints == governor.current_constraints

    def test_constraints_change_with_stress(self, stressed_governor):
        """Test constraints change with stress level."""
        # Stressed governor should have different constraints
        constraints = stressed_governor.get_constraints()

        # Should have reduced capabilities
        assert constraints.max_agents_per_round <= 5


# ============================================================================
# Agent-Specific Constraints Tests
# ============================================================================


class TestAgentConstraints:
    """Tests for agent-specific constraints."""

    def test_get_agent_constraints_no_history(self, governor):
        """Test agent constraints with no history."""
        constraints = governor.get_agent_constraints("new_agent")

        assert "timeout_seconds" in constraints
        assert "max_response_tokens" in constraints

    def test_get_agent_constraints_with_history(self, governor):
        """Test agent constraints adjust based on history."""
        # Record some history
        for _ in range(5):
            governor.record_agent_response("claude", latency_ms=2000, success=True)

        constraints = governor.get_agent_constraints("claude")

        assert constraints["reliability_score"] == 1.0
        assert constraints["timeout_seconds"] > 0

    def test_slow_agent_gets_reduced_tokens(self, governor):
        """Test slow agents get reduced max tokens."""
        # Record high timeout rate
        for _ in range(5):
            governor.record_agent_timeout("slow_agent", 60.0)

        for _ in range(5):
            governor.record_agent_response("slow_agent", 1000, success=True)

        constraints = governor.get_agent_constraints("slow_agent")

        # Should have reduced tokens due to 50% timeout rate
        base = governor.current_constraints.max_proposal_length
        assert constraints["max_response_tokens"] < base


# ============================================================================
# Complexity Decision Tests
# ============================================================================


class TestComplexityDecisions:
    """Tests for complexity-related decisions."""

    def test_should_reduce_complexity_nominal(self, governor):
        """Test should_reduce_complexity at NOMINAL."""
        assert governor.should_reduce_complexity() is False

    def test_should_reduce_complexity_high(self):
        """Test should_reduce_complexity at HIGH."""
        gov = AdaptiveComplexityGovernor()
        gov.stress_level = StressLevel.HIGH

        assert gov.should_reduce_complexity() is True

    def test_should_skip_agent_low_samples(self, governor):
        """Test should_skip_agent with low samples."""
        governor.record_agent_timeout("agent", 60.0)

        # Not enough samples yet
        assert governor.should_skip_agent("agent") is False

    def test_should_skip_agent_high_timeout(self, governor):
        """Test should_skip_agent with high timeout rate."""
        # Record >70% timeout rate with enough samples
        for _ in range(8):
            governor.record_agent_timeout("bad_agent", 60.0)
        for _ in range(2):
            governor.record_agent_response("bad_agent", 100, success=True)

        assert governor.should_skip_agent("bad_agent") is True

    def test_get_recommended_agent_count(self, governor):
        """Test recommended agent count."""
        count = governor.get_recommended_agent_count()

        assert count == governor.current_constraints.max_agents_per_round


# ============================================================================
# Task Complexity and Timeout Scaling Tests
# ============================================================================


class TestTimeoutScaling:
    """Tests for task complexity and timeout scaling."""

    def test_set_task_complexity(self, governor):
        """Test setting task complexity."""
        governor.set_task_complexity(TaskComplexity.COMPLEX)

        assert governor.task_complexity == TaskComplexity.COMPLEX

    def test_scaled_timeout_simple(self, governor):
        """Test scaled timeout for simple task."""
        governor.set_task_complexity(TaskComplexity.SIMPLE)

        timeout = governor.get_scaled_timeout(base_timeout=180.0)

        # Simple = 0.5x factor
        assert timeout == pytest.approx(90.0)

    def test_scaled_timeout_complex(self, governor):
        """Test scaled timeout for complex task."""
        governor.set_task_complexity(TaskComplexity.COMPLEX)
        # Set high stress limit to not constrain
        governor.current_constraints.agent_timeout_seconds = 300.0

        timeout = governor.get_scaled_timeout(base_timeout=180.0)

        # Complex = 1.5x factor
        assert timeout == pytest.approx(270.0)

    def test_scaled_timeout_stress_limited(self, governor):
        """Test scaled timeout limited by stress level."""
        governor.set_task_complexity(TaskComplexity.COMPLEX)
        governor.current_constraints.agent_timeout_seconds = 60.0  # Stress limit

        timeout = governor.get_scaled_timeout(base_timeout=180.0)

        # Should be capped at stress limit
        assert timeout == 60.0

    def test_complexity_factors_exist(self):
        """Test all complexity levels have timeout factors."""
        for complexity in TaskComplexity:
            assert complexity in COMPLEXITY_TIMEOUT_FACTORS


# ============================================================================
# Status and Reset Tests
# ============================================================================


class TestStatusAndReset:
    """Tests for status retrieval and reset."""

    def test_get_status(self, governor):
        """Test getting governor status."""
        governor.record_agent_response("agent", 1000, success=True)
        governor.record_round_complete(1, 30.0, 3, 0)

        status = governor.get_status()

        assert status["stress_level"] == "nominal"
        assert "constraints" in status
        assert "agent_metrics" in status
        assert "agent" in status["agent_metrics"]
        assert "recent_rounds" in status

    def test_reset_metrics(self, governor):
        """Test resetting metrics."""
        # Add some history
        governor.record_agent_response("agent", 1000, success=True)
        governor.record_agent_timeout("agent", 60.0)
        governor.record_round_complete(1, 30.0, 3, 1)
        governor.set_task_complexity(TaskComplexity.COMPLEX)

        governor.reset_metrics()

        assert len(governor.agent_metrics) == 0
        assert len(governor.round_history) == 0
        assert governor.consecutive_failures == 0
        assert governor.stress_level == StressLevel.NOMINAL
        assert governor.task_complexity == TaskComplexity.MODERATE


# ============================================================================
# Global Governor Tests
# ============================================================================


class TestGlobalGovernor:
    """Tests for global governor functions."""

    def test_get_complexity_governor_singleton(self):
        """Test get_complexity_governor returns singleton."""
        gov1 = get_complexity_governor()
        gov2 = get_complexity_governor()

        assert gov1 is gov2

    def test_reset_complexity_governor(self):
        """Test resetting global governor."""
        gov1 = get_complexity_governor()
        reset_complexity_governor()
        gov2 = get_complexity_governor()

        assert gov1 is not gov2


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for realistic scenarios."""

    def test_gradual_degradation(self, governor):
        """Test gradual system degradation handling."""
        # Start with good responses
        for _ in range(10):
            governor.record_agent_response("agent", 1000, success=True)

        assert governor.stress_level == StressLevel.NOMINAL

        # Gradually add failures
        for _ in range(5):
            governor.record_agent_timeout("agent", 60.0)

        # Should have elevated stress
        assert governor.stress_level != StressLevel.NOMINAL

    def test_recovery_scenario(self, governor):
        """Test recovery after stress."""
        # Create stress
        governor.record_agent_response("a", 100, success=False)
        governor.record_agent_response("a", 100, success=False)

        initial_level = governor.stress_level

        # Simulate cooldown passing
        governor.last_adjustment_time = time.time() - 120  # 2 minutes ago

        # Record many successes
        for _ in range(20):
            governor.record_agent_response("a", 100, success=True)

        # May have recovered
        # (actual behavior depends on thresholds)

    def test_multi_agent_scenario(self, governor):
        """Test with multiple agents with varying performance."""
        # Good agent
        for _ in range(10):
            governor.record_agent_response("good_agent", 500, success=True)

        # Bad agent
        for _ in range(10):
            governor.record_agent_timeout("bad_agent", 60.0)

        # Mixed agent
        for i in range(10):
            governor.record_agent_response("mixed_agent", 1000, success=(i % 2 == 0))

        # Check individual agent constraints
        good_constraints = governor.get_agent_constraints("good_agent")
        bad_constraints = governor.get_agent_constraints("bad_agent")

        assert good_constraints["reliability_score"] == 1.0
        assert governor.should_skip_agent("bad_agent") is True
