"""Tests for the unified WorkflowExecutor protocol and factory."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.workflow import (
    WorkflowEngine,
    EnhancedWorkflowEngine,
    get_workflow_executor,
    get_workflow_engine,
    reset_workflow_engine,
    WorkflowExecutor,
    ResumableExecutor,
    ResourceAwareExecutor,
    TaskQueueExecutorAdapter,
)
from aragora.workflow.resource_tracker import (
    ResourceTracker,
    ResourceLimits,
    ResourceUsage,
    ResourceType,
)
from aragora.workflow.types import (
    StepDefinition,
    WorkflowConfig,
    WorkflowDefinition,
    WorkflowResult,
)


class TestWorkflowExecutorProtocol:
    """Test that all executors implement the WorkflowExecutor protocol."""

    def test_workflow_engine_is_executor(self):
        """WorkflowEngine should implement WorkflowExecutor protocol."""
        engine = WorkflowEngine()
        assert isinstance(engine, WorkflowExecutor)

    def test_enhanced_engine_is_executor(self):
        """EnhancedWorkflowEngine should implement WorkflowExecutor protocol."""
        engine = EnhancedWorkflowEngine()
        assert isinstance(engine, WorkflowExecutor)

    def test_enhanced_engine_is_resource_aware(self):
        """EnhancedWorkflowEngine should implement ResourceAwareExecutor protocol."""
        engine = EnhancedWorkflowEngine()
        assert isinstance(engine, ResourceAwareExecutor)

    def test_workflow_engine_is_resumable(self):
        """WorkflowEngine should implement ResumableExecutor protocol."""
        engine = WorkflowEngine()
        assert isinstance(engine, ResumableExecutor)

    def test_queue_adapter_is_executor(self):
        """TaskQueueExecutorAdapter should implement WorkflowExecutor protocol."""
        adapter = TaskQueueExecutorAdapter()
        assert isinstance(adapter, WorkflowExecutor)


class TestGetWorkflowExecutor:
    """Test the unified get_workflow_executor factory."""

    def test_default_mode_returns_workflow_engine(self):
        """Default mode should return WorkflowEngine."""
        executor = get_workflow_executor(mode="default")
        assert isinstance(executor, WorkflowEngine)

    def test_enhanced_mode_returns_enhanced_engine(self):
        """Enhanced mode should return EnhancedWorkflowEngine."""
        executor = get_workflow_executor(mode="enhanced")
        assert isinstance(executor, EnhancedWorkflowEngine)

    def test_enhanced_mode_with_limits(self):
        """Enhanced mode should accept resource limits."""
        limits = ResourceLimits(max_tokens=50000, max_cost_usd=2.5)
        executor = get_workflow_executor(mode="enhanced", resource_limits=limits)
        assert isinstance(executor, EnhancedWorkflowEngine)
        assert executor.limits.max_tokens == 50000
        assert executor.limits.max_cost_usd == 2.5

    def test_queue_mode_returns_adapter(self):
        """Queue mode should return TaskQueueExecutorAdapter."""
        executor = get_workflow_executor(mode="queue")
        assert isinstance(executor, TaskQueueExecutorAdapter)

    def test_invalid_mode_raises_error(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown executor mode"):
            get_workflow_executor(mode="invalid")

    def test_config_passed_to_engine(self):
        """Config should be passed to engine when creating new instance."""
        # Reset singleton to ensure fresh instance
        reset_workflow_engine()
        config = WorkflowConfig(enable_checkpointing=False, stop_on_failure=False)
        executor = get_workflow_executor(mode="default", config=config)
        assert executor._config.enable_checkpointing is False
        assert executor._config.stop_on_failure is False
        # Clean up
        reset_workflow_engine()


class TestResourceTracker:
    """Test the composable ResourceTracker."""

    def test_basic_token_tracking(self):
        """Should track tokens and calculate cost."""
        tracker = ResourceTracker()
        tracker.start()

        cost = tracker.add_tokens("step1", "claude", input_tokens=500, output_tokens=200)

        assert tracker.usage.tokens_used == 700
        assert cost > 0
        assert tracker.usage.cost_usd > 0

    def test_api_call_tracking(self):
        """Should track API calls."""
        tracker = ResourceTracker()
        tracker.start()

        tracker.add_api_call()
        tracker.add_api_call()
        tracker.add_api_call()

        assert tracker.usage.api_calls == 3

    def test_limit_checking(self):
        """Should check limits correctly."""
        tracker = ResourceTracker(ResourceLimits(max_tokens=100))
        tracker.start()

        assert tracker.check_limits() is True

        tracker.add_tokens("step1", "claude", input_tokens=50, output_tokens=60)

        assert tracker.check_limits() is False
        assert tracker.limit_exceeded_type == ResourceType.TOKENS

    def test_limit_raise(self):
        """Should raise ResourceExhaustedError when limits exceeded."""
        from aragora.workflow.resource_tracker import ResourceExhaustedError

        tracker = ResourceTracker(ResourceLimits(max_api_calls=2))
        tracker.start()

        tracker.add_api_call()
        tracker.add_api_call()

        with pytest.raises(ResourceExhaustedError, match="API call limit"):
            tracker.check_limits_or_raise()

    def test_warning_callback(self):
        """Should call warning callback when threshold crossed."""
        warnings_received: list[tuple[ResourceType, float]] = []

        def on_warning(resource_type: ResourceType, percentage: float) -> None:
            warnings_received.append((resource_type, percentage))

        tracker = ResourceTracker(
            ResourceLimits(max_tokens=100, warning_threshold=0.5),
            on_warning=on_warning,
        )
        tracker.start()

        # Add tokens to cross 50% threshold
        tracker.add_tokens("step1", "claude", input_tokens=30, output_tokens=30)

        assert len(warnings_received) == 1
        assert warnings_received[0][0] == ResourceType.TOKENS
        assert warnings_received[0][1] >= 0.5

    def test_metrics_output(self):
        """Should provide comprehensive metrics."""
        tracker = ResourceTracker()
        tracker.start()
        tracker.add_tokens("step1", "claude", 100, 50)
        tracker.add_api_call()

        metrics = tracker.get_metrics()

        assert "usage" in metrics
        assert "limits" in metrics
        assert "within_limits" in metrics
        assert metrics["usage"]["tokens_used"] == 150
        assert metrics["usage"]["api_calls"] == 1

    def test_reset(self):
        """Should reset all counters."""
        tracker = ResourceTracker()
        tracker.start()
        tracker.add_tokens("step1", "claude", 100, 50)
        tracker.add_api_call()

        tracker.usage.reset()

        assert tracker.usage.tokens_used == 0
        assert tracker.usage.api_calls == 0
        assert tracker.usage.cost_usd == 0.0

    def test_cost_estimation(self):
        """Should estimate cost for different models."""
        tracker = ResourceTracker()

        claude_cost = tracker.estimate_cost("claude", estimated_tokens=1000)
        gpt4_cost = tracker.estimate_cost("gpt-4", estimated_tokens=1000)

        # GPT-4 should be more expensive than Claude Sonnet
        assert gpt4_cost > claude_cost
        assert claude_cost > 0

    def test_per_step_tracking(self):
        """Should track usage per step."""
        tracker = ResourceTracker()
        tracker.start()

        tracker.add_tokens("step1", "claude", 100, 50)
        tracker.add_tokens("step2", "gpt-4", 200, 100)
        tracker.add_step_duration("step1", 1.5)
        tracker.add_step_duration("step2", 2.0)

        assert "step1" in tracker.usage.step_tokens
        assert "step2" in tracker.usage.step_tokens
        assert tracker.usage.step_tokens["step1"] == 150
        assert tracker.usage.step_tokens["step2"] == 300
        assert tracker.usage.step_durations["step1"] == 1.5

    def test_per_agent_tracking(self):
        """Should track usage per agent type."""
        tracker = ResourceTracker()
        tracker.start()

        tracker.add_tokens("step1", "claude", 100, 50)
        tracker.add_tokens("step2", "claude", 100, 50)
        tracker.add_tokens("step3", "gpt-4", 200, 100)

        assert tracker.usage.agent_tokens["claude"] == 300
        assert tracker.usage.agent_tokens["gpt-4"] == 300


class TestTaskQueueAdapter:
    """Test the TaskQueueExecutorAdapter."""

    def test_adapter_initialization(self):
        """Should initialize with correct defaults."""
        adapter = TaskQueueExecutorAdapter()
        assert adapter._max_concurrent == 10
        assert adapter._default_timeout == 300.0

    def test_adapter_custom_config(self):
        """Should accept custom configuration."""
        adapter = TaskQueueExecutorAdapter(
            max_concurrent=5,
            default_timeout=60.0,
            max_retries=3,
        )
        assert adapter._max_concurrent == 5
        assert adapter._default_timeout == 60.0
        assert adapter._max_retries == 3

    def test_get_metrics_empty(self):
        """Should return empty metrics before execution."""
        adapter = TaskQueueExecutorAdapter()
        metrics = adapter.get_metrics()

        assert metrics["total_steps"] == 0
        assert metrics["completed_steps"] == 0
        assert metrics["failed_steps"] == 0

    def test_request_termination(self):
        """Should handle termination request."""
        adapter = TaskQueueExecutorAdapter()
        adapter.request_termination("User cancelled")

        assert adapter._should_terminate is True
        assert adapter._termination_reason == "User cancelled"


class TestExecutorCommonInterface:
    """Test that all executors share a common interface."""

    @pytest.fixture
    def simple_definition(self) -> WorkflowDefinition:
        """Create a simple workflow definition for testing."""
        return WorkflowDefinition(
            id="test-workflow",
            name="Test Workflow",
            steps=[
                StepDefinition(
                    id="step1",
                    name="Step 1",
                    step_type="agent",
                    config={"agent_type": "demo"},
                ),
            ],
            entry_step="step1",
        )

    def test_all_executors_have_execute(self, simple_definition: WorkflowDefinition):
        """All executors should have execute method."""
        executors = [
            WorkflowEngine(),
            EnhancedWorkflowEngine(),
            TaskQueueExecutorAdapter(),
        ]

        for executor in executors:
            assert hasattr(executor, "execute")
            assert callable(executor.execute)

    def test_all_executors_have_get_metrics(self):
        """All executors should have get_metrics method."""
        executors = [
            WorkflowEngine(),
            EnhancedWorkflowEngine(),
            TaskQueueExecutorAdapter(),
        ]

        for executor in executors:
            assert hasattr(executor, "get_metrics")
            metrics = executor.get_metrics()
            assert isinstance(metrics, dict)
            assert "total_steps" in metrics

    def test_all_executors_have_request_termination(self):
        """All executors should have request_termination method."""
        executors = [
            WorkflowEngine(),
            EnhancedWorkflowEngine(),
            TaskQueueExecutorAdapter(),
        ]

        for executor in executors:
            assert hasattr(executor, "request_termination")
            # Should not raise
            executor.request_termination("Test termination")


class TestResourceLimits:
    """Test ResourceLimits dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        limits = ResourceLimits()
        assert limits.max_tokens == 100000
        assert limits.max_cost_usd == 10.0
        assert limits.timeout_seconds == 600.0
        assert limits.max_api_calls == 100

    def test_to_dict(self):
        """Should convert to dictionary."""
        limits = ResourceLimits(max_tokens=50000)
        d = limits.to_dict()
        assert d["max_tokens"] == 50000
        assert "max_cost_usd" in d

    def test_from_dict(self):
        """Should create from dictionary (resource_tracker version only)."""
        # from_dict is only available on resource_tracker.ResourceLimits
        from aragora.workflow.resource_tracker import ResourceLimits as TrackerLimits

        limits = TrackerLimits.from_dict(
            {
                "max_tokens": 25000,
                "max_cost_usd": 5.0,
            }
        )
        assert limits.max_tokens == 25000
        assert limits.max_cost_usd == 5.0


class TestResourceUsage:
    """Test ResourceUsage dataclass."""

    def test_default_values(self):
        """Should start with zero usage."""
        usage = ResourceUsage()
        assert usage.tokens_used == 0
        assert usage.cost_usd == 0.0
        assert usage.api_calls == 0

    def test_add_tokens(self):
        """Should add tokens correctly."""
        usage = ResourceUsage()
        cost = usage.add_tokens("step1", "claude", 100, 50)

        assert usage.tokens_used == 150
        assert cost > 0
        assert usage.step_tokens["step1"] == 150
        assert usage.agent_tokens["claude"] == 150

    def test_to_dict(self):
        """Should convert to dictionary."""
        usage = ResourceUsage()
        usage.add_tokens("step1", "claude", 100, 50)

        d = usage.to_dict()
        assert d["tokens_used"] == 150
        assert "step_tokens" in d
        assert "agent_tokens" in d
