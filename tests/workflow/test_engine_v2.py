"""
Tests for the Enhanced Workflow Engine.

Tests cover:
- Resource limits and tracking
- Single and multi-step workflow execution
- Parallel step execution
- Cost estimation
- Error handling and retries
- Metrics callbacks
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from aragora.workflow.engine_v2 import (
    EnhancedWorkflowEngine,
    EnhancedWorkflowResult,
    MODEL_PRICING,
    ResourceExhaustedError,
    ResourceLimits,
    ResourceType,
    ResourceUsage,
)
from aragora.workflow.types import (
    ExecutionPattern,
    StepDefinition,
    StepStatus,
    WorkflowConfig,
    WorkflowDefinition,
)


# ============================================================================
# ResourceLimits Tests
# ============================================================================


class TestResourceLimits:
    """Tests for ResourceLimits configuration."""

    def test_default_limits(self):
        """Test default resource limit values."""
        limits = ResourceLimits()

        assert limits.max_tokens == 100000
        assert limits.max_cost_usd == 10.0
        assert limits.timeout_seconds == 600.0
        assert limits.max_api_calls == 100
        assert limits.max_parallel_agents == 5
        assert limits.max_retries_per_step == 3

    def test_custom_limits(self):
        """Test custom resource limit values."""
        limits = ResourceLimits(
            max_tokens=50000,
            max_cost_usd=5.0,
            timeout_seconds=300.0,
            max_api_calls=50,
        )

        assert limits.max_tokens == 50000
        assert limits.max_cost_usd == 5.0
        assert limits.timeout_seconds == 300.0
        assert limits.max_api_calls == 50

    def test_limits_to_dict(self):
        """Test serialization of limits."""
        limits = ResourceLimits(max_tokens=1000, max_cost_usd=1.0)
        data = limits.to_dict()

        assert data["max_tokens"] == 1000
        assert data["max_cost_usd"] == 1.0
        assert "timeout_seconds" in data


# ============================================================================
# ResourceUsage Tests
# ============================================================================


class TestResourceUsage:
    """Tests for ResourceUsage tracking."""

    def test_initial_usage(self):
        """Test initial usage values are zero."""
        usage = ResourceUsage()

        assert usage.tokens_used == 0
        assert usage.cost_usd == 0.0
        assert usage.api_calls == 0
        assert usage.time_elapsed_seconds == 0.0

    def test_add_tokens(self):
        """Test adding token usage."""
        usage = ResourceUsage()

        cost = usage.add_tokens(
            step_id="step1",
            agent_type="claude",
            input_tokens=100,
            output_tokens=50,
        )

        assert usage.tokens_used == 150
        assert usage.step_tokens["step1"] == 150
        assert usage.agent_tokens["claude"] == 150
        assert cost > 0

    def test_add_tokens_calculates_cost(self):
        """Test that token usage calculates correct cost."""
        usage = ResourceUsage()

        # Claude-3-sonnet pricing
        usage.add_tokens("step1", "claude", input_tokens=1000, output_tokens=1000)

        # Input: $0.003/1K, Output: $0.015/1K
        expected_cost = (1000 / 1000) * 0.003 + (1000 / 1000) * 0.015
        assert abs(usage.cost_usd - expected_cost) < 0.0001

    def test_add_api_call(self):
        """Test recording API calls."""
        usage = ResourceUsage()

        usage.add_api_call()
        usage.add_api_call()
        usage.add_api_call()

        assert usage.api_calls == 3

    def test_usage_to_dict(self):
        """Test serialization of usage."""
        usage = ResourceUsage()
        usage.add_tokens("step1", "claude", 100, 50)
        usage.add_api_call()

        data = usage.to_dict()

        assert data["tokens_used"] == 150
        assert data["api_calls"] == 1
        assert "step_tokens" in data
        assert "agent_tokens" in data


# ============================================================================
# Model Pricing Tests
# ============================================================================


class TestModelPricing:
    """Tests for model pricing configuration."""

    def test_pricing_includes_common_models(self):
        """Test that pricing includes common model types."""
        assert "claude-3-opus" in MODEL_PRICING
        assert "gpt-4" in MODEL_PRICING
        assert "gemini-pro" in MODEL_PRICING
        assert "default" in MODEL_PRICING

    def test_pricing_has_input_output(self):
        """Test that each model has input and output prices."""
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing
            assert "output" in pricing
            assert pricing["input"] >= 0
            assert pricing["output"] >= 0


# ============================================================================
# Engine Initialization Tests
# ============================================================================


class TestEngineInitialization:
    """Tests for EnhancedWorkflowEngine initialization."""

    def test_engine_creation(self, mock_step_registry, workflow_config):
        """Test basic engine creation."""
        engine = EnhancedWorkflowEngine(
            config=workflow_config,
            step_registry=mock_step_registry,
        )

        assert engine is not None
        assert engine.limits is not None

    def test_engine_with_custom_limits(self, mock_step_registry, workflow_config):
        """Test engine with custom resource limits."""
        limits = ResourceLimits(max_tokens=5000)

        engine = EnhancedWorkflowEngine(
            config=workflow_config,
            step_registry=mock_step_registry,
            limits=limits,
        )

        assert engine.limits.max_tokens == 5000

    def test_engine_set_limits(self, engine):
        """Test updating limits after creation."""
        new_limits = ResourceLimits(max_tokens=2000)
        engine.set_limits(new_limits)

        assert engine.limits.max_tokens == 2000


# ============================================================================
# Simple Workflow Execution Tests
# ============================================================================


class TestSimpleWorkflowExecution:
    """Tests for simple workflow execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self, engine, simple_workflow):
        """Test executing a simple single-step workflow."""
        result = await engine.execute(simple_workflow)

        assert result is not None
        assert result.success is True
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_multi_step_workflow(self, engine, multi_step_workflow):
        """Test executing a multi-step workflow."""
        result = await engine.execute(multi_step_workflow)

        assert result.success is True
        assert len(result.steps) == 3
        assert all(s.status == StepStatus.COMPLETED for s in result.steps)

    @pytest.mark.asyncio
    async def test_result_contains_metrics(self, engine, simple_workflow):
        """Test that result contains execution metrics."""
        result = await engine.execute(simple_workflow)

        metrics = result.metrics
        assert "total_tokens" in metrics
        assert "total_cost_usd" in metrics
        assert "total_duration_seconds" in metrics
        assert "api_calls" in metrics

    @pytest.mark.asyncio
    async def test_workflow_id_generation(self, engine, simple_workflow):
        """Test that workflow ID is generated if not provided."""
        result = await engine.execute(simple_workflow)

        assert result.workflow_id is not None
        assert result.workflow_id.startswith("wf_")

    @pytest.mark.asyncio
    async def test_custom_workflow_id(self, engine, simple_workflow):
        """Test using a custom workflow ID."""
        result = await engine.execute(simple_workflow, workflow_id="custom-id-123")

        assert result.workflow_id == "custom-id-123"


# ============================================================================
# Agent Workflow Tests
# ============================================================================


class TestAgentWorkflow:
    """Tests for workflows with mock agent steps."""

    @pytest.mark.asyncio
    async def test_execute_agent_workflow(self, engine, agent_workflow):
        """Test executing a mock agent workflow."""
        result = await engine.execute(agent_workflow)

        assert result.success is True
        assert result.final_output is not None
        assert "response" in result.final_output

    @pytest.mark.asyncio
    async def test_agent_workflow_tracks_tokens(self, engine, agent_workflow):
        """Test that agent workflow tracks token usage."""
        result = await engine.execute(agent_workflow)

        assert result.resource_usage.tokens_used > 0
        assert result.resource_usage.cost_usd > 0


# ============================================================================
# Debate Workflow Tests
# ============================================================================


class TestDebateWorkflow:
    """Tests for workflows with mock debate steps."""

    @pytest.mark.asyncio
    async def test_execute_debate_workflow(self, engine, debate_workflow):
        """Test executing a mock debate workflow."""
        result = await engine.execute(debate_workflow)

        assert result.success is True
        assert result.final_output is not None
        assert result.final_output.get("consensus") is True

    @pytest.mark.asyncio
    async def test_debate_workflow_tracks_tokens(self, engine, debate_workflow):
        """Test that mock debate workflow tracks token usage."""
        result = await engine.execute(debate_workflow)

        # Debate has multiple agents and rounds, should have significant tokens
        assert result.resource_usage.tokens_used > 0


# ============================================================================
# Parallel Execution Tests
# ============================================================================


class TestParallelExecution:
    """Tests for parallel step execution."""

    @pytest.mark.asyncio
    async def test_execute_parallel_workflow(self, engine, parallel_workflow):
        """Test executing parallel workflow steps."""
        result = await engine.execute(parallel_workflow)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_parallel_respects_semaphore(self, mock_step_registry, workflow_config):
        """Test that parallel execution respects the semaphore limit."""
        limits = ResourceLimits(max_parallel_agents=2)
        engine = EnhancedWorkflowEngine(
            config=workflow_config,
            step_registry=mock_step_registry,
            limits=limits,
        )

        # Create workflow with many parallel steps
        workflow = WorkflowDefinition(
            id="test-parallel-limit",
            name="Parallel Limit Test",
            steps=[
                StepDefinition(
                    id="coordinator",
                    name="Coordinator",
                    step_type="mock",
                    execution_pattern=ExecutionPattern.PARALLEL,
                    config={"parallel_steps": ["w1", "w2", "w3", "w4"]},
                ),
                StepDefinition(id="w1", name="W1", step_type="mock"),
                StepDefinition(id="w2", name="W2", step_type="mock"),
                StepDefinition(id="w3", name="W3", step_type="mock"),
                StepDefinition(id="w4", name="W4", step_type="mock"),
            ],
            entry_step="coordinator",
        )

        result = await engine.execute(workflow)
        assert result.success is True


# ============================================================================
# Resource Limit Enforcement Tests
# ============================================================================


class TestResourceLimitEnforcement:
    """Tests for resource limit enforcement."""

    @pytest.mark.asyncio
    async def test_api_call_limit_exceeded(self, mock_step_registry, workflow_config):
        """Test that API call limit is enforced."""
        limits = ResourceLimits(max_api_calls=1)
        engine = EnhancedWorkflowEngine(
            config=workflow_config,
            step_registry=mock_step_registry,
            limits=limits,
        )

        # Create workflow with multiple steps
        workflow = WorkflowDefinition(
            id="test-api-limit",
            name="API Limit Test",
            steps=[
                StepDefinition(id="s1", name="S1", step_type="mock", next_steps=["s2"]),
                StepDefinition(id="s2", name="S2", step_type="mock", next_steps=["s3"]),
                StepDefinition(id="s3", name="S3", step_type="mock"),
            ],
            entry_step="s1",
        )

        result = await engine.execute(workflow)

        assert result.limits_exceeded is True
        assert result.limit_exceeded_type == ResourceType.API_CALLS

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self, mock_step_registry, workflow_config):
        """Test that timeout is enforced."""
        limits = ResourceLimits(timeout_seconds=0.1)
        engine = EnhancedWorkflowEngine(
            config=workflow_config,
            step_registry=mock_step_registry,
            limits=limits,
        )

        workflow = WorkflowDefinition(
            id="test-timeout",
            name="Timeout Test",
            steps=[
                StepDefinition(
                    id="slow",
                    name="Slow Step",
                    step_type="slow_mock",
                    config={"delay_seconds": 1.0},
                ),
            ],
            entry_step="slow",
        )

        result = await engine.execute(workflow)

        assert result.limits_exceeded is True
        assert result.limit_exceeded_type == ResourceType.TIME


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in workflow execution."""

    @pytest.mark.asyncio
    async def test_failing_step_stops_workflow(self, mock_step_registry, workflow_config):
        """Test that a failing step stops the workflow."""
        engine = EnhancedWorkflowEngine(
            config=workflow_config,
            step_registry=mock_step_registry,
        )

        workflow = WorkflowDefinition(
            id="test-failing",
            name="Failing Test",
            steps=[
                StepDefinition(id="fail", name="Failing Step", step_type="failing_mock"),
            ],
            entry_step="fail",
        )

        result = await engine.execute(workflow)

        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED

    @pytest.mark.asyncio
    async def test_optional_step_failure_continues(self, mock_step_registry, workflow_config):
        """Test that optional step failure allows workflow to continue."""
        workflow_config.stop_on_failure = True
        engine = EnhancedWorkflowEngine(
            config=workflow_config,
            step_registry=mock_step_registry,
        )

        workflow = WorkflowDefinition(
            id="test-optional",
            name="Optional Test",
            steps=[
                StepDefinition(
                    id="optional-fail",
                    name="Optional Failing",
                    step_type="failing_mock",
                    optional=True,
                    next_steps=["success"],
                ),
                StepDefinition(id="success", name="Success Step", step_type="mock"),
            ],
            entry_step="optional-fail",
        )

        result = await engine.execute(workflow)

        # First step fails but is optional, second step should execute
        assert len(result.steps) >= 1

    @pytest.mark.asyncio
    async def test_unknown_step_type_fails(self, engine):
        """Test that unknown step type causes failure."""
        workflow = WorkflowDefinition(
            id="test-unknown",
            name="Unknown Step Test",
            steps=[
                StepDefinition(
                    id="unknown",
                    name="Unknown Step",
                    step_type="nonexistent_step_type",
                ),
            ],
            entry_step="unknown",
        )

        result = await engine.execute(workflow)

        assert result.steps[0].status == StepStatus.FAILED
        assert "Unknown step type" in result.steps[0].error


# ============================================================================
# Cost Estimation Tests
# ============================================================================


class TestCostEstimation:
    """Tests for workflow cost estimation."""

    def test_estimate_agent_workflow(self, engine):
        """Test cost estimation for agent workflow with correct step type."""
        # Cost estimation looks for step_type="agent"
        workflow = WorkflowDefinition(
            id="test-estimate",
            name="Estimation Test",
            steps=[
                StepDefinition(
                    id="agent-step",
                    name="Agent",
                    step_type="agent",
                    config={"agent_type": "claude", "estimated_tokens": 2000},
                ),
            ],
            entry_step="agent-step",
        )

        estimates = engine.estimate_cost(workflow)

        assert "total" in estimates
        assert estimates["total"] > 0

    def test_estimate_debate_workflow(self, engine):
        """Test cost estimation for debate workflow."""
        # Cost estimation looks for step_type="debate" or "quick_debate"
        workflow = WorkflowDefinition(
            id="test-debate-estimate",
            name="Debate Estimation",
            steps=[
                StepDefinition(
                    id="debate-step",
                    name="Debate",
                    step_type="debate",
                    config={"agents": ["claude", "gpt4", "gemini"], "rounds": 3},
                ),
            ],
            entry_step="debate-step",
        )

        estimates = engine.estimate_cost(workflow)

        assert estimates["total"] > 0
        # Debate has multiple agents
        assert any(agent in estimates for agent in ["claude", "gpt4", "gemini"])

    def test_estimate_returns_per_agent_costs(self, engine):
        """Test that estimation returns per-agent breakdown."""
        workflow = WorkflowDefinition(
            id="test-multi-agent",
            name="Multi Agent",
            steps=[
                StepDefinition(
                    id="debate",
                    name="Debate",
                    step_type="debate",
                    config={"agents": ["claude", "gpt4"], "rounds": 2},
                ),
            ],
            entry_step="debate",
        )

        estimates = engine.estimate_cost(workflow)

        # Should have per-agent costs in addition to total
        non_total_keys = [k for k in estimates.keys() if k != "total"]
        assert len(non_total_keys) > 0


# ============================================================================
# Metrics Callback Tests
# ============================================================================


class TestMetricsCallback:
    """Tests for metrics callback functionality."""

    @pytest.mark.asyncio
    async def test_metrics_callback_called(self, mock_step_registry, workflow_config, simple_workflow):
        """Test that metrics callback is called during execution."""
        callback_data: List[Dict[str, Any]] = []

        def metrics_callback(metrics: Dict[str, Any]) -> None:
            callback_data.append(metrics)

        engine = EnhancedWorkflowEngine(
            config=workflow_config,
            step_registry=mock_step_registry,
            metrics_callback=metrics_callback,
        )

        await engine.execute(simple_workflow)

        assert len(callback_data) > 0
        assert "step_id" in callback_data[0]
        assert "tokens_used" in callback_data[0]

    @pytest.mark.asyncio
    async def test_metrics_callback_error_ignored(self, mock_step_registry, workflow_config, simple_workflow):
        """Test that errors in metrics callback don't break execution."""

        def failing_callback(metrics: Dict[str, Any]) -> None:
            raise Exception("Callback error")

        engine = EnhancedWorkflowEngine(
            config=workflow_config,
            step_registry=mock_step_registry,
            metrics_callback=failing_callback,
        )

        # Should not raise, callback errors are ignored
        result = await engine.execute(simple_workflow)
        assert result.success is True


# ============================================================================
# Enhanced Result Tests
# ============================================================================


class TestEnhancedResult:
    """Tests for EnhancedWorkflowResult."""

    @pytest.mark.asyncio
    async def test_result_has_resource_usage(self, engine, simple_workflow):
        """Test that result includes resource usage."""
        result = await engine.execute(simple_workflow)

        assert isinstance(result, EnhancedWorkflowResult)
        assert result.resource_usage is not None
        assert isinstance(result.resource_usage, ResourceUsage)

    @pytest.mark.asyncio
    async def test_result_metrics_property(self, engine, agent_workflow):
        """Test the metrics property provides useful data."""
        result = await engine.execute(agent_workflow)

        metrics = result.metrics

        assert "total_tokens" in metrics
        assert "total_cost_usd" in metrics
        assert "total_duration_seconds" in metrics
        assert "api_calls" in metrics
        assert "steps_completed" in metrics
        assert "steps_failed" in metrics
        assert "per_step_costs" in metrics
        assert "per_agent_costs" in metrics

    @pytest.mark.asyncio
    async def test_result_duration_tracking(self, engine, simple_workflow):
        """Test that result tracks execution duration."""
        result = await engine.execute(simple_workflow)

        assert result.total_duration_ms > 0
        assert result.resource_usage.time_elapsed_seconds > 0


# ============================================================================
# Workflow Validation Tests
# ============================================================================


class TestWorkflowValidation:
    """Tests for workflow definition validation."""

    def test_empty_workflow_fails(self, engine):
        """Test that empty workflow is detected."""
        workflow = WorkflowDefinition(
            id="empty",
            name="Empty Workflow",
            steps=[],
        )

        is_valid, errors = workflow.validate()
        assert is_valid is False
        assert any("at least one step" in e for e in errors)

    def test_missing_entry_step_detected(self):
        """Test that missing entry step is detected."""
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            steps=[
                StepDefinition(id="step1", name="Step 1", step_type="mock"),
            ],
            entry_step="nonexistent",
        )

        is_valid, errors = workflow.validate()
        assert is_valid is False
        assert any("not found" in e for e in errors)

    def test_invalid_next_step_detected(self):
        """Test that invalid next_step reference is detected."""
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            steps=[
                StepDefinition(
                    id="step1",
                    name="Step 1",
                    step_type="mock",
                    next_steps=["nonexistent"],
                ),
            ],
            entry_step="step1",
        )

        is_valid, errors = workflow.validate()
        assert is_valid is False
        assert any("unknown next step" in e for e in errors)
