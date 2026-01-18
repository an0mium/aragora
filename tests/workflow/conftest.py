"""
Workflow Engine test fixtures.

Provides fixtures specific to testing workflow components:
- EnhancedWorkflowEngine
- WorkflowDefinition and StepDefinition
- ResourceLimits and ResourceUsage
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.workflow.types import (
    ExecutionPattern,
    StepDefinition,
    StepResult,
    StepStatus,
    TransitionRule,
    WorkflowConfig,
    WorkflowDefinition,
)
from aragora.workflow.step import WorkflowStep, WorkflowContext


# ============================================================================
# Mock Workflow Step
# ============================================================================


class MockWorkflowStep:
    """Mock workflow step for testing."""

    step_type = "mock"

    def __init__(
        self,
        name: str = "mock",
        config: Optional[Dict[str, Any]] = None,
    ):
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, context: WorkflowContext) -> Any:
        # Check if we should fail (from config)
        if self._config.get("should_fail"):
            raise Exception("Mock step failure")

        # Check for delay
        delay = self._config.get("delay_seconds", 0.0)
        if delay > 0:
            await asyncio.sleep(delay)

        result = {"message": self._config.get("message", "Mock step completed")}

        # Add token info if provided
        input_tokens = self._config.get("input_tokens", 0)
        output_tokens = self._config.get("output_tokens", 0)
        if input_tokens or output_tokens:
            result["input_tokens"] = input_tokens
            result["output_tokens"] = output_tokens

        return result


class MockAgentStep:
    """Mock agent workflow step for testing."""

    step_type = "mock_agent"

    def __init__(
        self,
        name: str = "mock_agent",
        config: Optional[Dict[str, Any]] = None,
    ):
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, context: WorkflowContext) -> Any:
        agent_type = self._config.get("agent_type", "default")
        return {
            "response": f"Response from {agent_type}",
            "agent_type": agent_type,
            "input_tokens": self._config.get("input_tokens", 100),
            "output_tokens": self._config.get("output_tokens", 50),
        }


class MockDebateStep:
    """Mock debate workflow step for testing."""

    step_type = "mock_debate"

    def __init__(
        self,
        name: str = "mock_debate",
        config: Optional[Dict[str, Any]] = None,
    ):
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, context: WorkflowContext) -> Any:
        agents = self._config.get("agents", ["claude", "gpt4"])
        rounds = self._config.get("rounds", 3)
        return {
            "consensus": True,
            "agents": agents,
            "rounds_used": rounds,
            "conclusion": "Debate concluded",
            "input_tokens": len(agents) * rounds * 500,
            "output_tokens": len(agents) * rounds * 300,
        }


class FailingWorkflowStep:
    """Step that always fails for testing error handling."""

    step_type = "failing_mock"

    def __init__(
        self,
        name: str = "failing",
        config: Optional[Dict[str, Any]] = None,
    ):
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, context: WorkflowContext) -> Any:
        raise Exception("Mock step failure")


class SlowWorkflowStep:
    """Step with configurable delay for testing timeouts."""

    step_type = "slow_mock"

    def __init__(
        self,
        name: str = "slow",
        config: Optional[Dict[str, Any]] = None,
    ):
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, context: WorkflowContext) -> Any:
        delay = self._config.get("delay_seconds", 1.0)
        await asyncio.sleep(delay)
        return {"message": "Slow step completed"}


# ============================================================================
# Engine Fixtures
# ============================================================================


@pytest.fixture
def mock_step_registry():
    """Create a mock step registry with common step types."""
    return {
        "mock": MockWorkflowStep,
        "mock_agent": MockAgentStep,
        "mock_debate": MockDebateStep,
        "failing_mock": FailingWorkflowStep,
        "slow_mock": SlowWorkflowStep,
    }


@pytest.fixture
def workflow_config():
    """Create a default workflow configuration."""
    return WorkflowConfig(
        total_timeout_seconds=60.0,
        step_timeout_seconds=10.0,
        stop_on_failure=True,
        enable_checkpointing=False,
    )


@pytest.fixture
def engine(mock_step_registry, workflow_config):
    """Create an EnhancedWorkflowEngine with mock steps."""
    from aragora.workflow.engine_v2 import EnhancedWorkflowEngine, ResourceLimits

    return EnhancedWorkflowEngine(
        config=workflow_config,
        step_registry=mock_step_registry,
        limits=ResourceLimits(
            max_tokens=100000,
            max_cost_usd=10.0,
            timeout_seconds=60.0,
            max_api_calls=100,
        ),
    )


# ============================================================================
# Workflow Definition Fixtures
# ============================================================================


@pytest.fixture
def simple_workflow():
    """Create a simple single-step workflow."""
    return WorkflowDefinition(
        id="test-simple",
        name="Simple Test Workflow",
        description="A simple single-step workflow for testing",
        steps=[
            StepDefinition(
                id="step1",
                name="Mock Step",
                step_type="mock",
                config={"message": "Hello from step 1"},
            ),
        ],
        entry_step="step1",
    )


@pytest.fixture
def multi_step_workflow():
    """Create a multi-step sequential workflow."""
    return WorkflowDefinition(
        id="test-multi",
        name="Multi-Step Test Workflow",
        steps=[
            StepDefinition(
                id="step1",
                name="First Step",
                step_type="mock",
                config={"message": "Step 1"},
                next_steps=["step2"],
            ),
            StepDefinition(
                id="step2",
                name="Second Step",
                step_type="mock",
                config={"message": "Step 2"},
                next_steps=["step3"],
            ),
            StepDefinition(
                id="step3",
                name="Third Step",
                step_type="mock",
                config={"message": "Step 3"},
            ),
        ],
        entry_step="step1",
    )


@pytest.fixture
def agent_workflow():
    """Create a workflow with a mock agent step."""
    return WorkflowDefinition(
        id="test-agent",
        name="Agent Workflow",
        steps=[
            StepDefinition(
                id="agent-step",
                name="Agent Call",
                step_type="mock_agent",  # Use mock agent type
                config={
                    "agent_type": "claude",
                    "prompt": "Hello agent",
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            ),
        ],
        entry_step="agent-step",
    )


@pytest.fixture
def debate_workflow():
    """Create a workflow with a mock debate step."""
    return WorkflowDefinition(
        id="test-debate",
        name="Debate Workflow",
        steps=[
            StepDefinition(
                id="debate-step",
                name="Multi-Agent Debate",
                step_type="mock_debate",  # Use mock debate type
                config={
                    "agents": ["claude", "gpt4", "gemini"],
                    "rounds": 3,
                    "topic": "Test topic",
                },
            ),
        ],
        entry_step="debate-step",
    )


@pytest.fixture
def conditional_workflow():
    """Create a workflow with conditional transitions."""
    return WorkflowDefinition(
        id="test-conditional",
        name="Conditional Workflow",
        steps=[
            StepDefinition(
                id="decision",
                name="Decision Point",
                step_type="mock",
                config={"decision": True},
            ),
            StepDefinition(
                id="path-a",
                name="Path A",
                step_type="mock",
                config={"path": "A"},
            ),
            StepDefinition(
                id="path-b",
                name="Path B",
                step_type="mock",
                config={"path": "B"},
            ),
        ],
        transitions=[
            TransitionRule(
                id="tr1",
                from_step="decision",
                to_step="path-a",
                condition="output.get('decision') == True",
                priority=10,
            ),
            TransitionRule(
                id="tr2",
                from_step="decision",
                to_step="path-b",
                condition="output.get('decision') == False",
                priority=5,
            ),
        ],
        entry_step="decision",
    )


@pytest.fixture
def parallel_workflow():
    """Create a workflow with parallel execution."""
    return WorkflowDefinition(
        id="test-parallel",
        name="Parallel Workflow",
        steps=[
            StepDefinition(
                id="parallel-coordinator",
                name="Parallel Coordinator",
                step_type="mock",
                execution_pattern=ExecutionPattern.PARALLEL,
                config={"parallel_steps": ["worker-1", "worker-2", "worker-3"]},
            ),
            StepDefinition(
                id="worker-1",
                name="Worker 1",
                step_type="mock",
                config={"worker": 1},
            ),
            StepDefinition(
                id="worker-2",
                name="Worker 2",
                step_type="mock",
                config={"worker": 2},
            ),
            StepDefinition(
                id="worker-3",
                name="Worker 3",
                step_type="mock",
                config={"worker": 3},
            ),
        ],
        entry_step="parallel-coordinator",
    )


@pytest.fixture
def failing_workflow():
    """Create a workflow that will fail."""
    return WorkflowDefinition(
        id="test-failing",
        name="Failing Workflow",
        steps=[
            StepDefinition(
                id="failing-step",
                name="Failing Step",
                step_type="mock",
                config={"should_fail": True},
            ),
        ],
        entry_step="failing-step",
    )


# ============================================================================
# Resource Limit Fixtures
# ============================================================================


@pytest.fixture
def strict_limits():
    """Create strict resource limits for testing limit enforcement."""
    from aragora.workflow.engine_v2 import ResourceLimits

    return ResourceLimits(
        max_tokens=1000,
        max_cost_usd=0.01,
        timeout_seconds=5.0,
        max_api_calls=5,
        max_parallel_agents=2,
    )


@pytest.fixture
def relaxed_limits():
    """Create relaxed resource limits for normal testing."""
    from aragora.workflow.engine_v2 import ResourceLimits

    return ResourceLimits(
        max_tokens=1000000,
        max_cost_usd=100.0,
        timeout_seconds=3600.0,
        max_api_calls=10000,
    )
