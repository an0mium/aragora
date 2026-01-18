"""
Workflow Engine for Aragora.

The Workflow Engine generalizes Aragora's PhaseExecutor pattern to support
arbitrary multi-step workflows with:

- Sequential, parallel, and conditional execution patterns
- Checkpointing and resume for long-running workflows
- Transitions based on step outputs
- Integration with Knowledge Mound

This is the core runtime for the Enterprise Multi-Agent Control Plane.

Usage:
    from aragora.workflow import (
        WorkflowEngine,
        WorkflowDefinition,
        StepDefinition,
        WorkflowContext,
    )

    # Define a workflow
    definition = WorkflowDefinition(
        id="my-workflow",
        name="My Workflow",
        steps=[
            StepDefinition(
                id="step1",
                name="First Step",
                step_type="agent",
                config={"agent_type": "claude", "prompt": "..."},
                next_steps=["step2"],
            ),
            StepDefinition(
                id="step2",
                name="Second Step",
                step_type="agent",
                config={"agent_type": "gpt4", "prompt": "..."},
            ),
        ],
    )

    # Execute
    engine = WorkflowEngine()
    result = await engine.execute(definition, inputs={"task": "..."})
"""

from aragora.workflow.types import (
    ExecutionPattern,
    StepDefinition,
    StepResult,
    StepStatus,
    TransitionRule,
    WorkflowCheckpoint,
    WorkflowConfig,
    WorkflowDefinition,
    WorkflowResult,
)
from aragora.workflow.step import (
    WorkflowStep,
    WorkflowContext,
    BaseStep,
    AgentStep,
    ParallelStep,
    ConditionalStep,
    LoopStep,
)
from aragora.workflow.engine import WorkflowEngine
from aragora.workflow.engine_v2 import (
    EnhancedWorkflowEngine,
    ResourceLimits,
    ResourceUsage,
    ResourceType,
    ResourceExhaustedError,
    EnhancedWorkflowResult,
)
from aragora.workflow.schema import (
    validate_workflow,
    validate_workflow_file,
    ValidationResult,
    ValidationMessage,
    WorkflowValidator,
)

__all__ = [
    # Engines
    "WorkflowEngine",
    "EnhancedWorkflowEngine",
    # Resource management
    "ResourceLimits",
    "ResourceUsage",
    "ResourceType",
    "ResourceExhaustedError",
    "EnhancedWorkflowResult",
    # Validation
    "validate_workflow",
    "validate_workflow_file",
    "ValidationResult",
    "ValidationMessage",
    "WorkflowValidator",
    # Types
    "ExecutionPattern",
    "StepDefinition",
    "StepResult",
    "StepStatus",
    "TransitionRule",
    "WorkflowCheckpoint",
    "WorkflowConfig",
    "WorkflowDefinition",
    "WorkflowResult",
    # Steps
    "WorkflowStep",
    "WorkflowContext",
    "BaseStep",
    "AgentStep",
    "ParallelStep",
    "ConditionalStep",
    "LoopStep",
]
