"""
Workflow Engine for Aragora.

Generalizes the PhaseExecutor pattern from debate orchestration to support
arbitrary multi-step workflows with:
- Sequential, parallel, and conditional execution
- Checkpointing and resume
- Transitions based on step outputs
- Integration with Knowledge Mound

This is the core runtime for the Enterprise Control Plane.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from aragora.workflow.safe_eval import SafeEvalError, safe_eval_bool
from aragora.workflow.types import (
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
    AgentStep,
    ParallelStep,
    ConditionalStep,
    LoopStep,
)

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """
    Executes workflows defined by WorkflowDefinition.

    The engine supports:
    - Sequential step execution with configurable order
    - Parallel execution for hive-mind patterns
    - Conditional transitions based on step outputs
    - Checkpointing for long-running workflows
    - Integration with the Phase protocol for Aragora debates

    Usage:
        engine = WorkflowEngine()

        # Register step implementations
        engine.register_step_type("agent", AgentStep)
        engine.register_step_type("parallel", ParallelStep)

        # Execute workflow
        result = await engine.execute(definition, inputs)

        # Resume from checkpoint
        result = await engine.resume(workflow_id, checkpoint)
    """

    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        step_registry: Optional[Dict[str, Type[WorkflowStep]]] = None,
    ):
        self._config = config or WorkflowConfig()

        # Step type registry
        self._step_types: Dict[str, Type[WorkflowStep]] = step_registry or {}
        self._register_default_step_types()

        # Step instance cache
        self._step_instances: Dict[str, WorkflowStep] = {}

        # Execution state
        self._current_step: Optional[str] = None
        self._should_terminate: bool = False
        self._termination_reason: Optional[str] = None
        self._results: List[StepResult] = []

        # Checkpoint storage (in-memory for now, will be persisted)
        self._checkpoints: Dict[str, WorkflowCheckpoint] = {}

    def _register_default_step_types(self) -> None:
        """Register built-in step types."""
        self._step_types["agent"] = AgentStep
        self._step_types["parallel"] = ParallelStep
        self._step_types["conditional"] = ConditionalStep
        self._step_types["loop"] = LoopStep

        # Phase 2: Register new step types for workflow builder
        try:
            from aragora.workflow.nodes import (
                HumanCheckpointStep,
                MemoryReadStep,
                MemoryWriteStep,
                DebateStep,
                DecisionStep,
                TaskStep,
            )
            from aragora.workflow.nodes.decision import SwitchStep
            from aragora.workflow.nodes.debate import QuickDebateStep

            self._step_types["human_checkpoint"] = HumanCheckpointStep
            self._step_types["memory_read"] = MemoryReadStep
            self._step_types["memory_write"] = MemoryWriteStep
            self._step_types["debate"] = DebateStep
            self._step_types["quick_debate"] = QuickDebateStep
            self._step_types["decision"] = DecisionStep
            self._step_types["switch"] = SwitchStep
            self._step_types["task"] = TaskStep
        except ImportError as e:
            logger.debug(f"Some Phase 2 step types not available: {e}")

    def register_step_type(self, type_name: str, step_class: Type[WorkflowStep]) -> None:
        """
        Register a custom step type.

        Args:
            type_name: Name to identify the step type
            step_class: Class implementing WorkflowStep protocol
        """
        self._step_types[type_name] = step_class
        logger.debug(f"Registered step type: {type_name}")

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def execute(
        self,
        definition: WorkflowDefinition,
        inputs: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Execute a workflow from the beginning.

        Args:
            definition: Workflow definition to execute
            inputs: Input parameters for the workflow
            workflow_id: Optional ID (generated if not provided)

        Returns:
            WorkflowResult with step results and final output
        """
        workflow_id = workflow_id or f"wf_{uuid.uuid4().hex[:12]}"
        inputs = inputs or {}

        logger.info(f"Starting workflow {workflow_id}: {definition.name}")

        # Create execution context
        context = WorkflowContext(
            workflow_id=workflow_id,
            definition_id=definition.id,
            inputs=inputs,
        )

        # Reset execution state
        self._results = []
        self._should_terminate = False
        self._termination_reason = None

        start_time = time.time()
        checkpoints_created = 0

        try:
            # Execute with overall timeout
            final_output = await asyncio.wait_for(
                self._execute_workflow(definition, context),
                timeout=self._config.total_timeout_seconds,
            )
            success = all(r.success for r in self._results)
            error = None

        except asyncio.TimeoutError:
            logger.error(f"Workflow timed out after {self._config.total_timeout_seconds}s")
            success = False
            error = f"Workflow timed out after {self._config.total_timeout_seconds}s"
            final_output = None

        except Exception as e:
            logger.exception(f"Workflow execution failed: {e}")
            success = False
            error = str(e)
            final_output = None

        total_duration = (time.time() - start_time) * 1000

        return WorkflowResult(
            workflow_id=workflow_id,
            definition_id=definition.id,
            success=success,
            steps=self._results.copy(),
            total_duration_ms=total_duration,
            final_output=final_output,
            error=error,
            checkpoints_created=checkpoints_created,
        )

    async def resume(
        self,
        workflow_id: str,
        checkpoint: WorkflowCheckpoint,
        definition: WorkflowDefinition,
    ) -> WorkflowResult:
        """
        Resume a workflow from a checkpoint.

        Args:
            workflow_id: ID of the workflow to resume
            checkpoint: Checkpoint to resume from
            definition: Workflow definition

        Returns:
            WorkflowResult from resumed execution
        """
        logger.info(f"Resuming workflow {workflow_id} from step {checkpoint.current_step}")

        # Restore context from checkpoint
        context = WorkflowContext(
            workflow_id=workflow_id,
            definition_id=checkpoint.definition_id,
            inputs=checkpoint.context_state.get("inputs", {}),
            step_outputs=checkpoint.step_outputs,
            state=checkpoint.context_state.get("state", {}),
        )

        # Reset execution state
        self._results = []
        self._should_terminate = False

        start_time = time.time()

        try:
            # Execute remaining steps
            final_output = await asyncio.wait_for(
                self._execute_from_step(
                    definition, context, checkpoint.current_step, checkpoint.completed_steps
                ),
                timeout=self._config.total_timeout_seconds,
            )
            success = all(r.success for r in self._results)
            error = None

        except asyncio.TimeoutError:
            success = False
            error = "Workflow timed out"
            final_output = None

        except Exception as e:
            logger.exception(f"Workflow resume failed: {e}")
            success = False
            error = str(e)
            final_output = None

        total_duration = (time.time() - start_time) * 1000

        return WorkflowResult(
            workflow_id=workflow_id,
            definition_id=definition.id,
            success=success,
            steps=self._results.copy(),
            total_duration_ms=total_duration,
            final_output=final_output,
            error=error,
        )

    # =========================================================================
    # Internal Execution
    # =========================================================================

    async def _execute_workflow(
        self,
        definition: WorkflowDefinition,
        context: WorkflowContext,
    ) -> Any:
        """Execute workflow from entry step."""
        if not definition.entry_step:
            raise ValueError("Workflow has no entry step")

        return await self._execute_from_step(definition, context, definition.entry_step, set())

    async def _execute_from_step(
        self,
        definition: WorkflowDefinition,
        context: WorkflowContext,
        start_step: str,
        completed_steps: set,
    ) -> Any:
        """Execute workflow starting from a specific step."""
        current_step_id = start_step
        final_output = None
        step_count = 0

        while current_step_id and not self._should_terminate:
            step_def = definition.get_step(current_step_id)
            if not step_def:
                logger.error(f"Step '{current_step_id}' not found in definition")
                break

            # Skip already completed steps
            if current_step_id in completed_steps:
                current_step_id = self._get_next_step(definition, current_step_id, context)
                continue

            # Execute the step
            result = await self._execute_step(step_def, context)
            self._results.append(result)
            step_count += 1

            # Store output in context
            if result.output is not None:
                context.step_outputs[current_step_id] = result.output
                final_output = result.output

            # Handle failure
            if not result.success:
                if self._config.stop_on_failure and not step_def.optional:
                    logger.error(f"Step '{current_step_id}' failed, stopping workflow")
                    break
                elif step_def.optional:
                    logger.warning(f"Optional step '{current_step_id}' failed, continuing")

            # Create checkpoint if enabled
            if (
                self._config.enable_checkpointing
                and step_count % self._config.checkpoint_interval_steps == 0
            ):
                await self._create_checkpoint(
                    context.workflow_id,
                    definition.id,
                    current_step_id,
                    set(r.step_id for r in self._results if r.success),
                    context,
                )

            # Determine next step
            current_step_id = self._get_next_step(definition, current_step_id, context)

        return final_output

    async def _execute_step(
        self,
        step_def: StepDefinition,
        context: WorkflowContext,
    ) -> StepResult:
        """Execute a single workflow step."""
        self._current_step = step_def.id
        started_at = datetime.now(timezone.utc)
        start_time = time.time()

        logger.debug(f"Executing step: {step_def.name} ({step_def.id})")

        # Update context with current step info
        context.current_step_id = step_def.id
        context.current_step_config = step_def.config

        # Get or create step instance
        step = self._get_step_instance(step_def)
        if step is None:
            return StepResult(
                step_id=step_def.id,
                step_name=step_def.name,
                status=StepStatus.FAILED,
                error=f"Unknown step type: {step_def.step_type}",
            )

        # Execute with retries
        retry_count = 0
        last_error = None

        while retry_count <= step_def.retries:
            try:
                output = await asyncio.wait_for(
                    step.execute(context),
                    timeout=step_def.timeout_seconds,
                )

                duration_ms = (time.time() - start_time) * 1000
                logger.debug(f"Completed step '{step_def.name}' in {duration_ms:.1f}ms")

                return StepResult(
                    step_id=step_def.id,
                    step_name=step_def.name,
                    status=StepStatus.COMPLETED,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                    output=output,
                    retry_count=retry_count,
                )

            except asyncio.TimeoutError:
                last_error = f"Timed out after {step_def.timeout_seconds}s"
                retry_count += 1
                if retry_count <= step_def.retries:
                    logger.warning(f"Step '{step_def.name}' timed out, retrying ({retry_count}/{step_def.retries})")

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                if retry_count <= step_def.retries:
                    logger.warning(f"Step '{step_def.name}' failed, retrying ({retry_count}/{step_def.retries})")

        # All retries exhausted
        duration_ms = (time.time() - start_time) * 1000

        if step_def.optional and self._config.skip_optional_on_timeout:
            return StepResult(
                step_id=step_def.id,
                step_name=step_def.name,
                status=StepStatus.SKIPPED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                error=last_error,
                retry_count=retry_count,
            )
        else:
            return StepResult(
                step_id=step_def.id,
                step_name=step_def.name,
                status=StepStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                error=last_error,
                retry_count=retry_count,
            )

    def _get_step_instance(self, step_def: StepDefinition) -> Optional[WorkflowStep]:
        """Get or create a step instance."""
        cache_key = f"{step_def.id}:{step_def.step_type}"

        if cache_key in self._step_instances:
            return self._step_instances[cache_key]

        step_class = self._step_types.get(step_def.step_type)
        if step_class is None:
            return None

        try:
            # Create step instance with config
            # Step classes are registered dynamically and may have various signatures
            step = step_class(name=step_def.name, config=step_def.config)  # type: ignore[call-arg]
            self._step_instances[cache_key] = step
            return step
        except Exception as e:
            logger.error(f"Failed to create step instance: {e}")
            return None

    def _get_next_step(
        self,
        definition: WorkflowDefinition,
        current_step_id: str,
        context: WorkflowContext,
    ) -> Optional[str]:
        """Determine the next step based on transitions and step output."""
        step_def = definition.get_step(current_step_id)
        if not step_def:
            return None

        # Check conditional transitions first
        transitions = definition.get_transitions_from(current_step_id)
        for transition in transitions:
            if self._evaluate_transition(transition, context):
                logger.debug(f"Taking transition {current_step_id} -> {transition.to_step}")
                return transition.to_step

        # Fall back to default next steps
        if step_def.next_steps:
            return step_def.next_steps[0]

        return None

    def _evaluate_transition(
        self,
        transition: TransitionRule,
        context: WorkflowContext,
    ) -> bool:
        """Evaluate a transition condition using AST-based evaluator."""
        try:
            namespace = {
                "inputs": context.inputs,
                "outputs": context.step_outputs,
                "state": context.state,
                "step_output": context.step_outputs.get(transition.from_step),
            }
            return safe_eval_bool(transition.condition, namespace)
        except SafeEvalError as e:
            logger.warning(f"Failed to evaluate transition condition: {e}")
            return False

    # =========================================================================
    # Checkpointing
    # =========================================================================

    async def _create_checkpoint(
        self,
        workflow_id: str,
        definition_id: str,
        current_step: str,
        completed_steps: set,
        context: WorkflowContext,
    ) -> WorkflowCheckpoint:
        """Create a checkpoint of current workflow state."""
        checkpoint_id = f"cp_{uuid.uuid4().hex[:12]}"

        # Create state snapshot
        context_state = {
            "inputs": context.inputs,
            "state": context.state,
            "metadata": context.metadata,
        }

        # Compute checksum
        state_json = json.dumps(context_state, sort_keys=True, default=str)
        checksum = hashlib.sha256(state_json.encode()).hexdigest()[:16]

        checkpoint = WorkflowCheckpoint(
            id=checkpoint_id,
            workflow_id=workflow_id,
            definition_id=definition_id,
            current_step=current_step,
            completed_steps=list(completed_steps),
            step_outputs=context.step_outputs.copy(),
            context_state=context_state,
            created_at=datetime.now(timezone.utc),
            checksum=checksum,
        )

        # Store checkpoint
        self._checkpoints[checkpoint_id] = checkpoint
        logger.debug(f"Created checkpoint {checkpoint_id} at step {current_step}")

        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    def list_checkpoints(self, workflow_id: str) -> List[WorkflowCheckpoint]:
        """List all checkpoints for a workflow."""
        return [
            cp for cp in self._checkpoints.values()
            if cp.workflow_id == workflow_id
        ]

    # =========================================================================
    # Termination Control
    # =========================================================================

    def request_termination(self, reason: str = "Requested") -> None:
        """Request early termination of workflow execution."""
        self._should_terminate = True
        self._termination_reason = reason
        logger.info(f"Workflow termination requested: {reason}")

    def check_termination(self) -> tuple[bool, Optional[str]]:
        """Check if termination has been requested."""
        return self._should_terminate, self._termination_reason

    @property
    def current_step(self) -> Optional[str]:
        """Get currently executing step ID."""
        return self._current_step

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        total_duration = sum(r.duration_ms for r in self._results)
        completed = sum(1 for r in self._results if r.status == StepStatus.COMPLETED)
        failed = sum(1 for r in self._results if r.status == StepStatus.FAILED)
        skipped = sum(1 for r in self._results if r.status == StepStatus.SKIPPED)

        return {
            "total_steps": len(self._results),
            "completed_steps": completed,
            "failed_steps": failed,
            "skipped_steps": skipped,
            "total_duration_ms": total_duration,
            "step_durations": {r.step_id: r.duration_ms for r in self._results},
            "current_step": self._current_step,
            "terminated_early": self._should_terminate,
            "termination_reason": self._termination_reason,
        }
