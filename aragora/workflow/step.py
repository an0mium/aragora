"""
Workflow Step Protocol and Base Classes.

Steps are the building blocks of workflows. Each step has:
- A name for identification
- Configuration for parameterization
- An execute method that performs the work
- Optional checkpoint/resume support
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from aragora.workflow.safe_eval import SafeEvalError, safe_eval_bool


@runtime_checkable
class WorkflowStep(Protocol):
    """
    Protocol for workflow steps.

    Any class implementing this protocol can be used as a workflow step.
    This mirrors the Phase protocol from PhaseExecutor but is more generic.
    """

    @property
    def name(self) -> str:
        """Step name for identification."""
        ...

    async def execute(self, context: "WorkflowContext") -> Any:
        """
        Execute the step with given context.

        Args:
            context: Workflow context with inputs and state

        Returns:
            Step output (will be stored in context.step_outputs)
        """
        ...


@dataclass
class WorkflowContext:
    """
    Context passed to workflow steps during execution.

    Provides access to:
    - Workflow inputs
    - Previous step outputs
    - Shared state
    - Step configuration
    """

    workflow_id: str
    definition_id: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    step_outputs: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Current step info (set by engine during execution)
    current_step_id: Optional[str] = None
    current_step_config: Dict[str, Any] = field(default_factory=dict)

    def get_input(self, key: str, default: Any = None) -> Any:
        """Get a workflow input value."""
        return self.inputs.get(key, default)

    def get_step_output(self, step_id: str, default: Any = None) -> Any:
        """Get output from a previous step."""
        return self.step_outputs.get(step_id, default)

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get shared state value."""
        return self.state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set shared state value."""
        self.state[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get current step configuration value."""
        return self.current_step_config.get(key, default)


class BaseStep(ABC):
    """
    Base class for workflow steps with common functionality.

    Extend this class to create custom steps with:
    - Automatic configuration handling
    - Checkpoint support
    - Logging integration
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        """Step name."""
        return self._name

    @property
    def config(self) -> Dict[str, Any]:
        """Step configuration."""
        return self._config

    @abstractmethod
    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the step."""
        ...

    async def checkpoint(self) -> Dict[str, Any]:
        """
        Save step state for checkpointing.

        Override to save custom state that can be restored on resume.
        """
        return {}

    async def restore(self, state: Dict[str, Any]) -> None:
        """
        Restore step state from checkpoint.

        Override to restore custom state on resume.
        """
        pass

    def validate_config(self) -> bool:
        """
        Validate step configuration.

        Override to add custom validation logic.
        """
        return True


class AgentStep(BaseStep):
    """
    Step that delegates work to an AI agent.

    Used for steps that require LLM inference, such as:
    - Content generation
    - Analysis and review
    - Decision making
    """

    def __init__(
        self,
        name: str,
        agent_type: str,
        prompt_template: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, config)
        self.agent_type = agent_type
        self.prompt_template = prompt_template

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the agent step."""
        # Import here to avoid circular dependencies
        from aragora.agents import create_agent

        # Build prompt from template and context
        prompt = self._build_prompt(context)

        # Create and run agent
        agent = create_agent(self.agent_type)
        response = await agent.generate(prompt)

        return {"response": response, "agent_type": self.agent_type}

    def _build_prompt(self, context: WorkflowContext) -> str:
        """Build prompt from template and context."""
        # Simple template substitution
        prompt = self.prompt_template
        for key, value in context.inputs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        for step_id, output in context.step_outputs.items():
            prompt = prompt.replace(f"{{step.{step_id}}}", str(output))
        return prompt


class ParallelStep(BaseStep):
    """
    Step that executes multiple sub-steps in parallel.

    Used for hive-mind patterns where multiple agents work concurrently.
    """

    def __init__(
        self,
        name: str,
        sub_steps: list[WorkflowStep],
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, config)
        self.sub_steps = sub_steps

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute all sub-steps in parallel."""
        import asyncio

        tasks = [step.execute(context) for step in self.sub_steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results by step name
        outputs = {}
        for i, result in enumerate(results):
            step_name = self.sub_steps[i].name
            if isinstance(result, Exception):
                outputs[step_name] = {"error": str(result)}
            else:
                outputs[step_name] = result

        return outputs


class ConditionalStep(BaseStep):
    """
    Step that executes based on a condition.

    The condition is evaluated against the context and determines
    whether to execute the wrapped step or skip it.
    """

    def __init__(
        self,
        name: str,
        wrapped_step: WorkflowStep,
        condition: str,  # Python expression
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, config)
        self.wrapped_step = wrapped_step
        self.condition = condition

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute wrapped step if condition is true."""
        # Evaluate condition
        should_execute = self._evaluate_condition(context)

        if should_execute:
            return await self.wrapped_step.execute(context)
        else:
            return {"skipped": True, "condition": self.condition}

    def _evaluate_condition(self, context: WorkflowContext) -> bool:
        """Safely evaluate condition expression using AST-based evaluator."""
        try:
            # Create evaluation namespace
            namespace = {
                "inputs": context.inputs,
                "outputs": context.step_outputs,
                "state": context.state,
            }
            return safe_eval_bool(self.condition, namespace)
        except SafeEvalError:
            return False


class LoopStep(BaseStep):
    """
    Step that repeats until a condition is met.

    Useful for iterative refinement patterns.
    """

    def __init__(
        self,
        name: str,
        wrapped_step: WorkflowStep,
        condition: str,  # Exit condition (stop when True)
        max_iterations: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, config)
        self.wrapped_step = wrapped_step
        self.condition = condition
        self.max_iterations = max_iterations

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute wrapped step until condition is met."""
        iterations = 0
        outputs = []

        while iterations < self.max_iterations:
            # Execute wrapped step
            result = await self.wrapped_step.execute(context)
            outputs.append(result)
            iterations += 1

            # Update state for condition evaluation
            context.state["loop_iteration"] = iterations
            context.state["loop_last_output"] = result

            # Check exit condition
            if self._evaluate_condition(context):
                break

        return {
            "iterations": iterations,
            "outputs": outputs,
            "final_output": outputs[-1] if outputs else None,
        }

    def _evaluate_condition(self, context: WorkflowContext) -> bool:
        """Safely evaluate exit condition using AST-based evaluator."""
        try:
            namespace = {
                "inputs": context.inputs,
                "outputs": context.step_outputs,
                "state": context.state,
                "iteration": context.state.get("loop_iteration", 0),
                "last_output": context.state.get("loop_last_output"),
            }
            return safe_eval_bool(self.condition, namespace)
        except SafeEvalError:
            return False
