"""
Workflow Step Protocol and Base Classes.

Steps are the building blocks of workflows. Each step has:
- A name for identification
- Configuration for parameterization
- An execute method that performs the work
- Optional checkpoint/resume support
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable
from collections.abc import Callable

from aragora.agents.base import AgentType
from aragora.workflow.safe_eval import SafeEvalError, safe_eval_bool

logger = logging.getLogger(__name__)


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

    async def execute(self, context: WorkflowContext) -> Any:
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
    inputs: dict[str, Any] = field(default_factory=dict)
    step_outputs: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Current step info (set by engine during execution)
    current_step_id: str | None = None
    current_step_config: dict[str, Any] = field(default_factory=dict)
    event_callback: Callable[[str, dict[str, Any]], None] | None = None

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

    def emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit a workflow event if callback is configured."""
        if self.event_callback is None:
            return
        try:
            self.event_callback(event_type, payload)
        except (RuntimeError, ValueError, TypeError, OSError, AttributeError) as exc:
            logger.debug("Workflow event callback failed: %s", exc)


class BaseStep(ABC):
    """
    Base class for workflow steps with common functionality.

    Extend this class to create custom steps with:
    - Automatic configuration handling
    - Checkpoint support
    - Logging integration
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        """Step name."""
        return self._name

    @property
    def config(self) -> dict[str, Any]:
        """Step configuration."""
        return self._config

    @abstractmethod
    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the step."""
        ...

    async def checkpoint(self) -> dict[str, Any]:
        """
        Save step state for checkpointing.

        Override to save custom state that can be restored on resume.
        """
        return {}

    async def restore(self, state: dict[str, Any]) -> None:
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

    Config options:
    - agent_type: The type of agent to use (claude, codex, gemini, etc.)
    - prompt_template: Template string for the prompt
    - coding_harness: Optional harness config for agents without native coding support
      - harness: "kilocode"
      - provider_id: KiloCode provider (e.g., "gemini-explorer")
      - mode: KiloCode mode (code, architect, ask, debug)
    """

    def __init__(
        self,
        name: str,
        agent_type: str | None = None,
        prompt_template: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(name, config)
        # Extract from config if not passed directly (for engine compatibility)
        self.agent_type = agent_type or self._config.get("agent_type", "claude")
        self.prompt_template = prompt_template or self._config.get("prompt_template", "")
        self.coding_harness = self._config.get("coding_harness")
        self.agent_pool = self._normalize_agent_pool(self._config.get("agent_pool"))

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the agent step."""
        # Import here to avoid circular dependencies
        from aragora.agents import create_agent

        step_config = {**self._config, **context.current_step_config}

        # Build prompt from template and context
        prompt = self._build_prompt(context)

        # Heterogeneous agent pool orchestration
        pool = self._normalize_agent_pool(step_config.get("agent_pool")) or self.agent_pool
        if pool:
            return await self._execute_agent_pool(prompt, context, pool, step_config)

        # Check if we should use a coding harness (e.g., KiloCode for Gemini)
        harness_config = step_config.get("coding_harness") or self.coding_harness
        if isinstance(harness_config, dict) and harness_config.get("harness") == "kilocode":
            return await self._execute_with_kilocode(prompt, context, harness_config)

        # Single-agent power sampling
        power_sampling = step_config.get("power_sampling")
        if power_sampling:
            return await self._execute_with_power_sampling(
                prompt,
                step_config,
                power_sampling,
            )

        # Create and run agent normally
        agent_type = str(step_config.get("agent_type", self.agent_type))
        agent = create_agent(cast(AgentType, agent_type))
        response = await agent.generate(prompt)

        return {"response": response, "agent_type": agent_type}

    def _normalize_agent_pool(self, value: Any | None) -> list[str]:
        """Normalize agent pool input into a list of agent type strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _get_response_scorer(self, config: dict[str, Any]) -> Any:
        """Resolve a response scorer for selection and power sampling."""
        scorer_fn = config.get("response_scorer") or config.get("quality_scorer")
        if scorer_fn and callable(scorer_fn):

            class _CallableScorer:
                def __init__(self, fn: Callable[[str, str], Any]) -> None:
                    self._fn = fn

                async def score(self, response: str, prompt: str) -> float:
                    value = self._fn(response, prompt)
                    if inspect.isawaitable(value):
                        value = await value
                    return float(value)

            return _CallableScorer(scorer_fn)

        try:
            from aragora.reasoning.sampling.power_sampling import DefaultScorer

            return DefaultScorer()
        except ImportError:
            return None

    async def _execute_with_power_sampling(
        self,
        prompt: str,
        config: dict[str, Any],
        power_sampling: Any,
    ) -> dict[str, Any]:
        """Execute a single agent with power sampling."""
        from aragora.agents import create_agent

        agent_type = str(config.get("agent_type", self.agent_type))
        agent = create_agent(cast(AgentType, agent_type))

        scorer = self._get_response_scorer(config)
        if scorer is None:
            response = await agent.generate(prompt)
            return {"response": response, "agent_type": agent_type}

        try:
            from aragora.reasoning.sampling.power_sampling import PowerSampler, PowerSamplingConfig

            cfg_dict: dict[str, Any] = {}
            if isinstance(power_sampling, dict):
                cfg_dict.update(power_sampling)
            if "n_samples" not in cfg_dict:
                samples_override = config.get("samples_per_agent")
                if isinstance(samples_override, int) and samples_override > 0:
                    cfg_dict["n_samples"] = samples_override

            valid_keys = set(PowerSamplingConfig.__dataclass_fields__.keys())
            filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
            sampler_config = PowerSamplingConfig(**filtered)
            sampler = PowerSampler(config=sampler_config)
            result = await sampler.sample_best_reasoning(agent.generate, prompt, scorer)

            return {
                "response": result.best_response,
                "agent_type": agent_type,
                "sampling": {
                    "best_score": result.best_score,
                    "confidence": result.confidence,
                    "samples": len(result.all_samples),
                },
            }
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.debug("Power sampling unavailable; falling back to single sample: %s", exc)
            response = await agent.generate(prompt)
            return {"response": response, "agent_type": agent_type}

    async def _execute_agent_pool(
        self,
        prompt: str,
        context: WorkflowContext,
        pool: list[str],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute multiple agents in parallel and select the best response."""
        from aragora.agents import create_agent

        samples_per_agent = int(config.get("samples_per_agent", 1) or 1)
        selection_strategy = str(config.get("selection_strategy", "best_score"))
        include_candidates = bool(config.get("include_candidates", False))
        max_parallel = int(config.get("max_parallel_agents", len(pool)) or len(pool))
        max_parallel = max(1, min(max_parallel, len(pool)))
        semaphore = asyncio.Semaphore(max_parallel)
        scorer = self._get_response_scorer(config)

        async def _generate_one(agent_type: str) -> dict[str, Any] | None:
            async with semaphore:
                agent = create_agent(cast(AgentType, agent_type))

                # Use power sampling when configured or sampling count > 1
                if (samples_per_agent > 1 or config.get("power_sampling")) and scorer is not None:
                    try:
                        from aragora.reasoning.sampling.power_sampling import (
                            PowerSampler,
                            PowerSamplingConfig,
                        )

                        cfg_dict: dict[str, Any] = {}
                        if isinstance(config.get("power_sampling"), dict):
                            cfg_dict.update(config["power_sampling"])
                        if "n_samples" not in cfg_dict:
                            cfg_dict["n_samples"] = samples_per_agent

                        valid_keys = set(PowerSamplingConfig.__dataclass_fields__.keys())
                        filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
                        sampler = PowerSampler(config=PowerSamplingConfig(**filtered))
                        result = await sampler.sample_best_reasoning(agent.generate, prompt, scorer)
                        return {
                            "agent_type": agent_type,
                            "response": result.best_response,
                            "score": result.best_score,
                            "confidence": result.confidence,
                            "samples": len(result.all_samples),
                        }
                    except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
                        logger.debug("Agent pool power sampling failed for %s: %s", agent_type, exc)

                try:
                    response = await agent.generate(prompt)
                except (RuntimeError, ValueError, TypeError, OSError, ConnectionError) as exc:
                    logger.debug("Agent pool generation failed for %s: %s", agent_type, exc)
                    return None

                score = None
                if scorer is not None:
                    try:
                        score = await scorer.score(response, prompt)
                    except (ValueError, TypeError, RuntimeError):
                        score = None

                return {"agent_type": agent_type, "response": response, "score": score}

        results = await asyncio.gather(*[_generate_one(agent) for agent in pool])
        candidates = [c for c in results if c is not None]
        if not candidates:
            return {"success": False, "error": "All agent pool generations failed"}

        if scorer is not None:
            for candidate in candidates:
                if candidate.get("score") is None and candidate.get("response"):
                    try:
                        candidate["score"] = await scorer.score(candidate["response"], prompt)
                    except (ValueError, TypeError, RuntimeError):
                        candidate["score"] = 0.0

        best = candidates[0]
        if selection_strategy in {"best_score", "power_law"}:
            best = max(
                candidates,
                key=lambda c: float(c.get("score")) if c.get("score") is not None else 0.0,
            )

        output: dict[str, Any] = {
            "response": best.get("response", ""),
            "agent_type": best.get("agent_type"),
            "selection_strategy": selection_strategy,
        }
        if best.get("score") is not None:
            output["best_score"] = best.get("score")

        if include_candidates:
            output["candidates"] = [
                {
                    "agent_type": c.get("agent_type"),
                    "response": c.get("response"),
                    "score": c.get("score"),
                    "confidence": c.get("confidence"),
                    "samples": c.get("samples"),
                }
                for c in candidates
            ]

        return output

    async def _execute_with_kilocode(
        self,
        prompt: str,
        context: WorkflowContext,
        harness_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute using KiloCode as the coding harness."""
        from aragora.agents.cli_agents import KiloCodeAgent

        # coding_harness is guaranteed to be a dict when this method is called
        harness_config = harness_config or self.coding_harness or {}
        provider_id = harness_config.get("provider_id", "gemini-explorer")
        mode = harness_config.get("mode", "code")

        # Create KiloCode agent with the specified provider
        agent = KiloCodeAgent(
            name=f"kilocode-{provider_id}",
            provider_id=provider_id,
            mode=mode,
        )

        response = await agent.generate(prompt)

        return {
            "response": response,
            "agent_type": self.agent_type,
            "coding_harness": "kilocode",
            "provider_id": provider_id,
        }

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
        config: dict[str, Any] | None = None,
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
                outputs[step_name] = cast(dict[str, Any], result)

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
        config: dict[str, Any] | None = None,
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
        config: dict[str, Any] | None = None,
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
