"""
Proposal phase step executors.

This module contains executors for steps related to the proposal phase:
- StepExecutor: Abstract base class for all executors
- AgentStepExecutor: Execute steps using AI agents
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from aragora.nomic.molecules.base import MoleculeStep

__all__ = ["AgentStepExecutor", "StepExecutor"]

logger = logging.getLogger(__name__)


def _step_name(step: MoleculeStep) -> str:
    """Return a stable step name for logging and default results."""
    return getattr(step, "name", type(step).__name__)


class StepExecutor(ABC):
    """Abstract base class for step executors."""

    @abstractmethod
    async def execute(
        self,
        step: MoleculeStep,
        context: dict[str, Any],
    ) -> Any:
        """
        Execute a step.

        Args:
            step: The step to execute
            context: Execution context (previous results, etc.)

        Returns:
            Step result
        """
        raise NotImplementedError("Step executors must implement execute().")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class AgentStepExecutor(StepExecutor):
    """Execute steps using AI agents."""

    def __init__(self, agent_fn: Callable[..., Any] | None = None) -> None:
        self._agent_fn = agent_fn

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute step via agent."""
        step_name = _step_name(step)
        logger.info("Agent executing step: %s", step_name)
        if self._agent_fn is not None:
            result = self._agent_fn(step, context)
            if inspect.isawaitable(result):
                return await result
            return result
        return {"status": "executed", "step": step_name}

    def __repr__(self) -> str:
        has_fn = self._agent_fn is not None
        return f"AgentStepExecutor(agent_fn={'set' if has_fn else 'None'})"
