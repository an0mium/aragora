"""
Task Router for Aragora.

Routes task types to their corresponding workflow definitions,
mapping high-level task intents (debate, code_edit, computer_use,
analysis, composite) to concrete workflow step sequences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Valid task types
VALID_TASK_TYPES = frozenset(
    {
        "debate",
        "code_edit",
        "computer_use",
        "analysis",
        "composite",
    }
)


@dataclass
class TaskRoute:
    """A route mapping a task type to its workflow definition.

    Attributes:
        task_type: The type of task (e.g. "debate", "code_edit").
        workflow_steps: List of step definitions for the workflow engine.
        required_capabilities: Agent capabilities needed for this task type.
        description: Human-readable description of what this route does.
    """

    task_type: str
    workflow_steps: list[dict[str, Any]]
    required_capabilities: list[str] = field(default_factory=list)
    description: str = ""


class TaskRouter:
    """Routes task types to workflow definitions.

    The router maintains a registry of task routes, each mapping a task
    type string to a sequence of workflow steps. Default routes are
    registered on initialization covering the five standard task types.

    Usage:
        router = TaskRouter()
        route = router.route("debate", "Should we adopt microservices?", {})
        # route.workflow_steps contains the step definitions
    """

    def __init__(self) -> None:
        self._routes: dict[str, TaskRoute] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default task routes for all standard task types."""
        self._routes["debate"] = TaskRoute(
            task_type="debate",
            workflow_steps=[
                {
                    "id": "debate_step",
                    "type": "debate",
                    "name": "Multi-Agent Debate",
                    "config": {
                        "rounds": 3,
                        "consensus": "majority",
                    },
                },
            ],
            required_capabilities=["debate", "reasoning"],
            description="Run a multi-agent debate to reach consensus on a question.",
        )

        self._routes["code_edit"] = TaskRoute(
            task_type="code_edit",
            workflow_steps=[
                {
                    "id": "analysis_step",
                    "type": "analysis",
                    "name": "Code Analysis",
                    "config": {
                        "mode": "code_review",
                    },
                },
                {
                    "id": "implementation_step",
                    "type": "implementation",
                    "name": "Code Implementation",
                    "config": {
                        "mode": "edit",
                    },
                },
                {
                    "id": "verification_step",
                    "type": "verification",
                    "name": "Change Verification",
                    "config": {
                        "run_tests": True,
                    },
                },
            ],
            required_capabilities=["code_generation", "code_review"],
            description="Analyze, implement, and verify code changes.",
        )

        self._routes["computer_use"] = TaskRoute(
            task_type="computer_use",
            workflow_steps=[
                {
                    "id": "computer_use_step",
                    "type": "computer_use",
                    "name": "Computer Use Task",
                    "config": {
                        "mode": "autonomous",
                        "screenshot_interval": 2,
                    },
                },
            ],
            required_capabilities=["computer_use", "vision"],
            description="Execute a task using computer use capabilities.",
        )

        self._routes["analysis"] = TaskRoute(
            task_type="analysis",
            workflow_steps=[
                {
                    "id": "analysis_debate_step",
                    "type": "debate",
                    "name": "Analysis Debate",
                    "config": {
                        "rounds": 2,
                        "consensus": "majority",
                        "prompt_prefix": "Analyze the following: ",
                    },
                },
            ],
            required_capabilities=["reasoning", "analysis"],
            description="Run an analysis-focused debate to evaluate a topic.",
        )

        self._routes["composite"] = TaskRoute(
            task_type="composite",
            workflow_steps=[
                {
                    "id": "planning_step",
                    "type": "debate",
                    "name": "Task Planning",
                    "config": {
                        "rounds": 1,
                        "consensus": "majority",
                        "prompt_prefix": "Plan the following multi-step task: ",
                    },
                },
                {
                    "id": "execution_step",
                    "type": "implementation",
                    "name": "Task Execution",
                    "config": {
                        "mode": "multi_step",
                    },
                },
                {
                    "id": "review_step",
                    "type": "debate",
                    "name": "Result Review",
                    "config": {
                        "rounds": 1,
                        "consensus": "majority",
                        "prompt_prefix": "Review the results of: ",
                    },
                },
            ],
            required_capabilities=["reasoning", "code_generation"],
            description="Multi-step workflow: plan, execute, and review.",
        )

    def register(self, route: TaskRoute) -> None:
        """Register a custom task route.

        Args:
            route: The TaskRoute to register. Overwrites any existing
                   route for the same task_type.
        """
        self._routes[route.task_type] = route
        logger.info("Registered task route: %s", route.task_type)

    def route(self, task_type: str, goal: str, context: dict[str, Any]) -> TaskRoute:
        """Return the appropriate route for a given task type.

        If the task type has a registered route, returns it directly.
        For unknown types, returns a fallback debate route.

        Args:
            task_type: The type of task to route.
            goal: The goal description (used for context-aware routing).
            context: Additional context for routing decisions.

        Returns:
            A TaskRoute with workflow step definitions.

        Raises:
            ValueError: If task_type is empty or None.
        """
        if not task_type:
            raise ValueError("task_type must not be empty")

        if task_type in self._routes:
            return self._routes[task_type]

        # Fallback: wrap unknown types in a debate route
        logger.warning("Unknown task type '%s', falling back to debate route", task_type)
        return TaskRoute(
            task_type=task_type,
            workflow_steps=[
                {
                    "id": "fallback_debate_step",
                    "type": "debate",
                    "name": f"Debate: {task_type}",
                    "config": {
                        "rounds": 2,
                        "consensus": "majority",
                    },
                },
            ],
            required_capabilities=["reasoning"],
            description=f"Fallback debate route for unknown type '{task_type}'.",
        )

    @property
    def registered_types(self) -> list[str]:
        """Return list of registered task type names."""
        return sorted(self._routes.keys())

    def get_route(self, task_type: str) -> TaskRoute | None:
        """Get a route by task type without fallback.

        Args:
            task_type: The task type to look up.

        Returns:
            The TaskRoute if registered, None otherwise.
        """
        return self._routes.get(task_type)
