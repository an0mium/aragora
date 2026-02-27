"""Execute a crystallized spec via HardenedOrchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from aragora.interrogation.crystallizer import Spec

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRequest:
    """Request to execute a spec."""

    spec: Spec
    target: str = "self"  # "self" = aragora codebase, or git URL
    dry_run: bool = False
    budget_limit: float | None = None
    require_approval: bool = True


@dataclass
class ExecutionResult:
    """Result of executing a spec."""

    success: bool
    dry_run: bool = False
    goal_text: str = ""
    subtasks_completed: int = 0
    subtasks_failed: int = 0
    pr_url: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InterrogationExecutor:
    """Bridges the Interrogation Engine output to HardenedOrchestrator."""

    def __init__(self, orchestrator: Any | None = None) -> None:
        self._orchestrator = orchestrator

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        goal_text = request.spec.to_goal_text()

        if request.dry_run:
            return ExecutionResult(
                success=True,
                dry_run=True,
                goal_text=goal_text,
                metadata={"requirements_count": len(request.spec.requirements)},
            )

        if not self._orchestrator:
            try:
                from aragora.nomic.hardened_orchestrator import HardenedOrchestrator

                self._orchestrator = HardenedOrchestrator()
            except ImportError:
                logger.warning("HardenedOrchestrator not available")
                return ExecutionResult(
                    success=False,
                    goal_text=goal_text,
                    error="Orchestrator not available",
                )

        try:
            result = await self._orchestrator.execute_goal(
                goal=goal_text,
                context={"spec": request.spec.problem_statement},
            )
            return ExecutionResult(
                success=getattr(result, "success", False),
                goal_text=goal_text,
                subtasks_completed=getattr(result, "completed_subtasks", 0),
                subtasks_failed=getattr(result, "failed_subtasks", 0),
            )
        except Exception:
            logger.exception("Execution failed")
            return ExecutionResult(
                success=False,
                goal_text=goal_text,
                error="Execution failed",
            )
