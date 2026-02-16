"""Pipeline Bridge - Route Nomic Loop output through the DecisionPlan pipeline.

Converts Nomic Loop debate output (goal, subtasks, consensus) into a
DecisionPlan, then executes it via PlanExecutor. This gives self-improvement
access to risk registers, verification plans, execution receipts, and KM
ingestion for free.

Usage:
    from aragora.nomic.pipeline_bridge import NomicPipelineBridge

    bridge = NomicPipelineBridge(repo_path=Path.cwd())
    outcome = await bridge.execute_via_pipeline(
        goal="Improve error handling",
        subtasks=decomposition.subtasks,
        consensus_result=debate_result,  # Optional DebateResult
        execution_mode="hybrid",
    )

The bridge is intentionally thin -- it transforms Nomic types to Pipeline
types and delegates execution to PlanExecutor.

Stability: ALPHA
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

from aragora.nomic.task_decomposer import SubTask, TaskDecomposition

logger = logging.getLogger(__name__)


def _subtask_to_implement_task(
    subtask: SubTask,
    index: int,
) -> Any:
    """Convert a Nomic SubTask to a pipeline ImplementTask.

    Args:
        subtask: The Nomic SubTask from task decomposition.
        index: 1-based index for generating task IDs.

    Returns:
        An ImplementTask ready for the pipeline.
    """
    from aragora.implement.types import ImplementTask

    # Map Nomic complexity levels to pipeline complexity levels
    complexity_map = {
        "low": "simple",
        "medium": "moderate",
        "high": "complex",
    }
    complexity = complexity_map.get(subtask.estimated_complexity, "moderate")

    # Convert Nomic dependency IDs to pipeline task IDs
    # Nomic subtasks use their own ID scheme; we remap to task-N format
    dependencies: list[str] = []
    # Dependencies will be resolved after all tasks are created

    return ImplementTask(
        id=f"task-{index}",
        description=subtask.description,
        files=subtask.file_scope,
        complexity=complexity,
        dependencies=dependencies,
    )


def _build_synthetic_debate_result(
    goal: str,
    subtasks: list[SubTask],
    dissent: list[str] | None = None,
) -> Any:
    """Build a minimal DebateResult for the DecisionPlanFactory.

    When the Nomic Loop debate phase does not produce a full DebateResult
    (e.g., when using heuristic decomposition), this constructs a synthetic
    one with enough data for the factory to generate useful risk registers
    and verification plans.

    Args:
        goal: The high-level goal that was debated/decomposed.
        subtasks: The decomposed subtasks.
        dissent: Optional dissenting views from the debate.

    Returns:
        A DebateResult populated with synthetic data.
    """
    from aragora.core_types import DebateResult

    debate_id = f"nomic-{uuid.uuid4().hex[:12]}"

    # Build a final answer from the subtask descriptions
    final_answer_lines = [f"Implementation plan for: {goal}\n"]
    for i, st in enumerate(subtasks, 1):
        files_str = ""
        if st.file_scope:
            files_str = " (" + ", ".join(f"`{f}`" for f in st.file_scope[:3]) + ")"
            final_answer_lines.append(f"{i}. {st.description}{files_str}")
        else:
            final_answer_lines.append(f"{i}. {st.description}")

    return DebateResult(
        debate_id=debate_id,
        task=goal,
        final_answer="\n".join(final_answer_lines),
        confidence=0.75,  # Moderate confidence for synthetic results
        consensus_reached=True,
        rounds_used=1,
        status="completed",
        participants=["nomic-orchestrator"],
        dissenting_views=dissent or [],
    )


def _resolve_dependencies(
    subtasks: list[SubTask],
    implement_tasks: list[Any],
) -> None:
    """Resolve Nomic subtask dependencies to pipeline task IDs.

    Modifies implement_tasks in-place to set dependency references.

    Args:
        subtasks: Original Nomic subtasks (with their ID scheme).
        implement_tasks: Pipeline ImplementTasks (with task-N IDs).
    """
    # Build a mapping from Nomic subtask ID to pipeline task ID
    nomic_to_pipeline: dict[str, str] = {}
    for i, st in enumerate(subtasks):
        nomic_to_pipeline[st.id] = f"task-{i + 1}"

    # Resolve dependencies
    for i, st in enumerate(subtasks):
        deps: list[str] = []
        for dep_id in st.dependencies:
            pipeline_id = nomic_to_pipeline.get(dep_id)
            if pipeline_id:
                deps.append(pipeline_id)
        implement_tasks[i].dependencies = deps


class NomicPipelineBridge:
    """Bridge between Nomic Loop and the DecisionPlan execution pipeline.

    Transforms Nomic subtasks and debate output into a DecisionPlan,
    then executes it via PlanExecutor to get risk registers, verification
    plans, receipts, and KM ingestion.

    Args:
        repo_path: Path to the repository root.
        budget_limit_usd: Optional budget cap for execution.
        execution_mode: Execution mode for PlanExecutor
            ("workflow", "hybrid", "fabric", "computer_use").
    """

    def __init__(
        self,
        repo_path: Path | None = None,
        budget_limit_usd: float | None = None,
        execution_mode: str = "hybrid",
    ) -> None:
        self._repo_path = repo_path or Path.cwd()
        self._budget_limit_usd = budget_limit_usd
        self._execution_mode = execution_mode

    def build_decision_plan(
        self,
        goal: str,
        subtasks: list[SubTask],
        debate_result: Any | None = None,
        dissent: list[str] | None = None,
    ) -> Any:
        """Build a DecisionPlan from Nomic Loop output.

        Args:
            goal: The high-level goal.
            subtasks: Decomposed subtasks from TaskDecomposer.
            debate_result: Optional DebateResult from the debate phase.
                If None, a synthetic one is constructed.
            dissent: Optional dissenting views (used when debate_result
                is None to populate the risk register).

        Returns:
            A DecisionPlan ready for approval and execution.
        """
        from aragora.implement.types import ImplementPlan, ImplementTask
        from aragora.pipeline.decision_plan.core import ApprovalMode
        from aragora.pipeline.decision_plan.factory import DecisionPlanFactory

        # Convert Nomic subtasks to ImplementTasks
        implement_tasks = [
            _subtask_to_implement_task(st, i + 1) for i, st in enumerate(subtasks)
        ]

        # Resolve cross-task dependencies
        _resolve_dependencies(subtasks, implement_tasks)

        # Build the ImplementPlan
        design_text = f"Nomic Loop plan for: {goal}"
        design_hash = hashlib.sha256(design_text.encode()).hexdigest()
        implement_plan = ImplementPlan(
            design_hash=design_hash,
            tasks=implement_tasks,
        )

        # Use real debate result or build synthetic one
        result = debate_result
        if result is None:
            result = _build_synthetic_debate_result(goal, subtasks, dissent)

        # Create the DecisionPlan via the factory
        plan = DecisionPlanFactory.from_debate_result(
            result,
            budget_limit_usd=self._budget_limit_usd,
            approval_mode=ApprovalMode.NEVER,  # Self-improvement is automated
            repo_path=self._repo_path,
            implement_plan=implement_plan,
            metadata={
                "source": "nomic_loop",
                "goal": goal,
                "subtask_count": len(subtasks),
            },
        )

        logger.info(
            "Built DecisionPlan %s from %d Nomic subtasks (risks=%d, verifications=%d)",
            plan.id,
            len(subtasks),
            len(plan.risk_register.risks) if plan.risk_register else 0,
            len(plan.verification_plan.test_cases) if plan.verification_plan else 0,
        )

        return plan

    async def execute_via_pipeline(
        self,
        goal: str,
        subtasks: list[SubTask],
        debate_result: Any | None = None,
        dissent: list[str] | None = None,
        execution_mode: str | None = None,
    ) -> Any:
        """Build a DecisionPlan and execute it via PlanExecutor.

        This is the main entry point for routing Nomic Loop output through
        the production pipeline.

        Args:
            goal: The high-level goal.
            subtasks: Decomposed subtasks from TaskDecomposer.
            debate_result: Optional DebateResult from the debate phase.
            dissent: Optional dissenting views for risk analysis.
            execution_mode: Override the default execution mode.

        Returns:
            A PlanOutcome with execution results, receipt ID, and lessons.
        """
        from aragora.pipeline.executor import PlanExecutor

        plan = self.build_decision_plan(
            goal=goal,
            subtasks=subtasks,
            debate_result=debate_result,
            dissent=dissent,
        )

        mode = execution_mode or self._execution_mode

        executor = PlanExecutor(
            execution_mode=mode,  # type: ignore[arg-type]
            repo_path=self._repo_path,
        )

        logger.info(
            "Executing DecisionPlan %s via PlanExecutor (mode=%s, tasks=%d)",
            plan.id,
            mode,
            len(plan.implement_plan.tasks) if plan.implement_plan else 0,
        )

        outcome = await executor.execute(plan, execution_mode=mode)  # type: ignore[arg-type]

        logger.info(
            "PlanExecutor completed: success=%s, tasks=%d/%d, receipt=%s",
            outcome.success,
            outcome.tasks_completed,
            outcome.tasks_total,
            outcome.receipt_id or "none",
        )

        return outcome

    async def execute_decomposition_via_pipeline(
        self,
        goal: str,
        decomposition: TaskDecomposition,
        debate_result: Any | None = None,
        dissent: list[str] | None = None,
        execution_mode: str | None = None,
    ) -> Any:
        """Convenience method: execute a full TaskDecomposition via pipeline.

        Args:
            goal: The high-level goal.
            decomposition: The TaskDecomposition from TaskDecomposer.
            debate_result: Optional DebateResult from debate phase.
            dissent: Optional dissenting views for risk analysis.
            execution_mode: Override the default execution mode.

        Returns:
            A PlanOutcome with execution results.
        """
        return await self.execute_via_pipeline(
            goal=goal,
            subtasks=decomposition.subtasks,
            debate_result=debate_result,
            dissent=dissent,
            execution_mode=execution_mode,
        )
