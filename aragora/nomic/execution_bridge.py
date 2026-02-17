"""
Execution Bridge -- translates debate decisions into agent instructions.

Converts PrioritizedGoal and SubTask objects into structured execution
plans that can be dispatched to Claude Code sessions (or any OpenClaw-compatible
agent) working in isolated worktrees.

The bridge handles:
1. Instruction generation: SubTask -> structured Markdown prompt
2. Context packaging: debate rationale + file hints + constraints -> agent context
3. Result ingestion: execution output -> Knowledge Mound observations
4. Verification dispatch: changed files -> PRReviewRunner.review_diff()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.nomic.task_decomposer import SubTask
    from aragora.nomic.meta_planner import PrioritizedGoal

logger = logging.getLogger(__name__)


@dataclass
class ExecutionInstruction:
    """Structured instruction for an execution agent."""

    subtask_id: str
    track: str
    objective: str  # What to accomplish
    context: str  # Debate rationale, constraints, historical learnings
    file_hints: list[str]  # Files likely to be modified
    success_criteria: list[str]  # How to verify success
    constraints: list[str]  # Don'ts and guardrails
    worktree_path: str | None = None  # Isolated worktree if applicable
    budget_limit_usd: float = 0.0  # Cost cap

    def to_agent_prompt(self) -> str:
        """Render as a structured Markdown prompt for Claude Code."""
        sections = [
            f"# Task: {self.objective}",
            "",
            "## Context",
            self.context,
            "",
        ]
        if self.file_hints:
            sections.append("## Relevant Files")
            for f in self.file_hints:
                sections.append(f"- `{f}`")
            sections.append("")
        if self.success_criteria:
            sections.append("## Success Criteria")
            for c in self.success_criteria:
                sections.append(f"- {c}")
            sections.append("")
        if self.constraints:
            sections.append("## Constraints")
            for c in self.constraints:
                sections.append(f"- {c}")
            sections.append("")
        sections.append("## Verification")
        sections.append("After making changes, run relevant tests to verify:")
        sections.append("```bash")
        sections.append("python -m pytest <relevant_test_files> -x -q --timeout=30")
        sections.append("```")
        return "\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON transport."""
        return {
            "subtask_id": self.subtask_id,
            "track": self.track,
            "objective": self.objective,
            "context": self.context,
            "file_hints": self.file_hints,
            "success_criteria": self.success_criteria,
            "constraints": self.constraints,
            "worktree_path": self.worktree_path,
            "budget_limit_usd": self.budget_limit_usd,
        }


@dataclass
class ExecutionResult:
    """Result from an execution agent."""

    subtask_id: str
    success: bool
    files_changed: list[str] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    error: str | None = None
    duration_seconds: float = 0.0
    tokens_used: int = 0
    diff_summary: str = ""
    agent_observations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON transport."""
        return {
            "subtask_id": self.subtask_id,
            "success": self.success,
            "files_changed": self.files_changed,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used,
            "diff_summary": self.diff_summary[:500],
        }


class ExecutionBridge:
    """Translates debate decisions into executable agent instructions.

    The bridge sits between MetaPlanner/TaskDecomposer (planning) and
    WorktreeManager/BranchCoordinator (execution), providing:

    1. Instruction generation from SubTasks
    2. Context packaging with debate rationale
    3. Result ingestion back into Knowledge Mound
    4. Verification via PRReviewRunner

    Usage:
        bridge = ExecutionBridge()

        # Generate instructions from a subtask
        instruction = bridge.create_instruction(subtask, goal, debate_context)

        # After execution, ingest results
        bridge.ingest_result(result)

        # Verify changes via review
        verification = await bridge.verify_changes(result)
    """

    def __init__(
        self,
        base_constraints: list[str] | None = None,
        enable_km_ingestion: bool = True,
        enable_verification: bool = True,
    ):
        self.base_constraints = base_constraints or [
            "Do not modify CLAUDE.md or protected files without approval",
            "Always run tests after changes",
            "Use specific exception types, never bare except",
            "Preserve existing functionality",
        ]
        self.enable_km_ingestion = enable_km_ingestion
        self.enable_verification = enable_verification
        self._results: list[ExecutionResult] = []

    def create_instruction(
        self,
        subtask: SubTask,
        goal: PrioritizedGoal | None = None,
        debate_context: str = "",
        extra_constraints: list[str] | None = None,
        worktree_path: str | None = None,
        budget_limit_usd: float = 0.0,
    ) -> ExecutionInstruction:
        """Create an ExecutionInstruction from a SubTask.

        Packages the subtask with debate context, file hints, success criteria,
        and constraints into a structured instruction that an execution agent
        can follow.

        Args:
            subtask: The SubTask to translate into an instruction.
            goal: Optional PrioritizedGoal providing higher-level context.
            debate_context: Free-form text with debate rationale.
            extra_constraints: Additional constraints beyond the base set.
            worktree_path: Path to an isolated worktree for execution.
            budget_limit_usd: Maximum cost budget for this execution.

        Returns:
            ExecutionInstruction ready for dispatch to an agent.
        """
        context_parts: list[str] = []
        if debate_context:
            context_parts.append(debate_context)
        if goal:
            context_parts.append(f"This subtask is part of: {goal.description}")
            context_parts.append(f"Rationale: {goal.rationale}")
            context_parts.append(f"Estimated impact: {goal.estimated_impact}")

        constraints = list(self.base_constraints)
        if extra_constraints:
            constraints.extend(extra_constraints)

        # SubTask uses file_scope, not file_hints
        file_hints = getattr(subtask, "file_scope", []) or []

        # SubTask.success_criteria is a dict, extract values as list
        raw_criteria = getattr(subtask, "success_criteria", {}) or {}
        if isinstance(raw_criteria, dict):
            success_criteria = [
                f"{key}: {value}" for key, value in raw_criteria.items()
            ]
        elif isinstance(raw_criteria, list):
            success_criteria = list(raw_criteria)
        else:
            success_criteria = []

        return ExecutionInstruction(
            subtask_id=getattr(subtask, "id", "") or f"subtask_{time.time_ns()}",
            track=getattr(subtask, "track", "core") or "core",
            objective=subtask.description,
            context="\n".join(context_parts),
            file_hints=file_hints,
            success_criteria=success_criteria,
            constraints=constraints,
            worktree_path=worktree_path,
            budget_limit_usd=budget_limit_usd,
        )

    def ingest_result(self, result: ExecutionResult) -> None:
        """Ingest an execution result back into the Knowledge Mound.

        Records which files were changed, what succeeded/failed, and any
        observations the agent made during execution. Uses the NomicCycleAdapter
        to persist a GoalOutcome representing this execution.

        Args:
            result: The ExecutionResult to record.
        """
        self._results.append(result)

        if not self.enable_km_ingestion:
            return

        try:
            from aragora.knowledge.mound.adapters.nomic_cycle_adapter import (
                GoalOutcome,
                CycleStatus,
                NomicCycleOutcome,
                get_nomic_cycle_adapter,
            )
            from datetime import datetime, timezone

            adapter = get_nomic_cycle_adapter()

            status = CycleStatus.SUCCESS if result.success else CycleStatus.FAILED
            goal_outcome = GoalOutcome(
                goal_id=result.subtask_id,
                description=f"Execution of subtask {result.subtask_id}",
                track="core",
                status=status,
                error=result.error[:500] if result.error else None,
                files_changed=result.files_changed,
                tests_passed=result.tests_passed,
                tests_failed=result.tests_failed,
                learnings=result.agent_observations[:10],
            )

            # Build a minimal cycle outcome wrapping this single goal
            now = datetime.now(timezone.utc)
            cycle_outcome = NomicCycleOutcome(
                cycle_id=f"exec_{result.subtask_id}",
                objective=f"Execute subtask {result.subtask_id}",
                status=status,
                started_at=now,
                completed_at=now,
                goal_outcomes=[goal_outcome],
                goals_attempted=1,
                goals_succeeded=1 if result.success else 0,
                goals_failed=0 if result.success else 1,
                total_files_changed=len(result.files_changed),
                total_tests_passed=result.tests_passed,
                total_tests_failed=result.tests_failed,
                metadata={"source": "execution_bridge"},
            )

            # ingest_cycle_outcome is async; schedule if possible
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(adapter.ingest_cycle_outcome(cycle_outcome))
            except RuntimeError:
                # No running event loop, skip async ingestion
                logger.debug(
                    "No event loop available for KM ingestion, result stored locally"
                )

            logger.info(
                "execution_result_ingested subtask=%s success=%s files=%d",
                result.subtask_id[:20],
                result.success,
                len(result.files_changed),
            )
        except ImportError:
            logger.debug("KM adapter not available, skipping result ingestion")
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.debug("KM ingestion failed (non-critical): %s", e)

    async def verify_changes(self, result: ExecutionResult) -> dict[str, Any]:
        """Verify execution changes using PRReviewRunner.

        Runs the changed files through Aragora's review pipeline to check
        for security issues, code quality, and correctness.

        Args:
            result: The ExecutionResult whose diff_summary will be reviewed.

        Returns:
            Dictionary with verification outcome including 'verified' bool,
            findings count, and agreement score.
        """
        if not self.enable_verification or not result.diff_summary:
            return {"verified": True, "skipped": True, "reason": "no diff to verify"}

        try:
            from aragora.compat.openclaw.pr_review_runner import PRReviewRunner

            runner = PRReviewRunner()
            review = await runner.review_diff(
                diff=result.diff_summary,
                label=f"subtask-{result.subtask_id[:12]}",
            )

            return {
                "verified": not review.has_critical,
                "findings_count": len(review.findings),
                "critical_count": review.critical_count,
                "agreement_score": review.agreement_score,
            }
        except ImportError:
            logger.debug("PRReviewRunner not available, skipping verification")
            return {"verified": True, "skipped": True, "reason": "reviewer unavailable"}
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Verification failed: %s", e)
            return {"verified": False, "error": str(e)}

    def get_session_summary(self) -> dict[str, Any]:
        """Get a summary of all execution results in this session.

        Returns:
            Dictionary with counts of total subtasks, successes, failures,
            total files changed, total tests run, and per-result details.
        """
        total = len(self._results)
        successes = sum(1 for r in self._results if r.success)
        total_files = sum(len(r.files_changed) for r in self._results)
        total_tests = sum(r.tests_passed + r.tests_failed for r in self._results)

        return {
            "total_subtasks": total,
            "successful": successes,
            "failed": total - successes,
            "total_files_changed": total_files,
            "total_tests_run": total_tests,
            "results": [r.to_dict() for r in self._results],
        }

    def create_batch_instructions(
        self,
        subtasks: list[SubTask],
        goal: PrioritizedGoal | None = None,
        debate_context: str = "",
        extra_constraints: list[str] | None = None,
    ) -> list[ExecutionInstruction]:
        """Create instructions for a batch of subtasks.

        Convenience method that calls create_instruction for each subtask
        in the list.

        Args:
            subtasks: List of SubTasks to translate.
            goal: Optional PrioritizedGoal for all subtasks.
            debate_context: Shared debate rationale.
            extra_constraints: Additional constraints for all subtasks.

        Returns:
            List of ExecutionInstructions, one per subtask.
        """
        return [
            self.create_instruction(
                subtask=st,
                goal=goal,
                debate_context=debate_context,
                extra_constraints=extra_constraints,
            )
            for st in subtasks
        ]
