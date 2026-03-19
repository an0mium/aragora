"""Pause-refresh controller for long unattended self-improvement shifts.

Wraps canonical assessment, bounded self-improvement execution, durable
checkpoint receipts, and a mandatory refresh gate before resuming.

Two controller flavours:

1. ``PauseRefreshShiftController`` — production-grade, state-machine backed
   by ``SelfImproveRunStore``, with pause/resume and durable receipts.

2. ``ShiftController`` — lightweight bounded shift runner for simpler
   use-cases.  Uses ``CanonicalAssessmentCompiler`` for refresh and
   ``HardenedOrchestrator`` for execution.  No external store dependency.
"""

from __future__ import annotations

import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from aragora.nomic.canonical_assessment import (
    CanonicalAssessmentCompiler,
    CanonicalRepoAssessment,
    compute_delta,
    load_assessment,
    save_assessment,
)
from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline
from aragora.nomic.stores.run_store import RunStatus, SelfImproveRun, SelfImproveRunStore
from aragora.swarm.boss_loop import check_runner_freshness

logger = logging.getLogger(__name__)
UTC = timezone.utc


# ---------------------------------------------------------------------------
# Lightweight shift types (used by ShiftController)
# ---------------------------------------------------------------------------


@dataclass
class ShiftConfig:
    """Configuration for a lightweight autonomous improvement shift."""

    max_duration_seconds: float = 28800.0  # 8 hours
    max_items_per_shift: int = 10
    checkpoint_interval_seconds: float = 900.0  # 15 minutes
    budget_limit_usd: float = 50.0
    require_assessment_before_start: bool = True
    require_assessment_after_shift: bool = True
    stop_on_regression: bool = True
    stop_on_dirty_worktree: bool = True
    stop_on_budget_exceeded: bool = True


@dataclass
class ShiftCheckpoint:
    """Snapshot of shift state at a checkpoint."""

    checkpoint_id: str = ""
    shift_id: str = ""
    timestamp: float = 0.0
    phase: str = ""  # "assessment", "planning", "executing", "paused", "completed", "stopped"
    items_attempted: int = 0
    items_landed: int = 0
    items_failed: int = 0
    items_reverted: int = 0
    wall_clock_seconds: float = 0.0
    budget_spent_usd: float = 0.0
    health_score: float = 0.0
    stop_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            self.checkpoint_id = f"chk-{uuid.uuid4().hex[:8]}"
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class ShiftResult:
    """Result of a completed shift."""

    shift_id: str
    started_at: float
    completed_at: float
    duration_seconds: float
    items_attempted: int
    items_landed: int
    items_failed: int
    items_reverted: int
    budget_spent_usd: float
    health_score_before: float
    health_score_after: float
    checkpoints: list[ShiftCheckpoint]
    stop_reason: str  # "completed", "budget_exceeded", "time_exceeded", "regression", "dirty_worktree", "no_work", "error"
    goals_executed: list[str]
    error: str | None = None


class ShiftController:
    """Bounded autonomous improvement shift controller.

    Lightweight alternative to ``PauseRefreshShiftController`` for simple
    use-cases.  Runs the cycle:

        assess -> plan -> execute N items -> checkpoint -> repeat

    until shift budget/time/item limits are reached, then stops and emits
    a final assessment for the next shift.
    """

    def __init__(
        self,
        repo_path: Path | None = None,
        config: ShiftConfig | None = None,
    ) -> None:
        self._repo_path = repo_path or Path(".")
        self._config = config or ShiftConfig()
        self._shift_id = ""
        self._checkpoints: list[ShiftCheckpoint] = []
        self._items_attempted = 0
        self._items_landed = 0
        self._items_failed = 0
        self._items_reverted = 0
        self._budget_spent = 0.0
        self._goals_executed: list[str] = []
        self._start_time = 0.0
        self._running = False

    # -- Public API ----------------------------------------------------------

    async def run_shift(self) -> ShiftResult:
        """Execute one bounded autonomous improvement shift."""
        self._shift_id = f"shift-{uuid.uuid4().hex[:8]}"
        self._start_time = time.time()
        self._running = True
        self._checkpoints = []
        self._items_attempted = 0
        self._items_landed = 0
        self._items_failed = 0
        self._items_reverted = 0
        self._budget_spent = 0.0
        self._goals_executed = []
        health_before = 0.0
        health_after = 0.0
        stop_reason = "completed"
        error = None
        assessment: dict[str, Any] | None = None

        try:
            # Phase 1: Pre-shift assessment
            if self._config.require_assessment_before_start:
                assessment = await self._run_assessment()
                health_before = assessment.get("health_score", 0.0)
                self._emit_checkpoint("assessment", health_score=health_before)

            # Phase 2: Plan from assessment
            goals = await self._plan_from_assessment(assessment)
            if not goals:
                stop_reason = "no_work"
                self._emit_checkpoint("stopped", stop_reason="no_work")
                return self._build_result(health_before, health_before, stop_reason)

            self._emit_checkpoint("planning", details={"goals_count": len(goals)})

            # Phase 3: Execute goals with checkpoints
            for i, goal in enumerate(goals[: self._config.max_items_per_shift]):
                # Check stop conditions before each goal
                stop = self._check_stop_conditions()
                if stop:
                    stop_reason = stop
                    self._emit_checkpoint("stopped", stop_reason=stop)
                    break

                # Execute one goal
                self._emit_checkpoint("executing", details={"goal": goal, "index": i})
                result = await self._execute_goal(goal)
                self._items_attempted += 1

                if result.get("success"):
                    self._items_landed += 1
                    self._goals_executed.append(goal)
                elif result.get("reverted"):
                    self._items_reverted += 1
                else:
                    self._items_failed += 1
                    if self._config.stop_on_regression and result.get("regression"):
                        stop_reason = "regression"
                        self._emit_checkpoint("stopped", stop_reason="regression")
                        break

                self._budget_spent += result.get("cost_usd", 0.0)

            # Phase 4: Post-shift assessment
            if self._config.require_assessment_after_shift:
                post_assessment = await self._run_assessment()
                health_after = post_assessment.get("health_score", 0.0)
                self._emit_checkpoint("completed", health_score=health_after)
            else:
                health_after = health_before

        except Exception as exc:
            stop_reason = "error"
            error = str(exc)
            logger.error("Shift %s failed: %s", self._shift_id, exc)
            self._emit_checkpoint("stopped", stop_reason="error", details={"error": str(exc)})

        self._running = False
        return self._build_result(health_before, health_after, stop_reason, error)

    def stop(self) -> None:
        """Request graceful stop of the current shift."""
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def checkpoints(self) -> list[ShiftCheckpoint]:
        return list(self._checkpoints)

    # -- Internal helpers ----------------------------------------------------

    async def _run_assessment(self) -> dict[str, Any]:
        """Run canonical assessment compiler."""
        try:
            compiler = CanonicalAssessmentCompiler(repo_path=self._repo_path)
            compiled = await compiler.compile()
            return {
                "health_score": (
                    compiled.health_report.get("health_score", 0.0)
                    if isinstance(compiled.health_report, dict)
                    else 0.0
                ),
                "improvement_candidates": compiled.improvement_candidates,
                "assessment_id": compiled.assessment_id,
            }
        except Exception as exc:
            logger.warning("Assessment failed, proceeding with empty: %s", exc)
            return {
                "health_score": 0.0,
                "improvement_candidates": [],
                "assessment_id": "",
            }

    async def _plan_from_assessment(self, assessment: dict[str, Any] | None) -> list[str]:
        """Convert assessment into executable goal strings."""
        if not assessment or not assessment.get("improvement_candidates"):
            return []
        candidates = assessment["improvement_candidates"]
        goals: list[str] = []
        for c in candidates[: self._config.max_items_per_shift * 2]:
            if isinstance(c, dict):
                desc = c.get("description") or c.get("title") or c.get("goal") or str(c)
            else:
                desc = str(c)
            if desc:
                goals.append(desc)
        return goals

    async def _execute_goal(self, goal: str) -> dict[str, Any]:
        """Execute a single goal using HardenedOrchestrator."""
        try:
            from aragora.nomic.hardened_orchestrator import HardenedOrchestrator

            remaining_budget = max(0, self._config.budget_limit_usd - self._budget_spent)
            orch = HardenedOrchestrator(
                aragora_path=self._repo_path,
                budget_limit_usd=remaining_budget,
            )
            result = await orch.execute_goal_coordinated(goal, max_cycles=1)
            return {
                "success": result.success,
                "reverted": getattr(result, "reverted", False),
                "regression": result.failed_subtasks > 0,
                "cost_usd": getattr(result, "total_cost_usd", 0.0),
                "summary": result.summary,
            }
        except Exception as exc:
            logger.warning("Goal execution failed: %s", exc)
            return {
                "success": False,
                "reverted": False,
                "regression": False,
                "cost_usd": 0.0,
                "error": str(exc),
            }

    def _check_stop_conditions(self) -> str | None:
        """Check whether any stop condition is met."""
        if not self._running:
            return "manual_stop"

        elapsed = time.time() - self._start_time
        if elapsed >= self._config.max_duration_seconds:
            return "time_exceeded"

        if (
            self._config.stop_on_budget_exceeded
            and self._budget_spent >= self._config.budget_limit_usd
        ):
            return "budget_exceeded"

        if self._items_attempted >= self._config.max_items_per_shift:
            return "completed"

        if self._config.stop_on_dirty_worktree:
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(self._repo_path),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.stdout.strip():
                    return "dirty_worktree"
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass  # Can't check — don't block the shift

        return None

    def _emit_checkpoint(
        self,
        phase: str,
        *,
        health_score: float = 0.0,
        stop_reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ShiftCheckpoint:
        """Create and record a shift checkpoint."""
        cp = ShiftCheckpoint(
            shift_id=self._shift_id,
            phase=phase,
            items_attempted=self._items_attempted,
            items_landed=self._items_landed,
            items_failed=self._items_failed,
            items_reverted=self._items_reverted,
            wall_clock_seconds=time.time() - self._start_time,
            budget_spent_usd=self._budget_spent,
            health_score=health_score,
            stop_reason=stop_reason,
            details=details or {},
        )
        self._checkpoints.append(cp)
        logger.info(
            "Shift %s checkpoint [%s]: %d attempted, %d landed, %d failed, %.2f USD",
            self._shift_id,
            phase,
            cp.items_attempted,
            cp.items_landed,
            cp.items_failed,
            cp.budget_spent_usd,
        )
        return cp

    def _build_result(
        self,
        health_before: float,
        health_after: float,
        stop_reason: str,
        error: str | None = None,
    ) -> ShiftResult:
        """Build the final shift result."""
        now = time.time()
        return ShiftResult(
            shift_id=self._shift_id,
            started_at=self._start_time,
            completed_at=now,
            duration_seconds=now - self._start_time,
            items_attempted=self._items_attempted,
            items_landed=self._items_landed,
            items_failed=self._items_failed,
            items_reverted=self._items_reverted,
            budget_spent_usd=self._budget_spent,
            health_score_before=health_before,
            health_score_after=health_after,
            checkpoints=list(self._checkpoints),
            stop_reason=stop_reason,
            goals_executed=list(self._goals_executed),
            error=error,
        )


# ---------------------------------------------------------------------------
# Production-grade shift types (used by PauseRefreshShiftController)
# ---------------------------------------------------------------------------


class ShiftStopReason(str, Enum):
    """Truthful stop reasons for unattended self-improvement shifts."""

    ASSESSMENT_REFRESH_REQUIRED = "assessment_refresh_required"
    STALE_RUNNERS = "stale_runners"
    DIRTY_WORKTREE = "dirty_worktree"
    COLLIDING_WORKTREES = "colliding_worktrees"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MISSING_RECEIPTS = "missing_receipts"
    QUALITY_GATES_FAILED = "quality_gates_failed"
    OWNERSHIP_AMBIGUOUS = "ownership_ambiguous"
    NO_BACKLOG = "no_backlog"
    TIME_LIMIT_REACHED = "time_limit_reached"


class ShiftStateError(RuntimeError):
    """Raised when a shift run cannot be started or resumed safely."""


class ShiftRefreshRequiredError(ShiftStateError):
    """Raised when a paused shift is resumed without a fresh assessment."""


@dataclass
class ShiftControllerConfig:
    """Configuration for pause-refresh unattended shifts."""

    max_shift_hours: float = 8.0
    max_cycles_per_shift: int = 3
    backlog_slice_size: int = 3
    budget_limit_usd: float = 20.0
    require_approval: bool = False
    run_tests: bool = True
    use_worktrees: bool = True
    enforce_runner_freshness: bool = True
    require_runner_owner_context: bool = False
    enforce_clean_repo: bool = True
    enforce_worktree_collision_check: bool = True
    require_file_evidence: bool = True
    require_fresh_assessment_on_resume: bool = True


@dataclass
class ShiftCheckpointReceipt:
    """Durable summary of a shift checkpoint or pause decision."""

    checkpoint_id: str
    run_id: str
    timestamp: str
    assessment_id: str | None
    objective: str
    accepted_backlog_slice: list[dict[str, Any]] = field(default_factory=list)
    work_completed: list[dict[str, Any]] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    next_refresh_actions: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    status: str = "paused"

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "assessment_id": self.assessment_id,
            "objective": self.objective,
            "accepted_backlog_slice": list(self.accepted_backlog_slice),
            "work_completed": list(self.work_completed),
            "blocked_reasons": list(self.blocked_reasons),
            "next_refresh_actions": list(self.next_refresh_actions),
            "stop_reason": self.stop_reason,
            "status": self.status,
        }


class PauseRefreshShiftController:
    """Run bounded unattended self-improvement shifts with refresh checkpoints."""

    def __init__(
        self,
        *,
        config: ShiftControllerConfig | None = None,
        repo_path: Path | None = None,
        run_store: SelfImproveRunStore | None = None,
    ) -> None:
        self.config = config or ShiftControllerConfig()
        self.repo_path = Path(repo_path or Path.cwd()).resolve()
        self.run_store = run_store or SelfImproveRunStore()

    async def run_shift(
        self,
        *,
        goal: str | None = None,
        tracks: list[str] | None = None,
        run_id: str | None = None,
        assessment_id: str | None = None,
        dry_run: bool = False,
    ) -> SelfImproveRun:
        """Start or resume a bounded unattended shift."""
        if run_id:
            run, assessment = await self._resume_shift_state(
                run_id=run_id,
                assessment_id=assessment_id,
                tracks=tracks,
            )
        else:
            run, assessment = await self._create_shift_state(
                goal=goal,
                tracks=tracks,
                assessment_id=assessment_id,
                dry_run=dry_run,
            )

        return await self._execute_shift(run, assessment, dry_run=dry_run)

    def get_shift(self, run_id: str) -> SelfImproveRun | None:
        """Return a persisted shift run by id."""
        run = self.run_store.get_run(run_id)
        if run is None:
            return None
        if not self._is_shift_run(run):
            return None
        return run

    def list_shifts(self, limit: int = 20) -> list[SelfImproveRun]:
        """List persisted pause-refresh shifts newest first."""
        runs = self.run_store.list_runs(limit=limit)
        return [run for run in runs if self._is_shift_run(run)]

    async def _create_shift_state(
        self,
        *,
        goal: str | None,
        tracks: list[str] | None,
        assessment_id: str | None,
        dry_run: bool,
    ) -> tuple[SelfImproveRun, CanonicalRepoAssessment]:
        assessment = await self._resolve_assessment(assessment_id)
        accepted_backlog_slice = self._build_backlog_slice(assessment, completed_objectives=set())
        objective = goal or self._default_goal(accepted_backlog_slice)
        shift_plan = {
            "mode": "pause_refresh_shift",
            "repo_path": str(self.repo_path),
            "assessment_id": assessment.assessment_id,
            "assessment_history": [self._assessment_ref(assessment)],
            "assessment_deltas": [],
            "accepted_backlog_slice": accepted_backlog_slice,
            "completed_work": [],
            "checkpoint_receipts": [],
            "requires_refresh": False,
            "stop_reason": None,
            "next_refresh_actions": [],
            "current_objective": "",
            "budget_used_usd": 0.0,
        }
        run = self.run_store.create_run(
            goal=objective,
            tracks=tracks or [],
            mode="shift",
            budget_limit_usd=self.config.budget_limit_usd,
            max_cycles=self.config.max_cycles_per_shift,
            dry_run=dry_run,
            plan=shift_plan,
        )
        updated = self.run_store.update_run(
            run.run_id,
            status=RunStatus.RUNNING,
            started_at=self._now_iso(),
            summary=f"Starting pause-refresh shift from assessment {assessment.assessment_id}",
        )
        if updated is None:
            raise ShiftStateError("Failed to initialize shift run state")
        return updated, assessment

    async def _resume_shift_state(
        self,
        *,
        run_id: str,
        assessment_id: str | None,
        tracks: list[str] | None,
    ) -> tuple[SelfImproveRun, CanonicalRepoAssessment]:
        run = self.run_store.get_run(run_id)
        if run is None:
            raise ShiftStateError(f"Unknown shift run: {run_id}")
        if not self._is_shift_run(run):
            raise ShiftStateError(f"Run {run_id} is not a pause-refresh shift")
        if run.status not in (RunStatus.PAUSED, RunStatus.PENDING, RunStatus.RUNNING):
            raise ShiftStateError(f"Run {run_id} is in terminal status {run.status.value}")

        previous_plan = self._shift_plan(run)
        previous_assessment_id = previous_plan.get("assessment_id")
        assessment = await self._resolve_assessment(assessment_id)
        if (
            self.config.require_fresh_assessment_on_resume
            and previous_assessment_id
            and assessment.assessment_id == previous_assessment_id
        ):
            raise ShiftRefreshRequiredError(
                f"Run {run_id} requires a fresh assessment before resuming"
            )

        previous_assessment = (
            load_assessment(previous_assessment_id) if previous_assessment_id else None
        )
        if previous_assessment is not None:
            previous_plan.setdefault("assessment_deltas", []).append(
                compute_delta(assessment, previous_assessment).to_dict()
            )

        completed_objectives = {
            str(item.get("objective", "")).strip()
            for item in previous_plan.get("completed_work", [])
            if str(item.get("objective", "")).strip()
        }
        previous_plan["assessment_id"] = assessment.assessment_id
        previous_plan.setdefault("assessment_history", []).append(self._assessment_ref(assessment))
        previous_plan["accepted_backlog_slice"] = self._build_backlog_slice(
            assessment,
            completed_objectives=completed_objectives,
        )
        previous_plan["requires_refresh"] = False
        previous_plan["stop_reason"] = None
        previous_plan["next_refresh_actions"] = []
        if tracks:
            run.tracks = list(tracks)

        updated = self.run_store.update_run(
            run_id,
            status=RunStatus.RUNNING,
            plan=run.plan,
            tracks=run.tracks,
            summary=f"Resumed shift with refreshed assessment {assessment.assessment_id}",
        )
        if updated is None:
            raise ShiftStateError(f"Failed to resume shift {run_id}")
        return updated, assessment

    async def _execute_shift(
        self,
        run: SelfImproveRun,
        assessment: CanonicalRepoAssessment,
        *,
        dry_run: bool,
    ) -> SelfImproveRun:
        shift_plan = self._shift_plan(run)
        accepted_backlog_slice = list(shift_plan.get("accepted_backlog_slice", []))
        elapsed_start = time.time()
        cycles_executed = 0

        if not accepted_backlog_slice:
            return self._pause_run(
                run,
                stop_reason=ShiftStopReason.NO_BACKLOG,
                blocked_reasons=["No assessment-backed backlog slice was available to execute."],
                next_refresh_actions=self._refresh_actions(ShiftStopReason.NO_BACKLOG),
            )

        while cycles_executed < self.config.max_cycles_per_shift:
            current_item = self._next_backlog_item(run)
            if current_item is None:
                break

            stop_reason, blocked_reasons = self._preflight_stop_reason(
                run=run,
                backlog_item=current_item,
                started_at=elapsed_start,
            )
            if stop_reason is not None:
                return self._pause_run(
                    run,
                    stop_reason=stop_reason,
                    blocked_reasons=blocked_reasons,
                    next_refresh_actions=self._refresh_actions(stop_reason),
                )

            shift_plan["current_objective"] = current_item["objective"]
            updated = self.run_store.update_run(
                run.run_id,
                status=RunStatus.RUNNING,
                plan=run.plan,
                summary=f"Executing shift objective: {current_item['objective']}",
            )
            if updated is not None:
                run = updated

            result = await self._execute_objective(
                objective=current_item["objective"],
                remaining_budget=max(
                    0.0,
                    float(self.config.budget_limit_usd) - self._budget_used(run),
                ),
                dry_run=dry_run,
            )
            completed_summary = self._summarize_result(current_item, result)
            shift_plan.setdefault("completed_work", []).append(completed_summary)
            shift_plan["budget_used_usd"] = self._budget_used(run) + float(
                getattr(result, "total_cost_usd", 0.0) or 0.0
            )
            cycles_executed += 1

            post_stop_reason, blocked_reasons = self._post_execution_stop_reason(
                run=run,
                result=result,
                started_at=elapsed_start,
            )
            if post_stop_reason is not None:
                return self._pause_run(
                    run,
                    stop_reason=post_stop_reason,
                    blocked_reasons=blocked_reasons,
                    next_refresh_actions=self._refresh_actions(post_stop_reason),
                )

        return self._pause_run(
            run,
            stop_reason=ShiftStopReason.ASSESSMENT_REFRESH_REQUIRED,
            blocked_reasons=[
                "Shift reached a bounded pause point and now requires a fresh assessment."
            ],
            next_refresh_actions=self._refresh_actions(ShiftStopReason.ASSESSMENT_REFRESH_REQUIRED),
        )

    async def _resolve_assessment(
        self,
        assessment_id: str | None,
    ) -> CanonicalRepoAssessment:
        if assessment_id:
            assessment = load_assessment(assessment_id)
            if assessment is None:
                raise ShiftStateError(f"Unknown canonical assessment: {assessment_id}")
            return assessment

        compiler = CanonicalAssessmentCompiler(repo_path=self.repo_path)
        assessment = await compiler.compile()
        save_assessment(assessment)
        return assessment

    async def _execute_objective(
        self,
        *,
        objective: str,
        remaining_budget: float,
        dry_run: bool,
    ) -> Any:
        config = SelfImproveConfig(
            use_worktrees=self.config.use_worktrees,
            budget_limit_usd=remaining_budget or self.config.budget_limit_usd,
            require_approval=self.config.require_approval,
            autonomous=not self.config.require_approval,
            run_tests=self.config.run_tests,
        )
        pipeline = SelfImprovePipeline(config)
        if dry_run:
            plan = await pipeline.dry_run(objective)
            return type(
                "DryRunShiftResult",
                (),
                {
                    "cycle_id": f"dryrun-{uuid.uuid4().hex[:8]}",
                    "objective": objective,
                    "subtasks_total": len(plan.get("subtasks", [])),
                    "subtasks_completed": 0,
                    "subtasks_failed": 0,
                    "tests_passed": 0,
                    "tests_failed": 0,
                    "regressions_detected": False,
                    "files_changed": [],
                    "total_cost_usd": 0.0,
                },
            )()
        return await pipeline.run(objective)

    def _preflight_stop_reason(
        self,
        *,
        run: SelfImproveRun,
        backlog_item: dict[str, Any],
        started_at: float,
    ) -> tuple[ShiftStopReason | None, list[str]]:
        blocked_reasons: list[str] = []

        if self.config.enforce_clean_repo and self._is_repo_dirty():
            blocked_reasons.append("Repository has uncommitted changes.")
            return ShiftStopReason.DIRTY_WORKTREE, blocked_reasons

        if self.config.enforce_worktree_collision_check and self._has_colliding_worktrees():
            blocked_reasons.append("Git worktree registry reports duplicate paths or branches.")
            return ShiftStopReason.COLLIDING_WORKTREES, blocked_reasons

        if self.config.enforce_runner_freshness:
            freshness = check_runner_freshness()
            if not freshness.fresh:
                if freshness.blocked_reason == "missing_owner_context":
                    if self.config.require_runner_owner_context:
                        blocked_reasons.append("Runner freshness could not be proven.")
                        return ShiftStopReason.STALE_RUNNERS, blocked_reasons
                else:
                    blocked_reasons.append(
                        f"Runner freshness failed: {freshness.blocked_reason or 'unknown'}."
                    )
                    return ShiftStopReason.STALE_RUNNERS, blocked_reasons

        if self.config.require_file_evidence and not self._has_file_evidence(backlog_item):
            blocked_reasons.append(
                "Selected backlog item lacks file-level evidence, so ownership is ambiguous."
            )
            return ShiftStopReason.OWNERSHIP_AMBIGUOUS, blocked_reasons

        if self._budget_used(run) >= self.config.budget_limit_usd:
            blocked_reasons.append("Shift budget has already been exhausted.")
            return ShiftStopReason.BUDGET_EXHAUSTED, blocked_reasons

        if self._elapsed_hours(started_at) >= self.config.max_shift_hours:
            blocked_reasons.append("Shift exceeded its bounded duration.")
            return ShiftStopReason.TIME_LIMIT_REACHED, blocked_reasons

        return None, blocked_reasons

    def _post_execution_stop_reason(
        self,
        *,
        run: SelfImproveRun,
        result: Any,
        started_at: float,
    ) -> tuple[ShiftStopReason | None, list[str]]:
        blocked_reasons: list[str] = []

        if not getattr(result, "cycle_id", None):
            blocked_reasons.append(
                "Execution completed without a durable cycle/checkpoint receipt id."
            )
            return ShiftStopReason.MISSING_RECEIPTS, blocked_reasons

        if (
            getattr(result, "regressions_detected", False)
            or getattr(result, "subtasks_failed", 0) > 0
            or getattr(result, "tests_failed", 0) > 0
        ):
            blocked_reasons.append(
                "Execution reported regressions, failed subtasks, or failing tests."
            )
            return ShiftStopReason.QUALITY_GATES_FAILED, blocked_reasons

        if self._budget_used(run) >= self.config.budget_limit_usd:
            blocked_reasons.append("Shift budget has been exhausted.")
            return ShiftStopReason.BUDGET_EXHAUSTED, blocked_reasons

        if self._elapsed_hours(started_at) >= self.config.max_shift_hours:
            blocked_reasons.append("Shift exceeded its bounded duration.")
            return ShiftStopReason.TIME_LIMIT_REACHED, blocked_reasons

        return None, blocked_reasons

    def _pause_run(
        self,
        run: SelfImproveRun,
        *,
        stop_reason: ShiftStopReason,
        blocked_reasons: list[str],
        next_refresh_actions: list[str],
    ) -> SelfImproveRun:
        shift_plan = self._shift_plan(run)
        receipt = ShiftCheckpointReceipt(
            checkpoint_id=f"shift-{uuid.uuid4().hex[:10]}",
            run_id=run.run_id,
            timestamp=self._now_iso(),
            assessment_id=shift_plan.get("assessment_id"),
            objective=str(shift_plan.get("current_objective", "") or ""),
            accepted_backlog_slice=list(shift_plan.get("accepted_backlog_slice", [])),
            work_completed=list(shift_plan.get("completed_work", [])),
            blocked_reasons=list(blocked_reasons),
            next_refresh_actions=list(next_refresh_actions),
            stop_reason=stop_reason.value,
        )
        shift_plan.setdefault("checkpoint_receipts", []).append(receipt.to_dict())
        shift_plan["requires_refresh"] = True
        shift_plan["stop_reason"] = stop_reason.value
        shift_plan["next_refresh_actions"] = list(next_refresh_actions)
        shift_plan["current_objective"] = ""

        updated = self.run_store.update_run(
            run.run_id,
            status=RunStatus.PAUSED,
            plan=run.plan,
            summary=f"Shift paused: {stop_reason.value}",
        )
        if updated is None:
            raise ShiftStateError(f"Failed to persist checkpoint for shift {run.run_id}")
        return updated

    def _build_backlog_slice(
        self,
        assessment: CanonicalRepoAssessment,
        *,
        completed_objectives: set[str],
    ) -> list[dict[str, Any]]:
        accepted: list[dict[str, Any]] = []

        for candidate in getattr(assessment, "improvement_candidates", []):
            item = self._candidate_to_backlog_item(candidate)
            if item is None:
                continue
            if item["objective"] in completed_objectives:
                continue
            accepted.append(item)
            if len(accepted) >= self.config.backlog_slice_size:
                return accepted

        for feature in getattr(assessment, "feature_inventory", []):
            status = getattr(feature, "status", "")
            if status not in {"gap", "scaffolding", "stale"}:
                continue
            objective = f"Harden {getattr(feature, 'name', 'feature')}"
            if objective in completed_objectives:
                continue
            accepted.append(
                {
                    "objective": objective,
                    "priority": getattr(feature, "priority", "P3"),
                    "source": "feature_inventory",
                    "evidence": list(getattr(feature, "evidence", []) or []),
                }
            )
            if len(accepted) >= self.config.backlog_slice_size:
                break

        return accepted

    def _candidate_to_backlog_item(self, candidate: Any) -> dict[str, Any] | None:
        if isinstance(candidate, dict):
            objective = str(candidate.get("description", "")).strip()
            evidence = candidate.get("files") or candidate.get("evidence") or []
            priority = candidate.get("priority", 0.0)
            source = candidate.get("source", "assessment")
        else:
            objective = str(getattr(candidate, "description", "")).strip()
            evidence = (
                getattr(candidate, "files", None) or getattr(candidate, "evidence", None) or []
            )
            priority = getattr(candidate, "priority", 0.0)
            source = getattr(candidate, "source", "assessment")

        if not objective:
            return None
        return {
            "objective": objective,
            "priority": priority,
            "source": source,
            "evidence": list(evidence),
        }

    def _summarize_result(self, backlog_item: dict[str, Any], result: Any) -> dict[str, Any]:
        return {
            "objective": backlog_item["objective"],
            "source": backlog_item.get("source"),
            "cycle_id": getattr(result, "cycle_id", None),
            "subtasks_total": int(getattr(result, "subtasks_total", 0) or 0),
            "subtasks_completed": int(getattr(result, "subtasks_completed", 0) or 0),
            "subtasks_failed": int(getattr(result, "subtasks_failed", 0) or 0),
            "tests_passed": int(getattr(result, "tests_passed", 0) or 0),
            "tests_failed": int(getattr(result, "tests_failed", 0) or 0),
            "regressions_detected": bool(getattr(result, "regressions_detected", False)),
            "files_changed": list(getattr(result, "files_changed", []) or []),
            "total_cost_usd": float(getattr(result, "total_cost_usd", 0.0) or 0.0),
        }

    def _default_goal(self, accepted_backlog_slice: list[dict[str, Any]]) -> str:
        if accepted_backlog_slice:
            return accepted_backlog_slice[0]["objective"]
        return "Autonomous repo improvement shift"

    def _assessment_ref(self, assessment: CanonicalRepoAssessment) -> dict[str, Any]:
        return {
            "assessment_id": assessment.assessment_id,
            "timestamp": assessment.timestamp,
            "health_score": assessment.health_report.get("health_score"),
        }

    def _refresh_actions(self, stop_reason: ShiftStopReason) -> list[str]:
        actions = [
            "Run `aragora assess --save --diff` to capture a fresh ground-up assessment.",
            "Review the canonical backlog slice before resuming the shift.",
        ]
        if stop_reason == ShiftStopReason.DIRTY_WORKTREE:
            actions.insert(0, "Reconcile or clean the dirty worktree before resuming.")
        elif stop_reason == ShiftStopReason.COLLIDING_WORKTREES:
            actions.insert(0, "Reconcile duplicate/colliding worktrees before resuming.")
        elif stop_reason == ShiftStopReason.STALE_RUNNERS:
            actions.insert(0, "Restore a fresh eligible runner registration before resuming.")
        elif stop_reason == ShiftStopReason.QUALITY_GATES_FAILED:
            actions.insert(0, "Inspect the failing quality gates before accepting more work.")
        elif stop_reason == ShiftStopReason.OWNERSHIP_AMBIGUOUS:
            actions.insert(
                0, "Tighten file ownership or scope evidence for the next backlog slice."
            )
        return actions

    def _shift_plan(self, run: SelfImproveRun) -> dict[str, Any]:
        if run.plan is None:
            run.plan = {}
        return run.plan

    def _next_backlog_item(self, run: SelfImproveRun) -> dict[str, Any] | None:
        shift_plan = self._shift_plan(run)
        completed = {
            str(item.get("objective", "")).strip()
            for item in shift_plan.get("completed_work", [])
            if str(item.get("objective", "")).strip()
        }
        for item in shift_plan.get("accepted_backlog_slice", []):
            objective = str(item.get("objective", "")).strip()
            if objective and objective not in completed:
                return item
        return None

    def _budget_used(self, run: SelfImproveRun) -> float:
        shift_plan = self._shift_plan(run)
        return float(shift_plan.get("budget_used_usd", 0.0) or 0.0)

    def _has_file_evidence(self, backlog_item: dict[str, Any]) -> bool:
        evidence = list(backlog_item.get("evidence", []) or [])
        if not evidence:
            return False
        return any(str(item).strip() for item in evidence)

    def _is_repo_dirty(self) -> bool:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if proc.returncode != 0:
            logger.debug("git status failed while checking shift repo cleanliness")
            return True
        return bool(proc.stdout.strip())

    def _has_colliding_worktrees(self) -> bool:
        proc = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if proc.returncode != 0:
            logger.debug("git worktree list failed while checking collisions")
            return True

        paths: set[str] = set()
        branches: set[str] = set()
        current_path: str | None = None

        for line in proc.stdout.splitlines():
            if line.startswith("worktree "):
                current_path = line.split(" ", 1)[1].strip()
                if current_path in paths:
                    return True
                paths.add(current_path)
            elif line.startswith("branch "):
                branch = line.split(" ", 1)[1].strip()
                if current_path and branch in branches:
                    return True
                if branch:
                    branches.add(branch)
        return False

    def _elapsed_hours(self, started_at: float) -> float:
        return (time.time() - started_at) / 3600.0

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def _is_shift_run(self, run: SelfImproveRun) -> bool:
        return bool((run.plan or {}).get("mode") == "pause_refresh_shift")


__all__ = [
    "PauseRefreshShiftController",
    "ShiftCheckpoint",
    "ShiftCheckpointReceipt",
    "ShiftConfig",
    "ShiftController",
    "ShiftControllerConfig",
    "ShiftRefreshRequiredError",
    "ShiftResult",
    "ShiftStateError",
    "ShiftStopReason",
]
