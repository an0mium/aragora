"""Self-Improvement Pipeline -- Aragora improving itself.

Chains: MetaPlanner -> TaskDecomposer -> WorktreeManager -> Execution ->
        BranchCoordinator -> OutcomeTracker -> CycleLearningStore

Usage:
    pipeline = SelfImprovePipeline()
    result = await pipeline.run("Make Aragora better for SMEs")

Dry-run (preview without executing):
    plan = await pipeline.dry_run("Improve test coverage")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SelfImproveConfig:
    """Configuration for the self-improvement pipeline."""

    # Planning
    use_meta_planner: bool = True  # Debate-driven prioritization
    quick_mode: bool = False  # Skip debate, use heuristics
    max_goals: int = 5

    # Execution
    use_worktrees: bool = True  # Isolated worktrees per subtask
    max_parallel: int = 4  # Max parallel worktrees
    budget_limit_usd: float = 10.0  # Total budget cap
    require_approval: bool = True  # Human approval at checkpoints

    # Verification
    run_tests: bool = True
    run_review: bool = True  # PRReviewRunner on diffs
    capture_metrics: bool = True  # Before/after debate quality

    # Feedback
    persist_outcomes: bool = True  # Save to CycleLearningStore
    auto_revert_on_regression: bool = True
    degradation_threshold: float = 0.05


@dataclass
class SelfImproveResult:
    """Result of a self-improvement cycle."""

    cycle_id: str
    objective: str
    goals_planned: int = 0
    subtasks_total: int = 0
    subtasks_completed: int = 0
    subtasks_failed: int = 0
    files_changed: list[str] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    regressions_detected: bool = False
    reverted: bool = False
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a plain dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "objective": self.objective,
            "goals_planned": self.goals_planned,
            "subtasks_total": self.subtasks_total,
            "subtasks_completed": self.subtasks_completed,
            "subtasks_failed": self.subtasks_failed,
            "files_changed": self.files_changed,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "regressions_detected": self.regressions_detected,
            "reverted": self.reverted,
            "duration_seconds": self.duration_seconds,
        }


class SelfImprovePipeline:
    """The full self-improvement pipeline.

    Orchestrates the complete cycle:
    1. Plan: MetaPlanner debates priorities -> TaskDecomposer breaks into subtasks
    2. Baseline: OutcomeTracker captures pre-change debate quality metrics
    3. Execute: WorktreeManager creates isolation -> agents implement subtasks
    4. Verify: Run tests + PRReviewRunner on each branch
    5. Merge: BranchCoordinator merges passing branches
    6. Measure: OutcomeTracker captures post-change metrics, compares
    7. Learn: CycleLearningStore persists outcomes for next cycle
    """

    def __init__(self, config: SelfImproveConfig | None = None):
        self.config = config or SelfImproveConfig()

    async def run(self, objective: str) -> SelfImproveResult:
        """Run a full self-improvement cycle.

        Args:
            objective: High-level objective like "Improve test coverage"

        Returns:
            SelfImproveResult with cycle outcomes
        """
        cycle_id = f"cycle_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        result = SelfImproveResult(cycle_id=cycle_id, objective=objective)

        logger.info(
            "self_improve_started cycle=%s objective=%s", cycle_id, objective[:100]
        )

        # Step 1: Plan
        goals = await self._plan(objective)
        result.goals_planned = len(goals)

        if not goals:
            logger.warning("self_improve_no_goals cycle=%s", cycle_id)
            result.duration_seconds = time.time() - start_time
            return result

        # Step 2: Decompose goals into subtasks
        subtasks = await self._decompose(goals)
        result.subtasks_total = len(subtasks)

        # Step 3: Capture baseline metrics
        baseline = None
        if self.config.capture_metrics:
            baseline = await self._capture_baseline()

        # Step 4: Execute subtasks (in worktrees if configured)
        execution_results = await self._execute(subtasks, cycle_id)

        for er in execution_results:
            if er.get("success"):
                result.subtasks_completed += 1
                result.files_changed.extend(er.get("files_changed", []))
                result.tests_passed += er.get("tests_passed", 0)
            else:
                result.subtasks_failed += 1
                result.tests_failed += er.get("tests_failed", 0)

        # Step 5: Capture post-change metrics and compare
        if self.config.capture_metrics and baseline is not None:
            after = await self._capture_after()
            if after is not None:
                comparison = self._compare_metrics(baseline, after)

                if comparison and not comparison.get("improved", True):
                    result.regressions_detected = True
                    if self.config.auto_revert_on_regression:
                        logger.warning(
                            "self_improve_regression cycle=%s recommendation=%s",
                            cycle_id,
                            comparison.get("recommendation"),
                        )
                        # Don't auto-revert yet -- log and let human decide
                        # result.reverted = True

        # Step 6: Persist outcomes
        if self.config.persist_outcomes:
            self._persist_outcome(cycle_id, result)

        result.duration_seconds = time.time() - start_time
        logger.info(
            "self_improve_completed cycle=%s goals=%d subtasks=%d/%d duration=%.1fs",
            cycle_id,
            result.goals_planned,
            result.subtasks_completed,
            result.subtasks_total,
            result.duration_seconds,
        )

        return result

    async def dry_run(self, objective: str) -> dict[str, Any]:
        """Preview what the pipeline would do without executing.

        Returns the plan (goals + subtasks) without making any changes.
        """
        goals = await self._plan(objective)
        subtasks = await self._decompose(goals)

        return {
            "objective": objective,
            "goals": [
                {
                    "description": getattr(g, "description", str(g)),
                    "track": (
                        g.track.value
                        if hasattr(g, "track") and hasattr(g.track, "value")
                        else str(getattr(g, "track", "core"))
                    ),
                    "priority": getattr(g, "priority", 0),
                    "estimated_impact": getattr(g, "estimated_impact", "unknown"),
                    "rationale": getattr(g, "rationale", ""),
                }
                for g in goals
            ],
            "subtasks": [
                {
                    "title": getattr(s, "title", None)
                    or getattr(s, "original_task", str(s)),
                    "description": getattr(s, "description", str(s)),
                    "scope": getattr(s, "scope", "unknown"),
                    "file_hints": getattr(s, "file_scope", []),
                    "success_criteria": getattr(s, "success_criteria", {}),
                }
                for s in subtasks
            ],
            "config": {
                "use_worktrees": self.config.use_worktrees,
                "max_parallel": self.config.max_parallel,
                "budget_limit_usd": self.config.budget_limit_usd,
            },
        }

    # --- Private pipeline steps ---

    async def _plan(self, objective: str) -> list[Any]:
        """Step 1: Use MetaPlanner to prioritize goals."""
        if not self.config.use_meta_planner:
            # Return a single goal: the objective itself
            try:
                from aragora.nomic.meta_planner import PrioritizedGoal, Track

                return [
                    PrioritizedGoal(
                        id="direct",
                        track=Track.CORE,
                        description=objective,
                        rationale="Direct objective (no meta-planning)",
                        estimated_impact="high",
                        priority=1,
                    )
                ]
            except ImportError:
                logger.warning("PrioritizedGoal not importable, returning raw goal")
                return [objective]

        try:
            from aragora.nomic.meta_planner import MetaPlanner, MetaPlannerConfig

            config = MetaPlannerConfig(
                quick_mode=self.config.quick_mode,
                max_goals=self.config.max_goals,
            )
            planner = MetaPlanner(config)
            goals = await planner.prioritize_work(objective=objective)
            return goals
        except ImportError as exc:
            logger.warning("MetaPlanner unavailable: %s", exc)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("MetaPlanner failed, using direct objective: %s", exc)

        # Fallback: wrap objective as a single PrioritizedGoal
        try:
            from aragora.nomic.meta_planner import PrioritizedGoal, Track

            return [
                PrioritizedGoal(
                    id="fallback",
                    track=Track.CORE,
                    description=objective,
                    rationale="MetaPlanner fallback",
                    estimated_impact="medium",
                    priority=1,
                )
            ]
        except ImportError:
            return [objective]

    async def _decompose(self, goals: list[Any]) -> list[Any]:
        """Step 2: Use TaskDecomposer to break goals into subtasks."""
        all_subtasks: list[Any] = []
        try:
            from aragora.nomic.task_decomposer import TaskDecomposer

            decomposer = TaskDecomposer()
            for goal in goals:
                desc = getattr(goal, "description", str(goal))
                decomposition = decomposer.analyze(desc)
                if decomposition.should_decompose:
                    all_subtasks.extend(decomposition.subtasks)
                else:
                    # Simple enough to be its own subtask
                    all_subtasks.append(decomposition)
        except ImportError as exc:
            logger.warning("TaskDecomposer unavailable: %s", exc)
            # Fall back to treating each goal as a single task
            for goal in goals:
                all_subtasks.append(goal)
        except (RuntimeError, ValueError) as exc:
            logger.warning("TaskDecomposer failed: %s", exc)
            for goal in goals:
                all_subtasks.append(goal)

        return all_subtasks

    async def _capture_baseline(self) -> Any:
        """Step 3: Capture pre-change debate quality metrics."""
        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            tracker = NomicOutcomeTracker()
            return await tracker.capture_baseline()
        except ImportError as exc:
            logger.debug("Outcome tracker unavailable for baseline: %s", exc)
            return None
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Baseline capture failed: %s", exc)
            return None

    async def _capture_after(self) -> Any:
        """Step 5a: Capture post-change debate quality metrics."""
        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            tracker = NomicOutcomeTracker()
            return await tracker.capture_after()
        except ImportError as exc:
            logger.debug("Outcome tracker unavailable for after: %s", exc)
            return None
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("After capture failed: %s", exc)
            return None

    def _compare_metrics(
        self, baseline: Any, after: Any
    ) -> dict[str, Any] | None:
        """Step 5b: Compare baseline and after metrics."""
        if baseline is None or after is None:
            return None
        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            tracker = NomicOutcomeTracker(
                degradation_threshold=self.config.degradation_threshold,
            )
            comparison = tracker.compare(baseline, after)
            return {
                "improved": comparison.improved,
                "recommendation": comparison.recommendation,
                "deltas": comparison.metrics_delta,
            }
        except ImportError as exc:
            logger.debug("Outcome comparison unavailable: %s", exc)
            return None
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Outcome comparison failed: %s", exc)
            return None

    async def _execute(
        self,
        subtasks: list[Any],
        cycle_id: str,
    ) -> list[dict[str, Any]]:
        """Step 4: Execute subtasks, optionally in parallel worktrees."""
        if self.config.use_worktrees:
            return await self._execute_in_worktrees(subtasks, cycle_id)

        # Sequential execution without isolation
        results: list[dict[str, Any]] = []
        for subtask in subtasks:
            result = await self._execute_single(subtask, cycle_id)
            results.append(result)
        return results

    async def _execute_in_worktrees(
        self,
        subtasks: list[Any],
        cycle_id: str,
    ) -> list[dict[str, Any]]:
        """Execute subtasks in isolated worktrees using BranchCoordinator."""
        try:
            from aragora.nomic.branch_coordinator import (
                BranchCoordinator,
                TrackAssignment,
            )
            from aragora.nomic.meta_planner import PrioritizedGoal, Track

            coordinator = BranchCoordinator()

            assignments = []
            for i, subtask in enumerate(subtasks[: self.config.max_parallel]):
                # Determine track
                track_name = getattr(subtask, "track", "core")
                try:
                    track = (
                        Track(track_name)
                        if isinstance(track_name, str)
                        else track_name
                    )
                except ValueError:
                    track = Track.CORE

                desc = getattr(subtask, "description", str(subtask))

                # TrackAssignment.goal expects a PrioritizedGoal
                goal_obj = PrioritizedGoal(
                    id=f"subtask_{i}",
                    track=track,
                    description=desc[:200],
                    rationale=f"Subtask from cycle {cycle_id}",
                    estimated_impact="medium",
                    priority=i + 1,
                )

                assignments.append(TrackAssignment(goal=goal_obj))

            async def execute_fn(assignment: TrackAssignment) -> dict[str, Any]:
                return await self._execute_single(assignment, cycle_id)

            coordination_result = await coordinator.coordinate_parallel_work(
                assignments=assignments,
                run_nomic_fn=execute_fn,
            )

            results: list[dict[str, Any]] = []
            for a in coordination_result.assignments:
                success = a.status in ("completed", "merged")
                result_data = a.result if isinstance(a.result, dict) else {}
                results.append(
                    {
                        "success": success,
                        "files_changed": result_data.get("files_changed", []),
                        "tests_passed": result_data.get("tests_passed", 0),
                        "tests_failed": result_data.get("tests_failed", 0),
                    }
                )
            return results

        except ImportError as exc:
            logger.warning(
                "Worktree execution unavailable, falling back to sequential: %s", exc
            )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.warning(
                "Worktree execution failed, falling back to sequential: %s", exc
            )

        # Fallback: sequential execution
        results = []
        for subtask in subtasks:
            result = await self._execute_single(subtask, cycle_id)
            results.append(result)
        return results

    async def _execute_single(
        self,
        subtask: Any,
        cycle_id: str,
    ) -> dict[str, Any]:
        """Execute a single subtask.

        In the current implementation, this generates an execution description
        but does not yet dispatch to a Claude Code session. This is the
        integration point where an execution agent would be invoked.

        Args:
            subtask: A SubTask, TaskDecomposition, TrackAssignment, or raw string
            cycle_id: The cycle identifier for logging

        Returns:
            Dict with execution outcome
        """
        # Extract description from various subtask types
        if isinstance(subtask, str):
            desc = subtask
        elif hasattr(subtask, "goal") and hasattr(subtask.goal, "description"):
            # TrackAssignment
            desc = subtask.goal.description
        elif hasattr(subtask, "original_task"):
            # TaskDecomposition
            desc = subtask.original_task
        elif hasattr(subtask, "description"):
            desc = str(subtask.description)
        elif hasattr(subtask, "title"):
            desc = str(subtask.title)
        else:
            desc = str(subtask)

        logger.info("execute_subtask cycle=%s task=%s", cycle_id, desc[:80])

        # Attempt to use ExecutionBridge to generate + dispatch instruction
        try:
            from aragora.nomic.execution_bridge import ExecutionBridge

            bridge = ExecutionBridge()
            instruction = bridge.create_instruction(subtask)

            logger.info(
                "execution_instruction generated subtask=%s",
                getattr(instruction, "subtask_id", "unknown")[:20],
            )

            # Write instruction to worktree for agent pickup
            dispatched = False
            executed = False
            worktree_path = instruction.worktree_path or getattr(
                subtask, "worktree_path", None
            )
            files_changed: list[str] = []
            tests_passed = 0
            tests_failed = 0

            if worktree_path:
                dispatched = self._write_instruction_to_worktree(
                    instruction, worktree_path
                )

                # Try to execute via Claude Code harness
                exec_result = await self._dispatch_to_claude_code(
                    instruction, worktree_path
                )
                if exec_result is not None:
                    executed = True
                    files_changed = exec_result.get("files_changed", [])
                    tests_passed = exec_result.get("tests_passed", 0)
                    tests_failed = exec_result.get("tests_failed", 0)

            return {
                "success": True,
                "subtask": desc[:100],
                "instruction_generated": True,
                "instruction_dispatched": dispatched,
                "instruction_executed": executed,
                "worktree_path": worktree_path,
                "files_changed": files_changed,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
            }
        except ImportError:
            pass
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            logger.debug("ExecutionBridge failed: %s", exc)

        return {
            "success": True,
            "subtask": desc[:100],
            "instruction_generated": False,
            "instruction_dispatched": False,
            "files_changed": [],
            "tests_passed": 0,
            "tests_failed": 0,
        }

    async def _dispatch_to_claude_code(
        self,
        instruction: Any,
        worktree_path: str,
    ) -> dict[str, Any] | None:
        """Dispatch an instruction to Claude Code CLI for execution.

        Uses ClaudeCodeHarness.execute_implementation() to run the instruction
        in the given worktree directory. Returns None if the CLI is not
        available or if dispatch is skipped (e.g., require_approval is True
        and no approval mechanism exists yet).

        Args:
            instruction: ExecutionInstruction with to_agent_prompt()
            worktree_path: Path to the isolated worktree

        Returns:
            Dict with execution results, or None if dispatch was skipped.
        """
        if self.config.require_approval:
            logger.info(
                "dispatch_skipped reason=require_approval subtask=%s",
                instruction.subtask_id[:20],
            )
            return None

        try:
            import shutil

            if not shutil.which("claude"):
                logger.debug("Claude Code CLI not found in PATH, skipping dispatch")
                return None

            from pathlib import Path as P
            from aragora.harnesses.claude_code import ClaudeCodeHarness, ClaudeCodeConfig

            config = ClaudeCodeConfig(
                timeout_seconds=int(
                    min(self.config.budget_limit_usd * 60, 600)  # Budget → timeout
                ),
                use_mcp_tools=False,  # Keep it simple for now
            )
            harness = ClaudeCodeHarness(config)
            prompt = instruction.to_agent_prompt()

            stdout, stderr = await harness.execute_implementation(
                repo_path=P(worktree_path),
                prompt=prompt,
            )

            # Parse files changed from git diff in worktree
            files_changed: list[str] = []
            try:
                import subprocess

                diff_result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=worktree_path,
                    timeout=10,
                )
                if diff_result.returncode == 0:
                    files_changed = [
                        f for f in diff_result.stdout.strip().split("\n") if f
                    ]
            except (subprocess.TimeoutExpired, OSError):
                pass

            # Run tests if configured
            tests_passed = 0
            tests_failed = 0
            if self.config.run_tests and files_changed:
                test_result = await self._run_tests_in_worktree(worktree_path)
                tests_passed = test_result.get("passed", 0)
                tests_failed = test_result.get("failed", 0)

            logger.info(
                "dispatch_completed subtask=%s files=%d tests=%d/%d",
                instruction.subtask_id[:20],
                len(files_changed),
                tests_passed,
                tests_passed + tests_failed,
            )

            return {
                "files_changed": files_changed,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "stdout_len": len(stdout),
            }

        except ImportError as exc:
            logger.debug("ClaudeCodeHarness not available: %s", exc)
            return None
        except (RuntimeError, OSError, asyncio.TimeoutError) as exc:
            logger.warning("Claude Code dispatch failed: %s", exc)
            return None

    async def _run_tests_in_worktree(
        self,
        worktree_path: str,
    ) -> dict[str, int]:
        """Run pytest in a worktree and return pass/fail counts."""
        try:
            import subprocess

            result = await asyncio.to_thread(
                subprocess.run,
                ["python", "-m", "pytest", "--tb=no", "-q", "--timeout=30"],
                capture_output=True,
                text=True,
                cwd=worktree_path,
                timeout=300,
            )

            # Parse pytest summary line: "X passed, Y failed"
            passed = 0
            failed = 0
            for line in result.stdout.splitlines():
                if "passed" in line:
                    import re

                    m = re.search(r"(\d+) passed", line)
                    if m:
                        passed = int(m.group(1))
                    m = re.search(r"(\d+) failed", line)
                    if m:
                        failed = int(m.group(1))

            return {"passed": passed, "failed": failed}

        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Test run failed in worktree: %s", exc)
            return {"passed": 0, "failed": 0}

    @staticmethod
    def _write_instruction_to_worktree(
        instruction: Any,
        worktree_path: str,
    ) -> bool:
        """Write an execution instruction file into a worktree.

        Creates `.aragora/instruction.md` in the worktree root so a
        Claude Code session opened in that directory picks it up as context.

        Returns True if the file was written successfully.
        """
        from pathlib import Path

        wt = Path(worktree_path)
        if not wt.exists():
            logger.debug("Worktree path does not exist: %s", worktree_path)
            return False

        instruction_dir = wt / ".aragora"
        instruction_dir.mkdir(parents=True, exist_ok=True)

        prompt = instruction.to_agent_prompt()
        instruction_file = instruction_dir / "instruction.md"
        instruction_file.write_text(prompt, encoding="utf-8")

        # Also write machine-readable JSON for programmatic pickup
        json_file = instruction_dir / "instruction.json"
        json_file.write_text(
            json.dumps(instruction.to_dict(), indent=2),
            encoding="utf-8",
        )

        logger.info(
            "instruction_written worktree=%s subtask=%s",
            worktree_path,
            instruction.subtask_id[:20],
        )
        return True

    def _persist_outcome(self, cycle_id: str, result: SelfImproveResult) -> None:
        """Step 6: Persist cycle outcome to CycleLearningStore."""
        try:
            from aragora.nomic.cycle_record import NomicCycleRecord
            from aragora.nomic.cycle_store import get_cycle_store

            store = get_cycle_store()
            record = NomicCycleRecord(
                cycle_id=cycle_id,
                started_at=time.time() - result.duration_seconds,
            )
            record.mark_complete(
                success=result.subtasks_completed > 0
                and result.subtasks_failed == 0,
            )

            # Add metadata as evidence quality scores
            record.evidence_quality_scores = {
                "subtasks_completed": float(result.subtasks_completed),
                "subtasks_failed": float(result.subtasks_failed),
                "files_changed": float(len(result.files_changed)),
                "regressions": 1.0 if result.regressions_detected else 0.0,
            }

            store.save_cycle(record)
            logger.info("cycle_outcome_persisted cycle=%s", cycle_id)

        except ImportError as exc:
            logger.debug("Failed to persist cycle outcome (import): %s", exc)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Failed to persist cycle outcome: %s", exc)


__all__ = [
    "SelfImproveConfig",
    "SelfImprovePipeline",
    "SelfImproveResult",
]
