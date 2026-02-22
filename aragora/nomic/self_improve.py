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
import subprocess
import sys
import time
import types
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised when the self-improvement pipeline exceeds its budget."""


@dataclass
class SelfImproveConfig:
    """Configuration for the self-improvement pipeline."""

    # Planning
    use_meta_planner: bool = True  # Debate-driven prioritization
    quick_mode: bool = False  # Skip debate, use heuristics
    scan_mode: bool = True  # Use codebase signals, no LLM calls
    max_goals: int = 5

    # Execution
    use_worktrees: bool = True  # Isolated worktrees per subtask
    max_parallel: int = 4  # Max parallel worktrees
    budget_limit_usd: float = 10.0  # Total budget cap
    require_approval: bool = True  # Human approval at checkpoints
    autonomous: bool = False  # Skip approval gates for fully autonomous runs
    auto_mode: bool = False  # Risk-based auto-execution (low=execute, high=defer)
    approval_callback_url: str | None = None  # URL for API approval mode
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None

    # Verification
    run_tests: bool = True
    run_review: bool = True  # PRReviewRunner on diffs
    capture_metrics: bool = True  # Before/after debate quality

    # Codebase metrics (before/after measurement)
    enable_codebase_metrics: bool = True
    metrics_test_scope: list[str] = field(default_factory=list)
    metrics_test_timeout: int = 120

    # Codebase knowledge indexing (semantic module summaries + dependency graph)
    enable_codebase_indexing: bool = True

    # Debug loop: iterative test-failure-feedback-retry execution
    enable_debug_loop: bool = True
    debug_loop_max_retries: int = 3

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
    # Codebase metrics delta (from MetricsCollector)
    metrics_delta: dict[str, Any] = field(default_factory=dict)
    improvement_score: float = 0.0
    # Budget tracking
    total_cost_usd: float = 0.0
    # KnowledgeMound persistence
    km_persisted: bool = False

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
            "metrics_delta": self.metrics_delta,
            "improvement_score": self.improvement_score,
            "total_cost_usd": self.total_cost_usd,
            "km_persisted": self.km_persisted,
        }


class _SelfImproveOrchestrationAdapter:
    """Adapter to make SelfImproveResult look like OrchestrationResult for NomicPipelineBridge.

    NomicPipelineBridge.create_pipeline_from_cycle() expects an OrchestrationResult
    with .goal, .summary, .assignments, .success, .duration_seconds, .improvement_score.
    This adapter wraps a SelfImproveResult to provide that interface.
    """

    def __init__(self, cycle_id: str, result: SelfImproveResult) -> None:
        self.goal = result.objective
        self.summary = f"Cycle {cycle_id}: {result.subtasks_completed}/{result.subtasks_total} subtasks completed"
        self.success = result.subtasks_completed > 0 and not result.regressions_detected
        self.duration_seconds = result.duration_seconds
        self.improvement_score = result.improvement_score
        # Build minimal assignment-like objects from result data
        self.assignments = _build_assignments_from_result(result)


def _build_assignments_from_result(result: SelfImproveResult) -> list[Any]:
    """Build minimal assignment-like objects for pipeline bridge consumption."""
    assignments = []
    for i, f in enumerate(result.files_changed[:20]):  # Cap at 20 for visualization
        assignment = types.SimpleNamespace(
            status="completed",
            agent_type="self_improve",
            track=types.SimpleNamespace(value="core"),
            subtask=types.SimpleNamespace(
                id=f"si-{i}",
                title=f"Modified: {f}",
                description="File changed during self-improvement cycle",
                file_scope=[f],
                estimated_complexity="medium",
                dependencies=[],
            ),
        )
        assignments.append(assignment)
    return assignments


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
        self._total_spend_usd: float = 0.0

    def _emit_progress(self, event: str, data: dict[str, Any]) -> None:
        """Emit a progress event via the configured callback."""
        if self.config.progress_callback:
            try:
                self.config.progress_callback(event, data)
            except Exception:
                pass

    async def run(self, objective: str | None = None) -> SelfImproveResult:
        """Run a full self-improvement cycle.

        Args:
            objective: High-level objective like "Improve test coverage".
                       None for self-directing mode (scan generates goals).

        Returns:
            SelfImproveResult with cycle outcomes
        """
        cycle_id = f"cycle_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        effective_objective = objective or "self-directed codebase improvement"
        result = SelfImproveResult(cycle_id=cycle_id, objective=effective_objective)

        logger.info(
            "self_improve_started cycle=%s objective=%s self_directing=%s",
            cycle_id,
            effective_objective[:100],
            objective is None,
        )
        self._emit_progress("cycle_started", {"cycle_id": cycle_id, "objective": effective_objective})

        # Step 0: Index codebase for richer planning context
        if self.config.enable_codebase_indexing:
            await self._index_codebase()

        # Step 1: Plan (None objective triggers scan-based goal synthesis)
        goals = await self._plan(objective)
        result.goals_planned = len(goals)
        self._emit_progress("planning_complete", {"goals": len(goals)})

        # In self-directing mode, synthesize objective from top goal
        if objective is None and goals:
            top_desc = getattr(goals[0], "description", str(goals[0]))
            effective_objective = f"[scan] {top_desc[:100]}"
            result.objective = effective_objective

        if not goals:
            logger.warning("self_improve_no_goals cycle=%s", cycle_id)
            result.duration_seconds = time.time() - start_time
            return result

        # Step 2: Decompose goals into subtasks
        subtasks = await self._decompose(goals)
        result.subtasks_total = len(subtasks)
        self._emit_progress("decomposition_complete", {"subtasks": len(subtasks)})

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

        self._emit_progress("execution_complete", {
            "completed": result.subtasks_completed,
            "failed": result.subtasks_failed,
        })

        # Step 5: Capture post-change metrics and compare
        outcome_comparison = None  # OutcomeComparison for feedback loop
        after = None
        if self.config.capture_metrics and baseline is not None:
            after = await self._capture_after()
            if after is not None:
                comparison = self._compare_metrics(baseline, after)

                if comparison:
                    result.metrics_delta = comparison.get("deltas", {})
                    result.improvement_score = comparison.get(
                        "improvement_score", 0.0
                    )
                    outcome_comparison = comparison.get(
                        "_outcome_comparison"
                    )

                    if not comparison.get("improved", True):
                        result.regressions_detected = True
                        if self.config.auto_revert_on_regression:
                            logger.warning(
                                "self_improve_regression cycle=%s recommendation=%s",
                                cycle_id,
                                comparison.get("recommendation"),
                            )
                            # Don't auto-revert yet -- log and let human decide
                            # result.reverted = True

        # Step 5b: Semantic goal evaluation
        goal_eval = self._evaluate_goal(
            effective_objective, subtasks, result, baseline, after
        )
        if goal_eval is not None:
            result.metrics_delta["goal_achievement"] = goal_eval.achievement_score
            result.metrics_delta["goal_scope_coverage"] = goal_eval.scope_coverage
            result.metrics_delta["goal_diff_relevance"] = goal_eval.diff_relevance
            if not goal_eval.achieved:
                logger.warning(
                    "self_improve_goal_not_achieved cycle=%s score=%.2f",
                    cycle_id,
                    goal_eval.achievement_score,
                )

        # Step 6: Persist outcomes (with OutcomeComparison for feedback loop)
        if self.config.persist_outcomes:
            self._persist_outcome(cycle_id, result, outcome_comparison)

        # Step 7: Run feedback orchestrator (6-step audit→active bridge)
        self._run_feedback_orchestrator(cycle_id, execution_results)

        # Step 8: Publish to pipeline visualization graph
        self._publish_to_pipeline_graph(cycle_id, result)

        result.duration_seconds = time.time() - start_time
        result.total_cost_usd = self._total_spend_usd
        logger.info(
            "self_improve_completed cycle=%s goals=%d subtasks=%d/%d "
            "duration=%.1fs cost=$%.4f",
            cycle_id,
            result.goals_planned,
            result.subtasks_completed,
            result.subtasks_total,
            result.duration_seconds,
            result.total_cost_usd,
        )
        self._emit_progress("cycle_complete", {
            "completed": result.subtasks_completed,
            "failed": result.subtasks_failed,
            "duration": result.duration_seconds,
        })

        return result

    async def dry_run(self, objective: str | None = None) -> dict[str, Any]:
        """Preview what the pipeline would do without executing.

        Returns the plan (goals + subtasks) without making any changes.
        """
        goals = await self._plan(objective)
        subtasks = await self._decompose(goals)

        # Synthesize objective label from top goal in self-directing mode
        effective_objective = objective
        if effective_objective is None and goals:
            top_desc = getattr(goals[0], "description", str(goals[0]))
            effective_objective = f"[scan] {top_desc[:100]}"
        elif effective_objective is None:
            effective_objective = "self-directed codebase improvement"

        return {
            "objective": effective_objective,
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

    async def _index_codebase(self) -> None:
        """Index codebase structure into MemoryFabric for planning context.

        Runs CodebaseKnowledgeBuilder to ingest module summaries and
        dependency graph. Non-fatal: failures are logged and swallowed.
        """
        try:
            from pathlib import Path as P

            from aragora.memory.codebase_builder import CodebaseKnowledgeBuilder
            from aragora.memory.fabric import MemoryFabric

            fabric = MemoryFabric()
            builder = CodebaseKnowledgeBuilder(fabric, P.cwd())

            summary_stats = await builder.ingest_module_summaries(max_modules=30)
            dep_stats = await builder.ingest_dependency_graph(max_files=100)

            logger.info(
                "codebase_indexed summaries=%d deps=%d errors=%d",
                summary_stats.items_ingested,
                dep_stats.items_ingested,
                summary_stats.errors + dep_stats.errors,
            )
        except ImportError as exc:
            logger.debug("CodebaseKnowledgeBuilder unavailable: %s", exc)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Codebase indexing failed: %s", exc)

    async def _plan(self, objective: str | None) -> list[Any]:
        """Step 1: Use MetaPlanner to prioritize goals.

        When objective is None (self-directing mode), MetaPlanner uses
        scan_mode to derive goals purely from codebase signals.
        """
        if not self.config.use_meta_planner:
            if objective is None:
                # Can't do direct objective mode without an objective
                logger.warning("self_directing requires use_meta_planner=True, enabling scan")
                self.config.use_meta_planner = True
                self.config.scan_mode = True
            else:
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
                scan_mode=self.config.scan_mode,
                max_goals=self.config.max_goals,
            )
            planner = MetaPlanner(config)
            # objective=None triggers self-directing scan in MetaPlanner
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
                    subtasks = decomposition.subtasks
                    # Enrich with KM learnings (async overlay)
                    try:
                        subtasks = await decomposer.enrich_subtasks_from_km(
                            desc, subtasks
                        )
                    except (RuntimeError, ValueError, OSError) as km_exc:
                        logger.debug("KM enrichment skipped: %s", km_exc)
                    all_subtasks.extend(subtasks)
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

        # Enrich subtasks with file_scope from CodebaseIndexer
        if all_subtasks and self.config.enable_codebase_indexing:
            await self._enrich_file_scope(all_subtasks)

        return all_subtasks

    async def _enrich_file_scope(self, subtasks: list[Any]) -> None:
        """Populate empty file_scope on subtasks using CodebaseIndexer.

        For each subtask with an empty ``file_scope``, queries the
        CodebaseIndexer for modules matching the subtask's title and
        description, then sets ``file_scope`` to the matching paths.
        """
        try:
            from aragora.nomic.codebase_indexer import CodebaseIndexer

            indexer = CodebaseIndexer(repo_path=".", max_modules=50)
            await indexer.index()

            for subtask in subtasks:
                scope = getattr(subtask, "file_scope", None)
                if scope is not None and len(scope) == 0:
                    title = getattr(subtask, "title", "")
                    desc = getattr(subtask, "description", str(subtask))
                    query = f"{title} {desc}".strip()
                    if query:
                        matches = await indexer.query(query, limit=5)
                        subtask.file_scope = [
                            str(m.path) for m in matches
                        ]

            logger.info(
                "file_scope_enrichment subtasks=%d",
                sum(
                    1
                    for s in subtasks
                    if getattr(s, "file_scope", None)
                ),
            )
        except ImportError:
            logger.debug("CodebaseIndexer unavailable, skipping file scope enrichment")
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("File scope enrichment failed: %s", exc)

    async def _capture_baseline(self) -> Any:
        """Step 3: Capture pre-change metrics (debate quality + codebase health)."""
        combined: dict[str, Any] = {"debate": None, "codebase": None}

        # Debate quality baseline
        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            tracker = NomicOutcomeTracker()
            combined["debate"] = await tracker.capture_baseline()
        except ImportError as exc:
            logger.debug("Outcome tracker unavailable for baseline: %s", exc)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Debate baseline capture failed: %s", exc)

        # Codebase health baseline
        if self.config.enable_codebase_metrics:
            try:
                from aragora.nomic.metrics_collector import (
                    MetricsCollector,
                    MetricsCollectorConfig,
                )

                mc_config = MetricsCollectorConfig(
                    test_timeout=self.config.metrics_test_timeout,
                    test_scope_dirs=list(self.config.metrics_test_scope),
                )
                collector = MetricsCollector(mc_config)
                snapshot = await collector.collect_baseline("self-improve")
                combined["codebase"] = snapshot.to_dict()
                # Stash collector for reuse in _capture_after
                self._metrics_collector = collector
            except ImportError as exc:
                logger.debug("MetricsCollector unavailable for baseline: %s", exc)
            except (RuntimeError, ValueError, OSError) as exc:
                logger.debug("Codebase baseline capture failed: %s", exc)

        if combined["debate"] is None and combined["codebase"] is None:
            return None
        return combined

    async def _capture_after(self) -> Any:
        """Step 5a: Capture post-change metrics (debate quality + codebase health)."""
        combined: dict[str, Any] = {"debate": None, "codebase": None}

        # Debate quality after
        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            tracker = NomicOutcomeTracker()
            combined["debate"] = await tracker.capture_after()
        except ImportError as exc:
            logger.debug("Outcome tracker unavailable for after: %s", exc)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Debate after capture failed: %s", exc)

        # Codebase health after (reuse collector from baseline for consistent config)
        if self.config.enable_codebase_metrics:
            try:
                collector = getattr(self, "_metrics_collector", None)
                if collector is None:
                    from aragora.nomic.metrics_collector import (
                        MetricsCollector,
                        MetricsCollectorConfig,
                    )

                    mc_config = MetricsCollectorConfig(
                        test_timeout=self.config.metrics_test_timeout,
                        test_scope_dirs=list(self.config.metrics_test_scope),
                    )
                    collector = MetricsCollector(mc_config)

                snapshot = await collector.collect_after("self-improve")
                combined["codebase"] = snapshot.to_dict()
            except ImportError as exc:
                logger.debug("MetricsCollector unavailable for after: %s", exc)
            except (RuntimeError, ValueError, OSError) as exc:
                logger.debug("Codebase after capture failed: %s", exc)

        if combined["debate"] is None and combined["codebase"] is None:
            return None
        return combined

    def _compare_metrics(
        self, baseline: Any, after: Any
    ) -> dict[str, Any] | None:
        """Step 5b: Compare baseline and after metrics (debate + codebase)."""
        if baseline is None or after is None:
            return None

        result: dict[str, Any] = {"improved": True, "recommendation": None, "deltas": {}}

        # Compare debate quality
        debate_baseline = baseline.get("debate") if isinstance(baseline, dict) else baseline
        debate_after = after.get("debate") if isinstance(after, dict) else after

        if debate_baseline is not None and debate_after is not None:
            try:
                from aragora.nomic.outcome_tracker import NomicOutcomeTracker

                tracker = NomicOutcomeTracker(
                    degradation_threshold=self.config.degradation_threshold,
                )
                comparison = tracker.compare(debate_baseline, debate_after)
                result["improved"] = comparison.improved
                result["recommendation"] = comparison.recommendation
                result["deltas"]["debate"] = comparison.metrics_delta
                # Stash the OutcomeComparison for the feedback loop
                result["_outcome_comparison"] = comparison
            except ImportError as exc:
                logger.debug("Outcome comparison unavailable: %s", exc)
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.debug("Outcome comparison failed: %s", exc)

        # Compare codebase health
        codebase_baseline = baseline.get("codebase") if isinstance(baseline, dict) else None
        codebase_after = after.get("codebase") if isinstance(after, dict) else None

        if codebase_baseline is not None and codebase_after is not None:
            try:
                from aragora.nomic.metrics_collector import MetricSnapshot

                base_snap = MetricSnapshot.from_dict(codebase_baseline)
                after_snap = MetricSnapshot.from_dict(codebase_after)

                collector = getattr(self, "_metrics_collector", None)
                if collector is None:
                    from aragora.nomic.metrics_collector import MetricsCollector

                    collector = MetricsCollector()

                delta = collector.compare(base_snap, after_snap)
                result["deltas"]["codebase"] = delta.to_dict()
                result["codebase_improved"] = delta.improved
                result["improvement_score"] = delta.improvement_score

                # If codebase regressed, mark as not improved
                if not delta.improved and delta.improvement_score < 0.3:
                    result["improved"] = False
                    if not result["recommendation"]:
                        result["recommendation"] = delta.summary
            except ImportError as exc:
                logger.debug("MetricsCollector comparison unavailable: %s", exc)
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.debug("Codebase comparison failed: %s", exc)

        return result

    def _evaluate_goal(
        self,
        objective: str,
        subtasks: list[Any],
        result: SelfImproveResult,
        baseline: Any,
        after: Any,
    ) -> Any:
        """Step 5b: Semantic evaluation of whether the goal was achieved.

        Uses GoalEvaluator to score scope coverage, test delta, and diff
        relevance. Returns a GoalEvaluation or None if the evaluator is
        unavailable.
        """
        try:
            from aragora.nomic.goal_evaluator import GoalEvaluator

            evaluator = GoalEvaluator()

            # Collect intended file scope from all subtasks
            file_scope: list[str] = []
            for st in subtasks:
                for f in getattr(st, "file_scope", []):
                    if f not in file_scope:
                        file_scope.append(f)

            # Test counts (before/after from codebase metrics if available)
            tests_before: dict[str, int] = {}
            tests_after: dict[str, int] = {}
            if isinstance(baseline, dict) and baseline.get("codebase"):
                cb = baseline["codebase"]
                tests_before = {
                    "passed": cb.get("test_pass_count", 0),
                    "failed": cb.get("test_fail_count", 0),
                }
            if isinstance(after, dict) and after.get("codebase"):
                ca = after["codebase"]
                tests_after = {
                    "passed": ca.get("test_pass_count", 0),
                    "failed": ca.get("test_fail_count", 0),
                }

            # Build diff summary from files changed (just the list for now)
            diff_summary = " ".join(result.files_changed)

            return evaluator.evaluate(
                goal=objective,
                file_scope=file_scope,
                files_changed=result.files_changed,
                diff_summary=diff_summary,
                tests_before=tests_before,
                tests_after=tests_after,
            )
        except ImportError:
            logger.debug("GoalEvaluator not available")
            return None
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Goal evaluation failed: %s", exc)
            return None

    async def _execute(
        self,
        subtasks: list[Any],
        cycle_id: str,
    ) -> list[dict[str, Any]]:
        """Step 4: Execute subtasks, respecting dependency order.

        Groups subtasks into dependency waves using ``SubTask.dependencies``.
        Subtasks within a wave (no unmet dependencies) execute in parallel;
        waves execute sequentially.
        """
        waves = self._group_dependency_waves(subtasks)

        if self.config.use_worktrees:
            all_results: list[dict[str, Any]] = []
            for wave in waves:
                wave_results = await self._execute_in_worktrees(wave, cycle_id)
                all_results.extend(wave_results)
            return all_results

        # Sequential execution respecting wave order
        results: list[dict[str, Any]] = []
        for wave in waves:
            for subtask in wave:
                result = await self._execute_single(subtask, cycle_id)
                results.append(result)
        return results

    @staticmethod
    def _group_dependency_waves(subtasks: list[Any]) -> list[list[Any]]:
        """Group subtasks into dependency waves for ordered execution.

        Wave 0: subtasks with no dependencies (or no dependency field).
        Wave N: subtasks whose dependencies are all in waves 0..N-1.

        Falls back to a single wave (all parallel) if dependency data is
        missing or if a cycle is detected.
        """
        # Build id -> subtask map
        by_id: dict[str, Any] = {}
        for st in subtasks:
            st_id = getattr(st, "id", None)
            if st_id:
                by_id[st_id] = st

        # If no subtask has dependencies, return single wave
        has_deps = any(getattr(st, "dependencies", None) for st in subtasks)
        if not has_deps:
            return [subtasks]

        assigned: set[str] = set()
        waves: list[list[Any]] = []
        remaining = list(subtasks)
        max_iterations = len(subtasks) + 1  # cycle guard

        for _ in range(max_iterations):
            if not remaining:
                break

            wave: list[Any] = []
            still_remaining: list[Any] = []

            for st in remaining:
                deps = getattr(st, "dependencies", None) or []
                # A subtask is ready if all its deps are already assigned
                # or if the dep ID doesn't exist in the subtask set
                unmet = [d for d in deps if d in by_id and d not in assigned]
                if not unmet:
                    wave.append(st)
                else:
                    still_remaining.append(st)

            if not wave:
                # All remaining have unmet deps (cycle) — flush them
                wave = still_remaining
                still_remaining = []

            waves.append(wave)
            for st in wave:
                st_id = getattr(st, "id", None)
                if st_id:
                    assigned.add(st_id)
            remaining = still_remaining

        return waves

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

        # Fallback: wave-based parallel execution
        results: list[dict[str, Any]] = []
        waves = self._dependency_waves(subtasks)
        for wave in waves:
            wave_results = await asyncio.gather(
                *[self._execute_single(st, cycle_id) for st in wave],
                return_exceptions=True,
            )
            for r in wave_results:
                if isinstance(r, BaseException):
                    logger.warning("Subtask in wave failed: %s", r)
                    results.append({
                        "success": False,
                        "files_changed": [],
                        "tests_passed": 0,
                        "tests_failed": 0,
                    })
                else:
                    results.append(r)

            # Check budget after each wave
            wave_cost = sum(
                r.get("cost_usd", 0.0) for r in results
                if isinstance(r, dict)
            )
            self._total_spend_usd += wave_cost
            if self._total_spend_usd > self.config.budget_limit_usd:
                logger.warning(
                    "budget_exceeded spend=%.2f limit=%.2f",
                    self._total_spend_usd,
                    self.config.budget_limit_usd,
                )
                break
        return results

    def _dependency_waves(self, subtasks: list[Any]) -> list[list[Any]]:
        """Group subtasks into dependency waves for parallel execution.

        Subtasks without explicit dependencies go in wave 0, etc.
        Currently all subtasks are treated as independent and batched
        by ``max_parallel``.

        Returns:
            List of waves, each wave is a list of subtasks to run concurrently.
        """
        max_p = max(self.config.max_parallel, 1)
        waves: list[list[Any]] = []
        batch: list[Any] = []
        for st in subtasks:
            batch.append(st)
            if len(batch) >= max_p:
                waves.append(batch)
                batch = []
        if batch:
            waves.append(batch)
        return waves

    async def _execute_single(
        self,
        subtask: Any,
        cycle_id: str,
        goal: Any | None = None,
    ) -> dict[str, Any]:
        """Execute a single subtask.

        In the current implementation, this generates an execution description
        but does not yet dispatch to a Claude Code session. This is the
        integration point where an execution agent would be invoked.

        Args:
            subtask: A SubTask, TaskDecomposition, TrackAssignment, or raw string
            cycle_id: The cycle identifier for logging
            goal: Optional PrioritizedGoal for richer instruction context

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

        # Read file contents for richer prompts
        file_contents = self._read_file_contents(subtask)

        # Extract worktree_path hint early so create_instruction can embed it
        wt_hint = getattr(subtask, "worktree_path", None)
        worktree_path = str(wt_hint) if wt_hint is not None else None

        # Attempt to use ExecutionBridge to generate + dispatch instruction
        try:
            from aragora.nomic.execution_bridge import ExecutionBridge

            bridge = ExecutionBridge()
            instruction = bridge.create_instruction(
                subtask,
                file_contents=file_contents,
                goal=goal,
                worktree_path=worktree_path,
            )

            logger.info(
                "execution_instruction generated subtask=%s",
                getattr(instruction, "subtask_id", "unknown")[:20],
            )

            # Write instruction to worktree for agent pickup
            dispatched = False
            executed = False
            # Prefer worktree_path from instruction (may have been enriched)
            if instruction.worktree_path:
                worktree_path = instruction.worktree_path
            files_changed: list[str] = []
            tests_passed = 0
            tests_failed = 0

            if worktree_path:
                dispatched = self._write_instruction_to_worktree(
                    instruction, worktree_path
                )

                # Try debug loop first (iterative test-feedback-retry)
                debug_result = await self._execute_with_debug_loop(
                    instruction, worktree_path, subtask
                )
                if debug_result is not None:
                    executed = True
                    files_changed = debug_result.get("files_changed", [])
                    tests_passed = debug_result.get("tests_passed", 0)
                    tests_failed = debug_result.get("tests_failed", 0)
                else:
                    # Fallback: single dispatch via Claude Code harness
                    exec_result = await self._dispatch_to_claude_code(
                        instruction, worktree_path
                    )
                    if exec_result is not None:
                        executed = True
                        files_changed = exec_result.get("files_changed", [])
                        tests_passed = exec_result.get("tests_passed", 0)
                        tests_failed = exec_result.get("tests_failed", 0)
                    else:
                        logger.warning(
                            "execute_subtask dispatch returned None for %s",
                            desc[:80],
                        )

            # Verify changes via PRReviewRunner if available
            if files_changed and worktree_path:
                try:
                    from aragora.nomic.execution_bridge import ExecutionResult

                    exec_result_obj = ExecutionResult(
                        subtask_id=getattr(instruction, "subtask_id", "unknown"),
                        success=True,
                        files_changed=files_changed,
                        diff_summary="\n".join(files_changed),
                    )
                    verification = await bridge.verify_changes(exec_result_obj)
                    logger.info(
                        "Verification result for %s: %s",
                        desc[:40],
                        verification.get("verified", "unknown"),
                    )
                except (ImportError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
                    logger.debug("Verification skipped: %s", exc)

            # Auto-commit changes in worktree for downstream merge
            if files_changed and worktree_path:
                try:
                    commit_result = subprocess.run(
                        ["git", "add", "-A"],
                        capture_output=True,
                        text=True,
                        cwd=worktree_path,
                        timeout=10,
                    )
                    if commit_result.returncode == 0:
                        subprocess.run(
                            [
                                "git", "commit", "-m",
                                f"self-improve: {getattr(instruction, 'subtask_id', 'unknown')[:40]}",
                            ],
                            capture_output=True,
                            text=True,
                            cwd=worktree_path,
                            timeout=10,
                            check=False,
                        )
                except (subprocess.TimeoutExpired, OSError):
                    pass

            # Generate per-subtask execution receipt
            receipt_hash = self._generate_subtask_receipt(
                subtask_id=getattr(instruction, "subtask_id", "unknown"),
                cycle_id=cycle_id,
                desc=desc,
                files_changed=files_changed,
                success=executed and tests_failed == 0,
            )

            return {
                "success": executed and len(files_changed) > 0,
                "subtask": desc[:100],
                "instruction_generated": True,
                "instruction_dispatched": dispatched,
                "instruction_executed": executed,
                "worktree_path": worktree_path,
                "files_changed": files_changed,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "receipt_hash": receipt_hash,
            }
        except ImportError:
            pass
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            logger.debug("ExecutionBridge failed: %s", exc)

        return {
            "success": False,
            "subtask": desc[:100],
            "instruction_generated": False,
            "instruction_dispatched": False,
            "files_changed": [],
            "tests_passed": 0,
            "tests_failed": 0,
        }

    async def _execute_with_debug_loop(
        self,
        instruction: Any,
        worktree_path: str,
        subtask: Any = None,
    ) -> dict[str, Any] | None:
        """Execute via iterative debug loop with test-failure-retry.

        Returns None if the debug loop is disabled or unavailable,
        falling back to the single-dispatch path.
        """
        if not self.config.enable_debug_loop:
            return None

        try:
            from aragora.nomic.debug_loop import DebugLoop, DebugLoopConfig

            config = DebugLoopConfig(
                max_retries=self.config.debug_loop_max_retries,
                test_timeout=self.config.metrics_test_timeout,
            )
            debug = DebugLoop(config)

            prompt = instruction.to_agent_prompt()
            subtask_id = getattr(instruction, "subtask_id", "unknown")

            # Infer test scope from file_hints
            test_scope = self._infer_test_scope(subtask)

            debug_result = await debug.execute_with_retry(
                instruction=prompt,
                worktree_path=worktree_path,
                test_scope=test_scope or None,
                subtask_id=subtask_id,
            )

            return {
                "files_changed": debug_result.final_files_changed,
                "tests_passed": debug_result.final_tests_passed,
                "tests_failed": debug_result.final_tests_failed,
                "debug_loop_attempts": debug_result.total_attempts,
                "debug_loop_success": debug_result.success,
            }

        except ImportError:
            logger.debug("DebugLoop not available, falling back to single dispatch")
            return None
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Debug loop failed, falling back: %s", exc)
            return None

    def _infer_test_scope(self, subtask: Any) -> list[str]:
        """Infer test directories from subtask file hints."""
        test_dirs: list[str] = []
        file_hints = getattr(subtask, "file_scope", [])
        if not file_hints and hasattr(subtask, "goal"):
            file_hints = getattr(subtask.goal, "file_hints", [])

        for hint in file_hints:
            if hint.startswith("aragora/"):
                parts = hint.split("/")
                if len(parts) >= 2:
                    test_dir = f"tests/{parts[1]}"
                    if test_dir not in test_dirs:
                        test_dirs.append(test_dir)
            elif hint.startswith("tests/"):
                if hint not in test_dirs:
                    test_dirs.append(hint)

        return test_dirs

    @staticmethod
    def _read_file_contents(
        subtask: Any,
        max_chars_per_file: int = 2000,
        max_total_chars: int = 10000,
    ) -> dict[str, str]:
        """Read truncated file contents from subtask file_scope.

        Gives the execution agent real code context instead of just file paths.
        Reads the first ``max_chars_per_file`` characters of each file,
        capped at ``max_total_chars`` total across all files.

        Returns:
            Dict mapping file path -> truncated content.
        """
        from pathlib import Path as P

        file_hints: list[str] = getattr(subtask, "file_scope", [])
        if not file_hints and hasattr(subtask, "goal"):
            file_hints = getattr(subtask.goal, "file_hints", [])

        contents: dict[str, str] = {}
        total = 0

        for hint in file_hints:
            if total >= max_total_chars:
                break
            path = P(hint)
            if not path.exists() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                budget = min(max_chars_per_file, max_total_chars - total)
                snippet = text[:budget]
                if len(text) > budget:
                    snippet += "\n# ... [truncated]"
                contents[hint] = snippet
                total += len(snippet)
            except OSError:
                continue

        return contents

    def _assess_execution_risk(self, instruction: Any) -> str:
        """Assess risk level of an execution instruction.

        Returns "low", "medium", or "high" based on:
        - File count and scope (tests-only = low, core modules = high)
        - Whether files are in protected paths (CLAUDE.md, __init__.py, etc.)
        - Whether the instruction involves deletions vs additions
        """
        file_hints: list[str] = getattr(instruction, "file_hints", [])
        if not file_hints:
            file_hints = getattr(instruction, "file_scope", [])

        # Protected files -> always high risk
        protected = {"CLAUDE.md", "__init__.py", ".env", "nomic_loop.py"}
        if any(any(p in f for p in protected) for f in file_hints):
            return "high"

        # Tests-only changes -> low risk
        if all(f.startswith("tests/") for f in file_hints) and file_hints:
            return "low"

        # Many files -> high risk
        if len(file_hints) > 10:
            return "high"

        # Core modules -> medium risk
        core_paths = {"aragora/debate/", "aragora/server/", "aragora/nomic/"}
        if any(any(f.startswith(c) for c in core_paths) for f in file_hints):
            return "medium"

        return "low"

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
            try:
                from aragora.nomic.approval import ApprovalGate, ApprovalDecision

                mode = "auto" if self.config.autonomous else "cli"
                gate = ApprovalGate(
                    mode=mode,
                    callback_url=getattr(self.config, "approval_callback_url", None),
                )
                decision = await gate.request_approval(instruction)

                if decision == ApprovalDecision.REJECT:
                    logger.info(
                        "dispatch_rejected subtask=%s",
                        instruction.subtask_id[:20],
                    )
                    return None
                if decision == ApprovalDecision.DEFER:
                    logger.info(
                        "dispatch_deferred subtask=%s",
                        instruction.subtask_id[:20],
                    )
                    return {"deferred": True, "risk_level": "high", "files_changed": []}
                if decision == ApprovalDecision.SKIP:
                    logger.info(
                        "dispatch_skipped subtask=%s",
                        instruction.subtask_id[:20],
                    )
                    return {"skipped": True, "files_changed": []}
                # APPROVE: fall through to execution
            except ImportError:
                # Fallback to legacy gate if approval module unavailable
                if not self.config.autonomous:
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

            # Parse cost from Claude Code output
            cost_usd = self._parse_cost_from_output(stdout)
            self._total_spend_usd += cost_usd

            logger.info(
                "dispatch_completed subtask=%s files=%d tests=%d/%d cost=$%.4f",
                instruction.subtask_id[:20],
                len(files_changed),
                tests_passed,
                tests_passed + tests_failed,
                cost_usd,
            )

            # Check budget
            if self._total_spend_usd > self.config.budget_limit_usd:
                logger.warning(
                    "budget_exceeded spend=%.2f limit=%.2f",
                    self._total_spend_usd,
                    self.config.budget_limit_usd,
                )
                raise BudgetExceededError(
                    f"Spent ${self._total_spend_usd:.2f} of "
                    f"${self.config.budget_limit_usd:.2f} budget"
                )

            return {
                "files_changed": files_changed,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "stdout_len": len(stdout),
                "cost_usd": cost_usd,
            }

        except BudgetExceededError:
            raise  # Let budget errors propagate
        except ImportError as exc:
            logger.debug("ClaudeCodeHarness not available: %s", exc)
            return None
        except (RuntimeError, OSError, asyncio.TimeoutError) as exc:
            logger.warning("Claude Code dispatch failed: %s", exc)
            return None

    @staticmethod
    def _parse_cost_from_output(output: str) -> float:
        """Parse cost from Claude Code CLI output.

        Looks for patterns like ``Total cost: $0.42`` or token counts
        to estimate cost.
        """
        import re

        # Direct cost reporting: "Total cost: $X.XX"
        m = re.search(r"(?:Total cost|Cost):\s*\$?([\d.]+)", output)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

        # Estimate from token counts: "input=1234, output=5678"
        m = re.search(r"input[=:]\s*(\d+).*?output[=:]\s*(\d+)", output)
        if m:
            input_tokens = int(m.group(1))
            output_tokens = int(m.group(2))
            # Rough estimate: $3/M input, $15/M output (Claude pricing)
            return (input_tokens * 3 + output_tokens * 15) / 1_000_000

        return 0.0

    async def _run_tests_in_worktree(
        self,
        worktree_path: str,
    ) -> dict[str, int]:
        """Run pytest in a worktree and return pass/fail counts."""
        try:
            import subprocess

            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-m", "pytest", "--tb=no", "-q", "--timeout=30"],
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

    @staticmethod
    def _generate_subtask_receipt(
        subtask_id: str,
        cycle_id: str,
        desc: str,
        files_changed: list[str],
        success: bool,
    ) -> str | None:
        """Generate a DecisionReceipt for a completed subtask.

        Returns the receipt hash string, or None if receipt generation fails.
        """
        try:
            import hashlib

            # Build a deterministic receipt content hash
            content = f"{cycle_id}:{subtask_id}:{desc}:{','.join(sorted(files_changed))}:{success}"
            receipt_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            try:
                from aragora.export.decision_receipt import DecisionReceipt

                receipt = DecisionReceipt(
                    receipt_id=f"si_{subtask_id}_{receipt_hash}",
                    verdict="approved" if success else "rejected",
                    summary=desc[:200],
                    evidence=[f"files: {', '.join(files_changed[:5])}"],
                    metadata={
                        "cycle_id": cycle_id,
                        "subtask_id": subtask_id,
                        "type": "self_improve_subtask",
                    },
                )

                # Persist receipt to KM
                from aragora.knowledge.mound.adapters.receipt_adapter import ReceiptAdapter
                import asyncio

                adapter = ReceiptAdapter()
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(adapter.ingest_receipt(
                        receipt,
                        tags=["self_improve", "subtask", cycle_id],
                    ))
                except RuntimeError:
                    pass
            except ImportError:
                pass

            logger.info(
                "subtask_receipt generated=%s subtask=%s success=%s",
                receipt_hash,
                subtask_id[:20],
                success,
            )
            return receipt_hash

        except (RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Subtask receipt generation failed: %s", exc)
            return None

    def _persist_outcome(
        self,
        cycle_id: str,
        result: SelfImproveResult,
        outcome_comparison: Any = None,
    ) -> None:
        """Step 6: Persist cycle outcome to CycleLearningStore.

        Args:
            cycle_id: The cycle identifier.
            result: The pipeline result.
            outcome_comparison: Optional OutcomeComparison from debate metrics
                comparison. When provided, also records to OutcomeTracker for
                the cross-cycle feedback loop.
        """
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
                "improvement_score": result.improvement_score,
            }
            if result.metrics_delta:
                record.evidence_quality_scores["has_metrics"] = 1.0

            store.save_cycle(record)
            logger.info("cycle_outcome_persisted cycle=%s", cycle_id)

            # Record OutcomeComparison to close the feedback loop
            if outcome_comparison is not None:
                try:
                    from aragora.nomic.outcome_tracker import NomicOutcomeTracker

                    tracker = NomicOutcomeTracker()
                    tracker.record_cycle_outcome(cycle_id, outcome_comparison)
                    logger.info(
                        "outcome_comparison_recorded cycle=%s improved=%s",
                        cycle_id,
                        outcome_comparison.improved,
                    )
                except (ImportError, RuntimeError, ValueError, OSError) as exc:
                    logger.debug(
                        "Failed to record outcome comparison: %s", exc
                    )

            # Feed outcomes to MetaPlanner for cross-cycle learning
            try:
                from aragora.nomic.meta_planner import MetaPlanner

                goal_outcomes = []
                # Build goal outcomes from result
                if result.subtasks_completed > 0 or result.subtasks_failed > 0:
                    goal_outcomes.append({
                        "track": "core",
                        "success": result.subtasks_completed > 0 and result.subtasks_failed == 0,
                        "description": result.objective,
                    })

                if goal_outcomes:
                    planner = MetaPlanner()
                    planner.record_outcome(
                        goal_outcomes=goal_outcomes,
                        objective=result.objective,
                    )
                    logger.info(
                        "meta_planner_outcome_recorded cycle=%s goals=%d",
                        cycle_id,
                        len(goal_outcomes),
                    )
            except (ImportError, RuntimeError, ValueError, TypeError) as exc:
                logger.debug("MetaPlanner outcome recording skipped: %s", exc)

            # Persist to KnowledgeMound for cross-cycle learning
            try:
                from aragora.pipeline.km_bridge import PipelineKMBridge

                bridge = PipelineKMBridge()
                if bridge.available:
                    stored = bridge.store_pipeline_result({
                        "cycle_id": cycle_id,
                        "objective": result.objective,
                        "success": (
                            result.subtasks_completed > 0
                            and result.subtasks_failed == 0
                        ),
                        "subtasks_completed": result.subtasks_completed,
                        "subtasks_failed": result.subtasks_failed,
                        "files_changed": result.files_changed,
                        "improvement_score": result.improvement_score,
                        "metrics_delta": result.metrics_delta,
                        "duration_seconds": result.duration_seconds,
                        "total_cost_usd": result.total_cost_usd,
                    })
                    if stored:
                        result.km_persisted = True
                        logger.info("km_bridge_persisted cycle=%s", cycle_id)
            except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
                logger.debug("KM bridge persistence skipped: %s", exc)

        except ImportError as exc:
            logger.debug("Failed to persist cycle outcome (import): %s", exc)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Failed to persist cycle outcome: %s", exc)

    def _run_feedback_orchestrator(
        self,
        cycle_id: str,
        execution_results: list[dict[str, Any]],
    ) -> None:
        """Step 7: Run the 6-step feedback orchestrator.

        Bridges Gauntlet, Introspection, Genesis, Learning, Workspace,
        and Pulse into active feedback goals persisted to ImprovementQueue.
        Next cycle's scan mode reads these as Signal 8.
        """
        try:
            from aragora.nomic.feedback_orchestrator import SelfImproveFeedbackOrchestrator

            orchestrator = SelfImproveFeedbackOrchestrator()
            feedback = orchestrator.run(cycle_id, execution_results)

            logger.info(
                "feedback_orchestrator_result cycle=%s goals=%d steps=%d/%d",
                cycle_id,
                feedback.goals_generated,
                feedback.steps_completed,
                feedback.steps_completed + feedback.steps_failed,
            )
        except ImportError:
            logger.debug("FeedbackOrchestrator not available")
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Feedback orchestrator failed (non-critical): %s", exc)

    def _publish_to_pipeline_graph(
        self,
        cycle_id: str,
        result: SelfImproveResult,
    ) -> None:
        """Publish cycle results as a pipeline visualization graph.

        Creates a UniversalGraph from the cycle results and persists it
        to GraphStore, making self-improvement cycles visible in the
        /pipeline canvas UI.
        """
        try:
            from aragora.nomic.pipeline_bridge import NomicPipelineBridge
            from aragora.pipeline.graph_store import get_graph_store

            # Build a minimal OrchestrationResult-like object for the bridge
            # We use a lightweight wrapper since SelfImproveResult doesn't
            # have the same shape as OrchestrationResult
            cycle_data = _SelfImproveOrchestrationAdapter(cycle_id, result)

            bridge = NomicPipelineBridge()
            graph = bridge.create_pipeline_from_cycle(cycle_data)

            store = get_graph_store()
            store.create(graph)

            logger.info(
                "pipeline_graph_published cycle=%s graph_id=%s nodes=%d",
                cycle_id,
                graph.id,
                len(graph.nodes),
            )
        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as e:
            logger.debug("Pipeline graph publication skipped: %s", e)


__all__ = [
    "SelfImproveConfig",
    "SelfImprovePipeline",
    "SelfImproveResult",
]
