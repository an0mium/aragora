"""Self-Improvement Daemon — continuous autonomous improvement loop.

Ties together AutonomousAssessmentEngine, GoalGenerator, and
SelfImprovePipeline into a daemon that continuously identifies and
implements its own best improvements.

Core loop:
1. assess() → health report
2. Skip if health_score > threshold (codebase is healthy enough)
3. generate_goals(report) → ranked goals
4. SelfImprovePipeline.run(objective=top_goal) → execute
5. assess() again → measure health delta
6. record_outcome() → feed back to MetaPlanner for next cycle
7. Sleep → repeat

Usage:
    daemon = SelfImprovementDaemon()
    await daemon.start()       # Start continuous loop
    await daemon.stop()        # Graceful shutdown
    result = await daemon.trigger_cycle()  # Manual one-shot

    status = daemon.status()   # Query state
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DaemonState(Enum):
    """Daemon lifecycle states."""

    IDLE = "idle"
    RUNNING = "running"
    ASSESSING = "assessing"
    EXECUTING = "executing"
    STOPPED = "stopped"


@dataclass
class CycleResult:
    """Result of a single daemon improvement cycle."""

    cycle_number: int
    health_before: float
    health_after: float | None = None
    health_delta: float = 0.0
    goal_executed: str = ""
    success: bool = False
    skipped: bool = False
    skip_reason: str = ""
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_number": self.cycle_number,
            "health_before": self.health_before,
            "health_after": self.health_after,
            "health_delta": self.health_delta,
            "goal_executed": self.goal_executed,
            "success": self.success,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass
class DaemonConfig:
    """Configuration for the self-improvement daemon."""

    # Assessment thresholds
    health_threshold: float = 0.95  # Skip if health above this
    min_candidates: int = 1  # Skip if fewer candidates than this

    # Execution limits
    budget_limit_per_cycle_usd: float = 5.0
    budget_limit_cumulative_usd: float = 50.0
    max_consecutive_failures: int = 3
    max_cycles: int = 0  # 0 = unlimited

    # Timing
    interval_seconds: float = 3600.0  # 1 hour between cycles
    cooldown_after_failure_seconds: float = 300.0  # 5 min after failure

    # Safety
    dry_run: bool = False
    require_approval: bool = True

    # Pipeline config passthrough
    use_worktrees: bool = True
    run_tests: bool = True
    autonomous: bool = False


@dataclass
class DaemonStatus:
    """Current daemon status snapshot."""

    state: str
    cycles_completed: int = 0
    cycles_failed: int = 0
    consecutive_failures: int = 0
    last_health_score: float | None = None
    last_cycle_time: float | None = None
    cumulative_budget_used_usd: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "cycles_completed": self.cycles_completed,
            "cycles_failed": self.cycles_failed,
            "consecutive_failures": self.consecutive_failures,
            "last_health_score": self.last_health_score,
            "last_cycle_time": self.last_cycle_time,
            "cumulative_budget_used_usd": self.cumulative_budget_used_usd,
            "history": self.history[-10:],  # Last 10 cycles
        }


class SelfImprovementDaemon:
    """Continuous autonomous self-improvement daemon.

    Runs an assess→generate→execute→measure loop on a configurable
    interval, with safety guards for budget, failure limits, and
    health thresholds.
    """

    def __init__(self, config: DaemonConfig | None = None) -> None:
        self.config = config or DaemonConfig()
        self._state = DaemonState.IDLE
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._cycles_completed = 0
        self._cycles_failed = 0
        self._consecutive_failures = 0
        self._cumulative_budget_usd = 0.0
        self._last_health_score: float | None = None
        self._last_cycle_time: float | None = None
        self._history: list[CycleResult] = []

    @property
    def state(self) -> DaemonState:
        return self._state

    async def start(self) -> None:
        """Start the continuous improvement loop."""
        if self._state == DaemonState.RUNNING:
            logger.warning("daemon_already_running")
            return

        self._state = DaemonState.RUNNING
        self._stop_event.clear()

        logger.info(
            "daemon_started interval=%ds threshold=%.2f dry_run=%s",
            self.config.interval_seconds,
            self.config.health_threshold,
            self.config.dry_run,
        )

        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Gracefully stop the daemon."""
        if self._state == DaemonState.STOPPED:
            return

        logger.info("daemon_stopping")
        self._stop_event.set()

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._state = DaemonState.STOPPED
        logger.info(
            "daemon_stopped cycles_completed=%d cycles_failed=%d",
            self._cycles_completed,
            self._cycles_failed,
        )

    async def trigger_cycle(self) -> CycleResult:
        """Manually trigger a single improvement cycle."""
        return await self._run_cycle(self._cycles_completed + 1)

    def get_status(self) -> DaemonStatus:
        """Get current daemon status."""
        return DaemonStatus(
            state=self._state.value,
            cycles_completed=self._cycles_completed,
            cycles_failed=self._cycles_failed,
            consecutive_failures=self._consecutive_failures,
            last_health_score=self._last_health_score,
            last_cycle_time=self._last_cycle_time,
            cumulative_budget_used_usd=self._cumulative_budget_usd,
            history=[r.to_dict() for r in self._history[-10:]],
        )

    # --- Private ---

    async def _loop(self) -> None:
        """Main daemon loop."""
        cycle_num = 0

        while not self._stop_event.is_set():
            cycle_num += 1

            # Check max cycles
            if self.config.max_cycles > 0 and cycle_num > self.config.max_cycles:
                logger.info("daemon_max_cycles_reached max=%d", self.config.max_cycles)
                break

            # Check consecutive failure limit
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                logger.warning(
                    "daemon_too_many_failures consecutive=%d limit=%d",
                    self._consecutive_failures,
                    self.config.max_consecutive_failures,
                )
                break

            # Check cumulative budget
            if self._cumulative_budget_usd >= self.config.budget_limit_cumulative_usd:
                logger.warning(
                    "daemon_budget_exhausted cumulative=%.2f limit=%.2f",
                    self._cumulative_budget_usd,
                    self.config.budget_limit_cumulative_usd,
                )
                break

            # Run a cycle
            result = await self._run_cycle(cycle_num)
            self._history.append(result)

            if result.success:
                self._cycles_completed += 1
                self._consecutive_failures = 0
            elif not result.skipped:
                self._cycles_failed += 1
                self._consecutive_failures += 1

            self._last_cycle_time = time.time()

            # Sleep between cycles (with failure cooldown)
            sleep_time = self.config.interval_seconds
            if result.error:
                sleep_time = self.config.cooldown_after_failure_seconds

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=sleep_time,
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Normal — sleep elapsed, continue loop

        self._state = DaemonState.STOPPED

    async def _run_cycle(self, cycle_num: int) -> CycleResult:
        """Run a single assess→generate→execute→measure cycle."""
        start = time.time()
        result = CycleResult(cycle_number=cycle_num, health_before=0.0)

        try:
            # Step 1: Assess
            self._state = DaemonState.ASSESSING
            report = await self._assess()
            result.health_before = report.health_score
            self._last_health_score = report.health_score

            # Step 2: Check threshold
            if report.health_score >= self.config.health_threshold:
                result.skipped = True
                result.skip_reason = (
                    f"Health score {report.health_score:.2f} >= "
                    f"threshold {self.config.health_threshold:.2f}"
                )
                result.duration_seconds = time.time() - start
                logger.info(
                    "daemon_cycle_skipped cycle=%d reason=%s",
                    cycle_num,
                    result.skip_reason,
                )
                return result

            # Step 3: Check minimum candidates
            candidates = getattr(report, "improvement_candidates", [])
            if len(candidates) < self.config.min_candidates:
                result.skipped = True
                result.skip_reason = (
                    f"Only {len(candidates)} candidates < min {self.config.min_candidates}"
                )
                result.duration_seconds = time.time() - start
                return result

            # Step 4: Generate goals
            goals = self._generate_goals(report)
            if not goals:
                result.skipped = True
                result.skip_reason = "No goals generated"
                result.duration_seconds = time.time() - start
                return result

            # Step 5: Execute top goal
            self._state = DaemonState.EXECUTING
            top_goal = goals[0]
            objective = getattr(top_goal, "description", str(top_goal))
            result.goal_executed = objective

            if self.config.dry_run:
                result.skipped = True
                result.skip_reason = "Dry run mode"
                result.success = True
                result.duration_seconds = time.time() - start
                logger.info(
                    "daemon_dry_run cycle=%d goal=%s",
                    cycle_num,
                    objective[:80],
                )
                return result

            pipeline_result = await self._execute(objective)

            # Step 6: Measure health delta
            self._state = DaemonState.ASSESSING
            after_report = await self._assess()
            result.health_after = after_report.health_score
            result.health_delta = after_report.health_score - report.health_score
            self._last_health_score = after_report.health_score

            # Step 7: Record outcome
            self._record_outcome(objective, pipeline_result)

            result.success = getattr(pipeline_result, "subtasks_completed", 0) > 0 and not getattr(
                pipeline_result, "regressions_detected", False
            )
            result.duration_seconds = time.time() - start

            logger.info(
                "daemon_cycle_complete cycle=%d goal=%s health=%.2f→%.2f delta=%.3f",
                cycle_num,
                objective[:60],
                result.health_before,
                result.health_after or 0.0,
                result.health_delta,
            )

        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as e:
            result.error = f"{type(e).__name__}: {e}"
            result.duration_seconds = time.time() - start
            logger.warning("daemon_cycle_error cycle=%d error=%s", cycle_num, result.error)

        return result

    async def _assess(self) -> Any:
        """Run autonomous assessment."""
        from aragora.nomic.assessment_engine import AutonomousAssessmentEngine

        engine = AutonomousAssessmentEngine()
        return await engine.assess()

    def _generate_goals(self, report: Any) -> list[Any]:
        """Generate improvement goals from assessment report."""
        from aragora.nomic.goal_generator import GoalGenerator

        generator = GoalGenerator()
        return generator.generate_goals(report)

    async def _execute(self, objective: str) -> Any:
        """Execute a self-improvement cycle via SelfImprovePipeline."""
        from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline

        config = SelfImproveConfig(
            use_worktrees=self.config.use_worktrees,
            budget_limit_usd=self.config.budget_limit_per_cycle_usd,
            require_approval=self.config.require_approval,
            autonomous=self.config.autonomous,
            run_tests=self.config.run_tests,
        )
        pipeline = SelfImprovePipeline(config)
        return await pipeline.run(objective)

    def _record_outcome(self, objective: str, pipeline_result: Any) -> None:
        """Record execution outcome to MetaPlanner."""
        try:
            from aragora.nomic.meta_planner import MetaPlanner

            success = (
                getattr(pipeline_result, "subtasks_completed", 0) > 0
                and getattr(pipeline_result, "subtasks_failed", 0) == 0
            )

            goal_outcomes = [
                {
                    "track": "core",
                    "success": success,
                    "description": objective,
                }
            ]

            planner = MetaPlanner()
            planner.record_outcome(goal_outcomes=goal_outcomes, objective=objective)
        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            logger.debug("Outcome recording skipped: %s", e)


__all__ = [
    "CycleResult",
    "DaemonConfig",
    "DaemonState",
    "DaemonStatus",
    "SelfImprovementDaemon",
]
