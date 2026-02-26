"""SwarmCommander: interrogate -> spec -> dispatch -> merge -> report.

Top-level orchestrator that wraps HardenedOrchestrator with user-facing
interrogation and plain-English reporting phases.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from aragora.swarm.config import SwarmCommanderConfig
from aragora.swarm.interrogator import SwarmInterrogator
from aragora.swarm.reporter import SwarmReport, SwarmReporter
from aragora.swarm.spec import SwarmSpec

logger = logging.getLogger(__name__)


class SwarmCommander:
    """Top-level orchestrator: interrogate -> spec -> dispatch -> merge -> report.

    Wraps HardenedOrchestrator with user-facing phases. The user interacts
    only with the interrogation and reporting phases; the dispatch phase
    delegates entirely to the existing orchestration infrastructure.

    Usage:
        commander = SwarmCommander()
        report = await commander.run("Make the dashboard faster")
        print(report.to_plain_text())

    Or with a pre-built spec:
        spec = SwarmSpec.from_yaml(Path("my-spec.yaml").read_text())
        report = await commander.run_from_spec(spec)
    """

    def __init__(self, config: SwarmCommanderConfig | None = None) -> None:
        self.config = config or SwarmCommanderConfig()
        self._interrogator = SwarmInterrogator(self.config.interrogator)
        self._reporter = SwarmReporter()
        self._spec: SwarmSpec | None = None
        self._result: Any = None

    async def run(
        self,
        initial_goal: str,
        input_fn: Any | None = None,
        print_fn: Any | None = None,
    ) -> SwarmReport:
        """Full swarm lifecycle: interrogate -> dispatch -> report.

        Args:
            initial_goal: The user's goal in plain language.
            input_fn: Custom input function (default: builtin input).
            print_fn: Custom print function (default: builtin print).

        Returns:
            SwarmReport with plain-English summary.
        """
        _print = print_fn or print

        # Phase 1: Interrogation
        _print("\n[Phase 1/3] Gathering requirements...\n")
        self._spec = await self._interrogator.interrogate(
            initial_goal, input_fn=input_fn, print_fn=print_fn
        )

        # Phase 2: Dispatch
        _print("\n[Phase 2/3] Dispatching agents...\n")
        start_time = time.monotonic()
        self._result = await self._dispatch(self._spec)
        duration = time.monotonic() - start_time

        # Phase 3: Report
        _print("\n[Phase 3/3] Generating report...\n")
        report = await self._reporter.generate(
            spec=self._spec,
            result=self._result,
            duration_seconds=duration,
        )

        _print(report.to_plain_text())
        return report

    async def run_from_spec(
        self,
        spec: SwarmSpec,
        print_fn: Any | None = None,
    ) -> SwarmReport:
        """Skip interrogation, run from a pre-built spec.

        Args:
            spec: A pre-built SwarmSpec (from YAML, JSON, or previous run).
            print_fn: Custom print function.

        Returns:
            SwarmReport with plain-English summary.
        """
        _print = print_fn or print
        self._spec = spec

        _print("\n[Phase 1/2] Dispatching agents...\n")
        start_time = time.monotonic()
        self._result = await self._dispatch(spec)
        duration = time.monotonic() - start_time

        _print("\n[Phase 2/2] Generating report...\n")
        report = await self._reporter.generate(
            spec=spec,
            result=self._result,
            duration_seconds=duration,
        )

        _print(report.to_plain_text())
        return report

    async def dry_run(
        self,
        initial_goal: str,
        input_fn: Any | None = None,
        print_fn: Any | None = None,
    ) -> SwarmSpec:
        """Run interrogation only, produce a spec without executing.

        Args:
            initial_goal: The user's goal in plain language.
            input_fn: Custom input function.
            print_fn: Custom print function.

        Returns:
            The produced SwarmSpec (no execution).
        """
        _print = print_fn or print

        _print("\n[DRY RUN] Gathering requirements only (no agents will be dispatched)...\n")
        spec = await self._interrogator.interrogate(
            initial_goal, input_fn=input_fn, print_fn=print_fn
        )

        _print("\n" + "=" * 60)
        _print("SPEC (would be used for dispatch)")
        _print("=" * 60)
        _print(spec.to_json(indent=2))
        _print("")

        return spec

    async def _dispatch(self, spec: SwarmSpec) -> Any:
        """Dispatch the swarm using HardenedOrchestrator.

        Translates the SwarmSpec into HardenedOrchestrator parameters
        and executes the goal.
        """
        orchestrator = self._build_orchestrator(spec)

        context: dict[str, Any] = {}
        if spec.acceptance_criteria:
            context["acceptance_criteria"] = spec.acceptance_criteria
        if spec.constraints:
            context["constraints"] = spec.constraints
        if spec.user_expertise:
            context["user_expertise"] = spec.user_expertise
        if spec.file_scope_hints:
            context["file_scope_hints"] = spec.file_scope_hints

        tracks = spec.track_hints if spec.track_hints else None

        try:
            result = await orchestrator.execute_goal_coordinated(
                goal=spec.refined_goal or spec.raw_goal,
                tracks=tracks,
                max_cycles=self.config.max_cycles,
                context=context if context else None,
            )
            return result
        except Exception as exc:
            logger.error("Swarm dispatch failed: %s", exc)
            # Return a minimal result so reporting still works
            return _ErrorResult(str(exc))

    async def run_iterative(
        self,
        initial_goal: str,
        input_fn: Any | None = None,
        print_fn: Any | None = None,
    ) -> list[SwarmReport]:
        """Run the swarm in an iterative loop: run -> report -> 'what next?' -> repeat.

        Args:
            initial_goal: The user's first goal in plain language.
            input_fn: Custom input function (default: builtin input).
            print_fn: Custom print function (default: builtin print).

        Returns:
            List of SwarmReports from each cycle.
        """
        _input = input_fn or input
        _print = print_fn or print
        reports: list[SwarmReport] = []
        goal = initial_goal
        cycle = 1

        while True:
            _print(f"\n{'=' * 60}")
            _print(f"  Cycle {cycle}")
            _print(f"{'=' * 60}")

            report = await self.run(goal, input_fn=input_fn, print_fn=print_fn)
            reports.append(report)

            if not self.config.iterative_mode:
                break

            _print("\n" + "-" * 60)
            _print("What would you like to do next?")
            _print('(Type "done", "quit", or "exit" to finish)')
            _print("-" * 60)
            next_input = _input("> ")

            if next_input.strip().lower() in ("done", "quit", "exit", ""):
                _print("\nAll done! Here's a summary of what was accomplished:\n")
                for i, r in enumerate(reports, 1):
                    _print(f"  Cycle {i}: {r.summary}")
                break

            goal = next_input.strip()
            # Reset interrogator for new goal
            self._interrogator = SwarmInterrogator(self.config.interrogator)
            cycle += 1

        return reports

    def _build_orchestrator(self, spec: SwarmSpec) -> Any:
        """Configure HardenedOrchestrator from spec and config."""
        from aragora.nomic.hardened_orchestrator import HardenedOrchestrator

        orchestrator = HardenedOrchestrator(
            require_human_approval=spec.requires_approval or self.config.require_approval,
            budget_limit_usd=spec.budget_limit_usd or self.config.budget_limit_usd,
            use_worktree_isolation=self.config.use_worktree_isolation,
            enable_gauntlet_validation=self.config.enable_gauntlet_validation,
            enable_mode_enforcement=self.config.enable_mode_enforcement,
            enable_meta_planning=self.config.enable_meta_planning,
            generate_receipts=self.config.generate_receipts,
            spectate_stream=self.config.spectate_stream,
            max_parallel_tasks=self.config.max_parallel_tasks,
        )

        # Post-configure task decomposer if available
        if hasattr(orchestrator, "task_decomposer"):
            decomposer = orchestrator.task_decomposer
            if hasattr(decomposer, "config") and hasattr(decomposer.config, "max_subtasks"):
                decomposer.config.max_subtasks = self.config.max_subtasks

        return orchestrator

    @property
    def spec(self) -> SwarmSpec | None:
        """The spec from the last run."""
        return self._spec

    @property
    def result(self) -> Any:
        """The orchestration result from the last run."""
        return self._result


class _ErrorResult:
    """Minimal result object for when dispatch fails entirely."""

    def __init__(self, error: str) -> None:
        self.error = error
        self.total_subtasks = 0
        self.completed_subtasks = 0
        self.failed_subtasks = 1
        self.skipped_subtasks = 0
        self.assignments: list[Any] = []
        self.total_cost_usd = 0.0
