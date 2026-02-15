"""
Parallel Orchestrator - Production-grade multi-agent execution.

Composes BranchCoordinator (worktree isolation) with AutonomousOrchestrator
(parallel execution, gauntlet gate, convoy tracking) to implement the
Planner/Worker/Judge hierarchy:

- **Planner**: TaskDecomposer.analyze_with_debate() + DecisionPlanFactory
- **Workers**: HybridExecutor running in isolated git worktrees
- **Judge**: GauntletRunner via workflow pipeline

Usage:
    from aragora.nomic.parallel_orchestrator import ParallelOrchestrator

    orchestrator = ParallelOrchestrator()
    result = await orchestrator.execute_goal(
        goal="Improve error handling across connectors",
        tracks=["developer", "qa"],
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from collections.abc import Callable

from aragora.nomic.autonomous_orchestrator import (
    AutonomousOrchestrator,
    HierarchyConfig,
    OrchestrationResult,
    Track,
    TrackConfig,
)
from aragora.nomic.branch_coordinator import (
    BranchCoordinator,
    BranchCoordinatorConfig,
)
from aragora.nomic.task_decomposer import TaskDecomposer
from aragora.workflow.engine import WorkflowEngine

logger = logging.getLogger(__name__)


class ParallelOrchestrator:
    """
    Entry point for production-grade parallel multi-agent execution.

    Composes worktree isolation, gauntlet validation, decision planning,
    and convoy tracking into a single orchestrator. Each feature can be
    toggled independently.
    """

    def __init__(
        self,
        aragora_path: Path | None = None,
        *,
        # Worktree isolation
        use_worktrees: bool = True,
        worktrees_base: str = ".worktrees",
        # Gauntlet red-team gate
        enable_gauntlet: bool = True,
        # Decision plan (risk-aware workflow from debate results)
        use_decision_plan: bool = False,
        # Convoy/bead crash-resilient tracking
        enable_convoy_tracking: bool = False,
        # Planner/Worker/Judge hierarchy
        hierarchy: HierarchyConfig | None = None,
        # Standard orchestrator options
        track_configs: dict[Track, TrackConfig] | None = None,
        workflow_engine: WorkflowEngine | None = None,
        task_decomposer: TaskDecomposer | None = None,
        require_human_approval: bool = False,
        max_parallel_tasks: int = 4,
        on_checkpoint: Callable[[str, dict[str, Any]], None] | None = None,
        use_debate_decomposition: bool = False,
        enable_curriculum: bool = True,
        budget_limit_usd: float | None = None,
    ):
        self.aragora_path = aragora_path or Path.cwd()
        self._budget_limit_usd = budget_limit_usd

        # Build BranchCoordinator for worktree isolation
        branch_coordinator: BranchCoordinator | None = None
        if use_worktrees:
            bc_config = BranchCoordinatorConfig(
                use_worktrees=True,
                worktree_base_dir=worktrees_base,
            )
            branch_coordinator = BranchCoordinator(
                repo_path=str(self.aragora_path),
                config=bc_config,
            )
            logger.info(
                "Worktree isolation enabled (base: %s)",
                worktrees_base,
            )

        # Build workspace manager for convoy tracking
        workspace_manager: Any = None
        if enable_convoy_tracking:
            try:
                from aragora.workspace.manager import WorkspaceManager

                workspace_manager = WorkspaceManager(
                    workspace_root=self.aragora_path,
                )
                logger.info("Convoy/bead tracking enabled")
            except ImportError:
                logger.warning(
                    "Convoy tracking requested but workspace module unavailable"
                )

        # Compose the AutonomousOrchestrator with all wiring
        self._orchestrator = AutonomousOrchestrator(
            aragora_path=self.aragora_path,
            track_configs=track_configs,
            workflow_engine=workflow_engine,
            task_decomposer=task_decomposer,
            require_human_approval=require_human_approval,
            max_parallel_tasks=max_parallel_tasks,
            on_checkpoint=on_checkpoint,
            use_debate_decomposition=use_debate_decomposition,
            enable_curriculum=enable_curriculum,
            branch_coordinator=branch_coordinator,
            hierarchy=hierarchy,
            enable_gauntlet_gate=enable_gauntlet,
            use_decision_plan=use_decision_plan,
            enable_convoy_tracking=enable_convoy_tracking,
            workspace_manager=workspace_manager,
        )

        self._branch_coordinator = branch_coordinator
        self._use_worktrees = use_worktrees

    async def execute_goal(
        self,
        goal: str,
        tracks: list[str] | None = None,
        max_cycles: int = 5,
    ) -> OrchestrationResult:
        """
        Execute a high-level goal with full parallel orchestration.

        Delegates to AutonomousOrchestrator.execute_goal() which handles
        branch creation, semaphore enforcement, gauntlet gating, decision
        planning, convoy tracking, and worktree cleanup internally.

        Args:
            goal: High-level development goal.
            tracks: Optional list of track names to constrain.
            max_cycles: Maximum improvement cycles per subtask.

        Returns:
            OrchestrationResult with completion stats and assignment details.
        """
        logger.info(
            "ParallelOrchestrator starting goal: %s (worktrees=%s)",
            goal,
            self._use_worktrees,
        )

        return await self._orchestrator.execute_goal(
            goal=goal,
            tracks=tracks,
            max_cycles=max_cycles,
        )

    async def cleanup(self) -> None:
        """Clean up worktrees and resources after execution."""
        if self._branch_coordinator:
            try:
                self._branch_coordinator.cleanup_all_worktrees()
                logger.info("Worktrees cleaned up")
            except Exception:
                logger.warning("Failed to clean up worktrees", exc_info=True)
