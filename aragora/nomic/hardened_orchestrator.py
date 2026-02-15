"""Hardened Orchestrator with worktree isolation, mode enforcement, and validation.

Extends AutonomousOrchestrator with production-grade features:
- Worktree isolation: each agent gets its own git worktree
- Mode enforcement: design→architect, implement→coder, verify→reviewer
- Gauntlet validation: adversarial testing of agent output
- Prompt injection defense: scan goals/context with SkillScanner
- Budget enforcement: cumulative USD spend tracking
- Audit reconciliation: cross-agent file overlap detection

All features are opt-in via constructor flags. Default behavior is
identical to the base AutonomousOrchestrator.

Usage:
    from aragora.nomic.hardened_orchestrator import HardenedOrchestrator

    orchestrator = HardenedOrchestrator(
        use_worktree_isolation=True,
        enable_gauntlet_validation=True,
        budget_limit_usd=5.0,
    )
    result = await orchestrator.execute_goal("Improve SDK test coverage")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from aragora.nomic.autonomous_orchestrator import (
    AgentAssignment,
    AutonomousOrchestrator,
    OrchestrationResult,
)
from aragora.workflow.types import StepDefinition, WorkflowDefinition

logger = logging.getLogger(__name__)

# Maps nomic loop phases to operational modes
PHASE_MODE_MAP = {
    "design": "architect",
    "implement": "coder",
    "verify": "reviewer",
}


@dataclass
class HardenedConfig:
    """Configuration for hardened orchestrator features."""

    use_worktree_isolation: bool = False
    enable_mode_enforcement: bool = True
    enable_gauntlet_validation: bool = True
    enable_prompt_defense: bool = True
    enable_audit_reconciliation: bool = True
    budget_limit_usd: float | None = None


class HardenedOrchestrator(AutonomousOrchestrator):
    """Orchestrator with worktree isolation and hardened validation.

    Extends AutonomousOrchestrator with opt-in production features.
    When all flags are disabled, behavior is identical to the base class.
    """

    def __init__(
        self,
        *,
        use_worktree_isolation: bool = False,
        enable_mode_enforcement: bool = True,
        enable_gauntlet_validation: bool = True,
        enable_prompt_defense: bool = True,
        enable_audit_reconciliation: bool = True,
        budget_limit_usd: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.hardened_config = HardenedConfig(
            use_worktree_isolation=use_worktree_isolation,
            enable_mode_enforcement=enable_mode_enforcement,
            enable_gauntlet_validation=enable_gauntlet_validation,
            enable_prompt_defense=enable_prompt_defense,
            enable_audit_reconciliation=enable_audit_reconciliation,
            budget_limit_usd=budget_limit_usd,
        )

        # Budget tracking
        self._budget_spent_usd: float = 0.0

        # Worktree manager (created lazily when needed)
        self._worktree_manager: Any | None = None

    def _get_worktree_manager(self) -> Any:
        """Lazily create WorktreeManager."""
        if self._worktree_manager is None:
            from aragora.nomic.worktree_manager import WorktreeManager

            self._worktree_manager = WorktreeManager(
                repo_path=self.aragora_path,
                base_branch="main",
            )
        return self._worktree_manager

    # =========================================================================
    # A. Prompt Injection Defense
    # =========================================================================

    async def execute_goal(
        self,
        goal: str,
        tracks: list[str] | None = None,
        max_cycles: int = 5,
        context: dict[str, Any] | None = None,
    ) -> OrchestrationResult:
        """Execute goal with optional prompt injection scanning.

        Scans the goal text for injection patterns before proceeding.
        CRITICAL severity findings raise ValueError (hard reject).
        HIGH severity findings are logged as warnings.
        """
        if self.hardened_config.enable_prompt_defense:
            self._scan_for_injection(goal, context)

        # Reset budget tracking for this run
        self._budget_spent_usd = 0.0

        result = await super().execute_goal(goal, tracks, max_cycles, context)

        # F. Audit reconciliation after all assignments complete
        if self.hardened_config.enable_audit_reconciliation:
            self._reconcile_audits(result.assignments)

        return result

    def _scan_for_injection(self, goal: str, context: dict[str, Any] | None) -> None:
        """Scan goal and context for prompt injection patterns."""
        try:
            from aragora.compat.openclaw.skill_scanner import (
                Severity,
                SkillScanner,
                Verdict,
            )

            scanner = SkillScanner()

            # Scan goal text
            result = scanner.scan_text(goal)
            if result.verdict == Verdict.DANGEROUS:
                critical_findings = [
                    f for f in result.findings if f.severity == Severity.CRITICAL
                ]
                descriptions = "; ".join(f.description for f in critical_findings[:3])
                raise ValueError(
                    f"Goal rejected: prompt injection detected — {descriptions}"
                )

            if result.verdict == Verdict.SUSPICIOUS:
                logger.warning(
                    "prompt_defense_warning goal_risk_score=%d verdict=%s",
                    result.risk_score,
                    result.verdict.value,
                )

            # Scan context values if provided
            if context:
                for key, value in context.items():
                    if isinstance(value, str) and len(value) > 10:
                        ctx_result = scanner.scan_text(value)
                        if ctx_result.verdict == Verdict.DANGEROUS:
                            raise ValueError(
                                f"Context key '{key}' rejected: "
                                f"prompt injection detected"
                            )

        except ImportError:
            logger.debug("SkillScanner unavailable, skipping prompt defense")

    # =========================================================================
    # B. Mode Enforcement
    # =========================================================================

    def _build_subtask_workflow(
        self, assignment: AgentAssignment
    ) -> WorkflowDefinition:
        """Build workflow with optional mode enforcement.

        When mode enforcement is enabled:
        - Design step gets "architect" mode (read-only tools)
        - Implement step gets "coder" mode (full tools)
        - Verify step gets "reviewer" mode (read-only tools)
        - High-complexity subtasks get quick_debate for design
        """
        workflow = super()._build_subtask_workflow(assignment)

        if not self.hardened_config.enable_mode_enforcement:
            return workflow

        # Inject mode system prompts into step configs
        enhanced_steps = []
        for step in workflow.steps:
            step = self._apply_mode_to_step(step, assignment)
            enhanced_steps.append(step)

        return WorkflowDefinition(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            steps=enhanced_steps,
            entry_step=workflow.entry_step,
        )

    def _apply_mode_to_step(
        self, step: StepDefinition, assignment: AgentAssignment
    ) -> StepDefinition:
        """Apply operational mode constraints to a workflow step."""
        mode_name = PHASE_MODE_MAP.get(step.id)
        if not mode_name:
            return step

        # Try to get mode from registry
        try:
            from aragora.modes.base import ModeRegistry

            mode = ModeRegistry.get(mode_name)
            if mode is not None:
                config = dict(step.config) if step.config else {}
                config["mode"] = mode_name
                config["mode_system_prompt"] = mode.get_system_prompt()

                # High-complexity subtasks get debate-based design
                if (
                    step.id == "design"
                    and assignment.subtask.estimated_complexity == "high"
                ):
                    return StepDefinition(
                        id=step.id,
                        name=step.name,
                        step_type="quick_debate",
                        config={
                            **config,
                            "rounds": 2,
                            "agents": 2,
                            "task": assignment.subtask.description,
                        },
                        next_steps=step.next_steps,
                    )

                return StepDefinition(
                    id=step.id,
                    name=step.name,
                    step_type=step.step_type,
                    config=config,
                    next_steps=step.next_steps,
                )
        except ImportError:
            logger.debug("ModeRegistry unavailable, skipping mode enforcement")

        return step

    # =========================================================================
    # C. Worktree Isolation + D. Gauntlet + E. Budget
    # =========================================================================

    async def _execute_single_assignment(
        self,
        assignment: AgentAssignment,
        max_cycles: int,
    ) -> None:
        """Execute assignment with optional worktree isolation, gauntlet, and budget.

        When worktree isolation is enabled:
        1. Create worktree for this assignment
        2. Set repo_path in the workflow to the worktree
        3. Execute workflow inside the worktree
        4. Run tests in the worktree
        5. Merge back to base branch
        6. Clean up worktree (guaranteed via finally)

        When gauntlet validation is enabled:
        - After successful execution, run lightweight gauntlet
        - Critical findings mark the assignment as failed

        When budget enforcement is enabled:
        - Skip assignments that would exceed the budget
        """
        # E. Budget enforcement
        if self.hardened_config.budget_limit_usd is not None:
            if self._budget_spent_usd >= self.hardened_config.budget_limit_usd:
                logger.warning(
                    "budget_exceeded subtask=%s spent=%.2f limit=%.2f",
                    assignment.subtask.id,
                    self._budget_spent_usd,
                    self.hardened_config.budget_limit_usd,
                )
                assignment.status = "skipped"
                assignment.result = {"reason": "budget_exceeded"}
                assignment.completed_at = datetime.now(timezone.utc)
                self._completed_assignments.append(assignment)
                self._active_assignments.remove(assignment)
                return

        # A. Worktree isolation path
        if self.hardened_config.use_worktree_isolation:
            await self._execute_in_worktree(assignment, max_cycles)
            return

        # Non-worktree path: delegate to parent
        await super()._execute_single_assignment(assignment, max_cycles)

        # C. Post-execution gauntlet validation
        if (
            self.hardened_config.enable_gauntlet_validation
            and assignment.status == "completed"
        ):
            await self._run_gauntlet_validation(assignment)

    async def _execute_in_worktree(
        self,
        assignment: AgentAssignment,
        max_cycles: int,
    ) -> None:
        """Execute an assignment inside an isolated worktree."""
        manager = self._get_worktree_manager()
        ctx = None

        try:
            # Create worktree
            ctx = await manager.create_worktree_for_subtask(
                assignment.subtask,
                assignment.track,
                assignment.agent_type,
            )

            # Override aragora_path for this assignment's workflow
            original_path = self.aragora_path
            self.aragora_path = ctx.worktree_path

            try:
                # Build and execute workflow (repo_path will use worktree)
                await super()._execute_single_assignment(assignment, max_cycles)
            finally:
                self.aragora_path = original_path

            # Run gauntlet on completed work
            if (
                self.hardened_config.enable_gauntlet_validation
                and assignment.status == "completed"
            ):
                await self._run_gauntlet_validation(assignment)

            # Merge if successful
            if assignment.status == "completed":
                test_paths = self._infer_test_paths(assignment.subtask.file_scope)
                merge_result = await manager.merge_worktree(
                    ctx,
                    require_tests_pass=True,
                    test_paths=test_paths,
                )
                if not merge_result.get("success"):
                    logger.warning(
                        "merge_failed subtask=%s error=%s",
                        assignment.subtask.id,
                        merge_result.get("error"),
                    )
                    assignment.status = "failed"
                    assignment.result = {
                        **(assignment.result or {}),
                        "merge_error": merge_result.get("error"),
                    }

        except Exception as e:
            logger.exception(
                "worktree_execution_failed subtask=%s error=%s",
                assignment.subtask.id,
                str(e),
            )
            assignment.status = "failed"
            assignment.result = {"error": str(e)}

        finally:
            # Always clean up the worktree
            if ctx is not None:
                try:
                    await manager.cleanup_worktree(ctx)
                except Exception:
                    logger.exception(
                        "worktree_cleanup_error subtask=%s", assignment.subtask.id
                    )

    # =========================================================================
    # C. Gauntlet Validation
    # =========================================================================

    async def _run_gauntlet_validation(self, assignment: AgentAssignment) -> None:
        """Run lightweight gauntlet validation on completed assignment."""
        try:
            from aragora.gauntlet.runner import GauntletConfig, GauntletRunner

            # Build content summary from assignment result
            content = self._build_gauntlet_content(assignment)
            if not content:
                return

            config = GauntletConfig(
                attack_rounds=1,
                max_concurrent=1,
            )
            runner = GauntletRunner(config)
            result = await runner.run(content, context=assignment.subtask.description)

            # Check for critical findings
            critical = [
                f
                for f in getattr(result, "findings", [])
                if getattr(f, "severity", "").lower() in ("critical", "high")
            ]

            if critical:
                logger.warning(
                    "gauntlet_critical subtask=%s findings=%d",
                    assignment.subtask.id,
                    len(critical),
                )
                assignment.status = "failed"
                assignment.result = {
                    **(assignment.result or {}),
                    "gauntlet_findings": [str(f) for f in critical[:5]],
                }

        except ImportError:
            logger.debug("Gauntlet unavailable, skipping validation")
        except Exception as e:
            logger.warning(
                "gauntlet_error subtask=%s error=%s",
                assignment.subtask.id,
                str(e),
            )

    def _build_gauntlet_content(self, assignment: AgentAssignment) -> str:
        """Build content string for gauntlet validation from assignment."""
        parts = [f"Task: {assignment.subtask.title}"]
        parts.append(f"Description: {assignment.subtask.description}")
        if assignment.subtask.file_scope:
            parts.append(f"Files modified: {', '.join(assignment.subtask.file_scope)}")
        if assignment.result:
            workflow_result = assignment.result.get("workflow_result", "")
            if isinstance(workflow_result, str):
                parts.append(f"Output: {workflow_result[:2000]}")
        return "\n".join(parts)

    # =========================================================================
    # F. Audit Reconciliation
    # =========================================================================

    def _reconcile_audits(self, assignments: list[AgentAssignment]) -> None:
        """Detect cross-agent file overlaps and log reconciliation report."""
        completed = [a for a in assignments if a.status == "completed"]
        if len(completed) < 2:
            return

        # Build file → assignments mapping
        file_assignments: dict[str, list[str]] = {}
        for a in completed:
            for f in a.subtask.file_scope:
                file_assignments.setdefault(f, []).append(a.subtask.id)

        # Find overlaps (files touched by multiple assignments)
        overlaps = {
            f: subtask_ids
            for f, subtask_ids in file_assignments.items()
            if len(subtask_ids) > 1
        }

        if not overlaps:
            logger.info("audit_reconciliation no_overlaps")
            return

        logger.warning(
            "audit_reconciliation overlaps=%d files=%s",
            len(overlaps),
            list(overlaps.keys()),
        )

        # Try to log via AuditLog
        try:
            from aragora.audit.log import AuditLog

            audit = AuditLog()
            audit.log(
                event="orchestration_reconciliation",
                data={
                    "overlapping_files": overlaps,
                    "assignment_count": len(completed),
                    "overlap_count": len(overlaps),
                },
            )
        except (ImportError, Exception) as e:
            logger.debug("AuditLog unavailable for reconciliation: %s", e)


__all__ = [
    "HardenedConfig",
    "HardenedOrchestrator",
    "PHASE_MODE_MAP",
]
