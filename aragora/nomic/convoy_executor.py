"""
Gastown-style convoy executor for Nomic ImplementPhase.

Bridges ImplementPlan tasks to Beads + Convoys + HookQueues, then
executes tasks with Claude/Codex while using additional agents for
cross-check review.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aragora.implement.executor import HybridExecutor
from aragora.implement.types import ImplementTask, TaskResult
from aragora.nomic.agent_roles import AgentHierarchy, AgentRole
from aragora.nomic.beads import Bead, BeadPriority, BeadStatus, BeadStore, BeadType
from aragora.nomic.convoy_coordinator import AssignmentStatus, ConvoyCoordinator
from aragora.nomic.convoys import ConvoyManager, ConvoyPriority
from aragora.nomic.hook_queue import HookQueue

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    approved: bool
    notes: str


class GastownConvoyExecutor:
    """Execute ImplementPlan tasks via Gastown-style convoys and beads."""

    def __init__(
        self,
        repo_path: Path,
        implementers: List[Any],
        reviewers: Optional[List[Any]] = None,
        bead_dir: Optional[Path] = None,
        convoy_dir: Optional[Path] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        stream_emit_fn: Optional[Callable[..., None]] = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.implementers = [a for a in implementers if a is not None]
        self.reviewers = [a for a in (reviewers or []) if a is not None]
        self._log = log_fn or (lambda msg: logger.info(msg))
        self._stream_emit = stream_emit_fn or (lambda *args, **kwargs: None)

        base_dir = bead_dir or (self.repo_path / ".nomic" / "convoys")
        self.bead_dir = Path(base_dir)
        self.convoy_dir = Path(convoy_dir or base_dir)

        self.bead_store = BeadStore(bead_dir=self.bead_dir, git_enabled=True, auto_commit=False)
        self.convoy_manager = ConvoyManager(bead_store=self.bead_store, convoy_dir=self.convoy_dir)
        self.hierarchy = AgentHierarchy(self.repo_path / ".nomic" / "agents")
        self.coordinator = ConvoyCoordinator(
            convoy_manager=self.convoy_manager,
            hierarchy=self.hierarchy,
            bead_store=self.bead_store,
        )
        self._executor = HybridExecutor(self.repo_path)
        self._hook_queues: Dict[str, HookQueue] = {}
        self._initialized = False

    async def _register_agents(self) -> None:
        for agent in self.implementers:
            try:
                await self.hierarchy.register_agent(agent.name, AgentRole.CREW)
            except Exception:
                continue
        for agent in self.reviewers:
            if any(a.name == agent.name for a in self.implementers):
                continue
            try:
                await self.hierarchy.register_agent(agent.name, AgentRole.WITNESS)
            except Exception:
                continue

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        await self.bead_store.initialize()
        await self.convoy_manager.initialize()
        await self.hierarchy.initialize()
        await self._register_agents()
        await self.coordinator.initialize()
        for agent in self.implementers:
            queue = HookQueue(
                agent_id=getattr(agent, "name", str(agent)), bead_store=self.bead_store
            )
            await queue.initialize()
            self._hook_queues[queue.agent_id] = queue
        self._initialized = True

    async def execute_plan(
        self,
        tasks: List[ImplementTask],
        completed: set[str],
        on_task_complete=None,
    ) -> List[TaskResult]:
        await self._ensure_initialized()
        if not self.implementers:
            raise RuntimeError("No implementer agents available for convoy execution")

        # Create beads for tasks
        convoy_id = f"nomic-{uuid.uuid4().hex[:8]}"
        bead_ids: Dict[str, str] = {}
        for task in tasks:
            bead = Bead.create(
                bead_type=BeadType.TASK,
                title=task.id,
                description=task.description,
                priority=BeadPriority.NORMAL,
                metadata={"task_id": task.id, "files": task.files, "complexity": task.complexity},
            )
            bead.id = f"{convoy_id}-{task.id}"
            # Preserve dependencies using bead IDs
            bead.dependencies = []
            bead_ids[task.id] = bead.id
            try:
                await self.bead_store.create(bead)
            except ValueError:
                # Already exists; ignore
                pass

        # Map dependencies to bead IDs
        for task in tasks:
            bead = await self.bead_store.get(bead_ids[task.id])
            if bead:
                bead.dependencies = [bead_ids[d] for d in task.dependencies if d in bead_ids]
                await self.bead_store.update(bead)

        # Create convoy
        convoy = await self.convoy_manager.create_convoy(
            title="Nomic Implement Plan",
            description=f"{len(tasks)} tasks",
            bead_ids=[bead_ids[t.id] for t in tasks],
            priority=ConvoyPriority.HIGH,
            metadata={"convoy_id": convoy_id},
            convoy_id=convoy_id,
        )

        # Distribute beads to implementers
        assignments = await self.coordinator.distribute_convoy(
            convoy_id=convoy.id,
            agent_ids=[a.name for a in self.implementers],
        )
        if assignments:
            self._log(
                f"  [convoy] Assigned {len(assignments)} beads across {len(self.implementers)} agents"
            )

        # Push to hooks
        for assignment in assignments or []:
            hook = self._hook_queues.get(assignment.agent_id)
            if hook:
                await hook.push(assignment.bead_id, priority=assignment.priority)

        # Execute tasks respecting dependencies
        results: List[TaskResult] = []
        # Iterate tasks in plan order (dependencies already encoded)
        for task in tasks:
            if task.id in completed:
                continue
            if not all(dep in completed for dep in task.dependencies):
                self._log(f"  [convoy] Skipping {task.id} (dependencies unmet)")
                continue

            bead_id = bead_ids[task.id]
            assignment = await self.coordinator.get_assignment(bead_id)
            agent = self._select_agent_for_assignment(assignment.agent_id if assignment else None)

            await self.coordinator.update_assignment_status(bead_id, AssignmentStatus.ACTIVE)
            await self.bead_store.claim(bead_id, agent.name)
            await self.bead_store.update_status(bead_id, BeadStatus.RUNNING)

            result = await self._execute_task_with_agent(task, agent)
            review = await self._review_task(task, agent, result.diff)

            if result.success and review.approved:
                completed.add(task.id)
                await self.bead_store.update_status(bead_id, BeadStatus.COMPLETED)
                await self.coordinator.update_assignment_status(bead_id, AssignmentStatus.COMPLETED)
                hook = self._hook_queues.get(agent.name)
                if hook:
                    await hook.complete(bead_id)
                if on_task_complete:
                    on_task_complete(task.id, result)
            else:
                reason = result.error or review.notes or "review_blocked"
                result = TaskResult(
                    task_id=task.id,
                    success=False,
                    diff=result.diff,
                    error=reason,
                    model_used=result.model_used,
                    duration_seconds=result.duration_seconds,
                )
                await self.bead_store.update_status(bead_id, BeadStatus.FAILED)
                await self.coordinator.update_assignment_status(
                    bead_id, AssignmentStatus.FAILED, error_message=reason
                )
                hook = self._hook_queues.get(agent.name)
                if hook:
                    await hook.fail(bead_id, reason)

            results.append(result)

        return results

    def _select_agent_for_assignment(self, agent_id: Optional[str]) -> Any:
        if agent_id:
            for agent in self.implementers:
                if agent.name == agent_id:
                    return agent
        return self.implementers[0]

    async def _execute_task_with_agent(self, task: ImplementTask, agent: Any) -> TaskResult:
        # Use HybridExecutor to leverage its prompt and timeout logic.
        use_codex = self._is_codex_agent(agent)
        return await self._executor.execute_task(task, attempt=1, use_fallback=use_codex)

    async def _review_task(self, task: ImplementTask, implementer: Any, diff: str) -> ReviewResult:
        if not diff.strip():
            return ReviewResult(approved=True, notes="no_diff")

        reviewers = [r for r in self.reviewers if r.name != implementer.name]
        if not reviewers:
            # Fallback to codex review if configured
            try:
                codex_review = await self._executor.review_with_codex(diff)
                approved = bool(codex_review.get("approved", True))
                notes = "codex_review" if approved else "codex_blocked"
                return ReviewResult(approved=approved, notes=notes)
            except Exception as e:
                return ReviewResult(approved=True, notes=f"review_skipped:{e}")

        prompt = f"""Review the following code changes for correctness and safety.

## Task
{task.id}: {task.description}

## Diff
```
{diff[:10000]}
```

Respond with one of:
- APPROVE: <short reason>
- BLOCKER: <short reason>

Be concise. If unsure, choose APPROVE and note uncertainty.
"""

        async def review_with(agent: Any) -> str:
            try:
                return await agent.generate(prompt, context=[])
            except Exception as e:
                return f"BLOCKER: review failed ({e})"

        review_tasks = [review_with(r) for r in reviewers[:2]]
        responses = await asyncio.gather(*review_tasks, return_exceptions=True)

        blocker_reasons = []
        for resp in responses:
            if isinstance(resp, Exception):
                blocker_reasons.append(str(resp))
                continue
            text = str(resp).strip().lower()
            if text.startswith("blocker"):
                blocker_reasons.append(str(resp))

        if blocker_reasons:
            return ReviewResult(approved=False, notes="; ".join(blocker_reasons)[:500])

        return ReviewResult(approved=True, notes="approved")

    @staticmethod
    def _is_codex_agent(agent: Any) -> bool:
        # Works with AirlockProxy by inspecting wrapped_agent
        wrapped = getattr(agent, "wrapped_agent", agent)
        name = getattr(wrapped, "name", "").lower()
        cls_name = wrapped.__class__.__name__.lower()
        return "codex" in name or "codex" in cls_name


__all__ = ["GastownConvoyExecutor"]
