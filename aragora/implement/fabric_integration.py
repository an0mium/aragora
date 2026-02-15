"""
Implementation-Fabric integration for multi-agent execution.

Bridges DecisionPlan implementation tasks to AgentFabric for
heterogeneous, parallel task execution with policy/budget support.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from collections.abc import Iterable
from uuid import uuid4

from aragora.fabric.models import (
    AgentConfig,
    AgentHandle,
    PolicyContext,
    Priority,
    Task,
    TaskHandle,
    Usage,
)

if TYPE_CHECKING:
    from aragora.fabric.fabric import AgentFabric
from aragora.implement.executor import HybridExecutor
from aragora.implement.types import ImplementTask, TaskResult
from aragora.pipeline.decision_plan.core import ImplementationProfile

logger = logging.getLogger(__name__)


@dataclass
class FabricImplementationConfig:
    """Configuration for fabric-managed implementation execution."""

    pool_id: str | None = None
    models: list[str] = field(default_factory=list)
    min_agents: int = 1
    max_agents: int | None = None
    priority: Priority = Priority.NORMAL
    timeout_seconds: float = 1800.0
    org_id: str = ""
    user_id: str = ""
    workspace_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


class FabricImplementationRunner:
    """Runs implementation tasks through AgentFabric."""

    def __init__(
        self,
        fabric: AgentFabric,
        *,
        repo_path: Path,
        implementation_profile: ImplementationProfile | None = None,
    ) -> None:
        self.fabric = fabric
        self.repo_path = repo_path
        self.profile = implementation_profile
        self._executors: dict[str, HybridExecutor] = {}
        self._agent_overrides: dict[str, Any] = {}

    def _create_executor(self) -> HybridExecutor:
        profile = self.profile
        return HybridExecutor(
            repo_path=self.repo_path,
            max_retries=2,
            strategy=profile.strategy if profile else None,
            implementers=profile.implementers if profile else None,
            critic=profile.critic if profile else None,
            reviser=profile.reviser if profile else None,
            max_revisions=profile.max_revisions if profile else None,
            complexity_router=profile.complexity_router if profile else None,
            task_type_router=profile.task_type_router if profile else None,
            capability_router=profile.capability_router if profile else None,
        )

    def _get_executor(self, agent_id: str) -> HybridExecutor:
        executor = self._executors.get(agent_id)
        if executor is None:
            executor = self._create_executor()
            self._executors[agent_id] = executor
        return executor

    def _get_agent_override(
        self, agent_handle: AgentHandle, executor: HybridExecutor
    ) -> tuple[Any | None, str | None]:
        agent_type = agent_handle.config.model
        if not agent_type:
            return None, None
        if agent_handle.agent_id in self._agent_overrides:
            return self._agent_overrides[agent_handle.agent_id], agent_type

        agent = executor._get_dynamic_agent(
            agent_type,
            role="implementer",
            timeout=executor.claude_timeout,
            system_prompt="""You are implementing code changes in a repository.
Be precise, follow existing patterns, and make only necessary changes.
Include proper type hints and docstrings.""",
        )
        if agent is None:
            return None, None
        self._agent_overrides[agent_handle.agent_id] = agent
        return agent, agent_type

    @staticmethod
    def _parse_complexity_router(raw: str) -> dict[str, str]:
        mapping: dict[str, str] = {}
        if not raw:
            return mapping
        for entry in raw.split(","):
            if ":" not in entry:
                continue
            key, value = entry.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key and value:
                mapping[key] = value
        return mapping

    @staticmethod
    def _normalize_router_value(value: Any) -> str | None:
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        if isinstance(value, (list, tuple)):
            for item in value:
                if item:
                    cleaned = str(item).strip()
                    if cleaned:
                        return cleaned
        if value is not None:
            cleaned = str(value).strip()
            return cleaned or None
        return None

    @staticmethod
    def _resolve_task(task: Task | ImplementTask) -> ImplementTask:
        if isinstance(task, ImplementTask):
            return task
        payload = task.payload.get("task") if isinstance(task.payload, dict) else None
        if isinstance(payload, dict):
            return ImplementTask.from_dict(payload)
        if isinstance(task.payload, dict):
            return ImplementTask.from_dict(task.payload)
        raise ValueError("Implementation task payload is missing")

    @staticmethod
    def _estimate_cost(task: ImplementTask) -> float:
        complexity = str(task.complexity or "moderate").lower()
        if complexity == "simple":
            return 0.005
        if complexity == "complex":
            return 0.05
        return 0.02

    async def _execute_task(
        self,
        task: Task | ImplementTask,
        agent_handle: AgentHandle,
        *,
        cost_hint: float | None = None,
    ) -> TaskResult:
        impl_task = self._resolve_task(task)

        executor = self._get_executor(agent_handle.agent_id)
        agent, model_label = self._get_agent_override(agent_handle, executor)

        if agent is None:
            result = await executor.execute_task_with_retry(impl_task)
            if cost_hint and getattr(result, "cost_usd", 0.0) <= 0.0:
                result.cost_usd = cost_hint
            return result

        result = await executor.execute_task(
            impl_task,
            attempt=1,
            use_fallback=False,
            agent_override=agent,
            model_label=model_label or agent_handle.agent_id,
        )

        if result.success and executor._should_review():
            result = await executor._review_and_revise(impl_task, result)

        if cost_hint and getattr(result, "cost_usd", 0.0) <= 0.0:
            result.cost_usd = cost_hint

        return result

    async def run_plan(
        self,
        tasks: Iterable[ImplementTask],
        *,
        config: FabricImplementationConfig,
        on_task_complete: Any | None = None,
    ) -> list[TaskResult]:
        """Schedule and execute implementation tasks via fabric."""
        task_list = list(tasks)
        if not task_list:
            return []

        agent_ids: list[str] = []

        # Use provided pool when available
        if config.pool_id:
            pool = await self.fabric.get_pool(config.pool_id)
            if not pool:
                raise ValueError(f"Pool {config.pool_id} not found")
            if config.max_agents:
                await self.fabric.scale_pool(config.pool_id, config.max_agents)
            agent_ids = list(pool.current_agents)
        else:
            models = config.models or ["claude"]
            target_agents = max(config.min_agents, len(models))
            if config.max_agents is not None:
                target_agents = max(1, min(target_agents, config.max_agents))
            for idx in range(target_agents):
                model = models[idx % len(models)]
                agent_id = f"impl-{uuid4().hex[:8]}-{idx}"
                await self.fabric.spawn(
                    AgentConfig(
                        id=agent_id,
                        model=model,
                        metadata={"role": "implementer"},
                    )
                )
                agent_ids.append(agent_id)

        if not agent_ids:
            raise RuntimeError("No fabric agents available for implementation")

        agent_models: dict[str, str] = {}
        for agent_id in agent_ids:
            agent_handle = await self.fabric.lifecycle.get_agent(agent_id)
            if agent_handle:
                agent_models[agent_id] = agent_handle.config.model

        enforce_policy = False
        try:
            policies = await self.fabric.policy.list_policies()
            enforce_policy = len(policies) > 0
        except (AttributeError, RuntimeError, OSError):
            enforce_policy = False

        complexity_router: dict[str, str] = {}
        if self.profile and self.profile.complexity_router:
            complexity_router = dict(self.profile.complexity_router)
        else:
            import os as _os

            complexity_router = self._parse_complexity_router(
                _os.environ.get("IMPL_AGENT_BY_COMPLEXITY", "")
            )

        task_type_router: dict[str, str] = {}
        if self.profile and self.profile.task_type_router:
            task_type_router = dict(self.profile.task_type_router)
        else:
            import os as _os

            task_type_router = self._parse_complexity_router(
                _os.environ.get("IMPL_AGENT_BY_TASK_TYPE", "")
            )

        capability_router: dict[str, str] = {}
        if self.profile and self.profile.capability_router:
            capability_router = dict(self.profile.capability_router)
        else:
            import os as _os

            capability_router = self._parse_complexity_router(
                _os.environ.get("IMPL_AGENT_BY_CAPABILITY", "")
            )

        rr_index: dict[str, int] = {}
        default_index = 0

        # Schedule tasks with dependencies
        handles: list[TaskHandle] = []
        for idx, task in enumerate(task_list):
            complexity_key = str(task.complexity or "moderate").lower()
            desired_model: str | None = None
            task_type = str(getattr(task, "task_type", "") or "").lower()
            if task_type_router and task_type:
                desired_model = self._normalize_router_value(task_type_router.get(task_type))
            if desired_model is None and capability_router:
                for cap in getattr(task, "capabilities", []) or []:
                    cap_key = str(cap).lower()
                    if cap_key in capability_router:
                        desired_model = self._normalize_router_value(capability_router.get(cap_key))
                        if desired_model:
                            break
            if desired_model is None:
                desired_model = complexity_router.get(complexity_key)
            candidates = agent_ids
            if desired_model:
                candidates = [
                    candidate
                    for candidate in agent_ids
                    if agent_models.get(candidate) == desired_model
                ] or agent_ids

            if desired_model:
                rr_index.setdefault(desired_model, 0)
                agent_id = candidates[rr_index[desired_model] % len(candidates)]
                rr_index[desired_model] += 1
            else:
                agent_id = candidates[default_index % len(candidates)]
                default_index += 1
            fabric_task = Task(
                id=task.id,
                type="implementation",
                payload={"task": task.to_dict()},
                depends_on=list(task.dependencies),
                timeout_seconds=config.timeout_seconds,
                metadata={
                    "org_id": config.org_id,
                    "user_id": config.user_id,
                    "workspace_id": config.workspace_id,
                    **config.metadata,
                },
            )
            task_handle = await self.fabric.schedule(
                fabric_task,
                agent_id=agent_id,
                priority=config.priority,
                depends_on=task.dependencies,
            )
            handles.append(task_handle)

        stop_event = asyncio.Event()
        poll_delay = 0.05

        async def worker(agent_id: str) -> None:
            while not stop_event.is_set():
                next_task = await self.fabric.pop_next_task(agent_id)
                if next_task is None:
                    await asyncio.sleep(poll_delay)
                    continue
                agent_handle = await self.fabric.lifecycle.get_agent(agent_id)
                if not agent_handle:
                    await self.fabric.complete_task(next_task.id, error="Agent missing")
                    continue

                try:
                    impl_task = self._resolve_task(next_task)
                    estimated_cost = self._estimate_cost(impl_task)

                    policy_context = PolicyContext(
                        agent_id=agent_id,
                        user_id=config.user_id or None,
                        tenant_id=config.org_id or None,
                        workspace_id=config.workspace_id or None,
                        action="execute",
                        resource="implementation",
                        attributes={
                            "task_id": impl_task.id,
                            "complexity": impl_task.complexity,
                            "files": len(impl_task.files),
                            "task_type": getattr(impl_task, "task_type", None),
                            "capabilities": getattr(impl_task, "capabilities", []) or [],
                            "requires_approval": getattr(impl_task, "requires_approval", False),
                        },
                    )
                    if enforce_policy:
                        decision = await self.fabric.check_policy(
                            "implementation.execute", policy_context
                        )
                        if not decision.allowed:
                            await self.fabric.complete_task(
                                next_task.id, error=f"Policy denied: {decision.reason}"
                            )
                            continue

                    can_proceed, _status = await self.fabric.check_budget(
                        agent_id,
                        estimated_tokens=0,
                        estimated_cost_usd=estimated_cost,
                    )
                    if not can_proceed:
                        await self.fabric.complete_task(
                            next_task.id, error="Budget exceeded for implementation task"
                        )
                        continue

                    start = time.monotonic()
                    timeout = next_task.timeout_seconds or config.timeout_seconds
                    result = await asyncio.wait_for(
                        self._execute_task(impl_task, agent_handle, cost_hint=estimated_cost),
                        timeout=timeout,
                    )
                    try:
                        await self.fabric.track_usage(
                            Usage(
                                agent_id=agent_id,
                                tokens_input=0,
                                tokens_output=0,
                                compute_seconds=result.duration_seconds,
                                cost_usd=result.cost_usd,
                                model=result.model_used or agent_handle.config.model,
                                task_id=next_task.id,
                            )
                        )
                    except (ImportError, AttributeError, RuntimeError):
                        logger.debug(
                            "Failed to track usage for task %s", next_task.id, exc_info=True
                        )
                    await self.fabric.complete_task(next_task.id, result=result)
                    if on_task_complete:
                        on_task_complete(next_task.id, result)
                    elapsed = time.monotonic() - start
                    logger.debug("Fabric task %s completed in %.2fs", next_task.id, elapsed)
                except asyncio.TimeoutError:
                    await self.fabric.complete_task(
                        next_task.id,
                        error=f"Timeout after {next_task.timeout_seconds or config.timeout_seconds}s",
                    )
                except Exception as exc:
                    await self.fabric.complete_task(next_task.id, error=str(exc))

        worker_tasks = [asyncio.create_task(worker(agent_id)) for agent_id in agent_ids]

        try:
            overall_timeout = config.timeout_seconds * max(1, len(handles))
            deadline = time.monotonic() + overall_timeout
            pending = {h.task_id for h in handles}
            while pending and time.monotonic() < deadline:
                for task_id in list(pending):
                    th = await self.fabric.get_task(task_id)
                    if th and th.status.name.lower() in {
                        "completed",
                        "failed",
                        "cancelled",
                        "timeout",
                    }:
                        pending.discard(task_id)
                if pending:
                    await asyncio.sleep(0.05)
            if pending:
                logger.warning(
                    "Fabric implementation run timed out after %.1fs (%d pending)",
                    overall_timeout,
                    len(pending),
                )
        finally:
            stop_event.set()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

        results: list[TaskResult] = []
        for scheduled_handle in handles:
            final_handle = await self.fabric.get_task(scheduled_handle.task_id)
            if final_handle and isinstance(final_handle.result, TaskResult):
                results.append(final_handle.result)
            elif final_handle and final_handle.error:
                results.append(
                    TaskResult(
                        task_id=scheduled_handle.task_id,
                        success=False,
                        error=final_handle.error,
                    )
                )
            else:
                results.append(
                    TaskResult(
                        task_id=scheduled_handle.task_id,
                        success=False,
                        error="Unknown fabric execution failure",
                    )
                )

        return results


async def register_implementation_executor(fabric: AgentFabric) -> None:
    """Register the implementation executor with fabric (optional)."""

    runner = FabricImplementationRunner(
        fabric,
        repo_path=Path.cwd(),
        implementation_profile=None,
    )

    async def execute_implementation(task: Task, agent_handle: AgentHandle) -> TaskResult:
        return await runner._execute_task(task, agent_handle)

    fabric.register_executor("implementation", execute_implementation)
    logger.info("Registered implementation executor with Agent Fabric")


__all__ = [
    "FabricImplementationConfig",
    "FabricImplementationRunner",
    "register_implementation_executor",
]
