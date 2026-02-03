"""Execution progress notifier for decision implementation.

Provides callback factories that push task-level progress updates to
originating channels during multi-agent plan execution.  Integrates
with both the PlanExecutor (workflow-based) and HybridExecutor (direct).

Usage:
    notifier = ExecutionNotifier(debate_id="d-123")
    callback = notifier.on_task_complete
    results = await executor.execute_plan(tasks, set(), on_task_complete=callback)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ExecutionProgress:
    """Snapshot of execution progress for channel delivery."""

    debate_id: str
    plan_id: str | None = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    current_task_id: str | None = None
    current_task_description: str | None = None
    elapsed_seconds: float = 0.0
    task_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks + self.failed_tasks) / self.total_tasks * 100

    @property
    def is_complete(self) -> bool:
        return (self.completed_tasks + self.failed_tasks) >= self.total_tasks

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "plan_id": self.plan_id,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "current_task_id": self.current_task_id,
            "current_task_description": self.current_task_description,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "progress_pct": round(self.progress_pct, 1),
            "is_complete": self.is_complete,
        }


class ExecutionNotifier:
    """Pushes execution progress to originating channels.

    Create one per execution, then pass ``on_task_complete`` as the
    callback to ``HybridExecutor.execute_plan`` or
    ``PlanExecutor.execute``.
    """

    def __init__(
        self,
        debate_id: str,
        plan_id: str | None = None,
        total_tasks: int = 0,
        notify_channel: bool = True,
        notify_websocket: bool = True,
    ) -> None:
        self.progress = ExecutionProgress(
            debate_id=debate_id,
            plan_id=plan_id,
            total_tasks=total_tasks,
        )
        self._notify_channel = notify_channel
        self._notify_websocket = notify_websocket
        self._start_time = time.monotonic()
        self._task_descriptions: dict[str, str] = {}
        self._listeners: list[Callable[[ExecutionProgress], Any]] = []

    def set_task_descriptions(self, tasks: list[Any]) -> None:
        """Pre-populate task descriptions from ImplementTask list."""
        for task in tasks:
            tid = getattr(task, "id", None) or str(task)
            desc = getattr(task, "description", "") or ""
            self._task_descriptions[tid] = desc
        self.progress.total_tasks = len(tasks)

    def add_listener(self, fn: Callable[[ExecutionProgress], Any]) -> None:
        """Register an additional listener for progress events."""
        self._listeners.append(fn)

    def on_task_complete(self, task_id: str, result: Any) -> None:
        """Callback for HybridExecutor.execute_plan's on_task_complete.

        Matches the signature ``(task_id: str, result: TaskResult) -> None``.
        """
        self.progress.elapsed_seconds = time.monotonic() - self._start_time
        success = getattr(result, "success", True)

        if success:
            self.progress.completed_tasks += 1
        else:
            self.progress.failed_tasks += 1

        self.progress.current_task_id = task_id
        self.progress.current_task_description = self._task_descriptions.get(task_id)
        self.progress.task_results.append(
            {
                "task_id": task_id,
                "success": success,
                "model_used": getattr(result, "model_used", None),
                "duration_seconds": getattr(result, "duration_seconds", 0.0),
                "error": getattr(result, "error", None),
            }
        )

        # Fire-and-forget channel notification
        self._dispatch_progress()

    def _dispatch_progress(self) -> None:
        """Send progress update to channels and WebSocket."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            loop.create_task(self._async_dispatch())
        else:
            # Fallback: log progress for non-async contexts
            logger.info(
                "Execution progress [%s]: %d/%d tasks (%.0f%%)",
                self.progress.debate_id,
                self.progress.completed_tasks + self.progress.failed_tasks,
                self.progress.total_tasks,
                self.progress.progress_pct,
            )

    async def _async_dispatch(self) -> None:
        """Send progress to channel and WebSocket asynchronously."""
        progress_dict = self.progress.to_dict()
        event_payload = {
            "debate_id": self.progress.debate_id,
            "event": "execution_progress",
            "progress": progress_dict,
        }

        # 1. Route to originating channel (Slack/Teams/Discord/etc.)
        if self._notify_channel:
            try:
                from aragora.server.result_router import route_result

                await route_result(self.progress.debate_id, event_payload)
            except Exception as exc:
                logger.debug("Channel progress notification failed: %s", exc)

        # 2. Broadcast to WebSocket subscribers
        if self._notify_websocket:
            try:
                from aragora.server.stream.broadcast import broadcast_event

                await broadcast_event(
                    "execution_progress",
                    progress_dict,
                    debate_id=self.progress.debate_id,
                )
            except Exception as exc:
                logger.debug("WebSocket progress broadcast failed: %s", exc)

        # 3. Custom listeners
        for listener in self._listeners:
            try:
                result = listener(self.progress)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.debug("Progress listener failed: %s", exc)

    async def send_completion_summary(self) -> None:
        """Send a final summary when execution completes."""
        self.progress.elapsed_seconds = time.monotonic() - self._start_time
        summary_payload = {
            "debate_id": self.progress.debate_id,
            "event": "execution_complete",
            "summary": {
                **self.progress.to_dict(),
                "task_results": self.progress.task_results,
            },
        }

        if self._notify_channel:
            try:
                from aragora.server.result_router import route_result

                await route_result(self.progress.debate_id, summary_payload)
            except Exception as exc:
                logger.debug("Channel completion notification failed: %s", exc)

        if self._notify_websocket:
            try:
                from aragora.server.stream.broadcast import broadcast_event

                await broadcast_event(
                    "execution_complete",
                    summary_payload.get("summary", {}),
                    debate_id=self.progress.debate_id,
                )
            except Exception as exc:
                logger.debug("WebSocket completion broadcast failed: %s", exc)
