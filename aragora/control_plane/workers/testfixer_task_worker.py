"""Worker loop for control plane TestFixer tasks."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aragora.control_plane.integration import ControlPlaneIntegration
from aragora.control_plane.testfixer import TESTFIXER_TASK_TYPE, TestFixerControlPlane

logger = logging.getLogger(__name__)


class TestFixerTaskWorker:
    """Worker that claims TestFixer tasks from the control plane scheduler."""

    def __init__(self, integration: ControlPlaneIntegration, worker_id: str = "testfixer-worker"):
        self._integration = integration
        self._worker_id = worker_id
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("TestFixer task worker started")
        while self._running:
            task = await self._integration._coordinator._scheduler_bridge.claim_task(
                agent_id=self._worker_id,
                capabilities=["testfixer"],
                block_ms=2000,
            )
            if not task:
                await asyncio.sleep(0.5)
                continue
            if task.task_type != TESTFIXER_TASK_TYPE:
                # Release non-testfixer tasks
                await self._integration._coordinator._scheduler_bridge.fail_task(
                    task.id,
                    "Unsupported task type",
                    agent_id=self._worker_id,
                    requeue=True,
                )
                continue
            await self._handle_task(task)

    async def stop(self) -> None:
        self._running = False

    async def _handle_task(self, task: Any) -> None:
        handler = TestFixerControlPlane(self._integration)
        try:
            result = await handler.execute(task.payload)
            await self._integration._coordinator._scheduler_bridge.complete_task(
                task.id, result=result, agent_id=self._worker_id
            )
        except Exception as exc:
            logger.error(f"TestFixer task {task.id} failed: {exc}")
            await self._integration._coordinator._scheduler_bridge.fail_task(
                task.id, error=str(exc), agent_id=self._worker_id
            )
