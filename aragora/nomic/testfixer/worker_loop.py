"""Standalone worker loop for TestFixer (control-plane or local)."""

from __future__ import annotations

import asyncio
import logging

from aragora.control_plane.integration import setup_control_plane_integration
from aragora.control_plane.workers.testfixer_task_worker import TestFixerTaskWorker

logger = logging.getLogger(__name__)


async def start_testfixer_worker() -> TestFixerTaskWorker:
    integration = await setup_control_plane_integration()
    worker = TestFixerTaskWorker(integration)
    asyncio.create_task(worker.start())
    logger.info("TestFixer control-plane worker started")
    return worker
