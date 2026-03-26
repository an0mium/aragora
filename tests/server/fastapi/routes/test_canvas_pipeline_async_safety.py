from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("ARAGORA_USE_SECRETS_MANAGER", "0")

from aragora.server.fastapi.routes import canvas_pipeline
from aragora.server.fastapi.routes.canvas_pipeline import (
    SaveCanvasRequest,
    get_pipeline,
    save_canvas_state,
)


class _SlowLoadStore:
    def load(self, pipeline_id: str):
        time.sleep(0.3)
        return {
            "pipeline_id": pipeline_id,
            "stage_status": {"ideas": "complete"},
        }


class _SlowSaveStore:
    def load(self, pipeline_id: str):
        time.sleep(0.2)
        return {"pipeline_id": pipeline_id}

    def save(self, pipeline_id: str, data: dict):
        time.sleep(0.2)
        return None


async def _ticker(duration: float) -> int:
    ticks = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        await asyncio.sleep(0.01)
        ticks += 1
    return ticks


@pytest.mark.asyncio
async def test_get_pipeline_does_not_block_event_loop_for_sync_store() -> None:
    canvas_pipeline._pipeline_objects.clear()
    task = asyncio.create_task(_ticker(0.4))
    await asyncio.sleep(0)

    with patch.object(canvas_pipeline, "_get_store", return_value=_SlowLoadStore()):
        response = await get_pipeline("pipe-123")
    ticks = await task

    assert response.pipeline_id == "pipe-123"
    assert ticks >= 10


@pytest.mark.asyncio
async def test_save_canvas_state_does_not_block_event_loop_for_sync_store() -> None:
    task = asyncio.create_task(_ticker(0.45))
    await asyncio.sleep(0)

    with patch.object(canvas_pipeline, "_get_store", return_value=_SlowSaveStore()):
        response = await save_canvas_state(
            "pipe-123",
            SaveCanvasRequest(stage="ideas", canvas_data={"nodes": []}),
            auth=SimpleNamespace(user_id="user-1"),
        )
    ticks = await task

    assert response == {"saved": True, "pipeline_id": "pipe-123"}
    assert ticks >= 10
