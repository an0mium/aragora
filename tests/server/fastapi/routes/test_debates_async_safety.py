from __future__ import annotations

import asyncio
import os
import time

import pytest

os.environ.setdefault("ARAGORA_USE_SECRETS_MANAGER", "0")

from aragora.server.fastapi.routes.debates import list_debates


class _SlowSyncStorage:
    def list_debates(self, **kwargs):
        time.sleep(0.2)
        return []

    def count_debates(self, status: str | None = None) -> int:
        time.sleep(0.2)
        return 0


@pytest.mark.asyncio
async def test_list_debates_does_not_block_event_loop_for_sync_storage() -> None:
    async def ticker() -> int:
        ticks = 0
        start = time.perf_counter()
        while time.perf_counter() - start < 0.35:
            await asyncio.sleep(0.01)
            ticks += 1
        return ticks

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)

    response = await list_debates(
        request=None,
        limit=10,
        offset=0,
        status=None,
        storage=_SlowSyncStorage(),
    )
    ticks = await task

    assert response.total == 0
    assert response.debates == []
    assert ticks >= 10
