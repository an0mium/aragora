from __future__ import annotations

import os

import pytest

os.environ.setdefault("ARAGORA_USE_SECRETS_MANAGER", "0")

from aragora.server.fastapi.routes.debates import get_debate, list_debates


class _ReceiptAwareStorage:
    def __init__(self) -> None:
        self._debate = {
            "id": "debate-1",
            "task": "Ship receipt links",
            "status": "completed",
            "result": {"receipt_id": "rcpt-123"},
            "metadata": {"result": {"receipt_id": "rcpt-123"}},
        }

    def get_debate(self, debate_id: str) -> dict[str, object]:
        return {**self._debate, "id": debate_id}

    def list_debates(self, **kwargs) -> list[dict[str, object]]:
        return [self._debate]

    def count_debates(self, status: str | None = None) -> int:
        return 1


@pytest.mark.asyncio
async def test_get_debate_surfaces_receipt_id_from_persisted_result() -> None:
    response = await get_debate("debate-1", storage=_ReceiptAwareStorage())

    assert response.receipt_id == "rcpt-123"


@pytest.mark.asyncio
async def test_list_debates_surfaces_receipt_id_from_persisted_result() -> None:
    response = await list_debates(
        request=None,
        limit=10,
        offset=0,
        status=None,
        storage=_ReceiptAwareStorage(),
    )

    assert response.debates[0].receipt_id == "rcpt-123"
