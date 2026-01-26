"""Tests for Knowledge Mound CRUD update normalization."""

from __future__ import annotations

from datetime import datetime

import pytest

from aragora.knowledge.mound.api.crud import CRUDOperationsMixin


class DummyCRUD(CRUDOperationsMixin):
    """Minimal CRUD mixin harness to capture update payloads."""

    def __init__(self) -> None:
        self._cache = None
        self.updated_payload = None

    def _ensure_initialized(self) -> None:
        return

    async def _update_node(self, node_id: str, updates: dict) -> None:
        self.updated_payload = updates

    async def _get_node(self, node_id: str):
        return None


@pytest.mark.asyncio
async def test_update_sets_iso_updated_at():
    """Updated nodes should carry ISO formatted timestamps."""
    crud = DummyCRUD()
    await crud.update("node_123", {"confidence": 0.9})

    assert crud.updated_payload is not None
    updated_at = crud.updated_payload["updated_at"]
    assert isinstance(updated_at, str)
    # Should parse as ISO-8601 without error
    datetime.fromisoformat(updated_at)
