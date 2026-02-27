"""Tests for ObsidianAdapter live sync via VaultWatcher events."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.connectors.knowledge.obsidian_watcher import VaultChangeEvent
from aragora.knowledge.mound.adapters.obsidian_adapter import ObsidianAdapter


def _make_adapter() -> ObsidianAdapter:
    """Create an ObsidianAdapter with a stubbed connector."""
    connector = MagicMock()
    connector.is_configured = True
    connector._config = MagicMock()
    connector._config.watch_tags = []
    return ObsidianAdapter(connector=connector, workspace_id="test-ws")


@pytest.mark.asyncio
async def test_watcher_change_triggers_sync() -> None:
    """A 'modified' VaultChangeEvent should trigger sync_to_km."""
    adapter = _make_adapter()
    adapter.sync_to_km = AsyncMock()  # type: ignore[method-assign]

    event = VaultChangeEvent(
        path="/vault/notes/test.md",
        change_type="modified",
        timestamp=1700000000.0,
    )

    await adapter._on_vault_change(event)

    adapter.sync_to_km.assert_called_once()


@pytest.mark.asyncio
async def test_created_file_triggers_sync() -> None:
    """A 'created' VaultChangeEvent should also trigger sync_to_km."""
    adapter = _make_adapter()
    adapter.sync_to_km = AsyncMock()  # type: ignore[method-assign]

    event = VaultChangeEvent(
        path="/vault/notes/new_note.md",
        change_type="created",
        timestamp=1700000000.0,
    )

    await adapter._on_vault_change(event)

    adapter.sync_to_km.assert_called_once()


@pytest.mark.asyncio
async def test_deleted_file_skips_sync() -> None:
    """A 'deleted' VaultChangeEvent should NOT trigger sync_to_km."""
    adapter = _make_adapter()
    adapter.sync_to_km = AsyncMock()  # type: ignore[method-assign]

    event = VaultChangeEvent(
        path="/vault/notes/removed.md",
        change_type="deleted",
        timestamp=1700000000.0,
    )

    await adapter._on_vault_change(event)

    adapter.sync_to_km.assert_not_called()
