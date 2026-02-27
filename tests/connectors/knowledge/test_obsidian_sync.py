"""Tests for bidirectional Obsidian sync with conflict detection."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.connectors.knowledge.obsidian_sync import (
    BidirectionalSyncManager,
    ConflictResolution,
    SyncConflict,
    SyncDirection,
    SyncRecord,
)


@pytest.fixture
def connector() -> MagicMock:
    mock = MagicMock()
    mock.update_note_frontmatter = AsyncMock(return_value=MagicMock())
    mock.get_note = MagicMock()
    return mock


@pytest.fixture
def adapter() -> MagicMock:
    mock = MagicMock()
    mock.sync_to_km = AsyncMock()
    mock.sync_from_km = AsyncMock()
    return mock


@pytest.fixture
def sync_manager(connector: MagicMock, adapter: MagicMock) -> BidirectionalSyncManager:
    return BidirectionalSyncManager(connector=connector, adapter=adapter)


class TestConflictDetection:
    def test_no_conflict_when_only_vault_changed(
        self, sync_manager: BidirectionalSyncManager
    ) -> None:
        record = SyncRecord(
            note_path="test.md",
            vault_modified=datetime.now(),
            km_modified=datetime.now() - timedelta(hours=1),
            last_synced=datetime.now() - timedelta(minutes=30),
        )
        conflict = sync_manager.detect_conflict(record)
        assert conflict is None

    def test_no_conflict_when_only_km_changed(self, sync_manager: BidirectionalSyncManager) -> None:
        record = SyncRecord(
            note_path="test.md",
            vault_modified=datetime.now() - timedelta(hours=1),
            km_modified=datetime.now(),
            last_synced=datetime.now() - timedelta(minutes=30),
        )
        conflict = sync_manager.detect_conflict(record)
        assert conflict is None

    def test_conflict_when_both_changed(self, sync_manager: BidirectionalSyncManager) -> None:
        record = SyncRecord(
            note_path="test.md",
            vault_modified=datetime.now(),
            km_modified=datetime.now(),
            last_synced=datetime.now() - timedelta(hours=1),
        )
        conflict = sync_manager.detect_conflict(record)
        assert conflict is not None
        assert isinstance(conflict, SyncConflict)

    def test_default_resolution_prefers_vault_content(
        self, sync_manager: BidirectionalSyncManager
    ) -> None:
        conflict = SyncConflict(
            note_path="test.md",
            vault_modified=datetime.now(),
            km_modified=datetime.now(),
        )
        resolution = sync_manager.resolve_conflict(conflict)
        assert resolution == ConflictResolution.PREFER_VAULT_CONTENT

    @pytest.mark.asyncio
    async def test_sync_forward_calls_adapter(
        self,
        sync_manager: BidirectionalSyncManager,
        adapter: MagicMock,
    ) -> None:
        await sync_manager.sync_forward()
        adapter.sync_to_km.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_reverse_calls_adapter(
        self,
        sync_manager: BidirectionalSyncManager,
        adapter: MagicMock,
    ) -> None:
        await sync_manager.sync_reverse()
        adapter.sync_from_km.assert_called_once()
