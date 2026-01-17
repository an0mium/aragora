"""
Tests for the directory sync module.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.sync import (
    DirectoryWatcher,
    FileChange,
    FileChangeType,
    SyncConfig,
    SyncManager,
    SyncResult,
    SyncState,
    SyncStatus,
    WatcherConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def populated_dir(temp_dir: Path) -> Path:
    """Create a temporary directory with some files."""
    # Create subdirectory
    subdir = temp_dir / "subdir"
    subdir.mkdir()

    # Create files
    (temp_dir / "file1.txt").write_text("Hello World")
    (temp_dir / "file2.md").write_text("# Markdown\n\nContent here.")
    (subdir / "nested.py").write_text('print("Hello")')

    # Create excluded files
    gitdir = temp_dir / ".git"
    gitdir.mkdir()
    (gitdir / "config").write_text("[core]")

    return temp_dir


@pytest.fixture
def watcher_config() -> WatcherConfig:
    """Create test watcher configuration."""
    return WatcherConfig(
        debounce_ms=100,
        poll_delay_ms=50,
        step_ms=10,
        exclude_patterns=[".git", "__pycache__", "*.pyc"],
    )


@pytest.fixture
def sync_config() -> SyncConfig:
    """Create test sync configuration."""
    return SyncConfig(
        debounce_ms=100,
        exclude_patterns=[".git", "__pycache__"],
        delete_missing=True,
        update_existing=True,
    )


# =============================================================================
# FileChange Tests
# =============================================================================


class TestFileChange:
    """Tests for FileChange model."""

    def test_create_file_change(self):
        """Test creating a FileChange."""
        change = FileChange(
            path="test.txt",
            absolute_path="/tmp/test.txt",
            change_type=FileChangeType.ADDED,
            size_bytes=100,
            mime_type="text/plain",
        )

        assert change.path == "test.txt"
        assert change.change_type == FileChangeType.ADDED
        assert change.size_bytes == 100
        assert not change.processed

    def test_file_change_to_dict(self):
        """Test FileChange serialization."""
        change = FileChange(
            path="test.txt",
            absolute_path="/tmp/test.txt",
            change_type=FileChangeType.MODIFIED,
            content_hash="abc123",
        )

        data = change.to_dict()

        assert data["path"] == "test.txt"
        assert data["change_type"] == "modified"
        assert data["content_hash"] == "abc123"
        assert "detected_at" in data


class TestSyncState:
    """Tests for SyncState model."""

    def test_create_sync_state(self):
        """Test creating a SyncState."""
        state = SyncState(
            workspace_id="ws_123",
            root_path="/tmp/docs",
        )

        assert state.workspace_id == "ws_123"
        assert state.status == SyncStatus.IDLE
        assert state.total_files == 0

    def test_sync_state_to_dict(self):
        """Test SyncState serialization."""
        state = SyncState(
            workspace_id="ws_123",
            root_path="/tmp/docs",
            status=SyncStatus.SYNCING,
            total_files=42,
        )

        data = state.to_dict()

        assert data["workspace_id"] == "ws_123"
        assert data["status"] == "syncing"
        assert data["total_files"] == 42


class TestSyncResult:
    """Tests for SyncResult model."""

    def test_create_sync_result(self):
        """Test creating a SyncResult."""
        result = SyncResult(
            workspace_id="ws_123",
            root_path="/tmp/docs",
            success=True,
            files_added=5,
            files_modified=3,
        )

        assert result.success
        assert result.files_added == 5
        assert result.total_processed == 8

    def test_sync_result_duration(self):
        """Test SyncResult duration calculation."""
        result = SyncResult(
            workspace_id="ws_123",
            root_path="/tmp/docs",
            success=True,
        )

        # No completed_at yet
        assert result.duration_seconds is None

        # Set completed
        result.completed_at = datetime.utcnow()
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0


# =============================================================================
# DirectoryWatcher Tests
# =============================================================================


class TestDirectoryWatcher:
    """Tests for DirectoryWatcher."""

    @pytest.mark.asyncio
    async def test_start_watching(self, populated_dir: Path, watcher_config: WatcherConfig):
        """Test starting to watch a directory."""
        watcher = DirectoryWatcher(config=watcher_config)

        try:
            state = await watcher.start_watching(
                populated_dir,
                workspace_id="ws_test",
                initial_scan=True,
            )

            assert state.workspace_id == "ws_test"
            # Path may resolve differently on macOS (/var -> /private/var)
            assert Path(state.root_path).name == populated_dir.name
            assert state.status == SyncStatus.IDLE
            # Should have 3 files (excluding .git)
            assert state.total_files == 3
            assert "file1.txt" in state.known_files
            assert "subdir/nested.py" in state.known_files
        finally:
            await watcher.stop_all()

    @pytest.mark.asyncio
    async def test_stop_watching(self, temp_dir: Path, watcher_config: WatcherConfig):
        """Test stopping a directory watch."""
        watcher = DirectoryWatcher(config=watcher_config)

        state = await watcher.start_watching(temp_dir, "ws_test")
        assert watcher.get_state(temp_dir) is not None

        await watcher.stop_watching(temp_dir)
        assert watcher.get_state(temp_dir) is None

    @pytest.mark.asyncio
    async def test_invalid_path(self, watcher_config: WatcherConfig):
        """Test watching a non-existent path."""
        watcher = DirectoryWatcher(config=watcher_config)

        with pytest.raises(ValueError, match="does not exist"):
            await watcher.start_watching("/nonexistent/path", "ws_test")

    @pytest.mark.asyncio
    async def test_get_changes_added(self, populated_dir: Path, watcher_config: WatcherConfig):
        """Test detecting added files."""
        watcher = DirectoryWatcher(config=watcher_config)

        try:
            state = await watcher.start_watching(populated_dir, "ws_test")

            # Add a new file
            new_file = populated_dir / "new_file.txt"
            new_file.write_text("New content")

            # Get changes
            changes = await watcher.get_changes(populated_dir)

            assert len(changes) == 1
            assert changes[0].path == "new_file.txt"
            assert changes[0].change_type == FileChangeType.ADDED
        finally:
            await watcher.stop_all()

    @pytest.mark.asyncio
    async def test_get_changes_modified(self, populated_dir: Path, watcher_config: WatcherConfig):
        """Test detecting modified files."""
        watcher = DirectoryWatcher(config=watcher_config)

        try:
            state = await watcher.start_watching(populated_dir, "ws_test")

            # Modify existing file
            (populated_dir / "file1.txt").write_text("Modified content")

            # Get changes
            changes = await watcher.get_changes(populated_dir)

            assert len(changes) == 1
            assert changes[0].path == "file1.txt"
            assert changes[0].change_type == FileChangeType.MODIFIED
        finally:
            await watcher.stop_all()

    @pytest.mark.asyncio
    async def test_get_changes_deleted(self, populated_dir: Path, watcher_config: WatcherConfig):
        """Test detecting deleted files."""
        watcher = DirectoryWatcher(config=watcher_config)

        try:
            state = await watcher.start_watching(populated_dir, "ws_test")

            # Delete a file
            (populated_dir / "file1.txt").unlink()

            # Get changes
            changes = await watcher.get_changes(populated_dir)

            assert len(changes) == 1
            assert changes[0].path == "file1.txt"
            assert changes[0].change_type == FileChangeType.DELETED
        finally:
            await watcher.stop_all()

    @pytest.mark.asyncio
    async def test_exclude_patterns(self, temp_dir: Path, watcher_config: WatcherConfig):
        """Test that excluded files are not tracked."""
        # Create files that should be excluded
        (temp_dir / "normal.txt").write_text("normal")
        pycache = temp_dir / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_bytes(b"bytecode")

        watcher = DirectoryWatcher(config=watcher_config)

        try:
            state = await watcher.start_watching(temp_dir, "ws_test")

            # Only normal.txt should be tracked
            assert state.total_files == 1
            assert "normal.txt" in state.known_files
            assert "__pycache__/module.pyc" not in state.known_files
        finally:
            await watcher.stop_all()


# =============================================================================
# SyncManager Tests
# =============================================================================


class TestSyncManager:
    """Tests for SyncManager."""

    @pytest.mark.asyncio
    async def test_sync_directory(self, populated_dir: Path, sync_config: SyncConfig):
        """Test syncing a directory."""
        manager = SyncManager(config=sync_config)

        try:
            result = await manager.sync_directory(
                populated_dir,
                workspace_id="ws_test",
            )

            assert result.success
            assert result.workspace_id == "ws_test"
            # First sync should have no changes (all files are "known")
            assert result.files_unchanged == 3

            # Add a new file
            (populated_dir / "new.txt").write_text("New file")

            # Second sync should detect the new file
            result2 = await manager.sync_directory(populated_dir, "ws_test")
            assert result2.success
            assert result2.files_added == 1
        finally:
            await manager.stop_all()

    @pytest.mark.asyncio
    async def test_sync_full_sync(self, populated_dir: Path, sync_config: SyncConfig):
        """Test full sync mode."""
        manager = SyncManager(config=sync_config)

        try:
            # Full sync should process all files as modified
            result = await manager.sync_directory(
                populated_dir,
                workspace_id="ws_test",
                full_sync=True,
            )

            assert result.success
            assert result.files_modified == 3  # All files processed
            assert result.files_unchanged == 0
        finally:
            await manager.stop_all()

    @pytest.mark.asyncio
    async def test_sync_with_document_store(self, populated_dir: Path, sync_config: SyncConfig):
        """Test sync with mock document store."""
        # Create mock document store
        mock_store = MagicMock()
        mock_store.ingest_file = AsyncMock(return_value="doc_123")
        mock_store.update_document = AsyncMock()
        mock_store.delete_document = AsyncMock()

        manager = SyncManager(document_store=mock_store, config=sync_config)

        try:
            # Initial sync
            await manager.sync_directory(populated_dir, "ws_test")

            # Add a file
            (populated_dir / "new.txt").write_text("New content")

            # Sync again
            result = await manager.sync_directory(populated_dir, "ws_test")

            assert result.files_added == 1
            mock_store.ingest_file.assert_called()
        finally:
            await manager.stop_all()

    @pytest.mark.asyncio
    async def test_sync_delete_missing(self, populated_dir: Path):
        """Test deleting documents for removed files."""
        config = SyncConfig(delete_missing=True)
        mock_store = MagicMock()
        mock_store.ingest_file = AsyncMock(return_value="doc_123")
        mock_store.delete_document = AsyncMock()

        manager = SyncManager(document_store=mock_store, config=config)

        try:
            # Initial sync
            await manager.sync_directory(populated_dir, "ws_test")

            # Manually set document mapping
            state = manager.get_state(populated_dir)
            state.document_map["file1.txt"] = "doc_file1"

            # Delete the file
            (populated_dir / "file1.txt").unlink()

            # Sync again
            result = await manager.sync_directory(populated_dir, "ws_test")

            assert result.files_deleted == 1
            mock_store.delete_document.assert_called_with("doc_file1")
        finally:
            await manager.stop_all()

    @pytest.mark.asyncio
    async def test_progress_callback(self, populated_dir: Path, sync_config: SyncConfig):
        """Test progress callback."""
        progress_calls = []

        def on_progress(workspace_id: str, processed: int, total: int):
            progress_calls.append((workspace_id, processed, total))

        manager = SyncManager(config=sync_config, on_progress=on_progress)

        try:
            # Full sync to trigger progress for each file
            await manager.sync_directory(populated_dir, "ws_test", full_sync=True)

            # Should have 3 progress calls (one per file)
            assert len(progress_calls) == 3
            assert all(ws == "ws_test" for ws, _, _ in progress_calls)
        finally:
            await manager.stop_all()

    @pytest.mark.asyncio
    async def test_start_stop_watching(self, temp_dir: Path, sync_config: SyncConfig):
        """Test start and stop watching."""
        manager = SyncManager(config=sync_config)

        state = await manager.start_watching(temp_dir, "ws_test", auto_sync=False)
        assert state is not None
        assert manager.get_state(temp_dir) is not None

        await manager.stop_watching(temp_dir)
        assert manager.get_state(temp_dir) is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestSyncIntegration:
    """Integration tests for the sync module."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_full_sync_workflow(self, temp_dir: Path):
        """Test complete sync workflow."""
        config = SyncConfig(
            debounce_ms=50,
            delete_missing=True,
            update_existing=True,
        )

        manager = SyncManager(config=config)

        try:
            # Create initial files
            (temp_dir / "doc1.txt").write_text("Document 1")
            (temp_dir / "doc2.md").write_text("# Document 2")

            # Initial sync
            result1 = await manager.sync_directory(temp_dir, "ws_test")
            assert result1.success
            assert result1.files_unchanged == 2

            # Modify a file
            (temp_dir / "doc1.txt").write_text("Document 1 - Modified")

            # Second sync
            result2 = await manager.sync_directory(temp_dir, "ws_test")
            assert result2.success
            assert result2.files_modified == 1

            # Add a file
            (temp_dir / "doc3.json").write_text('{"key": "value"}')

            # Third sync
            result3 = await manager.sync_directory(temp_dir, "ws_test")
            assert result3.success
            assert result3.files_added == 1

            # Delete a file
            (temp_dir / "doc2.md").unlink()

            # Fourth sync
            result4 = await manager.sync_directory(temp_dir, "ws_test")
            assert result4.success
            assert result4.files_deleted == 1

            # Verify final state
            state = manager.get_state(temp_dir)
            assert state.total_files == 2  # doc1.txt and doc3.json

        finally:
            await manager.stop_all()


# =============================================================================
# Edge Cases
# =============================================================================


class TestSyncEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_directory(self, temp_dir: Path):
        """Test syncing an empty directory."""
        manager = SyncManager()

        try:
            result = await manager.sync_directory(temp_dir, "ws_test")
            assert result.success
            assert result.files_unchanged == 0
            assert result.total_processed == 0
        finally:
            await manager.stop_all()

    @pytest.mark.asyncio
    async def test_nested_directories(self, temp_dir: Path):
        """Test syncing deeply nested directories."""
        # Create nested structure
        deep = temp_dir / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "deep.txt").write_text("Deep file")

        manager = SyncManager()

        try:
            result = await manager.sync_directory(temp_dir, "ws_test")
            assert result.success

            state = manager.get_state(temp_dir)
            assert "a/b/c/d/deep.txt" in state.known_files
        finally:
            await manager.stop_all()

    @pytest.mark.asyncio
    async def test_binary_files(self, temp_dir: Path):
        """Test handling binary files."""
        # Create a binary file
        (temp_dir / "image.bin").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        manager = SyncManager()

        try:
            result = await manager.sync_directory(temp_dir, "ws_test")
            assert result.success

            state = manager.get_state(temp_dir)
            assert "image.bin" in state.known_files
        finally:
            await manager.stop_all()

    @pytest.mark.asyncio
    async def test_special_characters_in_filename(self, temp_dir: Path):
        """Test files with special characters in names."""
        # Create file with special characters
        special_file = temp_dir / "file with spaces & symbols!.txt"
        special_file.write_text("Content")

        manager = SyncManager()

        try:
            result = await manager.sync_directory(temp_dir, "ws_test")
            assert result.success

            state = manager.get_state(temp_dir)
            assert "file with spaces & symbols!.txt" in state.known_files
        finally:
            await manager.stop_all()
