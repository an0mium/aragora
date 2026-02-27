"""
Tests for Obsidian VaultWatcher.

Tests cover:
- Initialization with valid vault and custom options
- File filtering (_should_process)
- Async change handling and debouncing
- Start/stop lifecycle
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.knowledge.obsidian_watcher import (
    VaultChangeEvent,
    VaultWatcher,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_vault(tmp_path: Path) -> Path:
    """Create a temporary Obsidian vault directory."""
    vault = tmp_path / "test_vault"
    vault.mkdir()
    # Create .obsidian marker so it looks like a real vault
    (vault / ".obsidian").mkdir()
    return vault


# =============================================================================
# Initialization Tests
# =============================================================================


class TestVaultWatcherInit:
    """Tests for VaultWatcher initialization."""

    def test_init_with_valid_vault(self, temp_vault: Path) -> None:
        """VaultWatcher accepts a valid vault path and sets defaults."""
        watcher = VaultWatcher(vault_path=str(temp_vault))

        assert watcher.vault_path == Path(temp_vault)
        assert watcher.debounce_ms == 500
        assert watcher.watch_tags is None
        assert set(watcher.ignore_folders) == {".obsidian", ".trash", "templates"}
        assert watcher.is_running is False

    def test_init_with_custom_debounce(self, temp_vault: Path) -> None:
        """VaultWatcher accepts a custom debounce interval."""
        watcher = VaultWatcher(vault_path=str(temp_vault), debounce_ms=200)

        assert watcher.debounce_ms == 200

    def test_init_with_tag_filter(self, temp_vault: Path) -> None:
        """VaultWatcher accepts watch_tags for filtering."""
        tags = ["#debate", "#decision"]
        watcher = VaultWatcher(vault_path=str(temp_vault), watch_tags=tags)

        assert watcher.watch_tags == ["#debate", "#decision"]

    def test_init_with_ignore_folders(self, temp_vault: Path) -> None:
        """VaultWatcher accepts custom ignore_folders, replacing defaults."""
        folders = [".obsidian", "archive"]
        watcher = VaultWatcher(vault_path=str(temp_vault), ignore_folders=folders)

        assert watcher.ignore_folders == [".obsidian", "archive"]


# =============================================================================
# File Filtering Tests
# =============================================================================


class TestShouldProcess:
    """Tests for _should_process path filtering."""

    def test_should_process_markdown_file(self, temp_vault: Path) -> None:
        """Markdown files in non-ignored folders are accepted."""
        watcher = VaultWatcher(vault_path=str(temp_vault))

        md_path = str(temp_vault / "notes" / "my-note.md")
        assert watcher._should_process(md_path) is True

    def test_should_process_ignores_configured_folders(self, temp_vault: Path) -> None:
        """Files inside ignored folders are rejected."""
        watcher = VaultWatcher(vault_path=str(temp_vault))

        # Default ignored folders
        assert watcher._should_process(str(temp_vault / ".obsidian" / "config.json")) is False
        assert watcher._should_process(str(temp_vault / ".trash" / "old.md")) is False
        assert watcher._should_process(str(temp_vault / "templates" / "daily.md")) is False

    def test_should_process_non_vault_path(self, temp_vault: Path) -> None:
        """Non-markdown files are rejected even in valid locations."""
        watcher = VaultWatcher(vault_path=str(temp_vault))

        assert watcher._should_process(str(temp_vault / "notes" / "image.png")) is False
        assert watcher._should_process(str(temp_vault / "data.json")) is False
        assert watcher._should_process(str(temp_vault / "readme.txt")) is False


# =============================================================================
# Async Change Handling Tests
# =============================================================================


class TestChangeHandling:
    """Tests for async change handling and debouncing."""

    @pytest.mark.asyncio
    async def test_on_change_calls_callback(self, temp_vault: Path) -> None:
        """_handle_change queues an event and _flush_pending delivers it to the callback."""
        callback = AsyncMock()
        watcher = VaultWatcher(vault_path=str(temp_vault), on_change=callback)

        file_path = str(temp_vault / "notes" / "test.md")
        await watcher._handle_change(file_path, "created")
        await watcher._flush_pending()

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, VaultChangeEvent)
        assert event.path == file_path
        assert event.change_type == "created"
        assert isinstance(event.timestamp, float)

    @pytest.mark.asyncio
    async def test_debounce_coalesces_rapid_changes(self, temp_vault: Path) -> None:
        """Multiple rapid changes to the same file produce only one event on flush."""
        callback = AsyncMock()
        watcher = VaultWatcher(vault_path=str(temp_vault), on_change=callback)

        file_path = str(temp_vault / "notes" / "test.md")

        # Simulate rapid edits to the same file
        await watcher._handle_change(file_path, "modified")
        await watcher._handle_change(file_path, "modified")
        await watcher._handle_change(file_path, "modified")

        await watcher._flush_pending()

        # Should only call once despite three changes
        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.change_type == "modified"


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for start/stop lifecycle."""

    def test_start_stop_lifecycle(self, temp_vault: Path) -> None:
        """Start creates an Observer, stop shuts it down."""
        with patch("aragora.connectors.knowledge.obsidian_watcher.Observer") as MockObserver:
            mock_observer = MagicMock()
            MockObserver.return_value = mock_observer

            watcher = VaultWatcher(vault_path=str(temp_vault))

            assert watcher.is_running is False

            watcher.start()
            assert watcher.is_running is True
            mock_observer.schedule.assert_called_once()
            mock_observer.start.assert_called_once()

            watcher.stop()
            assert watcher.is_running is False
            mock_observer.stop.assert_called_once()
            mock_observer.join.assert_called_once()
