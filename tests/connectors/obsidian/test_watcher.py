"""Tests for the Obsidian vault watcher."""

from __future__ import annotations

import asyncio
import os
import pytest
import tempfile
from pathlib import Path

from aragora.connectors.obsidian.watcher import (
    FileChange,
    ObsidianVaultWatcher,
    WatcherState,
)


@pytest.fixture
def vault(tmp_path):
    """Create a temporary Obsidian vault with sample notes."""
    # Create some markdown files
    (tmp_path / "note1.md").write_text("# Note 1\nHello world")
    (tmp_path / "note2.md").write_text("# Note 2\nGoodbye world")

    # Create a subfolder with notes
    sub = tmp_path / "projects"
    sub.mkdir()
    (sub / "project-a.md").write_text("# Project A\nDetails here")

    # Create .obsidian folder (should be ignored)
    obs = tmp_path / ".obsidian"
    obs.mkdir()
    (obs / "config.json").write_text("{}")

    return tmp_path


class TestWatcherState:
    def test_initial_state(self):
        state = WatcherState()
        assert state.files == {}
        assert state.last_scan == 0.0


class TestFileChange:
    def test_creation(self):
        change = FileChange(
            path="note1.md",
            abs_path="/vault/note1.md",
            change_type="created",
            mtime=1234.0,
        )
        assert change.path == "note1.md"
        assert change.change_type == "created"


class TestObsidianVaultWatcher:
    def test_init(self, vault):
        watcher = ObsidianVaultWatcher(vault_path=str(vault))
        assert watcher.vault_path == vault
        assert watcher.poll_interval == 5.0
        assert ".obsidian" in watcher.ignore_folders

    def test_scan_initial(self, vault):
        """First scan after init should return all files as created."""
        watcher = ObsidianVaultWatcher(vault_path=str(vault))
        changes = watcher.scan_changes()

        assert len(changes) == 3  # note1, note2, projects/project-a
        assert all(c.change_type == "created" for c in changes)

        paths = {c.path for c in changes}
        assert "note1.md" in paths
        assert "note2.md" in paths
        assert os.path.join("projects", "project-a.md") in paths

    def test_scan_no_changes(self, vault):
        """Second scan with no changes should return empty."""
        watcher = ObsidianVaultWatcher(vault_path=str(vault))
        watcher.scan_changes()  # Initial scan

        changes = watcher.scan_changes()  # Second scan
        assert changes == []

    def test_scan_modified_file(self, vault):
        """Detect modified files."""
        watcher = ObsidianVaultWatcher(vault_path=str(vault))
        watcher.scan_changes()  # Initial scan

        # Modify a file
        (vault / "note1.md").write_text("# Note 1\nUpdated content!")

        changes = watcher.scan_changes()
        assert len(changes) == 1
        assert changes[0].path == "note1.md"
        assert changes[0].change_type == "modified"

    def test_scan_new_file(self, vault):
        """Detect new files."""
        watcher = ObsidianVaultWatcher(vault_path=str(vault))
        watcher.scan_changes()  # Initial scan

        # Add a new file
        (vault / "note3.md").write_text("# Note 3\nNew note")

        changes = watcher.scan_changes()
        assert len(changes) == 1
        assert changes[0].path == "note3.md"
        assert changes[0].change_type == "created"

    def test_scan_deleted_file(self, vault):
        """Detect deleted files."""
        watcher = ObsidianVaultWatcher(vault_path=str(vault))
        watcher.scan_changes()  # Initial scan

        # Delete a file
        (vault / "note2.md").unlink()

        changes = watcher.scan_changes()
        assert len(changes) == 1
        assert changes[0].path == "note2.md"
        assert changes[0].change_type == "deleted"

    def test_ignores_obsidian_folder(self, vault):
        """Should not detect files in .obsidian folder."""
        watcher = ObsidianVaultWatcher(vault_path=str(vault))
        changes = watcher.scan_changes()

        paths = {c.path for c in changes}
        assert not any(".obsidian" in p for p in paths)

    def test_ignores_custom_folders(self, vault):
        """Should respect custom ignore folders."""
        # Create a folder to ignore
        ignored = vault / "private"
        ignored.mkdir()
        (ignored / "secret.md").write_text("# Secret")

        watcher = ObsidianVaultWatcher(
            vault_path=str(vault),
            ignore_folders=["private"],
        )
        changes = watcher.scan_changes()

        paths = {c.path for c in changes}
        assert not any("private" in p for p in paths)

    def test_only_markdown_files(self, vault):
        """Should only detect .md files."""
        (vault / "image.png").write_bytes(b"\x89PNG")
        (vault / "data.json").write_text("{}")

        watcher = ObsidianVaultWatcher(vault_path=str(vault))
        changes = watcher.scan_changes()

        assert all(c.path.endswith(".md") for c in changes)

    def test_nonexistent_vault(self, tmp_path):
        """Should handle nonexistent vault gracefully."""
        watcher = ObsidianVaultWatcher(
            vault_path=str(tmp_path / "nonexistent"),
        )
        changes = watcher.scan_changes()
        assert changes == []

    def test_full_scan_populates_state(self, vault):
        """_full_scan should populate state without returning changes."""
        watcher = ObsidianVaultWatcher(vault_path=str(vault))
        watcher._full_scan()

        assert len(watcher._state.files) == 3
        assert watcher._state.last_scan > 0

    @pytest.mark.asyncio
    async def test_start_stop(self, vault):
        """Test start/stop lifecycle."""
        changes_received = []

        async def on_change(changes):
            changes_received.extend(changes)

        watcher = ObsidianVaultWatcher(
            vault_path=str(vault),
            poll_interval=0.1,
            on_change=on_change,
        )

        await watcher.start()
        assert watcher._running is True
        assert watcher._task is not None

        # Modify a file to trigger change detection
        (vault / "note1.md").write_text("# Updated")
        await asyncio.sleep(0.3)

        await watcher.stop()
        assert watcher._running is False
        assert watcher._task is None

        # Should have detected the change
        assert len(changes_received) >= 1

    @pytest.mark.asyncio
    async def test_start_idempotent(self, vault):
        """Starting twice should be safe."""
        watcher = ObsidianVaultWatcher(vault_path=str(vault), poll_interval=0.1)
        await watcher.start()
        await watcher.start()  # Should not create second task
        await watcher.stop()
