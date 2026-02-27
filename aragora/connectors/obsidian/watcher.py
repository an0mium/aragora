"""Obsidian vault watcher — detect changes and trigger sync.

Uses polling-based change detection (no external dependencies required).
Tracks file modification times to detect new, changed, and deleted notes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Folders Obsidian uses internally — always ignore
_OBSIDIAN_IGNORE = {".obsidian", ".trash", "templates", ".git", "__pycache__"}


@dataclass
class FileChange:
    """A detected file change in the vault."""

    path: str  # Relative path within vault
    abs_path: str
    change_type: str  # "created" | "modified" | "deleted"
    mtime: float = 0.0


@dataclass
class WatcherState:
    """Tracked state of all files in the vault."""

    # path -> (mtime, content_hash)
    files: dict[str, tuple[float, str]] = field(default_factory=dict)
    last_scan: float = 0.0


class ObsidianVaultWatcher:
    """Poll an Obsidian vault for markdown file changes.

    Parameters
    ----------
    vault_path:
        Absolute path to the Obsidian vault root.
    poll_interval:
        Seconds between scans.
    ignore_folders:
        Additional folder names to ignore.
    on_change:
        Async callback ``(changes: list[FileChange]) -> None``.
    """

    def __init__(
        self,
        vault_path: str,
        poll_interval: float = 5.0,
        ignore_folders: list[str] | None = None,
        on_change: Callable[[list[FileChange]], Any] | None = None,
    ) -> None:
        self.vault_path = Path(vault_path).resolve()
        self.poll_interval = poll_interval
        self.ignore_folders = _OBSIDIAN_IGNORE | set(ignore_folders or [])
        self.on_change = on_change
        self._state = WatcherState()
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background watcher loop."""
        if self._running:
            return
        self._running = True
        # Initial scan to populate state
        self._full_scan()
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("Obsidian watcher started: %s", self.vault_path)

    async def stop(self) -> None:
        """Stop the watcher."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Obsidian watcher stopped")

    def scan_changes(self) -> list[FileChange]:
        """Perform a single scan and return detected changes."""
        changes: list[FileChange] = []
        current_files: dict[str, tuple[float, str]] = {}

        for md_path in self._iter_markdown_files():
            rel_path = str(md_path.relative_to(self.vault_path))
            try:
                stat = md_path.stat()
                mtime = stat.st_mtime
            except OSError:
                continue

            # Quick content hash for change detection
            content_hash = self._hash_file(md_path)
            current_files[rel_path] = (mtime, content_hash)

            prev = self._state.files.get(rel_path)
            if prev is None:
                changes.append(
                    FileChange(
                        path=rel_path,
                        abs_path=str(md_path),
                        change_type="created",
                        mtime=mtime,
                    )
                )
            elif prev[1] != content_hash:
                changes.append(
                    FileChange(
                        path=rel_path,
                        abs_path=str(md_path),
                        change_type="modified",
                        mtime=mtime,
                    )
                )

        # Detect deletions
        for rel_path in self._state.files:
            if rel_path not in current_files:
                changes.append(
                    FileChange(
                        path=rel_path,
                        abs_path=str(self.vault_path / rel_path),
                        change_type="deleted",
                    )
                )

        self._state.files = current_files
        self._state.last_scan = time.time()
        return changes

    # -- internal -----------------------------------------------------------

    def _full_scan(self) -> None:
        """Populate state without emitting changes."""
        for md_path in self._iter_markdown_files():
            rel_path = str(md_path.relative_to(self.vault_path))
            try:
                mtime = md_path.stat().st_mtime
            except OSError:
                continue
            content_hash = self._hash_file(md_path)
            self._state.files[rel_path] = (mtime, content_hash)
        self._state.last_scan = time.time()

    async def _watch_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            try:
                await asyncio.sleep(self.poll_interval)
                changes = self.scan_changes()
                if changes and self.on_change:
                    await self.on_change(changes)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in vault watcher loop")

    def _iter_markdown_files(self):
        """Yield all .md files in the vault, respecting ignore rules."""
        if not self.vault_path.is_dir():
            return
        for root, dirs, files in os.walk(self.vault_path):
            # Filter out ignored directories in-place
            dirs[:] = [d for d in dirs if d not in self.ignore_folders and not d.startswith(".")]
            for fname in files:
                if fname.endswith(".md"):
                    yield Path(root) / fname

    @staticmethod
    def _hash_file(path: Path) -> str:
        """SHA-256 hash of file contents for change detection."""
        try:
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        except OSError:
            return ""
