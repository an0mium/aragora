"""
Directory Watcher for incremental synchronization.

Uses watchfiles for efficient cross-platform file system monitoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from watchfiles import Change, awatch

from aragora.sync.models import (
    FileChange,
    FileChangeType,
    SyncState,
    SyncStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class WatcherConfig:
    """Configuration for the directory watcher."""

    debounce_ms: int = 1000
    poll_delay_ms: int = 300
    step_ms: int = 50
    recursive: bool = True

    # Filter patterns (fnmatch style)
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            ".git",
            "__pycache__",
            ".venv",
            "node_modules",
            ".DS_Store",
            "*.pyc",
            "*.swp",
            "*.tmp",
        ]
    )

    # Max file size to track (bytes)
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50 MB


def _should_exclude(path: str, patterns: list[str]) -> bool:
    """Check if a path matches any exclude pattern."""
    import fnmatch

    path_parts = Path(path).parts

    for pattern in patterns:
        # Check if any path component matches
        if any(fnmatch.fnmatch(part, pattern) for part in path_parts):
            return True
        # Check full path
        if fnmatch.fnmatch(path, pattern):
            return True

    return False


def _compute_file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of file contents."""
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (OSError, IOError) as e:
        logger.warning(f"Could not hash file {path}: {e}")
        return ""


class DirectoryWatcher:
    """
    Watches directories for file changes and yields FileChange events.

    Uses watchfiles for efficient cross-platform file monitoring.
    Supports:
    - Debouncing rapid changes
    - Pattern-based filtering
    - Content hash-based change detection
    - Graceful start/stop
    """

    def __init__(
        self,
        config: Optional[WatcherConfig] = None,
        on_change: Optional[Callable[[FileChange], None]] = None,
    ):
        """
        Initialize the directory watcher.

        Args:
            config: Watcher configuration
            on_change: Optional callback for file changes
        """
        self.config = config or WatcherConfig()
        self.on_change = on_change

        self._watching: dict[str, SyncState] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def start_watching(
        self,
        path: str | Path,
        workspace_id: str,
        initial_scan: bool = True,
    ) -> SyncState:
        """
        Start watching a directory for changes.

        Args:
            path: Directory path to watch
            workspace_id: Associated workspace ID
            initial_scan: Whether to do an initial full scan

        Returns:
            SyncState for the watched directory
        """
        root = Path(path).resolve()
        path_str = str(root)

        if not root.exists():
            raise ValueError(f"Path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {root}")

        # Check if already watching
        if path_str in self._watching:
            logger.info(f"Already watching: {path_str}")
            return self._watching[path_str]

        # Create sync state
        state = SyncState(
            workspace_id=workspace_id,
            root_path=path_str,
            status=SyncStatus.IDLE,
        )

        # Initial scan to populate known files
        if initial_scan:
            logger.info(f"Initial scan of {path_str}")
            state.status = SyncStatus.SCANNING
            await self._initial_scan(root, state)
            state.status = SyncStatus.IDLE

        # Store state and start watch task
        self._watching[path_str] = state
        self._stop_events[path_str] = asyncio.Event()

        task = asyncio.create_task(self._watch_loop(root, state))
        self._tasks[path_str] = task

        logger.info(f"Started watching: {path_str} ({state.total_files} files)")
        return state

    async def stop_watching(self, path: str | Path) -> None:
        """
        Stop watching a directory.

        Args:
            path: Directory path to stop watching
        """
        root = Path(path).resolve()
        path_str = str(root)

        if path_str not in self._watching:
            logger.warning(f"Not watching: {path_str}")
            return

        # Signal stop
        self._stop_events[path_str].set()

        # Cancel task
        if path_str in self._tasks:
            self._tasks[path_str].cancel()
            try:
                await self._tasks[path_str]
            except asyncio.CancelledError:
                pass
            del self._tasks[path_str]

        # Clean up
        del self._watching[path_str]
        del self._stop_events[path_str]

        logger.info(f"Stopped watching: {path_str}")

    async def stop_all(self) -> None:
        """Stop watching all directories."""
        paths = list(self._watching.keys())
        for path in paths:
            await self.stop_watching(path)

    def get_state(self, path: str | Path) -> Optional[SyncState]:
        """Get the sync state for a watched directory."""
        root = Path(path).resolve()
        return self._watching.get(str(root))

    def get_all_states(self) -> dict[str, SyncState]:
        """Get all sync states."""
        return dict(self._watching)

    async def get_changes(
        self,
        path: str | Path,
        since: Optional[datetime] = None,
    ) -> list[FileChange]:
        """
        Get pending changes for a watched directory.

        Args:
            path: Directory path
            since: Only return changes after this time

        Returns:
            List of FileChange objects
        """
        root = Path(path).resolve()
        path_str = str(root)

        state = self._watching.get(path_str)
        if not state:
            raise ValueError(f"Not watching: {path_str}")

        # Scan for changes compared to known state
        changes = []
        current_files: dict[str, str] = {}

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue

            rel_path = str(file_path.relative_to(root))

            if _should_exclude(rel_path, self.config.exclude_patterns):
                continue

            try:
                stat = file_path.stat()
                if stat.st_size > self.config.max_file_size_bytes:
                    continue

                content_hash = _compute_file_hash(file_path)
                current_files[rel_path] = content_hash

                if rel_path not in state.known_files:
                    # New file
                    change = self._create_file_change(
                        file_path, root, FileChangeType.ADDED, content_hash
                    )
                    if since is None or change.detected_at > since:
                        changes.append(change)

                elif state.known_files[rel_path] != content_hash:
                    # Modified file
                    change = self._create_file_change(
                        file_path, root, FileChangeType.MODIFIED, content_hash
                    )
                    if since is None or change.detected_at > since:
                        changes.append(change)

            except (OSError, IOError) as e:
                logger.warning(f"Error scanning {file_path}: {e}")

        # Check for deleted files
        for rel_path in state.known_files:
            if rel_path not in current_files:
                abs_path = str(root / rel_path)
                change = FileChange(
                    path=rel_path,
                    absolute_path=abs_path,
                    change_type=FileChangeType.DELETED,
                )
                if since is None or change.detected_at > since:
                    changes.append(change)

        return changes

    async def watch_changes(
        self,
        path: str | Path,
    ) -> AsyncIterator[FileChange]:
        """
        Async iterator that yields file changes as they occur.

        Args:
            path: Directory path to watch

        Yields:
            FileChange objects as changes are detected
        """
        root = Path(path).resolve()
        path_str = str(root)

        if path_str not in self._watching:
            raise ValueError(f"Not watching: {path_str}. Call start_watching first.")

        state = self._watching[path_str]
        stop_event = self._stop_events[path_str]

        async for changes in awatch(
            root,
            debounce=self.config.debounce_ms,
            poll_delay_ms=self.config.poll_delay_ms,
            step=self.config.step_ms,
            recursive=self.config.recursive,
            stop_event=stop_event,
        ):
            for change_type, change_path in changes:
                file_path = Path(change_path)
                rel_path = str(file_path.relative_to(root))

                if _should_exclude(rel_path, self.config.exclude_patterns):
                    continue

                if change_type == Change.added:
                    if file_path.is_file():
                        change = self._create_file_change(file_path, root, FileChangeType.ADDED)
                        self._update_state(state, change)
                        yield change

                elif change_type == Change.modified:
                    if file_path.is_file():
                        # Check if content actually changed (hash-based)
                        new_hash = _compute_file_hash(file_path)
                        old_hash = state.known_files.get(rel_path)

                        if old_hash != new_hash:
                            change = self._create_file_change(
                                file_path, root, FileChangeType.MODIFIED, new_hash
                            )
                            self._update_state(state, change)
                            yield change

                elif change_type == Change.deleted:
                    if rel_path in state.known_files:
                        change = FileChange(
                            path=rel_path,
                            absolute_path=change_path,
                            change_type=FileChangeType.DELETED,
                        )
                        self._update_state(state, change)
                        yield change

    async def _initial_scan(self, root: Path, state: SyncState) -> None:
        """Perform initial scan to populate known files."""
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue

            rel_path = str(file_path.relative_to(root))

            if _should_exclude(rel_path, self.config.exclude_patterns):
                continue

            try:
                stat = file_path.stat()
                if stat.st_size > self.config.max_file_size_bytes:
                    continue

                content_hash = _compute_file_hash(file_path)
                state.known_files[rel_path] = content_hash
                state.total_files += 1

            except (OSError, IOError) as e:
                logger.warning(f"Error scanning {file_path}: {e}")

            # Yield control periodically
            await asyncio.sleep(0)

        state.last_sync_at = datetime.now(timezone.utc)
        state.updated_at = datetime.now(timezone.utc)

    async def _watch_loop(self, root: Path, state: SyncState) -> None:
        """Main watch loop that processes file changes."""
        stop_event = self._stop_events[str(root)]

        try:
            async for changes in awatch(
                root,
                debounce=self.config.debounce_ms,
                poll_delay_ms=self.config.poll_delay_ms,
                step=self.config.step_ms,
                recursive=self.config.recursive,
                stop_event=stop_event,
            ):
                for change_type, change_path in changes:
                    try:
                        await self._handle_change(root, state, change_type, change_path)
                    except Exception as e:
                        logger.error(f"Error handling change {change_path}: {e}")
                        state.error_count += 1
                        state.last_error = str(e)

        except asyncio.CancelledError:
            logger.info(f"Watch loop cancelled for {root}")
            raise

    async def _handle_change(
        self,
        root: Path,
        state: SyncState,
        change_type: Change,
        change_path: str,
    ) -> None:
        """Handle a single file change."""
        file_path = Path(change_path)
        rel_path = str(file_path.relative_to(root))

        if _should_exclude(rel_path, self.config.exclude_patterns):
            return

        if change_type == Change.added:
            if file_path.is_file():
                change = self._create_file_change(file_path, root, FileChangeType.ADDED)
                self._update_state(state, change)
                if self.on_change:
                    self.on_change(change)

        elif change_type == Change.modified:
            if file_path.is_file():
                new_hash = _compute_file_hash(file_path)
                old_hash = state.known_files.get(rel_path)

                if old_hash != new_hash:
                    change = self._create_file_change(
                        file_path, root, FileChangeType.MODIFIED, new_hash
                    )
                    self._update_state(state, change)
                    if self.on_change:
                        self.on_change(change)

        elif change_type == Change.deleted:
            if rel_path in state.known_files:
                change = FileChange(
                    path=rel_path,
                    absolute_path=change_path,
                    change_type=FileChangeType.DELETED,
                )
                self._update_state(state, change)
                if self.on_change:
                    self.on_change(change)

    def _create_file_change(
        self,
        file_path: Path,
        root: Path,
        change_type: FileChangeType,
        content_hash: Optional[str] = None,
    ) -> FileChange:
        """Create a FileChange object for a file."""
        rel_path = str(file_path.relative_to(root))

        try:
            stat = file_path.stat()
            size = stat.st_size
        except (OSError, IOError):
            size = None

        mime_type, _ = mimetypes.guess_type(str(file_path))

        if content_hash is None and change_type != FileChangeType.DELETED:
            content_hash = _compute_file_hash(file_path)

        return FileChange(
            path=rel_path,
            absolute_path=str(file_path),
            change_type=change_type,
            size_bytes=size,
            mime_type=mime_type,
            extension=file_path.suffix.lower(),
            content_hash=content_hash,
        )

    def _update_state(self, state: SyncState, change: FileChange) -> None:
        """Update sync state based on a file change."""
        state.last_change_at = change.detected_at
        state.updated_at = datetime.now(timezone.utc)

        if change.change_type == FileChangeType.ADDED:
            state.known_files[change.path] = change.content_hash or ""
            state.total_files += 1

        elif change.change_type == FileChangeType.MODIFIED:
            state.known_files[change.path] = change.content_hash or ""

        elif change.change_type == FileChangeType.DELETED:
            state.known_files.pop(change.path, None)
            state.document_map.pop(change.path, None)
            state.total_files = max(0, state.total_files - 1)


__all__ = [
    "DirectoryWatcher",
    "WatcherConfig",
]
