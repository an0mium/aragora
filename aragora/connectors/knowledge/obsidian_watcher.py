"""
Obsidian Vault Watcher.

Filesystem watcher for bidirectional Obsidian vault synchronization.
Detects file creation, modification, and deletion events in an Obsidian vault
and delivers debounced change events to an async callback.

Uses watchdog for cross-platform filesystem monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None  # type: ignore[assignment,misc]
    FileSystemEventHandler = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class VaultChangeEvent:
    """Represents a single change event within an Obsidian vault.

    Attributes:
        path: Absolute filesystem path of the changed file.
        change_type: One of "created", "modified", "deleted".
        timestamp: Unix timestamp (seconds since epoch) of when the change was detected.
    """

    path: str
    change_type: str
    timestamp: float


# =============================================================================
# Watchdog Bridge
# =============================================================================

# Map watchdog event types to our change_type strings.
_WATCHDOG_EVENT_MAP = {
    "created": "created",
    "modified": "modified",
    "deleted": "deleted",
    "moved": "modified",  # Treat moves as modifications.
}


class _WatchdogHandler(FileSystemEventHandler):  # type: ignore[misc]
    """Bridges synchronous watchdog events into the async VaultWatcher pipeline.

    The watchdog Observer fires callbacks on a background thread.  This handler
    checks ``_should_process`` and, for qualifying events, schedules
    ``_handle_change`` on the watcher's event loop.
    """

    def __init__(self, watcher: VaultWatcher) -> None:
        super().__init__()
        self._watcher = watcher

    # FileSystemEventHandler interface ------------------------------------------

    def on_created(self, event: Any) -> None:
        self._dispatch(event, "created")

    def on_modified(self, event: Any) -> None:
        self._dispatch(event, "modified")

    def on_deleted(self, event: Any) -> None:
        self._dispatch(event, "deleted")

    def on_moved(self, event: Any) -> None:
        # Treat the destination as a modification.
        dest_path = getattr(event, "dest_path", None) or getattr(event, "src_path", "")
        if not getattr(event, "is_directory", False) and self._watcher._should_process(dest_path):
            loop = self._watcher._loop
            if loop is not None and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._watcher._handle_change(dest_path, "modified"),
                    loop,
                )

    # Helpers -------------------------------------------------------------------

    def _dispatch(self, event: Any, change_type: str) -> None:
        """Filter and forward a watchdog event to the async handler."""
        if getattr(event, "is_directory", False):
            return
        src_path: str = getattr(event, "src_path", "")
        if not self._watcher._should_process(src_path):
            return
        loop = self._watcher._loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._watcher._handle_change(src_path, change_type),
                loop,
            )


# =============================================================================
# VaultWatcher
# =============================================================================

# Default folders to ignore (Obsidian internal, trash, templates).
_DEFAULT_IGNORE_FOLDERS: list[str] = [".obsidian", ".trash", "templates"]


class VaultWatcher:
    """Watch an Obsidian vault for file changes and deliver debounced events.

    Args:
        vault_path: Filesystem path to the Obsidian vault root.
        on_change: Async callback ``(VaultChangeEvent) -> None`` invoked for
            each debounced change.
        debounce_ms: Minimum interval in milliseconds before a pending change
            is flushed.  Rapid edits within this window are coalesced.
        watch_tags: Optional list of Obsidian tags to filter on (reserved for
            future tag-aware filtering; currently stored but not enforced).
        ignore_folders: Folder names (relative to vault root) whose contents
            should be ignored.  Defaults to ``[".obsidian", ".trash", "templates"]``.
    """

    def __init__(
        self,
        vault_path: str,
        on_change: Callable[[VaultChangeEvent], Awaitable[None]] | None = None,
        debounce_ms: int = 500,
        watch_tags: list[str] | None = None,
        ignore_folders: list[str] | None = None,
    ) -> None:
        self._vault_path = Path(vault_path).expanduser().resolve()
        self._on_change = on_change
        self._debounce_ms = debounce_ms
        self._watch_tags = watch_tags
        self._ignore_folders = (
            ignore_folders if ignore_folders is not None else list(_DEFAULT_IGNORE_FOLDERS)
        )

        # Pending changes keyed by absolute path; latest event wins.
        self._pending: dict[str, VaultChangeEvent] = {}

        # Watchdog observer (created on start).
        self._observer: Any | None = None
        self._running = False

        # Event loop reference for thread-safe scheduling.
        self._loop: asyncio.AbstractEventLoop | None = None

    # =========================================================================
    # Public properties
    # =========================================================================

    @property
    def vault_path(self) -> Path:
        """Resolved vault root path."""
        return self._vault_path

    @property
    def debounce_ms(self) -> int:
        """Debounce interval in milliseconds."""
        return self._debounce_ms

    @property
    def watch_tags(self) -> list[str] | None:
        """Tag filter list (may be ``None``)."""
        return self._watch_tags

    @property
    def ignore_folders(self) -> list[str]:
        """Folder names whose contents are ignored."""
        return self._ignore_folders

    @property
    def is_running(self) -> bool:
        """Whether the watcher is actively monitoring the vault."""
        return self._running

    # =========================================================================
    # Path filtering
    # =========================================================================

    def _should_process(self, path: str) -> bool:
        """Determine whether a filesystem path should trigger a change event.

        Returns ``True`` only for ``.md`` files that are **not** inside any
        ignored folder.
        """
        try:
            p = Path(path)
        except (TypeError, ValueError):
            return False

        # Only markdown files.
        if p.suffix.lower() != ".md":
            return False

        # Reject if any path component matches an ignored folder.
        try:
            rel = p.relative_to(self._vault_path)
        except ValueError:
            # Path is not under the vault at all -- still check parts.
            rel = p

        for part in rel.parts:
            if part in self._ignore_folders:
                return False

        return True

    # =========================================================================
    # Async change pipeline
    # =========================================================================

    async def _handle_change(self, path: str, change_type: str) -> None:
        """Record a pending change (latest event per path wins)."""
        self._pending[path] = VaultChangeEvent(
            path=path,
            change_type=change_type,
            timestamp=time.time(),
        )

    async def _flush_pending(self) -> None:
        """Deliver all pending events to the ``on_change`` callback and clear the queue."""
        if not self._pending:
            return

        events = list(self._pending.values())
        self._pending.clear()

        if self._on_change is None:
            return

        for event in events:
            try:
                await self._on_change(event)
            except Exception:  # noqa: BLE001 -- external callback
                logger.warning("on_change callback failed for %s", event.path, exc_info=True)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start watching the vault for filesystem changes.

        Raises:
            RuntimeError: If watchdog is not installed.
        """
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError("watchdog is required for VaultWatcher -- pip install watchdog")

        if self._running:
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        handler = _WatchdogHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._vault_path), recursive=True)
        self._observer.start()
        self._running = True
        logger.info("VaultWatcher started for %s", self._vault_path)

    def stop(self) -> None:
        """Stop watching and clean up resources."""
        if not self._running or self._observer is None:
            return

        self._observer.stop()
        self._observer.join()
        self._observer = None
        self._running = False
        self._loop = None
        logger.info("VaultWatcher stopped for %s", self._vault_path)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "VaultChangeEvent",
    "VaultWatcher",
    "WATCHDOG_AVAILABLE",
]
