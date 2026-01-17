"""
Sync Manager for orchestrating directory synchronization.

Coordinates between the directory watcher and document ingestion.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from aragora.sync.models import (
    FileChange,
    FileChangeType,
    SyncConfig,
    SyncResult,
    SyncState,
    SyncStatus,
)
from aragora.sync.watcher import DirectoryWatcher, WatcherConfig

if TYPE_CHECKING:
    from aragora.documents.store import DocumentStore

logger = logging.getLogger(__name__)


class SyncManager:
    """
    Orchestrates directory synchronization with document storage.

    Provides:
    - One-time sync operations
    - Continuous watching with auto-sync
    - Incremental change processing
    - Progress callbacks
    """

    def __init__(
        self,
        document_store: Optional["DocumentStore"] = None,
        config: Optional[SyncConfig] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        """
        Initialize the sync manager.

        Args:
            document_store: Document store for ingestion
            config: Sync configuration
            on_progress: Callback(workspace_id, processed, total) for progress updates
        """
        self.document_store = document_store
        self.config = config or SyncConfig()
        self.on_progress = on_progress

        # Create watcher with matching config
        watcher_config = WatcherConfig(
            debounce_ms=self.config.debounce_ms,
            exclude_patterns=self.config.exclude_patterns,
            max_file_size_bytes=int(self.config.max_file_size_mb * 1024 * 1024),
        )
        self._watcher = DirectoryWatcher(
            config=watcher_config,
            on_change=self._handle_change,
        )

        # Sync state
        self._sync_states: dict[str, SyncState] = {}
        self._sync_locks: dict[str, asyncio.Lock] = {}
        self._pending_changes: dict[str, list[FileChange]] = {}

    async def sync_directory(
        self,
        path: str | Path,
        workspace_id: str,
        full_sync: bool = False,
    ) -> SyncResult:
        """
        Synchronize a directory with the document store.

        Args:
            path: Directory path to sync
            workspace_id: Target workspace ID
            full_sync: If True, re-process all files even if unchanged

        Returns:
            SyncResult with statistics
        """
        root = Path(path).resolve()
        path_str = str(root)

        # Get or create sync state
        state = self._watcher.get_state(path)
        if state is None:
            state = await self._watcher.start_watching(path, workspace_id, initial_scan=True)

        # Track in our own states dict
        self._sync_states[path_str] = state

        # Get sync lock
        if path_str not in self._sync_locks:
            self._sync_locks[path_str] = asyncio.Lock()

        async with self._sync_locks[path_str]:
            result = SyncResult(
                workspace_id=workspace_id,
                root_path=path_str,
                success=True,
            )

            try:
                state.status = SyncStatus.SYNCING

                if full_sync:
                    # Process all known files
                    changes = [
                        FileChange(
                            path=rel_path,
                            absolute_path=str(root / rel_path),
                            change_type=FileChangeType.MODIFIED,
                            content_hash=content_hash,
                        )
                        for rel_path, content_hash in state.known_files.items()
                    ]
                else:
                    # Get only changed files
                    changes = await self._watcher.get_changes(path, since=state.last_sync_at)

                result.files_unchanged = state.total_files - len(changes)

                # Process changes in batches
                await self._process_changes(changes, state, result)

                # Update state
                state.last_sync_at = datetime.utcnow()
                state.sync_count += 1
                state.status = SyncStatus.COMPLETED

            except Exception as e:
                logger.error(f"Sync failed for {path_str}: {e}")
                result.success = False
                result.errors.append(str(e))
                state.status = SyncStatus.FAILED
                state.error_count += 1
                state.last_error = str(e)

            result.completed_at = datetime.utcnow()
            return result

    async def start_watching(
        self,
        path: str | Path,
        workspace_id: str,
        auto_sync: bool = True,
    ) -> SyncState:
        """
        Start watching a directory for changes.

        Args:
            path: Directory path to watch
            workspace_id: Target workspace ID
            auto_sync: If True, automatically sync changes

        Returns:
            SyncState for the watched directory
        """
        root = Path(path).resolve()
        path_str = str(root)

        # Initialize pending changes list
        self._pending_changes[path_str] = []

        # Start watching
        state = await self._watcher.start_watching(path, workspace_id, initial_scan=True)
        self._sync_states[path_str] = state

        if auto_sync:
            # Start auto-sync task
            asyncio.create_task(self._auto_sync_loop(root, workspace_id))

        return state

    async def stop_watching(self, path: str | Path) -> None:
        """Stop watching a directory."""
        root = Path(path).resolve()
        path_str = str(root)

        await self._watcher.stop_watching(path)

        self._sync_states.pop(path_str, None)
        self._pending_changes.pop(path_str, None)

    async def stop_all(self) -> None:
        """Stop all watchers."""
        await self._watcher.stop_all()
        self._sync_states.clear()
        self._pending_changes.clear()

    def get_state(self, path: str | Path) -> Optional[SyncState]:
        """Get sync state for a path."""
        root = Path(path).resolve()
        return self._sync_states.get(str(root))

    def get_all_states(self) -> dict[str, SyncState]:
        """Get all sync states."""
        return dict(self._sync_states)

    async def _auto_sync_loop(self, root: Path, workspace_id: str) -> None:
        """Background loop that syncs pending changes."""
        path_str = str(root)

        while path_str in self._sync_states:
            try:
                # Wait for debounce period
                await asyncio.sleep(self.config.debounce_ms / 1000)

                # Check for pending changes
                pending = self._pending_changes.get(path_str, [])
                if pending:
                    # Move to processing
                    self._pending_changes[path_str] = []

                    state = self._sync_states.get(path_str)
                    if state:
                        result = SyncResult(
                            workspace_id=workspace_id,
                            root_path=path_str,
                            success=True,
                        )
                        await self._process_changes(pending, state, result)

                        if result.errors:
                            logger.warning(f"Auto-sync errors for {path_str}: {result.errors}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-sync error for {path_str}: {e}")

    def _handle_change(self, change: FileChange) -> None:
        """Handle a file change from the watcher."""
        # Find the watched path for this change
        for path_str, state in self._sync_states.items():
            if change.absolute_path.startswith(path_str):
                if path_str not in self._pending_changes:
                    self._pending_changes[path_str] = []
                self._pending_changes[path_str].append(change)
                state.last_change_at = change.detected_at
                break

    async def _process_changes(
        self,
        changes: list[FileChange],
        state: SyncState,
        result: SyncResult,
    ) -> None:
        """Process a batch of file changes."""
        total = len(changes)

        for i, change in enumerate(changes):
            try:
                if change.change_type == FileChangeType.ADDED:
                    await self._handle_add(change, state)
                    result.files_added += 1
                    result.changes.append(change)

                elif change.change_type == FileChangeType.MODIFIED:
                    if self.config.update_existing:
                        await self._handle_modify(change, state)
                        result.files_modified += 1
                        result.changes.append(change)
                    else:
                        result.files_unchanged += 1

                elif change.change_type == FileChangeType.DELETED:
                    if self.config.delete_missing:
                        await self._handle_delete(change, state)
                        result.files_deleted += 1
                        result.changes.append(change)

                change.processed = True

            except Exception as e:
                logger.error(f"Error processing {change.path}: {e}")
                change.error = str(e)
                result.files_failed += 1
                result.errors.append(f"{change.path}: {e}")

            # Progress callback
            if self.on_progress:
                self.on_progress(state.workspace_id, i + 1, total)

            # Yield control
            await asyncio.sleep(0)

    async def _handle_add(self, change: FileChange, state: SyncState) -> None:
        """Handle a newly added file."""
        if not self.document_store:
            logger.debug(f"No document store, skipping add: {change.path}")
            return

        try:
            # Read file content
            content = Path(change.absolute_path).read_bytes()

            # Ingest document
            doc_id = await self.document_store.ingest_file(
                content=content,
                filename=Path(change.path).name,
                workspace_id=state.workspace_id,
                metadata={
                    "source": "sync",
                    "sync_path": change.path,
                    "content_hash": change.content_hash,
                },
            )

            # Update state mapping
            state.document_map[change.path] = doc_id
            change.document_id = doc_id

            logger.debug(f"Added document: {change.path} -> {doc_id}")

        except Exception as e:
            logger.error(f"Failed to ingest {change.path}: {e}")
            raise

    async def _handle_modify(self, change: FileChange, state: SyncState) -> None:
        """Handle a modified file."""
        if not self.document_store:
            logger.debug(f"No document store, skipping modify: {change.path}")
            return

        # Get existing document ID
        doc_id = state.document_map.get(change.path)

        if doc_id:
            try:
                # Read updated content
                content = Path(change.absolute_path).read_bytes()

                # Update document
                await self.document_store.update_document(
                    document_id=doc_id,
                    content=content,
                    metadata={
                        "content_hash": change.content_hash,
                        "updated_via": "sync",
                    },
                )

                change.document_id = doc_id
                logger.debug(f"Updated document: {change.path} ({doc_id})")

            except Exception as e:
                logger.error(f"Failed to update {change.path}: {e}")
                raise
        else:
            # No existing document, treat as add
            await self._handle_add(change, state)

    async def _handle_delete(self, change: FileChange, state: SyncState) -> None:
        """Handle a deleted file."""
        if not self.document_store:
            logger.debug(f"No document store, skipping delete: {change.path}")
            return

        doc_id = state.document_map.get(change.path)

        if doc_id:
            try:
                await self.document_store.delete_document(doc_id)
                del state.document_map[change.path]
                change.document_id = doc_id
                logger.debug(f"Deleted document: {change.path} ({doc_id})")

            except Exception as e:
                logger.error(f"Failed to delete {change.path}: {e}")
                raise


__all__ = [
    "SyncManager",
]
