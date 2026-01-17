"""
Directory Sync Module.

Provides incremental synchronization for document workspaces:
- Watch directories for file changes
- Detect added, modified, and deleted files
- Trigger incremental document ingestion
- Track sync state and history

Usage:
    from aragora.sync import DirectoryWatcher, SyncManager, FileChangeType

    # Start watching a directory
    watcher = DirectoryWatcher()
    await watcher.start_watching("/path/to/docs", workspace_id="ws_123")

    # Or use the sync manager for full workflow
    sync = SyncManager(document_store)
    await sync.sync_directory("/path/to/docs", workspace_id="ws_123")
"""

from aragora.sync.models import (
    FileChange,
    FileChangeType,
    SyncConfig,
    SyncResult,
    SyncState,
    SyncStatus,
)
from aragora.sync.watcher import DirectoryWatcher, WatcherConfig
from aragora.sync.sync_manager import SyncManager

__all__ = [
    # Models
    "FileChange",
    "FileChangeType",
    "SyncConfig",
    "SyncResult",
    "SyncState",
    "SyncStatus",
    # Watcher
    "DirectoryWatcher",
    "WatcherConfig",
    # Manager
    "SyncManager",
]
