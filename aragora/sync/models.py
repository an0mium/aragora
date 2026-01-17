"""
Data models for directory synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class FileChangeType(str, Enum):
    """Type of file change detected."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


class SyncStatus(str, Enum):
    """Status of a sync operation."""

    IDLE = "idle"
    SCANNING = "scanning"
    SYNCING = "syncing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class FileChange:
    """Represents a detected file change."""

    path: str  # Relative path from watched root
    absolute_path: str
    change_type: FileChangeType
    detected_at: datetime = field(default_factory=datetime.utcnow)

    # File metadata (not available for deleted files)
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    extension: Optional[str] = None

    # Content hash for change detection
    content_hash: Optional[str] = None

    # Processing state
    processed: bool = False
    document_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "absolute_path": self.absolute_path,
            "change_type": self.change_type.value,
            "detected_at": self.detected_at.isoformat(),
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
            "extension": self.extension,
            "content_hash": self.content_hash,
            "processed": self.processed,
            "document_id": self.document_id,
            "error": self.error,
        }


@dataclass
class SyncConfig:
    """Configuration for directory synchronization."""

    # Watching settings
    debounce_ms: int = 1000  # Debounce file changes
    poll_interval_ms: int = 500  # Poll interval for watcher

    # Filtering
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "**/.git/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/node_modules/**",
            "**/.DS_Store",
            "**/*.pyc",
            "**/*.swp",
            "**/*.tmp",
        ]
    )
    include_patterns: Optional[list[str]] = None
    max_file_size_mb: float = 50.0

    # Sync behavior
    delete_missing: bool = False  # Delete documents for removed files
    update_existing: bool = True  # Update documents for modified files
    hash_check: bool = True  # Use content hash to detect real changes

    # Batch settings
    batch_size: int = 10  # Files to process per batch
    max_concurrent: int = 4  # Concurrent processing tasks


@dataclass
class SyncState:
    """Persistent state for a watched directory."""

    workspace_id: str
    root_path: str
    status: SyncStatus = SyncStatus.IDLE

    # File tracking
    known_files: dict[str, str] = field(default_factory=dict)  # path -> content_hash
    document_map: dict[str, str] = field(default_factory=dict)  # path -> document_id

    # Statistics
    total_files: int = 0
    last_sync_at: Optional[datetime] = None
    last_change_at: Optional[datetime] = None

    # History
    sync_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "root_path": self.root_path,
            "status": self.status.value,
            "total_files": self.total_files,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "last_change_at": self.last_change_at.isoformat() if self.last_change_at else None,
            "sync_count": self.sync_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class SyncResult:
    """Result of a sync operation."""

    workspace_id: str
    root_path: str
    success: bool

    # Changes processed
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    files_unchanged: int = 0
    files_failed: int = 0

    # Details
    changes: list[FileChange] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get sync duration in seconds."""
        if not self.completed_at:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def total_processed(self) -> int:
        """Total files processed."""
        return self.files_added + self.files_modified + self.files_deleted

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "root_path": self.root_path,
            "success": self.success,
            "files_added": self.files_added,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
            "files_unchanged": self.files_unchanged,
            "files_failed": self.files_failed,
            "total_processed": self.total_processed,
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


__all__ = [
    "FileChange",
    "FileChangeType",
    "SyncConfig",
    "SyncResult",
    "SyncState",
    "SyncStatus",
]
