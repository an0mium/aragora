"""
Configuration dataclasses for folder upload.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExclusionReason(Enum):
    """Reason why a file was excluded."""

    PATTERN = "pattern"
    SIZE = "size"
    COUNT = "count"
    AGENT = "agent"
    PERMISSION = "permission"
    SYMLINK = "symlink"
    DEPTH = "depth"


@dataclass
class FolderUploadConfig:
    """Configuration for folder upload with filtering."""

    # Traversal settings
    max_depth: int = 10  # -1 for unlimited
    follow_symlinks: bool = False  # Security: default off

    # Explicit exclusion patterns (gitignore-style)
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "**/.git/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "**/.env",
            "**/*.pyc",
            "**/dist/**",
            "**/build/**",
            "**/.DS_Store",
            "**/._*",
        ]
    )
    include_patterns: list[str] = field(default_factory=list)  # Empty = all files

    # Size limits
    max_file_size_mb: int = 100  # Per-file limit
    max_total_size_mb: int = 500  # Batch total limit
    max_file_count: int = 1000  # Max files per upload

    # Agent-based filtering (optional)
    enable_agent_filter: bool = False
    agent_filter_model: str = "gemini-2.0-flash"  # Fast, cheap model
    agent_filter_prompt: str = ""  # Custom relevance criteria
    agent_filter_batch_size: int = 50  # Files per agent call

    # Processing options
    chunking_strategy: str = "auto"  # auto, semantic, sliding, recursive
    preserve_folder_structure: bool = True  # Track original paths

    # Timeout protection
    scan_timeout_seconds: float = 300.0  # 5 minutes max scan time

    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    def max_total_size_bytes(self) -> int:
        """Get max total size in bytes."""
        return self.max_total_size_mb * 1024 * 1024


@dataclass
class FileInfo:
    """Information about a file to be uploaded."""

    path: str  # Relative to root folder
    absolute_path: str
    size_bytes: int
    extension: str
    mime_type: str = "application/octet-stream"

    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)


@dataclass
class ExcludedFile:
    """Information about an excluded file."""

    path: str  # Relative to root folder
    reason: ExclusionReason
    details: str  # e.g., "Matched pattern: **/node_modules/**"


@dataclass
class FolderScanResult:
    """Result of scanning a folder for upload."""

    # Root folder info
    root_path: str
    root_absolute_path: str

    # Statistics
    total_files_found: int = 0
    total_size_bytes: int = 0
    files_excluded_by_pattern: int = 0
    files_excluded_by_size: int = 0
    files_excluded_by_count: int = 0
    files_excluded_by_agent: int = 0
    files_excluded_by_permission: int = 0
    directories_scanned: int = 0
    max_depth_reached: int = 0

    # File lists
    included_files: list[FileInfo] = field(default_factory=list)
    excluded_files: list[ExcludedFile] = field(default_factory=list)

    # Warnings and errors
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Timing
    scan_duration_ms: float = 0.0

    @property
    def included_count(self) -> int:
        """Number of files that will be uploaded."""
        return len(self.included_files)

    @property
    def excluded_count(self) -> int:
        """Number of files that were excluded."""
        return len(self.excluded_files)

    @property
    def included_size_bytes(self) -> int:
        """Total size of files to be uploaded."""
        return sum(f.size_bytes for f in self.included_files)

    @property
    def included_size_mb(self) -> float:
        """Total size in MB of files to be uploaded."""
        return self.included_size_bytes / (1024 * 1024)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "root_path": self.root_path,
            "root_absolute_path": self.root_absolute_path,
            "statistics": {
                "total_files_found": self.total_files_found,
                "total_size_bytes": self.total_size_bytes,
                "included_count": self.included_count,
                "included_size_bytes": self.included_size_bytes,
                "excluded_count": self.excluded_count,
                "files_excluded_by_pattern": self.files_excluded_by_pattern,
                "files_excluded_by_size": self.files_excluded_by_size,
                "files_excluded_by_count": self.files_excluded_by_count,
                "files_excluded_by_agent": self.files_excluded_by_agent,
                "files_excluded_by_permission": self.files_excluded_by_permission,
                "directories_scanned": self.directories_scanned,
                "max_depth_reached": self.max_depth_reached,
            },
            "included_files": [
                {
                    "path": f.path,
                    "size_bytes": f.size_bytes,
                    "extension": f.extension,
                    "mime_type": f.mime_type,
                }
                for f in self.included_files
            ],
            "excluded_files": [
                {"path": f.path, "reason": f.reason.value, "details": f.details}
                for f in self.excluded_files
            ],
            "warnings": self.warnings,
            "errors": self.errors,
            "scan_duration_ms": self.scan_duration_ms,
        }


@dataclass
class FolderUploadProgress:
    """Progress information during folder upload."""

    total_files: int
    uploaded_files: int
    failed_files: int
    current_file: Optional[str] = None
    bytes_uploaded: int = 0
    total_bytes: int = 0

    @property
    def progress_percent(self) -> float:
        """Get upload progress as percentage."""
        if self.total_files == 0:
            return 100.0
        return (self.uploaded_files / self.total_files) * 100


@dataclass
class FolderUploadResult:
    """Result of uploading a folder."""

    folder_id: str
    scan_result: FolderScanResult

    # Upload statistics
    files_uploaded: int = 0
    files_failed: int = 0
    document_ids: list[str] = field(default_factory=list)

    # Errors
    upload_errors: list[dict] = field(default_factory=list)

    # Timing
    upload_duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        """Check if upload was successful (no failures)."""
        return self.files_failed == 0
