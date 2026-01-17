"""
Folder upload module for Aragora.

Provides recursive folder scanning with configurable filtering:
- Gitignore-style pattern exclusions
- Size and file count limits
- Agent-based relevance filtering (optional)
- Depth-limited traversal

Example usage:
    from aragora.documents.folder import FolderScanner, FolderUploadConfig

    config = FolderUploadConfig(
        max_depth=5,
        exclude_patterns=["**/.git/**", "**/node_modules/**"],
        max_file_size_mb=50,
        max_total_size_mb=200,
        max_file_count=500,
    )

    scanner = FolderScanner(config)
    result = await scanner.scan("./my-documents/")

    print(f"Found {result.included_count} files to upload")
    print(f"Excluded {result.excluded_count} files")
"""

from .config import (
    ExcludedFile,
    ExclusionReason,
    FileInfo,
    FolderScanResult,
    FolderUploadConfig,
    FolderUploadProgress,
    FolderUploadResult,
)
from .filters import (
    PatternMatcher,
    SizeFilter,
    format_size_bytes,
    parse_size_string,
)
from .scanner import FolderScanner, get_folder_scanner
from .agent_filter import AgentFileFilter, FilterDecision, get_agent_filter

__all__ = [
    # Config classes
    "FolderUploadConfig",
    "FileInfo",
    "ExcludedFile",
    "ExclusionReason",
    "FolderScanResult",
    "FolderUploadProgress",
    "FolderUploadResult",
    # Filters
    "PatternMatcher",
    "SizeFilter",
    "parse_size_string",
    "format_size_bytes",
    # Scanner
    "FolderScanner",
    "get_folder_scanner",
    # Agent filter
    "AgentFileFilter",
    "FilterDecision",
    "get_agent_filter",
]
