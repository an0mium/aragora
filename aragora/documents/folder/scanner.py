"""
Folder scanner for recursive directory traversal with filtering.
"""

import asyncio
import mimetypes
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from .config import (
    ExcludedFile,
    ExclusionReason,
    FileInfo,
    FolderScanResult,
    FolderUploadConfig,
)
from .filters import PatternMatcher, SizeFilter


class FolderScanner:
    """
    Scans folders with filtering and depth control.

    Supports:
    - Recursive traversal up to configurable depth
    - Gitignore-style pattern exclusions
    - Size limits (per-file and total)
    - File count limits
    - Symlink handling (configurable)
    - Timeout protection
    """

    def __init__(self, config: Optional[FolderUploadConfig] = None):
        """
        Initialize the folder scanner.

        Args:
            config: Upload configuration with filtering options
        """
        self.config = config or FolderUploadConfig()
        self._pattern_matcher = PatternMatcher(
            exclude_patterns=self.config.exclude_patterns,
            include_patterns=self.config.include_patterns or None,
        )
        self._size_filter = SizeFilter(
            max_file_size_bytes=self.config.max_file_size_bytes(),
            max_total_size_bytes=self.config.max_total_size_bytes(),
            max_file_count=self.config.max_file_count,
        )

    async def scan(self, root_path: str | Path) -> FolderScanResult:
        """
        Scan a folder and return filtered file list.

        Args:
            root_path: Path to the folder to scan

        Returns:
            FolderScanResult with included/excluded files and statistics

        Raises:
            ValueError: If root_path is invalid
            asyncio.TimeoutError: If scan exceeds timeout
        """
        start_time = time.monotonic()
        root = Path(root_path).resolve()

        # Validate root path
        if not root.exists():
            raise ValueError(f"Path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {root}")

        # Initialize result
        result = FolderScanResult(
            root_path=str(root_path),
            root_absolute_path=str(root),
        )

        # Reset size filter for new scan
        self._size_filter.reset()

        try:
            # Scan with timeout
            await asyncio.wait_for(
                self._scan_directory(root, root, 0, result),
                timeout=self.config.scan_timeout_seconds,
            )
        except asyncio.TimeoutError:
            result.errors.append(f"Scan timed out after {self.config.scan_timeout_seconds} seconds")

        # Calculate final statistics
        result.scan_duration_ms = (time.monotonic() - start_time) * 1000

        return result

    async def _scan_directory(
        self,
        current_dir: Path,
        root: Path,
        depth: int,
        result: FolderScanResult,
    ) -> None:
        """
        Recursively scan a directory.

        Args:
            current_dir: Current directory being scanned
            root: Root directory (for relative path calculation)
            depth: Current depth in the directory tree
            result: Result object to populate
        """
        # Check depth limit
        if self.config.max_depth >= 0 and depth > self.config.max_depth:
            result.warnings.append(
                f"Max depth ({self.config.max_depth}) reached at: {current_dir.relative_to(root)}"
            )
            return

        result.directories_scanned += 1
        if depth > result.max_depth_reached:
            result.max_depth_reached = depth

        # Get directory contents
        try:
            entries = list(current_dir.iterdir())
        except PermissionError:
            result.warnings.append(f"Permission denied: {current_dir.relative_to(root)}")
            return
        except OSError as e:
            result.errors.append(f"Error reading directory {current_dir}: {e}")
            return

        # Sort for consistent ordering
        entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

        for entry in entries:
            # Yield control to event loop periodically
            await asyncio.sleep(0)

            try:
                rel_path = str(entry.relative_to(root))

                # Handle symlinks
                if entry.is_symlink():
                    if not self.config.follow_symlinks:
                        result.excluded_files.append(
                            ExcludedFile(
                                path=rel_path,
                                reason=ExclusionReason.SYMLINK,
                                details="Symlinks are not followed (enable with follow_symlinks=True)",
                            )
                        )
                        continue
                    else:
                        # Validate symlink target is within root
                        try:
                            target = entry.resolve()
                            target.relative_to(root)
                        except ValueError:
                            result.warnings.append(f"Symlink points outside root: {rel_path}")
                            result.excluded_files.append(
                                ExcludedFile(
                                    path=rel_path,
                                    reason=ExclusionReason.SYMLINK,
                                    details="Symlink target is outside root directory",
                                )
                            )
                            continue

                # Handle directories
                if entry.is_dir():
                    # Check if directory should be excluded
                    is_excluded, pattern = self._pattern_matcher.is_directory_excluded(rel_path)
                    if is_excluded:
                        result.excluded_files.append(
                            ExcludedFile(
                                path=rel_path + "/",
                                reason=ExclusionReason.PATTERN,
                                details=f"Matched pattern: {pattern}",
                            )
                        )
                        continue

                    # Recurse into directory
                    await self._scan_directory(entry, root, depth + 1, result)

                # Handle files
                elif entry.is_file():
                    await self._process_file(entry, root, rel_path, result)

            except PermissionError:
                result.excluded_files.append(
                    ExcludedFile(
                        path=str(entry.relative_to(root)),
                        reason=ExclusionReason.PERMISSION,
                        details="Permission denied",
                    )
                )
                result.files_excluded_by_permission += 1
            except OSError as e:
                result.errors.append(f"Error processing {entry}: {e}")

    async def _process_file(
        self,
        file_path: Path,
        root: Path,
        rel_path: str,
        result: FolderScanResult,
    ) -> None:
        """
        Process a single file for inclusion/exclusion.

        Args:
            file_path: Absolute path to the file
            root: Root directory
            rel_path: Relative path from root
            result: Result object to populate
        """
        result.total_files_found += 1

        # Get file size
        try:
            size = file_path.stat().st_size
            result.total_size_bytes += size
        except OSError as e:
            result.excluded_files.append(
                ExcludedFile(
                    path=rel_path,
                    reason=ExclusionReason.PERMISSION,
                    details=f"Could not stat file: {e}",
                )
            )
            result.files_excluded_by_permission += 1
            return

        # Check pattern exclusion
        is_excluded, pattern = self._pattern_matcher.is_excluded(rel_path)
        if is_excluded:
            result.excluded_files.append(
                ExcludedFile(
                    path=rel_path,
                    reason=ExclusionReason.PATTERN,
                    details=(
                        f"Matched pattern: {pattern}" if pattern else "No include pattern matched"
                    ),
                )
            )
            result.files_excluded_by_pattern += 1
            return

        # Check size limits
        size_excluded, size_reason = self._size_filter.check_file(size)
        if size_excluded:
            reason_text = size_reason or "Size limit exceeded"
            if "count" in reason_text.lower():
                result.excluded_files.append(
                    ExcludedFile(
                        path=rel_path,
                        reason=ExclusionReason.COUNT,
                        details=reason_text,
                    )
                )
                result.files_excluded_by_count += 1
            else:
                result.excluded_files.append(
                    ExcludedFile(
                        path=rel_path,
                        reason=ExclusionReason.SIZE,
                        details=reason_text,
                    )
                )
                result.files_excluded_by_size += 1
            return

        # File passes all filters - include it
        self._size_filter.accept_file(size)

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))

        result.included_files.append(
            FileInfo(
                path=rel_path,
                absolute_path=str(file_path),
                size_bytes=size,
                extension=file_path.suffix.lower(),
                mime_type=mime_type or "application/octet-stream",
            )
        )

    async def scan_iter(self, root_path: str | Path) -> AsyncIterator[FileInfo | ExcludedFile]:
        """
        Scan a folder and yield files as they are discovered.

        This is useful for progress updates during scanning.

        Args:
            root_path: Path to the folder to scan

        Yields:
            FileInfo for included files, ExcludedFile for excluded files
        """
        root = Path(root_path).resolve()

        if not root.exists():
            raise ValueError(f"Path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {root}")

        self._size_filter.reset()

        async for item in self._scan_directory_iter(root, root, 0):
            yield item

    async def _scan_directory_iter(
        self,
        current_dir: Path,
        root: Path,
        depth: int,
    ) -> AsyncIterator[FileInfo | ExcludedFile]:
        """
        Recursively scan a directory and yield files.

        Args:
            current_dir: Current directory being scanned
            root: Root directory
            depth: Current depth

        Yields:
            FileInfo or ExcludedFile objects
        """
        if self.config.max_depth >= 0 and depth > self.config.max_depth:
            return

        try:
            entries = list(current_dir.iterdir())
        except (PermissionError, OSError):
            return

        entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

        for entry in entries:
            await asyncio.sleep(0)

            try:
                rel_path = str(entry.relative_to(root))

                if entry.is_symlink() and not self.config.follow_symlinks:
                    yield ExcludedFile(
                        path=rel_path,
                        reason=ExclusionReason.SYMLINK,
                        details="Symlinks not followed",
                    )
                    continue

                if entry.is_dir():
                    is_excluded, pattern = self._pattern_matcher.is_directory_excluded(rel_path)
                    if is_excluded:
                        yield ExcludedFile(
                            path=rel_path + "/",
                            reason=ExclusionReason.PATTERN,
                            details=f"Matched: {pattern}",
                        )
                        continue

                    async for item in self._scan_directory_iter(entry, root, depth + 1):
                        yield item

                elif entry.is_file():
                    size = entry.stat().st_size

                    is_excluded, pattern = self._pattern_matcher.is_excluded(rel_path)
                    if is_excluded:
                        yield ExcludedFile(
                            path=rel_path,
                            reason=ExclusionReason.PATTERN,
                            details=f"Matched: {pattern}" if pattern else "No include match",
                        )
                        continue

                    size_excluded, reason = self._size_filter.check_file(size)
                    if size_excluded:
                        reason_text = reason or "Size limit exceeded"
                        yield ExcludedFile(
                            path=rel_path,
                            reason=(
                                ExclusionReason.SIZE
                                if "size" in reason_text.lower()
                                else ExclusionReason.COUNT
                            ),
                            details=reason_text,
                        )
                        continue

                    self._size_filter.accept_file(size)
                    mime_type, _ = mimetypes.guess_type(str(entry))

                    yield FileInfo(
                        path=rel_path,
                        absolute_path=str(entry),
                        size_bytes=size,
                        extension=entry.suffix.lower(),
                        mime_type=mime_type or "application/octet-stream",
                    )

            except (PermissionError, OSError):
                yield ExcludedFile(
                    path=str(entry.relative_to(root)),
                    reason=ExclusionReason.PERMISSION,
                    details="Permission denied",
                )


def get_folder_scanner(config: Optional[FolderUploadConfig] = None) -> FolderScanner:
    """
    Get a folder scanner instance.

    Args:
        config: Optional configuration

    Returns:
        FolderScanner instance
    """
    return FolderScanner(config)
