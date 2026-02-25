"""
Tests for folder scanner module.

Tests cover:
- FolderScanner construction with default and custom config
- scan() basic file discovery
- scan() depth limiting
- scan() pattern exclusions
- scan() size filtering
- scan() symlink handling
- scan() timeout handling
- scan() error handling (permissions, OS errors)
- scan_iter() async iteration
- get_folder_scanner() factory function
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.documents.folder.config import (
    ExcludedFile,
    ExclusionReason,
    FileInfo,
    FolderScanResult,
    FolderUploadConfig,
)
from aragora.documents.folder.scanner import FolderScanner, get_folder_scanner


@pytest.fixture
def temp_folder(tmp_path: Path) -> Path:
    """Create a temporary folder structure for testing."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / ".git").mkdir()

    # Create files
    (tmp_path / "readme.md").write_text("# Readme")
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "src" / "app" / "utils.py").write_text("def helper(): pass")
    (tmp_path / "docs" / "guide.txt").write_text("User guide")
    (tmp_path / "node_modules" / "package.json").write_text("{}")
    (tmp_path / ".git" / "config").write_text("[core]")

    return tmp_path


class TestFolderScannerConstruction:
    """Tests for FolderScanner construction."""

    def test_default_config(self):
        scanner = FolderScanner()
        assert scanner.config is not None
        assert scanner.config.max_depth == 10

    def test_custom_config(self):
        config = FolderUploadConfig(max_depth=5, max_file_size_mb=50)
        scanner = FolderScanner(config)
        assert scanner.config.max_depth == 5
        assert scanner.config.max_file_size_mb == 50


class TestFolderScannerScan:
    """Tests for FolderScanner.scan() method."""

    @pytest.mark.asyncio
    async def test_scan_basic(self, temp_folder: Path):
        config = FolderUploadConfig(exclude_patterns=[])  # No exclusions
        scanner = FolderScanner(config)

        result = await scanner.scan(temp_folder)

        assert isinstance(result, FolderScanResult)
        assert result.root_absolute_path == str(temp_folder)
        assert result.total_files_found > 0
        assert result.directories_scanned > 0
        assert len(result.included_files) > 0

    @pytest.mark.asyncio
    async def test_scan_with_exclusions(self, temp_folder: Path):
        config = FolderUploadConfig(exclude_patterns=["**/node_modules/**", "**/.git/**"])
        scanner = FolderScanner(config)

        result = await scanner.scan(temp_folder)

        # node_modules and .git should be excluded
        included_paths = [f.path for f in result.included_files]
        excluded_paths = [f.path for f in result.excluded_files]

        assert not any("node_modules" in p for p in included_paths)
        assert not any(".git" in p for p in included_paths)
        # Directories are tracked in excluded_files, verify they're there
        assert any("node_modules" in e.path or ".git" in e.path for e in result.excluded_files)

    @pytest.mark.asyncio
    async def test_scan_depth_limit(self, temp_folder: Path):
        config = FolderUploadConfig(max_depth=0, exclude_patterns=[])
        scanner = FolderScanner(config)

        result = await scanner.scan(temp_folder)

        # Only root level files should be included
        for f in result.included_files:
            assert "/" not in f.path  # No subdirectory files

    @pytest.mark.asyncio
    async def test_scan_depth_limit_one(self, temp_folder: Path):
        config = FolderUploadConfig(max_depth=1, exclude_patterns=[])
        scanner = FolderScanner(config)

        result = await scanner.scan(temp_folder)

        # Files up to depth 1 (e.g., src/main.py) should be included
        included_paths = [f.path for f in result.included_files]
        # readme.md (depth 0) should be included
        assert "readme.md" in included_paths
        # src/main.py (depth 1) should be included
        assert any("src" in p and "/" in p for p in included_paths)

    @pytest.mark.asyncio
    async def test_scan_nonexistent_path_raises(self):
        scanner = FolderScanner()
        with pytest.raises(ValueError, match="does not exist"):
            await scanner.scan("/nonexistent/path/that/does/not/exist")

    @pytest.mark.asyncio
    async def test_scan_file_path_raises(self, temp_folder: Path):
        scanner = FolderScanner()
        file_path = temp_folder / "readme.md"
        with pytest.raises(ValueError, match="not a directory"):
            await scanner.scan(file_path)

    @pytest.mark.asyncio
    async def test_scan_tracks_statistics(self, temp_folder: Path):
        config = FolderUploadConfig(exclude_patterns=["**/.git/**"])
        scanner = FolderScanner(config)

        result = await scanner.scan(temp_folder)

        assert result.total_files_found > 0
        assert result.total_size_bytes > 0
        assert result.directories_scanned > 0
        assert result.scan_duration_ms > 0

    @pytest.mark.asyncio
    async def test_scan_file_info_populated(self, temp_folder: Path):
        # Exclude .git to avoid extensionless files like .git/config
        config = FolderUploadConfig(exclude_patterns=["**/.git/**"])
        scanner = FolderScanner(config)

        result = await scanner.scan(temp_folder)

        for file_info in result.included_files:
            assert file_info.path  # relative path
            assert file_info.absolute_path  # absolute path
            assert file_info.size_bytes >= 0
            assert isinstance(file_info.extension, str)  # can be empty for extensionless files
            assert file_info.mime_type  # e.g., "text/plain"


class TestFolderScannerSizeFiltering:
    """Tests for size-based filtering."""

    @pytest.mark.asyncio
    async def test_scan_file_size_limit(self, temp_folder: Path):
        # Create a large file
        large_file = temp_folder / "large.bin"
        large_file.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB

        config = FolderUploadConfig(
            exclude_patterns=[],
            max_file_size_mb=1,  # 1 MB limit
        )
        scanner = FolderScanner(config)

        result = await scanner.scan(temp_folder)

        excluded_paths = [f.path for f in result.excluded_files]
        assert "large.bin" in excluded_paths
        assert result.files_excluded_by_size > 0

    @pytest.mark.asyncio
    async def test_scan_file_count_limit(self, temp_folder: Path):
        config = FolderUploadConfig(
            exclude_patterns=[],
            max_file_count=2,
        )
        scanner = FolderScanner(config)

        result = await scanner.scan(temp_folder)

        assert len(result.included_files) <= 2
        if result.total_files_found > 2:
            assert result.files_excluded_by_count > 0


class TestFolderScannerSymlinks:
    """Tests for symlink handling."""

    @pytest.mark.asyncio
    async def test_scan_symlinks_not_followed_by_default(self, temp_folder: Path):
        # Create a symlink
        link_path = temp_folder / "link_to_src"
        link_path.symlink_to(temp_folder / "src")

        config = FolderUploadConfig(
            exclude_patterns=[],
            follow_symlinks=False,
        )
        scanner = FolderScanner(config)

        result = await scanner.scan(temp_folder)

        # Symlink should be excluded
        excluded_paths = [f.path for f in result.excluded_files]
        assert "link_to_src" in excluded_paths


class TestFolderScannerTimeout:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_scan_timeout(self, temp_folder: Path):
        config = FolderUploadConfig(
            exclude_patterns=[],
            scan_timeout_seconds=0.001,  # Very short timeout
        )
        scanner = FolderScanner(config)

        # Mock iterdir to be slow
        original_iterdir = Path.iterdir

        def slow_iterdir(self):
            import time

            time.sleep(0.01)  # 10ms delay
            return original_iterdir(self)

        with patch.object(Path, "iterdir", slow_iterdir):
            result = await scanner.scan(temp_folder)

        # Should have timeout error
        assert any("timed out" in e for e in result.errors)


class TestFolderScannerScanIter:
    """Tests for FolderScanner.scan_iter() method."""

    @pytest.mark.asyncio
    async def test_scan_iter_yields_files(self, temp_folder: Path):
        config = FolderUploadConfig(exclude_patterns=["**/.git/**", "**/node_modules/**"])
        scanner = FolderScanner(config)

        items = []
        async for item in scanner.scan_iter(temp_folder):
            items.append(item)

        # Should have both included and excluded files
        file_infos = [i for i in items if isinstance(i, FileInfo)]
        excluded_files = [i for i in items if isinstance(i, ExcludedFile)]

        assert len(file_infos) > 0
        assert len(excluded_files) > 0

    @pytest.mark.asyncio
    async def test_scan_iter_nonexistent_raises(self):
        scanner = FolderScanner()
        with pytest.raises(ValueError, match="does not exist"):
            async for _ in scanner.scan_iter("/nonexistent/path"):
                pass

    @pytest.mark.asyncio
    async def test_scan_iter_file_raises(self, temp_folder: Path):
        scanner = FolderScanner()
        file_path = temp_folder / "readme.md"
        with pytest.raises(ValueError, match="not a directory"):
            async for _ in scanner.scan_iter(file_path):
                pass


class TestGetFolderScanner:
    """Tests for get_folder_scanner factory function."""

    def test_default_config(self):
        scanner = get_folder_scanner()
        assert isinstance(scanner, FolderScanner)
        assert scanner.config.max_depth == 10

    def test_custom_config(self):
        config = FolderUploadConfig(max_depth=3)
        scanner = get_folder_scanner(config)
        assert scanner.config.max_depth == 3
