"""
Tests for folder upload configuration module.

Tests cover:
- ExclusionReason enum values
- FolderUploadConfig defaults and computed properties
- FileInfo dataclass and size_mb property
- ExcludedFile dataclass
- FolderScanResult properties and to_dict serialization
- FolderUploadProgress properties
- FolderUploadResult properties
"""

from __future__ import annotations

from aragora.documents.folder.config import (
    ExcludedFile,
    ExclusionReason,
    FileInfo,
    FolderScanResult,
    FolderUploadConfig,
    FolderUploadProgress,
    FolderUploadResult,
)


class TestExclusionReason:
    """Tests for ExclusionReason enum."""

    def test_all_values(self):
        assert ExclusionReason.PATTERN.value == "pattern"
        assert ExclusionReason.SIZE.value == "size"
        assert ExclusionReason.COUNT.value == "count"
        assert ExclusionReason.AGENT.value == "agent"
        assert ExclusionReason.PERMISSION.value == "permission"
        assert ExclusionReason.SYMLINK.value == "symlink"
        assert ExclusionReason.DEPTH.value == "depth"

    def test_member_count(self):
        assert len(ExclusionReason) == 7


class TestFolderUploadConfig:
    """Tests for FolderUploadConfig dataclass."""

    def test_defaults(self):
        config = FolderUploadConfig()
        assert config.max_depth == 10
        assert config.follow_symlinks is False
        assert config.max_file_size_mb == 100
        assert config.max_total_size_mb == 500
        assert config.max_file_count == 1000
        assert config.enable_agent_filter is False
        assert config.chunking_strategy == "auto"
        assert config.preserve_folder_structure is True
        assert config.scan_timeout_seconds == 300.0

    def test_default_exclude_patterns(self):
        config = FolderUploadConfig()
        assert "**/.git/**" in config.exclude_patterns
        assert "**/node_modules/**" in config.exclude_patterns
        assert "**/__pycache__/**" in config.exclude_patterns
        assert "**/.venv/**" in config.exclude_patterns

    def test_custom_values(self):
        config = FolderUploadConfig(
            max_depth=5,
            follow_symlinks=True,
            max_file_size_mb=50,
            max_total_size_mb=200,
            max_file_count=500,
            enable_agent_filter=True,
        )
        assert config.max_depth == 5
        assert config.follow_symlinks is True
        assert config.max_file_size_mb == 50
        assert config.max_total_size_mb == 200
        assert config.max_file_count == 500
        assert config.enable_agent_filter is True

    def test_max_file_size_bytes(self):
        config = FolderUploadConfig(max_file_size_mb=100)
        assert config.max_file_size_bytes() == 100 * 1024 * 1024

    def test_max_total_size_bytes(self):
        config = FolderUploadConfig(max_total_size_mb=500)
        assert config.max_total_size_bytes() == 500 * 1024 * 1024

    def test_empty_include_patterns_by_default(self):
        config = FolderUploadConfig()
        assert config.include_patterns == []


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_creation(self):
        info = FileInfo(
            path="docs/readme.md",
            absolute_path="/home/user/project/docs/readme.md",
            size_bytes=1024,
            extension=".md",
            mime_type="text/markdown",
        )
        assert info.path == "docs/readme.md"
        assert info.absolute_path == "/home/user/project/docs/readme.md"
        assert info.size_bytes == 1024
        assert info.extension == ".md"
        assert info.mime_type == "text/markdown"

    def test_default_mime_type(self):
        info = FileInfo(
            path="file.unknown",
            absolute_path="/file.unknown",
            size_bytes=100,
            extension=".unknown",
        )
        assert info.mime_type == "application/octet-stream"

    def test_size_mb(self):
        info = FileInfo(
            path="large.bin",
            absolute_path="/large.bin",
            size_bytes=5 * 1024 * 1024,  # 5 MB
            extension=".bin",
        )
        assert info.size_mb() == 5.0

    def test_size_mb_small_file(self):
        info = FileInfo(
            path="tiny.txt",
            absolute_path="/tiny.txt",
            size_bytes=512,
            extension=".txt",
        )
        assert info.size_mb() < 0.001


class TestExcludedFile:
    """Tests for ExcludedFile dataclass."""

    def test_creation(self):
        excluded = ExcludedFile(
            path="node_modules/package.json",
            reason=ExclusionReason.PATTERN,
            details="Matched pattern: **/node_modules/**",
        )
        assert excluded.path == "node_modules/package.json"
        assert excluded.reason == ExclusionReason.PATTERN
        assert excluded.details == "Matched pattern: **/node_modules/**"

    def test_different_reasons(self):
        for reason in ExclusionReason:
            excluded = ExcludedFile(
                path="test.txt",
                reason=reason,
                details=f"Excluded by {reason.value}",
            )
            assert excluded.reason == reason


class TestFolderScanResult:
    """Tests for FolderScanResult dataclass."""

    def test_creation_defaults(self):
        result = FolderScanResult(
            root_path="./project",
            root_absolute_path="/home/user/project",
        )
        assert result.root_path == "./project"
        assert result.total_files_found == 0
        assert result.total_size_bytes == 0
        assert result.files_excluded_by_pattern == 0
        assert result.directories_scanned == 0
        assert result.included_files == []
        assert result.excluded_files == []
        assert result.warnings == []
        assert result.errors == []

    def test_included_count(self):
        result = FolderScanResult(
            root_path="./",
            root_absolute_path="/",
        )
        result.included_files = [
            FileInfo(path="a.txt", absolute_path="/a.txt", size_bytes=100, extension=".txt"),
            FileInfo(path="b.txt", absolute_path="/b.txt", size_bytes=200, extension=".txt"),
        ]
        assert result.included_count == 2

    def test_excluded_count(self):
        result = FolderScanResult(root_path="./", root_absolute_path="/")
        result.excluded_files = [
            ExcludedFile(path="x.tmp", reason=ExclusionReason.PATTERN, details="excluded"),
        ]
        assert result.excluded_count == 1

    def test_included_size_bytes(self):
        result = FolderScanResult(root_path="./", root_absolute_path="/")
        result.included_files = [
            FileInfo(path="a.txt", absolute_path="/a.txt", size_bytes=1000, extension=".txt"),
            FileInfo(path="b.txt", absolute_path="/b.txt", size_bytes=2000, extension=".txt"),
        ]
        assert result.included_size_bytes == 3000

    def test_included_size_mb(self):
        result = FolderScanResult(root_path="./", root_absolute_path="/")
        result.included_files = [
            FileInfo(
                path="big.bin",
                absolute_path="/big.bin",
                size_bytes=2 * 1024 * 1024,
                extension=".bin",
            ),
        ]
        assert result.included_size_mb == 2.0

    def test_to_dict(self):
        result = FolderScanResult(root_path="./proj", root_absolute_path="/home/proj")
        result.total_files_found = 10
        result.directories_scanned = 3
        result.included_files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=500, extension=".py"),
        ]
        result.excluded_files = [
            ExcludedFile(path=".git/HEAD", reason=ExclusionReason.PATTERN, details="git"),
        ]
        result.scan_duration_ms = 123.45

        data = result.to_dict()

        assert data["root_path"] == "./proj"
        assert data["root_absolute_path"] == "/home/proj"
        assert data["statistics"]["total_files_found"] == 10
        assert data["statistics"]["directories_scanned"] == 3
        assert data["statistics"]["included_count"] == 1
        assert len(data["included_files"]) == 1
        assert data["included_files"][0]["path"] == "a.py"
        assert len(data["excluded_files"]) == 1
        assert data["excluded_files"][0]["reason"] == "pattern"
        assert data["scan_duration_ms"] == 123.45


class TestFolderUploadProgress:
    """Tests for FolderUploadProgress dataclass."""

    def test_creation(self):
        progress = FolderUploadProgress(
            total_files=100,
            uploaded_files=50,
            failed_files=2,
        )
        assert progress.total_files == 100
        assert progress.uploaded_files == 50
        assert progress.failed_files == 2
        assert progress.current_file is None
        assert progress.bytes_uploaded == 0
        assert progress.total_bytes == 0

    def test_progress_percent(self):
        progress = FolderUploadProgress(
            total_files=100,
            uploaded_files=25,
            failed_files=0,
        )
        assert progress.progress_percent == 25.0

    def test_progress_percent_empty(self):
        progress = FolderUploadProgress(
            total_files=0,
            uploaded_files=0,
            failed_files=0,
        )
        assert progress.progress_percent == 100.0

    def test_progress_percent_complete(self):
        progress = FolderUploadProgress(
            total_files=10,
            uploaded_files=10,
            failed_files=0,
        )
        assert progress.progress_percent == 100.0


class TestFolderUploadResult:
    """Tests for FolderUploadResult dataclass."""

    def test_creation(self):
        scan_result = FolderScanResult(root_path="./", root_absolute_path="/")
        result = FolderUploadResult(
            folder_id="folder-123",
            scan_result=scan_result,
        )
        assert result.folder_id == "folder-123"
        assert result.files_uploaded == 0
        assert result.files_failed == 0
        assert result.document_ids == []
        assert result.upload_errors == []
        assert result.upload_duration_ms == 0.0

    def test_success_no_failures(self):
        scan_result = FolderScanResult(root_path="./", root_absolute_path="/")
        result = FolderUploadResult(
            folder_id="f-1",
            scan_result=scan_result,
            files_uploaded=10,
            files_failed=0,
        )
        assert result.success is True

    def test_success_with_failures(self):
        scan_result = FolderScanResult(root_path="./", root_absolute_path="/")
        result = FolderUploadResult(
            folder_id="f-2",
            scan_result=scan_result,
            files_uploaded=8,
            files_failed=2,
        )
        assert result.success is False

    def test_success_all_failed(self):
        scan_result = FolderScanResult(root_path="./", root_absolute_path="/")
        result = FolderUploadResult(
            folder_id="f-3",
            scan_result=scan_result,
            files_uploaded=0,
            files_failed=5,
        )
        assert result.success is False
