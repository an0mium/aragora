"""
Tests for aragora.cli.documents module.

Tests document CLI commands: upload, list, show.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.documents import (
    cmd_list,
    cmd_show,
    cmd_upload,
    create_documents_parser,
    documents_cli,
)


# ===========================================================================
# Test Fixtures and Mock Classes
# ===========================================================================


@dataclass
class MockScanResult:
    """Mock folder scan result."""

    total_files_found: int = 15
    included_count: int = 10
    included_size_bytes: int = 1024 * 1024 * 5  # 5MB
    excluded_count: int = 5
    files_excluded_by_pattern: int = 3
    files_excluded_by_size: int = 1
    files_excluded_by_count: int = 1
    included_files: list = field(default_factory=list)

    def to_dict(self):
        return {
            "total_files_found": self.total_files_found,
            "included_count": self.included_count,
            "excluded_count": self.excluded_count,
        }


@dataclass
class MockFileInfo:
    """Mock file info from scan."""

    absolute_path: str = "/tmp/test/file.pdf"


@dataclass
class MockDocument:
    """Mock document."""

    id: str = "doc-123"


@dataclass
class MockJobResult:
    """Mock job result."""

    document: MockDocument = field(default_factory=MockDocument)


# ===========================================================================
# Tests: create_documents_parser
# ===========================================================================


class TestCreateDocumentsParser:
    """Tests for create_documents_parser function."""

    def test_creates_documents_subparser(self):
        """Test that documents parser is created with subcommands."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_documents_parser(subparsers)

        # Parse upload command
        args = parser.parse_args(["documents", "upload", "file.pdf"])
        assert args.paths == ["file.pdf"]
        assert args.recursive is False
        assert args.dry_run is False

    def test_upload_with_recursive_option(self):
        """Test upload command with recursive option."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_documents_parser(subparsers)

        args = parser.parse_args(["documents", "upload", "./folder", "-r"])
        assert args.recursive is True

    def test_upload_with_all_options(self):
        """Test upload command with all options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_documents_parser(subparsers)

        args = parser.parse_args(
            [
                "documents",
                "upload",
                "file1.pdf",
                "file2.pdf",
                "-r",
                "--max-depth",
                "5",
                "--exclude",
                "*.log",
                "--exclude",
                "*.tmp",
                "--include",
                "*.pdf",
                "--max-size",
                "1gb",
                "--max-file-size",
                "50mb",
                "--max-files",
                "500",
                "--agent-filter",
                "--filter-prompt",
                "Only relevant docs",
                "--filter-model",
                "gpt-4",
                "--dry-run",
                "--config",
                "config.yaml",
                "--follow-symlinks",
                "--json",
            ]
        )
        assert args.paths == ["file1.pdf", "file2.pdf"]
        assert args.recursive is True
        assert args.max_depth == 5
        assert args.exclude == ["*.log", "*.tmp"]
        assert args.include == ["*.pdf"]
        assert args.max_size == "1gb"
        assert args.max_file_size == "50mb"
        assert args.max_files == 500
        assert args.agent_filter is True
        assert args.filter_prompt == "Only relevant docs"
        assert args.filter_model == "gpt-4"
        assert args.dry_run is True
        assert args.config == "config.yaml"
        assert args.follow_symlinks is True
        assert args.json_output is True

    def test_list_command_options(self):
        """Test list command with options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_documents_parser(subparsers)

        args = parser.parse_args(["documents", "list", "--limit", "100", "--json"])
        assert args.limit == 100
        assert args.json_output is True

    def test_list_command_default_limit(self):
        """Test list command default limit."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_documents_parser(subparsers)

        args = parser.parse_args(["documents", "list"])
        assert args.limit == 50

    def test_show_command_options(self):
        """Test show command with options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_documents_parser(subparsers)

        args = parser.parse_args(["documents", "show", "doc-123", "--chunks", "--json"])
        assert args.doc_id == "doc-123"
        assert args.chunks is True
        assert args.json_output is True


# ===========================================================================
# Tests: documents_cli
# ===========================================================================


class TestDocumentsCli:
    """Tests for documents_cli function."""

    def test_upload_command_dispatches(self):
        """Test upload command dispatches to cmd_upload."""
        args = argparse.Namespace()
        args.doc_command = "upload"
        args.paths = []

        with patch("aragora.cli.documents.cmd_upload", return_value=0) as mock_cmd:
            result = documents_cli(args)

        assert result == 0
        mock_cmd.assert_called_once_with(args)

    def test_list_command_dispatches(self):
        """Test list command dispatches to cmd_list."""
        args = argparse.Namespace()
        args.doc_command = "list"

        with patch("aragora.cli.documents.cmd_list", return_value=0) as mock_cmd:
            result = documents_cli(args)

        assert result == 0
        mock_cmd.assert_called_once_with(args)

    def test_show_command_dispatches(self):
        """Test show command dispatches to cmd_show."""
        args = argparse.Namespace()
        args.doc_command = "show"
        args.doc_id = "doc-123"

        with patch("aragora.cli.documents.cmd_show", return_value=0) as mock_cmd:
            result = documents_cli(args)

        assert result == 0
        mock_cmd.assert_called_once_with(args)

    def test_unknown_command_returns_error(self, capsys):
        """Test unknown command returns error."""
        args = argparse.Namespace()
        args.doc_command = "unknown"

        result = documents_cli(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown documents command" in captured.out


# ===========================================================================
# Tests: cmd_list
# ===========================================================================


class TestCmdList:
    """Tests for cmd_list function."""

    @pytest.fixture
    def list_args(self):
        """Create base list args."""
        args = argparse.Namespace()
        args.limit = 50
        args.json_output = False
        return args

    def test_list_text_output(self, list_args, capsys):
        """Test list with text output."""
        result = cmd_list(list_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "DOCUMENTS LIST" in captured.out
        assert "Connect to server" in captured.out

    def test_list_json_output(self, list_args, capsys):
        """Test list with JSON output."""
        list_args.json_output = True

        result = cmd_list(list_args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["documents"] == []
        assert output["total"] == 0
        assert "Connect to server" in output["message"]


# ===========================================================================
# Tests: cmd_show
# ===========================================================================


class TestCmdShow:
    """Tests for cmd_show function."""

    @pytest.fixture
    def show_args(self):
        """Create base show args."""
        args = argparse.Namespace()
        args.doc_id = "doc-123"
        args.json_output = False
        args.chunks = False
        return args

    def test_show_text_output(self, show_args, capsys):
        """Test show with text output."""
        result = cmd_show(show_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "DOCUMENT: doc-123" in captured.out
        assert "Connect to server" in captured.out

    def test_show_json_output(self, show_args, capsys):
        """Test show with JSON output."""
        show_args.json_output = True

        result = cmd_show(show_args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["doc_id"] == "doc-123"
        assert "Connect to server" in output["message"]

    def test_show_with_chunks(self, show_args, capsys):
        """Test show with chunks option."""
        show_args.chunks = True

        result = cmd_show(show_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "CHUNKS:" in captured.out


# ===========================================================================
# Tests: cmd_upload
# ===========================================================================


class TestCmdUpload:
    """Tests for cmd_upload function."""

    @pytest.fixture
    def upload_args(self, tmp_path):
        """Create base upload args with temp file."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"test content")

        args = argparse.Namespace()
        args.paths = [str(test_file)]
        args.recursive = False
        args.max_depth = 10
        args.exclude = []
        args.include = []
        args.max_size = "500mb"
        args.max_file_size = "100mb"
        args.max_files = 1000
        args.agent_filter = False
        args.filter_prompt = ""
        args.filter_model = "gemini-2.0-flash"
        args.dry_run = False
        args.config = None
        args.follow_symlinks = False
        args.json_output = False
        return args

    def test_upload_no_files_error(self, capsys):
        """Test upload with no files returns error."""
        args = argparse.Namespace()
        args.paths = ["/nonexistent/path"]
        args.recursive = False
        args.max_size = "500mb"
        args.max_file_size = "100mb"
        args.max_files = 1000
        args.exclude = []
        args.include = []
        args.agent_filter = False
        args.filter_prompt = ""
        args.filter_model = "gemini-2.0-flash"
        args.dry_run = False
        args.config = None
        args.follow_symlinks = False
        args.json_output = False
        args.max_depth = 10

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No files to upload" in captured.out or "not found" in captured.out

    def test_upload_dry_run_single_file(self, upload_args, capsys):
        """Test upload dry run with single file."""
        upload_args.dry_run = True

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024
        mock_folder.format_size_bytes.side_effect = lambda x: f"{x} bytes"

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(upload_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "Total files: 1" in captured.out

    def test_upload_dry_run_json_output(self, upload_args, capsys):
        """Test upload dry run with JSON output."""
        upload_args.dry_run = True
        upload_args.json_output = True

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024
        mock_folder.format_size_bytes.side_effect = lambda x: f"{x} bytes"

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(upload_args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["dry_run"] is True
        assert output["total_files"] == 1

    def test_upload_invalid_size_string(self, upload_args, capsys):
        """Test upload with invalid size string."""
        upload_args.max_size = "invalid"

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = ValueError("Invalid size format")

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(upload_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_upload_with_config_file(self, upload_args, tmp_path, capsys):
        """Test upload with config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("max_depth: 5\nmax_files: 100")
        upload_args.config = str(config_file)
        upload_args.dry_run = True

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024
        mock_folder.format_size_bytes.side_effect = lambda x: f"{x} bytes"

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(upload_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Loaded config from" in captured.out

    def test_upload_missing_config_file(self, upload_args, capsys):
        """Test upload with missing config file."""
        upload_args.config = "/nonexistent/config.yaml"
        upload_args.dry_run = True

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024
        mock_folder.format_size_bytes.side_effect = lambda x: f"{x} bytes"

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(upload_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Warning: Config file not found" in captured.out

    def test_upload_directory_without_recursive(self, upload_args, tmp_path, capsys):
        """Test uploading directory without -r flag."""
        test_dir = tmp_path / "folder"
        test_dir.mkdir()
        upload_args.paths = [str(test_dir)]
        upload_args.recursive = False

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(upload_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Skipping directory" in captured.out or "No files to upload" in captured.out

    def test_upload_directory_recursive(self, upload_args, tmp_path, capsys):
        """Test uploading directory with -r flag."""
        test_dir = tmp_path / "folder"
        test_dir.mkdir()
        test_file = test_dir / "doc.pdf"
        test_file.write_bytes(b"test")
        upload_args.paths = [str(test_dir)]
        upload_args.recursive = True
        upload_args.dry_run = True

        mock_result = MockScanResult(included_files=[MockFileInfo(absolute_path=str(test_file))])
        mock_scanner = MagicMock()
        mock_scanner.scan = AsyncMock(return_value=mock_result)

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.FolderScanner.return_value = mock_scanner
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024
        mock_folder.format_size_bytes.side_effect = lambda x: f"{x // 1024}KB"

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(upload_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Scanning" in captured.out
        assert "DRY RUN" in captured.out

    def test_upload_actual_success(self, upload_args, capsys):
        """Test actual upload success."""
        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024

        mock_job_result = MockJobResult()
        mock_processor = MagicMock()
        mock_processor.submit = AsyncMock(return_value="job-123")
        mock_processor.wait_for_job = AsyncMock(return_value=mock_job_result)

        mock_ingestion = MagicMock()
        mock_ingestion.get_batch_processor = AsyncMock(return_value=mock_processor)

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            with patch.dict("sys.modules", {"aragora.documents.ingestion": mock_ingestion}):
                result = cmd_upload(upload_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Uploading" in captured.out
        assert "succeeded" in captured.out

    def test_upload_actual_json_output(self, upload_args, capsys):
        """Test actual upload with JSON output."""
        upload_args.json_output = True

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024

        mock_job_result = MockJobResult()
        mock_processor = MagicMock()
        mock_processor.submit = AsyncMock(return_value="job-123")
        mock_processor.wait_for_job = AsyncMock(return_value=mock_job_result)

        mock_ingestion = MagicMock()
        mock_ingestion.get_batch_processor = AsyncMock(return_value=mock_processor)

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            with patch.dict("sys.modules", {"aragora.documents.ingestion": mock_ingestion}):
                result = cmd_upload(upload_args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["total_files"] == 1
        assert output["successful"] == 1
        assert output["failed"] == 0

    def test_upload_with_failed_file(self, upload_args, capsys):
        """Test upload with a file that fails."""
        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024

        mock_processor = MagicMock()
        mock_processor.submit = AsyncMock(return_value="job-123")
        mock_processor.wait_for_job = AsyncMock(return_value=None)  # No result

        mock_ingestion = MagicMock()
        mock_ingestion.get_batch_processor = AsyncMock(return_value=mock_processor)

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            with patch.dict("sys.modules", {"aragora.documents.ingestion": mock_ingestion}):
                result = cmd_upload(upload_args)

        assert result == 1  # Failed upload
        captured = capsys.readouterr()
        assert "FAILED" in captured.out or "failed" in captured.out

    def test_upload_with_exception(self, upload_args, capsys):
        """Test upload when exception occurs."""
        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024

        mock_processor = MagicMock()
        mock_processor.submit = AsyncMock(side_effect=Exception("Upload failed"))

        mock_ingestion = MagicMock()
        mock_ingestion.get_batch_processor = AsyncMock(return_value=mock_processor)

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            with patch.dict("sys.modules", {"aragora.documents.ingestion": mock_ingestion}):
                result = cmd_upload(upload_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.out or "failed" in captured.out

    def test_upload_glob_pattern(self, tmp_path, capsys):
        """Test upload with glob pattern."""
        # Create a separate subdirectory for glob test
        glob_dir = tmp_path / "glob_test"
        glob_dir.mkdir()
        (glob_dir / "file1.txt").write_bytes(b"test1")
        (glob_dir / "file2.txt").write_bytes(b"test2")

        args = argparse.Namespace()
        args.paths = [str(glob_dir / "*.txt")]
        args.recursive = False
        args.max_depth = 10
        args.exclude = []
        args.include = []
        args.max_size = "500mb"
        args.max_file_size = "100mb"
        args.max_files = 1000
        args.agent_filter = False
        args.filter_prompt = ""
        args.filter_model = "gemini-2.0-flash"
        args.dry_run = True
        args.config = None
        args.follow_symlinks = False
        args.json_output = False

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024
        mock_folder.format_size_bytes.side_effect = lambda x: f"{x} bytes"

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "Total files: 2" in captured.out

    def test_upload_removes_duplicates(self, upload_args, tmp_path, capsys):
        """Test upload removes duplicate files."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"test")

        # Same file specified multiple times
        upload_args.paths = [str(test_file), str(test_file)]
        upload_args.dry_run = True

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024
        mock_folder.format_size_bytes.side_effect = lambda x: f"{x} bytes"

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(upload_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Total files: 1" in captured.out  # Deduplicated

    def test_upload_many_files_truncated_display(self, upload_args, tmp_path, capsys):
        """Test upload display is truncated for many files."""
        # Create 25 files
        files = []
        for i in range(25):
            f = tmp_path / f"file{i:02d}.pdf"
            f.write_bytes(b"test")
            files.append(str(f))

        upload_args.paths = files
        upload_args.dry_run = True

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024
        mock_folder.format_size_bytes.side_effect = lambda x: f"{x} bytes"

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(upload_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "more files" in captured.out  # Truncated display

    def test_upload_scan_error(self, upload_args, tmp_path, capsys):
        """Test upload when folder scan fails."""
        test_dir = tmp_path / "folder"
        test_dir.mkdir()
        upload_args.paths = [str(test_dir)]
        upload_args.recursive = True

        mock_scanner = MagicMock()
        mock_scanner.scan = AsyncMock(side_effect=ValueError("Scan failed"))

        mock_folder = MagicMock()
        mock_folder.FolderUploadConfig.return_value = MagicMock(exclude_patterns=[])
        mock_folder.FolderScanner.return_value = mock_scanner
        mock_folder.parse_size_string.side_effect = lambda x: 500 * 1024 * 1024

        with patch.dict("sys.modules", {"aragora.documents.folder": mock_folder}):
            result = cmd_upload(upload_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error scanning" in captured.out
