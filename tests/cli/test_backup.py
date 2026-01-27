"""Tests for CLI backup module."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.backup import (
    _format_duration,
    _format_size,
    add_backup_subparsers,
    cmd_backup,
    cmd_backup_cleanup,
    cmd_backup_create,
    cmd_backup_list,
    cmd_backup_restore,
    cmd_backup_verify,
)


class TestFormatSize:
    """Test size formatting helper."""

    def test_format_bytes(self):
        """Test formatting bytes."""
        assert _format_size(500) == "500.0 B"
        assert _format_size(0) == "0.0 B"

    def test_format_kilobytes(self):
        """Test formatting kilobytes."""
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(2048) == "2.0 KB"
        assert _format_size(1536) == "1.5 KB"

    def test_format_megabytes(self):
        """Test formatting megabytes."""
        assert _format_size(1024 * 1024) == "1.0 MB"
        assert _format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_format_gigabytes(self):
        """Test formatting gigabytes."""
        assert _format_size(1024 * 1024 * 1024) == "1.0 GB"
        assert _format_size(2.5 * 1024 * 1024 * 1024) == "2.5 GB"

    def test_format_terabytes(self):
        """Test formatting terabytes."""
        assert _format_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"


class TestFormatDuration:
    """Test duration formatting helper."""

    def test_format_seconds(self):
        """Test formatting seconds."""
        assert _format_duration(0.5) == "0.5s"
        assert _format_duration(30) == "30.0s"
        assert _format_duration(59.9) == "59.9s"

    def test_format_minutes(self):
        """Test formatting minutes."""
        assert _format_duration(60) == "1.0m"
        assert _format_duration(90) == "1.5m"
        assert _format_duration(3540) == "59.0m"

    def test_format_hours(self):
        """Test formatting hours."""
        assert _format_duration(3600) == "1.0h"
        assert _format_duration(7200) == "2.0h"
        assert _format_duration(5400) == "1.5h"


class TestCmdBackupCreate:
    """Test backup create command."""

    def test_create_database_not_found(self, capsys, tmp_path):
        """Test error when database not found."""
        args = argparse.Namespace(
            database=str(tmp_path / "missing.db"),
            output=None,
            incremental=False,
            no_compress=False,
            skip_verify=False,
            notes=None,
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            dry_run=False,
        )

        result = cmd_backup_create(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Database not found" in captured.out

    def test_create_dry_run(self, capsys, tmp_path):
        """Test backup create with dry run."""
        # Create a mock database
        db_path = tmp_path / "aragora.db"
        db_path.touch()

        args = argparse.Namespace(
            database=str(db_path),
            output=str(tmp_path / "backups"),
            incremental=False,
            no_compress=False,
            skip_verify=False,
            notes=None,
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            dry_run=True,
        )

        result = cmd_backup_create(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out

    def test_create_success(self, capsys, tmp_path):
        """Test successful backup creation."""
        db_path = tmp_path / "aragora.db"
        db_path.touch()

        mock_result = MagicMock()
        mock_result.id = "backup-123"
        mock_result.backup_path = tmp_path / "backup.tar.gz"
        mock_result.compressed_size_bytes = 1024 * 1024
        mock_result.duration_seconds = 5.5
        mock_result.status.value = "completed"
        mock_result.verified = True
        mock_result.tables = ["table1", "table2"]

        args = argparse.Namespace(
            database=str(db_path),
            output=str(tmp_path / "backups"),
            incremental=False,
            no_compress=False,
            skip_verify=False,
            notes="Test backup",
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            dry_run=False,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.create_backup.return_value = mock_result

            result = cmd_backup_create(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Backup created successfully" in captured.out
        assert "backup-123" in captured.out

    def test_create_incremental(self, capsys, tmp_path):
        """Test incremental backup creation."""
        db_path = tmp_path / "aragora.db"
        db_path.touch()

        args = argparse.Namespace(
            database=str(db_path),
            output=str(tmp_path / "backups"),
            incremental=True,
            no_compress=False,
            skip_verify=False,
            notes=None,
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            dry_run=True,
        )

        result = cmd_backup_create(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "incremental" in captured.out

    def test_create_error(self, capsys, tmp_path):
        """Test backup creation failure."""
        db_path = tmp_path / "aragora.db"
        db_path.touch()

        args = argparse.Namespace(
            database=str(db_path),
            output=str(tmp_path / "backups"),
            incremental=False,
            no_compress=False,
            skip_verify=False,
            notes=None,
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            dry_run=False,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.create_backup.side_effect = Exception("Disk full")

            result = cmd_backup_create(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Backup failed" in captured.out
        assert "Disk full" in captured.out


class TestCmdBackupList:
    """Test backup list command."""

    def test_list_no_backup_dir(self, capsys, tmp_path):
        """Test listing when backup directory doesn't exist."""
        args = argparse.Namespace(
            backup_dir=str(tmp_path / "nonexistent"),
            limit=20,
            json=False,
        )

        result = cmd_backup_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No backups found" in captured.out

    def test_list_no_backups(self, capsys, tmp_path):
        """Test listing when no backups exist."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        args = argparse.Namespace(
            backup_dir=str(backup_dir),
            limit=20,
            json=False,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = {}

            result = cmd_backup_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No backups found" in captured.out

    def test_list_table_output(self, capsys, tmp_path):
        """Test listing backups as table."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        mock_backup = MagicMock()
        mock_backup.id = "backup-123"
        mock_backup.created_at = datetime(2026, 1, 15, 10, 30, 0)
        mock_backup.backup_type.value = "full"
        mock_backup.compressed_size_bytes = 1024 * 1024
        mock_backup.status.value = "completed"

        args = argparse.Namespace(
            backup_dir=str(backup_dir),
            limit=20,
            json=False,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = {"backup-123": mock_backup}

            result = cmd_backup_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "backup-123" in captured.out
        assert "full" in captured.out
        assert "completed" in captured.out
        assert "Total: 1 backups" in captured.out

    def test_list_json_output(self, capsys, tmp_path):
        """Test listing backups as JSON."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        mock_backup = MagicMock()
        mock_backup.id = "backup-123"
        mock_backup.created_at = datetime(2026, 1, 15, 10, 30, 0)
        mock_backup.to_dict.return_value = {"id": "backup-123", "type": "full"}

        args = argparse.Namespace(
            backup_dir=str(backup_dir),
            limit=20,
            json=True,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = {"backup-123": mock_backup}

            result = cmd_backup_list(args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert len(output) == 1
        assert output[0]["id"] == "backup-123"

    def test_list_with_limit(self, capsys, tmp_path):
        """Test listing with limit."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        # Create mock backups
        mock_backups = {}
        for i in range(5):
            backup = MagicMock()
            backup.id = f"backup-{i}"
            backup.created_at = datetime(2026, 1, i + 1)
            backup.backup_type.value = "full"
            backup.compressed_size_bytes = 1024
            backup.status.value = "completed"
            mock_backups[f"backup-{i}"] = backup

        args = argparse.Namespace(
            backup_dir=str(backup_dir),
            limit=3,
            json=False,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = mock_backups

            result = cmd_backup_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "... and 2 more backups" in captured.out


class TestCmdBackupRestore:
    """Test backup restore command."""

    def test_restore_backup_dir_not_found(self, capsys, tmp_path):
        """Test error when backup directory doesn't exist."""
        args = argparse.Namespace(
            backup_id="backup-123",
            backup_dir=str(tmp_path / "nonexistent"),
            output=None,
            force=False,
            skip_verify=False,
            dry_run=False,
        )

        result = cmd_backup_restore(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Backup directory not found" in captured.out

    def test_restore_backup_not_found(self, capsys, tmp_path):
        """Test error when backup ID not found."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        args = argparse.Namespace(
            backup_id="missing-123",
            backup_dir=str(backup_dir),
            output=None,
            force=False,
            skip_verify=False,
            dry_run=False,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = {}

            result = cmd_backup_restore(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Backup not found" in captured.out

    def test_restore_dry_run(self, capsys, tmp_path):
        """Test restore with dry run."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        mock_backup = MagicMock()
        mock_backup.id = "backup-123"
        mock_backup.backup_path = backup_dir / "backup.tar.gz"
        mock_backup.source_path = tmp_path / "original.db"
        mock_backup.created_at = datetime(2026, 1, 15)

        args = argparse.Namespace(
            backup_id="backup-123",
            backup_dir=str(backup_dir),
            output=None,
            force=False,
            skip_verify=False,
            dry_run=True,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = {"backup-123": mock_backup}

            result = cmd_backup_restore(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out

    def test_restore_target_exists_no_force(self, capsys, tmp_path):
        """Test restore fails when target exists without --force."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        target_path = tmp_path / "existing.db"
        target_path.touch()

        mock_backup = MagicMock()
        mock_backup.id = "backup-123"
        mock_backup.backup_path = backup_dir / "backup.tar.gz"
        mock_backup.source_path = target_path
        mock_backup.created_at = datetime(2026, 1, 15)

        args = argparse.Namespace(
            backup_id="backup-123",
            backup_dir=str(backup_dir),
            output=str(target_path),
            force=False,
            skip_verify=False,
            dry_run=False,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = {"backup-123": mock_backup}

            result = cmd_backup_restore(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Target already exists" in captured.out
        assert "--force" in captured.out

    def test_restore_partial_id_match(self, capsys, tmp_path):
        """Test restore with partial ID match."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        mock_backup = MagicMock()
        mock_backup.id = "backup-12345"
        mock_backup.backup_path = backup_dir / "backup.tar.gz"
        mock_backup.source_path = tmp_path / "original.db"
        mock_backup.created_at = datetime(2026, 1, 15)

        args = argparse.Namespace(
            backup_id="backup-123",  # Partial match
            backup_dir=str(backup_dir),
            output=None,
            force=False,
            skip_verify=False,
            dry_run=True,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = {"backup-12345": mock_backup}

            result = cmd_backup_restore(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "backup-12345" in captured.out


class TestCmdBackupVerify:
    """Test backup verify command."""

    def test_verify_backup_not_found(self, capsys, tmp_path):
        """Test error when backup not found."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        args = argparse.Namespace(
            backup_id="missing-123",
            backup_dir=str(backup_dir),
            skip_restore_test=False,
            json=False,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = {}

            result = cmd_backup_verify(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Backup not found" in captured.out

    def test_verify_success(self, capsys, tmp_path):
        """Test successful verification."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        mock_backup = MagicMock()
        mock_backup.id = "backup-123"

        mock_result = MagicMock()
        mock_result.backup_id = "backup-123"
        mock_result.verified = True
        mock_result.checksum_valid = True
        mock_result.restore_tested = True
        mock_result.tables_valid = True
        mock_result.row_counts_valid = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.duration_seconds = 2.5

        args = argparse.Namespace(
            backup_id="backup-123",
            backup_dir=str(backup_dir),
            skip_restore_test=False,
            json=False,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = {"backup-123": mock_backup}
            mock_manager.verify_backup.return_value = mock_result

            result = cmd_backup_verify(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Checksum: Valid" in captured.out
        assert "verified successfully" in captured.out

    def test_verify_failure(self, capsys, tmp_path):
        """Test failed verification."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        mock_backup = MagicMock()
        mock_backup.id = "backup-123"

        mock_result = MagicMock()
        mock_result.backup_id = "backup-123"
        mock_result.verified = False
        mock_result.checksum_valid = False
        mock_result.restore_tested = False
        mock_result.tables_valid = True
        mock_result.row_counts_valid = True
        mock_result.errors = ["Checksum mismatch"]
        mock_result.warnings = []
        mock_result.duration_seconds = 1.0

        args = argparse.Namespace(
            backup_id="backup-123",
            backup_dir=str(backup_dir),
            skip_restore_test=False,
            json=False,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = {"backup-123": mock_backup}
            mock_manager.verify_backup.return_value = mock_result

            result = cmd_backup_verify(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Checksum: INVALID" in captured.out
        assert "FAILED" in captured.out

    def test_verify_json_output(self, capsys, tmp_path):
        """Test verification with JSON output."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        mock_backup = MagicMock()
        mock_backup.id = "backup-123"

        mock_result = MagicMock()
        mock_result.backup_id = "backup-123"
        mock_result.verified = True
        mock_result.checksum_valid = True
        mock_result.restore_tested = True
        mock_result.tables_valid = True
        mock_result.row_counts_valid = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.duration_seconds = 2.5

        args = argparse.Namespace(
            backup_id="backup-123",
            backup_dir=str(backup_dir),
            skip_restore_test=False,
            json=True,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager._backups = {"backup-123": mock_backup}
            mock_manager.verify_backup.return_value = mock_result

            result = cmd_backup_verify(args)

        assert result == 0
        captured = capsys.readouterr()
        # Output contains "Verifying backup..." then multiline JSON
        # Find the start of JSON output (starts with {) and extract it
        out = captured.out
        json_start = out.find("{")
        assert json_start != -1, f"No JSON found in output: {out}"
        json_str = out[json_start:]
        output = json.loads(json_str)
        assert output["verified"] is True
        assert output["backup_id"] == "backup-123"


class TestCmdBackupCleanup:
    """Test backup cleanup command."""

    def test_cleanup_no_backup_dir(self, capsys, tmp_path):
        """Test cleanup when backup directory doesn't exist."""
        args = argparse.Namespace(
            backup_dir=str(tmp_path / "nonexistent"),
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            force=False,
            dry_run=False,
        )

        result = cmd_backup_cleanup(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No backups found" in captured.out

    def test_cleanup_nothing_to_remove(self, capsys, tmp_path):
        """Test cleanup when nothing needs removal."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        args = argparse.Namespace(
            backup_dir=str(backup_dir),
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            force=False,
            dry_run=False,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.apply_retention_policy.return_value = []

            result = cmd_backup_cleanup(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No backups need to be removed" in captured.out

    def test_cleanup_dry_run(self, capsys, tmp_path):
        """Test cleanup with dry run."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        mock_backup = MagicMock()
        mock_backup.id = "old-backup"
        mock_backup.created_at = datetime(2026, 1, 1)
        mock_backup.compressed_size_bytes = 1024 * 1024

        args = argparse.Namespace(
            backup_dir=str(backup_dir),
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            force=False,
            dry_run=True,
        )

        with patch("aragora.backup.manager.BackupManager") as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.apply_retention_policy.return_value = ["old-backup"]
            mock_manager._backups = {"old-backup": mock_backup}

            result = cmd_backup_cleanup(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "old-backup" in captured.out


class TestAddBackupSubparsers:
    """Test parser creation."""

    def test_parser_has_all_subcommands(self):
        """Test that parser has all expected subcommands."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()

        add_backup_subparsers(subparsers)

        # Parse each subcommand
        args = main_parser.parse_args(["backup", "create"])
        assert hasattr(args, "func")

        args = main_parser.parse_args(["backup", "list"])
        assert hasattr(args, "func")

        args = main_parser.parse_args(["backup", "verify", "backup-id"])
        assert hasattr(args, "func")

        args = main_parser.parse_args(["backup", "restore", "backup-id"])
        assert hasattr(args, "func")

        args = main_parser.parse_args(["backup", "cleanup"])
        assert hasattr(args, "func")

    def test_create_has_all_options(self):
        """Test create subcommand has expected options."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()

        add_backup_subparsers(subparsers)

        args = main_parser.parse_args(
            [
                "backup",
                "create",
                "--database",
                "test.db",
                "--output",
                "/backups",
                "--incremental",
                "--no-compress",
                "--skip-verify",
                "--notes",
                "Test backup",
                "--dry-run",
            ]
        )

        assert args.database == "test.db"
        assert args.output == "/backups"
        assert args.incremental is True
        assert args.no_compress is True
        assert args.skip_verify is True
        assert args.notes == "Test backup"
        assert args.dry_run is True


class TestCmdBackup:
    """Test main backup command dispatcher."""

    def test_backup_no_subcommand(self, capsys):
        """Test backup command without subcommand shows help."""
        args = argparse.Namespace()

        result = cmd_backup(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Usage: aragora backup" in captured.out
        assert "create" in captured.out
        assert "list" in captured.out
        assert "restore" in captured.out

    def test_backup_with_func(self):
        """Test backup command dispatches to func."""
        mock_func = MagicMock(return_value=0)
        args = argparse.Namespace(backup_command="create", func=mock_func)

        result = cmd_backup(args)

        assert result == 0
        mock_func.assert_called_once_with(args)
