"""Tests for CLI backup command - database backup and restore."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.backup import (
    _format_size,
    _format_duration,
    cmd_backup_create,
    cmd_backup_list,
    cmd_backup_restore,
    cmd_backup_verify,
    cmd_backup_cleanup,
    cmd_backup,
    add_backup_subparsers,
)


class TestFormatSize:
    """Test _format_size helper."""

    def test_bytes(self):
        assert _format_size(500) == "500.0 B"

    def test_kilobytes(self):
        assert _format_size(1024) == "1.0 KB"

    def test_megabytes(self):
        assert _format_size(1024 * 1024) == "1.0 MB"

    def test_gigabytes(self):
        assert _format_size(1024**3) == "1.0 GB"

    def test_terabytes(self):
        assert _format_size(1024**4) == "1.0 TB"

    def test_zero(self):
        assert _format_size(0) == "0.0 B"

    def test_fractional(self):
        result = _format_size(512)
        assert "B" in result


class TestFormatDuration:
    """Test _format_duration helper."""

    def test_seconds(self):
        assert _format_duration(30) == "30.0s"

    def test_minutes(self):
        assert _format_duration(120) == "2.0m"

    def test_hours(self):
        assert _format_duration(7200) == "2.0h"

    def test_fractional_seconds(self):
        assert _format_duration(1.5) == "1.5s"


class TestCmdBackupCreate:
    """Test cmd_backup_create command."""

    def test_dry_run(self, capsys):
        args = argparse.Namespace(
            output=None,
            database=None,
            incremental=False,
            no_compress=False,
            skip_verify=False,
            notes=None,
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            dry_run=True,
        )
        with patch("aragora.cli.backup.resolve_db_path") as mock_resolve:
            mock_resolve.return_value = "/tmp/test.db"
            with patch("pathlib.Path.exists", return_value=True):
                result = cmd_backup_create(args)
        assert result == 0
        output = capsys.readouterr().out
        assert "DRY RUN" in output

    def test_database_not_found(self, capsys):
        args = argparse.Namespace(
            output=None,
            database="nonexistent.db",
            incremental=False,
            no_compress=False,
            skip_verify=False,
            notes=None,
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            dry_run=False,
        )
        with patch("aragora.cli.backup.resolve_db_path", return_value="nonexistent.db"):
            result = cmd_backup_create(args)
        assert result == 1

    def test_successful_backup(self, capsys):
        mock_result = MagicMock()
        mock_result.id = "backup-001"
        mock_result.backup_path = "/backups/001.gz"
        mock_result.compressed_size_bytes = 1024
        mock_result.duration_seconds = 2.5
        mock_result.status.value = "completed"
        mock_result.verified = True
        mock_result.tables = ["t1", "t2"]

        mock_manager_cls = MagicMock()
        mock_manager_cls.return_value.create_backup.return_value = mock_result
        mock_retention_cls = MagicMock()
        mock_backup_type = MagicMock()
        mock_backup_type.INCREMENTAL = "incremental"
        mock_backup_type.FULL = "full"

        args = argparse.Namespace(
            output="/tmp/backups",
            database=None,
            incremental=False,
            no_compress=False,
            skip_verify=False,
            notes="test backup",
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            dry_run=False,
        )

        with patch("aragora.cli.backup.resolve_db_path", return_value="/tmp/test.db"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.dict(
                    "sys.modules",
                    {
                        "aragora.backup.manager": MagicMock(
                            BackupManager=mock_manager_cls,
                            BackupType=mock_backup_type,
                            RetentionPolicy=mock_retention_cls,
                        )
                    },
                ):
                    result = cmd_backup_create(args)

        assert result == 0
        output = capsys.readouterr().out
        assert "successfully" in output
        assert "backup-001" in output

    def test_backup_failure(self, capsys):
        mock_manager_cls = MagicMock()
        mock_manager_cls.return_value.create_backup.side_effect = RuntimeError("disk full")
        mock_retention_cls = MagicMock()
        mock_backup_type = MagicMock()
        mock_backup_type.INCREMENTAL = "incremental"
        mock_backup_type.FULL = "full"

        args = argparse.Namespace(
            output=None,
            database=None,
            incremental=True,
            no_compress=True,
            skip_verify=True,
            notes=None,
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            dry_run=False,
        )

        with patch("aragora.cli.backup.resolve_db_path", return_value="/tmp/test.db"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.dict(
                    "sys.modules",
                    {
                        "aragora.backup.manager": MagicMock(
                            BackupManager=mock_manager_cls,
                            BackupType=mock_backup_type,
                            RetentionPolicy=mock_retention_cls,
                        )
                    },
                ):
                    result = cmd_backup_create(args)

        assert result == 1
        output = capsys.readouterr().out
        assert "Error" in output


class TestCmdBackupList:
    """Test cmd_backup_list command."""

    def test_no_backup_dir(self, capsys):
        args = argparse.Namespace(backup_dir="/nonexistent", limit=20, json=False)
        result = cmd_backup_list(args)
        assert result == 0
        assert "No backups found" in capsys.readouterr().out

    def test_empty_backups(self, capsys, tmp_path):
        mock_manager_cls = MagicMock()
        mock_manager_cls.return_value._backups = {}

        args = argparse.Namespace(backup_dir=str(tmp_path), limit=20, json=False)
        with patch.dict(
            "sys.modules",
            {"aragora.backup.manager": MagicMock(BackupManager=mock_manager_cls)},
        ):
            result = cmd_backup_list(args)
        assert result == 0

    def test_json_output(self, capsys, tmp_path):
        from datetime import datetime

        mock_backup = MagicMock()
        mock_backup.id = "b1"
        mock_backup.created_at = datetime(2025, 1, 1)
        mock_backup.backup_type.value = "full"
        mock_backup.compressed_size_bytes = 1024
        mock_backup.status.value = "completed"
        mock_backup.to_dict.return_value = {"id": "b1"}

        mock_manager_cls = MagicMock()
        mock_manager_cls.return_value._backups = {"b1": mock_backup}

        args = argparse.Namespace(backup_dir=str(tmp_path), limit=20, json=True)
        with patch.dict(
            "sys.modules",
            {"aragora.backup.manager": MagicMock(BackupManager=mock_manager_cls)},
        ):
            result = cmd_backup_list(args)
        assert result == 0


class TestCmdBackupRestore:
    """Test cmd_backup_restore command."""

    def test_backup_dir_not_found(self, capsys):
        args = argparse.Namespace(
            backup_dir="/nonexistent",
            backup_id="b1",
            output=None,
            force=False,
            skip_verify=False,
            dry_run=False,
        )
        result = cmd_backup_restore(args)
        assert result == 1

    def test_backup_not_found(self, capsys, tmp_path):
        mock_manager_cls = MagicMock()
        mock_manager_cls.return_value._backups = {}

        args = argparse.Namespace(
            backup_dir=str(tmp_path),
            backup_id="nonexistent",
            output=None,
            force=False,
            skip_verify=False,
            dry_run=False,
        )
        with patch.dict(
            "sys.modules",
            {"aragora.backup.manager": MagicMock(BackupManager=mock_manager_cls)},
        ):
            result = cmd_backup_restore(args)
        assert result == 1

    def test_dry_run(self, capsys, tmp_path):
        mock_backup = MagicMock()
        mock_backup.id = "b1"
        mock_backup.backup_path = "/backups/b1.gz"
        mock_backup.source_path = "/data/db.sqlite"
        mock_backup.created_at = MagicMock()

        mock_manager_cls = MagicMock()
        mock_manager_cls.return_value._backups = {"b1": mock_backup}

        args = argparse.Namespace(
            backup_dir=str(tmp_path),
            backup_id="b1",
            output=None,
            force=False,
            skip_verify=False,
            dry_run=True,
        )
        with patch.dict(
            "sys.modules",
            {"aragora.backup.manager": MagicMock(BackupManager=mock_manager_cls)},
        ):
            result = cmd_backup_restore(args)
        assert result == 0
        assert "DRY RUN" in capsys.readouterr().out


class TestCmdBackupVerify:
    """Test cmd_backup_verify command."""

    def test_backup_dir_not_found(self, capsys):
        args = argparse.Namespace(
            backup_dir="/nonexistent",
            backup_id="b1",
            skip_restore_test=False,
            json=False,
        )
        result = cmd_backup_verify(args)
        assert result == 1

    def test_backup_not_found(self, capsys, tmp_path):
        mock_manager_cls = MagicMock()
        mock_manager_cls.return_value._backups = {}

        args = argparse.Namespace(
            backup_dir=str(tmp_path),
            backup_id="nonexistent",
            skip_restore_test=False,
            json=False,
        )
        with patch.dict(
            "sys.modules",
            {"aragora.backup.manager": MagicMock(BackupManager=mock_manager_cls)},
        ):
            result = cmd_backup_verify(args)
        assert result == 1

    def test_successful_verify(self, capsys, tmp_path):
        mock_backup = MagicMock()
        mock_backup.id = "b1"

        mock_verify_result = MagicMock()
        mock_verify_result.backup_id = "b1"
        mock_verify_result.verified = True
        mock_verify_result.checksum_valid = True
        mock_verify_result.restore_tested = True
        mock_verify_result.tables_valid = True
        mock_verify_result.row_counts_valid = True
        mock_verify_result.errors = []
        mock_verify_result.warnings = []
        mock_verify_result.duration_seconds = 1.0

        mock_manager_cls = MagicMock()
        mock_manager_cls.return_value._backups = {"b1": mock_backup}
        mock_manager_cls.return_value.verify_backup.return_value = mock_verify_result

        args = argparse.Namespace(
            backup_dir=str(tmp_path),
            backup_id="b1",
            skip_restore_test=False,
            json=False,
        )
        with patch.dict(
            "sys.modules",
            {"aragora.backup.manager": MagicMock(BackupManager=mock_manager_cls)},
        ):
            result = cmd_backup_verify(args)
        assert result == 0
        assert "successfully" in capsys.readouterr().out


class TestCmdBackupCleanup:
    """Test cmd_backup_cleanup command."""

    def test_no_backups_to_remove(self, capsys, tmp_path):
        mock_manager_cls = MagicMock()
        mock_manager_cls.return_value.apply_retention_policy.return_value = []
        mock_retention_cls = MagicMock()

        args = argparse.Namespace(
            backup_dir=str(tmp_path),
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            force=False,
            dry_run=False,
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.backup.manager": MagicMock(
                    BackupManager=mock_manager_cls,
                    RetentionPolicy=mock_retention_cls,
                )
            },
        ):
            result = cmd_backup_cleanup(args)
        assert result == 0

    def test_dry_run(self, capsys, tmp_path):
        mock_backup = MagicMock()
        mock_backup.id = "old-backup"
        mock_backup.compressed_size_bytes = 1024
        mock_backup.created_at.strftime.return_value = "2025-01-01 00:00"

        mock_manager_cls = MagicMock()
        mock_manager_cls.return_value.apply_retention_policy.return_value = ["old-backup"]
        mock_manager_cls.return_value._backups = {"old-backup": mock_backup}
        mock_retention_cls = MagicMock()

        args = argparse.Namespace(
            backup_dir=str(tmp_path),
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            force=False,
            dry_run=True,
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.backup.manager": MagicMock(
                    BackupManager=mock_manager_cls,
                    RetentionPolicy=mock_retention_cls,
                )
            },
        ):
            result = cmd_backup_cleanup(args)
        assert result == 0
        assert "DRY RUN" in capsys.readouterr().out


class TestCmdBackupDispatcher:
    """Test cmd_backup dispatcher."""

    def test_no_subcommand(self, capsys):
        args = argparse.Namespace()
        result = cmd_backup(args)
        assert result == 0
        assert "Usage" in capsys.readouterr().out

    def test_with_func(self):
        mock_func = MagicMock(return_value=0)
        args = argparse.Namespace(backup_command="create", func=mock_func)
        result = cmd_backup(args)
        assert result == 0
        mock_func.assert_called_once_with(args)


class TestAddBackupSubparsers:
    """Test that subparsers are properly configured."""

    def test_creates_subparsers(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_backup_subparsers(subparsers)

        args = parser.parse_args(["backup", "create", "--dry-run"])
        assert args.dry_run is True

    def test_list_subparser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_backup_subparsers(subparsers)

        args = parser.parse_args(["backup", "list", "--json", "--limit", "5"])
        assert args.json is True
        assert args.limit == 5

    def test_restore_subparser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_backup_subparsers(subparsers)

        args = parser.parse_args(["backup", "restore", "my-backup-id", "--force"])
        assert args.backup_id == "my-backup-id"
        assert args.force is True
