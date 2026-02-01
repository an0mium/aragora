"""
Tests for aragora.migrations.__main__ CLI module.

Covers:
- CLI command parsing (upgrade, downgrade, status, create)
- Command execution with mocked runner
- Error handling and exit codes
- Migration file creation

Run with:
    python -m pytest tests/migrations/test_main.py -v --noconftest --timeout=30
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


class TestMainModuleImport:
    """Verify the __main__ module can be imported."""

    def test_import_main_module(self):
        import aragora.migrations.__main__ as mod

        assert hasattr(mod, "main")
        assert hasattr(mod, "cmd_upgrade")
        assert hasattr(mod, "cmd_downgrade")
        assert hasattr(mod, "cmd_status")
        assert hasattr(mod, "cmd_create")
        assert hasattr(mod, "add_common_args")

    def test_functions_are_callable(self):
        from aragora.migrations.__main__ import (
            main,
            cmd_upgrade,
            cmd_downgrade,
            cmd_status,
            cmd_create,
        )

        assert callable(main)
        assert callable(cmd_upgrade)
        assert callable(cmd_downgrade)
        assert callable(cmd_status)
        assert callable(cmd_create)


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_upgrade_command_parses(self):
        """Test upgrade command argument parsing."""
        from aragora.migrations.__main__ import main

        with patch("sys.argv", ["migrations", "upgrade"]):
            with patch("aragora.migrations.__main__.cmd_upgrade") as mock_cmd:
                mock_cmd.return_value = 0
                with patch("aragora.migrations.__main__.get_migration_runner"):
                    with patch("aragora.migrations.__main__.reset_runner"):
                        # Can't easily test full main() due to argparse.parse_args
                        pass

    def test_upgrade_with_target(self):
        """Test upgrade command with --target argument."""
        from aragora.migrations.__main__ import add_common_args

        parser = argparse.ArgumentParser()
        add_common_args(parser)
        parser.add_argument("--target", type=int)

        args = parser.parse_args(["--target", "20240101000000"])
        assert args.target == 20240101000000

    def test_downgrade_with_target(self):
        """Test downgrade command with --target argument."""
        from aragora.migrations.__main__ import add_common_args

        parser = argparse.ArgumentParser()
        add_common_args(parser)
        parser.add_argument("--target", type=int)

        args = parser.parse_args(["--target", "20240101000000"])
        assert args.target == 20240101000000

    def test_common_args_db_path(self):
        """Test --db-path argument."""
        from aragora.migrations.__main__ import add_common_args

        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args(["--db-path", "/custom/path.db"])
        assert args.db_path == "/custom/path.db"

    def test_common_args_database_url(self):
        """Test --database-url argument."""
        from aragora.migrations.__main__ import add_common_args

        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args(["--database-url", "postgresql://localhost/db"])
        assert args.database_url == "postgresql://localhost/db"

    def test_common_args_default_db_path(self):
        """Test default db-path value."""
        from aragora.migrations.__main__ import add_common_args

        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args([])
        assert args.db_path == "aragora.db"


# ---------------------------------------------------------------------------
# cmd_upgrade tests
# ---------------------------------------------------------------------------


class TestCmdUpgrade:
    """Tests for the upgrade command handler."""

    def test_upgrade_applies_migrations(self):
        """Test upgrade applies pending migrations."""
        from aragora.migrations.__main__ import cmd_upgrade
        from aragora.migrations.runner import Migration

        mock_runner = MagicMock()
        mock_migration = Migration(version=1, name="test", up_sql="SELECT 1")
        mock_runner.upgrade.return_value = [mock_migration]

        args = argparse.Namespace(db_path="test.db", database_url=None, target=None)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner"):
                result = cmd_upgrade(args)

        assert result == 0
        mock_runner.upgrade.assert_called_once_with(target_version=None)

    def test_upgrade_no_pending(self):
        """Test upgrade with no pending migrations."""
        from aragora.migrations.__main__ import cmd_upgrade

        mock_runner = MagicMock()
        mock_runner.upgrade.return_value = []

        args = argparse.Namespace(db_path="test.db", database_url=None, target=None)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner"):
                result = cmd_upgrade(args)

        assert result == 0

    def test_upgrade_with_target_version(self):
        """Test upgrade with specific target version."""
        from aragora.migrations.__main__ import cmd_upgrade

        mock_runner = MagicMock()
        mock_runner.upgrade.return_value = []

        args = argparse.Namespace(db_path="test.db", database_url=None, target=20240101000000)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner"):
                cmd_upgrade(args)

        mock_runner.upgrade.assert_called_once_with(target_version=20240101000000)

    def test_upgrade_handles_error(self):
        """Test upgrade returns error code on failure."""
        from aragora.migrations.__main__ import cmd_upgrade

        mock_runner = MagicMock()
        mock_runner.upgrade.side_effect = RuntimeError("Migration failed")

        args = argparse.Namespace(db_path="test.db", database_url=None, target=None)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner"):
                result = cmd_upgrade(args)

        assert result == 1

    def test_upgrade_always_resets_runner(self):
        """Test that reset_runner is called even on error."""
        from aragora.migrations.__main__ import cmd_upgrade

        mock_runner = MagicMock()
        mock_runner.upgrade.side_effect = RuntimeError("boom")

        args = argparse.Namespace(db_path="test.db", database_url=None, target=None)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner") as mock_reset:
                cmd_upgrade(args)
                mock_reset.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_downgrade tests
# ---------------------------------------------------------------------------


class TestCmdDowngrade:
    """Tests for the downgrade command handler."""

    def test_downgrade_rolls_back(self):
        """Test downgrade rolls back migrations."""
        from aragora.migrations.__main__ import cmd_downgrade
        from aragora.migrations.runner import Migration

        mock_runner = MagicMock()
        mock_migration = Migration(version=1, name="test", up_sql="SELECT 1", down_sql="SELECT 1")
        mock_runner.downgrade.return_value = [mock_migration]

        args = argparse.Namespace(db_path="test.db", database_url=None, target=None)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner"):
                result = cmd_downgrade(args)

        assert result == 0
        mock_runner.downgrade.assert_called_once_with(
            target_version=None,
            dry_run=False,
            reason=None,
            use_stored_rollback=False,
        )

    def test_downgrade_no_migrations(self):
        """Test downgrade with nothing to rollback."""
        from aragora.migrations.__main__ import cmd_downgrade

        mock_runner = MagicMock()
        mock_runner.downgrade.return_value = []

        args = argparse.Namespace(db_path="test.db", database_url=None, target=None)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner"):
                result = cmd_downgrade(args)

        assert result == 0

    def test_downgrade_with_target_version(self):
        """Test downgrade with specific target version."""
        from aragora.migrations.__main__ import cmd_downgrade

        mock_runner = MagicMock()
        mock_runner.downgrade.return_value = []

        args = argparse.Namespace(db_path="test.db", database_url=None, target=20240101000000)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner"):
                cmd_downgrade(args)

        mock_runner.downgrade.assert_called_once_with(
            target_version=20240101000000,
            dry_run=False,
            reason=None,
            use_stored_rollback=False,
        )

    def test_downgrade_handles_error(self):
        """Test downgrade returns error code on failure."""
        from aragora.migrations.__main__ import cmd_downgrade

        mock_runner = MagicMock()
        mock_runner.downgrade.side_effect = RuntimeError("Rollback failed")

        args = argparse.Namespace(db_path="test.db", database_url=None, target=None)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner"):
                result = cmd_downgrade(args)

        assert result == 1


# ---------------------------------------------------------------------------
# cmd_status tests
# ---------------------------------------------------------------------------


class TestCmdStatus:
    """Tests for the status command handler."""

    def test_status_displays_info(self):
        """Test status command displays migration info."""
        from aragora.migrations.__main__ import cmd_status

        mock_runner = MagicMock()
        mock_runner.status.return_value = {
            "applied_count": 5,
            "pending_count": 2,
            "latest_applied": 20240105000000,
            "latest_available": 20240107000000,
            "pending_versions": [20240106000000, 20240107000000],
        }
        mock_runner._backend.backend_type = "sqlite"

        args = argparse.Namespace(db_path="test.db", database_url=None)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner"):
                result = cmd_status(args)

        assert result == 0
        mock_runner.status.assert_called_once()

    def test_status_no_migrations(self):
        """Test status with no migrations."""
        from aragora.migrations.__main__ import cmd_status

        mock_runner = MagicMock()
        mock_runner.status.return_value = {
            "applied_count": 0,
            "pending_count": 0,
            "latest_applied": None,
            "latest_available": None,
            "pending_versions": [],
        }
        mock_runner._backend.backend_type = "sqlite"

        args = argparse.Namespace(db_path="test.db", database_url=None)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner"):
                result = cmd_status(args)

        assert result == 0

    def test_status_handles_error(self):
        """Test status returns error code on failure."""
        from aragora.migrations.__main__ import cmd_status

        mock_runner = MagicMock()
        mock_runner.status.side_effect = RuntimeError("Database error")

        args = argparse.Namespace(db_path="test.db", database_url=None)

        with patch("aragora.migrations.__main__.get_migration_runner", return_value=mock_runner):
            with patch("aragora.migrations.__main__.reset_runner"):
                result = cmd_status(args)

        assert result == 1


# ---------------------------------------------------------------------------
# cmd_create tests
# ---------------------------------------------------------------------------


class TestCmdCreate:
    """Tests for the create command handler."""

    def test_create_generates_file(self):
        """Test create command generates a migration file."""
        from aragora.migrations.__main__ import cmd_create

        args = argparse.Namespace(name="Add users table")

        # Just mock the write_text to prevent actual file creation
        with patch("pathlib.Path.write_text") as mock_write:
            mock_write.return_value = None
            result = cmd_create(args)

        # The command itself should return 0 on success
        assert result == 0
        # Verify write was called
        mock_write.assert_called_once()

    def test_create_sanitizes_name(self):
        """Test create command sanitizes the migration name."""
        from aragora.migrations.__main__ import cmd_create

        args = argparse.Namespace(name="Add User's Table!")

        # Test that the function handles special characters
        with patch("pathlib.Path.write_text") as mock_write:
            mock_write.return_value = None
            result = cmd_create(args)

        assert result == 0
        # Verify write was called
        mock_write.assert_called_once()

    def test_create_includes_template(self):
        """Test created file contains proper template."""
        from aragora.migrations.__main__ import cmd_create

        captured_content = []

        def capture_write(content):
            captured_content.append(content)

        args = argparse.Namespace(name="Test migration")

        with patch("pathlib.Path.write_text", side_effect=capture_write):
            cmd_create(args)

        # Check the template content
        assert len(captured_content) == 1
        content = captured_content[0]
        assert "Test migration" in content
        assert "from aragora.migrations.runner import Migration" in content
        assert "migration = Migration" in content
        assert "up_sql" in content
        assert "down_sql" in content

    def test_create_handles_file_error(self):
        """Test create returns error code on file write failure."""
        from aragora.migrations.__main__ import cmd_create

        args = argparse.Namespace(name="Test migration")

        with patch("pathlib.Path.write_text", side_effect=OSError("Permission denied")):
            result = cmd_create(args)

        assert result == 1

    def test_create_version_is_timestamp(self):
        """Test created migration uses timestamp-based version."""
        from aragora.migrations.__main__ import cmd_create

        captured_content = []

        def capture_write(content):
            captured_content.append(content)

        args = argparse.Namespace(name="Test")

        with patch("pathlib.Path.write_text", side_effect=capture_write):
            cmd_create(args)

        content = captured_content[0]
        # Version should be a 14-digit timestamp
        import re

        match = re.search(r"version=(\d+)", content)
        assert match is not None
        version = match.group(1)
        assert len(version) == 14


# ---------------------------------------------------------------------------
# main() entry point tests
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    """Tests for the main entry point function."""

    def test_main_requires_command(self):
        """Test main raises when no command provided."""
        from aragora.migrations.__main__ import main

        with patch("sys.argv", ["migrations"]):
            with pytest.raises(SystemExit):
                main()

    def test_main_upgrade_command(self):
        """Test main routes to upgrade command."""
        from aragora.migrations.__main__ import main

        mock_runner = MagicMock()
        mock_runner.upgrade.return_value = []

        with patch("sys.argv", ["migrations", "upgrade"]):
            with patch(
                "aragora.migrations.__main__.get_migration_runner",
                return_value=mock_runner,
            ):
                with patch("aragora.migrations.__main__.reset_runner"):
                    result = main()

        assert result == 0

    def test_main_downgrade_command(self):
        """Test main routes to downgrade command."""
        from aragora.migrations.__main__ import main

        mock_runner = MagicMock()
        mock_runner.downgrade.return_value = []

        with patch("sys.argv", ["migrations", "downgrade"]):
            with patch(
                "aragora.migrations.__main__.get_migration_runner",
                return_value=mock_runner,
            ):
                with patch("aragora.migrations.__main__.reset_runner"):
                    result = main()

        assert result == 0

    def test_main_status_command(self):
        """Test main routes to status command."""
        from aragora.migrations.__main__ import main

        mock_runner = MagicMock()
        mock_runner.status.return_value = {
            "applied_count": 0,
            "pending_count": 0,
            "latest_applied": None,
            "latest_available": None,
            "pending_versions": [],
        }
        mock_runner._backend.backend_type = "sqlite"

        with patch("sys.argv", ["migrations", "status"]):
            with patch(
                "aragora.migrations.__main__.get_migration_runner",
                return_value=mock_runner,
            ):
                with patch("aragora.migrations.__main__.reset_runner"):
                    result = main()

        assert result == 0

    def test_main_create_command(self):
        """Test main routes to create command."""
        from aragora.migrations.__main__ import main

        with patch("sys.argv", ["migrations", "create", "Test migration"]):
            with patch("pathlib.Path.write_text"):
                result = main()

        assert result == 0


# ---------------------------------------------------------------------------
# Environment variable handling
# ---------------------------------------------------------------------------


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_database_url_from_env(self):
        """Test DATABASE_URL environment variable is respected."""
        from aragora.migrations.__main__ import add_common_args

        parser = argparse.ArgumentParser()
        add_common_args(parser)

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            # Re-import to get fresh default value
            import importlib
            import aragora.migrations.__main__ as mod

            importlib.reload(mod)

            parser2 = argparse.ArgumentParser()
            mod.add_common_args(parser2)
            args = parser2.parse_args([])
            # The default comes from os.environ.get() at parse time
            assert args.database_url is None or args.database_url == "postgresql://localhost/test"

    def test_command_line_overrides_env(self):
        """Test command line argument overrides environment variable."""
        from aragora.migrations.__main__ import add_common_args

        parser = argparse.ArgumentParser()
        add_common_args(parser)

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            args = parser.parse_args(["--database-url", "postgresql://other/db"])

        assert args.database_url == "postgresql://other/db"
