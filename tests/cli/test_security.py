"""
Tests for aragora.cli.security module.

Tests security CLI commands: status, rotate-key, migrate, health.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.security import (
    cmd_health,
    cmd_migrate,
    cmd_rotate_key,
    cmd_security_status,
    create_security_parser,
)


# ===========================================================================
# Test Fixtures and Mock Classes
# ===========================================================================


@dataclass
class MockEncryptionKey:
    """Mock encryption key."""

    key_id: str = "key-123"
    version: int = 1
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=30)
    )


@dataclass
class MockRotationResult:
    """Mock key rotation result."""

    success: bool = True
    old_key_version: int = 1
    new_key_version: int = 2
    stores_processed: int = 3
    records_reencrypted: int = 100
    failed_records: int = 0
    duration_seconds: float = 5.5
    errors: list = field(default_factory=list)


@dataclass
class MockMigrationResult:
    """Mock migration result."""

    success: bool = True
    store_name: str = "integration"
    total_records: int = 50
    migrated_records: int = 30
    already_encrypted: int = 20
    failed_records: int = 0
    duration_seconds: float = 2.5


# ===========================================================================
# Tests: create_security_parser
# ===========================================================================


class TestCreateSecurityParser:
    """Tests for create_security_parser function."""

    def test_creates_security_subparser(self):
        """Test that security parser is created with subcommands."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_security_parser(subparsers)

        # Parse a simple status command
        args = parser.parse_args(["security", "status"])
        assert args.security_action == "status"

    def test_rotate_key_options(self):
        """Test rotate-key subcommand options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_security_parser(subparsers)

        args = parser.parse_args(
            ["security", "rotate-key", "--dry-run", "--stores", "gmail,sync", "--force"]
        )
        assert args.dry_run is True
        assert args.stores == "gmail,sync"
        assert args.force is True

    def test_migrate_options(self):
        """Test migrate subcommand options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_security_parser(subparsers)

        args = parser.parse_args(["security", "migrate", "--dry-run", "--stores", "integration"])
        assert args.dry_run is True
        assert args.stores == "integration"

    def test_health_options(self):
        """Test health subcommand options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_security_parser(subparsers)

        args = parser.parse_args(["security", "health", "--detailed"])
        assert args.detailed is True


# ===========================================================================
# Tests: cmd_security_status
# ===========================================================================


class TestCmdSecurityStatus:
    """Tests for cmd_security_status function."""

    @pytest.fixture
    def status_args(self):
        """Create base status args."""
        return argparse.Namespace()

    def test_status_success(self, status_args, capsys):
        """Test status command shows encryption info."""
        mock_service = MagicMock()
        mock_service.get_active_key_id.return_value = "key-123"
        mock_service.get_active_key.return_value = MockEncryptionKey()
        mock_service.list_keys.return_value = [MockEncryptionKey()]

        mock_encryption = MagicMock()
        mock_encryption.CRYPTO_AVAILABLE = True
        mock_encryption.get_encryption_service.return_value = mock_service

        with patch.dict("sys.modules", {"aragora.security.encryption": mock_encryption}):
            result = cmd_security_status(status_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Encryption Status" in captured.out
        assert "key-123" in captured.out

    def test_status_crypto_not_available(self, status_args, capsys):
        """Test status when crypto is not available."""
        mock_encryption = MagicMock()
        mock_encryption.CRYPTO_AVAILABLE = False

        with patch.dict("sys.modules", {"aragora.security.encryption": mock_encryption}):
            result = cmd_security_status(status_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not available" in captured.out

    def test_status_no_active_key(self, status_args, capsys):
        """Test status when no active key."""
        mock_service = MagicMock()
        mock_service.get_active_key_id.return_value = None
        mock_service.get_active_key.return_value = None
        mock_service.list_keys.return_value = []

        mock_encryption = MagicMock()
        mock_encryption.CRYPTO_AVAILABLE = True
        mock_encryption.get_encryption_service.return_value = mock_service

        with patch.dict("sys.modules", {"aragora.security.encryption": mock_encryption}):
            result = cmd_security_status(status_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No active key found" in captured.out

    def test_status_key_age_warning(self, status_args, capsys):
        """Test status shows warning for old key."""
        old_key = MockEncryptionKey(created_at=datetime.now(timezone.utc) - timedelta(days=100))
        mock_service = MagicMock()
        mock_service.get_active_key_id.return_value = "key-old"
        mock_service.get_active_key.return_value = old_key
        mock_service.list_keys.return_value = [old_key]

        mock_encryption = MagicMock()
        mock_encryption.CRYPTO_AVAILABLE = True
        mock_encryption.get_encryption_service.return_value = mock_service

        with patch.dict("sys.modules", {"aragora.security.encryption": mock_encryption}):
            result = cmd_security_status(status_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Consider rotation" in captured.out

    def test_status_import_error(self, status_args, capsys):
        """Test status when import fails."""
        with patch.dict("sys.modules", {"aragora.security.encryption": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module"),
            ):
                result = cmd_security_status(status_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Import error" in captured.out


# ===========================================================================
# Tests: cmd_rotate_key
# ===========================================================================


class TestCmdRotateKey:
    """Tests for cmd_rotate_key function."""

    @pytest.fixture
    def rotate_args(self):
        """Create base rotate args."""
        args = argparse.Namespace()
        args.stores = "integration,gmail,sync"
        args.dry_run = True
        args.force = False
        return args

    def test_rotate_dry_run_success(self, rotate_args, capsys):
        """Test key rotation in dry run mode."""
        mock_result = MockRotationResult()

        mock_migration = MagicMock()
        mock_migration.rotate_encryption_key.return_value = mock_result

        with patch.dict("sys.modules", {"aragora.security.migration": mock_migration}):
            result = cmd_rotate_key(rotate_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Key Rotation" in captured.out
        assert "DRY RUN" in captured.out
        assert "completed successfully" in captured.out

    def test_rotate_live_with_force(self, rotate_args, capsys):
        """Test live key rotation with force flag."""
        rotate_args.dry_run = False
        rotate_args.force = True

        mock_result = MockRotationResult()
        mock_migration = MagicMock()
        mock_migration.rotate_encryption_key.return_value = mock_result

        with patch.dict("sys.modules", {"aragora.security.migration": mock_migration}):
            result = cmd_rotate_key(rotate_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "LIVE ROTATION" in captured.out

    def test_rotate_user_aborts(self, rotate_args, capsys, monkeypatch):
        """Test user aborts rotation."""
        rotate_args.dry_run = False
        rotate_args.force = False

        mock_migration = MagicMock()
        monkeypatch.setattr("builtins.input", lambda _: "n")

        with patch.dict("sys.modules", {"aragora.security.migration": mock_migration}):
            result = cmd_rotate_key(rotate_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Aborted" in captured.out

    def test_rotate_failure(self, rotate_args, capsys):
        """Test key rotation failure."""
        mock_result = MockRotationResult(
            success=False, failed_records=5, errors=["Error 1", "Error 2"]
        )

        mock_migration = MagicMock()
        mock_migration.rotate_encryption_key.return_value = mock_result

        with patch.dict("sys.modules", {"aragora.security.migration": mock_migration}):
            result = cmd_rotate_key(rotate_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "failed" in captured.out
        assert "Failed records: 5" in captured.out

    def test_rotate_import_error(self, rotate_args, capsys):
        """Test rotation when import fails."""
        with patch.dict("sys.modules", {"aragora.security.migration": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module"),
            ):
                result = cmd_rotate_key(rotate_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Import error" in captured.out


# ===========================================================================
# Tests: cmd_migrate
# ===========================================================================


class TestCmdMigrate:
    """Tests for cmd_migrate function."""

    @pytest.fixture
    def migrate_args(self):
        """Create base migrate args."""
        args = argparse.Namespace()
        args.stores = "integration,gmail,sync"
        args.dry_run = True
        return args

    def test_migrate_dry_run_success(self, migrate_args, capsys):
        """Test migration in dry run mode."""
        mock_results = [
            MockMigrationResult(store_name="integration"),
            MockMigrationResult(store_name="gmail"),
            MockMigrationResult(store_name="sync"),
        ]

        mock_migration = MagicMock()
        mock_migration.run_startup_migration.return_value = mock_results
        mock_migration.StartupMigrationConfig = MagicMock()

        with patch.dict("sys.modules", {"aragora.security.migration": mock_migration}):
            result = cmd_migrate(migrate_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Data Migration" in captured.out
        assert "DRY RUN" in captured.out

    def test_migrate_user_aborts(self, migrate_args, capsys, monkeypatch):
        """Test user aborts migration."""
        migrate_args.dry_run = False

        mock_migration = MagicMock()
        mock_migration.StartupMigrationConfig = MagicMock()
        monkeypatch.setattr("builtins.input", lambda _: "n")

        with patch.dict("sys.modules", {"aragora.security.migration": mock_migration}):
            result = cmd_migrate(migrate_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Aborted" in captured.out

    def test_migrate_with_failures(self, migrate_args, capsys):
        """Test migration with some failures."""
        mock_results = [
            MockMigrationResult(store_name="integration"),
            MockMigrationResult(store_name="gmail", success=False, failed_records=5),
        ]

        mock_migration = MagicMock()
        mock_migration.run_startup_migration.return_value = mock_results
        mock_migration.StartupMigrationConfig = MagicMock()

        with patch.dict("sys.modules", {"aragora.security.migration": mock_migration}):
            result = cmd_migrate(migrate_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Failed: 5" in captured.out

    def test_migrate_import_error(self, migrate_args, capsys):
        """Test migration when import fails."""
        with patch.dict("sys.modules", {"aragora.security.migration": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module"),
            ):
                result = cmd_migrate(migrate_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Import error" in captured.out


# ===========================================================================
# Tests: cmd_health
# ===========================================================================


class TestCmdHealth:
    """Tests for cmd_health function."""

    @pytest.fixture
    def health_args(self):
        """Create base health args."""
        args = argparse.Namespace()
        args.detailed = False
        return args

    def test_health_all_checks_pass(self, health_args, capsys):
        """Test health when all checks pass."""
        mock_key = MockEncryptionKey()
        mock_service = MagicMock()
        mock_service.get_active_key_id.return_value = "key-123"
        mock_service.get_active_key.return_value = mock_key
        mock_service.encrypt.return_value = b"encrypted"
        mock_service.decrypt.return_value = b"health_check_test_data"

        mock_encryption = MagicMock()
        mock_encryption.CRYPTO_AVAILABLE = True
        mock_encryption.get_encryption_service.return_value = mock_service

        with patch.dict("sys.modules", {"aragora.security.encryption": mock_encryption}):
            result = cmd_health(health_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Health Check" in captured.out
        assert "All health checks passed" in captured.out

    def test_health_crypto_not_available(self, health_args, capsys):
        """Test health when crypto is not available."""
        mock_encryption = MagicMock()
        mock_encryption.CRYPTO_AVAILABLE = False

        with patch.dict("sys.modules", {"aragora.security.encryption": mock_encryption}):
            result = cmd_health(health_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not installed" in captured.out

    def test_health_no_active_key(self, health_args, capsys):
        """Test health when no active key."""
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = None

        mock_encryption = MagicMock()
        mock_encryption.CRYPTO_AVAILABLE = True
        mock_encryption.get_encryption_service.return_value = mock_service

        with patch.dict("sys.modules", {"aragora.security.encryption": mock_encryption}):
            result = cmd_health(health_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No active encryption key" in captured.out

    def test_health_key_age_warning(self, health_args, capsys):
        """Test health shows warning for old key."""
        old_key = MockEncryptionKey(created_at=datetime.now(timezone.utc) - timedelta(days=95))
        mock_service = MagicMock()
        mock_service.get_active_key_id.return_value = "key-old"
        mock_service.get_active_key.return_value = old_key
        mock_service.encrypt.return_value = b"encrypted"
        mock_service.decrypt.return_value = b"health_check_test_data"

        mock_encryption = MagicMock()
        mock_encryption.CRYPTO_AVAILABLE = True
        mock_encryption.get_encryption_service.return_value = mock_service

        with patch.dict("sys.modules", {"aragora.security.encryption": mock_encryption}):
            result = cmd_health(health_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "rotation recommended" in captured.out
        assert "warning" in captured.out.lower()

    def test_health_encrypt_decrypt_failure(self, health_args, capsys):
        """Test health when encrypt/decrypt fails."""
        mock_key = MockEncryptionKey()
        mock_service = MagicMock()
        mock_service.get_active_key_id.return_value = "key-123"
        mock_service.get_active_key.return_value = mock_key
        mock_service.encrypt.side_effect = Exception("Encryption failed")

        mock_encryption = MagicMock()
        mock_encryption.CRYPTO_AVAILABLE = True
        mock_encryption.get_encryption_service.return_value = mock_service

        with patch.dict("sys.modules", {"aragora.security.encryption": mock_encryption}):
            result = cmd_health(health_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Encrypt/decrypt error" in captured.out

    def test_health_encrypt_decrypt_mismatch(self, health_args, capsys):
        """Test health when decrypt doesn't match original."""
        mock_key = MockEncryptionKey()
        mock_service = MagicMock()
        mock_service.get_active_key_id.return_value = "key-123"
        mock_service.get_active_key.return_value = mock_key
        mock_service.encrypt.return_value = b"encrypted"
        mock_service.decrypt.return_value = b"different_data"

        mock_encryption = MagicMock()
        mock_encryption.CRYPTO_AVAILABLE = True
        mock_encryption.get_encryption_service.return_value = mock_service

        with patch.dict("sys.modules", {"aragora.security.encryption": mock_encryption}):
            result = cmd_health(health_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "round-trip failed" in captured.out

    def test_health_service_init_error(self, health_args, capsys):
        """Test health when service initialization fails."""
        mock_encryption = MagicMock()
        mock_encryption.CRYPTO_AVAILABLE = True
        mock_encryption.get_encryption_service.side_effect = Exception("Service init failed")

        with patch.dict("sys.modules", {"aragora.security.encryption": mock_encryption}):
            result = cmd_health(health_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Service init failed" in captured.out
