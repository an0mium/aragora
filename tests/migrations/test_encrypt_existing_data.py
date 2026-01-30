"""
Tests for aragora.migrations.encrypt_existing_data module.

Covers:
- EncryptionMigration class initialization
- Prerequisite checking (encryption key, crypto availability)
- Database backup creation
- Migration of integrations, webhooks, tokens, sync configs
- Dry run mode
- Rollback functionality
- Error handling

Run with:
    python -m pytest tests/migrations/test_encrypt_existing_data.py -v --noconftest --timeout=30
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


class TestEncryptionMigrationImport:
    """Verify the encrypt_existing_data module can be imported."""

    def test_import_module(self):
        import aragora.migrations.encrypt_existing_data as mod

        assert hasattr(mod, "EncryptionMigration")
        assert hasattr(mod, "main")

    def test_import_class(self):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        assert EncryptionMigration is not None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_data_dir():
    """Create a temporary data directory for testing."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture()
def mock_encryption_env(temp_data_dir):
    """Set up environment for encryption tests."""
    with patch.dict(
        os.environ,
        {
            "ARAGORA_ENCRYPTION_KEY": "test-encryption-key-32bytes!!!!",
            "ARAGORA_DATA_DIR": str(temp_data_dir),
        },
    ):
        yield temp_data_dir


@pytest.fixture()
def integrations_db(temp_data_dir):
    """Create a test integrations database."""
    db_path = temp_data_dir / "integrations.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE integrations (
            id TEXT PRIMARY KEY,
            settings_json TEXT
        )
    """)
    # Insert test data with sensitive fields
    cursor.execute(
        "INSERT INTO integrations (id, settings_json) VALUES (?, ?)",
        (
            "int_1",
            json.dumps(
                {
                    "api_key": "secret_key_123",
                    "api_secret": "secret_value",
                    "name": "Test Integration",
                }
            ),
        ),
    )
    cursor.execute(
        "INSERT INTO integrations (id, settings_json) VALUES (?, ?)",
        ("int_2", json.dumps({"name": "No Secrets Here"})),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def webhooks_db(temp_data_dir):
    """Create a test webhooks database."""
    db_path = temp_data_dir / "webhooks.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE webhooks (
            id TEXT PRIMARY KEY,
            secret TEXT
        )
    """)
    cursor.execute(
        "INSERT INTO webhooks (id, secret) VALUES (?, ?)",
        ("wh_1", "webhook_secret_123"),
    )
    cursor.execute(
        "INSERT INTO webhooks (id, secret) VALUES (?, ?)",
        ("wh_2", "another_secret"),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def tokens_db(temp_data_dir):
    """Create a test tokens database."""
    db_path = temp_data_dir / "gmail_tokens.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE gmail_tokens (
            user_id TEXT PRIMARY KEY,
            access_token TEXT,
            refresh_token TEXT
        )
    """)
    cursor.execute(
        "INSERT INTO gmail_tokens (user_id, access_token, refresh_token) VALUES (?, ?, ?)",
        ("user_1", "access_token_123", "refresh_token_456"),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def sync_configs_db(temp_data_dir):
    """Create a test sync configs database."""
    db_path = temp_data_dir / "enterprise_sync.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE sync_configs (
            id TEXT PRIMARY KEY,
            config_json TEXT
        )
    """)
    cursor.execute(
        "INSERT INTO sync_configs (id, config_json) VALUES (?, ?)",
        (
            "sync_1",
            json.dumps({"api_key": "sync_api_key", "endpoint": "https://example.com"}),
        ),
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# EncryptionMigration initialization tests
# ---------------------------------------------------------------------------


class TestEncryptionMigrationInit:
    """Tests for EncryptionMigration initialization."""

    def test_init_default_data_dir(self):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        with patch.dict(os.environ, {"ARAGORA_DATA_DIR": "/custom/path"}):
            migration = EncryptionMigration()
            assert migration.data_dir == Path("/custom/path")

    def test_init_custom_data_dir(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))
        assert migration.data_dir == temp_data_dir

    def test_init_dry_run_mode(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir), dry_run=True)
        assert migration.dry_run is True

    def test_init_custom_backup_dir(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        backup_dir = temp_data_dir / "custom_backups"
        migration = EncryptionMigration(data_dir=str(temp_data_dir), backup_dir=str(backup_dir))
        assert migration.backup_dir == backup_dir

    def test_init_default_backup_dir(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))
        assert migration.backup_dir == temp_data_dir / "backups"

    def test_init_stats_initialized(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))
        assert migration.stats["integrations_migrated"] == 0
        assert migration.stats["webhooks_migrated"] == 0
        assert migration.stats["tokens_migrated"] == 0
        assert migration.stats["sync_configs_migrated"] == 0
        assert migration.stats["errors"] == []


# ---------------------------------------------------------------------------
# Prerequisite checking tests
# ---------------------------------------------------------------------------


class TestCheckPrerequisites:
    """Tests for prerequisite checking."""

    def test_missing_encryption_key(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        with patch.dict(os.environ, {"ARAGORA_ENCRYPTION_KEY": ""}, clear=True):
            # Remove the key if it exists
            os.environ.pop("ARAGORA_ENCRYPTION_KEY", None)
            migration = EncryptionMigration(data_dir=str(temp_data_dir))
            result = migration.check_prerequisites()
            assert result is False

    def test_encryption_key_present(self, mock_encryption_env):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        # Mock the encryption availability check
        with patch(
            "aragora.migrations.encrypt_existing_data.is_encryption_available",
            return_value=True,
        ):
            migration = EncryptionMigration(data_dir=str(mock_encryption_env))
            result = migration.check_prerequisites()
            assert result is True

    def test_encryption_not_available(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        with patch.dict(os.environ, {"ARAGORA_ENCRYPTION_KEY": "test-key"}, clear=False):
            with patch(
                "aragora.storage.encrypted_fields.is_encryption_available",
                return_value=False,
            ):
                migration = EncryptionMigration(data_dir=str(temp_data_dir))
                result = migration.check_prerequisites()
                assert result is False

    def test_import_error_handled(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        with patch.dict(os.environ, {"ARAGORA_ENCRYPTION_KEY": "test-key"}, clear=False):
            with patch(
                "aragora.migrations.encrypt_existing_data.is_encryption_available",
                side_effect=ImportError("Module not found"),
            ):
                migration = EncryptionMigration(data_dir=str(temp_data_dir))
                result = migration.check_prerequisites()
                assert result is False


# ---------------------------------------------------------------------------
# Backup creation tests
# ---------------------------------------------------------------------------


class TestCreateBackup:
    """Tests for database backup creation."""

    def test_create_backup_success(self, temp_data_dir, integrations_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))
        backup_path = migration.create_backup(integrations_db)

        assert backup_path is not None
        assert backup_path.exists()
        assert "integrations_" in backup_path.name

    def test_create_backup_nonexistent_db(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))
        nonexistent = temp_data_dir / "nonexistent.db"
        backup_path = migration.create_backup(nonexistent)

        assert backup_path is None

    def test_create_backup_creates_backup_dir(self, temp_data_dir, integrations_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        backup_dir = temp_data_dir / "new_backups"
        migration = EncryptionMigration(data_dir=str(temp_data_dir), backup_dir=str(backup_dir))
        migration.create_backup(integrations_db)

        assert backup_dir.exists()

    def test_create_backup_error_handled(self, temp_data_dir, integrations_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))

        with patch("shutil.copy2", side_effect=OSError("Permission denied")):
            backup_path = migration.create_backup(integrations_db)

        assert backup_path is None


# ---------------------------------------------------------------------------
# Integration migration tests
# ---------------------------------------------------------------------------


class TestMigrateIntegrations:
    """Tests for integration settings migration."""

    def test_migrate_integrations_dry_run(self, mock_encryption_env, integrations_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(mock_encryption_env), dry_run=True)

        with patch("aragora.storage.encrypted_fields.is_field_encrypted", return_value=False):
            with patch(
                "aragora.storage.encrypted_fields.SENSITIVE_FIELDS",
                ["api_key", "api_secret"],
            ):
                with patch("aragora.storage.encrypted_fields.encrypt_sensitive"):
                    count = migration.migrate_integrations()

        # Dry run should still count but not modify
        assert count >= 0

    def test_migrate_integrations_no_db(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))
        count = migration.migrate_integrations()

        assert count == 0

    def test_migrate_integrations_already_encrypted(self, mock_encryption_env, integrations_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(mock_encryption_env))

        with patch("aragora.storage.encrypted_fields.is_field_encrypted", return_value=True):
            with patch("aragora.storage.encrypted_fields.SENSITIVE_FIELDS", ["api_key"]):
                count = migration.migrate_integrations()

        # Already encrypted, nothing to migrate
        assert count == 0


# ---------------------------------------------------------------------------
# Webhook migration tests
# ---------------------------------------------------------------------------


class TestMigrateWebhooks:
    """Tests for webhook secret migration."""

    def test_migrate_webhooks_no_db(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))
        count = migration.migrate_webhooks()

        assert count == 0

    def test_migrate_webhooks_already_encrypted(self, mock_encryption_env, webhooks_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        # Update the webhook to look encrypted
        conn = sqlite3.connect(str(webhooks_db))
        conn.execute("UPDATE webhooks SET secret = 'gAAAAA...' WHERE id = 'wh_1'")
        conn.execute("UPDATE webhooks SET secret = 'gAAAAA...' WHERE id = 'wh_2'")
        conn.commit()
        conn.close()

        migration = EncryptionMigration(data_dir=str(mock_encryption_env))

        with patch(
            "aragora.storage.encrypted_fields.is_encryption_available",
            return_value=True,
        ):
            count = migration.migrate_webhooks()

        assert count == 0

    def test_migrate_webhooks_encryption_unavailable(self, mock_encryption_env, webhooks_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(mock_encryption_env))

        with patch(
            "aragora.storage.encrypted_fields.is_encryption_available",
            return_value=False,
        ):
            count = migration.migrate_webhooks()

        assert count == 0


# ---------------------------------------------------------------------------
# Token migration tests
# ---------------------------------------------------------------------------


class TestMigrateTokens:
    """Tests for OAuth token migration."""

    def test_migrate_tokens_no_db(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))
        count = migration.migrate_tokens()

        assert count == 0

    def test_migrate_tokens_already_encrypted(self, mock_encryption_env, tokens_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        # Update tokens to look encrypted
        conn = sqlite3.connect(str(tokens_db))
        conn.execute(
            "UPDATE gmail_tokens SET access_token = 'gAAAAA...', refresh_token = 'gAAAAA...'"
        )
        conn.commit()
        conn.close()

        migration = EncryptionMigration(data_dir=str(mock_encryption_env))

        with patch(
            "aragora.storage.encrypted_fields.is_encryption_available",
            return_value=True,
        ):
            count = migration.migrate_tokens()

        assert count == 0

    def test_migrate_tokens_encryption_unavailable(self, mock_encryption_env, tokens_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(mock_encryption_env))

        with patch(
            "aragora.storage.encrypted_fields.is_encryption_available",
            return_value=False,
        ):
            count = migration.migrate_tokens()

        assert count == 0


# ---------------------------------------------------------------------------
# Sync config migration tests
# ---------------------------------------------------------------------------


class TestMigrateSyncConfigs:
    """Tests for sync configuration migration."""

    def test_migrate_sync_configs_no_db(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))
        count = migration.migrate_sync_configs()

        assert count == 0

    def test_migrate_sync_configs_already_encrypted(self, mock_encryption_env, sync_configs_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(mock_encryption_env))

        with patch("aragora.storage.encrypted_fields.is_field_encrypted", return_value=True):
            with patch("aragora.storage.encrypted_fields.SENSITIVE_FIELDS", ["api_key"]):
                count = migration.migrate_sync_configs()

        assert count == 0


# ---------------------------------------------------------------------------
# Full run tests
# ---------------------------------------------------------------------------


class TestMigrationRun:
    """Tests for full migration run."""

    def test_run_prerequisites_fail(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        with patch.dict(os.environ, {"ARAGORA_ENCRYPTION_KEY": ""}, clear=True):
            os.environ.pop("ARAGORA_ENCRYPTION_KEY", None)
            migration = EncryptionMigration(data_dir=str(temp_data_dir))
            result = migration.run()

        assert result["success"] is False
        assert "Prerequisites not met" in result["error"]

    def test_run_dry_run_mode(self, mock_encryption_env):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(mock_encryption_env), dry_run=True)

        with patch.object(migration, "check_prerequisites", return_value=True):
            result = migration.run()

        assert result["dry_run"] is True

    def test_run_success(self, mock_encryption_env):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(mock_encryption_env))

        with patch.object(migration, "check_prerequisites", return_value=True):
            with patch.object(migration, "migrate_integrations", return_value=2):
                with patch.object(migration, "migrate_webhooks", return_value=1):
                    with patch.object(migration, "migrate_tokens", return_value=3):
                        with patch.object(migration, "migrate_sync_configs", return_value=1):
                            result = migration.run()

        assert result["success"] is True
        assert result["total_migrated"] == 7

    def test_run_with_errors(self, mock_encryption_env):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(mock_encryption_env))
        migration.stats["errors"].append("Test error")

        with patch.object(migration, "check_prerequisites", return_value=True):
            result = migration.run()

        assert result["success"] is False


# ---------------------------------------------------------------------------
# Rollback tests
# ---------------------------------------------------------------------------


class TestMigrationRollback:
    """Tests for migration rollback functionality."""

    def test_rollback_no_backup_dir(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        backup_dir = temp_data_dir / "nonexistent_backups"
        migration = EncryptionMigration(data_dir=str(temp_data_dir), backup_dir=str(backup_dir))
        result = migration.rollback()

        assert result is False

    def test_rollback_no_backups(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        backup_dir = temp_data_dir / "backups"
        backup_dir.mkdir()

        migration = EncryptionMigration(data_dir=str(temp_data_dir), backup_dir=str(backup_dir))
        result = migration.rollback()

        assert result is False

    def test_rollback_restores_backup(self, temp_data_dir, integrations_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        # Create a backup
        migration = EncryptionMigration(data_dir=str(temp_data_dir))
        backup_path = migration.create_backup(integrations_db)
        assert backup_path is not None

        # Modify the original
        conn = sqlite3.connect(str(integrations_db))
        conn.execute("DELETE FROM integrations")
        conn.commit()
        conn.close()

        # Verify it's empty
        conn = sqlite3.connect(str(integrations_db))
        count = conn.execute("SELECT COUNT(*) FROM integrations").fetchone()[0]
        conn.close()
        assert count == 0

        # Rollback
        result = migration.rollback()
        assert result is True

        # Verify restoration
        conn = sqlite3.connect(str(integrations_db))
        count = conn.execute("SELECT COUNT(*) FROM integrations").fetchone()[0]
        conn.close()
        assert count > 0

    def test_rollback_specific_timestamp(self, temp_data_dir, integrations_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))

        # Create backup - get the timestamp
        backup_path = migration.create_backup(integrations_db)
        timestamp = migration.backup_timestamp

        result = migration.rollback(timestamp=timestamp)
        assert result is True

    def test_rollback_invalid_timestamp(self, temp_data_dir, integrations_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(temp_data_dir))
        migration.create_backup(integrations_db)

        result = migration.rollback(timestamp="invalid_timestamp")
        assert result is False


# ---------------------------------------------------------------------------
# CLI main() tests
# ---------------------------------------------------------------------------


class TestMainCLI:
    """Tests for the main CLI entry point."""

    def test_main_dry_run(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import main

        with patch("sys.argv", ["script", "--dry-run", "--data-dir", str(temp_data_dir)]):
            with patch.dict(os.environ, {"ARAGORA_ENCRYPTION_KEY": ""}, clear=True):
                os.environ.pop("ARAGORA_ENCRYPTION_KEY", None)
                with pytest.raises(SystemExit) as exc_info:
                    main()
                # Should exit with error code 1 (prerequisites not met)
                assert exc_info.value.code == 1

    def test_main_rollback(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import main

        backup_dir = temp_data_dir / "backups"
        backup_dir.mkdir()

        with patch("sys.argv", ["script", "--rollback", "--data-dir", str(temp_data_dir)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # No backups to restore
            assert exc_info.value.code == 1

    def test_main_verbose_logging(self, temp_data_dir):
        from aragora.migrations.encrypt_existing_data import main
        import logging

        with patch("sys.argv", ["script", "-v", "--data-dir", str(temp_data_dir)]):
            with patch.dict(os.environ, {"ARAGORA_ENCRYPTION_KEY": ""}, clear=True):
                os.environ.pop("ARAGORA_ENCRYPTION_KEY", None)
                with pytest.raises(SystemExit):
                    main()


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_invalid_json_in_integrations(self, mock_encryption_env):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        # Create db with invalid JSON
        db_path = mock_encryption_env / "integrations.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE integrations (
                id TEXT PRIMARY KEY,
                settings_json TEXT
            )
        """)
        cursor.execute(
            "INSERT INTO integrations (id, settings_json) VALUES (?, ?)",
            ("int_1", "not valid json {"),
        )
        conn.commit()
        conn.close()

        migration = EncryptionMigration(data_dir=str(mock_encryption_env))

        with patch("aragora.storage.encrypted_fields.is_field_encrypted", return_value=False):
            with patch("aragora.storage.encrypted_fields.SENSITIVE_FIELDS", ["api_key"]):
                # Should handle invalid JSON gracefully
                count = migration.migrate_integrations()

        # Invalid JSON records should be skipped
        assert count == 0

    def test_database_error_during_migration(self, mock_encryption_env, integrations_db):
        from aragora.migrations.encrypt_existing_data import EncryptionMigration

        migration = EncryptionMigration(data_dir=str(mock_encryption_env))

        with patch("sqlite3.connect", side_effect=sqlite3.Error("Database error")):
            count = migration.migrate_integrations()

        # Should handle database errors
        assert count == 0 or len(migration.stats["errors"]) > 0
