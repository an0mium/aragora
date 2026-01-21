"""
Tests for encryption migration utilities.

Tests automatic detection, migration, and startup migration functionality.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from aragora.security.migration import (
    MigrationResult,
    EncryptionMigrator,
    is_field_encrypted,
    needs_migration,
    migrate_integration_store,
    migrate_gmail_token_store,
    migrate_sync_store,
    StartupMigrationConfig,
    get_startup_migration_config,
    run_startup_migration,
)


class TestMigrationResult:
    """Tests for MigrationResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = MigrationResult(store_name="test_store")

        assert result.store_name == "test_store"
        assert result.total_records == 0
        assert result.migrated_records == 0
        assert result.already_encrypted == 0
        assert result.failed_records == 0
        assert result.errors == []
        assert result.completed_at is None
        assert result.duration_seconds == 0.0

    def test_success_property_no_failures(self):
        """Test success is True when no failures."""
        result = MigrationResult(
            store_name="test",
            total_records=10,
            migrated_records=8,
            already_encrypted=2,
            failed_records=0,
        )
        assert result.success is True

    def test_success_property_with_failures(self):
        """Test success is False when there are failures."""
        result = MigrationResult(
            store_name="test",
            total_records=10,
            migrated_records=7,
            failed_records=3,
        )
        assert result.success is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.now(timezone.utc)
        result = MigrationResult(
            store_name="test_store",
            total_records=10,
            migrated_records=5,
            already_encrypted=3,
            failed_records=2,
            errors=["Error 1", "Error 2"],
            started_at=now,
            completed_at=now,
            duration_seconds=1.5,
        )

        data = result.to_dict()

        assert data["store_name"] == "test_store"
        assert data["total_records"] == 10
        assert data["migrated_records"] == 5
        assert data["already_encrypted"] == 3
        assert data["failed_records"] == 2
        assert data["errors"] == ["Error 1", "Error 2"]
        assert data["duration_seconds"] == 1.5
        assert data["success"] is False

    def test_to_dict_truncates_errors(self):
        """Test that to_dict truncates error list to 10 items."""
        result = MigrationResult(
            store_name="test",
            errors=[f"Error {i}" for i in range(20)],
        )

        data = result.to_dict()
        assert len(data["errors"]) == 10


class TestFieldEncryptionDetection:
    """Tests for is_field_encrypted and needs_migration."""

    def test_is_field_encrypted_dict_with_marker(self):
        """Test detection of encrypted dict field."""
        value = {"_encrypted": True, "ciphertext": "abc123", "nonce": "xyz"}
        assert is_field_encrypted(value) is True

    def test_is_field_encrypted_dict_without_marker(self):
        """Test detection of unencrypted dict field."""
        value = {"api_key": "secret123"}
        assert is_field_encrypted(value) is False

    def test_is_field_encrypted_string(self):
        """Test that string values are not encrypted."""
        assert is_field_encrypted("secret123") is False

    def test_is_field_encrypted_none(self):
        """Test that None values are not encrypted."""
        assert is_field_encrypted(None) is False

    def test_needs_migration_plaintext_fields(self):
        """Test detection of records needing migration."""
        record = {
            "id": "123",
            "name": "Test",
            "api_key": "secret_value",
        }
        sensitive_fields = ["api_key", "api_secret"]

        assert needs_migration(record, sensitive_fields) is True

    def test_needs_migration_already_encrypted(self):
        """Test that already encrypted records don't need migration."""
        record = {
            "id": "123",
            "name": "Test",
            "api_key": {"_encrypted": True, "ciphertext": "abc"},
        }
        sensitive_fields = ["api_key"]

        assert needs_migration(record, sensitive_fields) is False

    def test_needs_migration_no_sensitive_fields(self):
        """Test records without sensitive fields don't need migration."""
        record = {
            "id": "123",
            "name": "Test",
        }
        sensitive_fields = ["api_key", "api_secret"]

        assert needs_migration(record, sensitive_fields) is False

    def test_needs_migration_null_sensitive_field(self):
        """Test that null sensitive fields don't trigger migration."""
        record = {
            "id": "123",
            "api_key": None,
        }
        sensitive_fields = ["api_key"]

        assert needs_migration(record, sensitive_fields) is False


class TestEncryptionMigrator:
    """Tests for EncryptionMigrator class."""

    def test_init_defaults(self):
        """Test default initialization."""
        migrator = EncryptionMigrator()

        assert migrator._encryption_service is None
        assert migrator._batch_size == 100
        assert migrator._dry_run is False

    def test_init_custom_values(self):
        """Test custom initialization."""
        mock_service = MagicMock()
        migrator = EncryptionMigrator(
            encryption_service=mock_service,
            batch_size=50,
            dry_run=True,
        )

        assert migrator._encryption_service is mock_service
        assert migrator._batch_size == 50
        assert migrator._dry_run is True

    def test_get_encryption_service_lazy_load(self):
        """Test lazy loading of encryption service."""
        migrator = EncryptionMigrator()

        with patch("aragora.security.encryption.get_encryption_service") as mock_get:
            mock_service = MagicMock()
            mock_get.return_value = mock_service

            service = migrator._get_encryption_service()

            assert service is mock_service
            mock_get.assert_called_once()

            # Second call should use cached service
            service2 = migrator._get_encryption_service()
            assert service2 is mock_service
            mock_get.assert_called_once()  # Still only one call

    def test_migrate_record(self):
        """Test single record migration."""
        mock_service = MagicMock()
        mock_service.encrypt_fields.return_value = {
            "id": "123",
            "api_key": {"_encrypted": True, "ciphertext": "encrypted"},
        }

        migrator = EncryptionMigrator(encryption_service=mock_service)

        record = {"id": "123", "api_key": "secret"}
        result = migrator.migrate_record(record, ["api_key"], record_id="123")

        mock_service.encrypt_fields.assert_called_once_with(
            record, ["api_key"], associated_data="123"
        )
        assert result["api_key"]["_encrypted"] is True

    def test_migrate_store_success(self):
        """Test successful store migration."""
        mock_service = MagicMock()
        mock_service.encrypt_fields.return_value = {
            "id": "1",
            "api_key": {"_encrypted": True, "ciphertext": "enc"},
        }

        migrator = EncryptionMigrator(encryption_service=mock_service)

        records = [
            {"id": "1", "api_key": "secret1"},
            {"id": "2", "api_key": {"_encrypted": True, "ciphertext": "already"}},
            {"id": "3", "api_key": "secret3"},
        ]

        def list_fn():
            return records

        def save_fn(record_id, record):
            return True

        result = migrator.migrate_store(
            store_name="test_store",
            list_fn=list_fn,
            save_fn=save_fn,
            sensitive_fields=["api_key"],
            id_field="id",
        )

        assert result.store_name == "test_store"
        assert result.total_records == 3
        assert result.migrated_records == 2
        assert result.already_encrypted == 1
        assert result.failed_records == 0
        assert result.success is True

    def test_migrate_store_dry_run(self):
        """Test dry run mode doesn't save."""
        mock_service = MagicMock()
        migrator = EncryptionMigrator(encryption_service=mock_service, dry_run=True)

        records = [{"id": "1", "api_key": "secret"}]
        save_called = []

        def list_fn():
            return records

        def save_fn(record_id, record):
            save_called.append(record_id)
            return True

        result = migrator.migrate_store(
            store_name="test_store",
            list_fn=list_fn,
            save_fn=save_fn,
            sensitive_fields=["api_key"],
        )

        assert result.migrated_records == 1
        assert len(save_called) == 0  # Save should not be called
        mock_service.encrypt_fields.assert_not_called()

    def test_migrate_store_save_failure(self):
        """Test handling of save failures."""
        mock_service = MagicMock()
        mock_service.encrypt_fields.return_value = {"id": "1", "api_key": {"_encrypted": True}}

        migrator = EncryptionMigrator(encryption_service=mock_service)

        records = [{"id": "1", "api_key": "secret"}]

        def list_fn():
            return records

        def save_fn(record_id, record):
            return False  # Simulate save failure

        result = migrator.migrate_store(
            store_name="test_store",
            list_fn=list_fn,
            save_fn=save_fn,
            sensitive_fields=["api_key"],
        )

        assert result.failed_records == 1
        assert result.migrated_records == 0
        assert result.success is False
        assert len(result.errors) == 1

    def test_migrate_store_encryption_error(self):
        """Test handling of encryption errors."""
        mock_service = MagicMock()
        mock_service.encrypt_fields.side_effect = Exception("Encryption failed")

        migrator = EncryptionMigrator(encryption_service=mock_service)

        records = [{"id": "1", "api_key": "secret"}]

        def list_fn():
            return records

        def save_fn(record_id, record):
            return True

        result = migrator.migrate_store(
            store_name="test_store",
            list_fn=list_fn,
            save_fn=save_fn,
            sensitive_fields=["api_key"],
        )

        assert result.failed_records == 1
        assert result.success is False


class TestStoreMigrations:
    """Tests for store-specific migration functions."""

    def test_migrate_integration_store_import_error(self):
        """Test graceful handling of import error."""
        with patch("aragora.storage.integration_store.get_integration_store", side_effect=ImportError("Not found")):
            result = migrate_integration_store()

            assert result.store_name == "integration_store"
            assert len(result.errors) > 0

    def test_migrate_gmail_token_store_import_error(self):
        """Test graceful handling of import error."""
        with patch("aragora.storage.gmail_token_store.get_gmail_token_store", side_effect=ImportError("Not found")):
            result = migrate_gmail_token_store()

            assert result.store_name == "gmail_token_store"
            assert len(result.errors) > 0

    def test_migrate_sync_store_empty(self):
        """Test migration with empty sync store."""
        result = migrate_sync_store(dry_run=True)

        assert result.store_name == "sync_store"
        # Should complete successfully even with empty store
        assert result.failed_records == 0


class TestStartupMigrationConfig:
    """Tests for startup migration configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StartupMigrationConfig()

        assert config.enabled is False
        assert config.dry_run is False
        assert config.stores == ["integration", "gmail", "sync"]
        assert config.fail_on_error is False

    def test_get_config_from_env_enabled(self):
        """Test config loading with migration enabled."""
        with patch.dict("os.environ", {
            "ARAGORA_MIGRATE_ON_STARTUP": "true",
            "ARAGORA_MIGRATION_DRY_RUN": "1",
            "ARAGORA_MIGRATION_STORES": "integration,sync",
            "ARAGORA_MIGRATION_FAIL_ON_ERROR": "yes",
        }):
            config = get_startup_migration_config()

            assert config.enabled is True
            assert config.dry_run is True
            assert config.stores == ["integration", "sync"]
            assert config.fail_on_error is True

    def test_get_config_from_env_disabled(self):
        """Test config loading with migration disabled."""
        with patch.dict("os.environ", {
            "ARAGORA_MIGRATE_ON_STARTUP": "false",
        }, clear=True):
            config = get_startup_migration_config()

            assert config.enabled is False


class TestRunStartupMigration:
    """Tests for run_startup_migration function."""

    def test_disabled_returns_empty(self):
        """Test that disabled migration returns empty list."""
        config = StartupMigrationConfig(enabled=False)

        results = run_startup_migration(config)

        assert results == []

    def test_enabled_runs_migrations(self):
        """Test that enabled migration runs store migrations."""
        config = StartupMigrationConfig(
            enabled=True,
            dry_run=True,
            stores=["integration"],
        )

        mock_result = MigrationResult(store_name="integration_store", migrated_records=5)

        with patch("aragora.security.migration.migrate_integration_store", return_value=mock_result) as mock_migrate:
            results = run_startup_migration(config)

            assert len(results) == 1
            assert results[0].store_name == "integration_store"
            mock_migrate.assert_called_once_with(dry_run=True)

    def test_unknown_store_skipped(self):
        """Test that unknown store names are skipped."""
        config = StartupMigrationConfig(
            enabled=True,
            stores=["unknown_store"],
        )

        results = run_startup_migration(config)

        assert len(results) == 0

    def test_fail_on_error_raises(self):
        """Test that fail_on_error causes exception on failure."""
        config = StartupMigrationConfig(
            enabled=True,
            stores=["integration"],
            fail_on_error=True,
        )

        mock_result = MigrationResult(
            store_name="integration_store",
            failed_records=5,
            errors=["Something went wrong"],
        )

        with patch("aragora.security.migration.migrate_integration_store", return_value=mock_result):
            with pytest.raises(RuntimeError) as exc_info:
                run_startup_migration(config)

            assert "Migration failed for integration" in str(exc_info.value)

    def test_fail_on_error_false_continues(self):
        """Test that fail_on_error=False continues after failure."""
        config = StartupMigrationConfig(
            enabled=True,
            stores=["integration", "sync"],
            fail_on_error=False,
        )

        failed_result = MigrationResult(
            store_name="integration_store",
            failed_records=5,
        )
        success_result = MigrationResult(
            store_name="sync_store",
            migrated_records=3,
        )

        with patch("aragora.security.migration.migrate_integration_store", return_value=failed_result):
            with patch("aragora.security.migration.migrate_sync_store", return_value=success_result):
                results = run_startup_migration(config)

                assert len(results) == 2
                assert results[0].success is False
                assert results[1].success is True

    def test_migration_exception_with_fail_on_error(self):
        """Test exception handling with fail_on_error=True."""
        config = StartupMigrationConfig(
            enabled=True,
            stores=["integration"],
            fail_on_error=True,
        )

        with patch("aragora.security.migration.migrate_integration_store", side_effect=Exception("DB error")):
            with pytest.raises(Exception) as exc_info:
                run_startup_migration(config)

            assert "DB error" in str(exc_info.value)

    def test_migration_exception_without_fail_on_error(self):
        """Test exception handling with fail_on_error=False."""
        config = StartupMigrationConfig(
            enabled=True,
            stores=["integration", "sync"],
            fail_on_error=False,
        )

        success_result = MigrationResult(store_name="sync_store")

        with patch("aragora.security.migration.migrate_integration_store", side_effect=Exception("DB error")):
            with patch("aragora.security.migration.migrate_sync_store", return_value=success_result):
                results = run_startup_migration(config)

                # Should have one result for sync store
                assert len(results) == 1
                assert results[0].store_name == "sync_store"

    def test_all_stores_run(self):
        """Test that all configured stores are migrated."""
        config = StartupMigrationConfig(
            enabled=True,
            stores=["integration", "gmail", "sync"],
        )

        results_map = {
            "integration": MigrationResult(store_name="integration_store"),
            "gmail": MigrationResult(store_name="gmail_token_store"),
            "sync": MigrationResult(store_name="sync_store"),
        }

        with patch("aragora.security.migration.migrate_integration_store", return_value=results_map["integration"]):
            with patch("aragora.security.migration.migrate_gmail_token_store", return_value=results_map["gmail"]):
                with patch("aragora.security.migration.migrate_sync_store", return_value=results_map["sync"]):
                    results = run_startup_migration(config)

                    assert len(results) == 3
                    store_names = {r.store_name for r in results}
                    assert store_names == {"integration_store", "gmail_token_store", "sync_store"}
