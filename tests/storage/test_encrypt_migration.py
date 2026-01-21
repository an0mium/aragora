"""
Tests for the data encryption migration utility.

Tests cover:
- Migration result tracking
- Detection of unencrypted data
- Dry run mode
- Error handling
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.storage.migrations.encrypt_existing_data import (
    MigrationResult,
    _needs_migration,
    migrate_sync_store,
    migrate_integration_store,
    migrate_all,
)


class TestMigrationResult:
    """Tests for MigrationResult dataclass."""

    def test_success_when_no_failures(self):
        """Result is successful when no records failed."""
        result = MigrationResult(
            store_name="TestStore",
            total_records=100,
            migrated=50,
            already_encrypted=50,
            failed=0,
            errors=[],
            dry_run=False,
        )
        assert result.success is True

    def test_failure_when_records_failed(self):
        """Result is not successful when records failed."""
        result = MigrationResult(
            store_name="TestStore",
            total_records=100,
            migrated=45,
            already_encrypted=50,
            failed=5,
            errors=["Error 1", "Error 2"],
            dry_run=False,
        )
        assert result.success is False

    def test_str_representation(self):
        """String representation includes key metrics."""
        result = MigrationResult(
            store_name="TestStore",
            total_records=100,
            migrated=50,
            already_encrypted=50,
            failed=0,
            errors=[],
            dry_run=False,
        )
        s = str(result)
        assert "TestStore" in s
        assert "100" in s  # total
        assert "50" in s   # migrated

    def test_str_shows_dry_run_mode(self):
        """String representation shows dry run indicator."""
        result = MigrationResult(
            store_name="TestStore",
            total_records=10,
            migrated=5,
            already_encrypted=5,
            failed=0,
            errors=[],
            dry_run=True,
        )
        s = str(result)
        assert "DRY RUN" in s


class TestNeedsMigration:
    """Tests for _needs_migration function."""

    def test_empty_config_does_not_need_migration(self):
        """Empty config doesn't need migration."""
        assert _needs_migration({}, ["password", "token"]) is False

    def test_none_config_does_not_need_migration(self):
        """None config doesn't need migration."""
        assert _needs_migration(None, ["password", "token"]) is False

    def test_non_sensitive_fields_dont_need_migration(self):
        """Config with only non-sensitive fields doesn't need migration."""
        config = {"name": "test", "host": "localhost", "port": 5432}
        assert _needs_migration(config, ["password", "token"]) is False

    def test_plaintext_sensitive_field_needs_migration(self):
        """Config with plaintext sensitive field needs migration."""
        config = {"name": "test", "password": "secret123"}
        assert _needs_migration(config, ["password", "token"]) is True

    def test_encrypted_sensitive_field_does_not_need_migration(self):
        """Config with encrypted sensitive field doesn't need migration."""
        config = {
            "name": "test",
            "password": {"_encrypted": True, "_value": "encrypted_data"},
        }
        assert _needs_migration(config, ["password", "token"]) is False

    def test_mixed_fields_need_migration(self):
        """Config with mix of encrypted and plaintext needs migration."""
        config = {
            "password": {"_encrypted": True, "_value": "encrypted_data"},
            "api_token": "plaintext_token",  # This needs migration
        }
        assert _needs_migration(config, ["password", "token"]) is True

    def test_none_sensitive_value_does_not_need_migration(self):
        """Config with None sensitive value doesn't need migration."""
        config = {"name": "test", "password": None}
        assert _needs_migration(config, ["password", "token"]) is False

    def test_case_insensitive_keyword_matching(self):
        """Keyword matching is case-insensitive."""
        config = {"API_TOKEN": "secret", "Password": "abc123"}
        assert _needs_migration(config, ["password", "token"]) is True


class TestMigrateSyncStore:
    """Tests for migrate_sync_store function."""

    @pytest.mark.asyncio
    async def test_handles_encryption_not_available(self):
        """Returns error when encryption not available."""
        with patch("aragora.storage.migrations.encrypt_existing_data.CRYPTO_AVAILABLE", False):
            result = await migrate_sync_store(dry_run=True)

        assert result.failed == 0
        assert len(result.errors) > 0
        assert "not available" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_returns_result_with_store_name(self):
        """Result has correct store name."""
        with patch("aragora.storage.migrations.encrypt_existing_data.CRYPTO_AVAILABLE", False):
            result = await migrate_sync_store(dry_run=True)

        assert result.store_name == "SyncStore"
        assert result.dry_run is True


class TestMigrateIntegrationStore:
    """Tests for migrate_integration_store function."""

    @pytest.mark.asyncio
    async def test_handles_encryption_not_available(self):
        """Returns error when encryption not available."""
        with patch("aragora.storage.migrations.encrypt_existing_data.CRYPTO_AVAILABLE", False):
            result = await migrate_integration_store(dry_run=True)

        assert len(result.errors) > 0


class TestMigrateAll:
    """Tests for migrate_all function."""

    @pytest.mark.asyncio
    async def test_runs_all_migrations(self):
        """migrate_all runs all individual migrations."""
        with patch("aragora.storage.migrations.encrypt_existing_data.CRYPTO_AVAILABLE", True):
            with patch("aragora.storage.migrations.encrypt_existing_data.get_encryption_service") as mock_service:
                mock_service.return_value.get_active_key_id.return_value = "test-key"

                with patch("aragora.storage.migrations.encrypt_existing_data.migrate_sync_store") as mock_sync:
                    with patch("aragora.storage.migrations.encrypt_existing_data.migrate_integration_store") as mock_int:
                        with patch("aragora.storage.migrations.encrypt_existing_data.migrate_gmail_tokens") as mock_gmail:
                            mock_sync.return_value = MigrationResult(
                                store_name="SyncStore", total_records=0, migrated=0,
                                already_encrypted=0, failed=0, errors=[], dry_run=True
                            )
                            mock_int.return_value = MigrationResult(
                                store_name="IntegrationStore", total_records=0, migrated=0,
                                already_encrypted=0, failed=0, errors=[], dry_run=True
                            )
                            mock_gmail.return_value = MigrationResult(
                                store_name="GmailTokenStore", total_records=0, migrated=0,
                                already_encrypted=0, failed=0, errors=[], dry_run=True
                            )

                            results = await migrate_all(dry_run=True)

                            assert len(results) == 3
                            mock_sync.assert_called_once_with(dry_run=True)
                            mock_int.assert_called_once_with(dry_run=True)
                            mock_gmail.assert_called_once_with(dry_run=True)
