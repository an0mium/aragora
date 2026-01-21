"""
Encryption Migration Utilities.

Provides tools for migrating plaintext secrets to encrypted format
and running migrations on application startup.

Features:
- Automatic detection of plaintext vs encrypted fields
- Background migration for existing records
- Startup migration option
- Migration progress tracking and reporting
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    store_name: str
    total_records: int = 0
    migrated_records: int = 0
    already_encrypted: int = 0
    failed_records: int = 0
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """Check if migration was successful."""
        return self.failed_records == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "store_name": self.store_name,
            "total_records": self.total_records,
            "migrated_records": self.migrated_records,
            "already_encrypted": self.already_encrypted,
            "failed_records": self.failed_records,
            "errors": self.errors[:10],  # Limit error details
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
        }


def is_field_encrypted(value: Any) -> bool:
    """Check if a field value is already encrypted."""
    if isinstance(value, dict):
        return value.get("_encrypted") is True
    return False


def needs_migration(record: Dict[str, Any], sensitive_fields: List[str]) -> bool:
    """Check if a record has plaintext sensitive fields that need migration."""
    for field_name in sensitive_fields:
        if field_name in record:
            value = record[field_name]
            if value is not None and not is_field_encrypted(value):
                return True
    return False


class EncryptionMigrator:
    """
    Handles migration of plaintext data to encrypted format.

    This migrator works with any store that implements a basic interface:
    - list_all() -> List[Dict] or similar
    - save(record) or update(id, record)
    """

    def __init__(
        self,
        encryption_service: Optional[Any] = None,
        batch_size: int = 100,
        dry_run: bool = False,
    ):
        """
        Initialize the migrator.

        Args:
            encryption_service: Encryption service instance (uses global if not provided)
            batch_size: Number of records to process in each batch
            dry_run: If True, report what would be migrated without making changes
        """
        self._encryption_service = encryption_service
        self._batch_size = batch_size
        self._dry_run = dry_run

    def _get_encryption_service(self):
        """Get encryption service lazily."""
        if self._encryption_service is None:
            from aragora.security.encryption import get_encryption_service

            self._encryption_service = get_encryption_service()
        return self._encryption_service

    def migrate_record(
        self,
        record: Dict[str, Any],
        sensitive_fields: List[str],
        record_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Migrate a single record's sensitive fields to encrypted format.

        Args:
            record: The record to migrate
            sensitive_fields: Fields that should be encrypted
            record_id: Optional record ID for associated data

        Returns:
            Migrated record with encrypted fields
        """
        service = self._get_encryption_service()
        return service.encrypt_fields(record, sensitive_fields, associated_data=record_id)

    def migrate_store(
        self,
        store_name: str,
        list_fn: Callable[[], List[Dict[str, Any]]],
        save_fn: Callable[[str, Dict[str, Any]], bool],
        sensitive_fields: List[str],
        id_field: str = "id",
    ) -> MigrationResult:
        """
        Migrate all records in a store.

        Args:
            store_name: Name of the store (for logging/reporting)
            list_fn: Function that returns all records
            save_fn: Function that saves a record (id, record) -> success
            sensitive_fields: Fields that should be encrypted
            id_field: Name of the ID field in records

        Returns:
            MigrationResult with statistics
        """
        result = MigrationResult(store_name=store_name)

        try:
            records = list_fn()
            result.total_records = len(records)

            logger.info(
                f"Starting migration for {store_name}: {result.total_records} records"
            )

            for record in records:
                record_id = record.get(id_field, "unknown")

                try:
                    if not needs_migration(record, sensitive_fields):
                        result.already_encrypted += 1
                        continue

                    if self._dry_run:
                        result.migrated_records += 1
                        logger.debug(f"[DRY RUN] Would migrate record: {record_id}")
                        continue

                    # Migrate the record
                    migrated = self.migrate_record(
                        record, sensitive_fields, record_id=str(record_id)
                    )

                    # Save back
                    if save_fn(record_id, migrated):
                        result.migrated_records += 1
                        logger.debug(f"Migrated record: {record_id}")
                    else:
                        result.failed_records += 1
                        result.errors.append(f"Failed to save record: {record_id}")

                except Exception as e:
                    result.failed_records += 1
                    result.errors.append(f"Error migrating {record_id}: {str(e)}")
                    logger.warning(f"Failed to migrate record {record_id}: {e}")

            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()

            logger.info(
                f"Migration complete for {store_name}: "
                f"{result.migrated_records} migrated, "
                f"{result.already_encrypted} already encrypted, "
                f"{result.failed_records} failed"
            )

        except Exception as e:
            result.errors.append(f"Migration failed: {str(e)}")
            result.failed_records = result.total_records
            logger.error(f"Migration failed for {store_name}: {e}")

        return result


# Store-specific migration functions


def migrate_integration_store(dry_run: bool = False) -> MigrationResult:
    """Migrate integration store secrets."""
    try:
        from aragora.storage.integration_store import get_integration_store

        store = get_integration_store()
        migrator = EncryptionMigrator(dry_run=dry_run)

        # Sensitive fields in integration configs
        sensitive_fields = [
            "api_key",
            "api_secret",
            "access_token",
            "refresh_token",
            "password",
            "secret",
            "credentials",
            "token",
        ]

        def list_all():
            return list(store.list_all())

        def save(integration_id, record):
            return store.save(record)

        return migrator.migrate_store(
            store_name="integration_store",
            list_fn=list_all,
            save_fn=save,
            sensitive_fields=sensitive_fields,
            id_field="integration_id",
        )
    except ImportError as e:
        logger.warning(f"Integration store not available: {e}")
        return MigrationResult(store_name="integration_store", errors=[str(e)])


def migrate_gmail_token_store(dry_run: bool = False) -> MigrationResult:
    """Migrate Gmail token store secrets."""
    try:
        from aragora.storage.gmail_token_store import get_gmail_token_store

        store = get_gmail_token_store()
        migrator = EncryptionMigrator(dry_run=dry_run)

        sensitive_fields = ["access_token", "refresh_token"]

        def list_all():
            # GmailTokenStore may not have list_all, get states instead
            if hasattr(store, "list_all"):
                return list(store.list_all())
            return []

        def save(user_id, record):
            return store.save_state(user_id, record)

        return migrator.migrate_store(
            store_name="gmail_token_store",
            list_fn=list_all,
            save_fn=save,
            sensitive_fields=sensitive_fields,
            id_field="user_id",
        )
    except ImportError as e:
        logger.warning(f"Gmail token store not available: {e}")
        return MigrationResult(store_name="gmail_token_store", errors=[str(e)])


def migrate_sync_store(dry_run: bool = False) -> MigrationResult:
    """Migrate connector sync store secrets."""
    try:
        from aragora.connectors.enterprise.sync_store import get_sync_store

        store = get_sync_store()
        migrator = EncryptionMigrator(dry_run=dry_run)

        # Connector credentials
        sensitive_fields = [
            "api_key",
            "api_secret",
            "token",
            "password",
            "auth_token",
            "secret",
        ]

        def list_all():
            if hasattr(store, "list_all"):
                return list(store.list_all())
            return []

        def save(job_id, record):
            if hasattr(store, "save"):
                return store.save(record)
            return False

        return migrator.migrate_store(
            store_name="sync_store",
            list_fn=list_all,
            save_fn=save,
            sensitive_fields=sensitive_fields,
            id_field="job_id",
        )
    except ImportError as e:
        logger.warning(f"Sync store not available: {e}")
        return MigrationResult(store_name="sync_store", errors=[str(e)])


@dataclass
class StartupMigrationConfig:
    """Configuration for startup migration."""

    enabled: bool = False
    dry_run: bool = False
    stores: List[str] = field(default_factory=lambda: ["integration", "gmail", "sync"])
    fail_on_error: bool = False


def get_startup_migration_config() -> StartupMigrationConfig:
    """Get startup migration config from environment."""
    return StartupMigrationConfig(
        enabled=os.environ.get("ARAGORA_MIGRATE_ON_STARTUP", "").lower() in (
            "true",
            "1",
            "yes",
        ),
        dry_run=os.environ.get("ARAGORA_MIGRATION_DRY_RUN", "").lower() in (
            "true",
            "1",
            "yes",
        ),
        stores=os.environ.get("ARAGORA_MIGRATION_STORES", "integration,gmail,sync").split(
            ","
        ),
        fail_on_error=os.environ.get("ARAGORA_MIGRATION_FAIL_ON_ERROR", "").lower() in (
            "true",
            "1",
            "yes",
        ),
    )


def run_startup_migration(
    config: Optional[StartupMigrationConfig] = None,
) -> List[MigrationResult]:
    """
    Run encryption migration on startup.

    This function can be called during application initialization to
    migrate any plaintext secrets to encrypted format.

    Args:
        config: Migration configuration (uses env vars if not provided)

    Returns:
        List of migration results for each store

    Environment Variables:
        ARAGORA_MIGRATE_ON_STARTUP: Set to "true" to enable
        ARAGORA_MIGRATION_DRY_RUN: Set to "true" for dry run mode
        ARAGORA_MIGRATION_STORES: Comma-separated list of stores to migrate
        ARAGORA_MIGRATION_FAIL_ON_ERROR: Set to "true" to fail on errors
    """
    if config is None:
        config = get_startup_migration_config()

    if not config.enabled:
        logger.debug("Startup migration disabled")
        return []

    logger.info(
        f"Running startup migration (dry_run={config.dry_run}, stores={config.stores})"
    )

    results = []

    store_migrations = {
        "integration": migrate_integration_store,
        "gmail": migrate_gmail_token_store,
        "sync": migrate_sync_store,
    }

    for store_name in config.stores:
        store_name = store_name.strip()
        migrate_fn = store_migrations.get(store_name)

        if migrate_fn is None:
            logger.warning(f"Unknown store for migration: {store_name}")
            continue

        try:
            result = migrate_fn(dry_run=config.dry_run)
            results.append(result)

            if not result.success and config.fail_on_error:
                raise RuntimeError(
                    f"Migration failed for {store_name}: {result.errors}"
                )

        except Exception as e:
            logger.error(f"Migration error for {store_name}: {e}")
            if config.fail_on_error:
                raise

    logger.info(f"Startup migration complete: {len(results)} stores processed")
    return results


__all__ = [
    "EncryptionMigrator",
    "MigrationResult",
    "StartupMigrationConfig",
    "is_field_encrypted",
    "needs_migration",
    "migrate_integration_store",
    "migrate_gmail_token_store",
    "migrate_sync_store",
    "run_startup_migration",
    "get_startup_migration_config",
]
