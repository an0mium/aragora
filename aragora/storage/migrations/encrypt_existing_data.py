"""
Migration utility to encrypt existing plaintext data.

This module provides functions to migrate existing unencrypted data to use
the new field-level encryption. It's designed to be run during a maintenance
window or as a background job.

Features:
- Detects which records need migration (missing _encrypted markers)
- Re-saves records to trigger encryption
- Provides progress logging and error handling
- Supports dry-run mode for testing

Usage:
    # From Python
    from aragora.storage.migrations.encrypt_existing_data import (
        migrate_sync_store,
        migrate_integration_store,
        migrate_gmail_tokens,
    )

    # Migrate all stores
    await migrate_sync_store(dry_run=False)
    await migrate_integration_store(dry_run=False)
    await migrate_gmail_tokens(dry_run=False)

    # CLI
    python -m aragora.storage.migrations.encrypt_existing_data --all
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Check encryption availability
try:
    from aragora.security.encryption import CRYPTO_AVAILABLE, get_encryption_service
except ImportError:
    CRYPTO_AVAILABLE = False

# Check metrics availability
try:
    from aragora.observability.metrics import record_migration_record, record_migration_error

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

    def record_migration_record(*args, **kwargs):
        pass

    def record_migration_error(*args, **kwargs):
        pass


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    store_name: str
    total_records: int
    migrated: int
    already_encrypted: int
    failed: int
    errors: List[str]
    dry_run: bool

    @property
    def success(self) -> bool:
        return self.failed == 0

    def __str__(self) -> str:
        mode = "(DRY RUN)" if self.dry_run else ""
        return (
            f"{self.store_name} Migration {mode}:\n"
            f"  Total records: {self.total_records}\n"
            f"  Already encrypted: {self.already_encrypted}\n"
            f"  Migrated: {self.migrated}\n"
            f"  Failed: {self.failed}"
        )


def _needs_migration(config: Dict[str, Any], sensitive_keywords: List[str]) -> bool:
    """
    Check if a config has any plaintext sensitive fields.

    Returns True if any sensitive field exists but is not encrypted.
    """
    if not config:
        return False

    for key, value in config.items():
        # Check if this is a sensitive key
        key_lower = key.lower()
        is_sensitive = any(kw in key_lower for kw in sensitive_keywords)

        if is_sensitive and value is not None:
            # Check if it's already encrypted
            if isinstance(value, dict) and value.get("_encrypted"):
                continue
            # Has sensitive data that's not encrypted
            return True

    return False


async def migrate_sync_store(
    dry_run: bool = True,
    batch_size: int = 100,
) -> MigrationResult:
    """
    Migrate SyncStore connector configs to use encryption.

    Args:
        dry_run: If True, only report what would be migrated
        batch_size: Number of records to process in each batch

    Returns:
        MigrationResult with counts and any errors
    """
    from aragora.connectors.enterprise.sync_store import (
        SyncStore,
        CREDENTIAL_KEYWORDS,
    )

    result = MigrationResult(
        store_name="SyncStore",
        total_records=0,
        migrated=0,
        already_encrypted=0,
        failed=0,
        errors=[],
        dry_run=dry_run,
    )

    if not CRYPTO_AVAILABLE:
        result.errors.append("Encryption not available (cryptography library missing)")
        return result

    try:
        store = SyncStore(use_encryption=True)
        await store.initialize()

        # Get all connectors
        connectors = await store.list_connectors()
        result.total_records = len(connectors)

        for connector in connectors:
            connector_id = connector.id
            config = connector.config

            try:
                if _needs_migration(config, list(CREDENTIAL_KEYWORDS)):
                    if dry_run:
                        logger.info(f"[DRY RUN] Would migrate connector: {connector_id}")
                    else:
                        # Re-save triggers encryption
                        await store.save_connector(connector_id, connector)
                        logger.info(f"Migrated connector: {connector_id}")
                    result.migrated += 1
                    record_migration_record("sync_store", "migrated")
                else:
                    result.already_encrypted += 1
                    record_migration_record("sync_store", "skipped")
            except Exception as e:
                result.failed += 1
                result.errors.append(f"Connector {connector_id}: {str(e)}")
                logger.error(f"Failed to migrate connector {connector_id}: {e}")
                record_migration_record("sync_store", "failed")
                record_migration_error("sync_store", type(e).__name__)

    except Exception as e:
        result.errors.append(f"Store initialization failed: {str(e)}")
        logger.error(f"Failed to initialize SyncStore: {e}")

    return result


async def migrate_integration_store(
    dry_run: bool = True,
    batch_size: int = 100,
) -> MigrationResult:
    """
    Migrate IntegrationStore settings to use encryption.

    Args:
        dry_run: If True, only report what would be migrated
        batch_size: Number of records to process in each batch

    Returns:
        MigrationResult with counts and any errors
    """
    result = MigrationResult(
        store_name="IntegrationStore",
        total_records=0,
        migrated=0,
        already_encrypted=0,
        failed=0,
        errors=[],
        dry_run=dry_run,
    )

    if not CRYPTO_AVAILABLE:
        result.errors.append("Encryption not available (cryptography library missing)")
        return result

    try:
        from aragora.storage.integration_store import (
            get_integration_store,
            SENSITIVE_KEYS,
        )

        store = get_integration_store(use_encryption=True)

        # Get all integrations
        integrations = store.list_all()
        result.total_records = len(integrations)

        for integration in integrations:
            integration_name = integration.name
            settings = integration.settings

            try:
                if _needs_migration(settings, list(SENSITIVE_KEYS)):
                    if dry_run:
                        logger.info(f"[DRY RUN] Would migrate integration: {integration_name}")
                    else:
                        # Re-save triggers encryption
                        store.save(integration)
                        logger.info(f"Migrated integration: {integration_name}")
                    result.migrated += 1
                    record_migration_record("integration_store", "migrated")
                else:
                    result.already_encrypted += 1
                    record_migration_record("integration_store", "skipped")
            except Exception as e:
                result.failed += 1
                result.errors.append(f"Integration {integration_name}: {str(e)}")
                logger.error(f"Failed to migrate integration {integration_name}: {e}")
                record_migration_record("integration_store", "failed")
                record_migration_error("integration_store", type(e).__name__)

    except Exception as e:
        result.errors.append(f"Store initialization failed: {str(e)}")
        logger.error(f"Failed to initialize IntegrationStore: {e}")

    return result


async def migrate_gmail_tokens(
    dry_run: bool = True,
    batch_size: int = 100,
) -> MigrationResult:
    """
    Migrate GmailTokenStore tokens to use encryption.

    Args:
        dry_run: If True, only report what would be migrated
        batch_size: Number of records to process in each batch

    Returns:
        MigrationResult with counts and any errors
    """
    result = MigrationResult(
        store_name="GmailTokenStore",
        total_records=0,
        migrated=0,
        already_encrypted=0,
        failed=0,
        errors=[],
        dry_run=dry_run,
    )

    if not CRYPTO_AVAILABLE:
        result.errors.append("Encryption not available (cryptography library missing)")
        return result

    try:
        from aragora.storage.gmail_token_store import (
            get_gmail_token_store,
            ENCRYPTED_FIELDS,
        )

        store = get_gmail_token_store(use_encryption=True)

        # Get all users with tokens
        users = await store.list_users()
        result.total_records = len(users)

        for user_id in users:
            try:
                state = await store.get_user_state(user_id)
                if state is None:
                    continue

                # Check if tokens need migration
                token_dict = {
                    "access_token": state.access_token,
                    "refresh_token": state.refresh_token,
                }

                if _needs_migration(token_dict, ENCRYPTED_FIELDS):
                    if dry_run:
                        logger.info(f"[DRY RUN] Would migrate Gmail tokens for: {user_id}")
                    else:
                        # Re-save triggers encryption
                        await store.save_user_state(user_id, state)
                        logger.info(f"Migrated Gmail tokens for: {user_id}")
                    result.migrated += 1
                    record_migration_record("gmail_token_store", "migrated")
                else:
                    result.already_encrypted += 1
                    record_migration_record("gmail_token_store", "skipped")
            except Exception as e:
                result.failed += 1
                result.errors.append(f"User {user_id}: {str(e)}")
                logger.error(f"Failed to migrate Gmail tokens for {user_id}: {e}")
                record_migration_record("gmail_token_store", "failed")
                record_migration_error("gmail_token_store", type(e).__name__)

    except ImportError:
        result.errors.append("GmailTokenStore not available")
    except Exception as e:
        result.errors.append(f"Store initialization failed: {str(e)}")
        logger.error(f"Failed to initialize GmailTokenStore: {e}")

    return result


async def migrate_all(dry_run: bool = True) -> List[MigrationResult]:
    """
    Run all migration functions.

    Args:
        dry_run: If True, only report what would be migrated

    Returns:
        List of MigrationResult for each store
    """
    results = []

    logger.info(f"Starting data encryption migration (dry_run={dry_run})")

    # Check encryption key is set
    if not CRYPTO_AVAILABLE:
        logger.error("Encryption not available - install cryptography library")
        return results

    try:
        service = get_encryption_service()
        key_id = service.get_active_key_id()
        logger.info(f"Using encryption key: {key_id}")
    except Exception as e:
        logger.error(f"Failed to get encryption service: {e}")
        logger.error("Set ARAGORA_ENCRYPTION_KEY environment variable")
        return results

    # Run migrations
    results.append(await migrate_sync_store(dry_run=dry_run))
    results.append(await migrate_integration_store(dry_run=dry_run))
    results.append(await migrate_gmail_tokens(dry_run=dry_run))

    # Summary
    total_migrated = sum(r.migrated for r in results)
    total_failed = sum(r.failed for r in results)

    if dry_run:
        logger.info(f"Migration preview complete: {total_migrated} records would be migrated")
    else:
        logger.info(f"Migration complete: {total_migrated} records migrated, {total_failed} failed")

    return results


def main():
    """CLI entry point for migration."""
    parser = argparse.ArgumentParser(
        description="Migrate existing data to use field-level encryption"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate all stores",
    )
    parser.add_argument(
        "--sync-store",
        action="store_true",
        help="Migrate SyncStore connector configs",
    )
    parser.add_argument(
        "--integration-store",
        action="store_true",
        help="Migrate IntegrationStore settings",
    )
    parser.add_argument(
        "--gmail-tokens",
        action="store_true",
        help="Migrate GmailTokenStore tokens",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform migration (default is dry-run)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dry_run = not args.execute

    if dry_run:
        pass

    async def run():
        results = []

        if args.all or not any([args.sync_store, args.integration_store, args.gmail_tokens]):
            results = await migrate_all(dry_run=dry_run)
        else:
            if args.sync_store:
                results.append(await migrate_sync_store(dry_run=dry_run))
            if args.integration_store:
                results.append(await migrate_integration_store(dry_run=dry_run))
            if args.gmail_tokens:
                results.append(await migrate_gmail_tokens(dry_run=dry_run))

        # Print results
        for result in results:
            if result.errors:
                for error in result.errors[:5]:
                    pass
                if len(result.errors) > 5:
                    pass

        # Exit with error if any failures
        if any(not r.success for r in results):
            sys.exit(1)

    asyncio.run(run())


if __name__ == "__main__":
    main()
