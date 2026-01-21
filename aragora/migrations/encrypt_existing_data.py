"""
Data Migration Utility for Encrypting Existing Secrets.

This utility migrates unencrypted sensitive data to encrypted format.
It handles all stores that contain sensitive fields.

Usage:
    # Dry run (preview changes)
    python -m aragora.migrations.encrypt_existing_data --dry-run

    # Execute migration
    python -m aragora.migrations.encrypt_existing_data

    # Rollback (if backup exists)
    python -m aragora.migrations.encrypt_existing_data --rollback

Requirements:
    - ARAGORA_ENCRYPTION_KEY must be set
    - Database access required
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EncryptionMigration:
    """Handles migration of unencrypted data to encrypted format."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        dry_run: bool = False,
        backup_dir: Optional[str] = None,
    ):
        """Initialize the migration.

        Args:
            data_dir: Path to the data directory (default: .nomic)
            dry_run: If True, preview changes without applying
            backup_dir: Path for backups (default: data_dir/backups)
        """
        self.data_dir = Path(data_dir or os.environ.get("ARAGORA_DATA_DIR", ".nomic"))
        self.dry_run = dry_run
        self.backup_dir = Path(backup_dir) if backup_dir else self.data_dir / "backups"
        self.backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Statistics
        self.stats: dict[str, Any] = {
            "integrations_migrated": 0,
            "webhooks_migrated": 0,
            "tokens_migrated": 0,
            "sync_configs_migrated": 0,
            "errors": [],
        }

    def check_prerequisites(self) -> bool:
        """Check if prerequisites are met for migration."""
        # Check encryption key
        if not os.environ.get("ARAGORA_ENCRYPTION_KEY"):
            logger.error("ARAGORA_ENCRYPTION_KEY must be set")
            return False

        # Check if encryption is available
        try:
            from aragora.storage.encrypted_fields import is_encryption_available

            if not is_encryption_available():
                logger.error("Encryption not available - install cryptography package")
                return False
        except ImportError as e:
            logger.error(f"Cannot import encryption module: {e}")
            return False

        return True

    def create_backup(self, db_path: Path) -> Optional[Path]:
        """Create a backup of the database.

        Args:
            db_path: Path to the database file

        Returns:
            Path to the backup file, or None if backup failed
        """
        if not db_path.exists():
            logger.warning(f"Database not found: {db_path}")
            return None

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = self.backup_dir / f"{db_path.stem}_{self.backup_timestamp}.db"

        try:
            shutil.copy2(db_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    def migrate_integrations(self) -> int:
        """Migrate integration settings to encrypted format.

        Returns:
            Number of records migrated
        """
        from aragora.storage.encrypted_fields import (
            encrypt_sensitive,
            is_field_encrypted,
            SENSITIVE_FIELDS,
        )

        db_path = self.data_dir / "integrations.db"
        if not db_path.exists():
            logger.info("No integrations database found")
            return 0

        if not self.dry_run:
            self.create_backup(db_path)

        migrated = 0
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            # Get all integrations with settings
            cursor.execute(
                "SELECT id, settings_json FROM integrations WHERE settings_json IS NOT NULL"
            )
            rows = cursor.fetchall()

            for row in rows:
                record_id, settings_json = row
                if not settings_json:
                    continue

                try:
                    settings = json.loads(settings_json)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON for integration {record_id}")
                    continue

                # Check if any sensitive fields need encryption
                needs_encryption = False
                for field in SENSITIVE_FIELDS:
                    if field in settings:
                        if not is_field_encrypted(settings, field):
                            needs_encryption = True
                            break

                if not needs_encryption:
                    continue

                # Encrypt sensitive fields
                encrypted_settings = encrypt_sensitive(settings, record_id=record_id)

                if self.dry_run:
                    logger.info(f"Would encrypt integration {record_id}")
                else:
                    cursor.execute(
                        "UPDATE integrations SET settings_json = ? WHERE id = ?",
                        (json.dumps(encrypted_settings), record_id),
                    )
                    logger.debug(f"Encrypted integration {record_id}")

                migrated += 1

            if not self.dry_run:
                conn.commit()

        except Exception as e:
            logger.error(f"Error migrating integrations: {e}")
            self.stats["errors"].append(f"integrations: {e}")
            conn.rollback()
        finally:
            conn.close()

        self.stats["integrations_migrated"] = migrated
        return migrated

    def migrate_webhooks(self) -> int:
        """Migrate webhook secrets to encrypted format.

        Returns:
            Number of records migrated
        """
        from aragora.storage.encrypted_fields import is_encryption_available

        if not is_encryption_available():
            logger.warning("Encryption not available, skipping webhooks")
            return 0

        db_path = self.data_dir / "webhooks.db"
        if not db_path.exists():
            logger.info("No webhooks database found")
            return 0

        if not self.dry_run:
            self.create_backup(db_path)

        migrated = 0
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            from aragora.security.encryption import get_encryption_service

            service = get_encryption_service()

            # Get all webhooks with secrets
            cursor.execute(
                "SELECT id, secret FROM webhooks WHERE secret IS NOT NULL AND secret != ''"
            )
            rows = cursor.fetchall()

            for row in rows:
                webhook_id, secret = row

                # Check if already encrypted (base64 encoded EncryptedValue)
                if secret.startswith("gAAAAA") or "._encrypted" in secret:
                    continue

                # Encrypt the secret
                encrypted = service.encrypt(secret)
                encrypted_str = encrypted.to_base64()

                if self.dry_run:
                    logger.info(f"Would encrypt webhook {webhook_id}")
                else:
                    cursor.execute(
                        "UPDATE webhooks SET secret = ? WHERE id = ?",
                        (encrypted_str, webhook_id),
                    )
                    logger.debug(f"Encrypted webhook {webhook_id}")

                migrated += 1

            if not self.dry_run:
                conn.commit()

        except Exception as e:
            logger.error(f"Error migrating webhooks: {e}")
            self.stats["errors"].append(f"webhooks: {e}")
            conn.rollback()
        finally:
            conn.close()

        self.stats["webhooks_migrated"] = migrated
        return migrated

    def migrate_tokens(self) -> int:
        """Migrate OAuth tokens to encrypted format.

        Returns:
            Number of records migrated
        """
        from aragora.storage.encrypted_fields import is_encryption_available

        if not is_encryption_available():
            logger.warning("Encryption not available, skipping tokens")
            return 0

        db_path = self.data_dir / "gmail_tokens.db"
        if not db_path.exists():
            logger.info("No tokens database found")
            return 0

        if not self.dry_run:
            self.create_backup(db_path)

        migrated = 0
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            from aragora.security.encryption import get_encryption_service

            service = get_encryption_service()

            # Get all tokens
            cursor.execute("SELECT user_id, access_token, refresh_token FROM gmail_tokens")
            rows = cursor.fetchall()

            for row in rows:
                user_id, access_token, refresh_token = row

                # Check if already encrypted
                if access_token and (
                    access_token.startswith("gAAAAA") or "._encrypted" in access_token
                ):
                    continue

                updates = []
                params = []

                if access_token:
                    encrypted = service.encrypt(access_token)
                    updates.append("access_token = ?")
                    params.append(encrypted.to_base64())

                if refresh_token:
                    encrypted = service.encrypt(refresh_token)
                    updates.append("refresh_token = ?")
                    params.append(encrypted.to_base64())

                if not updates:
                    continue

                if self.dry_run:
                    logger.info(f"Would encrypt tokens for user {user_id}")
                else:
                    params.append(user_id)
                    cursor.execute(
                        f"UPDATE gmail_tokens SET {', '.join(updates)} WHERE user_id = ?",
                        params,
                    )
                    logger.debug(f"Encrypted tokens for user {user_id}")

                migrated += 1

            if not self.dry_run:
                conn.commit()

        except Exception as e:
            logger.error(f"Error migrating tokens: {e}")
            self.stats["errors"].append(f"tokens: {e}")
            conn.rollback()
        finally:
            conn.close()

        self.stats["tokens_migrated"] = migrated
        return migrated

    def migrate_sync_configs(self) -> int:
        """Migrate sync store configs to encrypted format.

        Returns:
            Number of records migrated
        """
        from aragora.storage.encrypted_fields import (
            encrypt_sensitive,
            is_field_encrypted,
            SENSITIVE_FIELDS,
        )

        db_path = self.data_dir / "enterprise_sync.db"
        if not db_path.exists():
            logger.info("No sync configs database found")
            return 0

        if not self.dry_run:
            self.create_backup(db_path)

        migrated = 0
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            # Get all sync configs
            cursor.execute("SELECT id, config_json FROM sync_configs WHERE config_json IS NOT NULL")
            rows = cursor.fetchall()

            for row in rows:
                config_id, config_json = row
                if not config_json:
                    continue

                try:
                    config = json.loads(config_json)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON for sync config {config_id}")
                    continue

                # Check if needs encryption
                needs_encryption = False
                for field in SENSITIVE_FIELDS:
                    if field in config:
                        if not is_field_encrypted(config, field):
                            needs_encryption = True
                            break

                if not needs_encryption:
                    continue

                # Encrypt sensitive fields
                encrypted_config = encrypt_sensitive(config, record_id=config_id)

                if self.dry_run:
                    logger.info(f"Would encrypt sync config {config_id}")
                else:
                    cursor.execute(
                        "UPDATE sync_configs SET config_json = ? WHERE id = ?",
                        (json.dumps(encrypted_config), config_id),
                    )
                    logger.debug(f"Encrypted sync config {config_id}")

                migrated += 1

            if not self.dry_run:
                conn.commit()

        except Exception as e:
            logger.error(f"Error migrating sync configs: {e}")
            self.stats["errors"].append(f"sync_configs: {e}")
            conn.rollback()
        finally:
            conn.close()

        self.stats["sync_configs_migrated"] = migrated
        return migrated

    def run(self) -> dict[str, Any]:
        """Run the full migration.

        Returns:
            Migration statistics
        """
        logger.info(f"Starting encryption migration {'(dry run)' if self.dry_run else ''}")

        if not self.check_prerequisites():
            logger.error("Prerequisites not met, aborting migration")
            return {"success": False, "error": "Prerequisites not met"}

        # Run migrations
        self.migrate_integrations()
        self.migrate_webhooks()
        self.migrate_tokens()
        self.migrate_sync_configs()

        # Summary
        total = (
            self.stats["integrations_migrated"]
            + self.stats["webhooks_migrated"]
            + self.stats["tokens_migrated"]
            + self.stats["sync_configs_migrated"]
        )

        logger.info(f"Migration complete: {total} records processed")
        logger.info(f"  Integrations: {self.stats['integrations_migrated']}")
        logger.info(f"  Webhooks: {self.stats['webhooks_migrated']}")
        logger.info(f"  Tokens: {self.stats['tokens_migrated']}")
        logger.info(f"  Sync configs: {self.stats['sync_configs_migrated']}")

        if self.stats["errors"]:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
            for error in self.stats["errors"]:
                logger.warning(f"  - {error}")

        return {
            "success": len(self.stats["errors"]) == 0,
            "dry_run": self.dry_run,
            "total_migrated": total,
            **self.stats,
        }

    def rollback(self, timestamp: Optional[str] = None) -> bool:
        """Rollback to a previous backup.

        Args:
            timestamp: Backup timestamp to restore (default: most recent)

        Returns:
            True if rollback succeeded
        """
        if not self.backup_dir.exists():
            logger.error("No backup directory found")
            return False

        # Find backups
        backups = list(self.backup_dir.glob("*_*.db"))
        if not backups:
            logger.error("No backups found")
            return False

        # Group by timestamp
        by_timestamp: dict[str, list[Path]] = {}
        for backup in backups:
            ts = backup.stem.split("_")[-2] + "_" + backup.stem.split("_")[-1]
            if ts not in by_timestamp:
                by_timestamp[ts] = []
            by_timestamp[ts].append(backup)

        if timestamp:
            if timestamp not in by_timestamp:
                logger.error(f"No backups found for timestamp {timestamp}")
                return False
            restore_backups = by_timestamp[timestamp]
        else:
            # Most recent
            latest_ts = sorted(by_timestamp.keys())[-1]
            restore_backups = by_timestamp[latest_ts]
            logger.info(f"Restoring from most recent backup: {latest_ts}")

        # Restore each backup
        for backup in restore_backups:
            # Extract original filename
            original_name = "_".join(backup.stem.split("_")[:-2]) + ".db"
            original_path = self.data_dir / original_name

            try:
                shutil.copy2(backup, original_path)
                logger.info(f"Restored: {original_path}")
            except Exception as e:
                logger.error(f"Failed to restore {original_path}: {e}")
                return False

        logger.info("Rollback complete")
        return True


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Migrate unencrypted data to encrypted format")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to most recent backup",
    )
    parser.add_argument(
        "--rollback-timestamp",
        type=str,
        help="Specific backup timestamp to restore",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to data directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    migration = EncryptionMigration(
        data_dir=args.data_dir,
        dry_run=args.dry_run,
    )

    if args.rollback:
        success = migration.rollback(args.rollback_timestamp)
        sys.exit(0 if success else 1)

    result = migration.run()
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
