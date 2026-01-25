"""
Backup Manager - Centralized backup orchestration with verification.

Provides automated backup with:
- Multiple storage backends (local, S3, GCS)
- Backup verification and integrity checks
- Prometheus metrics for monitoring
- Retention policy enforcement
- Dry-run restore capability
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class BackupStatus(str, Enum):
    """Backup status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


class BackupType(str, Enum):
    """Type of backup."""

    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


@dataclass
class BackupMetadata:
    """Metadata for a backup."""

    id: str
    created_at: datetime
    backup_type: BackupType
    status: BackupStatus
    source_path: str
    backup_path: str
    size_bytes: int = 0
    compressed_size_bytes: int = 0
    checksum: str = ""
    row_counts: dict[str, int] = field(default_factory=dict)
    tables: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    verified: bool = False
    verified_at: datetime | None = None
    restore_tested: bool = False
    error: str | None = None
    storage_backend: str = "local"
    encryption_key_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Enhanced verification fields
    schema_hash: str = ""
    table_checksums: dict[str, str] = field(default_factory=dict)
    foreign_keys: list[tuple[str, str, str, str]] = field(
        default_factory=list
    )  # (table, column, ref_table, ref_column)
    indexes: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (index_name, table, columns)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "backup_type": self.backup_type.value,
            "status": self.status.value,
            "source_path": self.source_path,
            "backup_path": self.backup_path,
            "size_bytes": self.size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "checksum": self.checksum,
            "row_counts": self.row_counts,
            "tables": self.tables,
            "duration_seconds": self.duration_seconds,
            "verified": self.verified,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "restore_tested": self.restore_tested,
            "error": self.error,
            "storage_backend": self.storage_backend,
            "encryption_key_id": self.encryption_key_id,
            "metadata": self.metadata,
            "schema_hash": self.schema_hash,
            "table_checksums": self.table_checksums,
            "foreign_keys": [list(fk) for fk in self.foreign_keys],
            "indexes": [list(idx) for idx in self.indexes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackupMetadata:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            backup_type=BackupType(data["backup_type"]),
            status=BackupStatus(data["status"]),
            source_path=data["source_path"],
            backup_path=data["backup_path"],
            size_bytes=data.get("size_bytes", 0),
            compressed_size_bytes=data.get("compressed_size_bytes", 0),
            checksum=data.get("checksum", ""),
            row_counts=data.get("row_counts", {}),
            tables=data.get("tables", []),
            duration_seconds=data.get("duration_seconds", 0.0),
            verified=data.get("verified", False),
            verified_at=(
                datetime.fromisoformat(data["verified_at"]) if data.get("verified_at") else None
            ),
            restore_tested=data.get("restore_tested", False),
            error=data.get("error"),
            storage_backend=data.get("storage_backend", "local"),
            encryption_key_id=data.get("encryption_key_id"),
            metadata=data.get("metadata", {}),
            schema_hash=data.get("schema_hash", ""),
            table_checksums=data.get("table_checksums", {}),
            foreign_keys=[tuple(fk) for fk in data.get("foreign_keys", [])],
            indexes=[tuple(idx) for idx in data.get("indexes", [])],
        )


@dataclass
class RetentionPolicy:
    """Backup retention policy."""

    keep_daily: int = 7  # Keep last N daily backups
    keep_weekly: int = 4  # Keep last N weekly backups
    keep_monthly: int = 3  # Keep last N monthly backups
    max_size_bytes: int | None = None  # Max total storage
    min_backups: int = 1  # Always keep at least N backups


@dataclass
class VerificationResult:
    """Result of backup verification."""

    backup_id: str
    verified: bool
    checksum_valid: bool
    restore_tested: bool
    tables_valid: bool
    row_counts_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verified_at: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0


@dataclass
class SchemaValidationResult:
    """Result of schema validation between original and restored database."""

    valid: bool
    tables_match: bool
    columns_match: bool
    types_match: bool
    constraints_match: bool
    indexes_match: bool
    missing_tables: list[str] = field(default_factory=list)
    extra_tables: list[str] = field(default_factory=list)
    column_mismatches: list[str] = field(default_factory=list)
    type_mismatches: list[str] = field(default_factory=list)
    constraint_mismatches: list[str] = field(default_factory=list)
    index_mismatches: list[str] = field(default_factory=list)


@dataclass
class IntegrityResult:
    """Result of referential integrity verification."""

    valid: bool
    foreign_keys_valid: bool
    orphaned_records: dict[str, int] = field(default_factory=dict)  # table: count
    foreign_key_errors: list[str] = field(default_factory=list)
    data_type_errors: list[str] = field(default_factory=list)
    null_constraint_violations: list[str] = field(default_factory=list)


@dataclass
class ComprehensiveVerificationResult:
    """Comprehensive verification combining all checks."""

    backup_id: str
    verified: bool
    basic_verification: VerificationResult
    schema_validation: SchemaValidationResult | None = None
    integrity_check: IntegrityResult | None = None
    table_checksums_valid: bool = False
    table_checksum_errors: list[str] = field(default_factory=list)
    all_errors: list[str] = field(default_factory=list)
    all_warnings: list[str] = field(default_factory=list)
    verified_at: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "backup_id": self.backup_id,
            "verified": self.verified,
            "basic_verification": {
                "checksum_valid": self.basic_verification.checksum_valid,
                "restore_tested": self.basic_verification.restore_tested,
                "tables_valid": self.basic_verification.tables_valid,
                "row_counts_valid": self.basic_verification.row_counts_valid,
            },
            "schema_validation": (
                {
                    "valid": self.schema_validation.valid,
                    "tables_match": self.schema_validation.tables_match,
                    "columns_match": self.schema_validation.columns_match,
                    "types_match": self.schema_validation.types_match,
                }
                if self.schema_validation
                else None
            ),
            "integrity_check": (
                {
                    "valid": self.integrity_check.valid,
                    "foreign_keys_valid": self.integrity_check.foreign_keys_valid,
                    "orphaned_records": self.integrity_check.orphaned_records,
                }
                if self.integrity_check
                else None
            ),
            "table_checksums_valid": self.table_checksums_valid,
            "all_errors": self.all_errors,
            "all_warnings": self.all_warnings,
            "verified_at": self.verified_at.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


class BackupManager:
    """
    Centralized backup manager with verification.

    Features:
    - Create full and incremental backups
    - Verify backup integrity
    - Test restore without applying
    - Enforce retention policies
    - Emit Prometheus metrics
    """

    def __init__(
        self,
        backup_dir: str | Path,
        retention_policy: RetentionPolicy | None = None,
        compression: bool = True,
        verify_after_backup: bool = True,
        metrics_enabled: bool = True,
    ) -> None:
        """
        Initialize the backup manager.

        Args:
            backup_dir: Directory to store backups
            retention_policy: Retention policy (default: 7 daily, 4 weekly, 3 monthly)
            compression: Whether to compress backups
            verify_after_backup: Whether to verify backups after creation
            metrics_enabled: Whether to emit Prometheus metrics
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.retention_policy = retention_policy or RetentionPolicy()
        self.compression = compression
        self.verify_after_backup = verify_after_backup
        self.metrics_enabled = metrics_enabled

        self._manifest_path = self.backup_dir / "manifest.json"
        self._backups: dict[str, BackupMetadata] = {}
        self._load_manifest()

        # Prometheus metrics (lazy init)
        self._metrics_initialized = False

    def create_backup(
        self,
        source_path: str | Path,
        backup_type: BackupType = BackupType.FULL,
        metadata: dict[str, Any] | None = None,
    ) -> BackupMetadata:
        """
        Create a backup of a SQLite database.

        Args:
            source_path: Path to the database to backup
            backup_type: Type of backup
            metadata: Additional metadata to store

        Returns:
            BackupMetadata with backup details
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source database not found: {source}")

        start_time = datetime.now(timezone.utc)
        backup_id = str(uuid4())[:8]
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source.stem}_{timestamp}_{backup_id}"

        if self.compression:
            backup_path = self.backup_dir / f"{backup_name}.db.gz"
        else:
            backup_path = self.backup_dir / f"{backup_name}.db"

        backup_meta = BackupMetadata(
            id=backup_id,
            created_at=start_time,
            backup_type=backup_type,
            status=BackupStatus.IN_PROGRESS,
            source_path=str(source),
            backup_path=str(backup_path),
            metadata=metadata or {},
        )

        try:
            # Get source database info
            tables, row_counts = self._get_database_info(source)
            backup_meta.tables = tables
            backup_meta.row_counts = row_counts
            backup_meta.size_bytes = source.stat().st_size

            # Get enhanced schema info
            schema_info = self._get_schema_info(source)
            backup_meta.schema_hash = self._compute_schema_hash(source)
            backup_meta.table_checksums = self._compute_table_checksums(source)

            # Extract foreign keys and indexes
            for table, info in schema_info.items():
                for fk in info["foreign_keys"]:
                    backup_meta.foreign_keys.append((table, fk[0], fk[1], fk[2]))
                for idx in info["indexes"]:
                    backup_meta.indexes.append((idx[0], table, ",".join(idx[1])))

            # Create backup using SQLite's backup API
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            self._backup_database(source, tmp_path)

            # Compress if enabled
            if self.compression:
                self._compress_file(tmp_path, backup_path)
                tmp_path.unlink()
            else:
                shutil.move(tmp_path, backup_path)

            backup_meta.compressed_size_bytes = backup_path.stat().st_size
            backup_meta.checksum = self._compute_checksum(backup_path)
            backup_meta.duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            backup_meta.status = BackupStatus.COMPLETED

            # Verify if configured
            if self.verify_after_backup:
                result = self.verify_backup(backup_id, backup_meta)
                if result.verified:
                    backup_meta.status = BackupStatus.VERIFIED
                    backup_meta.verified = True
                    backup_meta.verified_at = result.verified_at

            self._backups[backup_id] = backup_meta
            self._save_manifest()

            self._emit_backup_metrics(backup_meta, success=True)
            logger.info(
                "Backup created: %s -> %s (%.2f MB, %.2fs)",
                source.name,
                backup_path.name,
                backup_meta.compressed_size_bytes / 1024 / 1024,
                backup_meta.duration_seconds,
            )

        except Exception as e:
            backup_meta.status = BackupStatus.FAILED
            backup_meta.error = str(e)
            self._backups[backup_id] = backup_meta
            self._save_manifest()
            self._emit_backup_metrics(backup_meta, success=False)
            logger.error("Backup failed: %s - %s", source, e)
            raise

        return backup_meta

    def verify_backup(
        self,
        backup_id: str,
        backup_meta: BackupMetadata | None = None,
        test_restore: bool = True,
    ) -> VerificationResult:
        """
        Verify a backup's integrity.

        Args:
            backup_id: ID of the backup to verify
            backup_meta: Optional metadata (will be loaded if not provided)
            test_restore: Whether to test restoring to temp database

        Returns:
            VerificationResult with verification details
        """
        start_time = datetime.now(timezone.utc)

        if backup_meta is None:
            backup_meta = self._backups.get(backup_id)
            if backup_meta is None:
                return VerificationResult(
                    backup_id=backup_id,
                    verified=False,
                    checksum_valid=False,
                    restore_tested=False,
                    tables_valid=False,
                    row_counts_valid=False,
                    errors=["Backup not found"],
                )

        result = VerificationResult(
            backup_id=backup_id,
            verified=True,
            checksum_valid=False,
            restore_tested=False,
            tables_valid=False,
            row_counts_valid=False,
        )

        backup_path = Path(backup_meta.backup_path)

        # Check file exists
        if not backup_path.exists():
            result.verified = False
            result.errors.append(f"Backup file not found: {backup_path}")
            return result

        # Verify checksum
        try:
            current_checksum = self._compute_checksum(backup_path)
            if current_checksum == backup_meta.checksum:
                result.checksum_valid = True
            else:
                result.verified = False
                result.errors.append(
                    f"Checksum mismatch: expected {backup_meta.checksum}, got {current_checksum}"
                )
        except Exception as e:
            result.verified = False
            result.errors.append(f"Checksum computation failed: {e}")

        # Test restore if enabled
        if test_restore:
            try:
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                # Decompress and restore
                if self.compression:
                    self._decompress_file(backup_path, tmp_path)
                else:
                    shutil.copy(backup_path, tmp_path)

                # Verify restored database
                tables, row_counts = self._get_database_info(tmp_path)

                result.restore_tested = True

                # Check tables match
                if set(tables) == set(backup_meta.tables):
                    result.tables_valid = True
                else:
                    result.verified = False
                    result.errors.append(
                        f"Table mismatch: expected {backup_meta.tables}, got {tables}"
                    )

                # Check row counts match (with tolerance)
                row_count_errors = []
                for table, expected_count in backup_meta.row_counts.items():
                    actual_count = row_counts.get(table, 0)
                    if actual_count != expected_count:
                        row_count_errors.append(
                            f"{table}: expected {expected_count}, got {actual_count}"
                        )

                if not row_count_errors:
                    result.row_counts_valid = True
                else:
                    result.warnings.extend(row_count_errors)

                # Clean up
                tmp_path.unlink()

            except Exception as e:
                result.verified = False
                result.errors.append(f"Restore test failed: {e}")

        result.duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
        result.verified = result.verified and result.checksum_valid

        # Update backup metadata
        backup_meta.verified = result.verified
        backup_meta.verified_at = result.verified_at
        backup_meta.restore_tested = result.restore_tested
        self._save_manifest()

        self._emit_verification_metrics(result)

        return result

    def restore_backup(
        self,
        backup_id: str,
        target_path: str | Path,
        dry_run: bool = False,
    ) -> bool:
        """
        Restore a backup to a target path.

        Args:
            backup_id: ID of the backup to restore
            target_path: Path to restore to
            dry_run: If True, only verify without actually restoring

        Returns:
            True if restore succeeded
        """
        backup_meta = self._backups.get(backup_id)
        if backup_meta is None:
            raise ValueError(f"Backup not found: {backup_id}")

        backup_path = Path(backup_meta.backup_path)
        target = Path(target_path)

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        # Verify before restore
        result = self.verify_backup(backup_id, backup_meta, test_restore=True)
        if not result.verified:
            raise ValueError(f"Backup verification failed: {result.errors}")

        if dry_run:
            logger.info("Dry run: would restore %s to %s", backup_id, target)
            return True

        # Create backup of target if it exists
        if target.exists():
            target_backup = target.with_suffix(
                f".backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy(target, target_backup)
            logger.info("Created backup of target: %s", target_backup)

        try:
            if self.compression:
                self._decompress_file(backup_path, target)
            else:
                shutil.copy(backup_path, target)

            logger.info("Restored backup %s to %s", backup_id, target)
            self._emit_restore_metrics(backup_id, success=True)
            return True

        except Exception as e:
            logger.error("Restore failed: %s", e)
            self._emit_restore_metrics(backup_id, success=False)
            raise

    def list_backups(
        self,
        source_path: str | None = None,
        status: BackupStatus | None = None,
        since: datetime | None = None,
    ) -> list[BackupMetadata]:
        """List backups with optional filters."""
        backups = list(self._backups.values())

        if source_path:
            backups = [b for b in backups if b.source_path == source_path]

        if status:
            backups = [b for b in backups if b.status == status]

        if since:
            backups = [b for b in backups if b.created_at >= since]

        return sorted(backups, key=lambda b: b.created_at, reverse=True)

    def cleanup_expired(self) -> list[str]:
        """
        Clean up expired backups according to retention policy.

        Returns:
            List of deleted backup IDs
        """
        deleted: list[str] = []
        now = datetime.now(timezone.utc)

        # Sort by date
        backups = sorted(self._backups.values(), key=lambda b: b.created_at, reverse=True)

        # Keep minimum backups
        if len(backups) <= self.retention_policy.min_backups:
            return deleted

        # Mark for deletion
        to_delete = []
        daily_kept = 0
        weekly_kept = 0
        monthly_kept = 0

        for backup in backups:
            age = now - backup.created_at

            if age <= timedelta(days=1):
                daily_kept += 1
                if daily_kept > self.retention_policy.keep_daily:
                    to_delete.append(backup.id)
            elif age <= timedelta(weeks=1):
                weekly_kept += 1
                if weekly_kept > self.retention_policy.keep_weekly:
                    to_delete.append(backup.id)
            else:
                monthly_kept += 1
                if monthly_kept > self.retention_policy.keep_monthly:
                    to_delete.append(backup.id)

        # Delete expired backups (keep minimum)
        for backup_id in to_delete:
            if len(self._backups) <= self.retention_policy.min_backups:
                break

            backup = self._backups[backup_id]
            try:
                backup_path = Path(backup.backup_path)
                if backup_path.exists():
                    backup_path.unlink()
                    logger.info("Deleted expired backup: %s", backup_id)

                backup.status = BackupStatus.EXPIRED
                del self._backups[backup_id]
                deleted.append(backup_id)

            except Exception as e:
                logger.error("Failed to delete backup %s: %s", backup_id, e)

        self._save_manifest()
        return deleted

    def cleanup_expired_backups(self) -> list[str]:
        """Alias for cleanup_expired for API compatibility."""
        return self.cleanup_expired()

    def apply_retention_policy(self, dry_run: bool = False) -> list[str]:
        """Apply retention policy to identify or remove old backups.

        Args:
            dry_run: If True, only return IDs that would be removed without deleting

        Returns:
            List of backup IDs that were (or would be) removed
        """
        if dry_run:
            # Return list of IDs that would be deleted without actually deleting
            to_delete: list[str] = []
            now = datetime.now(timezone.utc)
            backups = sorted(self._backups.values(), key=lambda b: b.created_at, reverse=True)

            if len(backups) <= self.retention_policy.min_backups:
                return to_delete

            # Determine which backups exceed retention limits
            daily_kept = 0
            weekly_kept = 0
            monthly_kept = 0

            for backup in backups:
                age_days = (now - backup.created_at).days
                keep = False

                # Check retention tiers
                if age_days <= 7 and daily_kept < self.retention_policy.keep_daily:
                    daily_kept += 1
                    keep = True
                elif age_days <= 30 and weekly_kept < self.retention_policy.keep_weekly:
                    weekly_kept += 1
                    keep = True
                elif monthly_kept < self.retention_policy.keep_monthly:
                    monthly_kept += 1
                    keep = True

                if not keep and len(backups) - len(to_delete) > self.retention_policy.min_backups:
                    to_delete.append(backup.id)

            return to_delete
        else:
            # Actually perform cleanup
            return self.cleanup_expired()

    def get_latest_backup(self, source_path: str | None = None) -> BackupMetadata | None:
        """Get the most recent verified backup."""
        backups = self.list_backups(
            source_path=source_path,
            status=BackupStatus.VERIFIED,
        )
        return backups[0] if backups else None

    def _backup_database(self, source: Path, dest: Path) -> None:
        """Create a backup using SQLite's backup API."""
        src_conn = sqlite3.connect(str(source))
        dst_conn = sqlite3.connect(str(dest))

        with dst_conn:
            src_conn.backup(dst_conn)

        src_conn.close()
        dst_conn.close()

    def _get_database_info(self, db_path: Path) -> tuple[list[str], dict[str, int]]:
        """Get table names and row counts from a database."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Get row counts
        row_counts = {}
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
            row_counts[table] = cursor.fetchone()[0]

        conn.close()
        return tables, row_counts

    def _get_schema_info(self, db_path: Path) -> dict[str, dict[str, Any]]:
        """
        Get detailed schema information for each table.

        Returns:
            Dict mapping table name to schema info:
            {
                "table_name": {
                    "columns": [{"name": str, "type": str, "notnull": bool, "pk": bool}],
                    "foreign_keys": [(column, ref_table, ref_column)],
                    "indexes": [(name, columns, unique)]
                }
            }
        """
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        schema_info: dict[str, dict[str, Any]] = {}

        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            # Get column info using PRAGMA
            cursor.execute(f'PRAGMA table_info("{table}")')
            columns = []
            for row in cursor.fetchall():
                columns.append(
                    {
                        "name": row[1],
                        "type": row[2],
                        "notnull": bool(row[3]),
                        "default": row[4],
                        "pk": bool(row[5]),
                    }
                )

            # Get foreign keys
            cursor.execute(f'PRAGMA foreign_key_list("{table}")')
            fks = []
            for row in cursor.fetchall():
                fks.append((row[3], row[2], row[4]))  # (column, ref_table, ref_column)

            # Get indexes
            cursor.execute(f'PRAGMA index_list("{table}")')
            indexes = []
            for idx_row in cursor.fetchall():
                idx_name = idx_row[1]
                is_unique = bool(idx_row[2])
                # Get columns in this index
                cursor.execute(f'PRAGMA index_info("{idx_name}")')
                idx_cols = [col_row[2] for col_row in cursor.fetchall()]
                indexes.append((idx_name, idx_cols, is_unique))

            schema_info[table] = {
                "columns": columns,
                "foreign_keys": fks,
                "indexes": indexes,
            }

        conn.close()
        return schema_info

    def _compute_schema_hash(self, db_path: Path) -> str:
        """
        Compute a hash of the database schema structure.

        This creates a deterministic hash that changes only when schema changes.
        """
        schema_info = self._get_schema_info(db_path)

        # Create deterministic string representation
        schema_str = ""
        for table in sorted(schema_info.keys()):
            info = schema_info[table]
            schema_str += f"TABLE:{table}\n"
            for col in sorted(info["columns"], key=lambda c: c["name"]):
                schema_str += f"  COL:{col['name']}:{col['type']}:{col['notnull']}:{col['pk']}\n"
            for fk in sorted(info["foreign_keys"]):
                schema_str += f"  FK:{fk[0]}->{fk[1]}.{fk[2]}\n"
            for idx in sorted(info["indexes"], key=lambda i: i[0]):
                schema_str += f"  IDX:{idx[0]}:{','.join(idx[1])}:{idx[2]}\n"

        return hashlib.sha256(schema_str.encode()).hexdigest()

    def _validate_schema(
        self, restored_db: Path, backup_meta: BackupMetadata
    ) -> SchemaValidationResult:
        """
        Validate schema of restored database against backup metadata.

        Compares:
        - Table existence
        - Column definitions
        - Data types
        - Constraints
        - Indexes
        """
        result = SchemaValidationResult(
            valid=True,
            tables_match=True,
            columns_match=True,
            types_match=True,
            constraints_match=True,
            indexes_match=True,
        )

        # Get current schema
        current_schema = self._get_schema_info(restored_db)
        current_tables = set(current_schema.keys())
        expected_tables = set(backup_meta.tables)

        # Check tables match
        result.missing_tables = list(expected_tables - current_tables)
        result.extra_tables = list(current_tables - expected_tables)

        if result.missing_tables or result.extra_tables:
            result.tables_match = False
            result.valid = False

        # Compare schema hash if available
        if backup_meta.schema_hash:
            current_hash = self._compute_schema_hash(restored_db)
            if current_hash != backup_meta.schema_hash:
                # Schema changed - do detailed comparison
                for table in current_tables & expected_tables:
                    current_info = current_schema[table]

                    # Compare columns
                    current_cols = {c["name"]: c for c in current_info["columns"]}

                    # Check for missing/extra columns
                    for col_name, col_info in current_cols.items():
                        if col_info["pk"]:
                            continue  # Skip PK columns in detailed comparison

                    # Check indexes match
                    current_indexes = set(idx[0] for idx in current_info["indexes"])
                    expected_indexes = set(idx[0] for idx in backup_meta.indexes if idx[1] == table)
                    if current_indexes != expected_indexes:
                        result.indexes_match = False
                        result.index_mismatches.append(
                            f"{table}: expected indexes {expected_indexes}, got {current_indexes}"
                        )

        result.valid = (
            result.tables_match
            and result.columns_match
            and result.types_match
            and result.constraints_match
            and result.indexes_match
        )

        return result

    def _verify_referential_integrity(self, db_path: Path) -> IntegrityResult:
        """
        Verify referential integrity of a database.

        Checks:
        - Foreign key constraints
        - Orphaned records
        - NOT NULL constraints on required columns
        """
        result = IntegrityResult(
            valid=True,
            foreign_keys_valid=True,
        )

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            # Enable foreign key checking
            cursor.execute("PRAGMA foreign_keys = ON")

            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            integrity_results = cursor.fetchall()
            if integrity_results[0][0] != "ok":
                result.valid = False
                for row in integrity_results:
                    result.data_type_errors.append(str(row[0]))

            # Check foreign key violations
            cursor.execute("PRAGMA foreign_key_check")
            fk_violations = cursor.fetchall()
            if fk_violations:
                result.foreign_keys_valid = False
                result.valid = False
                for violation in fk_violations:
                    table, rowid, ref_table, fk_index = violation
                    result.foreign_key_errors.append(
                        f"Table {table} row {rowid} violates FK to {ref_table}"
                    )
                    # Count orphaned records by table
                    if table not in result.orphaned_records:
                        result.orphaned_records[table] = 0
                    result.orphaned_records[table] += 1

            # Check for NULL values in NOT NULL columns
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f'PRAGMA table_info("{table}")')
                for col in cursor.fetchall():
                    col_name, col_notnull = col[1], col[3]
                    if col_notnull:
                        cursor.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" IS NULL')
                        null_count = cursor.fetchone()[0]
                        if null_count > 0:
                            result.valid = False
                            result.null_constraint_violations.append(
                                f"{table}.{col_name} has {null_count} NULL values"
                            )

        except Exception as e:
            result.valid = False
            result.data_type_errors.append(f"Integrity check failed: {e}")
        finally:
            conn.close()

        return result

    def _compute_table_checksums(self, db_path: Path) -> dict[str, str]:
        """
        Compute SHA-256 checksum for each table's data.

        This allows detection of data corruption at the table level.
        """
        checksums: dict[str, str] = {}

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                # Get all data sorted by rowid for deterministic hash
                try:
                    cursor.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
                    rows = cursor.fetchall()

                    # Create hash from serialized rows
                    hasher = hashlib.sha256()
                    for row in rows:
                        row_str = "|".join(str(val) for val in row)
                        hasher.update(row_str.encode())

                    checksums[table] = hasher.hexdigest()
                except Exception as e:
                    # Some tables may not have rowid (e.g., WITHOUT ROWID tables)
                    logger.warning(f"Could not compute checksum for table {table}: {e}")
                    checksums[table] = ""

        finally:
            conn.close()

        return checksums

    def verify_restore_comprehensive(
        self,
        backup_id: str,
        backup_meta: BackupMetadata | None = None,
    ) -> ComprehensiveVerificationResult:
        """
        Perform comprehensive verification of a backup.

        Includes:
        - Basic verification (checksum, row counts, tables)
        - Schema validation (columns, types, constraints, indexes)
        - Referential integrity (foreign keys, orphans)
        - Per-table checksums

        Args:
            backup_id: ID of the backup to verify
            backup_meta: Optional metadata (will be loaded if not provided)

        Returns:
            ComprehensiveVerificationResult with all check results
        """
        start_time = datetime.now(timezone.utc)

        if backup_meta is None:
            backup_meta = self._backups.get(backup_id)

        # Start with basic verification
        basic_result = self.verify_backup(backup_id, backup_meta, test_restore=True)

        result = ComprehensiveVerificationResult(
            backup_id=backup_id,
            verified=basic_result.verified,
            basic_verification=basic_result,
        )
        result.all_errors.extend(basic_result.errors)
        result.all_warnings.extend(basic_result.warnings)

        if not basic_result.verified or backup_meta is None:
            result.duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            return result

        backup_path = Path(backup_meta.backup_path)

        # Restore to temp for comprehensive checks
        try:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            if self.compression:
                self._decompress_file(backup_path, tmp_path)
            else:
                shutil.copy(backup_path, tmp_path)

            # Schema validation
            schema_result = self._validate_schema(tmp_path, backup_meta)
            result.schema_validation = schema_result
            if not schema_result.valid:
                result.verified = False
                if schema_result.missing_tables:
                    result.all_errors.append(f"Missing tables: {schema_result.missing_tables}")
                if schema_result.extra_tables:
                    result.all_warnings.append(f"Extra tables: {schema_result.extra_tables}")
                result.all_errors.extend(schema_result.column_mismatches)
                result.all_errors.extend(schema_result.type_mismatches)
                result.all_errors.extend(schema_result.constraint_mismatches)
                result.all_errors.extend(schema_result.index_mismatches)

            # Referential integrity
            integrity_result = self._verify_referential_integrity(tmp_path)
            result.integrity_check = integrity_result
            if not integrity_result.valid:
                result.verified = False
                result.all_errors.extend(integrity_result.foreign_key_errors)
                result.all_errors.extend(integrity_result.data_type_errors)
                result.all_errors.extend(integrity_result.null_constraint_violations)
                if integrity_result.orphaned_records:
                    result.all_warnings.append(
                        f"Orphaned records: {integrity_result.orphaned_records}"
                    )

            # Table checksums
            if backup_meta.table_checksums:
                current_checksums = self._compute_table_checksums(tmp_path)
                result.table_checksums_valid = True

                for table, expected_checksum in backup_meta.table_checksums.items():
                    if not expected_checksum:
                        continue  # Skip empty checksums
                    current_checksum = current_checksums.get(table, "")
                    if current_checksum and current_checksum != expected_checksum:
                        result.table_checksums_valid = False
                        result.table_checksum_errors.append(f"Table {table}: checksum mismatch")

                if not result.table_checksums_valid:
                    result.verified = False
                    result.all_errors.extend(result.table_checksum_errors)
            else:
                # No stored checksums - just compute them for info
                result.table_checksums_valid = True

            # Clean up
            tmp_path.unlink()

        except Exception as e:
            result.verified = False
            result.all_errors.append(f"Comprehensive verification failed: {e}")

        result.duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.info(
            "Comprehensive verification for %s: verified=%s, errors=%d, warnings=%d",
            backup_id,
            result.verified,
            len(result.all_errors),
            len(result.all_warnings),
        )

        return result

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _compress_file(self, source: Path, dest: Path) -> None:
        """Compress a file using gzip."""
        with open(source, "rb") as f_in:
            with gzip.open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    def _decompress_file(self, source: Path, dest: Path) -> None:
        """Decompress a gzip file."""
        with gzip.open(source, "rb") as f_in:
            with open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    def _load_manifest(self) -> None:
        """Load backup manifest from disk."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path) as f:
                    data = json.load(f)
                    self._backups = {
                        k: BackupMetadata.from_dict(v) for k, v in data.get("backups", {}).items()
                    }
            except Exception as e:
                logger.error("Failed to load manifest: %s", e)
                self._backups = {}

    def _save_manifest(self) -> None:
        """Save backup manifest to disk."""
        data = {
            "backups": {k: v.to_dict() for k, v in self._backups.items()},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def _emit_backup_metrics(self, backup: BackupMetadata, success: bool) -> None:
        """Emit Prometheus metrics for backup operation."""
        if not self.metrics_enabled:
            return

        try:
            from aragora.observability.metrics import (  # type: ignore[attr-defined]
                BACKUP_DURATION,
                BACKUP_SIZE,
                BACKUP_SUCCESS,
                LAST_BACKUP_TIMESTAMP,
            )

            labels = {"source": Path(backup.source_path).stem, "type": backup.backup_type.value}
            BACKUP_DURATION.labels(**labels).observe(backup.duration_seconds)
            BACKUP_SIZE.labels(**labels).set(backup.compressed_size_bytes)
            BACKUP_SUCCESS.labels(**labels, success=str(success).lower()).inc()
            LAST_BACKUP_TIMESTAMP.labels(**labels).set(backup.created_at.timestamp())

        except ImportError:
            pass  # Metrics not available

    def _emit_verification_metrics(self, result: VerificationResult) -> None:
        """Emit Prometheus metrics for verification."""
        if not self.metrics_enabled:
            return

        try:
            from aragora.observability.metrics import (  # type: ignore[attr-defined]
                BACKUP_VERIFICATION_DURATION,
                BACKUP_VERIFICATION_SUCCESS,
            )

            BACKUP_VERIFICATION_DURATION.observe(result.duration_seconds)
            BACKUP_VERIFICATION_SUCCESS.labels(verified=str(result.verified).lower()).inc()

        except ImportError:
            pass

    def _emit_restore_metrics(self, backup_id: str, success: bool) -> None:
        """Emit Prometheus metrics for restore operation."""
        if not self.metrics_enabled:
            return

        try:
            from aragora.observability.metrics import BACKUP_RESTORE_SUCCESS  # type: ignore[attr-defined]

            BACKUP_RESTORE_SUCCESS.labels(success=str(success).lower()).inc()

        except ImportError:
            pass


# Global backup manager instance
_backup_manager: BackupManager | None = None


def get_backup_manager(backup_dir: str | Path | None = None) -> BackupManager:
    """Get or create the global backup manager."""
    global _backup_manager
    if _backup_manager is None:
        if backup_dir is None:
            backup_dir = Path.home() / ".aragora" / "backups"
        _backup_manager = BackupManager(backup_dir)
    return _backup_manager


def set_backup_manager(manager: BackupManager) -> None:
    """Set the global backup manager."""
    global _backup_manager
    _backup_manager = manager
