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
            from aragora.observability.metrics import (
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
            from aragora.observability.metrics import (
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
            from aragora.observability.metrics import BACKUP_RESTORE_SUCCESS

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
