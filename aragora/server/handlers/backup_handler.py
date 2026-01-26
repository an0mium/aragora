"""
Backup HTTP Handlers for Aragora.

Provides REST API endpoints for backup and disaster recovery:
- List and manage backups
- Trigger manual backups
- Verify backup integrity
- Test restore (dry-run)
- Cleanup expired backups

Endpoints:
    GET  /api/v2/backups                            - List backups with filters
    POST /api/v2/backups                            - Create new backup
    GET  /api/v2/backups/:backup_id                 - Get specific backup metadata
    POST /api/v2/backups/:backup_id/verify          - Verify backup integrity
    POST /api/v2/backups/:backup_id/verify-comprehensive - Comprehensive verification
    POST /api/v2/backups/:backup_id/restore-test    - Dry-run restore test
    DELETE /api/v2/backups/:backup_id               - Delete a backup
    POST /api/v2/backups/cleanup                    - Run retention policy cleanup
    GET  /api/v2/backups/stats                      - Backup statistics

These endpoints support enterprise disaster recovery requirements.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from aragora.rbac.decorators import require_permission
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    ServerContext,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


class BackupHandler(BaseHandler):
    """
    HTTP handler for backup and disaster recovery operations.

    Provides REST API access to backup management with verification
    and dry-run restore capabilities.
    """

    ROUTES = [
        "/api/v2/backups",
        "/api/v2/backups/*",
    ]

    def __init__(self, server_context: ServerContext):
        """Initialize with server context."""
        super().__init__(server_context)
        self._manager = None  # Lazy initialization

    def _get_manager(self):
        """Get or create backup manager (lazy initialization)."""
        if self._manager is None:
            from aragora.backup.manager import get_backup_manager

            self._manager = get_backup_manager()
        return self._manager

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the request."""
        if path.startswith("/api/v2/backups"):
            return method in ("GET", "POST", "DELETE")
        return False

    @rate_limit(requests_per_minute=30)
    async def handle(  # type: ignore[override]
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> HandlerResult:
        """Route request to appropriate handler method."""
        query_params = query_params or {}
        body = body or {}

        try:
            # Stats endpoint
            if path == "/api/v2/backups/stats" and method == "GET":
                return await self._get_stats()

            # Cleanup endpoint
            if path == "/api/v2/backups/cleanup" and method == "POST":
                return await self._cleanup_expired(body)

            # List backups
            if path == "/api/v2/backups" and method == "GET":
                return await self._list_backups(query_params)

            # Create backup
            if path == "/api/v2/backups" and method == "POST":
                return await self._create_backup(body)

            # Backup-specific routes
            if path.startswith("/api/v2/backups/"):
                parts = path.split("/")
                if len(parts) < 5:
                    return error_response("Invalid backup path", 400)

                backup_id = parts[4]

                # Verify endpoint
                if len(parts) > 5 and parts[5] == "verify" and method == "POST":
                    return await self._verify_backup(backup_id)

                # Comprehensive verify endpoint
                if len(parts) > 5 and parts[5] == "verify-comprehensive" and method == "POST":
                    return await self._verify_comprehensive(backup_id)

                # Restore test endpoint
                if len(parts) > 5 and parts[5] == "restore-test" and method == "POST":
                    return await self._restore_test(backup_id, body)

                # Delete backup
                if method == "DELETE":
                    return await self._delete_backup(backup_id)

                # Get single backup
                if method == "GET":
                    return await self._get_backup(backup_id)

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error handling backup request: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    @require_permission("backups:read")
    async def _list_backups(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        List backups with filtering and pagination.

        Query params:
            limit: Max results (default 20, max 100)
            offset: Pagination offset
            source: Filter by source database path
            status: Filter by status (completed, verified, failed)
            since: ISO date/timestamp for start
            backup_type: Filter by type (full, incremental, differential)
        """
        manager = self._get_manager()

        # Parse filters
        source_path = query_params.get("source")
        status_str = query_params.get("status")
        since_str = query_params.get("since")

        # Parse status enum
        status = None
        if status_str:
            from aragora.backup.manager import BackupStatus

            try:
                status = BackupStatus(status_str.lower())
            except ValueError:
                return error_response(
                    f"Invalid status: {status_str}. Valid: pending, in_progress, "
                    "completed, verified, failed, expired",
                    400,
                )

        # Parse since timestamp
        since = None
        if since_str:
            since = self._parse_timestamp(since_str)

        # Get backups from manager
        backups = manager.list_backups(
            source_path=source_path,
            status=status,
            since=since,
        )

        # Apply pagination
        limit = min(int(query_params.get("limit", "20")), 100)
        offset = int(query_params.get("offset", "0"))
        total = len(backups)
        backups = backups[offset : offset + limit]

        return json_response(
            {
                "backups": [b.to_dict() for b in backups],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total,
                    "has_more": offset + len(backups) < total,
                },
            }
        )

    @require_permission("backups:read")
    async def _get_backup(self, backup_id: str) -> HandlerResult:
        """Get a specific backup by ID."""
        manager = self._get_manager()
        backups = manager.list_backups()

        # Find backup by ID
        backup = next((b for b in backups if b.id == backup_id), None)

        if not backup:
            return error_response("Backup not found", 404)

        return json_response(backup.to_dict())

    @require_permission("backups:create")
    async def _create_backup(self, body: Dict[str, Any]) -> HandlerResult:
        """
        Create a new backup.

        Body:
            source_path: Path to the database to backup (required)
            backup_type: Type of backup (full, incremental, differential)
            metadata: Additional metadata to store
        """
        source_path = body.get("source_path")
        if not source_path:
            return error_response("source_path is required", 400)

        backup_type_str = body.get("backup_type", "full")
        metadata = body.get("metadata", {})

        from aragora.backup.manager import BackupType

        try:
            backup_type = BackupType(backup_type_str.lower())
        except ValueError:
            return error_response(
                f"Invalid backup_type: {backup_type_str}. Valid: full, incremental, differential",
                400,
            )

        manager = self._get_manager()

        try:
            backup = manager.create_backup(
                source_path=source_path,
                backup_type=backup_type,
                metadata=metadata,
            )

            return json_response(
                {
                    "backup": backup.to_dict(),
                    "message": f"Backup created: {backup.id}",
                },
                status=201,
            )

        except FileNotFoundError as e:
            return error_response(str(e), 404)
        except Exception as e:
            logger.exception(f"Backup creation failed: {e}")
            return error_response(f"Backup failed: {str(e)}", 500)

    @require_permission("backups:verify")
    async def _verify_backup(self, backup_id: str) -> HandlerResult:
        """Verify backup integrity with restore test."""
        manager = self._get_manager()

        result = manager.verify_backup(backup_id, test_restore=True)

        return json_response(
            {
                "backup_id": result.backup_id,
                "verified": result.verified,
                "checksum_valid": result.checksum_valid,
                "restore_tested": result.restore_tested,
                "tables_valid": result.tables_valid,
                "row_counts_valid": result.row_counts_valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "verified_at": result.verified_at.isoformat(),
                "duration_seconds": result.duration_seconds,
            }
        )

    @require_permission("backups:verify")
    async def _verify_comprehensive(self, backup_id: str) -> HandlerResult:
        """
        Perform comprehensive verification of a backup.

        Includes:
        - Basic verification (checksum, row counts, tables)
        - Schema validation (columns, types, constraints, indexes)
        - Referential integrity (foreign keys, orphans)
        - Per-table checksums
        """
        manager = self._get_manager()

        result = manager.verify_restore_comprehensive(backup_id)

        return json_response(result.to_dict())

    @require_permission("backups:restore")
    async def _restore_test(self, backup_id: str, body: Dict[str, Any]) -> HandlerResult:
        """
        Test restore a backup (dry-run).

        Body:
            target_path: Optional target path for restore test info
        """
        target_path = body.get("target_path", "/tmp/restore_test.db")

        manager = self._get_manager()

        try:
            # Dry run - doesn't actually restore
            success = manager.restore_backup(
                backup_id=backup_id,
                target_path=target_path,
                dry_run=True,
            )

            return json_response(
                {
                    "backup_id": backup_id,
                    "restore_test_passed": success,
                    "target_path": target_path,
                    "dry_run": True,
                    "message": "Dry-run restore test completed successfully",
                }
            )

        except ValueError as e:
            return error_response(str(e), 400)
        except FileNotFoundError as e:
            return error_response(str(e), 404)

    @require_permission("backups:delete")
    async def _delete_backup(self, backup_id: str) -> HandlerResult:
        """Delete a backup by ID."""
        manager = self._get_manager()
        backups = manager.list_backups()

        # Find backup by ID
        backup = next((b for b in backups if b.id == backup_id), None)

        if not backup:
            return error_response("Backup not found", 404)

        from pathlib import Path

        try:
            backup_path = Path(backup.backup_path)
            if backup_path.exists():
                backup_path.unlink()
                logger.info(f"Deleted backup file: {backup_path}")

            # Remove from manager's tracking
            if backup_id in manager._backups:
                del manager._backups[backup_id]
                manager._save_manifest()

            return json_response(
                {
                    "deleted": True,
                    "backup_id": backup_id,
                    "message": f"Backup {backup_id} deleted",
                }
            )

        except Exception as e:
            logger.exception(f"Failed to delete backup: {e}")
            return error_response(f"Delete failed: {str(e)}", 500)

    @require_permission("backups:delete")
    async def _cleanup_expired(self, body: Dict[str, Any]) -> HandlerResult:
        """
        Run retention policy cleanup.

        Body:
            dry_run: If true, only report what would be deleted (default: true)
        """
        dry_run = body.get("dry_run", True)

        manager = self._get_manager()

        deleted_ids = manager.apply_retention_policy(dry_run=dry_run)

        return json_response(
            {
                "dry_run": dry_run,
                "backup_ids": deleted_ids,
                "count": len(deleted_ids),
                "message": (
                    f"Would delete {len(deleted_ids)} backups"
                    if dry_run
                    else f"Deleted {len(deleted_ids)} expired backups"
                ),
            }
        )

    @require_permission("backups:read")
    async def _get_stats(self) -> HandlerResult:
        """Get backup statistics."""
        manager = self._get_manager()
        backups = manager.list_backups()

        from aragora.backup.manager import BackupStatus

        # Compute statistics
        total_size = sum(b.compressed_size_bytes for b in backups)
        verified_count = sum(1 for b in backups if b.verified)
        failed_count = sum(1 for b in backups if b.status == BackupStatus.FAILED)

        # Get latest backup
        latest = manager.get_latest_backup()

        stats = {
            "total_backups": len(backups),
            "verified_backups": verified_count,
            "failed_backups": failed_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "latest_backup": latest.to_dict() if latest else None,
            "retention_policy": {
                "keep_daily": manager.retention_policy.keep_daily,
                "keep_weekly": manager.retention_policy.keep_weekly,
                "keep_monthly": manager.retention_policy.keep_monthly,
                "min_backups": manager.retention_policy.min_backups,
            },
        }

        return json_response(
            {
                "stats": stats,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _parse_timestamp(self, value: Optional[str]) -> Optional[datetime]:
        """Parse timestamp from string (ISO date or unix timestamp)."""
        if not value:
            return None

        try:
            # Try as unix timestamp
            ts = float(value)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except ValueError:
            pass

        try:
            # Try as ISO date
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt
        except (ValueError, AttributeError):
            pass

        return None


# Handler factory function for registration
def create_backup_handler(server_context: ServerContext) -> BackupHandler:
    """Factory function for handler registration."""
    return BackupHandler(server_context)
