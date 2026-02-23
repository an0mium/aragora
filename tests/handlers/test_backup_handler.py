"""
Tests for BackupHandler.

Tests cover:
- Handler routing (can_handle for all endpoints)
- List backups with filters and pagination
- Get single backup
- Create backup with path validation
- Verify backup (basic + comprehensive)
- Restore test (dry-run) with path validation
- Delete backup
- Cleanup expired backups
- Stats endpoint
- Error handling and edge cases
- Input validation
- Timestamp parsing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Lightweight stubs so we never touch the real BackupManager / disk
# ---------------------------------------------------------------------------


class _BackupStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


class _BackupType(str, Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


@dataclass
class _RetentionPolicy:
    keep_daily: int = 7
    keep_weekly: int = 4
    keep_monthly: int = 3
    max_size_bytes: int | None = None
    min_backups: int = 1


@dataclass
class _BackupMetadata:
    id: str
    created_at: datetime
    backup_type: _BackupType
    status: _BackupStatus
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
    schema_hash: str = ""
    table_checksums: dict[str, str] = field(default_factory=dict)
    foreign_keys: list[tuple[str, str, str, str]] = field(default_factory=list)
    indexes: list[tuple[str, str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
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


@dataclass
class _VerificationResult:
    backup_id: str
    verified: bool
    checksum_valid: bool
    restore_tested: bool
    tables_valid: bool
    row_counts_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 0.0


@dataclass
class _ComprehensiveVerificationResult:
    backup_id: str
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {"backup_id": self.backup_id, "verified": self.verified}


def _make_backup(
    backup_id: str = "bk-001",
    status: _BackupStatus = _BackupStatus.COMPLETED,
    verified: bool = False,
    compressed_size: int = 1024,
    backup_type: _BackupType = _BackupType.FULL,
) -> _BackupMetadata:
    return _BackupMetadata(
        id=backup_id,
        created_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        backup_type=backup_type,
        status=status,
        source_path="/var/aragora/data/main.db",
        backup_path=f"/tmp/backups/{backup_id}.gz",
        size_bytes=2048,
        compressed_size_bytes=compressed_size,
        verified=verified,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_manager():
    """Create a mock BackupManager with sane defaults."""
    mgr = MagicMock()
    mgr.list_backups.return_value = []
    mgr.get_latest_backup.return_value = None
    mgr.retention_policy = _RetentionPolicy()
    mgr._backups = {}
    mgr.verify_backup.return_value = _VerificationResult(
        backup_id="bk-001",
        verified=True,
        checksum_valid=True,
        restore_tested=True,
        tables_valid=True,
        row_counts_valid=True,
    )
    mgr.verify_restore_comprehensive.return_value = _ComprehensiveVerificationResult(
        backup_id="bk-001", verified=True
    )
    mgr.restore_backup.return_value = True
    mgr.apply_retention_policy.return_value = []
    mgr.create_backup.return_value = _make_backup()
    return mgr


@pytest.fixture
def handler(mock_manager):
    """Create a BackupHandler with an injected mock manager."""
    from aragora.server.handlers.backup_handler import BackupHandler

    h = BackupHandler({"nomic_dir": "/tmp/test"})
    h._manager = mock_manager
    return h


def _http(command: str = "GET", body: dict | None = None) -> MagicMock:
    """Return a mock HTTP handler object."""
    mock = MagicMock()
    mock.command = command
    mock.client_address = ("127.0.0.1", 54321)
    mock.headers = {}
    if body is not None:
        raw = json.dumps(body).encode()
        mock.rfile.read.return_value = raw
        mock.headers["Content-Length"] = str(len(raw))
    else:
        mock.rfile.read.return_value = b"{}"
        mock.headers["Content-Length"] = "2"
    return mock


def _body(result) -> dict[str, Any]:
    """Parse the JSON body out of a HandlerResult."""
    return json.loads(result.body)


# ============================================================================
# Routing Tests (can_handle)
# ============================================================================


class TestCanHandle:
    """Tests for BackupHandler.can_handle routing."""

    def test_list_backups_get(self, handler):
        assert handler.can_handle("/api/v2/backups", method="GET")

    def test_create_backup_post(self, handler):
        assert handler.can_handle("/api/v2/backups", method="POST")

    def test_get_single_backup(self, handler):
        assert handler.can_handle("/api/v2/backups/bk-001", method="GET")

    def test_delete_backup(self, handler):
        assert handler.can_handle("/api/v2/backups/bk-001", method="DELETE")

    def test_verify_backup(self, handler):
        assert handler.can_handle("/api/v2/backups/bk-001/verify", method="POST")

    def test_verify_comprehensive(self, handler):
        assert handler.can_handle("/api/v2/backups/bk-001/verify-comprehensive", method="POST")

    def test_restore_test(self, handler):
        assert handler.can_handle("/api/v2/backups/bk-001/restore-test", method="POST")

    def test_cleanup(self, handler):
        assert handler.can_handle("/api/v2/backups/cleanup", method="POST")

    def test_stats(self, handler):
        assert handler.can_handle("/api/v2/backups/stats", method="GET")

    def test_rejects_patch_method(self, handler):
        assert not handler.can_handle("/api/v2/backups", method="PATCH")

    def test_rejects_put_method(self, handler):
        assert not handler.can_handle("/api/v2/backups", method="PUT")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v2/debates", method="GET")

    def test_rejects_v1_prefix(self, handler):
        # can_handle checks only /api/v2/backups
        assert not handler.can_handle("/api/v1/backups", method="GET")


# ============================================================================
# List Backups
# ============================================================================


class TestListBackups:
    """Tests for GET /api/v2/backups."""

    @pytest.mark.asyncio
    async def test_list_empty(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        http = _http("GET")
        result = await handler.handle("/api/v2/backups", {}, http)
        assert result.status_code == 200
        data = _body(result)
        assert data["backups"] == []
        assert data["pagination"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_returns_backups(self, handler, mock_manager):
        mock_manager.list_backups.return_value = [_make_backup("bk-1"), _make_backup("bk-2")]
        result = await handler.handle("/api/v2/backups", {}, _http("GET"))
        data = _body(result)
        assert len(data["backups"]) == 2
        assert data["pagination"]["total"] == 2

    @pytest.mark.asyncio
    async def test_list_pagination_limit(self, handler, mock_manager):
        backups = [_make_backup(f"bk-{i}") for i in range(5)]
        mock_manager.list_backups.return_value = backups
        result = await handler.handle("/api/v2/backups", {"limit": "2"}, _http("GET"))
        data = _body(result)
        assert len(data["backups"]) == 2
        assert data["pagination"]["limit"] == 2
        assert data["pagination"]["total"] == 5
        assert data["pagination"]["has_more"] is True

    @pytest.mark.asyncio
    async def test_list_pagination_offset(self, handler, mock_manager):
        backups = [_make_backup(f"bk-{i}") for i in range(5)]
        mock_manager.list_backups.return_value = backups
        result = await handler.handle("/api/v2/backups", {"limit": "2", "offset": "3"}, _http("GET"))
        data = _body(result)
        assert len(data["backups"]) == 2
        assert data["pagination"]["offset"] == 3

    @pytest.mark.asyncio
    async def test_list_pagination_no_more(self, handler, mock_manager):
        backups = [_make_backup(f"bk-{i}") for i in range(3)]
        mock_manager.list_backups.return_value = backups
        result = await handler.handle("/api/v2/backups", {"limit": "20"}, _http("GET"))
        data = _body(result)
        assert data["pagination"]["has_more"] is False

    @pytest.mark.asyncio
    async def test_list_filter_by_source(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        result = await handler.handle(
            "/api/v2/backups", {"source": "/var/aragora/data/main.db"}, _http("GET")
        )
        assert result.status_code == 200
        mock_manager.list_backups.assert_called_once()
        call_kwargs = mock_manager.list_backups.call_args
        assert call_kwargs.kwargs.get("source_path") == "/var/aragora/data/main.db"

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        result = await handler.handle(
            "/api/v2/backups", {"status": "completed"}, _http("GET")
        )
        assert result.status_code == 200
        call_kwargs = mock_manager.list_backups.call_args
        status_val = call_kwargs.kwargs.get("status")
        assert status_val is not None
        assert status_val.value == "completed"

    @pytest.mark.asyncio
    async def test_list_invalid_status(self, handler, mock_manager):
        result = await handler.handle(
            "/api/v2/backups", {"status": "bogus"}, _http("GET")
        )
        assert result.status_code == 400
        data = _body(result)
        assert "Invalid status" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_list_filter_by_since_unix(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        result = await handler.handle(
            "/api/v2/backups", {"since": "1700000000"}, _http("GET")
        )
        assert result.status_code == 200
        call_kwargs = mock_manager.list_backups.call_args
        since_val = call_kwargs.kwargs.get("since")
        assert since_val is not None

    @pytest.mark.asyncio
    async def test_list_filter_by_since_iso(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        result = await handler.handle(
            "/api/v2/backups", {"since": "2026-01-01T00:00:00Z"}, _http("GET")
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_filter_since_invalid_string_ignored(self, handler, mock_manager):
        """An unparseable since value results in since=None (no filter)."""
        mock_manager.list_backups.return_value = []
        result = await handler.handle(
            "/api/v2/backups", {"since": "not-a-date"}, _http("GET")
        )
        assert result.status_code == 200
        call_kwargs = mock_manager.list_backups.call_args
        assert call_kwargs.kwargs.get("since") is None


# ============================================================================
# Get Single Backup
# ============================================================================


class TestGetBackup:
    """Tests for GET /api/v2/backups/:id."""

    @pytest.mark.asyncio
    async def test_get_existing_backup(self, handler, mock_manager):
        bk = _make_backup("bk-found")
        mock_manager.list_backups.return_value = [bk]
        result = await handler.handle("/api/v2/backups/bk-found", {}, _http("GET"))
        assert result.status_code == 200
        data = _body(result)
        assert data["id"] == "bk-found"

    @pytest.mark.asyncio
    async def test_get_missing_backup(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        result = await handler.handle("/api/v2/backups/nonexistent", {}, _http("GET"))
        assert result.status_code == 404
        data = _body(result)
        assert "not found" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_selects_correct_backup(self, handler, mock_manager):
        bk1 = _make_backup("bk-a")
        bk2 = _make_backup("bk-b")
        mock_manager.list_backups.return_value = [bk1, bk2]
        result = await handler.handle("/api/v2/backups/bk-b", {}, _http("GET"))
        assert result.status_code == 200
        assert _body(result)["id"] == "bk-b"


# ============================================================================
# Create Backup
# ============================================================================


class TestCreateBackup:
    """Tests for POST /api/v2/backups."""

    @pytest.mark.asyncio
    async def test_create_missing_source_path(self, handler):
        result = await handler.handle("/api/v2/backups", {}, _http("POST", body={}))
        assert result.status_code == 400
        assert "source_path" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_invalid_backup_type(self, handler):
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            return_value="/var/aragora/data/main.db",
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_BACKUP_SOURCE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups",
                {},
                _http("POST", body={"source_path": "main.db", "backup_type": "snapshot"}),
            )
        assert result.status_code == 400
        assert "Invalid backup_type" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_create_success(self, handler, mock_manager):
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            return_value="/var/aragora/data/main.db",
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_BACKUP_SOURCE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups",
                {},
                _http("POST", body={"source_path": "main.db"}),
            )
        assert result.status_code == 201
        data = _body(result)
        assert "backup" in data
        assert "message" in data

    @pytest.mark.asyncio
    async def test_create_with_metadata(self, handler, mock_manager):
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            return_value="/var/aragora/data/main.db",
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_BACKUP_SOURCE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups",
                {},
                _http("POST", body={
                    "source_path": "main.db",
                    "backup_type": "incremental",
                    "metadata": {"triggered_by": "cron"},
                }),
            )
        assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_create_path_traversal_blocked(self, handler):
        """Path traversal attempts should be rejected."""
        from aragora.utils.paths import PathTraversalError

        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            side_effect=PathTraversalError("traversal"),
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_BACKUP_SOURCE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups",
                {},
                _http("POST", body={"source_path": "../../etc/passwd"}),
            )
        assert result.status_code == 400
        assert "Invalid source path" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_create_source_not_found(self, handler, mock_manager):
        """safe_path succeeds but manager raises FileNotFoundError."""
        mock_manager.create_backup.side_effect = FileNotFoundError("gone")
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            return_value="/var/aragora/data/main.db",
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_BACKUP_SOURCE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups",
                {},
                _http("POST", body={"source_path": "main.db"}),
            )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_create_manager_runtime_error(self, handler, mock_manager):
        mock_manager.create_backup.side_effect = RuntimeError("disk full")
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            return_value="/var/aragora/data/main.db",
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_BACKUP_SOURCE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups",
                {},
                _http("POST", body={"source_path": "main.db"}),
            )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_create_no_allowed_dirs_exist(self, handler):
        """If no allowed dir exists, source path is rejected."""
        with patch(
            "aragora.server.handlers.backup_handler._ALLOWED_BACKUP_SOURCE_DIRS",
            [MagicMock(exists=MagicMock(return_value=False))],
        ):
            result = await handler.handle(
                "/api/v2/backups",
                {},
                _http("POST", body={"source_path": "main.db"}),
            )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_file_not_found_in_safe_path(self, handler):
        """safe_path raises FileNotFoundError -- try next base, then fail."""
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            side_effect=FileNotFoundError("no file"),
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_BACKUP_SOURCE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups",
                {},
                _http("POST", body={"source_path": "nonexistent.db"}),
            )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_os_error_from_manager(self, handler, mock_manager):
        mock_manager.create_backup.side_effect = OSError("permission denied")
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            return_value="/var/aragora/data/main.db",
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_BACKUP_SOURCE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups",
                {},
                _http("POST", body={"source_path": "main.db"}),
            )
        assert result.status_code == 500


# ============================================================================
# Verify Backup
# ============================================================================


class TestVerifyBackup:
    """Tests for POST /api/v2/backups/:id/verify."""

    @pytest.mark.asyncio
    async def test_verify_success(self, handler, mock_manager):
        result = await handler.handle(
            "/api/v2/backups/bk-001/verify", {}, _http("POST")
        )
        assert result.status_code == 200
        data = _body(result)
        assert data["backup_id"] == "bk-001"
        assert data["verified"] is True
        assert data["checksum_valid"] is True
        assert data["restore_tested"] is True
        assert data["tables_valid"] is True
        assert data["row_counts_valid"] is True
        assert "verified_at" in data
        assert "duration_seconds" in data
        assert isinstance(data["errors"], list)
        assert isinstance(data["warnings"], list)

    @pytest.mark.asyncio
    async def test_verify_failed(self, handler, mock_manager):
        mock_manager.verify_backup.return_value = _VerificationResult(
            backup_id="bk-bad",
            verified=False,
            checksum_valid=False,
            restore_tested=False,
            tables_valid=False,
            row_counts_valid=False,
            errors=["checksum mismatch"],
        )
        result = await handler.handle(
            "/api/v2/backups/bk-bad/verify", {}, _http("POST")
        )
        assert result.status_code == 200
        data = _body(result)
        assert data["verified"] is False
        assert "checksum mismatch" in data["errors"]

    @pytest.mark.asyncio
    async def test_verify_passes_backup_id(self, handler, mock_manager):
        await handler.handle("/api/v2/backups/xyz-99/verify", {}, _http("POST"))
        mock_manager.verify_backup.assert_called_once_with("xyz-99", test_restore=True)


# ============================================================================
# Comprehensive Verify
# ============================================================================


class TestVerifyComprehensive:
    """Tests for POST /api/v2/backups/:id/verify-comprehensive."""

    @pytest.mark.asyncio
    async def test_comprehensive_success(self, handler, mock_manager):
        result = await handler.handle(
            "/api/v2/backups/bk-001/verify-comprehensive", {}, _http("POST")
        )
        assert result.status_code == 200
        data = _body(result)
        assert data["backup_id"] == "bk-001"
        assert data["verified"] is True

    @pytest.mark.asyncio
    async def test_comprehensive_passes_backup_id(self, handler, mock_manager):
        await handler.handle(
            "/api/v2/backups/comp-42/verify-comprehensive", {}, _http("POST")
        )
        mock_manager.verify_restore_comprehensive.assert_called_once_with("comp-42")


# ============================================================================
# Restore Test (dry-run)
# ============================================================================


class TestRestoreTest:
    """Tests for POST /api/v2/backups/:id/restore-test."""

    @pytest.mark.asyncio
    async def test_restore_test_success(self, handler, mock_manager):
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            return_value="/tmp/aragora_restore/restore_test.db",
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_RESTORE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups/bk-001/restore-test", {}, _http("POST")
            )
        assert result.status_code == 200
        data = _body(result)
        assert data["backup_id"] == "bk-001"
        assert data["restore_test_passed"] is True
        assert data["dry_run"] is True

    @pytest.mark.asyncio
    async def test_restore_test_with_custom_target(self, handler, mock_manager):
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            return_value="/tmp/aragora_restore/custom.db",
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_RESTORE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups/bk-001/restore-test",
                {},
                _http("POST", body={"target_path": "custom.db"}),
            )
        assert result.status_code == 200
        mock_manager.restore_backup.assert_called_once_with(
            backup_id="bk-001",
            target_path="/tmp/aragora_restore/custom.db",
            dry_run=True,
        )

    @pytest.mark.asyncio
    async def test_restore_test_path_traversal_blocked(self, handler):
        from aragora.utils.paths import PathTraversalError

        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            side_effect=PathTraversalError("traversal"),
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_RESTORE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups/bk-001/restore-test",
                {},
                _http("POST", body={"target_path": "../../../etc/shadow"}),
            )
        assert result.status_code == 400
        assert "Invalid target path" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_restore_test_value_error(self, handler, mock_manager):
        mock_manager.restore_backup.side_effect = ValueError("bad input")
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            return_value="/tmp/aragora_restore/test.db",
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_RESTORE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups/bk-001/restore-test", {}, _http("POST")
            )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_restore_test_file_not_found(self, handler, mock_manager):
        mock_manager.restore_backup.side_effect = FileNotFoundError("missing")
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            return_value="/tmp/aragora_restore/test.db",
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_RESTORE_DIRS",
            [MagicMock(exists=MagicMock(return_value=True))],
        ):
            result = await handler.handle(
                "/api/v2/backups/bk-001/restore-test", {}, _http("POST")
            )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_restore_no_allowed_dirs(self, handler):
        """All allowed restore dirs fail validation."""
        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            side_effect=OSError("cannot create"),
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_RESTORE_DIRS",
            [MagicMock(exists=MagicMock(return_value=False), mkdir=MagicMock(side_effect=OSError))],
        ):
            result = await handler.handle(
                "/api/v2/backups/bk-001/restore-test", {}, _http("POST")
            )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_restore_creates_missing_dir(self, handler, mock_manager):
        """If restore dir doesn't exist, handler should try to create it."""
        mock_dir = MagicMock()
        mock_dir.exists.return_value = False
        mock_dir.mkdir.return_value = None  # mkdir succeeds

        with patch(
            "aragora.server.handlers.backup_handler.safe_path",
            return_value="/tmp/aragora_restore/test.db",
        ), patch(
            "aragora.server.handlers.backup_handler._ALLOWED_RESTORE_DIRS",
            [mock_dir],
        ):
            result = await handler.handle(
                "/api/v2/backups/bk-001/restore-test", {}, _http("POST")
            )
        assert result.status_code == 200
        mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)


# ============================================================================
# Delete Backup
# ============================================================================


class TestDeleteBackup:
    """Tests for DELETE /api/v2/backups/:id."""

    @pytest.mark.asyncio
    async def test_delete_existing_backup(self, handler, mock_manager):
        bk = _make_backup("bk-del")
        mock_manager.list_backups.return_value = [bk]
        mock_manager._backups = {"bk-del": bk}

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.unlink"):
            http = _http("DELETE")
            result = await handler.handle("/api/v2/backups/bk-del", {}, http)

        assert result.status_code == 200
        data = _body(result)
        assert data["deleted"] is True
        assert data["backup_id"] == "bk-del"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_backup(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        result = await handler.handle("/api/v2/backups/gone", {}, _http("DELETE"))
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_file_not_on_disk(self, handler, mock_manager):
        """Backup tracked but file already gone from disk."""
        bk = _make_backup("bk-nofile")
        mock_manager.list_backups.return_value = [bk]
        mock_manager._backups = {"bk-nofile": bk}

        with patch("pathlib.Path.exists", return_value=False):
            result = await handler.handle("/api/v2/backups/bk-nofile", {}, _http("DELETE"))

        assert result.status_code == 200
        data = _body(result)
        assert data["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_os_error(self, handler, mock_manager):
        bk = _make_backup("bk-err")
        mock_manager.list_backups.return_value = [bk]
        mock_manager._backups = {"bk-err": bk}

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.unlink", side_effect=OSError("perm denied")):
            result = await handler.handle("/api/v2/backups/bk-err", {}, _http("DELETE"))

        assert result.status_code == 500


# ============================================================================
# Cleanup Expired
# ============================================================================


class TestCleanupExpired:
    """Tests for POST /api/v2/backups/cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_dry_run_default(self, handler, mock_manager):
        mock_manager.apply_retention_policy.return_value = ["bk-old-1", "bk-old-2"]
        result = await handler.handle("/api/v2/backups/cleanup", {}, _http("POST"))
        assert result.status_code == 200
        data = _body(result)
        assert data["dry_run"] is True
        assert data["count"] == 2
        assert "Would delete" in data["message"]
        mock_manager.apply_retention_policy.assert_called_once_with(dry_run=True)

    @pytest.mark.asyncio
    async def test_cleanup_actual_delete(self, handler, mock_manager):
        mock_manager.apply_retention_policy.return_value = ["bk-old-1"]
        result = await handler.handle(
            "/api/v2/backups/cleanup", {}, _http("POST", body={"dry_run": False})
        )
        assert result.status_code == 200
        data = _body(result)
        assert data["dry_run"] is False
        assert "Deleted" in data["message"]

    @pytest.mark.asyncio
    async def test_cleanup_nothing_to_delete(self, handler, mock_manager):
        mock_manager.apply_retention_policy.return_value = []
        result = await handler.handle("/api/v2/backups/cleanup", {}, _http("POST"))
        assert result.status_code == 200
        data = _body(result)
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_returns_backup_ids(self, handler, mock_manager):
        ids = ["a1", "b2", "c3"]
        mock_manager.apply_retention_policy.return_value = ids
        result = await handler.handle("/api/v2/backups/cleanup", {}, _http("POST"))
        data = _body(result)
        assert data["backup_ids"] == ids


# ============================================================================
# Stats
# ============================================================================


class TestStats:
    """Tests for GET /api/v2/backups/stats."""

    @pytest.mark.asyncio
    async def test_stats_empty(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        mock_manager.get_latest_backup.return_value = None
        result = await handler.handle("/api/v2/backups/stats", {}, _http("GET"))
        assert result.status_code == 200
        data = _body(result)
        stats = data["stats"]
        assert stats["total_backups"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["latest_backup"] is None
        assert "retention_policy" in stats
        assert "generated_at" in data

    @pytest.mark.asyncio
    async def test_stats_with_backups(self, handler, mock_manager):
        bk1 = _make_backup("bk-1", verified=True, compressed_size=1024 * 1024)
        bk2 = _make_backup("bk-2", verified=False, compressed_size=2 * 1024 * 1024)
        bk3 = _make_backup(
            "bk-3", status=_BackupStatus.FAILED, compressed_size=0
        )
        mock_manager.list_backups.return_value = [bk1, bk2, bk3]
        mock_manager.get_latest_backup.return_value = bk2

        result = await handler.handle("/api/v2/backups/stats", {}, _http("GET"))
        data = _body(result)
        stats = data["stats"]
        assert stats["total_backups"] == 3
        assert stats["verified_backups"] == 1
        assert stats["failed_backups"] == 1
        assert stats["total_size_bytes"] == 3 * 1024 * 1024
        assert stats["total_size_mb"] == 3.0
        assert stats["latest_backup"] is not None

    @pytest.mark.asyncio
    async def test_stats_retention_policy(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        mock_manager.retention_policy = _RetentionPolicy(
            keep_daily=14, keep_weekly=8, keep_monthly=6, min_backups=3
        )
        result = await handler.handle("/api/v2/backups/stats", {}, _http("GET"))
        rp = _body(result)["stats"]["retention_policy"]
        assert rp["keep_daily"] == 14
        assert rp["keep_weekly"] == 8
        assert rp["keep_monthly"] == 6
        assert rp["min_backups"] == 3


# ============================================================================
# Invalid Path / 404
# ============================================================================


class TestNotFound:
    """Tests for unmatched routes that reach the handler."""

    @pytest.mark.asyncio
    async def test_invalid_short_path(self, handler):
        """Path /api/v2/backups/ with trailing slash: backup_id is empty string, returns 404."""
        result = await handler.handle("/api/v2/backups/", {}, _http("GET"))
        # The empty backup_id won't match any backup, so we get 404
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_sub_route(self, handler):
        """Unknown action under a backup_id returns 404."""
        result = await handler.handle(
            "/api/v2/backups/bk-001/unknown-action", {}, _http("POST")
        )
        assert result.status_code == 404


# ============================================================================
# Error Handling (generic exception catch)
# ============================================================================


class TestGenericErrorHandling:
    """Test that handled exceptions produce 500."""

    @pytest.mark.asyncio
    async def test_value_error_in_list(self, handler, mock_manager):
        mock_manager.list_backups.side_effect = ValueError("boom")
        result = await handler.handle("/api/v2/backups", {}, _http("GET"))
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_key_error_in_stats(self, handler, mock_manager):
        mock_manager.list_backups.side_effect = KeyError("missing")
        result = await handler.handle("/api/v2/backups/stats", {}, _http("GET"))
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_type_error_in_verify(self, handler, mock_manager):
        mock_manager.verify_backup.side_effect = TypeError("bad type")
        result = await handler.handle(
            "/api/v2/backups/bk-001/verify", {}, _http("POST")
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_runtime_error_in_cleanup(self, handler, mock_manager):
        mock_manager.apply_retention_policy.side_effect = RuntimeError("oops")
        result = await handler.handle(
            "/api/v2/backups/cleanup", {}, _http("POST")
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_os_error_in_get_backup(self, handler, mock_manager):
        mock_manager.list_backups.side_effect = OSError("disk")
        result = await handler.handle("/api/v2/backups/bk-001", {}, _http("GET"))
        assert result.status_code == 500


# ============================================================================
# Timestamp Parsing
# ============================================================================


class TestTimestampParsing:
    """Tests for _parse_timestamp helper."""

    def test_parse_none(self, handler):
        assert handler._parse_timestamp(None) is None

    def test_parse_empty_string(self, handler):
        assert handler._parse_timestamp("") is None

    def test_parse_unix_timestamp(self, handler):
        dt = handler._parse_timestamp("1700000000")
        assert dt is not None
        assert dt.year == 2023

    def test_parse_unix_float(self, handler):
        dt = handler._parse_timestamp("1700000000.5")
        assert dt is not None

    def test_parse_iso_date(self, handler):
        dt = handler._parse_timestamp("2026-01-15T12:00:00+00:00")
        assert dt is not None
        assert dt.year == 2026

    def test_parse_iso_date_z_suffix(self, handler):
        dt = handler._parse_timestamp("2026-01-15T12:00:00Z")
        assert dt is not None

    def test_parse_invalid_string(self, handler):
        assert handler._parse_timestamp("not-a-date") is None


# ============================================================================
# Handler Factory
# ============================================================================


class TestFactory:
    """Tests for the create_backup_handler factory."""

    def test_factory_returns_handler(self):
        from aragora.server.handlers.backup_handler import create_backup_handler

        h = create_backup_handler({"nomic_dir": "/tmp/test"})
        assert h is not None
        assert hasattr(h, "handle")
        assert hasattr(h, "can_handle")


# ============================================================================
# Lazy Manager Initialization
# ============================================================================


class TestLazyManager:
    """Test _get_manager lazy initialization."""

    def test_returns_injected_manager(self, handler, mock_manager):
        assert handler._get_manager() is mock_manager

    def test_lazy_init_when_none(self):
        from aragora.server.handlers.backup_handler import BackupHandler

        h = BackupHandler({"nomic_dir": "/tmp/test"})
        h._manager = None
        h._manager_factory = MagicMock()
        sentinel = object()
        h._manager_factory.get.return_value = sentinel
        assert h._get_manager() is sentinel

    def test_caches_after_first_call(self):
        from aragora.server.handlers.backup_handler import BackupHandler

        h = BackupHandler({"nomic_dir": "/tmp/test"})
        h._manager = None
        h._manager_factory = MagicMock()
        sentinel = object()
        h._manager_factory.get.return_value = sentinel
        h._get_manager()
        h._get_manager()
        # Factory only called once
        h._manager_factory.get.assert_called_once()


# ============================================================================
# Method Routing within handle()
# ============================================================================


class TestMethodRouting:
    """Edge cases in method extraction and routing."""

    @pytest.mark.asyncio
    async def test_default_method_is_get(self, handler, mock_manager):
        """When handler is None, method defaults to GET."""
        mock_manager.list_backups.return_value = []
        result = await handler.handle("/api/v2/backups", {}, None)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handler_without_command_defaults_get(self, handler, mock_manager):
        """Handler object without .command attribute defaults to GET."""
        mock_manager.list_backups.return_value = []
        # Need a handler that has headers/rfile (for read_json_body) but no .command
        plain = MagicMock(spec=[])  # No attributes by default
        plain.headers = {"Content-Length": "0"}
        plain.rfile = MagicMock()
        plain.rfile.read.return_value = b""
        result = await handler.handle("/api/v2/backups", {}, plain)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_query_params_none_handled(self, handler, mock_manager):
        """None query_params is coerced to empty dict."""
        mock_manager.list_backups.return_value = []
        result = await handler.handle("/api/v2/backups", None, _http("GET"))
        assert result.status_code == 200


# ============================================================================
# ROUTES Class Variable
# ============================================================================


class TestRoutes:
    """Tests that the ROUTES class variable is correct."""

    def test_routes_contains_v2_backups(self, handler):
        assert "/api/v2/backups" in handler.ROUTES

    def test_routes_contains_wildcard(self, handler):
        assert "/api/v2/backups/*" in handler.ROUTES

    def test_routes_has_v1_cleanup(self, handler):
        assert "/api/v1/backups/cleanup" in handler.ROUTES

    def test_routes_has_v1_stats(self, handler):
        assert "/api/v1/backups/stats" in handler.ROUTES
