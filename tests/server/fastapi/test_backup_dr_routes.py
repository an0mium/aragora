"""FastAPI tests for backup and DR bridge routes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app
from aragora.server.fastapi.dependencies.auth import require_authenticated


class _BackupRecord:
    def __init__(self, backup_id: str = "backup-001") -> None:
        self.id = backup_id
        self.created_at = datetime.now(timezone.utc)
        self.backup_type = "full"
        self.status = "verified"
        self.source_path = "/tmp/core.db"
        self.backup_path = "/tmp/backup.db.gz"
        self.size_bytes = 1024
        self.compressed_size_bytes = 512
        self.checksum = "abc123"
        self.row_counts: dict[str, int] = {}
        self.tables: list[str] = []
        self.duration_seconds = 1.2
        self.verified = True
        self.verified_at = None
        self.restore_tested = True
        self.error = None
        self.storage_backend = "local"
        self.encryption_key_id = None
        self.metadata: dict[str, Any] = {}
        self.schema_hash = ""
        self.table_checksums: dict[str, str] = {}
        self.foreign_keys: list[tuple[str, ...]] = []
        self.indexes: list[tuple[str, ...]] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "backup_type": self.backup_type,
            "status": self.status,
            "source_path": self.source_path,
            "backup_path": self.backup_path,
            "size_bytes": self.size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "checksum": self.checksum,
            "row_counts": self.row_counts,
            "tables": self.tables,
            "duration_seconds": self.duration_seconds,
            "verified": self.verified,
            "verified_at": self.verified_at,
            "restore_tested": self.restore_tested,
            "error": self.error,
            "storage_backend": self.storage_backend,
            "encryption_key_id": self.encryption_key_id,
            "metadata": self.metadata,
            "schema_hash": self.schema_hash,
            "table_checksums": self.table_checksums,
            "foreign_keys": [],
            "indexes": [],
        }


@pytest.fixture
def backup_manager() -> MagicMock:
    manager = MagicMock()
    record = _BackupRecord()
    manager.list_backups.return_value = [record]
    manager.get_latest_backup.return_value = record
    manager.create_backup.return_value = _BackupRecord("backup-002")
    manager.verify_restore_comprehensive.return_value = MagicMock(
        verified=True,
        schema_validation=MagicMock(valid=True),
        integrity_check=MagicMock(valid=True),
        to_dict=lambda: {"verified": True},
    )
    manager.restore_backup.return_value = True
    manager.retention_policy.keep_daily = 7
    manager.retention_policy.keep_weekly = 4
    manager.retention_policy.keep_monthly = 3
    manager.retention_policy.min_backups = 1
    return manager


@pytest.fixture
def app(backup_manager: MagicMock):
    app = create_app()
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
        "backup_manager": backup_manager,
    }
    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _override_auth(client: TestClient, permissions: set[str]) -> None:
    from aragora.rbac.models import AuthorizationContext

    auth_ctx = AuthorizationContext(
        user_id="user-1",
        org_id="org-1",
        workspace_id="ws-1",
        roles={"admin"},
        permissions=permissions,
    )
    client.app.dependency_overrides[require_authenticated] = lambda: auth_ctx


def test_backup_and_dr_routes_are_registered(client: TestClient) -> None:
    _override_auth(client, {"backups:read", "dr:read"})
    try:
        assert client.get("/api/v2/backups/stats").status_code == 200
        assert client.get("/api/v2/backups?limit=1").status_code == 200
        assert client.get("/api/v2/dr/status").status_code == 200
        assert client.get("/api/v2/dr/objectives").status_code == 200
    finally:
        client.app.dependency_overrides.clear()


def test_create_backup_route_uses_default_source_path(
    client: TestClient,
    backup_manager: MagicMock,
    tmp_path: Path,
) -> None:
    default_db = tmp_path / "core.db"
    default_db.write_text("sqlite", encoding="utf-8")

    _override_auth(client, {"backups:create"})
    try:
        with patch(
            "aragora.server.handlers.backup_handler.get_default_backup_source_path",
            return_value=default_db,
        ):
            response = client.post("/api/v2/backups", json={"backup_type": "full"})
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 201
    backup_manager.create_backup.assert_called_once()
    assert backup_manager.create_backup.call_args.kwargs["source_path"] == default_db


def test_dr_drill_route_is_registered(client: TestClient) -> None:
    _override_auth(client, {"dr:drill"})
    try:
        response = client.post("/api/v2/dr/drill", json={"drill_type": "restore_test"})
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["drill_type"] == "restore_test"
    assert body["success"] is True
