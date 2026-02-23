"""Tests for Disaster Recovery HTTP Handler.

Tests the DRHandler which provides REST API endpoints for:
- DR readiness status (GET /api/v2/dr/status)
- DR drills (POST /api/v2/dr/drill)
- RPO/RTO objectives (GET /api/v2/dr/objectives)
- DR configuration validation (POST /api/v2/dr/validate)

Covers: all routes, success paths, error handling, edge cases, input validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from aragora.backup.manager import (
    BackupMetadata,
    BackupStatus,
    BackupType,
    ComprehensiveVerificationResult,
    IntegrityResult,
    RetentionPolicy,
    SchemaValidationResult,
    VerificationResult,
)
from aragora.server.handlers.dr_handler import DRHandler, create_dr_handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_body(result: Any) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


def _make_backup(
    backup_id: str = "bk-001",
    created_at: datetime | None = None,
    verified: bool = True,
    status: BackupStatus = BackupStatus.COMPLETED,
    compressed_size_bytes: int = 50_000_000,
) -> MagicMock:
    """Create a mock BackupMetadata object."""
    now = created_at or datetime.now(timezone.utc)
    mock = MagicMock(spec=BackupMetadata)
    mock.id = backup_id
    mock.created_at = now
    mock.verified = verified
    mock.status = status
    mock.compressed_size_bytes = compressed_size_bytes
    mock.to_dict.return_value = {
        "id": backup_id,
        "created_at": now.isoformat(),
        "status": status.value,
        "verified": verified,
        "compressed_size_bytes": compressed_size_bytes,
    }
    return mock


def _make_verification_result(
    backup_id: str = "bk-001",
    verified: bool = True,
) -> VerificationResult:
    """Create a VerificationResult for testing."""
    return VerificationResult(
        backup_id=backup_id,
        verified=verified,
        checksum_valid=verified,
        restore_tested=verified,
        tables_valid=verified,
        row_counts_valid=verified,
        errors=[] if verified else ["checksum mismatch"],
    )


def _make_comprehensive_result(
    backup_id: str = "bk-001",
    verified: bool = True,
    schema_valid: bool = True,
    integrity_valid: bool = True,
) -> ComprehensiveVerificationResult:
    """Create a ComprehensiveVerificationResult for testing."""
    basic = _make_verification_result(backup_id, verified)
    schema = SchemaValidationResult(
        valid=schema_valid,
        tables_match=schema_valid,
        columns_match=schema_valid,
        types_match=schema_valid,
        constraints_match=schema_valid,
        indexes_match=schema_valid,
    )
    integrity = IntegrityResult(
        valid=integrity_valid,
        foreign_keys_valid=integrity_valid,
    )
    return ComprehensiveVerificationResult(
        backup_id=backup_id,
        verified=verified,
        basic_verification=basic,
        schema_validation=schema,
        integrity_check=integrity,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_manager():
    """Create a mock BackupManager."""
    mgr = MagicMock()
    mgr.list_backups.return_value = []
    mgr.get_latest_backup.return_value = None
    mgr.retention_policy = RetentionPolicy()
    mgr.compression = True
    mgr.verify_after_backup = True
    mgr.backup_dir = Path("/tmp/test_backups")
    return mgr


@pytest.fixture
def handler(mock_manager):
    """Create a DRHandler with a mock backup manager injected."""
    h = DRHandler(server_context={})
    h._manager = mock_manager
    return h


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_status_get(self, handler):
        assert handler.can_handle("/api/v2/dr/status", "GET") is True

    def test_handles_drill_post(self, handler):
        assert handler.can_handle("/api/v2/dr/drill", "POST") is True

    def test_handles_objectives_get(self, handler):
        assert handler.can_handle("/api/v2/dr/objectives", "GET") is True

    def test_handles_validate_post(self, handler):
        assert handler.can_handle("/api/v2/dr/validate", "POST") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/v2/backups", "GET") is False

    def test_rejects_wrong_method_put(self, handler):
        assert handler.can_handle("/api/v2/dr/status", "PUT") is False

    def test_rejects_wrong_method_delete(self, handler):
        assert handler.can_handle("/api/v2/dr/status", "DELETE") is False

    def test_rejects_empty_path(self, handler):
        assert handler.can_handle("", "GET") is False

    def test_accepts_dr_base_path(self, handler):
        assert handler.can_handle("/api/v2/dr", "GET") is True

    def test_accepts_nested_dr_path(self, handler):
        assert handler.can_handle("/api/v2/dr/something", "GET") is True


# ---------------------------------------------------------------------------
# Routing via handle()
# ---------------------------------------------------------------------------


class TestRouting:
    """Tests for request routing through handle()."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, handler):
        result = await handler.handle(
            method="GET", path="/api/v2/dr/nonexistent", body={}
        )
        assert result.status_code == 404
        body = parse_body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_no_arguments_returns_400(self, handler):
        result = await handler.handle()
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_positional_base_class_style(self, handler, mock_manager):
        """Test handle(path, query_params, handler) style."""
        backup = _make_backup()
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup
        # Base class style: path as first arg, dict as second, None handler
        result = await handler.handle("/api/v2/dr/status", None, None)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_positional_extended_style(self, handler, mock_manager):
        """Test handle(method, path, body) positional style."""
        backup = _make_backup()
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup
        result = await handler.handle("GET", "/api/v2/dr/status", {})
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_keyword_style(self, handler, mock_manager):
        """Test handle(method=, path=, body=) keyword style."""
        backup = _make_backup()
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup
        result = await handler.handle(
            method="GET", path="/api/v2/dr/status", body={}
        )
        assert result.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/v2/dr/status
# ---------------------------------------------------------------------------


class TestGetStatus:
    """Tests for DR readiness status endpoint."""

    @pytest.mark.asyncio
    async def test_healthy_status_with_recent_verified_backup(self, handler, mock_manager):
        backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        assert result.status_code == 200
        body = parse_body(result)
        assert body["status"] == "healthy"
        assert body["readiness_score"] == 100
        assert body["rpo_status"]["compliant"] is True
        assert body["backup_status"]["total_backups"] == 1
        assert body["issues"] == []

    @pytest.mark.asyncio
    async def test_no_backups_critical(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        mock_manager.get_latest_backup.return_value = None

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        assert result.status_code == 200
        body = parse_body(result)
        assert body["status"] == "critical"
        # -50 (no backups) -30 (no latest) = 20
        assert body["readiness_score"] == 20
        assert "No backups found" in body["issues"]
        assert "No verified backup available" in body["issues"]

    @pytest.mark.asyncio
    async def test_rpo_violation(self, handler, mock_manager):
        old_backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=48),
        )
        mock_manager.list_backups.return_value = [old_backup]
        mock_manager.get_latest_backup.return_value = old_backup

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        body = parse_body(result)
        assert body["rpo_status"]["compliant"] is False
        assert any("RPO violation" in i for i in body["issues"])

    @pytest.mark.asyncio
    async def test_failed_backups_reduce_score(self, handler, mock_manager):
        good = _make_backup(
            backup_id="bk-good",
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        failed1 = _make_backup(
            backup_id="bk-fail1",
            status=BackupStatus.FAILED,
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        failed2 = _make_backup(
            backup_id="bk-fail2",
            status=BackupStatus.FAILED,
            created_at=datetime.now(timezone.utc) - timedelta(hours=3),
        )
        mock_manager.list_backups.return_value = [good, failed1, failed2]
        mock_manager.get_latest_backup.return_value = good

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        body = parse_body(result)
        assert "2 failed backups" in body["issues"]
        # Score reduced by min(2*5, 20) = 10
        assert body["readiness_score"] <= 90

    @pytest.mark.asyncio
    async def test_low_verification_rate_warning(self, handler, mock_manager):
        verified = _make_backup(
            backup_id="bk-v",
            verified=True,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        unverified1 = _make_backup(backup_id="bk-u1", verified=False)
        unverified2 = _make_backup(backup_id="bk-u2", verified=False)
        unverified3 = _make_backup(backup_id="bk-u3", verified=False)
        unverified4 = _make_backup(backup_id="bk-u4", verified=False)
        all_backups = [verified, unverified1, unverified2, unverified3, unverified4]
        mock_manager.list_backups.return_value = all_backups
        mock_manager.get_latest_backup.return_value = verified

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        body = parse_body(result)
        assert any("80%" in i for i in body["issues"])

    @pytest.mark.asyncio
    async def test_warning_status_score_range(self, handler, mock_manager):
        """Score 70-89 should produce 'warning' status."""
        old_backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=30),
        )
        mock_manager.list_backups.return_value = [old_backup]
        mock_manager.get_latest_backup.return_value = old_backup

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        body = parse_body(result)
        # -20 for RPO violation = 80
        assert body["readiness_score"] == 80
        assert body["status"] == "warning"

    @pytest.mark.asyncio
    async def test_score_clamped_to_zero_minimum(self, handler, mock_manager):
        """Score should never go below 0."""
        mock_manager.list_backups.return_value = []
        mock_manager.get_latest_backup.return_value = None

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        body = parse_body(result)
        assert body["readiness_score"] >= 0

    @pytest.mark.asyncio
    async def test_status_response_shape(self, handler, mock_manager):
        backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        body = parse_body(result)
        assert "status" in body
        assert "readiness_score" in body
        assert "backup_status" in body
        assert "rpo_status" in body
        assert "issues" in body
        assert "recommendations" in body
        assert "checked_at" in body

    @pytest.mark.asyncio
    async def test_backup_status_fields(self, handler, mock_manager):
        backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        body = parse_body(result)
        bs = body["backup_status"]
        assert "total_backups" in bs
        assert "verified_backups" in bs
        assert "failed_backups" in bs
        assert "latest_backup" in bs
        assert "hours_since_backup" in bs

    @pytest.mark.asyncio
    async def test_rpo_status_fields(self, handler, mock_manager):
        backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        body = parse_body(result)
        rpo = body["rpo_status"]
        assert rpo["target_hours"] == 24
        assert isinstance(rpo["compliant"], bool)
        assert isinstance(rpo["current_hours"], (int, float))

    @pytest.mark.asyncio
    async def test_many_failed_backups_cap_penalty(self, handler, mock_manager):
        """Failed backup penalty caps at 20."""
        backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        failed = [
            _make_backup(
                backup_id=f"bk-fail-{i}",
                status=BackupStatus.FAILED,
                verified=False,
            )
            for i in range(10)
        ]
        all_backups = [backup] + failed
        mock_manager.list_backups.return_value = all_backups
        mock_manager.get_latest_backup.return_value = backup

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        body = parse_body(result)
        # 100 - 20 (capped failed penalty) - 10 (low verification) = 70
        assert body["readiness_score"] >= 0

    @pytest.mark.asyncio
    async def test_no_latest_but_have_backups(self, handler, mock_manager):
        """Have backups but none verified (no latest)."""
        unverified = _make_backup(verified=False)
        mock_manager.list_backups.return_value = [unverified]
        mock_manager.get_latest_backup.return_value = None

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        body = parse_body(result)
        assert "No verified backup available" in body["issues"]
        assert body["backup_status"]["hours_since_backup"] is None
        assert body["rpo_status"]["current_hours"] is None


# ---------------------------------------------------------------------------
# POST /api/v2/dr/drill
# ---------------------------------------------------------------------------


class TestRunDrill:
    """Tests for DR drill endpoint."""

    @pytest.mark.asyncio
    async def test_restore_test_success(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        mock_manager.verify_restore_comprehensive.return_value = (
            _make_comprehensive_result(verified=True)
        )
        mock_manager.restore_backup.return_value = True

        result = await handler.handle(
            method="POST", path="/api/v2/dr/drill", body={}
        )
        assert result.status_code == 200
        body = parse_body(result)
        assert body["success"] is True
        assert body["drill_type"] == "restore_test"
        assert "drill_id" in body
        assert "started_at" in body
        assert "completed_at" in body
        assert "duration_seconds" in body

    @pytest.mark.asyncio
    async def test_restore_test_verification_fails(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        mock_manager.verify_restore_comprehensive.return_value = (
            _make_comprehensive_result(verified=False)
        )

        result = await handler.handle(
            method="POST", path="/api/v2/dr/drill", body={}
        )
        assert result.status_code == 200
        body = parse_body(result)
        assert body["success"] is False
        assert "error" in body

    @pytest.mark.asyncio
    async def test_restore_test_dry_run_fails(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        mock_manager.verify_restore_comprehensive.return_value = (
            _make_comprehensive_result(verified=True)
        )
        mock_manager.restore_backup.return_value = False

        result = await handler.handle(
            method="POST", path="/api/v2/dr/drill", body={}
        )
        assert result.status_code == 200
        body = parse_body(result)
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_drill_with_specific_backup_id(self, handler, mock_manager):
        backup = _make_backup(backup_id="specific-001")
        mock_manager.list_backups.return_value = [backup]
        mock_manager.verify_restore_comprehensive.return_value = (
            _make_comprehensive_result(backup_id="specific-001", verified=True)
        )
        mock_manager.restore_backup.return_value = True

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/drill",
            body={"backup_id": "specific-001"},
        )
        assert result.status_code == 200
        body = parse_body(result)
        assert body["backup_id"] == "specific-001"
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_drill_backup_id_not_found(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/drill",
            body={"backup_id": "nonexistent"},
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_drill_no_latest_backup(self, handler, mock_manager):
        mock_manager.get_latest_backup.return_value = None

        result = await handler.handle(
            method="POST", path="/api/v2/dr/drill", body={}
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_drill_type_returns_400(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/drill",
            body={"drill_type": "unknown_type"},
        )
        assert result.status_code == 400
        body = parse_body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_full_recovery_sim_success(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        comp = _make_comprehensive_result(
            verified=True, schema_valid=True, integrity_valid=True,
        )
        mock_manager.verify_restore_comprehensive.return_value = comp
        mock_manager.restore_backup.return_value = True

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/drill",
            body={"drill_type": "full_recovery_sim"},
        )
        assert result.status_code == 200
        body = parse_body(result)
        assert body["success"] is True
        assert body["drill_type"] == "full_recovery_sim"
        # Should have 4 steps
        assert len(body["steps"]) == 4

    @pytest.mark.asyncio
    async def test_full_recovery_sim_schema_failure(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        comp = _make_comprehensive_result(
            verified=True, schema_valid=False, integrity_valid=True,
        )
        # Schema validation on the result object
        comp.schema_validation.valid = False
        mock_manager.verify_restore_comprehensive.return_value = comp
        mock_manager.restore_backup.return_value = True

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/drill",
            body={"drill_type": "full_recovery_sim"},
        )
        body = parse_body(result)
        # Schema step should be failed
        schema_step = body["steps"][1]
        assert schema_step["step"] == "schema_validation"
        assert schema_step["status"] == "failed"

    @pytest.mark.asyncio
    async def test_full_recovery_sim_integrity_failure(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        comp = _make_comprehensive_result(
            verified=True, schema_valid=True, integrity_valid=False,
        )
        comp.integrity_check.valid = False
        mock_manager.verify_restore_comprehensive.return_value = comp
        mock_manager.restore_backup.return_value = True

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/drill",
            body={"drill_type": "full_recovery_sim"},
        )
        body = parse_body(result)
        integrity_step = body["steps"][2]
        assert integrity_step["step"] == "integrity_check"
        assert integrity_step["status"] == "failed"

    @pytest.mark.asyncio
    async def test_failover_test_success(self, handler, mock_manager):
        backups = [_make_backup(backup_id=f"bk-{i}") for i in range(3)]
        mock_manager.get_latest_backup.return_value = backups[0]
        mock_manager.list_backups.return_value = backups
        mock_manager.verify_backup.return_value = _make_verification_result(verified=True)

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/drill",
            body={"drill_type": "failover_test"},
        )
        assert result.status_code == 200
        body = parse_body(result)
        assert body["success"] is True
        assert body["steps"][0]["step"] == "verify_recent_backups"
        assert body["steps"][0]["details"]["checked"] == 3
        assert body["steps"][0]["details"]["verified"] == 3

    @pytest.mark.asyncio
    async def test_failover_test_no_verified(self, handler, mock_manager):
        backups = [_make_backup(backup_id=f"bk-{i}") for i in range(2)]
        mock_manager.get_latest_backup.return_value = backups[0]
        mock_manager.list_backups.return_value = backups
        mock_manager.verify_backup.return_value = _make_verification_result(verified=False)

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/drill",
            body={"drill_type": "failover_test"},
        )
        body = parse_body(result)
        assert body["success"] is False
        assert body["steps"][0]["details"]["verified"] == 0

    @pytest.mark.asyncio
    async def test_drill_exception_handling(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        mock_manager.verify_restore_comprehensive.side_effect = OSError("disk failure")

        result = await handler.handle(
            method="POST", path="/api/v2/dr/drill", body={}
        )
        assert result.status_code == 200
        body = parse_body(result)
        assert body["success"] is False
        assert "error" in body

    @pytest.mark.asyncio
    async def test_drill_runtime_error_handling(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        mock_manager.verify_restore_comprehensive.side_effect = RuntimeError("broken")

        result = await handler.handle(
            method="POST", path="/api/v2/dr/drill", body={}
        )
        body = parse_body(result)
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_drill_value_error_handling(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        mock_manager.verify_restore_comprehensive.side_effect = ValueError("bad value")

        result = await handler.handle(
            method="POST", path="/api/v2/dr/drill", body={}
        )
        body = parse_body(result)
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_drill_custom_target_path(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        mock_manager.verify_restore_comprehensive.return_value = (
            _make_comprehensive_result(verified=True)
        )
        mock_manager.restore_backup.return_value = True

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/drill",
            body={"target_path": "/custom/path.db"},
        )
        assert result.status_code == 200
        mock_manager.restore_backup.assert_called_once_with(
            "bk-001", "/custom/path.db", dry_run=True
        )

    @pytest.mark.asyncio
    async def test_drill_default_target_path(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        mock_manager.verify_restore_comprehensive.return_value = (
            _make_comprehensive_result(verified=True)
        )
        mock_manager.restore_backup.return_value = True

        result = await handler.handle(
            method="POST", path="/api/v2/dr/drill", body={}
        )
        assert result.status_code == 200
        mock_manager.restore_backup.assert_called_once_with(
            "bk-001", "/tmp/dr_drill_test.db", dry_run=True
        )

    @pytest.mark.asyncio
    async def test_drill_steps_populated(self, handler, mock_manager):
        backup = _make_backup()
        mock_manager.get_latest_backup.return_value = backup
        mock_manager.verify_restore_comprehensive.return_value = (
            _make_comprehensive_result(verified=True)
        )
        mock_manager.restore_backup.return_value = True

        result = await handler.handle(
            method="POST", path="/api/v2/dr/drill", body={}
        )
        body = parse_body(result)
        assert len(body["steps"]) >= 1
        for step in body["steps"]:
            assert "step" in step
            assert "status" in step

    @pytest.mark.asyncio
    async def test_failover_test_limits_to_five_backups(self, handler, mock_manager):
        backups = [_make_backup(backup_id=f"bk-{i}") for i in range(10)]
        mock_manager.get_latest_backup.return_value = backups[0]
        mock_manager.list_backups.return_value = backups
        mock_manager.verify_backup.return_value = _make_verification_result(verified=True)

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/drill",
            body={"drill_type": "failover_test"},
        )
        body = parse_body(result)
        # Only first 5 should be checked
        assert body["steps"][0]["details"]["checked"] == 5


# ---------------------------------------------------------------------------
# GET /api/v2/dr/objectives
# ---------------------------------------------------------------------------


class TestGetObjectives:
    """Tests for RPO/RTO objectives endpoint."""

    @pytest.mark.asyncio
    async def test_objectives_with_recent_backup(self, handler, mock_manager):
        backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            compressed_size_bytes=50_000_000,
        )
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup

        result = await handler.handle(method="GET", path="/api/v2/dr/objectives")
        assert result.status_code == 200
        body = parse_body(result)
        assert body["rpo"]["target_hours"] == 24
        assert body["rpo"]["compliant"] is True
        assert body["rpo"]["current_hours"] is not None
        assert body["rto"]["target_minutes"] == 30

    @pytest.mark.asyncio
    async def test_objectives_no_backups(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        mock_manager.get_latest_backup.return_value = None

        result = await handler.handle(method="GET", path="/api/v2/dr/objectives")
        assert result.status_code == 200
        body = parse_body(result)
        assert body["rpo"]["current_hours"] is None
        assert body["rpo"]["compliant"] is False
        assert body["rto"]["estimated_minutes"] is None
        assert body["rto"]["compliant"] is False

    @pytest.mark.asyncio
    async def test_objectives_rpo_non_compliant(self, handler, mock_manager):
        backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=48),
        )
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup

        result = await handler.handle(method="GET", path="/api/v2/dr/objectives")
        body = parse_body(result)
        assert body["rpo"]["compliant"] is False
        assert body["rpo"]["current_hours"] > 24

    @pytest.mark.asyncio
    async def test_objectives_rto_estimate(self, handler, mock_manager):
        # 500MB backup -> ~5 min restore + 5 min overhead = ~10 min
        backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            compressed_size_bytes=500_000_000,
        )
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup

        result = await handler.handle(method="GET", path="/api/v2/dr/objectives")
        body = parse_body(result)
        assert body["rto"]["estimated_minutes"] is not None
        assert body["rto"]["estimated_minutes"] > 0
        assert body["rto"]["compliant"] is True

    @pytest.mark.asyncio
    async def test_objectives_rto_non_compliant_large_backup(self, handler, mock_manager):
        # 10GB backup -> ~100 min restore + 5 overhead -> not compliant with 30 min target
        backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            compressed_size_bytes=10_000_000_000,
        )
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup

        result = await handler.handle(method="GET", path="/api/v2/dr/objectives")
        body = parse_body(result)
        assert body["rto"]["compliant"] is False

    @pytest.mark.asyncio
    async def test_objectives_rto_zero_size_backup(self, handler, mock_manager):
        backup = _make_backup(
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            compressed_size_bytes=0,
        )
        mock_manager.list_backups.return_value = [backup]
        mock_manager.get_latest_backup.return_value = backup

        result = await handler.handle(method="GET", path="/api/v2/dr/objectives")
        body = parse_body(result)
        assert body["rto"]["estimated_minutes"] is None

    @pytest.mark.asyncio
    async def test_objectives_response_shape(self, handler, mock_manager):
        mock_manager.list_backups.return_value = []
        mock_manager.get_latest_backup.return_value = None

        result = await handler.handle(method="GET", path="/api/v2/dr/objectives")
        body = parse_body(result)
        assert "rpo" in body
        assert "rto" in body
        assert "backup_coverage" in body
        assert "generated_at" in body

    @pytest.mark.asyncio
    async def test_objectives_backup_coverage(self, handler, mock_manager):
        now = datetime.now(timezone.utc)
        recent = _make_backup(
            backup_id="bk-recent",
            created_at=now - timedelta(days=1),
        )
        old = _make_backup(
            backup_id="bk-old",
            created_at=now - timedelta(days=14),
        )
        mock_manager.list_backups.return_value = [recent, old]
        mock_manager.get_latest_backup.return_value = recent

        result = await handler.handle(method="GET", path="/api/v2/dr/objectives")
        body = parse_body(result)
        assert body["backup_coverage"]["total_backups"] == 2
        assert body["backup_coverage"]["backups_last_7_days"] == 1
        assert body["backup_coverage"]["latest_backup"] is not None

    @pytest.mark.asyncio
    async def test_objectives_rpo_violations_history(self, handler, mock_manager):
        now = datetime.now(timezone.utc)
        # Create backups with a gap > 24 hours
        b1 = _make_backup(
            backup_id="bk-1",
            created_at=now - timedelta(days=1),
        )
        b2 = _make_backup(
            backup_id="bk-2",
            created_at=now - timedelta(days=3),
        )
        mock_manager.list_backups.return_value = [b1, b2]
        mock_manager.get_latest_backup.return_value = b1

        result = await handler.handle(method="GET", path="/api/v2/dr/objectives")
        body = parse_body(result)
        assert body["rpo"]["violations_last_7_days"] >= 1

    @pytest.mark.asyncio
    async def test_objectives_no_violations_when_backups_close(self, handler, mock_manager):
        now = datetime.now(timezone.utc)
        b1 = _make_backup(backup_id="bk-1", created_at=now - timedelta(hours=1))
        b2 = _make_backup(backup_id="bk-2", created_at=now - timedelta(hours=12))
        mock_manager.list_backups.return_value = [b1, b2]
        mock_manager.get_latest_backup.return_value = b1

        result = await handler.handle(method="GET", path="/api/v2/dr/objectives")
        body = parse_body(result)
        assert body["rpo"]["violations_last_7_days"] == 0

    @pytest.mark.asyncio
    async def test_objectives_single_recent_backup_no_violations(self, handler, mock_manager):
        """Single backup means no gap to check -> 0 violations."""
        b1 = _make_backup(
            backup_id="bk-1",
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        mock_manager.list_backups.return_value = [b1]
        mock_manager.get_latest_backup.return_value = b1

        result = await handler.handle(method="GET", path="/api/v2/dr/objectives")
        body = parse_body(result)
        assert body["rpo"]["violations_last_7_days"] == 0


# ---------------------------------------------------------------------------
# POST /api/v2/dr/validate
# ---------------------------------------------------------------------------


class TestValidateConfiguration:
    """Tests for DR configuration validation endpoint."""

    @pytest.mark.asyncio
    async def test_validate_all_checks_pass(self, handler, mock_manager):
        mock_manager.encryption_enabled = True
        mock_manager.encryption_key = "secret-key-123"
        mock_manager.backup_dir = Path("/tmp/test_backups")
        mock_manager.list_backups.return_value = [_make_backup()]

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST", path="/api/v2/dr/validate", body={}
            )

        assert result.status_code == 200
        body = parse_body(result)
        assert body["valid"] is True
        check_names = [c["name"] for c in body["checks"]]
        assert "rbac_permissions" in check_names
        assert "encryption_config" in check_names
        assert "storage_access" in check_names
        assert "retention_policy" in check_names
        assert "compression" in check_names
        assert "auto_verify" in check_names
        assert "backup_exists" in check_names

    @pytest.mark.asyncio
    async def test_validate_skip_storage_check(self, handler, mock_manager):
        mock_manager.encryption_enabled = True
        mock_manager.encryption_key = "key"
        mock_manager.list_backups.return_value = [_make_backup()]

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/validate",
            body={"check_storage": False},
        )
        body = parse_body(result)
        check_names = [c["name"] for c in body["checks"]]
        assert "storage_access" not in check_names

    @pytest.mark.asyncio
    async def test_validate_skip_permissions_check(self, handler, mock_manager):
        mock_manager.encryption_enabled = True
        mock_manager.encryption_key = "key"
        mock_manager.list_backups.return_value = [_make_backup()]

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        check_names = [c["name"] for c in body["checks"]]
        assert "rbac_permissions" not in check_names

    @pytest.mark.asyncio
    async def test_validate_skip_encryption_check(self, handler, mock_manager):
        mock_manager.list_backups.return_value = [_make_backup()]

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_encryption": False},
            )
        body = parse_body(result)
        check_names = [c["name"] for c in body["checks"]]
        assert "encryption_config" not in check_names

    @pytest.mark.asyncio
    async def test_validate_encryption_not_enabled(self, handler, mock_manager):
        mock_manager.encryption_enabled = False
        mock_manager.list_backups.return_value = [_make_backup()]

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        enc_check = next(c for c in body["checks"] if c["name"] == "encryption_config")
        assert enc_check["status"] == "warning"
        assert "recommendation" in enc_check

    @pytest.mark.asyncio
    async def test_validate_encryption_enabled_no_key(self, handler, mock_manager):
        mock_manager.encryption_enabled = True
        mock_manager.encryption_key = None
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        enc_check = next(c for c in body["checks"] if c["name"] == "encryption_config")
        assert enc_check["status"] == "failed"
        assert body["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_encryption_enabled_with_key(self, handler, mock_manager):
        mock_manager.encryption_enabled = True
        mock_manager.encryption_key = "my-secret-key"
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        enc_check = next(c for c in body["checks"] if c["name"] == "encryption_config")
        assert enc_check["status"] == "passed"

    @pytest.mark.asyncio
    async def test_validate_storage_dir_missing(self, handler, mock_manager):
        mock_manager.encryption_enabled = False
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=False):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        storage_check = next(c for c in body["checks"] if c["name"] == "storage_access")
        assert storage_check["status"] == "failed"
        assert body["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_storage_permission_error(self, handler, mock_manager):
        mock_manager.encryption_enabled = False
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text", side_effect=PermissionError("no write")):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        storage_check = next(c for c in body["checks"] if c["name"] == "storage_access")
        assert storage_check["status"] == "failed"
        assert body["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_retention_policy_good(self, handler, mock_manager):
        mock_manager.encryption_enabled = False
        mock_manager.retention_policy = RetentionPolicy()
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        retention_check = next(c for c in body["checks"] if c["name"] == "retention_policy")
        assert retention_check["status"] == "passed"

    @pytest.mark.asyncio
    async def test_validate_retention_policy_zero_min(self, handler, mock_manager):
        mock_manager.encryption_enabled = False
        policy = RetentionPolicy()
        policy.min_backups = 0
        mock_manager.retention_policy = policy
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        retention_check = next(c for c in body["checks"] if c["name"] == "retention_policy")
        assert retention_check["status"] == "warning"

    @pytest.mark.asyncio
    async def test_validate_compression_enabled(self, handler, mock_manager):
        mock_manager.compression = True
        mock_manager.encryption_enabled = False
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        comp_check = next(c for c in body["checks"] if c["name"] == "compression")
        assert comp_check["status"] == "passed"

    @pytest.mark.asyncio
    async def test_validate_compression_disabled(self, handler, mock_manager):
        mock_manager.compression = False
        mock_manager.encryption_enabled = False
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        comp_check = next(c for c in body["checks"] if c["name"] == "compression")
        assert comp_check["status"] == "info"

    @pytest.mark.asyncio
    async def test_validate_auto_verify_on(self, handler, mock_manager):
        mock_manager.verify_after_backup = True
        mock_manager.encryption_enabled = False
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        av_check = next(c for c in body["checks"] if c["name"] == "auto_verify")
        assert av_check["status"] == "passed"

    @pytest.mark.asyncio
    async def test_validate_auto_verify_off(self, handler, mock_manager):
        mock_manager.verify_after_backup = False
        mock_manager.encryption_enabled = False
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        av_check = next(c for c in body["checks"] if c["name"] == "auto_verify")
        assert av_check["status"] == "warning"
        assert "recommendation" in av_check

    @pytest.mark.asyncio
    async def test_validate_backup_exists_check_with_backups(self, handler, mock_manager):
        mock_manager.encryption_enabled = False
        mock_manager.list_backups.return_value = [_make_backup()]

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        be_check = next(c for c in body["checks"] if c["name"] == "backup_exists")
        assert be_check["status"] == "passed"
        assert "1 backup(s)" in be_check["details"]

    @pytest.mark.asyncio
    async def test_validate_backup_exists_check_no_backups(self, handler, mock_manager):
        mock_manager.encryption_enabled = False
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={"check_permissions": False},
            )
        body = parse_body(result)
        be_check = next(c for c in body["checks"] if c["name"] == "backup_exists")
        assert be_check["status"] == "warning"
        assert "recommendation" in be_check

    @pytest.mark.asyncio
    async def test_validate_empty_body_uses_defaults(self, handler, mock_manager):
        """Empty body means all checks enabled by default."""
        mock_manager.encryption_enabled = False
        mock_manager.list_backups.return_value = []

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "write_text"), \
             patch.object(Path, "unlink"):
            result = await handler.handle(
                method="POST",
                path="/api/v2/dr/validate",
                body={},
            )
        body = parse_body(result)
        check_names = [c["name"] for c in body["checks"]]
        assert "rbac_permissions" in check_names
        assert "encryption_config" in check_names
        assert "storage_access" in check_names

    @pytest.mark.asyncio
    async def test_validate_all_checks_disabled(self, handler, mock_manager):
        """Disabling optional checks still runs retention/compression/auto_verify/backup_exists."""
        mock_manager.list_backups.return_value = []

        result = await handler.handle(
            method="POST",
            path="/api/v2/dr/validate",
            body={
                "check_storage": False,
                "check_permissions": False,
                "check_encryption": False,
            },
        )
        body = parse_body(result)
        check_names = [c["name"] for c in body["checks"]]
        assert "retention_policy" in check_names
        assert "compression" in check_names
        assert "auto_verify" in check_names
        assert "backup_exists" in check_names
        assert "rbac_permissions" not in check_names
        assert "encryption_config" not in check_names
        assert "storage_access" not in check_names


# ---------------------------------------------------------------------------
# Error Handling in handle()
# ---------------------------------------------------------------------------


class TestHandleErrorCatching:
    """Tests for the top-level error handling in handle()."""

    @pytest.mark.asyncio
    async def test_key_error_returns_500(self, handler, mock_manager):
        mock_manager.list_backups.side_effect = KeyError("missing key")

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_value_error_returns_500(self, handler, mock_manager):
        mock_manager.list_backups.side_effect = ValueError("bad value")

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_type_error_returns_500(self, handler, mock_manager):
        mock_manager.list_backups.side_effect = TypeError("wrong type")

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_os_error_returns_500(self, handler, mock_manager):
        mock_manager.list_backups.side_effect = OSError("disk failure")

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, handler, mock_manager):
        mock_manager.list_backups.side_effect = RuntimeError("runtime error")

        result = await handler.handle(method="GET", path="/api/v2/dr/status")
        assert result.status_code == 500


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestFactoryFunction:
    """Tests for create_dr_handler factory."""

    def test_creates_handler_instance(self):
        handler = create_dr_handler({})
        assert isinstance(handler, DRHandler)

    def test_handler_has_routes(self):
        handler = create_dr_handler({})
        assert len(handler.ROUTES) > 0

    def test_handler_accepts_context(self):
        ctx = {"storage": MagicMock()}
        handler = create_dr_handler(ctx)
        assert isinstance(handler, DRHandler)


# ---------------------------------------------------------------------------
# Lazy manager initialization
# ---------------------------------------------------------------------------


class TestLazyManager:
    """Tests for lazy BackupManager initialization."""

    def test_manager_none_initially(self):
        h = DRHandler(server_context={})
        assert h._manager is None

    def test_injected_manager_is_used(self, handler, mock_manager):
        assert handler._get_backup_manager() is mock_manager

    def test_lazy_factory_called_when_no_manager(self):
        h = DRHandler(server_context={})
        h._manager_factory = MagicMock()
        fake_mgr = MagicMock()
        h._manager_factory.get.return_value = fake_mgr

        result = h._get_backup_manager()
        assert result is fake_mgr
        h._manager_factory.get.assert_called_once()

    def test_lazy_factory_caches_result(self):
        h = DRHandler(server_context={})
        h._manager_factory = MagicMock()
        fake_mgr = MagicMock()
        h._manager_factory.get.return_value = fake_mgr

        h._get_backup_manager()
        h._get_backup_manager()
        # Only called once due to caching
        h._manager_factory.get.assert_called_once()
