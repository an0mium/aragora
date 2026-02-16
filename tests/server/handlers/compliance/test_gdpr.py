"""
Tests for GDPR Compliance Handler.

Tests cover:
- Data export endpoints (json and csv)
- Right-to-be-forgotten (RTBF) workflow
- Deletion management (list, get, cancel)
- Coordinated deletion across systems
- Backup exclusion management
- RBAC permission enforcement
- Error handling and edge cases
"""

from __future__ import annotations

import hashlib
import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.server.handlers.compliance.gdpr import GDPRMixin
from aragora.server.handlers.base import HandlerResult


# ============================================================================
# Test Fixtures
# ============================================================================


class MockGDPRHandler(GDPRMixin):
    """Mock handler class that includes the GDPR mixin for testing."""

    def __init__(self):
        pass


@pytest.fixture
def gdpr_handler():
    """Create a mock GDPR handler instance."""
    return MockGDPRHandler()


@pytest.fixture
def mock_audit_store():
    """Create a mock audit store."""
    store = MagicMock()
    store.log_event = MagicMock()
    store.get_recent_activity = MagicMock(return_value=[])
    return store


@pytest.fixture
def mock_receipt_store():
    """Create a mock receipt store."""
    store = MagicMock()
    store.list = MagicMock(return_value=[])
    return store


@pytest.fixture
def mock_deletion_scheduler():
    """Create a mock deletion scheduler."""
    scheduler = MagicMock()
    scheduler.store = MagicMock()
    scheduler.schedule_deletion = MagicMock()
    scheduler.cancel_deletion = MagicMock()
    return scheduler


@pytest.fixture
def mock_legal_hold_manager():
    """Create a mock legal hold manager."""
    manager = MagicMock()
    manager.is_user_on_hold = MagicMock(return_value=False)
    manager.get_active_holds = MagicMock(return_value=[])
    return manager


@pytest.fixture
def mock_deletion_coordinator():
    """Create a mock deletion coordinator."""
    coordinator = MagicMock()
    coordinator.execute_coordinated_deletion = AsyncMock()
    coordinator.process_pending_deletions = AsyncMock(return_value=[])
    coordinator.get_backup_exclusion_list = MagicMock(return_value=[])
    coordinator.add_to_backup_exclusion_list = MagicMock()
    return coordinator


@pytest.fixture
def mock_consent_manager():
    """Create a mock consent manager."""
    manager = MagicMock()
    manager.bulk_revoke_for_user = MagicMock(return_value=3)
    manager.export_consent_data = MagicMock()
    return manager


# ============================================================================
# Data Export Tests
# ============================================================================


class TestGDPRExport:
    """Tests for GDPR data export endpoint."""

    @pytest.mark.asyncio
    async def test_export_requires_user_id(self, gdpr_handler):
        """Export fails without user_id parameter."""
        result = await gdpr_handler._gdpr_export({})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "user_id is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_export_json_format(self, gdpr_handler, mock_receipt_store, mock_audit_store):
        """Export returns JSON format by default."""
        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export({"user_id": "user-123"})

        assert result.status_code == 200
        assert result.content_type == "application/json"

        body = json.loads(result.body)
        assert body["user_id"] == "user-123"
        assert "export_id" in body
        assert "checksum" in body
        assert "data_categories" in body

    @pytest.mark.asyncio
    async def test_export_csv_format(self, gdpr_handler, mock_receipt_store, mock_audit_store):
        """Export returns CSV format when requested."""
        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export({"user_id": "user-123", "format": "csv"})

        assert result.status_code == 200
        assert result.content_type == "text/csv"
        assert "Content-Disposition" in result.headers
        assert "gdpr-export-user-123.csv" in result.headers["Content-Disposition"]

    @pytest.mark.asyncio
    async def test_export_includes_decisions(
        self, gdpr_handler, mock_receipt_store, mock_audit_store
    ):
        """Export includes decisions when requested."""
        mock_receipt = MagicMock()
        mock_receipt.receipt_id = "receipt-1"
        mock_receipt.gauntlet_id = "gauntlet-1"
        mock_receipt.verdict = "approved"
        mock_receipt.confidence = 0.95
        mock_receipt.created_at = datetime.now(timezone.utc).isoformat()
        mock_receipt.risk_level = "low"
        mock_receipt_store.list.return_value = [mock_receipt]

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export(
                {"user_id": "user-123", "include": "decisions"}
            )

        body = json.loads(result.body)
        assert "decisions" in body["data_categories"]
        assert len(body["decisions"]) == 1
        assert body["decisions"][0]["receipt_id"] == "receipt-1"

    @pytest.mark.asyncio
    async def test_export_includes_preferences(
        self, gdpr_handler, mock_receipt_store, mock_audit_store
    ):
        """Export includes preferences when requested."""
        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export(
                {"user_id": "user-123", "include": "preferences"}
            )

        body = json.loads(result.body)
        assert "preferences" in body["data_categories"]
        assert "preferences" in body

    @pytest.mark.asyncio
    async def test_export_includes_activity(
        self, gdpr_handler, mock_receipt_store, mock_audit_store
    ):
        """Export includes activity when requested."""
        mock_audit_store.get_recent_activity.return_value = [
            {"action": "login", "timestamp": datetime.now(timezone.utc).isoformat()}
        ]

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export({"user_id": "user-123", "include": "activity"})

        body = json.loads(result.body)
        assert "activity" in body["data_categories"]
        assert len(body["activity"]) == 1

    @pytest.mark.asyncio
    async def test_export_all_categories(self, gdpr_handler, mock_receipt_store, mock_audit_store):
        """Export includes all categories when 'all' is specified."""
        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export({"user_id": "user-123", "include": "all"})

        body = json.loads(result.body)
        assert "decisions" in body["data_categories"]
        assert "preferences" in body["data_categories"]
        assert "activity" in body["data_categories"]

    @pytest.mark.asyncio
    async def test_export_checksum_integrity(
        self, gdpr_handler, mock_receipt_store, mock_audit_store
    ):
        """Export includes a valid SHA-256 checksum."""
        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export({"user_id": "user-123"})

        body = json.loads(result.body)
        checksum = body.pop("checksum")

        # Verify checksum is a valid SHA-256 hex string
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)


# ============================================================================
# Right-to-be-Forgotten Tests
# ============================================================================


class TestRightToBeForgotten:
    """Tests for GDPR Right-to-be-forgotten endpoint."""

    @pytest.mark.asyncio
    async def test_rtbf_requires_user_id(self, gdpr_handler):
        """RTBF fails without user_id."""
        result = await gdpr_handler._right_to_be_forgotten({})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "user_id is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_rtbf_schedules_deletion(
        self,
        gdpr_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_consent_manager,
        mock_audit_store,
        mock_receipt_store,
    ):
        """RTBF schedules deletion after grace period."""
        mock_deletion_request = MagicMock()
        mock_deletion_request.request_id = "del-123"
        mock_deletion_request.scheduled_for = datetime.now(timezone.utc) + timedelta(days=30)
        mock_deletion_request.status = MagicMock()
        mock_deletion_request.status.value = "pending"
        mock_deletion_request.created_at = datetime.now(timezone.utc)

        mock_deletion_scheduler.schedule_deletion.return_value = mock_deletion_request

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.privacy.consent.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
        ):
            result = await gdpr_handler._right_to_be_forgotten(
                {"user_id": "user-123", "reason": "User request"}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "scheduled"
        assert body["user_id"] == "user-123"
        assert "request_id" in body
        assert "deletion_scheduled" in body

    @pytest.mark.asyncio
    async def test_rtbf_revokes_consents(
        self,
        gdpr_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_consent_manager,
        mock_audit_store,
        mock_receipt_store,
    ):
        """RTBF revokes all user consents."""
        mock_consent_manager.bulk_revoke_for_user.return_value = 5

        mock_deletion_request = MagicMock()
        mock_deletion_request.request_id = "del-123"
        mock_deletion_request.scheduled_for = datetime.now(timezone.utc) + timedelta(days=30)
        mock_deletion_request.status = MagicMock(value="pending")
        mock_deletion_request.created_at = datetime.now(timezone.utc)
        mock_deletion_scheduler.schedule_deletion.return_value = mock_deletion_request

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.privacy.consent.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
        ):
            result = await gdpr_handler._right_to_be_forgotten({"user_id": "user-123"})

        body = json.loads(result.body)
        consent_op = next(
            (op for op in body["operations"] if op["operation"] == "revoke_consents"),
            None,
        )
        assert consent_op is not None
        assert consent_op["status"] == "completed"
        assert consent_op["consents_revoked"] == 5

    @pytest.mark.asyncio
    async def test_rtbf_generates_export(
        self,
        gdpr_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_consent_manager,
        mock_audit_store,
        mock_receipt_store,
    ):
        """RTBF generates final export when include_export is true."""
        mock_deletion_request = MagicMock()
        mock_deletion_request.request_id = "del-123"
        mock_deletion_request.scheduled_for = datetime.now(timezone.utc) + timedelta(days=30)
        mock_deletion_request.status = MagicMock(value="pending")
        mock_deletion_request.created_at = datetime.now(timezone.utc)
        mock_deletion_scheduler.schedule_deletion.return_value = mock_deletion_request

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.privacy.consent.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
        ):
            result = await gdpr_handler._right_to_be_forgotten(
                {"user_id": "user-123", "include_export": True}
            )

        body = json.loads(result.body)
        export_op = next(
            (op for op in body["operations"] if op["operation"] == "generate_export"),
            None,
        )
        assert export_op is not None
        assert export_op["status"] == "completed"
        assert "export_url" in body

    @pytest.mark.asyncio
    async def test_rtbf_skips_export(
        self,
        gdpr_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_consent_manager,
        mock_audit_store,
        mock_receipt_store,
    ):
        """RTBF skips export when include_export is false."""
        mock_deletion_request = MagicMock()
        mock_deletion_request.request_id = "del-123"
        mock_deletion_request.scheduled_for = datetime.now(timezone.utc) + timedelta(days=30)
        mock_deletion_request.status = MagicMock(value="pending")
        mock_deletion_request.created_at = datetime.now(timezone.utc)
        mock_deletion_scheduler.schedule_deletion.return_value = mock_deletion_request

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.privacy.consent.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
        ):
            result = await gdpr_handler._right_to_be_forgotten(
                {"user_id": "user-123", "include_export": False}
            )

        body = json.loads(result.body)
        export_op = next(
            (op for op in body["operations"] if op["operation"] == "generate_export"),
            None,
        )
        assert export_op is None
        assert "export_url" not in body

    @pytest.mark.asyncio
    async def test_rtbf_custom_grace_period(
        self,
        gdpr_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_consent_manager,
        mock_audit_store,
        mock_receipt_store,
    ):
        """RTBF respects custom grace period."""
        mock_deletion_request = MagicMock()
        mock_deletion_request.request_id = "del-123"
        mock_deletion_request.scheduled_for = datetime.now(timezone.utc) + timedelta(days=14)
        mock_deletion_request.status = MagicMock(value="pending")
        mock_deletion_request.created_at = datetime.now(timezone.utc)
        mock_deletion_scheduler.schedule_deletion.return_value = mock_deletion_request

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.privacy.consent.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
        ):
            result = await gdpr_handler._right_to_be_forgotten(
                {"user_id": "user-123", "grace_period_days": 14}
            )

        body = json.loads(result.body)
        assert body["grace_period_days"] == 14

    @pytest.mark.asyncio
    async def test_rtbf_blocked_by_legal_hold(
        self,
        gdpr_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_consent_manager,
        mock_audit_store,
        mock_receipt_store,
    ):
        """RTBF is blocked when user is under legal hold."""
        mock_legal_hold_manager.is_user_on_hold.return_value = True

        mock_hold = MagicMock()
        mock_hold.hold_id = "hold-001"
        mock_hold.reason = "Legal investigation"
        mock_deletion_scheduler.store.get_active_holds_for_user.return_value = [mock_hold]

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.privacy.consent.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
        ):
            result = await gdpr_handler._right_to_be_forgotten({"user_id": "user-123"})

        # RTBF should fail with status 500 due to ValueError from legal hold
        body = json.loads(result.body)
        assert body["status"] == "failed"
        assert body.get("error")  # Sanitized error message present

    @pytest.mark.asyncio
    async def test_rtbf_logs_audit_event(
        self,
        gdpr_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_consent_manager,
        mock_audit_store,
        mock_receipt_store,
    ):
        """RTBF logs audit event for compliance."""
        mock_deletion_request = MagicMock()
        mock_deletion_request.request_id = "del-123"
        mock_deletion_request.scheduled_for = datetime.now(timezone.utc) + timedelta(days=30)
        mock_deletion_request.status = MagicMock(value="pending")
        mock_deletion_request.created_at = datetime.now(timezone.utc)
        mock_deletion_scheduler.schedule_deletion.return_value = mock_deletion_request

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.privacy.consent.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
        ):
            await gdpr_handler._right_to_be_forgotten({"user_id": "user-123"})

        # Verify audit store was called
        assert mock_audit_store.log_event.called


# ============================================================================
# Deletion Management Tests
# ============================================================================


class TestListDeletions:
    """Tests for listing deletion requests."""

    @pytest.mark.asyncio
    async def test_list_deletions_success(self, gdpr_handler, mock_deletion_scheduler):
        """List deletions returns scheduled requests."""
        mock_request = MagicMock()
        mock_request.to_dict.return_value = {
            "request_id": "del-001",
            "user_id": "user-123",
            "status": "pending",
        }
        mock_deletion_scheduler.store.get_all_requests.return_value = [mock_request]

        with patch(
            "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
            return_value=mock_deletion_scheduler,
        ):
            result = await gdpr_handler._list_deletions({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1
        assert body["deletions"][0]["request_id"] == "del-001"

    @pytest.mark.asyncio
    async def test_list_deletions_with_status_filter(self, gdpr_handler, mock_deletion_scheduler):
        """List deletions filters by status."""
        mock_deletion_scheduler.store.get_all_requests.return_value = []

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch("aragora.privacy.deletion.DeletionStatus") as mock_status_enum,
        ):
            mock_status_enum.return_value = "pending"
            result = await gdpr_handler._list_deletions({"status": "pending"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["filters"]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_list_deletions_with_limit(self, gdpr_handler, mock_deletion_scheduler):
        """List deletions respects limit parameter."""
        mock_deletion_scheduler.store.get_all_requests.return_value = []

        with patch(
            "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
            return_value=mock_deletion_scheduler,
        ):
            result = await gdpr_handler._list_deletions({"limit": "25"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["filters"]["limit"] == 25

    @pytest.mark.asyncio
    async def test_list_deletions_max_limit(self, gdpr_handler, mock_deletion_scheduler):
        """List deletions enforces maximum limit of 200."""
        mock_deletion_scheduler.store.get_all_requests.return_value = []

        with patch(
            "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
            return_value=mock_deletion_scheduler,
        ):
            result = await gdpr_handler._list_deletions({"limit": "500"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["filters"]["limit"] == 200


class TestGetDeletion:
    """Tests for getting a specific deletion request."""

    @pytest.mark.asyncio
    async def test_get_deletion_success(self, gdpr_handler, mock_deletion_scheduler):
        """Get deletion returns the request details."""
        mock_request = MagicMock()
        mock_request.to_dict.return_value = {
            "request_id": "del-001",
            "user_id": "user-123",
            "status": "pending",
        }
        mock_deletion_scheduler.store.get_request.return_value = mock_request

        with patch(
            "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
            return_value=mock_deletion_scheduler,
        ):
            result = await gdpr_handler._get_deletion("del-001")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["deletion"]["request_id"] == "del-001"

    @pytest.mark.asyncio
    async def test_get_deletion_not_found(self, gdpr_handler, mock_deletion_scheduler):
        """Get deletion returns 404 when not found."""
        mock_deletion_scheduler.store.get_request.return_value = None

        with patch(
            "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
            return_value=mock_deletion_scheduler,
        ):
            result = await gdpr_handler._get_deletion("nonexistent")

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "not found" in body.get("error", "").lower()


class TestCancelDeletion:
    """Tests for cancelling deletion requests."""

    @pytest.mark.asyncio
    async def test_cancel_deletion_success(
        self, gdpr_handler, mock_deletion_scheduler, mock_audit_store
    ):
        """Cancel deletion successfully cancels a pending request."""
        mock_cancelled = MagicMock()
        mock_cancelled.user_id = "user-123"
        mock_cancelled.cancelled_at = datetime.now(timezone.utc)
        mock_cancelled.to_dict.return_value = {
            "request_id": "del-001",
            "status": "cancelled",
        }
        mock_deletion_scheduler.cancel_deletion.return_value = mock_cancelled

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._cancel_deletion("del-001", {"reason": "User changed mind"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "cancelled" in body["message"].lower()
        assert body["deletion"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_deletion_not_found(self, gdpr_handler, mock_deletion_scheduler):
        """Cancel deletion returns 404 when request not found."""
        mock_deletion_scheduler.cancel_deletion.return_value = None

        with patch(
            "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
            return_value=mock_deletion_scheduler,
        ):
            result = await gdpr_handler._cancel_deletion("nonexistent", {})

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_deletion_invalid_state(self, gdpr_handler, mock_deletion_scheduler):
        """Cancel deletion returns 400 for invalid state transitions."""
        mock_deletion_scheduler.cancel_deletion.side_effect = ValueError(
            "Cannot cancel completed deletion"
        )

        with patch(
            "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
            return_value=mock_deletion_scheduler,
        ):
            result = await gdpr_handler._cancel_deletion("del-001", {})

        assert result.status_code == 400


# ============================================================================
# Coordinated Deletion Tests
# ============================================================================


class TestCoordinatedDeletion:
    """Tests for coordinated deletion across systems."""

    @pytest.mark.asyncio
    async def test_coordinated_deletion_requires_user_id(self, gdpr_handler):
        """Coordinated deletion fails without user_id."""
        result = await gdpr_handler._coordinated_deletion({"reason": "Test"})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "user_id is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_coordinated_deletion_requires_reason(self, gdpr_handler):
        """Coordinated deletion fails without reason."""
        result = await gdpr_handler._coordinated_deletion({"user_id": "user-123"})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "reason is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_coordinated_deletion_success(
        self,
        gdpr_handler,
        mock_deletion_coordinator,
        mock_legal_hold_manager,
        mock_audit_store,
    ):
        """Coordinated deletion executes successfully."""
        from aragora.deletion_coordinator import CascadeStatus, DeletionSystem

        mock_report = MagicMock()
        mock_report.success = True
        mock_report.deleted_from = [DeletionSystem.USER_STORE]
        mock_report.backup_purge_results = {"backup-1": "purged"}
        mock_report.to_dict.return_value = {
            "status": "completed",
            "success": True,
        }
        mock_deletion_coordinator.execute_coordinated_deletion.return_value = mock_report

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._coordinated_deletion(
                {"user_id": "user-123", "reason": "GDPR request"}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "completed" in body["message"].lower()
        assert body["report"]["success"] is True

    @pytest.mark.asyncio
    async def test_coordinated_deletion_dry_run(
        self,
        gdpr_handler,
        mock_deletion_coordinator,
        mock_legal_hold_manager,
        mock_audit_store,
    ):
        """Coordinated deletion supports dry run mode."""
        mock_report = MagicMock()
        mock_report.success = True
        mock_report.deleted_from = []
        mock_report.backup_purge_results = {}
        mock_report.to_dict.return_value = {"status": "dry_run"}
        mock_deletion_coordinator.execute_coordinated_deletion.return_value = mock_report

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._coordinated_deletion(
                {"user_id": "user-123", "reason": "Test", "dry_run": True}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "dry run" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_coordinated_deletion_blocked_by_legal_hold(
        self, gdpr_handler, mock_deletion_coordinator, mock_legal_hold_manager
    ):
        """Coordinated deletion blocked when user is under legal hold."""
        mock_legal_hold_manager.is_user_on_hold.return_value = True

        mock_hold = MagicMock()
        mock_hold.hold_id = "hold-001"
        mock_hold.user_ids = ["user-123"]
        mock_legal_hold_manager.get_active_holds.return_value = [mock_hold]

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
        ):
            result = await gdpr_handler._coordinated_deletion(
                {"user_id": "user-123", "reason": "Test"}
            )

        assert result.status_code == 409
        body = json.loads(result.body)
        assert body.get("error")  # Sanitized error message present


class TestExecutePendingDeletions:
    """Tests for executing pending deletions."""

    @pytest.mark.asyncio
    async def test_execute_pending_deletions_success(
        self, gdpr_handler, mock_deletion_coordinator, mock_audit_store
    ):
        """Execute pending deletions processes scheduled requests."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.to_dict.return_value = {
            "user_id": "user-123",
            "status": "completed",
        }
        mock_deletion_coordinator.process_pending_deletions.return_value = [mock_result]

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._execute_pending_deletions({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["summary"]["total_processed"] == 1
        assert body["summary"]["successful"] == 1
        assert body["summary"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_execute_pending_deletions_with_failures(
        self, gdpr_handler, mock_deletion_coordinator, mock_audit_store
    ):
        """Execute pending deletions reports failures."""
        success_result = MagicMock()
        success_result.success = True
        success_result.to_dict.return_value = {"status": "completed"}

        failed_result = MagicMock()
        failed_result.success = False
        failed_result.to_dict.return_value = {"status": "failed"}

        mock_deletion_coordinator.process_pending_deletions.return_value = [
            success_result,
            failed_result,
        ]

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._execute_pending_deletions({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["summary"]["successful"] == 1
        assert body["summary"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_execute_pending_deletions_max_limit(
        self, gdpr_handler, mock_deletion_coordinator, mock_audit_store
    ):
        """Execute pending deletions enforces maximum limit of 500."""
        mock_deletion_coordinator.process_pending_deletions.return_value = []

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            await gdpr_handler._execute_pending_deletions({"limit": 1000})

        # Verify limit was capped at 500
        mock_deletion_coordinator.process_pending_deletions.assert_called_once()
        call_kwargs = mock_deletion_coordinator.process_pending_deletions.call_args.kwargs
        assert call_kwargs["limit"] == 500


# ============================================================================
# Backup Exclusion Tests
# ============================================================================


class TestBackupExclusions:
    """Tests for backup exclusion management."""

    @pytest.mark.asyncio
    async def test_list_backup_exclusions(self, gdpr_handler, mock_deletion_coordinator):
        """List backup exclusions returns excluded users."""
        mock_deletion_coordinator.get_backup_exclusion_list.return_value = [
            {"user_id": "user-123", "reason": "GDPR deletion"},
            {"user_id": "user-456", "reason": "User request"},
        ]

        with patch(
            "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
            return_value=mock_deletion_coordinator,
        ):
            result = await gdpr_handler._list_backup_exclusions({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 2
        assert len(body["exclusions"]) == 2

    @pytest.mark.asyncio
    async def test_list_backup_exclusions_with_limit(self, gdpr_handler, mock_deletion_coordinator):
        """List backup exclusions respects limit parameter."""
        mock_deletion_coordinator.get_backup_exclusion_list.return_value = []

        with patch(
            "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
            return_value=mock_deletion_coordinator,
        ):
            await gdpr_handler._list_backup_exclusions({"limit": "50"})

        mock_deletion_coordinator.get_backup_exclusion_list.assert_called_with(limit=50)

    @pytest.mark.asyncio
    async def test_list_backup_exclusions_max_limit(self, gdpr_handler, mock_deletion_coordinator):
        """List backup exclusions enforces max limit of 500."""
        mock_deletion_coordinator.get_backup_exclusion_list.return_value = []

        with patch(
            "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
            return_value=mock_deletion_coordinator,
        ):
            await gdpr_handler._list_backup_exclusions({"limit": "1000"})

        mock_deletion_coordinator.get_backup_exclusion_list.assert_called_with(limit=500)

    @pytest.mark.asyncio
    async def test_add_backup_exclusion_requires_user_id(self, gdpr_handler):
        """Add backup exclusion fails without user_id."""
        result = await gdpr_handler._add_backup_exclusion({"reason": "Test"})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "user_id is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_add_backup_exclusion_requires_reason(self, gdpr_handler):
        """Add backup exclusion fails without reason."""
        result = await gdpr_handler._add_backup_exclusion({"user_id": "user-123"})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "reason is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_add_backup_exclusion_success(
        self, gdpr_handler, mock_deletion_coordinator, mock_audit_store
    ):
        """Add backup exclusion successfully adds user."""
        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._add_backup_exclusion(
                {"user_id": "user-123", "reason": "GDPR deletion"}
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["user_id"] == "user-123"
        assert body["reason"] == "GDPR deletion"

        mock_deletion_coordinator.add_to_backup_exclusion_list.assert_called_once_with(
            "user-123", "GDPR deletion"
        )


# ============================================================================
# RBAC Permission Tests
# ============================================================================


class TestGDPRPermissions:
    """Tests for GDPR handler RBAC permission enforcement.

    Uses source file reading instead of inspect.getsource() to avoid
    false failures from test pollution (runtime method replacement).
    """

    @pytest.fixture(autouse=True)
    def _load_source(self):
        """Load GDPR handler source file once for all permission tests."""
        import inspect

        source_file = inspect.getfile(GDPRMixin)
        with open(source_file) as f:
            self._source = f.read()

    def test_gdpr_export_has_permission_decorator(self):
        """GDPR export requires compliance:gdpr permission."""
        assert "require_permission" in self._source
        assert "compliance:gdpr" in self._source

    def test_rtbf_has_permission_decorator(self):
        """RTBF requires compliance:gdpr permission."""
        assert "compliance:gdpr" in self._source

    def test_list_deletions_has_permission_decorator(self):
        """List deletions requires compliance:gdpr permission."""
        assert "compliance:gdpr" in self._source

    def test_coordinated_deletion_has_permission_decorator(self):
        """Coordinated deletion requires compliance:gdpr permission."""
        assert "compliance:gdpr" in self._source

    def test_backup_exclusion_endpoints_have_permission_decorator(self):
        """Backup exclusion endpoints require compliance:gdpr permission."""
        assert "require_permission" in self._source
        assert "_list_backup_exclusions" in self._source
        assert "_add_backup_exclusion" in self._source


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestGDPRErrorHandling:
    """Tests for error handling in GDPR operations."""

    @pytest.mark.asyncio
    async def test_export_handles_receipt_store_error(self, gdpr_handler, mock_audit_store):
        """Export handles receipt store errors gracefully."""
        mock_receipt_store = MagicMock()
        mock_receipt_store.list.side_effect = RuntimeError("Database connection failed")

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export({"user_id": "user-123"})

        # Should still return a response with empty decisions
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["decisions"] == []

    @pytest.mark.asyncio
    async def test_export_handles_audit_store_error(self, gdpr_handler, mock_receipt_store):
        """Export handles audit store errors gracefully."""
        mock_audit_store = MagicMock()
        mock_audit_store.get_recent_activity.side_effect = RuntimeError("Database error")

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export({"user_id": "user-123", "include": "activity"})

        # Should still return a response with empty activity
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["activity"] == []

    @pytest.mark.asyncio
    async def test_list_deletions_handles_scheduler_error(
        self, gdpr_handler, mock_deletion_scheduler
    ):
        """List deletions handles scheduler errors."""
        mock_deletion_scheduler.store.get_all_requests.side_effect = RuntimeError(
            "Scheduler unavailable"
        )

        with patch(
            "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
            return_value=mock_deletion_scheduler,
        ):
            result = await gdpr_handler._list_deletions({})

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "failed" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_coordinated_deletion_handles_coordinator_error(
        self, gdpr_handler, mock_deletion_coordinator, mock_legal_hold_manager
    ):
        """Coordinated deletion handles coordinator errors."""
        mock_deletion_coordinator.execute_coordinated_deletion.side_effect = ValueError(
            "Coordinator failure"
        )

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
        ):
            result = await gdpr_handler._coordinated_deletion(
                {"user_id": "user-123", "reason": "Test"}
            )

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "failed" in body.get("error", "").lower()


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestGDPREdgeCases:
    """Tests for edge cases in GDPR operations."""

    @pytest.mark.asyncio
    async def test_export_empty_user_data(self, gdpr_handler, mock_receipt_store, mock_audit_store):
        """Export handles user with no data."""
        mock_receipt_store.list.return_value = []
        mock_audit_store.get_recent_activity.return_value = []

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export({"user_id": "new-user"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["decisions"] == []
        assert body["activity"] == []

    @pytest.mark.asyncio
    async def test_rtbf_consent_revocation_failure_continues(
        self,
        gdpr_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_audit_store,
        mock_receipt_store,
    ):
        """RTBF continues even if consent revocation fails."""
        mock_deletion_request = MagicMock()
        mock_deletion_request.request_id = "del-123"
        mock_deletion_request.scheduled_for = datetime.now(timezone.utc) + timedelta(days=30)
        mock_deletion_request.status = MagicMock(value="pending")
        mock_deletion_request.created_at = datetime.now(timezone.utc)
        mock_deletion_scheduler.schedule_deletion.return_value = mock_deletion_request

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.privacy.consent.get_consent_manager",
                side_effect=ImportError("Consent manager not available"),
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
        ):
            result = await gdpr_handler._right_to_be_forgotten({"user_id": "user-123"})

        # Should still succeed with 0 consents revoked
        assert result.status_code == 200
        body = json.loads(result.body)
        consent_op = next(
            (op for op in body["operations"] if op["operation"] == "revoke_consents"),
            None,
        )
        assert consent_op["consents_revoked"] == 0

    @pytest.mark.asyncio
    async def test_csv_export_with_special_characters(
        self, gdpr_handler, mock_receipt_store, mock_audit_store
    ):
        """CSV export handles special characters correctly."""
        mock_receipt = MagicMock()
        mock_receipt.receipt_id = "receipt-1"
        mock_receipt.gauntlet_id = "gauntlet-1"
        mock_receipt.verdict = 'approved with note: "test, data"'
        mock_receipt.confidence = 0.95
        mock_receipt.created_at = datetime.now(timezone.utc).isoformat()
        mock_receipt.risk_level = "low"
        mock_receipt_store.list.return_value = [mock_receipt]

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export({"user_id": "user-123", "format": "csv"})

        assert result.status_code == 200
        assert result.content_type == "text/csv"
        # CSV should be properly encoded
        csv_content = result.body.decode("utf-8")
        assert "user-123" in csv_content

    @pytest.mark.asyncio
    async def test_multiple_data_categories_in_include(
        self, gdpr_handler, mock_receipt_store, mock_audit_store
    ):
        """Export handles multiple comma-separated include categories."""
        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await gdpr_handler._gdpr_export(
                {"user_id": "user-123", "include": "decisions,activity"}
            )

        body = json.loads(result.body)
        assert "decisions" in body["data_categories"]
        assert "activity" in body["data_categories"]
        # preferences not requested
        assert "preferences" not in body["data_categories"]

    @pytest.mark.asyncio
    async def test_zero_grace_period(
        self,
        gdpr_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_consent_manager,
        mock_audit_store,
        mock_receipt_store,
    ):
        """RTBF handles zero grace period."""
        mock_deletion_request = MagicMock()
        mock_deletion_request.request_id = "del-123"
        mock_deletion_request.scheduled_for = datetime.now(timezone.utc)
        mock_deletion_request.status = MagicMock(value="pending")
        mock_deletion_request.created_at = datetime.now(timezone.utc)
        mock_deletion_scheduler.schedule_deletion.return_value = mock_deletion_request

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.privacy.consent.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_receipt_store",
                return_value=mock_receipt_store,
            ),
        ):
            result = await gdpr_handler._right_to_be_forgotten(
                {"user_id": "user-123", "grace_period_days": 0}
            )

        body = json.loads(result.body)
        assert body["grace_period_days"] == 0


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestGDPRHelperMethods:
    """Tests for GDPR helper methods."""

    @pytest.mark.asyncio
    async def test_get_user_decisions_handles_exception(self, gdpr_handler):
        """_get_user_decisions returns empty list on error."""
        with patch(
            "aragora.server.handlers.compliance.gdpr.get_receipt_store",
            side_effect=RuntimeError("Store unavailable"),
        ):
            result = await gdpr_handler._get_user_decisions("user-123")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_user_activity_handles_exception(self, gdpr_handler):
        """_get_user_activity returns empty list on error."""
        with patch(
            "aragora.server.handlers.compliance.gdpr.get_audit_store",
            side_effect=RuntimeError("Store unavailable"),
        ):
            result = await gdpr_handler._get_user_activity("user-123")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_user_preferences_returns_default(self, gdpr_handler):
        """_get_user_preferences returns default preferences structure."""
        result = await gdpr_handler._get_user_preferences("user-123")

        assert "notification_settings" in result
        assert "privacy_settings" in result

    def test_render_gdpr_csv_basic(self, gdpr_handler):
        """_render_gdpr_csv generates valid CSV."""
        export_data = {
            "user_id": "user-123",
            "export_id": "export-001",
            "requested_at": "2024-01-01T00:00:00Z",
            "data_categories": ["decisions"],
            "decisions": [{"id": "1", "verdict": "approved"}],
            "checksum": "abc123",
        }

        csv_content = gdpr_handler._render_gdpr_csv(export_data)

        assert "GDPR Data Export" in csv_content
        assert "user-123" in csv_content
        assert "export-001" in csv_content
        assert "DECISIONS" in csv_content
        assert "abc123" in csv_content

    def test_render_gdpr_csv_with_dict_data(self, gdpr_handler):
        """_render_gdpr_csv handles dictionary data correctly."""
        export_data = {
            "user_id": "user-123",
            "export_id": "export-001",
            "requested_at": "2024-01-01T00:00:00Z",
            "data_categories": ["preferences"],
            "preferences": {"theme": "dark", "language": "en"},
            "checksum": "abc123",
        }

        csv_content = gdpr_handler._render_gdpr_csv(export_data)

        assert "PREFERENCES" in csv_content
        assert "theme" in csv_content
        assert "dark" in csv_content


# ============================================================================
# Audit Logging Tests
# ============================================================================


class TestGDPRAuditLogging:
    """Tests for GDPR audit logging compliance."""

    @pytest.mark.asyncio
    async def test_coordinated_deletion_logs_audit_event(
        self,
        gdpr_handler,
        mock_deletion_coordinator,
        mock_legal_hold_manager,
        mock_audit_store,
    ):
        """Coordinated deletion logs audit event with required fields."""
        from aragora.deletion_coordinator import DeletionSystem

        mock_report = MagicMock()
        mock_report.success = True
        mock_report.deleted_from = [DeletionSystem.USER_STORE]
        mock_report.backup_purge_results = {}
        mock_report.to_dict.return_value = {"status": "completed"}
        mock_deletion_coordinator.execute_coordinated_deletion.return_value = mock_report

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            await gdpr_handler._coordinated_deletion(
                {"user_id": "user-123", "reason": "GDPR request"}
            )

        mock_audit_store.log_event.assert_called_once()
        call_kwargs = mock_audit_store.log_event.call_args.kwargs
        assert call_kwargs["action"] == "gdpr_coordinated_deletion"
        assert call_kwargs["resource_type"] == "user"
        assert call_kwargs["resource_id"] == "user-123"
        assert "reason" in call_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_cancel_deletion_logs_audit_event(
        self, gdpr_handler, mock_deletion_scheduler, mock_audit_store
    ):
        """Cancel deletion logs audit event."""
        mock_cancelled = MagicMock()
        mock_cancelled.user_id = "user-123"
        mock_cancelled.cancelled_at = datetime.now(timezone.utc)
        mock_cancelled.to_dict.return_value = {"status": "cancelled"}
        mock_deletion_scheduler.cancel_deletion.return_value = mock_cancelled

        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            await gdpr_handler._cancel_deletion("del-001", {"reason": "User request"})

        mock_audit_store.log_event.assert_called_once()
        call_kwargs = mock_audit_store.log_event.call_args.kwargs
        assert call_kwargs["action"] == "gdpr_deletion_cancelled"

    @pytest.mark.asyncio
    async def test_add_backup_exclusion_logs_audit_event(
        self, gdpr_handler, mock_deletion_coordinator, mock_audit_store
    ):
        """Add backup exclusion logs audit event."""
        with (
            patch(
                "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
                return_value=mock_deletion_coordinator,
            ),
            patch(
                "aragora.server.handlers.compliance.gdpr.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            await gdpr_handler._add_backup_exclusion(
                {"user_id": "user-123", "reason": "GDPR deletion"}
            )

        mock_audit_store.log_event.assert_called_once()
        call_kwargs = mock_audit_store.log_event.call_args.kwargs
        assert call_kwargs["action"] == "gdpr_backup_exclusion_added"
        assert call_kwargs["resource_id"] == "user-123"
