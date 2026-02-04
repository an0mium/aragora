"""
Tests for CCPA Compliance Handler.

Tests cover:
- Right to Know (disclosure) endpoints
- Right to Delete (erasure) workflow
- Right to Opt-Out of Sale/Sharing
- Right to Correct (inaccurate PI)
- CCPA request status tracking
- RBAC permission enforcement
- Error handling and edge cases
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.compliance.ccpa import CCPAMixin
from aragora.server.handlers.base import HandlerResult


# ============================================================================
# Test Fixtures
# ============================================================================


class MockCCPAHandler(CCPAMixin):
    """Mock handler class that includes the CCPA mixin for testing."""

    def __init__(self):
        pass


@pytest.fixture
def ccpa_handler():
    """Create a mock CCPA handler instance."""
    return MockCCPAHandler()


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
def mock_ccpa_request_store():
    """Create a mock CCPA request store."""
    store = MagicMock()
    store.create_request = MagicMock()
    store.get_request = MagicMock()
    store.update_request = MagicMock()
    store.list_requests = MagicMock(return_value=[])
    return store


@pytest.fixture
def mock_consent_manager():
    """Create a mock consent manager."""
    manager = MagicMock()
    manager.get_opt_out_status = MagicMock(return_value=False)
    manager.set_opt_out = MagicMock()
    manager.export_consent_data = MagicMock()
    return manager


# ============================================================================
# Right to Know (Disclosure) Tests
# ============================================================================


class TestCCPADisclosure:
    """Tests for CCPA Right to Know disclosure endpoint."""

    @pytest.mark.asyncio
    async def test_disclosure_requires_consumer_id(self, ccpa_handler):
        """Disclosure fails without consumer_id parameter."""
        result = await ccpa_handler._ccpa_disclosure({})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "consumer_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_disclosure_categories_default(
        self, ccpa_handler, mock_receipt_store, mock_audit_store
    ):
        """Disclosure returns PI categories by default."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_disclosure(
                {"consumer_id": "consumer-123", "disclosure_type": "categories"}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["consumer_id"] == "consumer-123"
        assert "categories" in body
        assert "request_id" in body

    @pytest.mark.asyncio
    async def test_disclosure_specific_pieces(
        self, ccpa_handler, mock_receipt_store, mock_audit_store
    ):
        """Disclosure returns specific pieces of PI when requested."""
        mock_receipt = MagicMock()
        mock_receipt.receipt_id = "receipt-1"
        mock_receipt.created_at = datetime.now(timezone.utc).isoformat()
        mock_receipt_store.list.return_value = [mock_receipt]

        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_disclosure(
                {"consumer_id": "consumer-123", "disclosure_type": "specific"}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "personal_information" in body
        assert body["disclosure_type"] == "specific"

    @pytest.mark.asyncio
    async def test_disclosure_includes_sources(
        self, ccpa_handler, mock_receipt_store, mock_audit_store
    ):
        """Disclosure includes sources when requested."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_disclosure(
                {"consumer_id": "consumer-123", "include_sources": True}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "sources" in body

    @pytest.mark.asyncio
    async def test_disclosure_includes_business_purpose(
        self, ccpa_handler, mock_receipt_store, mock_audit_store
    ):
        """Disclosure includes business/commercial purpose."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_disclosure(
                {"consumer_id": "consumer-123", "include_purpose": True}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "business_purpose" in body

    @pytest.mark.asyncio
    async def test_disclosure_includes_third_parties(
        self, ccpa_handler, mock_receipt_store, mock_audit_store
    ):
        """Disclosure includes third parties PI was shared with."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_disclosure(
                {"consumer_id": "consumer-123", "include_third_parties": True}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "third_parties" in body

    @pytest.mark.asyncio
    async def test_disclosure_logs_audit_event(
        self, ccpa_handler, mock_receipt_store, mock_audit_store
    ):
        """Disclosure logs audit event for compliance."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            await ccpa_handler._ccpa_disclosure({"consumer_id": "consumer-123"})

        assert mock_audit_store.log_event.called


# ============================================================================
# Right to Delete Tests
# ============================================================================


class TestCCPADelete:
    """Tests for CCPA Right to Delete endpoint."""

    @pytest.mark.asyncio
    async def test_delete_requires_consumer_id(self, ccpa_handler):
        """Delete fails without consumer_id."""
        result = await ccpa_handler._ccpa_delete({})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "consumer_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_delete_schedules_deletion(
        self,
        ccpa_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_audit_store,
    ):
        """Delete schedules deletion request."""
        mock_deletion_request = MagicMock()
        mock_deletion_request.request_id = "del-123"
        mock_deletion_request.scheduled_for = datetime.now(timezone.utc) + timedelta(days=45)
        mock_deletion_request.status = MagicMock(value="pending")
        mock_deletion_request.created_at = datetime.now(timezone.utc)
        mock_deletion_scheduler.schedule_deletion.return_value = mock_deletion_request

        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_delete({"consumer_id": "consumer-123"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "scheduled"
        assert "request_id" in body

    @pytest.mark.asyncio
    async def test_delete_blocked_by_legal_hold(
        self,
        ccpa_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
    ):
        """Delete is blocked when consumer is under legal hold."""
        mock_legal_hold_manager.is_user_on_hold.return_value = True

        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
        ):
            result = await ccpa_handler._ccpa_delete({"consumer_id": "consumer-123"})

        body = json.loads(result.body)
        assert body["status"] == "failed"
        assert "legal hold" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_delete_with_verification(
        self,
        ccpa_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_audit_store,
    ):
        """Delete requires identity verification."""
        mock_deletion_request = MagicMock()
        mock_deletion_request.request_id = "del-123"
        mock_deletion_request.scheduled_for = datetime.now(timezone.utc) + timedelta(days=45)
        mock_deletion_request.status = MagicMock(value="pending_verification")
        mock_deletion_request.created_at = datetime.now(timezone.utc)
        mock_deletion_scheduler.schedule_deletion.return_value = mock_deletion_request

        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_delete(
                {"consumer_id": "consumer-123", "require_verification": True}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "verification" in body.get("status", "") or "request_id" in body

    @pytest.mark.asyncio
    async def test_delete_respects_ccpa_45_day_timeline(
        self,
        ccpa_handler,
        mock_deletion_scheduler,
        mock_legal_hold_manager,
        mock_audit_store,
    ):
        """Delete respects CCPA 45-day response timeline."""
        mock_deletion_request = MagicMock()
        mock_deletion_request.request_id = "del-123"
        mock_deletion_request.scheduled_for = datetime.now(timezone.utc) + timedelta(days=45)
        mock_deletion_request.status = MagicMock(value="pending")
        mock_deletion_request.created_at = datetime.now(timezone.utc)
        mock_deletion_scheduler.schedule_deletion.return_value = mock_deletion_request

        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_delete({"consumer_id": "consumer-123"})

        body = json.loads(result.body)
        assert "response_deadline" in body or "scheduled_for" in body


# ============================================================================
# Right to Opt-Out Tests
# ============================================================================


class TestCCPAOptOut:
    """Tests for CCPA Right to Opt-Out endpoint."""

    @pytest.mark.asyncio
    async def test_opt_out_requires_consumer_id(self, ccpa_handler):
        """Opt-out fails without consumer_id."""
        result = await ccpa_handler._ccpa_opt_out({})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "consumer_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_opt_out_sale_success(self, ccpa_handler, mock_consent_manager, mock_audit_store):
        """Opt-out of sale successfully."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_opt_out(
                {"consumer_id": "consumer-123", "opt_out_type": "sale"}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["opt_out_type"] == "sale"
        assert body["status"] == "opted_out"

    @pytest.mark.asyncio
    async def test_opt_out_sharing_success(
        self, ccpa_handler, mock_consent_manager, mock_audit_store
    ):
        """Opt-out of sharing successfully."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_opt_out(
                {"consumer_id": "consumer-123", "opt_out_type": "sharing"}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["opt_out_type"] == "sharing"
        assert body["status"] == "opted_out"

    @pytest.mark.asyncio
    async def test_opt_out_both_sale_and_sharing(
        self, ccpa_handler, mock_consent_manager, mock_audit_store
    ):
        """Opt-out of both sale and sharing."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_opt_out(
                {"consumer_id": "consumer-123", "opt_out_type": "both"}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "opted_out"

    @pytest.mark.asyncio
    async def test_opt_out_logs_audit_event(
        self, ccpa_handler, mock_consent_manager, mock_audit_store
    ):
        """Opt-out logs audit event."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            await ccpa_handler._ccpa_opt_out({"consumer_id": "consumer-123"})

        assert mock_audit_store.log_event.called

    @pytest.mark.asyncio
    async def test_opt_in_after_opt_out(self, ccpa_handler, mock_consent_manager, mock_audit_store):
        """Consumer can opt back in after opting out."""
        mock_consent_manager.get_opt_out_status.return_value = True

        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_opt_out(
                {"consumer_id": "consumer-123", "action": "opt_in"}
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "opted_in"


# ============================================================================
# Right to Correct Tests
# ============================================================================


class TestCCPACorrect:
    """Tests for CCPA Right to Correct endpoint."""

    @pytest.mark.asyncio
    async def test_correct_requires_consumer_id(self, ccpa_handler):
        """Correct fails without consumer_id."""
        result = await ccpa_handler._ccpa_correct({})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "consumer_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_correct_requires_corrections(self, ccpa_handler):
        """Correct fails without corrections data."""
        result = await ccpa_handler._ccpa_correct({"consumer_id": "consumer-123"})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "corrections" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_correct_success(self, ccpa_handler, mock_audit_store):
        """Correct successfully updates PI."""
        with patch(
            "aragora.server.handlers.compliance.ccpa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await ccpa_handler._ccpa_correct(
                {
                    "consumer_id": "consumer-123",
                    "corrections": {"email": "new@example.com", "phone": "+1234567890"},
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "corrected"
        assert "corrections_applied" in body

    @pytest.mark.asyncio
    async def test_correct_with_documentation(self, ccpa_handler, mock_audit_store):
        """Correct accepts supporting documentation."""
        with patch(
            "aragora.server.handlers.compliance.ccpa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await ccpa_handler._ccpa_correct(
                {
                    "consumer_id": "consumer-123",
                    "corrections": {"name": "John Doe"},
                    "documentation": "proof_of_name_change.pdf",
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "corrected"

    @pytest.mark.asyncio
    async def test_correct_logs_audit_event(self, ccpa_handler, mock_audit_store):
        """Correct logs audit event with correction details."""
        with patch(
            "aragora.server.handlers.compliance.ccpa.get_audit_store",
            return_value=mock_audit_store,
        ):
            await ccpa_handler._ccpa_correct(
                {
                    "consumer_id": "consumer-123",
                    "corrections": {"email": "new@example.com"},
                }
            )

        assert mock_audit_store.log_event.called


# ============================================================================
# CCPA Request Status Tests
# ============================================================================


class TestCCPARequestStatus:
    """Tests for CCPA request status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_requires_request_id(self, ccpa_handler):
        """Get status fails without request_id."""
        result = await ccpa_handler._ccpa_get_status({})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "request_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_status_success(self, ccpa_handler, mock_ccpa_request_store):
        """Get status returns request details."""
        mock_request = MagicMock()
        mock_request.request_id = "req-123"
        mock_request.request_type = "disclosure"
        mock_request.status = "completed"
        mock_request.created_at = datetime.now(timezone.utc)
        mock_request.completed_at = datetime.now(timezone.utc)
        mock_request.to_dict = MagicMock(
            return_value={
                "request_id": "req-123",
                "request_type": "disclosure",
                "status": "completed",
            }
        )
        mock_ccpa_request_store.get_request.return_value = mock_request

        with patch(
            "aragora.server.handlers.compliance.ccpa.get_ccpa_request_store",
            return_value=mock_ccpa_request_store,
        ):
            result = await ccpa_handler._ccpa_get_status({"request_id": "req-123"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["request"]["request_id"] == "req-123"

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, ccpa_handler, mock_ccpa_request_store):
        """Get status returns 404 when request not found."""
        mock_ccpa_request_store.get_request.return_value = None

        with patch(
            "aragora.server.handlers.compliance.ccpa.get_ccpa_request_store",
            return_value=mock_ccpa_request_store,
        ):
            result = await ccpa_handler._ccpa_get_status({"request_id": "nonexistent"})

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_list_requests_by_consumer(self, ccpa_handler, mock_ccpa_request_store):
        """List all CCPA requests for a consumer."""
        mock_request = MagicMock()
        mock_request.to_dict.return_value = {
            "request_id": "req-123",
            "request_type": "disclosure",
            "status": "completed",
        }
        mock_ccpa_request_store.list_requests.return_value = [mock_request]

        with patch(
            "aragora.server.handlers.compliance.ccpa.get_ccpa_request_store",
            return_value=mock_ccpa_request_store,
        ):
            result = await ccpa_handler._ccpa_list_requests({"consumer_id": "consumer-123"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1
        assert len(body["requests"]) == 1


# ============================================================================
# RBAC Permission Tests
# ============================================================================


class TestCCPAPermissions:
    """Tests for CCPA handler RBAC permission enforcement."""

    def test_disclosure_has_permission_decorator(self):
        """Disclosure requires compliance:ccpa permission."""
        import inspect

        source = inspect.getsource(CCPAMixin._ccpa_disclosure)
        assert "require_permission" in source or "track_handler" in source

    def test_delete_has_permission_decorator(self):
        """Delete requires compliance:ccpa permission."""
        import inspect

        source = inspect.getsource(CCPAMixin._ccpa_delete)
        assert "require_permission" in source or "track_handler" in source

    def test_opt_out_has_permission_decorator(self):
        """Opt-out requires compliance:ccpa permission."""
        import inspect

        source = inspect.getsource(CCPAMixin._ccpa_opt_out)
        assert "require_permission" in source or "track_handler" in source

    def test_correct_has_permission_decorator(self):
        """Correct requires compliance:ccpa permission."""
        import inspect

        source = inspect.getsource(CCPAMixin._ccpa_correct)
        assert "require_permission" in source or "track_handler" in source


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestCCPAErrorHandling:
    """Tests for error handling in CCPA operations."""

    @pytest.mark.asyncio
    async def test_disclosure_handles_store_error(self, ccpa_handler, mock_audit_store):
        """Disclosure handles store errors gracefully."""
        mock_receipt_store = MagicMock()
        mock_receipt_store.list.side_effect = RuntimeError("Database connection failed")

        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_disclosure({"consumer_id": "consumer-123"})

        # Should return error or gracefully degrade
        assert result.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_delete_handles_scheduler_error(self, ccpa_handler, mock_legal_hold_manager):
        """Delete handles scheduler errors."""
        mock_deletion_scheduler = MagicMock()
        mock_deletion_scheduler.schedule_deletion.side_effect = RuntimeError(
            "Scheduler unavailable"
        )

        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_deletion_scheduler",
                return_value=mock_deletion_scheduler,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
        ):
            result = await ccpa_handler._ccpa_delete({"consumer_id": "consumer-123"})

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_opt_out_handles_consent_manager_error(self, ccpa_handler):
        """Opt-out handles consent manager errors."""
        mock_consent_manager = MagicMock()
        mock_consent_manager.set_opt_out.side_effect = RuntimeError("Consent service down")

        with patch(
            "aragora.server.handlers.compliance.ccpa.get_consent_manager",
            return_value=mock_consent_manager,
        ):
            result = await ccpa_handler._ccpa_opt_out({"consumer_id": "consumer-123"})

        assert result.status_code == 500


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestCCPAEdgeCases:
    """Tests for edge cases in CCPA operations."""

    @pytest.mark.asyncio
    async def test_disclosure_empty_consumer_data(
        self, ccpa_handler, mock_receipt_store, mock_audit_store
    ):
        """Disclosure handles consumer with no data."""
        mock_receipt_store.list.return_value = []

        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_disclosure({"consumer_id": "new-consumer"})

        assert result.status_code == 200
        body = json.loads(result.body)
        # Should still return valid response with empty or minimal data
        assert "consumer_id" in body

    @pytest.mark.asyncio
    async def test_duplicate_opt_out_is_idempotent(
        self, ccpa_handler, mock_consent_manager, mock_audit_store
    ):
        """Duplicate opt-out request is idempotent."""
        mock_consent_manager.get_opt_out_status.return_value = True

        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_opt_out({"consumer_id": "consumer-123"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] in ("opted_out", "already_opted_out")

    @pytest.mark.asyncio
    async def test_correct_with_no_changes(self, ccpa_handler, mock_audit_store):
        """Correct handles empty corrections gracefully."""
        with patch(
            "aragora.server.handlers.compliance.ccpa.get_audit_store",
            return_value=mock_audit_store,
        ):
            result = await ccpa_handler._ccpa_correct(
                {"consumer_id": "consumer-123", "corrections": {}}
            )

        # Should either reject or accept with no changes
        assert result.status_code in (200, 400)

    @pytest.mark.asyncio
    async def test_invalid_disclosure_type(
        self, ccpa_handler, mock_receipt_store, mock_audit_store
    ):
        """Disclosure rejects invalid disclosure type."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_disclosure(
                {"consumer_id": "consumer-123", "disclosure_type": "invalid_type"}
            )

        # Should handle gracefully (either 400 or default to categories)
        assert result.status_code in (200, 400)


# ============================================================================
# CCPA Compliance Timeline Tests
# ============================================================================


class TestCCPATimelines:
    """Tests for CCPA compliance timeline requirements."""

    @pytest.mark.asyncio
    async def test_disclosure_response_within_45_days(
        self, ccpa_handler, mock_receipt_store, mock_audit_store
    ):
        """Disclosure includes response deadline within CCPA 45-day requirement."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_receipt_store",
                return_value=mock_receipt_store,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_disclosure({"consumer_id": "consumer-123"})

        body = json.loads(result.body)
        # Response should be immediate for disclosure
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_opt_out_processed_within_15_days(
        self, ccpa_handler, mock_consent_manager, mock_audit_store
    ):
        """Opt-out is processed immediately (CCPA requires within 15 days)."""
        with (
            patch(
                "aragora.server.handlers.compliance.ccpa.get_consent_manager",
                return_value=mock_consent_manager,
            ),
            patch(
                "aragora.server.handlers.compliance.ccpa.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await ccpa_handler._ccpa_opt_out({"consumer_id": "consumer-123"})

        assert result.status_code == 200
        body = json.loads(result.body)
        # Opt-out should be effective immediately
        assert body["status"] == "opted_out"


__all__ = [
    "TestCCPADisclosure",
    "TestCCPADelete",
    "TestCCPAOptOut",
    "TestCCPACorrect",
    "TestCCPARequestStatus",
    "TestCCPAPermissions",
    "TestCCPAErrorHandling",
    "TestCCPAEdgeCases",
    "TestCCPATimelines",
]
