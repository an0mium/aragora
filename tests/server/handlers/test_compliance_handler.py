"""
Tests for ComplianceHandler - Compliance HTTP endpoints.

Tests cover:
- Compliance status endpoint
- SOC 2 report generation
- GDPR data export
- Audit verification
- SIEM-compatible event export
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.compliance_handler import (
    ComplianceHandler,
    create_compliance_handler,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def handler(mock_server_context):
    """Create handler with mocked dependencies."""
    return ComplianceHandler(mock_server_context)


# ===========================================================================
# Handler Tests
# ===========================================================================


class TestComplianceHandlerRouting:
    """Test request routing."""

    def test_can_handle_compliance_paths(self, handler):
        """Test that handler recognizes compliance paths."""
        assert handler.can_handle("/api/v2/compliance/status", "GET")
        assert handler.can_handle("/api/v2/compliance/soc2-report", "GET")
        assert handler.can_handle("/api/v2/compliance/gdpr-export", "GET")
        assert handler.can_handle("/api/v2/compliance/audit-verify", "POST")
        assert handler.can_handle("/api/v2/compliance/audit-events", "GET")

    def test_cannot_handle_other_paths(self, handler):
        """Test that handler rejects non-compliance paths."""
        assert not handler.can_handle("/api/v2/backups", "GET")
        assert not handler.can_handle("/api/v1/compliance/status", "GET")


class TestComplianceStatus:
    """Test compliance status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status(self, handler):
        """Test getting compliance status."""
        result = await handler.handle("GET", "/api/v2/compliance/status")
        assert result.status_code == 200


class TestSOC2Report:
    """Test SOC 2 report endpoint."""

    @pytest.mark.asyncio
    async def test_get_soc2_report_json(self, handler):
        """Test getting SOC 2 report in JSON format."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/soc2-report",
            query_params={"format": "json"},
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_soc2_report_html(self, handler):
        """Test getting SOC 2 report in HTML format."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/soc2-report",
            query_params={"format": "html"},
        )
        assert result.status_code == 200
        assert result.content_type == "text/html"

    @pytest.mark.asyncio
    async def test_get_soc2_report_with_period(self, handler):
        """Test SOC 2 report with date range."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/soc2-report",
            query_params={
                "period_start": "2023-01-01T00:00:00Z",
                "period_end": "2023-12-31T23:59:59Z",
            },
        )
        assert result.status_code == 200


class TestGDPRExport:
    """Test GDPR export endpoint."""

    @pytest.mark.asyncio
    async def test_gdpr_export_requires_user_id(self, handler):
        """Test GDPR export requires user_id parameter."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/gdpr-export",
            query_params={},
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_gdpr_export_json(self, handler):
        """Test GDPR export in JSON format."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/gdpr-export",
            query_params={"user_id": "user-001", "format": "json"},
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_gdpr_export_csv(self, handler):
        """Test GDPR export in CSV format."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/gdpr-export",
            query_params={"user_id": "user-001", "format": "csv"},
        )
        assert result.status_code == 200
        assert result.content_type == "text/csv"

    @pytest.mark.asyncio
    async def test_gdpr_export_with_categories(self, handler):
        """Test GDPR export with specific data categories."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/gdpr-export",
            query_params={
                "user_id": "user-001",
                "include": "decisions,activity",
            },
        )
        assert result.status_code == 200


class TestAuditVerify:
    """Test audit verification endpoint."""

    @pytest.mark.asyncio
    async def test_verify_trail(self, handler):
        """Test verifying an audit trail."""
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/audit-verify",
            body={"trail_id": "trail-001"},
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_verify_receipts(self, handler):
        """Test verifying receipts."""
        with patch("aragora.storage.receipt_store.get_receipt_store") as mock_store:
            mock_store.return_value.verify_batch.return_value = ([], {"total": 0, "valid": 0})
            result = await handler.handle(
                "POST",
                "/api/v2/compliance/audit-verify",
                body={"receipt_ids": ["receipt-001", "receipt-002"]},
            )
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_verify_date_range(self, handler):
        """Test verifying events in date range."""
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/audit-verify",
            body={
                "date_range": {
                    "from": "2023-01-01T00:00:00Z",
                    "to": "2023-12-31T23:59:59Z",
                }
            },
        )
        assert result.status_code == 200


class TestAuditEvents:
    """Test audit events export endpoint."""

    @pytest.mark.asyncio
    async def test_get_events_json(self, handler):
        """Test getting audit events in JSON format."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/audit-events",
            query_params={"format": "json"},
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_events_elasticsearch(self, handler):
        """Test getting audit events in Elasticsearch format."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/audit-events",
            query_params={"format": "elasticsearch"},
        )
        assert result.status_code == 200
        assert result.content_type == "application/x-ndjson"

    @pytest.mark.asyncio
    async def test_get_events_ndjson(self, handler):
        """Test getting audit events in NDJSON format."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/audit-events",
            query_params={"format": "ndjson"},
        )
        assert result.status_code == 200
        assert result.content_type == "application/x-ndjson"

    @pytest.mark.asyncio
    async def test_get_events_with_filters(self, handler):
        """Test getting audit events with filters."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/audit-events",
            query_params={
                "from": "1700000000",
                "to": "1800000000",
                "limit": "100",
                "event_type": "gauntlet_start",
            },
        )
        assert result.status_code == 200


class TestGDPRDeletion:
    """Test GDPR deletion endpoints."""

    @pytest.mark.asyncio
    async def test_right_to_be_forgotten(self, handler):
        """Test GDPR right to be forgotten endpoint."""
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/right-to-be-forgotten",
            body={"user_id": "user-001", "reason": "User request"},
        )
        # May return 200 or 202 (accepted for async processing)
        assert result.status_code in (200, 202)

    @pytest.mark.asyncio
    async def test_right_to_be_forgotten_requires_user_id(self, handler):
        """Test GDPR deletion requires user_id."""
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/right-to-be-forgotten",
            body={},
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_list_deletions(self, handler):
        """Test listing GDPR deletion requests."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/gdpr/deletions",
            query_params={"status": "pending"},
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_deletion_status(self, handler):
        """Test getting deletion request status."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/gdpr/deletions/del-001",
        )
        # Returns 200 if found, 404 if not
        assert result.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_cancel_deletion(self, handler):
        """Test cancelling a deletion request."""
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/deletions/del-001/cancel",
            body={"reason": "Legal hold applied"},
        )
        # Returns 200 if cancelled, 404 if not found
        assert result.status_code in (200, 404)


class TestGDPRLegalHolds:
    """Test GDPR legal hold endpoints."""

    @pytest.mark.asyncio
    async def test_list_legal_holds(self, handler):
        """Test listing legal holds."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/gdpr/legal-holds",
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_create_legal_hold(self, handler):
        """Test creating a legal hold."""
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/legal-holds",
            body={
                "user_ids": ["user-001"],
                "reason": "Litigation pending",
                "case_id": "case-001",
            },
        )
        assert result.status_code in (200, 201)

    @pytest.mark.asyncio
    async def test_release_legal_hold(self, handler):
        """Test releasing a legal hold."""
        result = await handler.handle(
            "DELETE",
            "/api/v2/compliance/gdpr/legal-holds/hold-001",
        )
        # Returns 200/204 if released, 404 if not found
        assert result.status_code in (200, 204, 404)


class TestGDPRCoordinatedDeletion:
    """Test GDPR coordinated deletion endpoints."""

    @pytest.mark.asyncio
    async def test_coordinated_deletion(self, handler):
        """Test backup-aware coordinated deletion."""
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/coordinated-deletion",
            body={"user_id": "user-001", "reason": "User deletion request"},
        )
        assert result.status_code in (200, 202)

    @pytest.mark.asyncio
    async def test_execute_pending_deletions(self, handler):
        """Test executing pending deletions."""
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/execute-pending",
        )
        assert result.status_code in (200, 202)

    @pytest.mark.asyncio
    async def test_list_backup_exclusions(self, handler):
        """Test listing backup exclusions."""
        result = await handler.handle(
            "GET",
            "/api/v2/compliance/gdpr/backup-exclusions",
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_add_backup_exclusion(self, handler):
        """Test adding a backup exclusion."""
        result = await handler.handle(
            "POST",
            "/api/v2/compliance/gdpr/backup-exclusions",
            body={"user_id": "user-001", "reason": "GDPR deletion request"},
        )
        assert result.status_code in (200, 201)


class TestComplianceHandlerRouting_Extended:
    """Extended routing tests for GDPR endpoints."""

    def test_can_handle_gdpr_deletion_paths(self, handler):
        """Test handler recognizes GDPR deletion paths."""
        assert handler.can_handle("/api/v2/compliance/gdpr/right-to-be-forgotten", "POST")
        assert handler.can_handle("/api/v2/compliance/gdpr/deletions", "GET")
        assert handler.can_handle("/api/v2/compliance/gdpr/deletions/del-001", "GET")
        assert handler.can_handle("/api/v2/compliance/gdpr/deletions/del-001/cancel", "POST")

    def test_can_handle_legal_hold_paths(self, handler):
        """Test handler recognizes legal hold paths."""
        assert handler.can_handle("/api/v2/compliance/gdpr/legal-holds", "GET")
        assert handler.can_handle("/api/v2/compliance/gdpr/legal-holds", "POST")
        assert handler.can_handle("/api/v2/compliance/gdpr/legal-holds/hold-001", "DELETE")

    def test_can_handle_coordinated_deletion_paths(self, handler):
        """Test handler recognizes coordinated deletion paths."""
        assert handler.can_handle("/api/v2/compliance/gdpr/coordinated-deletion", "POST")
        assert handler.can_handle("/api/v2/compliance/gdpr/execute-pending", "POST")
        assert handler.can_handle("/api/v2/compliance/gdpr/backup-exclusions", "GET")
        assert handler.can_handle("/api/v2/compliance/gdpr/backup-exclusions", "POST")


class TestFactoryFunction:
    """Test handler factory function."""

    def test_create_compliance_handler(self, mock_server_context):
        """Test factory function creates handler."""
        handler = create_compliance_handler(mock_server_context)
        assert isinstance(handler, ComplianceHandler)
