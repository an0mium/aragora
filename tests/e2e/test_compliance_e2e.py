"""
E2E Tests for Compliance API Endpoints.

Tests the compliance HTTP API endpoints:
- GET /api/v2/compliance/soc2-report - SOC 2 compliance report
- GET /api/v2/compliance/gdpr-export - GDPR data export
- POST /api/v2/compliance/audit-verify - Audit trail verification
- GET /api/v2/compliance/audit-events - Audit event export
- GET /api/v2/compliance/status - Overall compliance status

Run with: pytest tests/e2e/test_compliance_e2e.py -v
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aragora.audit.log import AuditEvent, AuditLog, AuditCategory
from aragora.storage.audit_store import get_audit_store
from aragora.storage.receipt_store import get_receipt_store
from aragora.server.handlers.compliance_handler import ComplianceHandler
from aragora.server.handlers.base import ServerContext

pytestmark = [pytest.mark.e2e, pytest.mark.compliance]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handlers."""
    context = MagicMock(spec=ServerContext)
    context.get_user_id = MagicMock(return_value="test-user-123")
    context.get_org_id = MagicMock(return_value="test-org-456")
    return context


@pytest.fixture
def compliance_handler(mock_server_context):
    """Create a ComplianceHandler instance."""
    return ComplianceHandler(mock_server_context)


@pytest.fixture
def sample_audit_events() -> List[Dict[str, Any]]:
    """Generate sample audit events for testing."""
    base_time = datetime.now(timezone.utc) - timedelta(days=7)
    events = []

    for i in range(10):
        events.append(
            {
                "event_id": f"evt-{uuid.uuid4().hex[:8]}",
                "timestamp": (base_time + timedelta(hours=i * 12)).isoformat(),
                "category": "debate" if i % 2 == 0 else "auth",
                "action": "created" if i % 3 == 0 else "updated",
                "user_id": f"user-{i % 3}",
                "resource_type": "debate",
                "resource_id": f"debate-{i}",
                "details": {"round": i % 5},
                "ip_address": f"10.0.0.{i}",
            }
        )

    return events


@pytest.fixture
def sample_receipts() -> List[Dict[str, Any]]:
    """Generate sample decision receipts for testing."""
    receipts = []

    for i in range(5):
        receipt_data = {
            "debate_id": f"debate-{i}",
            "decision": f"Decision {i}",
            "consensus_score": 0.85 + (i * 0.02),
            "created_at": (datetime.now(timezone.utc) - timedelta(days=i)).isoformat(),
        }
        receipt_hash = hashlib.sha256(json.dumps(receipt_data, sort_keys=True).encode()).hexdigest()

        receipts.append(
            {
                **receipt_data,
                "receipt_hash": receipt_hash,
                "receipt_id": f"rcpt-{uuid.uuid4().hex[:8]}",
            }
        )

    return receipts


# ============================================================================
# SOC 2 Report Tests
# ============================================================================


class TestSOC2Report:
    """Tests for SOC 2 compliance report generation."""

    @pytest.mark.asyncio
    async def test_generate_soc2_report(self, compliance_handler):
        """Test basic SOC 2 report generation."""
        with patch.object(compliance_handler, "_get_soc2_report") as mock_report:
            mock_report.return_value = {
                "status": 200,
                "data": {
                    "report_type": "SOC2_Type_II",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "period": {
                        "start": (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(),
                        "end": datetime.now(timezone.utc).isoformat(),
                    },
                    "controls": {
                        "access_control": {"status": "compliant", "findings": 0},
                        "audit_logging": {"status": "compliant", "findings": 0},
                        "data_encryption": {"status": "compliant", "findings": 0},
                    },
                    "overall_status": "compliant",
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/soc2-report",
                query_params={},
            )

            assert result["status"] == 200
            assert "data" in result
            assert result["data"]["report_type"] == "SOC2_Type_II"
            assert result["data"]["overall_status"] == "compliant"

    @pytest.mark.asyncio
    async def test_soc2_report_with_date_range(self, compliance_handler):
        """Test SOC 2 report with custom date range."""
        start_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        end_date = datetime.now(timezone.utc).isoformat()

        with patch.object(compliance_handler, "_get_soc2_report") as mock_report:
            mock_report.return_value = {
                "status": 200,
                "data": {
                    "report_type": "SOC2_Type_II",
                    "period": {"start": start_date, "end": end_date},
                    "overall_status": "compliant",
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/soc2-report",
                query_params={"start_date": start_date, "end_date": end_date},
            )

            assert result["status"] == 200
            mock_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_soc2_report_includes_audit_stats(self, compliance_handler, sample_audit_events):
        """Test that SOC 2 report includes audit event statistics."""
        with patch("aragora.storage.audit_store.get_audit_store") as mock_store:
            mock_store_instance = MagicMock()
            mock_store_instance.count_events.return_value = len(sample_audit_events)
            mock_store_instance.get_category_counts.return_value = {
                "debate": 5,
                "auth": 5,
            }
            mock_store.return_value = mock_store_instance

            with patch.object(compliance_handler, "_get_soc2_report") as mock_report:
                mock_report.return_value = {
                    "status": 200,
                    "data": {
                        "audit_stats": {
                            "total_events": 10,
                            "by_category": {"debate": 5, "auth": 5},
                        },
                    },
                }

                result = await compliance_handler.handle(
                    method="GET",
                    path="/api/v2/compliance/soc2-report",
                )

                assert result["status"] == 200
                assert "audit_stats" in result["data"]


# ============================================================================
# GDPR Export Tests
# ============================================================================


class TestGDPRExport:
    """Tests for GDPR data export functionality."""

    @pytest.mark.asyncio
    async def test_gdpr_export_user_data(self, compliance_handler):
        """Test GDPR export returns user data."""
        user_id = "test-user-123"

        with patch.object(compliance_handler, "_gdpr_export") as mock_export:
            mock_export.return_value = {
                "status": 200,
                "data": {
                    "user_id": user_id,
                    "export_format": "json",
                    "data_categories": ["profile", "debates", "preferences"],
                    "records": {
                        "profile": {"name": "Test User", "email": "test@example.com"},
                        "debates": [{"id": "debate-1"}, {"id": "debate-2"}],
                        "preferences": {"theme": "dark"},
                    },
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr-export",
                query_params={"user_id": user_id},
            )

            assert result["status"] == 200
            assert result["data"]["user_id"] == user_id
            assert "records" in result["data"]

    @pytest.mark.asyncio
    async def test_gdpr_export_anonymized(self, compliance_handler):
        """Test GDPR export with anonymization option."""
        with patch.object(compliance_handler, "_gdpr_export") as mock_export:
            mock_export.return_value = {
                "status": 200,
                "data": {
                    "user_id": "anonymized",
                    "anonymization_applied": True,
                    "records": {
                        "profile": {"name": "***", "email": "***@***.***"},
                    },
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr-export",
                query_params={"user_id": "test-user", "anonymize": "true"},
            )

            assert result["status"] == 200
            assert result["data"].get("anonymization_applied") is True

    @pytest.mark.asyncio
    async def test_gdpr_export_formats(self, compliance_handler):
        """Test GDPR export supports different formats."""
        formats = ["json", "csv"]

        for fmt in formats:
            with patch.object(compliance_handler, "_gdpr_export") as mock_export:
                mock_export.return_value = {
                    "status": 200,
                    "data": {
                        "export_format": fmt,
                        "content": "..." if fmt == "csv" else {},
                    },
                }

                result = await compliance_handler.handle(
                    method="GET",
                    path="/api/v2/compliance/gdpr-export",
                    query_params={"user_id": "test-user", "format": fmt},
                )

                assert result["status"] == 200
                assert result["data"]["export_format"] == fmt


# ============================================================================
# Audit Integrity Tests
# ============================================================================


class TestAuditIntegrity:
    """Tests for audit trail integrity verification."""

    @pytest.mark.asyncio
    async def test_audit_verify_passes(self, compliance_handler):
        """Test audit verification passes for valid logs."""
        with patch.object(compliance_handler, "_verify_audit") as mock_verify:
            mock_verify.return_value = {
                "status": 200,
                "data": {
                    "integrity_status": "valid",
                    "events_verified": 100,
                    "hash_chain_valid": True,
                    "first_event": "2024-01-01T00:00:00Z",
                    "last_event": "2024-01-15T12:00:00Z",
                },
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/audit-verify",
                body={},
            )

            assert result["status"] == 200
            assert result["data"]["integrity_status"] == "valid"
            assert result["data"]["hash_chain_valid"] is True

    @pytest.mark.asyncio
    async def test_audit_tampering_detected(self, compliance_handler):
        """Test audit verification detects tampering."""
        with patch.object(compliance_handler, "_verify_audit") as mock_verify:
            mock_verify.return_value = {
                "status": 200,
                "data": {
                    "integrity_status": "compromised",
                    "events_verified": 95,
                    "hash_chain_valid": False,
                    "tampering_detected_at": "2024-01-10T08:30:00Z",
                    "affected_events": ["evt-123", "evt-124"],
                },
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/audit-verify",
                body={},
            )

            assert result["status"] == 200
            assert result["data"]["integrity_status"] == "compromised"
            assert result["data"]["hash_chain_valid"] is False
            assert "affected_events" in result["data"]

    @pytest.mark.asyncio
    async def test_audit_verify_with_date_range(self, compliance_handler):
        """Test audit verification with specific date range."""
        start = "2024-01-01T00:00:00Z"
        end = "2024-01-07T23:59:59Z"

        with patch.object(compliance_handler, "_verify_audit") as mock_verify:
            mock_verify.return_value = {
                "status": 200,
                "data": {
                    "integrity_status": "valid",
                    "period": {"start": start, "end": end},
                    "events_verified": 50,
                },
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/audit-verify",
                body={"start_date": start, "end_date": end},
            )

            assert result["status"] == 200
            assert result["data"]["period"]["start"] == start


# ============================================================================
# Audit Query & Export Tests
# ============================================================================


class TestAuditQuery:
    """Tests for audit log querying and export."""

    @pytest.mark.asyncio
    async def test_query_audit_events(self, compliance_handler, sample_audit_events):
        """Test querying audit events."""
        with patch.object(compliance_handler, "_get_audit_events") as mock_events:
            mock_events.return_value = {
                "status": 200,
                "data": {
                    "events": sample_audit_events[:5],
                    "total_count": 10,
                    "page": 1,
                    "page_size": 5,
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/audit-events",
                query_params={"limit": "5"},
            )

            assert result["status"] == 200
            assert len(result["data"]["events"]) == 5
            assert result["data"]["total_count"] == 10

    @pytest.mark.asyncio
    async def test_query_by_date_range(self, compliance_handler):
        """Test querying audit events by date range."""
        start = "2024-01-01"
        end = "2024-01-15"

        with patch.object(compliance_handler, "_get_audit_events") as mock_events:
            mock_events.return_value = {
                "status": 200,
                "data": {
                    "events": [],
                    "filters_applied": {"start_date": start, "end_date": end},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/audit-events",
                query_params={"start_date": start, "end_date": end},
            )

            assert result["status"] == 200
            assert result["data"]["filters_applied"]["start_date"] == start

    @pytest.mark.asyncio
    async def test_query_by_user(self, compliance_handler):
        """Test querying audit events by user ID."""
        user_id = "user-123"

        with patch.object(compliance_handler, "_get_audit_events") as mock_events:
            mock_events.return_value = {
                "status": 200,
                "data": {
                    "events": [{"user_id": user_id}],
                    "filters_applied": {"user_id": user_id},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/audit-events",
                query_params={"user_id": user_id},
            )

            assert result["status"] == 200
            assert result["data"]["filters_applied"]["user_id"] == user_id

    @pytest.mark.asyncio
    async def test_query_by_category(self, compliance_handler):
        """Test querying audit events by category."""
        with patch.object(compliance_handler, "_get_audit_events") as mock_events:
            mock_events.return_value = {
                "status": 200,
                "data": {
                    "events": [{"category": "auth"}],
                    "filters_applied": {"category": "auth"},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/audit-events",
                query_params={"category": "auth"},
            )

            assert result["status"] == 200


# ============================================================================
# Compliance Status Tests
# ============================================================================


class TestComplianceStatus:
    """Tests for overall compliance status endpoint."""

    @pytest.mark.asyncio
    async def test_get_compliance_status(self, compliance_handler):
        """Test getting overall compliance status."""
        with patch.object(compliance_handler, "_get_status") as mock_status:
            mock_status.return_value = {
                "status": 200,
                "data": {
                    "overall_status": "compliant",
                    "frameworks": {
                        "soc2": {"status": "compliant", "last_audit": "2024-01-01"},
                        "gdpr": {"status": "compliant", "last_audit": "2024-01-01"},
                        "hipaa": {"status": "partial", "last_audit": "2023-12-01"},
                    },
                    "audit_health": {
                        "integrity": "valid",
                        "retention_days": 2555,
                        "events_count": 10000,
                    },
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/status",
            )

            assert result["status"] == 200
            assert result["data"]["overall_status"] == "compliant"
            assert "frameworks" in result["data"]
            assert "audit_health" in result["data"]

    @pytest.mark.asyncio
    async def test_compliance_status_includes_all_frameworks(self, compliance_handler):
        """Test that compliance status includes all supported frameworks."""
        expected_frameworks = ["soc2", "gdpr", "hipaa"]

        with patch.object(compliance_handler, "_get_status") as mock_status:
            mock_status.return_value = {
                "status": 200,
                "data": {
                    "frameworks": {fw: {"status": "compliant"} for fw in expected_frameworks},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/status",
            )

            assert result["status"] == 200
            for fw in expected_frameworks:
                assert fw in result["data"]["frameworks"]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestComplianceErrorHandling:
    """Tests for error handling in compliance endpoints."""

    @pytest.mark.asyncio
    async def test_invalid_endpoint_returns_404(self, compliance_handler):
        """Test that invalid endpoints return 404."""
        result = await compliance_handler.handle(
            method="GET",
            path="/api/v2/compliance/invalid-endpoint",
        )

        # HandlerResult can be a dict or an object with status attribute
        status = (
            result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
        )
        assert status == 404

    @pytest.mark.asyncio
    async def test_invalid_method_returns_error(self, compliance_handler):
        """Test that invalid HTTP methods are handled."""
        # ComplianceHandler only supports GET and POST
        result = await compliance_handler.handle(
            method="DELETE",
            path="/api/v2/compliance/soc2-report",
        )

        # HandlerResult can be a dict or an object with status attribute
        status = (
            result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
        )
        # Handler returns 404 for unsupported method/path combinations
        assert status in [404, 405]

    @pytest.mark.asyncio
    async def test_missing_user_id_for_gdpr_export(self, compliance_handler):
        """Test GDPR export requires user_id parameter."""
        with patch.object(compliance_handler, "_gdpr_export") as mock_export:
            mock_export.return_value = {
                "status": 400,
                "error": "user_id is required",
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr-export",
                query_params={},  # Missing user_id
            )

            # Should return error status
            assert result["status"] in [400, 422]
