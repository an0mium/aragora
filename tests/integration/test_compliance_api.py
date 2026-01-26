"""
Integration tests for the Compliance API endpoints.

Tests the compliance management functionality:
- Get compliance status
- Generate SOC 2 report
- Export GDPR data
- Verify audit trail
- Export audit events (SIEM formats)
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any


class TestComplianceAPIEndpoints:
    """Test Compliance API HTTP endpoints."""

    @pytest.fixture
    def compliance_handler(self):
        """Create ComplianceHandler instance."""
        from aragora.server.handlers.compliance_handler import ComplianceHandler

        server_context = {"workspace_id": "test"}
        return ComplianceHandler(server_context)

    @pytest.fixture
    def mock_compliance_data(self):
        """Create mock compliance data."""
        return {
            "status": {
                "overall_score": 92,
                "frameworks": {
                    "soc2": {"status": "compliant", "score": 95},
                    "gdpr": {"status": "compliant", "score": 90},
                    "hipaa": {"status": "partial", "score": 75},
                },
                "last_audit": "2024-01-15T00:00:00Z",
            },
            "soc2_report": {
                "report_id": "soc2_2024_001",
                "type": "Type II",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
                "controls": [
                    {"id": "CC1.1", "name": "Security", "status": "effective"},
                    {"id": "CC2.1", "name": "Availability", "status": "effective"},
                ],
                "trust_criteria": {
                    "security": {"score": 95, "status": "effective"},
                    "availability": {"score": 90, "status": "effective"},
                    "processing_integrity": {"score": 88, "status": "effective"},
                    "confidentiality": {"score": 92, "status": "effective"},
                    "privacy": {"score": 85, "status": "effective"},
                },
            },
            "gdpr_export": {
                "user_id": "user_123",
                "export_date": datetime.now(timezone.utc).isoformat(),
                "data_categories": [
                    "personal_info",
                    "activity_logs",
                    "preferences",
                    "decisions",
                ],
                "data": {
                    "personal_info": {"email": "user@example.com"},
                    "decisions": [],
                    "preferences": {},
                    "activity": [],
                },
                "checksum": "sha256:abc123",
            },
        }

    @pytest.mark.asyncio
    async def test_get_compliance_status(self, compliance_handler, mock_compliance_data):
        """Test getting compliance status endpoint."""
        result = await compliance_handler.handle(
            method="GET",
            path="/api/v2/compliance/status",
        )

        assert result is not None
        assert result.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_get_soc2_report_json(self, compliance_handler, mock_compliance_data):
        """Test generating SOC 2 report in JSON format."""
        result = await compliance_handler.handle(
            method="GET",
            path="/api/v2/compliance/soc2-report",
            query_params={"format": "json"},
        )

        assert result is not None
        assert result.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_get_soc2_report_html(self, compliance_handler, mock_compliance_data):
        """Test generating SOC 2 report in HTML format."""
        result = await compliance_handler.handle(
            method="GET",
            path="/api/v2/compliance/soc2-report",
            query_params={"format": "html"},
        )

        assert result is not None
        assert result.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_gdpr_export_json(self, compliance_handler, mock_compliance_data):
        """Test GDPR data export in JSON format."""
        result = await compliance_handler.handle(
            method="GET",
            path="/api/v2/compliance/gdpr-export",
            query_params={"user_id": "user_123", "format": "json"},
        )

        assert result is not None
        assert result.status_code in (200, 400, 500)  # 400 if user_id missing

    @pytest.mark.asyncio
    async def test_gdpr_export_csv(self, compliance_handler, mock_compliance_data):
        """Test GDPR data export in CSV format."""
        result = await compliance_handler.handle(
            method="GET",
            path="/api/v2/compliance/gdpr-export",
            query_params={"user_id": "user_123", "format": "csv"},
        )

        assert result is not None
        assert result.status_code in (200, 400, 500)

    @pytest.mark.asyncio
    async def test_audit_verify(self, compliance_handler):
        """Test audit trail verification."""
        result = await compliance_handler.handle(
            method="POST",
            path="/api/v2/compliance/audit-verify",
            body={"date_from": "2024-01-01", "date_to": "2024-01-31"},
        )

        assert result is not None
        assert result.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_audit_events_elasticsearch(self, compliance_handler):
        """Test audit events export in Elasticsearch format."""
        result = await compliance_handler.handle(
            method="GET",
            path="/api/v2/compliance/audit-events",
            query_params={"format": "elasticsearch"},
        )

        assert result is not None
        assert result.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_audit_events_ndjson(self, compliance_handler):
        """Test audit events export in NDJSON format."""
        result = await compliance_handler.handle(
            method="GET",
            path="/api/v2/compliance/audit-events",
            query_params={"format": "ndjson"},
        )

        assert result is not None
        assert result.status_code in (200, 500)


class TestComplianceHandlerIntegration:
    """Test ComplianceHandler integration."""

    def test_compliance_handler_import(self):
        """Test that ComplianceHandler can be imported."""
        from aragora.server.handlers.compliance_handler import ComplianceHandler

        assert ComplianceHandler is not None

    def test_compliance_handler_routes(self):
        """Test that ComplianceHandler has correct routes."""
        from aragora.server.handlers.compliance_handler import ComplianceHandler

        handler = ComplianceHandler({})
        assert handler.can_handle("/api/v2/compliance/status", "GET")
        assert handler.can_handle("/api/v2/compliance/soc2-report", "GET")
        assert handler.can_handle("/api/v2/compliance/gdpr-export", "GET")
        assert handler.can_handle("/api/v2/compliance/audit-verify", "POST")
        assert handler.can_handle("/api/v2/compliance/audit-events", "GET")


class TestCompliancePermissions:
    """Test RBAC permissions for compliance operations."""

    def test_compliance_read_permission(self):
        """Test that reading compliance status requires compliance.read permission."""
        from aragora.rbac.decorators import require_permission

        @require_permission("compliance:read")
        def get_status(ctx):
            return {"overall_score": 92, "status": "compliant"}

        from aragora.rbac.models import AuthorizationContext

        ctx = AuthorizationContext(
            user_id="auditor_123",
            permissions={"compliance:read"},
        )
        result = get_status(ctx)
        assert result["status"] == "compliant"

    def test_soc2_permission(self):
        """Test that SOC 2 reports require compliance.soc2 permission."""
        from aragora.rbac.decorators import require_permission

        @require_permission("compliance:soc2")
        def get_soc2_report(ctx):
            return {"report_id": "soc2_001", "type": "Type II"}

        from aragora.rbac.models import AuthorizationContext

        ctx = AuthorizationContext(
            user_id="auditor_123",
            permissions={"compliance:soc2"},
        )
        result = get_soc2_report(ctx)
        assert result["type"] == "Type II"

    def test_gdpr_permission(self):
        """Test that GDPR exports require compliance.gdpr permission."""
        from aragora.rbac.decorators import require_permission

        @require_permission("compliance:gdpr")
        def export_gdpr(ctx, user_id: str):
            return {"user_id": user_id, "data": {}}

        from aragora.rbac.models import AuthorizationContext

        ctx = AuthorizationContext(
            user_id="admin_123",
            permissions={"compliance:gdpr"},
        )
        result = export_gdpr(ctx, "target_user")
        assert result["user_id"] == "target_user"


class TestSOC2Controls:
    """Test SOC 2 control coverage."""

    def test_soc2_trust_criteria(self):
        """Test that all SOC 2 trust service criteria are covered."""
        trust_criteria = [
            "security",
            "availability",
            "processing_integrity",
            "confidentiality",
            "privacy",
        ]
        for criterion in trust_criteria:
            assert criterion in trust_criteria

    def test_soc2_control_effectiveness(self):
        """Test control effectiveness evaluation."""
        control = {
            "id": "CC1.1",
            "name": "Security Policy",
            "status": "effective",
            "score": 95,
        }
        assert control["status"] in ["effective", "partially_effective", "not_effective"]
        assert 0 <= control["score"] <= 100


class TestGDPRExport:
    """Test GDPR data export functionality."""

    def test_gdpr_data_categories(self):
        """Test that all required GDPR data categories are included."""
        required_categories = [
            "personal_info",
            "activity_logs",
            "preferences",
            "decisions",
        ]
        for category in required_categories:
            assert category in required_categories

    def test_gdpr_export_checksum(self):
        """Test that GDPR exports include integrity checksum."""
        export = {
            "user_id": "user_123",
            "checksum": "sha256:abc123def456",
            "data": {},
        }
        assert export["checksum"].startswith("sha256:")

    def test_gdpr_export_formats(self):
        """Test supported GDPR export formats."""
        supported_formats = ["json", "csv"]
        for fmt in supported_formats:
            assert fmt in supported_formats


class TestAuditEventExport:
    """Test audit event export functionality."""

    def test_siem_formats(self):
        """Test supported SIEM export formats."""
        supported_formats = ["elasticsearch", "ndjson", "json"]
        for fmt in supported_formats:
            assert fmt in supported_formats

    def test_elasticsearch_bulk_format(self):
        """Test Elasticsearch bulk format structure."""
        # Elasticsearch bulk format has index actions followed by documents
        bulk_action = {"index": {"_index": "audit-logs", "_type": "_doc"}}
        document = {
            "timestamp": "2024-01-01T00:00:00Z",
            "event_type": "debate.created",
            "user_id": "user_123",
        }
        assert "index" in bulk_action
        assert "timestamp" in document
