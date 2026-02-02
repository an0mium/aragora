"""
Tests for SOC 2 Compliance Handler.

Tests cover:
- Compliance status endpoint
- SOC 2 Type II report generation (JSON and HTML)
- Trust service criteria assessment
- Control evaluation
- HTML rendering with XSS protection
- RBAC permission enforcement
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.compliance.handler import ComplianceHandler
from aragora.server.handlers.base import HandlerResult


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def compliance_handler():
    """Create a compliance handler instance."""
    return ComplianceHandler(server_context={})


# ============================================================================
# Compliance Status Tests
# ============================================================================


class TestComplianceStatus:
    """Tests for compliance status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_returns_compliant(self, compliance_handler):
        """Status returns compliant when all controls pass."""
        result = await compliance_handler._get_status()

        assert result.status_code == 200
        body = json.loads(result.body)

        # Check overall status
        assert body["status"] == "compliant"
        assert body["compliance_score"] == 100

        # Check frameworks section
        assert "frameworks" in body
        assert "soc2_type2" in body["frameworks"]
        assert "gdpr" in body["frameworks"]
        assert "hipaa" in body["frameworks"]

        # Check controls summary
        assert body["controls_summary"]["total"] == 12
        assert body["controls_summary"]["compliant"] == 12
        assert body["controls_summary"]["non_compliant"] == 0

    @pytest.mark.asyncio
    async def test_get_status_includes_timestamps(self, compliance_handler):
        """Status includes audit timestamps."""
        result = await compliance_handler._get_status()
        body = json.loads(result.body)

        assert "last_audit" in body
        assert "next_audit_due" in body
        assert "generated_at" in body

        # Verify timestamps are valid ISO format
        datetime.fromisoformat(body["last_audit"].replace("Z", "+00:00"))
        datetime.fromisoformat(body["next_audit_due"].replace("Z", "+00:00"))
        datetime.fromisoformat(body["generated_at"].replace("Z", "+00:00"))

    @pytest.mark.asyncio
    async def test_get_status_gdpr_framework(self, compliance_handler):
        """Status includes GDPR framework details."""
        result = await compliance_handler._get_status()
        body = json.loads(result.body)

        gdpr = body["frameworks"]["gdpr"]
        assert gdpr["status"] == "supported"
        assert gdpr["data_export"] is True
        assert gdpr["consent_tracking"] is True
        assert gdpr["retention_policy"] is True

    @pytest.mark.asyncio
    async def test_get_status_hipaa_framework(self, compliance_handler):
        """Status includes HIPAA framework details."""
        result = await compliance_handler._get_status()
        body = json.loads(result.body)

        hipaa = body["frameworks"]["hipaa"]
        assert hipaa["status"] == "partial"
        assert "note" in hipaa

    @pytest.mark.asyncio
    async def test_status_score_calculation_mostly_compliant(self, compliance_handler):
        """Score between 80-95 returns mostly_compliant."""
        # Mock evaluate_controls to return some non-compliant
        with patch.object(
            compliance_handler,
            "_evaluate_controls",
            new_callable=AsyncMock,
            return_value=[
                {"control_id": f"CC{i}", "status": "compliant" if i < 9 else "non_compliant"}
                for i in range(10)
            ],
        ):
            result = await compliance_handler._get_status()

        body = json.loads(result.body)
        assert body["status"] == "mostly_compliant"
        assert body["compliance_score"] == 90

    @pytest.mark.asyncio
    async def test_status_score_calculation_partial(self, compliance_handler):
        """Score between 60-80 returns partial."""
        with patch.object(
            compliance_handler,
            "_evaluate_controls",
            new_callable=AsyncMock,
            return_value=[
                {"control_id": f"CC{i}", "status": "compliant" if i < 7 else "non_compliant"}
                for i in range(10)
            ],
        ):
            result = await compliance_handler._get_status()

        body = json.loads(result.body)
        assert body["status"] == "partial"
        assert body["compliance_score"] == 70

    @pytest.mark.asyncio
    async def test_status_score_calculation_non_compliant(self, compliance_handler):
        """Score below 60 returns non_compliant."""
        with patch.object(
            compliance_handler,
            "_evaluate_controls",
            new_callable=AsyncMock,
            return_value=[
                {"control_id": f"CC{i}", "status": "compliant" if i < 5 else "non_compliant"}
                for i in range(10)
            ],
        ):
            result = await compliance_handler._get_status()

        body = json.loads(result.body)
        assert body["status"] == "non_compliant"
        assert body["compliance_score"] == 50


# ============================================================================
# SOC 2 Report Tests
# ============================================================================


class TestSOC2Report:
    """Tests for SOC 2 Type II report generation."""

    @pytest.mark.asyncio
    async def test_soc2_report_json_format(self, compliance_handler):
        """SOC 2 report returns JSON by default."""
        result = await compliance_handler._get_soc2_report({})

        assert result.status_code == 200
        assert result.content_type == "application/json"

        body = json.loads(result.body)
        assert body["report_type"] == "SOC 2 Type II"
        assert "report_id" in body
        assert body["report_id"].startswith("soc2-")

    @pytest.mark.asyncio
    async def test_soc2_report_html_format(self, compliance_handler):
        """SOC 2 report returns HTML when requested."""
        result = await compliance_handler._get_soc2_report({"format": "html"})

        assert result.status_code == 200
        assert result.content_type == "text/html"

        html_content = result.body.decode("utf-8")
        assert "<!DOCTYPE html>" in html_content
        assert "SOC 2 Type II" in html_content

    @pytest.mark.asyncio
    async def test_soc2_report_default_period(self, compliance_handler):
        """SOC 2 report defaults to last 90 days."""
        result = await compliance_handler._get_soc2_report({})
        body = json.loads(result.body)

        assert "period" in body
        assert body["period"]["days"] == 90

    @pytest.mark.asyncio
    async def test_soc2_report_custom_period(self, compliance_handler):
        """SOC 2 report respects custom period."""
        start = "2025-01-01T00:00:00Z"
        end = "2025-01-31T00:00:00Z"

        result = await compliance_handler._get_soc2_report(
            {"period_start": start, "period_end": end}
        )
        body = json.loads(result.body)

        assert body["period"]["days"] == 30

    @pytest.mark.asyncio
    async def test_soc2_report_includes_trust_criteria(self, compliance_handler):
        """SOC 2 report includes all trust service criteria."""
        result = await compliance_handler._get_soc2_report({})
        body = json.loads(result.body)

        criteria = body["trust_service_criteria"]
        assert "security" in criteria
        assert "availability" in criteria
        assert "processing_integrity" in criteria
        assert "confidentiality" in criteria
        assert "privacy" in criteria

    @pytest.mark.asyncio
    async def test_soc2_report_security_criteria(self, compliance_handler):
        """Security criteria assessment is complete."""
        result = await compliance_handler._get_soc2_report({})
        body = json.loads(result.body)

        security = body["trust_service_criteria"]["security"]
        assert security["status"] == "effective"
        assert security["controls_tested"] == 8
        assert security["controls_effective"] == 8
        assert len(security["key_findings"]) > 0

    @pytest.mark.asyncio
    async def test_soc2_report_availability_criteria(self, compliance_handler):
        """Availability criteria includes uptime target."""
        result = await compliance_handler._get_soc2_report({})
        body = json.loads(result.body)

        availability = body["trust_service_criteria"]["availability"]
        assert availability["uptime_target"] == "99.9%"

    @pytest.mark.asyncio
    async def test_soc2_report_includes_controls(self, compliance_handler):
        """SOC 2 report includes all evaluated controls."""
        result = await compliance_handler._get_soc2_report({})
        body = json.loads(result.body)

        assert "controls" in body
        assert len(body["controls"]) == 12

        # Check control structure
        control = body["controls"][0]
        assert "control_id" in control
        assert "category" in control
        assert "name" in control
        assert "description" in control
        assert "status" in control
        assert "evidence" in control

    @pytest.mark.asyncio
    async def test_soc2_report_summary(self, compliance_handler):
        """SOC 2 report includes summary statistics."""
        result = await compliance_handler._get_soc2_report({})
        body = json.loads(result.body)

        summary = body["summary"]
        assert summary["total_controls"] == 12
        assert summary["controls_tested"] == 12
        assert summary["controls_effective"] == 12
        assert summary["exceptions"] == 0

    @pytest.mark.asyncio
    async def test_soc2_report_organization_info(self, compliance_handler):
        """SOC 2 report includes organization info."""
        result = await compliance_handler._get_soc2_report({})
        body = json.loads(result.body)

        assert body["organization"] == "Aragora AI Decision Platform"
        assert "scope" in body


# ============================================================================
# Control Evaluation Tests
# ============================================================================


class TestControlEvaluation:
    """Tests for SOC 2 control evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_controls_returns_list(self, compliance_handler):
        """Control evaluation returns list of controls."""
        controls = await compliance_handler._evaluate_controls()

        assert isinstance(controls, list)
        assert len(controls) > 0

    @pytest.mark.asyncio
    async def test_control_categories(self, compliance_handler):
        """Controls cover all required categories."""
        controls = await compliance_handler._evaluate_controls()
        categories = {c["category"] for c in controls}

        assert "Security" in categories
        assert "Availability" in categories
        assert "Processing Integrity" in categories
        assert "Confidentiality" in categories
        assert "Privacy" in categories

    @pytest.mark.asyncio
    async def test_control_ids_are_standard(self, compliance_handler):
        """Control IDs follow SOC 2 standard format."""
        controls = await compliance_handler._evaluate_controls()

        # CC prefix for Common Criteria, P prefix for Privacy
        for control in controls:
            assert control["control_id"].startswith("CC") or control["control_id"].startswith("P")

    @pytest.mark.asyncio
    async def test_control_evidence_provided(self, compliance_handler):
        """Each control includes evidence."""
        controls = await compliance_handler._evaluate_controls()

        for control in controls:
            assert "evidence" in control
            assert isinstance(control["evidence"], list)
            assert len(control["evidence"]) > 0


# ============================================================================
# Trust Service Criteria Assessment Tests
# ============================================================================


class TestTrustServiceCriteria:
    """Tests for trust service criteria assessment."""

    @pytest.mark.asyncio
    async def test_security_criteria(self, compliance_handler):
        """Security criteria assessment is complete."""
        criteria = await compliance_handler._assess_security_criteria()

        assert criteria["status"] == "effective"
        assert criteria["controls_tested"] > 0
        assert criteria["controls_effective"] <= criteria["controls_tested"]
        assert "key_findings" in criteria

    @pytest.mark.asyncio
    async def test_availability_criteria(self, compliance_handler):
        """Availability criteria assessment is complete."""
        criteria = await compliance_handler._assess_availability_criteria()

        assert criteria["status"] == "effective"
        assert "uptime_target" in criteria
        assert "key_findings" in criteria

    @pytest.mark.asyncio
    async def test_integrity_criteria(self, compliance_handler):
        """Processing integrity criteria assessment is complete."""
        criteria = await compliance_handler._assess_integrity_criteria()

        assert criteria["status"] == "effective"
        assert "key_findings" in criteria

    @pytest.mark.asyncio
    async def test_confidentiality_criteria(self, compliance_handler):
        """Confidentiality criteria assessment is complete."""
        criteria = await compliance_handler._assess_confidentiality_criteria()

        assert criteria["status"] == "effective"
        assert "key_findings" in criteria

    @pytest.mark.asyncio
    async def test_privacy_criteria(self, compliance_handler):
        """Privacy criteria assessment is complete."""
        criteria = await compliance_handler._assess_privacy_criteria()

        assert criteria["status"] == "effective"
        assert "key_findings" in criteria


# ============================================================================
# HTML Rendering Tests
# ============================================================================


class TestSOC2HTMLRendering:
    """Tests for SOC 2 HTML report rendering."""

    def test_render_soc2_html_basic(self, compliance_handler):
        """HTML rendering produces valid HTML."""
        report = {
            "report_type": "SOC 2 Type II",
            "report_id": "soc2-20250202",
            "period": {"start": "2025-01-01", "end": "2025-02-01"},
            "organization": "Test Org",
            "summary": {
                "controls_tested": 10,
                "controls_effective": 9,
                "exceptions": 1,
            },
            "controls": [
                {
                    "control_id": "CC1.1",
                    "category": "Security",
                    "name": "Test Control",
                    "status": "compliant",
                }
            ],
            "generated_at": "2025-02-02T00:00:00Z",
        }

        html = compliance_handler._render_soc2_html(report)

        assert "<!DOCTYPE html>" in html
        assert "SOC 2 Type II" in html
        assert "soc2-20250202" in html
        assert "Test Org" in html

    def test_render_soc2_html_escapes_xss(self, compliance_handler):
        """HTML rendering escapes XSS attempts."""
        report = {
            "report_type": "<script>alert('xss')</script>",
            "report_id": "soc2-<img onerror=alert(1)>",
            "period": {"start": "2025-01-01", "end": "2025-02-01"},
            "organization": "<script>malicious</script>",
            "summary": {},
            "controls": [
                {
                    "control_id": "<script>bad</script>",
                    "category": "<img src=x>",
                    "name": "Test",
                    "status": "compliant",
                }
            ],
            "generated_at": "2025-02-02",
        }

        html = compliance_handler._render_soc2_html(report)

        # XSS should be escaped - script tags should be converted to HTML entities
        assert "<script>alert('xss')</script>" not in html
        assert "&lt;script&gt;" in html
        # The img tag should be escaped (< becomes &lt;)
        assert "<img onerror=" not in html
        assert "&lt;img onerror=" in html

    def test_render_soc2_html_control_status_classes(self, compliance_handler):
        """HTML uses correct CSS classes for control status."""
        report = {
            "report_type": "SOC 2 Type II",
            "report_id": "test",
            "period": {"start": "2025-01-01", "end": "2025-02-01"},
            "organization": "Test",
            "summary": {},
            "controls": [
                {"control_id": "CC1", "category": "Sec", "name": "A", "status": "compliant"},
                {"control_id": "CC2", "category": "Sec", "name": "B", "status": "non_compliant"},
            ],
            "generated_at": "2025-02-02",
        }

        html = compliance_handler._render_soc2_html(report)

        assert 'class="success"' in html
        assert 'class="warning"' in html


# ============================================================================
# RBAC Permission Tests
# ============================================================================


class TestSOC2Permissions:
    """Tests for SOC 2 handler RBAC permission enforcement."""

    def test_get_status_has_permission_decorator(self):
        """Status endpoint requires compliance:read permission."""
        import inspect

        source = inspect.getsource(ComplianceHandler._get_status)
        assert "require_permission" in source
        assert "compliance:read" in source

    def test_get_soc2_report_has_permission_decorator(self):
        """SOC 2 report requires compliance:soc2 permission."""
        import inspect

        source = inspect.getsource(ComplianceHandler._get_soc2_report)
        assert "require_permission" in source
        assert "compliance:soc2" in source


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestSOC2EdgeCases:
    """Tests for edge cases in SOC 2 operations."""

    @pytest.mark.asyncio
    async def test_report_with_empty_controls(self, compliance_handler):
        """Report handles empty controls list."""
        with patch.object(
            compliance_handler,
            "_evaluate_controls",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await compliance_handler._get_soc2_report({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["summary"]["total_controls"] == 0

    @pytest.mark.asyncio
    async def test_status_with_zero_controls(self, compliance_handler):
        """Status handles zero controls without division error."""
        with patch.object(
            compliance_handler,
            "_evaluate_controls",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await compliance_handler._get_status()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["compliance_score"] == 0

    @pytest.mark.asyncio
    async def test_report_invalid_period_format(self, compliance_handler):
        """Report handles invalid period format gracefully."""
        # This should raise ValueError for invalid date format
        with pytest.raises(ValueError):
            await compliance_handler._get_soc2_report(
                {"period_start": "not-a-date", "period_end": "2025-02-01"}
            )

    @pytest.mark.asyncio
    async def test_report_period_only_end(self, compliance_handler):
        """Report with only end date defaults start to 90 days prior."""
        result = await compliance_handler._get_soc2_report({"period_end": "2025-02-01T00:00:00Z"})
        body = json.loads(result.body)

        assert body["period"]["days"] == 90


# ============================================================================
# Handler Tracking Tests
# ============================================================================


class TestSOC2Tracking:
    """Tests for handler metrics tracking."""

    def test_status_has_track_handler_decorator(self):
        """Status endpoint has metrics tracking."""
        import inspect

        source = inspect.getsource(ComplianceHandler._get_status)
        assert "track_handler" in source
        assert "compliance/status" in source

    def test_soc2_report_has_track_handler_decorator(self):
        """SOC 2 report has metrics tracking."""
        import inspect

        source = inspect.getsource(ComplianceHandler._get_soc2_report)
        assert "track_handler" in source
        assert "compliance/soc2-report" in source
