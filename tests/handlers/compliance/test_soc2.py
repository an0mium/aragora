"""Comprehensive tests for the SOC2Mixin handler (aragora/server/handlers/compliance/soc2.py).

Covers all SOC 2 endpoints routed via ComplianceHandler:
- GET  /api/v2/compliance/status       - Overall compliance status
- GET  /api/v2/compliance/soc2-report  - SOC 2 Type II report (JSON/HTML)

Also covers internal helpers:
- _evaluate_controls
- _assess_security_criteria, _assess_availability_criteria
- _assess_integrity_criteria, _assess_confidentiality_criteria
- _assess_privacy_criteria
- _render_soc2_html
"""

from __future__ import annotations

import html as html_module
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.compliance.handler import ComplianceHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to ComplianceHandler.handle."""

    def __init__(self, method: str = "GET", body: dict[str, Any] | None = None):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a ComplianceHandler with minimal server context."""
    return ComplianceHandler({})


@pytest.fixture(autouse=True)
def _patch_external_deps(monkeypatch):
    """Patch external stores and schedulers that ComplianceHandler mixins import.

    Even though we only test SOC2 paths, the ComplianceHandler constructor
    eagerly imports all mixins, so we must satisfy their runtime dependencies.
    """
    mock_audit_store = MagicMock()
    mock_audit_store.get_log.return_value = []
    mock_audit_store.get_recent_activity.return_value = []
    mock_audit_store.log_event.return_value = None

    mock_receipt_store = MagicMock()
    mock_receipt_store.list.return_value = []
    mock_receipt_store.get.return_value = None
    mock_receipt_store.get_by_gauntlet.return_value = None
    mock_receipt_store.verify_batch.return_value = ([], {"total": 0, "valid": 0})

    mock_scheduler = MagicMock()
    mock_scheduler.store = MagicMock()
    mock_scheduler.store.get_all_requests.return_value = []
    mock_scheduler.store.get_request.return_value = None
    mock_scheduler.cancel_deletion.return_value = None

    mock_hold_manager = MagicMock()
    mock_hold_manager.is_user_on_hold.return_value = False
    mock_hold_manager.get_active_holds.return_value = []

    mock_coordinator = MagicMock()
    mock_coordinator.get_backup_exclusion_list.return_value = []

    # Patch all mixin module-level store getters
    for module in [
        "aragora.server.handlers.compliance.gdpr",
        "aragora.server.handlers.compliance.ccpa",
        "aragora.server.handlers.compliance.hipaa",
        "aragora.server.handlers.compliance.audit_verify",
        "aragora.server.handlers.compliance.legal_hold",
    ]:
        try:
            monkeypatch.setattr(f"{module}.get_audit_store", lambda: mock_audit_store)
        except AttributeError:
            pass
        try:
            monkeypatch.setattr(f"{module}.get_receipt_store", lambda: mock_receipt_store)
        except AttributeError:
            pass
        try:
            monkeypatch.setattr(f"{module}.get_deletion_scheduler", lambda: mock_scheduler)
        except AttributeError:
            pass
        try:
            monkeypatch.setattr(f"{module}.get_legal_hold_manager", lambda: mock_hold_manager)
        except AttributeError:
            pass
        try:
            monkeypatch.setattr(f"{module}.get_deletion_coordinator", lambda: mock_coordinator)
        except AttributeError:
            pass

    # Patch handler_events emit
    for module in [
        "aragora.server.handlers.compliance.handler",
        "aragora.server.handlers.compliance.gdpr",
        "aragora.server.handlers.compliance.ccpa",
        "aragora.server.handlers.compliance.hipaa",
    ]:
        try:
            monkeypatch.setattr(f"{module}.emit_handler_event", lambda *a, **kw: None)
        except AttributeError:
            pass


# ============================================================================
# GET /api/v2/compliance/status
# ============================================================================


class TestGetStatus:
    """Tests for GET /api/v2/compliance/status."""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, handler):
        """Status endpoint returns 200."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_status_contains_overall_status(self, handler):
        """Response includes the top-level status field."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert "status" in body
        assert body["status"] in ("compliant", "mostly_compliant", "partial", "non_compliant")

    @pytest.mark.asyncio
    async def test_status_contains_compliance_score(self, handler):
        """Response includes a numeric compliance_score."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert "compliance_score" in body
        assert isinstance(body["compliance_score"], int)
        assert 0 <= body["compliance_score"] <= 100

    @pytest.mark.asyncio
    async def test_status_all_controls_compliant_gives_100(self, handler):
        """All 12 controls are compliant, so score should be 100."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        # All default controls in _evaluate_controls are "compliant"
        assert body["compliance_score"] == 100
        assert body["status"] == "compliant"

    @pytest.mark.asyncio
    async def test_status_contains_frameworks(self, handler):
        """Response includes frameworks section with soc2, gdpr, hipaa."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert "frameworks" in body
        assert "soc2_type2" in body["frameworks"]
        assert "gdpr" in body["frameworks"]
        assert "hipaa" in body["frameworks"]

    @pytest.mark.asyncio
    async def test_status_soc2_framework_fields(self, handler):
        """SOC 2 framework entry includes controls_assessed and controls_compliant."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        soc2 = _body(result)["frameworks"]["soc2_type2"]
        assert soc2["status"] == "in_progress"
        assert "controls_assessed" in soc2
        assert "controls_compliant" in soc2
        assert soc2["controls_assessed"] == 12
        assert soc2["controls_compliant"] == 12

    @pytest.mark.asyncio
    async def test_status_gdpr_framework_fields(self, handler):
        """GDPR framework entry includes data_export, consent_tracking, retention_policy."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        gdpr = _body(result)["frameworks"]["gdpr"]
        assert gdpr["status"] == "supported"
        assert gdpr["data_export"] is True
        assert gdpr["consent_tracking"] is True
        assert gdpr["retention_policy"] is True

    @pytest.mark.asyncio
    async def test_status_hipaa_framework_fields(self, handler):
        """HIPAA framework entry includes status and note."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        hipaa = _body(result)["frameworks"]["hipaa"]
        assert hipaa["status"] == "partial"
        assert "note" in hipaa

    @pytest.mark.asyncio
    async def test_status_controls_summary(self, handler):
        """Response includes controls_summary with total, compliant, non_compliant."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        summary = body["controls_summary"]
        assert summary["total"] == 12
        assert summary["compliant"] == 12
        assert summary["non_compliant"] == 0

    @pytest.mark.asyncio
    async def test_status_contains_audit_dates(self, handler):
        """Response includes last_audit and next_audit_due as ISO strings."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert "last_audit" in body
        assert "next_audit_due" in body
        # Should parse without error
        datetime.fromisoformat(body["last_audit"])
        datetime.fromisoformat(body["next_audit_due"])

    @pytest.mark.asyncio
    async def test_status_last_audit_is_7_days_ago(self, handler):
        """last_audit should be approximately 7 days before generated_at."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        generated = datetime.fromisoformat(body["generated_at"])
        last_audit = datetime.fromisoformat(body["last_audit"])
        diff = generated - last_audit
        assert diff.days == 7

    @pytest.mark.asyncio
    async def test_status_next_audit_is_83_days_ahead(self, handler):
        """next_audit_due should be approximately 83 days after generated_at."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        generated = datetime.fromisoformat(body["generated_at"])
        next_audit = datetime.fromisoformat(body["next_audit_due"])
        diff = next_audit - generated
        assert diff.days == 83

    @pytest.mark.asyncio
    async def test_status_generated_at_is_recent(self, handler):
        """generated_at is within the last few seconds."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        generated = datetime.fromisoformat(body["generated_at"])
        now = datetime.now(timezone.utc)
        assert abs((now - generated).total_seconds()) < 5


# ============================================================================
# GET /api/v2/compliance/soc2-report  (JSON format)
# ============================================================================


class TestSOC2ReportJSON:
    """Tests for GET /api/v2/compliance/soc2-report (JSON output)."""

    @pytest.mark.asyncio
    async def test_returns_200(self, handler):
        """SOC 2 report returns 200."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_report_type(self, handler):
        """Report type is SOC 2 Type II."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        body = _body(result)
        assert body["report_type"] == "SOC 2 Type II"

    @pytest.mark.asyncio
    async def test_report_id_format(self, handler):
        """Report ID follows soc2-YYYYMMDD-HHMMSS format."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        body = _body(result)
        assert body["report_id"].startswith("soc2-")
        # Should contain date and time portions
        parts = body["report_id"].split("-")
        assert len(parts) == 3  # "soc2", "YYYYMMDD", "HHMMSS"

    @pytest.mark.asyncio
    async def test_default_period_90_days(self, handler):
        """Default period is last 90 days when no dates specified."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        body = _body(result)
        assert body["period"]["days"] == 90

    @pytest.mark.asyncio
    async def test_custom_period(self, handler):
        """Custom period_start and period_end are respected."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {
                "period_start": "2026-01-01T00:00:00+00:00",
                "period_end": "2026-01-31T00:00:00+00:00",
            },
            mock_h,
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["period"]["days"] == 30
        assert "2026-01-01" in body["period"]["start"]
        assert "2026-01-31" in body["period"]["end"]

    @pytest.mark.asyncio
    async def test_custom_period_start_only(self, handler):
        """Providing period_start only uses now as end."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"period_start": "2026-01-01T00:00:00+00:00"},
            mock_h,
        )
        body = _body(result)
        assert _status(result) == 200
        # Days should be from Jan 1 to ~now
        assert body["period"]["days"] > 0

    @pytest.mark.asyncio
    async def test_custom_period_end_only(self, handler):
        """Providing period_end only uses end - 90 days as start."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"period_end": "2026-06-01T00:00:00+00:00"},
            mock_h,
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["period"]["days"] == 90

    @pytest.mark.asyncio
    async def test_invalid_date_returns_400(self, handler):
        """Invalid date format returns 400."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"period_start": "not-a-date"},
            mock_h,
        )
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid date format" in body["error"]

    @pytest.mark.asyncio
    async def test_invalid_period_end_returns_400(self, handler):
        """Invalid period_end returns 400."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"period_end": "bad-date"},
            mock_h,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_z_suffix_date_format(self, handler):
        """Dates with Z suffix are parsed correctly."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {
                "period_start": "2026-01-01T00:00:00Z",
                "period_end": "2026-02-01T00:00:00Z",
            },
            mock_h,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["period"]["days"] == 31

    @pytest.mark.asyncio
    async def test_organization_field(self, handler):
        """Report includes organization name."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        body = _body(result)
        assert body["organization"] == "Aragora AI Decision Platform"

    @pytest.mark.asyncio
    async def test_scope_field(self, handler):
        """Report includes scope description."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        body = _body(result)
        assert "scope" in body
        assert "multi-agent" in body["scope"].lower()

    @pytest.mark.asyncio
    async def test_trust_service_criteria_present(self, handler):
        """Report includes all 5 trust service criteria."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        body = _body(result)
        tsc = body["trust_service_criteria"]
        assert "security" in tsc
        assert "availability" in tsc
        assert "processing_integrity" in tsc
        assert "confidentiality" in tsc
        assert "privacy" in tsc

    @pytest.mark.asyncio
    async def test_security_criteria_effective(self, handler):
        """Security trust service criteria status is effective."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        sec = _body(result)["trust_service_criteria"]["security"]
        assert sec["status"] == "effective"
        assert sec["controls_tested"] == 8
        assert sec["controls_effective"] == 8
        assert isinstance(sec["key_findings"], list)
        assert len(sec["key_findings"]) > 0

    @pytest.mark.asyncio
    async def test_availability_criteria_effective(self, handler):
        """Availability criteria is effective with uptime target."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        avail = _body(result)["trust_service_criteria"]["availability"]
        assert avail["status"] == "effective"
        assert avail["controls_tested"] == 4
        assert avail["uptime_target"] == "99.9%"

    @pytest.mark.asyncio
    async def test_integrity_criteria_effective(self, handler):
        """Processing integrity criteria is effective."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        integrity = _body(result)["trust_service_criteria"]["processing_integrity"]
        assert integrity["status"] == "effective"
        assert integrity["controls_tested"] == 3

    @pytest.mark.asyncio
    async def test_confidentiality_criteria_effective(self, handler):
        """Confidentiality criteria is effective."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        conf = _body(result)["trust_service_criteria"]["confidentiality"]
        assert conf["status"] == "effective"
        assert conf["controls_tested"] == 3

    @pytest.mark.asyncio
    async def test_privacy_criteria_effective(self, handler):
        """Privacy criteria is effective."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        priv = _body(result)["trust_service_criteria"]["privacy"]
        assert priv["status"] == "effective"
        assert priv["controls_tested"] == 4

    @pytest.mark.asyncio
    async def test_controls_list(self, handler):
        """Report includes full list of controls."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        body = _body(result)
        controls = body["controls"]
        assert isinstance(controls, list)
        assert len(controls) == 12

    @pytest.mark.asyncio
    async def test_control_structure(self, handler):
        """Each control has required fields."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        control = _body(result)["controls"][0]
        assert "control_id" in control
        assert "category" in control
        assert "name" in control
        assert "description" in control
        assert "status" in control
        assert "evidence" in control

    @pytest.mark.asyncio
    async def test_control_ids_are_expected(self, handler):
        """Control IDs match expected SOC 2 control identifiers."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        ids = [c["control_id"] for c in _body(result)["controls"]]
        expected = [
            "CC1.1", "CC2.1", "CC3.1", "CC5.1", "CC6.1", "CC6.6",
            "CC7.1", "CC7.2", "CC8.1", "CC9.1", "P1.1", "P4.1",
        ]
        assert ids == expected

    @pytest.mark.asyncio
    async def test_control_categories(self, handler):
        """Controls span Security, Availability, Processing Integrity, Confidentiality, Privacy."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        categories = {c["category"] for c in _body(result)["controls"]}
        assert "Security" in categories
        assert "Availability" in categories
        assert "Processing Integrity" in categories
        assert "Confidentiality" in categories
        assert "Privacy" in categories

    @pytest.mark.asyncio
    async def test_all_controls_compliant(self, handler):
        """All default controls have status compliant."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        for control in _body(result)["controls"]:
            assert control["status"] == "compliant"

    @pytest.mark.asyncio
    async def test_summary_section(self, handler):
        """Report summary has correct counts."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        summary = _body(result)["summary"]
        assert summary["total_controls"] == 12
        assert summary["controls_tested"] == 12
        assert summary["controls_effective"] == 12
        assert summary["exceptions"] == 0

    @pytest.mark.asyncio
    async def test_generated_at_iso_format(self, handler):
        """generated_at is a valid ISO 8601 timestamp."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        body = _body(result)
        datetime.fromisoformat(body["generated_at"])

    @pytest.mark.asyncio
    async def test_generated_at_is_recent(self, handler):
        """generated_at should be within the last few seconds."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        body = _body(result)
        generated = datetime.fromisoformat(body["generated_at"])
        now = datetime.now(timezone.utc)
        assert abs((now - generated).total_seconds()) < 5

    @pytest.mark.asyncio
    async def test_default_format_is_json(self, handler):
        """Without format param, response is JSON."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        assert result.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_explicit_json_format(self, handler):
        """format=json returns JSON."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "json"},
            mock_h,
        )
        assert _status(result) == 200
        assert result.content_type == "application/json"


# ============================================================================
# GET /api/v2/compliance/soc2-report  (HTML format)
# ============================================================================


class TestSOC2ReportHTML:
    """Tests for GET /api/v2/compliance/soc2-report?format=html."""

    @pytest.mark.asyncio
    async def test_html_returns_200(self, handler):
        """HTML format returns 200."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_html_content_type(self, handler):
        """HTML format returns text/html content type."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        assert result.content_type == "text/html"

    @pytest.mark.asyncio
    async def test_html_is_valid_document(self, handler):
        """HTML contains doctype and basic structure."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<html>" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "<body>" in content

    @pytest.mark.asyncio
    async def test_html_contains_report_title(self, handler):
        """HTML contains the report type as heading."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        assert "SOC 2 Type II" in content

    @pytest.mark.asyncio
    async def test_html_contains_report_id(self, handler):
        """HTML contains the report ID."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        assert "soc2-" in content

    @pytest.mark.asyncio
    async def test_html_contains_organization(self, handler):
        """HTML contains the organization name."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        assert "Aragora AI Decision Platform" in content

    @pytest.mark.asyncio
    async def test_html_contains_period(self, handler):
        """HTML contains the report period."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        assert "Period" in content

    @pytest.mark.asyncio
    async def test_html_contains_controls_table(self, handler):
        """HTML contains a table with control data."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        assert "<table>" in content
        assert "Control ID" in content
        assert "Category" in content
        assert "Name" in content
        assert "Status" in content

    @pytest.mark.asyncio
    async def test_html_contains_all_control_ids(self, handler):
        """HTML table contains all 12 control IDs."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        expected_ids = [
            "CC1.1", "CC2.1", "CC3.1", "CC5.1", "CC6.1", "CC6.6",
            "CC7.1", "CC7.2", "CC8.1", "CC9.1", "P1.1", "P4.1",
        ]
        for cid in expected_ids:
            assert cid in content

    @pytest.mark.asyncio
    async def test_html_contains_summary(self, handler):
        """HTML contains summary section with counts."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        assert "Summary" in content
        assert "Controls Tested" in content
        assert "Controls Effective" in content
        assert "Exceptions" in content

    @pytest.mark.asyncio
    async def test_html_contains_generated_timestamp(self, handler):
        """HTML contains generated timestamp."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        assert "Generated:" in content

    @pytest.mark.asyncio
    async def test_html_contains_css_styles(self, handler):
        """HTML contains embedded CSS styles."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        assert "<style>" in content
        assert ".success" in content
        assert ".warning" in content

    @pytest.mark.asyncio
    async def test_html_compliant_status_has_success_class(self, handler):
        """Compliant controls use the success CSS class."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        # All controls are compliant so all should have success class
        assert 'class="success"' in content

    @pytest.mark.asyncio
    async def test_html_with_custom_period(self, handler):
        """HTML format works with custom period."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {
                "format": "html",
                "period_start": "2026-01-01T00:00:00Z",
                "period_end": "2026-02-01T00:00:00Z",
            },
            mock_h,
        )
        assert _status(result) == 200
        content = result.body.decode("utf-8")
        assert "2026-01-01" in content
        assert "2026-02-01" in content

    @pytest.mark.asyncio
    async def test_html_invalid_date_still_returns_400(self, handler):
        """Invalid date with HTML format still returns 400 JSON error."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html", "period_start": "bad"},
            mock_h,
        )
        assert _status(result) == 400


# ============================================================================
# Internal helpers: _evaluate_controls
# ============================================================================


class TestEvaluateControls:
    """Tests for _evaluate_controls internal method."""

    @pytest.mark.asyncio
    async def test_returns_list(self, handler):
        """Returns a list of control dicts."""
        controls = await handler._evaluate_controls()
        assert isinstance(controls, list)
        assert len(controls) == 12

    @pytest.mark.asyncio
    async def test_each_control_has_required_keys(self, handler):
        """Each control dict has control_id, category, name, description, status, evidence."""
        controls = await handler._evaluate_controls()
        for ctrl in controls:
            assert "control_id" in ctrl
            assert "category" in ctrl
            assert "name" in ctrl
            assert "description" in ctrl
            assert "status" in ctrl
            assert "evidence" in ctrl

    @pytest.mark.asyncio
    async def test_evidence_is_list_of_strings(self, handler):
        """Evidence for each control is a list of strings."""
        controls = await handler._evaluate_controls()
        for ctrl in controls:
            assert isinstance(ctrl["evidence"], list)
            for item in ctrl["evidence"]:
                assert isinstance(item, str)

    @pytest.mark.asyncio
    async def test_all_statuses_compliant(self, handler):
        """All default controls have compliant status."""
        controls = await handler._evaluate_controls()
        for ctrl in controls:
            assert ctrl["status"] == "compliant"

    @pytest.mark.asyncio
    async def test_security_controls_count(self, handler):
        """There are 6 Security category controls."""
        controls = await handler._evaluate_controls()
        security = [c for c in controls if c["category"] == "Security"]
        assert len(security) == 6

    @pytest.mark.asyncio
    async def test_availability_controls_count(self, handler):
        """There are 2 Availability category controls."""
        controls = await handler._evaluate_controls()
        avail = [c for c in controls if c["category"] == "Availability"]
        assert len(avail) == 2

    @pytest.mark.asyncio
    async def test_processing_integrity_controls_count(self, handler):
        """There is 1 Processing Integrity category control."""
        controls = await handler._evaluate_controls()
        integrity = [c for c in controls if c["category"] == "Processing Integrity"]
        assert len(integrity) == 1

    @pytest.mark.asyncio
    async def test_confidentiality_controls_count(self, handler):
        """There is 1 Confidentiality category control."""
        controls = await handler._evaluate_controls()
        conf = [c for c in controls if c["category"] == "Confidentiality"]
        assert len(conf) == 1

    @pytest.mark.asyncio
    async def test_privacy_controls_count(self, handler):
        """There are 2 Privacy category controls."""
        controls = await handler._evaluate_controls()
        priv = [c for c in controls if c["category"] == "Privacy"]
        assert len(priv) == 2


# ============================================================================
# Internal helpers: _assess_*_criteria
# ============================================================================


class TestAssessSecurityCriteria:
    """Tests for _assess_security_criteria."""

    @pytest.mark.asyncio
    async def test_returns_effective(self, handler):
        result = await handler._assess_security_criteria()
        assert result["status"] == "effective"

    @pytest.mark.asyncio
    async def test_all_controls_effective(self, handler):
        result = await handler._assess_security_criteria()
        assert result["controls_tested"] == result["controls_effective"]

    @pytest.mark.asyncio
    async def test_has_key_findings(self, handler):
        result = await handler._assess_security_criteria()
        assert len(result["key_findings"]) > 0

    @pytest.mark.asyncio
    async def test_rbac_mentioned(self, handler):
        result = await handler._assess_security_criteria()
        findings = " ".join(result["key_findings"])
        assert "RBAC" in findings

    @pytest.mark.asyncio
    async def test_encryption_mentioned(self, handler):
        result = await handler._assess_security_criteria()
        findings = " ".join(result["key_findings"])
        assert "ncryption" in findings  # covers Encryption or encryption


class TestAssessAvailabilityCriteria:
    """Tests for _assess_availability_criteria."""

    @pytest.mark.asyncio
    async def test_returns_effective(self, handler):
        result = await handler._assess_availability_criteria()
        assert result["status"] == "effective"

    @pytest.mark.asyncio
    async def test_uptime_target_present(self, handler):
        result = await handler._assess_availability_criteria()
        assert result["uptime_target"] == "99.9%"

    @pytest.mark.asyncio
    async def test_all_controls_effective(self, handler):
        result = await handler._assess_availability_criteria()
        assert result["controls_tested"] == result["controls_effective"]

    @pytest.mark.asyncio
    async def test_has_key_findings(self, handler):
        result = await handler._assess_availability_criteria()
        assert len(result["key_findings"]) > 0


class TestAssessIntegrityCriteria:
    """Tests for _assess_integrity_criteria."""

    @pytest.mark.asyncio
    async def test_returns_effective(self, handler):
        result = await handler._assess_integrity_criteria()
        assert result["status"] == "effective"

    @pytest.mark.asyncio
    async def test_all_controls_effective(self, handler):
        result = await handler._assess_integrity_criteria()
        assert result["controls_tested"] == result["controls_effective"]

    @pytest.mark.asyncio
    async def test_decision_receipts_mentioned(self, handler):
        result = await handler._assess_integrity_criteria()
        findings = " ".join(result["key_findings"])
        assert "receipt" in findings.lower()


class TestAssessConfidentialityCriteria:
    """Tests for _assess_confidentiality_criteria."""

    @pytest.mark.asyncio
    async def test_returns_effective(self, handler):
        result = await handler._assess_confidentiality_criteria()
        assert result["status"] == "effective"

    @pytest.mark.asyncio
    async def test_all_controls_effective(self, handler):
        result = await handler._assess_confidentiality_criteria()
        assert result["controls_tested"] == result["controls_effective"]

    @pytest.mark.asyncio
    async def test_tenant_isolation_mentioned(self, handler):
        result = await handler._assess_confidentiality_criteria()
        findings = " ".join(result["key_findings"])
        assert "isolation" in findings.lower()


class TestAssessPrivacyCriteria:
    """Tests for _assess_privacy_criteria."""

    @pytest.mark.asyncio
    async def test_returns_effective(self, handler):
        result = await handler._assess_privacy_criteria()
        assert result["status"] == "effective"

    @pytest.mark.asyncio
    async def test_all_controls_effective(self, handler):
        result = await handler._assess_privacy_criteria()
        assert result["controls_tested"] == result["controls_effective"]

    @pytest.mark.asyncio
    async def test_gdpr_mentioned(self, handler):
        result = await handler._assess_privacy_criteria()
        findings = " ".join(result["key_findings"])
        assert "GDPR" in findings


# ============================================================================
# Internal helper: _render_soc2_html
# ============================================================================


class TestRenderSOC2HTML:
    """Tests for _render_soc2_html."""

    def _sample_report(self) -> dict[str, Any]:
        """Build a minimal report dict for testing."""
        return {
            "report_type": "SOC 2 Type II",
            "report_id": "soc2-20260101-120000",
            "organization": "Aragora AI Decision Platform",
            "period": {
                "start": "2025-10-03T00:00:00+00:00",
                "end": "2026-01-01T00:00:00+00:00",
            },
            "controls": [
                {
                    "control_id": "CC1.1",
                    "category": "Security",
                    "name": "COSO Principle 1",
                    "status": "compliant",
                },
                {
                    "control_id": "CC7.1",
                    "category": "Availability",
                    "name": "System Monitoring",
                    "status": "non_compliant",
                },
            ],
            "summary": {
                "controls_tested": 2,
                "controls_effective": 1,
                "exceptions": 1,
            },
            "generated_at": "2026-01-01T12:00:00+00:00",
        }

    def test_returns_string(self, handler):
        """Returns a string."""
        html = handler._render_soc2_html(self._sample_report())
        assert isinstance(html, str)

    def test_contains_doctype(self, handler):
        html = handler._render_soc2_html(self._sample_report())
        assert "<!DOCTYPE html>" in html

    def test_contains_report_id_in_title(self, handler):
        html = handler._render_soc2_html(self._sample_report())
        assert "soc2-20260101-120000" in html

    def test_contains_organization(self, handler):
        html = handler._render_soc2_html(self._sample_report())
        assert "Aragora AI Decision Platform" in html

    def test_contains_period_dates(self, handler):
        html = handler._render_soc2_html(self._sample_report())
        assert "2025-10-03" in html
        assert "2026-01-01" in html

    def test_compliant_row_has_success_class(self, handler):
        html = handler._render_soc2_html(self._sample_report())
        assert 'class="success"' in html

    def test_non_compliant_row_has_warning_class(self, handler):
        html = handler._render_soc2_html(self._sample_report())
        assert 'class="warning"' in html

    def test_summary_counts_rendered(self, handler):
        html = handler._render_soc2_html(self._sample_report())
        assert "Controls Tested:" in html
        assert "Controls Effective:" in html
        assert "Exceptions:" in html

    def test_generated_at_rendered(self, handler):
        html = handler._render_soc2_html(self._sample_report())
        assert "2026-01-01T12:00:00" in html

    def test_xss_prevention_control_id(self, handler):
        """Control ID with HTML is escaped."""
        report = self._sample_report()
        report["controls"][0]["control_id"] = '<script>alert("xss")</script>'
        html = handler._render_soc2_html(report)
        assert "<script>" not in html
        assert html_module.escape('<script>alert("xss")</script>') in html

    def test_xss_prevention_control_name(self, handler):
        """Control name with HTML is escaped."""
        report = self._sample_report()
        report["controls"][0]["name"] = '<img onerror="alert(1)" src=x>'
        html = handler._render_soc2_html(report)
        assert 'onerror="alert(1)"' not in html

    def test_xss_prevention_organization(self, handler):
        """Organization field with HTML is escaped."""
        report = self._sample_report()
        report["organization"] = '<b onmouseover="alert(1)">Evil Corp</b>'
        html = handler._render_soc2_html(report)
        assert 'onmouseover="alert(1)"' not in html

    def test_xss_prevention_report_id(self, handler):
        """Report ID with HTML is escaped."""
        report = self._sample_report()
        report["report_id"] = '"><script>alert(1)</script>'
        html = handler._render_soc2_html(report)
        assert "<script>alert(1)</script>" not in html

    def test_empty_controls_list(self, handler):
        """Report with no controls renders without error."""
        report = self._sample_report()
        report["controls"] = []
        html = handler._render_soc2_html(report)
        assert "<!DOCTYPE html>" in html
        assert "<table>" in html

    def test_missing_summary_keys(self, handler):
        """Report with empty summary still renders."""
        report = self._sample_report()
        report["summary"] = {}
        html = handler._render_soc2_html(report)
        assert "Controls Tested:" in html

    def test_missing_period(self, handler):
        """Report with empty period still renders."""
        report = self._sample_report()
        report["period"] = {}
        html = handler._render_soc2_html(report)
        assert "<!DOCTYPE html>" in html


# ============================================================================
# Compliance Score Thresholds
# ============================================================================


class TestComplianceScoreThresholds:
    """Test that different compliance scores map to correct statuses.

    Since _evaluate_controls returns hardcoded data, we test the logic by
    patching the method to return custom controls.
    """

    @pytest.mark.asyncio
    async def test_score_100_is_compliant(self, handler, monkeypatch):
        """Score >= 95 -> compliant."""
        async def mock_controls(self_inner):
            return [{"status": "compliant"}] * 10

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert body["status"] == "compliant"
        assert body["compliance_score"] == 100

    @pytest.mark.asyncio
    async def test_score_90_is_mostly_compliant(self, handler, monkeypatch):
        """Score 80-94 -> mostly_compliant."""
        async def mock_controls(self_inner):
            return [{"status": "compliant"}] * 9 + [{"status": "non_compliant"}]

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert body["status"] == "mostly_compliant"
        assert body["compliance_score"] == 90

    @pytest.mark.asyncio
    async def test_score_70_is_partial(self, handler, monkeypatch):
        """Score 60-79 -> partial."""
        async def mock_controls(self_inner):
            return [{"status": "compliant"}] * 7 + [{"status": "non_compliant"}] * 3

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert body["status"] == "partial"
        assert body["compliance_score"] == 70

    @pytest.mark.asyncio
    async def test_score_50_is_non_compliant(self, handler, monkeypatch):
        """Score < 60 -> non_compliant."""
        async def mock_controls(self_inner):
            return [{"status": "compliant"}] * 5 + [{"status": "non_compliant"}] * 5

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert body["status"] == "non_compliant"
        assert body["compliance_score"] == 50

    @pytest.mark.asyncio
    async def test_score_0_with_no_controls(self, handler, monkeypatch):
        """Score 0 when no controls exist."""
        async def mock_controls(self_inner):
            return []

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert body["compliance_score"] == 0
        assert body["status"] == "non_compliant"

    @pytest.mark.asyncio
    async def test_boundary_95_is_compliant(self, handler, monkeypatch):
        """Score exactly 95 -> compliant."""
        async def mock_controls(self_inner):
            return [{"status": "compliant"}] * 19 + [{"status": "non_compliant"}]

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert body["compliance_score"] == 95
        assert body["status"] == "compliant"

    @pytest.mark.asyncio
    async def test_boundary_80_is_mostly_compliant(self, handler, monkeypatch):
        """Score exactly 80 -> mostly_compliant."""
        async def mock_controls(self_inner):
            return [{"status": "compliant"}] * 8 + [{"status": "non_compliant"}] * 2

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert body["compliance_score"] == 80
        assert body["status"] == "mostly_compliant"

    @pytest.mark.asyncio
    async def test_boundary_60_is_partial(self, handler, monkeypatch):
        """Score exactly 60 -> partial."""
        async def mock_controls(self_inner):
            return [{"status": "compliant"}] * 6 + [{"status": "non_compliant"}] * 4

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert body["compliance_score"] == 60
        assert body["status"] == "partial"

    @pytest.mark.asyncio
    async def test_controls_summary_counts_non_compliant(self, handler, monkeypatch):
        """controls_summary correctly reflects non-compliant controls."""
        async def mock_controls(self_inner):
            return [{"status": "compliant"}] * 3 + [{"status": "non_compliant"}] * 2

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        body = _body(result)
        assert body["controls_summary"]["total"] == 5
        assert body["controls_summary"]["compliant"] == 3
        assert body["controls_summary"]["non_compliant"] == 2


# ============================================================================
# Route dispatch verification
# ============================================================================


class TestSOC2RouteDispatch:
    """Verify that SOC 2 routes dispatch correctly."""

    @pytest.mark.asyncio
    async def test_status_route(self, handler):
        """GET /api/v2/compliance/status dispatches correctly."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "compliance_score" in body

    @pytest.mark.asyncio
    async def test_soc2_report_route(self, handler):
        """GET /api/v2/compliance/soc2-report dispatches correctly."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["report_type"] == "SOC 2 Type II"

    @pytest.mark.asyncio
    async def test_post_to_status_returns_404(self, handler):
        """POST /api/v2/compliance/status returns 404."""
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle("/api/v2/compliance/status", {}, mock_h)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_to_soc2_report_returns_404(self, handler):
        """POST /api/v2/compliance/soc2-report returns 404."""
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_unknown_compliance_path_returns_404(self, handler):
        """Unknown compliance path returns 404."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/nonexistent", {}, mock_h)
        assert _status(result) == 404


# ============================================================================
# SOC 2 Report with custom controls (via monkeypatch)
# ============================================================================


class TestSOC2ReportWithCustomControls:
    """Test SOC 2 report generation with patched control data."""

    @pytest.mark.asyncio
    async def test_report_reflects_non_compliant_controls(self, handler, monkeypatch):
        """Non-compliant controls are reflected in the report summary."""
        async def mock_controls(self_inner):
            return [
                {
                    "control_id": "CC1.1",
                    "category": "Security",
                    "name": "Integrity",
                    "description": "Integrity values",
                    "status": "compliant",
                    "evidence": ["Evidence A"],
                },
                {
                    "control_id": "CC2.1",
                    "category": "Security",
                    "name": "Objectives",
                    "description": "Clear objectives",
                    "status": "non_compliant",
                    "evidence": ["Needs work"],
                },
            ]

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        body = _body(result)
        assert body["summary"]["total_controls"] == 2
        assert body["summary"]["controls_effective"] == 1
        assert body["summary"]["exceptions"] == 1

    @pytest.mark.asyncio
    async def test_html_report_shows_warning_for_non_compliant(self, handler, monkeypatch):
        """Non-compliant controls render with warning class in HTML."""
        async def mock_controls(self_inner):
            return [
                {
                    "control_id": "CC1.1",
                    "category": "Security",
                    "name": "Test",
                    "description": "Test",
                    "status": "non_compliant",
                    "evidence": [],
                },
            ]

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/soc2-report",
            {"format": "html"},
            mock_h,
        )
        content = result.body.decode("utf-8")
        assert 'class="warning"' in content
        assert "non_compliant" in content

    @pytest.mark.asyncio
    async def test_empty_controls_in_report(self, handler, monkeypatch):
        """Report with zero controls has correct summary."""
        async def mock_controls(self_inner):
            return []

        monkeypatch.setattr(type(handler), "_evaluate_controls", mock_controls)
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/soc2-report", {}, mock_h)
        body = _body(result)
        assert body["summary"]["total_controls"] == 0
        assert body["summary"]["controls_effective"] == 0
        assert body["summary"]["exceptions"] == 0
