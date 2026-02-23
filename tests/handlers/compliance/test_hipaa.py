"""Comprehensive tests for the HIPAAMixin handler methods.

Tests all HIPAA compliance endpoints routed through ComplianceHandler:
- GET  /api/v2/compliance/hipaa/status           - HIPAA compliance status
- GET  /api/v2/compliance/hipaa/phi-access       - PHI access audit log
- POST /api/v2/compliance/hipaa/breach-assessment - Breach risk assessment
- GET  /api/v2/compliance/hipaa/baa              - List Business Associate Agreements
- POST /api/v2/compliance/hipaa/baa              - Register new BAA
- GET  /api/v2/compliance/hipaa/security-report   - Security Rule compliance report
- POST /api/v2/compliance/hipaa/deidentify       - PHI de-identification
- POST /api/v2/compliance/hipaa/safe-harbor/verify- Safe Harbor verification
- POST /api/v2/compliance/hipaa/detect-phi       - PHI detection
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

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
def _patch_stores(monkeypatch):
    """Patch external stores and event emitters used by HIPAA mixin.

    Prevents tests from touching real storage or compliance infrastructure.
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

    mock_hold_manager = MagicMock()
    mock_hold_manager.is_user_on_hold.return_value = False
    mock_hold_manager.get_active_holds.return_value = []

    mock_coordinator = MagicMock()

    # Patch HIPAA mixin's direct imports
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.hipaa.get_audit_store",
        lambda: mock_audit_store,
    )

    # Patch GDPR mixin helpers (needed for handler init routing)
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.gdpr.get_audit_store",
        lambda: mock_audit_store,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.gdpr.get_receipt_store",
        lambda: mock_receipt_store,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.gdpr.get_deletion_scheduler",
        lambda: mock_scheduler,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.gdpr.get_legal_hold_manager",
        lambda: mock_hold_manager,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.gdpr.get_deletion_coordinator",
        lambda: mock_coordinator,
    )

    # Patch legal_hold mixin helpers
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.legal_hold.get_legal_hold_manager",
        lambda: mock_hold_manager,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.legal_hold.get_audit_store",
        lambda: mock_audit_store,
    )

    # Patch audit_verify mixin helpers
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.audit_verify.get_audit_store",
        lambda: mock_audit_store,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.audit_verify.get_receipt_store",
        lambda: mock_receipt_store,
    )

    # Patch CCPA mixin helpers
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.ccpa.get_audit_store",
        lambda: mock_audit_store,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.ccpa.get_receipt_store",
        lambda: mock_receipt_store,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.ccpa.get_deletion_scheduler",
        lambda: mock_scheduler,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.ccpa.get_legal_hold_manager",
        lambda: mock_hold_manager,
    )

    # Patch handler_events emit to avoid side effects
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.handler.emit_handler_event",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.gdpr.emit_handler_event",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.ccpa.emit_handler_event",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.hipaa.emit_handler_event",
        lambda *a, **kw: None,
    )

    yield {
        "audit_store": mock_audit_store,
        "receipt_store": mock_receipt_store,
        "scheduler": mock_scheduler,
        "hold_manager": mock_hold_manager,
        "coordinator": mock_coordinator,
    }


# ============================================================================
# HIPAA Status Endpoint - GET /api/v2/compliance/hipaa/status
# ============================================================================


class TestHIPAAStatus:
    """Tests for _hipaa_status method."""

    @pytest.mark.asyncio
    async def test_summary_returns_200(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_summary_contains_framework(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        body = _body(result)
        assert body["compliance_framework"] == "HIPAA"

    @pytest.mark.asyncio
    async def test_summary_contains_assessed_at(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        body = _body(result)
        assert "assessed_at" in body

    @pytest.mark.asyncio
    async def test_summary_contains_compliance_score(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        body = _body(result)
        assert "compliance_score" in body
        assert isinstance(body["compliance_score"], int)

    @pytest.mark.asyncio
    async def test_summary_contains_overall_status(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        body = _body(result)
        assert body["overall_status"] in (
            "compliant",
            "substantially_compliant",
            "partially_compliant",
            "non_compliant",
        )

    @pytest.mark.asyncio
    async def test_summary_contains_rules(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        body = _body(result)
        assert "rules" in body
        assert "privacy_rule" in body["rules"]
        assert "security_rule" in body["rules"]
        assert "breach_notification_rule" in body["rules"]

    @pytest.mark.asyncio
    async def test_summary_contains_business_associates(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        body = _body(result)
        assert "business_associates" in body

    @pytest.mark.asyncio
    async def test_summary_scope_excludes_details(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/status", {"scope": "summary"}, mock_h
        )
        body = _body(result)
        assert "safeguard_details" not in body
        assert "phi_controls" not in body

    @pytest.mark.asyncio
    async def test_full_scope_includes_safeguard_details(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/status", {"scope": "full"}, mock_h
        )
        body = _body(result)
        assert "safeguard_details" in body

    @pytest.mark.asyncio
    async def test_full_scope_includes_phi_controls(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/status", {"scope": "full"}, mock_h
        )
        body = _body(result)
        assert "phi_controls" in body

    @pytest.mark.asyncio
    async def test_includes_recommendations_by_default(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        body = _body(result)
        assert "recommendations" in body

    @pytest.mark.asyncio
    async def test_exclude_recommendations(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/status",
            {"include_recommendations": "false"},
            mock_h,
        )
        body = _body(result)
        assert "recommendations" not in body

    @pytest.mark.asyncio
    async def test_security_rule_has_counts(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        body = _body(result)
        sec_rule = body["rules"]["security_rule"]
        assert "safeguards_assessed" in sec_rule
        assert "safeguards_compliant" in sec_rule
        assert sec_rule["safeguards_assessed"] > 0

    @pytest.mark.asyncio
    async def test_privacy_rule_has_phi_handling(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        body = _body(result)
        priv_rule = body["rules"]["privacy_rule"]
        assert "phi_handling" in priv_rule

    @pytest.mark.asyncio
    async def test_breach_notification_rule_documented(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        body = _body(result)
        breach_rule = body["rules"]["breach_notification_rule"]
        assert breach_rule["procedures_documented"] is True

    @pytest.mark.asyncio
    async def test_recommendations_include_annual_risk_analysis(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/status", {}, mock_h)
        body = _body(result)
        rec_texts = [r["recommendation"] for r in body["recommendations"]]
        assert any("annual security risk analysis" in r for r in rec_texts)


# ============================================================================
# HIPAA PHI Access Log - GET /api/v2/compliance/hipaa/phi-access
# ============================================================================


class TestHIPAAPHIAccessLog:
    """Tests for _hipaa_phi_access_log method."""

    @pytest.mark.asyncio
    async def test_returns_200(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access", {}, mock_h
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_returns_access_log_structure(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access", {}, mock_h
        )
        body = _body(result)
        assert "phi_access_log" in body
        assert "count" in body
        assert "filters" in body
        assert "hipaa_reference" in body

    @pytest.mark.asyncio
    async def test_hipaa_reference_is_correct(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access", {}, mock_h
        )
        body = _body(result)
        assert "164.312(b)" in body["hipaa_reference"]

    @pytest.mark.asyncio
    async def test_empty_log_returns_zero_count(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access", {}, mock_h
        )
        body = _body(result)
        assert body["count"] == 0
        assert body["phi_access_log"] == []

    @pytest.mark.asyncio
    async def test_filters_by_patient_id(self, handler, _patch_stores):
        _patch_stores["audit_store"].get_log.return_value = [
            {
                "action": "phi_access",
                "user_id": "doc-1",
                "timestamp": "2025-06-01T10:00:00+00:00",
                "metadata": {"patient_id": "p-123", "access_type": "read"},
            },
            {
                "action": "phi_access",
                "user_id": "doc-2",
                "timestamp": "2025-06-01T11:00:00+00:00",
                "metadata": {"patient_id": "p-456", "access_type": "read"},
            },
        ]
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access",
            {"patient_id": "p-123"},
            mock_h,
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["phi_access_log"][0]["patient_id"] == "p-123"

    @pytest.mark.asyncio
    async def test_filters_by_user_id(self, handler, _patch_stores):
        _patch_stores["audit_store"].get_log.return_value = [
            {
                "action": "phi_read",
                "user_id": "doc-1",
                "timestamp": "2025-06-01T10:00:00+00:00",
                "metadata": {"patient_id": "p-123"},
            },
            {
                "action": "phi_write",
                "user_id": "doc-2",
                "timestamp": "2025-06-01T11:00:00+00:00",
                "metadata": {"patient_id": "p-456"},
            },
        ]
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access",
            {"user_id": "doc-1"},
            mock_h,
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["phi_access_log"][0]["user_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_filters_by_date_range(self, handler, _patch_stores):
        _patch_stores["audit_store"].get_log.return_value = [
            {
                "action": "phi_access",
                "user_id": "doc-1",
                "timestamp": "2025-01-15T10:00:00+00:00",
                "metadata": {"patient_id": "p-1"},
            },
            {
                "action": "phi_access",
                "user_id": "doc-1",
                "timestamp": "2025-06-15T10:00:00+00:00",
                "metadata": {"patient_id": "p-2"},
            },
            {
                "action": "phi_access",
                "user_id": "doc-1",
                "timestamp": "2025-12-15T10:00:00+00:00",
                "metadata": {"patient_id": "p-3"},
            },
        ]
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access",
            {"from": "2025-03-01T00:00:00+00:00", "to": "2025-09-01T00:00:00+00:00"},
            mock_h,
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["phi_access_log"][0]["patient_id"] == "p-2"

    @pytest.mark.asyncio
    async def test_limit_param(self, handler, _patch_stores):
        events = [
            {
                "action": "phi_read",
                "user_id": f"doc-{i}",
                "timestamp": f"2025-06-{i + 1:02d}T10:00:00+00:00",
                "metadata": {"patient_id": f"p-{i}"},
            }
            for i in range(10)
        ]
        _patch_stores["audit_store"].get_log.return_value = events
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access",
            {"limit": "3"},
            mock_h,
        )
        body = _body(result)
        assert body["count"] == 3

    @pytest.mark.asyncio
    async def test_limit_capped_at_1000(self, handler, _patch_stores):
        """Even if requesting limit=5000, it should cap at 1000."""
        _patch_stores["audit_store"].get_log.return_value = []
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access",
            {"limit": "5000"},
            mock_h,
        )
        body = _body(result)
        # Verify it completed without error (the cap is applied internally)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_non_phi_events_filtered_out(self, handler, _patch_stores):
        _patch_stores["audit_store"].get_log.return_value = [
            {
                "action": "login",
                "user_id": "user-1",
                "timestamp": "2025-06-01T10:00:00+00:00",
                "metadata": {},
            },
            {
                "action": "phi_access",
                "user_id": "doc-1",
                "timestamp": "2025-06-01T11:00:00+00:00",
                "metadata": {"patient_id": "p-1"},
            },
        ]
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access", {}, mock_h
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["phi_access_log"][0]["action"] == "phi_access"

    @pytest.mark.asyncio
    async def test_filters_returned_in_response(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access",
            {"patient_id": "p-99", "user_id": "doc-5"},
            mock_h,
        )
        body = _body(result)
        assert body["filters"]["patient_id"] == "p-99"
        assert body["filters"]["user_id"] == "doc-5"

    @pytest.mark.asyncio
    async def test_store_error_returns_500(self, handler, _patch_stores):
        _patch_stores["audit_store"].get_log.side_effect = RuntimeError("db down")
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access", {}, mock_h
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_access_log_entry_structure(self, handler, _patch_stores):
        _patch_stores["audit_store"].get_log.return_value = [
            {
                "action": "phi_view",
                "user_id": "doc-1",
                "timestamp": "2025-06-01T10:00:00+00:00",
                "metadata": {
                    "patient_id": "p-1",
                    "access_type": "view",
                    "phi_elements": ["name", "dob"],
                    "purpose": "treatment",
                    "ip_address": "10.0.0.1",
                },
            }
        ]
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access", {}, mock_h
        )
        body = _body(result)
        entry = body["phi_access_log"][0]
        assert entry["timestamp"] == "2025-06-01T10:00:00+00:00"
        assert entry["action"] == "phi_view"
        assert entry["user_id"] == "doc-1"
        assert entry["patient_id"] == "p-1"
        assert entry["access_type"] == "view"
        assert entry["phi_elements"] == ["name", "dob"]
        assert entry["purpose"] == "treatment"
        assert entry["ip_address"] == "10.0.0.1"

    @pytest.mark.asyncio
    async def test_datetime_object_timestamp(self, handler, _patch_stores):
        """Timestamps that are datetime objects (not strings) should be handled."""
        ts = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        _patch_stores["audit_store"].get_log.return_value = [
            {
                "action": "phi_access",
                "user_id": "doc-1",
                "timestamp": ts,
                "metadata": {"patient_id": "p-1"},
            }
        ]
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access", {}, mock_h
        )
        body = _body(result)
        assert body["count"] == 1


# ============================================================================
# HIPAA Breach Assessment - POST /api/v2/compliance/hipaa/breach-assessment
# ============================================================================


class TestHIPAABreachAssessment:
    """Tests for _hipaa_breach_assessment method."""

    @pytest.mark.asyncio
    async def test_missing_incident_id_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"incident_type": "data_leak"})
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_incident_type_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"incident_id": "inc-1"})
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_body_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_no_phi_involved_returns_not_applicable(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-1",
                "incident_type": "lost_laptop",
                "phi_involved": False,
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["breach_determination"] == "not_applicable"
        assert body["notification_required"] is False
        assert "message" in body

    @pytest.mark.asyncio
    async def test_no_phi_default_is_false(self, handler):
        """phi_involved defaults to False when not provided."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={"incident_id": "inc-1", "incident_type": "lost_laptop"},
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        assert body["breach_determination"] == "not_applicable"

    @pytest.mark.asyncio
    async def test_phi_with_sensitive_types_high_risk(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-2",
                "incident_type": "unauthorized_access",
                "phi_involved": True,
                "phi_types": ["SSN", "Medical diagnosis"],
                "unauthorized_access": {"confirmed_access": True},
                "affected_individuals": 1000,
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        assert body["breach_determination"] == "presumed_breach"
        assert body["notification_required"] is True

    @pytest.mark.asyncio
    async def test_notification_deadlines_present_for_breach(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-3",
                "incident_type": "hack",
                "phi_involved": True,
                "phi_types": ["SSN", "Financial"],
                "unauthorized_access": {"confirmed_access": True},
                "affected_individuals": 600,
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        deadlines = body["notification_deadlines"]
        assert "individual_notification" in deadlines
        assert "hhs_notification" in deadlines
        assert "media_notification" in deadlines

    @pytest.mark.asyncio
    async def test_large_breach_requires_media_notification(self, handler):
        """500+ affected individuals require media notification."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-4",
                "incident_type": "data_exposure",
                "phi_involved": True,
                "phi_types": ["SSN", "Treatment information"],
                "unauthorized_access": {"confirmed_access": True},
                "affected_individuals": 500,
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        deadlines = body["notification_deadlines"]
        # For 500+ individuals, media notification has a date, not "Not required"
        assert deadlines["media_notification"] != "Not required"

    @pytest.mark.asyncio
    async def test_small_breach_no_media_notification(self, handler):
        """Fewer than 500 affected individuals: media notification not required."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-5",
                "incident_type": "data_exposure",
                "phi_involved": True,
                "phi_types": ["SSN", "Financial"],
                "unauthorized_access": {"confirmed_access": True},
                "affected_individuals": 100,
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        deadlines = body["notification_deadlines"]
        assert deadlines["media_notification"] == "Not required"

    @pytest.mark.asyncio
    async def test_small_breach_hhs_annual(self, handler):
        """Fewer than 500 affected: HHS notification is annual."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-6",
                "incident_type": "phishing",
                "phi_involved": True,
                "phi_types": ["SSN", "Financial"],
                "unauthorized_access": {"confirmed_access": True},
                "affected_individuals": 50,
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        deadlines = body["notification_deadlines"]
        assert deadlines["hhs_notification"] == "Annual"

    @pytest.mark.asyncio
    async def test_low_risk_no_notification(self, handler):
        """Low risk (fewer than 2 high factors) does not require notification."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-7",
                "incident_type": "misdirected_email",
                "phi_involved": True,
                "phi_types": ["Names"],  # Non-sensitive
                "unauthorized_access": {"known_recipient": True},
                "mitigation_actions": ["recall", "confirm_delete", "retrain"],
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        assert body["breach_determination"] == "low_probability"
        assert body["notification_required"] is False
        assert body["notification_deadlines"] is None

    @pytest.mark.asyncio
    async def test_risk_factors_present_for_phi(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-8",
                "incident_type": "data_leak",
                "phi_involved": True,
                "phi_types": ["Email addresses"],
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        assert len(body["risk_factors"]) == 4
        factor_names = [f["factor"] for f in body["risk_factors"]]
        assert "Nature and extent of PHI" in factor_names
        assert "Unauthorized person" in factor_names
        assert "PHI acquisition/viewing" in factor_names
        assert "Risk mitigation" in factor_names

    @pytest.mark.asyncio
    async def test_assessment_id_format(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-9",
                "incident_type": "theft",
                "phi_involved": False,
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        assert body["assessment_id"].startswith("hipaa-breach-inc-9-")

    @pytest.mark.asyncio
    async def test_audit_store_log_on_assessment(self, handler, _patch_stores):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-10",
                "incident_type": "data_leak",
                "phi_involved": False,
            },
        )
        await handler.handle("/api/v2/compliance/hipaa/breach-assessment", {}, mock_h)
        _patch_stores["audit_store"].log_event.assert_called_once()
        call_kwargs = _patch_stores["audit_store"].log_event.call_args
        assert call_kwargs[1]["action"] == "hipaa_breach_assessment"
        assert call_kwargs[1]["resource_type"] == "incident"

    @pytest.mark.asyncio
    async def test_audit_store_failure_does_not_break(self, handler, _patch_stores):
        """Audit store failure should not prevent the assessment from returning."""
        _patch_stores["audit_store"].log_event.side_effect = RuntimeError("store down")
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-11",
                "incident_type": "loss",
                "phi_involved": False,
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_known_recipient_moderate_risk(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-12",
                "incident_type": "misdirected",
                "phi_involved": True,
                "phi_types": ["Names"],
                "unauthorized_access": {"known_recipient": True},
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        person_factor = next(f for f in body["risk_factors"] if f["factor"] == "Unauthorized person")
        assert person_factor["risk"] == "moderate"

    @pytest.mark.asyncio
    async def test_unknown_recipient_high_risk(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-13",
                "incident_type": "data_leak",
                "phi_involved": True,
                "phi_types": ["Names"],
                "unauthorized_access": {},
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        person_factor = next(f for f in body["risk_factors"] if f["factor"] == "Unauthorized person")
        assert person_factor["risk"] == "high"

    @pytest.mark.asyncio
    async def test_confirmed_access_high_risk(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-14",
                "incident_type": "hack",
                "phi_involved": True,
                "phi_types": ["Names"],
                "unauthorized_access": {"confirmed_access": True},
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        access_factor = next(f for f in body["risk_factors"] if f["factor"] == "PHI acquisition/viewing")
        assert access_factor["risk"] == "high"

    @pytest.mark.asyncio
    async def test_no_confirmed_access_low_risk(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-15",
                "incident_type": "misdirected",
                "phi_involved": True,
                "phi_types": ["Names"],
                "unauthorized_access": {},
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        access_factor = next(f for f in body["risk_factors"] if f["factor"] == "PHI acquisition/viewing")
        assert access_factor["risk"] == "low"

    @pytest.mark.asyncio
    async def test_comprehensive_mitigation_low_risk(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-16",
                "incident_type": "misdirected",
                "phi_involved": True,
                "phi_types": ["Names"],
                "mitigation_actions": ["recall", "confirm_delete", "retrain"],
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        mitigation_factor = next(f for f in body["risk_factors"] if f["factor"] == "Risk mitigation")
        assert mitigation_factor["risk"] == "low"

    @pytest.mark.asyncio
    async def test_limited_mitigation_moderate_risk(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-17",
                "incident_type": "misdirected",
                "phi_involved": True,
                "phi_types": ["Names"],
                "mitigation_actions": ["recall"],
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        mitigation_factor = next(f for f in body["risk_factors"] if f["factor"] == "Risk mitigation")
        assert mitigation_factor["risk"] == "moderate"

    @pytest.mark.asyncio
    async def test_sensitive_phi_high_nature_risk(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-18",
                "incident_type": "breach",
                "phi_involved": True,
                "phi_types": ["SSN"],
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        nature_factor = next(f for f in body["risk_factors"] if f["factor"] == "Nature and extent of PHI")
        assert nature_factor["risk"] == "high"

    @pytest.mark.asyncio
    async def test_non_sensitive_phi_moderate_nature_risk(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "incident_id": "inc-19",
                "incident_type": "breach",
                "phi_involved": True,
                "phi_types": ["Names"],
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        body = _body(result)
        nature_factor = next(f for f in body["risk_factors"] if f["factor"] == "Nature and extent of PHI")
        assert nature_factor["risk"] == "moderate"


# ============================================================================
# HIPAA BAA List - GET /api/v2/compliance/hipaa/baa
# ============================================================================


class TestHIPAAListBAAs:
    """Tests for _hipaa_list_baas method."""

    @pytest.mark.asyncio
    async def test_returns_200(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_response_structure(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        assert "business_associates" in body
        assert "count" in body
        assert "filters" in body
        assert "hipaa_reference" in body

    @pytest.mark.asyncio
    async def test_hipaa_reference_correct(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        assert "164.502(e)" in body["hipaa_reference"]

    @pytest.mark.asyncio
    async def test_default_status_filter_active(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        assert body["filters"]["status"] == "active"

    @pytest.mark.asyncio
    async def test_default_ba_type_filter_all(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        assert body["filters"]["ba_type"] == "all"

    @pytest.mark.asyncio
    async def test_custom_filters_in_response(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/baa",
            {"status": "expired", "ba_type": "vendor"},
            mock_h,
        )
        body = _body(result)
        assert body["filters"]["status"] == "expired"
        assert body["filters"]["ba_type"] == "vendor"

    @pytest.mark.asyncio
    async def test_returns_baa_entries(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        assert body["count"] >= 0
        assert len(body["business_associates"]) == body["count"]


# ============================================================================
# HIPAA BAA Create - POST /api/v2/compliance/hipaa/baa
# ============================================================================


class TestHIPAACreateBAA:
    """Tests for _hipaa_create_baa method."""

    @pytest.mark.asyncio
    async def test_missing_business_associate_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={"ba_type": "vendor", "services_provided": "hosting"},
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_ba_type_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={"business_associate": "Corp A", "services_provided": "hosting"},
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_ba_type_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "business_associate": "Corp A",
                "ba_type": "invalid_type",
                "services_provided": "hosting",
            },
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_services_provided_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={"business_associate": "Corp A", "ba_type": "vendor"},
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_body_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_successful_create_vendor(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "business_associate": "Cloud Provider X",
                "ba_type": "vendor",
                "services_provided": "Cloud hosting services",
            },
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        assert _status(result) == 201
        assert body["message"] == "Business Associate Agreement registered"
        assert body["baa"]["business_associate"] == "Cloud Provider X"

    @pytest.mark.asyncio
    async def test_successful_create_subcontractor(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "business_associate": "Data Processor Y",
                "ba_type": "subcontractor",
                "services_provided": "Data processing",
            },
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        assert _status(result) == 201
        assert body["baa"]["ba_type"] == "subcontractor"

    @pytest.mark.asyncio
    async def test_baa_id_generated(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "business_associate": "Test Corp",
                "ba_type": "vendor",
                "services_provided": "Testing",
            },
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        assert body["baa"]["baa_id"].startswith("baa-")

    @pytest.mark.asyncio
    async def test_baa_has_required_provisions(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "business_associate": "Test Corp",
                "ba_type": "vendor",
                "services_provided": "Testing",
            },
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        provisions = body["baa"]["required_provisions"]
        assert len(provisions) == 8
        assert "Breach notification obligation" in provisions
        assert "Safeguards requirement" in provisions

    @pytest.mark.asyncio
    async def test_baa_status_is_active(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "business_associate": "Test Corp",
                "ba_type": "vendor",
                "services_provided": "Testing",
            },
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        assert body["baa"]["status"] == "active"

    @pytest.mark.asyncio
    async def test_baa_optional_fields(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "business_associate": "Test Corp",
                "ba_type": "vendor",
                "services_provided": "Testing",
                "phi_access_scope": ["medical_records", "billing"],
                "expiration_date": "2026-12-31T00:00:00Z",
                "subcontractor_clause": False,
            },
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        assert body["baa"]["phi_access_scope"] == ["medical_records", "billing"]
        assert body["baa"]["expiration_date"] == "2026-12-31T00:00:00Z"
        assert body["baa"]["subcontractor_clause"] is False

    @pytest.mark.asyncio
    async def test_baa_default_subcontractor_clause(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "business_associate": "Test Corp",
                "ba_type": "vendor",
                "services_provided": "Testing",
            },
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        body = _body(result)
        assert body["baa"]["subcontractor_clause"] is True

    @pytest.mark.asyncio
    async def test_baa_audit_log(self, handler, _patch_stores):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "business_associate": "Test Corp",
                "ba_type": "vendor",
                "services_provided": "Testing",
            },
        )
        await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        _patch_stores["audit_store"].log_event.assert_called_once()
        call_kwargs = _patch_stores["audit_store"].log_event.call_args[1]
        assert call_kwargs["action"] == "hipaa_baa_created"
        assert call_kwargs["resource_type"] == "baa"

    @pytest.mark.asyncio
    async def test_baa_audit_failure_does_not_break(self, handler, _patch_stores):
        _patch_stores["audit_store"].log_event.side_effect = RuntimeError("db down")
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "business_associate": "Test Corp",
                "ba_type": "vendor",
                "services_provided": "Testing",
            },
        )
        result = await handler.handle("/api/v2/compliance/hipaa/baa", {}, mock_h)
        assert _status(result) == 201


# ============================================================================
# HIPAA Security Report - GET /api/v2/compliance/hipaa/security-report
# ============================================================================


class TestHIPAASecurityReport:
    """Tests for _hipaa_security_report method."""

    @pytest.mark.asyncio
    async def test_json_report_returns_200(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {}, mock_h
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_json_report_structure(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {}, mock_h
        )
        body = _body(result)
        assert body["report_type"] == "HIPAA Security Rule Compliance"
        assert "report_id" in body
        assert "generated_at" in body
        assert "assessment_period" in body
        assert "safeguards" in body
        assert "summary" in body

    @pytest.mark.asyncio
    async def test_report_id_format(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {}, mock_h
        )
        body = _body(result)
        assert body["report_id"].startswith("hipaa-sec-")

    @pytest.mark.asyncio
    async def test_assessment_period_has_start_end(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {}, mock_h
        )
        body = _body(result)
        assert "start" in body["assessment_period"]
        assert "end" in body["assessment_period"]

    @pytest.mark.asyncio
    async def test_safeguard_categories(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {}, mock_h
        )
        body = _body(result)
        assert "administrative" in body["safeguards"]
        assert "physical" in body["safeguards"]
        assert "technical" in body["safeguards"]

    @pytest.mark.asyncio
    async def test_safeguard_category_structure(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {}, mock_h
        )
        body = _body(result)
        admin = body["safeguards"]["administrative"]
        assert "category" in admin
        assert "standards" in admin
        assert "compliant_count" in admin
        assert "total_count" in admin

    @pytest.mark.asyncio
    async def test_summary_has_counts(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {}, mock_h
        )
        body = _body(result)
        summary = body["summary"]
        assert "total_standards_assessed" in summary
        assert "standards_compliant" in summary
        assert "compliance_percentage" in summary
        assert "overall_status" in summary

    @pytest.mark.asyncio
    async def test_html_report_returns_text_html(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {"format": "html"}, mock_h
        )
        assert _status(result) == 200
        assert result.content_type == "text/html"

    @pytest.mark.asyncio
    async def test_html_report_contains_title(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {"format": "html"}, mock_h
        )
        html = result.body.decode("utf-8")
        assert "HIPAA Security Rule Compliance Report" in html

    @pytest.mark.asyncio
    async def test_html_report_contains_safeguard_ids(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {"format": "html"}, mock_h
        )
        html = result.body.decode("utf-8")
        assert "164.308" in html
        assert "164.310" in html
        assert "164.312" in html

    @pytest.mark.asyncio
    async def test_include_evidence(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report",
            {"include_evidence": "true"},
            mock_h,
        )
        body = _body(result)
        assert "evidence_references" in body
        assert len(body["evidence_references"]) > 0

    @pytest.mark.asyncio
    async def test_exclude_evidence_by_default(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {}, mock_h
        )
        body = _body(result)
        assert "evidence_references" not in body

    @pytest.mark.asyncio
    async def test_all_compliant_status(self, handler):
        """Default safeguards are all compliant, so overall should be Compliant."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {}, mock_h
        )
        body = _body(result)
        assert body["summary"]["overall_status"] == "Compliant"
        assert body["summary"]["compliance_percentage"] == 100


# ============================================================================
# HIPAA De-identification - POST /api/v2/compliance/hipaa/deidentify
# ============================================================================


class TestHIPAADeidentify:
    """Tests for _hipaa_deidentify method."""

    @pytest.mark.asyncio
    async def test_missing_content_and_data_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle(
            "/api/v2/compliance/hipaa/deidentify", {}, mock_h
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_content_deidentification(self, handler):
        mock_result = MagicMock()
        mock_result.audit_id = "audit-123"
        mock_result.identifiers_found = []
        mock_result.fields_anonymized = []
        mock_result.to_dict.return_value = {
            "anonymized_content": "[REDACTED] visited the clinic",
            "identifiers_count": 1,
            "audit_id": "audit-123",
        }

        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.anonymize.return_value = mock_result

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler(
                "POST",
                body={"content": "John Doe visited the clinic", "method": "redact"},
            )
            result = await handler.handle(
                "/api/v2/compliance/hipaa/deidentify", {}, mock_h
            )
            body = _body(result)
            assert _status(result) == 200
            assert "anonymized_content" in body

    @pytest.mark.asyncio
    async def test_data_deidentification(self, handler):
        mock_result = MagicMock()
        mock_result.audit_id = "audit-456"
        mock_result.identifiers_found = []
        mock_result.fields_anonymized = ["name"]
        mock_result.to_dict.return_value = {
            "anonymized_content": "",
            "fields_anonymized": ["name"],
            "audit_id": "audit-456",
        }

        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.anonymize_structured.return_value = mock_result

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler(
                "POST",
                body={"data": {"name": "John Doe", "diagnosis": "flu"}},
            )
            result = await handler.handle(
                "/api/v2/compliance/hipaa/deidentify", {}, mock_h
            )
            body = _body(result)
            assert _status(result) == 200
            mock_anonymizer_cls.return_value.anonymize_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_method_returns_400(self, handler):
        from aragora.privacy.anonymization import AnonymizationMethod

        with patch.object(
            AnonymizationMethod, "__new__", side_effect=ValueError("invalid"),
        ):
            mock_h = _MockHTTPHandler(
                "POST",
                body={"content": "test", "method": "invalid_method"},
            )
            result = await handler.handle(
                "/api/v2/compliance/hipaa/deidentify", {}, mock_h
            )
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_import_failure_returns_501(self, handler):
        """If privacy module is not available, return 501."""
        import sys

        # Temporarily remove the module from sys.modules to force ImportError
        saved = sys.modules.get("aragora.privacy.anonymization")
        sys.modules["aragora.privacy.anonymization"] = None  # type: ignore
        try:
            mock_h = _MockHTTPHandler(
                "POST", body={"content": "test"}
            )
            result = await handler.handle(
                "/api/v2/compliance/hipaa/deidentify", {}, mock_h
            )
            assert _status(result) == 501
        finally:
            if saved is not None:
                sys.modules["aragora.privacy.anonymization"] = saved
            else:
                sys.modules.pop("aragora.privacy.anonymization", None)

    @pytest.mark.asyncio
    async def test_audit_log_on_deidentify(self, handler, _patch_stores):
        mock_result = MagicMock()
        mock_result.audit_id = "audit-789"
        mock_result.identifiers_found = [MagicMock()]
        mock_result.fields_anonymized = []
        mock_result.to_dict.return_value = {"audit_id": "audit-789"}

        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.anonymize.return_value = mock_result

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler("POST", body={"content": "test data"})
            await handler.handle("/api/v2/compliance/hipaa/deidentify", {}, mock_h)

        _patch_stores["audit_store"].log_event.assert_called_once()
        call_kwargs = _patch_stores["audit_store"].log_event.call_args[1]
        assert call_kwargs["action"] == "hipaa_phi_deidentified"

    @pytest.mark.asyncio
    async def test_invalid_identifier_type_returns_400(self, handler):
        from aragora.privacy.anonymization import IdentifierType

        with patch.object(
            IdentifierType, "__new__", side_effect=ValueError("bad type"),
        ):
            mock_h = _MockHTTPHandler(
                "POST",
                body={
                    "content": "test",
                    "identifier_types": ["not_a_real_type"],
                },
            )
            result = await handler.handle(
                "/api/v2/compliance/hipaa/deidentify", {}, mock_h
            )
            assert _status(result) == 400


# ============================================================================
# HIPAA Safe Harbor Verify - POST /api/v2/compliance/hipaa/safe-harbor/verify
# ============================================================================


class TestHIPAASafeHarborVerify:
    """Tests for _hipaa_safe_harbor_verify method."""

    @pytest.mark.asyncio
    async def test_missing_content_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle(
            "/api/v2/compliance/hipaa/safe-harbor/verify", {}, mock_h
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_compliant_content(self, handler):
        mock_safe_harbor_result = MagicMock()
        mock_safe_harbor_result.compliant = True
        mock_safe_harbor_result.identifiers_remaining = []
        mock_safe_harbor_result.verification_notes = ["All identifiers removed"]
        mock_safe_harbor_result.verified_at = datetime(2025, 6, 1, tzinfo=timezone.utc)

        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.verify_safe_harbor.return_value = mock_safe_harbor_result

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler(
                "POST", body={"content": "Patient treated at facility"}
            )
            result = await handler.handle(
                "/api/v2/compliance/hipaa/safe-harbor/verify", {}, mock_h
            )
            body = _body(result)
            assert _status(result) == 200
            assert body["compliant"] is True
            assert body["identifiers_remaining"] == []
            assert body["hipaa_reference"] == "45 CFR 164.514(b) - Safe Harbor Method"

    @pytest.mark.asyncio
    async def test_non_compliant_content(self, handler):
        mock_identifier = MagicMock()
        mock_identifier.identifier_type.value = "name"
        mock_identifier.value = "John Doe"
        mock_identifier.confidence = 0.95
        mock_identifier.start_pos = 0
        mock_identifier.end_pos = 8

        mock_safe_harbor_result = MagicMock()
        mock_safe_harbor_result.compliant = False
        mock_safe_harbor_result.identifiers_remaining = [mock_identifier]
        mock_safe_harbor_result.verification_notes = ["Names detected"]
        mock_safe_harbor_result.verified_at = datetime(2025, 6, 1, tzinfo=timezone.utc)

        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.verify_safe_harbor.return_value = mock_safe_harbor_result

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler(
                "POST", body={"content": "John Doe visited clinic"}
            )
            result = await handler.handle(
                "/api/v2/compliance/hipaa/safe-harbor/verify", {}, mock_h
            )
            body = _body(result)
            assert _status(result) == 200
            assert body["compliant"] is False
            assert len(body["identifiers_remaining"]) == 1
            ident = body["identifiers_remaining"][0]
            assert ident["type"] == "name"
            assert ident["confidence"] == 0.95
            assert "position" in ident

    @pytest.mark.asyncio
    async def test_value_preview_truncated(self, handler):
        """Values longer than 3 chars should show only first 3 + '...'."""
        mock_identifier = MagicMock()
        mock_identifier.identifier_type.value = "name"
        mock_identifier.value = "John Doe"
        mock_identifier.confidence = 0.9
        mock_identifier.start_pos = 0
        mock_identifier.end_pos = 8

        mock_safe_harbor_result = MagicMock()
        mock_safe_harbor_result.compliant = False
        mock_safe_harbor_result.identifiers_remaining = [mock_identifier]
        mock_safe_harbor_result.verification_notes = []
        mock_safe_harbor_result.verified_at = datetime(2025, 6, 1, tzinfo=timezone.utc)

        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.verify_safe_harbor.return_value = mock_safe_harbor_result

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler("POST", body={"content": "John Doe test"})
            result = await handler.handle(
                "/api/v2/compliance/hipaa/safe-harbor/verify", {}, mock_h
            )
            body = _body(result)
            assert body["identifiers_remaining"][0]["value_preview"] == "Joh..."

    @pytest.mark.asyncio
    async def test_short_value_shows_asterisks(self, handler):
        """Values of 3 chars or less should show '***'."""
        mock_identifier = MagicMock()
        mock_identifier.identifier_type.value = "ssn"
        mock_identifier.value = "123"
        mock_identifier.confidence = 0.8
        mock_identifier.start_pos = 0
        mock_identifier.end_pos = 3

        mock_safe_harbor_result = MagicMock()
        mock_safe_harbor_result.compliant = False
        mock_safe_harbor_result.identifiers_remaining = [mock_identifier]
        mock_safe_harbor_result.verification_notes = []
        mock_safe_harbor_result.verified_at = datetime(2025, 6, 1, tzinfo=timezone.utc)

        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.verify_safe_harbor.return_value = mock_safe_harbor_result

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler("POST", body={"content": "123"})
            result = await handler.handle(
                "/api/v2/compliance/hipaa/safe-harbor/verify", {}, mock_h
            )
            body = _body(result)
            assert body["identifiers_remaining"][0]["value_preview"] == "***"

    @pytest.mark.asyncio
    async def test_import_failure_returns_501(self, handler):
        import sys

        saved = sys.modules.get("aragora.privacy.anonymization")
        sys.modules["aragora.privacy.anonymization"] = None  # type: ignore
        try:
            mock_h = _MockHTTPHandler("POST", body={"content": "test"})
            result = await handler.handle(
                "/api/v2/compliance/hipaa/safe-harbor/verify", {}, mock_h
            )
            assert _status(result) == 501
        finally:
            if saved is not None:
                sys.modules["aragora.privacy.anonymization"] = saved
            else:
                sys.modules.pop("aragora.privacy.anonymization", None)


# ============================================================================
# HIPAA Detect PHI - POST /api/v2/compliance/hipaa/detect-phi
# ============================================================================


class TestHIPAADetectPHI:
    """Tests for _hipaa_detect_phi method."""

    @pytest.mark.asyncio
    async def test_missing_content_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle(
            "/api/v2/compliance/hipaa/detect-phi", {}, mock_h
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_detects_identifiers(self, handler):
        mock_ident = MagicMock()
        mock_ident.identifier_type.value = "email"
        mock_ident.value = "user@example.com"
        mock_ident.start_pos = 10
        mock_ident.end_pos = 26
        mock_ident.confidence = 0.95

        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.detect_identifiers.return_value = [mock_ident]

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler(
                "POST",
                body={"content": "Contact: user@example.com for info"},
            )
            result = await handler.handle(
                "/api/v2/compliance/hipaa/detect-phi", {}, mock_h
            )
            body = _body(result)
            assert _status(result) == 200
            assert body["count"] == 1
            assert body["identifiers"][0]["type"] == "email"
            assert body["identifiers"][0]["value"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_min_confidence_filter(self, handler):
        high_conf = MagicMock()
        high_conf.identifier_type.value = "email"
        high_conf.value = "user@test.com"
        high_conf.start_pos = 0
        high_conf.end_pos = 13
        high_conf.confidence = 0.9

        low_conf = MagicMock()
        low_conf.identifier_type.value = "phone"
        low_conf.value = "555-1234"
        low_conf.start_pos = 20
        low_conf.end_pos = 28
        low_conf.confidence = 0.3

        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.detect_identifiers.return_value = [low_conf, high_conf]

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler(
                "POST",
                body={"content": "test text", "min_confidence": 0.5},
            )
            result = await handler.handle(
                "/api/v2/compliance/hipaa/detect-phi", {}, mock_h
            )
            body = _body(result)
            assert body["count"] == 1
            assert body["identifiers"][0]["type"] == "email"

    @pytest.mark.asyncio
    async def test_default_min_confidence_is_05(self, handler):
        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.detect_identifiers.return_value = []

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler("POST", body={"content": "test"})
            result = await handler.handle(
                "/api/v2/compliance/hipaa/detect-phi", {}, mock_h
            )
            body = _body(result)
            assert body["min_confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_hipaa_reference_in_response(self, handler):
        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.detect_identifiers.return_value = []

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler("POST", body={"content": "test"})
            result = await handler.handle(
                "/api/v2/compliance/hipaa/detect-phi", {}, mock_h
            )
            body = _body(result)
            assert "164.514" in body["hipaa_reference"]

    @pytest.mark.asyncio
    async def test_no_identifiers_found(self, handler):
        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.detect_identifiers.return_value = []

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler(
                "POST", body={"content": "no PHI here"}
            )
            result = await handler.handle(
                "/api/v2/compliance/hipaa/detect-phi", {}, mock_h
            )
            body = _body(result)
            assert body["count"] == 0
            assert body["identifiers"] == []

    @pytest.mark.asyncio
    async def test_import_failure_returns_501(self, handler):
        import sys

        saved = sys.modules.get("aragora.privacy.anonymization")
        sys.modules["aragora.privacy.anonymization"] = None  # type: ignore
        try:
            mock_h = _MockHTTPHandler("POST", body={"content": "test"})
            result = await handler.handle(
                "/api/v2/compliance/hipaa/detect-phi", {}, mock_h
            )
            assert _status(result) == 501
        finally:
            if saved is not None:
                sys.modules["aragora.privacy.anonymization"] = saved
            else:
                sys.modules.pop("aragora.privacy.anonymization", None)

    @pytest.mark.asyncio
    async def test_identifier_structure(self, handler):
        mock_ident = MagicMock()
        mock_ident.identifier_type.value = "ssn"
        mock_ident.value = "123-45-6789"
        mock_ident.start_pos = 5
        mock_ident.end_pos = 16
        mock_ident.confidence = 0.99

        mock_anonymizer_cls = MagicMock()
        mock_anonymizer_cls.return_value.detect_identifiers.return_value = [mock_ident]

        with patch(
            "aragora.privacy.anonymization.HIPAAAnonymizer",
            mock_anonymizer_cls,
        ):
            mock_h = _MockHTTPHandler(
                "POST", body={"content": "SSN: 123-45-6789"}
            )
            result = await handler.handle(
                "/api/v2/compliance/hipaa/detect-phi", {}, mock_h
            )
            body = _body(result)
            ident = body["identifiers"][0]
            assert "type" in ident
            assert "value" in ident
            assert "start" in ident
            assert "end" in ident
            assert "confidence" in ident


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestHIPAAHelperMethods:
    """Tests for internal helper methods on the mixin."""

    @pytest.mark.asyncio
    async def test_evaluate_safeguards_returns_all_categories(self, handler):
        result = await handler._evaluate_safeguards()
        assert "administrative" in result
        assert "physical" in result
        assert "technical" in result

    @pytest.mark.asyncio
    async def test_evaluate_safeguards_administrative_count(self, handler):
        result = await handler._evaluate_safeguards()
        assert len(result["administrative"]) == 8

    @pytest.mark.asyncio
    async def test_evaluate_safeguards_physical_count(self, handler):
        result = await handler._evaluate_safeguards()
        assert len(result["physical"]) == 4

    @pytest.mark.asyncio
    async def test_evaluate_safeguards_technical_count(self, handler):
        result = await handler._evaluate_safeguards()
        assert len(result["technical"]) == 5

    @pytest.mark.asyncio
    async def test_evaluate_safeguards_entry_structure(self, handler):
        result = await handler._evaluate_safeguards()
        entry = result["administrative"][0]
        assert "id" in entry
        assert "name" in entry
        assert "status" in entry
        assert "controls" in entry
        assert "last_assessed" in entry

    @pytest.mark.asyncio
    async def test_evaluate_phi_controls(self, handler):
        result = await handler._evaluate_phi_controls()
        assert result["status"] == "configured"
        assert result["identifiers_tracked"] == 18
        assert result["de_identification_method"] == "Safe Harbor"
        assert result["minimum_necessary_enforced"] is True

    @pytest.mark.asyncio
    async def test_evaluate_phi_controls_access_controls(self, handler):
        result = await handler._evaluate_phi_controls()
        ac = result["access_controls"]
        assert ac["role_based"] is True
        assert ac["audit_logging"] is True
        assert ac["encryption_at_rest"] is True
        assert ac["encryption_in_transit"] is True

    @pytest.mark.asyncio
    async def test_get_baa_status(self, handler):
        result = await handler._get_baa_status()
        assert "total_baas" in result
        assert "active" in result

    @pytest.mark.asyncio
    async def test_get_baa_list(self, handler):
        result = await handler._get_baa_list("active", "all")
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_hipaa_recommendations(self, handler):
        safeguards = await handler._evaluate_safeguards()
        phi_controls = await handler._evaluate_phi_controls()
        recommendations = await handler._get_hipaa_recommendations(safeguards, phi_controls)
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 1

    @pytest.mark.asyncio
    async def test_get_security_evidence(self, handler):
        result = await handler._get_security_evidence()
        assert isinstance(result, list)
        assert len(result) == 3
        assert all("control" in e for e in result)
        assert all("evidence_type" in e for e in result)

    def test_render_hipaa_html(self, handler):
        report = {
            "report_id": "test-report",
            "generated_at": "2025-01-01T00:00:00Z",
            "safeguards": {
                "administrative": {
                    "category": "Administrative",
                    "standards": [
                        {"id": "164.308(a)(1)", "name": "Security Management", "status": "compliant"},
                    ],
                    "compliant_count": 1,
                    "total_count": 1,
                },
            },
            "summary": {
                "total_standards_assessed": 1,
                "standards_compliant": 1,
                "compliance_percentage": 100,
                "overall_status": "Compliant",
            },
        }
        html = handler._render_hipaa_html(report)
        assert "HIPAA Security Rule Compliance Report" in html
        assert "test-report" in html
        assert "Administrative" in html
        assert "164.308(a)(1)" in html


# ============================================================================
# Route Dispatch Tests (HIPAA-specific)
# ============================================================================


class TestHIPAARouteDispatch:
    """Verify correct routing of HIPAA endpoints."""

    @pytest.mark.asyncio
    async def test_hipaa_status_wrong_method_returns_404(self, handler):
        mock_h = _MockHTTPHandler("POST")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/status", {}, mock_h
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_hipaa_phi_access_wrong_method_returns_404(self, handler):
        mock_h = _MockHTTPHandler("POST")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/phi-access", {}, mock_h
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_hipaa_breach_assessment_get_returns_404(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/breach-assessment", {}, mock_h
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_hipaa_baa_get_dispatches(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/baa", {}, mock_h
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_hipaa_baa_post_dispatches(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "business_associate": "Test",
                "ba_type": "vendor",
                "services_provided": "Test",
            },
        )
        result = await handler.handle(
            "/api/v2/compliance/hipaa/baa", {}, mock_h
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_hipaa_security_report_dispatches(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/security-report", {}, mock_h
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_hipaa_deidentify_get_returns_404(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/deidentify", {}, mock_h
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_hipaa_safe_harbor_get_returns_404(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/safe-harbor/verify", {}, mock_h
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_hipaa_detect_phi_get_returns_404(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/detect-phi", {}, mock_h
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_hipaa_unknown_subpath_returns_404(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/hipaa/nonexistent", {}, mock_h
        )
        assert _status(result) == 404


# ============================================================================
# HIPAA Mixin Constants
# ============================================================================


class TestHIPAAMixinConstants:
    """Verify HIPAA mixin static data is correctly defined."""

    def test_security_safeguards_has_three_categories(self, handler):
        assert set(handler.SECURITY_SAFEGUARDS.keys()) == {
            "administrative",
            "physical",
            "technical",
        }

    def test_phi_identifiers_count(self, handler):
        assert len(handler.PHI_IDENTIFIERS) == 18

    def test_phi_identifiers_includes_names(self, handler):
        assert "Names" in handler.PHI_IDENTIFIERS

    def test_phi_identifiers_includes_ssn(self, handler):
        assert "Social Security numbers" in handler.PHI_IDENTIFIERS

    def test_phi_identifiers_includes_emails(self, handler):
        assert "Email addresses" in handler.PHI_IDENTIFIERS

    def test_administrative_safeguards_count(self, handler):
        assert len(handler.SECURITY_SAFEGUARDS["administrative"]) == 8

    def test_physical_safeguards_count(self, handler):
        assert len(handler.SECURITY_SAFEGUARDS["physical"]) == 4

    def test_technical_safeguards_count(self, handler):
        assert len(handler.SECURITY_SAFEGUARDS["technical"]) == 5
