"""Comprehensive tests for the CCPAMixin handler (aragora/server/handlers/compliance/ccpa.py).

Covers all CCPA endpoints routed through ComplianceHandler:
- GET  /api/v2/compliance/ccpa/disclosure  - Right to Know (categories/specific)
- POST /api/v2/compliance/ccpa/delete      - Right to Delete
- POST /api/v2/compliance/ccpa/opt-out     - Right to Opt-Out (Do Not Sell/Share)
- POST /api/v2/compliance/ccpa/correct     - Right to Correct
- GET  /api/v2/compliance/ccpa/status      - CCPA request status

Also covers internal helpers:
- _get_ccpa_categories, _get_categories_disclosed, _get_business_purposes
- _get_specific_pi, _get_pi_sources, _get_third_party_disclosures
- _verify_ccpa_request, _store_ccpa_preference, _log_ccpa_request
- CCPA_CATEGORIES constant
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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
    """Patch external stores and event emitters used by all compliance mixins.

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

    # Patch CCPA mixin's module-level helpers
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

    # Patch GDPR mixin helpers (needed because ComplianceHandler includes all mixins)
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

    # Patch HIPAA mixin helpers
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.hipaa.get_audit_store",
        lambda: mock_audit_store,
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

    # Patch handler_events emit to avoid side effects
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.handler.emit_handler_event",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.ccpa.emit_handler_event",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.compliance.gdpr.emit_handler_event",
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
# CCPA Disclosure Endpoint - GET /api/v2/compliance/ccpa/disclosure
# ============================================================================


class TestCCPADisclosure:
    """Tests for _ccpa_disclosure method (Right to Know)."""

    @pytest.mark.asyncio
    async def test_returns_200(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure", {"user_id": "user-42"}, mock_h
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_missing_user_id_returns_400(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/ccpa/disclosure", {}, mock_h)
        assert _status(result) == 400
        assert "user_id" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_response_contains_request_id(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert body["request_id"].startswith("ccpa-disc-user-42-")

    @pytest.mark.asyncio
    async def test_response_contains_user_id(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert body["user_id"] == "user-42"

    @pytest.mark.asyncio
    async def test_response_contains_regulatory_basis(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert body["regulatory_basis"] == "California Consumer Privacy Act (CCPA)"

    @pytest.mark.asyncio
    async def test_response_contains_response_deadline(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert "response_deadline" in body
        # Deadline should parse as ISO datetime
        datetime.fromisoformat(body["response_deadline"])

    @pytest.mark.asyncio
    async def test_response_contains_requested_at(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert "requested_at" in body
        datetime.fromisoformat(body["requested_at"])

    @pytest.mark.asyncio
    async def test_response_contains_verification_hash(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert "verification_hash" in body
        # Should be a valid 64-char hex SHA-256
        assert len(body["verification_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in body["verification_hash"])

    @pytest.mark.asyncio
    async def test_default_disclosure_type_categories(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert body["disclosure_type"] == "categories"

    @pytest.mark.asyncio
    async def test_categories_disclosure_contains_categories_collected(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure",
            {"user_id": "user-42", "disclosure_type": "categories"},
            mock_h,
        )
        body = _body(result)
        assert "categories_collected" in body
        assert isinstance(body["categories_collected"], list)
        assert len(body["categories_collected"]) > 0

    @pytest.mark.asyncio
    async def test_categories_disclosure_contains_categories_disclosed(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure",
            {"user_id": "user-42", "disclosure_type": "categories"},
            mock_h,
        )
        body = _body(result)
        assert "categories_disclosed" in body
        assert isinstance(body["categories_disclosed"], list)

    @pytest.mark.asyncio
    async def test_categories_disclosure_categories_sold_empty(self, handler):
        """CCPA requires disclosing categories sold. Aragora does not sell data."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure",
            {"user_id": "user-42", "disclosure_type": "categories"},
            mock_h,
        )
        body = _body(result)
        assert body["categories_sold"] == []

    @pytest.mark.asyncio
    async def test_categories_disclosure_contains_business_purpose(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure",
            {"user_id": "user-42", "disclosure_type": "categories"},
            mock_h,
        )
        body = _body(result)
        assert "business_purpose" in body
        assert isinstance(body["business_purpose"], list)
        assert len(body["business_purpose"]) > 0

    @pytest.mark.asyncio
    async def test_specific_disclosure_contains_personal_information(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure",
            {"user_id": "user-42", "disclosure_type": "specific"},
            mock_h,
        )
        body = _body(result)
        assert "personal_information" in body

    @pytest.mark.asyncio
    async def test_specific_disclosure_contains_sources(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure",
            {"user_id": "user-42", "disclosure_type": "specific"},
            mock_h,
        )
        body = _body(result)
        assert "sources" in body
        assert isinstance(body["sources"], list)

    @pytest.mark.asyncio
    async def test_specific_disclosure_contains_third_parties(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure",
            {"user_id": "user-42", "disclosure_type": "specific"},
            mock_h,
        )
        body = _body(result)
        assert "third_parties" in body
        assert isinstance(body["third_parties"], list)

    @pytest.mark.asyncio
    async def test_pdf_format_queues_generation(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure",
            {"user_id": "user-42", "format": "pdf"},
            mock_h,
        )
        body = _body(result)
        assert "pdf_generation" in body
        assert body["pdf_generation"]["status"] == "queued"

    @pytest.mark.asyncio
    async def test_pdf_format_contains_download_url(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure",
            {"user_id": "user-42", "format": "pdf"},
            mock_h,
        )
        body = _body(result)
        assert "download_url" in body["pdf_generation"]
        assert body["pdf_generation"]["download_url"].startswith(
            "/api/v2/compliance/ccpa/disclosures/"
        )

    @pytest.mark.asyncio
    async def test_pdf_format_contains_estimated_completion(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure",
            {"user_id": "user-42", "format": "pdf"},
            mock_h,
        )
        body = _body(result)
        assert "estimated_completion" in body["pdf_generation"]
        datetime.fromisoformat(body["pdf_generation"]["estimated_completion"])

    @pytest.mark.asyncio
    async def test_json_format_no_pdf_generation(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure",
            {"user_id": "user-42", "format": "json"},
            mock_h,
        )
        body = _body(result)
        assert "pdf_generation" not in body

    @pytest.mark.asyncio
    async def test_response_deadline_is_45_days(self, handler):
        """CCPA requires 45-day response window."""
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        requested = datetime.fromisoformat(body["requested_at"])
        deadline = datetime.fromisoformat(body["response_deadline"])
        delta = deadline - requested
        assert delta.days == 45

    @pytest.mark.asyncio
    async def test_logs_ccpa_request(self, handler, _patch_stores):
        mock_h = _MockHTTPHandler("GET")
        await handler.handle("/api/v2/compliance/ccpa/disclosure", {"user_id": "user-42"}, mock_h)
        _patch_stores["audit_store"].log_event.assert_called()
        call_kwargs = _patch_stores["audit_store"].log_event.call_args[1]
        assert call_kwargs["action"] == "ccpa_disclosure_request"
        assert call_kwargs["resource_id"] == "user-42"


# ============================================================================
# CCPA Delete Endpoint - POST /api/v2/compliance/ccpa/delete
# ============================================================================


class TestCCPADelete:
    """Tests for _ccpa_delete method (Right to Delete)."""

    @pytest.mark.asyncio
    async def test_missing_user_id_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={"verification_method": "email", "verification_code": "abc123"},
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        assert _status(result) == 400
        assert "user_id" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_missing_verification_method_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={"user_id": "user-42", "verification_code": "abc123"},
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        assert _status(result) == 400
        assert "verification_method" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_missing_verification_code_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={"user_id": "user-42", "verification_method": "email"},
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        assert _status(result) == 400
        assert "verification" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_missing_both_verification_fields_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={"user_id": "user-42"},
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_body_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_verification_failure_returns_401(self, handler):
        """Failed verification returns 401."""
        with patch.object(handler, "_verify_ccpa_request", return_value=False):
            mock_h = _MockHTTPHandler(
                "POST",
                body={
                    "user_id": "user-42",
                    "verification_method": "email",
                    "verification_code": "wrong-code",
                },
            )
            result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_successful_deletion_returns_scheduled(self, handler, _patch_stores):
        mock_del = MagicMock()
        mock_del.scheduled_for = datetime.now(timezone.utc) + timedelta(days=45)
        _patch_stores["scheduler"].schedule_deletion.return_value = mock_del

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "verification_method": "email",
                "verification_code": "valid123",
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "scheduled"
        assert body["user_id"] == "user-42"

    @pytest.mark.asyncio
    async def test_successful_deletion_contains_request_id(self, handler, _patch_stores):
        mock_del = MagicMock()
        mock_del.scheduled_for = datetime.now(timezone.utc) + timedelta(days=45)
        _patch_stores["scheduler"].schedule_deletion.return_value = mock_del

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "verification_method": "email",
                "verification_code": "valid123",
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        body = _body(result)
        assert body["request_id"].startswith("ccpa-del-user-42-")

    @pytest.mark.asyncio
    async def test_successful_deletion_contains_message(self, handler, _patch_stores):
        mock_del = MagicMock()
        mock_del.scheduled_for = datetime.now(timezone.utc) + timedelta(days=45)
        _patch_stores["scheduler"].schedule_deletion.return_value = mock_del

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "verification_method": "email",
                "verification_code": "valid123",
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        body = _body(result)
        assert "45 days" in body["message"]

    @pytest.mark.asyncio
    async def test_successful_deletion_contains_response_deadline(self, handler, _patch_stores):
        mock_del = MagicMock()
        mock_del.scheduled_for = datetime.now(timezone.utc) + timedelta(days=45)
        _patch_stores["scheduler"].schedule_deletion.return_value = mock_del

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "verification_method": "email",
                "verification_code": "valid123",
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        body = _body(result)
        assert "response_deadline" in body
        datetime.fromisoformat(body["response_deadline"])

    @pytest.mark.asyncio
    async def test_legal_hold_blocks_deletion(self, handler, _patch_stores):
        """User under legal hold returns 409."""
        _patch_stores["hold_manager"].is_user_on_hold.return_value = True

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "verification_method": "email",
                "verification_code": "valid123",
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        assert _status(result) == 409
        assert "legal hold" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_retain_exceptions_default_empty(self, handler, _patch_stores):
        mock_del = MagicMock()
        mock_del.scheduled_for = datetime.now(timezone.utc) + timedelta(days=45)
        _patch_stores["scheduler"].schedule_deletion.return_value = mock_del

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "verification_method": "email",
                "verification_code": "valid123",
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        body = _body(result)
        assert body["retained_categories"] == []

    @pytest.mark.asyncio
    async def test_retain_exceptions_with_values(self, handler, _patch_stores):
        mock_del = MagicMock()
        mock_del.scheduled_for = datetime.now(timezone.utc) + timedelta(days=45)
        _patch_stores["scheduler"].schedule_deletion.return_value = mock_del

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "verification_method": "email",
                "verification_code": "valid123",
                "retain_for_exceptions": ["security_incidents", "legal_obligations"],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        body = _body(result)
        assert body["retained_categories"] == ["security_incidents", "legal_obligations"]

    @pytest.mark.asyncio
    async def test_deletion_logs_audit_event(self, handler, _patch_stores):
        mock_del = MagicMock()
        mock_del.scheduled_for = datetime.now(timezone.utc) + timedelta(days=45)
        _patch_stores["scheduler"].schedule_deletion.return_value = mock_del

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "verification_method": "email",
                "verification_code": "valid123",
            },
        )
        await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        _patch_stores["audit_store"].log_event.assert_called()

    @pytest.mark.asyncio
    async def test_scheduler_error_returns_500(self, handler, _patch_stores):
        _patch_stores["scheduler"].schedule_deletion.side_effect = ValueError("DB down")

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "verification_method": "email",
                "verification_code": "valid123",
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_scheduler_os_error_returns_500(self, handler, _patch_stores):
        _patch_stores["scheduler"].schedule_deletion.side_effect = OSError("Disk full")

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "verification_method": "email",
                "verification_code": "valid123",
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        assert _status(result) == 500


# ============================================================================
# CCPA Opt-Out Endpoint - POST /api/v2/compliance/ccpa/opt-out
# ============================================================================


class TestCCPAOptOut:
    """Tests for _ccpa_opt_out method (Right to Opt-Out of Sale/Sharing)."""

    @pytest.mark.asyncio
    async def test_missing_user_id_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        assert _status(result) == 400
        assert "user_id" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_successful_opt_out_returns_confirmed(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42"})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "confirmed"

    @pytest.mark.asyncio
    async def test_response_contains_request_id(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42"})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        body = _body(result)
        assert body["request_id"].startswith("ccpa-opt-user-42-")

    @pytest.mark.asyncio
    async def test_response_contains_user_id(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42"})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        body = _body(result)
        assert body["user_id"] == "user-42"

    @pytest.mark.asyncio
    async def test_default_opt_out_type_both(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42"})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        body = _body(result)
        assert body["opt_out_type"] == "both"

    @pytest.mark.asyncio
    async def test_opt_out_type_sale(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42", "opt_out_type": "sale"})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        body = _body(result)
        assert body["opt_out_type"] == "sale"

    @pytest.mark.asyncio
    async def test_opt_out_type_sharing(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42", "opt_out_type": "sharing"})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        body = _body(result)
        assert body["opt_out_type"] == "sharing"

    @pytest.mark.asyncio
    async def test_default_sensitive_pi_limit_false(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42"})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        body = _body(result)
        assert body["sensitive_pi_limit"] is False

    @pytest.mark.asyncio
    async def test_sensitive_pi_limit_true(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={"user_id": "user-42", "sensitive_pi_limit": True},
        )
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        body = _body(result)
        assert body["sensitive_pi_limit"] is True

    @pytest.mark.asyncio
    async def test_sensitive_pi_limit_augments_message(self, handler):
        """When sensitive_pi_limit is True, message mentions sensitive PI."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={"user_id": "user-42", "sensitive_pi_limit": True},
        )
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        body = _body(result)
        assert "sensitive personal information" in body["message"]

    @pytest.mark.asyncio
    async def test_message_without_sensitive_pi(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42"})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        body = _body(result)
        assert "will not sell or share" in body["message"]
        assert "sensitive personal information" not in body["message"]

    @pytest.mark.asyncio
    async def test_response_contains_effective_at(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42"})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        body = _body(result)
        assert "effective_at" in body
        datetime.fromisoformat(body["effective_at"])

    @pytest.mark.asyncio
    async def test_logs_ccpa_opt_out(self, handler, _patch_stores):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42"})
        await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        _patch_stores["audit_store"].log_event.assert_called()

    @pytest.mark.asyncio
    async def test_stores_preference(self, handler, _patch_stores):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42"})
        await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        # _store_ccpa_preference calls log_event on audit store
        assert _patch_stores["audit_store"].log_event.call_count >= 1

    @pytest.mark.asyncio
    async def test_store_preference_error_still_succeeds(self, handler, _patch_stores):
        """If _store_ccpa_preference fails, opt-out still succeeds (graceful degradation)."""
        _patch_stores["audit_store"].log_event.side_effect = ValueError("Store error")

        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42"})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        # _store_ccpa_preference and _log_ccpa_request catch errors internally
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "confirmed"


# ============================================================================
# CCPA Correct Endpoint - POST /api/v2/compliance/ccpa/correct
# ============================================================================


class TestCCPACorrect:
    """Tests for _ccpa_correct method (Right to Correct)."""

    @pytest.mark.asyncio
    async def test_missing_user_id_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={"corrections": [{"field": "name", "corrected_value": "Jane"}]},
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        assert _status(result) == 400
        assert "user_id" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_missing_corrections_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42"})
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        assert _status(result) == 400
        assert "corrections" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_empty_corrections_list_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "user-42", "corrections": []})
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_body_returns_400(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_successful_correction_returns_200(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [
                    {"field": "name", "current_value": "John", "corrected_value": "Jane"},
                ],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_response_contains_request_id(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [{"field": "name", "corrected_value": "Jane"}],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        body = _body(result)
        assert body["request_id"].startswith("ccpa-corr-user-42-")

    @pytest.mark.asyncio
    async def test_response_status_pending_review(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [{"field": "name", "corrected_value": "Jane"}],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        body = _body(result)
        assert body["status"] == "pending_review"

    @pytest.mark.asyncio
    async def test_response_contains_corrections_count(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [
                    {"field": "name", "corrected_value": "Jane"},
                    {"field": "email", "corrected_value": "jane@example.com"},
                ],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        body = _body(result)
        assert body["corrections_requested"] == 2

    @pytest.mark.asyncio
    async def test_corrections_redact_current_value(self, handler):
        """Current value should be redacted in the response."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [
                    {"field": "name", "current_value": "John Doe", "corrected_value": "Jane"},
                ],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        body = _body(result)
        assert body["corrections"][0]["current_value"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_corrections_have_pending_status(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [{"field": "name", "corrected_value": "Jane"}],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        body = _body(result)
        assert body["corrections"][0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_corrections_preserve_field_name(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [{"field": "email", "corrected_value": "new@example.com"}],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        body = _body(result)
        assert body["corrections"][0]["field"] == "email"

    @pytest.mark.asyncio
    async def test_response_contains_deadline(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [{"field": "name", "corrected_value": "Jane"}],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        body = _body(result)
        assert "response_deadline" in body

    @pytest.mark.asyncio
    async def test_response_deadline_is_45_days(self, handler):
        """CCPA/CPRA requires 45-day response window."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [{"field": "name", "corrected_value": "Jane"}],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        body = _body(result)
        requested = datetime.fromisoformat(body["requested_at"])
        deadline = datetime.fromisoformat(body["response_deadline"])
        delta = deadline - requested
        assert delta.days == 45

    @pytest.mark.asyncio
    async def test_response_message(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [{"field": "name", "corrected_value": "Jane"}],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        body = _body(result)
        assert "45 days" in body["message"]
        assert "CCPA/CPRA" in body["message"]

    @pytest.mark.asyncio
    async def test_logs_correction_request(self, handler, _patch_stores):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [{"field": "name", "corrected_value": "Jane"}],
            },
        )
        await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        _patch_stores["audit_store"].log_event.assert_called()
        call_kwargs = _patch_stores["audit_store"].log_event.call_args[1]
        assert call_kwargs["action"] == "ccpa_correction_request"

    @pytest.mark.asyncio
    async def test_multiple_corrections(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "user-42",
                "corrections": [
                    {"field": "name", "corrected_value": "Jane"},
                    {"field": "email", "corrected_value": "jane@example.com"},
                    {"field": "phone", "corrected_value": "555-1234"},
                ],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        body = _body(result)
        assert len(body["corrections"]) == 3
        assert body["corrections_requested"] == 3


# ============================================================================
# CCPA Status Endpoint - GET /api/v2/compliance/ccpa/status
# ============================================================================


class TestCCPAStatus:
    """Tests for _ccpa_get_status method."""

    @pytest.mark.asyncio
    async def test_missing_user_id_returns_400(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/ccpa/status", {}, mock_h)
        assert _status(result) == 400
        assert "user_id" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_returns_200(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/status", {"user_id": "user-42"}, mock_h
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_response_contains_user_id(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/status", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert body["user_id"] == "user-42"

    @pytest.mark.asyncio
    async def test_response_contains_requests_and_count(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/status", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert "requests" in body
        assert "count" in body
        assert isinstance(body["requests"], list)

    @pytest.mark.asyncio
    async def test_empty_requests_returns_zero_count(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/status", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert body["count"] == 0
        assert body["requests"] == []

    @pytest.mark.asyncio
    async def test_filters_ccpa_requests(self, handler, _patch_stores):
        _patch_stores["audit_store"].get_log.return_value = [
            {
                "resource_id": "user-42",
                "action": "ccpa_disclosure_request",
                "timestamp": "2026-01-01T00:00:00Z",
                "metadata": {"request_id": "ccpa-disc-1", "status": "completed"},
            },
            {
                "resource_id": "user-42",
                "action": "login",
                "timestamp": "2026-01-01T00:00:00Z",
                "metadata": {},
            },
        ]
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/status", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["requests"][0]["request_id"] == "ccpa-disc-1"

    @pytest.mark.asyncio
    async def test_filters_by_request_type(self, handler, _patch_stores):
        _patch_stores["audit_store"].get_log.return_value = [
            {
                "resource_id": "user-42",
                "action": "ccpa_disclosure_request",
                "timestamp": "2026-01-01T00:00:00Z",
                "metadata": {"request_id": "ccpa-disc-1"},
            },
            {
                "resource_id": "user-42",
                "action": "ccpa_deletion_request",
                "timestamp": "2026-01-02T00:00:00Z",
                "metadata": {"request_id": "ccpa-del-1"},
            },
        ]
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/status",
            {"user_id": "user-42", "request_type": "deletion"},
            mock_h,
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["requests"][0]["request_id"] == "ccpa-del-1"

    @pytest.mark.asyncio
    async def test_filters_by_user_id(self, handler, _patch_stores):
        _patch_stores["audit_store"].get_log.return_value = [
            {
                "resource_id": "user-42",
                "action": "ccpa_disclosure_request",
                "timestamp": "2026-01-01T00:00:00Z",
                "metadata": {"request_id": "ccpa-disc-1"},
            },
            {
                "resource_id": "user-99",
                "action": "ccpa_disclosure_request",
                "timestamp": "2026-01-01T00:00:00Z",
                "metadata": {"request_id": "ccpa-disc-2"},
            },
        ]
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/status", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["requests"][0]["request_id"] == "ccpa-disc-1"

    @pytest.mark.asyncio
    async def test_store_error_returns_500(self, handler, _patch_stores):
        _patch_stores["audit_store"].get_log.side_effect = ValueError("DB down")

        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/status", {"user_id": "user-42"}, mock_h
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_os_error_returns_500(self, handler, _patch_stores):
        _patch_stores["audit_store"].get_log.side_effect = OSError("Disk error")

        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/status", {"user_id": "user-42"}, mock_h
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_request_type_extraction(self, handler, _patch_stores):
        """CCPA request type is extracted from action string."""
        _patch_stores["audit_store"].get_log.return_value = [
            {
                "resource_id": "user-42",
                "action": "ccpa_opt_out_request",
                "timestamp": "2026-01-01T00:00:00Z",
                "metadata": {"request_id": "ccpa-opt-1"},
            },
        ]
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/status", {"user_id": "user-42"}, mock_h
        )
        body = _body(result)
        # action "ccpa_opt_out_request" -> type "opt_out"
        assert body["requests"][0]["type"] == "opt_out"


# ============================================================================
# Route Dispatch Tests (CCPA-specific)
# ============================================================================


class TestCCPARouteDispatch:
    """Verify correct routing of CCPA endpoints."""

    @pytest.mark.asyncio
    async def test_disclosure_wrong_method_returns_404(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle("/api/v2/compliance/ccpa/disclosure", {}, mock_h)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_delete_get_returns_404(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_opt_out_get_returns_404(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_correct_get_returns_404(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_status_post_returns_404(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = await handler.handle("/api/v2/compliance/ccpa/status", {}, mock_h)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_unknown_ccpa_subpath_returns_404(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/ccpa/nonexistent", {}, mock_h)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_disclosure_get_dispatches(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle(
            "/api/v2/compliance/ccpa/disclosure", {"user_id": "u1"}, mock_h
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_delete_post_dispatches(self, handler, _patch_stores):
        mock_del = MagicMock()
        mock_del.scheduled_for = datetime.now(timezone.utc) + timedelta(days=45)
        _patch_stores["scheduler"].schedule_deletion.return_value = mock_del

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "u1",
                "verification_method": "email",
                "verification_code": "code",
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/delete", {}, mock_h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_opt_out_post_dispatches(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"user_id": "u1"})
        result = await handler.handle("/api/v2/compliance/ccpa/opt-out", {}, mock_h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_correct_post_dispatches(self, handler):
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "user_id": "u1",
                "corrections": [{"field": "name", "corrected_value": "Jane"}],
            },
        )
        result = await handler.handle("/api/v2/compliance/ccpa/correct", {}, mock_h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_status_get_dispatches(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = await handler.handle("/api/v2/compliance/ccpa/status", {"user_id": "u1"}, mock_h)
        assert _status(result) == 200


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestCCPAHelperMethods:
    """Tests for internal helper methods on the CCPAMixin."""

    @pytest.mark.asyncio
    async def test_get_ccpa_categories_returns_all(self, handler):
        result = await handler._get_ccpa_categories("user-42")
        assert isinstance(result, list)
        assert len(result) == len(handler.CCPA_CATEGORIES)

    @pytest.mark.asyncio
    async def test_get_ccpa_categories_collected_flags(self, handler):
        """Categories A, D, F, K are collected; B, C, G are not."""
        result = await handler._get_ccpa_categories("user-42")
        collected_map = {cat["category"]: cat.get("collected", False) for cat in result}
        assert collected_map["A"] is True
        assert collected_map["D"] is True
        assert collected_map["F"] is True
        assert collected_map["K"] is True
        assert collected_map["B"] is False
        assert collected_map["C"] is False
        assert collected_map["G"] is False

    @pytest.mark.asyncio
    async def test_get_ccpa_categories_collected_have_examples(self, handler):
        """Collected categories include examples (up to 3)."""
        result = await handler._get_ccpa_categories("user-42")
        for cat in result:
            if cat.get("collected"):
                assert "examples" in cat
                assert len(cat["examples"]) <= 3

    @pytest.mark.asyncio
    async def test_get_ccpa_categories_not_collected_no_examples(self, handler):
        """Non-collected categories do not include examples."""
        result = await handler._get_ccpa_categories("user-42")
        for cat in result:
            if not cat.get("collected"):
                assert "examples" not in cat

    @pytest.mark.asyncio
    async def test_get_categories_disclosed(self, handler):
        result = await handler._get_categories_disclosed("user-42")
        assert isinstance(result, list)
        assert len(result) == 2
        categories = [c["category"] for c in result]
        assert "A" in categories
        assert "F" in categories

    @pytest.mark.asyncio
    async def test_get_categories_disclosed_structure(self, handler):
        result = await handler._get_categories_disclosed("user-42")
        for entry in result:
            assert "category" in entry
            assert "name" in entry
            assert "disclosed_to" in entry
            assert "purpose" in entry

    @pytest.mark.asyncio
    async def test_get_business_purposes(self, handler):
        result = await handler._get_business_purposes()
        assert isinstance(result, list)
        assert len(result) == 4
        purposes = [p["purpose"] for p in result]
        assert "Service delivery" in purposes
        assert "Security" in purposes
        assert "Improvement" in purposes
        assert "Communication" in purposes

    @pytest.mark.asyncio
    async def test_get_business_purposes_structure(self, handler):
        result = await handler._get_business_purposes()
        for p in result:
            assert "purpose" in p
            assert "description" in p

    @pytest.mark.asyncio
    async def test_get_specific_pi(self, handler):
        result = await handler._get_specific_pi("user-42")
        assert isinstance(result, dict)
        assert "account_information" in result
        assert "activity_records" in result
        assert "preferences" in result

    @pytest.mark.asyncio
    async def test_get_specific_pi_account_info(self, handler):
        result = await handler._get_specific_pi("user-42")
        assert result["account_information"]["user_id"] == "user-42"
        assert result["account_information"]["account_type"] == "standard"

    @pytest.mark.asyncio
    async def test_get_specific_pi_receipt_store_error(self, handler, _patch_stores):
        """Receipt store error returns error dict."""
        _patch_stores["receipt_store"].list.side_effect = ValueError("DB error")
        result = await handler._get_specific_pi("user-42")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_pi_sources(self, handler):
        result = await handler._get_pi_sources("user-42")
        assert isinstance(result, list)
        assert len(result) == 3
        sources = [s["source"] for s in result]
        assert "Direct collection" in sources
        assert "Automatic collection" in sources
        assert "Third-party authentication" in sources

    @pytest.mark.asyncio
    async def test_get_third_party_disclosures(self, handler):
        result = await handler._get_third_party_disclosures("user-42")
        assert isinstance(result, list)
        assert len(result) == 2
        party_types = [p["party_type"] for p in result]
        assert "Service providers" in party_types
        assert "AI model providers" in party_types

    @pytest.mark.asyncio
    async def test_get_third_party_disclosures_ai_note(self, handler):
        result = await handler._get_third_party_disclosures("user-42")
        ai_party = next(p for p in result if p["party_type"] == "AI model providers")
        assert "anonymized" in ai_party["note"].lower()

    @pytest.mark.asyncio
    async def test_verify_ccpa_request_returns_true(self, handler):
        """Placeholder verification always returns True."""
        result = await handler._verify_ccpa_request("user-42", "email", "code")
        assert result is True

    @pytest.mark.asyncio
    async def test_store_ccpa_preference(self, handler, _patch_stores):
        await handler._store_ccpa_preference("user-42", {"key": "value"})
        _patch_stores["audit_store"].log_event.assert_called_once()
        call_kwargs = _patch_stores["audit_store"].log_event.call_args[1]
        assert call_kwargs["action"] == "ccpa_preference_stored"
        assert call_kwargs["resource_id"] == "user-42"

    @pytest.mark.asyncio
    async def test_store_ccpa_preference_error_handled(self, handler, _patch_stores):
        """Store error is silently handled (logged, not raised)."""
        _patch_stores["audit_store"].log_event.side_effect = OSError("Disk full")
        # Should not raise
        await handler._store_ccpa_preference("user-42", {"key": "value"})

    @pytest.mark.asyncio
    async def test_log_ccpa_request(self, handler, _patch_stores):
        await handler._log_ccpa_request(
            request_type="disclosure",
            request_id="ccpa-disc-1",
            user_id="user-42",
            disclosure_type="categories",
        )
        _patch_stores["audit_store"].log_event.assert_called_once()
        call_kwargs = _patch_stores["audit_store"].log_event.call_args[1]
        assert call_kwargs["action"] == "ccpa_disclosure_request"
        assert call_kwargs["resource_type"] == "user"
        assert call_kwargs["resource_id"] == "user-42"
        assert call_kwargs["metadata"]["request_type"] == "disclosure"
        assert call_kwargs["metadata"]["disclosure_type"] == "categories"

    @pytest.mark.asyncio
    async def test_log_ccpa_request_error_handled(self, handler, _patch_stores):
        """Logging error is silently handled."""
        _patch_stores["audit_store"].log_event.side_effect = KeyError("missing")
        # Should not raise
        await handler._log_ccpa_request(
            request_type="deletion",
            request_id="ccpa-del-1",
            user_id="user-42",
        )


# ============================================================================
# CCPA Mixin Constants
# ============================================================================


class TestCCPAMixinConstants:
    """Verify CCPA mixin static data is correctly defined."""

    def test_ccpa_categories_count(self, handler):
        assert len(handler.CCPA_CATEGORIES) == 7

    def test_ccpa_categories_have_required_fields(self, handler):
        for cat in handler.CCPA_CATEGORIES:
            assert "category" in cat
            assert "name" in cat
            assert "examples" in cat

    def test_ccpa_categories_include_identifiers(self, handler):
        names = [cat["name"] for cat in handler.CCPA_CATEGORIES]
        assert "Identifiers" in names

    def test_ccpa_categories_include_commercial(self, handler):
        names = [cat["name"] for cat in handler.CCPA_CATEGORIES]
        assert "Commercial Information" in names

    def test_ccpa_categories_include_internet_activity(self, handler):
        names = [cat["name"] for cat in handler.CCPA_CATEGORIES]
        assert "Internet or Network Activity" in names

    def test_ccpa_categories_include_inferences(self, handler):
        names = [cat["name"] for cat in handler.CCPA_CATEGORIES]
        assert "Inferences" in names

    def test_ccpa_categories_include_geolocation(self, handler):
        names = [cat["name"] for cat in handler.CCPA_CATEGORIES]
        assert "Geolocation Data" in names

    def test_ccpa_category_letters(self, handler):
        """Verify CCPA category letter identifiers."""
        letters = {cat["category"] for cat in handler.CCPA_CATEGORIES}
        assert letters == {"A", "B", "C", "D", "F", "G", "K"}

    def test_ccpa_categories_examples_non_empty(self, handler):
        for cat in handler.CCPA_CATEGORIES:
            assert len(cat["examples"]) > 0

    def test_identifiers_category_examples(self, handler):
        cat_a = next(c for c in handler.CCPA_CATEGORIES if c["category"] == "A")
        assert "Real name" in cat_a["examples"]
        assert "Email address" in cat_a["examples"]
        assert "IP address" in cat_a["examples"]
