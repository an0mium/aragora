"""Tests for the ComplianceReportHandler.

Covers all routes:
- GET  /api/v1/compliance/reports/:id          (retrieve cached report)
- GET  /api/v1/compliance/reports/:id/download  (download in json/html/markdown)
- POST /api/v1/compliance/reports/generate      (generate report for framework+debate)

Includes happy paths, error cases, edge cases, and validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.compliance_reports import (
    ComplianceReportHandler,
    _report_cache,
)


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


def _raw_body(result) -> bytes:
    """Extract raw body bytes from a HandlerResult."""
    if isinstance(result, dict):
        return b""
    return result.body


def _content_type(result) -> str:
    """Extract content type from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("content_type", "")
    return result.content_type


def _headers(result) -> dict:
    """Extract headers from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("headers", {})
    return result.headers or {}


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to handle/handle_post."""

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
# Fake report objects to populate cache
# ---------------------------------------------------------------------------


class _FakeFramework(Enum):
    SOC2 = "soc2"
    GENERAL = "general"


@dataclass
class _FakeReport:
    report_id: str
    debate_id: str
    framework: _FakeFramework
    generated_at: datetime
    generated_by: str = "Test System"
    summary: str = "Test summary"
    sections: list[Any] = field(default_factory=list)
    attestation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "debate_id": self.debate_id,
            "framework": self.framework.value,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by,
            "summary": self.summary,
            "sections": [],
            "attestation": self.attestation,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Patch helpers -- imports are local in the handler, so we patch at origin
# ---------------------------------------------------------------------------

_PATCH_GENERATOR = "aragora.compliance.report_generator.ComplianceReportGenerator"
_PATCH_FRAMEWORK = "aragora.compliance.report_generator.ComplianceFramework"
_PATCH_DEBATE_RESULT = "aragora.core.DebateResult"


def _make_mock_report(
    report_id: str = "CR-GENERATED001",
    debate_id: str = "debate-001",
    framework_value: str = "general",
    summary: str = "Decision summary",
) -> MagicMock:
    """Create a mock ComplianceReport object."""
    mock_report = MagicMock()
    mock_report.report_id = report_id
    mock_report.debate_id = debate_id
    mock_report.framework.value = framework_value
    mock_report.generated_at.isoformat.return_value = "2026-01-15T12:00:00"
    mock_report.summary = summary
    return mock_report


def _make_generator_mock(report: MagicMock | None = None) -> MagicMock:
    """Create a mock ComplianceReportGenerator class."""
    gen_instance = MagicMock()
    if report is not None:
        gen_instance.generate.return_value = report
    gen_instance.export_json.return_value = '{"report_id": "test"}'
    gen_instance.export_markdown.return_value = "# Report\n\nSummary here."
    mock_cls = MagicMock(return_value=gen_instance)
    return mock_cls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a ComplianceReportHandler with minimal ctx."""
    return ComplianceReportHandler(ctx={})


@pytest.fixture(autouse=True)
def _clear_report_cache():
    """Clear the global report cache before and after each test."""
    _report_cache.clear()
    yield
    _report_cache.clear()


@pytest.fixture
def cached_report():
    """Insert a fake report into the cache and return it."""
    report = _FakeReport(
        report_id="CR-ABCDEF123456",
        debate_id="debate-001",
        framework=_FakeFramework.GENERAL,
        generated_at=datetime(2026, 1, 15, 12, 0, 0),
    )
    _report_cache["CR-ABCDEF123456"] = report
    return report


@pytest.fixture
def mock_storage():
    """Create a mock storage with a default debate."""
    storage = MagicMock()
    storage.get_debate.return_value = {
        "task": "Should we migrate to microservices?",
        "consensus_reached": True,
        "rounds_used": 3,
        "winner": "claude",
        "final_answer": "Yes, with careful planning.",
    }
    return storage


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    """Verify that can_handle correctly identifies matching paths."""

    def test_reports_root(self, handler):
        assert handler.can_handle("/api/v1/compliance/reports") is True

    def test_reports_with_id(self, handler):
        assert handler.can_handle("/api/v1/compliance/reports/CR-ABC123") is True

    def test_reports_download(self, handler):
        assert handler.can_handle("/api/v1/compliance/reports/CR-ABC123/download") is True

    def test_reports_generate(self, handler):
        assert handler.can_handle("/api/v1/compliance/reports/generate") is True

    def test_unrelated_compliance_path(self, handler):
        assert handler.can_handle("/api/v1/compliance/status") is False

    def test_different_api(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_match(self, handler):
        """Paths that share prefix but not /reports should not match."""
        assert handler.can_handle("/api/v1/compliance/report") is False

    def test_longer_valid_path(self, handler):
        assert handler.can_handle("/api/v1/compliance/reports/x/y/z") is True


# ===========================================================================
# GET /api/v1/compliance/reports/:id  (retrieve report)
# ===========================================================================


class TestGetReport:
    """GET /api/v1/compliance/reports/:id -- retrieve a cached report."""

    def test_get_existing_report(self, handler, cached_report):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/compliance/reports/CR-ABCDEF123456", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["report_id"] == "CR-ABCDEF123456"
        assert body["debate_id"] == "debate-001"
        assert body["framework"] == "general"

    def test_get_nonexistent_report(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/compliance/reports/CR-DOESNOTEXIST", {}, mock_h)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    def test_get_report_returns_full_dict(self, handler, cached_report):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/compliance/reports/CR-ABCDEF123456", {}, mock_h)
        body = _body(result)
        assert "generated_at" in body
        assert "summary" in body
        assert body["summary"] == "Test summary"
        assert "sections" in body
        assert "attestation" in body
        assert "metadata" in body

    def test_get_report_different_ids_all_404(self, handler):
        """Different IDs each produce 404 when not cached."""
        mock_h = _MockHTTPHandler("GET")
        for rid in ["CR-111", "CR-222", "CR-333"]:
            result = handler.handle(f"/api/v1/compliance/reports/{rid}", {}, mock_h)
            assert _status(result) == 404

    def test_get_report_id_case_sensitive(self, handler, cached_report):
        """Report IDs are case-sensitive."""
        mock_h = _MockHTTPHandler("GET")
        # The cached report has "CR-ABCDEF123456", lowercase should not match
        result = handler.handle("/api/v1/compliance/reports/cr-abcdef123456", {}, mock_h)
        assert _status(result) == 404


# ===========================================================================
# GET /api/v1/compliance/reports/:id/download  (download report)
# ===========================================================================


class TestDownloadReport:
    """GET /api/v1/compliance/reports/:id/download?format=..."""

    def test_download_json_format(self, handler, cached_report):
        mock_h = _MockHTTPHandler("GET")
        with patch(_PATCH_GENERATOR) as MockGen:
            gen_instance = MagicMock()
            gen_instance.export_json.return_value = '{"report_id": "CR-ABCDEF123456"}'
            MockGen.return_value = gen_instance

            result = handler.handle(
                "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                {"format": "json"},
                mock_h,
            )
            assert _status(result) == 200
            assert _content_type(result) == "application/json"
            assert b"CR-ABCDEF123456" in _raw_body(result)
            headers = _headers(result)
            assert "Content-Disposition" in headers
            assert "report-CR-ABCDEF123456.json" in headers["Content-Disposition"]

    def test_download_markdown_format(self, handler, cached_report):
        mock_h = _MockHTTPHandler("GET")
        with patch(_PATCH_GENERATOR) as MockGen:
            gen_instance = MagicMock()
            gen_instance.export_markdown.return_value = "# Report\n\nSummary here."
            MockGen.return_value = gen_instance

            result = handler.handle(
                "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                {"format": "markdown"},
                mock_h,
            )
            assert _status(result) == 200
            assert "text/markdown" in _content_type(result)
            assert b"# Report" in _raw_body(result)
            headers = _headers(result)
            assert "report-CR-ABCDEF123456.md" in headers["Content-Disposition"]

    def test_download_html_format(self, handler, cached_report):
        mock_h = _MockHTTPHandler("GET")
        with patch(_PATCH_GENERATOR) as MockGen:
            gen_instance = MagicMock()
            gen_instance.export_markdown.return_value = "# Report\n\nContent"
            MockGen.return_value = gen_instance

            result = handler.handle(
                "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                {"format": "html"},
                mock_h,
            )
            assert _status(result) == 200
            assert "text/html" in _content_type(result)
            body_bytes = _raw_body(result)
            assert b"<html>" in body_bytes
            assert b"<pre>" in body_bytes
            assert b"# Report" in body_bytes
            headers = _headers(result)
            assert "report-CR-ABCDEF123456.html" in headers["Content-Disposition"]

    def test_download_html_contains_title(self, handler, cached_report):
        """HTML download wraps content with <title> containing report_id."""
        mock_h = _MockHTTPHandler("GET")
        with patch(_PATCH_GENERATOR) as MockGen:
            gen_instance = MagicMock()
            gen_instance.export_markdown.return_value = "content"
            MockGen.return_value = gen_instance

            result = handler.handle(
                "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                {"format": "html"},
                mock_h,
            )
            body_text = _raw_body(result).decode("utf-8")
            assert "<title>Report CR-ABCDEF123456</title>" in body_text

    def test_download_default_format_is_json(self, handler, cached_report):
        """When no format param is specified, default is json."""
        mock_h = _MockHTTPHandler("GET")
        with patch(_PATCH_GENERATOR) as MockGen:
            gen_instance = MagicMock()
            gen_instance.export_json.return_value = '{"ok": true}'
            MockGen.return_value = gen_instance

            result = handler.handle(
                "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                {},
                mock_h,
            )
            assert _status(result) == 200
            assert _content_type(result) == "application/json"

    def test_download_invalid_format_pdf(self, handler, cached_report):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle(
            "/api/v1/compliance/reports/CR-ABCDEF123456/download",
            {"format": "pdf"},
            mock_h,
        )
        assert _status(result) == 400
        body = _body(result)
        error_msg = body.get("error", "").lower()
        assert "invalid format" in error_msg or "pdf" in error_msg

    def test_download_nonexistent_report(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle(
            "/api/v1/compliance/reports/CR-MISSING/download",
            {"format": "json"},
            mock_h,
        )
        assert _status(result) == 404

    def test_download_import_error(self, handler, cached_report):
        """If ComplianceReportGenerator cannot be imported, return 503."""
        mock_h = _MockHTTPHandler("GET")
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "report_generator" in name:
                raise ImportError("not available")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = handler.handle(
                "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                {"format": "json"},
                mock_h,
            )
            assert _status(result) == 503

    def test_download_various_valid_formats(self, handler, cached_report):
        """All three valid formats are accepted."""
        mock_h = _MockHTTPHandler("GET")
        for fmt in ["json", "html", "markdown"]:
            with patch(_PATCH_GENERATOR) as MockGen:
                gen_instance = MagicMock()
                gen_instance.export_json.return_value = '{"ok": true}'
                gen_instance.export_markdown.return_value = "# Report"
                MockGen.return_value = gen_instance

                result = handler.handle(
                    "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                    {"format": fmt},
                    mock_h,
                )
                assert _status(result) == 200, f"Format {fmt} should return 200"

    def test_download_invalid_format_names(self, handler, cached_report):
        """Various invalid format names should return 400."""
        mock_h = _MockHTTPHandler("GET")
        for fmt in ["pdf", "csv", "txt", "XML", "JSON", ""]:
            result = handler.handle(
                "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                {"format": fmt},
                mock_h,
            )
            assert _status(result) == 400, f"Format '{fmt}' should be rejected"

    def test_download_json_content_disposition(self, handler, cached_report):
        """JSON download has proper Content-Disposition attachment header."""
        mock_h = _MockHTTPHandler("GET")
        with patch(_PATCH_GENERATOR) as MockGen:
            gen_instance = MagicMock()
            gen_instance.export_json.return_value = "{}"
            MockGen.return_value = gen_instance

            result = handler.handle(
                "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                {"format": "json"},
                mock_h,
            )
            headers = _headers(result)
            assert headers["Content-Disposition"].startswith("attachment")

    def test_download_markdown_content_type_includes_charset(self, handler, cached_report):
        """Markdown download includes charset in content type."""
        mock_h = _MockHTTPHandler("GET")
        with patch(_PATCH_GENERATOR) as MockGen:
            gen_instance = MagicMock()
            gen_instance.export_markdown.return_value = "# test"
            MockGen.return_value = gen_instance

            result = handler.handle(
                "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                {"format": "markdown"},
                mock_h,
            )
            assert "charset=utf-8" in _content_type(result)


# ===========================================================================
# POST /api/v1/compliance/reports/generate  (generate report)
# ===========================================================================


class TestGenerateReport:
    """POST /api/v1/compliance/reports/generate."""

    def test_generate_success(self, handler, mock_storage):
        """Happy path: generate a report for a valid debate."""
        handler.ctx = {"storage": mock_storage}

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "debate_id": "debate-001",
                "framework": "general",
                "scope": {"include_evidence": True, "include_chain": False},
            },
        )

        mock_report = _make_mock_report()
        with (
            patch(_PATCH_GENERATOR) as MockGen,
            patch(_PATCH_FRAMEWORK),
            patch(_PATCH_DEBATE_RESULT),
        ):
            gen_instance = MagicMock()
            gen_instance.generate.return_value = mock_report
            MockGen.return_value = gen_instance

            result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
            assert _status(result) == 201
            body = _body(result)
            assert body["report_id"] == "CR-GENERATED001"
            assert body["debate_id"] == "debate-001"
            assert body["framework"] == "general"
            assert body["summary"] == "Decision summary"
            assert "generated_at" in body

            # Report should be cached
            assert "CR-GENERATED001" in _report_cache

    def test_generate_missing_debate_id(self, handler):
        """Missing debate_id should return 400."""
        mock_h = _MockHTTPHandler("POST", body={"framework": "soc2"})
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
        assert _status(result) == 400
        body = _body(result)
        assert "debate_id" in body.get("error", "").lower()

    def test_generate_invalid_framework(self, handler):
        """Invalid framework name should return 400."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={"debate_id": "debate-001", "framework": "invalid_fw"},
        )
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
        assert _status(result) == 400
        body = _body(result)
        error_msg = body.get("error", "").lower()
        assert "invalid framework" in error_msg or "invalid_fw" in error_msg

    def test_generate_debate_not_found(self, handler, mock_storage):
        """When storage returns no debate, return 404."""
        mock_storage.get_debate.return_value = None
        handler.ctx = {"storage": mock_storage}

        mock_h = _MockHTTPHandler(
            "POST",
            body={"debate_id": "nonexistent", "framework": "general"},
        )

        with patch(_PATCH_GENERATOR), patch(_PATCH_FRAMEWORK):
            result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
            assert _status(result) == 404
            body = _body(result)
            assert "not found" in body.get("error", "").lower()

    def test_generate_no_storage_returns_404(self, handler):
        """When no storage in ctx, debate_data is None, should return 404."""
        handler.ctx = {}
        mock_h = _MockHTTPHandler(
            "POST",
            body={"debate_id": "debate-001", "framework": "general"},
        )

        with patch(_PATCH_GENERATOR), patch(_PATCH_FRAMEWORK):
            result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
            assert _status(result) == 404

    def test_generate_import_error(self, handler):
        """If compliance module cannot be imported, return 503."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={"debate_id": "debate-001", "framework": "general"},
        )

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "report_generator" in name:
                raise ImportError("not available")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
            assert _status(result) == 503

    def test_generate_wrong_path_returns_none(self, handler):
        """POST to a non-generate path should return None."""
        mock_h = _MockHTTPHandler("POST", body={"debate_id": "d1"})
        result = handler.handle_post("/api/v1/compliance/reports/something-else", {}, mock_h)
        assert result is None

    def test_generate_all_valid_frameworks(self, handler, mock_storage):
        """All 6 valid framework names should be accepted."""
        valid_fws = ["soc2", "gdpr", "hipaa", "iso27001", "general", "custom"]
        handler.ctx = {"storage": mock_storage}

        for fw in valid_fws:
            mock_h = _MockHTTPHandler(
                "POST",
                body={"debate_id": f"debate-{fw}", "framework": fw},
            )

            mock_report = _make_mock_report(
                report_id=f"CR-{fw.upper()}",
                debate_id=f"debate-{fw}",
                framework_value=fw,
                summary=f"{fw} summary",
            )
            with (
                patch(_PATCH_GENERATOR) as MockGen,
                patch(_PATCH_FRAMEWORK),
                patch(_PATCH_DEBATE_RESULT),
            ):
                gen_instance = MagicMock()
                gen_instance.generate.return_value = mock_report
                MockGen.return_value = gen_instance

                result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
                assert _status(result) == 201, f"Framework '{fw}' should succeed"

    def test_generate_defaults_framework_to_general(self, handler, mock_storage):
        """When framework is omitted from body, defaults to 'general'."""
        handler.ctx = {"storage": mock_storage}

        mock_h = _MockHTTPHandler(
            "POST",
            body={"debate_id": "debate-default"},
        )

        mock_report = _make_mock_report(
            report_id="CR-DEFAULT",
            debate_id="debate-default",
            framework_value="general",
            summary="Default framework",
        )
        with (
            patch(_PATCH_GENERATOR) as MockGen,
            patch(_PATCH_FRAMEWORK),
            patch(_PATCH_DEBATE_RESULT),
        ):
            gen_instance = MagicMock()
            gen_instance.generate.return_value = mock_report
            MockGen.return_value = gen_instance

            result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
            assert _status(result) == 201
            body = _body(result)
            assert body["framework"] == "general"

    def test_generate_defaults_scope_to_empty(self, handler, mock_storage):
        """When scope is omitted, defaults apply: include_evidence=True, include_chain=True."""
        handler.ctx = {"storage": mock_storage}

        mock_h = _MockHTTPHandler(
            "POST",
            body={"debate_id": "debate-scope", "framework": "soc2"},
        )

        mock_report = _make_mock_report(report_id="CR-SCOPE")
        with (
            patch(_PATCH_GENERATOR) as MockGen,
            patch(_PATCH_FRAMEWORK),
            patch(_PATCH_DEBATE_RESULT),
        ):
            gen_instance = MagicMock()
            gen_instance.generate.return_value = mock_report
            MockGen.return_value = gen_instance

            result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
            assert _status(result) == 201

            # Verify generate was called with correct defaults
            call_kwargs = gen_instance.generate.call_args
            assert call_kwargs[1]["include_evidence"] is True
            assert call_kwargs[1]["include_chain"] is True
            assert call_kwargs[1]["include_full_transcript"] is False

    def test_generate_with_custom_scope(self, handler, mock_storage):
        """Scope values are passed through to generator."""
        handler.ctx = {"storage": mock_storage}

        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "debate_id": "debate-custom",
                "framework": "gdpr",
                "scope": {
                    "include_evidence": False,
                    "include_chain": False,
                    "include_transcript": True,
                },
            },
        )

        mock_report = _make_mock_report(report_id="CR-CUSTOM")
        with (
            patch(_PATCH_GENERATOR) as MockGen,
            patch(_PATCH_FRAMEWORK),
            patch(_PATCH_DEBATE_RESULT),
        ):
            gen_instance = MagicMock()
            gen_instance.generate.return_value = mock_report
            MockGen.return_value = gen_instance

            result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
            assert _status(result) == 201

            call_kwargs = gen_instance.generate.call_args
            assert call_kwargs[1]["include_evidence"] is False
            assert call_kwargs[1]["include_chain"] is False
            assert call_kwargs[1]["include_full_transcript"] is True

    def test_generate_empty_body(self, handler):
        """Empty body means no debate_id -> 400."""
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
        assert _status(result) == 400

    def test_generate_caches_report(self, handler, mock_storage):
        """After successful generation, the report is stored in _report_cache."""
        handler.ctx = {"storage": mock_storage}

        mock_h = _MockHTTPHandler(
            "POST",
            body={"debate_id": "debate-cache", "framework": "general"},
        )

        mock_report = _make_mock_report(report_id="CR-CACHED")
        with (
            patch(_PATCH_GENERATOR) as MockGen,
            patch(_PATCH_FRAMEWORK),
            patch(_PATCH_DEBATE_RESULT),
        ):
            gen_instance = MagicMock()
            gen_instance.generate.return_value = mock_report
            MockGen.return_value = gen_instance

            handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)

        assert "CR-CACHED" in _report_cache
        assert _report_cache["CR-CACHED"] is mock_report

    def test_generate_null_debate_id(self, handler):
        """Explicit null debate_id should return 400."""
        mock_h = _MockHTTPHandler("POST", body={"debate_id": None, "framework": "general"})
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
        assert _status(result) == 400

    def test_generate_empty_string_debate_id(self, handler):
        """Empty string debate_id should return 400."""
        mock_h = _MockHTTPHandler("POST", body={"debate_id": "", "framework": "general"})
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
        assert _status(result) == 400


# ===========================================================================
# handle() routing / dispatch
# ===========================================================================


class TestHandleRouting:
    """Test that handle() properly dispatches to the right internal method."""

    def test_unrelated_path_returns_none(self, handler):
        """Path not starting with /api/v1/compliance/reports returns None."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/something/else", {}, mock_h)
        assert result is None

    def test_root_path_returns_none(self, handler):
        """GET /api/v1/compliance/reports alone (4 parts) returns None since len != 5."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/compliance/reports", {}, mock_h)
        assert result is None

    def test_download_path_dispatches(self, handler, cached_report):
        """A path ending in /download routes to _handle_download."""
        mock_h = _MockHTTPHandler("GET")
        with patch(_PATCH_GENERATOR) as MockGen:
            gen_instance = MagicMock()
            gen_instance.export_json.return_value = '{"ok": true}'
            MockGen.return_value = gen_instance

            result = handler.handle(
                "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                {"format": "json"},
                mock_h,
            )
            assert _status(result) == 200

    def test_id_path_dispatches(self, handler, cached_report):
        """A 5-segment path routes to _handle_get_report."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/compliance/reports/CR-ABCDEF123456", {}, mock_h)
        assert _status(result) == 200

    def test_too_many_segments_returns_none(self, handler):
        """6+ segments that don't end in /download return None."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/compliance/reports/CR-ABC/extra/segment", {}, mock_h)
        assert result is None

    def test_download_takes_priority_over_id(self, handler, cached_report):
        """Path ending with /download is checked before segment-count routing."""
        mock_h = _MockHTTPHandler("GET")
        with patch(_PATCH_GENERATOR) as MockGen:
            gen_instance = MagicMock()
            gen_instance.export_json.return_value = '{"ok": true}'
            MockGen.return_value = gen_instance

            result = handler.handle(
                "/api/v1/compliance/reports/CR-ABCDEF123456/download",
                {"format": "json"},
                mock_h,
            )
            # Even though this path has 6 segments, /download suffix is matched first
            assert _status(result) == 200

    def test_generate_path_as_id(self, handler):
        """GET /api/v1/compliance/reports/generate -- 'generate' treated as report ID."""
        mock_h = _MockHTTPHandler("GET")
        # "generate" becomes report_id, not cached, so 404
        result = handler.handle("/api/v1/compliance/reports/generate", {}, mock_h)
        assert _status(result) == 404


# ===========================================================================
# handle_post() path matching
# ===========================================================================


class TestHandlePostRouting:
    """Test that handle_post only responds to the correct path."""

    def test_wrong_path_returns_none(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"debate_id": "d1"})
        result = handler.handle_post("/api/v1/compliance/reports/CR-123", {}, mock_h)
        assert result is None

    def test_similar_but_wrong_path(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"debate_id": "d1"})
        result = handler.handle_post("/api/v1/compliance/reports/generate/extra", {}, mock_h)
        assert result is None

    def test_correct_path_but_empty_body(self, handler):
        """Correct path but empty body (no debate_id) returns 400."""
        mock_h = _MockHTTPHandler("POST")
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
        assert _status(result) == 400

    def test_case_sensitive_path(self, handler):
        """Path matching is case-sensitive."""
        mock_h = _MockHTTPHandler("POST", body={"debate_id": "d1"})
        result = handler.handle_post("/api/v1/compliance/reports/Generate", {}, mock_h)
        assert result is None


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_handler_ctx_defaults_to_empty_dict(self):
        """Handler can be created with None ctx."""
        h = ComplianceReportHandler(ctx=None)
        assert h.ctx == {}

    def test_handler_with_kwargs(self):
        """Extra kwargs are accepted (BaseHandler pattern)."""
        h = ComplianceReportHandler(ctx={"key": "val"}, extra_param="ignored")
        assert h.ctx == {"key": "val"}

    def test_report_cache_is_module_level(self):
        """Verify _report_cache is shared across handler instances."""
        h1 = ComplianceReportHandler()
        h2 = ComplianceReportHandler()

        report = _FakeReport(
            report_id="CR-SHARED",
            debate_id="d1",
            framework=_FakeFramework.GENERAL,
            generated_at=datetime(2026, 1, 1),
        )
        _report_cache["CR-SHARED"] = report

        mock_h = _MockHTTPHandler("GET")
        result1 = h1.handle("/api/v1/compliance/reports/CR-SHARED", {}, mock_h)
        result2 = h2.handle("/api/v1/compliance/reports/CR-SHARED", {}, mock_h)
        assert _status(result1) == 200
        assert _status(result2) == 200
        assert _body(result1)["report_id"] == _body(result2)["report_id"]

    def test_generate_value_error_returns_400(self, handler, mock_storage):
        """ValueError during generation returns 400."""
        handler.ctx = {"storage": mock_storage}

        mock_h = _MockHTTPHandler(
            "POST",
            body={"debate_id": "debate-err", "framework": "general"},
        )

        with patch(
            _PATCH_FRAMEWORK,
            side_effect=ValueError("bad framework"),
        ):
            result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
            assert _status(result) == 400

    def test_generate_type_error_returns_400(self, handler, mock_storage):
        """TypeError during generation returns 400."""
        handler.ctx = {"storage": mock_storage}

        mock_h = _MockHTTPHandler(
            "POST",
            body={"debate_id": "debate-terr", "framework": "general"},
        )

        with patch(
            _PATCH_FRAMEWORK,
            side_effect=TypeError("bad type"),
        ):
            result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
            assert _status(result) == 400

    def test_report_id_with_special_chars(self, handler):
        """Report IDs with special chars still produce 404 (not cached)."""
        mock_h = _MockHTTPHandler("GET")
        for rid in ["CR-abc", "CR-123-456", "CR-test.report"]:
            result = handler.handle(f"/api/v1/compliance/reports/{rid}", {}, mock_h)
            assert _status(result) == 404

    def test_download_report_id_extracted_correctly(self, handler):
        """Download extracts report_id from parts[4]."""
        report = _FakeReport(
            report_id="CR-EXTRACT",
            debate_id="d1",
            framework=_FakeFramework.GENERAL,
            generated_at=datetime(2026, 1, 1),
        )
        _report_cache["CR-EXTRACT"] = report

        mock_h = _MockHTTPHandler("GET")
        with patch(_PATCH_GENERATOR) as MockGen:
            gen_instance = MagicMock()
            gen_instance.export_json.return_value = '{"ok": true}'
            MockGen.return_value = gen_instance

            result = handler.handle(
                "/api/v1/compliance/reports/CR-EXTRACT/download",
                {"format": "json"},
                mock_h,
            )
            assert _status(result) == 200

    def test_multiple_reports_in_cache(self, handler):
        """Multiple reports can coexist in cache."""
        for i in range(5):
            _report_cache[f"CR-{i:03d}"] = _FakeReport(
                report_id=f"CR-{i:03d}",
                debate_id=f"debate-{i}",
                framework=_FakeFramework.GENERAL,
                generated_at=datetime(2026, 1, i + 1),
            )

        mock_h = _MockHTTPHandler("GET")
        for i in range(5):
            result = handler.handle(f"/api/v1/compliance/reports/CR-{i:03d}", {}, mock_h)
            assert _status(result) == 200
            assert _body(result)["report_id"] == f"CR-{i:03d}"

    def test_generate_debate_data_missing_fields(self, handler):
        """Debate data with missing fields uses defaults."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {}  # Empty dict, all defaults
        handler.ctx = {"storage": mock_storage}

        mock_h = _MockHTTPHandler(
            "POST",
            body={"debate_id": "debate-sparse", "framework": "general"},
        )

        mock_report = _make_mock_report(report_id="CR-SPARSE")
        with (
            patch(_PATCH_GENERATOR) as MockGen,
            patch(_PATCH_FRAMEWORK),
            patch(_PATCH_DEBATE_RESULT) as MockDR,
        ):
            gen_instance = MagicMock()
            gen_instance.generate.return_value = mock_report
            MockGen.return_value = gen_instance

            result = handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h)
            assert _status(result) == 201

            # DebateResult was called with defaults from empty debate_data
            dr_call = MockDR.call_args
            assert dr_call[1]["task"] == ""
            assert dr_call[1]["consensus_reached"] is False
            assert dr_call[1]["rounds_used"] == 0
            assert dr_call[1]["winner"] is None
            assert dr_call[1]["final_answer"] == ""

    def test_generate_then_retrieve(self, handler, mock_storage):
        """End-to-end: generate a report, then retrieve it by ID."""
        handler.ctx = {"storage": mock_storage}

        mock_report = _make_mock_report(report_id="CR-E2E001")
        # Make to_dict work for retrieval
        mock_report.to_dict.return_value = {
            "report_id": "CR-E2E001",
            "debate_id": "debate-001",
            "framework": "general",
            "summary": "E2E test",
        }

        mock_h_post = _MockHTTPHandler(
            "POST",
            body={"debate_id": "debate-001", "framework": "general"},
        )

        with (
            patch(_PATCH_GENERATOR) as MockGen,
            patch(_PATCH_FRAMEWORK),
            patch(_PATCH_DEBATE_RESULT),
        ):
            gen_instance = MagicMock()
            gen_instance.generate.return_value = mock_report
            MockGen.return_value = gen_instance

            handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h_post)

        # Now retrieve it
        mock_h_get = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/compliance/reports/CR-E2E001", {}, mock_h_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["report_id"] == "CR-E2E001"

    def test_generate_then_download(self, handler, mock_storage):
        """End-to-end: generate a report, then download it."""
        handler.ctx = {"storage": mock_storage}

        mock_report = _make_mock_report(report_id="CR-DL001")

        mock_h_post = _MockHTTPHandler(
            "POST",
            body={"debate_id": "debate-001", "framework": "general"},
        )

        with (
            patch(_PATCH_GENERATOR) as MockGen,
            patch(_PATCH_FRAMEWORK),
            patch(_PATCH_DEBATE_RESULT),
        ):
            gen_instance = MagicMock()
            gen_instance.generate.return_value = mock_report
            MockGen.return_value = gen_instance

            handler.handle_post("/api/v1/compliance/reports/generate", {}, mock_h_post)

        # Now download it
        mock_h_dl = _MockHTTPHandler("GET")
        with patch(_PATCH_GENERATOR) as MockGen2:
            gen2 = MagicMock()
            gen2.export_markdown.return_value = "# Downloaded Report"
            MockGen2.return_value = gen2

            result = handler.handle(
                "/api/v1/compliance/reports/CR-DL001/download",
                {"format": "markdown"},
                mock_h_dl,
            )
            assert _status(result) == 200
            assert b"# Downloaded Report" in _raw_body(result)

    def test_handler_default_ctx(self):
        """Default init (no args) should give empty ctx."""
        h = ComplianceReportHandler()
        assert h.ctx == {}
