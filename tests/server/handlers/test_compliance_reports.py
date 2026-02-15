"""
Tests for Compliance Report Download Handler.

Covers:
- POST /api/v1/compliance/reports/generate
- GET  /api/v1/compliance/reports/:id
- GET  /api/v1/compliance/reports/:id/download?format=json|html|markdown
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.compliance_reports import (
    ComplianceReportHandler,
    _report_cache,
)


# ============================================================================
# Fixtures
# ============================================================================


def _make_mock_handler(method="GET", body=None):
    mock = MagicMock()
    mock.command = method
    if body is not None:
        body_bytes = json.dumps(body).encode()
    else:
        body_bytes = b"{}"
    mock.rfile = MagicMock()
    mock.rfile.read = MagicMock(return_value=body_bytes)
    mock.headers = {"Content-Length": str(len(body_bytes))}
    return mock


def _parse(result) -> dict[str, Any]:
    if result is None:
        return {}
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError:
        return {}


@pytest.fixture(autouse=True)
def clear_cache():
    _report_cache.clear()
    yield
    _report_cache.clear()


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.get_debate.return_value = {
        "task": "Evaluate rate limiter design",
        "consensus_reached": True,
        "rounds_used": 3,
        "winner": "claude",
        "final_answer": "Use token bucket algorithm",
    }
    return storage


@pytest.fixture
def handler(mock_storage):
    return ComplianceReportHandler(ctx={"storage": mock_storage})


@pytest.fixture
def handler_no_storage():
    return ComplianceReportHandler(ctx={})


# ============================================================================
# Route Tests
# ============================================================================


class TestRouting:
    def test_can_handle_reports_path(self, handler):
        assert handler.can_handle("/api/v1/compliance/reports") is True
        assert handler.can_handle("/api/v1/compliance/reports/generate") is True
        assert handler.can_handle("/api/v1/compliance/reports/CR-ABC/download") is True

    def test_cannot_handle_other_paths(self, handler):
        assert handler.can_handle("/api/v1/compliance/other") is False
        assert handler.can_handle("/api/v1/other") is False


# ============================================================================
# POST - Generate Report
# ============================================================================


class TestGenerateReport:
    def test_generate_with_defaults(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "debate_id": "debate-001",
            },
        )
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, http)
        assert result.status_code == 201
        body = _parse(result)
        assert "report_id" in body
        assert body["framework"] == "general"
        assert body["debate_id"] == "debate-001"

    def test_generate_with_framework(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "debate_id": "debate-001",
                "framework": "soc2",
            },
        )
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, http)
        assert result.status_code == 201
        body = _parse(result)
        assert body["framework"] == "soc2"

    def test_generate_missing_debate_id(self, handler):
        http = _make_mock_handler(method="POST", body={"framework": "gdpr"})
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, http)
        assert result.status_code == 400
        assert "debate_id" in _parse(result).get("error", "")

    def test_generate_invalid_framework(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "debate_id": "debate-001",
                "framework": "invalid",
            },
        )
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, http)
        assert result.status_code == 400

    def test_generate_debate_not_found(self, handler, mock_storage):
        mock_storage.get_debate.return_value = None
        http = _make_mock_handler(
            method="POST",
            body={
                "debate_id": "missing-debate",
            },
        )
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, http)
        assert result.status_code == 404

    def test_generate_returns_none_for_unhandled_path(self, handler):
        http = _make_mock_handler(method="POST", body={})
        result = handler.handle_post("/api/v1/other", {}, http)
        assert result is None

    def test_generate_gdpr_framework(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "debate_id": "debate-001",
                "framework": "gdpr",
            },
        )
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, http)
        assert result.status_code == 201
        body = _parse(result)
        assert body["framework"] == "gdpr"

    def test_generate_hipaa_framework(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "debate_id": "debate-001",
                "framework": "hipaa",
            },
        )
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, http)
        assert result.status_code == 201

    def test_generate_with_scope_options(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "debate_id": "debate-001",
                "scope": {
                    "include_evidence": False,
                    "include_chain": False,
                    "include_transcript": True,
                },
            },
        )
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, http)
        assert result.status_code == 201

    def test_generate_caches_report(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "debate_id": "debate-001",
            },
        )
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, http)
        body = _parse(result)
        report_id = body["report_id"]
        assert report_id in _report_cache


# ============================================================================
# GET - Retrieve Report
# ============================================================================


class TestGetReport:
    def _generate_report(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "debate_id": "debate-001",
            },
        )
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, http)
        return _parse(result)["report_id"]

    def test_get_existing_report(self, handler):
        report_id = self._generate_report(handler)
        http = _make_mock_handler()
        result = handler.handle(f"/api/v1/compliance/reports/{report_id}", {}, http)
        assert result.status_code == 200
        body = _parse(result)
        assert body["report_id"] == report_id

    def test_get_nonexistent_report(self, handler):
        http = _make_mock_handler()
        result = handler.handle("/api/v1/compliance/reports/MISSING", {}, http)
        assert result.status_code == 404

    def test_get_returns_full_report_data(self, handler):
        report_id = self._generate_report(handler)
        http = _make_mock_handler()
        result = handler.handle(f"/api/v1/compliance/reports/{report_id}", {}, http)
        body = _parse(result)
        assert "sections" in body
        assert "attestation" in body
        assert "summary" in body


# ============================================================================
# GET - Download Report
# ============================================================================


class TestDownloadReport:
    def _generate_report(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "debate_id": "debate-001",
            },
        )
        result = handler.handle_post("/api/v1/compliance/reports/generate", {}, http)
        return _parse(result)["report_id"]

    def test_download_json(self, handler):
        report_id = self._generate_report(handler)
        http = _make_mock_handler()
        result = handler.handle(
            f"/api/v1/compliance/reports/{report_id}/download",
            {"format": "json"},
            http,
        )
        assert result.status_code == 200
        assert "application/json" in result.content_type

    def test_download_markdown(self, handler):
        report_id = self._generate_report(handler)
        http = _make_mock_handler()
        result = handler.handle(
            f"/api/v1/compliance/reports/{report_id}/download",
            {"format": "markdown"},
            http,
        )
        assert result.status_code == 200
        assert "text/markdown" in result.content_type
        body_text = result.body.decode("utf-8") if isinstance(result.body, bytes) else result.body
        assert "# Compliance Report" in body_text

    def test_download_html(self, handler):
        report_id = self._generate_report(handler)
        http = _make_mock_handler()
        result = handler.handle(
            f"/api/v1/compliance/reports/{report_id}/download",
            {"format": "html"},
            http,
        )
        assert result.status_code == 200
        assert "text/html" in result.content_type

    def test_download_invalid_format(self, handler):
        report_id = self._generate_report(handler)
        http = _make_mock_handler()
        result = handler.handle(
            f"/api/v1/compliance/reports/{report_id}/download",
            {"format": "xml"},
            http,
        )
        assert result.status_code == 400

    def test_download_nonexistent_report(self, handler):
        http = _make_mock_handler()
        result = handler.handle(
            "/api/v1/compliance/reports/MISSING/download",
            {"format": "json"},
            http,
        )
        assert result.status_code == 404
