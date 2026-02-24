"""Comprehensive tests for the CrashTelemetryHandler.

Tests the three crash telemetry endpoints:

- POST /api/v1/observability/crashes         -> _ingest
- GET  /api/v1/observability/crashes         -> _list_crashes (admin)
- GET  /api/v1/observability/crashes/stats   -> _get_stats

Also tests:
- Deduplication by fingerprint
- Per-IP rate limiting
- Batch size enforcement
- Fingerprint computation stability
- Module-level store reset
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.observability.crashes import (
    CrashTelemetryHandler,
    _compute_fingerprint,
    get_crash_store,
    reset_crash_store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_handler_with_body(body: dict[str, Any]) -> MagicMock:
    """Create a mock HTTP handler with a JSON body."""
    body_bytes = json.dumps(body).encode()
    handler = MagicMock()
    handler.rfile.read.return_value = body_bytes
    handler.headers = {
        "Content-Length": str(len(body_bytes)),
        "Content-Type": "application/json",
    }
    handler.client_address = ("127.0.0.1", 12345)
    return handler


def _make_report(
    message: str = "TypeError: cannot read property 'x' of undefined",
    stack: str | None = "at Component.render (app.js:42)\n  at renderRoot",
    fingerprint: str | None = None,
    component_name: str | None = "DashboardWidget",
) -> dict[str, Any]:
    """Build a minimal crash report dict."""
    report: dict[str, Any] = {
        "message": message,
        "stack": stack,
        "url": "https://app.aragora.ai/dashboard",
        "timestamp": "2026-02-24T12:00:00Z",
        "userAgent": "Mozilla/5.0 TestBrowser",
        "sessionId": "sess_test123",
    }
    if fingerprint:
        report["fingerprint"] = fingerprint
    if component_name:
        report["componentName"] = component_name
    return report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_store():
    """Reset the crash store before and after each test."""
    reset_crash_store()
    yield
    reset_crash_store()


@pytest.fixture
def handler() -> CrashTelemetryHandler:
    """Create a CrashTelemetryHandler with empty server context."""
    return CrashTelemetryHandler(server_context={})


# ---------------------------------------------------------------------------
# Route matching
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching."""

    def test_matches_crashes_path(self, handler: CrashTelemetryHandler):
        assert handler.can_handle("/api/observability/crashes") is True

    def test_matches_versioned_crashes_path(self, handler: CrashTelemetryHandler):
        assert handler.can_handle("/api/v1/observability/crashes") is True

    def test_matches_stats_path(self, handler: CrashTelemetryHandler):
        assert handler.can_handle("/api/observability/crashes/stats") is True

    def test_rejects_unknown_path(self, handler: CrashTelemetryHandler):
        assert handler.can_handle("/api/observability/unknown") is False


# ---------------------------------------------------------------------------
# POST ingestion
# ---------------------------------------------------------------------------


class TestIngest:
    """Tests for crash report ingestion."""

    def test_single_report_accepted(self, handler: CrashTelemetryHandler):
        report = _make_report()
        mock = _make_handler_with_body({"reports": [report]})
        result = handler.handle_post("/api/v1/observability/crashes", {}, mock)
        assert _status(result) == 202
        body = _body(result)
        assert body["accepted"] == 1
        assert body["duplicates"] == 0
        assert body["total_stored"] == 1

    def test_multiple_reports_accepted(self, handler: CrashTelemetryHandler):
        reports = [
            _make_report(message=f"Error {i}", fingerprint=f"fp_{i}")
            for i in range(5)
        ]
        mock = _make_handler_with_body({"reports": reports})
        result = handler.handle_post("/api/v1/observability/crashes", {}, mock)
        assert _status(result) == 202
        assert _body(result)["accepted"] == 5
        assert len(get_crash_store()) == 5

    def test_empty_reports_list(self, handler: CrashTelemetryHandler):
        mock = _make_handler_with_body({"reports": []})
        result = handler.handle_post("/api/v1/observability/crashes", {}, mock)
        assert _status(result) == 202
        assert _body(result)["accepted"] == 0

    def test_invalid_json_body(self, handler: CrashTelemetryHandler):
        mock = MagicMock()
        mock.rfile.read.return_value = b"not json"
        mock.headers = {"Content-Length": "8"}
        mock.client_address = ("127.0.0.1", 12345)
        result = handler.handle_post("/api/v1/observability/crashes", {}, mock)
        assert _status(result) == 400

    def test_reports_not_a_list(self, handler: CrashTelemetryHandler):
        mock = _make_handler_with_body({"reports": "not-a-list"})
        result = handler.handle_post("/api/v1/observability/crashes", {}, mock)
        assert _status(result) == 400
        assert "list" in _body(result).get("error", "").lower()

    def test_batch_too_large(self, handler: CrashTelemetryHandler):
        reports = [_make_report(fingerprint=f"fp_{i}") for i in range(51)]
        mock = _make_handler_with_body({"reports": reports})
        result = handler.handle_post("/api/v1/observability/crashes", {}, mock)
        assert _status(result) == 400
        assert "batch" in _body(result).get("error", "").lower()

    def test_non_dict_reports_skipped(self, handler: CrashTelemetryHandler):
        mock = _make_handler_with_body({"reports": ["string", 42, None, _make_report()]})
        result = handler.handle_post("/api/v1/observability/crashes", {}, mock)
        assert _status(result) == 202
        # Only the dict report is accepted
        assert _body(result)["accepted"] == 1

    def test_message_truncated_to_2000_chars(self, handler: CrashTelemetryHandler):
        long_message = "x" * 5000
        mock = _make_handler_with_body(
            {"reports": [_make_report(message=long_message, fingerprint="fp_long")]}
        )
        handler.handle_post("/api/v1/observability/crashes", {}, mock)
        stored = list(get_crash_store())
        assert len(stored[0]["message"]) == 2000

    def test_wrong_path_returns_none(self, handler: CrashTelemetryHandler):
        mock = _make_handler_with_body({"reports": []})
        result = handler.handle_post("/api/v1/observability/other", {}, mock)
        assert result is None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for fingerprint-based deduplication."""

    def test_duplicate_fingerprint_rejected(self, handler: CrashTelemetryHandler):
        report = _make_report(fingerprint="dup_fp_1")
        mock1 = _make_handler_with_body({"reports": [report]})
        mock2 = _make_handler_with_body({"reports": [report]})

        result1 = handler.handle_post("/api/v1/observability/crashes", {}, mock1)
        assert _body(result1)["accepted"] == 1

        result2 = handler.handle_post("/api/v1/observability/crashes", {}, mock2)
        assert _body(result2)["accepted"] == 0
        assert _body(result2)["duplicates"] == 1

    def test_same_message_different_stack_are_separate(self, handler: CrashTelemetryHandler):
        r1 = _make_report(message="Error", stack="at A.js:1", fingerprint=None)
        r2 = _make_report(message="Error", stack="at B.js:99", fingerprint=None)
        mock = _make_handler_with_body({"reports": [r1, r2]})
        result = handler.handle_post("/api/v1/observability/crashes", {}, mock)
        assert _body(result)["accepted"] == 2

    def test_auto_computed_fingerprint_stable(self):
        fp1 = _compute_fingerprint("TypeError", "at render:42")
        fp2 = _compute_fingerprint("TypeError", "at render:42")
        assert fp1 == fp2
        assert len(fp1) == 16  # SHA-256 truncated to 16 hex chars

    def test_different_messages_different_fingerprints(self):
        fp1 = _compute_fingerprint("Error A", "stack")
        fp2 = _compute_fingerprint("Error B", "stack")
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for per-IP rate limiting on ingestion."""

    def test_ip_rate_limit_enforced(self, handler: CrashTelemetryHandler):
        """After IP_RATE_LIMIT requests, further requests are rejected."""
        from aragora.server.handlers.observability.crashes import IP_RATE_LIMIT

        for i in range(IP_RATE_LIMIT):
            report = _make_report(fingerprint=f"rate_{i}")
            mock = _make_handler_with_body({"reports": [report]})
            result = handler.handle_post("/api/v1/observability/crashes", {}, mock)
            assert _status(result) in (202, 429), f"Iteration {i}: unexpected status"

        # Next request should be rate limited
        report = _make_report(fingerprint="rate_overflow")
        mock = _make_handler_with_body({"reports": [report]})
        result = handler.handle_post("/api/v1/observability/crashes", {}, mock)
        assert _status(result) == 429


# ---------------------------------------------------------------------------
# GET /api/v1/observability/crashes (admin list)
# ---------------------------------------------------------------------------


class TestListCrashes:
    """Tests for the admin crash listing endpoint."""

    def _seed_crashes(self, handler: CrashTelemetryHandler, count: int = 5):
        """Seed the store with a number of crash reports."""
        for i in range(count):
            report = _make_report(
                message=f"Crash {i}", fingerprint=f"list_{i}", component_name=f"Comp{i}"
            )
            mock = _make_handler_with_body({"reports": [report]})
            handler.handle_post("/api/v1/observability/crashes", {}, mock)

    def test_list_returns_crashes(self, handler: CrashTelemetryHandler):
        self._seed_crashes(handler, 3)
        mock_http = MagicMock()
        result = handler.handle("/api/v1/observability/crashes", {}, mock_http)
        body = _body(result)
        assert body["total"] == 3
        assert len(body["crashes"]) == 3

    def test_list_pagination(self, handler: CrashTelemetryHandler):
        self._seed_crashes(handler, 10)
        mock_http = MagicMock()
        result = handler.handle(
            "/api/v1/observability/crashes", {"limit": "3", "offset": "2"}, mock_http
        )
        body = _body(result)
        assert len(body["crashes"]) == 3
        assert body["offset"] == 2
        assert body["total"] == 10

    def test_list_most_recent_first(self, handler: CrashTelemetryHandler):
        self._seed_crashes(handler, 5)
        mock_http = MagicMock()
        result = handler.handle("/api/v1/observability/crashes", {}, mock_http)
        crashes = _body(result)["crashes"]
        # Most recent (last inserted) should be first
        assert "Crash 4" in crashes[0]["message"]


# ---------------------------------------------------------------------------
# GET /api/v1/observability/crashes/stats
# ---------------------------------------------------------------------------


class TestStats:
    """Tests for the crash stats endpoint."""

    def test_stats_empty(self, handler: CrashTelemetryHandler):
        mock_http = MagicMock()
        result = handler.handle("/api/v1/observability/crashes/stats", {}, mock_http)
        body = _body(result)
        assert body["total_ingested"] == 0
        assert body["unique_fingerprints"] == 0
        assert body["last_hour"] == 0

    def test_stats_after_ingestion(self, handler: CrashTelemetryHandler):
        for i in range(3):
            report = _make_report(fingerprint=f"stat_{i}", component_name="StatsWidget")
            mock = _make_handler_with_body({"reports": [report]})
            handler.handle_post("/api/v1/observability/crashes", {}, mock)

        mock_http = MagicMock()
        result = handler.handle("/api/v1/observability/crashes/stats", {}, mock_http)
        body = _body(result)
        assert body["total_ingested"] == 3
        assert body["unique_fingerprints"] == 3
        assert body["last_hour"] == 3
        assert body["last_24h"] == 3

    def test_stats_top_components(self, handler: CrashTelemetryHandler):
        # Ingest 3 crashes for CompA and 1 for CompB
        for i in range(3):
            report = _make_report(
                fingerprint=f"comp_a_{i}", component_name="CompA"
            )
            mock = _make_handler_with_body({"reports": [report]})
            handler.handle_post("/api/v1/observability/crashes", {}, mock)

        report = _make_report(fingerprint="comp_b_0", component_name="CompB")
        mock = _make_handler_with_body({"reports": [report]})
        handler.handle_post("/api/v1/observability/crashes", {}, mock)

        mock_http = MagicMock()
        result = handler.handle("/api/v1/observability/crashes/stats", {}, mock_http)
        top = _body(result)["top_components"]
        assert top[0]["component"] == "CompA"
        assert top[0]["count"] == 3

    def test_stats_duplicate_counting(self, handler: CrashTelemetryHandler):
        report = _make_report(fingerprint="dup_stat")
        mock1 = _make_handler_with_body({"reports": [report]})
        mock2 = _make_handler_with_body({"reports": [report]})

        handler.handle_post("/api/v1/observability/crashes", {}, mock1)
        handler.handle_post("/api/v1/observability/crashes", {}, mock2)

        mock_http = MagicMock()
        result = handler.handle("/api/v1/observability/crashes/stats", {}, mock_http)
        body = _body(result)
        assert body["total_duplicates"] == 1
        assert body["total_ingested"] == 1

    def test_stats_top_fingerprints(self, handler: CrashTelemetryHandler):
        """Top fingerprints include both the original and duplicate counts."""
        report = _make_report(fingerprint="top_fp")
        for _ in range(3):
            mock = _make_handler_with_body({"reports": [report]})
            handler.handle_post("/api/v1/observability/crashes", {}, mock)

        mock_http = MagicMock()
        result = handler.handle("/api/v1/observability/crashes/stats", {}, mock_http)
        top = _body(result)["top_fingerprints"]
        assert any(fp["fingerprint"] == "top_fp" for fp in top)


# ---------------------------------------------------------------------------
# Store management
# ---------------------------------------------------------------------------


class TestStoreManagement:
    """Tests for reset and bounded store behavior."""

    def test_reset_clears_everything(self, handler: CrashTelemetryHandler):
        report = _make_report(fingerprint="reset_test")
        mock = _make_handler_with_body({"reports": [report]})
        handler.handle_post("/api/v1/observability/crashes", {}, mock)
        assert len(get_crash_store()) == 1

        reset_crash_store()
        assert len(get_crash_store()) == 0

    def test_store_bounded_at_max(self, handler: CrashTelemetryHandler):
        """The deque maxlen keeps the store bounded."""
        store = get_crash_store()
        assert store.maxlen == 1_000
