"""Tests for the CrashTelemetryHandler.

Covers all routes and behaviour of the CrashTelemetryHandler class:
- POST /api/observability/crashes         - Ingest crash reports
- GET  /api/observability/crashes         - List recent crashes (admin)
- GET  /api/observability/crashes/stats   - Crash frequency stats (admin)

Also covers:
- IP-based rate limiting
- Fingerprint-based deduplication
- Batch size limits
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
    IP_RATE_LIMIT,
    MAX_REPORTS_PER_BATCH,
    _check_ip_rate_limit,
    _compute_fingerprint,
    get_crash_store,
    reset_crash_store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_response(result) -> dict:
    """Extract data from json_response HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            body = json.loads(body)
        if isinstance(body, dict):
            return body
    if isinstance(result, tuple):
        body = result[0] if len(result) > 0 else {}
        if isinstance(body, str):
            body = json.loads(body)
        return body
    if isinstance(result, dict):
        return result
    return {}


def _status_code(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return 200


def _make_handler_with_body(body: dict | None = None, client_ip: str = "127.0.0.1") -> MagicMock:
    """Create a mock HTTP handler with a JSON body and client address."""
    h = MagicMock()
    body_data = body or {}
    body_bytes = json.dumps(body_data).encode("utf-8")
    h.rfile = MagicMock()
    h.rfile.read.return_value = body_bytes
    h.headers = {
        "Content-Length": str(len(body_bytes)),
        "Content-Type": "application/json",
    }
    h.client_address = (client_ip, 12345)
    return h


def _make_crash_report(
    message: str = "TypeError: undefined is not a function",
    stack: str | None = "at Component.render (app.js:42)",
    component_name: str | None = "ErrorBoundary",
    fingerprint: str | None = None,
    url: str = "https://example.com/dashboard",
    session_id: str = "sess-001",
    user_agent: str = "Mozilla/5.0",
) -> dict[str, Any]:
    """Build a single crash report dict."""
    report: dict[str, Any] = {
        "message": message,
        "url": url,
        "sessionId": session_id,
        "userAgent": user_agent,
        "timestamp": "2026-02-20T10:00:00Z",
    }
    if stack is not None:
        report["stack"] = stack
    if component_name is not None:
        report["componentName"] = component_name
    if fingerprint is not None:
        report["fingerprint"] = fingerprint
    return report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset the module-level crash store before and after each test."""
    reset_crash_store()
    yield
    reset_crash_store()


@pytest.fixture
def handler():
    """Create a CrashTelemetryHandler instance."""
    return CrashTelemetryHandler({})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test ROUTES and can_handle."""

    def test_routes_contains_all_endpoints(self):
        expected = [
            "/api/observability/crashes",
            "/api/observability/crashes/stats",
        ]
        for route in expected:
            assert route in CrashTelemetryHandler.ROUTES, f"Missing route: {route}"

    def test_can_handle_crashes_path(self, handler):
        assert handler.can_handle("/api/observability/crashes")

    def test_can_handle_stats_path(self, handler):
        assert handler.can_handle("/api/observability/crashes/stats")

    def test_can_handle_with_version_prefix(self, handler):
        assert handler.can_handle("/api/v1/observability/crashes")
        assert handler.can_handle("/api/v1/observability/crashes/stats")

    def test_can_handle_rejects_unknown(self, handler):
        assert not handler.can_handle("/api/observability/other")
        assert not handler.can_handle("/api/observability/crashes/unknown")


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------


class TestFingerprint:
    """Test fingerprint generation logic."""

    def test_fingerprint_deterministic(self):
        fp1 = _compute_fingerprint("error msg", "stack trace")
        fp2 = _compute_fingerprint("error msg", "stack trace")
        assert fp1 == fp2

    def test_fingerprint_different_for_different_messages(self):
        fp1 = _compute_fingerprint("error A", "stack")
        fp2 = _compute_fingerprint("error B", "stack")
        assert fp1 != fp2

    def test_fingerprint_handles_none_stack(self):
        fp = _compute_fingerprint("error msg", None)
        assert isinstance(fp, str)
        assert len(fp) == 16

    def test_fingerprint_length(self):
        fp = _compute_fingerprint("msg", "stack")
        assert len(fp) == 16  # sha256[:16]


# ---------------------------------------------------------------------------
# IP rate limiting
# ---------------------------------------------------------------------------


class TestIPRateLimit:
    """Test per-IP rate limiting."""

    def test_allows_within_limit(self):
        assert _check_ip_rate_limit("10.0.0.1") is True

    def test_blocks_over_limit(self):
        for _ in range(IP_RATE_LIMIT):
            _check_ip_rate_limit("10.0.0.2")
        assert _check_ip_rate_limit("10.0.0.2") is False

    def test_different_ips_independent(self):
        for _ in range(IP_RATE_LIMIT):
            _check_ip_rate_limit("10.0.0.3")
        # Different IP should still be allowed
        assert _check_ip_rate_limit("10.0.0.4") is True


# ---------------------------------------------------------------------------
# POST /api/observability/crashes (ingest)
# ---------------------------------------------------------------------------


class TestIngest:
    """Test crash report ingestion."""

    def test_ingest_single_report(self, handler):
        body = {"reports": [_make_crash_report()]}
        h = _make_handler_with_body(body)

        result = handler.handle_post("/api/observability/crashes", {}, h)

        assert _status_code(result) == 202
        data = _parse_response(result)
        assert data["accepted"] == 1
        assert data["duplicates"] == 0
        assert data["total_stored"] == 1

    def test_ingest_multiple_reports(self, handler):
        body = {
            "reports": [
                _make_crash_report(message="Error A"),
                _make_crash_report(message="Error B"),
                _make_crash_report(message="Error C"),
            ]
        }
        h = _make_handler_with_body(body)

        result = handler.handle_post("/api/observability/crashes", {}, h)

        data = _parse_response(result)
        assert data["accepted"] == 3
        assert data["total_stored"] == 3

    def test_ingest_deduplication(self, handler):
        report = _make_crash_report(fingerprint="fixed-fp-001")
        body = {"reports": [report]}

        h1 = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, h1)

        h2 = _make_handler_with_body(body)
        result = handler.handle_post("/api/observability/crashes", {}, h2)

        data = _parse_response(result)
        assert data["accepted"] == 0
        assert data["duplicates"] == 1

    def test_ingest_invalid_json(self, handler):
        h = MagicMock()
        h.rfile = MagicMock()
        h.rfile.read.return_value = b"not-json"
        h.headers = {"Content-Length": "8", "Content-Type": "application/json"}
        h.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/observability/crashes", {}, h)

        assert _status_code(result) == 400

    def test_ingest_reports_not_a_list(self, handler):
        body = {"reports": "not-a-list"}
        h = _make_handler_with_body(body)

        result = handler.handle_post("/api/observability/crashes", {}, h)

        assert _status_code(result) == 400
        data = _parse_response(result)
        assert "list" in str(data).lower()

    def test_ingest_batch_too_large(self, handler):
        reports = [
            _make_crash_report(message=f"Error {i}") for i in range(MAX_REPORTS_PER_BATCH + 1)
        ]
        body = {"reports": reports}
        h = _make_handler_with_body(body)

        result = handler.handle_post("/api/observability/crashes", {}, h)

        assert _status_code(result) == 400
        data = _parse_response(result)
        assert "batch" in str(data).lower() or "large" in str(data).lower()

    def test_ingest_skips_non_dict_reports(self, handler):
        body = {"reports": ["not-a-dict", _make_crash_report()]}
        h = _make_handler_with_body(body)

        result = handler.handle_post("/api/observability/crashes", {}, h)

        data = _parse_response(result)
        assert data["accepted"] == 1

    def test_ingest_truncates_long_message(self, handler):
        long_msg = "x" * 5000
        report = _make_crash_report(message=long_msg)
        body = {"reports": [report]}
        h = _make_handler_with_body(body)

        handler.handle_post("/api/observability/crashes", {}, h)

        store = get_crash_store()
        assert len(store[-1]["message"]) <= 2000

    def test_ingest_truncates_long_stack(self, handler):
        long_stack = "s" * 10000
        report = _make_crash_report(stack=long_stack)
        body = {"reports": [report]}
        h = _make_handler_with_body(body)

        handler.handle_post("/api/observability/crashes", {}, h)

        store = get_crash_store()
        stored_stack = store[-1]["stack"]
        assert stored_stack is not None
        assert len(stored_stack) <= 5000

    def test_ingest_rate_limit_exceeded(self, handler):
        # Exhaust rate limit for this IP
        for _ in range(IP_RATE_LIMIT):
            _check_ip_rate_limit("192.168.1.100")

        body = {"reports": [_make_crash_report()]}
        h = _make_handler_with_body(body, client_ip="192.168.1.100")

        result = handler.handle_post("/api/observability/crashes", {}, h)

        assert _status_code(result) == 429

    def test_ingest_stores_entry_fields(self, handler):
        report = _make_crash_report(
            message="Test error",
            stack="at render()",
            component_name="Dashboard",
            url="https://app.example.com",
            session_id="sess-xyz",
            user_agent="TestAgent/1.0",
        )
        body = {"reports": [report]}
        h = _make_handler_with_body(body)

        handler.handle_post("/api/observability/crashes", {}, h)

        store = get_crash_store()
        entry = store[-1]
        assert entry["message"] == "Test error"
        assert entry["stack"] == "at render()"
        assert entry["component_name"] == "Dashboard"
        assert entry["url"] == "https://app.example.com"
        assert entry["session_id"] == "sess-xyz"
        assert entry["user_agent"] == "TestAgent/1.0"
        assert "fingerprint" in entry
        assert "received_at" in entry
        assert "client_ip" in entry

    def test_ingest_post_wrong_path_returns_none(self, handler):
        h = _make_handler_with_body({})
        result = handler.handle_post("/api/observability/crashes/stats", {}, h)
        assert result is None

    def test_ingest_null_stack(self, handler):
        report = _make_crash_report(stack=None)
        body = {"reports": [report]}
        h = _make_handler_with_body(body)

        handler.handle_post("/api/observability/crashes", {}, h)

        store = get_crash_store()
        assert store[-1]["stack"] is None

    def test_ingest_non_string_stack_treated_as_none(self, handler):
        report = _make_crash_report()
        report["stack"] = 12345  # non-string
        body = {"reports": [report]}
        h = _make_handler_with_body(body)

        handler.handle_post("/api/observability/crashes", {}, h)

        store = get_crash_store()
        assert store[-1]["stack"] is None

    def test_ingest_extracts_ip_from_x_forwarded_for(self, handler):
        report = _make_crash_report()
        body = {"reports": [report]}
        h = MagicMock()
        body_bytes = json.dumps(body).encode("utf-8")
        h.rfile = MagicMock()
        h.rfile.read.return_value = body_bytes
        h.headers = {
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "application/json",
            "X-Forwarded-For": "10.0.0.5, 10.0.0.1",
            "X-Real-IP": "10.0.0.99",
        }
        # No client_address to force header fallback
        del h.client_address

        handler.handle_post("/api/observability/crashes", {}, h)

        store = get_crash_store()
        assert store[-1]["client_ip"] == "10.0.0.5"


# ---------------------------------------------------------------------------
# GET /api/observability/crashes (list)
# ---------------------------------------------------------------------------


class TestListCrashes:
    """Test crash listing endpoint."""

    def test_list_empty_store(self, handler):
        h = MagicMock()
        result = handler.handle("/api/observability/crashes", {}, h)

        data = _parse_response(result)
        assert data["crashes"] == []
        assert data["total"] == 0

    def test_list_with_data(self, handler):
        # Populate store first
        body = {"reports": [_make_crash_report(message=f"Error {i}") for i in range(5)]}
        ingest_h = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, ingest_h)

        h = MagicMock()
        result = handler.handle("/api/observability/crashes", {}, h)

        data = _parse_response(result)
        assert data["total"] == 5
        assert len(data["crashes"]) == 5

    def test_list_pagination_limit(self, handler):
        body = {"reports": [_make_crash_report(message=f"Error {i}") for i in range(10)]}
        ingest_h = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, ingest_h)

        h = MagicMock()
        result = handler.handle("/api/observability/crashes", {"limit": "3"}, h)

        data = _parse_response(result)
        assert len(data["crashes"]) == 3
        assert data["limit"] == 3

    def test_list_pagination_offset(self, handler):
        body = {"reports": [_make_crash_report(message=f"Error {i}") for i in range(10)]}
        ingest_h = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, ingest_h)

        h = MagicMock()
        result = handler.handle("/api/observability/crashes", {"limit": "3", "offset": "2"}, h)

        data = _parse_response(result)
        assert len(data["crashes"]) == 3
        assert data["offset"] == 2

    def test_list_limit_capped_at_200(self, handler):
        h = MagicMock()
        result = handler.handle("/api/observability/crashes", {"limit": "999"}, h)

        data = _parse_response(result)
        assert data["limit"] == 200

    def test_list_returns_most_recent_first(self, handler):
        body = {
            "reports": [
                _make_crash_report(message="First"),
                _make_crash_report(message="Second"),
            ]
        }
        ingest_h = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, ingest_h)

        h = MagicMock()
        result = handler.handle("/api/observability/crashes", {}, h)

        data = _parse_response(result)
        # Most recent (last appended) should come first
        assert data["crashes"][0]["message"] == "Second"


# ---------------------------------------------------------------------------
# GET /api/observability/crashes/stats
# ---------------------------------------------------------------------------


class TestStats:
    """Test crash statistics endpoint."""

    def test_stats_empty_store(self, handler):
        h = MagicMock()
        result = handler.handle("/api/observability/crashes/stats", {}, h)

        data = _parse_response(result)
        assert data["total_ingested"] == 0
        assert data["total_stored"] == 0
        assert data["total_duplicates"] == 0
        assert data["unique_fingerprints"] == 0

    def test_stats_after_ingestion(self, handler):
        body = {
            "reports": [
                _make_crash_report(message="Error A"),
                _make_crash_report(message="Error B"),
            ]
        }
        ingest_h = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, ingest_h)

        h = MagicMock()
        result = handler.handle("/api/observability/crashes/stats", {}, h)

        data = _parse_response(result)
        assert data["total_ingested"] == 2
        assert data["total_stored"] == 2
        assert data["unique_fingerprints"] == 2

    def test_stats_tracks_duplicates(self, handler):
        report = _make_crash_report(fingerprint="dup-fp-001")
        body = {"reports": [report]}

        h1 = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, h1)

        h2 = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, h2)

        h = MagicMock()
        result = handler.handle("/api/observability/crashes/stats", {}, h)

        data = _parse_response(result)
        assert data["total_duplicates"] == 1

    def test_stats_last_hour(self, handler):
        body = {"reports": [_make_crash_report()]}
        ingest_h = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, ingest_h)

        h = MagicMock()
        result = handler.handle("/api/observability/crashes/stats", {}, h)

        data = _parse_response(result)
        assert data["last_hour"] >= 1

    def test_stats_last_24h(self, handler):
        body = {"reports": [_make_crash_report()]}
        ingest_h = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, ingest_h)

        h = MagicMock()
        result = handler.handle("/api/observability/crashes/stats", {}, h)

        data = _parse_response(result)
        assert data["last_24h"] >= 1

    def test_stats_top_fingerprints(self, handler):
        body = {
            "reports": [
                _make_crash_report(fingerprint="fp-alpha"),
                _make_crash_report(fingerprint="fp-beta"),
            ]
        }
        ingest_h = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, ingest_h)

        h = MagicMock()
        result = handler.handle("/api/observability/crashes/stats", {}, h)

        data = _parse_response(result)
        assert "top_fingerprints" in data
        fps = [tf["fingerprint"] for tf in data["top_fingerprints"]]
        assert "fp-alpha" in fps
        assert "fp-beta" in fps

    def test_stats_top_components(self, handler):
        body = {
            "reports": [
                _make_crash_report(component_name="Dashboard"),
                _make_crash_report(component_name="Settings", message="Error B"),
                _make_crash_report(component_name="Dashboard", message="Error C"),
            ]
        }
        ingest_h = _make_handler_with_body(body)
        handler.handle_post("/api/observability/crashes", {}, ingest_h)

        h = MagicMock()
        result = handler.handle("/api/observability/crashes/stats", {}, h)

        data = _parse_response(result)
        assert "top_components" in data
        # Dashboard should be first (2 occurrences vs 1)
        if data["top_components"]:
            assert data["top_components"][0]["component"] == "Dashboard"
            assert data["top_components"][0]["count"] == 2


# ---------------------------------------------------------------------------
# GET dispatch
# ---------------------------------------------------------------------------


class TestHandleDispatch:
    """Test GET routing dispatch."""

    def test_handle_unknown_path_returns_none(self, handler):
        h = MagicMock()
        result = handler.handle("/api/observability/unknown", {}, h)
        assert result is None

    def test_handle_routes_to_stats(self, handler):
        h = MagicMock()
        result = handler.handle("/api/observability/crashes/stats", {}, h)
        assert result is not None
        data = _parse_response(result)
        assert "total_ingested" in data

    def test_handle_routes_to_list(self, handler):
        h = MagicMock()
        result = handler.handle("/api/observability/crashes", {}, h)
        assert result is not None
        data = _parse_response(result)
        assert "crashes" in data


# ---------------------------------------------------------------------------
# Module-level store
# ---------------------------------------------------------------------------


class TestCrashStore:
    """Test module-level crash store management."""

    def test_get_crash_store_returns_deque(self):
        store = get_crash_store()
        assert hasattr(store, "append")
        assert hasattr(store, "maxlen")

    def test_reset_crash_store(self):
        store = get_crash_store()
        store.append({"test": True})
        reset_crash_store()
        assert len(get_crash_store()) == 0
