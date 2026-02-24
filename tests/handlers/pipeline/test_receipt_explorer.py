"""Tests for the ReceiptExplorerHandler.

Covers:
  GET  /api/v1/receipts              - List receipts with filtering
  GET  /api/v1/receipts/:id          - Get full receipt + provenance
  GET  /api/v1/receipts/:id/verify   - Re-verify SHA-256 hashes
  register_receipt() and retrieval
  Rate limiting
  Invalid ID validation
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.pipeline.receipts import (
    ReceiptExplorerHandler,
    _receipt_store,
    register_receipt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler() -> ReceiptExplorerHandler:
    from aragora.server.handlers.base import BaseHandler

    orig = BaseHandler.__init__

    def _patched_init(self, server_context=None):
        self.ctx = server_context or {}

    BaseHandler.__init__ = _patched_init
    try:
        return ReceiptExplorerHandler()
    finally:
        BaseHandler.__init__ = orig


def _make_http_handler(client_ip: str = "127.0.0.1") -> MagicMock:
    handler = MagicMock()
    handler.client_address = (client_ip, 12345)
    handler.headers = {"Content-Length": "0"}
    handler.rfile.read.return_value = b"{}"
    return handler


def _body(result) -> dict[str, Any]:
    """Extract JSON body from a HandlerResult."""
    if result is None:
        return {}
    if hasattr(result, "body"):
        raw = result.body
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw) if raw else {}
    if isinstance(result, tuple):
        return result[0] if isinstance(result[0], dict) else json.loads(result[0])
    return {}


def _status(result) -> int:
    """Extract status code from a HandlerResult."""
    if result is None:
        return 0
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple):
        return result[1]
    return 0


def _make_receipt(
    receipt_id: str = "rcpt-abc123",
    pipeline_id: str = "pipe-1",
    status: str = "completed",
    content_hash: str | None = None,
) -> dict[str, Any]:
    """Build a sample receipt dict."""
    execution = {"status": status, "duration_ms": 120}
    provenance = {"stage_1": {"agent": "claude"}}
    if content_hash is None:
        content_str = f"{pipeline_id}:{execution}:{provenance}"
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
    return {
        "receipt_id": receipt_id,
        "pipeline_id": pipeline_id,
        "generated_at": "2026-02-23T10:00:00Z",
        "execution": execution,
        "provenance": provenance,
        "content_hash": content_hash,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_receipt_store():
    """Clear the in-memory receipt store between tests."""
    _receipt_store.clear()
    yield
    _receipt_store.clear()


@pytest.fixture(autouse=True)
def _bypass_rate_limit():
    """Bypass rate limiting for all tests."""
    with patch.object(
        ReceiptExplorerHandler,
        "_check_rate_limit",
        return_value=None,
    ):
        yield


# ===========================================================================
# register_receipt
# ===========================================================================


class TestRegisterReceipt:
    """Tests for the register_receipt helper function."""

    def test_register_stores_receipt(self):
        receipt = _make_receipt("rcpt-1")
        register_receipt(receipt)
        assert "rcpt-1" in _receipt_store
        assert _receipt_store["rcpt-1"] is receipt

    def test_register_empty_id_skipped(self):
        register_receipt({"receipt_id": "", "pipeline_id": "p1"})
        assert len(_receipt_store) == 0

    def test_register_missing_id_skipped(self):
        register_receipt({"pipeline_id": "p1"})
        assert len(_receipt_store) == 0

    def test_register_multiple(self):
        register_receipt(_make_receipt("r1"))
        register_receipt(_make_receipt("r2"))
        assert len(_receipt_store) == 2

    def test_register_overwrites_existing(self):
        register_receipt(_make_receipt("r1", pipeline_id="old"))
        register_receipt(_make_receipt("r1", pipeline_id="new"))
        assert _receipt_store["r1"]["pipeline_id"] == "new"


# ===========================================================================
# GET /api/v1/receipts  (list)
# ===========================================================================


class TestListReceipts:
    """Tests for listing receipts with filtering."""

    def test_list_empty(self):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["receipts"] == []
        assert body["count"] == 0

    def test_list_returns_receipts(self):
        register_receipt(_make_receipt("r1"))
        register_receipt(_make_receipt("r2"))
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts", {}, http)
        body = _body(result)
        assert body["count"] == 2
        ids = [r["receipt_id"] for r in body["receipts"]]
        assert "r1" in ids
        assert "r2" in ids

    def test_list_filter_by_pipeline_id(self):
        register_receipt(_make_receipt("r1", pipeline_id="pipe-A"))
        register_receipt(_make_receipt("r2", pipeline_id="pipe-B"))
        register_receipt(_make_receipt("r3", pipeline_id="pipe-A"))
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts", {"pipeline_id": "pipe-A"}, http)
        body = _body(result)
        assert body["count"] == 2
        for r in body["receipts"]:
            assert r["pipeline_id"] == "pipe-A"

    def test_list_filter_by_status(self):
        register_receipt(_make_receipt("r1", status="completed"))
        register_receipt(_make_receipt("r2", status="failed"))
        register_receipt(_make_receipt("r3", status="completed"))
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts", {"status": "failed"}, http)
        body = _body(result)
        assert body["count"] == 1
        assert body["receipts"][0]["status"] == "failed"

    def test_list_with_limit(self):
        for i in range(10):
            register_receipt(_make_receipt(f"r{i}"))
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts", {"limit": "3"}, http)
        body = _body(result)
        assert body["count"] == 3

    def test_list_combined_filters(self):
        register_receipt(_make_receipt("r1", pipeline_id="pipe-A", status="completed"))
        register_receipt(_make_receipt("r2", pipeline_id="pipe-A", status="failed"))
        register_receipt(_make_receipt("r3", pipeline_id="pipe-B", status="completed"))
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get(
            "/api/v1/receipts",
            {"pipeline_id": "pipe-A", "status": "completed"},
            http,
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["receipts"][0]["receipt_id"] == "r1"

    def test_list_receipt_fields(self):
        register_receipt(_make_receipt("r1"))
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts", {}, http)
        body = _body(result)
        r = body["receipts"][0]
        assert "receipt_id" in r
        assert "pipeline_id" in r
        assert "generated_at" in r
        assert "status" in r
        assert "content_hash" in r


# ===========================================================================
# GET /api/v1/receipts/:id  (get)
# ===========================================================================


class TestGetReceipt:
    """Tests for getting a full receipt."""

    def test_get_existing(self):
        register_receipt(_make_receipt("rcpt-1"))
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts/rcpt-1", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["receipt_id"] == "rcpt-1"
        assert "execution" in body
        assert "provenance" in body

    def test_get_not_found(self):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts/nonexistent", {}, http)
        assert _status(result) == 404

    def test_get_invalid_id(self):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts/<script>alert(1)</script>", {}, http)
        assert _status(result) == 400

    def test_get_returns_full_receipt(self):
        receipt = _make_receipt("rcpt-full")
        register_receipt(receipt)
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts/rcpt-full", {}, http)
        body = _body(result)
        assert body["pipeline_id"] == receipt["pipeline_id"]
        assert body["content_hash"] == receipt["content_hash"]


# ===========================================================================
# GET /api/v1/receipts/:id/verify  (verify)
# ===========================================================================


class TestVerifyReceipt:
    """Tests for receipt hash verification."""

    def test_verify_valid_hash(self):
        receipt = _make_receipt("rcpt-v1")
        register_receipt(receipt)
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts/rcpt-v1/verify", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is True
        assert body["receipt_id"] == "rcpt-v1"
        assert body["stored_hash"] == body["recomputed_hash"]

    def test_verify_invalid_hash(self):
        receipt = _make_receipt("rcpt-v2", content_hash="badhash000")
        register_receipt(receipt)
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts/rcpt-v2/verify", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is False
        assert body["stored_hash"] == "badhash000"
        assert body["recomputed_hash"] != "badhash000"

    def test_verify_not_found(self):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts/nonexistent/verify", {}, http)
        assert _status(result) == 404

    def test_verify_returns_pipeline_id(self):
        receipt = _make_receipt("rcpt-v3", pipeline_id="pipe-xyz")
        register_receipt(receipt)
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts/rcpt-v3/verify", {}, http)
        body = _body(result)
        assert body["pipeline_id"] == "pipe-xyz"

    def test_verify_invalid_id(self):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts/<script>/verify", {}, http)
        assert _status(result) == 400


# ===========================================================================
# Routing edge cases
# ===========================================================================


class TestRoutingEdgeCases:
    """Tests for unrecognized routes."""

    def test_unrecognized_path_returns_none(self):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/other", {}, http)
        assert result is None

    def test_too_short_path_returns_none(self):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1", {}, http)
        assert result is None

    def test_extra_segments_returns_none(self):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_get("/api/v1/receipts/rcpt-1/verify/extra", {}, http)
        assert result is None


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limit enforcement."""

    def test_rate_limited_returns_429(self):
        h = _make_handler()
        http = _make_http_handler()
        with patch.object(h, "_check_rate_limit") as mock_rl:
            mock_rl.return_value = MagicMock(
                status_code=429,
                body=json.dumps({"error": "Rate limit exceeded"}).encode(),
            )
            result = h.handle_get("/api/v1/receipts", {}, http)
        assert _status(result) == 429


__all__ = []
