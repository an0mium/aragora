"""
Tests for aragora.server.handlers.receipt_export - Receipt Export Handlers.

Tests cover:
- ReceiptExportHandler: instantiation, can_handle
- GET /api/v1/receipts/:id/export: json, html, pdf formats
- Invalid format parameter
- Missing receipt ID in path
- Receipt not found (neither store nor ctx)
- Receipt found in ctx fallback
- _VALID_FORMATS constant
- handle() routing: returns None for non-matching paths
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.receipt_export import (
    ReceiptExportHandler,
    _VALID_FORMATS,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Mock Objects
# ===========================================================================


class MockReceipt:
    """Mock receipt object."""

    def __init__(self, receipt_id: str = "receipt-001"):
        self.id = receipt_id
        self.debate_id = "debate-001"
        self.decision = "Approved"
        self.timestamp = "2026-02-14T10:00:00Z"
        self.hash = "abc123"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "debate_id": self.debate_id,
            "decision": self.decision,
            "timestamp": self.timestamp,
            "hash": self.hash,
        }


class MockReceiptStore:
    """Mock receipt store."""

    def __init__(self, receipts: dict[str, MockReceipt] | None = None):
        self._receipts = receipts or {}

    def get(self, receipt_id: str) -> MockReceipt | None:
        return self._receipts.get(receipt_id)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create a ReceiptExportHandler with no store."""
    with patch(
        "aragora.server.handlers.receipt_export._get_receipt_store",
        return_value=None,
    ):
        h = ReceiptExportHandler(ctx={})
        return h


@pytest.fixture
def handler_with_store():
    """Create a ReceiptExportHandler with a populated store."""
    store = MockReceiptStore({"receipt-001": MockReceipt("receipt-001")})
    with patch(
        "aragora.server.handlers.receipt_export._get_receipt_store",
        return_value=store,
    ):
        h = ReceiptExportHandler(ctx={})
        yield h


@pytest.fixture
def handler_with_ctx_store():
    """Create a ReceiptExportHandler with receipt in ctx."""
    receipt = MockReceipt("receipt-ctx")
    ctx = {"receipt_store": {"receipt-ctx": receipt}}
    with patch(
        "aragora.server.handlers.receipt_export._get_receipt_store",
        return_value=None,
    ):
        h = ReceiptExportHandler(ctx=ctx)
        yield h


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestReceiptExportHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, ReceiptExportHandler)

    def test_valid_formats(self):
        assert "json" in _VALID_FORMATS
        assert "html" in _VALID_FORMATS
        assert "pdf" in _VALID_FORMATS
        assert len(_VALID_FORMATS) == 3


# ===========================================================================
# Test can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle routing logic."""

    def test_can_handle_export_path(self, handler):
        assert handler.can_handle("/api/v1/receipts/receipt-001/export") is True

    def test_can_handle_different_id(self, handler):
        assert handler.can_handle("/api/v1/receipts/abc-xyz/export") is True

    def test_cannot_handle_without_export(self, handler):
        assert handler.can_handle("/api/v1/receipts/receipt-001") is False

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/v1/debates/123") is False

    def test_cannot_handle_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/receipt/123/export") is False


# ===========================================================================
# Test JSON Export
# ===========================================================================


class TestJsonExport:
    """Tests for JSON format export."""

    def test_export_json_success(self, handler_with_store):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.receipt_export._get_receipt_store",
            return_value=MockReceiptStore({"receipt-001": MockReceipt("receipt-001")}),
        ):
            result = handler_with_store.handle(
                "/api/v1/receipts/receipt-001/export",
                {"format": "json"},
                mock_handler,
            )
            assert result is not None
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["id"] == "receipt-001"
            assert data["decision"] == "Approved"

    def test_export_default_format_is_json(self, handler_with_store):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.receipt_export._get_receipt_store",
            return_value=MockReceiptStore({"receipt-001": MockReceipt("receipt-001")}),
        ):
            result = handler_with_store.handle(
                "/api/v1/receipts/receipt-001/export",
                {},
                mock_handler,
            )
            assert result is not None
            assert result.status_code == 200
            data = _parse_body(result)
            assert "id" in data


# ===========================================================================
# Test HTML Export
# ===========================================================================


class TestHtmlExport:
    """Tests for HTML format export."""

    def test_export_html_success(self, handler_with_store):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.receipt_export._get_receipt_store",
            return_value=MockReceiptStore({"receipt-001": MockReceipt("receipt-001")}),
        ):
            with patch(
                "aragora.gauntlet.export.receipt_to_html",
                return_value="<html><body>Receipt</body></html>",
            ):
                result = handler_with_store.handle(
                    "/api/v1/receipts/receipt-001/export",
                    {"format": "html"},
                    mock_handler,
                )
                assert result is not None
                assert result.status_code == 200
                assert result.content_type == "text/html; charset=utf-8"
                assert b"Receipt" in result.body
                assert result.headers["Content-Disposition"].startswith("attachment")


# ===========================================================================
# Test PDF Export
# ===========================================================================


class TestPdfExport:
    """Tests for PDF format export."""

    def test_export_pdf_success(self, handler_with_store):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.receipt_export._get_receipt_store",
            return_value=MockReceiptStore({"receipt-001": MockReceipt("receipt-001")}),
        ):
            with patch(
                "aragora.gauntlet.export.receipt_to_pdf",
                return_value=b"%PDF-1.4 fake content",
            ):
                result = handler_with_store.handle(
                    "/api/v1/receipts/receipt-001/export",
                    {"format": "pdf"},
                    mock_handler,
                )
                assert result is not None
                assert result.status_code == 200
                assert result.content_type == "application/pdf"
                assert b"%PDF" in result.body
                assert result.headers["Content-Disposition"].startswith("attachment")


# ===========================================================================
# Test Invalid Format
# ===========================================================================


class TestInvalidFormat:
    """Tests for invalid format parameter."""

    def test_invalid_format(self, handler_with_store):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.receipt_export._get_receipt_store",
            return_value=MockReceiptStore({"receipt-001": MockReceipt("receipt-001")}),
        ):
            result = handler_with_store.handle(
                "/api/v1/receipts/receipt-001/export",
                {"format": "xml"},
                mock_handler,
            )
            assert result is not None
            assert result.status_code == 400
            data = _parse_body(result)
            assert "xml" in data["error"].lower() or "invalid" in data["error"].lower()

    def test_invalid_format_csv(self, handler_with_store):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.receipt_export._get_receipt_store",
            return_value=MockReceiptStore({"receipt-001": MockReceipt("receipt-001")}),
        ):
            result = handler_with_store.handle(
                "/api/v1/receipts/receipt-001/export",
                {"format": "csv"},
                mock_handler,
            )
            assert result is not None
            assert result.status_code == 400


# ===========================================================================
# Test Receipt Not Found
# ===========================================================================


class TestReceiptNotFound:
    """Tests for receipt not found scenarios."""

    def test_receipt_not_in_store(self, handler):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.receipt_export._get_receipt_store",
            return_value=None,
        ):
            result = handler.handle(
                "/api/v1/receipts/nonexistent/export",
                {"format": "json"},
                mock_handler,
            )
            assert result is not None
            assert result.status_code == 404

    def test_receipt_not_in_store_or_ctx(self, handler):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.receipt_export._get_receipt_store",
            return_value=MockReceiptStore({}),
        ):
            result = handler.handle(
                "/api/v1/receipts/nonexistent/export",
                {"format": "json"},
                mock_handler,
            )
            assert result is not None
            assert result.status_code == 404


# ===========================================================================
# Test Receipt Found in Context Fallback
# ===========================================================================


class TestReceiptCtxFallback:
    """Tests for finding receipt in ctx when store returns None."""

    def test_receipt_found_in_ctx(self, handler_with_ctx_store):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.receipt_export._get_receipt_store",
            return_value=None,
        ):
            result = handler_with_ctx_store.handle(
                "/api/v1/receipts/receipt-ctx/export",
                {"format": "json"},
                mock_handler,
            )
            assert result is not None
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["id"] == "receipt-ctx"


# ===========================================================================
# Test handle() Routing
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() method routing."""

    def test_handle_non_matching_path_returns_none(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/debates/123", {"format": "json"}, mock_handler)
        assert result is None

    def test_handle_short_path_returns_400(self, handler):
        """Path too short to extract receipt ID."""
        mock_handler = _make_mock_handler()
        # Force can_handle to return True for this test
        with patch.object(handler, "can_handle", return_value=True):
            result = handler.handle("/api/export", {"format": "json"}, mock_handler)
            assert result is not None
            assert result.status_code == 400

    def test_handle_extracts_receipt_id(self, handler_with_store):
        """Verify the receipt ID is correctly extracted from the path."""
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.receipt_export._get_receipt_store",
            return_value=MockReceiptStore({"my-receipt-123": MockReceipt("my-receipt-123")}),
        ):
            result = handler_with_store.handle(
                "/api/v1/receipts/my-receipt-123/export",
                {"format": "json"},
                mock_handler,
            )
            assert result is not None
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["id"] == "my-receipt-123"
