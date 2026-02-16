"""
Tests for aragora.server.handlers.audit_trail - Audit Trail HTTP Handler.

Tests cover:
- Route handling and can_handle
- Audit trail listing with pagination and filtering
- Audit trail retrieval by ID
- Audit trail export (JSON, CSV, Markdown formats)
- Audit trail integrity verification
- Decision receipt listing with pagination and filtering
- Decision receipt retrieval by ID
- Decision receipt integrity verification
- Error handling (not found, invalid IDs, storage failures)
- Fallback to in-memory storage
- Loading trails/receipts from gauntlet handler

These tests verify the compliance-critical audit trail functionality.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.audit_trail import AuditTrailHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "user"
    permissions: list = field(
        default_factory=lambda: [
            "audit:read",
            "audit:export",
            "audit:verify",
            "audit:receipts.read",
            "audit:receipts.verify",
        ]
    )


class MockAuditTrailStore:
    """Mock audit trail store for testing."""

    def __init__(self):
        self.trails: dict[str, dict[str, Any]] = {}
        self.receipts: dict[str, dict[str, Any]] = {}

    def list_trails(
        self,
        limit: int = 20,
        offset: int = 0,
        verdict: str | None = None,
    ) -> list[dict[str, Any]]:
        trails = list(self.trails.values())
        if verdict:
            trails = [t for t in trails if t.get("verdict") == verdict]
        trails.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return trails[offset : offset + limit]

    def count_trails(self, verdict: str | None = None) -> int:
        if verdict:
            return len([t for t in self.trails.values() if t.get("verdict") == verdict])
        return len(self.trails)

    def get_trail(self, trail_id: str) -> dict[str, Any] | None:
        return self.trails.get(trail_id)

    def get_trail_by_gauntlet(self, gauntlet_id: str) -> dict[str, Any] | None:
        for trail in self.trails.values():
            if trail.get("gauntlet_id") == gauntlet_id:
                return trail
        return None

    def save_trail(self, trail_dict: dict[str, Any]) -> None:
        trail_id = trail_dict.get("trail_id", "")
        self.trails[trail_id] = trail_dict

    def list_receipts(
        self,
        limit: int = 20,
        offset: int = 0,
        verdict: str | None = None,
        risk_level: str | None = None,
    ) -> list[dict[str, Any]]:
        receipts = list(self.receipts.values())
        if verdict:
            receipts = [r for r in receipts if r.get("verdict") == verdict]
        if risk_level:
            receipts = [r for r in receipts if r.get("risk_level") == risk_level]
        receipts.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return receipts[offset : offset + limit]

    def count_receipts(
        self,
        verdict: str | None = None,
        risk_level: str | None = None,
    ) -> int:
        receipts = list(self.receipts.values())
        if verdict:
            receipts = [r for r in receipts if r.get("verdict") == verdict]
        if risk_level:
            receipts = [r for r in receipts if r.get("risk_level") == risk_level]
        return len(receipts)

    def get_receipt(self, receipt_id: str) -> dict[str, Any] | None:
        return self.receipts.get(receipt_id)

    def get_receipt_by_gauntlet(self, gauntlet_id: str) -> dict[str, Any] | None:
        for receipt in self.receipts.values():
            if receipt.get("gauntlet_id") == gauntlet_id:
                return receipt
        return None

    def save_receipt(self, receipt_dict: dict[str, Any]) -> None:
        receipt_id = receipt_dict.get("receipt_id", "")
        self.receipts[receipt_id] = receipt_dict


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    path: str = "/api/v1/audit-trails",
) -> MagicMock:
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.path = path
    handler.headers = {}
    handler.client_address = ("127.0.0.1", 12345)

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


def make_sample_trail(
    trail_id: str = "trail-test123", gauntlet_id: str = "test123"
) -> dict[str, Any]:
    """Create a sample audit trail dict."""
    return {
        "trail_id": trail_id,
        "gauntlet_id": gauntlet_id,
        "created_at": datetime.now().isoformat(),
        "verdict": "APPROVED",
        "confidence": 0.85,
        "total_findings": 3,
        "duration_seconds": 45.0,
        "input_summary": "Test input",
        "input_type": "spec",
        "agents_involved": ["claude", "gpt4"],
        "events": [
            {
                "event_id": "evt-00001",
                "event_type": "gauntlet_start",
                "timestamp": datetime.now().isoformat(),
                "source": "GauntletOrchestrator",
                "description": "Started Gauntlet stress-test",
                "details": {},
                "severity": "info",
                "agent": None,
                "parent_event_id": None,
            }
        ],
        "checksum": "abc123def456",
    }


def make_sample_receipt(
    receipt_id: str = "receipt-test123",
    gauntlet_id: str = "test123",
) -> dict[str, Any]:
    """Create a sample decision receipt dict."""
    content = json.dumps(
        {
            "receipt_id": receipt_id,
            "gauntlet_id": gauntlet_id,
            "verdict": "APPROVED",
            "confidence": 0.85,
        },
        sort_keys=True,
    )
    checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
    return {
        "receipt_id": receipt_id,
        "gauntlet_id": gauntlet_id,
        "timestamp": datetime.now().isoformat(),
        "verdict": "APPROVED",
        "confidence": 0.85,
        "risk_level": "LOW",
        "findings": [{"title": "Minor issue", "severity": "low"}],
        "checksum": checksum,
    }


@pytest.fixture
def mock_store():
    """Create a mock audit trail store."""
    return MockAuditTrailStore()


@pytest.fixture
def audit_trail_handler(mock_store):
    """Create AuditTrailHandler with mock context and store."""
    ctx = {"stream_emitter": None}
    # Patch the store import in the module where it's used
    with patch(
        "aragora.storage.audit_trail_store.get_audit_trail_store",
        return_value=mock_store,
    ):
        handler = AuditTrailHandler(ctx)
        # Override the store directly
        handler._store = mock_store
        # Clear class-level storage
        AuditTrailHandler._trails.clear()
        AuditTrailHandler._receipts.clear()
        yield handler
        # Clean up after test
        AuditTrailHandler._trails.clear()
        AuditTrailHandler._receipts.clear()


@pytest.fixture(autouse=True)
def clear_rate_limiters():
    """Clear rate limiters before each test."""
    from aragora.server.handlers.utils.rate_limit import _limiters

    for limiter in _limiters.values():
        limiter.clear()
    yield
    for limiter in _limiters.values():
        limiter.clear()


# ===========================================================================
# Test Routing (can_handle)
# ===========================================================================


class TestAuditTrailHandlerRouting:
    """Tests for AuditTrailHandler.can_handle."""

    def test_can_handle_audit_trails_list_get(self, audit_trail_handler):
        assert audit_trail_handler.can_handle("/api/v1/audit-trails", "GET") is True

    def test_can_handle_audit_trail_get(self, audit_trail_handler):
        assert audit_trail_handler.can_handle("/api/v1/audit-trails/trail-123", "GET") is True

    def test_can_handle_audit_trail_export(self, audit_trail_handler):
        assert (
            audit_trail_handler.can_handle("/api/v1/audit-trails/trail-123/export", "GET") is True
        )

    def test_can_handle_audit_trail_verify(self, audit_trail_handler):
        assert (
            audit_trail_handler.can_handle("/api/v1/audit-trails/trail-123/verify", "POST") is True
        )

    def test_can_handle_receipts_list_get(self, audit_trail_handler):
        assert audit_trail_handler.can_handle("/api/v1/receipts", "GET") is True

    def test_can_handle_receipt_get(self, audit_trail_handler):
        assert audit_trail_handler.can_handle("/api/v1/receipts/receipt-123", "GET") is True

    def test_can_handle_receipt_verify(self, audit_trail_handler):
        assert audit_trail_handler.can_handle("/api/v1/receipts/receipt-123/verify", "POST") is True

    def test_cannot_handle_other_paths(self, audit_trail_handler):
        assert audit_trail_handler.can_handle("/api/v1/debates", "GET") is False

    def test_cannot_handle_unsupported_method(self, audit_trail_handler):
        assert audit_trail_handler.can_handle("/api/v1/audit-trails", "DELETE") is False


# ===========================================================================
# Test List Audit Trails
# ===========================================================================


class TestAuditTrailList:
    """Tests for listing audit trails."""

    @pytest.mark.asyncio
    async def test_list_audit_trails_empty(self, audit_trail_handler, mock_store):
        handler = make_mock_handler()

        with patch.object(audit_trail_handler, "_list_audit_trails") as mock_method:
            mock_method.return_value = MagicMock(
                status_code=200,
                body=json.dumps({"trails": [], "total": 0, "limit": 20, "offset": 0}).encode(),
            )
            result = await audit_trail_handler.handle(
                "/api/v1/audit-trails", {"limit": "20", "offset": "0"}, handler
            )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_audit_trails_with_data(self, audit_trail_handler, mock_store):
        # Add test data
        mock_store.trails["trail-test123"] = make_sample_trail()
        mock_store.trails["trail-test456"] = make_sample_trail("trail-test456", "test456")

        handler = make_mock_handler()

        # Call internal method directly to bypass permission checks
        result = audit_trail_handler._list_audit_trails({"limit": "20", "offset": "0"})

        # Handle async if needed
        if hasattr(result, "__await__"):
            result = await result

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "trails" in data
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_list_audit_trails_with_pagination(self, audit_trail_handler, mock_store):
        # Add multiple trails
        for i in range(5):
            mock_store.trails[f"trail-test{i}"] = make_sample_trail(f"trail-test{i}", f"test{i}")

        handler = make_mock_handler()

        # Call internal method directly
        result = audit_trail_handler._list_audit_trails({"limit": "2", "offset": "1"})
        if hasattr(result, "__await__"):
            result = await result

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["trails"]) <= 2
        assert data["limit"] == 2
        assert data["offset"] == 1
        assert data["total"] == 5

    @pytest.mark.asyncio
    async def test_list_audit_trails_filter_by_verdict(self, audit_trail_handler, mock_store):
        # Add trails with different verdicts
        trail1 = make_sample_trail("trail-approved", "approved1")
        trail1["verdict"] = "APPROVED"
        trail2 = make_sample_trail("trail-rejected", "rejected1")
        trail2["verdict"] = "REJECTED"
        mock_store.trails["trail-approved"] = trail1
        mock_store.trails["trail-rejected"] = trail2

        handler = make_mock_handler()

        result = audit_trail_handler._list_audit_trails({"verdict": "APPROVED"})
        if hasattr(result, "__await__"):
            result = await result

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_list_audit_trails_fallback_to_memory(self, audit_trail_handler, mock_store):
        """Test fallback to in-memory storage when database is empty."""
        # Add data to in-memory cache only
        AuditTrailHandler._trails["trail-inmem"] = make_sample_trail("trail-inmem", "inmem1")

        handler = make_mock_handler()

        result = audit_trail_handler._list_audit_trails({})
        if hasattr(result, "__await__"):
            result = await result

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        # Should fall back to in-memory data
        assert data["total"] >= 1


# ===========================================================================
# Test Get Audit Trail
# ===========================================================================


class TestAuditTrailGet:
    """Tests for retrieving a specific audit trail."""

    @pytest.mark.asyncio
    async def test_get_audit_trail_success(self, audit_trail_handler, mock_store):
        trail = make_sample_trail()
        mock_store.trails["trail-test123"] = trail

        handler = make_mock_handler(path="/api/v1/audit-trails/trail-test123")

        result = await audit_trail_handler._get_audit_trail("trail-test123")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["trail_id"] == "trail-test123"

    @pytest.mark.asyncio
    async def test_get_audit_trail_not_found(self, audit_trail_handler, mock_store):
        handler = make_mock_handler(path="/api/v1/audit-trails/trail-nonexistent")

        result = await audit_trail_handler._get_audit_trail("trail-nonexistent")

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_audit_trail_from_memory(self, audit_trail_handler, mock_store):
        """Test retrieval from in-memory cache when not in database."""
        trail = make_sample_trail("trail-memory", "memory1")
        AuditTrailHandler._trails["trail-memory"] = trail

        result = await audit_trail_handler._get_audit_trail("trail-memory")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["trail_id"] == "trail-memory"


# ===========================================================================
# Test Export Audit Trail
# ===========================================================================


class TestAuditTrailExport:
    """Tests for audit trail export functionality."""

    @pytest.mark.asyncio
    async def test_export_audit_trail_json(self, audit_trail_handler, mock_store):
        trail = make_sample_trail()
        mock_store.trails["trail-test123"] = trail

        handler = make_mock_handler(path="/api/v1/audit-trails/trail-test123/export")

        result = await audit_trail_handler._export_audit_trail("trail-test123", {"format": "json"})

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert result.headers.get("Content-Disposition") is not None
        assert "trail-test123.json" in result.headers["Content-Disposition"]

    @pytest.mark.asyncio
    async def test_export_audit_trail_csv(self, audit_trail_handler, mock_store):
        trail = make_sample_trail()
        mock_store.trails["trail-test123"] = trail

        result = await audit_trail_handler._export_audit_trail("trail-test123", {"format": "csv"})

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "text/csv"
        assert "trail-test123.csv" in result.headers.get("Content-Disposition", "")

    @pytest.mark.asyncio
    async def test_export_audit_trail_markdown(self, audit_trail_handler, mock_store):
        trail = make_sample_trail()
        mock_store.trails["trail-test123"] = trail

        result = await audit_trail_handler._export_audit_trail("trail-test123", {"format": "md"})

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "text/markdown"
        assert "trail-test123.md" in result.headers.get("Content-Disposition", "")

    @pytest.mark.asyncio
    async def test_export_audit_trail_unknown_format(self, audit_trail_handler, mock_store):
        trail = make_sample_trail()
        mock_store.trails["trail-test123"] = trail

        result = await audit_trail_handler._export_audit_trail("trail-test123", {"format": "xml"})

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_export_audit_trail_not_found(self, audit_trail_handler, mock_store):
        result = await audit_trail_handler._export_audit_trail("trail-nonexistent", {})

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_export_audit_trail_default_json(self, audit_trail_handler, mock_store):
        """Test that default format is JSON."""
        trail = make_sample_trail()
        mock_store.trails["trail-test123"] = trail

        result = await audit_trail_handler._export_audit_trail("trail-test123", {})

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "application/json"


# ===========================================================================
# Test Verify Audit Trail
# ===========================================================================


class TestAuditTrailVerify:
    """Tests for audit trail integrity verification."""

    @pytest.mark.asyncio
    async def test_verify_audit_trail_valid(self, audit_trail_handler, mock_store):
        trail = make_sample_trail()
        AuditTrailHandler._trails["trail-test123"] = trail

        result = await audit_trail_handler._verify_audit_trail("trail-test123")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "trail_id" in data
        assert "valid" in data
        assert "stored_checksum" in data
        assert "computed_checksum" in data

    @pytest.mark.asyncio
    async def test_verify_audit_trail_not_found(self, audit_trail_handler, mock_store):
        result = await audit_trail_handler._verify_audit_trail("trail-nonexistent")

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_verify_audit_trail_with_error(self, audit_trail_handler, mock_store):
        """Test verification returns error info gracefully."""
        # Add trail with data that might cause export module issues
        trail = {"trail_id": "trail-broken", "gauntlet_id": "broken"}
        AuditTrailHandler._trails["trail-broken"] = trail

        with patch(
            "aragora.export.audit_trail.AuditTrail.from_json", side_effect=ValueError("Parse error")
        ):
            result = await audit_trail_handler._verify_audit_trail("trail-broken")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["valid"] is False
        assert "error" in data


# ===========================================================================
# Test List Receipts
# ===========================================================================


class TestReceiptList:
    """Tests for listing decision receipts."""

    @pytest.mark.asyncio
    async def test_list_receipts_empty(self, audit_trail_handler, mock_store):
        result = await audit_trail_handler._list_receipts({})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["receipts"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_receipts_with_data(self, audit_trail_handler, mock_store):
        mock_store.receipts["receipt-test123"] = make_sample_receipt()
        mock_store.receipts["receipt-test456"] = make_sample_receipt("receipt-test456", "test456")

        result = await audit_trail_handler._list_receipts({})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_list_receipts_with_pagination(self, audit_trail_handler, mock_store):
        for i in range(5):
            mock_store.receipts[f"receipt-test{i}"] = make_sample_receipt(
                f"receipt-test{i}", f"test{i}"
            )

        result = await audit_trail_handler._list_receipts({"limit": "2", "offset": "1"})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["receipts"]) <= 2
        assert data["limit"] == 2
        assert data["offset"] == 1

    @pytest.mark.asyncio
    async def test_list_receipts_filter_by_verdict(self, audit_trail_handler, mock_store):
        receipt1 = make_sample_receipt("receipt-approved", "approved1")
        receipt1["verdict"] = "APPROVED"
        receipt2 = make_sample_receipt("receipt-rejected", "rejected1")
        receipt2["verdict"] = "REJECTED"
        mock_store.receipts["receipt-approved"] = receipt1
        mock_store.receipts["receipt-rejected"] = receipt2

        result = await audit_trail_handler._list_receipts({"verdict": "APPROVED"})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_list_receipts_filter_by_risk_level(self, audit_trail_handler, mock_store):
        receipt1 = make_sample_receipt("receipt-low", "low1")
        receipt1["risk_level"] = "LOW"
        receipt2 = make_sample_receipt("receipt-high", "high1")
        receipt2["risk_level"] = "HIGH"
        mock_store.receipts["receipt-low"] = receipt1
        mock_store.receipts["receipt-high"] = receipt2

        result = await audit_trail_handler._list_receipts({"risk_level": "LOW"})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_list_receipts_fallback_to_memory(self, audit_trail_handler, mock_store):
        """Test fallback to in-memory storage when database is empty."""
        AuditTrailHandler._receipts["receipt-inmem"] = make_sample_receipt(
            "receipt-inmem", "inmem1"
        )

        result = await audit_trail_handler._list_receipts({})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total"] >= 1


# ===========================================================================
# Test Get Receipt
# ===========================================================================


class TestReceiptGet:
    """Tests for retrieving a specific decision receipt."""

    @pytest.mark.asyncio
    async def test_get_receipt_success(self, audit_trail_handler, mock_store):
        receipt = make_sample_receipt()
        mock_store.receipts["receipt-test123"] = receipt

        result = await audit_trail_handler._get_receipt("receipt-test123")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["receipt_id"] == "receipt-test123"

    @pytest.mark.asyncio
    async def test_get_receipt_not_found(self, audit_trail_handler, mock_store):
        result = await audit_trail_handler._get_receipt("receipt-nonexistent")

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_receipt_from_memory(self, audit_trail_handler, mock_store):
        """Test retrieval from in-memory cache when not in database."""
        receipt = make_sample_receipt("receipt-memory", "memory1")
        AuditTrailHandler._receipts["receipt-memory"] = receipt

        result = await audit_trail_handler._get_receipt("receipt-memory")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["receipt_id"] == "receipt-memory"


# ===========================================================================
# Test Verify Receipt
# ===========================================================================


class TestReceiptVerify:
    """Tests for decision receipt integrity verification."""

    @pytest.mark.asyncio
    async def test_verify_receipt_valid(self, audit_trail_handler, mock_store):
        receipt = make_sample_receipt()
        mock_store.receipts["receipt-test123"] = receipt

        result = await audit_trail_handler._verify_receipt("receipt-test123")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "receipt_id" in data
        assert "valid" in data
        assert "stored_checksum" in data
        assert "computed_checksum" in data
        assert data["match"] is True

    @pytest.mark.asyncio
    async def test_verify_receipt_not_found(self, audit_trail_handler, mock_store):
        result = await audit_trail_handler._verify_receipt("receipt-nonexistent")

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_verify_receipt_tampered(self, audit_trail_handler, mock_store):
        """Test verification detects tampered data."""
        receipt = make_sample_receipt()
        receipt["checksum"] = "tampered_checksum"
        mock_store.receipts["receipt-test123"] = receipt

        result = await audit_trail_handler._verify_receipt("receipt-test123")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["match"] is False

    @pytest.mark.asyncio
    async def test_verify_receipt_with_error(self, audit_trail_handler, mock_store):
        """Test verification returns error info gracefully on exception."""
        receipt = {"receipt_id": "receipt-broken"}
        mock_store.receipts["receipt-broken"] = receipt

        # Mock hashlib.sha256 to raise an exception during verification
        with patch("hashlib.sha256", side_effect=ValueError("Hash error")):
            result = await audit_trail_handler._verify_receipt("receipt-broken")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["valid"] is False
        assert "error" in data


# ===========================================================================
# Test Load from Gauntlet Handler
# ===========================================================================


class TestLoadFromGauntlet:
    """Tests for loading trails/receipts from gauntlet handler."""

    @pytest.mark.asyncio
    async def test_load_trail_from_gauntlet_by_trail_id(self, audit_trail_handler, mock_store):
        """Test loading trail when trail_id starts with 'trail-'."""
        trail = make_sample_trail("trail-gauntlet123", "gauntlet123")
        mock_store.trails["trail-gauntlet123"] = trail

        result = await audit_trail_handler._load_trail_from_gauntlet("trail-gauntlet123")

        assert result is not None
        assert result["trail_id"] == "trail-gauntlet123"

    @pytest.mark.asyncio
    async def test_load_trail_from_gauntlet_not_found(self, audit_trail_handler, mock_store):
        """Test loading trail returns None when not found."""
        result = await audit_trail_handler._load_trail_from_gauntlet("trail-nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_load_receipt_from_gauntlet_by_receipt_id(self, audit_trail_handler, mock_store):
        """Test loading receipt when receipt_id starts with 'receipt-'."""
        receipt = make_sample_receipt("receipt-gauntlet123", "gauntlet123")
        mock_store.receipts["receipt-gauntlet123"] = receipt

        # This should check the store by gauntlet_id
        result = await audit_trail_handler._load_receipt_from_gauntlet("receipt-gauntlet123")

        # The method extracts gauntlet_id from receipt_id and queries by that
        assert result is not None
        assert result["receipt_id"] == "receipt-gauntlet123"

    @pytest.mark.asyncio
    async def test_load_receipt_from_gauntlet_not_found(self, audit_trail_handler, mock_store):
        """Test loading receipt returns None when not found."""
        result = await audit_trail_handler._load_receipt_from_gauntlet("receipt-nonexistent")

        assert result is None


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in audit trail handler."""

    @pytest.mark.asyncio
    async def test_handle_not_found_path(self, audit_trail_handler):
        handler = make_mock_handler(path="/api/v1/unknown-path")

        result = await audit_trail_handler.handle("/api/v1/unknown-path", {}, handler)

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_handle_internal_error(self, audit_trail_handler, mock_store):
        """Test internal error handling."""
        handler = make_mock_handler(path="/api/v1/audit-trails")

        # Force an exception
        with patch.object(
            audit_trail_handler, "_list_audit_trails", side_effect=ValueError("Database error")
        ):
            result = await audit_trail_handler.handle("/api/v1/audit-trails", {}, handler)

        assert result is not None
        assert result.status_code == 500


# ===========================================================================
# Test Class Methods
# ===========================================================================


class TestClassMethods:
    """Tests for class-level storage methods."""

    def test_store_trail(self, audit_trail_handler):
        trail = make_sample_trail("trail-class", "class1")
        AuditTrailHandler.store_trail("trail-class", trail)

        assert "trail-class" in AuditTrailHandler._trails
        assert AuditTrailHandler._trails["trail-class"]["trail_id"] == "trail-class"

    def test_store_receipt(self, audit_trail_handler):
        receipt = make_sample_receipt("receipt-class", "class1")
        AuditTrailHandler.store_receipt("receipt-class", receipt)

        assert "receipt-class" in AuditTrailHandler._receipts
        assert AuditTrailHandler._receipts["receipt-class"]["receipt_id"] == "receipt-class"


# ===========================================================================
# Test Request Routing
# ===========================================================================


class TestRequestRouting:
    """Tests for the main handle method routing."""

    @pytest.mark.asyncio
    async def test_route_to_list_trails(self, audit_trail_handler, mock_store):
        handler = make_mock_handler(path="/api/v1/audit-trails")

        with patch.object(audit_trail_handler, "_list_audit_trails") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b'{"trails":[]}')
            await audit_trail_handler.handle("/api/v1/audit-trails", {}, handler)
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_get_trail(self, audit_trail_handler, mock_store):
        handler = make_mock_handler(path="/api/v1/audit-trails/trail-123")

        with patch.object(audit_trail_handler, "_get_audit_trail") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            await audit_trail_handler.handle("/api/v1/audit-trails/trail-123", {}, handler)
            mock_method.assert_called_once_with("trail-123")

    @pytest.mark.asyncio
    async def test_route_to_export_trail(self, audit_trail_handler, mock_store):
        handler = make_mock_handler(path="/api/v1/audit-trails/trail-123/export")

        with patch.object(audit_trail_handler, "_export_audit_trail") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            await audit_trail_handler.handle("/api/v1/audit-trails/trail-123/export", {}, handler)
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_verify_trail(self, audit_trail_handler, mock_store):
        handler = make_mock_handler(path="/api/v1/audit-trails/trail-123/verify", method="POST")
        handler.command = "POST"

        with patch.object(audit_trail_handler, "_verify_audit_trail") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            await audit_trail_handler.handle("/api/v1/audit-trails/trail-123/verify", {}, handler)
            mock_method.assert_called_once_with("trail-123")

    @pytest.mark.asyncio
    async def test_route_to_list_receipts(self, audit_trail_handler, mock_store):
        handler = make_mock_handler(path="/api/v1/receipts")

        with patch.object(audit_trail_handler, "_list_receipts") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b'{"receipts":[]}')
            await audit_trail_handler.handle("/api/v1/receipts", {}, handler)
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_get_receipt(self, audit_trail_handler, mock_store):
        handler = make_mock_handler(path="/api/v1/receipts/receipt-123")

        with patch.object(audit_trail_handler, "_get_receipt") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            await audit_trail_handler.handle("/api/v1/receipts/receipt-123", {}, handler)
            mock_method.assert_called_once_with("receipt-123")

    @pytest.mark.asyncio
    async def test_route_to_verify_receipt(self, audit_trail_handler, mock_store):
        handler = make_mock_handler(path="/api/v1/receipts/receipt-123/verify", method="POST")
        handler.command = "POST"

        with patch.object(audit_trail_handler, "_verify_receipt") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            await audit_trail_handler.handle("/api/v1/receipts/receipt-123/verify", {}, handler)
            mock_method.assert_called_once_with("receipt-123")


# ===========================================================================
# Test Query Parameter Handling
# ===========================================================================


class TestQueryParameters:
    """Tests for query parameter parsing."""

    @pytest.mark.asyncio
    async def test_limit_bounds(self, audit_trail_handler, mock_store):
        """Test that limit is bounded correctly."""
        # Add test data
        for i in range(10):
            mock_store.trails[f"trail-test{i}"] = make_sample_trail(f"trail-test{i}", f"test{i}")

        # Test max limit
        result = audit_trail_handler._list_audit_trails({"limit": "2000"})
        if hasattr(result, "__await__"):
            result = await result

        data = json.loads(result.body)
        assert data["limit"] <= 1000  # Max limit enforced

    @pytest.mark.asyncio
    async def test_offset_bounds(self, audit_trail_handler, mock_store):
        """Test that offset is bounded correctly."""
        result = audit_trail_handler._list_audit_trails({"offset": "2000000"})
        if hasattr(result, "__await__"):
            result = await result

        data = json.loads(result.body)
        assert data["offset"] <= 1000000  # Max offset enforced

    @pytest.mark.asyncio
    async def test_default_pagination(self, audit_trail_handler, mock_store):
        """Test default pagination values."""
        result = audit_trail_handler._list_audit_trails({})
        if hasattr(result, "__await__"):
            result = await result

        data = json.loads(result.body)
        assert data["limit"] == 20  # Default
        assert data["offset"] == 0  # Default
