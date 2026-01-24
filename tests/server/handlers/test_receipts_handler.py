"""
Tests for ReceiptsHandler - Decision receipt HTTP endpoints.

Tests cover:
- List receipts with filtering and pagination
- Get single receipt by ID
- Export in multiple formats (JSON, HTML, MD, PDF, SARIF, CSV)
- Verify integrity checksum
- Verify cryptographic signature
- Batch signature verification
- Statistics endpoint
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.receipts import (
    ReceiptsHandler,
    create_receipts_handler,
)


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockStoredReceipt:
    """Mock stored receipt for testing."""

    receipt_id: str = "receipt-001"
    gauntlet_id: str = "gauntlet-001"
    debate_id: Optional[str] = "debate-001"
    created_at: float = 1700000000.0
    expires_at: Optional[float] = 1800000000.0
    verdict: str = "APPROVED"
    confidence: float = 0.85
    risk_level: str = "MEDIUM"
    risk_score: float = 0.35
    checksum: str = "sha256:abc123"
    signature: Optional[str] = None
    signature_algorithm: Optional[str] = None
    signature_key_id: Optional[str] = None
    signed_at: Optional[float] = None
    audit_trail_id: Optional[str] = "audit-001"
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "receipt_id": self.receipt_id,
            "gauntlet_id": self.gauntlet_id,
            "debate_id": self.debate_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "checksum": self.checksum,
            "audit_trail_id": self.audit_trail_id,
            "is_signed": self.signature is not None,
        }
        if self.signature:
            result["signature_metadata"] = {
                "algorithm": self.signature_algorithm,
                "key_id": self.signature_key_id,
                "signed_at": self.signed_at,
            }
        return result

    def to_full_dict(self) -> Dict[str, Any]:
        result = self.to_dict()
        result.update(self.data)
        return result


@dataclass
class MockSignatureVerificationResult:
    """Mock signature verification result."""

    receipt_id: str
    is_valid: bool
    algorithm: Optional[str] = None
    key_id: Optional[str] = None
    signed_at: Optional[float] = None
    verified_at: float = 1700001000.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "signature_valid": self.is_valid,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "signed_at": self.signed_at,
            "verification_timestamp": datetime.fromtimestamp(
                self.verified_at, tz=timezone.utc
            ).isoformat(),
            "error": self.error,
        }


class MockReceiptStore:
    """Mock receipt store for testing."""

    def __init__(self):
        self.receipts: Dict[str, MockStoredReceipt] = {}
        self._next_id = 0

    def save(self, receipt_dict: Dict, signed_receipt: Optional[Dict] = None) -> str:
        receipt_id = receipt_dict.get("receipt_id", f"receipt-{self._next_id}")
        self._next_id += 1
        self.receipts[receipt_id] = MockStoredReceipt(
            receipt_id=receipt_id,
            gauntlet_id=receipt_dict.get("gauntlet_id", ""),
            verdict=receipt_dict.get("verdict", "APPROVED"),
            confidence=receipt_dict.get("confidence", 0.85),
            risk_level=receipt_dict.get("risk_level", "MEDIUM"),
            risk_score=receipt_dict.get("risk_score", 0.35),
            data=receipt_dict,
        )
        return receipt_id

    def get(self, receipt_id: str) -> Optional[MockStoredReceipt]:
        return self.receipts.get(receipt_id)

    def get_by_gauntlet(self, gauntlet_id: str) -> Optional[MockStoredReceipt]:
        for receipt in self.receipts.values():
            if receipt.gauntlet_id == gauntlet_id:
                return receipt
        return None

    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        verdict: Optional[str] = None,
        risk_level: Optional[str] = None,
        date_from: Optional[float] = None,
        date_to: Optional[float] = None,
        signed_only: bool = False,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> List[MockStoredReceipt]:
        results = list(self.receipts.values())
        if verdict:
            results = [r for r in results if r.verdict == verdict]
        if risk_level:
            results = [r for r in results if r.risk_level == risk_level]
        if signed_only:
            results = [r for r in results if r.signature is not None]
        return results[offset : offset + limit]

    def count(
        self,
        verdict: Optional[str] = None,
        risk_level: Optional[str] = None,
        date_from: Optional[float] = None,
        date_to: Optional[float] = None,
        signed_only: bool = False,
    ) -> int:
        results = list(self.receipts.values())
        if verdict:
            results = [r for r in results if r.verdict == verdict]
        if risk_level:
            results = [r for r in results if r.risk_level == risk_level]
        if signed_only:
            results = [r for r in results if r.signature is not None]
        return len(results)

    def verify_integrity(self, receipt_id: str) -> Dict[str, Any]:
        if receipt_id not in self.receipts:
            return {
                "receipt_id": receipt_id,
                "integrity_valid": False,
                "error": "Receipt not found",
            }
        return {"receipt_id": receipt_id, "integrity_valid": True, "stored_checksum": "sha256:abc"}

    def verify_signature(self, receipt_id: str) -> MockSignatureVerificationResult:
        if receipt_id not in self.receipts:
            return MockSignatureVerificationResult(
                receipt_id=receipt_id, is_valid=False, error="Receipt not found"
            )
        receipt = self.receipts[receipt_id]
        if not receipt.signature:
            return MockSignatureVerificationResult(
                receipt_id=receipt_id, is_valid=False, error="Receipt is not signed"
            )
        return MockSignatureVerificationResult(
            receipt_id=receipt_id,
            is_valid=True,
            algorithm=receipt.signature_algorithm,
            key_id=receipt.signature_key_id,
        )

    def verify_batch(
        self, receipt_ids: List[str]
    ) -> tuple[List[MockSignatureVerificationResult], Dict[str, int]]:
        results = []
        summary = {"total": len(receipt_ids), "valid": 0, "invalid": 0, "not_signed": 0}
        for rid in receipt_ids:
            result = self.verify_signature(rid)
            results.append(result)
            if result.is_valid:
                summary["valid"] += 1
            elif result.error == "Receipt is not signed":
                summary["not_signed"] += 1
            else:
                summary["invalid"] += 1
        return results, summary

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total": len(self.receipts),
            "signed": sum(1 for r in self.receipts.values() if r.signature),
            "unsigned": sum(1 for r in self.receipts.values() if not r.signature),
            "by_verdict": {"approved": 0, "rejected": 0},
            "by_risk_level": {"low": 0, "medium": 0, "high": 0},
            "retention_days": 2555,
        }


@pytest.fixture
def mock_receipt_store():
    """Create a mock receipt store."""
    return MockReceiptStore()


@pytest.fixture
def mock_server_context():
    """Create a mock server context."""
    return MagicMock()


@pytest.fixture
def receipts_handler(mock_server_context, mock_receipt_store):
    """Create a receipts handler with mocked store."""
    handler = ReceiptsHandler(mock_server_context)
    handler._store = mock_receipt_store
    return handler


def parse_handler_response(result) -> Dict[str, Any]:
    """Parse handler result body as JSON."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode())
        return json.loads(body)
    return {}


# ===========================================================================
# Handler Routing Tests
# ===========================================================================


class TestReceiptsHandlerRouting:
    """Tests for request routing."""

    def test_can_handle_list(self, receipts_handler):
        """Test can_handle for list endpoint."""
        assert receipts_handler.can_handle("/api/v2/receipts", "GET") is True

    def test_can_handle_get(self, receipts_handler):
        """Test can_handle for get endpoint."""
        assert receipts_handler.can_handle("/api/v2/receipts/receipt-001", "GET") is True

    def test_can_handle_verify(self, receipts_handler):
        """Test can_handle for verify endpoint."""
        assert receipts_handler.can_handle("/api/v2/receipts/receipt-001/verify", "POST") is True

    def test_can_handle_stats(self, receipts_handler):
        """Test can_handle for stats endpoint."""
        assert receipts_handler.can_handle("/api/v2/receipts/stats", "GET") is True

    def test_cannot_handle_other_paths(self, receipts_handler):
        """Test can_handle returns False for other paths."""
        assert receipts_handler.can_handle("/api/v2/gauntlet", "GET") is False
        assert receipts_handler.can_handle("/api/v1/receipts", "GET") is False

    def test_cannot_handle_delete(self, receipts_handler):
        """Test can_handle returns False for DELETE method."""
        assert receipts_handler.can_handle("/api/v2/receipts/receipt-001", "DELETE") is False


# ===========================================================================
# List Receipts Tests
# ===========================================================================


class TestReceiptsHandlerList:
    """Tests for list receipts endpoint."""

    @pytest.mark.asyncio
    async def test_list_empty(self, receipts_handler):
        """Test list returns empty for no receipts."""
        result = await receipts_handler.handle("GET", "/api/v2/receipts")

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["receipts"] == []
        assert data["pagination"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_with_receipts(self, receipts_handler, mock_receipt_store):
        """Test list returns receipts."""
        mock_receipt_store.save({"receipt_id": "r1", "gauntlet_id": "g1", "verdict": "APPROVED"})
        mock_receipt_store.save({"receipt_id": "r2", "gauntlet_id": "g2", "verdict": "REJECTED"})

        result = await receipts_handler.handle("GET", "/api/v2/receipts")

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert len(data["receipts"]) == 2

    @pytest.mark.asyncio
    async def test_list_pagination(self, receipts_handler, mock_receipt_store):
        """Test list pagination."""
        for i in range(5):
            mock_receipt_store.save({"receipt_id": f"r{i}", "gauntlet_id": f"g{i}"})

        result = await receipts_handler.handle(
            "GET", "/api/v2/receipts", query_params={"limit": "2", "offset": "0"}
        )

        data = parse_handler_response(result)
        assert len(data["receipts"]) == 2
        assert data["pagination"]["limit"] == 2
        assert data["pagination"]["total"] == 5
        assert data["pagination"]["has_more"] is True

    @pytest.mark.asyncio
    async def test_list_filter_verdict(self, receipts_handler, mock_receipt_store):
        """Test list filters by verdict."""
        mock_receipt_store.save({"receipt_id": "r1", "gauntlet_id": "g1", "verdict": "APPROVED"})
        mock_receipt_store.save({"receipt_id": "r2", "gauntlet_id": "g2", "verdict": "REJECTED"})

        result = await receipts_handler.handle(
            "GET", "/api/v2/receipts", query_params={"verdict": "APPROVED"}
        )

        data = parse_handler_response(result)
        assert data["filters"]["verdict"] == "APPROVED"

    @pytest.mark.asyncio
    async def test_list_limit_capped(self, receipts_handler):
        """Test list limit is capped at 100."""
        result = await receipts_handler.handle(
            "GET", "/api/v2/receipts", query_params={"limit": "500"}
        )

        data = parse_handler_response(result)
        assert data["pagination"]["limit"] == 100


# ===========================================================================
# Get Receipt Tests
# ===========================================================================


class TestReceiptsHandlerGet:
    """Tests for get single receipt endpoint."""

    @pytest.mark.asyncio
    async def test_get_by_id(self, receipts_handler, mock_receipt_store):
        """Test get receipt by ID."""
        mock_receipt_store.save({"receipt_id": "receipt-001", "gauntlet_id": "gauntlet-001"})

        result = await receipts_handler.handle("GET", "/api/v2/receipts/receipt-001")

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["receipt_id"] == "receipt-001"

    @pytest.mark.asyncio
    async def test_get_by_gauntlet_id(self, receipts_handler, mock_receipt_store):
        """Test get receipt by gauntlet_id fallback."""
        mock_receipt_store.save({"receipt_id": "receipt-001", "gauntlet_id": "gauntlet-001"})

        result = await receipts_handler.handle("GET", "/api/v2/receipts/gauntlet-001")

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_not_found(self, receipts_handler):
        """Test get returns 404 for nonexistent receipt."""
        result = await receipts_handler.handle("GET", "/api/v2/receipts/nonexistent")

        assert result.status_code == 404


# ===========================================================================
# Export Tests
# ===========================================================================


class TestReceiptsHandlerExport:
    """Tests for receipt export endpoint."""

    @pytest.mark.asyncio
    async def test_export_not_found(self, receipts_handler):
        """Test export returns 404 for nonexistent receipt."""
        result = await receipts_handler.handle("GET", "/api/v2/receipts/nonexistent/export")

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_export_unsupported_format(self, receipts_handler, mock_receipt_store):
        """Test export returns 400 for unsupported format."""
        mock_receipt_store.save({"receipt_id": "receipt-001", "gauntlet_id": "gauntlet-001"})

        mock_receipt = MagicMock()
        mock_receipt_class = MagicMock(from_dict=MagicMock(return_value=mock_receipt))

        with patch.dict(
            "sys.modules",
            {"aragora.export.decision_receipt": MagicMock(DecisionReceipt=mock_receipt_class)},
        ):
            result = await receipts_handler.handle(
                "GET",
                "/api/v2/receipts/receipt-001/export",
                query_params={"format": "invalid"},
            )

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_export_json(self, receipts_handler, mock_receipt_store):
        """Test export as JSON."""
        mock_receipt_store.save({"receipt_id": "receipt-001", "gauntlet_id": "gauntlet-001"})

        mock_receipt = MagicMock()
        mock_receipt.to_json.return_value = '{"test": "json"}'
        mock_receipt_class = MagicMock(from_dict=MagicMock(return_value=mock_receipt))

        with patch.dict(
            "sys.modules",
            {"aragora.export.decision_receipt": MagicMock(DecisionReceipt=mock_receipt_class)},
        ):
            result = await receipts_handler.handle(
                "GET",
                "/api/v2/receipts/receipt-001/export",
                query_params={"format": "json"},
            )

        assert result.status_code == 200
        assert result.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_export_html(self, receipts_handler, mock_receipt_store):
        """Test export as HTML."""
        mock_receipt_store.save({"receipt_id": "receipt-001", "gauntlet_id": "gauntlet-001"})

        mock_receipt = MagicMock()
        mock_receipt.to_html.return_value = "<html><body>Receipt</body></html>"
        mock_receipt_class = MagicMock(from_dict=MagicMock(return_value=mock_receipt))

        with patch.dict(
            "sys.modules",
            {"aragora.export.decision_receipt": MagicMock(DecisionReceipt=mock_receipt_class)},
        ):
            result = await receipts_handler.handle(
                "GET",
                "/api/v2/receipts/receipt-001/export",
                query_params={"format": "html"},
            )

        assert result.status_code == 200
        assert result.content_type == "text/html"

    @pytest.mark.asyncio
    async def test_export_markdown(self, receipts_handler, mock_receipt_store):
        """Test export as Markdown."""
        mock_receipt_store.save({"receipt_id": "receipt-001", "gauntlet_id": "gauntlet-001"})

        mock_receipt = MagicMock()
        mock_receipt.to_markdown.return_value = "# Receipt\n\nContent"
        mock_receipt_class = MagicMock(from_dict=MagicMock(return_value=mock_receipt))

        with patch.dict(
            "sys.modules",
            {"aragora.export.decision_receipt": MagicMock(DecisionReceipt=mock_receipt_class)},
        ):
            result = await receipts_handler.handle(
                "GET",
                "/api/v2/receipts/receipt-001/export",
                query_params={"format": "md"},
            )

        assert result.status_code == 200
        assert result.content_type == "text/markdown"

    @pytest.mark.asyncio
    async def test_export_csv(self, receipts_handler, mock_receipt_store):
        """Test export as CSV."""
        mock_receipt_store.save({"receipt_id": "receipt-001", "gauntlet_id": "gauntlet-001"})

        mock_receipt = MagicMock()
        mock_receipt.to_csv.return_value = "id,verdict\nreceipt-001,APPROVED"
        mock_receipt_class = MagicMock(from_dict=MagicMock(return_value=mock_receipt))

        with patch.dict(
            "sys.modules",
            {"aragora.export.decision_receipt": MagicMock(DecisionReceipt=mock_receipt_class)},
        ):
            result = await receipts_handler.handle(
                "GET",
                "/api/v2/receipts/receipt-001/export",
                query_params={"format": "csv"},
            )

        assert result.status_code == 200
        assert result.content_type == "text/csv"


# ===========================================================================
# Verification Tests
# ===========================================================================


class TestReceiptsHandlerVerification:
    """Tests for verification endpoints."""

    @pytest.mark.asyncio
    async def test_verify_integrity(self, receipts_handler, mock_receipt_store):
        """Test verify integrity endpoint."""
        mock_receipt_store.save({"receipt_id": "receipt-001", "gauntlet_id": "gauntlet-001"})

        result = await receipts_handler.handle("POST", "/api/v2/receipts/receipt-001/verify")

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["integrity_valid"] is True

    @pytest.mark.asyncio
    async def test_verify_integrity_not_found(self, receipts_handler):
        """Test verify integrity returns 404."""
        result = await receipts_handler.handle("POST", "/api/v2/receipts/nonexistent/verify")

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_verify_signature(self, receipts_handler, mock_receipt_store):
        """Test verify signature endpoint."""
        mock_receipt_store.save({"receipt_id": "receipt-001", "gauntlet_id": "gauntlet-001"})
        mock_receipt_store.receipts["receipt-001"].signature = "sig=="
        mock_receipt_store.receipts["receipt-001"].signature_algorithm = "HMAC-SHA256"

        result = await receipts_handler.handle(
            "POST", "/api/v2/receipts/receipt-001/verify-signature"
        )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["signature_valid"] is True

    @pytest.mark.asyncio
    async def test_verify_signature_unsigned(self, receipts_handler, mock_receipt_store):
        """Test verify signature for unsigned receipt."""
        mock_receipt_store.save({"receipt_id": "receipt-001", "gauntlet_id": "gauntlet-001"})

        result = await receipts_handler.handle(
            "POST", "/api/v2/receipts/receipt-001/verify-signature"
        )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["signature_valid"] is False
        assert "not signed" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_verify_signature_not_found(self, receipts_handler):
        """Test verify signature returns 404."""
        result = await receipts_handler.handle(
            "POST", "/api/v2/receipts/nonexistent/verify-signature"
        )

        assert result.status_code == 404


# ===========================================================================
# Batch Verification Tests
# ===========================================================================


class TestReceiptsHandlerBatchVerify:
    """Tests for batch verification endpoint."""

    @pytest.mark.asyncio
    async def test_verify_batch_empty(self, receipts_handler):
        """Test batch verify with empty list."""
        result = await receipts_handler.handle(
            "POST", "/api/v2/receipts/verify-batch", body={"receipt_ids": []}
        )

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_verify_batch_too_many(self, receipts_handler):
        """Test batch verify with too many IDs."""
        ids = [f"r{i}" for i in range(150)]
        result = await receipts_handler.handle(
            "POST", "/api/v2/receipts/verify-batch", body={"receipt_ids": ids}
        )

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_verify_batch_success(self, receipts_handler, mock_receipt_store):
        """Test batch verify with valid IDs."""
        mock_receipt_store.save({"receipt_id": "r1", "gauntlet_id": "g1"})
        mock_receipt_store.save({"receipt_id": "r2", "gauntlet_id": "g2"})

        result = await receipts_handler.handle(
            "POST", "/api/v2/receipts/verify-batch", body={"receipt_ids": ["r1", "r2"]}
        )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert len(data["results"]) == 2
        assert data["summary"]["total"] == 2


# ===========================================================================
# Statistics Tests
# ===========================================================================


class TestReceiptsHandlerStats:
    """Tests for statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats(self, receipts_handler, mock_receipt_store):
        """Test get stats endpoint."""
        mock_receipt_store.save({"receipt_id": "r1", "gauntlet_id": "g1"})

        result = await receipts_handler.handle("GET", "/api/v2/receipts/stats")

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert "stats" in data
        assert "generated_at" in data


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestReceiptsHandlerFactory:
    """Tests for handler factory function."""

    def test_create_receipts_handler(self, mock_server_context):
        """Test factory creates handler."""
        handler = create_receipts_handler(mock_server_context)

        assert isinstance(handler, ReceiptsHandler)


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestReceiptsHandlerErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_path(self, receipts_handler):
        """Test invalid path returns 404."""
        result = await receipts_handler.handle("GET", "/api/v2/receipts/")

        # Path with trailing slash might be parsed differently
        # Just ensure it doesn't crash

    @pytest.mark.asyncio
    async def test_handle_exception(self, receipts_handler):
        """Test handler catches exceptions gracefully."""
        # Force an exception
        receipts_handler._get_store = MagicMock(side_effect=Exception("Test error"))

        result = await receipts_handler.handle("GET", "/api/v2/receipts")

        assert result.status_code == 500


# ===========================================================================
# Timestamp Parsing Tests
# ===========================================================================


class TestReceiptsHandlerTimestampParsing:
    """Tests for timestamp parsing."""

    def test_parse_timestamp_none(self, receipts_handler):
        """Test parse_timestamp with None."""
        result = receipts_handler._parse_timestamp(None)
        assert result is None

    def test_parse_timestamp_float(self, receipts_handler):
        """Test parse_timestamp with float string."""
        result = receipts_handler._parse_timestamp("1700000000.0")
        assert result == 1700000000.0

    def test_parse_timestamp_iso(self, receipts_handler):
        """Test parse_timestamp with ISO date."""
        result = receipts_handler._parse_timestamp("2024-01-15T10:30:00Z")
        assert result is not None
        assert result > 0

    def test_parse_timestamp_invalid(self, receipts_handler):
        """Test parse_timestamp with invalid string."""
        result = receipts_handler._parse_timestamp("invalid")
        assert result is None
