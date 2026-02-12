"""
Tests for FastAPI receipt route endpoints.

Covers:
- List receipts with pagination
- Get receipt by ID
- Verify receipt integrity
- Export receipt in various formats
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    return create_app()


@pytest.fixture
def mock_receipt_store():
    """Create a mock receipt store."""
    store = MagicMock()
    store.list_recent = MagicMock(return_value=[])
    store.count = MagicMock(return_value=0)
    store.get = MagicMock(return_value=None)
    store.get_by_id = MagicMock(return_value=None)
    return store


@pytest.fixture
def client(app, mock_receipt_store):
    """Create a test client with mocked context."""
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
        "receipt_store": mock_receipt_store,
    }
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_receipt_dict():
    """Sample receipt data for testing."""
    return {
        "receipt_id": "rcpt_test123",
        "gauntlet_id": "gauntlet-20260211-abc123",
        "timestamp": "2026-02-11T12:00:00",
        "input_summary": "Test input content",
        "input_type": "spec",
        "schema_version": "1.0",
        "verdict": "APPROVED",
        "confidence": 0.85,
        "risk_level": "LOW",
        "risk_score": 0.15,
        "robustness_score": 0.85,
        "coverage_score": 0.9,
        "verification_coverage": 0.7,
        "findings": [
            {
                "id": "f-001",
                "severity": "MEDIUM",
                "category": "security",
                "title": "Input validation missing",
                "description": "The input lacks validation",
                "mitigation": "Add input validation",
                "source": "claude",
                "verified": False,
            }
        ],
        "critical_count": 0,
        "high_count": 0,
        "medium_count": 1,
        "low_count": 0,
        "mitigations": ["Add input validation"],
        "dissenting_views": [],
        "unresolved_tensions": [],
        "verified_claims": [],
        "unverified_claims": [],
        "agents_involved": ["claude", "codex"],
        "rounds_completed": 3,
        "duration_seconds": 45.2,
        "audit_trail_id": None,
        "checksum": "abc123",
    }


class TestListReceipts:
    """Tests for GET /api/v2/receipts."""

    def test_list_receipts_returns_200(self, client):
        """List receipts should return 200 with empty list."""
        response = client.get("/api/v2/receipts")
        assert response.status_code == 200
        data = response.json()
        assert "receipts" in data
        assert "total" in data
        assert data["receipts"] == []
        assert data["total"] == 0

    def test_list_receipts_with_pagination(self, client):
        """List receipts supports pagination params."""
        response = client.get("/api/v2/receipts?limit=10&offset=5")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 5

    def test_list_receipts_with_verdict_filter(self, client):
        """List receipts supports verdict filter."""
        response = client.get("/api/v2/receipts?verdict=APPROVED")
        assert response.status_code == 200

    def test_list_receipts_with_data(self, client, mock_receipt_store, sample_receipt_dict):
        """List receipts returns receipt summaries."""
        mock_receipt_store.list_recent.return_value = [sample_receipt_dict]
        mock_receipt_store.count.return_value = 1

        response = client.get("/api/v2/receipts")
        assert response.status_code == 200
        data = response.json()
        assert len(data["receipts"]) == 1
        assert data["receipts"][0]["receipt_id"] == "rcpt_test123"
        assert data["receipts"][0]["verdict"] == "APPROVED"
        assert data["total"] == 1

    def test_list_receipts_validation_limit_bounds(self, client):
        """Pagination limit must be between 1 and 100."""
        response = client.get("/api/v2/receipts?limit=0")
        assert response.status_code == 422

        response = client.get("/api/v2/receipts?limit=101")
        assert response.status_code == 422


class TestGetReceipt:
    """Tests for GET /api/v2/receipts/{receipt_id}."""

    def test_get_receipt_not_found(self, client):
        """Get nonexistent receipt returns 404."""
        response = client.get("/api/v2/receipts/nonexistent-id")
        assert response.status_code == 404

    def test_get_receipt_found(self, client, mock_receipt_store, sample_receipt_dict):
        """Get existing receipt returns full details."""
        mock_receipt_store.get.return_value = sample_receipt_dict

        response = client.get("/api/v2/receipts/rcpt_test123")
        assert response.status_code == 200
        data = response.json()
        assert data["receipt_id"] == "rcpt_test123"
        assert data["gauntlet_id"] == "gauntlet-20260211-abc123"
        assert data["verdict"] == "APPROVED"
        assert data["confidence"] == 0.85
        assert len(data["findings"]) == 1
        assert data["agents_involved"] == ["claude", "codex"]

    def test_get_receipt_with_nested_data(self, client, mock_receipt_store, sample_receipt_dict):
        """Get receipt handles stored receipts with nested 'data' key."""
        mock_receipt_store.get.return_value = {"data": sample_receipt_dict}

        response = client.get("/api/v2/receipts/rcpt_test123")
        assert response.status_code == 200
        data = response.json()
        assert data["receipt_id"] == "rcpt_test123"


class TestVerifyReceipt:
    """Tests for GET /api/v2/receipts/{receipt_id}/verify."""

    def test_verify_receipt_not_found(self, client):
        """Verify nonexistent receipt returns 404."""
        response = client.get("/api/v2/receipts/nonexistent-id/verify")
        assert response.status_code == 404

    def test_verify_receipt_returns_verification_result(
        self, client, mock_receipt_store, sample_receipt_dict
    ):
        """Verify receipt returns verification details."""
        mock_receipt_store.get.return_value = sample_receipt_dict

        response = client.get("/api/v2/receipts/rcpt_test123/verify")
        assert response.status_code == 200
        data = response.json()
        assert data["receipt_id"] == "rcpt_test123"
        assert "verified" in data
        assert "integrity_valid" in data
        assert "checksum_match" in data
        assert "details" in data

    def test_verify_receipt_with_valid_integrity(
        self, client, mock_receipt_store
    ):
        """Verify receipt with matching checksum succeeds."""
        # Create a receipt using the actual DecisionReceipt class for correct checksum
        try:
            from aragora.export.decision_receipt import DecisionReceipt

            receipt = DecisionReceipt(
                receipt_id="rcpt_integrity_test",
                gauntlet_id="gauntlet-test",
                verdict="APPROVED",
                confidence=0.9,
            )
            receipt_dict = receipt.to_dict()
            mock_receipt_store.get.return_value = receipt_dict

            response = client.get("/api/v2/receipts/rcpt_integrity_test/verify")
            assert response.status_code == 200
            data = response.json()
            assert data["integrity_valid"] is True
        except ImportError:
            pytest.skip("DecisionReceipt not available")


class TestExportReceipt:
    """Tests for GET /api/v2/receipts/{receipt_id}/export."""

    def test_export_receipt_not_found(self, client):
        """Export nonexistent receipt returns 404."""
        response = client.get("/api/v2/receipts/nonexistent-id/export")
        assert response.status_code == 404

    def test_export_receipt_json_format(self, client, mock_receipt_store, sample_receipt_dict):
        """Export receipt in JSON format."""
        mock_receipt_store.get.return_value = sample_receipt_dict

        response = client.get("/api/v2/receipts/rcpt_test123/export?format=json")
        assert response.status_code == 200
        data = response.json()
        assert data["receipt_id"] == "rcpt_test123"
        assert data["format"] == "json"
        assert "content" in data
        # Content should be valid JSON
        import json
        parsed = json.loads(data["content"])
        assert parsed["receipt_id"] == "rcpt_test123"

    def test_export_receipt_markdown_format(self, client, mock_receipt_store, sample_receipt_dict):
        """Export receipt in Markdown format."""
        mock_receipt_store.get.return_value = sample_receipt_dict

        response = client.get("/api/v2/receipts/rcpt_test123/export?format=markdown")
        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "markdown"
        assert "Decision Receipt" in data["content"]

    def test_export_receipt_sarif_format(self, client, mock_receipt_store, sample_receipt_dict):
        """Export receipt in SARIF format."""
        mock_receipt_store.get.return_value = sample_receipt_dict

        response = client.get("/api/v2/receipts/rcpt_test123/export?format=sarif")
        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "sarif"
        import json
        sarif = json.loads(data["content"])
        assert sarif["version"] == "2.1.0"

    def test_export_receipt_default_format_is_json(self, client, mock_receipt_store, sample_receipt_dict):
        """Export without format param defaults to JSON."""
        mock_receipt_store.get.return_value = sample_receipt_dict

        response = client.get("/api/v2/receipts/rcpt_test123/export")
        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "json"

    def test_export_receipt_invalid_format(self, client, mock_receipt_store, sample_receipt_dict):
        """Export with invalid format returns 422."""
        mock_receipt_store.get.return_value = sample_receipt_dict

        response = client.get("/api/v2/receipts/rcpt_test123/export?format=pdf_invalid")
        assert response.status_code == 422
