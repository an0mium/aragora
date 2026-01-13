"""
Tests for GauntletHandler endpoints.

Endpoints tested:
- GET /api/gauntlet/personas - List available personas
- POST /api/gauntlet/run - Start a gauntlet stress-test
- GET /api/gauntlet/results - List recent results with pagination
- GET /api/gauntlet/{id} - Get gauntlet status/results
- GET /api/gauntlet/{id}/receipt - Get decision receipt
- GET /api/gauntlet/{id}/heatmap - Get risk heatmap
- GET /api/gauntlet/{id}/compare/{id2} - Compare two gauntlet runs
- DELETE /api/gauntlet/{id} - Delete a gauntlet result
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass, field

from aragora.server.handlers import GauntletHandler, HandlerResult
from aragora.server.handlers.base import clear_cache
import aragora.server.handlers.gauntlet as gauntlet_module

# Import rate limiting module for clearing between tests
import importlib

_rate_limit_mod = importlib.import_module("aragora.server.handlers.utils.rate_limit")


def run_async(coro):
    """Helper to run async handler methods synchronously in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def mock_handler_with_query(path: str, query_params: dict = None) -> Mock:
    """Create a mock handler with path containing query params."""
    mock_handler = Mock()
    if query_params:
        query_str = "&".join(f"{k}={v}" for k, v in query_params.items())
        mock_handler.path = f"{path}?{query_str}"
    else:
        mock_handler.path = path
    # Ensure headers returns a proper dict for extract_client_ip
    mock_handler.headers = {}
    mock_handler.client_address = ("127.0.0.1", 8080)
    return mock_handler


def create_mock_handler(path: str = "/api/gauntlet/run") -> Mock:
    """Create a properly configured mock handler for gauntlet tests."""
    mock_handler = Mock()
    mock_handler.path = path
    mock_handler.headers = {}
    mock_handler.client_address = ("127.0.0.1", 8080)
    return mock_handler


# ============================================================================
# Mock Classes for Gauntlet Types
# ============================================================================


@dataclass
class MockGauntletRun:
    """Mock in-memory gauntlet run."""

    gauntlet_id: str = "gauntlet-20260111120000-abc123"
    status: str = "completed"
    input_type: str = "spec"
    input_summary: str = "Test input content..."
    input_hash: str = "abc123def456"
    persona: str = "gdpr"
    profile: str = "default"
    created_at: str = "2026-01-11T12:00:00"
    result: dict = field(
        default_factory=lambda: {
            "gauntlet_id": "gauntlet-20260111120000-abc123",
            "verdict": "PASS",
            "confidence": 0.85,
            "risk_score": 0.15,
            "robustness_score": 0.9,
            "coverage_score": 0.75,
            "total_findings": 3,
            "critical_count": 0,
            "high_count": 1,
            "medium_count": 2,
            "low_count": 0,
            "findings": [],
        }
    )


class MockDecisionReceipt:
    """Mock DecisionReceipt for tests."""

    def __init__(self, **kwargs):
        self.receipt_id = kwargs.get("receipt_id", "receipt-abc123")
        self.gauntlet_id = kwargs.get("gauntlet_id", "gauntlet-20260111120000-abc123")
        self.verdict = kwargs.get("verdict", "PASS")
        self.confidence = kwargs.get("confidence", 0.85)
        # Accept any other kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {
            "receipt_id": self.receipt_id,
            "gauntlet_id": self.gauntlet_id,
            "verdict": self.verdict,
            "confidence": self.confidence,
        }

    def to_markdown(self):
        return f"# Decision Receipt\n\n**Verdict:** {self.verdict}\n**Confidence:** {self.confidence:.1%}"

    def to_html(self):
        return f"<h1>Decision Receipt</h1><p>Verdict: {self.verdict}</p>"

    @classmethod
    def from_mode_result(cls, result, input_hash=None):
        return cls(
            gauntlet_id=(
                result.get("gauntlet_id", "unknown") if isinstance(result, dict) else "unknown"
            )
        )


@dataclass
class MockRiskHeatmap:
    """Mock RiskHeatmap for tests."""

    cells: list = field(default_factory=list)
    categories: list = field(default_factory=list)
    severities: list = field(default_factory=lambda: ["critical", "high", "medium", "low"])
    total_findings: int = 0

    def to_dict(self):
        return {
            "cells": self.cells,
            "categories": self.categories,
            "severities": self.severities,
            "total_findings": self.total_findings,
        }

    def to_svg(self):
        return "<svg></svg>"

    def to_ascii(self):
        return "Heatmap"


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_gauntlet_run():
    """Create a mock gauntlet run."""
    return {
        "gauntlet_id": "gauntlet-20260111120000-abc123",
        "status": "completed",
        "input_type": "spec",
        "input_summary": "Test input content...",
        "input_hash": "abc123def456",
        "persona": "gdpr",
        "profile": "default",
        "created_at": "2026-01-11T12:00:00",
        "completed_at": "2026-01-11T12:05:00",
        "result": {
            "gauntlet_id": "gauntlet-20260111120000-abc123",
            "verdict": "PASS",
            "confidence": 0.85,
            "risk_score": 0.15,
            "robustness_score": 0.9,
            "coverage_score": 0.75,
            "total_findings": 3,
            "critical_count": 0,
            "high_count": 1,
            "medium_count": 2,
            "low_count": 0,
            "findings": [],
        },
    }


@pytest.fixture
def gauntlet_handler(mock_gauntlet_run):
    """Create a GauntletHandler with mocked dependencies."""
    ctx = {"nomic_dir": "/tmp/nomic"}
    handler = GauntletHandler(ctx)

    # Add a mock result to the module-level in-memory storage
    gauntlet_module._gauntlet_runs["gauntlet-20260111120000-abc123"] = mock_gauntlet_run

    return handler


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches and rate limits before and after each test."""
    clear_cache()
    gauntlet_module._gauntlet_runs.clear()
    # Clear all rate limiters
    with _rate_limit_mod._limiters_lock:
        for limiter in _rate_limit_mod._limiters.values():
            limiter.clear()
    yield
    clear_cache()
    gauntlet_module._gauntlet_runs.clear()
    # Clear all rate limiters
    with _rate_limit_mod._limiters_lock:
        for limiter in _rate_limit_mod._limiters.values():
            limiter.clear()


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestGauntletHandlerRouting:
    """Tests for route matching."""

    def test_can_handle_run(self, gauntlet_handler):
        """Should handle POST /api/gauntlet/run."""
        assert gauntlet_handler.can_handle("/api/gauntlet/run", "POST") is True

    def test_can_handle_personas(self, gauntlet_handler):
        """Should handle GET /api/gauntlet/personas."""
        assert gauntlet_handler.can_handle("/api/gauntlet/personas", "GET") is True

    def test_can_handle_results_list(self, gauntlet_handler):
        """Should handle GET /api/gauntlet/results."""
        assert gauntlet_handler.can_handle("/api/gauntlet/results", "GET") is True

    def test_can_handle_status(self, gauntlet_handler):
        """Should handle GET /api/gauntlet/{id}."""
        assert gauntlet_handler.can_handle("/api/gauntlet/gauntlet-12345-abc", "GET") is True

    def test_can_handle_receipt(self, gauntlet_handler):
        """Should handle GET /api/gauntlet/{id}/receipt."""
        assert (
            gauntlet_handler.can_handle("/api/gauntlet/gauntlet-12345-abc/receipt", "GET") is True
        )

    def test_can_handle_heatmap(self, gauntlet_handler):
        """Should handle GET /api/gauntlet/{id}/heatmap."""
        assert (
            gauntlet_handler.can_handle("/api/gauntlet/gauntlet-12345-abc/heatmap", "GET") is True
        )

    def test_can_handle_delete(self, gauntlet_handler):
        """Should handle DELETE /api/gauntlet/{id}."""
        assert gauntlet_handler.can_handle("/api/gauntlet/gauntlet-12345-abc", "DELETE") is True

    def test_cannot_handle_unknown_route(self, gauntlet_handler):
        """Should not handle unknown routes."""
        assert gauntlet_handler.can_handle("/api/debates", "GET") is False
        assert gauntlet_handler.can_handle("/api/unknown", "GET") is False


# ============================================================================
# Personas Endpoint Tests
# ============================================================================


class TestGauntletPersonas:
    """Tests for GET /api/gauntlet/personas endpoint."""

    def test_list_personas_returns_list(self, gauntlet_handler):
        """Should return list of personas."""
        result = run_async(gauntlet_handler.handle("/api/gauntlet/personas", "GET", None))

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "personas" in data
        assert "count" in data
        assert isinstance(data["personas"], list)

    def test_list_personas_structure(self, gauntlet_handler):
        """Should return properly structured personas."""
        with patch("aragora.gauntlet.personas.list_personas", return_value=["gdpr", "hipaa"]):
            mock_persona = Mock()
            mock_persona.name = "GDPR"
            mock_persona.description = "GDPR compliance"
            mock_persona.regulation = "GDPR"
            mock_persona.attack_prompts = []

            with patch("aragora.gauntlet.personas.get_persona", return_value=mock_persona):
                result = run_async(gauntlet_handler.handle("/api/gauntlet/personas", "GET", None))

                assert result.status_code == 200
                data = json.loads(result.body)
                assert data["count"] == 2


# ============================================================================
# Results List Tests
# ============================================================================


class TestGauntletResultsList:
    """Tests for GET /api/gauntlet/results endpoint."""

    def test_list_results_empty(self, gauntlet_handler):
        """Should return empty list when no results."""
        with patch.object(gauntlet_module, "_get_storage") as mock_storage:
            mock_store = Mock()
            mock_store.list_recent.return_value = []
            mock_store.count.return_value = 0
            mock_storage.return_value = mock_store

            result = run_async(gauntlet_handler.handle("/api/gauntlet/results", "GET", None))

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["results"] == []
            assert data["total"] == 0

    def test_list_results_with_pagination(self, gauntlet_handler):
        """Should support pagination params."""
        mock_handler = mock_handler_with_query(
            "/api/gauntlet/results", {"limit": "10", "offset": "5"}
        )

        with patch.object(gauntlet_module, "_get_storage") as mock_storage:
            mock_store = Mock()
            mock_store.list_recent.return_value = []
            mock_store.count.return_value = 0
            mock_storage.return_value = mock_store

            result = run_async(
                gauntlet_handler.handle("/api/gauntlet/results", "GET", mock_handler)
            )

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["limit"] == 10
            assert data["offset"] == 5


# ============================================================================
# Status Endpoint Tests
# ============================================================================


class TestGauntletStatus:
    """Tests for GET /api/gauntlet/{id} endpoint."""

    def test_get_status_from_memory(self, gauntlet_handler, mock_gauntlet_run):
        """Should return status from in-memory storage."""
        gauntlet_module._gauntlet_runs["gauntlet-20260111120000-abc123"] = mock_gauntlet_run

        result = run_async(
            gauntlet_handler.handle("/api/gauntlet/gauntlet-20260111120000-abc123", "GET", None)
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["gauntlet_id"] == "gauntlet-20260111120000-abc123"
        assert data["status"] == "completed"

    def test_get_status_not_found(self, gauntlet_handler):
        """Should return 404 for unknown gauntlet."""
        with patch.object(gauntlet_module, "_get_storage") as mock_storage:
            mock_store = Mock()
            mock_store.get.return_value = None
            mock_storage.return_value = mock_store

            result = run_async(
                gauntlet_handler.handle("/api/gauntlet/gauntlet-nonexistent-000000", "GET", None)
            )

            assert result.status_code == 404

    def test_get_status_invalid_id(self, gauntlet_handler):
        """Should return 400 for invalid gauntlet ID."""
        result = run_async(gauntlet_handler.handle("/api/gauntlet/invalid-id-format", "GET", None))

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Invalid gauntlet ID" in data["error"]


# ============================================================================
# Receipt Endpoint Tests
# ============================================================================


class TestGauntletReceipt:
    """Tests for GET /api/gauntlet/{id}/receipt endpoint."""

    def test_get_receipt_json(self, gauntlet_handler, mock_gauntlet_run):
        """Should return receipt as JSON (default)."""
        gauntlet_module._gauntlet_runs["gauntlet-20260111120000-abc123"] = mock_gauntlet_run

        with patch("aragora.gauntlet.receipt.DecisionReceipt", MockDecisionReceipt):
            result = run_async(
                gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-20260111120000-abc123/receipt", "GET", None
                )
            )

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "receipt_id" in data

    def test_get_receipt_markdown(self, gauntlet_handler, mock_gauntlet_run):
        """Should return receipt as markdown."""
        gauntlet_module._gauntlet_runs["gauntlet-20260111120000-abc123"] = mock_gauntlet_run
        mock_handler = mock_handler_with_query(
            "/api/gauntlet/gauntlet-20260111120000-abc123/receipt", {"format": "md"}
        )

        with patch("aragora.gauntlet.receipt.DecisionReceipt", MockDecisionReceipt):
            result = run_async(
                gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-20260111120000-abc123/receipt", "GET", mock_handler
                )
            )

            # Result is a tuple (body, status, headers) for non-JSON
            if isinstance(result, tuple):
                assert result[1] == 200
                assert "markdown" in result[2].get("Content-Type", "")
            else:
                assert result.status_code == 200

    def test_get_receipt_html(self, gauntlet_handler, mock_gauntlet_run):
        """Should return receipt as HTML."""
        gauntlet_module._gauntlet_runs["gauntlet-20260111120000-abc123"] = mock_gauntlet_run
        mock_handler = mock_handler_with_query(
            "/api/gauntlet/gauntlet-20260111120000-abc123/receipt", {"format": "html"}
        )

        with patch("aragora.gauntlet.receipt.DecisionReceipt", MockDecisionReceipt):
            result = run_async(
                gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-20260111120000-abc123/receipt", "GET", mock_handler
                )
            )

            # Result is a tuple (body, status, headers) for non-JSON
            if isinstance(result, tuple):
                assert result[1] == 200
                assert "html" in result[2].get("Content-Type", "")
            else:
                assert result.status_code == 200

    def test_get_receipt_not_found(self, gauntlet_handler):
        """Should return 404 for unknown gauntlet."""
        with patch.object(gauntlet_module, "_get_storage") as mock_storage:
            mock_store = Mock()
            mock_store.get.return_value = None
            mock_storage.return_value = mock_store

            result = run_async(
                gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-nonexistent-000000/receipt", "GET", None
                )
            )

            assert result.status_code == 404

    def test_get_receipt_invalid_id(self, gauntlet_handler):
        """Should return 400 for invalid gauntlet ID."""
        result = run_async(
            gauntlet_handler.handle("/api/gauntlet/invalid-format/receipt", "GET", None)
        )

        assert result.status_code == 400


# ============================================================================
# Heatmap Endpoint Tests
# ============================================================================


class TestGauntletHeatmap:
    """Tests for GET /api/gauntlet/{id}/heatmap endpoint."""

    def test_get_heatmap_json(self, gauntlet_handler, mock_gauntlet_run):
        """Should return heatmap as JSON (default)."""
        gauntlet_module._gauntlet_runs["gauntlet-20260111120000-abc123"] = mock_gauntlet_run

        with patch("aragora.gauntlet.heatmap.RiskHeatmap", MockRiskHeatmap):
            with patch("aragora.gauntlet.heatmap.HeatmapCell", Mock):
                result = run_async(
                    gauntlet_handler.handle(
                        "/api/gauntlet/gauntlet-20260111120000-abc123/heatmap", "GET", None
                    )
                )

                assert result.status_code == 200

    def test_get_heatmap_invalid_id(self, gauntlet_handler):
        """Should return 400 for invalid gauntlet ID."""
        result = run_async(
            gauntlet_handler.handle("/api/gauntlet/invalid-format/heatmap", "GET", None)
        )

        assert result.status_code == 400


# ============================================================================
# Run Endpoint Tests (POST)
# ============================================================================


class TestGauntletRun:
    """Tests for POST /api/gauntlet/run endpoint."""

    def test_run_requires_input_content(self, gauntlet_handler):
        """Should require input_content in body."""
        mock_handler = create_mock_handler("/api/gauntlet/run")

        # Mock read_json_body to return empty dict
        with patch.object(gauntlet_handler, "read_json_body", return_value={}):
            result = run_async(gauntlet_handler.handle("/api/gauntlet/run", "POST", mock_handler))

            assert result.status_code == 400
            data = json.loads(result.body)
            assert "input_content" in data["error"].lower()

    def test_run_returns_accepted(self, gauntlet_handler):
        """Should return 202 Accepted with gauntlet ID."""
        mock_handler = create_mock_handler("/api/gauntlet/run")

        with patch.object(
            gauntlet_handler,
            "read_json_body",
            return_value={
                "input_content": "Test content to validate",
                "input_type": "spec",
            },
        ):
            result = run_async(gauntlet_handler.handle("/api/gauntlet/run", "POST", mock_handler))

            assert result.status_code == 202
            data = json.loads(result.body)
            assert "gauntlet_id" in data
            assert data["status"] == "pending"


# ============================================================================
# Delete Endpoint Tests
# ============================================================================


class TestGauntletDelete:
    """Tests for DELETE /api/gauntlet/{id} endpoint."""

    def test_delete_from_memory(self, gauntlet_handler, mock_gauntlet_run):
        """Should delete from in-memory storage."""
        gauntlet_module._gauntlet_runs["gauntlet-20260111120000-abc123"] = mock_gauntlet_run

        with patch.object(gauntlet_module, "_get_storage") as mock_storage:
            mock_store = Mock()
            mock_store.delete.return_value = True
            mock_storage.return_value = mock_store

            result = run_async(
                gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-20260111120000-abc123", "DELETE", None
                )
            )

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["deleted"] is True

    def test_delete_not_found(self, gauntlet_handler):
        """Should return 404 for unknown gauntlet."""
        with patch.object(gauntlet_module, "_get_storage") as mock_storage:
            mock_store = Mock()
            mock_store.delete.return_value = False
            mock_storage.return_value = mock_store

            result = run_async(
                gauntlet_handler.handle("/api/gauntlet/gauntlet-nonexistent-000000", "DELETE", None)
            )

            assert result.status_code == 404

    def test_delete_invalid_id(self, gauntlet_handler):
        """Should return 400 for invalid gauntlet ID."""
        result = run_async(gauntlet_handler.handle("/api/gauntlet/invalid-format", "DELETE", None))

        assert result.status_code == 400


# ============================================================================
# Security Tests
# ============================================================================


class TestGauntletSecurity:
    """Security and edge case tests."""

    def test_path_traversal_in_id(self, gauntlet_handler):
        """Should reject path traversal attempts."""
        result = run_async(
            gauntlet_handler.handle("/api/gauntlet/../../../etc/passwd", "GET", None)
        )

        # Should return 400 for invalid ID format
        assert result.status_code == 400

    def test_invalid_format_param_defaults_to_json(self, gauntlet_handler, mock_gauntlet_run):
        """Should default to JSON for invalid format parameter."""
        gauntlet_module._gauntlet_runs["gauntlet-20260111120000-abc123"] = mock_gauntlet_run
        mock_handler = mock_handler_with_query(
            "/api/gauntlet/gauntlet-20260111120000-abc123/receipt",
            {"format": "<script>alert(1)</script>"},
        )

        with patch("aragora.gauntlet.receipt.DecisionReceipt", MockDecisionReceipt):
            result = run_async(
                gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-20260111120000-abc123/receipt", "GET", mock_handler
                )
            )

            # Should default to JSON for invalid format
            assert result.status_code == 200

    def test_long_format_param(self, gauntlet_handler, mock_gauntlet_run):
        """Should handle overly long format parameter."""
        gauntlet_module._gauntlet_runs["gauntlet-20260111120000-abc123"] = mock_gauntlet_run
        mock_handler = mock_handler_with_query(
            "/api/gauntlet/gauntlet-20260111120000-abc123/receipt", {"format": "x" * 100}
        )

        with patch("aragora.gauntlet.receipt.DecisionReceipt", MockDecisionReceipt):
            result = run_async(
                gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-20260111120000-abc123/receipt", "GET", mock_handler
                )
            )

            # Should default to JSON
            assert result.status_code == 200


# ============================================================================
# Integration Tests
# ============================================================================


class TestGauntletIntegration:
    """Integration tests for full workflows."""

    def test_run_then_get_status(self, gauntlet_handler):
        """Test running gauntlet then getting status."""
        mock_handler = create_mock_handler("/api/gauntlet/run")

        with patch.object(
            gauntlet_handler,
            "read_json_body",
            return_value={
                "input_content": "Test content",
                "input_type": "spec",
            },
        ):
            run_result = run_async(
                gauntlet_handler.handle("/api/gauntlet/run", "POST", mock_handler)
            )

            assert run_result.status_code == 202
            data = json.loads(run_result.body)
            gauntlet_id = data["gauntlet_id"]

            # Status should be available immediately
            status_result = run_async(
                gauntlet_handler.handle(f"/api/gauntlet/{gauntlet_id}", "GET", None)
            )

            assert status_result.status_code == 200
            status_data = json.loads(status_result.body)
            assert status_data["gauntlet_id"] == gauntlet_id

    def test_status_to_receipt_workflow(self, gauntlet_handler, mock_gauntlet_run):
        """Test getting status then receipt."""
        gauntlet_module._gauntlet_runs["gauntlet-20260111120000-abc123"] = mock_gauntlet_run

        # Get status
        status_result = run_async(
            gauntlet_handler.handle("/api/gauntlet/gauntlet-20260111120000-abc123", "GET", None)
        )
        assert status_result.status_code == 200

        # Get receipt
        with patch("aragora.gauntlet.receipt.DecisionReceipt", MockDecisionReceipt):
            receipt_result = run_async(
                gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-20260111120000-abc123/receipt", "GET", None
                )
            )
            assert receipt_result.status_code == 200
