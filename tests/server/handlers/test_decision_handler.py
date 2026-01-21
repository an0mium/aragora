"""
Tests for the DecisionHandler.

Verifies the /api/decisions endpoints work correctly for
unified decision-making API.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestDecisionHandlerRouting:
    """Test route matching for DecisionHandler."""

    def test_can_handle_decisions_root(self):
        """Test that handler matches /api/decisions."""
        from aragora.server.handlers.decision import DecisionHandler

        handler = DecisionHandler({})
        assert handler.can_handle("/api/decisions") is True

    def test_can_handle_decisions_with_id(self):
        """Test that handler matches /api/decisions/:id."""
        from aragora.server.handlers.decision import DecisionHandler

        handler = DecisionHandler({})
        assert handler.can_handle("/api/decisions/req-123") is True

    def test_can_handle_decisions_status(self):
        """Test that handler matches /api/decisions/:id/status."""
        from aragora.server.handlers.decision import DecisionHandler

        handler = DecisionHandler({})
        assert handler.can_handle("/api/decisions/req-123/status") is True

    def test_cannot_handle_other_paths(self):
        """Test that handler doesn't match other paths."""
        from aragora.server.handlers.decision import DecisionHandler

        handler = DecisionHandler({})
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/decision") is False  # No 's'
        assert handler.can_handle("/other/decisions") is False


class TestDecisionHandlerGet:
    """Test GET endpoints for DecisionHandler."""

    def test_list_decisions_empty(self):
        """Test listing decisions when none exist."""
        from aragora.server.handlers.decision import DecisionHandler, _decision_results_fallback

        # Clear any existing results
        _decision_results_fallback.clear()

        # Patch the result store to use fallback
        with patch("aragora.server.handlers.decision._get_result_store", return_value=None):
            handler = DecisionHandler({})
            result = handler.handle("/api/decisions", {})

            assert result is not None
            assert result.status_code == 200

            body = json.loads(result.body)
            assert body["total"] == 0
            assert body["decisions"] == []

    def test_get_decision_not_found(self):
        """Test getting a non-existent decision."""
        from aragora.server.handlers.decision import DecisionHandler, _decision_results_fallback

        _decision_results_fallback.clear()

        with patch("aragora.server.handlers.decision._get_result_store", return_value=None):
            handler = DecisionHandler({})
            result = handler.handle("/api/decisions/nonexistent-123", {})

            assert result is not None
            assert result.status_code == 404

    def test_get_decision_found(self):
        """Test getting an existing decision."""
        from aragora.server.handlers.decision import DecisionHandler, _decision_results_fallback

        _decision_results_fallback.clear()
        _decision_results_fallback["test-req-456"] = {
            "request_id": "test-req-456",
            "status": "completed",
            "completed_at": "2026-01-21T00:00:00",
            "result": {"answer": "Test answer"},
        }

        with patch("aragora.server.handlers.decision._get_result_store", return_value=None):
            handler = DecisionHandler({})
            result = handler.handle("/api/decisions/test-req-456", {})

            assert result is not None
            assert result.status_code == 200

            body = json.loads(result.body)
            assert body["request_id"] == "test-req-456"
            assert body["status"] == "completed"

    def test_get_decision_status(self):
        """Test getting decision status for polling."""
        from aragora.server.handlers.decision import DecisionHandler, _decision_results_fallback

        _decision_results_fallback.clear()
        _decision_results_fallback["poll-req-789"] = {
            "request_id": "poll-req-789",
            "status": "completed",
            "completed_at": "2026-01-21T01:00:00",
        }

        with patch("aragora.server.handlers.decision._get_result_store", return_value=None):
            handler = DecisionHandler({})
            result = handler.handle("/api/decisions/poll-req-789/status", {})

            assert result is not None
            assert result.status_code == 200

            body = json.loads(result.body)
            assert body["request_id"] == "poll-req-789"
            assert body["status"] == "completed"

    def test_get_decision_status_not_found(self):
        """Test getting status for non-existent decision."""
        from aragora.server.handlers.decision import DecisionHandler, _decision_results_fallback

        _decision_results_fallback.clear()

        with patch("aragora.server.handlers.decision._get_result_store", return_value=None):
            handler = DecisionHandler({})
            result = handler.handle("/api/decisions/missing/status", {})

            assert result is not None
            assert result.status_code == 200  # Returns status, not 404

            body = json.loads(result.body)
            assert body["status"] == "not_found"


class TestDecisionHandlerPost:
    """Test POST endpoints for DecisionHandler."""

    def test_create_decision_missing_content(self):
        """Test that missing content returns 400."""
        from aragora.server.handlers.decision import DecisionHandler

        handler = DecisionHandler({})

        # Create mock request handler with empty body (no "content" field)
        body_bytes = b"{}"
        mock_handler = MagicMock()
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_handler.rfile.read.return_value = body_bytes

        result = handler.handle_post("/api/decisions", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        assert b"content" in result.body.lower()

    @patch("aragora.server.handlers.decision._get_decision_router")
    @patch("aragora.billing.auth.extract_user_from_request")
    def test_create_decision_router_unavailable(self, mock_auth, mock_get_router):
        """Test error when router is not available."""
        from aragora.server.handlers.decision import DecisionHandler

        mock_get_router.return_value = None
        mock_auth.return_value = MagicMock(authenticated=False)

        handler = DecisionHandler({})

        body_bytes = json.dumps({"content": "Test question"}).encode()
        mock_handler = MagicMock()
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_handler.rfile.read.return_value = body_bytes

        result = handler.handle_post("/api/decisions", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    @patch("aragora.server.handlers.decision._get_decision_router")
    @patch("aragora.billing.auth.extract_user_from_request")
    def test_create_decision_success(self, mock_auth, mock_get_router):
        """Test successful decision creation."""
        from aragora.server.handlers.decision import DecisionHandler, _decision_results_fallback
        from aragora.core.decision import DecisionResult, DecisionType

        _decision_results_fallback.clear()

        # Mock the router
        mock_router = MagicMock()
        mock_result = DecisionResult(
            request_id="new-req-001",
            decision_type=DecisionType.QUICK,
            success=True,
            answer="Test answer from mock",
            confidence=0.85,
            consensus_reached=True,
            reasoning="Mock reasoning",
        )

        async def mock_route(request):
            return mock_result

        mock_router.route = mock_route
        mock_get_router.return_value = mock_router

        mock_auth.return_value = MagicMock(
            authenticated=True,
            user_id="test-user",
            org_id="test-org",
        )

        handler = DecisionHandler({})

        body_bytes = json.dumps(
            {
                "content": "What is the capital of France?",
                "decision_type": "quick",
            }
        ).encode()
        mock_handler = MagicMock()
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_handler.rfile.read.return_value = body_bytes

        result = handler.handle_post("/api/decisions", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert body["status"] == "completed"
        assert body["answer"] == "Test answer from mock"
        assert body["confidence"] == 0.85


class TestDecisionResultCache:
    """Test the in-memory result caching."""

    def test_list_with_limit(self):
        """Test listing decisions with limit."""
        from aragora.server.handlers.decision import DecisionHandler, _decision_results_fallback

        _decision_results_fallback.clear()

        # Add multiple results
        for i in range(10):
            _decision_results_fallback[f"req-{i}"] = {
                "request_id": f"req-{i}",
                "status": "completed",
            }

        with patch("aragora.server.handlers.decision._get_result_store", return_value=None):
            handler = DecisionHandler({})
            result = handler.handle("/api/decisions", {"limit": "3"})

            assert result is not None
            body = json.loads(result.body)
            assert len(body["decisions"]) == 3
            assert body["total"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
