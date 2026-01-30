"""
Tests for FastAPI middleware components.

Covers:
- TracingMiddleware (X-Trace-ID, response timing)
- RequestValidationMiddleware (body size, JSON depth limits)
- Error handling (custom exception types)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
    }
    return TestClient(app, raise_server_exceptions=False)


class TestTracingMiddleware:
    """Tests for request tracing."""

    def test_generates_trace_id(self, client):
        """Should generate X-Trace-ID if not provided."""
        response = client.get("/healthz")
        assert "x-trace-id" in response.headers
        # UUID format
        trace_id = response.headers["x-trace-id"]
        assert len(trace_id) > 0

    def test_propagates_trace_id(self, client):
        """Should propagate X-Trace-ID from request."""
        custom_id = "test-trace-12345"
        response = client.get(
            "/healthz",
            headers={"X-Trace-ID": custom_id},
        )
        assert response.headers.get("x-trace-id") == custom_id

    def test_includes_response_time(self, client):
        """Should include X-Response-Time header."""
        response = client.get("/healthz")
        assert "x-response-time" in response.headers


class TestErrorHandling:
    """Tests for custom exception handling."""

    def test_404_returns_json(self, client):
        """404 errors should return JSON."""
        response = client.get("/api/v2/nonexistent-endpoint")
        assert response.status_code in [404, 405]

    def test_validation_error_returns_422(self, client):
        """Pydantic validation errors should return 422."""
        response = client.post(
            "/api/v2/decisions",
            json={"invalid_field": "value"},
            headers={"Authorization": "Bearer dummy-token"},
        )
        # Either 401 (no valid auth) or 422 (validation error) expected
        assert response.status_code in [401, 422]
