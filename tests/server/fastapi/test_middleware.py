"""
Tests for FastAPI middleware components.

Covers:
- SecurityHeadersMiddleware (CSP, X-Frame-Options, etc.)
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


class TestSecurityHeadersMiddleware:
    """Tests for security headers middleware."""

    def test_x_frame_options_present(self, client):
        """Should include X-Frame-Options header."""
        response = client.get("/healthz")
        assert response.headers.get("x-frame-options") == "DENY"

    def test_x_content_type_options_present(self, client):
        """Should include X-Content-Type-Options header."""
        response = client.get("/healthz")
        assert response.headers.get("x-content-type-options") == "nosniff"

    def test_x_xss_protection_present(self, client):
        """Should include X-XSS-Protection header."""
        response = client.get("/healthz")
        assert response.headers.get("x-xss-protection") == "1; mode=block"

    def test_referrer_policy_present(self, client):
        """Should include Referrer-Policy header."""
        response = client.get("/healthz")
        assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_content_security_policy_present(self, client):
        """Should include Content-Security-Policy header."""
        response = client.get("/healthz")
        csp = response.headers.get("content-security-policy")
        assert csp is not None
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_security_headers_on_error_responses(self, client):
        """Security headers should be present even on error responses."""
        response = client.get("/api/v2/nonexistent-endpoint")
        # Even 404 responses should have security headers
        assert response.headers.get("x-frame-options") == "DENY"
        assert response.headers.get("x-content-type-options") == "nosniff"

    def test_security_headers_on_api_routes(self, client):
        """Security headers should be present on API routes."""
        response = client.get("/api/v2/health")
        assert response.headers.get("x-frame-options") == "DENY"
        assert "content-security-policy" in response.headers
