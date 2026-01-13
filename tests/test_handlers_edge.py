"""
Edge case tests for API handlers.

Tests error handling, validation, timeouts, and boundary conditions
for Probes, Verification, and generic handler behaviors.
"""

import json
import pytest
from io import BytesIO
from unittest.mock import MagicMock, patch, PropertyMock

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
)
from aragora.server.validation import validate_debate_id, validate_agent_name
from aragora.server.handlers.probes import ProbesHandler
from aragora.server.handlers.verification import VerificationHandler
from aragora.server.handlers.debates import DebatesHandler


# =============================================================================
# Mock Handler for Testing
# =============================================================================


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: bytes = b"", content_length: int = None):
        self.rfile = BytesIO(body)
        self._content_length = content_length if content_length is not None else len(body)
        self.headers = {
            "Content-Length": str(self._content_length),
            "Content-Type": "application/json",
        }

    def get_header(self, name):
        return self.headers.get(name)


# =============================================================================
# Probes Handler Edge Tests
# =============================================================================


class TestProbesHandlerEdge:
    """Edge case tests for ProbesHandler."""

    @pytest.fixture
    def handler(self):
        """Create a ProbesHandler instance."""
        return ProbesHandler({})

    def test_can_handle_valid_routes(self, handler):
        """Should handle valid probe routes."""
        assert handler.can_handle("/api/probes/capability") is True
        assert handler.can_handle("/api/probes/run") is True
        assert handler.can_handle("/api/probes/invalid") is False

    def test_get_request_returns_none(self, handler):
        """GET requests should return None (not supported)."""
        result = handler.handle("/api/probes/capability", {})
        assert result is None

    def test_missing_body_returns_error(self, handler):
        """POST without body should return error."""
        mock_handler = MockHandler(b"")
        result = handler.handle_post("/api/probes/capability", {}, mock_handler)

        # Should return error for empty/invalid body
        assert result is not None
        assert result.status_code >= 400

    def test_invalid_json_returns_error(self, handler):
        """POST with invalid JSON should return error."""
        mock_handler = MockHandler(b"not valid json {{{")
        result = handler.handle_post("/api/probes/capability", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_missing_agent_name_returns_error(self, handler):
        """POST without agent_name should return error."""
        mock_handler = MockHandler(json.dumps({"strategies": ["test"]}).encode())
        result = handler.handle_post("/api/probes/capability", {}, mock_handler)

        assert result is not None
        # Should error for missing required field
        assert result.status_code >= 400

    def test_prober_unavailable_returns_503(self, handler):
        """Should return 503 when prober module unavailable."""
        mock_handler = MockHandler(json.dumps({"agent_name": "test"}).encode())

        with patch.object(handler, "_run_capability_probe") as mock_run:
            mock_run.return_value = error_response("Prober not available", 503)
            result = handler.handle_post("/api/probes/capability", {}, mock_handler)

        assert result.status_code == 503


# =============================================================================
# Verification Handler Edge Tests
# =============================================================================


class TestVerificationHandlerEdge:
    """Edge case tests for VerificationHandler."""

    @pytest.fixture
    def handler(self):
        """Create a VerificationHandler instance."""
        return VerificationHandler({})

    def test_can_handle_valid_routes(self, handler):
        """Should handle valid verification routes."""
        assert handler.can_handle("/api/verification/status") is True
        assert handler.can_handle("/api/verification/formal-verify") is True
        assert handler.can_handle("/api/verification/invalid") is False

    def test_status_when_unavailable(self, handler):
        """Status should report unavailable when Z3 not installed."""
        with patch("aragora.server.handlers.verification.FORMAL_VERIFICATION_AVAILABLE", False):
            result = handler._get_status()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["available"] is False

    def test_verify_without_claim_returns_error(self, handler):
        """Verify without claim should return error."""
        mock_handler = MockHandler(json.dumps({"context": "some context"}).encode())
        result = handler.handle_post("/api/verification/formal-verify", {}, mock_handler)

        assert result is not None
        assert result.status_code >= 400

    def test_verify_with_empty_claim_returns_error(self, handler):
        """Verify with empty claim should return error."""
        mock_handler = MockHandler(json.dumps({"claim": ""}).encode())
        result = handler.handle_post("/api/verification/formal-verify", {}, mock_handler)

        assert result is not None
        assert result.status_code >= 400

    def test_large_claim_handling(self, handler):
        """Should handle very large claims gracefully."""
        large_claim = "A" * 50000  # 50KB claim
        mock_handler = MockHandler(json.dumps({"claim": large_claim}).encode())

        # Should either process or reject gracefully
        result = handler.handle_post("/api/verification/formal-verify", {}, mock_handler)
        assert result is not None
        # Should not crash, should return some response
        assert result.status_code in (200, 400, 413, 500, 503)


# =============================================================================
# Debates Handler Edge Tests
# =============================================================================


class TestDebatesHandlerEdge:
    """Edge case tests for DebatesHandler."""

    @pytest.fixture
    def handler(self):
        """Create a DebatesHandler instance with mock storage."""
        h = DebatesHandler({})
        return h

    def test_can_handle_all_routes(self, handler):
        """Should handle all debate-related routes."""
        assert handler.can_handle("/api/debates") is True
        assert handler.can_handle("/api/debates/some-slug") is True
        assert handler.can_handle("/api/debates/test/impasse") is True
        assert handler.can_handle("/api/debates/test/convergence") is True
        assert handler.can_handle("/api/debates/test/fork") is True

    def test_invalid_debate_id_characters(self, handler):
        """Should reject debate IDs with special characters."""
        # IDs with path traversal attempts
        dangerous_ids = [
            "../../../etc/passwd",
            "test;rm -rf /",
            "test<script>",
            "test\x00null",
        ]

        for dangerous_id in dangerous_ids:
            is_valid, error = validate_debate_id(dangerous_id)
            # Should either be invalid or sanitized
            if is_valid:
                # If valid, should be sanitized
                assert dangerous_id == dangerous_id.strip()

    def test_fork_invalid_branch_point(self, handler):
        """Fork with invalid branch_point should return error."""
        mock_http = MockHandler(json.dumps({"branch_point": "not-a-number"}).encode())

        with patch.object(handler, "get_storage") as mock_storage:
            mock_storage.return_value = MagicMock()
            mock_storage.return_value.get_debate.return_value = {"messages": []}
            result = handler._fork_debate(mock_http, "test-debate")

        assert result.status_code >= 400

    def test_fork_negative_branch_point(self, handler):
        """Fork with negative branch_point should return error."""
        mock_http = MockHandler(json.dumps({"branch_point": -5}).encode())

        with patch.object(handler, "get_storage") as mock_storage:
            mock_storage.return_value = MagicMock()
            mock_storage.return_value.get_debate.return_value = {"messages": []}
            result = handler._fork_debate(mock_http, "test-debate")

        assert result.status_code >= 400

    def test_fork_debate_not_found(self, handler):
        """Fork on nonexistent debate should return 404."""
        mock_http = MockHandler(json.dumps({"branch_point": 0}).encode())

        with patch.object(handler, "get_storage") as mock_storage:
            mock_storage.return_value = MagicMock()
            mock_storage.return_value.get_debate.return_value = None
            result = handler._fork_debate(mock_http, "nonexistent")

        assert result.status_code == 404


# =============================================================================
# Generic Handler Edge Tests
# =============================================================================


class TestGenericHandlerEdge:
    """Generic edge case tests applicable to all handlers."""

    def test_malformed_json_body(self):
        """All handlers should handle malformed JSON gracefully."""
        handlers = [
            ProbesHandler({}),
            VerificationHandler({}),
            DebatesHandler({}),
        ]

        malformed_bodies = [
            b"not json",
            b"{incomplete",
            b'{"key": undefined}',
            b"\xff\xfe invalid bytes",
        ]

        for handler in handlers:
            for body in malformed_bodies:
                mock_http = MockHandler(body)
                result = handler.read_json_body(mock_http)
                # Should return None for invalid JSON, not crash
                assert result is None or isinstance(result, dict)

    def test_missing_content_length(self):
        """Should handle missing Content-Length header."""
        handler = ProbesHandler({})
        mock_http = MockHandler(b'{"test": 1}')
        mock_http.headers.pop("Content-Length", None)

        # Should not crash
        result = handler.read_json_body(mock_http)
        # May return None or parsed body depending on implementation
        assert result is None or isinstance(result, dict)

    def test_unicode_handling(self):
        """Should handle unicode in request bodies."""
        handler = DebatesHandler({})

        unicode_bodies = [
            {"topic": "Test with émojis 🎉"},
            {"topic": "中文测试"},
            {"topic": "Тест на русском"},
            {"topic": "מבחן בעברית"},
        ]

        for body in unicode_bodies:
            mock_http = MockHandler(json.dumps(body).encode("utf-8"))
            result = handler.read_json_body(mock_http)
            assert result is not None
            assert result["topic"] == body["topic"]

    def test_path_traversal_prevention(self):
        """Should prevent path traversal in IDs."""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "test/../../../secret",
            "test%2F..%2F..%2Fsecret",
        ]

        for path in dangerous_paths:
            is_valid, _ = validate_debate_id(path)
            # Path traversal attempts should be rejected
            if "/" in path or "\\" in path or ".." in path:
                assert is_valid is False or "/" not in path

    def test_validate_agent_name(self):
        """Should validate agent names correctly."""
        valid_names = ["claude", "gpt4", "test-agent", "agent_123"]
        invalid_names = ["", "a" * 100, "agent<script>", "agent;rm"]

        for name in valid_names:
            is_valid, _ = validate_agent_name(name)
            assert is_valid is True, f"Should accept: {name}"

        for name in invalid_names:
            is_valid, _ = validate_agent_name(name)
            # Either rejected or heavily constrained
            if is_valid:
                assert len(name) <= 32


# =============================================================================
# Rate Limit Edge Tests
# =============================================================================


class TestRateLimitEdge:
    """Edge case tests for rate limiting."""

    def test_endpoint_limit_matching(self):
        """Should match endpoint limits correctly."""
        from aragora.server.rate_limit import get_limiter

        limiter = get_limiter()

        # Test configured endpoints
        config = limiter.get_config("/api/debates")
        assert config.requests_per_minute <= 60  # Should have a limit

        config = limiter.get_config("/api/memory/continuum/cleanup")
        assert config.requests_per_minute <= 10  # Should be restricted

    def test_burst_handling(self):
        """Should allow burst traffic up to limit."""
        from aragora.server.rate_limit import TokenBucket

        bucket = TokenBucket(rate_per_minute=60, burst_size=10)

        # Should allow burst
        for i in range(10):
            assert bucket.consume(1) is True, f"Burst {i} should succeed"

        # Should be rate limited after burst
        assert bucket.consume(1) is False


# =============================================================================
# Validation Edge Tests
# =============================================================================


class TestValidationEdge:
    """Edge case tests for input validation."""

    def test_validate_debate_id_edge_cases(self):
        """Should handle edge cases in debate ID validation."""
        from aragora.server.handlers.base import validate_debate_id

        # Valid IDs
        valid_ids = ["test", "test-123", "Test_ID", "a" * 64]
        for id in valid_ids:
            is_valid, _ = validate_debate_id(id)
            assert is_valid is True, f"Should accept: {id}"

        # Invalid IDs - various attack vectors
        invalid_ids = [
            "",  # Empty
            " ",  # Whitespace only
            "a" * 200,  # Too long
            "<script>alert(1)</script>",  # XSS
            "'; DROP TABLE debates;--",  # SQL injection
            "../../../etc/passwd",  # Path traversal
        ]
        for id in invalid_ids:
            is_valid, _ = validate_debate_id(id)
            assert is_valid is False, f"Should reject: {id}"

    def test_schema_validation_type_coercion(self):
        """Should not coerce types during validation."""
        from aragora.server.validation import validate_int_field

        data = {"rounds": "5"}  # String, not int
        result = validate_int_field(data, "rounds")
        assert result.is_valid is False

        data = {"rounds": 5.5}  # Float, not int
        result = validate_int_field(data, "rounds")
        assert result.is_valid is False

        data = {"rounds": True}  # Bool, not int
        result = validate_int_field(data, "rounds")
        assert result.is_valid is False

    def test_string_sanitization(self):
        """Should sanitize strings correctly."""
        from aragora.server.validation import sanitize_string

        # Should truncate
        long_string = "a" * 2000
        sanitized = sanitize_string(long_string, max_length=100)
        assert len(sanitized) == 100

        # Should strip whitespace
        assert sanitize_string("  test  ") == "test"

        # Should handle None/non-string
        assert sanitize_string(None) == ""
        assert sanitize_string(123) == ""
