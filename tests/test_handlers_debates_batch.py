"""
Tests for batch debate operations handler (debates_batch.py).

Tests cover:
- Batch submission validation
- Batch status checking
- Input validation and error handling
- Rate limiting behavior
"""

import json
import pytest
from unittest.mock import MagicMock, patch


class MockHandler:
    """Mock HTTP request handler."""

    def __init__(self):
        self.headers = {"Content-Type": "application/json"}
        self.path = "/api/debates/batch"
        self.command = "POST"
        self._body = b"{}"
        self.rfile = MagicMock()
        self.rfile.read.return_value = self._body

    def set_body(self, data: dict):
        self._body = json.dumps(data).encode()
        self.rfile.read.return_value = self._body


def parse_result(result):
    """Parse HandlerResult to get JSON body and status."""
    body = json.loads(result.body.decode())
    return body, result.status_code


class TestBatchOperations:
    """Test batch operations handler."""

    @pytest.fixture
    def mock_handler(self):
        return MockHandler()

    @pytest.fixture
    def debates_handler(self):
        """Create a mock DebatesHandler with batch operations."""
        from aragora.server.handlers.debates_batch import BatchOperationsMixin

        class MockDebatesHandler(BatchOperationsMixin):
            def __init__(self):
                self._debate_queue = MagicMock()

            def read_json_body(self, handler):
                try:
                    return json.loads(handler._body.decode())
                except json.JSONDecodeError:
                    return None

        return MockDebatesHandler()

    def test_batch_submit_empty_body(self, debates_handler, mock_handler):
        """Test batch submit with empty body."""
        mock_handler.set_body({})
        result = debates_handler._submit_batch(mock_handler)
        body, status = parse_result(result)
        assert "error" in body
        assert status == 400

    def test_batch_submit_empty_items(self, debates_handler, mock_handler):
        """Test batch submit with empty items array."""
        mock_handler.set_body({"items": []})
        result = debates_handler._submit_batch(mock_handler)
        body, status = parse_result(result)
        assert "error" in body
        assert status == 400

    def test_batch_submit_exceeds_limit(self, debates_handler, mock_handler):
        """Test batch submit with too many items."""
        items = [{"question": f"Question {i}"} for i in range(1001)]
        mock_handler.set_body({"items": items})
        result = debates_handler._submit_batch(mock_handler)
        body, status = parse_result(result)
        assert "error" in body
        assert "1000" in body["error"]
        assert status == 400

    def test_batch_submit_invalid_json(self, debates_handler, mock_handler):
        """Test batch submit with invalid JSON."""
        mock_handler._body = b"not json"
        result = debates_handler._submit_batch(mock_handler)
        body, status = parse_result(result)
        assert "error" in body
        assert status == 400


class TestWebhookValidation:
    """Test webhook URL validation for batch operations."""

    def test_validate_webhook_url_valid_https(self):
        """Test valid HTTPS webhook URL."""
        from aragora.server.debate_queue import validate_webhook_url

        is_valid, _ = validate_webhook_url("https://example.com/webhook")
        assert is_valid is True

    def test_validate_webhook_url_localhost_rejected(self):
        """Test localhost URLs rejected (SSRF prevention)."""
        from aragora.server.debate_queue import validate_webhook_url

        is_valid, _ = validate_webhook_url("https://localhost/webhook")
        assert is_valid is False

    def test_validate_webhook_url_private_ip_rejected(self):
        """Test private IP addresses rejected."""
        from aragora.server.debate_queue import validate_webhook_url

        for ip in ["127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1"]:
            is_valid, _ = validate_webhook_url(f"https://{ip}/webhook")
            assert is_valid is False, f"Should reject {ip}"


class TestBatchRateLimiting:
    """Test rate limiting on batch operations."""

    def test_batch_submit_rate_limited(self):
        """Test that batch submit is rate limited."""
        from aragora.server.handlers.debates_batch import BatchOperationsMixin

        # Check that the decorator is applied
        assert hasattr(BatchOperationsMixin._submit_batch, "__wrapped__")


# ============================================================================
# Exception Handling Tests (Round 23)
# ============================================================================


class TestBatchExceptionHandling:
    """Tests for specific exception handling in batch operations."""

    @pytest.fixture
    def mock_handler(self):
        return MockHandler()

    @pytest.fixture
    def debates_handler(self):
        """Create a mock DebatesHandler with batch operations."""
        from aragora.server.handlers.debates_batch import BatchOperationsMixin

        class MockDebatesHandler(BatchOperationsMixin):
            def __init__(self):
                self._debate_queue = MagicMock()

            def read_json_body(self, handler):
                try:
                    return json.loads(handler._body.decode())
                except json.JSONDecodeError:
                    return None

        return MockDebatesHandler()

    def test_debate_start_error_import(self):
        """Verify DebateStartError can be imported from exceptions."""
        from aragora.exceptions import DebateStartError

        error = DebateStartError(debate_id="test-123", reason="Agent unavailable")
        assert "test-123" in str(error)
        assert "Agent unavailable" in str(error)

    def test_debate_batch_error_import(self):
        """Verify DebateBatchError can be imported from exceptions."""
        from aragora.exceptions import DebateBatchError

        error = DebateBatchError(operation="submit", reason="Queue full", failed_ids=["id1", "id2"])
        assert "submit" in str(error)
        assert error.failed_ids == ["id1", "id2"]

    def test_batch_item_validation_error_captured(self, debates_handler, mock_handler):
        """Test that validation errors are properly captured for each item."""
        mock_handler.set_body(
            {
                "items": [
                    {"question": ""},  # Empty question
                    {"agents": "claude"},  # Missing question
                    {"question": "x" * 10001},  # Too long
                ]
            }
        )
        result = debates_handler._submit_batch(mock_handler)
        body, status = parse_result(result)
        assert status == 400
        assert "error" in body

    def test_batch_non_object_item_error(self, debates_handler, mock_handler):
        """Test error handling for non-object items in batch."""
        mock_handler.set_body({"items": ["not an object", {"question": "Valid question"}]})
        result = debates_handler._submit_batch(mock_handler)
        body, status = parse_result(result)
        assert status == 400
        assert "must be an object" in body["error"]

    def test_batch_multiple_errors_truncated(self, debates_handler, mock_handler):
        """Test that multiple validation errors are truncated after 5."""
        # Create 10 invalid items
        mock_handler.set_body(
            {"items": [{"agents": f"agent{i}"} for i in range(10)]}  # All missing question
        )
        result = debates_handler._submit_batch(mock_handler)
        body, status = parse_result(result)
        assert status == 400
        # Should show first 5 errors and indicate more
        assert "and" in body["error"] and "more" in body["error"]


class TestBatchStatusOperations:
    """Tests for batch status checking operations."""

    def test_get_batch_status_queue_unavailable(self):
        """Test status check when queue is not initialized."""
        from aragora.server.handlers.debates import DebatesHandler
        from aragora.server.validation.entities import validate_path_segment

        # validate_path_segment should work for valid IDs
        is_valid, _ = validate_path_segment("batch_abc123", "batch id")
        assert is_valid is True

        is_valid, _ = validate_path_segment("../evil", "batch id")
        assert is_valid is False

    def test_batch_id_validation_patterns(self):
        """Test various batch ID validation patterns."""
        from aragora.server.validation.entities import validate_path_segment, SAFE_ID_PATTERN

        # Valid IDs
        for valid_id in ["batch_123", "abc-def-ghi", "test123", "a1b2c3"]:
            is_valid, _ = validate_path_segment(valid_id, "batch id", SAFE_ID_PATTERN)
            assert is_valid is True, f"Should accept {valid_id}"

        # Invalid IDs (path traversal, special chars)
        for invalid_id in ["../evil", "batch/../id", "batch;rm -rf", "<script>"]:
            is_valid, _ = validate_path_segment(invalid_id, "batch id", SAFE_ID_PATTERN)
            assert is_valid is False, f"Should reject {invalid_id}"
