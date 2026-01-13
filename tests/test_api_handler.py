"""
Tests for DebateAPIHandler in api.py.

Covers:
- Safe ID validation
- Debate listing and retrieval
- Replay listing and retrieval
- Replay forking
- Static file serving with path traversal protection
- Health check
- CORS headers
"""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from datetime import datetime

import pytest

from aragora.server.api import DebateAPIHandler, SAFE_ID_PATTERN


# ============================================================================
# Safe ID Pattern Tests
# ============================================================================


class TestSafeIDPattern:
    """Tests for SAFE_ID_PATTERN regex."""

    def test_valid_simple_id(self):
        """Simple alphanumeric IDs should be valid."""
        assert SAFE_ID_PATTERN.match("debate123") is not None
        assert SAFE_ID_PATTERN.match("test") is not None
        assert SAFE_ID_PATTERN.match("a") is not None

    def test_valid_id_with_dashes(self):
        """IDs with dashes should be valid."""
        assert SAFE_ID_PATTERN.match("my-debate-2024") is not None
        assert SAFE_ID_PATTERN.match("rate-limiter-2026-01-01") is not None

    def test_valid_id_with_underscores(self):
        """IDs with underscores should be valid."""
        assert SAFE_ID_PATTERN.match("my_debate") is not None
        assert SAFE_ID_PATTERN.match("test_123_abc") is not None

    def test_valid_id_with_dots(self):
        """IDs with dots should be valid."""
        assert SAFE_ID_PATTERN.match("v1.2.3") is not None
        assert SAFE_ID_PATTERN.match("config.json") is not None

    def test_invalid_empty_string(self):
        """Empty string should be invalid."""
        assert SAFE_ID_PATTERN.match("") is None

    def test_invalid_starts_with_special(self):
        """IDs starting with special chars should be invalid."""
        assert SAFE_ID_PATTERN.match("-start") is None
        assert SAFE_ID_PATTERN.match("_start") is None
        assert SAFE_ID_PATTERN.match(".start") is None

    def test_invalid_path_traversal(self):
        """Path traversal attempts should be invalid."""
        assert SAFE_ID_PATTERN.match("../etc/passwd") is None
        assert SAFE_ID_PATTERN.match("..") is None
        assert SAFE_ID_PATTERN.match("foo/../bar") is None

    def test_invalid_special_chars(self):
        """Special characters should be invalid."""
        assert SAFE_ID_PATTERN.match("foo$bar") is None
        assert SAFE_ID_PATTERN.match("foo bar") is None
        assert SAFE_ID_PATTERN.match("foo/bar") is None

    def test_invalid_too_long(self):
        """IDs over 128 chars should be invalid (security hardened in Round 9)."""
        assert SAFE_ID_PATTERN.match("a" * 128) is not None  # Exactly 128 is valid
        assert SAFE_ID_PATTERN.match("a" * 129) is None  # 129 is invalid


# ============================================================================
# Handler Tests
# ============================================================================


@dataclass
class MockDebate:
    """Mock debate object for testing."""

    slug: str = "test-debate"
    task: str = "Test task"
    agents: list = None
    consensus_reached: bool = True
    confidence: float = 0.9
    view_count: int = 10
    created_at: datetime = None

    def __post_init__(self):
        if self.agents is None:
            self.agents = ["agent1", "agent2"]
        if self.created_at is None:
            self.created_at = datetime.now()


class TestDebateAPIHandler:
    """Tests for DebateAPIHandler."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create a handler with mocked dependencies."""
        handler = DebateAPIHandler.__new__(DebateAPIHandler)
        handler.path = "/api/health"
        handler.headers = {}
        handler.wfile = BytesIO()

        # Mock storage
        handler.storage = Mock()
        handler.replay_storage = Mock()
        handler.static_dir = tmp_path

        # Mock response methods
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.send_error = Mock()

        return handler

    def test_health_check(self, handler):
        """Health check should return ok status."""
        handler._health_check()

        # Verify response was sent
        handler.send_response.assert_called_with(200)
        response = handler.wfile.getvalue()
        data = json.loads(response)
        assert data["status"] == "ok"

    def test_list_debates_empty(self, handler):
        """Empty storage should return empty list."""
        handler.storage.list_recent.return_value = []

        handler._list_debates(20)

        handler.send_response.assert_called_with(200)
        response = handler.wfile.getvalue()
        data = json.loads(response)
        assert data == []

    def test_list_debates_with_data(self, handler):
        """Should return formatted debate list."""
        mock_debate = MockDebate(slug="test-123", task="A test debate")
        handler.storage.list_recent.return_value = [mock_debate]

        handler._list_debates(20)

        response = handler.wfile.getvalue()
        data = json.loads(response)
        assert len(data) == 1
        assert data[0]["slug"] == "test-123"
        assert "task" in data[0]
        assert "consensus" in data[0]

    def test_list_debates_no_storage(self, handler):
        """No storage should return empty list."""
        handler.storage = None

        handler._list_debates(20)

        response = handler.wfile.getvalue()
        data = json.loads(response)
        assert data == []

    def test_get_debate_found(self, handler):
        """Should return debate when found."""
        mock_debate = {"slug": "test", "task": "Test"}
        handler.storage.get_by_slug.return_value = mock_debate

        handler._get_debate("test")

        handler.send_response.assert_called_with(200)
        response = handler.wfile.getvalue()
        data = json.loads(response)
        assert data["slug"] == "test"

    def test_get_debate_not_found(self, handler):
        """Should return 404 when debate not found."""
        handler.storage.get_by_slug.return_value = None

        handler._get_debate("nonexistent")

        handler.send_error.assert_called_with(404, "Debate not found: nonexistent")

    def test_get_debate_no_storage(self, handler):
        """Should return 503 JSON error when storage not configured (updated in Round 9)."""
        handler.storage = None
        handler._send_json_error = Mock()

        handler._get_debate("test")

        handler._send_json_error.assert_called_with("Storage not configured", 503)

    def test_cors_headers_valid_origin(self, handler):
        """CORS headers should be added for valid origin."""
        handler.headers = {"Origin": "http://localhost:3000"}

        with patch("aragora.server.api.ALLOWED_ORIGINS", {"http://localhost:3000"}):
            handler._add_cors_headers()

        # Check that Access-Control-Allow-Origin was set
        calls = [
            call
            for call in handler.send_header.call_args_list
            if call[0][0] == "Access-Control-Allow-Origin"
        ]
        assert len(calls) == 1
        assert calls[0][0][1] == "http://localhost:3000"

    def test_cors_headers_invalid_origin(self, handler):
        """CORS headers should not include origin for invalid origins."""
        handler.headers = {"Origin": "http://evil.com"}

        with patch("aragora.server.api.ALLOWED_ORIGINS", {"http://localhost:3000"}):
            handler._add_cors_headers()

        # Access-Control-Allow-Origin should not be set for invalid origins
        calls = [
            call
            for call in handler.send_header.call_args_list
            if call[0][0] == "Access-Control-Allow-Origin"
        ]
        assert len(calls) == 0


class TestStaticFileServing:
    """Tests for static file serving with security."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create a handler with static directory."""
        handler = DebateAPIHandler.__new__(DebateAPIHandler)
        handler.headers = {}
        handler.wfile = BytesIO()
        handler.static_dir = tmp_path

        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.send_error = Mock()

        return handler

    def test_serve_existing_file(self, handler, tmp_path):
        """Should serve existing files."""
        test_file = tmp_path / "test.html"
        test_file.write_text("<html>Test</html>")

        handler._serve_file("test.html")

        handler.send_response.assert_called_with(200)
        response = handler.wfile.getvalue()
        assert b"<html>Test</html>" in response

    def test_serve_missing_file(self, handler):
        """Should return 404 for missing files."""
        handler._serve_file("nonexistent.html")

        handler.send_error.assert_called_with(404, "File not found")

    def test_serve_file_path_traversal_blocked(self, handler, tmp_path):
        """Path traversal should be blocked."""
        # Create a file outside static dir
        outside_file = tmp_path.parent / "secret.txt"
        outside_file.write_text("secret")

        handler._serve_file("../secret.txt")

        handler.send_error.assert_called_with(403, "Access denied")

    def test_serve_file_no_static_dir(self, handler):
        """Should return 404 when static dir not configured."""
        handler.static_dir = None

        handler._serve_file("test.html")

        handler.send_error.assert_called_with(404, "Static directory not configured")

    def test_serve_file_content_type_css(self, handler, tmp_path):
        """Should set correct content type for CSS."""
        css_file = tmp_path / "style.css"
        css_file.write_text("body { color: red; }")

        handler._serve_file("style.css")

        # Check Content-Type header
        calls = [
            call for call in handler.send_header.call_args_list if call[0][0] == "Content-Type"
        ]
        assert any("text/css" in str(call) for call in calls)

    def test_serve_file_content_type_js(self, handler, tmp_path):
        """Should set correct content type for JS."""
        js_file = tmp_path / "app.js"
        js_file.write_text("console.log('hello');")

        handler._serve_file("app.js")

        calls = [
            call for call in handler.send_header.call_args_list if call[0][0] == "Content-Type"
        ]
        assert any("javascript" in str(call) for call in calls)


class TestReplayEndpoints:
    """Tests for replay-related endpoints."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create a handler with replay storage."""
        handler = DebateAPIHandler.__new__(DebateAPIHandler)
        handler.headers = {"Content-Length": "0"}
        handler.wfile = BytesIO()
        handler.rfile = BytesIO()

        handler.replay_storage = Mock()
        handler.replay_storage.storage_dir = tmp_path

        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.send_error = Mock()

        return handler

    def test_list_replays_empty(self, handler):
        """Empty replay storage should return empty list."""
        handler.replay_storage.list_recordings.return_value = []

        handler._list_replays(20)

        response = handler.wfile.getvalue()
        data = json.loads(response)
        assert data == []

    def test_list_replays_no_storage(self, handler):
        """No replay storage should return empty list."""
        handler.replay_storage = None

        handler._list_replays(20)

        response = handler.wfile.getvalue()
        data = json.loads(response)
        assert data == []

    def test_get_replay_not_found(self, handler):
        """Should return 404 for missing replay."""
        handler._get_replay("nonexistent")

        handler.send_error.assert_called()
        assert (
            "404" in str(handler.send_error.call_args) or handler.send_error.call_args[0][0] == 404
        )

    def test_get_replay_found(self, handler, tmp_path):
        """Should return replay bundle when found."""
        # Create replay files
        session_dir = tmp_path / "test-replay"
        session_dir.mkdir()
        (session_dir / "meta.json").write_text('{"topic": "Test"}')
        (session_dir / "events.jsonl").write_text('{"type": "start"}\n{"type": "end"}')

        handler._get_replay("test-replay")

        handler.send_response.assert_called_with(200)
        response = handler.wfile.getvalue()
        data = json.loads(response)
        assert "meta" in data
        assert "events" in data
        assert len(data["events"]) == 2

    def test_get_replay_no_storage(self, handler):
        """Should return 503 JSON error when replay storage not configured (updated in Round 9)."""
        handler.replay_storage = None
        handler._send_json_error = Mock()

        handler._get_replay("any-id")

        handler._send_json_error.assert_called_with("Replay storage not configured", 503)


class TestForkReplay:
    """Tests for replay forking endpoint."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create a handler for fork testing."""
        handler = DebateAPIHandler.__new__(DebateAPIHandler)
        handler.headers = {"Content-Length": "50"}
        handler.wfile = BytesIO()

        handler.replay_storage = Mock()
        handler.replay_storage.storage_dir = tmp_path

        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.send_error = Mock()

        return handler

    def test_fork_missing_body(self, handler):
        """Should return 400 for missing request body."""
        handler.headers = {"Content-Length": "0"}
        handler.rfile = BytesIO()

        handler._fork_replay("test-replay")

        handler.send_error.assert_called_with(400, "Missing request body")

    def test_fork_invalid_json(self, handler):
        """Should return 400 for invalid JSON."""
        handler.headers = {"Content-Length": "10"}
        handler.rfile = BytesIO(b"not json!")

        handler._fork_replay("test-replay")

        handler.send_error.assert_called_with(400, "Invalid JSON")

    def test_fork_missing_event_id(self, handler):
        """Should return 400 when event_id is missing."""
        body = json.dumps({"config": {}}).encode()
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = BytesIO(body)

        handler._fork_replay("test-replay")

        handler.send_error.assert_called_with(400, "Missing event_id")

    def test_fork_replay_not_found(self, handler):
        """Should return 404 when replay not found."""
        body = json.dumps({"event_id": "evt-1"}).encode()
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = BytesIO(body)

        handler._fork_replay("nonexistent")

        handler.send_error.assert_called()

    def test_fork_success(self, handler, tmp_path):
        """Should successfully fork a replay."""
        # Create replay files
        session_dir = tmp_path / "test-replay"
        session_dir.mkdir()
        (session_dir / "meta.json").write_text('{"topic": "Test"}')
        (session_dir / "events.jsonl").write_text(
            '{"event_id": "evt-1", "type": "start"}\n{"event_id": "evt-2", "type": "arg"}'
        )

        body = json.dumps({"event_id": "evt-1", "config": {}}).encode()
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = BytesIO(body)

        handler._fork_replay("test-replay")

        handler.send_response.assert_called_with(200)
        response = handler.wfile.getvalue()
        data = json.loads(response)
        assert "fork_id" in data
        assert data["parent_id"] == "test-replay"
        assert data["fork_point"] == "evt-1"
        assert len(data["events"]) == 1  # Only up to fork point

    def test_fork_no_storage(self, handler):
        """Should return 500 when replay storage not configured."""
        handler.replay_storage = None
        body = json.dumps({"event_id": "evt-1"}).encode()
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = BytesIO(body)

        handler._fork_replay("any-id")

        handler.send_error.assert_called_with(500, "Replay storage not configured")

    def test_fork_payload_too_large(self, handler):
        """Should return 413 for oversized payload."""
        handler.headers = {"Content-Length": str(60 * 1024 * 1024)}  # 60MB

        handler._fork_replay("test-replay")

        handler.send_error.assert_called_with(413, "Payload too large")

    def test_fork_invalid_content_length(self, handler):
        """Should return 400 for invalid Content-Length."""
        handler.headers = {"Content-Length": "not-a-number"}

        handler._fork_replay("test-replay")

        handler.send_error.assert_called_with(400, "Invalid Content-Length header")
