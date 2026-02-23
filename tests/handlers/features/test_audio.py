"""Tests for audio and podcast endpoint handler.

Tests the AudioHandler endpoints including:
- GET /audio/{id}.mp3 - Serve audio file
- GET /api/v1/podcast/feed.xml - iTunes-compatible RSS feed
- GET /api/v1/podcast/episodes - JSON episode listing

Also tests:
- Rate limiting (10 req/min)
- Debate ID validation (prevents path traversal)
- Path containment verification
- Private debate access control
- Podcast feed generation with MAX_PODCAST_EPISODES limit
- HTTPS scheme detection via X-Forwarded-Proto
"""

import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.features.audio import (
    MAX_PODCAST_EPISODES,
    AudioHandler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result: HandlerResult) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result: HandlerResult) -> dict[str, Any]:
    """Extract parsed JSON body from HandlerResult."""
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        return json.loads(raw.decode("utf-8"))
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler with headers and client_address."""

    headers: dict[str, str] = field(
        default_factory=lambda: {"Content-Length": "0", "Host": "localhost:8080"}
    )
    client_address: tuple = ("127.0.0.1", 12345)
    rfile: BytesIO = field(default_factory=lambda: BytesIO(b"{}"))


def _make_handler(
    client_ip: str = "127.0.0.1",
    host: str = "localhost:8080",
    proto: str | None = None,
) -> MockHTTPHandler:
    """Build a MockHTTPHandler with the given parameters."""
    headers: dict[str, str] = {"Content-Length": "0", "Host": host}
    if proto:
        headers["X-Forwarded-Proto"] = proto
    return MockHTTPHandler(headers=headers, client_address=(client_ip, 12345))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audio_store():
    """Create a mock audio store."""
    store = MagicMock()
    # Use a MagicMock for storage_dir so .resolve() can be controlled
    mock_storage_dir = MagicMock()
    mock_storage_dir.resolve.return_value = Path("/tmp/audio_store")
    # Also make str() work for path containment checks
    mock_storage_dir.__str__ = MagicMock(return_value="/tmp/audio_store")
    store.storage_dir = mock_storage_dir
    store.list_all.return_value = []
    return store


@pytest.fixture
def handler(audio_store):
    """Create an AudioHandler with mock context."""
    ctx = {"audio_store": audio_store}
    return AudioHandler(ctx=ctx)


@pytest.fixture
def handler_no_store():
    """Create an AudioHandler without audio store."""
    return AudioHandler(ctx={})


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter state between tests."""
    from aragora.server.handlers.features.audio import _audio_limiter

    _audio_limiter.clear()
    yield
    _audio_limiter.clear()


@pytest.fixture
def mock_http():
    """Create a default mock HTTP handler."""
    return _make_handler()


# ===========================================================================
# Constructor / Initialization Tests
# ===========================================================================


class TestAudioHandlerInit:
    """Tests for AudioHandler initialization."""

    def test_init_with_ctx(self):
        h = AudioHandler(ctx={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_init_with_server_context(self):
        h = AudioHandler(server_context={"key2": "val2"})
        assert h.ctx == {"key2": "val2"}

    def test_server_context_takes_precedence(self):
        h = AudioHandler(ctx={"a": 1}, server_context={"b": 2})
        assert h.ctx == {"b": 2}

    def test_defaults_to_empty_dict(self):
        h = AudioHandler()
        assert h.ctx == {}

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "audio"

    def test_routes_defined(self, handler):
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) > 0
        assert "/audio/*" in handler.ROUTES

    def test_routes_include_podcast(self, handler):
        assert "/api/v1/podcast/feed.xml" in handler.ROUTES
        assert "/api/v1/podcast/episodes" in handler.ROUTES


# ===========================================================================
# can_handle Tests
# ===========================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_mp3_audio_path(self, handler):
        assert handler.can_handle("/audio/test-123.mp3")

    def test_podcast_feed(self, handler):
        assert handler.can_handle("/api/v1/podcast/feed.xml")

    def test_podcast_episodes(self, handler):
        assert handler.can_handle("/api/v1/podcast/episodes")

    def test_rejects_non_mp3_audio(self, handler):
        assert not handler.can_handle("/audio/test-123.wav")

    def test_rejects_audio_without_extension(self, handler):
        assert not handler.can_handle("/audio/test-123")

    def test_rejects_non_audio_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_podcast_feed_without_xml(self, handler):
        assert not handler.can_handle("/api/v1/podcast/feed")

    def test_rejects_partial_audio_path(self, handler):
        assert not handler.can_handle("/audio/")

    def test_accepts_complex_debate_id(self, handler):
        assert handler.can_handle("/audio/abc-def-123_456.mp3")

    def test_rejects_other_api_paths(self, handler):
        assert not handler.can_handle("/api/v1/users")
        assert not handler.can_handle("/api/v1/debates/123")


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for audio endpoint rate limiting (10 req/min)."""

    def test_first_request_allowed(self, handler, mock_http):
        result = handler.handle("/audio/test-123.mp3", {}, mock_http)
        assert result is not None
        assert _status(result) != 429

    def test_under_rate_limit(self, handler, audio_store):
        """9 requests should all succeed."""
        audio_store.get_path.return_value = None
        for i in range(9):
            mock = _make_handler(client_ip="10.0.0.1")
            result = handler.handle(f"/audio/debate-{i}.mp3", {}, mock)
            assert result is not None
            assert _status(result) != 429, f"Request {i + 1} should not be rate limited"

    def test_exceeds_rate_limit(self, handler, audio_store):
        """11th request from same IP should be rate limited."""
        audio_store.get_path.return_value = None
        for i in range(10):
            mock = _make_handler(client_ip="10.0.0.2")
            handler.handle(f"/audio/debate-{i}.mp3", {}, mock)

        mock = _make_handler(client_ip="10.0.0.2")
        result = handler.handle("/audio/extra.mp3", {}, mock)
        assert _status(result) == 429
        body = _body(result)
        assert "Rate limit" in body["error"]

    def test_different_ips_independent(self, handler, audio_store):
        """Different IPs have independent rate limits."""
        audio_store.get_path.return_value = None
        for i in range(10):
            mock = _make_handler(client_ip="10.0.0.3")
            handler.handle(f"/audio/debate-{i}.mp3", {}, mock)

        mock = _make_handler(client_ip="10.0.0.4")
        result = handler.handle("/audio/test.mp3", {}, mock)
        assert _status(result) != 429

    def test_rate_limit_applies_to_podcast_feed(self, handler):
        """Rate limit also applies to podcast feed endpoint."""
        for i in range(10):
            mock = _make_handler(client_ip="10.0.0.5")
            handler.handle("/audio/debate-{}.mp3".format(i), {}, mock)

        mock = _make_handler(client_ip="10.0.0.5")
        result = handler.handle("/api/v1/podcast/feed.xml", {}, mock)
        assert _status(result) == 429

    def test_rate_limit_applies_to_podcast_episodes(self, handler):
        """Rate limit also applies to podcast episodes endpoint."""
        for i in range(10):
            mock = _make_handler(client_ip="10.0.0.6")
            handler.handle("/audio/debate-{}.mp3".format(i), {}, mock)

        mock = _make_handler(client_ip="10.0.0.6")
        result = handler.handle("/api/v1/podcast/episodes", {}, mock)
        assert _status(result) == 429

    def test_none_handler_ip(self, handler, audio_store):
        """None handler yields 'unknown' IP but still works."""
        audio_store.get_path.return_value = None
        result = handler.handle("/audio/test.mp3", {}, None)
        assert result is not None


# ===========================================================================
# Audio Serving - _serve_audio Tests
# ===========================================================================


class TestServeAudio:
    """Tests for audio file serving."""

    def test_no_audio_store(self, handler_no_store, mock_http):
        """Returns 404 when audio store not configured."""
        result = handler_no_store.handle("/audio/test-123.mp3", {}, mock_http)
        assert _status(result) == 404
        body = _body(result)
        assert "not configured" in body["error"]

    def test_invalid_debate_id_path_traversal(self, handler, mock_http):
        """Path traversal in debate ID returns 400."""
        result = handler.handle("/audio/../etc/passwd.mp3", {}, mock_http)
        assert _status(result) == 400

    def test_invalid_debate_id_special_chars(self, handler, mock_http):
        """Debate ID with special characters returns 400."""
        result = handler.handle("/audio/test<script>.mp3", {}, mock_http)
        assert _status(result) == 400

    def test_audio_not_found(self, handler, audio_store, mock_http):
        """Returns 404 when audio file does not exist."""
        audio_store.get_path.return_value = None
        result = handler.handle("/audio/valid-debate-123.mp3", {}, mock_http)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_audio_path_not_exists(self, handler, audio_store, mock_http):
        """Returns 404 when audio path object exists() returns False."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        audio_store.get_path.return_value = mock_path
        result = handler.handle("/audio/valid-debate-123.mp3", {}, mock_http)
        assert _status(result) == 404

    def test_successful_audio_serve(self, handler, audio_store, mock_http):
        """Successfully serves audio file with correct headers."""
        content = b"fake mp3 content bytes"
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.resolve.return_value = Path("/tmp/audio_store/valid-debate-123.mp3")
        mock_path.read_bytes.return_value = content
        audio_store.get_path.return_value = mock_path
        audio_store.storage_dir.resolve.return_value = Path("/tmp/audio_store")

        result = handler.handle("/audio/valid-debate-123.mp3", {}, mock_http)
        assert _status(result) == 200
        assert result.content_type == "audio/mpeg"
        assert result.body == content
        assert result.headers["Content-Length"] == str(len(content))
        assert result.headers["Accept-Ranges"] == "bytes"
        assert "max-age=86400" in result.headers["Cache-Control"]

    def test_path_containment_check(self, handler, audio_store, mock_http):
        """Returns 400 when resolved path is outside storage directory."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.resolve.return_value = Path("/etc/passwd")
        mock_path.read_bytes.return_value = b"content"
        audio_store.get_path.return_value = mock_path
        audio_store.storage_dir.resolve.return_value = Path("/tmp/audio_store")

        result = handler.handle("/audio/valid-debate-123.mp3", {}, mock_http)
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid path" in body["error"]

    def test_path_resolve_os_error(self, handler, audio_store, mock_http):
        """Returns 400 when path resolution raises OSError."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.resolve.side_effect = OSError("broken symlink")
        audio_store.get_path.return_value = mock_path

        result = handler.handle("/audio/valid-debate-123.mp3", {}, mock_http)
        assert _status(result) == 400
        body = _body(result)
        assert "validation failed" in body["error"].lower()

    def test_path_resolve_value_error(self, handler, audio_store, mock_http):
        """Returns 400 when path resolution raises ValueError."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.resolve.side_effect = ValueError("invalid path")
        audio_store.get_path.return_value = mock_path

        result = handler.handle("/audio/valid-debate-123.mp3", {}, mock_http)
        assert _status(result) == 400

    def test_read_bytes_os_error(self, handler, audio_store, mock_http):
        """Returns 500 when reading audio file raises OSError."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.resolve.return_value = Path("/tmp/audio_store/test.mp3")
        mock_path.read_bytes.side_effect = OSError("disk read error")
        audio_store.get_path.return_value = mock_path
        audio_store.storage_dir.resolve.return_value = Path("/tmp/audio_store")

        result = handler.handle("/audio/valid-debate-123.mp3", {}, mock_http)
        assert _status(result) == 500
        body = _body(result)
        assert "read audio" in body["error"].lower()

    def test_read_bytes_value_error(self, handler, audio_store, mock_http):
        """Returns 500 when reading audio file raises ValueError."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.resolve.return_value = Path("/tmp/audio_store/test.mp3")
        mock_path.read_bytes.side_effect = ValueError("bad encoding")
        audio_store.get_path.return_value = mock_path
        audio_store.storage_dir.resolve.return_value = Path("/tmp/audio_store")

        result = handler.handle("/audio/valid-debate-123.mp3", {}, mock_http)
        assert _status(result) == 500

    def test_debate_id_extraction(self, handler, audio_store, mock_http):
        """Correctly extracts debate ID from /audio/{id}.mp3 path."""
        audio_store.get_path.return_value = None
        handler.handle("/audio/my-debate-id.mp3", {}, mock_http)
        audio_store.get_path.assert_called_with("my-debate-id")


# ===========================================================================
# Debate Access Control Tests
# ===========================================================================


class TestDebateAccessControl:
    """Tests for _check_debate_access (private debate gating)."""

    def test_no_storage_allows_access(self, handler, mock_http):
        """When storage is unavailable, access is granted."""
        with patch.object(handler, "get_storage", return_value=None):
            result = handler._check_debate_access("test-id", mock_http)
            assert result is None  # None = access granted

    def test_debate_not_found_allows_access(self, handler, mock_http):
        """When debate not found, access is granted (let serve handle 404)."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = None
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._check_debate_access("test-id", mock_http)
            assert result is None

    def test_public_debate_allows_access(self, handler, mock_http):
        """Public debates don't require auth."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"is_public": True}
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._check_debate_access("test-id", mock_http)
            assert result is None

    def test_default_public_debate(self, handler, mock_http):
        """Debates without is_public field default to public."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"id": "test-id"}
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._check_debate_access("test-id", mock_http)
            assert result is None

    def test_private_debate_auth_error(self, handler, mock_http):
        """Private debate returns error when auth fails."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"is_public": False}
        with patch.object(handler, "get_storage", return_value=mock_storage):
            with patch.object(
                handler,
                "require_auth_or_error",
                return_value=(
                    None,
                    HandlerResult(
                        status_code=401,
                        content_type="application/json",
                        body=b'{"error":"Unauthorized"}',
                    ),
                ),
            ):
                result = handler._check_debate_access("test-id", mock_http)
                assert result is not None
                assert _status(result) == 401

    def test_private_debate_owner_access(self, handler, mock_http):
        """Owner of a private debate gets access."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {
            "is_public": False,
            "owner_id": "test-user-001",
        }

        mock_user = MagicMock()
        mock_user.user_id = "test-user-001"
        mock_user.org_id = "test-org"
        mock_user.roles = {"member"}

        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker.check_permission.return_value = mock_decision

        with patch.object(handler, "get_storage", return_value=mock_storage):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.rbac.checker.get_permission_checker",
                    return_value=mock_checker,
                ):
                    result = handler._check_debate_access("test-id", mock_http)
                    assert result is None  # Access granted

    def test_private_debate_non_owner_denied(self, handler, mock_http):
        """Non-owner, non-admin of a private debate is denied."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {
            "is_public": False,
            "owner_id": "other-user",
        }

        mock_user = MagicMock()
        mock_user.user_id = "test-user-001"
        mock_user.org_id = "test-org"
        mock_user.roles = {"member"}

        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker.check_permission.return_value = mock_decision

        with patch.object(handler, "get_storage", return_value=mock_storage):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.rbac.checker.get_permission_checker",
                    return_value=mock_checker,
                ):
                    result = handler._check_debate_access("test-id", mock_http)
                    assert result is not None
                    assert _status(result) == 403
                    body = _body(result)
                    assert "private debate" in body["error"].lower()

    def test_private_debate_admin_access(self, handler, mock_http):
        """Admin gets access to any private debate."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {
            "is_public": False,
            "owner_id": "other-user",
        }

        mock_user = MagicMock()
        mock_user.user_id = "test-user-001"
        mock_user.org_id = "test-org"
        mock_user.roles = {"admin", "member"}

        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker.check_permission.return_value = mock_decision

        with patch.object(handler, "get_storage", return_value=mock_storage):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.rbac.checker.get_permission_checker",
                    return_value=mock_checker,
                ):
                    result = handler._check_debate_access("test-id", mock_http)
                    assert result is None  # Admin gets access

    def test_private_debate_permission_denied(self, handler, mock_http):
        """debates.read permission denied returns 403."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"is_public": False}

        mock_user = MagicMock()
        mock_user.user_id = "test-user-001"
        mock_user.org_id = "test-org"
        mock_user.roles = {"member"}

        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_checker.check_permission.return_value = mock_decision

        with patch.object(handler, "get_storage", return_value=mock_storage):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.rbac.checker.get_permission_checker",
                    return_value=mock_checker,
                ):
                    result = handler._check_debate_access("test-id", mock_http)
                    assert result is not None
                    assert _status(result) == 403
                    body = _body(result)
                    assert "Permission denied" in body["error"]

    def test_auth_import_error_returns_401(self, handler, mock_http):
        """ImportError during auth check returns 401."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"is_public": False}

        mock_user = MagicMock()
        mock_user.user_id = "test-user-001"
        mock_user.org_id = None
        mock_user.roles = {"member"}

        with patch.object(handler, "get_storage", return_value=mock_storage):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.rbac.checker.get_permission_checker",
                    side_effect=ImportError("rbac not available"),
                ):
                    result = handler._check_debate_access("test-id", mock_http)
                    assert result is not None
                    assert _status(result) == 401

    def test_debate_with_user_id_field(self, handler, mock_http):
        """Access check handles 'user_id' field in debate instead of 'owner_id'."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {
            "is_public": False,
            "user_id": "test-user-001",
        }

        mock_user = MagicMock()
        mock_user.user_id = "test-user-001"
        mock_user.org_id = "test-org"
        mock_user.roles = {"member"}

        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker.check_permission.return_value = mock_decision

        with patch.object(handler, "get_storage", return_value=mock_storage):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.rbac.checker.get_permission_checker",
                    return_value=mock_checker,
                ):
                    result = handler._check_debate_access("test-id", mock_http)
                    assert result is None  # User is owner via user_id field


# ===========================================================================
# Podcast Feed Tests
# ===========================================================================


class TestPodcastFeed:
    """Tests for GET /api/v1/podcast/feed.xml."""

    def test_no_audio_store_returns_503(self, handler_no_store, mock_http):
        result = handler_no_store.handle("/api/v1/podcast/feed.xml", {}, mock_http)
        assert _status(result) == 503
        body = _body(result)
        assert "not configured" in body["error"]

    def test_podcast_not_available(self, handler, mock_http):
        """Returns 503 when podcast module is not available."""
        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", False):
            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
            assert _status(result) == 503
            body = _body(result)
            assert "not available" in body["error"]

    def test_empty_feed(self, handler, audio_store, mock_http):
        """Returns RSS feed with no episodes when no audio exists."""
        audio_store.list_all.return_value = []

        mock_config = MagicMock()
        mock_generator = MagicMock()
        mock_generator.generate_feed.return_value = '<?xml version="1.0"?><rss></rss>'

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.audio.PodcastConfig",
                return_value=mock_config,
            ):
                with patch(
                    "aragora.server.handlers.features.audio.PodcastFeedGenerator",
                    return_value=mock_generator,
                ):
                    with patch(
                        "aragora.server.handlers.features.audio.PodcastEpisode"
                    ) as MockEpisode:
                        with patch.object(handler, "get_storage", return_value=MagicMock()):
                            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                            assert _status(result) == 200
                            assert "rss+xml" in result.content_type
                            mock_generator.generate_feed.assert_called_once_with([])

    def test_feed_with_episodes(self, handler, audio_store, mock_http):
        """Returns RSS feed with episodes."""
        audio_store.list_all.return_value = [
            {"debate_id": "d1", "duration_seconds": 120, "file_size_bytes": 5000},
            {"debate_id": "d2", "duration_seconds": 90, "file_size_bytes": 3000},
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.side_effect = lambda did: {
            "d1": {"task": "Debate One", "agents": ["a", "b"], "verdict": "A wins"},
            "d2": {"task": "Debate Two", "agents": ["c", "d"]},
        }.get(did)

        mock_audio_path = MagicMock()
        mock_audio_path.exists.return_value = True
        audio_store.get_path.return_value = mock_audio_path

        mock_generator = MagicMock()
        mock_generator.generate_feed.return_value = "<rss>feed</rss>"

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.audio.PodcastConfig",
                return_value=MagicMock(),
            ):
                with patch(
                    "aragora.server.handlers.features.audio.PodcastFeedGenerator",
                    return_value=mock_generator,
                ):
                    with patch(
                        "aragora.server.handlers.features.audio.PodcastEpisode"
                    ) as MockEpisode:
                        MockEpisode.side_effect = lambda **kwargs: MagicMock(**kwargs)
                        with patch.object(handler, "get_storage", return_value=mock_storage):
                            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                            assert _status(result) == 200
                            # Two episodes created
                            assert mock_generator.generate_feed.call_count == 1
                            episodes_arg = mock_generator.generate_feed.call_args[0][0]
                            assert len(episodes_arg) == 2

    def test_feed_skips_missing_debate_id(self, handler, audio_store, mock_http):
        """Feed skips entries without debate_id."""
        audio_store.list_all.return_value = [
            {"duration_seconds": 120},  # No debate_id
            {"debate_id": "d1", "duration_seconds": 90},
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}
        mock_audio_path = MagicMock()
        mock_audio_path.exists.return_value = True
        audio_store.get_path.return_value = mock_audio_path

        mock_generator = MagicMock()
        mock_generator.generate_feed.return_value = "<rss/>"

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.audio.PodcastConfig",
                return_value=MagicMock(),
            ):
                with patch(
                    "aragora.server.handlers.features.audio.PodcastFeedGenerator",
                    return_value=mock_generator,
                ):
                    with patch(
                        "aragora.server.handlers.features.audio.PodcastEpisode"
                    ) as MockEpisode:
                        MockEpisode.side_effect = lambda **kwargs: MagicMock(**kwargs)
                        with patch.object(handler, "get_storage", return_value=mock_storage):
                            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                            assert _status(result) == 200
                            episodes_arg = mock_generator.generate_feed.call_args[0][0]
                            assert len(episodes_arg) == 1

    def test_feed_skips_missing_debate(self, handler, audio_store, mock_http):
        """Feed skips entries where debate is not found in storage."""
        audio_store.list_all.return_value = [
            {"debate_id": "gone"},
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = None

        mock_generator = MagicMock()
        mock_generator.generate_feed.return_value = "<rss/>"

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.audio.PodcastConfig",
                return_value=MagicMock(),
            ):
                with patch(
                    "aragora.server.handlers.features.audio.PodcastFeedGenerator",
                    return_value=mock_generator,
                ):
                    with patch("aragora.server.handlers.features.audio.PodcastEpisode"):
                        with patch.object(handler, "get_storage", return_value=mock_storage):
                            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                            assert _status(result) == 200
                            episodes_arg = mock_generator.generate_feed.call_args[0][0]
                            assert len(episodes_arg) == 0

    def test_feed_skips_missing_audio_path(self, handler, audio_store, mock_http):
        """Feed skips entries where audio file does not exist."""
        audio_store.list_all.return_value = [
            {"debate_id": "d1"},
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}
        audio_store.get_path.return_value = None

        mock_generator = MagicMock()
        mock_generator.generate_feed.return_value = "<rss/>"

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.audio.PodcastConfig",
                return_value=MagicMock(),
            ):
                with patch(
                    "aragora.server.handlers.features.audio.PodcastFeedGenerator",
                    return_value=mock_generator,
                ):
                    with patch("aragora.server.handlers.features.audio.PodcastEpisode"):
                        with patch.object(handler, "get_storage", return_value=mock_storage):
                            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                            assert _status(result) == 200
                            episodes_arg = mock_generator.generate_feed.call_args[0][0]
                            assert len(episodes_arg) == 0

    def test_feed_max_episodes_limit(self, handler, audio_store, mock_http):
        """Feed is capped at MAX_PODCAST_EPISODES."""
        entries = [
            {"debate_id": f"d{i}", "duration_seconds": 60} for i in range(MAX_PODCAST_EPISODES + 50)
        ]
        audio_store.list_all.return_value = entries

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}
        mock_audio_path = MagicMock()
        mock_audio_path.exists.return_value = True
        audio_store.get_path.return_value = mock_audio_path

        mock_generator = MagicMock()
        mock_generator.generate_feed.return_value = "<rss/>"

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.audio.PodcastConfig",
                return_value=MagicMock(),
            ):
                with patch(
                    "aragora.server.handlers.features.audio.PodcastFeedGenerator",
                    return_value=mock_generator,
                ):
                    with patch(
                        "aragora.server.handlers.features.audio.PodcastEpisode"
                    ) as MockEpisode:
                        MockEpisode.side_effect = lambda **kwargs: MagicMock(**kwargs)
                        with patch.object(handler, "get_storage", return_value=mock_storage):
                            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                            assert _status(result) == 200
                            episodes_arg = mock_generator.generate_feed.call_args[0][0]
                            assert len(episodes_arg) == MAX_PODCAST_EPISODES

    def test_feed_https_scheme(self, handler, audio_store):
        """Feed uses https scheme when X-Forwarded-Proto is https."""
        mock_http = _make_handler(proto="https", host="example.com")

        audio_store.list_all.return_value = [
            {"debate_id": "d1", "duration_seconds": 60, "file_size_bytes": 1000},
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test", "created_at": "2024-01-01"}
        mock_audio_path = MagicMock()
        mock_audio_path.exists.return_value = True
        audio_store.get_path.return_value = mock_audio_path

        mock_generator = MagicMock()
        mock_generator.generate_feed.return_value = "<rss/>"

        captured_episodes = []

        def capture_episode(**kwargs):
            captured_episodes.append(kwargs)
            return MagicMock(**kwargs)

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.audio.PodcastConfig",
                return_value=MagicMock(),
            ):
                with patch(
                    "aragora.server.handlers.features.audio.PodcastFeedGenerator",
                    return_value=mock_generator,
                ):
                    with patch(
                        "aragora.server.handlers.features.audio.PodcastEpisode",
                        side_effect=capture_episode,
                    ):
                        with patch.object(handler, "get_storage", return_value=mock_storage):
                            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                            assert _status(result) == 200
                            assert len(captured_episodes) == 1
                            assert captured_episodes[0]["audio_url"].startswith("https://")

    def test_feed_http_scheme_default(self, handler, audio_store, mock_http):
        """Feed uses http scheme by default (no X-Forwarded-Proto)."""
        audio_store.list_all.return_value = [
            {"debate_id": "d1", "duration_seconds": 60, "file_size_bytes": 1000},
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test", "created_at": "2024-01-01"}
        mock_audio_path = MagicMock()
        mock_audio_path.exists.return_value = True
        audio_store.get_path.return_value = mock_audio_path

        mock_generator = MagicMock()
        mock_generator.generate_feed.return_value = "<rss/>"

        captured_episodes = []

        def capture_episode(**kwargs):
            captured_episodes.append(kwargs)
            return MagicMock(**kwargs)

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.audio.PodcastConfig",
                return_value=MagicMock(),
            ):
                with patch(
                    "aragora.server.handlers.features.audio.PodcastFeedGenerator",
                    return_value=mock_generator,
                ):
                    with patch(
                        "aragora.server.handlers.features.audio.PodcastEpisode",
                        side_effect=capture_episode,
                    ):
                        with patch.object(handler, "get_storage", return_value=mock_storage):
                            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                            assert _status(result) == 200
                            assert len(captured_episodes) == 1
                            assert captured_episodes[0]["audio_url"].startswith("http://")

    def test_feed_cache_control_header(self, handler, audio_store, mock_http):
        """Feed response has correct Cache-Control header."""
        audio_store.list_all.return_value = []
        mock_generator = MagicMock()
        mock_generator.generate_feed.return_value = "<rss/>"

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.audio.PodcastConfig",
                return_value=MagicMock(),
            ):
                with patch(
                    "aragora.server.handlers.features.audio.PodcastFeedGenerator",
                    return_value=mock_generator,
                ):
                    with patch("aragora.server.handlers.features.audio.PodcastEpisode"):
                        with patch.object(handler, "get_storage", return_value=MagicMock()):
                            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                            assert _status(result) == 200
                            assert "max-age=300" in result.headers["Cache-Control"]

    def test_feed_runtime_error(self, handler, audio_store, mock_http):
        """Returns 500 when feed generation raises RuntimeError."""
        audio_store.list_all.side_effect = RuntimeError("storage error")

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch.object(handler, "get_storage", return_value=MagicMock()):
                result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                assert _status(result) == 500

    def test_feed_type_error(self, handler, audio_store, mock_http):
        """Returns 500 when feed generation raises TypeError."""
        audio_store.list_all.side_effect = TypeError("bad type")

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch.object(handler, "get_storage", return_value=MagicMock()):
                result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                assert _status(result) == 500

    def test_podcast_config_none_despite_available(self, handler, audio_store, mock_http):
        """Returns 500 if PodcastConfig is None despite PODCAST_AVAILABLE=True."""
        audio_store.list_all.return_value = []

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch("aragora.server.handlers.features.audio.PodcastConfig", None):
                with patch.object(handler, "get_storage", return_value=MagicMock()):
                    result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                    assert _status(result) == 500


# ===========================================================================
# Podcast Episodes Tests
# ===========================================================================


class TestPodcastEpisodes:
    """Tests for GET /api/v1/podcast/episodes."""

    def test_no_audio_store_returns_503(self, handler_no_store, mock_http):
        result = handler_no_store.handle("/api/v1/podcast/episodes", {}, mock_http)
        assert _status(result) == 503
        body = _body(result)
        assert "not configured" in body["error"]

    def test_empty_episodes(self, handler, audio_store, mock_http):
        """Returns empty episodes list."""
        audio_store.list_all.return_value = []
        with patch.object(handler, "get_storage", return_value=MagicMock()):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["episodes"] == []
            assert body["count"] == 0
            assert body["feed_url"] == "/api/v1/podcast/feed.xml"

    def test_episodes_with_data(self, handler, audio_store, mock_http):
        """Returns episodes with debate data."""
        audio_store.list_all.return_value = [
            {
                "debate_id": "d1",
                "duration_seconds": 120,
                "file_size_bytes": 5000,
                "generated_at": "2024-01-01",
            },
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {
            "task": "Test Debate",
            "agents": ["agent1", "agent2"],
        }

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 1
            ep = body["episodes"][0]
            assert ep["debate_id"] == "d1"
            assert ep["task"] == "Test Debate"
            assert ep["agents"] == ["agent1", "agent2"]
            assert ep["duration_seconds"] == 120
            assert ep["file_size_bytes"] == 5000
            assert ep["generated_at"] == "2024-01-01"
            assert ".mp3" in ep["audio_url"]

    def test_episodes_without_debate_info(self, handler, audio_store, mock_http):
        """Returns 'Unknown' for episodes without debate in storage."""
        audio_store.list_all.return_value = [
            {"debate_id": "missing"},
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = None

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            ep = body["episodes"][0]
            assert ep["task"] == "Unknown"
            assert ep["agents"] == []

    def test_episodes_limit_param(self, handler, audio_store, mock_http):
        """Respects limit query parameter."""
        audio_store.list_all.return_value = [{"debate_id": f"d{i}"} for i in range(20)]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {"limit": "5"}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 5

    def test_episodes_default_limit(self, handler, audio_store, mock_http):
        """Default limit is 50."""
        audio_store.list_all.return_value = [{"debate_id": f"d{i}"} for i in range(100)]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 50

    def test_episodes_skips_no_debate_id(self, handler, audio_store, mock_http):
        """Skips entries without debate_id."""
        audio_store.list_all.return_value = [
            {"duration_seconds": 120},  # No debate_id
            {"debate_id": "d1"},
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 1

    def test_episodes_https_url(self, handler, audio_store):
        """Episode audio URLs use https when forwarded."""
        mock_http = _make_handler(proto="https", host="podcast.example.com")
        audio_store.list_all.return_value = [
            {"debate_id": "d1"},
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            ep = body["episodes"][0]
            assert ep["audio_url"].startswith("https://podcast.example.com/audio/")

    def test_episodes_http_url_default(self, handler, audio_store, mock_http):
        """Episode audio URLs use http by default."""
        audio_store.list_all.return_value = [
            {"debate_id": "d1"},
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            ep = body["episodes"][0]
            assert ep["audio_url"].startswith("http://")

    def test_episodes_runtime_error(self, handler, audio_store, mock_http):
        """Returns 500 when episodes retrieval raises RuntimeError."""
        audio_store.list_all.side_effect = RuntimeError("storage error")

        with patch.object(handler, "get_storage", return_value=MagicMock()):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 500

    def test_episodes_attribute_error(self, handler, audio_store, mock_http):
        """Returns 500 when episodes retrieval raises AttributeError."""
        audio_store.list_all.side_effect = AttributeError("missing attr")

        with patch.object(handler, "get_storage", return_value=MagicMock()):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 500

    def test_episodes_no_storage(self, handler, audio_store, mock_http):
        """Episodes work even when storage is None (uses None for debate)."""
        audio_store.list_all.return_value = [
            {"debate_id": "d1"},
        ]

        with patch.object(handler, "get_storage", return_value=None):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            ep = body["episodes"][0]
            assert ep["task"] == "Unknown"


# ===========================================================================
# Handle Method Routing Tests
# ===========================================================================


class TestHandleRouting:
    """Tests for handle() GET request routing."""

    def test_routes_to_audio_serve(self, handler, audio_store, mock_http):
        """Routes /audio/{id}.mp3 to _serve_audio."""
        audio_store.get_path.return_value = None
        result = handler.handle("/audio/test-123.mp3", {}, mock_http)
        assert result is not None
        assert _status(result) == 404  # No audio found

    def test_routes_to_podcast_feed(self, handler, audio_store, mock_http):
        """Routes /api/v1/podcast/feed.xml to _get_podcast_feed."""
        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", False):
            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
            assert result is not None
            assert _status(result) == 503

    def test_routes_to_podcast_episodes(self, handler, audio_store, mock_http):
        """Routes /api/v1/podcast/episodes to _get_podcast_episodes."""
        audio_store.list_all.return_value = []
        with patch.object(handler, "get_storage", return_value=MagicMock()):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert result is not None
            assert _status(result) == 200

    def test_unmatched_path_returns_none(self, handler, mock_http):
        """Returns None for unmatched paths."""
        result = handler.handle("/api/v1/other", {}, mock_http)
        assert result is None

    def test_non_mp3_audio_returns_none(self, handler, mock_http):
        """Returns None for /audio/*.wav paths."""
        result = handler.handle("/audio/test.wav", {}, mock_http)
        assert result is None

    def test_podcast_feed_no_xml_returns_none(self, handler, mock_http):
        """Returns None for /api/v1/podcast/feed (no .xml)."""
        result = handler.handle("/api/v1/podcast/feed", {}, mock_http)
        assert result is None


# ===========================================================================
# Security Tests
# ===========================================================================


class TestSecurity:
    """Security-focused tests for audio handler."""

    def test_path_traversal_dotdot(self, handler, mock_http):
        """Path traversal with .. in debate ID is rejected."""
        result = handler.handle("/audio/../../etc/passwd.mp3", {}, mock_http)
        assert result is not None
        assert _status(result) == 400

    def test_path_traversal_encoded(self, handler, mock_http):
        """URL-encoded path traversal characters rejected."""
        result = handler.handle("/audio/%2e%2e%2f%2e%2e%2fetc%2fpasswd.mp3", {}, mock_http)
        assert result is not None
        # May get 400 from validation or another error
        assert _status(result) in (400, 404)

    def test_null_byte_injection(self, handler, mock_http):
        """Null byte in debate ID is rejected."""
        result = handler.handle("/audio/test\x00evil.mp3", {}, mock_http)
        assert result is not None
        assert _status(result) == 400

    def test_sql_injection(self, handler, mock_http):
        """SQL injection in debate ID is rejected."""
        result = handler.handle("/audio/'; DROP TABLE debates; --.mp3", {}, mock_http)
        assert result is not None
        assert _status(result) == 400

    def test_script_injection(self, handler, mock_http):
        """Script injection in debate ID is rejected."""
        result = handler.handle("/audio/<script>alert(1)</script>.mp3", {}, mock_http)
        assert result is not None
        assert _status(result) == 400

    def test_very_long_debate_id(self, handler, mock_http):
        """Very long debate ID is rejected."""
        long_id = "a" * 500
        result = handler.handle(f"/audio/{long_id}.mp3", {}, mock_http)
        assert result is not None
        assert _status(result) == 400

    def test_empty_debate_id(self, handler, mock_http):
        """Empty debate ID (just .mp3) is rejected."""
        result = handler.handle("/audio/.mp3", {}, mock_http)
        assert result is not None
        assert _status(result) == 400

    def test_whitespace_debate_id(self, handler, mock_http):
        """Whitespace-only debate ID is rejected."""
        result = handler.handle("/audio/   .mp3", {}, mock_http)
        assert result is not None
        assert _status(result) == 400

    def test_path_containment_prevents_symlink_escape(self, handler, audio_store, mock_http):
        """Audio path resolved outside storage dir is blocked."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.resolve.return_value = Path("/var/secrets/key.pem")
        audio_store.get_path.return_value = mock_path

        result = handler.handle("/audio/valid-debate-id.mp3", {}, mock_http)
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid path" in body["error"]


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_max_podcast_episodes(self):
        assert MAX_PODCAST_EPISODES == 200

    def test_podcast_available_is_bool(self):
        from aragora.server.handlers.features.audio import PODCAST_AVAILABLE

        assert isinstance(PODCAST_AVAILABLE, bool)


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_handle_with_none_handler(self, handler, audio_store):
        """Handle works with None handler (rate limit returns 'unknown' IP)."""
        audio_store.get_path.return_value = None
        result = handler.handle("/audio/test.mp3", {}, None)
        assert result is not None

    def test_debate_id_with_hyphens_and_underscores(self, handler, audio_store, mock_http):
        """Valid debate IDs with hyphens and underscores."""
        audio_store.get_path.return_value = None
        result = handler.handle("/audio/my-debate_id-123.mp3", {}, mock_http)
        assert result is not None
        # Should pass validation, get 404 since file not found
        assert _status(result) == 404

    def test_podcast_episodes_with_invalid_limit(self, handler, audio_store, mock_http):
        """Non-integer limit falls back to default."""
        audio_store.list_all.return_value = []
        with patch.object(handler, "get_storage", return_value=MagicMock()):
            result = handler.handle("/api/v1/podcast/episodes", {"limit": "invalid"}, mock_http)
            assert _status(result) == 200

    def test_podcast_episodes_with_zero_limit(self, handler, audio_store, mock_http):
        """Limit of 0 returns 0 episodes."""
        audio_store.list_all.return_value = [{"debate_id": "d1"}]
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {"limit": "0"}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 0

    def test_podcast_episodes_with_negative_limit(self, handler, audio_store, mock_http):
        """Negative limit returns empty (Python slice handles it)."""
        audio_store.list_all.return_value = [{"debate_id": "d1"}]
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {"limit": "-1"}, mock_http)
            assert _status(result) == 200

    def test_audio_serve_empty_content(self, handler, audio_store, mock_http):
        """Serving empty audio file works."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.resolve.return_value = Path("/tmp/audio_store/empty.mp3")
        mock_path.read_bytes.return_value = b""
        audio_store.get_path.return_value = mock_path
        audio_store.storage_dir.resolve.return_value = Path("/tmp/audio_store")

        result = handler.handle("/audio/valid-debate-123.mp3", {}, mock_http)
        assert _status(result) == 200
        assert result.body == b""
        assert result.headers["Content-Length"] == "0"

    def test_multiple_podcast_episodes_order(self, handler, audio_store, mock_http):
        """Episodes maintain insertion order from audio store."""
        audio_store.list_all.return_value = [
            {"debate_id": "first"},
            {"debate_id": "second"},
            {"debate_id": "third"},
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.side_effect = lambda did: {"task": f"Task-{did}"}

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 3
            assert body["episodes"][0]["debate_id"] == "first"
            assert body["episodes"][1]["debate_id"] == "second"
            assert body["episodes"][2]["debate_id"] == "third"

    def test_large_audio_file_content_length(self, handler, audio_store, mock_http):
        """Content-Length header correctly set for large files."""
        content = b"x" * 100_000
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.resolve.return_value = Path("/tmp/audio_store/big.mp3")
        mock_path.read_bytes.return_value = content
        audio_store.get_path.return_value = mock_path
        audio_store.storage_dir.resolve.return_value = Path("/tmp/audio_store")

        result = handler.handle("/audio/valid-debate-123.mp3", {}, mock_http)
        assert _status(result) == 200
        assert result.headers["Content-Length"] == "100000"

    def test_podcast_episodes_feed_url_always_present(self, handler, audio_store, mock_http):
        """Feed URL is always present in episodes response."""
        audio_store.list_all.return_value = [{"debate_id": "d1"}]
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            body = _body(result)
            assert body["feed_url"] == "/api/v1/podcast/feed.xml"

    def test_podcast_episodes_audio_url_format(self, handler, audio_store, mock_http):
        """Audio URL in episodes has correct format /audio/{id}.mp3."""
        audio_store.list_all.return_value = [
            {"debate_id": "debate-abc-123"},
        ]
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test"}

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            body = _body(result)
            assert body["episodes"][0]["audio_url"].endswith("/audio/debate-abc-123.mp3")

    def test_feed_episode_number_descending(self, handler, audio_store):
        """Feed episodes have descending episode numbers."""
        mock_http = _make_handler()
        audio_store.list_all.return_value = [
            {"debate_id": f"d{i}", "duration_seconds": 60, "file_size_bytes": 1000}
            for i in range(3)
        ]

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"task": "Test", "created_at": "2024-01-01"}
        mock_audio_path = MagicMock()
        mock_audio_path.exists.return_value = True
        audio_store.get_path.return_value = mock_audio_path

        mock_generator = MagicMock()
        mock_generator.generate_feed.return_value = "<rss/>"

        captured_episodes = []

        def capture_episode(**kwargs):
            captured_episodes.append(kwargs)
            return MagicMock(**kwargs)

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.audio.PodcastConfig",
                return_value=MagicMock(),
            ):
                with patch(
                    "aragora.server.handlers.features.audio.PodcastFeedGenerator",
                    return_value=mock_generator,
                ):
                    with patch(
                        "aragora.server.handlers.features.audio.PodcastEpisode",
                        side_effect=capture_episode,
                    ):
                        with patch.object(handler, "get_storage", return_value=mock_storage):
                            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                            assert _status(result) == 200
                            # Episode numbers should be descending: 3, 2, 1
                            assert captured_episodes[0]["episode_number"] == 3
                            assert captured_episodes[1]["episode_number"] == 2
                            assert captured_episodes[2]["episode_number"] == 1

    def test_feed_debate_defaults(self, handler, audio_store):
        """Feed uses defaults for missing debate fields."""
        mock_http = _make_handler()
        audio_store.list_all.return_value = [
            {
                "debate_id": "d1",
                "duration_seconds": 60,
                "file_size_bytes": 1000,
                "generated_at": "2024-06-01",
            },
        ]

        mock_storage = MagicMock()
        # Debate missing task and agents fields but has id (non-empty so it's truthy)
        mock_storage.get_debate.return_value = {"id": "d1"}
        mock_audio_path = MagicMock()
        mock_audio_path.exists.return_value = True
        audio_store.get_path.return_value = mock_audio_path

        mock_generator = MagicMock()
        mock_generator.generate_feed.return_value = "<rss/>"

        captured_episodes = []

        def capture_episode(**kwargs):
            captured_episodes.append(kwargs)
            return MagicMock(**kwargs)

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.features.audio.PodcastConfig",
                return_value=MagicMock(),
            ):
                with patch(
                    "aragora.server.handlers.features.audio.PodcastFeedGenerator",
                    return_value=mock_generator,
                ):
                    with patch(
                        "aragora.server.handlers.features.audio.PodcastEpisode",
                        side_effect=capture_episode,
                    ):
                        with patch.object(handler, "get_storage", return_value=mock_storage):
                            result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                            assert _status(result) == 200
                            assert captured_episodes[0]["title"] == "Untitled Debate"

    def test_debate_access_no_owner_field(self, handler, mock_http):
        """Private debate without owner_id/user_id allows access for authenticated user."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {
            "is_public": False,
            # No owner_id or user_id
        }

        mock_user = MagicMock()
        mock_user.user_id = "test-user-001"
        mock_user.org_id = "test-org"
        mock_user.roles = {"member"}

        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker.check_permission.return_value = mock_decision

        with patch.object(handler, "get_storage", return_value=mock_storage):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.rbac.checker.get_permission_checker",
                    return_value=mock_checker,
                ):
                    result = handler._check_debate_access("test-id", mock_http)
                    # debate_owner is None, so ownership check is skipped
                    assert result is None

    def test_episodes_value_error(self, handler, audio_store, mock_http):
        """Returns 500 when episodes retrieval raises ValueError."""
        audio_store.list_all.side_effect = ValueError("bad value")

        with patch.object(handler, "get_storage", return_value=MagicMock()):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 500

    def test_feed_os_error(self, handler, audio_store, mock_http):
        """Returns 500 when feed generation raises OSError."""
        audio_store.list_all.side_effect = OSError("filesystem error")

        with patch("aragora.server.handlers.features.audio.PODCAST_AVAILABLE", True):
            with patch.object(handler, "get_storage", return_value=MagicMock()):
                result = handler.handle("/api/v1/podcast/feed.xml", {}, mock_http)
                assert _status(result) == 500

    def test_episodes_key_error(self, handler, audio_store, mock_http):
        """Returns 500 when episodes retrieval raises KeyError."""
        audio_store.list_all.side_effect = KeyError("missing key")

        with patch.object(handler, "get_storage", return_value=MagicMock()):
            result = handler.handle("/api/v1/podcast/episodes", {}, mock_http)
            assert _status(result) == 500
