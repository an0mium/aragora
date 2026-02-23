"""
Tests for social media handler (aragora/server/handlers/social/social_media.py).

Covers all routes and behavior of the SocialMediaHandler class:
- can_handle() route matching
- GET  /api/v1/youtube/auth     - YouTube OAuth authorization URL
- GET  /api/v1/youtube/callback - YouTube OAuth callback
- GET  /api/v1/youtube/status   - YouTube connector status
- POST /api/v1/debates/{id}/publish/twitter  - Publish to Twitter/X
- POST /api/v1/debates/{id}/publish/youtube  - Publish to YouTube
- OAuth state management (_store_oauth_state, _validate_oauth_state)
- Error handling and edge cases
- Method not allowed (POST on GET routes, GET on POST routes)
- Connector not configured / not available scenarios
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Lazy import so conftest auto-auth patches run first
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_module():
    """Import the handler module lazily (after conftest patches)."""
    import aragora.server.handlers.social.social_media as mod

    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.SocialMediaHandler


# ---------------------------------------------------------------------------
# Mock connectors and stores
# ---------------------------------------------------------------------------


def _make_youtube_connector(
    *,
    configured: bool = True,
    client_id: str = "yt-client-id",
    client_secret: str = "yt-client-secret",
    refresh_token: str = "yt-refresh-token",
    remaining_quota: int = 9000,
    can_upload: bool = True,
    circuit_breaker_open: bool = False,
) -> MagicMock:
    yt = MagicMock()
    yt.is_configured = configured
    yt.client_id = client_id
    yt.client_secret = client_secret
    yt.refresh_token = refresh_token
    yt.rate_limiter = MagicMock()
    yt.rate_limiter.remaining_quota = remaining_quota
    yt.rate_limiter.can_upload.return_value = can_upload
    yt.circuit_breaker = MagicMock()
    yt.circuit_breaker.is_open = circuit_breaker_open
    return yt


def _make_twitter_connector(*, configured: bool = True) -> MagicMock:
    tw = MagicMock()
    tw.is_configured = configured
    return tw


def _make_audio_store(*, exists: bool = True, path: Path | None = None) -> MagicMock:
    store = MagicMock()
    store.exists.return_value = exists
    store.get_path.return_value = path or (Path("/tmp/audio/debate.mp3") if exists else None)
    return store


def _make_video_generator(*, video_path: Path | None = None) -> MagicMock:
    gen = MagicMock()
    gen.generate_waveform_video.return_value = video_path or Path("/tmp/video/debate.mp4")
    gen.generate_static_video.return_value = video_path or Path("/tmp/video/debate_static.mp4")
    return gen


def _make_storage(*, debate: dict | None = None, slug_debate: dict | None = None) -> MagicMock:
    storage = MagicMock()
    storage.get_debate.return_value = debate
    storage.get_debate_by_slug.return_value = slug_debate
    return storage


@pytest.fixture
def handler(handler_cls):
    """Create a SocialMediaHandler with empty context."""
    return handler_cls(ctx={})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with headers."""
    mock = MagicMock()
    mock.command = "GET"
    mock.headers = {
        "Host": "localhost:8080",
        "Content-Length": "2",
    }
    mock.rfile = MagicMock()
    mock.rfile.read.return_value = b"{}"
    return mock


@pytest.fixture(autouse=True)
def _reset_oauth_states(handler_module):
    """Reset module-level OAuth state storage between tests."""
    handler_module._oauth_states.clear()
    yield
    handler_module._oauth_states.clear()


@pytest.fixture(autouse=True)
def _set_dev_mode(handler_module, monkeypatch):
    """Ensure we're in dev mode (not production) and set allowed hosts."""
    monkeypatch.setattr(handler_module, "_IS_PRODUCTION", False)
    monkeypatch.setattr(
        handler_module,
        "ALLOWED_OAUTH_HOSTS",
        frozenset(["localhost:8080", "127.0.0.1:8080"]),
    )


# ===========================================================================
# can_handle() Tests
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_youtube_auth(self, handler):
        assert handler.can_handle("/api/v1/youtube/auth") is True

    def test_youtube_callback(self, handler):
        assert handler.can_handle("/api/v1/youtube/callback") is True

    def test_youtube_status(self, handler):
        assert handler.can_handle("/api/v1/youtube/status") is True

    def test_publish_twitter(self, handler):
        assert handler.can_handle("/api/v1/debates/my-debate/publish/twitter") is True

    def test_publish_youtube(self, handler):
        assert handler.can_handle("/api/v1/debates/my-debate/publish/youtube") is True

    def test_unrelated_path_rejected(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_youtube_unknown_subpath(self, handler):
        assert handler.can_handle("/api/v1/youtube/something") is True

    def test_non_publish_debate_path(self, handler):
        assert handler.can_handle("/api/v1/debates/my-debate/results") is False

    def test_root_path_rejected(self, handler):
        assert handler.can_handle("/") is False

    def test_empty_path_rejected(self, handler):
        assert handler.can_handle("") is False


# ===========================================================================
# handle() GET Dispatch Tests
# ===========================================================================


class TestHandleGet:
    """Tests for GET request dispatch."""

    def test_returns_none_for_post_method(self, handler, mock_http_handler):
        """handle() returns None when method is POST."""
        mock_http_handler.command = "POST"
        result = handler.handle("/api/v1/youtube/auth", {}, mock_http_handler, method="POST")
        assert result is None

    def test_returns_none_for_unknown_path(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/unknown", {}, mock_http_handler, method="GET")
        assert result is None

    def test_dispatches_youtube_auth(self, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        h = handler_cls(ctx={"youtube_connector": yt})
        result = h.handle("/api/v1/youtube/auth", {}, mock_http_handler, method="GET")
        assert result is not None
        assert _status(result) == 200

    def test_dispatches_youtube_callback(self, handler_module, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        yt.exchange_code.return_value = {"success": True}
        h = handler_cls(ctx={"youtube_connector": yt})
        state = "test-state-123"
        handler_module._store_oauth_state(state)
        with patch.object(handler_module, "_run_async", side_effect=lambda x: x):
            result = h.handle(
                "/api/v1/youtube/callback",
                {"code": "auth-code", "state": state},
                mock_http_handler,
                method="GET",
            )
        assert result is not None

    def test_dispatches_youtube_status(self, handler_cls):
        yt = _make_youtube_connector()
        h = handler_cls(ctx={"youtube_connector": yt})
        result = h.handle("/api/v1/youtube/status", {}, None, method="GET")
        assert result is not None
        body = _body(result)
        assert body["configured"] is True

    def test_uses_handler_command_attribute(self, handler, mock_http_handler):
        """If handler.command is set, it overrides the method param."""
        mock_http_handler.command = "POST"
        result = handler.handle("/api/v1/youtube/auth", {}, mock_http_handler, method="GET")
        assert result is None  # POST method -> None


# ===========================================================================
# YouTube Auth URL Endpoint
# ===========================================================================


class TestYouTubeAuth:
    """Tests for _get_youtube_auth_url."""

    def test_no_youtube_connector(self, handler_cls, mock_http_handler):
        h = handler_cls(ctx={})
        result = h.handle("/api/v1/youtube/auth", {}, mock_http_handler, method="GET")
        assert _status(result) == 500
        assert "not initialized" in _body(result)["error"]

    def test_no_client_id(self, handler_cls, mock_http_handler):
        yt = _make_youtube_connector(client_id="")
        h = handler_cls(ctx={"youtube_connector": yt})
        result = h.handle("/api/v1/youtube/auth", {}, mock_http_handler, method="GET")
        assert _status(result) == 400
        body = _body(result)
        assert "client ID" in body.get("error", "")

    def test_untrusted_host(self, handler_module, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        h = handler_cls(ctx={"youtube_connector": yt})
        mock_http_handler.headers["Host"] = "evil.example.com"
        result = h.handle("/api/v1/youtube/auth", {}, mock_http_handler, method="GET")
        assert _status(result) == 400
        assert "Untrusted" in _body(result)["error"]

    def test_success_returns_auth_url(self, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        yt.get_auth_url.return_value = "https://accounts.google.com/o/oauth2/auth?test=1"
        h = handler_cls(ctx={"youtube_connector": yt})
        result = h.handle("/api/v1/youtube/auth", {}, mock_http_handler, method="GET")
        assert _status(result) == 200
        body = _body(result)
        assert "auth_url" in body
        assert "state" in body
        assert body["auth_url"].startswith("https://accounts.google.com")

    def test_https_scheme_via_forwarded_proto(self, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        yt.get_auth_url.return_value = "https://accounts.google.com/o/oauth2/auth"
        mock_http_handler.headers["X-Forwarded-Proto"] = "https"
        h = handler_cls(ctx={"youtube_connector": yt})
        result = h.handle("/api/v1/youtube/auth", {}, mock_http_handler, method="GET")
        assert _status(result) == 200
        # The redirect URI passed should use https
        call_args = yt.get_auth_url.call_args
        assert call_args[0][0].startswith("https://")

    def test_http_scheme_default(self, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        yt.get_auth_url.return_value = "https://accounts.google.com/o/oauth2/auth"
        h = handler_cls(ctx={"youtube_connector": yt})
        result = h.handle("/api/v1/youtube/auth", {}, mock_http_handler, method="GET")
        assert _status(result) == 200
        call_args = yt.get_auth_url.call_args
        assert call_args[0][0].startswith("http://")

    def test_state_is_stored(self, handler_module, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        yt.get_auth_url.return_value = "https://example.com/auth"
        h = handler_cls(ctx={"youtube_connector": yt})
        result = h.handle("/api/v1/youtube/auth", {}, mock_http_handler, method="GET")
        state = _body(result)["state"]
        # State should be stored for validation
        assert state in handler_module._oauth_states


# ===========================================================================
# YouTube Callback Endpoint
# ===========================================================================


class TestYouTubeCallback:
    """Tests for _handle_youtube_callback."""

    def test_missing_code(self, handler_cls, mock_http_handler):
        h = handler_cls(ctx={})
        result = h.handle(
            "/api/v1/youtube/callback",
            {"state": "some-state"},
            mock_http_handler,
            method="GET",
        )
        assert _status(result) == 400
        assert "authorization code" in _body(result)["error"].lower()

    def test_missing_state(self, handler_cls, mock_http_handler):
        h = handler_cls(ctx={})
        result = h.handle(
            "/api/v1/youtube/callback",
            {"code": "some-code"},
            mock_http_handler,
            method="GET",
        )
        assert _status(result) == 400
        assert "state" in _body(result)["error"].lower()

    def test_invalid_state(self, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        h = handler_cls(ctx={"youtube_connector": yt})
        result = h.handle(
            "/api/v1/youtube/callback",
            {"code": "code-123", "state": "invalid-state"},
            mock_http_handler,
            method="GET",
        )
        assert _status(result) == 400
        assert (
            "expired" in _body(result)["error"].lower()
            or "invalid" in _body(result)["error"].lower()
        )

    def test_expired_state(self, handler_module, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        h = handler_cls(ctx={"youtube_connector": yt})
        # Store state that is already expired
        handler_module._oauth_states["expired-state"] = time.time() - 1
        result = h.handle(
            "/api/v1/youtube/callback",
            {"code": "code-123", "state": "expired-state"},
            mock_http_handler,
            method="GET",
        )
        assert _status(result) == 400

    def test_no_youtube_connector_after_state_validation(
        self, handler_module, handler_cls, mock_http_handler
    ):
        h = handler_cls(ctx={})
        state = "valid-state"
        handler_module._store_oauth_state(state)
        result = h.handle(
            "/api/v1/youtube/callback",
            {"code": "code-123", "state": state},
            mock_http_handler,
            method="GET",
        )
        assert _status(result) == 500
        assert "not initialized" in _body(result)["error"]

    def test_untrusted_host_in_callback(self, handler_module, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        h = handler_cls(ctx={"youtube_connector": yt})
        state = "valid-state"
        handler_module._store_oauth_state(state)
        mock_http_handler.headers["Host"] = "evil.com"
        result = h.handle(
            "/api/v1/youtube/callback",
            {"code": "code-123", "state": state},
            mock_http_handler,
            method="GET",
        )
        assert _status(result) == 400
        assert "Untrusted" in _body(result)["error"]

    def test_successful_exchange(self, handler_module, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        yt.exchange_code.return_value = {"success": True}
        h = handler_cls(ctx={"youtube_connector": yt})
        state = "valid-state"
        handler_module._store_oauth_state(state)
        with patch.object(handler_module, "_run_async", side_effect=lambda x: x):
            result = h.handle(
                "/api/v1/youtube/callback",
                {"code": "code-123", "state": state},
                mock_http_handler,
                method="GET",
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    def test_failed_exchange(self, handler_module, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        yt.exchange_code.return_value = {"success": False, "error": "Bad credentials"}
        h = handler_cls(ctx={"youtube_connector": yt})
        state = "valid-state"
        handler_module._store_oauth_state(state)
        with patch.object(handler_module, "_run_async", side_effect=lambda x: x):
            result = h.handle(
                "/api/v1/youtube/callback",
                {"code": "code-123", "state": state},
                mock_http_handler,
                method="GET",
            )
        assert _status(result) == 400
        body = _body(result)
        assert body["success"] is False
        assert body["error"] == "Bad credentials"

    def test_exchange_unknown_error(self, handler_module, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        yt.exchange_code.return_value = {"success": False}
        h = handler_cls(ctx={"youtube_connector": yt})
        state = "valid-state"
        handler_module._store_oauth_state(state)
        with patch.object(handler_module, "_run_async", side_effect=lambda x: x):
            result = h.handle(
                "/api/v1/youtube/callback",
                {"code": "code-123", "state": state},
                mock_http_handler,
                method="GET",
            )
        assert _status(result) == 400
        body = _body(result)
        assert body["error"] == "Unknown error"

    def test_exchange_exception(self, handler_module, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        h = handler_cls(ctx={"youtube_connector": yt})
        state = "valid-state"
        handler_module._store_oauth_state(state)
        with patch.object(handler_module, "_run_async", side_effect=ConnectionError("timeout")):
            result = h.handle(
                "/api/v1/youtube/callback",
                {"code": "code-123", "state": state},
                mock_http_handler,
                method="GET",
            )
        assert _status(result) == 500

    def test_callback_https_scheme(self, handler_module, handler_cls, mock_http_handler):
        yt = _make_youtube_connector()
        yt.exchange_code.return_value = {"success": True}
        h = handler_cls(ctx={"youtube_connector": yt})
        state = "valid-state"
        handler_module._store_oauth_state(state)
        mock_http_handler.headers["X-Forwarded-Proto"] = "https"
        with patch.object(handler_module, "_run_async", side_effect=lambda x: x):
            result = h.handle(
                "/api/v1/youtube/callback",
                {"code": "code-123", "state": state},
                mock_http_handler,
                method="GET",
            )
        assert _status(result) == 200
        # Verify redirect_uri used https
        call_args = yt.exchange_code.call_args
        assert call_args[0][1].startswith("https://")

    def test_state_is_consumed(self, handler_module, handler_cls, mock_http_handler):
        """State should be one-time use (consumed after validation)."""
        yt = _make_youtube_connector()
        yt.exchange_code.return_value = {"success": True}
        h = handler_cls(ctx={"youtube_connector": yt})
        state = "one-time-state"
        handler_module._store_oauth_state(state)
        with patch.object(handler_module, "_run_async", side_effect=lambda x: x):
            h.handle(
                "/api/v1/youtube/callback",
                {"code": "code-123", "state": state},
                mock_http_handler,
                method="GET",
            )
        # Second attempt with same state should fail
        handler_module._store_oauth_state(
            "another-state"
        )  # need a new valid state to ensure we're testing consumption
        result = h.handle(
            "/api/v1/youtube/callback",
            {"code": "code-456", "state": state},
            mock_http_handler,
            method="GET",
        )
        assert _status(result) == 400


# ===========================================================================
# YouTube Status Endpoint
# ===========================================================================


class TestYouTubeStatus:
    """Tests for _get_youtube_status."""

    def test_no_connector(self, handler):
        result = handler.handle("/api/v1/youtube/status", {}, None, method="GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["configured"] is False

    def test_configured_connector(self, handler_cls):
        yt = _make_youtube_connector()
        h = handler_cls(ctx={"youtube_connector": yt})
        result = h.handle("/api/v1/youtube/status", {}, None, method="GET")
        body = _body(result)
        assert body["configured"] is True
        assert body["has_client_id"] is True
        assert body["has_client_secret"] is True
        assert body["has_refresh_token"] is True
        assert body["quota_remaining"] == 9000
        assert body["circuit_breaker_open"] is False

    def test_partially_configured(self, handler_cls):
        yt = _make_youtube_connector(client_id="cid", client_secret="", refresh_token="")
        h = handler_cls(ctx={"youtube_connector": yt})
        result = h.handle("/api/v1/youtube/status", {}, None, method="GET")
        body = _body(result)
        assert body["has_client_id"] is True
        assert body["has_client_secret"] is False
        assert body["has_refresh_token"] is False

    def test_circuit_breaker_open(self, handler_cls):
        yt = _make_youtube_connector(circuit_breaker_open=True)
        h = handler_cls(ctx={"youtube_connector": yt})
        result = h.handle("/api/v1/youtube/status", {}, None, method="GET")
        body = _body(result)
        assert body["circuit_breaker_open"] is True


# ===========================================================================
# Twitter Publishing Endpoint
# ===========================================================================


class TestPublishToTwitter:
    """Tests for POST /api/v1/debates/{id}/publish/twitter."""

    def _make_handler(self, **ctx_overrides) -> Any:
        from aragora.server.handlers.social.social_media import SocialMediaHandler

        ctx = {}
        ctx.update(ctx_overrides)
        return SocialMediaHandler(ctx=ctx)

    def _mock_http(self, body: dict | None = None) -> MagicMock:
        mock = MagicMock()
        mock.command = "POST"
        mock.headers = {
            "Host": "localhost:8080",
            "Content-Length": str(len(json.dumps(body or {}).encode())),
        }
        mock.rfile = MagicMock()
        mock.rfile.read.return_value = json.dumps(body or {}).encode()
        return mock

    def test_no_twitter_connector(self, handler_module):
        h = self._make_handler()
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/twitter", {}, http)
        assert _status(result) == 500
        assert "not initialized" in _body(result)["error"]

    def test_twitter_not_configured(self, handler_module):
        tw = _make_twitter_connector(configured=False)
        h = self._make_handler(twitter_connector=tw)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/twitter", {}, http)
        assert _status(result) == 400
        body = _body(result)
        assert "credentials" in body["error"].lower()

    def test_no_storage(self, handler_module):
        tw = _make_twitter_connector()
        h = self._make_handler(twitter_connector=tw)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/twitter", {}, http)
        assert _status(result) == 503

    def test_debate_not_found(self, handler_module):
        tw = _make_twitter_connector()
        storage = _make_storage(debate=None, slug_debate=None)
        h = self._make_handler(twitter_connector=tw, storage=storage)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/twitter", {}, http)
        assert _status(result) == 404

    def test_debate_found_by_slug(self, handler_module):
        tw = _make_twitter_connector()
        tw.post_tweet.return_value = {
            "success": True,
            "tweet_id": "123",
            "url": "https://x.com/123",
        }
        storage = _make_storage(
            debate=None,
            slug_debate={"task": "Test debate", "agents": ["claude"], "verdict": "Yes"},
        )
        h = self._make_handler(twitter_connector=tw, storage=storage)
        http = self._mock_http()
        mock_formatter = MagicMock()
        mock_formatter.return_value.format_single_tweet.return_value = "A tweet"
        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch(
                "aragora.server.handlers.social.social_media.DebateContentFormatter",
                mock_formatter,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.twitter_poster": MagicMock(
                        DebateContentFormatter=mock_formatter
                    )
                },
            ),
        ):
            result = h.handle_post("/api/v1/debates/my-slug/publish/twitter", {}, http)
        assert _status(result) == 200

    def test_invalid_json_body(self, handler_module):
        tw = _make_twitter_connector()
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        h = self._make_handler(twitter_connector=tw, storage=storage)
        http = self._mock_http()
        # Simulate invalid JSON body
        http.rfile.read.return_value = b"not-json"
        http.headers["Content-Length"] = "8"
        result = h.handle_post("/api/v1/debates/d1/publish/twitter", {}, http)
        assert _status(result) == 400

    def test_single_tweet_success(self, handler_module):
        tw = _make_twitter_connector()
        tw.post_tweet.return_value = {
            "success": True,
            "tweet_id": "t-1",
            "url": "https://x.com/t-1",
        }
        debate = {
            "task": "Should we use microservices?",
            "agents": ["claude", "gpt"],
            "verdict": "Yes",
        }
        storage = _make_storage(debate=debate)
        h = self._make_handler(twitter_connector=tw, storage=storage)
        http = self._mock_http({"thread_mode": False})

        mock_formatter_cls = MagicMock()
        mock_formatter = MagicMock()
        mock_formatter.format_single_tweet.return_value = "AI debated microservices!"
        mock_formatter_cls.return_value = mock_formatter
        twitter_poster_mod = MagicMock(DebateContentFormatter=mock_formatter_cls)

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.twitter_poster": twitter_poster_mod}),
        ):
            result = h.handle_post("/api/v1/debates/d1/publish/twitter", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["debate_id"] == "d1"
        assert body["tweet_id"] == "t-1"

    def test_thread_mode(self, handler_module):
        tw = _make_twitter_connector()
        tw.post_thread.return_value = {"success": True, "thread_ids": ["t-1", "t-2"]}
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Maybe"}
        storage = _make_storage(debate=debate)
        h = self._make_handler(twitter_connector=tw, storage=storage)
        http = self._mock_http({"thread_mode": True})

        mock_formatter_cls = MagicMock()
        mock_formatter = MagicMock()
        mock_formatter.format_as_thread.return_value = ["Tweet 1", "Tweet 2"]
        mock_formatter_cls.return_value = mock_formatter
        twitter_poster_mod = MagicMock(DebateContentFormatter=mock_formatter_cls)

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.twitter_poster": twitter_poster_mod}),
        ):
            result = h.handle_post("/api/v1/debates/d1/publish/twitter", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["thread_ids"] == ["t-1", "t-2"]

    def test_tweet_failure_response(self, handler_module):
        tw = _make_twitter_connector()
        tw.post_tweet.return_value = {"success": False, "error": "Rate limited"}
        debate = {"task": "Test", "agents": ["claude"], "verdict": None}
        storage = _make_storage(debate=debate)
        h = self._make_handler(twitter_connector=tw, storage=storage)
        http = self._mock_http()

        mock_formatter_cls = MagicMock()
        mock_formatter = MagicMock()
        mock_formatter.format_single_tweet.return_value = "A tweet"
        mock_formatter_cls.return_value = mock_formatter
        twitter_poster_mod = MagicMock(DebateContentFormatter=mock_formatter_cls)

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.twitter_poster": twitter_poster_mod}),
        ):
            result = h.handle_post("/api/v1/debates/d1/publish/twitter", {}, http)

        assert _status(result) == 500
        body = _body(result)
        assert body["success"] is False
        assert body["error"] == "Rate limited"

    def test_twitter_exception(self, handler_module):
        tw = _make_twitter_connector()
        debate = {"task": "Test", "agents": ["claude"], "verdict": None}
        storage = _make_storage(debate=debate)
        h = self._make_handler(twitter_connector=tw, storage=storage)
        http = self._mock_http()

        twitter_poster_mod = MagicMock()
        twitter_poster_mod.DebateContentFormatter.side_effect = ImportError("missing module")

        with patch.dict("sys.modules", {"aragora.connectors.twitter_poster": twitter_poster_mod}):
            result = h.handle_post("/api/v1/debates/d1/publish/twitter", {}, http)

        assert _status(result) == 500

    def test_include_audio_link(self, handler_module):
        tw = _make_twitter_connector()
        tw.post_tweet.return_value = {"success": True, "tweet_id": "t-1"}
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=True)
        h = self._make_handler(twitter_connector=tw, storage=storage, audio_store=audio)
        http = self._mock_http({"include_audio_link": True})

        mock_formatter_cls = MagicMock()
        mock_formatter = MagicMock()
        mock_formatter.format_single_tweet.return_value = "Tweet with audio"
        mock_formatter_cls.return_value = mock_formatter
        twitter_poster_mod = MagicMock(DebateContentFormatter=mock_formatter_cls)

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.twitter_poster": twitter_poster_mod}),
        ):
            result = h.handle_post("/api/v1/debates/d1/publish/twitter", {}, http)

        assert _status(result) == 200
        # Verify formatter was called with audio_url
        call_kwargs = mock_formatter.format_single_tweet.call_args
        assert call_kwargs[1].get("audio_url") is not None or (
            len(call_kwargs[0]) > 3 and call_kwargs[0][3] is not None
        )

    def test_no_audio_store(self, handler_module):
        tw = _make_twitter_connector()
        tw.post_tweet.return_value = {"success": True, "tweet_id": "t-1"}
        debate = {"task": "Test", "agents": ["claude"], "verdict": None}
        storage = _make_storage(debate=debate)
        h = self._make_handler(twitter_connector=tw, storage=storage)
        http = self._mock_http({"include_audio_link": True})

        mock_formatter_cls = MagicMock()
        mock_formatter = MagicMock()
        mock_formatter.format_single_tweet.return_value = "A tweet"
        mock_formatter_cls.return_value = mock_formatter
        twitter_poster_mod = MagicMock(DebateContentFormatter=mock_formatter_cls)

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.twitter_poster": twitter_poster_mod}),
        ):
            result = h.handle_post("/api/v1/debates/d1/publish/twitter", {}, http)

        assert _status(result) == 200

    def test_invalid_debate_id(self, handler_module):
        """Debate IDs with special chars should be rejected by path param validation."""
        tw = _make_twitter_connector()
        h = self._make_handler(twitter_connector=tw)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/../../etc/passwd/publish/twitter", {}, http)
        assert _status(result) == 400

    def test_unmatched_post_path(self, handler_module):
        h = self._make_handler()
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/results", {}, http)
        assert result is None


# ===========================================================================
# YouTube Publishing Endpoint
# ===========================================================================


class TestPublishToYouTube:
    """Tests for POST /api/v1/debates/{id}/publish/youtube."""

    def _make_handler(self, **ctx_overrides) -> Any:
        from aragora.server.handlers.social.social_media import SocialMediaHandler

        ctx = {}
        ctx.update(ctx_overrides)
        return SocialMediaHandler(ctx=ctx)

    def _mock_http(self, body: dict | None = None) -> MagicMock:
        mock = MagicMock()
        mock.command = "POST"
        mock.headers = {
            "Host": "localhost:8080",
            "Content-Length": str(len(json.dumps(body or {}).encode())),
            "X-Forwarded-Proto": "http",
        }
        mock.rfile = MagicMock()
        mock.rfile.read.return_value = json.dumps(body or {}).encode()
        return mock

    def test_no_youtube_connector(self, handler_module):
        h = self._make_handler()
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)
        assert _status(result) == 500

    def test_youtube_not_configured(self, handler_module):
        yt = _make_youtube_connector(configured=False)
        h = self._make_handler(youtube_connector=yt)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)
        assert _status(result) == 400
        body = _body(result)
        assert "credentials" in body["error"].lower()

    def test_no_storage(self, handler_module):
        yt = _make_youtube_connector()
        h = self._make_handler(youtube_connector=yt)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)
        assert _status(result) == 503

    def test_debate_not_found(self, handler_module):
        yt = _make_youtube_connector()
        storage = _make_storage(debate=None, slug_debate=None)
        h = self._make_handler(youtube_connector=yt, storage=storage)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)
        assert _status(result) == 404

    def test_quota_exceeded(self, handler_module):
        yt = _make_youtube_connector(can_upload=False, remaining_quota=0)
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        h = self._make_handler(youtube_connector=yt, storage=storage)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)
        assert _status(result) == 429
        body = _body(result)
        assert "quota" in body["error"].lower()
        assert body["quota_remaining"] == 0

    def test_no_audio_store(self, handler_module):
        yt = _make_youtube_connector()
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        h = self._make_handler(youtube_connector=yt, storage=storage)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)
        assert _status(result) == 400
        assert "audio" in _body(result)["error"].lower()

    def test_no_audio_for_debate(self, handler_module):
        yt = _make_youtube_connector()
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=False)
        h = self._make_handler(youtube_connector=yt, storage=storage, audio_store=audio)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)
        assert _status(result) == 400
        assert "audio" in _body(result)["error"].lower()

    def test_audio_path_not_found(self, handler_module):
        yt = _make_youtube_connector()
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=True, path=None)
        audio.get_path.return_value = None
        h = self._make_handler(youtube_connector=yt, storage=storage, audio_store=audio)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)
        assert _status(result) == 404

    def test_no_video_generator(self, handler_module):
        yt = _make_youtube_connector()
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=True)
        h = self._make_handler(youtube_connector=yt, storage=storage, audio_store=audio)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)
        assert _status(result) == 503

    def test_video_generation_returns_none(self, handler_module):
        yt = _make_youtube_connector()
        yt.upload.return_value = {"success": True, "video_id": "v1"}
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=True)
        vgen = _make_video_generator()
        vgen.generate_waveform_video.return_value = None
        h = self._make_handler(
            youtube_connector=yt, storage=storage, audio_store=audio, video_generator=vgen
        )
        http = self._mock_http()

        yt_uploader_mod = MagicMock()
        yt_uploader_mod.YouTubeVideoMetadata = MagicMock()

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.youtube_uploader": yt_uploader_mod}),
        ):
            result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)

        assert _status(result) == 500
        assert "generation failed" in _body(result)["error"].lower()

    def test_successful_upload(self, handler_module):
        yt = _make_youtube_connector(remaining_quota=8000)
        yt.upload.return_value = {
            "success": True,
            "video_id": "v-abc",
            "url": "https://youtu.be/v-abc",
        }
        debate = {"task": "Microservices debate", "agents": ["claude", "gpt"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=True)
        vgen = _make_video_generator()
        h = self._make_handler(
            youtube_connector=yt, storage=storage, audio_store=audio, video_generator=vgen
        )
        http = self._mock_http({"title": "Custom Title", "tags": ["custom"]})

        yt_uploader_mod = MagicMock()
        metadata_cls = MagicMock()
        yt_uploader_mod.YouTubeVideoMetadata = metadata_cls

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.youtube_uploader": yt_uploader_mod}),
        ):
            result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["video_id"] == "v-abc"
        assert body["url"] == "https://youtu.be/v-abc"
        assert body["quota_remaining"] == 8000

    def test_upload_failure(self, handler_module):
        yt = _make_youtube_connector()
        yt.upload.return_value = {"success": False, "error": "Quota limit reached"}
        debate = {"task": "Test", "agents": ["claude"], "verdict": None}
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=True)
        vgen = _make_video_generator()
        h = self._make_handler(
            youtube_connector=yt, storage=storage, audio_store=audio, video_generator=vgen
        )
        http = self._mock_http()

        yt_uploader_mod = MagicMock()
        yt_uploader_mod.YouTubeVideoMetadata = MagicMock()

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.youtube_uploader": yt_uploader_mod}),
        ):
            result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)

        assert _status(result) == 500
        body = _body(result)
        assert body["success"] is False
        assert body["error"] == "Quota limit reached"

    def test_upload_exception(self, handler_module):
        yt = _make_youtube_connector()
        debate = {"task": "Test", "agents": ["claude"], "verdict": None}
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=True)
        vgen = _make_video_generator()
        h = self._make_handler(
            youtube_connector=yt, storage=storage, audio_store=audio, video_generator=vgen
        )
        http = self._mock_http()

        with patch.object(handler_module, "_run_async", side_effect=OSError("Disk full")):
            result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)

        assert _status(result) == 500

    def test_waveform_fails_falls_back_to_static(self, handler_module):
        yt = _make_youtube_connector()
        yt.upload.return_value = {"success": True, "video_id": "v-1"}
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=True)
        vgen = _make_video_generator()
        # generate_waveform_video raises (before _run_async),
        # then generate_static_video returns a valid path via _run_async
        static_path = Path("/tmp/static.mp4")
        vgen.generate_waveform_video.side_effect = RuntimeError("ffmpeg crash")
        vgen.generate_static_video.return_value = static_path

        h = self._make_handler(
            youtube_connector=yt, storage=storage, audio_store=audio, video_generator=vgen
        )
        http = self._mock_http()

        yt_uploader_mod = MagicMock()
        yt_uploader_mod.YouTubeVideoMetadata = MagicMock()

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.youtube_uploader": yt_uploader_mod}),
        ):
            result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    def test_custom_metadata(self, handler_module):
        yt = _make_youtube_connector()
        yt.upload.return_value = {"success": True, "video_id": "v-1"}
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=True)
        vgen = _make_video_generator()
        h = self._make_handler(
            youtube_connector=yt, storage=storage, audio_store=audio, video_generator=vgen
        )

        custom_body = {
            "title": "My Custom Title",
            "description": "My custom description",
            "tags": ["tag1", "tag2"],
            "privacy": "private",
        }
        http = self._mock_http(custom_body)

        yt_uploader_mod = MagicMock()
        metadata_cls = MagicMock()
        yt_uploader_mod.YouTubeVideoMetadata = metadata_cls

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.youtube_uploader": yt_uploader_mod}),
        ):
            result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)

        assert _status(result) == 200
        # Verify metadata was created with custom values
        call_kwargs = metadata_cls.call_args[1]
        assert call_kwargs["title"] == "My Custom Title"
        assert call_kwargs["description"] == "My custom description"
        assert call_kwargs["tags"] == ["tag1", "tag2"]
        assert call_kwargs["privacy_status"] == "private"

    def test_default_metadata(self, handler_module):
        yt = _make_youtube_connector()
        yt.upload.return_value = {"success": True, "video_id": "v-1"}
        debate = {
            "task": "Climate change",
            "agents": ["claude", "gpt"],
            "verdict": "Reduce emissions",
        }
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=True)
        vgen = _make_video_generator()
        h = self._make_handler(
            youtube_connector=yt, storage=storage, audio_store=audio, video_generator=vgen
        )
        http = self._mock_http({})

        yt_uploader_mod = MagicMock()
        metadata_cls = MagicMock()
        yt_uploader_mod.YouTubeVideoMetadata = metadata_cls

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.youtube_uploader": yt_uploader_mod}),
        ):
            result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)

        assert _status(result) == 200
        call_kwargs = metadata_cls.call_args[1]
        assert "AI Debate: Climate change" in call_kwargs["title"]
        assert "claude" in call_kwargs["description"]
        assert call_kwargs["privacy_status"] == "public"
        assert "AI" in call_kwargs["tags"]

    def test_invalid_json_body(self, handler_module):
        yt = _make_youtube_connector()
        debate = {"task": "Test", "agents": ["claude"], "verdict": "Yes"}
        storage = _make_storage(debate=debate)
        audio = _make_audio_store(exists=True)
        h = self._make_handler(youtube_connector=yt, storage=storage, audio_store=audio)
        http = self._mock_http()
        http.rfile.read.return_value = b"invalid-json"
        http.headers["Content-Length"] = "12"
        result = h.handle_post("/api/v1/debates/d1/publish/youtube", {}, http)
        assert _status(result) == 400

    def test_invalid_debate_id_path_traversal(self, handler_module):
        yt = _make_youtube_connector()
        h = self._make_handler(youtube_connector=yt)
        http = self._mock_http()
        result = h.handle_post("/api/v1/debates/../../etc/passwd/publish/youtube", {}, http)
        assert _status(result) == 400

    def test_debate_found_by_slug_fallback(self, handler_module):
        yt = _make_youtube_connector()
        yt.upload.return_value = {"success": True, "video_id": "v-1"}
        storage = _make_storage(
            debate=None,
            slug_debate={"task": "Test", "agents": ["claude"], "verdict": "Yes"},
        )
        audio = _make_audio_store(exists=True)
        vgen = _make_video_generator()
        h = self._make_handler(
            youtube_connector=yt, storage=storage, audio_store=audio, video_generator=vgen
        )
        http = self._mock_http()

        yt_uploader_mod = MagicMock()
        yt_uploader_mod.YouTubeVideoMetadata = MagicMock()

        with (
            patch.object(handler_module, "_run_async", side_effect=lambda x: x),
            patch.dict("sys.modules", {"aragora.connectors.youtube_uploader": yt_uploader_mod}),
        ):
            result = h.handle_post("/api/v1/debates/my-slug/publish/youtube", {}, http)

        assert _status(result) == 200
        # get_debate_by_slug should have been called
        storage.get_debate_by_slug.assert_called_once()


# ===========================================================================
# OAuth State Management Tests
# ===========================================================================


class TestOAuthStateManagement:
    """Tests for _store_oauth_state and _validate_oauth_state."""

    def test_store_and_validate(self, handler_module):
        handler_module._store_oauth_state("state-1")
        assert handler_module._validate_oauth_state("state-1") is True

    def test_validate_consumes_state(self, handler_module):
        handler_module._store_oauth_state("state-1")
        assert handler_module._validate_oauth_state("state-1") is True
        assert handler_module._validate_oauth_state("state-1") is False

    def test_validate_unknown_state(self, handler_module):
        assert handler_module._validate_oauth_state("nonexistent") is False

    def test_expired_state_rejected(self, handler_module):
        handler_module._oauth_states["expired"] = time.time() - 1
        assert handler_module._validate_oauth_state("expired") is False

    def test_cleanup_expired_on_store(self, handler_module):
        handler_module._oauth_states["old"] = time.time() - 100
        handler_module._store_oauth_state("new")
        assert "old" not in handler_module._oauth_states
        assert "new" in handler_module._oauth_states

    def test_max_states_eviction(self, handler_module):
        original_max = handler_module.MAX_OAUTH_STATES
        handler_module.MAX_OAUTH_STATES = 10
        try:
            for i in range(10):
                handler_module._oauth_states[f"state-{i}"] = time.time() + 600 + i
            handler_module._store_oauth_state("overflow")
            # Some old states should be evicted
            assert len(handler_module._oauth_states) <= 10
            assert "overflow" in handler_module._oauth_states
        finally:
            handler_module.MAX_OAUTH_STATES = original_max

    def test_multiple_states_stored(self, handler_module):
        handler_module._store_oauth_state("a")
        handler_module._store_oauth_state("b")
        handler_module._store_oauth_state("c")
        assert handler_module._validate_oauth_state("b") is True
        assert handler_module._validate_oauth_state("a") is True
        assert handler_module._validate_oauth_state("c") is True


# ===========================================================================
# Accessor Method Tests
# ===========================================================================


class TestAccessorMethods:
    """Tests for typed accessor methods on SocialMediaHandler."""

    def test_get_youtube_connector_none(self, handler):
        assert handler._get_youtube_connector() is None

    def test_get_youtube_connector_present(self, handler_cls):
        yt = _make_youtube_connector()
        h = handler_cls(ctx={"youtube_connector": yt})
        assert h._get_youtube_connector() is yt

    def test_get_twitter_connector_none(self, handler):
        assert handler._get_twitter_connector() is None

    def test_get_twitter_connector_present(self, handler_cls):
        tw = _make_twitter_connector()
        h = handler_cls(ctx={"twitter_connector": tw})
        assert h._get_twitter_connector() is tw

    def test_get_audio_store_none(self, handler):
        assert handler._get_audio_store() is None

    def test_get_audio_store_present(self, handler_cls):
        audio = _make_audio_store()
        h = handler_cls(ctx={"audio_store": audio})
        assert h._get_audio_store() is audio

    def test_get_video_generator_none(self, handler):
        assert handler._get_video_generator() is None

    def test_get_video_generator_present(self, handler_cls):
        vgen = _make_video_generator()
        h = handler_cls(ctx={"video_generator": vgen})
        assert h._get_video_generator() is vgen


# ===========================================================================
# ROUTES Class Attribute Tests
# ===========================================================================


class TestRoutes:
    """Tests for the ROUTES class attribute."""

    def test_routes_defined(self, handler_cls):
        assert hasattr(handler_cls, "ROUTES")
        assert len(handler_cls.ROUTES) == 5

    def test_youtube_auth_route(self, handler_cls):
        assert "/api/v1/youtube/auth" in handler_cls.ROUTES

    def test_youtube_callback_route(self, handler_cls):
        assert "/api/v1/youtube/callback" in handler_cls.ROUTES

    def test_youtube_status_route(self, handler_cls):
        assert "/api/v1/youtube/status" in handler_cls.ROUTES

    def test_twitter_publish_route(self, handler_cls):
        assert "/api/v1/debates/*/publish/twitter" in handler_cls.ROUTES

    def test_youtube_publish_route(self, handler_cls):
        assert "/api/v1/debates/*/publish/youtube" in handler_cls.ROUTES


# ===========================================================================
# Init / Constructor Tests
# ===========================================================================


class TestInit:
    """Tests for handler initialization."""

    def test_default_context(self, handler_cls):
        h = handler_cls()
        assert h.ctx == {}

    def test_none_context(self, handler_cls):
        h = handler_cls(ctx=None)
        assert h.ctx == {}

    def test_custom_context(self, handler_cls):
        ctx = {"youtube_connector": "yt", "twitter_connector": "tw"}
        h = handler_cls(ctx=ctx)
        assert h.ctx is ctx
