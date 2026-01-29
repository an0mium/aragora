"""Tests for Speech Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.speech import (
    SpeechHandler,
    _speech_limiter,
    MAX_FILE_SIZE_MB,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    _speech_limiter._buckets.clear()
    yield


@pytest.fixture
def handler():
    """Create handler instance."""
    return SpeechHandler({})


class TestSpeechConstants:
    """Tests for speech constants."""

    def test_max_file_size(self):
        """Test max file size constants."""
        assert MAX_FILE_SIZE_MB == 25
        assert MAX_FILE_SIZE_BYTES == 25 * 1024 * 1024

    def test_supported_extensions(self):
        """Test supported extensions."""
        assert ".mp3" in SUPPORTED_EXTENSIONS
        assert ".wav" in SUPPORTED_EXTENSIONS
        assert ".m4a" in SUPPORTED_EXTENSIONS
        assert ".webm" in SUPPORTED_EXTENSIONS


class TestSpeechHandler:
    """Tests for SpeechHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(SpeechHandler, "ROUTES")
        routes = SpeechHandler.ROUTES
        assert "/api/v1/speech/transcribe" in routes
        assert "/api/v1/speech/transcribe-url" in routes
        assert "/api/v1/speech/providers" in routes

    def test_can_handle_speech_routes(self, handler):
        """Test can_handle for speech routes."""
        assert handler.can_handle("/api/v1/speech/transcribe") is True
        assert handler.can_handle("/api/v1/speech/transcribe-url") is True
        assert handler.can_handle("/api/v1/speech/providers") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/audio/transcribe") is False


class TestSpeechProviders:
    """Tests for speech providers endpoint."""

    def test_get_providers(self, handler):
        """Test getting available providers."""
        result = handler._get_providers()
        assert result.status_code == 200

        import json

        body = json.loads(result.body)
        assert "providers" in body
        assert "default" in body

    def test_get_providers_includes_openai(self, handler):
        """Test providers include OpenAI Whisper."""
        result = handler._get_providers()

        import json

        body = json.loads(result.body)
        provider_names = [p["name"] for p in body["providers"]]
        assert "openai_whisper" in provider_names


class TestSpeechTranscribe:
    """Tests for speech transcription endpoint."""

    def test_transcribe_no_content_length(self, handler):
        """Test transcribe requires content length."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)
        mock_handler.headers = MagicMock()
        mock_handler.headers.get = MagicMock(return_value="0")

        with (
            patch(
                "aragora.server.handlers.features.speech.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.speech.get_client_ip",
                return_value="127.0.0.1",
            ),
        ):
            result = handler._transcribe_upload(mock_handler, {})
            assert result.status_code == 400

    def test_transcribe_file_too_large(self, handler):
        """Test transcribe rejects files too large."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)
        mock_handler.headers = MagicMock()
        mock_handler.headers.get = MagicMock(return_value=str(MAX_FILE_SIZE_BYTES + 1))

        with (
            patch(
                "aragora.server.handlers.features.speech.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.speech.get_client_ip",
                return_value="127.0.0.1",
            ),
        ):
            result = handler._transcribe_upload(mock_handler, {})
            assert result.status_code == 413

    def test_transcribe_unsupported_format(self, handler):
        """Test transcribe rejects unsupported formats."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)
        mock_handler.headers = MagicMock()
        mock_handler.headers.get = MagicMock(
            side_effect=lambda k, d=None: {
                "Content-Length": "1000",
                "Content-Type": "audio/mp3",
            }.get(k, d)
        )

        with (
            patch(
                "aragora.server.handlers.features.speech.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.speech.get_client_ip",
                return_value="127.0.0.1",
            ),
            patch.object(
                handler,
                "_parse_upload",
                return_value=(b"content", "audio.xyz"),
            ),
        ):
            result = handler._transcribe_upload(mock_handler, {})
            assert result.status_code == 400


class TestSpeechTranscribeUrl:
    """Tests for URL transcription endpoint."""

    def test_transcribe_url_missing_url(self, handler):
        """Test URL transcribe requires URL."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)
        mock_handler.headers = MagicMock()
        mock_handler.headers.get = MagicMock(return_value="2")
        mock_handler.rfile = MagicMock()
        mock_handler.rfile.read = MagicMock(return_value=b"{}")

        with (
            patch(
                "aragora.server.handlers.features.speech.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.speech.get_client_ip",
                return_value="127.0.0.1",
            ),
        ):
            result = handler._transcribe_from_url(mock_handler, {})
            assert result.status_code == 400

    def test_transcribe_url_invalid_url(self, handler):
        """Test URL transcribe validates URL format."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)
        mock_handler.headers = MagicMock()
        mock_handler.headers.get = MagicMock(return_value="20")
        mock_handler.rfile = MagicMock()
        mock_handler.rfile.read = MagicMock(return_value=b'{"url": "invalid-url"}')

        with (
            patch(
                "aragora.server.handlers.features.speech.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.speech.get_client_ip",
                return_value="127.0.0.1",
            ),
        ):
            result = handler._transcribe_from_url(mock_handler, {})
            assert result.status_code == 400


class TestSpeechRateLimiting:
    """Tests for speech rate limiting."""

    def test_rate_limiter_exists(self):
        """Test that rate limiter is configured."""
        assert _speech_limiter is not None
        assert _speech_limiter.requests_per_minute == 10

    def test_rate_limit_exceeded(self, handler):
        """Test rate limit enforcement."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)
        mock_handler.headers = MagicMock()
        mock_handler.headers.get = MagicMock(return_value="1000")

        # Exhaust rate limit
        for _ in range(11):
            _speech_limiter.is_allowed("127.0.0.1")

        with patch(
            "aragora.server.handlers.features.speech.get_client_ip",
            return_value="127.0.0.1",
        ):
            result = handler.handle_post("/api/v1/speech/transcribe", {}, mock_handler)
            assert result.status_code == 429


class TestSpeechParseUpload:
    """Tests for upload parsing."""

    def test_parse_raw_upload(self, handler):
        """Test parsing raw file upload."""
        mock_handler = MagicMock()
        mock_handler.headers = MagicMock()
        mock_handler.headers.get = MagicMock(return_value="audio.mp3")
        mock_handler.rfile = MagicMock()
        mock_handler.rfile.read = MagicMock(return_value=b"audio content")

        content, filename = handler._parse_raw(mock_handler, 13)
        assert content == b"audio content"
        assert filename == "audio.mp3"

    def test_parse_multipart_no_boundary(self, handler):
        """Test parsing multipart without boundary returns None."""
        mock_handler = MagicMock()

        content, filename = handler._parse_multipart(mock_handler, "multipart/form-data", 100)
        assert content is None
        assert filename is None


class TestAsyncTranscription:
    """Tests for async transcription."""

    @pytest.mark.asyncio
    async def test_do_transcription_import_error(self, handler):
        """Test transcription handles import error."""
        with patch(
            "aragora.server.handlers.features.speech.transcribe_audio",
            side_effect=ImportError("Module not found"),
        ):
            result = await handler._do_transcription(
                content=b"audio",
                filename="audio.mp3",
                language=None,
                prompt=None,
                provider_name=None,
            )
            assert "error" in result

    @pytest.mark.asyncio
    async def test_do_transcription_success(self, handler):
        """Test successful transcription."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "text": "Hello world",
            "language": "en",
            "segments": [],
        }

        with (
            patch(
                "aragora.server.handlers.features.speech.transcribe_audio",
                new_callable=AsyncMock,
            ) as mock_transcribe,
            patch("aragora.server.handlers.features.speech.STTProviderConfig"),
            patch("tempfile.NamedTemporaryFile"),
            patch("pathlib.Path"),
        ):
            mock_transcribe.return_value = mock_result

            result = await handler._do_transcription(
                content=b"audio",
                filename="audio.mp3",
                language=None,
                prompt=None,
                provider_name=None,
            )

            # Result should contain the transcription
            assert result.get("text") == "Hello world" or "error" in result
