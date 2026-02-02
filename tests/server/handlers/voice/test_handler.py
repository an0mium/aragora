"""Tests for VoiceHandler Twilio webhook handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.voice import handler as voice_mod


# =============================================================================
# Helpers
# =============================================================================


def _make_request(params: dict[str, str] | None = None, headers: dict | None = None):
    """Build a mock aiohttp Request with POST data and headers."""
    req = MagicMock()
    post_data = params or {}
    req.post = AsyncMock(return_value=post_data)
    req.headers = headers or {}
    req.url = "https://example.com/api/voice/inbound"
    return req


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_voice():
    """Create a mock TwilioVoiceIntegration."""
    v = MagicMock()
    v.handle_inbound_call.return_value = "<Response><Say>Hello</Say></Response>"
    v.handle_gather_result.return_value = "<Response><Say>Got it</Say></Response>"
    v.handle_confirmation.return_value = "<Response><Say>Confirmed</Say></Response>"
    v.handle_status_callback.return_value = None
    v.config.auto_start_debate = False
    v.config.default_agents = ["claude", "gpt"]
    v.get_session.return_value = None
    v.mark_debate_started.return_value = None
    return v


@pytest.fixture
def handler(mock_voice):
    """Create a VoiceHandler with mock integration."""
    return voice_mod.VoiceHandler(voice_integration=mock_voice)


@pytest.fixture
def handler_with_debate(mock_voice):
    """Create a VoiceHandler with a debate starter."""
    starter = AsyncMock(return_value="debate-123")
    return voice_mod.VoiceHandler(voice_integration=mock_voice, debate_starter=starter)


# =============================================================================
# Test setup_voice_routes
# =============================================================================


class TestSetupVoiceRoutes:
    """Tests for route registration."""

    def test_registers_v1_routes(self, mock_voice):
        app = web.Application()
        h = voice_mod.VoiceHandler(voice_integration=mock_voice)
        voice_mod.setup_voice_routes(app, handler=h)

        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/api/v1/voice/inbound" in routes
        assert "/api/v1/voice/status" in routes
        assert "/api/v1/voice/gather" in routes
        assert "/api/v1/voice/gather/confirm" in routes

    def test_registers_legacy_routes(self, mock_voice):
        app = web.Application()
        h = voice_mod.VoiceHandler(voice_integration=mock_voice)
        voice_mod.setup_voice_routes(app, handler=h)

        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/api/voice/inbound" in routes
        assert "/api/voice/status" in routes
        assert "/api/voice/gather" in routes
        assert "/api/voice/gather/confirm" in routes

    def test_route_count(self, mock_voice):
        app = web.Application()
        h = voice_mod.VoiceHandler(voice_integration=mock_voice)
        voice_mod.setup_voice_routes(app, handler=h)

        routes = list(app.router.routes())
        assert len(routes) == 8  # 4 v1 + 4 legacy


# =============================================================================
# Test _verify_signature
# =============================================================================


class TestVerifySignature:
    """Tests for Twilio signature verification."""

    @pytest.mark.asyncio
    async def test_no_validator_dev_mode_allows(self, handler):
        """Without twilio validator, dev mode allows requests."""
        with patch.object(voice_mod, "HAS_TWILIO_VALIDATOR", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                result = await handler._verify_signature(_make_request(), {})
                assert result is True

    @pytest.mark.asyncio
    async def test_no_validator_production_denies(self, handler):
        """Without twilio validator, production denies requests."""
        with patch.object(voice_mod, "HAS_TWILIO_VALIDATOR", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "production"}, clear=True):
                result = await handler._verify_signature(_make_request(), {})
                assert result is False

    @pytest.mark.asyncio
    async def test_no_auth_token_dev_mode_allows(self, handler):
        """Without TWILIO_AUTH_TOKEN, dev mode allows requests."""
        with patch.object(voice_mod, "HAS_TWILIO_VALIDATOR", True):
            with patch.dict("os.environ", {"ARAGORA_ENV": "test"}, clear=True):
                result = await handler._verify_signature(_make_request(), {})
                assert result is True

    @pytest.mark.asyncio
    async def test_no_auth_token_production_denies(self, handler):
        """Without TWILIO_AUTH_TOKEN, production denies requests."""
        with patch.object(voice_mod, "HAS_TWILIO_VALIDATOR", True):
            with patch.dict("os.environ", {"ARAGORA_ENV": "production"}, clear=True):
                result = await handler._verify_signature(_make_request(), {})
                assert result is False

    @pytest.mark.asyncio
    async def test_missing_signature_header(self, handler):
        """Missing X-Twilio-Signature header returns False."""
        with patch.object(voice_mod, "HAS_TWILIO_VALIDATOR", True):
            with patch.dict(
                "os.environ",
                {"TWILIO_AUTH_TOKEN": "token123", "ARAGORA_ENV": "production"},
                clear=True,
            ):
                req = _make_request(headers={})
                result = await handler._verify_signature(req, {})
                assert result is False

    @pytest.mark.asyncio
    async def test_valid_signature(self, handler):
        """Valid signature returns True."""
        mock_validator = MagicMock()
        mock_validator.validate.return_value = True

        with patch.object(voice_mod, "HAS_TWILIO_VALIDATOR", True):
            with patch.object(voice_mod, "RequestValidator", return_value=mock_validator):
                with patch.dict(
                    "os.environ",
                    {"TWILIO_AUTH_TOKEN": "token123"},
                ):
                    req = _make_request(headers={"X-Twilio-Signature": "sig123"})
                    result = await handler._verify_signature(req, {"k": "v"})
                    assert result is True
                    mock_validator.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_signature(self, handler):
        """Invalid signature returns False."""
        mock_validator = MagicMock()
        mock_validator.validate.return_value = False

        with patch.object(voice_mod, "HAS_TWILIO_VALIDATOR", True):
            with patch.object(voice_mod, "RequestValidator", return_value=mock_validator):
                with patch.dict(
                    "os.environ",
                    {"TWILIO_AUTH_TOKEN": "token123"},
                ):
                    req = _make_request(headers={"X-Twilio-Signature": "bad"})
                    result = await handler._verify_signature(req, {})
                    assert result is False

    @pytest.mark.asyncio
    async def test_validator_exception_returns_false(self, handler):
        """Exception during validation returns False."""
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = RuntimeError("boom")

        with patch.object(voice_mod, "HAS_TWILIO_VALIDATOR", True):
            with patch.object(voice_mod, "RequestValidator", return_value=mock_validator):
                with patch.dict(
                    "os.environ",
                    {"TWILIO_AUTH_TOKEN": "token123"},
                ):
                    req = _make_request(headers={"X-Twilio-Signature": "sig"})
                    result = await handler._verify_signature(req, {})
                    assert result is False


# =============================================================================
# Test handle_inbound
# =============================================================================


class TestHandleInbound:
    """Tests for inbound call handler."""

    @pytest.mark.asyncio
    async def test_twilio_unavailable(self, handler):
        """Returns unavailable TwiML when Twilio not installed."""
        with patch.object(voice_mod, "HAS_TWILIO", False):
            resp = await handler.handle_inbound(_make_request())
            assert "Voice service unavailable" in resp.text
            assert resp.content_type == "application/xml"

    @pytest.mark.asyncio
    async def test_invalid_signature(self, handler):
        """Returns 401 when signature verification fails."""
        with patch.object(voice_mod, "HAS_TWILIO", True):
            with patch.object(handler, "_verify_signature", AsyncMock(return_value=False)):
                resp = await handler.handle_inbound(_make_request())
                assert resp.status == 401
                assert "Unauthorized" in resp.text

    @pytest.mark.asyncio
    async def test_missing_call_sid(self, handler, mock_voice):
        """Returns error TwiML when CallSid is missing."""
        with patch.object(voice_mod, "HAS_TWILIO", True):
            with patch.object(handler, "_verify_signature", AsyncMock(return_value=True)):
                resp = await handler.handle_inbound(_make_request(params={}))
                assert "Invalid request" in resp.text

    @pytest.mark.asyncio
    async def test_success(self, handler, mock_voice):
        """Returns TwiML from voice integration on success."""
        params = {"CallSid": "CA123", "From": "+15551234567", "To": "+15559876543"}
        with patch.object(voice_mod, "HAS_TWILIO", True):
            with patch.object(handler, "_verify_signature", AsyncMock(return_value=True)):
                resp = await handler.handle_inbound(_make_request(params=params))
                assert resp.text == "<Response><Say>Hello</Say></Response>"
                assert resp.content_type == "application/xml"
                mock_voice.handle_inbound_call.assert_called_once_with(
                    call_sid="CA123",
                    caller="+15551234567",
                    called="+15559876543",
                )


# =============================================================================
# Test handle_gather
# =============================================================================


class TestHandleGather:
    """Tests for speech gather result handler."""

    @pytest.mark.asyncio
    async def test_twilio_unavailable(self, handler):
        """Returns unavailable TwiML when Twilio not installed."""
        with patch.object(voice_mod, "HAS_TWILIO", False):
            resp = await handler.handle_gather(_make_request())
            assert "Service unavailable" in resp.text

    @pytest.mark.asyncio
    async def test_invalid_signature(self, handler):
        """Returns 401 when signature fails."""
        with patch.object(voice_mod, "HAS_TWILIO", True):
            with patch.object(handler, "_verify_signature", AsyncMock(return_value=False)):
                resp = await handler.handle_gather(_make_request())
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_success_no_auto_debate(self, handler, mock_voice):
        """Processes gather without starting debate when auto_start disabled."""
        params = {"CallSid": "CA123", "SpeechResult": "What is AI?", "Confidence": "0.95"}
        with patch.object(voice_mod, "HAS_TWILIO", True):
            with patch.object(handler, "_verify_signature", AsyncMock(return_value=True)):
                resp = await handler.handle_gather(_make_request(params=params))
                assert resp.text == "<Response><Say>Got it</Say></Response>"
                mock_voice.handle_gather_result.assert_called_once_with(
                    call_sid="CA123",
                    speech_result="What is AI?",
                    confidence=0.95,
                )

    @pytest.mark.asyncio
    async def test_auto_debate_queued(self, handler, mock_voice):
        """Queues debate when speech result present and auto_start enabled."""
        mock_voice.config.auto_start_debate = True
        params = {"CallSid": "CA123", "SpeechResult": "Debate this topic", "Confidence": "0.9"}

        with patch.object(voice_mod, "HAS_TWILIO", True):
            with patch.object(handler, "_verify_signature", AsyncMock(return_value=True)):
                with patch.object(handler, "_queue_debate_from_voice", AsyncMock()) as mock_queue:
                    resp = await handler.handle_gather(_make_request(params=params))
                    assert resp.content_type == "application/xml"
                    mock_queue.assert_called_once_with("CA123", "Debate this topic")

    @pytest.mark.asyncio
    async def test_no_debate_without_speech(self, handler, mock_voice):
        """Does not queue debate when SpeechResult is empty."""
        mock_voice.config.auto_start_debate = True
        params = {"CallSid": "CA123", "SpeechResult": "", "Confidence": "0"}

        with patch.object(voice_mod, "HAS_TWILIO", True):
            with patch.object(handler, "_verify_signature", AsyncMock(return_value=True)):
                with patch.object(handler, "_queue_debate_from_voice", AsyncMock()) as mock_queue:
                    await handler.handle_gather(_make_request(params=params))
                    mock_queue.assert_not_called()


# =============================================================================
# Test handle_gather_confirm
# =============================================================================


class TestHandleGatherConfirm:
    """Tests for confirmation digit press handler."""

    @pytest.mark.asyncio
    async def test_twilio_unavailable(self, handler):
        with patch.object(voice_mod, "HAS_TWILIO", False):
            resp = await handler.handle_gather_confirm(_make_request())
            assert "Service unavailable" in resp.text

    @pytest.mark.asyncio
    async def test_invalid_signature(self, handler):
        with patch.object(voice_mod, "HAS_TWILIO", True):
            with patch.object(handler, "_verify_signature", AsyncMock(return_value=False)):
                resp = await handler.handle_gather_confirm(_make_request())
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_confirm_digit_1_with_session(self, handler, mock_voice):
        """Digit 1 starts debate when session has transcription."""
        session = MagicMock()
        session.transcription = "What is AI?"
        session.caller = "+15551234567"
        mock_voice.get_session.return_value = session

        params = {"CallSid": "CA123", "Digits": "1"}
        with patch.object(voice_mod, "HAS_TWILIO", True):
            with patch.object(handler, "_verify_signature", AsyncMock(return_value=True)):
                with patch.object(handler, "_queue_debate_from_voice", AsyncMock()) as mock_queue:
                    resp = await handler.handle_gather_confirm(_make_request(params=params))
                    assert resp.content_type == "application/xml"
                    mock_queue.assert_called_once_with("CA123", "What is AI?")

    @pytest.mark.asyncio
    async def test_confirm_digit_1_no_session(self, handler, mock_voice):
        """Digit 1 does not start debate when no session."""
        mock_voice.get_session.return_value = None

        params = {"CallSid": "CA123", "Digits": "1"}
        with patch.object(voice_mod, "HAS_TWILIO", True):
            with patch.object(handler, "_verify_signature", AsyncMock(return_value=True)):
                with patch.object(handler, "_queue_debate_from_voice", AsyncMock()) as mock_queue:
                    await handler.handle_gather_confirm(_make_request(params=params))
                    mock_queue.assert_not_called()

    @pytest.mark.asyncio
    async def test_digit_2_no_debate(self, handler, mock_voice):
        """Digit 2 (retry) does not start debate."""
        params = {"CallSid": "CA123", "Digits": "2"}
        with patch.object(voice_mod, "HAS_TWILIO", True):
            with patch.object(handler, "_verify_signature", AsyncMock(return_value=True)):
                with patch.object(handler, "_queue_debate_from_voice", AsyncMock()) as mock_queue:
                    await handler.handle_gather_confirm(_make_request(params=params))
                    mock_queue.assert_not_called()


# =============================================================================
# Test handle_status
# =============================================================================


class TestHandleStatus:
    """Tests for status callback handler."""

    @pytest.mark.asyncio
    async def test_invalid_signature(self, handler):
        with patch.object(handler, "_verify_signature", AsyncMock(return_value=False)):
            resp = await handler.handle_status(_make_request())
            assert resp.status == 401
            assert resp.text == "Unauthorized"

    @pytest.mark.asyncio
    async def test_success(self, handler, mock_voice):
        params = {
            "CallSid": "CA123",
            "CallStatus": "completed",
            "CallDuration": "60",
            "RecordingUrl": "https://api.twilio.com/rec/123",
        }
        with patch.object(handler, "_verify_signature", AsyncMock(return_value=True)):
            resp = await handler.handle_status(_make_request(params=params))
            assert resp.status == 200
            assert resp.text == "OK"
            mock_voice.handle_status_callback.assert_called_once_with(
                call_sid="CA123",
                call_status="completed",
                duration="60",
                recording_url="https://api.twilio.com/rec/123",
            )


# =============================================================================
# Test _queue_debate_from_voice
# =============================================================================


class TestQueueDebateFromVoice:
    """Tests for debate queuing from voice input."""

    @pytest.mark.asyncio
    async def test_no_debate_starter(self, handler):
        """Does nothing when no debate_starter configured."""
        await handler._queue_debate_from_voice("CA123", "test question")
        # Should not raise

    @pytest.mark.asyncio
    async def test_starts_debate(self, handler_with_debate, mock_voice):
        """Calls debate_starter and marks debate started."""
        session = MagicMock()
        session.caller = "+15551234567"
        mock_voice.get_session.return_value = session

        await handler_with_debate._queue_debate_from_voice("CA123", "What is AI?")

        handler_with_debate.debate_starter.assert_called_once_with(
            task="What is AI?",
            source="voice",
            source_id="CA123",
            callback_number="+15551234567",
            agents=["claude", "gpt"],
        )
        mock_voice.mark_debate_started.assert_called_once_with("CA123", "debate-123")

    @pytest.mark.asyncio
    async def test_debate_starter_returns_none(self, mock_voice):
        """Does not mark debate started when starter returns None."""
        starter = AsyncMock(return_value=None)
        h = voice_mod.VoiceHandler(voice_integration=mock_voice, debate_starter=starter)

        await h._queue_debate_from_voice("CA123", "question")

        mock_voice.mark_debate_started.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_session_uses_unknown_caller(self, handler_with_debate, mock_voice):
        """Uses 'unknown' when session not found."""
        mock_voice.get_session.return_value = None

        await handler_with_debate._queue_debate_from_voice("CA123", "q")

        call_kwargs = handler_with_debate.debate_starter.call_args
        assert call_kwargs.kwargs["callback_number"] == "unknown"

    @pytest.mark.asyncio
    async def test_exception_handled_gracefully(self, mock_voice):
        """Exception in debate_starter is caught and logged."""
        starter = AsyncMock(side_effect=RuntimeError("fail"))
        h = voice_mod.VoiceHandler(voice_integration=mock_voice, debate_starter=starter)

        # Should not raise
        await h._queue_debate_from_voice("CA123", "q")
