"""Comprehensive tests for VoiceHandler (aragora/server/handlers/voice/handler.py).

Covers all routes and methods:

  TestInit                       - Handler initialization and properties
  TestRoutes                     - ROUTES list validation
  TestHandleInbound              - POST /api/v1/voice/inbound
  TestHandleGather               - POST /api/v1/voice/gather
  TestHandleGatherConfirm        - POST /api/v1/voice/gather/confirm
  TestHandleStatus               - POST /api/v1/voice/status
  TestHandleDeviceAssociation    - POST /api/v1/voice/device
  TestAssociateCallWithDevice    - associate_call_with_device() method
  TestGetDeviceForCall           - get_device_for_call() method
  TestGetCallContext             - _get_call_context() helper
  TestVerifySignature            - _verify_signature() logic
  TestQueueDebateFromVoice       - _queue_debate_from_voice() internal
  TestSetupVoiceRoutes           - setup_voice_routes() function
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import make_mocked_request

from aragora.server.handlers.voice.handler import (
    VoiceHandler,
    setup_voice_routes,
    TWIML_CONTENT_TYPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(
    method: str = "POST",
    path: str = "/api/v1/voice/inbound",
    post_data: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock aiohttp request with POST form data or JSON body."""
    from multidict import CIMultiDict

    raw_headers = CIMultiDict(headers or {})
    request = make_mocked_request(method, path, headers=raw_headers)

    # Set up post() to return form data
    if post_data is not None:
        request.post = AsyncMock(return_value=post_data)
    else:
        request.post = AsyncMock(return_value={})

    # Set up json() for JSON body endpoints
    if json_body is not None:
        request.json = AsyncMock(return_value=json_body)
    else:
        request.json = AsyncMock(side_effect=ValueError("No JSON body"))

    return request


def _body(response) -> dict[str, Any]:
    """Parse JSON response body."""
    return json.loads(response.body)


def _status(response) -> int:
    """Get response HTTP status code."""
    return response.status


def _text(response) -> str:
    """Get response text body."""
    return response.text


def _make_voice_mock() -> MagicMock:
    """Create a mock TwilioVoiceIntegration with sensible defaults."""
    voice = MagicMock()
    voice.config = MagicMock()
    voice.config.auto_start_debate = False
    voice.config.default_agents = ["anthropic-api", "openai-api"]
    voice.config.require_confirmation = True

    voice.handle_inbound_call = MagicMock(
        return_value='<?xml version="1.0"?><Response><Say>Hello</Say></Response>'
    )
    voice.handle_gather_result = MagicMock(
        return_value='<?xml version="1.0"?><Response><Say>Got it</Say></Response>'
    )
    voice.handle_confirmation = MagicMock(
        return_value='<?xml version="1.0"?><Response><Say>Confirmed</Say></Response>'
    )
    voice.handle_status_callback = MagicMock()
    voice.get_session = MagicMock(return_value=None)
    voice.mark_debate_started = MagicMock()

    return voice


def _make_device_registry() -> MagicMock:
    """Create a mock DeviceRegistry."""
    registry = MagicMock()
    registry.get = AsyncMock(return_value=None)
    return registry


def _make_device_node(
    device_id: str = "device-001",
    name: str = "Test Phone",
    device_type: str = "phone",
    capabilities: list[str] | None = None,
) -> MagicMock:
    """Create a mock DeviceNode."""
    node = MagicMock()
    node.device_id = device_id
    node.name = name
    node.device_type = device_type
    node.capabilities = capabilities or ["voice", "sms"]
    return node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def voice_mock():
    """Mock TwilioVoiceIntegration."""
    return _make_voice_mock()


@pytest.fixture
def handler(voice_mock):
    """Create a VoiceHandler with mocked voice integration."""
    return VoiceHandler(voice_integration=voice_mock)


@pytest.fixture
def device_registry():
    """Mock DeviceRegistry."""
    return _make_device_registry()


@pytest.fixture
def handler_with_device(voice_mock, device_registry):
    """Create a VoiceHandler with mocked device registry."""
    return VoiceHandler(voice_integration=voice_mock, device_registry=device_registry)


# Patch ARAGORA_ENV to 'test' for signature verification bypass
@pytest.fixture(autouse=True)
def _env_test(monkeypatch):
    """Set ARAGORA_ENV=test so signature checks pass by default."""
    monkeypatch.setenv("ARAGORA_ENV", "test")
    # Ensure TWILIO_AUTH_TOKEN is NOT set (so we skip real signature checks)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """VoiceHandler initialization tests."""

    def test_init_with_voice(self, voice_mock):
        h = VoiceHandler(voice_integration=voice_mock)
        assert h.voice is voice_mock
        assert h.debate_starter is None
        assert h.device_registry is None

    def test_init_with_debate_starter(self, voice_mock):
        starter = AsyncMock()
        h = VoiceHandler(voice_integration=voice_mock, debate_starter=starter)
        assert h.debate_starter is starter

    def test_init_with_device_registry(self, voice_mock, device_registry):
        h = VoiceHandler(voice_integration=voice_mock, device_registry=device_registry)
        assert h.device_registry is device_registry

    @patch("aragora.server.handlers.voice.handler.get_twilio_voice")
    def test_init_default_voice(self, mock_get):
        """When no voice_integration is passed, get_twilio_voice() is called."""
        sentinel = MagicMock()
        mock_get.return_value = sentinel
        h = VoiceHandler()
        assert h.voice is sentinel
        mock_get.assert_called_once()

    def test_call_device_map_starts_empty(self, handler):
        assert handler._call_device_map == {}


# ===========================================================================
# TestRoutes
# ===========================================================================


class TestRoutes:
    """Validate the ROUTES class attribute."""

    def test_routes_contains_v1_inbound(self):
        assert "/api/v1/voice/inbound" in VoiceHandler.ROUTES

    def test_routes_contains_v1_status(self):
        assert "/api/v1/voice/status" in VoiceHandler.ROUTES

    def test_routes_contains_v1_gather(self):
        assert "/api/v1/voice/gather" in VoiceHandler.ROUTES

    def test_routes_contains_v1_gather_confirm(self):
        assert "/api/v1/voice/gather/confirm" in VoiceHandler.ROUTES

    def test_routes_contains_v1_device(self):
        assert "/api/v1/voice/device" in VoiceHandler.ROUTES

    def test_routes_contains_v1_synthesize(self):
        assert "/api/v1/voice/synthesize" in VoiceHandler.ROUTES

    def test_routes_contains_v1_voices(self):
        assert "/api/v1/voice/voices" in VoiceHandler.ROUTES

    def test_routes_contains_v1_sessions(self):
        assert "/api/v1/voice/sessions" in VoiceHandler.ROUTES

    def test_routes_contains_v1_config(self):
        assert "/api/v1/voice/config" in VoiceHandler.ROUTES

    def test_routes_contains_legacy_inbound(self):
        assert "/api/voice/inbound" in VoiceHandler.ROUTES

    def test_routes_contains_legacy_status(self):
        assert "/api/voice/status" in VoiceHandler.ROUTES

    def test_routes_contains_legacy_gather(self):
        assert "/api/voice/gather" in VoiceHandler.ROUTES

    def test_routes_contains_legacy_gather_confirm(self):
        assert "/api/voice/gather/confirm" in VoiceHandler.ROUTES

    def test_routes_has_expected_count(self):
        # 5 v1 webhook + 4 v1 TTS/session + 1 v1 config + 4 legacy = 14
        assert len(VoiceHandler.ROUTES) == 14

    def test_no_duplicate_routes(self):
        assert len(VoiceHandler.ROUTES) == len(set(VoiceHandler.ROUTES))


# ===========================================================================
# TestHandleInbound
# ===========================================================================


class TestHandleInbound:
    """Tests for POST /api/v1/voice/inbound."""

    @pytest.mark.asyncio
    async def test_success(self, handler, voice_mock):
        request = _req(
            post_data={
                "CallSid": "CA123",
                "From": "+15551234567",
                "To": "+15559876543",
            },
        )
        resp = await handler.handle_inbound(request)
        assert _status(resp) == 200
        assert resp.content_type == TWIML_CONTENT_TYPE
        voice_mock.handle_inbound_call.assert_called_once_with(
            call_sid="CA123",
            caller="+15551234567",
            called="+15559876543",
        )

    @pytest.mark.asyncio
    async def test_missing_call_sid(self, handler, voice_mock):
        request = _req(
            post_data={
                "From": "+15551234567",
                "To": "+15559876543",
            },
        )
        resp = await handler.handle_inbound(request)
        assert _status(resp) == 200
        assert "Invalid request" in _text(resp)
        voice_mock.handle_inbound_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_twilio_not_available(self, handler):
        """When HAS_TWILIO is False, return service unavailable TwiML."""
        request = _req(post_data={"CallSid": "CA123"})
        with patch("aragora.server.handlers.voice.handler.HAS_TWILIO", False):
            resp = await handler.handle_inbound(request)
        assert _status(resp) == 200
        assert "Voice service unavailable" in _text(resp)
        assert resp.content_type == TWIML_CONTENT_TYPE

    @pytest.mark.asyncio
    async def test_invalid_signature(self, handler, monkeypatch):
        """When signature check fails, return 401."""
        # Set production env so signature check doesn't auto-pass
        monkeypatch.setenv("ARAGORA_ENV", "production")
        request = _req(
            post_data={
                "CallSid": "CA123",
                "From": "+15551234567",
                "To": "+15559876543",
            },
        )
        resp = await handler.handle_inbound(request)
        assert _status(resp) == 401
        assert "Unauthorized" in _text(resp)

    @pytest.mark.asyncio
    async def test_empty_post_data(self, handler, voice_mock):
        """Empty POST body: CallSid missing -> Invalid request."""
        request = _req(post_data={})
        resp = await handler.handle_inbound(request)
        assert "Invalid request" in _text(resp)
        voice_mock.handle_inbound_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_twiml_response_text(self, handler, voice_mock):
        """Verify the TwiML from handle_inbound_call is returned."""
        expected_twiml = "<Response><Say>Custom greeting</Say></Response>"
        voice_mock.handle_inbound_call.return_value = expected_twiml
        request = _req(
            post_data={
                "CallSid": "CA456",
                "From": "+15550001111",
                "To": "+15550002222",
            },
        )
        resp = await handler.handle_inbound(request)
        assert _text(resp) == expected_twiml


# ===========================================================================
# TestHandleGather
# ===========================================================================


class TestHandleGather:
    """Tests for POST /api/v1/voice/gather."""

    @pytest.mark.asyncio
    async def test_success(self, handler, voice_mock):
        request = _req(
            path="/api/v1/voice/gather",
            post_data={
                "CallSid": "CA123",
                "SpeechResult": "What is the meaning of life?",
                "Confidence": "0.95",
            },
        )
        resp = await handler.handle_gather(request)
        assert _status(resp) == 200
        assert resp.content_type == TWIML_CONTENT_TYPE
        voice_mock.handle_gather_result.assert_called_once_with(
            call_sid="CA123",
            speech_result="What is the meaning of life?",
            confidence=0.95,
        )

    @pytest.mark.asyncio
    async def test_empty_speech_result(self, handler, voice_mock):
        request = _req(
            path="/api/v1/voice/gather",
            post_data={
                "CallSid": "CA123",
                "SpeechResult": "",
                "Confidence": "0",
            },
        )
        resp = await handler.handle_gather(request)
        assert _status(resp) == 200
        voice_mock.handle_gather_result.assert_called_once_with(
            call_sid="CA123",
            speech_result="",
            confidence=0.0,
        )

    @pytest.mark.asyncio
    async def test_missing_confidence_defaults_to_zero(self, handler, voice_mock):
        request = _req(
            path="/api/v1/voice/gather",
            post_data={"CallSid": "CA123", "SpeechResult": "test"},
        )
        resp = await handler.handle_gather(request)
        assert _status(resp) == 200
        voice_mock.handle_gather_result.assert_called_once_with(
            call_sid="CA123",
            speech_result="test",
            confidence=0.0,
        )

    @pytest.mark.asyncio
    async def test_auto_start_debate_enabled(self, handler, voice_mock):
        """When auto_start_debate is True and speech_result present, queue debate."""
        voice_mock.config.auto_start_debate = True
        handler.debate_starter = None  # No starter configured; won't fail

        request = _req(
            path="/api/v1/voice/gather",
            post_data={
                "CallSid": "CA123",
                "SpeechResult": "How do computers work?",
                "Confidence": "0.9",
            },
        )
        resp = await handler.handle_gather(request)
        assert _status(resp) == 200

    @pytest.mark.asyncio
    async def test_auto_start_debate_with_starter(self, voice_mock):
        """When auto_start_debate and debate_starter present, debate is queued."""
        voice_mock.config.auto_start_debate = True
        starter = AsyncMock(return_value="debate-001")
        voice_mock.get_session.return_value = MagicMock(
            caller="+15551234567",
            transcription="How do computers work?",
        )
        h = VoiceHandler(voice_integration=voice_mock, debate_starter=starter)

        request = _req(
            path="/api/v1/voice/gather",
            post_data={
                "CallSid": "CA123",
                "SpeechResult": "How do computers work?",
                "Confidence": "0.9",
            },
        )
        resp = await h.handle_gather(request)
        assert _status(resp) == 200
        starter.assert_called_once()
        voice_mock.mark_debate_started.assert_called_once_with("CA123", "debate-001")

    @pytest.mark.asyncio
    async def test_auto_start_no_speech_no_debate(self, handler, voice_mock):
        """Even with auto_start enabled, empty speech doesn't queue debate."""
        voice_mock.config.auto_start_debate = True
        handler.debate_starter = AsyncMock()

        request = _req(
            path="/api/v1/voice/gather",
            post_data={"CallSid": "CA123", "SpeechResult": ""},
        )
        resp = await handler.handle_gather(request)
        assert _status(resp) == 200
        handler.debate_starter.assert_not_called()

    @pytest.mark.asyncio
    async def test_twilio_not_available(self, handler):
        request = _req(
            path="/api/v1/voice/gather",
            post_data={"CallSid": "CA123", "SpeechResult": "test"},
        )
        with patch("aragora.server.handlers.voice.handler.HAS_TWILIO", False):
            resp = await handler.handle_gather(request)
        assert "Service unavailable" in _text(resp)

    @pytest.mark.asyncio
    async def test_invalid_signature(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        request = _req(
            path="/api/v1/voice/gather",
            post_data={"CallSid": "CA123", "SpeechResult": "test"},
        )
        resp = await handler.handle_gather(request)
        assert _status(resp) == 401
        assert "Unauthorized" in _text(resp)


# ===========================================================================
# TestHandleGatherConfirm
# ===========================================================================


class TestHandleGatherConfirm:
    """Tests for POST /api/v1/voice/gather/confirm."""

    @pytest.mark.asyncio
    async def test_confirm_digit_1(self, handler, voice_mock):
        voice_mock.get_session.return_value = None
        request = _req(
            path="/api/v1/voice/gather/confirm",
            post_data={"CallSid": "CA123", "Digits": "1"},
        )
        resp = await handler.handle_gather_confirm(request)
        assert _status(resp) == 200
        assert resp.content_type == TWIML_CONTENT_TYPE
        voice_mock.handle_confirmation.assert_called_once_with(
            call_sid="CA123",
            digits="1",
        )

    @pytest.mark.asyncio
    async def test_confirm_digit_1_queues_debate(self, voice_mock):
        """Pressing 1 with a transcription queues a debate."""
        session = MagicMock()
        session.transcription = "What is AI?"
        session.caller = "+15551234567"
        voice_mock.get_session.return_value = session

        starter = AsyncMock(return_value="debate-002")
        h = VoiceHandler(voice_integration=voice_mock, debate_starter=starter)

        request = _req(
            path="/api/v1/voice/gather/confirm",
            post_data={"CallSid": "CA123", "Digits": "1"},
        )
        resp = await h.handle_gather_confirm(request)
        assert _status(resp) == 200
        starter.assert_called_once()
        voice_mock.mark_debate_started.assert_called_once_with("CA123", "debate-002")

    @pytest.mark.asyncio
    async def test_confirm_digit_1_no_transcription(self, handler, voice_mock):
        """Pressing 1 but no transcription in session -> no debate queued."""
        session = MagicMock()
        session.transcription = ""
        voice_mock.get_session.return_value = session
        handler.debate_starter = AsyncMock()

        request = _req(
            path="/api/v1/voice/gather/confirm",
            post_data={"CallSid": "CA123", "Digits": "1"},
        )
        resp = await handler.handle_gather_confirm(request)
        assert _status(resp) == 200
        handler.debate_starter.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_digit_2(self, handler, voice_mock):
        request = _req(
            path="/api/v1/voice/gather/confirm",
            post_data={"CallSid": "CA123", "Digits": "2"},
        )
        resp = await handler.handle_gather_confirm(request)
        assert _status(resp) == 200
        voice_mock.handle_confirmation.assert_called_once_with(
            call_sid="CA123",
            digits="2",
        )

    @pytest.mark.asyncio
    async def test_digit_2_does_not_queue_debate(self, handler, voice_mock):
        """Pressing 2 (retry) should NOT start a debate."""
        handler.debate_starter = AsyncMock()
        request = _req(
            path="/api/v1/voice/gather/confirm",
            post_data={"CallSid": "CA123", "Digits": "2"},
        )
        resp = await handler.handle_gather_confirm(request)
        assert _status(resp) == 200
        handler.debate_starter.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_digits(self, handler, voice_mock):
        """No digits pressed -> handler still returns TwiML."""
        request = _req(
            path="/api/v1/voice/gather/confirm",
            post_data={"CallSid": "CA123"},
        )
        resp = await handler.handle_gather_confirm(request)
        assert _status(resp) == 200
        voice_mock.handle_confirmation.assert_called_once_with(
            call_sid="CA123",
            digits="",
        )

    @pytest.mark.asyncio
    async def test_twilio_not_available(self, handler):
        request = _req(
            path="/api/v1/voice/gather/confirm",
            post_data={"CallSid": "CA123", "Digits": "1"},
        )
        with patch("aragora.server.handlers.voice.handler.HAS_TWILIO", False):
            resp = await handler.handle_gather_confirm(request)
        assert "Service unavailable" in _text(resp)

    @pytest.mark.asyncio
    async def test_invalid_signature(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        request = _req(
            path="/api/v1/voice/gather/confirm",
            post_data={"CallSid": "CA123", "Digits": "1"},
        )
        resp = await handler.handle_gather_confirm(request)
        assert _status(resp) == 401


# ===========================================================================
# TestHandleStatus
# ===========================================================================


class TestHandleStatus:
    """Tests for POST /api/v1/voice/status."""

    @pytest.mark.asyncio
    async def test_success(self, handler, voice_mock):
        request = _req(
            path="/api/v1/voice/status",
            post_data={
                "CallSid": "CA123",
                "CallStatus": "completed",
                "CallDuration": "45",
                "RecordingUrl": "https://rec.example.com/abc.wav",
            },
        )
        resp = await handler.handle_status(request)
        assert _status(resp) == 200
        assert _text(resp) == "OK"
        voice_mock.handle_status_callback.assert_called_once_with(
            call_sid="CA123",
            call_status="completed",
            duration="45",
            recording_url="https://rec.example.com/abc.wav",
        )

    @pytest.mark.asyncio
    async def test_ringing_status(self, handler, voice_mock):
        request = _req(
            path="/api/v1/voice/status",
            post_data={
                "CallSid": "CA123",
                "CallStatus": "ringing",
            },
        )
        resp = await handler.handle_status(request)
        assert _status(resp) == 200
        voice_mock.handle_status_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_completed_cleans_device_map(self, handler_with_device):
        """Completed status cleans up device association."""
        handler_with_device._call_device_map["CA123"] = "device-001"
        request = _req(
            path="/api/v1/voice/status",
            post_data={"CallSid": "CA123", "CallStatus": "completed"},
        )
        resp = await handler_with_device.handle_status(request)
        assert _status(resp) == 200
        assert "CA123" not in handler_with_device._call_device_map

    @pytest.mark.asyncio
    async def test_failed_cleans_device_map(self, handler_with_device):
        handler_with_device._call_device_map["CA456"] = "device-002"
        request = _req(
            path="/api/v1/voice/status",
            post_data={"CallSid": "CA456", "CallStatus": "failed"},
        )
        resp = await handler_with_device.handle_status(request)
        assert _status(resp) == 200
        assert "CA456" not in handler_with_device._call_device_map

    @pytest.mark.asyncio
    async def test_busy_cleans_device_map(self, handler_with_device):
        handler_with_device._call_device_map["CA789"] = "device-003"
        request = _req(
            path="/api/v1/voice/status",
            post_data={"CallSid": "CA789", "CallStatus": "busy"},
        )
        resp = await handler_with_device.handle_status(request)
        assert "CA789" not in handler_with_device._call_device_map

    @pytest.mark.asyncio
    async def test_no_answer_cleans_device_map(self, handler_with_device):
        handler_with_device._call_device_map["CA100"] = "device-004"
        request = _req(
            path="/api/v1/voice/status",
            post_data={"CallSid": "CA100", "CallStatus": "no-answer"},
        )
        resp = await handler_with_device.handle_status(request)
        assert "CA100" not in handler_with_device._call_device_map

    @pytest.mark.asyncio
    async def test_canceled_cleans_device_map(self, handler_with_device):
        handler_with_device._call_device_map["CA200"] = "device-005"
        request = _req(
            path="/api/v1/voice/status",
            post_data={"CallSid": "CA200", "CallStatus": "canceled"},
        )
        resp = await handler_with_device.handle_status(request)
        assert "CA200" not in handler_with_device._call_device_map

    @pytest.mark.asyncio
    async def test_initiated_does_not_clean_device_map(self, handler_with_device):
        """Non-terminal statuses should NOT clean device map."""
        handler_with_device._call_device_map["CA300"] = "device-006"
        request = _req(
            path="/api/v1/voice/status",
            post_data={"CallSid": "CA300", "CallStatus": "initiated"},
        )
        resp = await handler_with_device.handle_status(request)
        assert "CA300" in handler_with_device._call_device_map

    @pytest.mark.asyncio
    async def test_missing_fields_still_ok(self, handler, voice_mock):
        """Missing optional fields default to empty strings."""
        request = _req(
            path="/api/v1/voice/status",
            post_data={"CallSid": "CA123"},
        )
        resp = await handler.handle_status(request)
        assert _status(resp) == 200
        voice_mock.handle_status_callback.assert_called_once_with(
            call_sid="CA123",
            call_status="",
            duration="",
            recording_url="",
        )

    @pytest.mark.asyncio
    async def test_invalid_signature(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        request = _req(
            path="/api/v1/voice/status",
            post_data={"CallSid": "CA123", "CallStatus": "completed"},
        )
        resp = await handler.handle_status(request)
        assert _status(resp) == 401
        assert _text(resp) == "Unauthorized"


# ===========================================================================
# TestHandleDeviceAssociation
# ===========================================================================


class TestHandleDeviceAssociation:
    """Tests for POST /api/v1/voice/device."""

    @pytest.mark.asyncio
    async def test_success(self, handler_with_device, device_registry):
        """Successful device association."""
        node = _make_device_node()
        device_registry.get = AsyncMock(return_value=node)

        request = _req(
            path="/api/v1/voice/device",
            json_body={"call_sid": "CA123", "device_id": "device-001"},
        )

        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            resp = await handler_with_device.handle_device_association(request)

        assert _status(resp) == 200
        body = _body(resp)
        assert body["status"] == "associated"
        assert body["call_sid"] == "CA123"
        assert body["device_id"] == "device-001"

    @pytest.mark.asyncio
    async def test_no_device_registry(self, handler):
        """Device runtime not available -> 503."""
        request = _req(
            path="/api/v1/voice/device",
            json_body={"call_sid": "CA123", "device_id": "device-001"},
        )
        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", False):
            resp = await handler.handle_device_association(request)
        assert _status(resp) == 503
        assert "not available" in _body(resp)["error"]

    @pytest.mark.asyncio
    async def test_invalid_json(self, handler_with_device):
        """Invalid JSON body -> 400."""
        request = _req(path="/api/v1/voice/device")
        request.json = AsyncMock(side_effect=ValueError("bad json"))

        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            resp = await handler_with_device.handle_device_association(request)

        assert _status(resp) == 400
        assert "Invalid JSON" in _body(resp)["error"]

    @pytest.mark.asyncio
    async def test_missing_call_sid(self, handler_with_device):
        request = _req(
            path="/api/v1/voice/device",
            json_body={"device_id": "device-001"},
        )
        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            resp = await handler_with_device.handle_device_association(request)
        assert _status(resp) == 400
        assert "required" in _body(resp)["error"]

    @pytest.mark.asyncio
    async def test_missing_device_id(self, handler_with_device):
        request = _req(
            path="/api/v1/voice/device",
            json_body={"call_sid": "CA123"},
        )
        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            resp = await handler_with_device.handle_device_association(request)
        assert _status(resp) == 400
        assert "required" in _body(resp)["error"]

    @pytest.mark.asyncio
    async def test_association_fails(self, handler_with_device, device_registry):
        """When associate_call_with_device returns False -> 400."""
        device_registry.get = AsyncMock(return_value=None)  # Device not found

        request = _req(
            path="/api/v1/voice/device",
            json_body={"call_sid": "CA123", "device_id": "nonexistent"},
        )
        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            resp = await handler_with_device.handle_device_association(request)
        assert _status(resp) == 400
        assert "Failed" in _body(resp)["error"]

    @pytest.mark.asyncio
    async def test_device_without_voice_capability(self, handler_with_device, device_registry):
        """Device without voice capability -> association fails."""
        node = _make_device_node(capabilities=["sms", "email"])
        device_registry.get = AsyncMock(return_value=node)

        request = _req(
            path="/api/v1/voice/device",
            json_body={"call_sid": "CA123", "device_id": "device-001"},
        )
        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            resp = await handler_with_device.handle_device_association(request)
        assert _status(resp) == 400

    @pytest.mark.asyncio
    async def test_type_error_json(self, handler_with_device):
        """TypeError from request.json() -> 400."""
        request = _req(path="/api/v1/voice/device")
        request.json = AsyncMock(side_effect=TypeError("no body"))

        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            resp = await handler_with_device.handle_device_association(request)
        assert _status(resp) == 400
        assert "Invalid JSON" in _body(resp)["error"]


# ===========================================================================
# TestAssociateCallWithDevice
# ===========================================================================


class TestAssociateCallWithDevice:
    """Tests for associate_call_with_device() method."""

    @pytest.mark.asyncio
    async def test_success(self, handler_with_device, device_registry):
        node = _make_device_node()
        device_registry.get = AsyncMock(return_value=node)

        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            result = await handler_with_device.associate_call_with_device("CA123", "device-001")
        assert result is True
        assert handler_with_device._call_device_map["CA123"] == "device-001"

    @pytest.mark.asyncio
    async def test_no_registry(self, handler):
        result = await handler.associate_call_with_device("CA123", "device-001")
        assert result is False

    @pytest.mark.asyncio
    async def test_device_not_found(self, handler_with_device, device_registry):
        device_registry.get = AsyncMock(return_value=None)

        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            result = await handler_with_device.associate_call_with_device("CA123", "nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_device_lacks_voice(self, handler_with_device, device_registry):
        node = _make_device_node(capabilities=["sms"])
        device_registry.get = AsyncMock(return_value=node)

        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            result = await handler_with_device.associate_call_with_device("CA123", "device-001")
        assert result is False

    @pytest.mark.asyncio
    async def test_has_device_registry_false(self, handler_with_device):
        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", False):
            result = await handler_with_device.associate_call_with_device("CA123", "device-001")
        assert result is False


# ===========================================================================
# TestGetDeviceForCall
# ===========================================================================


class TestGetDeviceForCall:
    """Tests for get_device_for_call() method."""

    @pytest.mark.asyncio
    async def test_found(self, handler_with_device, device_registry):
        node = _make_device_node()
        device_registry.get = AsyncMock(return_value=node)
        handler_with_device._call_device_map["CA123"] = "device-001"

        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            result = await handler_with_device.get_device_for_call("CA123")
        assert result is node

    @pytest.mark.asyncio
    async def test_no_mapping(self, handler_with_device, device_registry):
        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            result = await handler_with_device.get_device_for_call("CA999")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_registry(self, handler):
        result = await handler.get_device_for_call("CA123")
        assert result is None

    @pytest.mark.asyncio
    async def test_has_device_registry_false(self, handler_with_device):
        handler_with_device._call_device_map["CA123"] = "device-001"
        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", False):
            result = await handler_with_device.get_device_for_call("CA123")
        assert result is None


# ===========================================================================
# TestGetCallContext
# ===========================================================================


class TestGetCallContext:
    """Tests for _get_call_context() helper."""

    def test_basic(self, handler):
        ctx = handler._get_call_context("CA123")
        assert ctx == {"call_sid": "CA123"}

    def test_with_device(self, handler):
        handler._call_device_map["CA123"] = "device-001"
        ctx = handler._get_call_context("CA123")
        assert ctx["call_sid"] == "CA123"
        assert ctx["device_id"] == "device-001"

    def test_no_device(self, handler):
        ctx = handler._get_call_context("CA999")
        assert "device_id" not in ctx


# ===========================================================================
# TestVerifySignature
# ===========================================================================


class TestVerifySignature:
    """Tests for _verify_signature() logic."""

    @pytest.mark.asyncio
    async def test_dev_env_no_validator(self, handler, monkeypatch):
        """In test env, no validator -> allows requests."""
        monkeypatch.setenv("ARAGORA_ENV", "test")
        request = _req(post_data={})

        with patch("aragora.server.handlers.voice.handler.HAS_TWILIO_VALIDATOR", False):
            result = await handler._verify_signature(request, {})
        assert result is True

    @pytest.mark.asyncio
    async def test_prod_env_no_validator(self, handler, monkeypatch):
        """In production env, no validator -> blocks requests."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        request = _req(post_data={})

        with patch("aragora.server.handlers.voice.handler.HAS_TWILIO_VALIDATOR", False):
            result = await handler._verify_signature(request, {})
        assert result is False

    @pytest.mark.asyncio
    async def test_dev_env_no_auth_token(self, handler, monkeypatch):
        """In dev env, validator present but no auth token -> allows."""
        monkeypatch.setenv("ARAGORA_ENV", "development")
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        request = _req(post_data={})

        with (
            patch("aragora.server.handlers.voice.handler.HAS_TWILIO_VALIDATOR", True),
            patch("aragora.server.handlers.voice.handler.RequestValidator", MagicMock()),
        ):
            result = await handler._verify_signature(request, {})
        assert result is True

    @pytest.mark.asyncio
    async def test_prod_env_no_auth_token(self, handler, monkeypatch):
        """In production env, no auth token -> blocks."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        request = _req(post_data={})

        with (
            patch("aragora.server.handlers.voice.handler.HAS_TWILIO_VALIDATOR", True),
            patch("aragora.server.handlers.voice.handler.RequestValidator", MagicMock()),
        ):
            result = await handler._verify_signature(request, {})
        assert result is False

    @pytest.mark.asyncio
    async def test_missing_signature_header(self, handler, monkeypatch):
        """Auth token present but no X-Twilio-Signature header -> False."""
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret-token")
        request = _req(post_data={})
        # Ensure no signature header
        if "X-Twilio-Signature" in request.headers:
            del request.headers["X-Twilio-Signature"]

        with (
            patch("aragora.server.handlers.voice.handler.HAS_TWILIO_VALIDATOR", True),
            patch("aragora.server.handlers.voice.handler.RequestValidator", MagicMock()),
        ):
            result = await handler._verify_signature(request, {})
        assert result is False

    @pytest.mark.asyncio
    async def test_valid_signature(self, handler, monkeypatch):
        """With auth token and valid signature -> True."""
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret-token")

        mock_validator_cls = MagicMock()
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate.return_value = True
        mock_validator_cls.return_value = mock_validator_instance

        request = _req(
            post_data={"CallSid": "CA123"},
            headers={"X-Twilio-Signature": "valid-sig"},
        )

        with (
            patch("aragora.server.handlers.voice.handler.HAS_TWILIO_VALIDATOR", True),
            patch(
                "aragora.server.handlers.voice.handler.RequestValidator",
                mock_validator_cls,
            ),
        ):
            result = await handler._verify_signature(request, {"CallSid": "CA123"})
        assert result is True
        mock_validator_cls.assert_called_once_with("secret-token")
        mock_validator_instance.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_signature(self, handler, monkeypatch):
        """With auth token but invalid signature -> False."""
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret-token")

        mock_validator_cls = MagicMock()
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate.return_value = False
        mock_validator_cls.return_value = mock_validator_instance

        request = _req(
            post_data={"CallSid": "CA123"},
            headers={"X-Twilio-Signature": "bad-sig"},
        )

        with (
            patch("aragora.server.handlers.voice.handler.HAS_TWILIO_VALIDATOR", True),
            patch(
                "aragora.server.handlers.voice.handler.RequestValidator",
                mock_validator_cls,
            ),
        ):
            result = await handler._verify_signature(request, {"CallSid": "CA123"})
        assert result is False

    @pytest.mark.asyncio
    async def test_validator_raises_value_error(self, handler, monkeypatch):
        """If validator.validate() raises ValueError -> False."""
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret-token")

        mock_validator_cls = MagicMock()
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate.side_effect = ValueError("bad params")
        mock_validator_cls.return_value = mock_validator_instance

        request = _req(
            post_data={},
            headers={"X-Twilio-Signature": "some-sig"},
        )

        with (
            patch("aragora.server.handlers.voice.handler.HAS_TWILIO_VALIDATOR", True),
            patch(
                "aragora.server.handlers.voice.handler.RequestValidator",
                mock_validator_cls,
            ),
        ):
            result = await handler._verify_signature(request, {})
        assert result is False

    @pytest.mark.asyncio
    async def test_validator_raises_type_error(self, handler, monkeypatch):
        """If validator.validate() raises TypeError -> False."""
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret-token")

        mock_validator_cls = MagicMock()
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate.side_effect = TypeError("wrong types")
        mock_validator_cls.return_value = mock_validator_instance

        request = _req(
            post_data={},
            headers={"X-Twilio-Signature": "some-sig"},
        )

        with (
            patch("aragora.server.handlers.voice.handler.HAS_TWILIO_VALIDATOR", True),
            patch(
                "aragora.server.handlers.voice.handler.RequestValidator",
                mock_validator_cls,
            ),
        ):
            result = await handler._verify_signature(request, {})
        assert result is False

    @pytest.mark.asyncio
    async def test_validator_raises_runtime_error(self, handler, monkeypatch):
        """If validator.validate() raises RuntimeError -> False."""
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret-token")

        mock_validator_cls = MagicMock()
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate.side_effect = RuntimeError("broken")
        mock_validator_cls.return_value = mock_validator_instance

        request = _req(
            post_data={},
            headers={"X-Twilio-Signature": "some-sig"},
        )

        with (
            patch("aragora.server.handlers.voice.handler.HAS_TWILIO_VALIDATOR", True),
            patch(
                "aragora.server.handlers.voice.handler.RequestValidator",
                mock_validator_cls,
            ),
        ):
            result = await handler._verify_signature(request, {})
        assert result is False

    @pytest.mark.asyncio
    async def test_local_env_allowed(self, handler, monkeypatch):
        """ARAGORA_ENV=local also allows bypass."""
        monkeypatch.setenv("ARAGORA_ENV", "local")
        request = _req(post_data={})

        with patch("aragora.server.handlers.voice.handler.HAS_TWILIO_VALIDATOR", False):
            result = await handler._verify_signature(request, {})
        assert result is True


# ===========================================================================
# TestQueueDebateFromVoice
# ===========================================================================


class TestQueueDebateFromVoice:
    """Tests for _queue_debate_from_voice() internal method."""

    @pytest.mark.asyncio
    async def test_no_debate_starter(self, handler, voice_mock):
        """If no debate_starter configured, returns silently."""
        handler.debate_starter = None
        # Should not raise
        await handler._queue_debate_from_voice("CA123", "What is AI?")

    @pytest.mark.asyncio
    async def test_debate_started_successfully(self, voice_mock):
        starter = AsyncMock(return_value="debate-100")
        session = MagicMock()
        session.caller = "+15551234567"
        voice_mock.get_session.return_value = session

        h = VoiceHandler(voice_integration=voice_mock, debate_starter=starter)
        await h._queue_debate_from_voice("CA123", "What is consciousness?")

        starter.assert_called_once()
        call_kwargs = starter.call_args[1]
        assert call_kwargs["task"] == "What is consciousness?"
        assert call_kwargs["source"] == "voice"
        assert call_kwargs["source_id"] == "CA123"
        assert call_kwargs["callback_number"] == "+15551234567"
        voice_mock.mark_debate_started.assert_called_once_with("CA123", "debate-100")

    @pytest.mark.asyncio
    async def test_debate_starter_returns_none(self, voice_mock):
        """If debate_starter returns None, mark_debate_started is NOT called."""
        starter = AsyncMock(return_value=None)
        voice_mock.get_session.return_value = MagicMock(caller="+15551234567")
        h = VoiceHandler(voice_integration=voice_mock, debate_starter=starter)

        await h._queue_debate_from_voice("CA123", "test question")
        voice_mock.mark_debate_started.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_session_uses_unknown_caller(self, voice_mock):
        """If get_session returns None, caller defaults to 'unknown'."""
        starter = AsyncMock(return_value="debate-200")
        voice_mock.get_session.return_value = None
        h = VoiceHandler(voice_integration=voice_mock, debate_starter=starter)

        await h._queue_debate_from_voice("CA123", "test question")
        call_kwargs = starter.call_args[1]
        assert call_kwargs["callback_number"] == "unknown"

    @pytest.mark.asyncio
    async def test_with_device_context(self, voice_mock, device_registry):
        """Device info is included in debate context."""
        starter = AsyncMock(return_value="debate-300")
        voice_mock.get_session.return_value = MagicMock(caller="+15551234567")

        node = _make_device_node(
            name="Office Phone",
            device_type="voip",
            capabilities=["voice", "sms"],
        )
        device_registry.get = AsyncMock(return_value=node)

        h = VoiceHandler(
            voice_integration=voice_mock,
            debate_starter=starter,
            device_registry=device_registry,
        )
        h._call_device_map["CA123"] = "device-001"

        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            await h._queue_debate_from_voice("CA123", "test question")

        call_kwargs = starter.call_args[1]
        ctx = call_kwargs["context"]
        assert ctx["device_id"] == "device-001"
        assert ctx["device_name"] == "Office Phone"
        assert ctx["device_type"] == "voip"
        assert ctx["device_capabilities"] == ["voice", "sms"]

    @pytest.mark.asyncio
    async def test_debate_starter_raises(self, voice_mock):
        """If debate_starter raises, error is caught gracefully."""
        starter = AsyncMock(side_effect=RuntimeError("starter failed"))
        voice_mock.get_session.return_value = MagicMock(caller="+15551234567")
        h = VoiceHandler(voice_integration=voice_mock, debate_starter=starter)

        # Should not raise
        await h._queue_debate_from_voice("CA123", "test question")
        voice_mock.mark_debate_started.assert_not_called()

    @pytest.mark.asyncio
    async def test_debate_starter_raises_value_error(self, voice_mock):
        starter = AsyncMock(side_effect=ValueError("bad value"))
        voice_mock.get_session.return_value = MagicMock(caller="+15551234567")
        h = VoiceHandler(voice_integration=voice_mock, debate_starter=starter)

        await h._queue_debate_from_voice("CA123", "test")
        voice_mock.mark_debate_started.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_default_agents(self, voice_mock):
        """Default agents from config are passed to debate_starter."""
        voice_mock.config.default_agents = ["custom-agent-1", "custom-agent-2"]
        starter = AsyncMock(return_value="debate-400")
        voice_mock.get_session.return_value = MagicMock(caller="+15551234567")
        h = VoiceHandler(voice_integration=voice_mock, debate_starter=starter)

        await h._queue_debate_from_voice("CA123", "test question")
        call_kwargs = starter.call_args[1]
        assert call_kwargs["agents"] == ["custom-agent-1", "custom-agent-2"]


# ===========================================================================
# TestSetupVoiceRoutes
# ===========================================================================


class TestSetupVoiceRoutes:
    """Tests for setup_voice_routes() function."""

    def test_returns_handler(self):
        app = MagicMock()
        app.router = MagicMock()
        app.router.add_post = MagicMock()

        result = setup_voice_routes(app)
        assert isinstance(result, VoiceHandler)

    def test_uses_provided_handler(self, handler):
        app = MagicMock()
        app.router = MagicMock()
        app.router.add_post = MagicMock()

        result = setup_voice_routes(app, handler=handler)
        assert result is handler

    def test_registers_v1_routes(self):
        app = MagicMock()
        app.router = MagicMock()
        app.router.add_post = MagicMock()

        setup_voice_routes(app)

        # Collect all registered paths
        registered_paths = [call.args[0] for call in app.router.add_post.call_args_list]
        assert "/api/v1/voice/inbound" in registered_paths
        assert "/api/v1/voice/status" in registered_paths
        assert "/api/v1/voice/gather" in registered_paths
        assert "/api/v1/voice/gather/confirm" in registered_paths

    def test_registers_legacy_routes(self):
        app = MagicMock()
        app.router = MagicMock()
        app.router.add_post = MagicMock()

        setup_voice_routes(app)

        registered_paths = [call.args[0] for call in app.router.add_post.call_args_list]
        assert "/api/voice/inbound" in registered_paths
        assert "/api/voice/status" in registered_paths
        assert "/api/voice/gather" in registered_paths
        assert "/api/voice/gather/confirm" in registered_paths

    def test_device_route_registered_when_registry_available(self, voice_mock):
        app = MagicMock()
        app.router = MagicMock()
        app.router.add_post = MagicMock()

        registry = _make_device_registry()
        h = VoiceHandler(voice_integration=voice_mock, device_registry=registry)

        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            setup_voice_routes(app, handler=h)

        registered_paths = [call.args[0] for call in app.router.add_post.call_args_list]
        assert "/api/v1/voice/device" in registered_paths

    def test_device_route_not_registered_when_no_registry(self):
        app = MagicMock()
        app.router = MagicMock()
        app.router.add_post = MagicMock()

        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", False):
            setup_voice_routes(app)

        registered_paths = [call.args[0] for call in app.router.add_post.call_args_list]
        assert "/api/v1/voice/device" not in registered_paths

    def test_total_route_count_without_device(self):
        app = MagicMock()
        app.router = MagicMock()
        app.router.add_post = MagicMock()

        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", False):
            setup_voice_routes(app)

        # 4 v1 + 4 legacy = 8
        assert app.router.add_post.call_count == 8

    def test_accepts_device_registry_kwarg(self, voice_mock):
        app = MagicMock()
        app.router = MagicMock()
        app.router.add_post = MagicMock()

        registry = _make_device_registry()
        with patch("aragora.server.handlers.voice.handler.HAS_DEVICE_REGISTRY", True):
            h = setup_voice_routes(app, device_registry=registry)

        # The handler was created with the device registry
        assert h.device_registry is registry


# ===========================================================================
# TestGetPostParams
# ===========================================================================


class TestGetPostParams:
    """Tests for _get_post_params() method."""

    @pytest.mark.asyncio
    async def test_normal_post_data(self, handler):
        request = _req(
            post_data={"CallSid": "CA123", "From": "+15551234567"},
        )
        params = await handler._get_post_params(request)
        assert params == {"CallSid": "CA123", "From": "+15551234567"}

    @pytest.mark.asyncio
    async def test_empty_post_data(self, handler):
        request = _req(post_data={})
        params = await handler._get_post_params(request)
        assert params == {}

    @pytest.mark.asyncio
    async def test_post_raises_value_error(self, handler):
        request = _req()
        request.post = AsyncMock(side_effect=ValueError("bad data"))
        params = await handler._get_post_params(request)
        assert params == {}

    @pytest.mark.asyncio
    async def test_post_raises_type_error(self, handler):
        request = _req()
        request.post = AsyncMock(side_effect=TypeError("wrong type"))
        params = await handler._get_post_params(request)
        assert params == {}

    @pytest.mark.asyncio
    async def test_post_raises_os_error(self, handler):
        request = _req()
        request.post = AsyncMock(side_effect=OSError("connection reset"))
        params = await handler._get_post_params(request)
        assert params == {}
