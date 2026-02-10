"""
Twilio Voice Webhook Handler.

Handles incoming voice webhooks from Twilio:
- /api/voice/inbound - Inbound call handling
- /api/voice/status - Call status updates
- /api/voice/gather - Speech recognition results
- /api/voice/gather/confirm - Confirmation input

All endpoints return TwiML responses.

Device Runtime Integration:
- Voice sessions can be associated with registered devices
- Device capabilities determine voice routing options
- Call metadata includes device context when available
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from aiohttp import web

from aragora.integrations.twilio_voice import (
    TwilioVoiceIntegration,
    get_twilio_voice,
    HAS_TWILIO,
)

# Try to import device registry for device runtime integration
_DeviceRegistryCls: type[Any] | None = None
_DeviceNodeCls: type[Any] | None = None

try:
    from aragora.gateway.device_registry import (
        DeviceRegistry as _DR,
        DeviceNode as _DN,
    )

    _DeviceRegistryCls = _DR
    _DeviceNodeCls = _DN
    HAS_DEVICE_REGISTRY = True
except ImportError:
    HAS_DEVICE_REGISTRY = False

# Try to import Twilio request validator
_RequestValidatorCls: type[Any] | None = None

try:
    from twilio.request_validator import RequestValidator as _RV

    _RequestValidatorCls = _RV
    HAS_TWILIO_VALIDATOR = True
except ImportError:
    HAS_TWILIO_VALIDATOR = False

# Expose for test patching even when Twilio isn't installed
RequestValidator = _RequestValidatorCls

if TYPE_CHECKING:
    from aiohttp.web import Application, Request, Response
    from aragora.gateway.device_registry import DeviceRegistry, DeviceNode

logger = logging.getLogger(__name__)

# Content type for TwiML responses
TWIML_CONTENT_TYPE = "application/xml"


class VoiceHandler:
    """
    Handler for Twilio Voice webhooks.

    Provides endpoints for:
    - Inbound call handling with speech-to-text
    - Call status tracking
    - Debate initiation from voice input
    - Device runtime integration for device-associated calls
    """

    def __init__(
        self,
        voice_integration: TwilioVoiceIntegration | None = None,
        debate_starter: Any | None = None,
        device_registry: DeviceRegistry | None = None,
    ):
        """
        Initialize voice handler.

        Args:
            voice_integration: TwilioVoiceIntegration instance
            debate_starter: Optional callable to start debates from voice input
            device_registry: Optional DeviceRegistry for device runtime integration
        """
        self.voice = voice_integration or get_twilio_voice()
        self.debate_starter = debate_starter
        self._device_registry: DeviceRegistry | None = device_registry
        self._call_device_map: dict[str, str] = {}  # call_sid -> device_id

    @property
    def device_registry(self) -> DeviceRegistry | None:
        """Get the device registry if available."""
        return self._device_registry

    async def associate_call_with_device(self, call_sid: str, device_id: str) -> bool:
        """
        Associate a voice call with a registered device.

        Args:
            call_sid: Twilio call SID
            device_id: Device identifier from the device registry

        Returns:
            True if association succeeded, False otherwise
        """
        if not HAS_DEVICE_REGISTRY or not self._device_registry:
            logger.debug("Device registry not available for call association")
            return False

        # Verify device exists and has voice capability
        registry = self._device_registry
        device = await registry.get(device_id)  # type: ignore[union-attr]
        if not device:
            logger.warning(f"Device {device_id} not found for call {call_sid}")
            return False

        if "voice" not in device.capabilities:
            logger.warning(f"Device {device_id} lacks voice capability for call {call_sid}")
            return False

        self._call_device_map[call_sid] = device_id
        logger.info(f"Associated call {call_sid} with device {device_id}")
        return True

    async def get_device_for_call(self, call_sid: str) -> DeviceNode | None:
        """
        Get the device associated with a call.

        Args:
            call_sid: Twilio call SID

        Returns:
            DeviceNode if associated, None otherwise
        """
        if not HAS_DEVICE_REGISTRY or not self._device_registry:
            return None

        device_id = self._call_device_map.get(call_sid)
        if not device_id:
            return None

        registry = self._device_registry
        return await registry.get(device_id)  # type: ignore[union-attr]

    def _get_call_context(self, call_sid: str) -> dict[str, Any]:
        """
        Build context dict for a call including device info if available.

        Args:
            call_sid: Twilio call SID

        Returns:
            Dict with call context
        """
        context: dict[str, Any] = {"call_sid": call_sid}

        device_id = self._call_device_map.get(call_sid)
        if device_id:
            context["device_id"] = device_id

        return context

    async def _verify_signature(self, request: "Request", params: dict[str, str]) -> bool:
        """
        Verify Twilio webhook signature.

        Args:
            request: The incoming request
            params: POST parameters from the request

        Returns:
            True if signature is valid, False otherwise
        """
        # Check if validator is available
        if not HAS_TWILIO_VALIDATOR or RequestValidator is None:
            logger.warning("Twilio validator not available, skipping signature check")
            # In production, you should fail closed - return False
            # For development, we allow requests through
            return os.environ.get("ARAGORA_ENV", "").lower() in (
                "development",
                "dev",
                "local",
                "test",
            )

        # Get auth token from environment
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        if not auth_token:
            logger.warning("TWILIO_AUTH_TOKEN not configured, skipping signature check")
            return os.environ.get("ARAGORA_ENV", "").lower() in (
                "development",
                "dev",
                "local",
                "test",
            )

        # Get signature from header
        signature = request.headers.get("X-Twilio-Signature", "")
        if not signature:
            logger.warning("Missing Twilio signature header")
            return False

        # Get full URL (Twilio signs against the full URL)
        url = str(request.url)

        # Validate the signature
        try:
            validator = RequestValidator(auth_token)
            is_valid = validator.validate(url, params, signature)

            if not is_valid:
                logger.warning(f"Invalid Twilio signature for {url}")

            return is_valid
        except Exception as e:
            logger.error(f"Twilio signature verification failed: {e}")
            return False

    async def _get_post_params(self, request: "Request") -> dict[str, str]:
        """Extract POST parameters from request."""
        try:
            data = await request.post()
            return {k: str(v) for k, v in data.items()}
        except Exception as e:
            logger.debug(f"Failed to parse POST data: {e}")
            return {}

    # =========================================================================
    # Inbound Call Handler
    # =========================================================================

    async def handle_inbound(self, request: "Request") -> "Response":
        """
        Handle inbound call webhook.

        POST /api/voice/inbound

        Twilio sends this when a call comes in. Returns TwiML to:
        1. Greet the caller
        2. Prompt for their question
        3. Gather speech input
        """
        if not HAS_TWILIO:
            return web.Response(
                text='<?xml version="1.0"?><Response><Say>Voice service unavailable.</Say></Response>',
                content_type=TWIML_CONTENT_TYPE,
            )

        params = await self._get_post_params(request)

        # Verify Twilio signature
        if not await self._verify_signature(request, params):
            logger.warning("Unauthorized voice webhook request - invalid signature")
            return web.Response(
                text='<?xml version="1.0"?><Response><Say>Unauthorized.</Say></Response>',
                status=401,
                content_type=TWIML_CONTENT_TYPE,
            )

        call_sid = params.get("CallSid", "")
        caller = params.get("From", "")
        called = params.get("To", "")

        if not call_sid:
            logger.warning("Inbound call missing CallSid")
            return web.Response(
                text='<?xml version="1.0"?><Response><Say>Invalid request.</Say></Response>',
                content_type=TWIML_CONTENT_TYPE,
            )

        twiml = self.voice.handle_inbound_call(
            call_sid=call_sid,
            caller=caller,
            called=called,
        )

        return web.Response(text=twiml, content_type=TWIML_CONTENT_TYPE)

    # =========================================================================
    # Gather (Speech Recognition) Handler
    # =========================================================================

    async def handle_gather(self, request: "Request") -> "Response":
        """
        Handle speech gather result.

        POST /api/voice/gather

        Called when caller finishes speaking. Processes transcription
        and optionally starts a debate.
        """
        if not HAS_TWILIO:
            return web.Response(
                text='<?xml version="1.0"?><Response><Say>Service unavailable.</Say></Response>',
                content_type=TWIML_CONTENT_TYPE,
            )

        params = await self._get_post_params(request)

        # Verify Twilio signature
        if not await self._verify_signature(request, params):
            logger.warning("Unauthorized gather webhook request - invalid signature")
            return web.Response(
                text='<?xml version="1.0"?><Response><Say>Unauthorized.</Say></Response>',
                status=401,
                content_type=TWIML_CONTENT_TYPE,
            )

        call_sid = params.get("CallSid", "")
        speech_result = params.get("SpeechResult", "")
        confidence = float(params.get("Confidence", "0") or "0")

        twiml = self.voice.handle_gather_result(
            call_sid=call_sid,
            speech_result=speech_result,
            confidence=confidence,
        )

        # If we have a transcription and auto-start is enabled, queue debate
        if speech_result and self.voice.config.auto_start_debate:
            await self._queue_debate_from_voice(call_sid, speech_result)

        return web.Response(text=twiml, content_type=TWIML_CONTENT_TYPE)

    async def handle_gather_confirm(self, request: "Request") -> "Response":
        """
        Handle confirmation digit press.

        POST /api/voice/gather/confirm

        Called when user presses 1 (confirm) or 2 (retry).
        """
        if not HAS_TWILIO:
            return web.Response(
                text='<?xml version="1.0"?><Response><Say>Service unavailable.</Say></Response>',
                content_type=TWIML_CONTENT_TYPE,
            )

        params = await self._get_post_params(request)

        # Verify Twilio signature
        if not await self._verify_signature(request, params):
            logger.warning("Unauthorized confirm webhook request - invalid signature")
            return web.Response(
                text='<?xml version="1.0"?><Response><Say>Unauthorized.</Say></Response>',
                status=401,
                content_type=TWIML_CONTENT_TYPE,
            )

        call_sid = params.get("CallSid", "")
        digits = params.get("Digits", "")

        twiml = self.voice.handle_confirmation(
            call_sid=call_sid,
            digits=digits,
        )

        # If confirmed (digit 1), start debate
        if digits == "1":
            session = self.voice.get_session(call_sid)
            if session and session.transcription:
                await self._queue_debate_from_voice(call_sid, session.transcription)

        return web.Response(text=twiml, content_type=TWIML_CONTENT_TYPE)

    # =========================================================================
    # Status Callback Handler
    # =========================================================================

    async def handle_status(self, request: "Request") -> "Response":
        """
        Handle call status callback.

        POST /api/voice/status

        Twilio sends this for call status updates.
        """
        params = await self._get_post_params(request)

        # Verify Twilio signature
        if not await self._verify_signature(request, params):
            logger.warning("Unauthorized status webhook request - invalid signature")
            return web.Response(text="Unauthorized", status=401)

        call_sid = params.get("CallSid", "")
        call_status = params.get("CallStatus", "")
        call_duration = params.get("CallDuration", "")
        recording_url = params.get("RecordingUrl", "")

        self.voice.handle_status_callback(
            call_sid=call_sid,
            call_status=call_status,
            duration=call_duration,
            recording_url=recording_url,
        )

        # Clean up device association when call ends
        if call_status in ("completed", "failed", "busy", "no-answer", "canceled"):
            device_id = self._call_device_map.pop(call_sid, None)
            if device_id:
                logger.debug(f"Removed device association for ended call {call_sid}")

        # Return empty 200 OK
        return web.Response(text="OK", status=200)

    # =========================================================================
    # Device Runtime Integration
    # =========================================================================

    async def handle_device_association(self, request: "Request") -> "Response":
        """
        Associate a call with a device.

        POST /api/v1/voice/device
        Body: {"call_sid": "...", "device_id": "..."}

        This endpoint allows devices to register themselves as the source
        of a voice call, enabling device-specific routing and context.
        """
        if not HAS_DEVICE_REGISTRY or not self._device_registry:
            return web.json_response(
                {"error": "Device runtime not available"},
                status=503,
            )

        try:
            data = await request.json()
        except Exception:
            logger.debug("Invalid JSON in voice handler request", exc_info=True)
            return web.json_response(
                {"error": "Invalid JSON"},
                status=400,
            )

        call_sid = data.get("call_sid")
        device_id = data.get("device_id")

        if not call_sid or not device_id:
            return web.json_response(
                {"error": "call_sid and device_id are required"},
                status=400,
            )

        success = await self.associate_call_with_device(call_sid, device_id)
        if not success:
            return web.json_response(
                {"error": "Failed to associate call with device"},
                status=400,
            )

        return web.json_response(
            {
                "status": "associated",
                "call_sid": call_sid,
                "device_id": device_id,
            }
        )

    # =========================================================================
    # Debate Integration
    # =========================================================================

    async def _queue_debate_from_voice(self, call_sid: str, question: str) -> None:
        """
        Queue a debate from voice input.

        Args:
            call_sid: Call SID for tracking
            question: Transcribed question/topic
        """
        if not self.debate_starter:
            logger.info(f"No debate_starter configured, skipping debate for {call_sid}")
            return

        try:
            session = self.voice.get_session(call_sid)
            caller = session.caller if session else "unknown"

            # Build context with device info if available
            context = self._get_call_context(call_sid)
            device = await self.get_device_for_call(call_sid)
            if device:
                context["device_name"] = device.name
                context["device_type"] = device.device_type
                context["device_capabilities"] = device.capabilities

            logger.info(
                f"Queuing debate from voice: {question[:100]}... "
                f"(caller: {caller}, device: {context.get('device_id', 'none')})"
            )

            # Start debate asynchronously with device context
            debate_id = await self.debate_starter(
                task=question,
                source="voice",
                source_id=call_sid,
                callback_number=caller,
                agents=self.voice.config.default_agents,
                context=context,
            )

            if debate_id:
                self.voice.mark_debate_started(call_sid, debate_id)
                logger.info(f"Debate {debate_id} started from call {call_sid}")

        except Exception as e:
            logger.error(f"Failed to start debate from voice: {e}")


def setup_voice_routes(
    app: "Application",
    handler: VoiceHandler | None = None,
    device_registry: DeviceRegistry | None = None,
) -> VoiceHandler:
    """
    Set up voice webhook routes.

    Args:
        app: aiohttp Application
        handler: Optional VoiceHandler instance
        device_registry: Optional DeviceRegistry for device runtime integration

    Returns:
        The configured VoiceHandler instance
    """
    if handler is None:
        handler = VoiceHandler(device_registry=device_registry)

    # v1 canonical routes
    app.router.add_post("/api/v1/voice/inbound", handler.handle_inbound)
    app.router.add_post("/api/v1/voice/status", handler.handle_status)
    app.router.add_post("/api/v1/voice/gather", handler.handle_gather)
    app.router.add_post("/api/v1/voice/gather/confirm", handler.handle_gather_confirm)

    # Device association endpoint (requires device runtime)
    if HAS_DEVICE_REGISTRY and (handler.device_registry or device_registry):
        app.router.add_post("/api/v1/voice/device", handler.handle_device_association)

    # legacy routes
    app.router.add_post("/api/voice/inbound", handler.handle_inbound)
    app.router.add_post("/api/voice/status", handler.handle_status)
    app.router.add_post("/api/voice/gather", handler.handle_gather)
    app.router.add_post("/api/voice/gather/confirm", handler.handle_gather_confirm)

    logger.info(
        f"Voice webhook routes registered "
        f"(device_runtime={'enabled' if handler.device_registry else 'disabled'})"
    )
    return handler


__all__ = [
    "VoiceHandler",
    "setup_voice_routes",
    "HAS_DEVICE_REGISTRY",
    "HAS_TWILIO_VALIDATOR",
    "RequestValidator",
]
