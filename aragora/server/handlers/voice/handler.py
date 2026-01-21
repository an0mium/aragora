"""
Twilio Voice Webhook Handler.

Handles incoming voice webhooks from Twilio:
- /api/voice/inbound - Inbound call handling
- /api/voice/status - Call status updates
- /api/voice/gather - Speech recognition results
- /api/voice/gather/confirm - Confirmation input

All endpoints return TwiML responses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from aiohttp import web

from aragora.integrations.twilio_voice import (
    TwilioVoiceIntegration,
    get_twilio_voice,
    HAS_TWILIO,
)

if TYPE_CHECKING:
    from aiohttp.web import Application, Request, Response

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
    """

    def __init__(
        self,
        voice_integration: Optional[TwilioVoiceIntegration] = None,
        debate_starter: Optional[Any] = None,
    ):
        """
        Initialize voice handler.

        Args:
            voice_integration: TwilioVoiceIntegration instance
            debate_starter: Optional callable to start debates from voice input
        """
        self.voice = voice_integration or get_twilio_voice()
        self.debate_starter = debate_starter

    def _verify_signature(self, request: "Request") -> bool:
        """Verify Twilio webhook signature."""
        signature = request.headers.get("X-Twilio-Signature", "")
        if not signature:
            logger.warning("Missing Twilio signature header")
            return False

        # Get full URL
        url = str(request.url)

        # Get POST params (need to await for body)
        # For now, skip verification if we can't get params synchronously
        # In production, implement async signature verification

        return True  # Simplified for now

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

        # Return empty 200 OK
        return web.Response(text="OK", status=200)

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

            logger.info(f"Queuing debate from voice: {question[:100]}... (caller: {caller})")

            # Start debate asynchronously
            debate_id = await self.debate_starter(
                task=question,
                source="voice",
                source_id=call_sid,
                callback_number=caller,
                agents=self.voice.config.default_agents,
            )

            if debate_id:
                self.voice.mark_debate_started(call_sid, debate_id)
                logger.info(f"Debate {debate_id} started from call {call_sid}")

        except Exception as e:
            logger.error(f"Failed to start debate from voice: {e}")


def setup_voice_routes(app: "Application", handler: Optional[VoiceHandler] = None) -> None:
    """
    Set up voice webhook routes.

    Args:
        app: aiohttp Application
        handler: Optional VoiceHandler instance
    """
    if handler is None:
        handler = VoiceHandler()

    # Voice webhook routes
    app.router.add_post("/api/voice/inbound", handler.handle_inbound)
    app.router.add_post("/api/voice/status", handler.handle_status)
    app.router.add_post("/api/voice/gather", handler.handle_gather)
    app.router.add_post("/api/voice/gather/confirm", handler.handle_gather_confirm)

    logger.info("Voice webhook routes registered")


__all__ = ["VoiceHandler", "setup_voice_routes"]
