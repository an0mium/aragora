"""
Voice Namespace API.

Provides text-to-speech synthesis, voice session management,
and Twilio webhook integration endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class VoiceAPI:
    """Synchronous Voice API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # -- TTS and session management -------------------------------------------

    def synthesize(
        self, text: str, voice: str = "default", format: str = "mp3", **kwargs: Any
    ) -> Any:
        """Synthesize speech from text.

        @route POST /api/v1/voice/synthesize
        """
        return self._client.request(
            "POST",
            "/api/v1/voice/synthesize",
            json={"text": text, "voice": voice, "format": format, **kwargs},
        )

    def list_voices(self) -> Any:
        """List available voices.

        @route GET /api/v1/voice/voices
        """
        return self._client.request("GET", "/api/v1/voice/voices")

    def get_config(self) -> Any:
        """Get voice configuration.

        @route GET /api/v1/voice/config
        """
        return self._client.request("GET", "/api/v1/voice/config")

    def create_session(self, debate_id: str, **kwargs: Any) -> Any:
        """Create a voice streaming session.

        @route POST /api/v1/voice/sessions
        """
        return self._client.request(
            "POST",
            "/api/v1/voice/sessions",
            json={"debate_id": debate_id, **kwargs},
        )

    def end_session(self, session_id: str) -> Any:
        """End a voice session.

        @route DELETE /api/v1/voice/sessions/{session_id}
        """
        return self._client.request(
            "DELETE", f"/api/v1/voice/sessions/{session_id}"
        )

    # -- Twilio voice webhook endpoints ---------------------------------------

    def handle_inbound(self, call_sid: str, caller: str = "", called: str = "") -> Any:
        """Trigger the inbound call webhook handler.

        @route POST /api/v1/voice/inbound
        """
        return self._client.request(
            "POST",
            "/api/v1/voice/inbound",
            json={"CallSid": call_sid, "From": caller, "To": called},
        )

    def get_call_status(self, call_sid: str, call_status: str = "", **kwargs: Any) -> Any:
        """Send a call status callback.

        @route POST /api/v1/voice/status
        """
        return self._client.request(
            "POST",
            "/api/v1/voice/status",
            json={"CallSid": call_sid, "CallStatus": call_status, **kwargs},
        )

    def submit_gather(
        self, call_sid: str, speech_result: str = "", confidence: float = 0.0
    ) -> Any:
        """Submit speech gather result from a call.

        @route POST /api/v1/voice/gather
        """
        return self._client.request(
            "POST",
            "/api/v1/voice/gather",
            json={
                "CallSid": call_sid,
                "SpeechResult": speech_result,
                "Confidence": str(confidence),
            },
        )

    def confirm_gather(self, call_sid: str, digits: str = "") -> Any:
        """Submit confirmation digit press for a gather result.

        @route POST /api/v1/voice/gather/confirm
        """
        return self._client.request(
            "POST",
            "/api/v1/voice/gather/confirm",
            json={"CallSid": call_sid, "Digits": digits},
        )

    def associate_device(self, call_sid: str, device_id: str) -> Any:
        """Associate a voice call with a registered device.

        @route POST /api/v1/voice/device
        """
        return self._client.request(
            "POST",
            "/api/v1/voice/device",
            json={"call_sid": call_sid, "device_id": device_id},
        )


class AsyncVoiceAPI:
    """Asynchronous Voice API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # -- TTS and session management -------------------------------------------

    async def synthesize(
        self, text: str, voice: str = "default", format: str = "mp3", **kwargs: Any
    ) -> Any:
        """Synthesize speech from text.

        @route POST /api/v1/voice/synthesize
        """
        return await self._client.request(
            "POST",
            "/api/v1/voice/synthesize",
            json={"text": text, "voice": voice, "format": format, **kwargs},
        )

    async def list_voices(self) -> Any:
        """List available voices.

        @route GET /api/v1/voice/voices
        """
        return await self._client.request("GET", "/api/v1/voice/voices")

    async def get_config(self) -> Any:
        """Get voice configuration.

        @route GET /api/v1/voice/config
        """
        return await self._client.request("GET", "/api/v1/voice/config")

    async def create_session(self, debate_id: str, **kwargs: Any) -> Any:
        """Create a voice streaming session.

        @route POST /api/v1/voice/sessions
        """
        return await self._client.request(
            "POST",
            "/api/v1/voice/sessions",
            json={"debate_id": debate_id, **kwargs},
        )

    async def end_session(self, session_id: str) -> Any:
        """End a voice session.

        @route DELETE /api/v1/voice/sessions/{session_id}
        """
        return await self._client.request(
            "DELETE", f"/api/v1/voice/sessions/{session_id}"
        )

    # -- Twilio voice webhook endpoints ---------------------------------------

    async def handle_inbound(self, call_sid: str, caller: str = "", called: str = "") -> Any:
        """Trigger the inbound call webhook handler.

        @route POST /api/v1/voice/inbound
        """
        return await self._client.request(
            "POST",
            "/api/v1/voice/inbound",
            json={"CallSid": call_sid, "From": caller, "To": called},
        )

    async def get_call_status(self, call_sid: str, call_status: str = "", **kwargs: Any) -> Any:
        """Send a call status callback.

        @route POST /api/v1/voice/status
        """
        return await self._client.request(
            "POST",
            "/api/v1/voice/status",
            json={"CallSid": call_sid, "CallStatus": call_status, **kwargs},
        )

    async def submit_gather(
        self, call_sid: str, speech_result: str = "", confidence: float = 0.0
    ) -> Any:
        """Submit speech gather result from a call.

        @route POST /api/v1/voice/gather
        """
        return await self._client.request(
            "POST",
            "/api/v1/voice/gather",
            json={
                "CallSid": call_sid,
                "SpeechResult": speech_result,
                "Confidence": str(confidence),
            },
        )

    async def confirm_gather(self, call_sid: str, digits: str = "") -> Any:
        """Submit confirmation digit press for a gather result.

        @route POST /api/v1/voice/gather/confirm
        """
        return await self._client.request(
            "POST",
            "/api/v1/voice/gather/confirm",
            json={"CallSid": call_sid, "Digits": digits},
        )

    async def associate_device(self, call_sid: str, device_id: str) -> Any:
        """Associate a voice call with a registered device.

        @route POST /api/v1/voice/device
        """
        return await self._client.request(
            "POST",
            "/api/v1/voice/device",
            json={"call_sid": call_sid, "device_id": device_id},
        )
