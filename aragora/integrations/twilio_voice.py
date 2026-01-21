"""
Twilio Voice Integration for Aragora.

Enables bidirectional voice interactions:
- Inbound: Receive phone calls and transcribe to initiate debates
- Outbound: Call users with debate results via TTS

Requires:
    TWILIO_ACCOUNT_SID - Twilio Account SID
    TWILIO_AUTH_TOKEN - Twilio Auth Token
    TWILIO_PHONE_NUMBER - Twilio phone number for calls

Usage:
    voice = TwilioVoiceIntegration(TwilioVoiceConfig(
        account_sid="ACxxxxxxxxx",
        auth_token="xxxxxxxxx",
        phone_number="+15551234567",
    ))

    # Generate TwiML for inbound call
    twiml = voice.handle_inbound_call(caller="+1234567890")

    # Make outbound call with debate result
    await voice.call_with_result(to="+1234567890", result=debate_result)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# Optional: Twilio SDK
try:
    from twilio.rest import Client as TwilioClient
    from twilio.twiml.voice_response import VoiceResponse, Gather, Say, Record

    HAS_TWILIO = True
except ImportError:
    TwilioClient = None
    VoiceResponse = None
    Gather = None
    Say = None
    Record = None
    HAS_TWILIO = False


@dataclass
class TwilioVoiceConfig:
    """Configuration for Twilio Voice integration."""

    # Twilio credentials
    account_sid: str = ""
    auth_token: str = ""
    phone_number: str = ""  # Your Twilio phone number

    # Webhook URLs (set by server)
    webhook_base_url: str = ""  # e.g., https://aragora.example.com
    inbound_path: str = "/api/voice/inbound"
    status_path: str = "/api/voice/status"
    transcription_path: str = "/api/voice/transcription"
    gather_path: str = "/api/voice/gather"

    # Voice settings
    voice: str = "alice"  # Twilio voice (alice, man, woman, or Polly voices)
    language: str = "en-US"
    speech_timeout: int = 3  # Seconds of silence before ending speech input
    max_recording_length: int = 120  # Max recording length in seconds

    # Debate settings
    auto_start_debate: bool = True  # Auto-start debate from transcription
    require_confirmation: bool = True  # Confirm before starting debate
    default_agents: list[str] = field(default_factory=lambda: ["anthropic-api", "openai-api"])

    def __post_init__(self) -> None:
        """Load from environment if not provided."""
        if not self.account_sid:
            self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        if not self.auth_token:
            self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        if not self.phone_number:
            self.phone_number = os.environ.get("TWILIO_PHONE_NUMBER", "")
        if not self.webhook_base_url:
            self.webhook_base_url = os.environ.get("ARAGORA_WEBHOOK_BASE_URL", "")

    @property
    def is_configured(self) -> bool:
        """Check if Twilio Voice is properly configured."""
        return bool(self.account_sid and self.auth_token and self.phone_number)

    def get_webhook_url(self, path: str) -> str:
        """Get full webhook URL for a path."""
        base = self.webhook_base_url.rstrip("/")
        return f"{base}{path}" if base else path


@dataclass
class CallSession:
    """Tracks an active voice call session."""

    call_sid: str
    caller: str
    called: str
    direction: str  # "inbound" or "outbound"
    status: str = "initiated"
    transcription: str = ""
    debate_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class TwilioVoiceIntegration:
    """
    Twilio Voice integration for bidirectional voice debates.

    Handles:
    - Inbound calls: Transcribe caller speech → start debate
    - Outbound calls: Call users with debate results
    - Recording/transcription management
    """

    def __init__(self, config: Optional[TwilioVoiceConfig] = None):
        """Initialize Twilio Voice integration."""
        self.config = config or TwilioVoiceConfig()
        self._client: Optional[Any] = None
        self._sessions: dict[str, CallSession] = {}

    @property
    def is_available(self) -> bool:
        """Check if Twilio SDK is available."""
        return HAS_TWILIO

    @property
    def is_configured(self) -> bool:
        """Check if integration is properly configured."""
        return self.config.is_configured and HAS_TWILIO

    def _get_client(self) -> Any:
        """Get or create Twilio client."""
        if not HAS_TWILIO:
            raise RuntimeError("Twilio SDK not installed. Run: pip install twilio")

        if self._client is None:
            self._client = TwilioClient(
                self.config.account_sid,
                self.config.auth_token,
            )
        return self._client

    # =========================================================================
    # Webhook Signature Verification
    # =========================================================================

    def verify_webhook_signature(
        self,
        url: str,
        params: dict[str, str],
        signature: str,
    ) -> bool:
        """
        Verify Twilio webhook signature.

        Args:
            url: Full webhook URL
            params: POST parameters
            signature: X-Twilio-Signature header value

        Returns:
            True if signature is valid
        """
        if not self.config.auth_token:
            logger.warning("No auth token configured for signature verification")
            return False

        # Build validation string
        s = url
        if params:
            # Sort parameters and append to URL
            for key in sorted(params.keys()):
                s += key + params[key]

        # Calculate expected signature
        expected = b64encode(
            hmac.new(
                self.config.auth_token.encode("utf-8"),
                s.encode("utf-8"),
                hashlib.sha1,
            ).digest()
        ).decode("utf-8")

        return hmac.compare_digest(signature, expected)

    # =========================================================================
    # Inbound Call Handling
    # =========================================================================

    def handle_inbound_call(
        self,
        call_sid: str,
        caller: str,
        called: str,
    ) -> str:
        """
        Generate TwiML response for inbound call.

        Creates a voice response that:
        1. Greets the caller
        2. Prompts for their question/topic
        3. Records or gathers speech input

        Args:
            call_sid: Twilio Call SID
            caller: Caller phone number
            called: Called phone number (your Twilio number)

        Returns:
            TwiML XML string
        """
        if not HAS_TWILIO:
            return '<?xml version="1.0" encoding="UTF-8"?><Response><Say>Voice service unavailable.</Say></Response>'

        # Track session
        session = CallSession(
            call_sid=call_sid,
            caller=caller,
            called=called,
            direction="inbound",
            status="answered",
        )
        self._sessions[call_sid] = session

        # Build TwiML response
        response = VoiceResponse()

        # Greeting
        response.say(
            "Welcome to Aragora. You can ask me any question and I'll have "
            "AI agents debate it for you. Please state your question after the beep.",
            voice=self.config.voice,
            language=self.config.language,
        )

        # Gather speech input
        gather = Gather(
            input="speech",
            action=self.config.get_webhook_url(self.config.gather_path),
            method="POST",
            speech_timeout=str(self.config.speech_timeout),
            language=self.config.language,
        )
        gather.say(
            "Go ahead.",
            voice=self.config.voice,
            language=self.config.language,
        )
        response.append(gather)

        # Fallback if no speech detected
        response.say(
            "I didn't hear anything. Please call back when you're ready.",
            voice=self.config.voice,
            language=self.config.language,
        )
        response.hangup()

        logger.info(f"Inbound call from {caller}, SID: {call_sid}")
        return str(response)

    def handle_gather_result(
        self,
        call_sid: str,
        speech_result: str,
        confidence: float = 0.0,
    ) -> str:
        """
        Handle speech gather result.

        Called when caller finishes speaking. Creates debate from transcription.

        Args:
            call_sid: Twilio Call SID
            speech_result: Transcribed speech text
            confidence: Speech recognition confidence (0-1)

        Returns:
            TwiML XML string for next action
        """
        if not HAS_TWILIO:
            return '<?xml version="1.0" encoding="UTF-8"?><Response><Say>Service unavailable.</Say></Response>'

        # Update session
        session = self._sessions.get(call_sid)
        if session:
            session.transcription = speech_result
            session.updated_at = datetime.now()
            session.metadata["speech_confidence"] = confidence

        response = VoiceResponse()

        if not speech_result or len(speech_result.strip()) < 5:
            response.say(
                "I couldn't understand your question. Please try again.",
                voice=self.config.voice,
                language=self.config.language,
            )
            response.redirect(self.config.get_webhook_url(self.config.inbound_path))
            return str(response)

        # Confirm the question if configured
        if self.config.require_confirmation:
            response.say(
                f"I heard: {speech_result}. "
                "Press 1 to start the debate, or press 2 to try again.",
                voice=self.config.voice,
                language=self.config.language,
            )

            gather = Gather(
                num_digits=1,
                action=self.config.get_webhook_url(f"{self.config.gather_path}/confirm"),
                method="POST",
            )
            response.append(gather)

            response.say(
                "No input received. Starting debate with your question.",
                voice=self.config.voice,
                language=self.config.language,
            )

        # Start debate processing
        response.say(
            "Starting debate on your question. "
            "This may take a few moments. I'll call you back with the results.",
            voice=self.config.voice,
            language=self.config.language,
        )
        response.hangup()

        logger.info(f"Gather result for {call_sid}: {speech_result[:100]}...")
        return str(response)

    def handle_confirmation(
        self,
        call_sid: str,
        digits: str,
    ) -> str:
        """
        Handle confirmation digit press.

        Args:
            call_sid: Twilio Call SID
            digits: Pressed digits

        Returns:
            TwiML XML string
        """
        if not HAS_TWILIO:
            return '<?xml version="1.0" encoding="UTF-8"?><Response><Say>Service unavailable.</Say></Response>'

        response = VoiceResponse()

        if digits == "1":
            # Start debate
            response.say(
                "Starting debate. I'll call you back with the results.",
                voice=self.config.voice,
                language=self.config.language,
            )
            response.hangup()
        elif digits == "2":
            # Try again
            response.say(
                "Let's try again. Please state your question.",
                voice=self.config.voice,
                language=self.config.language,
            )
            response.redirect(self.config.get_webhook_url(self.config.inbound_path))
        else:
            response.say(
                "Invalid input. Press 1 to start, or 2 to try again.",
                voice=self.config.voice,
                language=self.config.language,
            )
            response.redirect(self.config.get_webhook_url(f"{self.config.gather_path}/confirm"))

        return str(response)

    # =========================================================================
    # Outbound Calls
    # =========================================================================

    async def call_with_message(
        self,
        to: str,
        message: str,
        status_callback: Optional[str] = None,
    ) -> Optional[str]:
        """
        Make outbound call with a message.

        Args:
            to: Phone number to call
            message: Message to speak (via TTS)
            status_callback: Optional status webhook URL

        Returns:
            Call SID if successful, None otherwise
        """
        if not self.is_configured:
            logger.warning("Twilio Voice not configured")
            return None

        try:
            client = self._get_client()

            # Create TwiML for the message
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="{self.config.voice}" language="{self.config.language}">{message}</Say>
    <Pause length="1"/>
    <Say voice="{self.config.voice}" language="{self.config.language}">Goodbye.</Say>
</Response>"""

            call = client.calls.create(
                to=to,
                from_=self.config.phone_number,
                twiml=twiml,
                status_callback=status_callback or self.config.get_webhook_url(self.config.status_path),
                status_callback_event=["initiated", "ringing", "answered", "completed"],
            )

            # Track session
            session = CallSession(
                call_sid=call.sid,
                caller=self.config.phone_number,
                called=to,
                direction="outbound",
                status="initiated",
            )
            self._sessions[call.sid] = session

            logger.info(f"Outbound call initiated to {to}, SID: {call.sid}")
            return call.sid

        except Exception as e:
            logger.error(f"Failed to initiate outbound call: {e}")
            return None

    async def call_with_result(
        self,
        to: str,
        result: Any,  # DebateResult
    ) -> Optional[str]:
        """
        Call user with debate result summary.

        Args:
            to: Phone number to call
            result: DebateResult object

        Returns:
            Call SID if successful
        """
        # Format result as spoken message
        if hasattr(result, "consensus_reached") and result.consensus_reached:
            confidence_pct = int(getattr(result, "confidence", 0) * 100)
            answer = getattr(result, "final_answer", "")[:500]
            message = (
                f"Your debate has completed with consensus at {confidence_pct} percent confidence. "
                f"The answer is: {answer}"
            )
        else:
            rounds = getattr(result, "rounds_used", 0)
            message = (
                f"Your debate completed after {rounds} rounds but did not reach consensus. "
                "The agents had differing opinions. Check the app for full details."
            )

        return await self.call_with_message(to, message)

    # =========================================================================
    # Status Handling
    # =========================================================================

    def handle_status_callback(
        self,
        call_sid: str,
        call_status: str,
        **kwargs: Any,
    ) -> None:
        """
        Handle call status webhook.

        Args:
            call_sid: Twilio Call SID
            call_status: Call status (initiated, ringing, answered, completed, etc.)
            **kwargs: Additional status parameters
        """
        session = self._sessions.get(call_sid)
        if session:
            session.status = call_status
            session.updated_at = datetime.now()
            session.metadata.update(kwargs)

        logger.info(f"Call {call_sid} status: {call_status}")

        # Clean up completed calls after a delay
        if call_status in ("completed", "failed", "busy", "no-answer", "canceled"):
            # In production, schedule cleanup instead of immediate removal
            pass

    # =========================================================================
    # Session Management
    # =========================================================================

    def get_session(self, call_sid: str) -> Optional[CallSession]:
        """Get call session by SID."""
        return self._sessions.get(call_sid)

    def get_pending_debates(self) -> list[CallSession]:
        """Get sessions waiting for debate start."""
        return [
            s for s in self._sessions.values()
            if s.transcription and not s.debate_id and s.direction == "inbound"
        ]

    def mark_debate_started(self, call_sid: str, debate_id: str) -> None:
        """Mark that a debate has been started for a call."""
        session = self._sessions.get(call_sid)
        if session:
            session.debate_id = debate_id
            session.updated_at = datetime.now()


# Singleton instance
_voice_integration: Optional[TwilioVoiceIntegration] = None


def get_twilio_voice(config: Optional[TwilioVoiceConfig] = None) -> TwilioVoiceIntegration:
    """Get or create the Twilio Voice integration singleton."""
    global _voice_integration
    if _voice_integration is None:
        _voice_integration = TwilioVoiceIntegration(config)
    return _voice_integration


__all__ = [
    "TwilioVoiceConfig",
    "TwilioVoiceIntegration",
    "CallSession",
    "get_twilio_voice",
    "HAS_TWILIO",
]
