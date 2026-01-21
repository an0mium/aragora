"""
Tests for Twilio Voice integration.

Tests the voice integration module for inbound/outbound calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.integrations.twilio_voice import (
    TwilioVoiceConfig,
    TwilioVoiceIntegration,
    CallSession,
    get_twilio_voice,
    HAS_TWILIO,
)


class TestTwilioVoiceConfig:
    """Test TwilioVoiceConfig."""

    def test_config_defaults(self):
        """Config has sensible defaults."""
        config = TwilioVoiceConfig()
        assert config.voice == "alice"
        assert config.language == "en-US"
        assert config.speech_timeout == 3
        assert config.max_recording_length == 120

    def test_config_from_env(self):
        """Config loads from environment."""
        with patch.dict(
            "os.environ",
            {
                "TWILIO_ACCOUNT_SID": "ACtest123",
                "TWILIO_AUTH_TOKEN": "token123",
                "TWILIO_PHONE_NUMBER": "+15551234567",
            },
        ):
            config = TwilioVoiceConfig()
            assert config.account_sid == "ACtest123"
            assert config.auth_token == "token123"
            assert config.phone_number == "+15551234567"

    def test_is_configured(self):
        """is_configured checks all required fields."""
        config = TwilioVoiceConfig()
        assert not config.is_configured

        config = TwilioVoiceConfig(
            account_sid="ACtest",
            auth_token="token",
            phone_number="+15551234567",
        )
        assert config.is_configured

    def test_get_webhook_url(self):
        """get_webhook_url builds full URL."""
        config = TwilioVoiceConfig(webhook_base_url="https://example.com")
        assert config.get_webhook_url("/api/voice/inbound") == "https://example.com/api/voice/inbound"

        # Handles trailing slash
        config = TwilioVoiceConfig(webhook_base_url="https://example.com/")
        assert config.get_webhook_url("/api/voice/inbound") == "https://example.com/api/voice/inbound"


class TestCallSession:
    """Test CallSession dataclass."""

    def test_session_creation(self):
        """CallSession initializes correctly."""
        session = CallSession(
            call_sid="CA123",
            caller="+15551234567",
            called="+15559876543",
            direction="inbound",
        )
        assert session.call_sid == "CA123"
        assert session.direction == "inbound"
        assert session.status == "initiated"
        assert session.transcription == ""
        assert session.debate_id is None


class TestTwilioVoiceIntegration:
    """Test TwilioVoiceIntegration class."""

    def test_is_available(self):
        """is_available reflects SDK availability."""
        voice = TwilioVoiceIntegration()
        assert voice.is_available == HAS_TWILIO

    def test_is_configured(self):
        """is_configured checks both SDK and config."""
        voice = TwilioVoiceIntegration()
        # Without SDK or config, should be False (or True if SDK available but no config)
        if HAS_TWILIO:
            assert not voice.is_configured  # No config

        config = TwilioVoiceConfig(
            account_sid="ACtest",
            auth_token="token",
            phone_number="+15551234567",
        )
        voice = TwilioVoiceIntegration(config)
        assert voice.is_configured == HAS_TWILIO

    @pytest.mark.skipif(not HAS_TWILIO, reason="Twilio SDK not installed")
    def test_handle_inbound_call(self):
        """handle_inbound_call generates valid TwiML."""
        config = TwilioVoiceConfig(
            account_sid="ACtest",
            auth_token="token",
            phone_number="+15551234567",
            webhook_base_url="https://example.com",
        )
        voice = TwilioVoiceIntegration(config)

        twiml = voice.handle_inbound_call(
            call_sid="CA123",
            caller="+15551234567",
            called="+15559876543",
        )

        assert '<?xml version="1.0"' in twiml
        assert "<Response>" in twiml
        assert "<Say" in twiml
        assert "<Gather" in twiml
        assert "Welcome to Aragora" in twiml

    @pytest.mark.skipif(not HAS_TWILIO, reason="Twilio SDK not installed")
    def test_handle_gather_result(self):
        """handle_gather_result processes speech."""
        config = TwilioVoiceConfig(
            account_sid="ACtest",
            auth_token="token",
            phone_number="+15551234567",
            require_confirmation=False,
        )
        voice = TwilioVoiceIntegration(config)

        # First create session
        voice.handle_inbound_call("CA123", "+1", "+2")

        twiml = voice.handle_gather_result(
            call_sid="CA123",
            speech_result="What is the best programming language?",
            confidence=0.95,
        )

        assert "<Response>" in twiml
        assert "Starting debate" in twiml

        # Session should have transcription
        session = voice.get_session("CA123")
        assert session is not None
        assert session.transcription == "What is the best programming language?"

    @pytest.mark.skipif(not HAS_TWILIO, reason="Twilio SDK not installed")
    def test_handle_gather_empty_speech(self):
        """handle_gather_result handles empty speech."""
        config = TwilioVoiceConfig()
        voice = TwilioVoiceIntegration(config)

        twiml = voice.handle_gather_result(
            call_sid="CA123",
            speech_result="",
            confidence=0.0,
        )

        assert "couldn't understand" in twiml

    @pytest.mark.skipif(not HAS_TWILIO, reason="Twilio SDK not installed")
    def test_handle_confirmation(self):
        """handle_confirmation processes digit input."""
        config = TwilioVoiceConfig()
        voice = TwilioVoiceIntegration(config)

        # Confirm
        twiml = voice.handle_confirmation("CA123", "1")
        assert "Starting debate" in twiml

        # Retry
        twiml = voice.handle_confirmation("CA123", "2")
        assert "try again" in twiml

        # Invalid
        twiml = voice.handle_confirmation("CA123", "9")
        assert "Invalid" in twiml

    def test_session_management(self):
        """Session management works correctly."""
        voice = TwilioVoiceIntegration()

        # No session initially
        assert voice.get_session("CA123") is None

        # After handling call, session exists
        if HAS_TWILIO:
            voice.handle_inbound_call("CA123", "+1", "+2")
            session = voice.get_session("CA123")
            assert session is not None
            assert session.call_sid == "CA123"

    def test_mark_debate_started(self):
        """mark_debate_started updates session."""
        voice = TwilioVoiceIntegration()

        if HAS_TWILIO:
            voice.handle_inbound_call("CA123", "+1", "+2")
            voice.mark_debate_started("CA123", "debate-456")

            session = voice.get_session("CA123")
            assert session.debate_id == "debate-456"

    def test_get_pending_debates(self):
        """get_pending_debates returns sessions awaiting debate."""
        voice = TwilioVoiceIntegration()

        if HAS_TWILIO:
            # Create session with transcription but no debate
            voice.handle_inbound_call("CA123", "+1", "+2")
            voice.handle_gather_result("CA123", "Test question", 0.9)

            pending = voice.get_pending_debates()
            assert len(pending) == 1
            assert pending[0].call_sid == "CA123"

            # Mark debate started
            voice.mark_debate_started("CA123", "debate-123")
            pending = voice.get_pending_debates()
            assert len(pending) == 0


class TestVoiceWebhookSignature:
    """Test webhook signature verification."""

    def test_verify_signature_missing_token(self):
        """Verification fails without auth token."""
        config = TwilioVoiceConfig()  # No auth token
        voice = TwilioVoiceIntegration(config)

        result = voice.verify_webhook_signature(
            url="https://example.com/webhook",
            params={"key": "value"},
            signature="invalid",
        )
        assert result is False

    def test_verify_signature_with_token(self):
        """Verification with valid token."""
        config = TwilioVoiceConfig(
            auth_token="test_token_12345",
        )
        voice = TwilioVoiceIntegration(config)

        # Calculate expected signature
        import hashlib
        import hmac
        from base64 import b64encode

        url = "https://example.com/webhook"
        params = {"AccountSid": "AC123", "ApiVersion": "2010-04-01"}

        s = url
        for key in sorted(params.keys()):
            s += key + params[key]

        expected = b64encode(
            hmac.new(
                config.auth_token.encode("utf-8"),
                s.encode("utf-8"),
                hashlib.sha1,
            ).digest()
        ).decode("utf-8")

        result = voice.verify_webhook_signature(url, params, expected)
        assert result is True

        # Wrong signature should fail
        result = voice.verify_webhook_signature(url, params, "wrong_signature")
        assert result is False


class TestGetTwilioVoice:
    """Test singleton accessor."""

    def test_get_twilio_voice_singleton(self):
        """get_twilio_voice returns singleton."""
        # Reset singleton
        import aragora.integrations.twilio_voice as module
        module._voice_integration = None

        voice1 = get_twilio_voice()
        voice2 = get_twilio_voice()
        assert voice1 is voice2

        # Cleanup
        module._voice_integration = None
