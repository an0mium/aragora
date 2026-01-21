"""
Tests for Voice Bridge - transcription integration for chat voice messages.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestVoiceBridge:
    """Tests for VoiceBridge initialization and configuration."""

    @pytest.fixture
    def bridge(self):
        """Create a VoiceBridge instance."""
        from aragora.connectors.chat.voice_bridge import VoiceBridge

        return VoiceBridge()

    def test_default_initialization(self, bridge):
        """Bridge should initialize with default values."""
        assert bridge.max_file_size == 25 * 1024 * 1024  # 25MB
        assert bridge.default_language == "en"

    def test_custom_initialization(self):
        """Bridge should accept custom configuration."""
        from aragora.connectors.chat.voice_bridge import VoiceBridge

        bridge = VoiceBridge(
            max_file_size=10 * 1024 * 1024,  # 10MB
            default_language="es",
            custom_option="value",
        )

        assert bridge.max_file_size == 10 * 1024 * 1024
        assert bridge.default_language == "es"
        assert bridge.config.get("custom_option") == "value"

    def test_whisper_lazy_loaded(self, bridge):
        """Whisper connector should be lazy-loaded."""
        assert bridge._whisper is None


class TestVoiceBridgeWhisperIntegration:
    """Tests for Whisper integration."""

    @pytest.fixture
    def bridge(self):
        """Create a VoiceBridge instance."""
        from aragora.connectors.chat.voice_bridge import VoiceBridge

        return VoiceBridge()

    def test_get_whisper_method_exists(self, bridge):
        """Bridge should have _get_whisper method for lazy loading."""
        assert hasattr(bridge, "_get_whisper")
        assert callable(bridge._get_whisper)

    def test_whisper_initially_none(self, bridge):
        """Whisper should be None before first use."""
        assert bridge._whisper is None


class TestVoiceBridgeModels:
    """Tests for VoiceMessage model usage."""

    def test_voice_message_import(self):
        """VoiceMessage should be importable from models."""
        from aragora.connectors.chat.models import VoiceMessage

        assert VoiceMessage is not None

    def test_file_attachment_import(self):
        """FileAttachment should be importable from models."""
        from aragora.connectors.chat.models import FileAttachment

        assert FileAttachment is not None

    def test_file_attachment_creation(self):
        """FileAttachment should be creatable with correct fields."""
        from aragora.connectors.chat.models import FileAttachment

        attachment = FileAttachment(
            id="file-123",
            filename="test.txt",
            content_type="text/plain",
            size=100,
            url="https://example.com/test.txt",
            content=b"test content",
        )

        assert attachment.id == "file-123"
        assert attachment.filename == "test.txt"
        assert attachment.content_type == "text/plain"
        assert attachment.size == 100
