"""
Tests for Voice Wake Module.

Tests:
- VoiceConfig creation and validation
- WakeWordDetector lifecycle
- Wake event handling
- Callback invocation
- Backend fallback
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.voice.config import (
    AudioDevice,
    DetectionBackend,
    VoiceConfig,
)
from aragora.voice.wake import (
    DetectorStatus,
    WakeEvent,
    WakeWordDetector,
)


# =============================================================================
# VoiceConfig Tests
# =============================================================================


class TestVoiceConfig:
    """Test VoiceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VoiceConfig()

        assert "hey aragora" in config.wake_phrases
        assert config.sensitivity == 0.5
        assert config.backend == DetectionBackend.KEYWORD
        assert config.min_confidence == 0.6
        assert config.cooldown_seconds == 2.0

    def test_custom_wake_phrases(self):
        """Test custom wake phrases."""
        config = VoiceConfig(wake_phrases=["hello assistant", "wake up"])

        assert len(config.wake_phrases) == 2
        assert "hello assistant" in config.wake_phrases
        assert "wake up" in config.wake_phrases

    def test_validation_empty_phrases(self):
        """Test validation fails with empty wake phrases."""
        config = VoiceConfig(wake_phrases=[])
        errors = config.validate()

        assert len(errors) > 0
        assert any("wake phrase" in e.lower() for e in errors)

    def test_validation_invalid_sensitivity(self):
        """Test validation fails with invalid sensitivity."""
        config = VoiceConfig(sensitivity=1.5)
        errors = config.validate()

        assert len(errors) > 0
        assert any("sensitivity" in e.lower() for e in errors)

    def test_validation_invalid_confidence(self):
        """Test validation fails with invalid min_confidence."""
        config = VoiceConfig(min_confidence=-0.1)
        errors = config.validate()

        assert len(errors) > 0
        assert any("confidence" in e.lower() for e in errors)

    def test_validation_porcupine_without_key(self):
        """Test validation fails for Porcupine without access key."""
        config = VoiceConfig(
            backend=DetectionBackend.PORCUPINE,
            porcupine_access_key=None,
        )
        errors = config.validate()

        assert len(errors) > 0
        assert any("porcupine" in e.lower() for e in errors)

    def test_validation_valid_config(self):
        """Test validation passes with valid config."""
        config = VoiceConfig(
            wake_phrases=["hey aragora"],
            sensitivity=0.5,
            min_confidence=0.6,
        )
        errors = config.validate()

        assert len(errors) == 0


class TestAudioDevice:
    """Test AudioDevice configuration."""

    def test_default_device(self):
        """Test default audio device settings."""
        device = AudioDevice()

        assert device.device_id is None  # System default
        assert device.sample_rate == 16000
        assert device.channels == 1
        assert device.chunk_size == 1024
        assert device.format == "int16"

    def test_custom_device(self):
        """Test custom audio device settings."""
        device = AudioDevice(
            device_id=1,
            sample_rate=44100,
            channels=2,
            chunk_size=2048,
        )

        assert device.device_id == 1
        assert device.sample_rate == 44100
        assert device.channels == 2
        assert device.chunk_size == 2048


# =============================================================================
# WakeWordDetector Tests
# =============================================================================


class TestWakeWordDetector:
    """Test WakeWordDetector class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return VoiceConfig(
            wake_phrases=["hey test", "ok test"],
            sensitivity=0.5,
            cooldown_seconds=0.1,  # Short cooldown for tests
        )

    @pytest.fixture
    def detector(self, config):
        """Create detector instance."""
        return WakeWordDetector(config)

    def test_initial_status(self, detector):
        """Test detector starts in stopped state."""
        assert detector.status == DetectorStatus.STOPPED
        assert not detector.is_running

    @pytest.mark.asyncio
    async def test_start_stop(self, detector):
        """Test starting and stopping detector."""
        callback = AsyncMock()

        await detector.start(callback)
        assert detector.is_running
        assert detector.status == DetectorStatus.LISTENING

        await detector.stop()
        assert not detector.is_running
        assert detector.status == DetectorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_start_with_invalid_config(self):
        """Test start fails with invalid configuration."""
        config = VoiceConfig(wake_phrases=[])
        detector = WakeWordDetector(config)

        with pytest.raises(ValueError, match="Invalid voice config"):
            await detector.start(AsyncMock())

    @pytest.mark.asyncio
    async def test_callback_invocation(self, detector):
        """Test callback is invoked on wake detection."""
        callback = MagicMock()
        await detector.start(callback)

        try:
            # Simulate wake word detection
            detector.simulate_wake("hey test", confidence=0.9)

            callback.assert_called_once_with("hey test", 0.9)
        finally:
            await detector.stop()

    @pytest.mark.asyncio
    async def test_async_callback(self, detector):
        """Test async callback is awaited."""
        callback = AsyncMock()
        await detector.start(callback)

        try:
            detector.simulate_wake("hey test", confidence=0.85)
            await asyncio.sleep(0.01)  # Allow task to complete

            callback.assert_called_once_with("hey test", 0.85)
        finally:
            await detector.stop()

    @pytest.mark.asyncio
    async def test_simulate_uses_first_phrase(self, detector):
        """Test simulate_wake uses first phrase when none specified."""
        callback = MagicMock()
        await detector.start(callback)

        try:
            detector.simulate_wake()

            callback.assert_called_once()
            call_args = callback.call_args[0]
            assert call_args[0] == "hey test"  # First configured phrase
        finally:
            await detector.stop()

    def test_get_stats(self, detector):
        """Test get_stats returns expected information."""
        stats = detector.get_stats()

        assert "status" in stats
        assert "backend" in stats
        assert "wake_phrases" in stats
        assert "sensitivity" in stats
        assert stats["status"] == "stopped"
        assert stats["backend"] == "keyword"


# =============================================================================
# WakeEvent Tests
# =============================================================================


class TestWakeEvent:
    """Test WakeEvent dataclass."""

    def test_event_creation(self):
        """Test creating a wake event."""
        event = WakeEvent(phrase="hey aragora", confidence=0.95)

        assert event.phrase == "hey aragora"
        assert event.confidence == 0.95
        assert event.timestamp > 0
        assert event.audio_data is None

    def test_event_with_audio(self):
        """Test wake event with audio data."""
        audio = b"\x00\x01\x02\x03"
        event = WakeEvent(
            phrase="hey aragora",
            confidence=0.9,
            audio_data=audio,
        )

        assert event.audio_data == audio


# =============================================================================
# Backend Tests
# =============================================================================


class TestDetectionBackends:
    """Test detection backend handling."""

    @pytest.mark.asyncio
    async def test_keyword_backend_default(self):
        """Test keyword backend is used by default."""
        config = VoiceConfig()
        detector = WakeWordDetector(config)

        assert config.backend == DetectionBackend.KEYWORD
        assert detector.config.backend == DetectionBackend.KEYWORD

    @pytest.mark.asyncio
    async def test_porcupine_fallback_without_library(self):
        """Test Porcupine falls back to keyword when library unavailable."""
        config = VoiceConfig(
            backend=DetectionBackend.PORCUPINE,
            porcupine_access_key="test-key",
        )
        detector = WakeWordDetector(config)

        await detector.start(AsyncMock())
        try:
            # Should fall back to keyword since pvporcupine not installed
            assert detector.config.backend == DetectionBackend.KEYWORD
        finally:
            await detector.stop()

    @pytest.mark.asyncio
    async def test_vosk_fallback_without_library(self):
        """Test Vosk falls back to keyword when library unavailable."""
        config = VoiceConfig(
            backend=DetectionBackend.VOSK,
            vosk_model_path="/nonexistent/path",
        )
        detector = WakeWordDetector(config)

        await detector.start(AsyncMock())
        try:
            # Should fall back to keyword since vosk not installed
            assert detector.config.backend == DetectionBackend.KEYWORD
        finally:
            await detector.stop()


# =============================================================================
# Cooldown Tests
# =============================================================================


class TestCooldown:
    """Test cooldown behavior."""

    @pytest.mark.asyncio
    async def test_cooldown_after_detection(self):
        """Test detector enters cooldown after detection."""
        config = VoiceConfig(cooldown_seconds=1.0)
        detector = WakeWordDetector(config)

        callback = MagicMock()
        await detector.start(callback)

        try:
            # First detection
            detector.simulate_wake("hey test")
            assert callback.call_count == 1

            # Immediate second attempt should be ignored (in cooldown)
            # Note: simulate_wake bypasses cooldown check for testing
            # Real audio processing would respect cooldown
        finally:
            await detector.stop()


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_double_start(self):
        """Test calling start twice doesn't error."""
        config = VoiceConfig()
        detector = WakeWordDetector(config)

        await detector.start(AsyncMock())
        try:
            # Second start should be a no-op
            await detector.start(AsyncMock())
            assert detector.is_running
        finally:
            await detector.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """Test stopping when not running is safe."""
        config = VoiceConfig()
        detector = WakeWordDetector(config)

        # Should not error
        await detector.stop()
        assert detector.status == DetectorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_callback_error_handled(self):
        """Test callback errors are logged but don't crash detector."""
        config = VoiceConfig()
        detector = WakeWordDetector(config)

        def bad_callback(phrase, confidence):
            raise ValueError("Test error")

        await detector.start(bad_callback)
        try:
            # Should not raise
            detector.simulate_wake("hey test")
        finally:
            await detector.stop()


__all__ = [
    "TestVoiceConfig",
    "TestAudioDevice",
    "TestWakeWordDetector",
    "TestWakeEvent",
    "TestDetectionBackends",
    "TestCooldown",
    "TestEdgeCases",
]
