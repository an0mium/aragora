"""
Tests for Moltbot Voice Wake component.

Tests wake word detection, voice sessions, and command processing.
"""

import asyncio
import pytest
from pathlib import Path

from aragora.extensions.moltbot import (
    VoiceActivityState,
    VoiceCommand,
    VoiceWakeManager,
    WakeWordConfig,
    WakeWordEngine,
    WakeWordEvent,
)


class TestVoiceWakeManager:
    """Tests for VoiceWakeManager."""

    @pytest.fixture
    def config(self) -> WakeWordConfig:
        """Create test wake word config."""
        return WakeWordConfig(
            wake_words=["hey test", "test"],
            engine=WakeWordEngine.PORCUPINE,
            sensitivity=0.6,
            timeout_seconds=5.0,
        )

    @pytest.fixture
    def manager(self, config: WakeWordConfig, tmp_path: Path) -> VoiceWakeManager:
        """Create a voice wake manager for testing."""
        return VoiceWakeManager(config, storage_path=tmp_path / "voice")

    @pytest.mark.asyncio
    async def test_start_stop(self, manager: VoiceWakeManager):
        """Test starting and stopping the manager."""
        await manager.start()
        stats = await manager.get_stats()
        assert stats["running"] is True

        await manager.stop()
        stats = await manager.get_stats()
        assert stats["running"] is False

    @pytest.mark.asyncio
    async def test_start_listening(self, manager: VoiceWakeManager):
        """Test starting to listen on a device."""
        await manager.start()

        result = await manager.start_listening("device-1")
        assert result is True

        state = manager.get_device_state("device-1")
        assert state == VoiceActivityState.LISTENING

        await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_listening(self, manager: VoiceWakeManager):
        """Test stopping listening on a device."""
        await manager.start()
        await manager.start_listening("device-1")

        result = await manager.stop_listening("device-1")
        assert result is True

        state = manager.get_device_state("device-1")
        assert state == VoiceActivityState.IDLE

        await manager.stop()

    @pytest.mark.asyncio
    async def test_list_active_devices(self, manager: VoiceWakeManager):
        """Test listing active devices."""
        await manager.start()
        await manager.start_listening("device-1")
        await manager.start_listening("device-2")

        active = await manager.list_active_devices()
        assert len(active) == 2
        assert "device-1" in active
        assert "device-2" in active

        await manager.stop()


class TestWakeWordDetection:
    """Tests for wake word detection."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> VoiceWakeManager:
        """Create a voice wake manager for testing."""
        return VoiceWakeManager(storage_path=tmp_path / "voice")

    @pytest.mark.asyncio
    async def test_simulate_wake_detection(self, manager: VoiceWakeManager):
        """Test simulating wake word detection."""
        await manager.start()
        await manager.start_listening("device-1")

        event = await manager.simulate_wake_detection(
            device_id="device-1",
            wake_word="hey aragora",
            confidence=0.95,
        )

        assert event is not None
        assert event.wake_word == "hey aragora"
        assert event.confidence == 0.95
        assert event.device_id == "device-1"

        await manager.stop()

    @pytest.mark.asyncio
    async def test_wake_detection_creates_session(self, manager: VoiceWakeManager):
        """Test that wake detection creates a voice session."""
        await manager.start()
        await manager.start_listening("device-1")

        event = await manager.simulate_wake_detection("device-1")

        session = await manager.get_device_session("device-1")
        assert session is not None
        assert session.wake_event == event
        assert session.state == VoiceActivityState.DETECTING

        await manager.stop()

    @pytest.mark.asyncio
    async def test_wake_callback(self, manager: VoiceWakeManager):
        """Test wake detection callback."""
        await manager.start()
        await manager.start_listening("device-1")

        events_received = []
        manager.on_wake_detected(lambda e: events_received.append(e))

        await manager.simulate_wake_detection("device-1")

        assert len(events_received) == 1
        assert events_received[0].device_id == "device-1"

        await manager.stop()


class TestVoiceSessions:
    """Tests for voice session management."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> VoiceWakeManager:
        """Create a voice wake manager for testing."""
        return VoiceWakeManager(
            config=WakeWordConfig(timeout_seconds=10.0),
            storage_path=tmp_path / "voice",
        )

    @pytest.mark.asyncio
    async def test_get_session(self, manager: VoiceWakeManager):
        """Test getting a session by ID."""
        await manager.start()
        await manager.start_listening("device-1")
        await manager.simulate_wake_detection("device-1")

        session = await manager.get_device_session("device-1")
        assert session is not None

        retrieved = await manager.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

        await manager.stop()

    @pytest.mark.asyncio
    async def test_end_session(self, manager: VoiceWakeManager):
        """Test ending a session."""
        await manager.start()
        await manager.start_listening("device-1")
        await manager.simulate_wake_detection("device-1")

        session = await manager.get_device_session("device-1")
        result = await manager.end_session(session.id, reason="completed")
        assert result is True

        # Device should return to listening
        state = manager.get_device_state("device-1")
        assert state == VoiceActivityState.LISTENING

        await manager.stop()


class TestCommandProcessing:
    """Tests for voice command processing."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> VoiceWakeManager:
        """Create a voice wake manager for testing."""
        return VoiceWakeManager(storage_path=tmp_path / "voice")

    @pytest.mark.asyncio
    async def test_process_command(self, manager: VoiceWakeManager):
        """Test processing a voice command."""
        await manager.start()
        await manager.start_listening("device-1")
        await manager.simulate_wake_detection("device-1")

        session = await manager.get_device_session("device-1")
        command = await manager.process_command(
            session_id=session.id,
            transcript="play some music",
            confidence=0.9,
        )

        assert command is not None
        assert command.transcript == "play some music"
        assert command.confidence == 0.9
        assert command.intent == "media.play"

        await manager.stop()

    @pytest.mark.asyncio
    async def test_command_intent_parsing(self, manager: VoiceWakeManager):
        """Test intent parsing for various commands."""
        await manager.start()
        await manager.start_listening("device-1")
        await manager.simulate_wake_detection("device-1")

        session = await manager.get_device_session("device-1")

        test_cases = [
            ("play music", "media.play"),
            ("stop the music", "media.pause"),
            ("what's the weather like", "query.weather"),
            ("what time is it", "query.time"),
            ("turn on the lights", "home.lights"),
            ("set a reminder", "reminder.set"),
            ("search for recipes", "search.web"),
            ("start a debate about AI", "aragora.debate"),
        ]

        for transcript, expected_intent in test_cases:
            command = await manager.process_command(
                session_id=session.id,
                transcript=transcript,
            )
            assert command.intent == expected_intent, f"Failed for '{transcript}'"

        await manager.stop()

    @pytest.mark.asyncio
    async def test_process_command_invalid_session(self, manager: VoiceWakeManager):
        """Test processing command with invalid session."""
        await manager.start()

        with pytest.raises(ValueError, match="not found"):
            await manager.process_command(
                session_id="nonexistent",
                transcript="test",
            )

        await manager.stop()

    @pytest.mark.asyncio
    async def test_list_commands(self, manager: VoiceWakeManager):
        """Test listing commands."""
        await manager.start()
        await manager.start_listening("device-1")
        await manager.simulate_wake_detection("device-1")

        session = await manager.get_device_session("device-1")

        for i in range(3):
            await manager.process_command(
                session_id=session.id,
                transcript=f"command {i}",
            )

        commands = await manager.list_commands()
        assert len(commands) == 3

        device_commands = await manager.list_commands(device_id="device-1")
        assert len(device_commands) == 3

        await manager.stop()

    @pytest.mark.asyncio
    async def test_command_callback(self, manager: VoiceWakeManager):
        """Test command processed callback."""
        await manager.start()
        await manager.start_listening("device-1")
        await manager.simulate_wake_detection("device-1")

        commands_received = []
        manager.on_command_processed(lambda c: commands_received.append(c))

        session = await manager.get_device_session("device-1")
        await manager.process_command(
            session_id=session.id,
            transcript="test command",
        )

        assert len(commands_received) == 1
        assert commands_received[0].transcript == "test command"

        await manager.stop()


class TestVoiceWakeStats:
    """Tests for voice wake statistics."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> VoiceWakeManager:
        """Create a voice wake manager for testing."""
        return VoiceWakeManager(storage_path=tmp_path / "voice")

    @pytest.mark.asyncio
    async def test_get_stats(self, manager: VoiceWakeManager):
        """Test getting voice wake stats."""
        await manager.start()
        await manager.start_listening("device-1")
        await manager.simulate_wake_detection("device-1")

        session = await manager.get_device_session("device-1")
        await manager.process_command(session.id, "play music")

        stats = await manager.get_stats()

        assert stats["running"] is True
        assert stats["active_listeners"] == 1
        assert stats["total_wake_events"] == 1
        assert stats["total_commands"] == 1
        assert "media.play" in stats["commands_by_intent"]

        await manager.stop()
