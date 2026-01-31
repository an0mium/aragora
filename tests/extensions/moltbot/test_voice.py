"""
Tests for Moltbot VoiceProcessor - Speech-to-Text and Text-to-Speech Integration.

Tests voice session management, STT/TTS processing, and transcript handling.
"""

import asyncio
import pytest
from pathlib import Path

from aragora.extensions.moltbot import VoiceProcessor, VoiceSessionConfig


class TestVoiceProcessorBasic:
    """Tests for basic VoiceProcessor operations."""

    @pytest.fixture
    def processor(self, tmp_path: Path) -> VoiceProcessor:
        """Create a voice processor for testing."""
        return VoiceProcessor(storage_path=tmp_path / "voice")

    def test_create_processor(self, processor: VoiceProcessor):
        """Test creating a voice processor."""
        assert processor is not None

    def test_default_providers(self, processor: VoiceProcessor):
        """Test default mock providers are registered."""
        # Mock providers should be registered by default
        assert "mock" in processor._stt_handlers
        assert "mock" in processor._tts_handlers


class TestVoiceSessionManagement:
    """Tests for voice session management."""

    @pytest.fixture
    def processor(self, tmp_path: Path) -> VoiceProcessor:
        """Create a voice processor for testing."""
        return VoiceProcessor(storage_path=tmp_path / "voice")

    @pytest.mark.asyncio
    async def test_create_session(self, processor: VoiceProcessor):
        """Test creating a voice session."""
        config = VoiceSessionConfig()

        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        assert session is not None
        assert session.id is not None
        assert session.user_id == "user-1"
        assert session.channel_id == "channel-1"
        assert session.status == "active"
        assert session.started_at is not None

    @pytest.mark.asyncio
    async def test_create_session_with_tenant(self, processor: VoiceProcessor):
        """Test creating session with tenant."""
        config = VoiceSessionConfig()

        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
            tenant_id="tenant-1",
        )

        assert session.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_create_session_custom_config(self, processor: VoiceProcessor):
        """Test creating session with custom config."""
        config = VoiceSessionConfig(
            language="es-ES",
            sample_rate=48000,
            enable_vad=False,
        )

        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        assert session.config.language == "es-ES"
        assert session.config.sample_rate == 48000
        assert session.config.enable_vad is False

    @pytest.mark.asyncio
    async def test_get_session(self, processor: VoiceProcessor):
        """Test getting a session by ID."""
        config = VoiceSessionConfig()
        created = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        session = await processor.get_session(created.id)

        assert session is not None
        assert session.id == created.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, processor: VoiceProcessor):
        """Test getting nonexistent session."""
        session = await processor.get_session("nonexistent")
        assert session is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, processor: VoiceProcessor):
        """Test listing sessions."""
        config = VoiceSessionConfig()

        for i in range(3):
            await processor.create_session(
                config=config,
                user_id=f"user-{i}",
                channel_id="channel-1",
            )

        sessions = await processor.list_sessions()
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_list_sessions_by_user(self, processor: VoiceProcessor):
        """Test listing sessions by user."""
        config = VoiceSessionConfig()

        await processor.create_session(config=config, user_id="user-1", channel_id="ch-1")
        await processor.create_session(config=config, user_id="user-2", channel_id="ch-1")
        await processor.create_session(config=config, user_id="user-1", channel_id="ch-2")

        user1_sessions = await processor.list_sessions(user_id="user-1")
        assert len(user1_sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_by_status(self, processor: VoiceProcessor):
        """Test listing sessions by status."""
        config = VoiceSessionConfig()

        session1 = await processor.create_session(
            config=config, user_id="user-1", channel_id="ch-1"
        )
        session2 = await processor.create_session(
            config=config, user_id="user-2", channel_id="ch-1"
        )

        await processor.end_session(session1.id)

        active = await processor.list_sessions(status="active")
        ended = await processor.list_sessions(status="ended")

        assert len(active) == 1
        assert len(ended) == 1

    @pytest.mark.asyncio
    async def test_end_session(self, processor: VoiceProcessor):
        """Test ending a session."""
        config = VoiceSessionConfig()
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        ended = await processor.end_session(session.id, reason="completed")

        assert ended is not None
        assert ended.status == "ended"
        assert ended.ended_at is not None
        assert ended.duration_seconds >= 0
        assert ended.metadata["end_reason"] == "completed"

    @pytest.mark.asyncio
    async def test_end_nonexistent_session(self, processor: VoiceProcessor):
        """Test ending nonexistent session."""
        result = await processor.end_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_pause_session(self, processor: VoiceProcessor):
        """Test pausing a session."""
        config = VoiceSessionConfig()
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        paused = await processor.pause_session(session.id)

        assert paused is not None
        assert paused.status == "paused"

    @pytest.mark.asyncio
    async def test_resume_session(self, processor: VoiceProcessor):
        """Test resuming a paused session."""
        config = VoiceSessionConfig()
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        await processor.pause_session(session.id)
        resumed = await processor.resume_session(session.id)

        assert resumed is not None
        assert resumed.status == "active"

    @pytest.mark.asyncio
    async def test_resume_non_paused_session(self, processor: VoiceProcessor):
        """Test resuming non-paused session returns same session."""
        config = VoiceSessionConfig()
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        resumed = await processor.resume_session(session.id)

        assert resumed is not None
        assert resumed.status == "active"  # Was already active


class TestSpeechToText:
    """Tests for speech-to-text processing."""

    @pytest.fixture
    def processor(self, tmp_path: Path) -> VoiceProcessor:
        """Create a voice processor for testing."""
        return VoiceProcessor(storage_path=tmp_path / "voice")

    @pytest.mark.asyncio
    async def test_process_audio(self, processor: VoiceProcessor):
        """Test processing audio."""
        config = VoiceSessionConfig()
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        result = await processor.process_audio(
            session_id=session.id,
            audio_data=b"fake_audio_data",
        )

        assert result["success"] is True
        assert "transcript" in result
        assert result["is_final"] is True

    @pytest.mark.asyncio
    async def test_process_audio_updates_session(self, processor: VoiceProcessor):
        """Test processing audio updates session."""
        config = VoiceSessionConfig()
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        await processor.process_audio(session_id=session.id, audio_data=b"audio")

        updated = await processor.get_session(session.id)

        assert len(updated.transcripts) == 1
        assert updated.transcripts[0]["type"] == "user"
        assert updated.turns == 1
        assert updated.words_heard > 0

    @pytest.mark.asyncio
    async def test_process_audio_nonexistent_session(self, processor: VoiceProcessor):
        """Test processing audio for nonexistent session."""
        result = await processor.process_audio(
            session_id="nonexistent",
            audio_data=b"audio",
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_process_audio_ended_session(self, processor: VoiceProcessor):
        """Test processing audio for ended session."""
        config = VoiceSessionConfig()
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )
        await processor.end_session(session.id)

        result = await processor.process_audio(
            session_id=session.id,
            audio_data=b"audio",
        )

        assert result["success"] is False
        assert "ended" in result["error"]

    @pytest.mark.asyncio
    async def test_process_audio_stt_disabled(self, processor: VoiceProcessor):
        """Test processing audio with STT disabled."""
        config = VoiceSessionConfig(enable_stt=False)
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        result = await processor.process_audio(
            session_id=session.id,
            audio_data=b"audio",
        )

        assert result["success"] is False
        assert "not enabled" in result["error"]


class TestTextToSpeech:
    """Tests for text-to-speech processing."""

    @pytest.fixture
    def processor(self, tmp_path: Path) -> VoiceProcessor:
        """Create a voice processor for testing."""
        return VoiceProcessor(storage_path=tmp_path / "voice")

    @pytest.mark.asyncio
    async def test_synthesize_speech(self, processor: VoiceProcessor):
        """Test synthesizing speech."""
        config = VoiceSessionConfig()
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        result = await processor.synthesize_speech(
            session_id=session.id,
            text="Hello, how can I help you?",
        )

        assert result["success"] is True
        assert "audio" in result
        assert result["format"] == "pcm"
        assert result["sample_rate"] == 16000

    @pytest.mark.asyncio
    async def test_synthesize_speech_updates_session(self, processor: VoiceProcessor):
        """Test synthesizing speech updates session."""
        config = VoiceSessionConfig()
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        await processor.synthesize_speech(
            session_id=session.id,
            text="Hello world",
        )

        updated = await processor.get_session(session.id)

        assert len(updated.transcripts) == 1
        assert updated.transcripts[0]["type"] == "system"
        assert updated.transcripts[0]["text"] == "Hello world"
        assert updated.words_spoken == 2

    @pytest.mark.asyncio
    async def test_synthesize_speech_nonexistent_session(self, processor: VoiceProcessor):
        """Test synthesizing speech for nonexistent session."""
        result = await processor.synthesize_speech(
            session_id="nonexistent",
            text="Hello",
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_synthesize_speech_tts_disabled(self, processor: VoiceProcessor):
        """Test synthesizing speech with TTS disabled."""
        config = VoiceSessionConfig(enable_tts=False)
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        result = await processor.synthesize_speech(
            session_id=session.id,
            text="Hello",
        )

        assert result["success"] is False
        assert "not enabled" in result["error"]


class TestIntentDetection:
    """Tests for voice intent detection."""

    @pytest.fixture
    def processor(self, tmp_path: Path) -> VoiceProcessor:
        """Create a voice processor for testing."""
        return VoiceProcessor(storage_path=tmp_path / "voice")

    def test_detect_end_conversation(self, processor: VoiceProcessor):
        """Test detecting end conversation intent."""
        assert processor._detect_intent("stop") == "end_conversation"
        assert processor._detect_intent("goodbye") == "end_conversation"
        assert processor._detect_intent("end the session") == "end_conversation"

    def test_detect_help_request(self, processor: VoiceProcessor):
        """Test detecting help request intent."""
        assert processor._detect_intent("I need help") == "help_request"
        assert processor._detect_intent("can you assist me") == "help_request"
        assert processor._detect_intent("support please") == "help_request"

    def test_detect_clarification(self, processor: VoiceProcessor):
        """Test detecting clarification intent."""
        assert processor._detect_intent("can you repeat that") == "clarification"
        assert processor._detect_intent("say that again") == "clarification"
        assert processor._detect_intent("what did you say") == "clarification"

    def test_detect_question(self, processor: VoiceProcessor):
        """Test detecting question intent."""
        assert processor._detect_intent("How does this work?") == "question"
        # Note: "What" triggers clarification first, so use a different phrase
        assert processor._detect_intent("Is this correct?") == "question"
        assert processor._detect_intent("Can you explain?") == "question"

    def test_no_intent(self, processor: VoiceProcessor):
        """Test no specific intent detected."""
        assert processor._detect_intent("Thanks for the info") is None
        assert processor._detect_intent("That sounds good") is None


class TestProviderManagement:
    """Tests for STT/TTS provider management."""

    @pytest.fixture
    def processor(self, tmp_path: Path) -> VoiceProcessor:
        """Create a voice processor for testing."""
        return VoiceProcessor(storage_path=tmp_path / "voice")

    @pytest.mark.asyncio
    async def test_register_stt_provider(self, processor: VoiceProcessor):
        """Test registering an STT provider."""

        async def custom_stt(audio_data, config):
            return "Custom transcription"

        processor.register_stt_provider("custom", custom_stt)

        assert "custom" in processor._stt_handlers

    @pytest.mark.asyncio
    async def test_register_tts_provider(self, processor: VoiceProcessor):
        """Test registering a TTS provider."""

        async def custom_tts(text, config):
            return b"custom_audio"

        processor.register_tts_provider("custom", custom_tts)

        assert "custom" in processor._tts_handlers

    @pytest.mark.asyncio
    async def test_set_stt_provider(self, processor: VoiceProcessor):
        """Test setting active STT provider."""

        # Register custom provider
        async def custom_stt(audio_data, config):
            return "Custom"

        processor.register_stt_provider("custom", custom_stt)

        # Set as active
        result = processor.set_stt_provider("custom")
        assert result is True
        assert processor._stt_provider == "custom"

    @pytest.mark.asyncio
    async def test_set_stt_provider_unknown(self, processor: VoiceProcessor):
        """Test setting unknown STT provider."""
        result = processor.set_stt_provider("unknown")
        assert result is False

    @pytest.mark.asyncio
    async def test_set_tts_provider(self, processor: VoiceProcessor):
        """Test setting active TTS provider."""

        async def custom_tts(text, config):
            return b"custom"

        processor.register_tts_provider("custom", custom_tts)

        result = processor.set_tts_provider("custom")
        assert result is True
        assert processor._tts_provider == "custom"

    @pytest.mark.asyncio
    async def test_set_tts_provider_unknown(self, processor: VoiceProcessor):
        """Test setting unknown TTS provider."""
        result = processor.set_tts_provider("unknown")
        assert result is False


class TestTranscriptManagement:
    """Tests for transcript management."""

    @pytest.fixture
    def processor(self, tmp_path: Path) -> VoiceProcessor:
        """Create a voice processor for testing."""
        return VoiceProcessor(storage_path=tmp_path / "voice")

    @pytest.mark.asyncio
    async def test_get_transcript(self, processor: VoiceProcessor):
        """Test getting session transcript."""
        config = VoiceSessionConfig()
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        await processor.process_audio(session.id, b"audio1")
        await processor.synthesize_speech(session.id, "Response 1")
        await processor.process_audio(session.id, b"audio2")

        transcript = await processor.get_transcript(session.id)

        assert len(transcript) == 3
        assert transcript[0]["type"] == "user"
        assert transcript[1]["type"] == "system"
        assert transcript[2]["type"] == "user"

    @pytest.mark.asyncio
    async def test_get_transcript_nonexistent(self, processor: VoiceProcessor):
        """Test getting transcript for nonexistent session."""
        transcript = await processor.get_transcript("nonexistent")
        assert transcript == []

    @pytest.mark.asyncio
    async def test_transcript_persisted_on_end(self, processor: VoiceProcessor):
        """Test transcript is persisted when session ends."""
        config = VoiceSessionConfig()
        session = await processor.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        await processor.process_audio(session.id, b"audio")
        await processor.synthesize_speech(session.id, "Response")
        await processor.end_session(session.id)

        # Check if transcript file was created
        transcript_file = processor._storage_path / f"{session.id}.json"
        assert transcript_file.exists()


class TestVoiceStats:
    """Tests for voice processor statistics."""

    @pytest.fixture
    def processor(self, tmp_path: Path) -> VoiceProcessor:
        """Create a voice processor for testing."""
        return VoiceProcessor(storage_path=tmp_path / "voice")

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, processor: VoiceProcessor):
        """Test getting stats with no sessions."""
        stats = await processor.get_stats()

        assert stats["sessions_total"] == 0
        assert stats["sessions_active"] == 0
        assert stats["sessions_ended"] == 0
        assert stats["total_turns"] == 0
        assert stats["total_words_spoken"] == 0
        assert stats["total_words_heard"] == 0
        assert stats["stt_provider"] == "mock"
        assert stats["tts_provider"] == "mock"

    @pytest.mark.asyncio
    async def test_get_stats_with_sessions(self, processor: VoiceProcessor):
        """Test getting stats with sessions."""
        config = VoiceSessionConfig()

        # Create sessions
        session1 = await processor.create_session(config=config, user_id="u1", channel_id="c1")
        session2 = await processor.create_session(config=config, user_id="u2", channel_id="c1")

        # Process some audio
        await processor.process_audio(session1.id, b"audio")
        await processor.synthesize_speech(session1.id, "Hello there user")

        # End one session
        await processor.end_session(session2.id)

        stats = await processor.get_stats()

        assert stats["sessions_total"] == 2
        assert stats["sessions_active"] == 1
        assert stats["sessions_ended"] == 1
        assert stats["total_turns"] == 1
        assert stats["total_words_spoken"] == 3  # "Hello there user"
        assert "mock" in stats["stt_providers_available"]
        assert "mock" in stats["tts_providers_available"]
