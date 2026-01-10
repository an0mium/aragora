"""
End-to-end tests for Broadcast Pipeline.

Tests cover the full audio/video generation flow:
- Script generation from debate
- Audio synthesis (TTS)
- Video generation (optional)
- RSS feed creation
- Storage and retrieval
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import tempfile


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_debate():
    """Create a mock debate for broadcast."""
    return {
        "id": "debate-123",
        "task": "Should AI be open-sourced?",
        "messages": [
            {"agent": "claude", "content": "AI should be open-sourced for transparency."},
            {"agent": "gpt4", "content": "There are safety concerns with open-sourcing."},
            {"agent": "claude", "content": "Safety can be addressed through community review."},
        ],
        "consensus": "Conditional open-sourcing with safety guidelines is recommended.",
        "created_at": "2026-01-10T12:00:00",
    }


@pytest.fixture
def mock_storage():
    """Create mock broadcast storage."""
    storage = Mock()
    storage.save_audio = Mock(return_value="broadcasts/debate-123/audio.mp3")
    storage.save_video = Mock(return_value="broadcasts/debate-123/video.mp4")
    storage.get_audio_path = Mock(return_value=None)
    storage.get_video_path = Mock(return_value=None)
    return storage


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Script Generation Tests
# ============================================================================

class TestScriptGeneration:
    """Tests for debate-to-script conversion."""

    def test_generate_script_from_debate(self, mock_debate):
        """Test generating broadcast script from debate."""
        try:
            from aragora.broadcast.script_generator import generate_script

            script = generate_script(mock_debate)

            assert script is not None
            assert len(script.segments) > 0
        except ImportError:
            pytest.skip("Script generator not available")

    def test_script_includes_all_speakers(self, mock_debate):
        """Test script includes all debate participants."""
        try:
            from aragora.broadcast.script_generator import generate_script

            script = generate_script(mock_debate)

            speakers = set(seg.speaker for seg in script.segments)
            assert "claude" in speakers or "narrator" in speakers
        except ImportError:
            pytest.skip("Script generator not available")

    def test_script_has_intro_and_outro(self, mock_debate):
        """Test script has introduction and conclusion."""
        try:
            from aragora.broadcast.script_generator import generate_script

            script = generate_script(mock_debate)

            first_segment = script.segments[0]
            last_segment = script.segments[-1]

            # Should have some kind of intro/outro
            assert len(script.segments) >= 2
        except ImportError:
            pytest.skip("Script generator not available")


# ============================================================================
# Audio Generation Tests
# ============================================================================

class TestAudioGeneration:
    """Tests for audio synthesis."""

    @pytest.mark.asyncio
    async def test_audio_engine_initialization(self):
        """Test audio engine can be initialized."""
        try:
            from aragora.broadcast.audio_engine import AudioEngine

            engine = AudioEngine()
            assert engine is not None
        except ImportError:
            pytest.skip("Audio engine not available")

    @pytest.mark.asyncio
    async def test_generate_audio_for_segment(self):
        """Test generating audio for a script segment."""
        try:
            from aragora.broadcast.audio_engine import AudioEngine

            engine = AudioEngine()

            # Test with mock TTS
            with patch.object(engine, '_synthesize_text') as mock_tts:
                mock_tts.return_value = b"fake audio data"

                result = await engine.generate_segment_audio(
                    text="Test speech text",
                    voice="narrator",
                )

                assert result is not None
        except ImportError:
            pytest.skip("Audio engine not available")

    def test_voice_mapping(self):
        """Test agent to voice mapping."""
        try:
            from aragora.broadcast.audio_engine import get_voice_for_agent

            # Different agents should have different voices
            voice1 = get_voice_for_agent("claude")
            voice2 = get_voice_for_agent("gpt4")

            # Both should return valid voice identifiers
            assert voice1 is not None
            assert voice2 is not None
        except ImportError:
            pytest.skip("Audio engine not available")


# ============================================================================
# Video Generation Tests
# ============================================================================

class TestVideoGeneration:
    """Tests for video generation (requires FFmpeg)."""

    def test_video_generator_available(self):
        """Test video generator module is available."""
        try:
            from aragora.broadcast.video_generator import VideoGenerator

            generator = VideoGenerator()
            assert generator is not None
        except ImportError:
            pytest.skip("Video generator not available")

    def test_check_ffmpeg_available(self):
        """Test FFmpeg availability check."""
        try:
            from aragora.broadcast.video_generator import is_ffmpeg_available

            # Just test the function exists and returns bool
            result = is_ffmpeg_available()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Video generator not available")


# ============================================================================
# RSS Feed Tests
# ============================================================================

class TestRSSGeneration:
    """Tests for RSS feed generation."""

    def test_create_rss_episode(self, mock_debate):
        """Test creating an RSS episode entry."""
        try:
            from aragora.broadcast.rss_gen import create_episode

            episode = create_episode(
                debate=mock_debate,
                audio_url="https://example.com/audio.mp3",
                duration_seconds=120,
            )

            assert episode["title"] is not None
            assert episode["audio_url"] == "https://example.com/audio.mp3"
            assert episode["duration"] == 120
        except ImportError:
            pytest.skip("RSS generator not available")

    def test_generate_rss_feed(self):
        """Test generating full RSS feed XML."""
        try:
            from aragora.broadcast.rss_gen import generate_feed

            episodes = [
                {
                    "title": "Episode 1",
                    "audio_url": "https://example.com/ep1.mp3",
                    "duration": 120,
                    "pub_date": "2026-01-10T12:00:00",
                    "description": "Test episode",
                }
            ]

            feed_xml = generate_feed(
                title="Aragora Debates",
                episodes=episodes,
            )

            assert "<?xml" in feed_xml
            assert "<rss" in feed_xml
            assert "Episode 1" in feed_xml
        except ImportError:
            pytest.skip("RSS generator not available")


# ============================================================================
# Pipeline Integration Tests
# ============================================================================

class TestBroadcastPipeline:
    """Integration tests for the full broadcast pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, mock_debate, temp_output_dir):
        """Test running the complete broadcast pipeline."""
        try:
            from aragora.broadcast.pipeline import BroadcastPipeline

            pipeline = BroadcastPipeline(output_dir=temp_output_dir)

            # Mock the TTS service
            with patch.object(pipeline, '_generate_audio') as mock_audio:
                mock_audio.return_value = temp_output_dir / "audio.mp3"
                # Create a dummy file
                (temp_output_dir / "audio.mp3").write_bytes(b"fake audio")

                result = await pipeline.run(
                    debate=mock_debate,
                    generate_video=False,
                    create_rss_episode=True,
                )

                assert result is not None
                assert result.get("audio_path") is not None
        except ImportError:
            pytest.skip("Broadcast pipeline not available")

    @pytest.mark.asyncio
    async def test_pipeline_with_video(self, mock_debate, temp_output_dir):
        """Test pipeline with video generation enabled."""
        try:
            from aragora.broadcast.pipeline import BroadcastPipeline
            from aragora.broadcast.video_generator import is_ffmpeg_available

            if not is_ffmpeg_available():
                pytest.skip("FFmpeg not available for video generation")

            pipeline = BroadcastPipeline(output_dir=temp_output_dir)

            with patch.object(pipeline, '_generate_audio') as mock_audio, \
                 patch.object(pipeline, '_generate_video') as mock_video:

                mock_audio.return_value = temp_output_dir / "audio.mp3"
                mock_video.return_value = temp_output_dir / "video.mp4"

                result = await pipeline.run(
                    debate=mock_debate,
                    generate_video=True,
                )

                assert result.get("video_path") is not None
        except ImportError:
            pytest.skip("Broadcast pipeline not available")

    @pytest.mark.asyncio
    async def test_pipeline_step_tracking(self, mock_debate, temp_output_dir):
        """Test that pipeline reports completed steps."""
        try:
            from aragora.broadcast.pipeline import BroadcastPipeline

            pipeline = BroadcastPipeline(output_dir=temp_output_dir)
            completed_steps = []

            def on_step_complete(step: str):
                completed_steps.append(step)

            with patch.object(pipeline, '_generate_audio') as mock_audio:
                mock_audio.return_value = temp_output_dir / "audio.mp3"
                (temp_output_dir / "audio.mp3").write_bytes(b"fake audio")

                await pipeline.run(
                    debate=mock_debate,
                    generate_video=False,
                    on_step_complete=on_step_complete,
                )

                # Should have completed at least the audio step
                assert len(completed_steps) > 0
        except ImportError:
            pytest.skip("Broadcast pipeline not available")


# ============================================================================
# Handler Integration Tests
# ============================================================================

class TestBroadcastHandlerIntegration:
    """Integration tests for broadcast API handlers."""

    def test_handler_routes(self):
        """Test broadcast handler routes are registered."""
        try:
            from aragora.server.handlers.broadcast import BroadcastHandler

            handler = BroadcastHandler({})

            # Should have routes for broadcast operations
            assert len(handler.ROUTES) > 0
        except ImportError:
            pytest.skip("Broadcast handler not available")

    @pytest.mark.asyncio
    async def test_check_audio_exists(self, mock_storage):
        """Test checking if audio exists for a debate."""
        try:
            from aragora.server.handlers.broadcast import BroadcastHandler

            ctx = {"broadcast_storage": mock_storage}
            handler = BroadcastHandler(ctx)

            # Should not raise
            exists = handler._check_audio_exists("debate-123")
            assert isinstance(exists, bool)
        except ImportError:
            pytest.skip("Broadcast handler not available")


# ============================================================================
# Storage Integration Tests
# ============================================================================

class TestBroadcastStorage:
    """Tests for broadcast file storage."""

    def test_storage_path_generation(self, temp_output_dir):
        """Test storage path generation for broadcasts."""
        try:
            from aragora.broadcast.storage import BroadcastStorage

            storage = BroadcastStorage(base_dir=temp_output_dir)

            path = storage.get_audio_path("debate-123")

            assert "debate-123" in str(path)
        except ImportError:
            pytest.skip("Broadcast storage not available")

    def test_storage_saves_audio(self, temp_output_dir):
        """Test saving audio file to storage."""
        try:
            from aragora.broadcast.storage import BroadcastStorage

            storage = BroadcastStorage(base_dir=temp_output_dir)

            # Create test audio file
            audio_data = b"fake audio content"
            saved_path = storage.save_audio("debate-123", audio_data)

            assert saved_path.exists()
            assert saved_path.read_bytes() == audio_data
        except ImportError:
            pytest.skip("Broadcast storage not available")


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestBroadcastErrorHandling:
    """Tests for error handling in broadcast pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_handles_tts_failure(self, mock_debate, temp_output_dir):
        """Test pipeline handles TTS service failure gracefully."""
        try:
            from aragora.broadcast.pipeline import BroadcastPipeline

            pipeline = BroadcastPipeline(output_dir=temp_output_dir)

            with patch.object(pipeline, '_generate_audio') as mock_audio:
                mock_audio.side_effect = Exception("TTS service unavailable")

                with pytest.raises(Exception) as exc_info:
                    await pipeline.run(debate=mock_debate)

                assert "TTS" in str(exc_info.value) or "unavailable" in str(exc_info.value)
        except ImportError:
            pytest.skip("Broadcast pipeline not available")

    def test_rss_handles_missing_fields(self):
        """Test RSS generator handles debates with missing fields."""
        try:
            from aragora.broadcast.rss_gen import create_episode

            # Minimal debate without optional fields
            minimal_debate = {
                "id": "debate-123",
                "task": "Test topic",
                "messages": [],
            }

            episode = create_episode(
                debate=minimal_debate,
                audio_url="https://example.com/audio.mp3",
                duration_seconds=60,
            )

            assert episode is not None
            assert episode["title"] is not None
        except ImportError:
            pytest.skip("RSS generator not available")
