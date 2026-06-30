"""
End-to-end tests for the broadcast pipeline.

Tests the full debate-to-publication flow:
1. Audio generation from debate traces
2. Video generation (when FFmpeg available)
3. RSS feed creation
4. Storage persistence

These tests require mocking TTS backends but test the real pipeline logic.
"""

import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.broadcast.pipeline import (
    BroadcastPipeline,
    BroadcastOptions,
    PipelineResult,
)


@dataclass
class MockDebateTrace:
    """Mock debate trace for testing."""

    id: str = "test-debate-123"
    task: str = "Should AI systems have human oversight?"
    agents: list = None
    messages: list = None
    consensus_reached: bool = True
    confidence: float = 0.85
    rounds_used: int = 3

    def __post_init__(self):
        if self.agents is None:
            self.agents = ["claude-3-opus", "gpt-4", "gemini-pro"]
        if self.messages is None:
            self.messages = [
                {
                    "role": "proposer",
                    "agent": "claude-3-opus",
                    "content": "AI systems should have human oversight...",
                },
                {
                    "role": "critic",
                    "agent": "gpt-4",
                    "content": "I agree with the core premise but...",
                },
                {
                    "role": "synthesizer",
                    "agent": "gemini-pro",
                    "content": "Building on both perspectives...",
                },
            ]

    @classmethod
    def load(cls, path: Path) -> "MockDebateTrace":
        """Load trace from JSON file."""
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return cls(**data)
        return cls()


@pytest.fixture
def temp_nomic_dir():
    """Create a temporary nomic directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / "traces").mkdir()
        (base / "audio").mkdir()
        (base / "video").mkdir()
        yield base


@pytest.fixture
def sample_trace(temp_nomic_dir):
    """Create a sample debate trace file."""
    trace = MockDebateTrace()
    trace_path = temp_nomic_dir / "traces" / f"{trace.id}.json"
    with open(trace_path, "w") as f:
        json.dump(
            {
                "id": trace.id,
                "task": trace.task,
                "agents": trace.agents,
                "messages": trace.messages,
                "consensus_reached": trace.consensus_reached,
                "confidence": trace.confidence,
                "rounds_used": trace.rounds_used,
            },
            f,
        )
    return trace


@pytest.fixture
def pipeline(temp_nomic_dir):
    """Create a BroadcastPipeline instance."""
    return BroadcastPipeline(nomic_dir=temp_nomic_dir)


class TestBroadcastPipelineInit:
    """Tests for pipeline initialization."""

    def test_creates_output_directories(self, temp_nomic_dir):
        """Pipeline should create audio and video directories."""
        # Remove directories to test creation
        (temp_nomic_dir / "audio").rmdir()
        (temp_nomic_dir / "video").rmdir()

        pipeline = BroadcastPipeline(nomic_dir=temp_nomic_dir)

        assert pipeline.audio_dir.exists()
        assert pipeline.video_dir.exists()

    def test_accepts_custom_stores(self, temp_nomic_dir):
        """Pipeline should accept custom audio_store and rss_generator."""
        mock_audio_store = MagicMock()
        mock_rss_gen = MagicMock()

        pipeline = BroadcastPipeline(
            nomic_dir=temp_nomic_dir,
            audio_store=mock_audio_store,
            rss_generator=mock_rss_gen,
        )

        assert pipeline.audio_store is mock_audio_store
        assert pipeline.rss_generator is mock_rss_gen


class TestBroadcastPipelineRun:
    """Tests for the main pipeline run method."""

    @pytest.mark.asyncio
    async def test_returns_error_when_trace_not_found(self, pipeline):
        """Pipeline should return error when debate trace doesn't exist."""
        result = await pipeline.run("nonexistent-debate")

        assert result.success is False
        assert result.error_message == "Debate trace not found"
        assert result.steps_completed == []

    @pytest.mark.asyncio
    async def test_audio_generation_step(self, pipeline, sample_trace, temp_nomic_dir):
        """Pipeline should complete audio generation step."""
        # Create mock audio file
        mock_audio_path = temp_nomic_dir / "audio" / f"{sample_trace.id}.mp3"

        with patch.object(pipeline, "_load_trace", return_value=sample_trace):
            with patch.object(pipeline, "_generate_audio", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = mock_audio_path
                with patch.object(pipeline, "_get_audio_duration", return_value=120):
                    result = await pipeline.run(sample_trace.id)

        assert result.success is True
        assert "audio" in result.steps_completed
        assert result.audio_path == mock_audio_path
        assert result.duration_seconds == 120

    @pytest.mark.asyncio
    async def test_audio_failure_stops_pipeline(self, pipeline, sample_trace):
        """Pipeline should stop if audio generation fails."""
        with patch.object(pipeline, "_load_trace", return_value=sample_trace):
            with patch.object(pipeline, "_generate_audio", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = None
                result = await pipeline.run(sample_trace.id)

        assert result.success is False
        assert result.error_message == "Audio generation failed"
        assert "audio" not in result.steps_completed

    @pytest.mark.asyncio
    async def test_video_generation_step(self, pipeline, sample_trace, temp_nomic_dir):
        """Pipeline should complete video generation when enabled."""
        mock_audio_path = temp_nomic_dir / "audio" / f"{sample_trace.id}.mp3"
        mock_video_path = temp_nomic_dir / "video" / f"{sample_trace.id}.mp4"

        options = BroadcastOptions(video_enabled=True)

        with patch.object(pipeline, "_load_trace", return_value=sample_trace):
            with patch.object(pipeline, "_generate_audio", new_callable=AsyncMock) as mock_audio:
                mock_audio.return_value = mock_audio_path
                with patch.object(
                    pipeline, "_generate_video", new_callable=AsyncMock
                ) as mock_video:
                    mock_video.return_value = mock_video_path
                    with patch.object(pipeline, "_get_audio_duration", return_value=120):
                        result = await pipeline.run(sample_trace.id, options)

        assert result.success is True
        assert "audio" in result.steps_completed
        assert "video" in result.steps_completed
        assert result.video_path == mock_video_path

    @pytest.mark.asyncio
    async def test_video_failure_continues_pipeline(self, pipeline, sample_trace, temp_nomic_dir):
        """Pipeline should continue if video generation fails."""
        mock_audio_path = temp_nomic_dir / "audio" / f"{sample_trace.id}.mp3"

        options = BroadcastOptions(video_enabled=True)

        with patch.object(pipeline, "_load_trace", return_value=sample_trace):
            with patch.object(pipeline, "_generate_audio", new_callable=AsyncMock) as mock_audio:
                mock_audio.return_value = mock_audio_path
                with patch.object(
                    pipeline, "_generate_video", new_callable=AsyncMock
                ) as mock_video:
                    mock_video.return_value = None  # Video fails
                    with patch.object(pipeline, "_get_audio_duration", return_value=120):
                        result = await pipeline.run(sample_trace.id, options)

        # Pipeline should still succeed with audio
        assert result.success is True
        assert "audio" in result.steps_completed
        assert "video" not in result.steps_completed
        assert result.video_path is None

    @pytest.mark.asyncio
    async def test_rss_episode_creation(self, pipeline, sample_trace, temp_nomic_dir):
        """Pipeline should create RSS episode when enabled."""
        mock_audio_path = temp_nomic_dir / "audio" / f"{sample_trace.id}.mp3"
        mock_episode_guid = f"urn:uuid:{sample_trace.id}"

        options = BroadcastOptions(generate_rss_episode=True)

        with patch.object(pipeline, "_load_trace", return_value=sample_trace):
            with patch.object(pipeline, "_generate_audio", new_callable=AsyncMock) as mock_audio:
                mock_audio.return_value = mock_audio_path
                with patch.object(pipeline, "_create_rss_episode") as mock_rss:
                    mock_rss.return_value = mock_episode_guid
                    with patch.object(pipeline, "_get_audio_duration", return_value=120):
                        result = await pipeline.run(sample_trace.id, options)

        assert result.success is True
        assert "rss" in result.steps_completed
        assert result.rss_episode_guid == mock_episode_guid

    @pytest.mark.asyncio
    async def test_storage_persistence(self, pipeline, sample_trace, temp_nomic_dir):
        """Pipeline should persist to audio store when available."""
        mock_audio_path = temp_nomic_dir / "audio" / f"{sample_trace.id}.mp3"
        mock_stored_path = temp_nomic_dir / "stored" / f"{sample_trace.id}.mp3"

        mock_store = MagicMock()
        mock_store.save.return_value = mock_stored_path
        pipeline.audio_store = mock_store

        with patch.object(pipeline, "_load_trace", return_value=sample_trace):
            with patch.object(pipeline, "_generate_audio", new_callable=AsyncMock) as mock_audio:
                mock_audio.return_value = mock_audio_path
                with patch.object(pipeline, "_get_audio_duration", return_value=120):
                    result = await pipeline.run(sample_trace.id)

        assert result.success is True
        assert "storage" in result.steps_completed
        mock_store.save.assert_called_once()


class TestBroadcastOptions:
    """Tests for BroadcastOptions configuration."""

    def test_default_options(self):
        """Default options should enable audio and RSS, disable video."""
        options = BroadcastOptions()

        assert options.audio_enabled is True
        assert options.audio_format == "mp3"
        assert options.video_enabled is False
        assert options.generate_rss_episode is True

    def test_custom_options(self):
        """Custom options should be respected."""
        options = BroadcastOptions(
            audio_format="wav",
            video_enabled=True,
            video_resolution=(1280, 720),
            custom_title="Custom Title",
            episode_number=42,
        )

        assert options.audio_format == "wav"
        assert options.video_enabled is True
        assert options.video_resolution == (1280, 720)
        assert options.custom_title == "Custom Title"
        assert options.episode_number == 42


class TestPipelineResult:
    """Tests for PipelineResult data structure."""

    def test_result_initialization(self):
        """PipelineResult should initialize with correct defaults."""
        result = PipelineResult(debate_id="test-123", success=False)

        assert result.debate_id == "test-123"
        assert result.success is False
        assert result.audio_path is None
        assert result.video_path is None
        assert result.rss_episode_guid is None
        assert result.duration_seconds is None
        assert result.error_message is None
        assert result.steps_completed == []
        assert result.generated_at is not None  # Should have timestamp

    def test_result_with_all_fields(self):
        """PipelineResult should accept all fields."""
        result = PipelineResult(
            debate_id="test-123",
            success=True,
            audio_path=Path("/audio/test.mp3"),
            video_path=Path("/video/test.mp4"),
            rss_episode_guid="urn:uuid:123",
            duration_seconds=300,
            steps_completed=["audio", "video", "rss"],
        )

        assert result.success is True
        assert result.audio_path == Path("/audio/test.mp3")
        assert result.video_path == Path("/video/test.mp4")
        assert result.duration_seconds == 300


class TestFullPipelineIntegration:
    """Integration tests for the full pipeline (with mocked TTS)."""

    @pytest.mark.asyncio
    async def test_full_pipeline_audio_only(self, pipeline, sample_trace, temp_nomic_dir):
        """Test full pipeline with audio only (default settings)."""
        mock_audio_path = temp_nomic_dir / "audio" / f"{sample_trace.id}.mp3"
        mock_audio_path.write_bytes(b"fake audio data")

        with patch.object(pipeline, "_load_trace", return_value=sample_trace):
            with patch.object(pipeline, "_generate_audio", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = mock_audio_path
                with patch.object(pipeline, "_get_audio_duration", return_value=180):
                    with patch.object(pipeline, "_create_rss_episode", return_value="ep-123"):
                        result = await pipeline.run(sample_trace.id)

        assert result.success is True
        assert result.debate_id == sample_trace.id
        assert result.audio_path == mock_audio_path
        assert result.duration_seconds == 180
        assert "audio" in result.steps_completed
        assert "rss" in result.steps_completed

    @pytest.mark.asyncio
    async def test_full_pipeline_with_video(self, pipeline, sample_trace, temp_nomic_dir):
        """Test full pipeline with video enabled."""
        mock_audio_path = temp_nomic_dir / "audio" / f"{sample_trace.id}.mp3"
        mock_video_path = temp_nomic_dir / "video" / f"{sample_trace.id}.mp4"
        mock_audio_path.write_bytes(b"fake audio data")
        mock_video_path.write_bytes(b"fake video data")

        options = BroadcastOptions(
            video_enabled=True,
            custom_title="AI Oversight Debate",
        )

        with patch.object(pipeline, "_load_trace", return_value=sample_trace):
            with patch.object(pipeline, "_generate_audio", new_callable=AsyncMock) as mock_audio:
                mock_audio.return_value = mock_audio_path
                with patch.object(
                    pipeline, "_generate_video", new_callable=AsyncMock
                ) as mock_video:
                    mock_video.return_value = mock_video_path
                    with patch.object(pipeline, "_get_audio_duration", return_value=240):
                        with patch.object(pipeline, "_create_rss_episode", return_value="ep-456"):
                            result = await pipeline.run(sample_trace.id, options)

        assert result.success is True
        assert result.video_path == mock_video_path
        assert len(result.steps_completed) >= 3
        assert all(step in result.steps_completed for step in ["audio", "video", "rss"])

    @pytest.mark.asyncio
    async def test_pipeline_graceful_degradation(self, pipeline, sample_trace, temp_nomic_dir):
        """Test that pipeline degrades gracefully on partial failures."""
        mock_audio_path = temp_nomic_dir / "audio" / f"{sample_trace.id}.mp3"
        mock_audio_path.write_bytes(b"fake audio data")

        options = BroadcastOptions(
            video_enabled=True,
            generate_rss_episode=True,
        )

        with patch.object(pipeline, "_load_trace", return_value=sample_trace):
            with patch.object(pipeline, "_generate_audio", new_callable=AsyncMock) as mock_audio:
                mock_audio.return_value = mock_audio_path
                with patch.object(
                    pipeline, "_generate_video", new_callable=AsyncMock
                ) as mock_video:
                    mock_video.return_value = None  # Video fails
                    with patch.object(pipeline, "_create_rss_episode") as mock_rss:
                        mock_rss.return_value = None  # RSS also fails
                        with patch.object(pipeline, "_get_audio_duration", return_value=60):
                            result = await pipeline.run(sample_trace.id, options)

        # Should still succeed with just audio
        assert result.success is True
        assert "audio" in result.steps_completed
        assert "video" not in result.steps_completed
        assert "rss" not in result.steps_completed


class TestPipelineErrorHandling:
    """Tests for error handling in the pipeline."""

    @pytest.mark.asyncio
    async def test_handles_load_trace_exception(self, pipeline):
        """Pipeline should propagate exceptions during trace loading."""
        with patch.object(pipeline, "_load_trace", side_effect=Exception("Load failed")):
            # Pipeline propagates exceptions rather than catching them
            with pytest.raises(Exception, match="Load failed"):
                await pipeline.run("test-debate")

    @pytest.mark.asyncio
    async def test_handles_audio_generation_exception(self, pipeline, sample_trace):
        """Pipeline should propagate exceptions during audio generation."""
        with patch.object(pipeline, "_load_trace", return_value=sample_trace):
            with patch.object(pipeline, "_generate_audio", new_callable=AsyncMock) as mock_gen:
                mock_gen.side_effect = Exception("TTS service unavailable")
                # Pipeline propagates the exception
                with pytest.raises(Exception, match="TTS service unavailable"):
                    await pipeline.run(sample_trace.id)

    @pytest.mark.asyncio
    async def test_handles_storage_exception(self, pipeline, sample_trace, temp_nomic_dir):
        """Pipeline should handle storage exceptions gracefully."""
        mock_audio_path = temp_nomic_dir / "audio" / f"{sample_trace.id}.mp3"

        mock_store = MagicMock()
        mock_store.save.side_effect = Exception("Storage full")
        pipeline.audio_store = mock_store

        with patch.object(pipeline, "_load_trace", return_value=sample_trace):
            with patch.object(pipeline, "_generate_audio", new_callable=AsyncMock) as mock_audio:
                mock_audio.return_value = mock_audio_path
                with patch.object(pipeline, "_get_audio_duration", return_value=120):
                    result = await pipeline.run(sample_trace.id)

        # Should still succeed, just without storage step
        assert result.success is True
        assert "storage" not in result.steps_completed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
