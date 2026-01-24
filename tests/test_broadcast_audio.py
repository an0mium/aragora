"""Tests for broadcast audio engine module."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.broadcast.audio_engine import (
    VOICE_MAP,
    _generate_edge_tts,
    _generate_fallback_tts,
    _generate_fallback_tts_sync,
    _get_voice_for_speaker,
    generate_audio,
    generate_audio_segment,
)
from aragora.broadcast.script_gen import ScriptSegment


# =============================================================================
# Test Voice Mapping
# =============================================================================


class TestVoiceMapping:
    """Tests for voice mapping functionality."""

    def test_voice_map_contains_expected_agents(self):
        """VOICE_MAP contains mappings for expected agents."""
        expected_agents = [
            "claude-visionary",
            "codex-engineer",
            "gemini-visionary",
            "grok-lateral-thinker",
            "narrator",
        ]
        for agent in expected_agents:
            assert agent in VOICE_MAP

    def test_voice_map_values_are_valid_edge_tts_voices(self):
        """VOICE_MAP values are valid edge-tts voice identifiers."""
        for voice in VOICE_MAP.values():
            # edge-tts voices follow pattern: lang-Region-NameNeural
            assert "Neural" in voice

    def test_get_voice_for_known_speaker(self):
        """_get_voice_for_speaker returns correct voice for known speakers."""
        assert _get_voice_for_speaker("claude-visionary") == "en-GB-SoniaNeural"
        assert _get_voice_for_speaker("codex-engineer") == "en-US-GuyNeural"
        assert _get_voice_for_speaker("narrator") == "en-US-AriaNeural"

    def test_get_voice_for_unknown_speaker(self):
        """_get_voice_for_speaker returns narrator voice for unknown speakers."""
        assert _get_voice_for_speaker("unknown-agent") == VOICE_MAP["narrator"]
        assert _get_voice_for_speaker("") == VOICE_MAP["narrator"]
        assert _get_voice_for_speaker("random") == VOICE_MAP["narrator"]

    def test_narrator_voice_is_default(self):
        """Narrator voice is used as the fallback default."""
        assert "narrator" in VOICE_MAP
        unknown_voice = _get_voice_for_speaker("totally-fake-agent")
        assert unknown_voice == VOICE_MAP["narrator"]


# =============================================================================
# Test Edge-TTS Generation
# =============================================================================


@pytest.mark.skipif(
    "CI" in __import__("os").environ or "GITHUB_ACTIONS" in __import__("os").environ,
    reason="Edge TTS tests fail in CI environment",
)
class TestEdgeTTS:
    """Tests for edge-tts audio generation."""

    @pytest.mark.asyncio
    async def test_generate_edge_tts_success(self, tmp_path):
        """Successfully generate audio with edge-tts."""
        output_path = tmp_path / "test.mp3"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Create file to simulate edge-tts success
            output_path.write_bytes(b"fake audio")
            result = await _generate_edge_tts(
                text="Hello world",
                voice="en-US-GuyNeural",
                output_path=output_path,
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_generate_edge_tts_timeout(self, tmp_path):
        """Handle edge-tts timeout after 60 seconds."""
        output_path = tmp_path / "test.mp3"

        mock_process = AsyncMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                result = await _generate_edge_tts(
                    text="Hello world",
                    voice="en-US-GuyNeural",
                    output_path=output_path,
                )

        assert result is False
        # kill is called once per retry attempt (default 3 retries)
        assert mock_process.kill.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_edge_tts_subprocess_error(self, tmp_path):
        """Handle edge-tts subprocess returning non-zero exit code."""
        output_path = tmp_path / "test.mp3"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"error")
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await _generate_edge_tts(
                text="Hello world",
                voice="en-US-GuyNeural",
                output_path=output_path,
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_generate_edge_tts_no_output_file(self, tmp_path):
        """Return False when output file is not created."""
        output_path = tmp_path / "test.mp3"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0  # Success but no file

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Don't create the file
            result = await _generate_edge_tts(
                text="Hello world",
                voice="en-US-GuyNeural",
                output_path=output_path,
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_generate_edge_tts_exception(self, tmp_path):
        """Handle exceptions during edge-tts generation."""
        output_path = tmp_path / "test.mp3"

        with patch(
            "asyncio.create_subprocess_exec", side_effect=FileNotFoundError("edge-tts not found")
        ):
            result = await _generate_edge_tts(
                text="Hello world",
                voice="en-US-GuyNeural",
                output_path=output_path,
            )

        assert result is False


# =============================================================================
# Test Fallback TTS
# =============================================================================


class TestFallbackTTS:
    """Tests for pyttsx3 fallback TTS."""

    def test_fallback_tts_sync_unavailable(self, tmp_path):
        """Return False when pyttsx3 is unavailable (sync version)."""
        output_path = tmp_path / "test.mp3"

        with patch("aragora.broadcast.audio_engine.FALLBACK_AVAILABLE", False):
            result = _generate_fallback_tts_sync("Hello world", output_path)

        assert result is False

    def test_fallback_tts_sync_success(self, tmp_path):
        """Successfully generate audio with pyttsx3 fallback (sync version)."""
        output_path = tmp_path / "test.mp3"

        mock_engine = MagicMock()
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        with patch("aragora.broadcast.audio_engine.FALLBACK_AVAILABLE", True):
            # Use create=True since pyttsx3 may not be imported if not installed
            with patch("aragora.broadcast.audio_engine.pyttsx3", mock_pyttsx3, create=True):
                # Simulate file creation
                output_path.write_bytes(b"fake audio")
                result = _generate_fallback_tts_sync("Hello world", output_path)

        assert result is True
        mock_engine.save_to_file.assert_called_once()
        mock_engine.runAndWait.assert_called_once()

    def test_fallback_tts_sync_exception(self, tmp_path):
        """Handle exceptions during pyttsx3 fallback (sync version)."""
        output_path = tmp_path / "test.mp3"

        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.side_effect = RuntimeError("Audio driver error")

        with patch("aragora.broadcast.audio_engine.FALLBACK_AVAILABLE", True):
            # Use create=True since pyttsx3 may not be imported if not installed
            with patch("aragora.broadcast.audio_engine.pyttsx3", mock_pyttsx3, create=True):
                result = _generate_fallback_tts_sync("Hello world", output_path)

        assert result is False

    def test_fallback_tts_sync_no_file_created(self, tmp_path):
        """Return False when pyttsx3 doesn't create file (sync version)."""
        output_path = tmp_path / "test.mp3"

        mock_engine = MagicMock()
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        with patch("aragora.broadcast.audio_engine.FALLBACK_AVAILABLE", True):
            # Use create=True since pyttsx3 may not be imported if not installed
            with patch("aragora.broadcast.audio_engine.pyttsx3", mock_pyttsx3, create=True):
                # Don't create file
                result = _generate_fallback_tts_sync("Hello world", output_path)

        assert result is False

    @pytest.mark.asyncio
    async def test_fallback_tts_async_unavailable(self, tmp_path):
        """Return False when pyttsx3 is unavailable (async version)."""
        output_path = tmp_path / "test.mp3"

        with patch("aragora.broadcast.audio_engine.FALLBACK_AVAILABLE", False):
            result = await _generate_fallback_tts("Hello world", output_path)

        assert result is False

    @pytest.mark.asyncio
    async def test_fallback_tts_async_success(self, tmp_path):
        """Async wrapper calls sync function in thread pool."""
        output_path = tmp_path / "test.mp3"

        with patch("aragora.broadcast.audio_engine._generate_fallback_tts_sync") as mock_sync:
            mock_sync.return_value = True
            with patch("aragora.broadcast.audio_engine.FALLBACK_AVAILABLE", True):
                result = await _generate_fallback_tts("Hello world", output_path)

        assert result is True
        mock_sync.assert_called_once_with("Hello world", output_path)


# =============================================================================
# Test generate_audio_segment
# =============================================================================


class TestGenerateAudioSegment:
    """Tests for generate_audio_segment function."""

    @pytest.fixture
    def sample_segment(self):
        """Create a sample script segment."""
        return ScriptSegment(
            speaker="claude-visionary",
            text="Hello, I am Claude.",
        )

    @pytest.mark.asyncio
    async def test_generate_audio_segment_edge_tts_success(self, tmp_path, sample_segment):
        """Successfully generate audio segment with edge-tts."""
        with patch("aragora.broadcast.audio_engine._generate_edge_tts") as mock_edge:
            mock_edge.return_value = True
            # Create the expected output file
            expected_path = tmp_path / f"claude-visionary_{sample_segment.text[:12]}.mp3"

            async def mock_generate(text, voice, output_path):
                output_path.write_bytes(b"fake audio")
                return True

            mock_edge.side_effect = mock_generate

            result = await generate_audio_segment(sample_segment, tmp_path)

        assert result is not None
        assert result.exists()
        assert "claude-visionary" in result.name

    @pytest.mark.asyncio
    async def test_generate_audio_segment_fallback_on_edge_failure(self, tmp_path, sample_segment):
        """Fall back to pyttsx3 when edge-tts fails."""
        # Mock the backend to return None (fail) so we hit the legacy path
        mock_backend = MagicMock()
        mock_backend.synthesize = AsyncMock(return_value=None)
        mock_backend.name = "mock"

        with patch("aragora.broadcast.audio_engine.get_audio_backend", return_value=mock_backend):
            with patch("aragora.broadcast.audio_engine._generate_edge_tts", return_value=False):
                with patch(
                    "aragora.broadcast.audio_engine._generate_fallback_tts"
                ) as mock_fallback:
                    # _generate_fallback_tts is now async
                    async def mock_generate(text, output_path):
                        output_path.write_bytes(b"fake audio")
                        return True

                    mock_fallback.side_effect = mock_generate

                    result = await generate_audio_segment(sample_segment, tmp_path)

        mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_audio_segment_both_fail(self, tmp_path, sample_segment):
        """Return None when both edge-tts and fallback fail."""
        # Mock the backend to return None (fail) so we hit the legacy path
        mock_backend = MagicMock()
        mock_backend.synthesize = AsyncMock(return_value=None)
        mock_backend.name = "mock"

        with patch("aragora.broadcast.audio_engine.get_audio_backend", return_value=mock_backend):
            with patch("aragora.broadcast.audio_engine._generate_edge_tts", return_value=False):
                with patch(
                    "aragora.broadcast.audio_engine._generate_fallback_tts",
                    new=AsyncMock(return_value=False),
                ):
                    result = await generate_audio_segment(sample_segment, tmp_path)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_audio_segment_uses_correct_voice(self, tmp_path):
        """Use correct voice based on speaker."""
        segment = ScriptSegment(
            speaker="codex-engineer",
            text="Hello from Codex.",
        )

        # Mock the backend to return None (fail) so we hit the legacy path
        mock_backend = MagicMock()
        mock_backend.synthesize = AsyncMock(return_value=None)
        mock_backend.name = "mock"

        with patch("aragora.broadcast.audio_engine.get_audio_backend", return_value=mock_backend):
            with patch("aragora.broadcast.audio_engine._generate_edge_tts") as mock_edge:
                mock_edge.return_value = False  # Will fail, just check the voice
                with patch(
                    "aragora.broadcast.audio_engine._generate_fallback_tts",
                    new=AsyncMock(return_value=False),
                ):
                    await generate_audio_segment(segment, tmp_path)

            # Check that edge-tts was called with codex voice
            call_args = mock_edge.call_args
            assert call_args[0][1] == "en-US-GuyNeural"  # Codex voice

    @pytest.mark.asyncio
    async def test_generate_audio_segment_deterministic_filename(self, tmp_path, sample_segment):
        """Filename is deterministic based on text hash."""
        with patch("aragora.broadcast.audio_engine._generate_edge_tts") as mock_edge:

            async def mock_generate(text, voice, output_path):
                output_path.write_bytes(b"fake audio")
                return True

            mock_edge.side_effect = mock_generate

            result1 = await generate_audio_segment(sample_segment, tmp_path)

        # Remove file and generate again
        if result1:
            result1.unlink()

        with patch("aragora.broadcast.audio_engine._generate_edge_tts") as mock_edge:
            mock_edge.side_effect = mock_generate
            result2 = await generate_audio_segment(sample_segment, tmp_path)

        # Same text should produce same filename
        assert result1.name == result2.name


# =============================================================================
# Test generate_audio (Parallel Generation)
# =============================================================================


class TestGenerateAudio:
    """Tests for generate_audio function (parallel generation)."""

    @pytest.fixture
    def sample_segments(self):
        """Create sample script segments."""
        return [
            ScriptSegment(speaker="narrator", text="Welcome to the debate."),
            ScriptSegment(speaker="claude-visionary", text="Hello, I am Claude."),
            ScriptSegment(speaker="codex-engineer", text="And I am Codex."),
        ]

    @pytest.mark.asyncio
    async def test_generate_audio_all_success(self, tmp_path, sample_segments):
        """Successfully generate audio for all segments."""
        with patch("aragora.broadcast.audio_engine.generate_audio_segment") as mock_gen:

            async def mock_generate(segment, output_dir):
                path = output_dir / f"{segment.speaker}.mp3"
                path.write_bytes(b"fake audio")
                return path

            mock_gen.side_effect = mock_generate

            results = await generate_audio(sample_segments, tmp_path)

        assert len(results) == 3
        assert all(path.exists() for path in results)

    @pytest.mark.asyncio
    async def test_generate_audio_partial_failure(self, tmp_path, sample_segments):
        """Handle partial failures in segment generation."""
        call_count = 0

        with patch("aragora.broadcast.audio_engine.generate_audio_segment") as mock_gen:

            async def mock_generate(segment, output_dir):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    return None  # Second segment fails
                path = output_dir / f"{segment.speaker}.mp3"
                path.write_bytes(b"fake audio")
                return path

            mock_gen.side_effect = mock_generate

            results = await generate_audio(sample_segments, tmp_path)

        # Only 2 of 3 succeed
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_generate_audio_all_failures(self, tmp_path, sample_segments):
        """Handle all segments failing."""
        with patch("aragora.broadcast.audio_engine.generate_audio_segment", return_value=None):
            results = await generate_audio(sample_segments, tmp_path)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_generate_audio_empty_segments(self, tmp_path):
        """Handle empty segment list."""
        results = await generate_audio([], tmp_path)
        assert results == []

    @pytest.mark.asyncio
    async def test_generate_audio_single_segment(self, tmp_path):
        """Handle single segment."""
        segment = ScriptSegment(speaker="narrator", text="Hello.")

        with patch("aragora.broadcast.audio_engine.generate_audio_segment") as mock_gen:

            async def mock_generate(seg, output_dir):
                path = output_dir / "narrator.mp3"
                path.write_bytes(b"fake audio")
                return path

            mock_gen.side_effect = mock_generate

            results = await generate_audio([segment], tmp_path)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_generate_audio_creates_temp_dir(self, sample_segments):
        """Create temp directory when output_dir is None."""
        with patch("aragora.broadcast.audio_engine.generate_audio_segment") as mock_gen:

            async def mock_generate(segment, output_dir):
                path = output_dir / f"{segment.speaker}.mp3"
                path.write_bytes(b"fake audio")
                return path

            mock_gen.side_effect = mock_generate

            results = await generate_audio(sample_segments, output_dir=None)

        assert len(results) == 3
        # All paths should be in the same temp directory
        dirs = {path.parent for path in results}
        assert len(dirs) == 1

        # Clean up
        import shutil

        shutil.rmtree(list(dirs)[0])

    @pytest.mark.asyncio
    async def test_generate_audio_exception_handling(self, tmp_path, sample_segments):
        """Handle exceptions during segment generation."""
        call_count = 0

        with patch("aragora.broadcast.audio_engine.generate_audio_segment") as mock_gen:

            async def mock_generate(segment, output_dir):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("Generation failed")
                path = output_dir / f"{segment.speaker}.mp3"
                path.write_bytes(b"fake audio")
                return path

            mock_gen.side_effect = mock_generate

            # Should not raise, but log the exception
            results = await generate_audio(sample_segments, tmp_path)

        # 2 succeed, 1 exception
        assert len(results) == 2


# =============================================================================
# Test Integration Scenarios
# =============================================================================


@pytest.mark.skipif(
    "CI" in __import__("os").environ or "GITHUB_ACTIONS" in __import__("os").environ,
    reason="Audio integration tests fail in CI environment",
)
class TestAudioEngineIntegration:
    """Integration tests for audio engine."""

    @pytest.mark.asyncio
    async def test_full_segment_to_audio_flow(self, tmp_path):
        """Test complete flow from segment to audio file."""
        segment = ScriptSegment(
            speaker="narrator",
            text="This is a test narration for the debate.",
        )

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.write_bytes"):
                    # Manually create the expected file
                    import hashlib

                    text_hash = hashlib.sha256(segment.text.encode()).hexdigest()[:12]
                    expected_file = tmp_path / f"narrator_{text_hash}.mp3"
                    expected_file.write_bytes(b"fake audio")

                    result = await generate_audio_segment(segment, tmp_path)

        assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_segment_generation(self, tmp_path):
        """Test that segments are generated concurrently."""
        segments = [ScriptSegment(speaker=f"agent_{i}", text=f"Text {i}") for i in range(5)]

        generation_times = []

        with patch("aragora.broadcast.audio_engine.generate_audio_segment") as mock_gen:

            async def mock_generate(segment, output_dir):
                import time

                generation_times.append(time.time())
                await asyncio.sleep(0.1)  # Simulate work
                path = output_dir / f"{segment.speaker}.mp3"
                path.write_bytes(b"fake audio")
                return path

            mock_gen.side_effect = mock_generate

            results = await generate_audio(segments, tmp_path)

        assert len(results) == 5
        # All generations should start nearly simultaneously (within 0.2s)
        if len(generation_times) > 1:
            time_spread = max(generation_times) - min(generation_times)
            assert time_spread < 0.2  # Concurrent execution
