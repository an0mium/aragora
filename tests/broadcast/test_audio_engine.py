"""Tests for broadcast audio engine module.

Tests the TTS audio generation functionality including backend selection
and fallback behavior.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.broadcast.script_gen import ScriptSegment


class TestGetAudioBackend:
    """Test get_audio_backend function."""

    def test_get_audio_backend_caches(self):
        """Test backend is cached on first call."""
        import aragora.broadcast.audio_engine as audio_engine

        # Reset global
        audio_engine._tts_backend = None

        with patch("aragora.broadcast.audio_engine.get_fallback_backend") as mock_get:
            mock_backend = MagicMock()
            mock_get.return_value = mock_backend

            backend1 = audio_engine.get_audio_backend()
            backend2 = audio_engine.get_audio_backend()

            assert backend1 is backend2
            mock_get.assert_called_once()

        # Clean up
        audio_engine._tts_backend = None

    def test_get_audio_backend_forced(self):
        """Test forced backend via environment variable."""
        import aragora.broadcast.audio_engine as audio_engine

        audio_engine._tts_backend = None

        with patch.dict("os.environ", {"ARAGORA_TTS_BACKEND": "elevenlabs"}):
            with patch("aragora.broadcast.audio_engine.get_tts_backend") as mock_get:
                mock_backend = MagicMock()
                mock_get.return_value = mock_backend

                backend = audio_engine.get_audio_backend()

                mock_get.assert_called_once_with("elevenlabs")
                assert backend is mock_backend

        audio_engine._tts_backend = None


class TestGetVoiceForSpeaker:
    """Test _get_voice_for_speaker function."""

    def test_get_voice_known_speaker(self):
        """Test getting voice for known speaker."""
        from aragora.broadcast.audio_engine import _get_voice_for_speaker

        # The voice map should have narrator
        voice = _get_voice_for_speaker("narrator")
        assert voice is not None
        assert isinstance(voice, str)

    def test_get_voice_unknown_speaker(self):
        """Test getting voice for unknown speaker falls back to default."""
        from aragora.broadcast.audio_engine import _get_voice_for_speaker

        voice = _get_voice_for_speaker("unknown_speaker_xyz")
        assert voice is not None
        assert isinstance(voice, str)


class TestEdgeTTSCommand:
    """Test _edge_tts_command function."""

    @patch("shutil.which")
    def test_edge_tts_command_in_path(self, mock_which):
        """Test edge-tts found in PATH."""
        from aragora.broadcast.audio_engine import _edge_tts_command

        mock_which.return_value = "/usr/local/bin/edge-tts"

        cmd = _edge_tts_command()

        assert cmd == ["/usr/local/bin/edge-tts"]

    @patch("shutil.which")
    @patch("importlib.util.find_spec")
    def test_edge_tts_as_module(self, mock_spec, mock_which):
        """Test edge-tts available as Python module."""
        from aragora.broadcast.audio_engine import _edge_tts_command

        mock_which.return_value = None
        mock_spec.return_value = MagicMock()  # Module exists

        cmd = _edge_tts_command()

        assert cmd is not None
        assert "-m" in cmd
        assert "edge_tts" in cmd

    @patch("shutil.which")
    @patch("importlib.util.find_spec")
    def test_edge_tts_not_available(self, mock_spec, mock_which):
        """Test edge-tts not available."""
        from aragora.broadcast.audio_engine import _edge_tts_command

        mock_which.return_value = None
        mock_spec.return_value = None

        cmd = _edge_tts_command()

        assert cmd is None


class TestGenerateEdgeTTS:
    """Test _generate_edge_tts function."""

    @pytest.mark.asyncio
    @patch("aragora.broadcast.audio_engine._edge_tts_command")
    async def test_generate_edge_tts_not_available(self, mock_cmd):
        """Test edge-tts returns False when not available."""
        from aragora.broadcast.audio_engine import _generate_edge_tts

        mock_cmd.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp3"
            result = await _generate_edge_tts("Hello", "en-US-AriaNeural", output_path)

        assert result is False

    @pytest.mark.asyncio
    @patch("aragora.broadcast.audio_engine._edge_tts_command")
    @patch("asyncio.create_subprocess_exec")
    async def test_generate_edge_tts_success(self, mock_exec, mock_cmd):
        """Test successful edge-tts generation."""
        from aragora.broadcast.audio_engine import _generate_edge_tts

        mock_cmd.return_value = ["edge-tts"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp3"
            # Create the output file to simulate success
            output_path.write_text("fake audio")

            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_process

            result = await _generate_edge_tts("Hello", "en-US-AriaNeural", output_path)

        assert result is True

    @pytest.mark.asyncio
    @patch("aragora.broadcast.audio_engine._edge_tts_command")
    @patch("asyncio.create_subprocess_exec")
    async def test_generate_edge_tts_failure(self, mock_exec, mock_cmd):
        """Test edge-tts generation failure."""
        from aragora.broadcast.audio_engine import _generate_edge_tts

        mock_cmd.return_value = ["edge-tts"]

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"error"))
        mock_exec.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp3"
            result = await _generate_edge_tts(
                "Hello",
                "en-US-AriaNeural",
                output_path,
                max_retries=1,
                base_delay=0.001,
            )

        assert result is False


class TestGenerateFallbackTTS:
    """Test pyttsx3 fallback TTS."""

    @patch("aragora.broadcast.audio_engine.FALLBACK_AVAILABLE", False)
    @pytest.mark.asyncio
    async def test_fallback_not_available(self):
        """Test fallback returns False when pyttsx3 not available."""
        from aragora.broadcast.audio_engine import _generate_fallback_tts

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp3"
            result = await _generate_fallback_tts("Hello", output_path)

        assert result is False


class TestGenerateAudioSegment:
    """Test generate_audio_segment function."""

    @pytest.mark.asyncio
    @patch("aragora.broadcast.audio_engine.get_audio_backend")
    async def test_generate_audio_segment_success(self, mock_get_backend):
        """Test successful audio segment generation."""
        from aragora.broadcast.audio_engine import generate_audio_segment

        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend
        mock_backend.name = "test"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            expected_output = output_dir / "test_audio.mp3"

            mock_backend.synthesize = AsyncMock(return_value=expected_output)

            segment = ScriptSegment(
                speaker="narrator",
                text="Hello, world!",
                voice_id="en-US-AriaNeural",
            )

            result = await generate_audio_segment(segment, output_dir)

            assert result == expected_output
            mock_backend.synthesize.assert_called_once()

    @pytest.mark.asyncio
    @patch("aragora.broadcast.audio_engine.get_audio_backend")
    @patch("aragora.broadcast.audio_engine._generate_edge_tts")
    async def test_generate_audio_segment_fallback(self, mock_edge, mock_get_backend):
        """Test fallback to edge-tts when backend fails."""
        from aragora.broadcast.audio_engine import generate_audio_segment

        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend
        mock_backend.name = "test"
        mock_backend.synthesize = AsyncMock(return_value=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            async def edge_success(*args, **kwargs):
                # Create the file
                args[2].write_text("fake")
                return True

            mock_edge.side_effect = edge_success

            segment = ScriptSegment(
                speaker="narrator",
                text="Hello, world!",
            )

            result = await generate_audio_segment(segment, output_dir)

            assert result is not None
            mock_edge.assert_called_once()


class TestGenerateAudio:
    """Test generate_audio function."""

    @pytest.mark.asyncio
    @patch("aragora.broadcast.audio_engine.generate_audio_segment")
    async def test_generate_audio_success(self, mock_gen_segment):
        """Test successful multi-segment audio generation."""
        from aragora.broadcast.audio_engine import generate_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            async def mock_generate(segment, out_dir):
                path = out_dir / f"{segment.speaker}_{hash(segment.text)}.mp3"
                path.write_text("fake")
                return path

            mock_gen_segment.side_effect = mock_generate

            segments = [
                ScriptSegment(speaker="narrator", text="Hello"),
                ScriptSegment(speaker="host", text="Welcome"),
            ]

            result = await generate_audio(segments, output_dir)

            assert len(result) == 2
            assert all(p.exists() for p in result)

    @pytest.mark.asyncio
    @patch("aragora.broadcast.audio_engine.generate_audio_segment")
    async def test_generate_audio_partial_failure(self, mock_gen_segment):
        """Test audio generation with some segments failing."""
        from aragora.broadcast.audio_engine import generate_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            call_count = 0

            async def mock_generate(segment, out_dir):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    path = out_dir / "success.mp3"
                    path.write_text("fake")
                    return path
                return None  # Second segment fails

            mock_gen_segment.side_effect = mock_generate

            segments = [
                ScriptSegment(speaker="narrator", text="Hello"),
                ScriptSegment(speaker="host", text="Welcome"),
            ]

            result = await generate_audio(segments, output_dir)

            # Should only have 1 successful result
            assert len(result) == 1

    @pytest.mark.asyncio
    @patch("aragora.broadcast.audio_engine.generate_audio_segment")
    async def test_generate_audio_creates_temp_dir(self, mock_gen_segment):
        """Test generate_audio creates temp directory if none provided."""
        import shutil

        from aragora.broadcast.audio_engine import generate_audio

        async def mock_generate(segment, out_dir):
            path = out_dir / "test.mp3"
            path.write_text("fake")
            return path

        mock_gen_segment.side_effect = mock_generate

        segments = [ScriptSegment(speaker="narrator", text="Hello")]

        result = await generate_audio(segments, output_dir=None)

        assert len(result) == 1
        assert result[0].exists()
        # Clean up - use shutil.rmtree to handle non-empty directory
        shutil.rmtree(result[0].parent)

    @pytest.mark.asyncio
    @patch("aragora.broadcast.audio_engine.generate_audio_segment")
    async def test_generate_audio_handles_exceptions(self, mock_gen_segment):
        """Test generate_audio handles exceptions from segment generation."""
        from aragora.broadcast.audio_engine import generate_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            mock_gen_segment.side_effect = Exception("TTS error")

            segments = [ScriptSegment(speaker="narrator", text="Hello")]

            result = await generate_audio(segments, output_dir)

            # Should return empty list, not raise
            assert result == []
