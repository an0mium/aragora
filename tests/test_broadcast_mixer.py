"""Tests for broadcast mixer module and pipeline integration."""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.broadcast.mixer import (
    mix_audio,
    mix_audio_with_ffmpeg,
    PYDUB_AVAILABLE,
)
from aragora.broadcast import (
    broadcast_debate,
    broadcast_debate_sync,
)
from aragora.broadcast.script_gen import ScriptSegment


# =============================================================================
# Test Pydub Mixer
# =============================================================================


class TestPydubMixer:
    """Tests for pydub-based audio mixing."""

    def test_mix_audio_pydub_unavailable(self, tmp_path):
        """Return False when pydub is not available."""
        output_path = tmp_path / "output.mp3"
        audio_files = [tmp_path / "test.mp3"]

        with patch("aragora.broadcast.mixer.PYDUB_AVAILABLE", False):
            result = mix_audio(audio_files, output_path)

        assert result is False

    def test_mix_audio_empty_list(self, tmp_path):
        """Return False when no audio files provided."""
        output_path = tmp_path / "output.mp3"

        with patch("aragora.broadcast.mixer.PYDUB_AVAILABLE", True):
            result = mix_audio([], output_path)

        assert result is False

    def test_mix_audio_success(self, tmp_path):
        """Successfully mix multiple audio files."""
        # Create fake audio files
        audio_files = []
        for i in range(3):
            audio_file = tmp_path / f"audio_{i}.mp3"
            audio_file.write_bytes(b"fake audio")
            audio_files.append(audio_file)

        output_path = tmp_path / "output.mp3"

        mock_segment = MagicMock()
        mock_combined = MagicMock()
        mock_audio = MagicMock()
        mock_audio.empty.return_value = mock_combined
        mock_audio.from_file.return_value = mock_segment
        mock_combined.__iadd__ = MagicMock(return_value=mock_combined)

        with patch("aragora.broadcast.mixer.PYDUB_AVAILABLE", True):
            # Use create=True since AudioSegment is conditionally imported
            with patch("aragora.broadcast.mixer.AudioSegment", mock_audio, create=True):
                result = mix_audio(audio_files, output_path)

        assert result is True
        assert mock_audio.from_file.call_count == 3
        mock_combined.export.assert_called_once()

    def test_mix_audio_missing_file_skipped(self, tmp_path):
        """Skip missing files during mixing."""
        # Create one real file and one non-existent
        existing_file = tmp_path / "existing.mp3"
        existing_file.write_bytes(b"fake audio")
        missing_file = tmp_path / "missing.mp3"

        audio_files = [existing_file, missing_file]
        output_path = tmp_path / "output.mp3"

        mock_segment = MagicMock()
        mock_combined = MagicMock()
        mock_audio = MagicMock()
        mock_audio.empty.return_value = mock_combined
        mock_audio.from_file.return_value = mock_segment
        mock_combined.__iadd__ = MagicMock(return_value=mock_combined)

        with patch("aragora.broadcast.mixer.PYDUB_AVAILABLE", True):
            with patch("aragora.broadcast.mixer.AudioSegment", mock_audio, create=True):
                result = mix_audio(audio_files, output_path)

        assert result is True
        # Only one file was loaded
        assert mock_audio.from_file.call_count == 1

    def test_mix_audio_all_files_missing(self, tmp_path):
        """Return False when all audio files are missing."""
        missing_files = [tmp_path / f"missing_{i}.mp3" for i in range(3)]
        output_path = tmp_path / "output.mp3"

        mock_combined = MagicMock()
        mock_audio = MagicMock()
        mock_audio.empty.return_value = mock_combined

        with patch("aragora.broadcast.mixer.PYDUB_AVAILABLE", True):
            with patch("aragora.broadcast.mixer.AudioSegment", mock_audio, create=True):
                result = mix_audio(missing_files, output_path)

        assert result is False

    def test_mix_audio_io_error(self, tmp_path):
        """Handle IO errors gracefully."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")
        output_path = tmp_path / "output.mp3"

        mock_audio = MagicMock()
        mock_audio.empty.side_effect = IOError("Disk full")

        with patch("aragora.broadcast.mixer.PYDUB_AVAILABLE", True):
            with patch("aragora.broadcast.mixer.AudioSegment", mock_audio, create=True):
                result = mix_audio([audio_file], output_path)

        assert result is False

    def test_mix_audio_permission_error(self, tmp_path):
        """Handle permission errors gracefully."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")
        output_path = tmp_path / "output.mp3"

        mock_audio = MagicMock()
        mock_audio.empty.side_effect = PermissionError("Access denied")

        with patch("aragora.broadcast.mixer.PYDUB_AVAILABLE", True):
            with patch("aragora.broadcast.mixer.AudioSegment", mock_audio, create=True):
                result = mix_audio([audio_file], output_path)

        assert result is False

    def test_mix_audio_unexpected_error(self, tmp_path):
        """Handle unexpected errors gracefully."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")
        output_path = tmp_path / "output.mp3"

        mock_audio = MagicMock()
        mock_audio.empty.side_effect = RuntimeError("Unexpected error")

        with patch("aragora.broadcast.mixer.PYDUB_AVAILABLE", True):
            with patch("aragora.broadcast.mixer.AudioSegment", mock_audio, create=True):
                result = mix_audio([audio_file], output_path)

        assert result is False

    def test_mix_audio_with_format(self, tmp_path):
        """Mix audio with specific format."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")
        output_path = tmp_path / "output.wav"

        mock_segment = MagicMock()
        mock_combined = MagicMock()
        mock_audio = MagicMock()
        mock_audio.empty.return_value = mock_combined
        mock_audio.from_file.return_value = mock_segment
        mock_combined.__iadd__ = MagicMock(return_value=mock_combined)

        with patch("aragora.broadcast.mixer.PYDUB_AVAILABLE", True):
            with patch("aragora.broadcast.mixer.AudioSegment", mock_audio, create=True):
                result = mix_audio([audio_file], output_path, format="wav")

        assert result is True
        mock_combined.export.assert_called_with(str(output_path), format="wav")


# =============================================================================
# Test FFmpeg Fallback Mixer
# =============================================================================


class TestFFmpegMixer:
    """Tests for ffmpeg-based fallback audio mixing."""

    def test_mix_audio_with_ffmpeg_empty_list(self, tmp_path):
        """Return False when no audio files provided."""
        output_path = tmp_path / "output.mp3"
        result = mix_audio_with_ffmpeg([], output_path)
        assert result is False

    def test_mix_audio_with_ffmpeg_success(self, tmp_path):
        """Successfully mix audio with ffmpeg."""
        audio_files = []
        for i in range(3):
            audio_file = tmp_path / f"audio_{i}.mp3"
            audio_file.write_bytes(b"fake audio")
            audio_files.append(audio_file)

        output_path = tmp_path / "output.mp3"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = mix_audio_with_ffmpeg(audio_files, output_path)

        assert result is True
        # Called multiple times: ffprobe for codec detection + ffmpeg for mixing
        assert mock_run.call_count >= 1
        # Check ffmpeg command structure (last call should be ffmpeg)
        ffmpeg_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "ffmpeg"]
        assert len(ffmpeg_calls) == 1
        call_args = ffmpeg_calls[0][0][0]
        assert call_args[0] == "ffmpeg"
        assert "concat" in call_args  # -f concat (not -concat)

    def test_mix_audio_with_ffmpeg_failure(self, tmp_path):
        """Handle ffmpeg failure."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")
        output_path = tmp_path / "output.mp3"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ffmpeg error"

        with patch("subprocess.run", return_value=mock_result):
            result = mix_audio_with_ffmpeg([audio_file], output_path)

        assert result is False

    def test_mix_audio_with_ffmpeg_timeout(self, tmp_path):
        """Handle ffmpeg timeout."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")
        output_path = tmp_path / "output.mp3"

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 300)):
            result = mix_audio_with_ffmpeg([audio_file], output_path)

        assert result is False

    def test_mix_audio_with_ffmpeg_not_found(self, tmp_path):
        """Handle ffmpeg not installed."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")
        output_path = tmp_path / "output.mp3"

        with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")):
            result = mix_audio_with_ffmpeg([audio_file], output_path)

        assert result is False

    def test_mix_audio_with_ffmpeg_io_error(self, tmp_path):
        """Handle IO errors in ffmpeg mixing."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")
        output_path = tmp_path / "output.mp3"

        with patch("tempfile.NamedTemporaryFile", side_effect=IOError("Disk full")):
            result = mix_audio_with_ffmpeg([audio_file], output_path)

        assert result is False

    def test_mix_audio_with_ffmpeg_creates_concat_file(self, tmp_path):
        """Verify ffmpeg concat file is created with correct format."""
        audio_files = [tmp_path / "audio1.mp3", tmp_path / "audio2.mp3"]
        for f in audio_files:
            f.write_bytes(b"fake")

        output_path = tmp_path / "output.mp3"
        concat_content = None

        def capture_run(cmd, **kwargs):
            nonlocal concat_content
            # Find the file list argument
            for i, arg in enumerate(cmd):
                if arg == "-i" and i + 1 < len(cmd):
                    file_list_path = cmd[i + 1]
                    if os.path.exists(file_list_path):
                        with open(file_list_path) as f:
                            concat_content = f.read()
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        with patch("subprocess.run", side_effect=capture_run):
            result = mix_audio_with_ffmpeg(audio_files, output_path)

        assert result is True
        assert concat_content is not None
        assert "file '" in concat_content
        assert "audio1.mp3" in concat_content
        assert "audio2.mp3" in concat_content

    def test_mix_audio_with_ffmpeg_escapes_quotes(self, tmp_path):
        """Handle file paths with single quotes."""
        # Create file with quote in name
        audio_file = tmp_path / "audio'test.mp3"
        audio_file.write_bytes(b"fake")

        output_path = tmp_path / "output.mp3"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = mix_audio_with_ffmpeg([audio_file], output_path)

        assert result is True

    def test_mix_audio_with_ffmpeg_cleans_up_temp(self, tmp_path):
        """Verify temp file is cleaned up after mixing."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")
        output_path = tmp_path / "output.mp3"

        temp_files_before = set(Path(tempfile.gettempdir()).glob("tmp*.txt"))

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            mix_audio_with_ffmpeg([audio_file], output_path)

        # No new temp .txt files should remain
        temp_files_after = set(Path(tempfile.gettempdir()).glob("tmp*.txt"))
        new_files = temp_files_after - temp_files_before
        assert len(new_files) == 0


# =============================================================================
# Test Broadcast Integration
# =============================================================================


class TestBroadcastIntegration:
    """Tests for the main broadcast_debate pipeline."""

    @pytest.fixture
    def mock_trace(self):
        """Create a mock debate trace."""
        trace = MagicMock()
        trace.id = "test-debate-123"
        trace.events = [
            MagicMock(type="message", speaker="claude-visionary", content="Hello"),
            MagicMock(type="message", speaker="codex-engineer", content="Hi there"),
        ]
        return trace

    @pytest.mark.asyncio
    async def test_broadcast_debate_success(self, tmp_path, mock_trace):
        """Successfully generate audio from debate trace."""
        output_path = tmp_path / "output.mp3"

        mock_segments = [
            ScriptSegment(speaker="narrator", text="Welcome"),
            ScriptSegment(speaker="claude-visionary", text="Hello"),
        ]

        with patch("aragora.broadcast.generate_script", return_value=mock_segments):
            with patch("aragora.broadcast.generate_audio") as mock_gen_audio:
                mock_gen_audio.return_value = [tmp_path / "seg1.mp3", tmp_path / "seg2.mp3"]
                with patch("aragora.broadcast.mix_audio", return_value=True):
                    result = await broadcast_debate(mock_trace, output_path)

        assert result == output_path

    @pytest.mark.asyncio
    async def test_broadcast_debate_empty_script(self, tmp_path, mock_trace):
        """Return None when script generation produces no segments."""
        output_path = tmp_path / "output.mp3"

        with patch("aragora.broadcast.generate_script", return_value=[]):
            result = await broadcast_debate(mock_trace, output_path)

        assert result is None

    @pytest.mark.asyncio
    async def test_broadcast_debate_no_audio_generated(self, tmp_path, mock_trace):
        """Return None when audio generation fails."""
        output_path = tmp_path / "output.mp3"

        mock_segments = [ScriptSegment(speaker="narrator", text="Hello")]

        with patch("aragora.broadcast.generate_script", return_value=mock_segments):
            with patch("aragora.broadcast.generate_audio", return_value=[]):
                result = await broadcast_debate(mock_trace, output_path)

        assert result is None

    @pytest.mark.asyncio
    async def test_broadcast_debate_mixing_fails(self, tmp_path, mock_trace):
        """Return None when both pydub and ffmpeg mixing fail."""
        output_path = tmp_path / "output.mp3"

        mock_segments = [ScriptSegment(speaker="narrator", text="Hello")]

        with patch("aragora.broadcast.generate_script", return_value=mock_segments):
            with patch("aragora.broadcast.generate_audio") as mock_gen:
                mock_gen.return_value = [tmp_path / "seg.mp3"]
                with patch("aragora.broadcast.mix_audio", return_value=False):
                    with patch("aragora.broadcast.mix_audio_with_ffmpeg", return_value=False):
                        result = await broadcast_debate(mock_trace, output_path)

        assert result is None

    @pytest.mark.asyncio
    async def test_broadcast_debate_ffmpeg_fallback(self, tmp_path, mock_trace):
        """Fall back to ffmpeg when pydub mixing fails."""
        output_path = tmp_path / "output.mp3"

        mock_segments = [ScriptSegment(speaker="narrator", text="Hello")]

        with patch("aragora.broadcast.generate_script", return_value=mock_segments):
            with patch("aragora.broadcast.generate_audio") as mock_gen:
                mock_gen.return_value = [tmp_path / "seg.mp3"]
                with patch("aragora.broadcast.mix_audio", return_value=False):
                    with patch("aragora.broadcast.mix_audio_with_ffmpeg", return_value=True) as mock_ffmpeg:
                        result = await broadcast_debate(mock_trace, output_path)

        assert result == output_path
        mock_ffmpeg.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_debate_auto_output_path(self, mock_trace):
        """Generate output path when not provided."""
        mock_segments = [ScriptSegment(speaker="narrator", text="Hello")]

        with patch("aragora.broadcast.generate_script", return_value=mock_segments):
            with patch("aragora.broadcast.generate_audio") as mock_gen:
                mock_gen.return_value = [Path("/tmp/seg.mp3")]
                with patch("aragora.broadcast.mix_audio", return_value=True):
                    result = await broadcast_debate(mock_trace, output_path=None)

        assert result is not None
        assert "test-debate-123" in str(result)
        assert result.suffix == ".mp3"

    @pytest.mark.asyncio
    async def test_broadcast_debate_cleanup_temp_dir(self, tmp_path, mock_trace):
        """Clean up temporary directory after completion."""
        output_path = tmp_path / "output.mp3"
        mock_segments = [ScriptSegment(speaker="narrator", text="Hello")]

        with patch("aragora.broadcast.generate_script", return_value=mock_segments):
            with patch("aragora.broadcast.generate_audio") as mock_gen:
                temp_dir_used = None

                async def capture_temp_dir(segments, output_dir):
                    nonlocal temp_dir_used
                    temp_dir_used = output_dir
                    return [output_dir / "seg.mp3"]

                mock_gen.side_effect = capture_temp_dir
                with patch("aragora.broadcast.mix_audio", return_value=True):
                    await broadcast_debate(mock_trace, output_path)

        # Temp dir should be cleaned up
        if temp_dir_used:
            assert not temp_dir_used.exists()

    @pytest.mark.asyncio
    async def test_broadcast_debate_cleanup_on_failure(self, tmp_path, mock_trace):
        """Clean up temporary directory even on failure."""
        output_path = tmp_path / "output.mp3"
        mock_segments = [ScriptSegment(speaker="narrator", text="Hello")]

        with patch("aragora.broadcast.generate_script", return_value=mock_segments):
            with patch("aragora.broadcast.generate_audio") as mock_gen:
                temp_dir_used = None

                async def capture_temp_dir(segments, output_dir):
                    nonlocal temp_dir_used
                    temp_dir_used = output_dir
                    return []  # Simulate failure

                mock_gen.side_effect = capture_temp_dir
                await broadcast_debate(mock_trace, output_path)

        # Temp dir should still be cleaned up
        if temp_dir_used:
            assert not temp_dir_used.exists()


class TestBroadcastDebateSync:
    """Tests for the synchronous wrapper."""

    def test_broadcast_debate_sync_calls_async(self):
        """Sync wrapper calls async function via asyncio.run."""
        mock_trace = MagicMock()
        mock_trace.id = "test-123"

        with patch("aragora.broadcast.broadcast_debate") as mock_async:
            mock_async.return_value = Path("/tmp/output.mp3")
            with patch("asyncio.run") as mock_run:
                mock_run.return_value = Path("/tmp/output.mp3")

                result = broadcast_debate_sync(mock_trace)

        mock_run.assert_called_once()
        assert result == Path("/tmp/output.mp3")

    def test_broadcast_debate_sync_passes_args(self):
        """Sync wrapper passes all arguments correctly."""
        mock_trace = MagicMock()
        mock_trace.id = "test-123"
        output_path = Path("/custom/output.wav")

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = output_path

            broadcast_debate_sync(mock_trace, output_path=output_path, format="wav")

        # Check that asyncio.run was called with a coroutine
        mock_run.assert_called_once()


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_mix_audio_single_file(self, tmp_path):
        """Mix a single file (essentially a copy)."""
        audio_file = tmp_path / "single.mp3"
        audio_file.write_bytes(b"fake audio")
        output_path = tmp_path / "output.mp3"

        mock_segment = MagicMock()
        mock_combined = MagicMock()
        mock_audio = MagicMock()
        mock_audio.empty.return_value = mock_combined
        mock_audio.from_file.return_value = mock_segment
        mock_combined.__iadd__ = MagicMock(return_value=mock_combined)

        with patch("aragora.broadcast.mixer.PYDUB_AVAILABLE", True):
            with patch("aragora.broadcast.mixer.AudioSegment", mock_audio, create=True):
                result = mix_audio([audio_file], output_path)

        assert result is True

    def test_mix_audio_large_file_list(self, tmp_path):
        """Handle large number of audio files."""
        audio_files = []
        for i in range(100):
            f = tmp_path / f"audio_{i}.mp3"
            f.write_bytes(b"fake")
            audio_files.append(f)

        output_path = tmp_path / "output.mp3"

        mock_segment = MagicMock()
        mock_combined = MagicMock()
        mock_audio = MagicMock()
        mock_audio.empty.return_value = mock_combined
        mock_audio.from_file.return_value = mock_segment
        mock_combined.__iadd__ = MagicMock(return_value=mock_combined)

        with patch("aragora.broadcast.mixer.PYDUB_AVAILABLE", True):
            with patch("aragora.broadcast.mixer.AudioSegment", mock_audio, create=True):
                result = mix_audio(audio_files, output_path)

        assert result is True
        assert mock_audio.from_file.call_count == 100

    def test_mix_audio_ffmpeg_large_file_list(self, tmp_path):
        """Handle large number of files with ffmpeg."""
        audio_files = []
        for i in range(50):
            f = tmp_path / f"audio_{i}.mp3"
            f.write_bytes(b"fake")
            audio_files.append(f)

        output_path = tmp_path / "output.mp3"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = mix_audio_with_ffmpeg(audio_files, output_path)

        assert result is True

    def test_pydub_available_constant(self):
        """PYDUB_AVAILABLE reflects actual pydub availability."""
        # This is a sanity check - the constant should be boolean
        assert isinstance(PYDUB_AVAILABLE, bool)
