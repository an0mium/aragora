"""Tests for broadcast audio mixer module.

Tests the audio mixing functionality including pydub and FFmpeg-based
concatenation of audio segments.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestMixAudio:
    """Test mix_audio function."""

    def test_mix_audio_pydub_unavailable(self):
        """Test mix_audio fails gracefully when pydub not available."""
        import aragora.broadcast.mixer as mixer

        original_available = mixer.PYDUB_AVAILABLE
        mixer.PYDUB_AVAILABLE = False
        try:
            result = mixer.mix_audio([Path("/fake/file.mp3")], Path("/output.mp3"))
            assert result is False
        finally:
            mixer.PYDUB_AVAILABLE = original_available

    def test_mix_audio_empty_list(self):
        """Test mix_audio with empty file list."""
        from aragora.broadcast.mixer import mix_audio

        result = mix_audio([], Path("/output.mp3"))
        assert result is False

    def test_mix_audio_missing_files(self):
        """Test mix_audio handles missing files gracefully."""
        import aragora.broadcast.mixer as mixer

        # If pydub not available, this test is not meaningful
        if not mixer.PYDUB_AVAILABLE:
            pytest.skip("pydub not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp3"

            # All files missing - should fail
            result = mixer.mix_audio(
                [Path("/nonexistent1.mp3"), Path("/nonexistent2.mp3")],
                output_path,
            )

            assert result is False


class TestMixAudioWithFFmpeg:
    """Test mix_audio_with_ffmpeg function."""

    def test_mix_audio_ffmpeg_empty_list(self):
        """Test mix_audio_with_ffmpeg with empty file list."""
        from aragora.broadcast.mixer import mix_audio_with_ffmpeg

        result = mix_audio_with_ffmpeg([], Path("/output.mp3"))
        assert result is False

    def test_mix_audio_ffmpeg_too_many_files(self):
        """Test mix_audio_with_ffmpeg rejects too many files."""
        from aragora.broadcast.mixer import MAX_AUDIO_FILES, mix_audio_with_ffmpeg

        # Create list exceeding limit
        files = [Path(f"/fake/file_{i}.mp3") for i in range(MAX_AUDIO_FILES + 1)]

        result = mix_audio_with_ffmpeg(files, Path("/output.mp3"))
        assert result is False

    @patch("subprocess.run")
    @patch("aragora.broadcast.mixer._has_mixed_codecs")
    def test_mix_audio_ffmpeg_same_codec(self, mock_mixed, mock_run):
        """Test FFmpeg mixing with same codec (concat demuxer)."""
        from aragora.broadcast.mixer import mix_audio_with_ffmpeg

        mock_mixed.return_value = False
        mock_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "audio1.mp3"
            file2 = Path(tmpdir) / "audio2.mp3"
            file1.write_text("fake")
            file2.write_text("fake")
            output_path = Path(tmpdir) / "output.mp3"

            result = mix_audio_with_ffmpeg([file1, file2], output_path)

            assert result is True
            # Should use concat demuxer with -c copy
            cmd = mock_run.call_args[0][0]
            assert "-f" in cmd
            assert "concat" in cmd

    @patch("subprocess.run")
    @patch("aragora.broadcast.mixer._has_mixed_codecs")
    def test_mix_audio_ffmpeg_mixed_codecs(self, mock_mixed, mock_run):
        """Test FFmpeg mixing with mixed codecs (filter_complex)."""
        from aragora.broadcast.mixer import mix_audio_with_ffmpeg

        mock_mixed.return_value = True
        mock_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "audio1.mp3"
            file2 = Path(tmpdir) / "audio2.wav"
            file1.write_text("fake")
            file2.write_text("fake")
            output_path = Path(tmpdir) / "output.mp3"

            result = mix_audio_with_ffmpeg([file1, file2], output_path)

            assert result is True
            # Should use filter_complex
            cmd = mock_run.call_args[0][0]
            assert "-filter_complex" in cmd

    @patch("subprocess.run")
    @patch("aragora.broadcast.mixer._has_mixed_codecs")
    def test_mix_audio_ffmpeg_failure(self, mock_mixed, mock_run):
        """Test FFmpeg mixing failure."""
        from aragora.broadcast.mixer import mix_audio_with_ffmpeg

        mock_mixed.return_value = False
        mock_run.return_value = MagicMock(returncode=1, stderr="Error message")

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "audio1.mp3"
            file1.write_text("fake")
            output_path = Path(tmpdir) / "output.mp3"

            result = mix_audio_with_ffmpeg([file1], output_path)

            assert result is False

    @patch("subprocess.run")
    @patch("aragora.broadcast.mixer._has_mixed_codecs")
    def test_mix_audio_ffmpeg_not_found(self, mock_mixed, mock_run):
        """Test FFmpeg not found."""
        from aragora.broadcast.mixer import mix_audio_with_ffmpeg

        mock_mixed.return_value = False
        mock_run.side_effect = FileNotFoundError("ffmpeg not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "audio1.mp3"
            file1.write_text("fake")
            output_path = Path(tmpdir) / "output.mp3"

            result = mix_audio_with_ffmpeg([file1], output_path)

            assert result is False

    @patch("subprocess.run")
    @patch("aragora.broadcast.mixer._has_mixed_codecs")
    def test_mix_audio_ffmpeg_timeout(self, mock_mixed, mock_run):
        """Test FFmpeg timeout."""
        import subprocess

        from aragora.broadcast.mixer import mix_audio_with_ffmpeg

        mock_mixed.return_value = False
        mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 300)

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "audio1.mp3"
            file1.write_text("fake")
            output_path = Path(tmpdir) / "output.mp3"

            result = mix_audio_with_ffmpeg([file1], output_path)

            assert result is False


class TestDetectAudioCodec:
    """Test _detect_audio_codec function."""

    @patch("subprocess.run")
    def test_detect_codec_success(self, mock_run):
        """Test successful codec detection."""
        from aragora.broadcast.mixer import _detect_audio_codec

        mock_run.return_value = MagicMock(returncode=0, stdout="mp3\n")

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            codec = _detect_audio_codec(Path(f.name))

        assert codec == "mp3"

    @patch("subprocess.run")
    def test_detect_codec_failure(self, mock_run):
        """Test codec detection failure."""
        from aragora.broadcast.mixer import _detect_audio_codec

        mock_run.return_value = MagicMock(returncode=1, stdout="")

        codec = _detect_audio_codec(Path("/fake/file.mp3"))

        assert codec is None

    @patch("subprocess.run")
    def test_detect_codec_ffprobe_not_found(self, mock_run):
        """Test codec detection when ffprobe not found."""
        from aragora.broadcast.mixer import _detect_audio_codec

        mock_run.side_effect = FileNotFoundError("ffprobe not found")

        codec = _detect_audio_codec(Path("/fake/file.mp3"))

        assert codec is None


class TestHasMixedCodecs:
    """Test _has_mixed_codecs function."""

    @patch("aragora.broadcast.mixer._detect_audio_codec")
    def test_has_mixed_codecs_true(self, mock_detect):
        """Test mixed codecs detection."""
        from aragora.broadcast.mixer import _has_mixed_codecs

        mock_detect.side_effect = ["mp3", "pcm_s16le"]

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "audio1.mp3"
            file2 = Path(tmpdir) / "audio2.wav"
            file1.write_text("fake")
            file2.write_text("fake")

            result = _has_mixed_codecs([file1, file2])

        assert result is True

    @patch("aragora.broadcast.mixer._detect_audio_codec")
    def test_has_mixed_codecs_false(self, mock_detect):
        """Test same codec detection."""
        from aragora.broadcast.mixer import _has_mixed_codecs

        mock_detect.side_effect = ["mp3", "mp3"]

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "audio1.mp3"
            file2 = Path(tmpdir) / "audio2.mp3"
            file1.write_text("fake")
            file2.write_text("fake")

            result = _has_mixed_codecs([file1, file2])

        assert result is False

    @patch("aragora.broadcast.mixer._detect_audio_codec")
    def test_has_mixed_codecs_single_file(self, mock_detect):
        """Test single file returns False."""
        from aragora.broadcast.mixer import _has_mixed_codecs

        mock_detect.return_value = "mp3"

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "audio1.mp3"
            file1.write_text("fake")

            result = _has_mixed_codecs([file1])

        assert result is False
