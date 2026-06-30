"""
Tests for video generator module.

Tests VideoGenerator, thumbnail generation, audio duration extraction,
and FFmpeg/ImageMagick subprocess handling.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.broadcast.video_gen import (
    VideoGenerator,
    VideoMetadata,
    get_audio_duration,
    generate_thumbnail,
    _check_ffmpeg,
    _check_ffprobe,
    FFPROBE_TIMEOUT,
    FFMPEG_TIMEOUT,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def mock_audio_file(temp_dir):
    """Create a mock audio file."""
    audio_path = temp_dir / "test_audio.mp3"
    audio_path.write_bytes(b"fake audio content")
    return audio_path


@pytest.fixture
def video_generator(temp_dir):
    """Create a VideoGenerator instance."""
    return VideoGenerator(output_dir=temp_dir)


# =============================================================================
# VideoMetadata Tests
# =============================================================================


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""

    def test_metadata_creation(self):
        """Should create metadata with required fields."""
        meta = VideoMetadata(
            title="Test Debate",
            description="A test debate",
            duration_seconds=120,
            file_size_bytes=1024000,
        )

        assert meta.title == "Test Debate"
        assert meta.duration_seconds == 120
        assert meta.format == "mp4"  # Default
        assert meta.resolution == "1920x1080"  # Default

    def test_metadata_custom_format(self):
        """Should accept custom format and resolution."""
        meta = VideoMetadata(
            title="Test",
            description="Test",
            duration_seconds=60,
            file_size_bytes=512000,
            format="webm",
            resolution="1280x720",
        )

        assert meta.format == "webm"
        assert meta.resolution == "1280x720"


# =============================================================================
# FFmpeg/FFprobe Check Tests
# =============================================================================


class TestFFmpegChecks:
    """Tests for FFmpeg availability checks."""

    def test_check_ffmpeg_available(self):
        """Should return True when ffmpeg is in PATH."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/ffmpeg"
            assert _check_ffmpeg() is True
            mock_which.assert_called_with("ffmpeg")

    def test_check_ffmpeg_unavailable(self):
        """Should return False when ffmpeg is not in PATH."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            assert _check_ffmpeg() is False

    def test_check_ffprobe_available(self):
        """Should return True when ffprobe is in PATH."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/ffprobe"
            assert _check_ffprobe() is True
            mock_which.assert_called_with("ffprobe")


# =============================================================================
# Audio Duration Tests
# =============================================================================


class TestGetAudioDuration:
    """Tests for audio duration extraction."""

    @pytest.mark.asyncio
    async def test_get_duration_success(self, mock_audio_file):
        """Should return duration when ffprobe succeeds."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"123.45", b""))

        with patch("shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                duration = await get_audio_duration(mock_audio_file)

        assert duration == 123

    @pytest.mark.asyncio
    async def test_get_duration_ffprobe_unavailable(self, mock_audio_file):
        """Should return None when ffprobe is not available."""
        with patch("shutil.which", return_value=None):
            duration = await get_audio_duration(mock_audio_file)

        assert duration is None

    @pytest.mark.asyncio
    async def test_get_duration_timeout(self, mock_audio_file):
        """Should return None and kill process on timeout."""
        mock_process = MagicMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                duration = await get_audio_duration(mock_audio_file)

        assert duration is None
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_duration_ffprobe_error(self, mock_audio_file):
        """Should return None when ffprobe fails."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                duration = await get_audio_duration(mock_audio_file)

        assert duration is None


# =============================================================================
# Thumbnail Generation Tests
# =============================================================================


class TestGenerateThumbnail:
    """Tests for thumbnail generation."""

    @pytest.mark.asyncio
    async def test_generate_thumbnail_pure_python(self, temp_dir):
        """Should create thumbnail using pure Python when ImageMagick unavailable."""
        output_path = temp_dir / "thumb.png"

        with patch("shutil.which", return_value=None):
            # Use small dimensions to avoid memory issues in pure Python PNG creation
            result = await generate_thumbnail(
                "Test Title",
                ["claude", "gpt4"],
                output_path,
                width=64,  # Small for testing
                height=36,
            )

        assert result is True
        assert output_path.exists()
        # Verify it's a PNG (magic bytes)
        content = output_path.read_bytes()
        assert content[:8] == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.asyncio
    async def test_generate_thumbnail_imagemagick(self, temp_dir):
        """Should use ImageMagick when available."""
        output_path = temp_dir / "thumb.png"

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        # Create output file to simulate ImageMagick success
        def create_file(*args, **kwargs):
            output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return mock_process

        with patch("shutil.which", return_value="/usr/bin/convert"):
            with patch("asyncio.create_subprocess_exec", side_effect=create_file):
                result = await generate_thumbnail(
                    "Test Title",
                    ["claude", "gpt4", "gemini", "extra"],
                    output_path,
                )

        assert result is True

    @pytest.mark.asyncio
    async def test_generate_thumbnail_imagemagick_timeout(self, temp_dir):
        """Should return False on ImageMagick timeout."""
        output_path = temp_dir / "thumb.png"

        mock_process = MagicMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("shutil.which", return_value="/usr/bin/convert"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await generate_thumbnail("Test", [], output_path)

        assert result is False


# =============================================================================
# VideoGenerator Tests
# =============================================================================


class TestVideoGenerator:
    """Tests for VideoGenerator class."""

    def test_init_creates_output_dir(self, temp_dir):
        """Should create output directory on init."""
        output_dir = temp_dir / "videos"
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            generator = VideoGenerator(output_dir=output_dir)

        assert output_dir.exists()
        assert generator.ffmpeg_available is True

    def test_init_ffmpeg_unavailable(self, temp_dir):
        """Should flag when ffmpeg unavailable."""
        with patch("shutil.which", return_value=None):
            generator = VideoGenerator(output_dir=temp_dir)

        assert generator.ffmpeg_available is False

    @pytest.mark.asyncio
    async def test_generate_static_video_no_ffmpeg(self, temp_dir, mock_audio_file):
        """Should return None when ffmpeg unavailable."""
        with patch("shutil.which", return_value=None):
            generator = VideoGenerator(output_dir=temp_dir)
            result = await generator.generate_static_video(
                mock_audio_file,
                "Test",
                ["claude"],
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_static_video_missing_audio(self, temp_dir):
        """Should return None when audio file doesn't exist."""
        fake_path = temp_dir / "nonexistent.mp3"

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            generator = VideoGenerator(output_dir=temp_dir)
            result = await generator.generate_static_video(
                fake_path,
                "Test",
                ["claude"],
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_static_video_success(self, temp_dir, mock_audio_file):
        """Should return video path on success."""
        output_path = temp_dir / "output.mp4"

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        def create_output(*args, **kwargs):
            # Create output file to simulate ffmpeg success
            if "ffmpeg" in str(args):
                output_path.write_bytes(b"fake video content")
            return mock_process

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("asyncio.create_subprocess_exec", side_effect=create_output):
                with patch.object(
                    VideoGenerator, "generate_static_video", return_value=output_path
                ):
                    generator = VideoGenerator(output_dir=temp_dir)
                    result = await generator.generate_static_video(
                        mock_audio_file,
                        "Test Debate",
                        ["claude", "gpt4"],
                        output_path=output_path,
                    )

        assert result == output_path

    @pytest.mark.asyncio
    async def test_generate_video_ffmpeg_timeout(self, temp_dir, mock_audio_file):
        """Should handle ffmpeg timeout gracefully."""
        mock_thumb_process = MagicMock()
        mock_thumb_process.returncode = 0
        mock_thumb_process.communicate = AsyncMock(return_value=(b"", b""))

        call_count = 0

        def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Thumbnail generation succeeds
                return mock_thumb_process
            else:
                # FFmpeg times out
                mock_ffmpeg = MagicMock()
                mock_ffmpeg.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
                mock_ffmpeg.kill = MagicMock()
                mock_ffmpeg.wait = AsyncMock()
                return mock_ffmpeg

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess):
                with patch("aragora.broadcast.video_gen.generate_thumbnail", return_value=True):
                    with patch("aragora.broadcast.video_gen.get_audio_duration", return_value=60):
                        generator = VideoGenerator(output_dir=temp_dir)
                        result = await generator.generate_static_video(
                            mock_audio_file,
                            "Test",
                            ["claude"],
                        )

        # Should return None on timeout
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_video_cleans_up_thumbnail(self, temp_dir, mock_audio_file):
        """Should clean up temporary thumbnail file."""
        thumb_path = temp_dir / f"{mock_audio_file.stem}_thumb.png"

        # Pre-create thumbnail to verify cleanup
        thumb_path.write_bytes(b"fake thumbnail")

        mock_process = MagicMock()
        mock_process.returncode = 1  # FFmpeg fails
        mock_process.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                with patch("aragora.broadcast.video_gen.generate_thumbnail", return_value=True):
                    with patch("aragora.broadcast.video_gen.get_audio_duration", return_value=60):
                        generator = VideoGenerator(output_dir=temp_dir)
                        await generator.generate_static_video(
                            mock_audio_file,
                            "Test",
                            ["claude"],
                        )

        # Thumbnail should be cleaned up
        assert not thumb_path.exists()
