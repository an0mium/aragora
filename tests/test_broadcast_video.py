"""Tests for broadcast video generation module."""

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.broadcast.video_gen import (
    FFMPEG_TIMEOUT,
    FFPROBE_TIMEOUT,
    IMAGEMAGICK_TIMEOUT,
    VideoGenerator,
    VideoMetadata,
    _check_ffmpeg,
    _check_ffprobe,
    generate_thumbnail,
    get_audio_duration,
)


# =============================================================================
# Test VideoMetadata Dataclass
# =============================================================================


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""

    def test_metadata_creation_minimal(self):
        """Create VideoMetadata with required fields only."""
        metadata = VideoMetadata(
            title="Test Video",
            description="A test description",
            duration_seconds=120,
            file_size_bytes=1024000,
        )

        assert metadata.title == "Test Video"
        assert metadata.description == "A test description"
        assert metadata.duration_seconds == 120
        assert metadata.file_size_bytes == 1024000
        assert metadata.format == "mp4"  # Default
        assert metadata.resolution == "1920x1080"  # Default

    def test_metadata_with_all_fields(self):
        """Create VideoMetadata with all fields specified."""
        metadata = VideoMetadata(
            title="Custom Video",
            description="Custom desc",
            duration_seconds=300,
            file_size_bytes=5000000,
            format="webm",
            resolution="1280x720",
        )

        assert metadata.format == "webm"
        assert metadata.resolution == "1280x720"

    def test_metadata_zero_values(self):
        """VideoMetadata accepts zero values."""
        metadata = VideoMetadata(
            title="",
            description="",
            duration_seconds=0,
            file_size_bytes=0,
        )

        assert metadata.duration_seconds == 0
        assert metadata.file_size_bytes == 0


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestCheckFfmpeg:
    """Tests for ffmpeg/ffprobe availability checks."""

    def test_check_ffmpeg_available(self):
        """_check_ffmpeg returns True when ffmpeg is found."""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            assert _check_ffmpeg() is True

    def test_check_ffmpeg_unavailable(self):
        """_check_ffmpeg returns False when ffmpeg is not found."""
        with patch("shutil.which", return_value=None):
            assert _check_ffmpeg() is False

    def test_check_ffprobe_available(self):
        """_check_ffprobe returns True when ffprobe is found."""
        with patch("shutil.which", return_value="/usr/bin/ffprobe"):
            assert _check_ffprobe() is True

    def test_check_ffprobe_unavailable(self):
        """_check_ffprobe returns False when ffprobe is not found."""
        with patch("shutil.which", return_value=None):
            assert _check_ffprobe() is False


# =============================================================================
# Test get_audio_duration
# =============================================================================


class TestGetAudioDuration:
    """Tests for get_audio_duration function."""

    @pytest.mark.asyncio
    async def test_get_audio_duration_success(self, tmp_path):
        """Successfully get audio duration from ffprobe."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"125.5\n", b"")
        mock_process.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                duration = await get_audio_duration(audio_file)

        assert duration == 125  # Truncated to int

    @pytest.mark.asyncio
    async def test_get_audio_duration_ffprobe_unavailable(self, tmp_path):
        """Return None when ffprobe is not available."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        with patch("shutil.which", return_value=None):
            duration = await get_audio_duration(audio_file)

        assert duration is None

    @pytest.mark.asyncio
    async def test_get_audio_duration_timeout(self, tmp_path):
        """Handle ffprobe timeout gracefully."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_process = AsyncMock()
        mock_process.communicate.side_effect = asyncio.TimeoutError()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                    duration = await get_audio_duration(audio_file)

        assert duration is None

    @pytest.mark.asyncio
    async def test_get_audio_duration_ffprobe_error(self, tmp_path):
        """Handle ffprobe non-zero exit code."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"error")
        mock_process.returncode = 1

        with patch("shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                duration = await get_audio_duration(audio_file)

        assert duration is None

    @pytest.mark.asyncio
    async def test_get_audio_duration_invalid_output(self, tmp_path):
        """Handle ffprobe returning non-numeric output."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"not a number\n", b"")
        mock_process.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                duration = await get_audio_duration(audio_file)

        assert duration is None


# =============================================================================
# Test generate_thumbnail
# =============================================================================


class TestGenerateThumbnail:
    """Tests for generate_thumbnail function."""

    @pytest.mark.asyncio
    async def test_generate_thumbnail_fallback_png(self, tmp_path):
        """Generate thumbnail using pure Python when ImageMagick unavailable."""
        output_path = tmp_path / "thumb.png"

        with patch("shutil.which", return_value=None):  # No ImageMagick
            # Use small dimensions to avoid slow pure-Python PNG generation
            result = await generate_thumbnail(
                title="Test Debate",
                agents=["Claude", "Gemini"],
                output_path=output_path,
                width=64,
                height=64,
            )

        assert result is True
        assert output_path.exists()
        # Check PNG signature
        data = output_path.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n'

    @pytest.mark.asyncio
    async def test_generate_thumbnail_with_imagemagick(self, tmp_path):
        """Generate thumbnail using ImageMagick when available."""
        output_path = tmp_path / "thumb.png"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/convert"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                # Create the file to simulate ImageMagick success
                output_path.write_bytes(b"fake png")
                result = await generate_thumbnail(
                    title="Test Debate",
                    agents=["Claude", "Gemini"],
                    output_path=output_path,
                )

        assert result is True

    @pytest.mark.asyncio
    async def test_generate_thumbnail_imagemagick_timeout(self, tmp_path):
        """Handle ImageMagick timeout gracefully."""
        output_path = tmp_path / "thumb.png"

        mock_process = AsyncMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("shutil.which", return_value="/usr/bin/convert"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                    result = await generate_thumbnail(
                        title="Test",
                        agents=["Agent"],
                        output_path=output_path,
                    )

        assert result is False

    @pytest.mark.asyncio
    async def test_generate_thumbnail_many_agents(self, tmp_path):
        """Thumbnail with many agents truncates display."""
        output_path = tmp_path / "thumb.png"

        with patch("shutil.which", return_value=None):  # Use fallback
            result = await generate_thumbnail(
                title="Test",
                agents=["A1", "A2", "A3", "A4", "A5"],
                output_path=output_path,
                width=64,
                height=64,
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_generate_thumbnail_long_title(self, tmp_path):
        """Thumbnail with long title gets truncated."""
        output_path = tmp_path / "thumb.png"

        with patch("shutil.which", return_value=None):  # Use fallback
            result = await generate_thumbnail(
                title="A" * 100,  # Very long title
                agents=["Agent"],
                output_path=output_path,
                width=64,
                height=64,
            )

        assert result is True


# =============================================================================
# Test VideoGenerator Class
# =============================================================================


class TestVideoGeneratorInit:
    """Tests for VideoGenerator initialization."""

    def test_init_creates_output_dir(self, tmp_path):
        """VideoGenerator creates output directory if it doesn't exist."""
        output_dir = tmp_path / "new_dir" / "videos"
        assert not output_dir.exists()

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            gen = VideoGenerator(output_dir=output_dir)

        assert output_dir.exists()
        assert gen.output_dir == output_dir

    def test_init_uses_temp_dir_by_default(self):
        """VideoGenerator uses temp directory when none specified."""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            gen = VideoGenerator()

        assert "aragora_videos" in str(gen.output_dir)

    def test_init_ffmpeg_unavailable(self, tmp_path):
        """VideoGenerator notes when ffmpeg is unavailable."""
        with patch("shutil.which", return_value=None):
            gen = VideoGenerator(output_dir=tmp_path)

        assert gen.ffmpeg_available is False


class TestGenerateStaticVideo:
    """Tests for VideoGenerator.generate_static_video method."""

    @pytest.fixture
    def video_gen(self, tmp_path):
        """Create VideoGenerator with mocked ffmpeg."""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            return VideoGenerator(output_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_generate_static_video_success(self, video_gen, tmp_path):
        """Successfully generate static video."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")
        output_file = tmp_path / "output.mp4"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0

        # Mock both ffprobe for duration and ffmpeg for encoding
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("aragora.broadcast.video_gen.get_audio_duration", return_value=120):
                with patch("aragora.broadcast.video_gen.generate_thumbnail", return_value=True):
                    # Create output file to simulate success
                    output_file.write_bytes(b"fake video")
                    result = await video_gen.generate_static_video(
                        audio_path=audio_file,
                        title="Test Video",
                        agents=["Claude"],
                        output_path=output_file,
                    )

        assert result == output_file

    @pytest.mark.asyncio
    async def test_generate_static_video_missing_audio(self, video_gen, tmp_path):
        """Return None when audio file doesn't exist."""
        audio_file = tmp_path / "nonexistent.mp3"

        result = await video_gen.generate_static_video(
            audio_path=audio_file,
            title="Test",
            agents=["Agent"],
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_static_video_ffmpeg_unavailable(self, tmp_path):
        """Return None when ffmpeg is unavailable."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        with patch("shutil.which", return_value=None):
            gen = VideoGenerator(output_dir=tmp_path)
            result = await gen.generate_static_video(
                audio_path=audio_file,
                title="Test",
                agents=["Agent"],
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_static_video_ffmpeg_failure(self, video_gen, tmp_path):
        """Return None when ffmpeg fails."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"encoding error")
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("aragora.broadcast.video_gen.generate_thumbnail", return_value=True):
                result = await video_gen.generate_static_video(
                    audio_path=audio_file,
                    title="Test",
                    agents=["Agent"],
                )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_static_video_timeout(self, video_gen, tmp_path):
        """Handle ffmpeg timeout gracefully."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_process = AsyncMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                with patch("aragora.broadcast.video_gen.generate_thumbnail", return_value=True):
                    result = await video_gen.generate_static_video(
                        audio_path=audio_file,
                        title="Test",
                        agents=["Agent"],
                    )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_static_video_auto_output_path(self, video_gen, tmp_path):
        """Auto-generate output path when not specified."""
        audio_file = tmp_path / "my_audio.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("aragora.broadcast.video_gen.generate_thumbnail", return_value=True):
                # Create the expected output file
                expected_output = video_gen.output_dir / "my_audio.mp4"
                expected_output.write_bytes(b"fake video")

                result = await video_gen.generate_static_video(
                    audio_path=audio_file,
                    title="Test",
                    agents=["Agent"],
                )

        assert result is not None
        assert result.name == "my_audio.mp4"


class TestGenerateWaveformVideo:
    """Tests for VideoGenerator.generate_waveform_video method."""

    @pytest.fixture
    def video_gen(self, tmp_path):
        """Create VideoGenerator with mocked ffmpeg."""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            return VideoGenerator(output_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_generate_waveform_video_success(self, video_gen, tmp_path):
        """Successfully generate waveform video."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")
        output_file = tmp_path / "waveform.mp4"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            output_file.write_bytes(b"fake video")
            result = await video_gen.generate_waveform_video(
                audio_path=audio_file,
                output_path=output_file,
            )

        assert result == output_file

    @pytest.mark.asyncio
    async def test_generate_waveform_video_missing_audio(self, video_gen, tmp_path):
        """Return None when audio file doesn't exist."""
        audio_file = tmp_path / "nonexistent.mp3"

        result = await video_gen.generate_waveform_video(audio_path=audio_file)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_waveform_video_custom_color(self, video_gen, tmp_path):
        """Use custom waveform color."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")
        output_file = tmp_path / "waveform.mp4"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            output_file.write_bytes(b"fake video")
            await video_gen.generate_waveform_video(
                audio_path=audio_file,
                output_path=output_file,
                color="0xff0000",
            )

        # Verify color was passed to ffmpeg
        call_args = mock_exec.call_args
        cmd_str = " ".join(str(arg) for arg in call_args[0])
        assert "0xff0000" in cmd_str

    @pytest.mark.asyncio
    async def test_generate_waveform_video_timeout(self, video_gen, tmp_path):
        """Handle ffmpeg waveform timeout."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_process = AsyncMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                result = await video_gen.generate_waveform_video(
                    audio_path=audio_file,
                )

        assert result is None


class TestGetVideoMetadata:
    """Tests for VideoGenerator.get_video_metadata method."""

    @pytest.fixture
    def video_gen(self, tmp_path):
        """Create VideoGenerator."""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            return VideoGenerator(output_dir=tmp_path)

    def test_get_video_metadata_success(self, video_gen, tmp_path):
        """Successfully get video metadata."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"x" * 1000)  # 1000 byte file

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "180.5"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            metadata = video_gen.get_video_metadata(video_file)

        assert metadata is not None
        assert metadata.title == "test"
        assert metadata.duration_seconds == 180
        assert metadata.file_size_bytes == 1000
        assert metadata.format == "mp4"

    def test_get_video_metadata_missing_file(self, video_gen, tmp_path):
        """Return None for missing video file."""
        video_file = tmp_path / "nonexistent.mp4"

        metadata = video_gen.get_video_metadata(video_file)

        assert metadata is None

    def test_get_video_metadata_ffprobe_timeout(self, video_gen, tmp_path):
        """Handle ffprobe timeout."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 30)):
            metadata = video_gen.get_video_metadata(video_file)

        assert metadata is None

    def test_get_video_metadata_ffprobe_error(self, video_gen, tmp_path):
        """Handle ffprobe returning error."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"

        with patch("subprocess.run", return_value=mock_result):
            metadata = video_gen.get_video_metadata(video_file)

        assert metadata is not None
        assert metadata.duration_seconds == 0  # Falls back to 0

    def test_get_video_metadata_invalid_duration(self, video_gen, tmp_path):
        """Handle ffprobe returning invalid duration string."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not a number"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            metadata = video_gen.get_video_metadata(video_file)

        assert metadata is not None
        assert metadata.duration_seconds == 0


class TestVideoGeneratorCleanup:
    """Tests for VideoGenerator.cleanup method."""

    def test_cleanup_existing_file(self, tmp_path):
        """cleanup removes existing video file."""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            gen = VideoGenerator(output_dir=tmp_path)

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")
        assert video_file.exists()

        gen.cleanup(video_file)

        assert not video_file.exists()

    def test_cleanup_nonexistent_file(self, tmp_path):
        """cleanup handles nonexistent file gracefully."""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            gen = VideoGenerator(output_dir=tmp_path)

        video_file = tmp_path / "nonexistent.mp4"

        # Should not raise
        gen.cleanup(video_file)


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_ffprobe_timeout_value(self):
        """FFPROBE_TIMEOUT is reasonable."""
        assert FFPROBE_TIMEOUT == 30
        assert FFPROBE_TIMEOUT > 0

    def test_ffmpeg_timeout_value(self):
        """FFMPEG_TIMEOUT is reasonable for video encoding."""
        assert FFMPEG_TIMEOUT == 600  # 10 minutes
        assert FFMPEG_TIMEOUT > 0

    def test_imagemagick_timeout_value(self):
        """IMAGEMAGICK_TIMEOUT is reasonable."""
        assert IMAGEMAGICK_TIMEOUT == 60
        assert IMAGEMAGICK_TIMEOUT > 0
