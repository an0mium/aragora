"""
Video generator for YouTube uploads.

Converts debate audio files to video format by combining
audio with static thumbnails or animated visualizations.
"""

import asyncio
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata for generated video."""

    title: str
    description: str
    duration_seconds: int
    file_size_bytes: int
    format: str = "mp4"
    resolution: str = "1920x1080"


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None


def _check_ffprobe() -> bool:
    """Check if ffprobe is available."""
    return shutil.which("ffprobe") is not None


# Subprocess timeout constants
FFPROBE_TIMEOUT = 30  # seconds
FFMPEG_TIMEOUT = 600  # 10 minutes for video encoding
IMAGEMAGICK_TIMEOUT = 60  # seconds


async def get_audio_duration(audio_path: Path) -> Optional[int]:
    """
    Get audio duration in seconds using ffprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds or None if failed
    """
    if not _check_ffprobe():
        logger.warning("ffprobe not available")
        return None

    process = None
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path),
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(
            process.communicate(),
            timeout=FFPROBE_TIMEOUT
        )

        if process.returncode == 0:
            duration_str = stdout.decode('utf-8', errors='replace').strip()
            return int(float(duration_str))
    except asyncio.TimeoutError:
        logger.error(f"ffprobe timed out after {FFPROBE_TIMEOUT}s for {audio_path}")
        if process:
            process.kill()
            await process.wait()
    except Exception as e:
        logger.error(f"Failed to get audio duration: {e}")

    return None


async def generate_thumbnail(
    title: str,
    agents: list[str],
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
) -> bool:
    """
    Generate a simple thumbnail image for the video.

    Uses ImageMagick if available, otherwise creates a basic solid color image.

    Args:
        title: Video title
        agents: List of agent names
        output_path: Where to save the thumbnail
        width: Image width
        height: Image height

    Returns:
        True if successful
    """
    # Check for ImageMagick
    if not shutil.which("convert"):
        # Create a simple blank PNG using pure Python
        # This creates a minimal valid 1920x1080 PNG
        try:
            # Use a simple approach - create solid color thumbnail
            # For production, you'd want a proper image generation library
            import struct
            import zlib

            def create_png(width: int, height: int, color: tuple) -> bytes:
                """Create a minimal PNG with solid color."""
                # PNG header
                header = b'\x89PNG\r\n\x1a\n'

                # IHDR chunk
                ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
                ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data)
                ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)

                # IDAT chunk - scanlines with filter byte
                raw_data = b''
                for _ in range(height):
                    raw_data += b'\x00'  # Filter byte
                    for _ in range(width):
                        raw_data += bytes(color)

                compressed = zlib.compress(raw_data, 9)
                idat_crc = zlib.crc32(b'IDAT' + compressed)
                idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)

                # IEND chunk
                iend_crc = zlib.crc32(b'IEND')
                iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)

                return header + ihdr + idat + iend

            # Dark blue background
            png_data = create_png(width, height, (30, 40, 80))
            output_path.write_bytes(png_data)
            return True

        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return False

    # Use ImageMagick for nicer thumbnail
    process = None
    try:
        agent_text = ", ".join(agents[:3])
        if len(agents) > 3:
            agent_text += f" +{len(agents) - 3}"

        # Truncate title if too long
        display_title = title[:60] + "..." if len(title) > 60 else title

        cmd = [
            "convert",
            "-size", f"{width}x{height}",
            "xc:#1e2850",  # Dark blue background
            "-font", "Helvetica",
            "-pointsize", "72",
            "-fill", "white",
            "-gravity", "center",
            "-annotate", "+0-100", display_title,
            "-pointsize", "36",
            "-fill", "#aaaaaa",
            "-annotate", "+0+100", f"Aragora Debate",
            "-annotate", "+0+160", agent_text,
            str(output_path),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(
            process.communicate(),
            timeout=IMAGEMAGICK_TIMEOUT
        )
        return process.returncode == 0 and output_path.exists()

    except asyncio.TimeoutError:
        logger.error(f"ImageMagick timed out after {IMAGEMAGICK_TIMEOUT}s")
        if process:
            process.kill()
            await process.wait()
        return False
    except Exception as e:
        logger.error(f"Failed to generate thumbnail with ImageMagick: {e}")
        return False


class VideoGenerator:
    """
    Convert audio to video for YouTube upload.

    Uses ffmpeg to combine audio with static or animated visuals.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize video generator.

        Args:
            output_dir: Directory for output videos. Defaults to temp dir.
        """
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "aragora_videos"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ffmpeg_available = _check_ffmpeg()
        if not self.ffmpeg_available:
            logger.warning("ffmpeg not available - video generation disabled")

    async def generate_static_video(
        self,
        audio_path: Path,
        title: str,
        agents: list[str],
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate video with static thumbnail and audio track.

        This is the simplest approach - just a still image with audio.

        Args:
            audio_path: Path to audio file
            title: Video title (used for thumbnail)
            agents: List of agents (used for thumbnail)
            output_path: Output video path. Auto-generated if not provided.

        Returns:
            Path to video file or None if failed
        """
        if not self.ffmpeg_available:
            logger.error("ffmpeg not available")
            return None

        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        # Generate output path
        if output_path is None:
            output_path = self.output_dir / f"{audio_path.stem}.mp4"

        # Create temporary thumbnail
        thumb_path = self.output_dir / f"{audio_path.stem}_thumb.png"

        try:
            # Generate thumbnail
            if not await generate_thumbnail(title, agents, thumb_path):
                logger.warning("Using fallback thumbnail generation")
                # Create minimal valid PNG as fallback
                thumb_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

            # Get audio duration
            duration = await get_audio_duration(audio_path)

            # Generate video with ffmpeg
            # -loop 1: loop the image
            # -tune stillimage: optimize for static content
            # -shortest: end when audio ends
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-loop", "1",
                "-i", str(thumb_path),
                "-i", str(audio_path),
                "-c:v", "libx264",
                "-tune", "stillimage",
                "-c:a", "aac",
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                "-movflags", "+faststart",  # Web optimization
                str(output_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=FFMPEG_TIMEOUT
            )

            if process.returncode != 0:
                logger.error(f"ffmpeg failed: {stderr.decode('utf-8', errors='replace')[:500]}")
                return None

            if output_path.exists():
                logger.info(f"Generated video: {output_path}")
                return output_path

        except asyncio.TimeoutError:
            logger.error(f"ffmpeg timed out after {FFMPEG_TIMEOUT}s for {audio_path}")
            if process:
                process.kill()
                await process.wait()

        except Exception as e:
            logger.error(f"Video generation failed: {e}")

        finally:
            # Cleanup thumbnail
            if thumb_path.exists():
                thumb_path.unlink()

        return None

    async def generate_waveform_video(
        self,
        audio_path: Path,
        output_path: Optional[Path] = None,
        color: str = "0x4488ff",
    ) -> Optional[Path]:
        """
        Generate video with animated audio waveform.

        More visually interesting than static image.

        Args:
            audio_path: Path to audio file
            output_path: Output video path
            color: Waveform color in hex

        Returns:
            Path to video file or None if failed
        """
        if not self.ffmpeg_available:
            logger.error("ffmpeg not available")
            return None

        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        if output_path is None:
            output_path = self.output_dir / f"{audio_path.stem}_waveform.mp4"

        try:
            # Generate waveform video using ffmpeg's showwaves filter
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(audio_path),
                "-filter_complex",
                f"[0:a]showwaves=s=1920x1080:mode=cline:colors={color}[v]",
                "-map", "[v]",
                "-map", "0:a",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=FFMPEG_TIMEOUT
            )

            if process.returncode != 0:
                logger.error(f"ffmpeg waveform failed: {stderr.decode('utf-8', errors='replace')[:500]}")
                return None

            if output_path.exists():
                logger.info(f"Generated waveform video: {output_path}")
                return output_path

        except asyncio.TimeoutError:
            logger.error(f"ffmpeg waveform timed out after {FFMPEG_TIMEOUT}s for {audio_path}")
            if process:
                process.kill()
                await process.wait()

        except Exception as e:
            logger.error(f"Waveform video generation failed: {e}")

        return None

    def get_video_metadata(self, video_path: Path) -> Optional[VideoMetadata]:
        """
        Get metadata for a generated video.

        Args:
            video_path: Path to video file

        Returns:
            VideoMetadata or None if failed
        """
        if not video_path.exists():
            return None

        try:
            file_size = video_path.stat().st_size

            # Get duration using ffprobe synchronously
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=FFPROBE_TIMEOUT,
            )

            if result.returncode != 0:
                logger.warning(f"ffprobe failed for {video_path}: {result.stderr}")
                duration = 0
            else:
                try:
                    duration = int(float(result.stdout.strip()))
                except ValueError:
                    logger.warning(f"ffprobe returned invalid duration: {result.stdout!r}")
                    duration = 0

            return VideoMetadata(
                title=video_path.stem,
                description="",
                duration_seconds=duration,
                file_size_bytes=file_size,
            )

        except subprocess.TimeoutExpired:
            logger.error(f"ffprobe timed out after {FFPROBE_TIMEOUT}s for {video_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
            return None

    def cleanup(self, video_path: Path) -> None:
        """Remove a generated video file."""
        if video_path.exists():
            video_path.unlink()
