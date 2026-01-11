"""
Audio mixing and concatenation for Aragora Broadcast.

Combines individual audio segments into a single podcast file.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def _detect_audio_codec(audio_file: Path) -> Optional[str]:
    """
    Detect audio codec of a file using ffprobe.

    Args:
        audio_file: Path to audio file

    Returns:
        Codec name (e.g., 'mp3', 'pcm_s16le') or None if detection fails
    """
    import subprocess

    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0",
            str(audio_file)
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, shell=False
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _has_mixed_codecs(audio_files: List[Path]) -> bool:
    """
    Check if audio files have different codecs.

    Args:
        audio_files: List of audio file paths

    Returns:
        True if files have different codecs (require re-encoding)
    """
    codecs = set()
    for audio_file in audio_files:
        if audio_file.exists():
            codec = _detect_audio_codec(audio_file)
            if codec:
                codecs.add(codec)
    return len(codecs) > 1


def mix_audio(audio_files: List[Path], output_path: Path, format: str = "mp3") -> bool:
    """
    Mix and concatenate audio files into a single output file.

    Args:
        audio_files: List of audio file paths to concatenate
        output_path: Path for the final mixed audio file
        format: Output format ('mp3', 'wav', etc.)

    Returns:
        True if successful, False otherwise
    """
    if not PYDUB_AVAILABLE:
        logger.error("pydub not available. Install with: pip install pydub")
        return False

    if not audio_files:
        logger.warning("No audio files to mix")
        return False

    try:
        # Load and concatenate audio segments
        combined = AudioSegment.empty()
        mixed_count = 0

        for audio_file in audio_files:
            if not audio_file.exists():
                logger.warning(f"Audio file not found: {audio_file}")
                continue

            segment = AudioSegment.from_file(str(audio_file))
            combined += segment
            mixed_count += 1

        # Fail if no files were actually mixed
        if mixed_count == 0:
            logger.error(f"No valid audio files to mix from {len(audio_files)} provided")
            return False

        # Export the combined audio
        combined.export(str(output_path), format=format)
        return True

    except (OSError, IOError) as e:
        logger.error(f"File I/O error mixing audio: {e}")
        return False
    except PermissionError as e:
        logger.error(f"Permission denied mixing audio: {e}")
        return False
    except Exception as e:
        # Catch pydub errors and other unexpected issues
        logger.error(f"Unexpected error mixing audio: {type(e).__name__}: {e}")
        return False


def mix_audio_with_ffmpeg(audio_files: List[Path], output_path: Path) -> bool:
    """
    Fallback mixing using ffmpeg directly.

    Args:
        audio_files: List of audio file paths
        output_path: Output path

    Returns:
        True if successful
    """
    if not audio_files:
        return False

    import subprocess

    try:
        # Use TemporaryDirectory for secure temp file handling (auto-cleanup, restricted perms)
        with tempfile.TemporaryDirectory(prefix="aragora_ffmpeg_") as temp_dir:
            file_list = os.path.join(temp_dir, "filelist.txt")

            # Write file list for ffmpeg concat
            with open(file_list, 'w') as f:
                for audio_file in audio_files:
                    # Escape single quotes in file paths for ffmpeg concat format
                    escaped_path = str(audio_file.absolute()).replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            # Detect if we need to re-encode (mixed codecs)
            needs_reencode = _has_mixed_codecs(audio_files)

            if needs_reencode:
                # Re-encode to mp3 for compatibility with mixed formats
                logger.info("Mixed audio formats detected, re-encoding to mp3")
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", file_list,
                    "-c:a", "libmp3lame", "-q:a", "2",  # High quality MP3
                    str(output_path)
                ]
            else:
                # Stream copy (faster, no quality loss) when codecs match
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", file_list, "-c", "copy", str(output_path)
                ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, shell=False)
            if result.returncode != 0:
                logger.error(f"FFmpeg mixing failed (exit {result.returncode}): {result.stderr}")
                return False
            return True

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg mixing timed out after 5 minutes")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Install ffmpeg to use audio mixing.")
        return False
    except (OSError, IOError) as e:
        logger.error(f"File I/O error in FFmpeg mixing: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in FFmpeg mixing: {type(e).__name__}: {e}")
        return False