"""
Audio generation engine for Aragora Broadcast.

Uses edge-tts for high-quality text-to-speech generation.
"""

import asyncio
import hashlib
import logging
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Dict, Optional
import subprocess
from aragora.broadcast.script_gen import ScriptSegment

# Voice mapping for agents
VOICE_MAP: Dict[str, str] = {
    "claude-visionary": "en-GB-SoniaNeural",
    "codex-engineer": "en-US-GuyNeural",
    "gemini-visionary": "en-AU-NatashaNeural",
    "grok-lateral-thinker": "en-US-ChristopherNeural",
    "narrator": "en-US-ZiraNeural",  # Narrator voice
}

# Fallback TTS if edge-tts fails
try:
    import pyttsx3
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False


def _get_voice_for_speaker(speaker: str) -> str:
    """Get voice ID for a speaker."""
    return VOICE_MAP.get(speaker, VOICE_MAP["narrator"])


async def _generate_edge_tts(
    text: str,
    voice: str,
    output_path: Path,
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout: float = 60.0,
) -> bool:
    """Generate audio using edge-tts with retry logic.

    Args:
        text: Text to convert to speech
        voice: Voice ID to use
        output_path: Path to save the audio file
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        timeout: Timeout in seconds for each attempt (default: 60.0)

    Returns:
        True if audio was generated successfully, False otherwise
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            cmd = [
                "edge-tts",
                "--voice", voice,
                "--text", text,
                "--write-media", str(output_path),
                "--write-subtitles", str(output_path.with_suffix('.vtt'))  # Optional subtitles
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.warning(
                    f"edge-tts timed out after {timeout}s (attempt {attempt + 1}/{max_retries})"
                )
                last_error = TimeoutError(f"edge-tts timed out after {timeout}s")
                # Continue to retry
            else:
                if process.returncode == 0 and output_path.exists():
                    if attempt > 0:
                        logger.info(f"edge-tts succeeded on attempt {attempt + 1}")
                    return True

                # Non-zero return code - capture error for logging
                error_msg = stderr.decode('utf-8', errors='replace').strip() if stderr else "unknown error"
                logger.debug(
                    f"edge-tts failed (attempt {attempt + 1}/{max_retries}): "
                    f"returncode={process.returncode}, error={error_msg[:200]}"
                )
                last_error = RuntimeError(f"edge-tts returned {process.returncode}: {error_msg[:100]}")

        except FileNotFoundError:
            # edge-tts not installed - no point retrying
            logger.debug("edge-tts not found in PATH")
            return False
        except Exception as e:
            logger.debug(f"edge-tts generation failed (attempt {attempt + 1}/{max_retries}): {e}")
            last_error = e

        # Exponential backoff before retry (except on last attempt)
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.debug(f"Retrying edge-tts in {delay:.1f}s...")
            await asyncio.sleep(delay)

    # All retries exhausted
    logger.warning(f"edge-tts failed after {max_retries} attempts: {last_error}")
    return False


def _generate_fallback_tts(text: str, output_path: Path) -> bool:
    """Generate audio using pyttsx3 fallback."""
    if not FALLBACK_AVAILABLE:
        return False

    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        return output_path.exists()
    except Exception as e:
        logger.debug("pyttsx3 fallback TTS failed: %s", e)
        return False


async def generate_audio_segment(segment: ScriptSegment, output_dir: Path) -> Optional[Path]:
    """
    Generate audio for a single script segment.

    Args:
        segment: The script segment to convert
        output_dir: Directory to save audio file

    Returns:
        Path to generated audio file, or None if failed
    """
    voice = _get_voice_for_speaker(segment.speaker)

    # Create safe filename using stable hash (sha256 is deterministic across sessions)
    text_hash = hashlib.sha256(segment.text.encode('utf-8')).hexdigest()[:12]
    safe_name = f"{segment.speaker}_{text_hash}.mp3"
    output_path = output_dir / safe_name

    # Try edge-tts first
    if await _generate_edge_tts(segment.text, voice, output_path):
        return output_path

    # Fallback to pyttsx3
    if _generate_fallback_tts(segment.text, output_path):
        return output_path

    return None


async def generate_audio(segments: list[ScriptSegment], output_dir: Optional[Path] = None) -> list[Path]:
    """
    Generate audio files for all script segments.

    Args:
        segments: List of script segments
        output_dir: Directory to save audio files. If None, creates a temp
            directory. Caller is responsible for cleanup of this directory
            after use (e.g., with shutil.rmtree).

    Returns:
        List of paths to generated audio files

    Note:
        When output_dir is None, a temporary directory is created that will
        NOT be automatically cleaned up. Use broadcast_debate() instead for
        automatic cleanup, or manually remove the parent directory of the
        returned paths when done.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="aragora_broadcast_"))

    output_dir.mkdir(exist_ok=True)

    tasks = [generate_audio_segment(seg, output_dir) for seg in segments]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None results and log exceptions
    valid_paths = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Audio segment generation failed: {type(result).__name__}: {result}")
        elif result is not None:
            valid_paths.append(result)
    return valid_paths