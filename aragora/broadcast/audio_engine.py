"""
Audio generation engine for Aragora Broadcast.

Supports multiple TTS backends with automatic fallback:
1. ElevenLabs - Highest quality (cloud, paid)
2. Amazon Polly - High quality neural voices (cloud, AWS)
3. Coqui XTTS v2 - High quality local (GPU recommended)
4. Edge-TTS - Good quality (cloud, free)
5. pyttsx3 - Offline fallback (low quality)

Configure via environment:
    ARAGORA_TTS_ORDER - Comma-separated backend priority (elevenlabs, polly, xtts, edge-tts, pyttsx3)
    ARAGORA_TTS_BACKEND - Force a specific backend
    ARAGORA_ELEVENLABS_API_KEY - Enable ElevenLabs (best quality)
    ARAGORA_XTTS_MODEL_PATH - Optional Coqui XTTS model override
    ARAGORA_POLLY_REGION - AWS region for Polly (fallback to AWS_REGION)
"""

import asyncio
import hashlib
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

from aragora.broadcast.script_gen import ScriptSegment
from aragora.broadcast.tts_backends import (
    TTSBackend,
    get_tts_backend,
    get_fallback_backend,
    EDGE_TTS_VOICES,
)

logger = logging.getLogger(__name__)

# Legacy voice mapping (for backward compatibility)
VOICE_MAP: Dict[str, str] = EDGE_TTS_VOICES

# Global TTS backend instance (lazy initialized)
_tts_backend: Optional[TTSBackend] = None

# Legacy fallback availability (pyttsx3)
try:
    import pyttsx3
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False


def get_audio_backend() -> TTSBackend:
    """Get the TTS backend, initializing if needed."""
    global _tts_backend

    if _tts_backend is None:
        # Check for forced backend
        forced = os.getenv("ARAGORA_TTS_BACKEND") or os.getenv("TTS_BACKEND")
        if forced:
            try:
                _tts_backend = get_tts_backend(forced)
                logger.info(f"Using forced TTS backend: {forced}")
            except Exception as e:
                logger.warning(f"Failed to use forced backend '{forced}': {e}")
                _tts_backend = get_fallback_backend()
        else:
            _tts_backend = get_fallback_backend()

    return _tts_backend


def _get_voice_for_speaker(speaker: str) -> str:
    """Get voice ID for a speaker (legacy compatibility)."""
    return VOICE_MAP.get(speaker, VOICE_MAP.get("narrator", "en-US-AriaNeural"))


def _edge_tts_command() -> Optional[list[str]]:
    """Resolve the edge-tts command in a venv/pyenv-safe way."""
    cmd = shutil.which("edge-tts")
    if cmd:
        return [cmd]
    if importlib.util.find_spec("edge_tts") is not None:
        return [sys.executable, "-m", "edge_tts"]
    return None


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
            cmd = _edge_tts_command()
            if not cmd:
                logger.debug("edge-tts not found in PATH or environment")
                return False
            cmd += [
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


def _generate_fallback_tts_sync(text: str, output_path: Path) -> bool:
    """Generate audio using pyttsx3 fallback (synchronous).

    This is a blocking function - use _generate_fallback_tts() for async contexts.
    """
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


async def _generate_fallback_tts(text: str, output_path: Path) -> bool:
    """Generate audio using pyttsx3 fallback (non-blocking).

    Runs the synchronous pyttsx3 engine in a thread pool to avoid
    blocking the event loop.
    """
    if not FALLBACK_AVAILABLE:
        return False

    return await asyncio.to_thread(_generate_fallback_tts_sync, text, output_path)


async def generate_audio_segment(segment: ScriptSegment, output_dir: Path) -> Optional[Path]:
    """
    Generate audio for a single script segment.

    Uses the best available TTS backend with automatic fallback.

    Args:
        segment: The script segment to convert
        output_dir: Directory to save audio file

    Returns:
        Path to generated audio file, or None if failed
    """
    # Create safe filename using stable hash (sha256 is deterministic across sessions)
    text_hash = hashlib.sha256(segment.text.encode('utf-8')).hexdigest()[:12]
    backend = get_audio_backend()
    backend_ext = ".wav" if backend.name == "xtts" else ".mp3"
    safe_name = f"{segment.speaker}_{text_hash}{backend_ext}"
    output_path = output_dir / safe_name

    # Use the TTS backend abstraction
    result = await backend.synthesize(
        text=segment.text,
        voice=segment.voice_id or segment.speaker,
        output_path=output_path,
    )

    if result:
        return result

    # Legacy fallback path (if new backend fails completely)
    voice = _get_voice_for_speaker(segment.speaker)

    # Try edge-tts directly
    if await _generate_edge_tts(segment.text, voice, output_path):
        return output_path

    # Fallback to pyttsx3
    if await _generate_fallback_tts(segment.text, output_path):
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

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [generate_audio_segment(seg, output_dir) for seg in segments]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None results and log exceptions
    valid_paths: list[Path] = []
    for result in results:
        if isinstance(result, BaseException):
            logger.error(f"Audio segment generation failed: {type(result).__name__}: {result}")
        elif result is not None:
            valid_paths.append(result)
    return valid_paths
