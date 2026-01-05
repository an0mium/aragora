"""
Audio generation engine for Aragora Broadcast.

Uses edge-tts for high-quality text-to-speech generation.
"""

import asyncio
import tempfile
import os
from pathlib import Path
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


async def _generate_edge_tts(text: str, voice: str, output_path: Path) -> bool:
    """Generate audio using edge-tts."""
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
        await process.communicate()
        return process.returncode == 0 and output_path.exists()
    except Exception:
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
    except Exception:
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

    # Create safe filename
    safe_name = f"{segment.speaker}_{hash(segment.text) % 10000}.mp3"
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
        output_dir: Directory to save audio files (temp dir if None)

    Returns:
        List of paths to generated audio files
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