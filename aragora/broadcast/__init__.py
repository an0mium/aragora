"""
Aragora Broadcast: Post-debate podcast engine.

Creates audio clips from debate traces for passive consumption and sharing.

Usage:
    pip install aragora[broadcast]

    from aragora.broadcast import broadcast_debate
    from aragora.debate.traces import DebateTrace

    trace = DebateTrace.load("debate.json")
    output_path = await broadcast_debate(trace, Path("output.mp3"))
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.debate.traces import DebateTrace

from .audio_engine import VOICE_MAP, generate_audio
from .mixer import mix_audio, mix_audio_with_ffmpeg
from .script_gen import ScriptSegment, generate_script


async def broadcast_debate(
    trace: "DebateTrace",  # Forward reference to avoid circular import
    output_path: Optional[Path] = None,
    format: str = "mp3",
) -> Optional[Path]:
    """
    Generate a complete audio podcast from a debate trace.

    This is the main entry point for converting debates to audio. It:
    1. Generates a podcast script from the debate events
    2. Synthesizes audio for each speaker turn using the configured TTS backend
    3. Mixes all segments into a single audio file

    Args:
        trace: The DebateTrace to convert to audio
        output_path: Where to save the final audio (auto-generated if None)
        format: Audio format ('mp3', 'wav', etc.)

    Returns:
        Path to the generated audio file, or None if failed

    Example:
        >>> from aragora.debate.traces import DebateTrace
        >>> from aragora.broadcast import broadcast_debate
        >>> import asyncio
        >>>
        >>> trace = DebateTrace.load("my_debate.json")
        >>> output = asyncio.run(broadcast_debate(trace))
        >>> print(f"Audio saved to: {output}")
    """
    import tempfile

    # Generate script from trace
    segments = generate_script(trace)
    if not segments:
        return None

    # Use TemporaryDirectory context manager for secure cleanup (prevents race conditions)
    with tempfile.TemporaryDirectory(prefix="aragora_broadcast_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Generate audio for each segment
        audio_files = await generate_audio(segments, temp_dir)
        if not audio_files:
            return None

        # Determine output path
        if output_path is None:
            trace_id = getattr(trace, "id", None) or getattr(trace, "debate_id", "unknown")
            output_path = Path(tempfile.gettempdir()) / f"aragora_debate_{trace_id}.{format}"

        # Mix audio files
        success = mix_audio(audio_files, output_path, format)
        if not success:
            # Try ffmpeg fallback
            success = mix_audio_with_ffmpeg(audio_files, output_path)

        return output_path if success else None
    # TemporaryDirectory auto-cleans on exit


def broadcast_debate_sync(
    trace: "DebateTrace",
    output_path: Optional[Path] = None,
    format: str = "mp3",
) -> Optional[Path]:
    """Synchronous wrapper for broadcast_debate."""
    return asyncio.run(broadcast_debate(trace, output_path, format))


__all__ = [
    "generate_script",
    "ScriptSegment",
    "generate_audio",
    "VOICE_MAP",
    "mix_audio",
    "broadcast_debate",
    "broadcast_debate_sync",
]
