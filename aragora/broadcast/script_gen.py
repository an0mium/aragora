"""
Script generation for Aragora Broadcast.

Parses DebateTrace to extract speaker turns and generates podcast script.
"""

from dataclasses import dataclass
from typing import List, Optional
from aragora.debate.traces import DebateTrace, TraceEvent, EventType


@dataclass
class ScriptSegment:
    """A segment of the podcast script."""
    speaker: str  # Agent name or "narrator"
    text: str
    voice_id: Optional[str] = None


def _summarize_code(text: str) -> str:
    """Summarize code blocks in text."""
    lines = text.split('\n')
    if len(lines) > 10:
        return f"Reading code block of {len(lines)} lines..."
    return text


def _extract_content_text(content: dict | str) -> str:
    """Extract text content from event content (dict or string)."""
    if isinstance(content, dict):
        # Try common keys for content text
        return content.get("text") or content.get("content") or str(content)
    return str(content)


def _extract_speaker_turns(trace: DebateTrace) -> List[ScriptSegment]:
    """Extract speaker turns from debate trace."""
    segments = []

    # Opening
    segments.append(ScriptSegment(
        speaker="narrator",
        text=f"Welcome to Aragora Broadcast. Today's debate is about: {trace.task[:200]}..."
    ))

    previous_agent = None
    for event in trace.events:
        if event.event_type == EventType.MESSAGE and event.agent:
            # Add transition if different agent
            if previous_agent and previous_agent != event.agent:
                segments.append(ScriptSegment(
                    speaker="narrator",
                    text=f"Now, {event.agent} responds."
                ))

            # Extract and summarize content
            text = _extract_content_text(event.content)
            content = _summarize_code(text)
            segments.append(ScriptSegment(
                speaker=event.agent,
                text=content
            ))
            previous_agent = event.agent

    # Closing
    segments.append(ScriptSegment(
        speaker="narrator",
        text="That concludes this Aragora debate. Thank you for listening."
    ))

    return segments


def generate_script(trace: DebateTrace) -> List[ScriptSegment]:
    """
    Generate podcast script from debate trace.

    Args:
        trace: The debate trace to convert

    Returns:
        List of script segments ready for TTS
    """
    return _extract_speaker_turns(trace)