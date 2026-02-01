"""
Script generation for Aragora Broadcast.

Parses DebateTrace to extract speaker turns and generates podcast script.
"""

from __future__ import annotations

from dataclasses import dataclass

from aragora.debate.traces import DebateTrace, EventType


@dataclass
class ScriptSegment:
    """A segment of the podcast script."""

    speaker: str  # Agent name or "narrator"
    text: str
    voice_id: str | None = None


def _summarize_code(text: str) -> str:
    """Summarize code blocks in text."""
    lines = text.split("\n")
    if len(lines) > 10:
        return f"Reading code block of {len(lines)} lines..."
    return text


def _extract_content_text(content: dict | str) -> str:
    """Extract text content from event content (dict or string)."""
    if isinstance(content, dict):
        # Try common keys for content text
        return content.get("text") or content.get("content") or str(content)
    return str(content)


def _extract_speaker_turns_from_trace(trace: DebateTrace) -> list[ScriptSegment]:
    """Extract speaker turns from debate trace object."""
    segments = []

    # Opening
    segments.append(
        ScriptSegment(
            speaker="narrator",
            text=f"Welcome to Aragora Broadcast. Today's debate is about: {trace.task[:200]}...",
        )
    )

    previous_agent = None
    for event in trace.events:
        if event.event_type == EventType.MESSAGE and event.agent:
            # Add transition if different agent
            if previous_agent and previous_agent != event.agent:
                segments.append(
                    ScriptSegment(speaker="narrator", text=f"Now, {event.agent} responds.")
                )

            # Extract and summarize content
            text = _extract_content_text(event.content)
            content = _summarize_code(text)
            segments.append(ScriptSegment(speaker=event.agent, text=content))
            previous_agent = event.agent

    # Closing
    segments.append(
        ScriptSegment(
            speaker="narrator", text="That concludes this Aragora debate. Thank you for listening."
        )
    )

    return segments


def _extract_speaker_turns_from_dict(debate: dict) -> list[ScriptSegment]:
    """Extract speaker turns from debate dict (test format)."""
    segments = []

    task = debate.get("task", "an important topic")

    # Opening
    segments.append(
        ScriptSegment(
            speaker="narrator",
            text=f"Welcome to Aragora Broadcast. Today's debate is about: {task[:200]}...",
        )
    )

    # Process messages
    messages = debate.get("messages", [])
    previous_agent = None
    for msg in messages:
        agent = msg.get("agent", "unknown")
        content = msg.get("content", "")

        # Add transition if different agent
        if previous_agent and previous_agent != agent:
            segments.append(ScriptSegment(speaker="narrator", text=f"Now, {agent} responds."))

        content = _summarize_code(content)
        segments.append(ScriptSegment(speaker=agent, text=content))
        previous_agent = agent

    # Closing
    segments.append(
        ScriptSegment(
            speaker="narrator", text="That concludes this Aragora debate. Thank you for listening."
        )
    )

    return segments


@dataclass
class Script:
    """A complete broadcast script."""

    segments: list[ScriptSegment]


def generate_script(trace: DebateTrace | dict) -> Script:
    """
    Generate podcast script from debate trace or dict.

    Args:
        trace: The debate trace to convert (DebateTrace object or dict)

    Returns:
        Script object containing list of segments
    """
    if isinstance(trace, dict):
        segments = _extract_speaker_turns_from_dict(trace)
    else:
        segments = _extract_speaker_turns_from_trace(trace)

    return Script(segments=segments)
