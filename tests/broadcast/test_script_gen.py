"""Tests for broadcast script generation."""

import pytest
from unittest.mock import MagicMock

from aragora.broadcast.script_gen import (
    ScriptSegment,
    _summarize_code,
    _extract_content_text,
    _extract_speaker_turns,
    generate_script,
)
from aragora.debate.traces import DebateTrace, TraceEvent, EventType


class TestScriptSegment:
    """Test ScriptSegment dataclass."""

    def test_basic_segment(self):
        """Test creating a basic segment."""
        segment = ScriptSegment(speaker="claude", text="Hello world")
        assert segment.speaker == "claude"
        assert segment.text == "Hello world"
        assert segment.voice_id is None

    def test_segment_with_voice_id(self):
        """Test segment with custom voice ID."""
        segment = ScriptSegment(speaker="narrator", text="Welcome", voice_id="voice-123")
        assert segment.voice_id == "voice-123"


class TestSummarizeCode:
    """Test code summarization."""

    def test_short_code_unchanged(self):
        """Test short code is returned unchanged."""
        code = "x = 1\ny = 2\nz = x + y"
        result = _summarize_code(code)
        assert result == code

    def test_long_code_summarized(self):
        """Test long code (>10 lines) is summarized."""
        code = "\n".join([f"line_{i} = {i}" for i in range(20)])
        result = _summarize_code(code)
        assert "Reading code block of 20 lines..." in result

    def test_exactly_10_lines(self):
        """Test code with exactly 10 lines is not summarized."""
        code = "\n".join([f"line_{i}" for i in range(10)])
        result = _summarize_code(code)
        assert result == code

    def test_11_lines_summarized(self):
        """Test code with 11 lines is summarized."""
        code = "\n".join([f"line_{i}" for i in range(11)])
        result = _summarize_code(code)
        assert "11 lines" in result


class TestExtractContentText:
    """Test content text extraction."""

    def test_string_content(self):
        """Test extracting from string content."""
        result = _extract_content_text("Hello world")
        assert result == "Hello world"

    def test_dict_with_text_key(self):
        """Test extracting from dict with 'text' key."""
        result = _extract_content_text({"text": "From text key"})
        assert result == "From text key"

    def test_dict_with_content_key(self):
        """Test extracting from dict with 'content' key."""
        result = _extract_content_text({"content": "From content key"})
        assert result == "From content key"

    def test_dict_text_takes_priority(self):
        """Test 'text' key takes priority over 'content'."""
        result = _extract_content_text({"text": "Primary", "content": "Secondary"})
        assert result == "Primary"

    def test_dict_without_known_keys(self):
        """Test dict without known keys converts to string."""
        result = _extract_content_text({"other": "value"})
        assert "other" in result
        assert "value" in result

    def test_empty_text_falls_back_to_content(self):
        """Test empty 'text' falls back to 'content'."""
        result = _extract_content_text({"text": "", "content": "Fallback"})
        assert result == "Fallback"


def _make_event(
    event_type: EventType,
    agent: str,
    content: str,
    event_id: str = "evt-001",
    round_num: int = 1,
    timestamp: str = "2025-01-18T12:00:00Z",
) -> TraceEvent:
    """Helper to create TraceEvent with required fields."""
    return TraceEvent(
        event_id=event_id,
        event_type=event_type,
        timestamp=timestamp,
        round_num=round_num,
        agent=agent,
        content={"text": content},
    )


def _make_trace(
    debate_id: str,
    task: str,
    agents: list,
    events: list = None,
    trace_id: str = "trace-001",
    random_seed: int = 42,
) -> DebateTrace:
    """Helper to create DebateTrace with required fields."""
    return DebateTrace(
        trace_id=trace_id,
        debate_id=debate_id,
        task=task,
        agents=agents,
        random_seed=random_seed,
        events=events or [],
    )


class TestExtractSpeakerTurns:
    """Test speaker turn extraction from debate trace."""

    def test_basic_extraction(self):
        """Test basic extraction from trace."""
        trace = _make_trace(
            debate_id="test-1",
            task="Test debate",
            agents=["claude", "gpt"],
            events=[
                _make_event(EventType.MESSAGE, "claude", "Hello from Claude", "evt-1"),
                _make_event(EventType.MESSAGE, "gpt", "Hello from GPT", "evt-2"),
            ],
        )
        segments = _extract_speaker_turns(trace)

        # Should have: opening, claude message, transition, gpt message, closing
        assert len(segments) >= 4
        assert segments[0].speaker == "narrator"  # Opening
        assert "Welcome" in segments[0].text
        assert any(s.speaker == "claude" for s in segments)
        assert any(s.speaker == "gpt" for s in segments)
        assert segments[-1].speaker == "narrator"  # Closing

    def test_includes_transitions(self):
        """Test transitions are added between different speakers."""
        trace = _make_trace(
            debate_id="test-2",
            task="Test",
            agents=["a", "b"],
            events=[
                _make_event(EventType.MESSAGE, "a", "From A", "evt-1"),
                _make_event(EventType.MESSAGE, "b", "From B", "evt-2"),
            ],
        )
        segments = _extract_speaker_turns(trace)

        # Find transition narrator segments
        transitions = [s for s in segments if s.speaker == "narrator" and "responds" in s.text]
        assert len(transitions) >= 1

    def test_no_transition_same_speaker(self):
        """Test no transition when same speaker continues."""
        trace = _make_trace(
            debate_id="test-3",
            task="Test",
            agents=["a"],
            events=[
                _make_event(EventType.MESSAGE, "a", "First", "evt-1"),
                _make_event(EventType.MESSAGE, "a", "Second", "evt-2"),
            ],
        )
        segments = _extract_speaker_turns(trace)

        # Only one speaker, so no "responds" transitions
        transitions = [s for s in segments if "responds" in s.text]
        assert len(transitions) == 0

    def test_skips_non_message_events(self):
        """Test non-MESSAGE events are skipped."""
        trace = _make_trace(
            debate_id="test-4",
            task="Test",
            agents=["a"],
            events=[
                _make_event(EventType.MESSAGE, "a", "Message", "evt-1"),
                _make_event(EventType.CONSENSUS_CHECK, "a", "yes", "evt-2"),
            ],
        )
        segments = _extract_speaker_turns(trace)

        # Only one MESSAGE event, plus opening/closing
        message_segments = [s for s in segments if s.speaker == "a"]
        assert len(message_segments) == 1


class TestGenerateScript:
    """Test full script generation."""

    def test_generate_basic_script(self):
        """Test generating a basic script."""
        trace = _make_trace(
            debate_id="test-script",
            task="Discuss testing",
            agents=["tester"],
            events=[
                _make_event(EventType.MESSAGE, "tester", "Testing is important", "evt-1"),
            ],
        )
        script = generate_script(trace)

        assert len(script) >= 3  # Opening, message, closing
        assert isinstance(script[0], ScriptSegment)
        assert script[0].speaker == "narrator"

    def test_script_includes_task_in_opening(self):
        """Test script opening includes the task."""
        trace = _make_trace(
            debate_id="test",
            task="Design a rate limiter algorithm",
            agents=["engineer"],
            events=[],
        )
        script = generate_script(trace)

        opening = script[0]
        assert "rate limiter" in opening.text.lower()

    def test_script_truncates_long_task(self):
        """Test very long tasks are truncated in opening."""
        long_task = "x" * 500
        trace = _make_trace(
            debate_id="test",
            task=long_task,
            agents=["agent"],
            events=[],
        )
        script = generate_script(trace)

        opening = script[0]
        # Should be truncated to ~200 chars + "..."
        assert len(opening.text) < 300
