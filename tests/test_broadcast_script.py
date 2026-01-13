"""Tests for broadcast script generation module."""

import pytest
from unittest.mock import Mock

from aragora.broadcast.script_gen import (
    ScriptSegment,
    _summarize_code,
    _extract_content_text,
    _extract_speaker_turns,
    generate_script,
)
from aragora.debate.traces import DebateTrace, TraceEvent, EventType


class TestScriptSegment:
    """Tests for ScriptSegment dataclass."""

    def test_create_segment(self):
        """Creates segment with required fields."""
        segment = ScriptSegment(speaker="agent-1", text="Hello world")
        assert segment.speaker == "agent-1"
        assert segment.text == "Hello world"
        assert segment.voice_id is None

    def test_create_segment_with_voice(self):
        """Creates segment with voice ID."""
        segment = ScriptSegment(speaker="narrator", text="Welcome", voice_id="en-US-ZiraNeural")
        assert segment.voice_id == "en-US-ZiraNeural"


class TestSummarizeCode:
    """Tests for _summarize_code helper."""

    def test_short_text_unchanged(self):
        """Short text is returned unchanged."""
        text = "x = 1\ny = 2"
        assert _summarize_code(text) == text

    def test_long_code_summarized(self):
        """Long code blocks are summarized."""
        lines = ["line " + str(i) for i in range(20)]
        text = "\n".join(lines)
        result = _summarize_code(text)
        assert "20 lines" in result
        assert "Reading code block" in result

    def test_exactly_10_lines_unchanged(self):
        """10 lines or fewer are unchanged."""
        lines = ["line " + str(i) for i in range(10)]
        text = "\n".join(lines)
        assert _summarize_code(text) == text

    def test_11_lines_summarized(self):
        """11 lines triggers summarization."""
        lines = ["line " + str(i) for i in range(11)]
        text = "\n".join(lines)
        result = _summarize_code(text)
        assert "11 lines" in result


class TestExtractContentText:
    """Tests for _extract_content_text helper."""

    def test_string_content(self):
        """String content is returned as-is."""
        assert _extract_content_text("hello") == "hello"

    def test_dict_with_text_key(self):
        """Extracts text from dict with 'text' key."""
        content = {"text": "main text", "other": "ignored"}
        assert _extract_content_text(content) == "main text"

    def test_dict_with_content_key(self):
        """Extracts from 'content' key if 'text' is missing."""
        content = {"content": "fallback text", "other": "ignored"}
        assert _extract_content_text(content) == "fallback text"

    def test_dict_converts_to_string(self):
        """Dicts without text/content keys are stringified."""
        content = {"key": "value"}
        result = _extract_content_text(content)
        assert "key" in result
        assert "value" in result

    def test_non_string_content(self):
        """Non-string content is converted."""
        assert _extract_content_text(123) == "123"


class TestExtractSpeakerTurns:
    """Tests for _extract_speaker_turns function."""

    @pytest.fixture
    def basic_trace(self):
        """Create a basic debate trace."""
        return DebateTrace(
            trace_id="trace-123",
            debate_id="debate-123",
            task="Should we use tabs or spaces?",
            agents=["agent-1", "agent-2"],
            random_seed=42,
            events=[],
        )

    def test_opening_and_closing(self, basic_trace):
        """Generates opening and closing narrator segments."""
        segments = _extract_speaker_turns(basic_trace)

        # Should have at least opening and closing
        assert len(segments) >= 2
        assert segments[0].speaker == "narrator"
        assert "Welcome" in segments[0].text
        assert segments[-1].speaker == "narrator"
        assert "concludes" in segments[-1].text

    def test_includes_task_in_opening(self, basic_trace):
        """Opening includes debate task."""
        segments = _extract_speaker_turns(basic_trace)
        assert "tabs or spaces" in segments[0].text

    def test_extracts_message_events(self, basic_trace):
        """Extracts agent messages from events."""
        basic_trace.events = [
            TraceEvent(
                event_id="e1",
                event_type=EventType.MESSAGE,
                timestamp="2024-01-01T00:00:00",
                round_num=1,
                agent="agent-1",
                content={"text": "I prefer tabs."},
            ),
            TraceEvent(
                event_id="e2",
                event_type=EventType.MESSAGE,
                timestamp="2024-01-01T00:00:01",
                round_num=1,
                agent="agent-2",
                content={"text": "Spaces are better."},
            ),
        ]

        segments = _extract_speaker_turns(basic_trace)

        # Filter out narrator segments
        agent_segments = [s for s in segments if s.speaker != "narrator"]
        assert len(agent_segments) == 2
        assert agent_segments[0].speaker == "agent-1"
        assert "tabs" in agent_segments[0].text
        assert agent_segments[1].speaker == "agent-2"
        assert "Spaces" in agent_segments[1].text

    def test_adds_transitions_between_speakers(self, basic_trace):
        """Adds narrator transitions between different speakers."""
        basic_trace.events = [
            TraceEvent(
                event_id="e1",
                event_type=EventType.MESSAGE,
                timestamp="2024-01-01T00:00:00",
                round_num=1,
                agent="agent-1",
                content={"text": "First point"},
            ),
            TraceEvent(
                event_id="e2",
                event_type=EventType.MESSAGE,
                timestamp="2024-01-01T00:00:01",
                round_num=1,
                agent="agent-2",
                content={"text": "Response"},
            ),
        ]

        segments = _extract_speaker_turns(basic_trace)

        # Should have transition between agent-1 and agent-2
        narrator_segments = [s for s in segments if s.speaker == "narrator"]
        transition_found = any("agent-2 responds" in s.text for s in narrator_segments)
        assert transition_found

    def test_no_transition_for_same_speaker(self, basic_trace):
        """No transition when same speaker continues."""
        basic_trace.events = [
            TraceEvent(
                event_id="e1",
                event_type=EventType.MESSAGE,
                timestamp="2024-01-01T00:00:00",
                round_num=1,
                agent="agent-1",
                content={"text": "First part"},
            ),
            TraceEvent(
                event_id="e2",
                event_type=EventType.MESSAGE,
                timestamp="2024-01-01T00:00:01",
                round_num=1,
                agent="agent-1",  # Same speaker
                content={"text": "Second part"},
            ),
        ]

        segments = _extract_speaker_turns(basic_trace)

        # Should NOT have transition "agent-1 responds"
        narrator_segments = [s for s in segments if s.speaker == "narrator"]
        transition_found = any("agent-1 responds" in s.text for s in narrator_segments)
        assert not transition_found

    def test_ignores_non_message_events(self, basic_trace):
        """Non-MESSAGE events are ignored."""
        basic_trace.events = [
            TraceEvent(
                event_id="e1",
                event_type=EventType.ROUND_START,
                timestamp="2024-01-01T00:00:00",
                round_num=1,
                agent=None,
                content={"round": 1},
            ),
            TraceEvent(
                event_id="e2",
                event_type=EventType.MESSAGE,
                timestamp="2024-01-01T00:00:01",
                round_num=1,
                agent="agent-1",
                content={"text": "Actual message"},
            ),
        ]

        segments = _extract_speaker_turns(basic_trace)
        agent_segments = [s for s in segments if s.speaker != "narrator"]
        assert len(agent_segments) == 1

    def test_long_task_truncated_in_opening(self, basic_trace):
        """Long task is truncated in opening."""
        basic_trace.task = "A" * 500
        segments = _extract_speaker_turns(basic_trace)

        # Task should be truncated to 200 chars + "..."
        assert len(segments[0].text) < 300


class TestGenerateScript:
    """Tests for generate_script function."""

    def test_returns_script_segments(self):
        """Returns list of ScriptSegment objects."""
        trace = DebateTrace(
            trace_id="trace-123",
            debate_id="debate-123",
            task="Test task",
            agents=["agent-1"],
            random_seed=42,
            events=[],
        )

        segments = generate_script(trace)

        assert isinstance(segments, list)
        assert all(isinstance(s, ScriptSegment) for s in segments)

    def test_delegates_to_extract_speaker_turns(self):
        """generate_script delegates to _extract_speaker_turns."""
        trace = DebateTrace(
            trace_id="trace-123",
            debate_id="debate-123",
            task="Test task",
            agents=["agent-1", "agent-2"],
            random_seed=42,
            events=[
                TraceEvent(
                    event_id="e1",
                    event_type=EventType.MESSAGE,
                    timestamp="2024-01-01T00:00:00",
                    round_num=1,
                    agent="agent-1",
                    content={"text": "Test message"},
                )
            ],
        )

        segments = generate_script(trace)

        # Should have opening, message, closing
        assert len(segments) >= 3
        assert segments[0].speaker == "narrator"  # Opening
        assert segments[-1].speaker == "narrator"  # Closing
