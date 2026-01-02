import io
import json
import contextlib
from unittest.mock import patch

from aragora.spectate.stream import SpectatorStream
from aragora.spectate.events import SpectatorEvents


@contextlib.contextmanager
def capture_output(stream_obj):
    """Context manager to safely capture output without monkeypatching sys.stdout globally."""
    original_output = stream_obj.output
    captured = io.StringIO()
    stream_obj.output = captured
    try:
        yield captured
    finally:
        stream_obj.output = original_output


def test_spectator_enabled_text_output():
    """Test that enabled spectator produces readable text output."""
    stream = SpectatorStream(enabled=True, format="plain")

    with capture_output(stream) as captured:
        stream.emit(SpectatorEvents.PROPOSAL, agent="TestBot", details="Hello World", metric=0.95)

    output = captured.getvalue()
    assert "TestBot" in output
    assert "Hello" in output  # Partial match for truncated content
    assert "0.95" in output
    # Should NOT contain ANSI codes in plain format
    assert "\033[" not in output


def test_spectator_disabled_no_output():
    """Test that disabled spectator produces no output."""
    stream = SpectatorStream(enabled=False)

    with capture_output(stream) as captured:
        stream.emit(SpectatorEvents.PROPOSAL, agent="TestBot", details="Hello World")

    assert captured.getvalue() == ""


def test_spectator_json_format():
    """Test JSON format produces valid JSON."""
    stream = SpectatorStream(enabled=True, format="json")

    with capture_output(stream) as captured:
        stream.emit(
            SpectatorEvents.CRITIQUE,
            agent="CriticBot",
            details="Good points",
            metric=0.8,
            round_number=2
        )

    output = captured.getvalue().strip()
    parsed = json.loads(output)

    assert parsed["type"] == "critique"
    assert parsed["agent"] == "CriticBot"
    assert parsed["details"] == "Good points"
    assert parsed["metric"] == 0.8
    assert parsed["round"] == 2
    assert "timestamp" in parsed


def test_spectator_encoding_fallback():
    """Test that spectator handles encoding errors gracefully."""
    stream = SpectatorStream(enabled=True, format="plain")

    # Mock a broken output that raises UnicodeEncodeError
    class BrokenOutput:
        def write(self, data):
            raise UnicodeEncodeError('utf-8', data, 0, 1, 'mock error')
        def flush(self):
            pass

    stream.output = BrokenOutput()

    # Should not raise exception
    try:
        stream.emit(SpectatorEvents.PROPOSAL, agent="TestBot", details="Hello World")
    except Exception:
        assert False, "SpectatorStream should handle encoding errors gracefully"


def test_spectator_content_truncation():
    """Test that long content is properly truncated."""
    stream = SpectatorStream(enabled=True, format="plain", preview_length=20)

    long_content = "A" * 100

    with capture_output(stream) as captured:
        stream.emit(SpectatorEvents.PROPOSAL, details=long_content)

    output = captured.getvalue()
    assert "..." in output
    assert len("A" * 100) > len(output)  # Should be shorter


def test_spectator_no_color_detection():
    """Test that NO_COLOR environment variable disables colors."""
    with patch.dict('os.environ', {'NO_COLOR': '1'}):
        stream = SpectatorStream(enabled=True, format="auto")
        # Force re-detection
        stream._detect_capabilities()

        assert stream._use_color is False
        assert stream._use_emoji is False


def test_spectator_unknown_event_type():
    """Test that unknown event types don't crash and use fallback."""
    stream = SpectatorStream(enabled=True, format="plain")

    with capture_output(stream) as captured:
        stream.emit("unknown_event_xyz", agent="TestBot", details="Test")

    output = captured.getvalue()
    assert "TestBot" in output
    assert "Test" in output
    # Should use fallback icon
    assert "[UNKNOWN_EVENT_XYZ]" in output or "•" in output


def test_spectator_round_number_display():
    """Test that round numbers are displayed when provided."""
    stream = SpectatorStream(enabled=True, format="plain")

    with capture_output(stream) as captured:
        stream.emit(SpectatorEvents.ROUND_START, round_number=3, details="Starting round 3")

    output = captured.getvalue()
    assert "R3" in output