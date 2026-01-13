"""Tests for spectate module - real-time event streaming."""

import io
import json
import os
from unittest.mock import patch

import pytest

from aragora.spectate.events import (
    EVENT_ASCII,
    EVENT_STYLES,
    SpectatorEvents,
)
from aragora.spectate.stream import (
    SpectatorStream,
    VALID_EVENT_TYPES,
)


class TestSpectatorEvents:
    """Tests for SpectatorEvents constants."""

    def test_debate_lifecycle_events(self):
        """Test debate lifecycle event constants."""
        assert SpectatorEvents.DEBATE_START == "debate_start"
        assert SpectatorEvents.DEBATE_END == "debate_end"

    def test_round_lifecycle_events(self):
        """Test round lifecycle event constants."""
        assert SpectatorEvents.ROUND_START == "round_start"
        assert SpectatorEvents.ROUND_END == "round_end"

    def test_agent_action_events(self):
        """Test agent action event constants."""
        assert SpectatorEvents.PROPOSAL == "proposal"
        assert SpectatorEvents.CRITIQUE == "critique"
        assert SpectatorEvents.REFINE == "refine"
        assert SpectatorEvents.VOTE == "vote"
        assert SpectatorEvents.JUDGE == "judge"

    def test_consensus_events(self):
        """Test consensus/convergence event constants."""
        assert SpectatorEvents.CONSENSUS == "consensus"
        assert SpectatorEvents.CONVERGENCE == "convergence"
        assert SpectatorEvents.CONVERGED == "converged"

    def test_memory_events(self):
        """Test memory-related event constants."""
        assert SpectatorEvents.MEMORY_RECALL == "memory_recall"

    def test_system_events(self):
        """Test system event constants."""
        assert SpectatorEvents.SYSTEM == "system"
        assert SpectatorEvents.ERROR == "error"

    def test_breakpoint_events(self):
        """Test breakpoint event constants."""
        assert SpectatorEvents.BREAKPOINT == "breakpoint"
        assert SpectatorEvents.BREAKPOINT_RESOLVED == "breakpoint_resolved"


class TestEventStyles:
    """Tests for event styling configuration."""

    def test_all_events_have_styles(self):
        """Test that all main events have styles defined."""
        assert SpectatorEvents.DEBATE_START in EVENT_STYLES
        assert SpectatorEvents.DEBATE_END in EVENT_STYLES
        assert SpectatorEvents.PROPOSAL in EVENT_STYLES
        assert SpectatorEvents.CRITIQUE in EVENT_STYLES
        assert SpectatorEvents.CONSENSUS in EVENT_STYLES
        assert SpectatorEvents.ERROR in EVENT_STYLES

    def test_style_format(self):
        """Test that styles are (emoji, color) tuples."""
        for event, style in EVENT_STYLES.items():
            assert isinstance(style, tuple), f"{event} style should be tuple"
            assert len(style) == 2, f"{event} style should have 2 elements"
            icon, color = style
            assert isinstance(icon, str), f"{event} icon should be string"
            assert isinstance(color, str), f"{event} color should be string"

    def test_all_events_have_ascii_fallback(self):
        """Test that all styled events have ASCII fallbacks."""
        assert SpectatorEvents.DEBATE_START in EVENT_ASCII
        assert SpectatorEvents.PROPOSAL in EVENT_ASCII
        assert SpectatorEvents.ERROR in EVENT_ASCII

    def test_ascii_fallback_format(self):
        """Test ASCII fallbacks are bracket-wrapped."""
        for event, ascii_icon in EVENT_ASCII.items():
            assert isinstance(ascii_icon, str)
            assert ascii_icon.startswith("["), f"{event} ASCII should start with ["
            assert ascii_icon.endswith("]"), f"{event} ASCII should end with ]"


class TestSpectatorStream:
    """Tests for SpectatorStream class."""

    def test_init_disabled_by_default(self):
        """Test stream is disabled by default."""
        stream = SpectatorStream()
        assert stream.enabled is False

    def test_init_enabled(self):
        """Test stream can be enabled."""
        stream = SpectatorStream(enabled=True)
        assert stream.enabled is True

    def test_init_custom_output(self):
        """Test stream can use custom output."""
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output)
        assert stream.output is output

    def test_init_format_options(self):
        """Test format options."""
        for fmt in ["auto", "ansi", "plain", "json"]:
            stream = SpectatorStream(enabled=True, format=fmt)
            assert stream.format == fmt

    def test_detect_capabilities_plain_format(self):
        """Test plain format disables color and emoji."""
        stream = SpectatorStream(enabled=True, format="plain")
        assert stream._use_color is False
        assert stream._use_emoji is False

    def test_detect_capabilities_ansi_format(self):
        """Test ANSI format enables color and emoji."""
        stream = SpectatorStream(enabled=True, format="ansi")
        assert stream._use_color is True
        assert stream._use_emoji is True

    def test_emit_when_disabled(self):
        """Test emit does nothing when disabled."""
        output = io.StringIO()
        stream = SpectatorStream(enabled=False, output=output)
        stream.emit(SpectatorEvents.DEBATE_START, agent="test")
        assert output.getvalue() == ""

    def test_emit_json_format(self):
        """Test JSON output format."""
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="json")

        stream.emit(
            SpectatorEvents.PROPOSAL,
            agent="claude",
            details="test details",
            metric=0.95,
            round_number=1,
        )

        line = output.getvalue().strip()
        data = json.loads(line)

        assert data["type"] == "proposal"
        assert data["agent"] == "claude"
        assert data["details"] == "test details"
        assert data["metric"] == 0.95
        assert data["round"] == 1
        assert "timestamp" in data

    def test_emit_text_format_plain(self):
        """Test plain text output format."""
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")

        stream.emit(
            SpectatorEvents.PROPOSAL,
            agent="claude",
            details="test details",
            round_number=1,
        )

        line = output.getvalue()
        assert "[PROPOSE]" in line  # ASCII fallback
        assert "claude" in line
        assert "test details" in line
        assert "R1" in line

    def test_emit_text_format_ansi(self):
        """Test ANSI text output format."""
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="ansi")

        stream.emit(
            SpectatorEvents.PROPOSAL,
            agent="claude",
            details="test",
        )

        line = output.getvalue()
        assert "claude" in line
        assert "\033[" in line  # ANSI codes present

    def test_emit_with_metric(self):
        """Test emit with metric value."""
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")

        stream.emit(SpectatorEvents.CONSENSUS, metric=0.85)

        line = output.getvalue()
        assert "(0.85)" in line

    def test_emit_handles_unknown_event(self):
        """Test emit handles unknown event types gracefully."""
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")

        # Should not raise, just log warning
        stream.emit("unknown_event_type", agent="test")
        assert output.getvalue()  # Some output was produced

    def test_emit_handles_exceptions(self):
        """Test emit catches and handles exceptions gracefully."""

        # Create a mock output that raises on write
        class FailingOutput:
            def write(self, s):
                raise IOError("Write failed")

            def flush(self):
                pass

            def isatty(self):
                return False

        stream = SpectatorStream(enabled=True, output=FailingOutput(), format="plain")

        # Should not raise
        stream.emit(SpectatorEvents.PROPOSAL, agent="test")

    def test_emit_reraises_critical_errors(self):
        """Test emit re-raises critical errors like KeyboardInterrupt."""

        class InterruptOutput:
            def write(self, s):
                raise KeyboardInterrupt()

            def flush(self):
                pass

            def isatty(self):
                return False

        stream = SpectatorStream(enabled=True, output=InterruptOutput(), format="plain")

        with pytest.raises(KeyboardInterrupt):
            stream.emit(SpectatorEvents.PROPOSAL, agent="test")

    def test_truncate_short_text(self):
        """Test truncate doesn't modify short text."""
        stream = SpectatorStream(enabled=True, preview_length=80)
        result = stream._truncate("short text")
        assert result == "short text"

    def test_truncate_long_text(self):
        """Test truncate clips long text with ellipsis."""
        stream = SpectatorStream(enabled=True, preview_length=20)
        result = stream._truncate("this is a very long text that exceeds the limit")
        assert len(result) == 20
        assert result.endswith("...")

    def test_safe_print_unicode(self):
        """Test safe_print handles unicode properly."""
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output)
        stream._safe_print("Test 🎬 emoji")
        assert "emoji" in output.getvalue()

    def test_preview_disabled(self):
        """Test that preview can be disabled."""
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain", show_preview=False)

        long_text = "x" * 200
        stream.emit(SpectatorEvents.PROPOSAL, details=long_text)

        line = output.getvalue()
        assert long_text in line  # Full text, not truncated


class MockTTYOutput(io.StringIO):
    """Mock TTY output with configurable encoding."""

    def __init__(self, encoding="utf-8"):
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self):
        return self._encoding

    def isatty(self):
        return True


class TestCapabilityDetection:
    """Tests for terminal capability detection."""

    def test_no_color_env_var(self):
        """Test NO_COLOR environment variable is respected."""
        output = MockTTYOutput(encoding="utf-8")

        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            stream = SpectatorStream(enabled=True, output=output, format="auto")
            assert stream._use_color is False

    def test_dumb_terminal(self):
        """Test dumb terminal disables features."""
        output = MockTTYOutput(encoding="utf-8")

        with patch.dict(os.environ, {"TERM": "dumb"}, clear=False):
            stream = SpectatorStream(enabled=True, output=output, format="auto")
            assert stream._use_color is False
            assert stream._use_emoji is False

    def test_non_tty_output(self):
        """Test non-TTY output disables color."""
        output = io.StringIO()
        # StringIO.isatty() returns False by default

        stream = SpectatorStream(enabled=True, output=output, format="auto")
        assert stream._use_color is False


class TestValidEventTypes:
    """Tests for valid event type validation."""

    def test_valid_events_contains_main_types(self):
        """Test VALID_EVENT_TYPES contains expected events."""
        assert SpectatorEvents.DEBATE_START in VALID_EVENT_TYPES
        assert SpectatorEvents.DEBATE_END in VALID_EVENT_TYPES
        assert SpectatorEvents.PROPOSAL in VALID_EVENT_TYPES
        assert SpectatorEvents.CRITIQUE in VALID_EVENT_TYPES
        assert SpectatorEvents.CONSENSUS in VALID_EVENT_TYPES
        assert SpectatorEvents.ERROR in VALID_EVENT_TYPES

    def test_valid_events_is_frozen(self):
        """Test VALID_EVENT_TYPES is immutable."""
        assert isinstance(VALID_EVENT_TYPES, frozenset)
