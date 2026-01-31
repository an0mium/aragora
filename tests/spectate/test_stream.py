"""Comprehensive tests for aragora.spectate.stream module.

Tests SpectatorStream class including initialization, capability detection,
emit methods (JSON and text), truncation, safe printing, error handling,
and environment-aware formatting.
"""

import io
import json
import logging
import os
import time
from unittest.mock import patch, MagicMock

import pytest

from aragora.spectate.events import EVENT_ASCII, EVENT_STYLES, SpectatorEvents
from aragora.spectate.stream import SpectatorStream, VALID_EVENT_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


class FailingWriteOutput:
    """Output that raises IOError on write."""

    def write(self, s):
        raise IOError("Simulated write failure")

    def flush(self):
        pass

    def isatty(self):
        return False


class UnicodeFailOutput:
    """Output that raises UnicodeEncodeError on first write then succeeds."""

    def __init__(self):
        self._call_count = 0
        self.written = []

    def write(self, s):
        self._call_count += 1
        if self._call_count == 1:
            raise UnicodeEncodeError("ascii", s, 0, 1, "mock")
        self.written.append(s)

    def flush(self):
        pass

    def isatty(self):
        return False


# ---------------------------------------------------------------------------
# VALID_EVENT_TYPES frozenset
# ---------------------------------------------------------------------------


class TestValidEventTypes:
    """Tests for the VALID_EVENT_TYPES frozenset."""

    def test_valid_event_types_is_frozenset(self):
        assert isinstance(VALID_EVENT_TYPES, frozenset)

    def test_contains_debate_start(self):
        assert SpectatorEvents.DEBATE_START in VALID_EVENT_TYPES

    def test_contains_debate_end(self):
        assert SpectatorEvents.DEBATE_END in VALID_EVENT_TYPES

    def test_contains_round_start(self):
        assert SpectatorEvents.ROUND_START in VALID_EVENT_TYPES

    def test_contains_round_end(self):
        assert SpectatorEvents.ROUND_END in VALID_EVENT_TYPES

    def test_contains_proposal(self):
        assert SpectatorEvents.PROPOSAL in VALID_EVENT_TYPES

    def test_contains_critique(self):
        assert SpectatorEvents.CRITIQUE in VALID_EVENT_TYPES

    def test_contains_refine(self):
        assert SpectatorEvents.REFINE in VALID_EVENT_TYPES

    def test_contains_vote(self):
        assert SpectatorEvents.VOTE in VALID_EVENT_TYPES

    def test_contains_judge(self):
        assert SpectatorEvents.JUDGE in VALID_EVENT_TYPES

    def test_contains_consensus(self):
        assert SpectatorEvents.CONSENSUS in VALID_EVENT_TYPES

    def test_contains_convergence(self):
        assert SpectatorEvents.CONVERGENCE in VALID_EVENT_TYPES

    def test_contains_converged(self):
        assert SpectatorEvents.CONVERGED in VALID_EVENT_TYPES

    def test_contains_memory_recall(self):
        assert SpectatorEvents.MEMORY_RECALL in VALID_EVENT_TYPES

    def test_contains_system(self):
        assert SpectatorEvents.SYSTEM in VALID_EVENT_TYPES

    def test_contains_error(self):
        assert SpectatorEvents.ERROR in VALID_EVENT_TYPES

    def test_does_not_contain_arbitrary_string(self):
        assert "not_an_event" not in VALID_EVENT_TYPES

    def test_does_not_contain_breakpoint(self):
        """Breakpoint events are not in VALID_EVENT_TYPES (intentional)."""
        assert SpectatorEvents.BREAKPOINT not in VALID_EVENT_TYPES

    def test_does_not_contain_breakpoint_resolved(self):
        assert SpectatorEvents.BREAKPOINT_RESOLVED not in VALID_EVENT_TYPES

    def test_immutable(self):
        """Frozenset cannot be mutated."""
        with pytest.raises(AttributeError):
            VALID_EVENT_TYPES.add("new_event")


# ---------------------------------------------------------------------------
# SpectatorStream initialization
# ---------------------------------------------------------------------------


class TestSpectatorStreamInit:
    """Tests for SpectatorStream dataclass initialization."""

    def test_default_disabled(self):
        stream = SpectatorStream()
        assert stream.enabled is False

    def test_default_format_is_auto(self):
        stream = SpectatorStream()
        assert stream.format == "auto"

    def test_default_show_preview(self):
        stream = SpectatorStream()
        assert stream.show_preview is True

    def test_default_preview_length(self):
        stream = SpectatorStream()
        assert stream.preview_length == 80

    def test_enabled_true(self):
        stream = SpectatorStream(enabled=True, format="plain")
        assert stream.enabled is True

    def test_custom_output(self):
        buf = io.StringIO()
        stream = SpectatorStream(enabled=True, output=buf, format="plain")
        assert stream.output is buf

    def test_custom_preview_length(self):
        stream = SpectatorStream(preview_length=120)
        assert stream.preview_length == 120

    def test_show_preview_false(self):
        stream = SpectatorStream(show_preview=False)
        assert stream.show_preview is False

    def test_disabled_stream_skips_detection(self):
        """When disabled, _use_color and _use_emoji stay False."""
        stream = SpectatorStream(enabled=False)
        assert stream._use_color is False
        assert stream._use_emoji is False


# ---------------------------------------------------------------------------
# __post_init__ format handling
# ---------------------------------------------------------------------------


class TestPostInit:
    """Tests for __post_init__ format branching."""

    def test_format_auto_calls_detect(self):
        """Auto format triggers capability detection."""
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="auto")
        # Non-TTY StringIO => no color
        assert stream._use_color is False

    def test_format_ansi_enables_both(self):
        stream = SpectatorStream(enabled=True, format="ansi")
        assert stream._use_color is True
        assert stream._use_emoji is True

    def test_format_plain_disables_both(self):
        stream = SpectatorStream(enabled=True, format="plain")
        assert stream._use_color is False
        assert stream._use_emoji is False

    def test_format_json_leaves_defaults(self):
        """JSON format does not set color/emoji flags."""
        stream = SpectatorStream(enabled=True, format="json")
        assert stream._use_color is False
        assert stream._use_emoji is False


# ---------------------------------------------------------------------------
# _detect_capabilities
# ---------------------------------------------------------------------------


class TestDetectCapabilities:
    """Tests for terminal capability detection."""

    def test_tty_with_utf8_enables_color_and_emoji(self):
        output = MockTTYOutput(encoding="utf-8")
        env = {"TERM": "xterm-256color"}
        with patch.dict(os.environ, env, clear=False):
            # Ensure NO_COLOR is not set
            os.environ.pop("NO_COLOR", None)
            stream = SpectatorStream(enabled=True, output=output, format="auto")
            assert stream._use_color is True
            assert stream._use_emoji is True

    def test_no_color_env_disables_color(self):
        output = MockTTYOutput(encoding="utf-8")
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            stream = SpectatorStream(enabled=True, output=output, format="auto")
            assert stream._use_color is False

    def test_no_color_empty_string_still_disables(self):
        """NO_COLOR="" should still disable color (presence check, not truthiness)."""
        output = MockTTYOutput(encoding="utf-8")
        with patch.dict(os.environ, {"NO_COLOR": ""}):
            stream = SpectatorStream(enabled=True, output=output, format="auto")
            assert stream._use_color is False

    def test_dumb_terminal_disables_everything(self):
        output = MockTTYOutput(encoding="utf-8")
        with patch.dict(os.environ, {"TERM": "dumb"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            stream = SpectatorStream(enabled=True, output=output, format="auto")
            assert stream._use_color is False
            assert stream._use_emoji is False

    def test_non_tty_disables_color(self):
        output = io.StringIO()  # isatty() returns False
        stream = SpectatorStream(enabled=True, output=output, format="auto")
        assert stream._use_color is False

    def test_non_utf8_encoding_disables_emoji(self):
        output = MockTTYOutput(encoding="ascii")
        with patch.dict(os.environ, {"TERM": "xterm"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            stream = SpectatorStream(enabled=True, output=output, format="auto")
            assert stream._use_emoji is False

    def test_utf16_encoding_enables_emoji(self):
        output = MockTTYOutput(encoding="utf-16")
        with patch.dict(os.environ, {"TERM": "xterm"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            stream = SpectatorStream(enabled=True, output=output, format="auto")
            assert stream._use_emoji is True

    def test_utf32_encoding_enables_emoji(self):
        output = MockTTYOutput(encoding="utf-32")
        with patch.dict(os.environ, {"TERM": "xterm"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            stream = SpectatorStream(enabled=True, output=output, format="auto")
            assert stream._use_emoji is True

    def test_none_encoding_treated_as_ascii(self):
        """If output.encoding is None, treat as ascii."""
        output = MockTTYOutput(encoding=None)
        # Override encoding property to return None
        type(output).encoding = property(lambda self: None)
        with patch.dict(os.environ, {"TERM": "xterm"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            stream = SpectatorStream(enabled=True, output=output, format="auto")
            assert stream._use_emoji is False

    def test_output_without_isatty(self):
        """Objects without isatty method are treated as non-TTY."""

        class NoIsatty:
            encoding = "utf-8"

            def write(self, s):
                pass

            def flush(self):
                pass

        stream = SpectatorStream(enabled=True, output=NoIsatty(), format="auto")
        assert stream._use_color is False


# ---------------------------------------------------------------------------
# emit method - disabled / guarding
# ---------------------------------------------------------------------------


class TestEmitGuarding:
    """Tests for emit early-return and error handling."""

    def test_emit_disabled_returns_immediately(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=False, output=output)
        stream.emit(SpectatorEvents.DEBATE_START, agent="test")
        assert output.getvalue() == ""

    def test_emit_unknown_event_logs_warning(self, caplog):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        with caplog.at_level(logging.WARNING, logger="aragora.spectate.stream"):
            stream.emit("totally_unknown", agent="test")
        assert "Unknown spectator event type" in caplog.text

    def test_emit_unknown_event_still_produces_output(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        stream.emit("totally_unknown", agent="test")
        assert len(output.getvalue()) > 0

    def test_emit_catches_io_error(self):
        """Non-critical IO errors are silently caught."""
        stream = SpectatorStream(enabled=True, output=FailingWriteOutput(), format="plain")
        # Should not raise
        stream.emit(SpectatorEvents.PROPOSAL, agent="test")

    def test_emit_reraises_keyboard_interrupt(self):
        class KBOutput:
            def write(self, s):
                raise KeyboardInterrupt()

            def flush(self):
                pass

            def isatty(self):
                return False

        stream = SpectatorStream(enabled=True, output=KBOutput(), format="plain")
        with pytest.raises(KeyboardInterrupt):
            stream.emit(SpectatorEvents.PROPOSAL, agent="test")

    def test_emit_reraises_system_exit(self):
        class ExitOutput:
            def write(self, s):
                raise SystemExit(1)

            def flush(self):
                pass

            def isatty(self):
                return False

        stream = SpectatorStream(enabled=True, output=ExitOutput(), format="plain")
        with pytest.raises(SystemExit):
            stream.emit(SpectatorEvents.PROPOSAL, agent="test")

    def test_emit_reraises_memory_error(self):
        class MemOutput:
            def write(self, s):
                raise MemoryError()

            def flush(self):
                pass

            def isatty(self):
                return False

        stream = SpectatorStream(enabled=True, output=MemOutput(), format="plain")
        with pytest.raises(MemoryError):
            stream.emit(SpectatorEvents.PROPOSAL, agent="test")

    def test_emit_reraises_recursion_error(self):
        class RecOutput:
            def write(self, s):
                raise RecursionError()

            def flush(self):
                pass

            def isatty(self):
                return False

        stream = SpectatorStream(enabled=True, output=RecOutput(), format="plain")
        with pytest.raises(RecursionError):
            stream.emit(SpectatorEvents.PROPOSAL, agent="test")

    def test_emit_catches_value_error(self):
        """ValueError is non-critical and should be caught."""

        class ValOutput:
            def write(self, s):
                raise ValueError("test error")

            def flush(self):
                pass

            def isatty(self):
                return False

        stream = SpectatorStream(enabled=True, output=ValOutput(), format="plain")
        # Should not raise
        stream.emit(SpectatorEvents.PROPOSAL, agent="test")


# ---------------------------------------------------------------------------
# emit - JSON format
# ---------------------------------------------------------------------------


class TestEmitJson:
    """Tests for JSON-formatted event output."""

    def _emit_json(self, **kwargs):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="json")
        stream.emit(**kwargs)
        return json.loads(output.getvalue().strip())

    def test_json_has_type(self):
        data = self._emit_json(event_type=SpectatorEvents.PROPOSAL)
        assert data["type"] == "proposal"

    def test_json_has_timestamp(self):
        before = time.time()
        data = self._emit_json(event_type=SpectatorEvents.PROPOSAL)
        after = time.time()
        assert before <= data["timestamp"] <= after

    def test_json_agent_present(self):
        data = self._emit_json(event_type=SpectatorEvents.PROPOSAL, agent="claude")
        assert data["agent"] == "claude"

    def test_json_agent_empty_becomes_none(self):
        data = self._emit_json(event_type=SpectatorEvents.PROPOSAL, agent="")
        assert data["agent"] is None

    def test_json_agent_default_is_none(self):
        data = self._emit_json(event_type=SpectatorEvents.PROPOSAL)
        assert data["agent"] is None

    def test_json_details_present(self):
        data = self._emit_json(event_type=SpectatorEvents.PROPOSAL, details="hello world")
        assert data["details"] == "hello world"

    def test_json_details_empty_becomes_none(self):
        data = self._emit_json(event_type=SpectatorEvents.PROPOSAL, details="")
        assert data["details"] is None

    def test_json_metric_float(self):
        data = self._emit_json(event_type=SpectatorEvents.CONSENSUS, metric=0.95)
        assert data["metric"] == 0.95

    def test_json_metric_int(self):
        data = self._emit_json(event_type=SpectatorEvents.CONSENSUS, metric=3)
        assert data["metric"] == 3

    def test_json_metric_none_default(self):
        data = self._emit_json(event_type=SpectatorEvents.PROPOSAL)
        assert data["metric"] is None

    def test_json_round_number(self):
        data = self._emit_json(event_type=SpectatorEvents.ROUND_START, round_number=5)
        assert data["round"] == 5

    def test_json_round_none_default(self):
        data = self._emit_json(event_type=SpectatorEvents.PROPOSAL)
        assert data["round"] is None

    def test_json_full_event(self):
        data = self._emit_json(
            event_type=SpectatorEvents.CRITIQUE,
            agent="gpt-4",
            details="weak argument",
            metric=0.42,
            round_number=2,
        )
        assert data["type"] == "critique"
        assert data["agent"] == "gpt-4"
        assert data["details"] == "weak argument"
        assert data["metric"] == 0.42
        assert data["round"] == 2

    def test_json_is_valid_json_line(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="json")
        stream.emit(SpectatorEvents.DEBATE_START)
        line = output.getvalue().strip()
        # Should not raise
        json.loads(line)

    def test_json_special_chars_in_details(self):
        data = self._emit_json(
            event_type=SpectatorEvents.PROPOSAL,
            details='quote "test" and\nnewline',
        )
        assert data["details"] == 'quote "test" and\nnewline'


# ---------------------------------------------------------------------------
# emit - text format (plain)
# ---------------------------------------------------------------------------


class TestEmitTextPlain:
    """Tests for plain text-formatted event output."""

    def _emit_plain(self, **kwargs):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        stream.emit(**kwargs)
        return output.getvalue()

    def test_plain_contains_timestamp(self):
        line = self._emit_plain(event_type=SpectatorEvents.PROPOSAL)
        # Timestamp format is HH:MM:SS in brackets
        assert "[" in line and "]" in line

    def test_plain_contains_ascii_icon(self):
        line = self._emit_plain(event_type=SpectatorEvents.PROPOSAL)
        assert "[PROPOSE]" in line

    def test_plain_contains_agent(self):
        line = self._emit_plain(event_type=SpectatorEvents.PROPOSAL, agent="claude")
        assert "claude" in line

    def test_plain_no_agent_no_crash(self):
        line = self._emit_plain(event_type=SpectatorEvents.PROPOSAL)
        assert len(line) > 0

    def test_plain_contains_details(self):
        line = self._emit_plain(event_type=SpectatorEvents.PROPOSAL, details="my proposal text")
        assert "my proposal text" in line

    def test_plain_round_number(self):
        line = self._emit_plain(event_type=SpectatorEvents.ROUND_START, round_number=3)
        assert "R3" in line

    def test_plain_no_round_number(self):
        line = self._emit_plain(event_type=SpectatorEvents.PROPOSAL)
        # Should not contain "R" followed by digit pattern
        assert "R0" not in line  # No default round

    def test_plain_metric_float(self):
        line = self._emit_plain(event_type=SpectatorEvents.CONSENSUS, metric=0.85)
        assert "(0.85)" in line

    def test_plain_metric_int(self):
        line = self._emit_plain(event_type=SpectatorEvents.CONSENSUS, metric=3)
        assert "(3)" in line

    def test_plain_no_ansi_codes(self):
        line = self._emit_plain(event_type=SpectatorEvents.PROPOSAL, agent="test", details="data")
        assert "\033[" not in line

    def test_plain_unknown_event_uses_uppercase_fallback(self):
        """Unknown events use [EVENT_TYPE.upper()] format."""
        line = self._emit_plain(event_type="custom_event")
        assert "[CUSTOM_EVENT]" in line

    def test_plain_all_known_events_produce_output(self):
        """Every valid event type produces non-empty output."""
        for event_type in VALID_EVENT_TYPES:
            line = self._emit_plain(event_type=event_type)
            assert len(line.strip()) > 0, f"No output for event {event_type}"


# ---------------------------------------------------------------------------
# emit - text format (ANSI)
# ---------------------------------------------------------------------------


class TestEmitTextAnsi:
    """Tests for ANSI-colored text output."""

    def _emit_ansi(self, **kwargs):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="ansi")
        stream.emit(**kwargs)
        return output.getvalue()

    def test_ansi_contains_escape_codes(self):
        line = self._emit_ansi(event_type=SpectatorEvents.PROPOSAL, agent="test")
        assert "\033[" in line

    def test_ansi_contains_reset_code(self):
        line = self._emit_ansi(event_type=SpectatorEvents.PROPOSAL)
        assert "\033[0m" in line

    def test_ansi_contains_bold_for_agent(self):
        line = self._emit_ansi(event_type=SpectatorEvents.PROPOSAL, agent="claude")
        assert "\033[1m" in line  # BOLD

    def test_ansi_contains_dim_for_timestamp(self):
        line = self._emit_ansi(event_type=SpectatorEvents.PROPOSAL)
        assert "\033[2m" in line  # DIM

    def test_ansi_emoji_icon_used(self):
        """ANSI format uses emoji icons, not ASCII."""
        line = self._emit_ansi(event_type=SpectatorEvents.PROPOSAL)
        # Should NOT contain ASCII fallback
        assert "[PROPOSE]" not in line

    def test_ansi_metric_has_dim(self):
        line = self._emit_ansi(event_type=SpectatorEvents.CONSENSUS, metric=0.9)
        assert "\033[2m" in line  # DIM for metric
        assert "0.90" in line


# ---------------------------------------------------------------------------
# _truncate method
# ---------------------------------------------------------------------------


class TestTruncate:
    """Tests for the _truncate helper method."""

    def test_short_text_unchanged(self):
        stream = SpectatorStream(preview_length=80)
        assert stream._truncate("hello") == "hello"

    def test_exact_length_unchanged(self):
        stream = SpectatorStream(preview_length=5)
        assert stream._truncate("hello") == "hello"

    def test_long_text_truncated_with_ellipsis(self):
        stream = SpectatorStream(preview_length=10)
        result = stream._truncate("this is a long sentence")
        assert len(result) == 10
        assert result.endswith("...")

    def test_truncated_preserves_prefix(self):
        stream = SpectatorStream(preview_length=10)
        result = stream._truncate("abcdefghijklmnop")
        assert result == "abcdefg..."

    def test_empty_string(self):
        stream = SpectatorStream(preview_length=80)
        assert stream._truncate("") == ""

    def test_preview_length_one(self):
        """Edge case: preview_length of 1 would yield negative slice."""
        stream = SpectatorStream(preview_length=1)
        result = stream._truncate("ab")
        # preview_length - 3 = -2, so text[:-2] + "..."
        # This tests the behavior, regardless of whether it's ideal
        assert isinstance(result, str)

    def test_preview_length_three(self):
        stream = SpectatorStream(preview_length=3)
        result = stream._truncate("abcdef")
        assert result == "..."

    def test_preview_length_four(self):
        stream = SpectatorStream(preview_length=4)
        result = stream._truncate("abcdef")
        assert result == "a..."
        assert len(result) == 4

    def test_default_preview_length(self):
        stream = SpectatorStream()
        text_79 = "x" * 79
        assert stream._truncate(text_79) == text_79

    def test_default_truncates_at_81(self):
        stream = SpectatorStream()
        text_81 = "x" * 81
        result = stream._truncate(text_81)
        assert len(result) == 80
        assert result.endswith("...")


# ---------------------------------------------------------------------------
# _safe_print method
# ---------------------------------------------------------------------------


class TestSafePrint:
    """Tests for the _safe_print method."""

    def test_prints_to_output(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output)
        stream._safe_print("hello world")
        assert "hello world" in output.getvalue()

    def test_prints_newline(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output)
        stream._safe_print("line1")
        assert output.getvalue().endswith("\n")

    def test_unicode_content(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output)
        stream._safe_print("emoji test: cafe\u0301")
        assert "cafe" in output.getvalue()

    def test_unicode_encode_error_fallback(self):
        """When UnicodeEncodeError occurs, fallback to ASCII-safe version."""
        output = UnicodeFailOutput()
        stream = SpectatorStream(enabled=True, output=output)
        # UnicodeFailOutput raises on first write, succeeds on second
        stream._safe_print("test \u2603 snowman")
        # Second write should have occurred with ASCII-safe version
        assert len(output.written) > 0

    def test_flush_called(self):
        """Output is flushed after printing."""
        mock_output = MagicMock()
        mock_output.isatty.return_value = False
        stream = SpectatorStream(enabled=True, output=mock_output, format="plain")
        stream._safe_print("test")
        mock_output.flush.assert_called()


# ---------------------------------------------------------------------------
# show_preview toggle
# ---------------------------------------------------------------------------


class TestShowPreview:
    """Tests for the show_preview parameter."""

    def test_preview_enabled_truncates_long_details(self):
        output = io.StringIO()
        stream = SpectatorStream(
            enabled=True,
            output=output,
            format="plain",
            show_preview=True,
            preview_length=20,
        )
        long_text = "a" * 100
        stream.emit(SpectatorEvents.PROPOSAL, details=long_text)
        line = output.getvalue()
        assert "..." in line
        assert long_text not in line

    def test_preview_disabled_shows_full_details(self):
        output = io.StringIO()
        stream = SpectatorStream(
            enabled=True,
            output=output,
            format="plain",
            show_preview=False,
            preview_length=20,
        )
        long_text = "a" * 100
        stream.emit(SpectatorEvents.PROPOSAL, details=long_text)
        line = output.getvalue()
        assert long_text in line

    def test_preview_enabled_short_text_not_truncated(self):
        output = io.StringIO()
        stream = SpectatorStream(
            enabled=True,
            output=output,
            format="plain",
            show_preview=True,
            preview_length=200,
        )
        short_text = "short"
        stream.emit(SpectatorEvents.PROPOSAL, details=short_text)
        line = output.getvalue()
        assert short_text in line
        assert "..." not in line


# ---------------------------------------------------------------------------
# emit with empty / missing / None parameters
# ---------------------------------------------------------------------------


class TestEmitEdgeCases:
    """Tests for edge cases in emit parameter handling."""

    def test_emit_empty_agent(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        stream.emit(SpectatorEvents.PROPOSAL, agent="")
        # Should produce output without agent portion
        assert len(output.getvalue()) > 0

    def test_emit_empty_details(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        stream.emit(SpectatorEvents.PROPOSAL, details="")
        assert len(output.getvalue()) > 0

    def test_emit_zero_metric(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        stream.emit(SpectatorEvents.CONSENSUS, metric=0.0)
        assert "(0.00)" in output.getvalue()

    def test_emit_negative_metric(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        stream.emit(SpectatorEvents.CONSENSUS, metric=-1.5)
        assert "(-1.50)" in output.getvalue()

    def test_emit_large_metric(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        stream.emit(SpectatorEvents.CONSENSUS, metric=99999.99)
        assert "(99999.99)" in output.getvalue()

    def test_emit_round_zero(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        stream.emit(SpectatorEvents.ROUND_START, round_number=0)
        assert "R0" in output.getvalue()

    def test_emit_negative_round(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        stream.emit(SpectatorEvents.ROUND_START, round_number=-1)
        assert "R-1" in output.getvalue()

    def test_emit_all_params_empty_or_none(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        stream.emit(SpectatorEvents.SYSTEM)
        assert len(output.getvalue().strip()) > 0

    def test_emit_json_with_no_optional_params(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="json")
        stream.emit(SpectatorEvents.SYSTEM)
        data = json.loads(output.getvalue().strip())
        assert data["type"] == "system"
        assert data["agent"] is None
        assert data["details"] is None
        assert data["metric"] is None
        assert data["round"] is None


# ---------------------------------------------------------------------------
# Integration: multiple events
# ---------------------------------------------------------------------------


class TestMultipleEmits:
    """Tests for emitting multiple events in sequence."""

    def test_multiple_events_produce_multiple_lines(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="plain")
        stream.emit(SpectatorEvents.DEBATE_START)
        stream.emit(SpectatorEvents.ROUND_START, round_number=1)
        stream.emit(SpectatorEvents.PROPOSAL, agent="claude")
        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 3

    def test_multiple_json_events_are_separate_lines(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="json")
        stream.emit(SpectatorEvents.DEBATE_START)
        stream.emit(SpectatorEvents.DEBATE_END)
        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # Each line should be valid JSON

    def test_json_timestamps_are_monotonically_increasing(self):
        output = io.StringIO()
        stream = SpectatorStream(enabled=True, output=output, format="json")
        stream.emit(SpectatorEvents.DEBATE_START)
        stream.emit(SpectatorEvents.DEBATE_END)
        lines = output.getvalue().strip().split("\n")
        t1 = json.loads(lines[0])["timestamp"]
        t2 = json.loads(lines[1])["timestamp"]
        assert t2 >= t1


# ---------------------------------------------------------------------------
# Module-level import
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Tests for module-level imports and exports."""

    def test_import_spectator_stream(self):
        from aragora.spectate.stream import SpectatorStream as SS

        assert SS is SpectatorStream

    def test_import_valid_event_types(self):
        from aragora.spectate.stream import VALID_EVENT_TYPES as VET

        assert VET is VALID_EVENT_TYPES

    def test_import_from_package(self):
        from aragora.spectate import SpectatorStream as SS

        assert SS is SpectatorStream

    def test_import_events_from_package(self):
        from aragora.spectate import SpectatorEvents as SE

        assert SE is SpectatorEvents
