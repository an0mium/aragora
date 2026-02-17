"""Comprehensive tests for aragora.spectate.events module.

Tests all event classes, constants, style mappings, and ASCII fallback
configurations for the Spectator Mode event system.
"""

import pytest

from aragora.spectate.events import (
    EVENT_ASCII,
    EVENT_STYLES,
    SpectatorEvents,
)


# ---------------------------------------------------------------------------
# SpectatorEvents class constants
# ---------------------------------------------------------------------------


class TestSpectatorEventsDebateLifecycle:
    """Tests for debate lifecycle event constants."""

    def test_debate_start_value(self):
        assert SpectatorEvents.DEBATE_START == "debate_start"

    def test_debate_end_value(self):
        assert SpectatorEvents.DEBATE_END == "debate_end"

    def test_debate_start_is_string(self):
        assert isinstance(SpectatorEvents.DEBATE_START, str)

    def test_debate_end_is_string(self):
        assert isinstance(SpectatorEvents.DEBATE_END, str)

    def test_debate_start_and_end_differ(self):
        assert SpectatorEvents.DEBATE_START != SpectatorEvents.DEBATE_END


class TestSpectatorEventsRoundLifecycle:
    """Tests for round lifecycle event constants."""

    def test_round_start_value(self):
        assert SpectatorEvents.ROUND_START == "round_start"

    def test_round_end_value(self):
        assert SpectatorEvents.ROUND_END == "round_end"

    def test_round_start_and_end_differ(self):
        assert SpectatorEvents.ROUND_START != SpectatorEvents.ROUND_END


class TestSpectatorEventsAgentActions:
    """Tests for agent action event constants."""

    def test_proposal_value(self):
        assert SpectatorEvents.PROPOSAL == "proposal"

    def test_critique_value(self):
        assert SpectatorEvents.CRITIQUE == "critique"

    def test_refine_value(self):
        assert SpectatorEvents.REFINE == "refine"

    def test_vote_value(self):
        assert SpectatorEvents.VOTE == "vote"

    def test_judge_value(self):
        assert SpectatorEvents.JUDGE == "judge"

    def test_all_agent_actions_unique(self):
        actions = [
            SpectatorEvents.PROPOSAL,
            SpectatorEvents.CRITIQUE,
            SpectatorEvents.REFINE,
            SpectatorEvents.VOTE,
            SpectatorEvents.JUDGE,
        ]
        assert len(actions) == len(set(actions))


class TestSpectatorEventsConsensus:
    """Tests for consensus and convergence event constants."""

    def test_consensus_value(self):
        assert SpectatorEvents.CONSENSUS == "consensus"

    def test_convergence_value(self):
        assert SpectatorEvents.CONVERGENCE == "convergence"

    def test_converged_value(self):
        assert SpectatorEvents.CONVERGED == "converged"

    def test_consensus_events_all_unique(self):
        events = [
            SpectatorEvents.CONSENSUS,
            SpectatorEvents.CONVERGENCE,
            SpectatorEvents.CONVERGED,
        ]
        assert len(events) == len(set(events))


class TestSpectatorEventsMemory:
    """Tests for memory-related event constants."""

    def test_memory_recall_value(self):
        assert SpectatorEvents.MEMORY_RECALL == "memory_recall"

    def test_memory_recall_is_string(self):
        assert isinstance(SpectatorEvents.MEMORY_RECALL, str)


class TestSpectatorEventsBreakpoints:
    """Tests for human-in-the-loop breakpoint event constants."""

    def test_breakpoint_value(self):
        assert SpectatorEvents.BREAKPOINT == "breakpoint"

    def test_breakpoint_resolved_value(self):
        assert SpectatorEvents.BREAKPOINT_RESOLVED == "breakpoint_resolved"

    def test_breakpoint_and_resolved_differ(self):
        assert SpectatorEvents.BREAKPOINT != SpectatorEvents.BREAKPOINT_RESOLVED


class TestSpectatorEventsSystem:
    """Tests for system event constants."""

    def test_system_value(self):
        assert SpectatorEvents.SYSTEM == "system"

    def test_error_value(self):
        assert SpectatorEvents.ERROR == "error"

    def test_system_and_error_differ(self):
        assert SpectatorEvents.SYSTEM != SpectatorEvents.ERROR


class TestSpectatorEventsAllUnique:
    """Verify no duplicate event values exist across all event types."""

    def _all_event_values(self):
        """Collect all public event values from SpectatorEvents."""
        return [
            v
            for k, v in vars(SpectatorEvents).items()
            if not k.startswith("_") and isinstance(v, str)
        ]

    def test_all_event_values_are_unique(self):
        values = self._all_event_values()
        assert len(values) == len(set(values)), "Duplicate event values detected"

    def test_all_event_values_are_lowercase(self):
        for val in self._all_event_values():
            assert val == val.lower(), f"Event value {val!r} is not lowercase"

    def test_all_event_values_use_underscores(self):
        """Event values should use underscores not hyphens or spaces."""
        for val in self._all_event_values():
            assert " " not in val, f"Event value {val!r} contains space"
            assert "-" not in val, f"Event value {val!r} contains hyphen"

    def test_all_event_values_are_nonempty(self):
        for val in self._all_event_values():
            assert len(val) > 0, "Found empty event value"

    def test_expected_event_count(self):
        """There should be exactly 18 event constants (including EARLY_STOP)."""
        values = self._all_event_values()
        assert len(values) == 18


# ---------------------------------------------------------------------------
# EVENT_STYLES dictionary
# ---------------------------------------------------------------------------


class TestEventStylesCompleteness:
    """Verify all events have entries in EVENT_STYLES."""

    def test_debate_start_in_styles(self):
        assert SpectatorEvents.DEBATE_START in EVENT_STYLES

    def test_debate_end_in_styles(self):
        assert SpectatorEvents.DEBATE_END in EVENT_STYLES

    def test_round_start_in_styles(self):
        assert SpectatorEvents.ROUND_START in EVENT_STYLES

    def test_round_end_in_styles(self):
        assert SpectatorEvents.ROUND_END in EVENT_STYLES

    def test_proposal_in_styles(self):
        assert SpectatorEvents.PROPOSAL in EVENT_STYLES

    def test_critique_in_styles(self):
        assert SpectatorEvents.CRITIQUE in EVENT_STYLES

    def test_refine_in_styles(self):
        assert SpectatorEvents.REFINE in EVENT_STYLES

    def test_vote_in_styles(self):
        assert SpectatorEvents.VOTE in EVENT_STYLES

    def test_judge_in_styles(self):
        assert SpectatorEvents.JUDGE in EVENT_STYLES

    def test_consensus_in_styles(self):
        assert SpectatorEvents.CONSENSUS in EVENT_STYLES

    def test_convergence_in_styles(self):
        assert SpectatorEvents.CONVERGENCE in EVENT_STYLES

    def test_converged_in_styles(self):
        assert SpectatorEvents.CONVERGED in EVENT_STYLES

    def test_memory_recall_in_styles(self):
        assert SpectatorEvents.MEMORY_RECALL in EVENT_STYLES

    def test_breakpoint_in_styles(self):
        assert SpectatorEvents.BREAKPOINT in EVENT_STYLES

    def test_breakpoint_resolved_in_styles(self):
        assert SpectatorEvents.BREAKPOINT_RESOLVED in EVENT_STYLES

    def test_system_in_styles(self):
        assert SpectatorEvents.SYSTEM in EVENT_STYLES

    def test_error_in_styles(self):
        assert SpectatorEvents.ERROR in EVENT_STYLES


class TestEventStylesFormat:
    """Verify the structure and content of each style entry."""

    def test_styles_is_dict(self):
        assert isinstance(EVENT_STYLES, dict)

    def test_all_styles_are_tuples(self):
        for key, value in EVENT_STYLES.items():
            assert isinstance(value, tuple), f"Style for {key!r} should be tuple"

    def test_all_styles_have_two_elements(self):
        for key, value in EVENT_STYLES.items():
            assert len(value) == 2, f"Style for {key!r} should have 2 elements"

    def test_all_icons_are_strings(self):
        for key, (icon, _) in EVENT_STYLES.items():
            assert isinstance(icon, str), f"Icon for {key!r} should be string"

    def test_all_icons_are_nonempty(self):
        for key, (icon, _) in EVENT_STYLES.items():
            assert len(icon) > 0, f"Icon for {key!r} should not be empty"

    def test_all_color_codes_are_strings(self):
        for key, (_, color) in EVENT_STYLES.items():
            assert isinstance(color, str), f"Color for {key!r} should be string"

    def test_all_color_codes_start_with_escape(self):
        """Color codes should be ANSI escape sequences."""
        for key, (_, color) in EVENT_STYLES.items():
            assert color.startswith("\033["), f"Color for {key!r} should start with ANSI escape"

    def test_specific_icon_debate_start(self):
        icon, _ = EVENT_STYLES[SpectatorEvents.DEBATE_START]
        assert icon is not None and len(icon) > 0

    def test_specific_icon_error(self):
        icon, _ = EVENT_STYLES[SpectatorEvents.ERROR]
        assert icon is not None and len(icon) > 0

    def test_styles_count_matches_ascii_count(self):
        """EVENT_STYLES and EVENT_ASCII should have same number of entries."""
        assert len(EVENT_STYLES) == len(EVENT_ASCII)

    def test_styles_keys_match_ascii_keys(self):
        """Both dictionaries should cover the same events."""
        assert set(EVENT_STYLES.keys()) == set(EVENT_ASCII.keys())


class TestEventStylesColors:
    """Verify specific color assignments align with semantic grouping."""

    def test_debate_start_and_end_same_color(self):
        _, c1 = EVENT_STYLES[SpectatorEvents.DEBATE_START]
        _, c2 = EVENT_STYLES[SpectatorEvents.DEBATE_END]
        assert c1 == c2

    def test_round_start_and_end_same_color(self):
        _, c1 = EVENT_STYLES[SpectatorEvents.ROUND_START]
        _, c2 = EVENT_STYLES[SpectatorEvents.ROUND_END]
        assert c1 == c2

    def test_consensus_events_share_color(self):
        """Consensus, convergence, and converged should share green color."""
        _, c1 = EVENT_STYLES[SpectatorEvents.CONSENSUS]
        _, c2 = EVENT_STYLES[SpectatorEvents.CONVERGENCE]
        _, c3 = EVENT_STYLES[SpectatorEvents.CONVERGED]
        assert c1 == c2 == c3

    def test_error_uses_red(self):
        """Error events should use red ANSI color."""
        _, color = EVENT_STYLES[SpectatorEvents.ERROR]
        assert color == "\033[91m"

    def test_critique_uses_red(self):
        _, color = EVENT_STYLES[SpectatorEvents.CRITIQUE]
        assert color == "\033[91m"


# ---------------------------------------------------------------------------
# EVENT_ASCII dictionary
# ---------------------------------------------------------------------------


class TestEventAsciiCompleteness:
    """Verify all events have entries in EVENT_ASCII."""

    def test_debate_start_in_ascii(self):
        assert SpectatorEvents.DEBATE_START in EVENT_ASCII

    def test_debate_end_in_ascii(self):
        assert SpectatorEvents.DEBATE_END in EVENT_ASCII

    def test_round_start_in_ascii(self):
        assert SpectatorEvents.ROUND_START in EVENT_ASCII

    def test_round_end_in_ascii(self):
        assert SpectatorEvents.ROUND_END in EVENT_ASCII

    def test_proposal_in_ascii(self):
        assert SpectatorEvents.PROPOSAL in EVENT_ASCII

    def test_critique_in_ascii(self):
        assert SpectatorEvents.CRITIQUE in EVENT_ASCII

    def test_refine_in_ascii(self):
        assert SpectatorEvents.REFINE in EVENT_ASCII

    def test_vote_in_ascii(self):
        assert SpectatorEvents.VOTE in EVENT_ASCII

    def test_judge_in_ascii(self):
        assert SpectatorEvents.JUDGE in EVENT_ASCII

    def test_consensus_in_ascii(self):
        assert SpectatorEvents.CONSENSUS in EVENT_ASCII

    def test_convergence_in_ascii(self):
        assert SpectatorEvents.CONVERGENCE in EVENT_ASCII

    def test_converged_in_ascii(self):
        assert SpectatorEvents.CONVERGED in EVENT_ASCII

    def test_memory_recall_in_ascii(self):
        assert SpectatorEvents.MEMORY_RECALL in EVENT_ASCII

    def test_breakpoint_in_ascii(self):
        assert SpectatorEvents.BREAKPOINT in EVENT_ASCII

    def test_breakpoint_resolved_in_ascii(self):
        assert SpectatorEvents.BREAKPOINT_RESOLVED in EVENT_ASCII

    def test_system_in_ascii(self):
        assert SpectatorEvents.SYSTEM in EVENT_ASCII

    def test_error_in_ascii(self):
        assert SpectatorEvents.ERROR in EVENT_ASCII


class TestEventAsciiFormat:
    """Verify the structure of ASCII fallback strings."""

    def test_ascii_is_dict(self):
        assert isinstance(EVENT_ASCII, dict)

    def test_all_ascii_values_are_strings(self):
        for key, val in EVENT_ASCII.items():
            assert isinstance(val, str), f"ASCII for {key!r} should be string"

    def test_all_ascii_values_start_with_bracket(self):
        for key, val in EVENT_ASCII.items():
            assert val.startswith("["), f"ASCII for {key!r} should start with ["

    def test_all_ascii_values_end_with_bracket(self):
        for key, val in EVENT_ASCII.items():
            assert val.endswith("]"), f"ASCII for {key!r} should end with ]"

    def test_all_ascii_values_are_nonempty_content(self):
        """Content inside brackets should not be empty."""
        for key, val in EVENT_ASCII.items():
            inner = val[1:-1]
            assert len(inner) > 0, f"ASCII content for {key!r} should not be empty"

    def test_all_ascii_values_are_pure_ascii(self):
        """ASCII fallbacks should only contain ASCII characters."""
        for key, val in EVENT_ASCII.items():
            try:
                val.encode("ascii")
            except UnicodeEncodeError:
                pytest.fail(f"ASCII fallback for {key!r} contains non-ASCII characters")

    def test_all_ascii_values_are_uppercase_content(self):
        """Content inside brackets should be uppercase."""
        for key, val in EVENT_ASCII.items():
            inner = val[1:-1]
            # Allow / character for closing tags like [/ROUND]
            stripped = inner.replace("/", "")
            assert stripped == stripped.upper(), (
                f"ASCII content for {key!r} should be uppercase, got {val!r}"
            )


class TestEventAsciiSpecificValues:
    """Verify specific ASCII fallback values."""

    def test_debate_start_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.DEBATE_START] == "[START]"

    def test_debate_end_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.DEBATE_END] == "[END]"

    def test_round_start_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.ROUND_START] == "[ROUND]"

    def test_round_end_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.ROUND_END] == "[/ROUND]"

    def test_proposal_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.PROPOSAL] == "[PROPOSE]"

    def test_critique_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.CRITIQUE] == "[CRITIQUE]"

    def test_refine_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.REFINE] == "[REFINE]"

    def test_vote_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.VOTE] == "[VOTE]"

    def test_judge_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.JUDGE] == "[JUDGE]"

    def test_consensus_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.CONSENSUS] == "[CONSENSUS]"

    def test_convergence_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.CONVERGENCE] == "[CONVERGE]"

    def test_converged_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.CONVERGED] == "[DONE]"

    def test_memory_recall_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.MEMORY_RECALL] == "[MEMORY]"

    def test_breakpoint_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.BREAKPOINT] == "[BREAK]"

    def test_breakpoint_resolved_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.BREAKPOINT_RESOLVED] == "[RESOLVED]"

    def test_system_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.SYSTEM] == "[SYS]"

    def test_error_ascii(self):
        assert EVENT_ASCII[SpectatorEvents.ERROR] == "[ERR]"


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------


class TestCrossModuleConsistency:
    """Ensure events, styles, and ASCII mappings are consistent."""

    def test_every_styled_event_has_ascii(self):
        for key in EVENT_STYLES:
            assert key in EVENT_ASCII, f"Styled event {key!r} missing ASCII fallback"

    def test_every_ascii_event_has_style(self):
        for key in EVENT_ASCII:
            assert key in EVENT_STYLES, f"ASCII event {key!r} missing style entry"

    def test_no_extra_keys_in_styles(self):
        """No keys in styles that are not defined as SpectatorEvents attributes."""
        all_events = {
            v
            for k, v in vars(SpectatorEvents).items()
            if not k.startswith("_") and isinstance(v, str)
        }
        for key in EVENT_STYLES:
            assert key in all_events, f"EVENT_STYLES key {key!r} not in SpectatorEvents"

    def test_no_extra_keys_in_ascii(self):
        """No keys in ASCII that are not defined as SpectatorEvents attributes."""
        all_events = {
            v
            for k, v in vars(SpectatorEvents).items()
            if not k.startswith("_") and isinstance(v, str)
        }
        for key in EVENT_ASCII:
            assert key in all_events, f"EVENT_ASCII key {key!r} not in SpectatorEvents"

    def test_spectator_events_class_has_no_instance_methods(self):
        """SpectatorEvents should be a pure constants class with no methods."""
        methods = [
            k
            for k in dir(SpectatorEvents)
            if not k.startswith("_") and callable(getattr(SpectatorEvents, k))
        ]
        assert methods == [], f"SpectatorEvents should have no methods, found: {methods}"

    def test_spectator_events_instantiation(self):
        """SpectatorEvents can be instantiated (pure data class)."""
        obj = SpectatorEvents()
        assert obj.DEBATE_START == "debate_start"
