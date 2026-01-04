"""
Tests for match_recorded WebSocket event loop_id inclusion.

Validates that the match_recorded event correctly includes loop_id
for client-side filtering in multi-loop leaderboard support.
"""

import pytest
from unittest.mock import Mock, MagicMock

from aragora.server.stream import (
    StreamEvent,
    StreamEventType,
    SyncEventEmitter,
)
from aragora.server.nomic_stream import create_nomic_hooks


class TestMatchRecordedEvent:
    """Test that match_recorded event includes loop_id field."""

    def test_match_recorded_includes_loop_id(self):
        """Verify that on_match_recorded emits event with loop_id."""
        # Create a mock emitter to capture events
        emitted_events = []
        mock_emitter = Mock(spec=SyncEventEmitter)
        mock_emitter.emit = lambda event: emitted_events.append(event)

        # Create hooks with the mock emitter
        hooks = create_nomic_hooks(mock_emitter)
        on_match_recorded = hooks["on_match_recorded"]

        # Call with all parameters including loop_id
        on_match_recorded(
            debate_id="cycle-42",
            participants=["claude", "gemini", "codex"],
            elo_changes={"claude": 15.5, "gemini": -7.2, "codex": -8.3},
            domain="architecture",
            winner="claude",
            loop_id="nomic-20260103-150000",
        )

        # Verify event was emitted
        assert len(emitted_events) == 1
        event = emitted_events[0]

        # Verify event type
        assert event.type == StreamEventType.MATCH_RECORDED

        # Verify all data fields including loop_id
        assert event.data["debate_id"] == "cycle-42"
        assert event.data["participants"] == ["claude", "gemini", "codex"]
        assert event.data["elo_changes"] == {"claude": 15.5, "gemini": -7.2, "codex": -8.3}
        assert event.data["domain"] == "architecture"
        assert event.data["winner"] == "claude"
        assert event.data["loop_id"] == "nomic-20260103-150000"

    def test_match_recorded_loop_id_optional(self):
        """Verify that loop_id is optional and defaults to None."""
        emitted_events = []
        mock_emitter = Mock(spec=SyncEventEmitter)
        mock_emitter.emit = lambda event: emitted_events.append(event)

        hooks = create_nomic_hooks(mock_emitter)
        on_match_recorded = hooks["on_match_recorded"]

        # Call without loop_id (backward compatibility)
        on_match_recorded(
            debate_id="cycle-1",
            participants=["agent1", "agent2"],
            elo_changes={"agent1": 10, "agent2": -10},
        )

        assert len(emitted_events) == 1
        event = emitted_events[0]

        # Verify loop_id is present but None
        assert "loop_id" in event.data
        assert event.data["loop_id"] is None

    def test_match_recorded_data_structure(self):
        """Verify the complete data structure of match_recorded event."""
        emitted_events = []
        mock_emitter = Mock(spec=SyncEventEmitter)
        mock_emitter.emit = lambda event: emitted_events.append(event)

        hooks = create_nomic_hooks(mock_emitter)
        on_match_recorded = hooks["on_match_recorded"]

        on_match_recorded(
            debate_id="test-debate",
            participants=["a", "b"],
            elo_changes={"a": 5, "b": -5},
            domain="testing",
            winner="a",
            loop_id="test-loop",
        )

        event = emitted_events[0]

        # Verify all expected keys are present
        expected_keys = {"debate_id", "participants", "elo_changes", "domain", "winner", "loop_id"}
        assert set(event.data.keys()) == expected_keys
