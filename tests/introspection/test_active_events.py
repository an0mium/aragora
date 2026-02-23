"""Tests for ActiveIntrospectionTracker event emissions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.introspection.active import (
    ActiveIntrospectionTracker,
    RoundMetrics,
    IntrospectionGoals,
)


class TestIntrospectionEvents:
    """Tests for _emit_introspection_event."""

    def test_emits_event_on_round_update(self) -> None:
        tracker = ActiveIntrospectionTracker()
        metrics = RoundMetrics(
            round_number=1,
            proposals_made=2,
            proposals_accepted=1,
            critiques_given=3,
            critiques_led_to_changes=2,
            argument_influence=0.7,
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            tracker.update_round("claude", metrics)

        mock_dispatch.assert_called_once()
        data = mock_dispatch.call_args[0][1]
        assert data["agent"] == "claude"
        assert data["round"] == 1
        assert data["acceptance_rate"] == 0.5
        assert data["total_proposals"] == 2

    def test_accumulates_across_rounds(self) -> None:
        tracker = ActiveIntrospectionTracker()

        events = []
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=lambda name, data: events.append(data),
        ):
            tracker.update_round(
                "claude",
                RoundMetrics(
                    round_number=1,
                    proposals_made=1,
                    proposals_accepted=1,
                    critiques_given=2,
                    critiques_led_to_changes=1,
                ),
            )
            tracker.update_round(
                "claude",
                RoundMetrics(
                    round_number=2,
                    proposals_made=1,
                    proposals_accepted=0,
                    critiques_given=1,
                    critiques_led_to_changes=0,
                ),
            )

        assert len(events) == 2
        # After round 2: 2 proposals, 1 accepted = 50%
        assert events[1]["total_proposals"] == 2
        assert events[1]["acceptance_rate"] == 0.5
        assert events[1]["rounds_completed"] == 2

    def test_multiple_agents_independent(self) -> None:
        tracker = ActiveIntrospectionTracker()

        events = []
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=lambda name, data: events.append(data),
        ):
            tracker.update_round(
                "claude", RoundMetrics(round_number=1, proposals_made=3, proposals_accepted=2)
            )
            tracker.update_round(
                "gemini", RoundMetrics(round_number=1, proposals_made=1, proposals_accepted=0)
            )

        assert len(events) == 2
        claude_event = [e for e in events if e["agent"] == "claude"][0]
        gemini_event = [e for e in events if e["agent"] == "gemini"][0]
        assert claude_event["total_proposals"] == 3
        assert gemini_event["total_proposals"] == 1

    def test_event_type_is_correct(self) -> None:
        tracker = ActiveIntrospectionTracker()

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            tracker.update_round("claude", RoundMetrics(round_number=1))

        assert mock_dispatch.call_args[0][0] == "agent_introspection_update"

    def test_handles_import_error(self) -> None:
        tracker = ActiveIntrospectionTracker()

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            tracker.update_round("claude", RoundMetrics(round_number=1))

        # Tracker state still updated
        summary = tracker.get_summary("claude")
        assert summary.rounds_completed == 1

    def test_critique_effectiveness_in_event(self) -> None:
        tracker = ActiveIntrospectionTracker()

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            tracker.update_round(
                "claude",
                RoundMetrics(
                    round_number=1,
                    critiques_given=4,
                    critiques_led_to_changes=3,
                ),
            )

        data = mock_dispatch.call_args[0][1]
        assert data["critique_effectiveness"] == 0.75

    def test_influence_in_event(self) -> None:
        tracker = ActiveIntrospectionTracker()

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            tracker.update_round(
                "claude",
                RoundMetrics(
                    round_number=1,
                    argument_influence=0.85,
                ),
            )

        data = mock_dispatch.call_args[0][1]
        assert data["average_influence"] == 0.85
