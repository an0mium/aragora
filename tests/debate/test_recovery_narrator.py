"""Tests for recovery narrator template-based commentary.

Covers RecoveryNarrative, RecoveryNarrator (narrate, handle_health_event,
get_recent_narratives, get_mood_summary), global narrator management,
and checkpoint integration helpers.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.recovery_narrator import (
    RecoveryNarrative,
    RecoveryNarrator,
    get_narrator,
    integrate_narrator_with_checkpoint_webhook,
    reset_narrator,
    setup_narrator_with_checkpoint_manager,
)


# ---------------------------------------------------------------------------
# RecoveryNarrative
# ---------------------------------------------------------------------------


class TestRecoveryNarrative:
    def test_fields(self):
        n = RecoveryNarrative(
            event_type="agent_timeout",
            agent="claude",
            headline="claude is taking a moment...",
            narrative="claude needs more time.",
            mood="tense",
        )
        assert n.event_type == "agent_timeout"
        assert n.agent == "claude"
        assert n.mood == "tense"
        assert isinstance(n.timestamp, float)

    def test_to_dict(self):
        n = RecoveryNarrative(
            event_type="agent_recovered",
            agent="gpt",
            headline="gpt bounces back!",
            narrative="gpt is back.",
            mood="triumphant",
            timestamp=1000.0,
        )
        d = n.to_dict()
        assert d["event_type"] == "agent_recovered"
        assert d["agent"] == "gpt"
        assert d["mood"] == "triumphant"
        assert d["timestamp"] == 1000.0


# ---------------------------------------------------------------------------
# RecoveryNarrator.narrate
# ---------------------------------------------------------------------------


class TestNarrate:
    def test_known_event_type(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        result = narrator.narrate("agent_started", "claude")
        assert result.event_type == "agent_started"
        assert result.agent == "claude"
        assert "claude" in result.headline
        assert "claude" in result.narrative
        assert result.mood == "neutral"

    def test_timeout_event(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        result = narrator.narrate("agent_timeout", "gpt")
        assert result.mood == "tense"
        assert "gpt" in result.headline

    def test_recovered_event(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        result = narrator.narrate("agent_recovered", "gemini")
        assert result.mood == "triumphant"

    def test_consensus_reached(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        result = narrator.narrate("consensus_reached", "System")
        assert result.mood == "triumphant"

    def test_circuit_opened(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        result = narrator.narrate("circuit_opened", "claude")
        assert result.mood == "cautionary"

    def test_details_format_in_narrative(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        result = narrator.narrate("checkpoint_created", "System", {"round": 3})
        # Some templates use {round} placeholder
        assert result.event_type == "checkpoint_created"

    def test_unknown_event_uses_completed_fallback(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        result = narrator.narrate("totally_unknown_event", "agent-x")
        # Falls back to agent_completed templates
        assert result.event_type == "totally_unknown_event"
        assert result.agent == "agent-x"

    def test_tracks_recent_narratives(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        narrator.narrate("agent_started", "a")
        narrator.narrate("agent_failed", "b")
        narrator.narrate("agent_recovered", "c")
        assert len(narrator.recent_narratives) == 3

    def test_recent_narratives_trimmed_at_100(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        for i in range(110):
            narrator.narrate("agent_started", f"agent-{i}")
        # Should trim to 50 when exceeding 100
        assert len(narrator.recent_narratives) <= 100

    def test_avoids_template_repetition(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        # agent_started has 3 headline templates
        headlines = set()
        for _ in range(6):
            result = narrator.narrate("agent_started", "claude")
            headlines.add(result.headline)
        # Should have used at least 2 different headlines
        assert len(headlines) >= 2

    def test_default_agent_system(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        result = narrator.narrate("debate_stalled")
        assert result.agent == "System"


# ---------------------------------------------------------------------------
# handle_health_event
# ---------------------------------------------------------------------------


class TestHandleHealthEvent:
    def test_ignores_non_health_events(self):
        narrator = RecoveryNarrator()
        result = narrator.handle_health_event({"type": "other_event"})
        assert result is None

    def test_ignores_unknown_event_types(self):
        narrator = RecoveryNarrator()
        result = narrator.handle_health_event(
            {
                "type": "health_event",
                "data": {"event_type": "some_random_thing"},
            }
        )
        assert result is None

    def test_generates_narrative_for_known_event(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        result = narrator.handle_health_event(
            {
                "type": "health_event",
                "data": {"event_type": "agent_timeout", "component": "claude"},
            }
        )
        assert result is not None
        assert result.event_type == "agent_timeout"
        assert result.agent == "claude"

    def test_broadcasts_when_callback_set(self):
        callback = MagicMock()
        narrator = RecoveryNarrator(broadcast_callback=callback)
        random.seed(42)
        narrator.handle_health_event(
            {
                "type": "health_event",
                "data": {"event_type": "agent_recovered", "component": "gpt"},
            }
        )
        callback.assert_called_once()
        call_arg = callback.call_args[0][0]
        assert call_arg["type"] == "recovery_narrative"

    def test_broadcast_failure_handled(self):
        callback = MagicMock(side_effect=RuntimeError("broadcast failed"))
        narrator = RecoveryNarrator(broadcast_callback=callback)
        random.seed(42)
        # Should not raise
        result = narrator.handle_health_event(
            {
                "type": "health_event",
                "data": {"event_type": "agent_failed", "component": "x"},
            }
        )
        assert result is not None

    def test_uses_details_from_event(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        result = narrator.handle_health_event(
            {
                "type": "health_event",
                "data": {
                    "event_type": "checkpoint_created",
                    "component": "System",
                    "details": {"round": 5},
                },
            }
        )
        assert result is not None


# ---------------------------------------------------------------------------
# get_recent_narratives / get_mood_summary
# ---------------------------------------------------------------------------


class TestSummaryMethods:
    def test_get_recent_narratives_empty(self):
        narrator = RecoveryNarrator()
        assert narrator.get_recent_narratives() == []

    def test_get_recent_narratives_limited(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        for i in range(15):
            narrator.narrate("agent_started", f"a{i}")
        recent = narrator.get_recent_narratives(limit=5)
        assert len(recent) == 5
        assert all(isinstance(n, dict) for n in recent)

    def test_mood_summary_empty(self):
        narrator = RecoveryNarrator()
        summary = narrator.get_mood_summary()
        assert summary["mood"] == "neutral"
        assert summary["distribution"] == {}

    def test_mood_summary_with_events(self):
        narrator = RecoveryNarrator()
        random.seed(42)
        narrator.narrate("agent_timeout", "a")
        narrator.narrate("agent_timeout", "b")
        narrator.narrate("agent_recovered", "a")
        summary = narrator.get_mood_summary()
        assert "tense" in summary["distribution"]
        assert "triumphant" in summary["distribution"]
        # tense should dominate (2 vs 1)
        assert summary["mood"] == "tense"


# ---------------------------------------------------------------------------
# Global narrator
# ---------------------------------------------------------------------------


class TestGlobalNarrator:
    def setup_method(self):
        reset_narrator()

    def teardown_method(self):
        reset_narrator()

    def test_get_narrator_creates_singleton(self):
        n1 = get_narrator()
        n2 = get_narrator()
        assert n1 is n2

    def test_reset_narrator_clears_singleton(self):
        n1 = get_narrator()
        reset_narrator()
        n2 = get_narrator()
        assert n1 is not n2


# ---------------------------------------------------------------------------
# Checkpoint integration
# ---------------------------------------------------------------------------


class TestCheckpointIntegration:
    def setup_method(self):
        reset_narrator()

    def teardown_method(self):
        reset_narrator()

    def test_setup_with_checkpoint_manager(self):
        narrator = setup_narrator_with_checkpoint_manager()
        assert hasattr(narrator, "_checkpoint_handlers")
        assert "on_checkpoint" in narrator._checkpoint_handlers
        assert "on_resume" in narrator._checkpoint_handlers

    def test_checkpoint_handler_generates_narrative(self):
        narrator = setup_narrator_with_checkpoint_manager()
        random.seed(42)
        handler = narrator._checkpoint_handlers["on_checkpoint"]
        handler({"checkpoint": {"current_round": 3, "debate_id": "d1"}})
        assert len(narrator.recent_narratives) == 1
        assert narrator.recent_narratives[0].event_type == "checkpoint_created"

    def test_resume_handler_generates_narrative(self):
        narrator = setup_narrator_with_checkpoint_manager()
        random.seed(42)
        handler = narrator._checkpoint_handlers["on_resume"]
        handler({"checkpoint": {"current_round": 2, "debate_id": "d2", "agent_states": [1, 2]}})
        assert len(narrator.recent_narratives) == 1
        assert narrator.recent_narratives[0].event_type == "debate_resumed"

    def test_checkpoint_broadcast(self):
        callback = MagicMock()
        narrator = RecoveryNarrator(broadcast_callback=callback)
        narrator = setup_narrator_with_checkpoint_manager(narrator)
        random.seed(42)
        handler = narrator._checkpoint_handlers["on_checkpoint"]
        handler({"checkpoint": {"current_round": 1}})
        callback.assert_called_once()

    def test_integrate_with_webhook(self):
        webhook = MagicMock()
        webhook.on_checkpoint = MagicMock()
        webhook.on_resume = MagicMock()
        narrator = RecoveryNarrator()
        integrate_narrator_with_checkpoint_webhook(webhook, narrator)
        webhook.on_checkpoint.assert_called_once()
        webhook.on_resume.assert_called_once()

    def test_integrate_with_webhook_missing_methods(self):
        # Webhook without on_checkpoint/on_resume
        webhook = MagicMock(spec=[])
        narrator = RecoveryNarrator()
        # Should not raise
        integrate_narrator_with_checkpoint_webhook(webhook, narrator)


# ---------------------------------------------------------------------------
# Template coverage
# ---------------------------------------------------------------------------


class TestAllTemplates:
    """Verify all template event types produce valid narratives."""

    ALL_EVENTS = [
        "agent_started",
        "agent_timeout",
        "agent_failed",
        "agent_recovered",
        "agent_completed",
        "circuit_opened",
        "circuit_closed",
        "fallback_used",
        "consensus_reached",
        "debate_stalled",
        "checkpoint_created",
        "debate_resumed",
        "debate_paused",
        "checkpoint_restored",
    ]

    @pytest.mark.parametrize("event_type", ALL_EVENTS)
    def test_template_produces_narrative(self, event_type):
        narrator = RecoveryNarrator()
        random.seed(42)
        # Some templates have {round}/{agent_count} placeholders
        details = {"round": 1, "agent_count": 3, "debate_id": "test"}
        result = narrator.narrate(event_type, "test-agent", details)
        assert result.event_type == event_type
        assert isinstance(result.headline, str)
        assert isinstance(result.narrative, str)
        assert result.mood in ("tense", "triumphant", "cautionary", "neutral")
