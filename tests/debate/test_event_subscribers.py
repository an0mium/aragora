"""Tests for the debate-domain event-subscriber home (P4a Batch E4).

Covers ``aragora.debate.event_subscribers``: the ELO/calibration/consensus/
rhetorical reactions relocated out of infrastructure
(``aragora.events.cross_subscribers.handlers.basic``) and the budget-alert/
meta-learning reactions relocated out of
``aragora.events.cross_subscribers.handlers.strategic`` into this domain home,
plus the self-registration surface (get-or-create accessor, ``register()``).

The budget-alert and meta-learning handlers already had dedicated test classes
in ``tests/events/test_strategic_handlers.py`` (``TestBudgetAlertToTeamSelection``,
``TestMetaLearningToTeamSelection``); those were repointed to
``DebateEventSubscriber`` in place rather than duplicated here (same precedent
as ``TestApprovalToKMReinforcement`` for the P4a Batch E2c knowledge relocation).
"""

from __future__ import annotations

import pytest

from aragora.events.cross_subscribers import (
    CrossSubscriberManager,
    get_registered_subscribers,
    reset_cross_subscriber_manager,
    reset_registry,
)
from aragora.events.types import StreamEvent, StreamEventType
from aragora.debate.event_subscribers import (
    DEBATE_EVENT_SUBSCRIBER_HANDLER_NAMES,
    DebateEventSubscriber,
    get_debate_event_subscriber,
    register,
)


def make_stream_event(event_type: StreamEventType, data: dict | None = None) -> StreamEvent:
    """Factory for creating test stream events."""
    return StreamEvent(type=event_type, data=data or {}, round=0, agent="test_agent")


@pytest.fixture(autouse=True)
def _clean_registry_and_manager():
    """Isolate each test: empty registry + fresh manager before and after."""
    reset_registry()
    reset_cross_subscriber_manager()
    yield
    reset_registry()
    reset_cross_subscriber_manager()


class TestDebateEventSubscriberHandlers:
    """Direct handler-execution tests."""

    def test_elo_to_debate_handler_executes_without_error(self):
        subscriber = DebateEventSubscriber()
        event = make_stream_event(
            StreamEventType.AGENT_ELO_UPDATED,
            data={"agent": "claude", "elo": 1600, "delta": 20, "debate_id": "d1"},
        )
        subscriber._handle_elo_to_debate(event)

    def test_elo_to_debate_handler_logs_significant_change(self):
        subscriber = DebateEventSubscriber()
        event = make_stream_event(
            StreamEventType.AGENT_ELO_UPDATED,
            data={"agent": "claude", "elo": 1650, "delta": 75, "debate_id": "d1"},
        )
        subscriber._handle_elo_to_debate(event)

    def test_calibration_to_agent_handler_executes_without_error(self):
        subscriber = DebateEventSubscriber()
        event = make_stream_event(
            StreamEventType.CALIBRATION_UPDATE,
            data={"agent": "claude", "score": 0.85, "prediction_count": 12},
        )
        subscriber._handle_calibration_to_agent(event)

    def test_consensus_to_learning_handler_executes_without_error(self):
        subscriber = DebateEventSubscriber()
        event = make_stream_event(
            StreamEventType.CONSENSUS,
            data={"debate_id": "d1", "confidence": 0.9, "agents": ["claude", "gpt"]},
        )
        subscriber._handle_consensus_to_learning(event)

    def test_consensus_to_learning_skips_low_confidence(self):
        subscriber = DebateEventSubscriber()
        event = make_stream_event(
            StreamEventType.CONSENSUS,
            data={"debate_id": "d1", "confidence": 0.2, "agents": ["claude", "gpt"]},
        )
        subscriber._handle_consensus_to_learning(event)

    def test_consensus_to_learning_skips_no_agents(self):
        subscriber = DebateEventSubscriber()
        event = make_stream_event(
            StreamEventType.CONSENSUS,
            data={"debate_id": "d1", "confidence": 0.9, "agents": []},
        )
        subscriber._handle_consensus_to_learning(event)

    def test_agent_message_to_rhetorical_handler_executes_without_error(self):
        subscriber = DebateEventSubscriber()
        event = make_stream_event(
            StreamEventType.AGENT_MESSAGE,
            data={"agent": "claude", "content": "This is a sufficiently long message body."},
        )
        subscriber._handle_agent_message_to_rhetorical(event)

    def test_agent_message_to_rhetorical_skips_short_content(self):
        subscriber = DebateEventSubscriber()
        event = make_stream_event(
            StreamEventType.AGENT_MESSAGE,
            data={"agent": "claude", "content": "too short"},
        )
        subscriber._handle_agent_message_to_rhetorical(event)


class TestDebateEventSubscriberRegistration:
    """Registration + self-registration surface tests."""

    def test_handler_names_frozenset(self):
        assert DEBATE_EVENT_SUBSCRIBER_HANDLER_NAMES == frozenset(
            {
                "elo_to_debate",
                "calibration_to_agent",
                "consensus_to_learning",
                "agent_message_to_rhetorical",
                "budget_alert_to_team_selection",
                "meta_learning_to_team_selection",
            }
        )

    def test_register_wires_all_handlers_into_manager(self):
        manager = CrossSubscriberManager()
        subscriber = DebateEventSubscriber()

        subscriber.register(manager)

        registered = set(manager.get_stats())
        assert DEBATE_EVENT_SUBSCRIBER_HANDLER_NAMES <= registered

    @pytest.mark.parametrize(
        ("handler_name", "event_type", "data"),
        [
            (
                "elo_to_debate",
                StreamEventType.AGENT_ELO_UPDATED,
                {"agent": "claude", "elo": 1600, "delta": 20},
            ),
            (
                "calibration_to_agent",
                StreamEventType.CALIBRATION_UPDATE,
                {"agent": "claude", "score": 0.7},
            ),
            (
                "consensus_to_learning",
                StreamEventType.CONSENSUS,
                {"debate_id": "d1", "confidence": 0.9, "agents": ["claude"]},
            ),
            (
                "agent_message_to_rhetorical",
                StreamEventType.AGENT_MESSAGE,
                {"agent": "claude", "content": "x" * 30},
            ),
            (
                "budget_alert_to_team_selection",
                StreamEventType.BUDGET_ALERT,
                {"alert_type": "soft_limit", "workspace_id": "ws1"},
            ),
            (
                "meta_learning_to_team_selection",
                StreamEventType.META_LEARNING_ADJUSTED,
                {"adjustments": {"elo_weight": 0.3}, "learning_rate": 0.01},
            ),
        ],
    )
    def test_register_dispatches_events_through_manager(self, handler_name, event_type, data):
        manager = CrossSubscriberManager()
        DebateEventSubscriber().register(manager)

        manager._dispatch_event(make_stream_event(event_type, data=data))

        stats = manager.get_stats()
        assert stats[handler_name]["events_processed"] == 1

    def test_get_debate_event_subscriber_returns_singleton(self):
        first = get_debate_event_subscriber()
        second = get_debate_event_subscriber()
        assert first is second
        assert get_registered_subscribers()["debate"] is first

    def test_register_function_is_idempotent(self):
        register()
        first = get_registered_subscribers()["debate"]
        register()
        second = get_registered_subscribers()["debate"]
        assert first is second
