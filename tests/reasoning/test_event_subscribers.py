"""Tests for the reasoning-domain event-subscriber home (P4a Batch E3).

Covers ``aragora.reasoning.event_subscribers``: the vote -> belief-network
reaction relocated out of infrastructure
(``aragora.events.cross_subscribers.handlers.basic``) into its domain home,
plus the self-registration surface (get-or-create accessor, ``register()``).
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
from aragora.reasoning.event_subscribers import (
    REASONING_EVENT_SUBSCRIBER_HANDLER_NAMES,
    ReasoningEventSubscriber,
    get_reasoning_event_subscriber,
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


class TestReasoningEventSubscriberHandlers:
    """Direct handler-execution tests."""

    def test_vote_to_belief_handler_executes_without_error(self):
        subscriber = ReasoningEventSubscriber()
        event = make_stream_event(
            StreamEventType.VOTE,
            data={
                "agent": "claude",
                "position": "affirmative",
                "confidence": 0.8,
                "debate_id": "d1",
            },
        )
        subscriber._handle_vote_to_belief(event)

    def test_vote_to_belief_handler_skips_missing_position(self):
        subscriber = ReasoningEventSubscriber()
        event = make_stream_event(StreamEventType.VOTE, data={"agent": "claude"})
        subscriber._handle_vote_to_belief(event)


class TestReasoningEventSubscriberRegistration:
    """Registration + self-registration surface tests."""

    def test_handler_names_frozenset(self):
        assert REASONING_EVENT_SUBSCRIBER_HANDLER_NAMES == frozenset({"vote_to_belief"})

    def test_register_wires_handler_into_manager(self):
        manager = CrossSubscriberManager()
        subscriber = ReasoningEventSubscriber()

        subscriber.register(manager)

        assert "vote_to_belief" in manager.get_stats()

    def test_register_dispatches_events_through_manager(self):
        manager = CrossSubscriberManager()
        ReasoningEventSubscriber().register(manager)

        manager._dispatch_event(
            make_stream_event(
                StreamEventType.VOTE,
                data={"agent": "gpt", "position": "negative", "confidence": 0.6},
            )
        )

        stats = manager.get_stats()
        assert stats["vote_to_belief"]["events_processed"] == 1

    def test_get_reasoning_event_subscriber_returns_singleton(self):
        first = get_reasoning_event_subscriber()
        second = get_reasoning_event_subscriber()
        assert first is second
        assert get_registered_subscribers()["reasoning"] is first

    def test_register_function_is_idempotent(self):
        register()
        first = get_registered_subscribers()["reasoning"]
        register()
        second = get_registered_subscribers()["reasoning"]
        assert first is second
