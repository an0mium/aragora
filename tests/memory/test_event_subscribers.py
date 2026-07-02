"""Tests for the memory-domain event-subscriber home (P4a Batch E3).

Covers ``aragora.memory.event_subscribers``: the knowledge/evidence/mound ->
memory reactions relocated out of infrastructure
(``aragora.events.cross_subscribers.handlers.basic``) into their domain home,
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
from aragora.memory.event_subscribers import (
    MEMORY_EVENT_SUBSCRIBER_HANDLER_NAMES,
    MemoryEventSubscriber,
    get_memory_event_subscriber,
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


class TestMemoryEventSubscriberHandlers:
    """Direct handler-execution tests."""

    def test_knowledge_to_memory_handler_executes_without_error(self):
        subscriber = MemoryEventSubscriber()
        event = make_stream_event(
            StreamEventType.KNOWLEDGE_INDEXED,
            data={"node_id": "node_001", "content": "Test content", "node_type": "fact"},
        )
        subscriber._handle_knowledge_to_memory(event)

    def test_evidence_to_insight_handler_executes_without_error(self):
        subscriber = MemoryEventSubscriber()
        event = make_stream_event(
            StreamEventType.EVIDENCE_FOUND,
            data={
                "evidence_id": "ev_001",
                "source": "github",
                "content": "x" * 60,
                "claim": "some claim",
                "confidence": 0.9,
            },
        )
        subscriber._handle_evidence_to_insight(event)

    def test_evidence_to_insight_skips_short_content(self):
        subscriber = MemoryEventSubscriber()
        event = make_stream_event(
            StreamEventType.EVIDENCE_FOUND,
            data={"evidence_id": "ev_002", "source": "github", "content": "too short"},
        )
        subscriber._handle_evidence_to_insight(event)

    def test_mound_to_memory_handler_culture_patterns(self):
        subscriber = MemoryEventSubscriber()
        event = make_stream_event(
            StreamEventType.MOUND_UPDATED,
            data={"update_type": "culture_patterns", "patterns_count": 3, "debate_id": "d1"},
        )
        subscriber._handle_mound_to_memory(event)

    def test_mound_to_memory_handler_node_deleted(self):
        subscriber = MemoryEventSubscriber()
        event = make_stream_event(
            StreamEventType.MOUND_UPDATED,
            data={"update_type": "node_deleted", "node_id": "node_001", "archived": True},
        )
        subscriber._handle_mound_to_memory(event)


class TestMemoryEventSubscriberRegistration:
    """Registration + self-registration surface tests."""

    def test_handler_names_frozenset(self):
        assert MEMORY_EVENT_SUBSCRIBER_HANDLER_NAMES == frozenset(
            {"knowledge_to_memory", "evidence_to_insight", "mound_to_memory"}
        )

    def test_register_wires_all_handlers_into_manager(self):
        manager = CrossSubscriberManager()
        subscriber = MemoryEventSubscriber()

        subscriber.register(manager)

        registered = set(manager.get_stats())
        assert MEMORY_EVENT_SUBSCRIBER_HANDLER_NAMES <= registered

    def test_register_dispatches_events_through_manager(self):
        manager = CrossSubscriberManager()
        MemoryEventSubscriber().register(manager)

        manager._dispatch_event(
            make_stream_event(
                StreamEventType.KNOWLEDGE_INDEXED,
                data={"node_id": "node_002", "content": "Test", "node_type": "claim"},
            )
        )

        stats = manager.get_stats()
        assert stats["knowledge_to_memory"]["events_processed"] == 1

    def test_get_memory_event_subscriber_returns_singleton(self):
        first = get_memory_event_subscriber()
        second = get_memory_event_subscriber()
        assert first is second
        assert get_registered_subscribers()["memory"] is first

    def test_register_function_is_idempotent(self):
        register()
        first = get_registered_subscribers()["memory"]
        register()
        second = get_registered_subscribers()["memory"]
        assert first is second
