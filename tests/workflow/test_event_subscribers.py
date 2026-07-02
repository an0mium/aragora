"""Tests for the workflow-domain (application-tier) event-subscriber home
(P4a Batch E5).

Covers ``aragora.workflow.event_subscribers``: the post-debate workflow
automation reaction relocated out of infrastructure
(``aragora.events.subscribers.workflow_automation``) and the alert-escalation
emergency-brake reaction relocated out of
``aragora.events.cross_subscribers.handlers.strategic``, into this
application home, plus the self-registration surface (get-or-create
accessor, ``register()``).

Unit tests for ``PostDebateWorkflowSubscriber``'s own outcome-classification
and workflow-triggering logic live in
``tests/events/test_post_debate_workflow.py`` (repointed in place); the
``_handle_alert_escalated_to_workflow_brake`` handler's direct-execution
tests live in ``TestAlertEscalatedToWorkflowBrake`` in
``tests/events/test_strategic_handlers.py`` (also repointed in place - same
precedent as ``TestBudgetAlertToTeamSelection`` for the P4a Batch E4
relocation). This module covers registration/dispatch integration plus the
E5-specific invocation-count acceptance test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.events.cross_subscribers import (
    CrossSubscriberManager,
    get_registered_subscribers,
    reset_cross_subscriber_manager,
    reset_registry,
)
from aragora.events.types import StreamEvent, StreamEventType
from aragora.workflow.event_subscribers import (
    WORKFLOW_EVENT_SUBSCRIBER_HANDLER_NAMES,
    PostDebateWorkflowSubscriber,
    WorkflowEventSubscriber,
    get_workflow_event_subscriber,
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


class TestWorkflowEventSubscriberHandlers:
    """Direct handler-execution tests."""

    def test_debate_end_to_workflow_delegates_to_post_debate_workflow(self):
        subscriber = WorkflowEventSubscriber()
        event = make_stream_event(
            StreamEventType.DEBATE_END,
            data={"debate_id": "d1", "consensus_reached": True, "confidence": 0.9},
        )

        with patch.object(PostDebateWorkflowSubscriber, "_trigger_workflow") as mock_trigger:
            subscriber._handle_debate_end_to_workflow(event)

        mock_trigger.assert_called_once()
        assert subscriber.post_debate_workflow.stats["events_processed"] == 1

    def test_alert_escalated_to_workflow_brake_handler_executes_without_error(self):
        subscriber = WorkflowEventSubscriber()
        event = make_stream_event(
            StreamEventType.ALERT_ESCALATED,
            data={"severity": "warning", "alert_id": "alert_1"},
        )
        subscriber._handle_alert_escalated_to_workflow_brake(event)

    def test_alert_escalated_to_workflow_brake_pauses_on_critical(self):
        subscriber = WorkflowEventSubscriber()
        event = make_stream_event(
            StreamEventType.ALERT_ESCALATED,
            data={"severity": "critical", "alert_id": "alert_2", "reason": "test reason"},
        )
        mock_engine = MagicMock()
        with patch("aragora.workflow.engine.get_workflow_engine", return_value=mock_engine):
            subscriber._handle_alert_escalated_to_workflow_brake(event)
        mock_engine.pause_all.assert_called_once()


class TestWorkflowEventSubscriberRegistration:
    """Registration + self-registration surface tests."""

    def test_handler_names_frozenset(self):
        assert WORKFLOW_EVENT_SUBSCRIBER_HANDLER_NAMES == frozenset(
            {
                "debate_end_to_workflow",
                "alert_escalated_to_workflow_brake",
            }
        )

    def test_register_wires_all_handlers_into_manager(self):
        manager = CrossSubscriberManager()
        subscriber = WorkflowEventSubscriber()

        subscriber.register(manager)

        registered = set(manager.get_stats())
        assert WORKFLOW_EVENT_SUBSCRIBER_HANDLER_NAMES <= registered

    def test_register_dispatches_debate_end_to_workflow_through_manager(self):
        manager = CrossSubscriberManager()
        WorkflowEventSubscriber().register(manager)

        # Patch _trigger_workflow so this dispatch-count check does not
        # construct a real aragora.workflow.engine.WorkflowEngine.
        with patch.object(PostDebateWorkflowSubscriber, "_trigger_workflow"):
            manager._dispatch_event(
                make_stream_event(
                    StreamEventType.DEBATE_END,
                    data={"debate_id": "d1", "consensus_reached": False, "confidence": 0.0},
                )
            )

        stats = manager.get_stats()
        assert stats["debate_end_to_workflow"]["events_processed"] == 1

    def test_register_dispatches_alert_escalated_to_workflow_brake_through_manager(self):
        manager = CrossSubscriberManager()
        WorkflowEventSubscriber().register(manager)

        manager._dispatch_event(
            make_stream_event(
                StreamEventType.ALERT_ESCALATED,
                data={"severity": "warning", "alert_id": "a1"},
            )
        )

        stats = manager.get_stats()
        assert stats["alert_escalated_to_workflow_brake"]["events_processed"] == 1

    def test_get_workflow_event_subscriber_returns_singleton(self):
        first = get_workflow_event_subscriber()
        second = get_workflow_event_subscriber()
        assert first is second
        assert get_registered_subscribers()["workflow"] is first

    def test_register_function_is_idempotent(self):
        register()
        first = get_registered_subscribers()["workflow"]
        register()
        second = get_registered_subscribers()["workflow"]
        assert first is second


class TestPostDebateWorkflowSubscriberFiresExactlyOnce:
    """E5-specific acceptance test (docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md
    §10 row E5, acceptance criterion 6): an INVOCATION-COUNT test proving
    ``PostDebateWorkflowSubscriber`` fires EXACTLY ONCE per ``debate_end``
    event after the coupling inversion.

    A registration-count parity check (e.g.
    ``test_register_wires_all_handlers_into_manager`` above, or the
    golden-name parity tests in
    ``tests/events/test_cross_subscriber_registry.py``) is INSUFFICIENT: it
    only proves ONE subscriber *name* was registered, not that the underlying
    ``PostDebateWorkflowSubscriber.handle_debate_end`` was invoked exactly
    once in total. Historically this reaction was reachable from TWO
    independent delegating call sites - ``subscribers/debate_handlers.py:457``
    (deleted by P4a Batch E4) and ``cross_subscribers/handlers/basic.py:577``
    (deleted by this batch) - each of which instantiated its own throwaway
    ``PostDebateWorkflowSubscriber`` and called ``.handle_debate_end()``. A
    residual second call site would raise the TOTAL invocation count to 2
    while every PER-NAME registration/dispatch-count check would still
    independently read 1, so only a class-level invocation count (patching
    ``PostDebateWorkflowSubscriber.handle_debate_end`` itself, observed across
    the whole dispatch) can catch that regression.
    """

    def test_fires_exactly_once_via_full_superset_bootstrap(self):
        """Bootstrap the real interface-superset composition root (not a
        hand-built manager) and dispatch ONE DEBATE_END event: the total
        number of ``PostDebateWorkflowSubscriber.handle_debate_end`` calls
        across the whole wired manager must be exactly 1."""
        from aragora.server.startup.event_subscribers import bootstrap_event_subscribers

        manager = bootstrap_event_subscribers()
        event = make_stream_event(
            StreamEventType.DEBATE_END,
            data={"debate_id": "d1", "consensus_reached": True, "confidence": 0.9},
        )

        with patch.object(PostDebateWorkflowSubscriber, "handle_debate_end") as mock_handle:
            manager._dispatch_event(event)

        mock_handle.assert_called_once()

    def test_fires_exactly_once_when_registered_directly_on_a_manager(self):
        """Same guarantee at the unit level: the callable
        ``WorkflowEventSubscriber.register`` wires into the manager calls
        ``handle_debate_end`` exactly once per dispatched event."""
        manager = CrossSubscriberManager()
        WorkflowEventSubscriber().register(manager)
        event = make_stream_event(StreamEventType.DEBATE_END, data={"debate_id": "d2"})

        with patch.object(PostDebateWorkflowSubscriber, "handle_debate_end") as mock_handle:
            manager._dispatch_event(event)

        mock_handle.assert_called_once()

    def test_fires_exactly_once_after_repeated_bootstrap_calls(self):
        """Bootstrap is documented as idempotent (repeated calls only apply
        newly-registered subscribers): calling it twice must not create a
        second live delegate, so DEBATE_END still reaches handle_debate_end
        exactly once."""
        from aragora.server.startup.event_subscribers import bootstrap_event_subscribers

        bootstrap_event_subscribers()
        manager = bootstrap_event_subscribers()  # simulate a second bootstrap call
        event = make_stream_event(StreamEventType.DEBATE_END, data={"debate_id": "d3"})

        with patch.object(PostDebateWorkflowSubscriber, "handle_debate_end") as mock_handle:
            manager._dispatch_event(event)

        mock_handle.assert_called_once()


class TestLegacyDelegatingSitesRemoved:
    """Structural regression guard: both pre-inversion delegating call sites
    for ``PostDebateWorkflowSubscriber`` are gone entirely, not merely
    unregistered (docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §5.3)."""

    def test_basic_handlers_mixin_has_no_debate_end_to_workflow_delegate(self):
        from aragora.events.cross_subscribers.handlers.basic import BasicHandlersMixin

        assert not hasattr(BasicHandlersMixin, "_handle_debate_end_to_workflow")

    def test_strategic_handlers_mixin_has_no_alert_escalated_to_workflow_brake(self):
        from aragora.events.cross_subscribers.handlers.strategic import StrategicHandlersMixin

        assert not hasattr(StrategicHandlersMixin, "_handle_alert_escalated_to_workflow_brake")

    def test_workflow_automation_module_removed_no_shim(self):
        """Relocate-UP no-shim exemption: no re-export module survives at the
        old infrastructure path."""
        with pytest.raises(ModuleNotFoundError):
            import aragora.events.subscribers.workflow_automation  # noqa: F401
