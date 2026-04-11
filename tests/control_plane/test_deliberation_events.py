"""Tests for aragora.control_plane.deliberation_events."""

from __future__ import annotations

from aragora.control_plane.deliberation_events import (
    CATEGORY_AGENT,
    CATEGORY_CONSENSUS,
    CATEGORY_ERROR,
    CATEGORY_LIFECYCLE,
    CATEGORY_PROGRESS,
    CATEGORY_ROUND,
    CATEGORY_SLA,
    EVENT_PREFIX,
    DeliberationEventType,
)


class TestDeliberationEventType:
    def test_all_values_start_with_prefix(self):
        for member in DeliberationEventType:
            assert member.value.startswith(EVENT_PREFIX)

    def test_unique_values(self):
        values = [m.value for m in DeliberationEventType]
        assert len(values) == len(set(values))

    def test_category_property(self):
        assert DeliberationEventType.DELIBERATION_STARTED.category == CATEGORY_LIFECYCLE
        assert DeliberationEventType.ROUND_START.category == CATEGORY_ROUND
        assert DeliberationEventType.AGENT_MESSAGE.category == CATEGORY_AGENT
        assert DeliberationEventType.VOTE.category == CATEGORY_CONSENSUS
        assert DeliberationEventType.SLA_WARNING.category == CATEGORY_SLA
        assert DeliberationEventType.PROGRESS_UPDATE.category == CATEGORY_PROGRESS
        assert DeliberationEventType.AGENT_ERROR.category == CATEGORY_ERROR

    def test_every_member_has_category(self):
        for member in DeliberationEventType:
            assert member.category != "unknown", f"{member} missing category"

    def test_is_terminal(self):
        assert DeliberationEventType.DELIBERATION_COMPLETED.is_terminal
        assert DeliberationEventType.DELIBERATION_FAILED.is_terminal
        assert DeliberationEventType.DELIBERATION_CANCELLED.is_terminal

    def test_non_terminal(self):
        assert not DeliberationEventType.DELIBERATION_STARTED.is_terminal
        assert not DeliberationEventType.ROUND_START.is_terminal
        assert not DeliberationEventType.AGENT_MESSAGE.is_terminal

    def test_by_category_lifecycle(self):
        lifecycle = DeliberationEventType.by_category(CATEGORY_LIFECYCLE)
        assert DeliberationEventType.DELIBERATION_STARTED in lifecycle
        assert DeliberationEventType.DELIBERATION_COMPLETED in lifecycle
        assert DeliberationEventType.ROUND_START not in lifecycle

    def test_by_category_unknown_returns_empty(self):
        assert DeliberationEventType.by_category("nonexistent") == frozenset()

    def test_categories_returns_all(self):
        cats = DeliberationEventType.categories()
        expected = {
            CATEGORY_LIFECYCLE,
            CATEGORY_ROUND,
            CATEGORY_AGENT,
            CATEGORY_CONSENSUS,
            CATEGORY_SLA,
            CATEGORY_PROGRESS,
            CATEGORY_ERROR,
        }
        assert cats == expected

    def test_str_enum_behavior(self):
        assert DeliberationEventType.VOTE.value == "deliberation.vote"
        assert DeliberationEventType.VOTE == "deliberation.vote"

    def test_exports(self):
        from aragora.control_plane import deliberation_events

        for name in deliberation_events.__all__:
            assert hasattr(deliberation_events, name)
