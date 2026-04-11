"""Unit tests for aragora.control_plane.deliberation_events."""

from __future__ import annotations

import pytest

from aragora.control_plane.deliberation_events import (
    CATEGORIES,
    DeliberationEventType,
)


class TestCategories:
    def test_categories_tuple_has_expected_entries(self):
        assert "lifecycle" in CATEGORIES
        assert "round" in CATEGORIES
        assert "agent" in CATEGORIES
        assert "consensus" in CATEGORIES
        assert "sla" in CATEGORIES
        assert "progress" in CATEGORIES
        assert "error" in CATEGORIES
        assert len(CATEGORIES) == 7

    def test_all_enum_members_have_a_category(self):
        for event in DeliberationEventType:
            assert event.category in CATEGORIES


class TestDeliberationEventType:
    def test_is_str_enum(self):
        assert isinstance(DeliberationEventType.DELIBERATION_STARTED, str)
        assert DeliberationEventType.DELIBERATION_STARTED == "deliberation.started"

    def test_unique_values(self):
        values = [e.value for e in DeliberationEventType]
        assert len(values) == len(set(values))

    def test_category_property_lifecycle(self):
        assert DeliberationEventType.DELIBERATION_STARTED.category == "lifecycle"
        assert DeliberationEventType.DELIBERATION_COMPLETED.category == "lifecycle"
        assert DeliberationEventType.DELIBERATION_FAILED.category == "lifecycle"
        assert DeliberationEventType.DELIBERATION_CANCELLED.category == "lifecycle"

    def test_category_property_agent(self):
        assert DeliberationEventType.AGENT_MESSAGE.category == "agent"
        assert DeliberationEventType.AGENT_PROPOSAL.category == "agent"
        assert DeliberationEventType.AGENT_CRITIQUE.category == "agent"
        assert DeliberationEventType.AGENT_REVISION.category == "agent"

    def test_is_terminal_true_for_terminal_events(self):
        assert DeliberationEventType.DELIBERATION_COMPLETED.is_terminal is True
        assert DeliberationEventType.DELIBERATION_FAILED.is_terminal is True
        assert DeliberationEventType.DELIBERATION_CANCELLED.is_terminal is True

    def test_is_terminal_false_for_non_terminal_events(self):
        assert DeliberationEventType.DELIBERATION_STARTED.is_terminal is False
        assert DeliberationEventType.ROUND_START.is_terminal is False
        assert DeliberationEventType.AGENT_MESSAGE.is_terminal is False
        assert DeliberationEventType.CONSENSUS_REACHED.is_terminal is False

    def test_by_category_returns_correct_events(self):
        lifecycle = DeliberationEventType.by_category("lifecycle")
        assert DeliberationEventType.DELIBERATION_STARTED in lifecycle
        assert DeliberationEventType.DELIBERATION_COMPLETED in lifecycle
        assert DeliberationEventType.ROUND_START not in lifecycle
        assert isinstance(lifecycle, frozenset)

    def test_by_category_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown category"):
            DeliberationEventType.by_category("nonexistent")

    def test_by_category_covers_all_members(self):
        all_events: set[DeliberationEventType] = set()
        for cat in CATEGORIES:
            all_events.update(DeliberationEventType.by_category(cat))
        assert all_events == set(DeliberationEventType)
