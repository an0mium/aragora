"""Tests for handler registry (aragora/server/handlers/_registry.py).

Covers ALL public symbols and behavior:
- ALL_HANDLERS list (module-level mutable list)
- HANDLER_STABILITY dict (module-level mutable dict)
- get_handler_stability() lookup with default fallback
- get_all_handler_stability() serialization for API responses
- __all__ exports
- Interaction between the two data structures
- Edge cases: empty state, unknown handlers, special characters, concurrent modification
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.config.stability import Stability
from aragora.server.handlers._registry import (
    ALL_HANDLERS,
    HANDLER_STABILITY,
    __all__,
    get_all_handler_stability,
    get_handler_stability,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    saved_handlers = ALL_HANDLERS[:]
    saved_stability = HANDLER_STABILITY.copy()
    ALL_HANDLERS.clear()
    HANDLER_STABILITY.clear()
    yield
    ALL_HANDLERS.clear()
    ALL_HANDLERS.extend(saved_handlers)
    HANDLER_STABILITY.clear()
    HANDLER_STABILITY.update(saved_stability)


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


class TestExports:
    """Verify module __all__ exports."""

    def test_all_contains_all_handlers(self):
        assert "ALL_HANDLERS" in __all__

    def test_all_contains_handler_stability(self):
        assert "HANDLER_STABILITY" in __all__

    def test_all_contains_get_handler_stability(self):
        assert "get_handler_stability" in __all__

    def test_all_contains_get_all_handler_stability(self):
        assert "get_all_handler_stability" in __all__

    def test_all_has_exactly_four_entries(self):
        assert len(__all__) == 4


# ---------------------------------------------------------------------------
# ALL_HANDLERS list
# ---------------------------------------------------------------------------


class TestAllHandlers:
    """Verify ALL_HANDLERS behaves as a mutable list populated externally."""

    def test_initial_state_is_empty_in_test(self):
        """After fixture cleanup, list should be empty."""
        assert ALL_HANDLERS == []

    def test_is_a_list(self):
        assert isinstance(ALL_HANDLERS, list)

    def test_can_append_handler_class(self):
        mock_cls = type("FakeHandler", (), {})
        ALL_HANDLERS.append(mock_cls)
        assert len(ALL_HANDLERS) == 1
        assert ALL_HANDLERS[0] is mock_cls

    def test_can_extend_with_multiple(self):
        cls_a = type("HandlerA", (), {})
        cls_b = type("HandlerB", (), {})
        ALL_HANDLERS.extend([cls_a, cls_b])
        assert len(ALL_HANDLERS) == 2


# ---------------------------------------------------------------------------
# HANDLER_STABILITY dict
# ---------------------------------------------------------------------------


class TestHandlerStability:
    """Verify HANDLER_STABILITY behaves as a mutable dict populated externally."""

    def test_initial_state_is_empty_in_test(self):
        assert HANDLER_STABILITY == {}

    def test_is_a_dict(self):
        assert isinstance(HANDLER_STABILITY, dict)

    def test_can_set_stability(self):
        HANDLER_STABILITY["TestHandler"] = Stability.STABLE
        assert HANDLER_STABILITY["TestHandler"] == Stability.STABLE

    def test_can_set_multiple(self):
        HANDLER_STABILITY["A"] = Stability.STABLE
        HANDLER_STABILITY["B"] = Stability.EXPERIMENTAL
        HANDLER_STABILITY["C"] = Stability.PREVIEW
        assert len(HANDLER_STABILITY) == 3


# ---------------------------------------------------------------------------
# get_handler_stability()
# ---------------------------------------------------------------------------


class TestGetHandlerStability:
    """Test get_handler_stability lookup and defaults."""

    def test_returns_experimental_for_unknown_handler(self):
        result = get_handler_stability("NonexistentHandler")
        assert result == Stability.EXPERIMENTAL

    def test_returns_stable_when_classified(self):
        HANDLER_STABILITY["DebatesHandler"] = Stability.STABLE
        assert get_handler_stability("DebatesHandler") == Stability.STABLE

    def test_returns_preview_when_classified(self):
        HANDLER_STABILITY["PreviewHandler"] = Stability.PREVIEW
        assert get_handler_stability("PreviewHandler") == Stability.PREVIEW

    def test_returns_deprecated_when_classified(self):
        HANDLER_STABILITY["OldHandler"] = Stability.DEPRECATED
        assert get_handler_stability("OldHandler") == Stability.DEPRECATED

    def test_returns_experimental_when_classified_as_experimental(self):
        HANDLER_STABILITY["NewHandler"] = Stability.EXPERIMENTAL
        assert get_handler_stability("NewHandler") == Stability.EXPERIMENTAL

    def test_default_for_empty_string_name(self):
        result = get_handler_stability("")
        assert result == Stability.EXPERIMENTAL

    def test_default_for_handler_not_in_dict(self):
        HANDLER_STABILITY["OnlyThisOne"] = Stability.STABLE
        result = get_handler_stability("SomeOtherHandler")
        assert result == Stability.EXPERIMENTAL

    def test_case_sensitive_lookup(self):
        HANDLER_STABILITY["DebatesHandler"] = Stability.STABLE
        # Different case should NOT match
        result = get_handler_stability("debateshandler")
        assert result == Stability.EXPERIMENTAL

    def test_handler_name_with_special_characters(self):
        HANDLER_STABILITY["Handler-With-Dashes"] = Stability.PREVIEW
        assert get_handler_stability("Handler-With-Dashes") == Stability.PREVIEW

    def test_handler_name_with_dots(self):
        HANDLER_STABILITY["module.HandlerClass"] = Stability.STABLE
        assert get_handler_stability("module.HandlerClass") == Stability.STABLE

    def test_handler_name_with_spaces(self):
        """Unusual but should work since it's just a dict lookup."""
        HANDLER_STABILITY["Handler With Spaces"] = Stability.DEPRECATED
        assert get_handler_stability("Handler With Spaces") == Stability.DEPRECATED

    def test_stability_updated_after_initial_set(self):
        HANDLER_STABILITY["Evolving"] = Stability.EXPERIMENTAL
        assert get_handler_stability("Evolving") == Stability.EXPERIMENTAL
        HANDLER_STABILITY["Evolving"] = Stability.STABLE
        assert get_handler_stability("Evolving") == Stability.STABLE

    def test_stability_after_key_deleted(self):
        HANDLER_STABILITY["Temporary"] = Stability.PREVIEW
        del HANDLER_STABILITY["Temporary"]
        assert get_handler_stability("Temporary") == Stability.EXPERIMENTAL


# ---------------------------------------------------------------------------
# get_all_handler_stability()
# ---------------------------------------------------------------------------


class TestGetAllHandlerStability:
    """Test get_all_handler_stability serialization."""

    def test_empty_registry_returns_empty_dict(self):
        result = get_all_handler_stability()
        assert result == {}

    def test_returns_string_values_not_enums(self):
        HANDLER_STABILITY["H1"] = Stability.STABLE
        result = get_all_handler_stability()
        assert result["H1"] == "stable"
        assert isinstance(result["H1"], str)

    def test_all_stability_levels_serialized(self):
        HANDLER_STABILITY["S"] = Stability.STABLE
        HANDLER_STABILITY["E"] = Stability.EXPERIMENTAL
        HANDLER_STABILITY["P"] = Stability.PREVIEW
        HANDLER_STABILITY["D"] = Stability.DEPRECATED
        result = get_all_handler_stability()
        assert result == {
            "S": "stable",
            "E": "experimental",
            "P": "preview",
            "D": "deprecated",
        }

    def test_returns_new_dict_each_call(self):
        HANDLER_STABILITY["A"] = Stability.STABLE
        r1 = get_all_handler_stability()
        r2 = get_all_handler_stability()
        assert r1 == r2
        assert r1 is not r2

    def test_result_is_plain_dict(self):
        HANDLER_STABILITY["X"] = Stability.PREVIEW
        result = get_all_handler_stability()
        assert isinstance(result, dict)

    def test_reflects_current_state(self):
        """Verify function always reads live state of HANDLER_STABILITY."""
        result1 = get_all_handler_stability()
        assert result1 == {}

        HANDLER_STABILITY["Added"] = Stability.STABLE
        result2 = get_all_handler_stability()
        assert "Added" in result2
        assert result2["Added"] == "stable"

    def test_many_handlers(self):
        for i in range(50):
            HANDLER_STABILITY[f"Handler{i}"] = Stability.STABLE
        result = get_all_handler_stability()
        assert len(result) == 50
        assert all(v == "stable" for v in result.values())

    def test_mutating_result_does_not_affect_source(self):
        HANDLER_STABILITY["Keep"] = Stability.STABLE
        result = get_all_handler_stability()
        result["Keep"] = "tampered"
        # Source should be unaffected
        assert HANDLER_STABILITY["Keep"] == Stability.STABLE

    def test_handler_names_preserved_exactly(self):
        HANDLER_STABILITY["CamelCaseHandler"] = Stability.STABLE
        HANDLER_STABILITY["snake_case_handler"] = Stability.EXPERIMENTAL
        HANDLER_STABILITY["UPPER"] = Stability.PREVIEW
        result = get_all_handler_stability()
        assert set(result.keys()) == {"CamelCaseHandler", "snake_case_handler", "UPPER"}


# ---------------------------------------------------------------------------
# Integration: both data structures together
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Test ALL_HANDLERS and HANDLER_STABILITY used together."""

    def test_handler_in_list_but_not_stability(self):
        """A handler can exist in ALL_HANDLERS without a stability entry."""
        cls = type("OrphanHandler", (), {})
        ALL_HANDLERS.append(cls)
        assert len(ALL_HANDLERS) == 1
        assert get_handler_stability("OrphanHandler") == Stability.EXPERIMENTAL

    def test_stability_without_handler_class(self):
        """A stability entry can exist without a corresponding class in ALL_HANDLERS."""
        HANDLER_STABILITY["GhostHandler"] = Stability.STABLE
        assert len(ALL_HANDLERS) == 0
        assert get_handler_stability("GhostHandler") == Stability.STABLE

    def test_full_registration_pattern(self):
        """Simulate the pattern used by __init__.py to populate both."""
        handler_cls = type("DebatesHandler", (), {})
        ALL_HANDLERS.append(handler_cls)
        HANDLER_STABILITY["DebatesHandler"] = Stability.STABLE

        assert len(ALL_HANDLERS) == 1
        assert get_handler_stability("DebatesHandler") == Stability.STABLE
        result = get_all_handler_stability()
        assert result == {"DebatesHandler": "stable"}

    def test_clear_both_resets_state(self):
        cls = type("H", (), {})
        ALL_HANDLERS.append(cls)
        HANDLER_STABILITY["H"] = Stability.STABLE

        ALL_HANDLERS.clear()
        HANDLER_STABILITY.clear()

        assert ALL_HANDLERS == []
        assert HANDLER_STABILITY == {}
        assert get_handler_stability("H") == Stability.EXPERIMENTAL
        assert get_all_handler_stability() == {}
