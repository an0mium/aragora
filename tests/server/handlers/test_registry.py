"""
Tests for aragora.server.handlers._registry.

Tests cover:
1. get_handler_stability - Stability lookup for handler names
2. get_all_handler_stability - Exporting all stability levels as strings
3. ALL_HANDLERS and HANDLER_STABILITY module-level lists
"""

from __future__ import annotations

import pytest

from aragora.config.stability import Stability
from aragora.server.handlers._registry import (
    ALL_HANDLERS,
    HANDLER_STABILITY,
    get_all_handler_stability,
    get_handler_stability,
)


class TestGetHandlerStability:
    """Test get_handler_stability function."""

    def test_unknown_handler_returns_experimental(self):
        """Unknown handler names should default to EXPERIMENTAL."""
        result = get_handler_stability("NonexistentHandler")
        assert result == Stability.EXPERIMENTAL

    def test_returns_stability_enum(self):
        """Return value should be a Stability enum member."""
        result = get_handler_stability("anything")
        assert isinstance(result, Stability)

    def test_registered_handler_returns_correct_stability(self):
        """If a handler is registered, its actual stability should be returned."""
        # Temporarily register a handler
        HANDLER_STABILITY["TestStableHandler"] = Stability.STABLE
        try:
            result = get_handler_stability("TestStableHandler")
            assert result == Stability.STABLE
        finally:
            del HANDLER_STABILITY["TestStableHandler"]

    def test_deprecated_handler_stability(self):
        """Deprecated handlers should return DEPRECATED."""
        HANDLER_STABILITY["OldHandler"] = Stability.DEPRECATED
        try:
            assert get_handler_stability("OldHandler") == Stability.DEPRECATED
        finally:
            del HANDLER_STABILITY["OldHandler"]

    def test_preview_handler_stability(self):
        """Preview handlers should return PREVIEW."""
        HANDLER_STABILITY["PreviewHandler"] = Stability.PREVIEW
        try:
            assert get_handler_stability("PreviewHandler") == Stability.PREVIEW
        finally:
            del HANDLER_STABILITY["PreviewHandler"]


class TestGetAllHandlerStability:
    """Test get_all_handler_stability function."""

    def test_returns_dict(self):
        result = get_all_handler_stability()
        assert isinstance(result, dict)

    def test_values_are_strings(self):
        """All values should be string representations of Stability enum."""
        # Set up some test data
        HANDLER_STABILITY["StableOne"] = Stability.STABLE
        HANDLER_STABILITY["ExperimentalTwo"] = Stability.EXPERIMENTAL
        try:
            result = get_all_handler_stability()
            assert result["StableOne"] == "stable"
            assert result["ExperimentalTwo"] == "experimental"
        finally:
            del HANDLER_STABILITY["StableOne"]
            del HANDLER_STABILITY["ExperimentalTwo"]

    def test_empty_registry(self):
        """When registry is empty, should return empty dict."""
        saved = dict(HANDLER_STABILITY)
        HANDLER_STABILITY.clear()
        try:
            result = get_all_handler_stability()
            assert result == {}
        finally:
            HANDLER_STABILITY.update(saved)

    def test_keys_match_handler_names(self):
        """Keys should match the registered handler names."""
        HANDLER_STABILITY["HandlerA"] = Stability.STABLE
        HANDLER_STABILITY["HandlerB"] = Stability.PREVIEW
        try:
            result = get_all_handler_stability()
            assert "HandlerA" in result
            assert "HandlerB" in result
        finally:
            del HANDLER_STABILITY["HandlerA"]
            del HANDLER_STABILITY["HandlerB"]


class TestModuleLevelStructures:
    """Test module-level data structures."""

    def test_all_handlers_is_list(self):
        assert isinstance(ALL_HANDLERS, list)

    def test_handler_stability_is_dict(self):
        assert isinstance(HANDLER_STABILITY, dict)
