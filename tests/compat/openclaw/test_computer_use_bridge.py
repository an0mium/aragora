"""Tests for OpenClaw Computer-Use Bridge."""

from __future__ import annotations

import pytest

from aragora.compat.openclaw.computer_use_bridge import (
    ComputerUseBridge,
    ExtractAction,
    NavigateAction,
)
from aragora.computer_use.actions import (
    ClickAction,
    ClickButton,
    KeyAction,
    MoveAction,
    ScreenshotAction,
    ScrollAction,
    ScrollDirection,
    TypeAction,
)


# ---------------------------------------------------------------------------
# Tests: from_openclaw()
# ---------------------------------------------------------------------------


class TestFromOpenClaw:
    """Test ComputerUseBridge.from_openclaw() conversions."""

    def test_navigate_creates_navigate_action(self) -> None:
        """'navigate' should produce a NavigateAction with url and defaults."""
        action = ComputerUseBridge.from_openclaw(
            "navigate",
            {"url": "https://example.com", "wait_for_load": False, "timeout_ms": 5000},
        )

        assert isinstance(action, NavigateAction)
        assert action.url == "https://example.com"
        assert action.wait_for_load is False
        assert action.timeout_ms == 5000

    def test_navigate_defaults(self) -> None:
        """'navigate' with empty params should use sensible defaults."""
        action = ComputerUseBridge.from_openclaw("navigate", {})

        assert isinstance(action, NavigateAction)
        assert action.url == ""
        assert action.wait_for_load is True
        assert action.timeout_ms == 30000

    def test_screenshot_creates_screenshot_action(self) -> None:
        """'screenshot' should produce a ScreenshotAction."""
        action = ComputerUseBridge.from_openclaw("screenshot", {})

        assert isinstance(action, ScreenshotAction)

    def test_click_with_coordinates(self) -> None:
        """'click' should produce a ClickAction with coordinates and button."""
        action = ComputerUseBridge.from_openclaw(
            "click",
            {"coordinate": [100, 200], "button": "right", "double_click": True},
        )

        assert isinstance(action, ClickAction)
        assert action.x == 100
        assert action.y == 200
        assert action.button == ClickButton.RIGHT
        assert action.double_click is True

    def test_click_defaults(self) -> None:
        """'click' with empty params should default to (0, 0) left single click."""
        action = ComputerUseBridge.from_openclaw("click", {})

        assert isinstance(action, ClickAction)
        assert action.x == 0
        assert action.y == 0
        assert action.button == ClickButton.LEFT
        assert action.double_click is False

    def test_type_with_text(self) -> None:
        """'type' should produce a TypeAction with the given text."""
        action = ComputerUseBridge.from_openclaw("type", {"text": "hello world"})

        assert isinstance(action, TypeAction)
        assert action.text == "hello world"

    def test_type_empty_text(self) -> None:
        """'type' with missing text should default to empty string."""
        action = ComputerUseBridge.from_openclaw("type", {})

        assert isinstance(action, TypeAction)
        assert action.text == ""

    def test_key_with_key(self) -> None:
        """'key' should produce a KeyAction with the specified key."""
        action = ComputerUseBridge.from_openclaw("key", {"key": "Return"})

        assert isinstance(action, KeyAction)
        assert action.key == "Return"

    def test_scroll_with_direction(self) -> None:
        """'scroll' should produce a ScrollAction with the given direction."""
        action = ComputerUseBridge.from_openclaw("scroll", {"direction": "up"})

        assert isinstance(action, ScrollAction)
        assert action.direction == ScrollDirection.UP

    def test_scroll_default_direction(self) -> None:
        """'scroll' with no direction should default to DOWN."""
        action = ComputerUseBridge.from_openclaw("scroll", {})

        assert isinstance(action, ScrollAction)
        assert action.direction == ScrollDirection.DOWN

    def test_scroll_invalid_direction_defaults_to_down(self) -> None:
        """'scroll' with an invalid direction string should default to DOWN."""
        action = ComputerUseBridge.from_openclaw("scroll", {"direction": "diagonal"})

        assert isinstance(action, ScrollAction)
        assert action.direction == ScrollDirection.DOWN

    def test_move_with_coordinates(self) -> None:
        """'move' should produce a MoveAction with the given coordinates."""
        action = ComputerUseBridge.from_openclaw("move", {"coordinate": [300, 400]})

        assert isinstance(action, MoveAction)
        assert action.x == 300
        assert action.y == 400

    def test_extract_creates_extract_action(self) -> None:
        """'extract' should produce an ExtractAction with selector and type."""
        action = ComputerUseBridge.from_openclaw(
            "extract",
            {
                "selector": "div.content",
                "extract_type": "html",
                "region": {"x": 0, "y": 0, "width": 100, "height": 100},
            },
        )

        assert isinstance(action, ExtractAction)
        assert action.selector == "div.content"
        assert action.extract_type == "html"
        assert action.region == {"x": 0, "y": 0, "width": 100, "height": 100}

    def test_extract_defaults(self) -> None:
        """'extract' with empty params should use sensible defaults."""
        action = ComputerUseBridge.from_openclaw("extract", {})

        assert isinstance(action, ExtractAction)
        assert action.selector == ""
        assert action.extract_type == "text"
        assert action.region is None

    def test_unknown_action_defaults_to_screenshot(self) -> None:
        """An unrecognized action type should default to ScreenshotAction."""
        action = ComputerUseBridge.from_openclaw("magic_wand", {"x": 42})

        assert isinstance(action, ScreenshotAction)

    def test_action_type_case_insensitive(self) -> None:
        """Action type lookup should be case insensitive."""
        action = ComputerUseBridge.from_openclaw("SCREENSHOT", {})
        assert isinstance(action, ScreenshotAction)

        action2 = ComputerUseBridge.from_openclaw("Navigate", {"url": "https://x.com"})
        assert isinstance(action2, NavigateAction)

    def test_action_type_stripped(self) -> None:
        """Leading/trailing whitespace on action type should be stripped."""
        action = ComputerUseBridge.from_openclaw("  click  ", {"coordinate": [10, 20]})
        assert isinstance(action, ClickAction)
        assert action.x == 10


# ---------------------------------------------------------------------------
# Tests: to_openclaw()
# ---------------------------------------------------------------------------


class TestToOpenClaw:
    """Test ComputerUseBridge.to_openclaw() conversions."""

    def test_navigate_action_to_openclaw(self) -> None:
        """NavigateAction should convert to an OpenClaw dict with url and action."""
        nav = NavigateAction(url="https://example.com", wait_for_load=True, timeout_ms=10000)
        result = ComputerUseBridge.to_openclaw(nav)

        assert result["action"] == "navigate"
        assert result["url"] == "https://example.com"
        assert result["wait_for_load"] is True
        assert result["timeout_ms"] == 10000

    def test_click_action_to_openclaw(self) -> None:
        """ClickAction should convert to an OpenClaw dict with coordinate."""
        click = ClickAction(x=150, y=250)
        result = ComputerUseBridge.to_openclaw(click)

        assert result["action"] == "click"
        assert result["coordinate"] == [150, 250]
        # Default left click should not include button or double_click
        assert "button" not in result
        assert "double_click" not in result

    def test_click_action_double_click_to_openclaw(self) -> None:
        """ClickAction with double_click should include that in the output."""
        click = ClickAction(x=10, y=20, double_click=True)
        result = ComputerUseBridge.to_openclaw(click)

        assert result["action"] == "click"
        assert result["double_click"] is True

    def test_click_action_right_button_to_openclaw(self) -> None:
        """ClickAction with right button should include button in the output."""
        click = ClickAction(x=10, y=20, button=ClickButton.RIGHT)
        result = ComputerUseBridge.to_openclaw(click)

        assert result["action"] == "click"
        assert result["button"] == "right"

    def test_type_action_to_openclaw(self) -> None:
        """TypeAction should convert to an OpenClaw dict with text."""
        type_act = TypeAction(text="test input")
        result = ComputerUseBridge.to_openclaw(type_act)

        assert result["action"] == "type"
        assert result["text"] == "test input"

    def test_screenshot_action_to_openclaw(self) -> None:
        """ScreenshotAction should convert to a simple dict."""
        screenshot = ScreenshotAction()
        result = ComputerUseBridge.to_openclaw(screenshot)

        assert result == {"action": "screenshot"}

    def test_key_action_to_openclaw(self) -> None:
        """KeyAction should convert to an OpenClaw dict with key."""
        key_act = KeyAction(key="Escape")
        result = ComputerUseBridge.to_openclaw(key_act)

        assert result["action"] == "key"
        assert result["key"] == "Escape"

    def test_scroll_action_to_openclaw(self) -> None:
        """ScrollAction should convert to an OpenClaw dict with direction."""
        scroll = ScrollAction(direction=ScrollDirection.LEFT)
        result = ComputerUseBridge.to_openclaw(scroll)

        assert result["action"] == "scroll"
        assert result["direction"] == "left"

    def test_move_action_to_openclaw(self) -> None:
        """MoveAction should convert to an OpenClaw dict with coordinate."""
        move = MoveAction(x=500, y=600)
        result = ComputerUseBridge.to_openclaw(move)

        assert result["action"] == "move"
        assert result["coordinate"] == [500, 600]

    def test_extract_action_to_openclaw(self) -> None:
        """ExtractAction should convert to an OpenClaw dict with selector and type."""
        extract = ExtractAction(selector="h1", extract_type="text")
        result = ComputerUseBridge.to_openclaw(extract)

        assert result["action"] == "extract"
        assert result["selector"] == "h1"
        assert result["extract_type"] == "text"

    def test_extract_action_with_region_to_openclaw(self) -> None:
        """ExtractAction with region should include it in the output."""
        extract = ExtractAction(
            selector="img",
            extract_type="screenshot_region",
            region={"x": 10, "y": 20, "width": 200, "height": 300},
        )
        result = ComputerUseBridge.to_openclaw(extract)

        assert result["region"] == {"x": 10, "y": 20, "width": 200, "height": 300}

    def test_extract_action_without_region_to_openclaw(self) -> None:
        """ExtractAction without region should not include 'region' key."""
        extract = ExtractAction(selector="p", extract_type="text")
        result = ComputerUseBridge.to_openclaw(extract)

        assert "region" not in result


# ---------------------------------------------------------------------------
# Tests: Round-trip conversion
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Test that from_openclaw -> to_openclaw preserves data."""

    def test_roundtrip_navigate(self) -> None:
        """Navigate action should survive a round trip."""
        params = {"url": "https://example.com", "wait_for_load": True, "timeout_ms": 30000}
        action = ComputerUseBridge.from_openclaw("navigate", params)
        result = ComputerUseBridge.to_openclaw(action)

        assert result["action"] == "navigate"
        assert result["url"] == params["url"]
        assert result["wait_for_load"] == params["wait_for_load"]
        assert result["timeout_ms"] == params["timeout_ms"]

    def test_roundtrip_click(self) -> None:
        """Click action should survive a round trip."""
        params = {"coordinate": [100, 200]}
        action = ComputerUseBridge.from_openclaw("click", params)
        result = ComputerUseBridge.to_openclaw(action)

        assert result["action"] == "click"
        assert result["coordinate"] == [100, 200]

    def test_roundtrip_type(self) -> None:
        """Type action should survive a round trip."""
        params = {"text": "hello"}
        action = ComputerUseBridge.from_openclaw("type", params)
        result = ComputerUseBridge.to_openclaw(action)

        assert result["action"] == "type"
        assert result["text"] == "hello"

    def test_roundtrip_scroll(self) -> None:
        """Scroll action should survive a round trip."""
        params = {"direction": "right"}
        action = ComputerUseBridge.from_openclaw("scroll", params)
        result = ComputerUseBridge.to_openclaw(action)

        assert result["action"] == "scroll"
        assert result["direction"] == "right"

    def test_roundtrip_move(self) -> None:
        """Move action should survive a round trip."""
        params = {"coordinate": [42, 84]}
        action = ComputerUseBridge.from_openclaw("move", params)
        result = ComputerUseBridge.to_openclaw(action)

        assert result["action"] == "move"
        assert result["coordinate"] == [42, 84]


# ---------------------------------------------------------------------------
# Tests: supported_openclaw_actions()
# ---------------------------------------------------------------------------


class TestSupportedActions:
    """Test ComputerUseBridge.supported_openclaw_actions()."""

    def test_returns_all_eight_action_types(self) -> None:
        """supported_openclaw_actions() should list all 8 supported types."""
        actions = ComputerUseBridge.supported_openclaw_actions()

        assert len(actions) == 8
        expected = {"navigate", "extract", "screenshot", "click", "type", "key", "scroll", "move"}
        assert set(actions) == expected

    def test_returns_list_of_strings(self) -> None:
        """Each item should be a string."""
        actions = ComputerUseBridge.supported_openclaw_actions()

        for action in actions:
            assert isinstance(action, str)


# ---------------------------------------------------------------------------
# Tests: NavigateAction and ExtractAction dataclasses
# ---------------------------------------------------------------------------


class TestNavigateActionDataclass:
    """Test NavigateAction standalone behavior."""

    def test_defaults(self) -> None:
        """NavigateAction should have sensible defaults."""
        nav = NavigateAction()
        assert nav.url == ""
        assert nav.wait_for_load is True
        assert nav.timeout_ms == 30000

    def test_to_openclaw_params(self) -> None:
        """to_openclaw_params should return a complete dict."""
        nav = NavigateAction(url="https://test.com", wait_for_load=False, timeout_ms=5000)
        params = nav.to_openclaw_params()

        assert params == {
            "action": "navigate",
            "url": "https://test.com",
            "wait_for_load": False,
            "timeout_ms": 5000,
        }


class TestExtractActionDataclass:
    """Test ExtractAction standalone behavior."""

    def test_defaults(self) -> None:
        """ExtractAction should have sensible defaults."""
        ext = ExtractAction()
        assert ext.selector == ""
        assert ext.extract_type == "text"
        assert ext.region is None

    def test_to_openclaw_params_without_region(self) -> None:
        """to_openclaw_params without region should not include region key."""
        ext = ExtractAction(selector="div", extract_type="html")
        params = ext.to_openclaw_params()

        assert params == {
            "action": "extract",
            "selector": "div",
            "extract_type": "html",
        }
        assert "region" not in params

    def test_to_openclaw_params_with_region(self) -> None:
        """to_openclaw_params with region should include it."""
        ext = ExtractAction(
            selector="img",
            extract_type="screenshot_region",
            region={"x": 0, "y": 0, "width": 50, "height": 50},
        )
        params = ext.to_openclaw_params()

        assert params["region"] == {"x": 0, "y": 0, "width": 50, "height": 50}
