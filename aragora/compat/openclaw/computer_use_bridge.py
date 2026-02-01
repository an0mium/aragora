"""
OpenClaw Computer-Use Bridge.

Maps between OpenClaw browser control actions and Aragora's
computer-use action system.

OpenClaw browser actions:
    BROWSER_NAVIGATE, BROWSER_CLICK, BROWSER_TYPE,
    BROWSER_SCREENSHOT, BROWSER_SCROLL

Aragora computer-use actions:
    ScreenshotAction, ClickAction, TypeAction, KeyAction,
    ScrollAction, MoveAction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from aragora.computer_use.actions import (
    Action,
    ClickAction,
    ClickButton,
    KeyAction,
    MoveAction,
    ScreenshotAction,
    ScrollAction,
    ScrollDirection,
    TypeAction,
)

logger = logging.getLogger(__name__)


@dataclass
class NavigateAction:
    """Browser URL navigation action (extends Aragora actions for OpenClaw)."""

    url: str = ""
    wait_for_load: bool = True
    timeout_ms: int = 30000

    def to_openclaw_params(self) -> dict[str, Any]:
        """Convert to OpenClaw browser action params."""
        return {
            "action": "navigate",
            "url": self.url,
            "wait_for_load": self.wait_for_load,
            "timeout_ms": self.timeout_ms,
        }


@dataclass
class ExtractAction:
    """Content extraction action (extends Aragora actions for OpenClaw)."""

    selector: str = ""
    extract_type: str = "text"  # text, html, screenshot_region
    region: dict[str, int] | None = None  # {x, y, width, height}

    def to_openclaw_params(self) -> dict[str, Any]:
        """Convert to OpenClaw browser action params."""
        result: dict[str, Any] = {
            "action": "extract",
            "selector": self.selector,
            "extract_type": self.extract_type,
        }
        if self.region:
            result["region"] = self.region
        return result


# OpenClaw action name -> converter function
_OPENCLAW_TO_ARAGORA_MAP: dict[str, type] = {
    "screenshot": ScreenshotAction,
    "click": ClickAction,
    "type": TypeAction,
    "key": KeyAction,
    "scroll": ScrollAction,
    "move": MoveAction,
}


class ComputerUseBridge:
    """
    Bridges OpenClaw browser actions and Aragora computer-use actions.

    Provides bidirectional conversion for seamless interop between
    OpenClaw's browser control and Aragora's computer-use system.
    """

    @staticmethod
    def from_openclaw(
        action_type: str, params: dict[str, Any]
    ) -> Action | NavigateAction | ExtractAction:
        """
        Convert an OpenClaw browser action to an Aragora action.

        Args:
            action_type: OpenClaw action type (e.g., "click", "type", "navigate")
            params: OpenClaw action parameters

        Returns:
            Aragora Action instance (or NavigateAction/ExtractAction for browser-specific ops).
        """
        action_type = action_type.lower().strip()

        if action_type == "navigate":
            return NavigateAction(
                url=params.get("url", ""),
                wait_for_load=params.get("wait_for_load", True),
                timeout_ms=params.get("timeout_ms", 30000),
            )

        if action_type == "extract":
            return ExtractAction(
                selector=params.get("selector", ""),
                extract_type=params.get("extract_type", "text"),
                region=params.get("region"),
            )

        if action_type == "screenshot":
            return ScreenshotAction()

        if action_type == "click":
            coords = params.get("coordinate", [0, 0])
            x = coords[0] if len(coords) > 0 else 0
            y = coords[1] if len(coords) > 1 else 0
            return ClickAction(
                x=x,
                y=y,
                button=ClickButton(params.get("button", "left")),
                double_click=params.get("double_click", False),
            )

        if action_type == "type":
            return TypeAction(text=params.get("text", ""))

        if action_type == "key":
            return KeyAction(key=params.get("key", ""))

        if action_type == "scroll":
            direction_str = params.get("direction", "down").lower()
            try:
                direction = ScrollDirection(direction_str)
            except ValueError:
                direction = ScrollDirection.DOWN
            return ScrollAction(direction=direction)

        if action_type == "move":
            coords = params.get("coordinate", [0, 0])
            x = coords[0] if len(coords) > 0 else 0
            y = coords[1] if len(coords) > 1 else 0
            return MoveAction(x=x, y=y)

        logger.warning(f"Unknown OpenClaw action type: {action_type}, defaulting to screenshot")
        return ScreenshotAction()

    @staticmethod
    def to_openclaw(action: Action | NavigateAction | ExtractAction) -> dict[str, Any]:
        """
        Convert an Aragora action to OpenClaw browser action format.

        Args:
            action: An Aragora Action instance.

        Returns:
            Dictionary with OpenClaw action parameters.
        """
        if isinstance(action, NavigateAction):
            return action.to_openclaw_params()

        if isinstance(action, ExtractAction):
            return action.to_openclaw_params()

        if isinstance(action, ScreenshotAction):
            return {"action": "screenshot"}

        if isinstance(action, ClickAction):
            result: dict[str, Any] = {
                "action": "click",
                "coordinate": [action.x, action.y],
            }
            if action.double_click:
                result["double_click"] = True
            if action.button != ClickButton.LEFT:
                result["button"] = action.button.value
            return result

        if isinstance(action, TypeAction):
            return {"action": "type", "text": action.text}

        if isinstance(action, KeyAction):
            return {"action": "key", "key": action.key}

        if isinstance(action, ScrollAction):
            return {"action": "scroll", "direction": action.direction.value}

        if isinstance(action, MoveAction):
            return {"action": "move", "coordinate": [action.x, action.y]}

        logger.warning(f"Unknown action type for OpenClaw conversion: {type(action).__name__}")
        return {"action": "unknown"}

    @staticmethod
    def supported_openclaw_actions() -> list[str]:
        """List all supported OpenClaw browser action types."""
        return [
            "navigate",
            "extract",
            "screenshot",
            "click",
            "type",
            "key",
            "scroll",
            "move",
        ]
