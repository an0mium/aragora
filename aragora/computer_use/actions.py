"""
Computer-Use Action Definitions.

Defines the core actions that can be executed in a computer-use session:
- Screenshot: Capture current screen state
- Click: Mouse click at coordinates
- Type: Keyboard text input
- Key: Special key presses (Enter, Escape, etc.)
- Scroll: Mouse scroll actions
- Wait: Pause for conditions or time

These actions map to Claude's computer_20241022 tool capabilities.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Types of computer-use actions."""

    SCREENSHOT = "screenshot"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    KEY = "key"
    SCROLL = "scroll"
    DRAG = "drag"
    WAIT = "wait"
    MOVE = "move"


class ClickButton(str, Enum):
    """Mouse button for click actions."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class ScrollDirection(str, Enum):
    """Direction for scroll actions."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class ActionResult:
    """Result of executing an action."""

    action_id: str
    action_type: ActionType
    success: bool
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    error: str | None = None
    screenshot_b64: str | None = None  # Base64 encoded screenshot after action
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "success": self.success,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "has_screenshot": self.screenshot_b64 is not None,
            "metadata": self.metadata,
        }


@dataclass
class Action:
    """Base class for all computer-use actions."""

    action_id: str = ""
    action_type: ActionType = ActionType.SCREENSHOT
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.action_id:
            self.action_id = f"action-{uuid.uuid4().hex[:8]}"

    def to_tool_input(self) -> dict[str, Any]:
        """Convert to Claude tool input format."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_tool_use(cls, tool_input: dict[str, Any]) -> Action:
        """Create action from Claude tool_use response."""
        action_name = tool_input.get("action", "screenshot")

        if action_name == "screenshot":
            return ScreenshotAction()
        elif action_name == "click":
            return ClickAction(
                x=tool_input.get("coordinate", [0, 0])[0],
                y=tool_input.get("coordinate", [0, 0])[1],
            )
        elif action_name == "double_click":
            return ClickAction(
                x=tool_input.get("coordinate", [0, 0])[0],
                y=tool_input.get("coordinate", [0, 0])[1],
                double_click=True,
            )
        elif action_name == "type":
            return TypeAction(text=tool_input.get("text", ""))
        elif action_name == "key":
            return KeyAction(key=tool_input.get("text", ""))
        elif action_name == "scroll":
            direction = ScrollDirection.DOWN
            if tool_input.get("coordinate"):
                # Scroll direction based on delta
                delta_y = tool_input.get("coordinate", [0, 0])[1]
                direction = ScrollDirection.UP if delta_y < 0 else ScrollDirection.DOWN
            return ScrollAction(direction=direction)
        elif action_name == "mouse_move":
            return MoveAction(
                x=tool_input.get("coordinate", [0, 0])[0],
                y=tool_input.get("coordinate", [0, 0])[1],
            )
        else:
            return ScreenshotAction()


@dataclass
class ScreenshotAction(Action):
    """Capture a screenshot of the current screen state."""

    action_type: ActionType = field(default=ActionType.SCREENSHOT, init=False)

    def to_tool_input(self) -> dict[str, Any]:
        """Convert to Claude tool input format."""
        return {"action": "screenshot"}


@dataclass
class ClickAction(Action):
    """Click at specific screen coordinates."""

    x: int = 0
    y: int = 0
    button: ClickButton = ClickButton.LEFT
    double_click: bool = False
    action_type: ActionType = field(default=ActionType.CLICK, init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.double_click:
            self.action_type = ActionType.DOUBLE_CLICK

    def to_tool_input(self) -> dict[str, Any]:
        """Convert to Claude tool input format."""
        action_name = "double_click" if self.double_click else "click"
        if self.button == ClickButton.RIGHT:
            action_name = "right_click"
        return {
            "action": action_name,
            "coordinate": [self.x, self.y],
        }


@dataclass
class TypeAction(Action):
    """Type text using the keyboard."""

    text: str = ""
    action_type: ActionType = field(default=ActionType.TYPE, init=False)

    def to_tool_input(self) -> dict[str, Any]:
        """Convert to Claude tool input format."""
        return {
            "action": "type",
            "text": self.text,
        }


@dataclass
class KeyAction(Action):
    """Press a special key or key combination."""

    key: str = ""  # e.g., "Return", "Escape", "ctrl+c", "alt+Tab"
    action_type: ActionType = field(default=ActionType.KEY, init=False)

    def to_tool_input(self) -> dict[str, Any]:
        """Convert to Claude tool input format."""
        return {
            "action": "key",
            "text": self.key,
        }


@dataclass
class ScrollAction(Action):
    """Scroll the screen or a specific element."""

    direction: ScrollDirection = ScrollDirection.DOWN
    amount: int = 3  # Number of scroll units
    x: int | None = None  # Optional coordinates for scroll position
    y: int | None = None
    action_type: ActionType = field(default=ActionType.SCROLL, init=False)

    def to_tool_input(self) -> dict[str, Any]:
        """Convert to Claude tool input format."""
        # Map direction to coordinate delta
        delta_map = {
            ScrollDirection.UP: (0, -self.amount * 100),
            ScrollDirection.DOWN: (0, self.amount * 100),
            ScrollDirection.LEFT: (-self.amount * 100, 0),
            ScrollDirection.RIGHT: (self.amount * 100, 0),
        }
        delta = delta_map[self.direction]

        result: dict[str, Any] = {
            "action": "scroll",
            "coordinate": list(delta),
        }
        return result


@dataclass
class MoveAction(Action):
    """Move the mouse cursor to coordinates."""

    x: int = 0
    y: int = 0
    action_type: ActionType = field(default=ActionType.MOVE, init=False)

    def to_tool_input(self) -> dict[str, Any]:
        """Convert to Claude tool input format."""
        return {
            "action": "mouse_move",
            "coordinate": [self.x, self.y],
        }


@dataclass
class DragAction(Action):
    """Drag from one position to another."""

    start_x: int = 0
    start_y: int = 0
    end_x: int = 0
    end_y: int = 0
    action_type: ActionType = field(default=ActionType.DRAG, init=False)

    def to_tool_input(self) -> dict[str, Any]:
        """Convert to Claude tool input format."""
        return {
            "action": "drag",
            "start_coordinate": [self.start_x, self.start_y],
            "end_coordinate": [self.end_x, self.end_y],
        }


@dataclass
class WaitAction(Action):
    """Wait for a condition or fixed time."""

    duration_ms: int = 1000  # Fixed wait time
    wait_for: str | None = None  # Optional condition description
    action_type: ActionType = field(default=ActionType.WAIT, init=False)

    def to_tool_input(self) -> dict[str, Any]:
        """Convert to Claude tool input format (no direct equivalent)."""
        return {
            "action": "wait",
            "duration_ms": self.duration_ms,
            "wait_for": self.wait_for,
        }


# Common key constants for KeyAction
class Keys:
    """Common key names for KeyAction."""

    ENTER = "Return"
    ESCAPE = "Escape"
    TAB = "Tab"
    BACKSPACE = "BackSpace"
    DELETE = "Delete"
    SPACE = "space"
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"
    HOME = "Home"
    END = "End"
    PAGE_UP = "Page_Up"
    PAGE_DOWN = "Page_Down"

    # Modifier combinations
    CTRL_A = "ctrl+a"
    CTRL_C = "ctrl+c"
    CTRL_V = "ctrl+v"
    CTRL_X = "ctrl+x"
    CTRL_Z = "ctrl+z"
    CTRL_S = "ctrl+s"
    ALT_TAB = "alt+Tab"
    ALT_F4 = "alt+F4"


__all__ = [
    "Action",
    "ActionResult",
    "ActionType",
    "ClickAction",
    "ClickButton",
    "DragAction",
    "KeyAction",
    "Keys",
    "MoveAction",
    "ScreenshotAction",
    "ScrollAction",
    "ScrollDirection",
    "TypeAction",
    "WaitAction",
]
