"""Tests for computer-use action definitions."""

import pytest

from aragora.computer_use.actions import (
    Action,
    ActionResult,
    ActionType,
    ClickAction,
    ClickButton,
    DragAction,
    KeyAction,
    Keys,
    MoveAction,
    ScreenshotAction,
    ScrollAction,
    ScrollDirection,
    TypeAction,
    WaitAction,
)


class TestActionType:
    """Test ActionType enum."""

    def test_all_action_types_defined(self):
        """Verify all expected action types exist."""
        expected = [
            "screenshot",
            "click",
            "double_click",
            "right_click",
            "type",
            "key",
            "scroll",
            "drag",
            "wait",
            "move",
        ]
        actual = [a.value for a in ActionType]
        assert set(expected) == set(actual)

    def test_action_type_is_string_enum(self):
        """Verify action type values are strings."""
        for action_type in ActionType:
            assert isinstance(action_type.value, str)


class TestActionResult:
    """Test ActionResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful result."""
        result = ActionResult(
            action_id="test-123",
            action_type=ActionType.CLICK,
            success=True,
        )
        assert result.action_id == "test-123"
        assert result.action_type == ActionType.CLICK
        assert result.success is True
        assert result.error is None

    def test_create_failure_result(self):
        """Test creating a failed result."""
        result = ActionResult(
            action_id="test-456",
            action_type=ActionType.TYPE,
            success=False,
            error="Element not found",
        )
        assert result.success is False
        assert result.error == "Element not found"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ActionResult(
            action_id="test-789",
            action_type=ActionType.SCREENSHOT,
            success=True,
            duration_ms=150.5,
        )
        data = result.to_dict()
        assert data["action_id"] == "test-789"
        assert data["action_type"] == "screenshot"
        assert data["success"] is True
        assert data["duration_ms"] == 150.5


class TestScreenshotAction:
    """Test ScreenshotAction."""

    def test_create_screenshot_action(self):
        """Test creating a screenshot action."""
        action = ScreenshotAction()
        assert action.action_type == ActionType.SCREENSHOT
        assert action.action_id.startswith("action-")

    def test_to_tool_input(self):
        """Test conversion to Claude tool input."""
        action = ScreenshotAction()
        input_data = action.to_tool_input()
        assert input_data == {"action": "screenshot"}


class TestClickAction:
    """Test ClickAction."""

    def test_create_click_action(self):
        """Test creating a click action."""
        action = ClickAction(x=100, y=200)
        assert action.action_type == ActionType.CLICK
        assert action.x == 100
        assert action.y == 200
        assert action.button == ClickButton.LEFT
        assert action.double_click is False

    def test_double_click_action(self):
        """Test creating a double-click action."""
        action = ClickAction(x=100, y=200, double_click=True)
        assert action.action_type == ActionType.DOUBLE_CLICK
        assert action.double_click is True

    def test_right_click_action(self):
        """Test creating a right-click action."""
        action = ClickAction(x=100, y=200, button=ClickButton.RIGHT)
        assert action.button == ClickButton.RIGHT

    def test_to_tool_input_single_click(self):
        """Test conversion for single click."""
        action = ClickAction(x=150, y=250)
        input_data = action.to_tool_input()
        assert input_data == {"action": "click", "coordinate": [150, 250]}

    def test_to_tool_input_double_click(self):
        """Test conversion for double click."""
        action = ClickAction(x=150, y=250, double_click=True)
        input_data = action.to_tool_input()
        assert input_data == {"action": "double_click", "coordinate": [150, 250]}

    def test_to_tool_input_right_click(self):
        """Test conversion for right click."""
        action = ClickAction(x=150, y=250, button=ClickButton.RIGHT)
        input_data = action.to_tool_input()
        assert input_data == {"action": "right_click", "coordinate": [150, 250]}


class TestTypeAction:
    """Test TypeAction."""

    def test_create_type_action(self):
        """Test creating a type action."""
        action = TypeAction(text="Hello, World!")
        assert action.action_type == ActionType.TYPE
        assert action.text == "Hello, World!"

    def test_empty_text(self):
        """Test creating action with empty text."""
        action = TypeAction()
        assert action.text == ""

    def test_to_tool_input(self):
        """Test conversion to Claude tool input."""
        action = TypeAction(text="test input")
        input_data = action.to_tool_input()
        assert input_data == {"action": "type", "text": "test input"}


class TestKeyAction:
    """Test KeyAction."""

    def test_create_key_action(self):
        """Test creating a key action."""
        action = KeyAction(key=Keys.ENTER)
        assert action.action_type == ActionType.KEY
        assert action.key == "Return"

    def test_key_combination(self):
        """Test key combination."""
        action = KeyAction(key=Keys.CTRL_C)
        assert action.key == "ctrl+c"

    def test_to_tool_input(self):
        """Test conversion to Claude tool input."""
        action = KeyAction(key="Escape")
        input_data = action.to_tool_input()
        assert input_data == {"action": "key", "text": "Escape"}


class TestScrollAction:
    """Test ScrollAction."""

    def test_create_scroll_action(self):
        """Test creating a scroll action."""
        action = ScrollAction(direction=ScrollDirection.DOWN, amount=3)
        assert action.action_type == ActionType.SCROLL
        assert action.direction == ScrollDirection.DOWN
        assert action.amount == 3

    def test_scroll_up(self):
        """Test scroll up direction."""
        action = ScrollAction(direction=ScrollDirection.UP)
        input_data = action.to_tool_input()
        assert input_data["coordinate"][1] < 0  # Negative Y for up

    def test_scroll_down(self):
        """Test scroll down direction."""
        action = ScrollAction(direction=ScrollDirection.DOWN)
        input_data = action.to_tool_input()
        assert input_data["coordinate"][1] > 0  # Positive Y for down


class TestMoveAction:
    """Test MoveAction."""

    def test_create_move_action(self):
        """Test creating a move action."""
        action = MoveAction(x=500, y=300)
        assert action.action_type == ActionType.MOVE
        assert action.x == 500
        assert action.y == 300

    def test_to_tool_input(self):
        """Test conversion to Claude tool input."""
        action = MoveAction(x=100, y=200)
        input_data = action.to_tool_input()
        assert input_data == {"action": "mouse_move", "coordinate": [100, 200]}


class TestDragAction:
    """Test DragAction."""

    def test_create_drag_action(self):
        """Test creating a drag action."""
        action = DragAction(start_x=100, start_y=100, end_x=200, end_y=200)
        assert action.action_type == ActionType.DRAG
        assert action.start_x == 100
        assert action.end_x == 200

    def test_to_tool_input(self):
        """Test conversion to Claude tool input."""
        action = DragAction(start_x=0, start_y=0, end_x=100, end_y=100)
        input_data = action.to_tool_input()
        assert input_data["action"] == "drag"
        assert input_data["start_coordinate"] == [0, 0]
        assert input_data["end_coordinate"] == [100, 100]


class TestWaitAction:
    """Test WaitAction."""

    def test_create_wait_action(self):
        """Test creating a wait action."""
        action = WaitAction(duration_ms=2000)
        assert action.action_type == ActionType.WAIT
        assert action.duration_ms == 2000

    def test_wait_with_condition(self):
        """Test wait with condition."""
        action = WaitAction(duration_ms=1000, wait_for="page load")
        assert action.wait_for == "page load"


class TestActionFromToolUse:
    """Test creating actions from Claude tool_use responses."""

    def test_from_screenshot_tool_use(self):
        """Test parsing screenshot tool use."""
        tool_input = {"action": "screenshot"}
        action = Action.from_tool_use(tool_input)
        assert isinstance(action, ScreenshotAction)

    def test_from_click_tool_use(self):
        """Test parsing click tool use."""
        tool_input = {"action": "click", "coordinate": [100, 200]}
        action = Action.from_tool_use(tool_input)
        assert isinstance(action, ClickAction)
        assert action.x == 100
        assert action.y == 200

    def test_from_type_tool_use(self):
        """Test parsing type tool use."""
        tool_input = {"action": "type", "text": "hello"}
        action = Action.from_tool_use(tool_input)
        assert isinstance(action, TypeAction)
        assert action.text == "hello"

    def test_from_key_tool_use(self):
        """Test parsing key tool use."""
        tool_input = {"action": "key", "text": "Return"}
        action = Action.from_tool_use(tool_input)
        assert isinstance(action, KeyAction)
        assert action.key == "Return"

    def test_from_scroll_tool_use(self):
        """Test parsing scroll tool use."""
        tool_input = {"action": "scroll", "coordinate": [0, 300]}
        action = Action.from_tool_use(tool_input)
        assert isinstance(action, ScrollAction)


class TestKeys:
    """Test Keys constants."""

    def test_common_keys_defined(self):
        """Verify common keys are defined."""
        assert Keys.ENTER == "Return"
        assert Keys.ESCAPE == "Escape"
        assert Keys.TAB == "Tab"
        assert Keys.BACKSPACE == "BackSpace"

    def test_modifier_combinations(self):
        """Test modifier key combinations."""
        assert Keys.CTRL_C == "ctrl+c"
        assert Keys.CTRL_V == "ctrl+v"
        assert Keys.ALT_TAB == "alt+Tab"
