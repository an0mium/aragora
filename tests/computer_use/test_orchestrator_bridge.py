"""Tests for orchestrator-bridge integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.computer_use.actions import (
    Action,
    ActionResult,
    ActionType,
    ClickAction,
    ScreenshotAction,
)
from aragora.computer_use.orchestrator import (
    ComputerUseOrchestrator,
    MockActionExecutor,
    StepResult,
)


class TestOrchestratorBridgeWiring:
    """Test that orchestrator delegates to bridge when available."""

    def test_init_no_bridge(self):
        """Without bridge or api_key, _bridge is None."""
        o = ComputerUseOrchestrator()
        assert o._bridge is None

    def test_init_with_bridge(self):
        """Explicit bridge is stored."""
        mock_bridge = MagicMock()
        o = ComputerUseOrchestrator(bridge=mock_bridge)
        assert o._bridge is mock_bridge

    def test_init_auto_creates_bridge_from_api_key(self):
        """When api_key is provided, bridge is auto-created."""
        o = ComputerUseOrchestrator(api_key="test-key-123")
        assert o._bridge is not None

    def test_init_no_bridge_without_api_key(self):
        """Without api_key, no bridge is created."""
        o = ComputerUseOrchestrator()
        assert o._bridge is None

    @pytest.mark.asyncio
    async def test_get_next_action_delegates_to_bridge(self):
        """_get_next_action should call bridge.get_next_action."""
        mock_bridge = AsyncMock()
        mock_bridge.get_next_action = AsyncMock(
            return_value=(ClickAction(x=100, y=200), "Clicking button", False)
        )

        o = ComputerUseOrchestrator(bridge=mock_bridge)
        action, text, complete = await o._get_next_action(
            goal="Click the button",
            screenshot_b64="base64data",
            previous_steps=[],
            initial_context="test",
        )

        assert isinstance(action, ClickAction)
        assert action.x == 100
        assert text == "Clicking button"
        assert complete is False
        mock_bridge.get_next_action.assert_called_once_with(
            goal="Click the button",
            screenshot_b64="base64data",
            previous_steps=[],
            initial_context="test",
        )

    @pytest.mark.asyncio
    async def test_get_next_action_bridge_completion(self):
        """Bridge signals completion via (None, text, True)."""
        mock_bridge = AsyncMock()
        mock_bridge.get_next_action = AsyncMock(return_value=(None, "Task is done", True))

        o = ComputerUseOrchestrator(bridge=mock_bridge)
        action, text, complete = await o._get_next_action(
            goal="test",
            screenshot_b64="data",
            previous_steps=[MagicMock()],
        )

        assert action is None
        assert complete is True

    @pytest.mark.asyncio
    async def test_get_next_action_fallback_stub(self):
        """Without bridge, falls back to stub behavior."""
        o = ComputerUseOrchestrator()

        # First call returns screenshot
        action, text, complete = await o._get_next_action(
            goal="test", screenshot_b64="data", previous_steps=[]
        )
        assert isinstance(action, ScreenshotAction)
        assert complete is False

        # Second call (with steps) signals completion
        action2, text2, complete2 = await o._get_next_action(
            goal="test",
            screenshot_b64="data",
            previous_steps=[MagicMock()],
        )
        assert action2 is None
        assert complete2 is True


class TestOrchestratorBridgeReset:
    """Test that bridge is reset after run_task."""

    @pytest.mark.asyncio
    async def test_bridge_reset_on_task_completion(self):
        """Bridge.reset() is called after run_task completes."""
        mock_bridge = AsyncMock()
        mock_bridge.get_next_action = AsyncMock(return_value=(None, "Done", True))
        mock_bridge.reset = MagicMock()

        executor = MockActionExecutor()
        o = ComputerUseOrchestrator(executor=executor, bridge=mock_bridge)
        await o.run_task(goal="test task", max_steps=5)

        mock_bridge.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_reset_without_bridge(self):
        """No error when bridge is None during cleanup."""
        executor = MockActionExecutor()
        o = ComputerUseOrchestrator(executor=executor)
        # Should not raise
        result = await o.run_task(goal="test task", max_steps=2)
        assert result is not None
