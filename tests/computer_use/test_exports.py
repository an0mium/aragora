"""Tests for computer_use module exports."""

from __future__ import annotations


class TestComputerUseExports:
    """Test that all expected classes are exported from aragora.computer_use."""

    def test_action_exports(self):
        from aragora.computer_use import (
            Action,
            ActionResult,
            ActionType,
            ClickAction,
            KeyAction,
            ScreenshotAction,
            ScrollAction,
            TypeAction,
            WaitAction,
        )

        assert Action is not None
        assert ActionType.CLICK is not None

    def test_policy_exports(self):
        from aragora.computer_use import (
            ComputerPolicy,
            ComputerPolicyChecker,
            create_default_computer_policy,
            create_readonly_computer_policy,
            create_strict_computer_policy,
        )

        assert ComputerPolicy is not None
        policy = create_default_computer_policy()
        assert policy is not None

    def test_orchestrator_exports(self):
        from aragora.computer_use import (
            ComputerUseConfig,
            ComputerUseMetrics,
            ComputerUseOrchestrator,
            MockActionExecutor,
            StepResult,
            TaskResult,
        )

        assert ComputerUseOrchestrator is not None

    def test_executor_exports(self):
        from aragora.computer_use import ExecutorConfig, PlaywrightActionExecutor

        assert ExecutorConfig is not None
        assert PlaywrightActionExecutor is not None

    def test_bridge_exports(self):
        from aragora.computer_use import (
            BridgeConfig,
            ClaudeComputerUseBridge,
            ConversationMessage,
        )

        assert BridgeConfig is not None
        assert ClaudeComputerUseBridge is not None
        assert ConversationMessage is not None


class TestTopLevelExports:
    """Test that computer_use classes are accessible from aragora package."""

    def test_orchestrator_from_aragora(self):
        from aragora import ComputerUseOrchestrator

        assert ComputerUseOrchestrator is not None

    def test_policy_from_aragora(self):
        from aragora import ComputerPolicy, ComputerPolicyChecker

        assert ComputerPolicy is not None
        assert ComputerPolicyChecker is not None

    def test_executor_from_aragora(self):
        from aragora import ExecutorConfig, PlaywrightActionExecutor

        assert ExecutorConfig is not None
        assert PlaywrightActionExecutor is not None

    def test_bridge_from_aragora(self):
        from aragora import BridgeConfig, ClaudeComputerUseBridge

        assert BridgeConfig is not None
        assert ClaudeComputerUseBridge is not None
