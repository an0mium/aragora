"""
Safe Computer-Use Foundation Module.

Provides controlled access to Claude Computer Use API with:
- Action definitions (screenshot, click, type, scroll, key)
- Policy enforcement for UI actions
- Multi-turn action orchestration
- Airlock-style resilience wrapping
- Full audit trails

Pattern: Claude Computer Use API integration
Safety: All actions validated against policies before execution

Usage:
    from aragora.computer_use import ComputerUseOrchestrator, ComputerPolicy

    # Create orchestrator with policy
    policy = ComputerPolicy.create_default()
    orchestrator = ComputerUseOrchestrator(policy=policy)

    # Execute computer-use task
    result = await orchestrator.run_task(
        goal="Open the settings page and enable dark mode",
        max_steps=10,
    )
"""

from aragora.computer_use.actions import (
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
from aragora.computer_use.policies import (
    ActionRule,
    ComputerPolicy,
    ComputerPolicyChecker,
    DomainRule,
    ElementRule,
    create_default_computer_policy,
    create_readonly_computer_policy,
    create_strict_computer_policy,
)
from aragora.computer_use.orchestrator import (
    ComputerUseConfig,
    ComputerUseMetrics,
    ComputerUseOrchestrator,
    MockActionExecutor,
    StepResult,
    TaskResult,
)

__all__ = [
    # Actions
    "Action",
    "ActionResult",
    "ActionType",
    "ClickAction",
    "KeyAction",
    "ScreenshotAction",
    "ScrollAction",
    "TypeAction",
    "WaitAction",
    # Policies
    "ActionRule",
    "ComputerPolicy",
    "ComputerPolicyChecker",
    "DomainRule",
    "ElementRule",
    "create_default_computer_policy",
    "create_readonly_computer_policy",
    "create_strict_computer_policy",
    # Orchestrator
    "ComputerUseConfig",
    "ComputerUseMetrics",
    "ComputerUseOrchestrator",
    "MockActionExecutor",
    "StepResult",
    "TaskResult",
]
