"""
Computer-Use Orchestrator.

Manages multi-turn computer-use sessions with:
- Claude API integration for tool calling
- Policy enforcement per action
- Screenshot capture and validation
- Error recovery and retry logic
- Session metrics and audit trails

Pattern: Agentic loop with tool calling
Inspired by: Anthropic Computer Use demo
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from aragora.computer_use.actions import (
    Action,
    ActionResult,
    ScreenshotAction,
)
from aragora.computer_use.policies import (
    ComputerPolicy,
    ComputerPolicyChecker,
    create_default_computer_policy,
)

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Status of a single step in the task."""

    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"  # Policy denied
    TIMEOUT = "timeout"


class TaskStatus(str, Enum):
    """Status of the overall task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class StepResult:
    """Result of a single step in the task."""

    step_number: int
    action: Action
    result: ActionResult
    status: StepStatus
    model_response: str = ""
    policy_check_passed: bool = True
    policy_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_number": self.step_number,
            "action": self.action.to_dict(),
            "result": self.result.to_dict(),
            "status": self.status.value,
            "model_response": self.model_response,
            "policy_check_passed": self.policy_check_passed,
            "policy_reason": self.policy_reason,
        }


@dataclass
class TaskResult:
    """Result of executing a computer-use task."""

    task_id: str
    goal: str
    status: TaskStatus
    steps: list[StepResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    final_screenshot_b64: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": (self.end_time - self.start_time) if self.end_time else None,
            "has_final_screenshot": self.final_screenshot_b64 is not None,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ComputerUseMetrics:
    """Metrics for computer-use sessions."""

    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    policy_blocked_actions: int = 0
    total_latency_ms: float = 0.0

    @property
    def task_success_rate(self) -> float:
        """Task success rate as percentage."""
        if self.total_tasks == 0:
            return 100.0
        return (self.successful_tasks / self.total_tasks) * 100

    @property
    def action_success_rate(self) -> float:
        """Action success rate as percentage."""
        if self.total_actions == 0:
            return 100.0
        return (self.successful_actions / self.total_actions) * 100

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "task_success_rate": round(self.task_success_rate, 2),
            "total_actions": self.total_actions,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "policy_blocked_actions": self.policy_blocked_actions,
            "action_success_rate": round(self.action_success_rate, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


@dataclass
class ComputerUseConfig:
    """Configuration for the orchestrator."""

    # Model settings
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.0

    # Display settings
    display_width: int = 1920
    display_height: int = 1080

    # Timeout settings
    action_timeout_seconds: float = 10.0
    total_timeout_seconds: float = 300.0

    # Retry settings
    max_retries_per_action: int = 2
    retry_delay_seconds: float = 1.0

    # Step limits
    max_steps: int = 50

    # Screenshot settings
    take_screenshot_after_action: bool = True
    screenshot_delay_ms: int = 500

    # Human approval callback
    require_approval_callback: Callable[[Action], bool] | None = None


class ActionExecutor(Protocol):
    """Protocol for executing actions on the computer."""

    async def execute(self, action: Action) -> ActionResult:
        """Execute an action and return the result."""
        ...

    async def take_screenshot(self) -> str:
        """Take a screenshot and return base64-encoded image."""
        ...

    async def get_current_url(self) -> str | None:
        """Get current browser URL if applicable."""
        ...


class ComputerUseOrchestrator:
    """
    Orchestrates multi-turn computer-use sessions.

    Manages the agentic loop of:
    1. Take screenshot
    2. Send to Claude with goal
    3. Receive action from Claude
    4. Validate action against policy
    5. Execute action
    6. Repeat until goal achieved or limits reached

    Usage:
        executor = PlaywrightExecutor()  # Or other implementation
        policy = create_default_computer_policy()
        orchestrator = ComputerUseOrchestrator(
            executor=executor,
            policy=policy,
        )

        result = await orchestrator.run_task(
            goal="Open settings and enable dark mode",
            max_steps=10,
        )
    """

    def __init__(
        self,
        executor: ActionExecutor | None = None,
        policy: ComputerPolicy | None = None,
        config: ComputerUseConfig | None = None,
        api_key: str | None = None,
        bridge: Any | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            executor: Action executor implementation
            policy: Computer-use policy
            config: Orchestrator configuration
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            bridge: ClaudeComputerUseBridge instance for Claude API integration.
                    If not provided but api_key is given, one is created automatically.
        """
        self._executor = executor
        self._policy = policy or create_default_computer_policy()
        self._config = config or ComputerUseConfig()
        self._api_key = api_key
        self._policy_checker = ComputerPolicyChecker(self._policy)
        self._metrics = ComputerUseMetrics()
        self._current_task: TaskResult | None = None
        self._bridge = bridge

        # Auto-create bridge if api_key provided but no bridge
        if self._bridge is None and self._api_key:
            try:
                from aragora.computer_use.claude_bridge import (
                    BridgeConfig,
                    ClaudeComputerUseBridge,
                )

                self._bridge = ClaudeComputerUseBridge(
                    api_key=self._api_key,
                    config=BridgeConfig(
                        display_width=self._config.display_width,
                        display_height=self._config.display_height,
                    ),
                )
            except ImportError:
                logger.warning("Claude bridge unavailable - using stub for _get_next_action")

    @property
    def metrics(self) -> ComputerUseMetrics:
        """Get session metrics."""
        return self._metrics

    @property
    def policy(self) -> ComputerPolicy:
        """Get the policy."""
        return self._policy

    async def run_task(
        self,
        goal: str,
        max_steps: int | None = None,
        initial_context: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskResult:
        """
        Execute a computer-use task.

        Args:
            goal: Natural language description of the goal
            max_steps: Maximum steps (overrides config)
            initial_context: Additional context for Claude
            metadata: Optional metadata to attach

        Returns:
            TaskResult with all step details
        """
        if not self._executor:
            raise RuntimeError("No executor configured")

        task_id = f"task-{uuid.uuid4().hex[:8]}"
        max_steps = max_steps or self._config.max_steps

        result = TaskResult(
            task_id=task_id,
            goal=goal,
            status=TaskStatus.RUNNING,
            metadata=metadata or {},
        )
        self._current_task = result
        self._metrics.total_tasks += 1

        logger.info(f"Starting computer-use task: {task_id} - {goal}")

        try:
            # Initial screenshot
            screenshot_b64 = await self._executor.take_screenshot()

            step_number = 0
            while step_number < max_steps:
                # Check total timeout
                elapsed = time.time() - result.start_time
                if elapsed > self._config.total_timeout_seconds:
                    result.status = TaskStatus.TIMEOUT
                    result.error = f"Total timeout exceeded ({self._config.total_timeout_seconds}s)"
                    break

                # Get current URL for policy checks
                current_url = await self._executor.get_current_url()

                # Call Claude to get next action
                step_number += 1
                action, model_response, is_complete = await self._get_next_action(
                    goal=goal,
                    screenshot_b64=screenshot_b64,
                    previous_steps=result.steps,
                    initial_context=initial_context,
                )

                if is_complete:
                    result.status = TaskStatus.COMPLETED
                    result.final_screenshot_b64 = screenshot_b64
                    logger.info(f"Task {task_id} completed successfully")
                    break

                if action is None:
                    # Model indicated completion or confusion
                    result.status = TaskStatus.COMPLETED
                    result.final_screenshot_b64 = screenshot_b64
                    break

                # Policy check
                allowed, reason = self._policy_checker.check_action(action, current_url)

                if not allowed:
                    step_result = StepResult(
                        step_number=step_number,
                        action=action,
                        result=ActionResult(
                            action_id=action.action_id,
                            action_type=action.action_type,
                            success=False,
                            error=f"Policy denied: {reason}",
                        ),
                        status=StepStatus.BLOCKED,
                        model_response=model_response,
                        policy_check_passed=False,
                        policy_reason=reason,
                    )
                    result.steps.append(step_result)
                    self._metrics.policy_blocked_actions += 1
                    self._policy_checker.record_error()

                    # Try to continue with screenshot for context
                    await asyncio.sleep(0.5)
                    screenshot_b64 = await self._executor.take_screenshot()
                    continue

                # Human approval if required
                if self._config.require_approval_callback:
                    if not self._config.require_approval_callback(action):
                        step_result = StepResult(
                            step_number=step_number,
                            action=action,
                            result=ActionResult(
                                action_id=action.action_id,
                                action_type=action.action_type,
                                success=False,
                                error="Human approval denied",
                            ),
                            status=StepStatus.BLOCKED,
                            model_response=model_response,
                            policy_check_passed=True,
                            policy_reason="human approval required",
                        )
                        result.steps.append(step_result)
                        continue

                # Execute action
                self._metrics.total_actions += 1
                action_result = await self._execute_with_timeout(action)

                step_status = StepStatus.SUCCESS if action_result.success else StepStatus.FAILED

                step_result = StepResult(
                    step_number=step_number,
                    action=action,
                    result=action_result,
                    status=step_status,
                    model_response=model_response,
                    policy_check_passed=True,
                )
                result.steps.append(step_result)

                if action_result.success:
                    self._metrics.successful_actions += 1
                    self._policy_checker.record_success()
                else:
                    self._metrics.failed_actions += 1
                    self._policy_checker.record_error()

                # Take screenshot after action
                if self._config.take_screenshot_after_action:
                    await asyncio.sleep(self._config.screenshot_delay_ms / 1000)
                    screenshot_b64 = await self._executor.take_screenshot()
                    action_result.screenshot_b64 = screenshot_b64

            # End of loop
            if result.status == TaskStatus.RUNNING:
                result.status = TaskStatus.COMPLETED
                result.final_screenshot_b64 = screenshot_b64

            self._metrics.successful_tasks += 1

        except asyncio.TimeoutError:
            result.status = TaskStatus.TIMEOUT
            result.error = "Task timeout"
            self._metrics.failed_tasks += 1

        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            self._metrics.failed_tasks += 1
            logger.exception(f"Task {task_id} failed: {e}")

        finally:
            result.end_time = time.time()
            self._current_task = None
            self._policy_checker.reset()
            if self._bridge is not None and hasattr(self._bridge, "reset"):
                self._bridge.reset()

        return result

    async def _get_next_action(
        self,
        goal: str,
        screenshot_b64: str,
        previous_steps: list[StepResult],
        initial_context: str = "",
    ) -> tuple[Action | None, str, bool]:
        """
        Call Claude to determine the next action.

        Delegates to ClaudeComputerUseBridge when available, otherwise
        falls back to a stub that completes after the first step.

        Returns:
            (action, model_response, is_complete) tuple
        """
        if self._bridge is not None:
            return await self._bridge.get_next_action(
                goal=goal,
                screenshot_b64=screenshot_b64,
                previous_steps=previous_steps,
                initial_context=initial_context,
            )

        # Fallback stub for testing without API key
        logger.debug("No bridge configured, using stub (completes after first step)")
        if previous_steps:
            return None, "Task appears complete", True
        return ScreenshotAction(), "Taking initial screenshot", False

    async def _execute_with_timeout(self, action: Action) -> ActionResult:
        """Execute action with timeout."""
        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                self._executor.execute(action),  # type: ignore
                timeout=self._config.action_timeout_seconds,
            )
            result.duration_ms = (time.time() - start_time) * 1000
            self._metrics.total_latency_ms += result.duration_ms
            return result

        except asyncio.TimeoutError:
            return ActionResult(
                action_id=action.action_id,
                action_type=action.action_type,
                success=False,
                error=f"Action timeout after {self._config.action_timeout_seconds}s",
            )

    async def cancel_task(self) -> bool:
        """Cancel the current running task."""
        if self._current_task:
            self._current_task.status = TaskStatus.CANCELLED
            self._current_task.end_time = time.time()
            return True
        return False

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get the policy audit log."""
        return self._policy_checker.get_audit_log()


class MockActionExecutor:
    """
    Mock executor for testing.

    Simulates action execution without actual computer control.
    """

    def __init__(self, screenshot_b64: str = ""):
        self._screenshot = screenshot_b64 or self._generate_blank_screenshot()
        self._current_url: str | None = "http://localhost:8080"

    def _generate_blank_screenshot(self) -> str:
        """Generate a minimal valid base64 image."""
        # 1x1 white PNG
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    async def execute(self, action: Action) -> ActionResult:
        """Simulate action execution."""
        # Simulate some delay
        await asyncio.sleep(0.1)

        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=True,
            screenshot_b64=self._screenshot,
        )

    async def take_screenshot(self) -> str:
        """Return mock screenshot."""
        return self._screenshot

    async def get_current_url(self) -> str | None:
        """Return mock URL."""
        return self._current_url


__all__ = [
    "ActionExecutor",
    "ComputerUseConfig",
    "ComputerUseMetrics",
    "ComputerUseOrchestrator",
    "MockActionExecutor",
    "StepResult",
    "StepStatus",
    "TaskResult",
    "TaskStatus",
]
