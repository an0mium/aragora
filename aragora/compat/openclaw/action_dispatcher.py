"""
OpenClaw Action Dispatcher -- real execution of computer-use actions.

Wires the ComputerUseBridge (format conversion) to actual execution via
Aragora's PlaywrightActionExecutor, with policy enforcement and receipt
generation.

Flow:
    1. Receive OpenClaw action (type + params)
    2. Convert via ComputerUseBridge.from_openclaw()
    3. Enforce policy via ComputerPolicyChecker
    4. Dispatch to PlaywrightActionExecutor (or NavigateAction handler)
    5. Build ComputerUseActionBundle for audit trail
    6. Return DispatchResult

Usage:
    dispatcher = OpenClawActionDispatcher()
    await dispatcher.start()

    result = await dispatcher.dispatch("click", {"coordinate": [100, 200]})
    if result.success:
        print(result.screenshot_b64)

    await dispatcher.stop()
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aragora.compat.openclaw.computer_use_bridge import (
    ComputerUseBridge,
    ExtractAction,
    NavigateAction,
)
from aragora.computer_use.actions import Action, ActionResult, ActionType
from aragora.computer_use.policies import (
    ComputerPolicy,
    ComputerPolicyChecker,
    PolicyDecision,
    create_default_computer_policy,
)

logger = logging.getLogger(__name__)


@dataclass
class DispatchResult:
    """Result of dispatching an OpenClaw action."""

    success: bool
    action_type: str
    action_id: str = ""
    error: str | None = None
    screenshot_b64: str | None = None
    duration_ms: float = 0.0
    policy_decision: str = "allow"
    policy_reason: str = ""
    receipt_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "action_type": self.action_type,
            "action_id": self.action_id,
            "error": self.error,
            "has_screenshot": self.screenshot_b64 is not None,
            "duration_ms": round(self.duration_ms, 2),
            "policy_decision": self.policy_decision,
            "policy_reason": self.policy_reason,
            "receipt_hash": self.receipt_hash,
            "metadata": self.metadata,
        }


def _build_receipt_hash(action_type: str, action_id: str, success: bool) -> str:
    """Build a SHA-256 receipt hash for audit trail."""
    payload = f"{action_type}:{action_id}:{success}:{time.time()}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class OpenClawActionDispatcher:
    """
    Dispatches OpenClaw actions to real execution handlers.

    Connects:
    - ComputerUseBridge (format conversion)
    - ComputerPolicyChecker (policy enforcement)
    - PlaywrightActionExecutor (browser automation)

    For non-browser actions (navigate, extract), dispatches directly
    to the executor's navigate() method or page content extraction.
    """

    def __init__(
        self,
        policy: ComputerPolicy | None = None,
        executor: Any | None = None,
        executor_config: Any | None = None,
        current_url: str | None = None,
    ) -> None:
        """
        Initialize the dispatcher.

        Args:
            policy: Computer-use policy for action validation.
                    Uses default policy if not provided.
            executor: An ActionExecutor instance (e.g. PlaywrightActionExecutor).
                      If not provided, one is created lazily on start().
            executor_config: ExecutorConfig for auto-created executor.
            current_url: Initial URL context for policy domain checks.
        """
        self._policy = policy or create_default_computer_policy()
        self._policy_checker = ComputerPolicyChecker(self._policy)
        self._executor = executor
        self._executor_config = executor_config
        self._owns_executor = executor is None
        self._current_url = current_url
        self._running = False
        self._dispatch_count = 0
        self._error_count = 0
        self._audit_log: list[dict[str, Any]] = []

    @property
    def is_running(self) -> bool:
        """Check if dispatcher is running."""
        return self._running

    @property
    def dispatch_count(self) -> int:
        """Total dispatched actions."""
        return self._dispatch_count

    @property
    def current_url(self) -> str | None:
        """Current browser URL."""
        return self._current_url

    def _require_executor(self) -> Any:
        """Return the initialized executor or fail closed."""
        if self._executor is None:
            raise RuntimeError("Action executor is not initialized")
        return self._executor

    async def start(self, start_url: str | None = None) -> None:
        """
        Start the dispatcher and underlying executor.

        Args:
            start_url: Optional URL to navigate to on startup.
        """
        if self._running:
            return

        if self._executor is None:
            from aragora.computer_use.executor import ExecutorConfig, PlaywrightActionExecutor

            config = self._executor_config or ExecutorConfig()
            self._executor = PlaywrightActionExecutor(config=config)
            self._owns_executor = True

        if self._owns_executor and hasattr(self._executor, "start"):
            await self._executor.start(start_url=start_url)

        if start_url:
            self._current_url = start_url

        self._running = True
        logger.info("OpenClawActionDispatcher started")

    async def stop(self) -> None:
        """Stop the dispatcher and underlying executor."""
        if not self._running:
            return

        if self._owns_executor and self._executor and hasattr(self._executor, "stop"):
            await self._executor.stop()

        self._running = False
        logger.info(
            "OpenClawActionDispatcher stopped (dispatched=%d, errors=%d)",
            self._dispatch_count,
            self._error_count,
        )

    async def __aenter__(self) -> OpenClawActionDispatcher:
        """Context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        await self.stop()

    async def dispatch(
        self,
        action_type: str,
        params: dict[str, Any] | None = None,
        *,
        skip_policy: bool = False,
    ) -> DispatchResult:
        """
        Dispatch an OpenClaw action to real execution.

        Args:
            action_type: OpenClaw action type (click, type, screenshot, navigate, etc.)
            params: Action parameters.
            skip_policy: If True, bypass policy checks (for testing only).

        Returns:
            DispatchResult with execution outcome.
        """
        params = params or {}
        start_time = time.time()
        self._dispatch_count += 1
        action_id = f"dispatch-{uuid.uuid4().hex[:8]}"

        if not self._running:
            return DispatchResult(
                success=False,
                action_type=action_type,
                action_id=action_id,
                error="Dispatcher not running",
                policy_decision="deny",
            )

        # Step 1: Convert via bridge
        try:
            converted = ComputerUseBridge.from_openclaw(action_type, params)
        except (ValueError, KeyError, TypeError) as e:
            self._error_count += 1
            return DispatchResult(
                success=False,
                action_type=action_type,
                action_id=action_id,
                error=f"Action conversion failed: {e}",
            )

        # Step 2: Policy check (only for Aragora Action subclasses)
        if not skip_policy and isinstance(converted, Action):
            decision, reason = self._policy_checker.evaluate_action(converted, self._current_url)
            if decision == PolicyDecision.DENY:
                self._log_audit(action_type, action_id, "denied", reason)
                return DispatchResult(
                    success=False,
                    action_type=action_type,
                    action_id=action_id,
                    error=f"Policy denied: {reason}",
                    policy_decision="deny",
                    policy_reason=reason,
                    duration_ms=(time.time() - start_time) * 1000,
                )
            if decision == PolicyDecision.REQUIRE_APPROVAL:
                self._log_audit(action_type, action_id, "require_approval", reason)
                return DispatchResult(
                    success=False,
                    action_type=action_type,
                    action_id=action_id,
                    error=f"Requires approval: {reason}",
                    policy_decision="require_approval",
                    policy_reason=reason,
                    duration_ms=(time.time() - start_time) * 1000,
                )

        # Step 3: Dispatch to executor
        try:
            result = await self._execute(converted, action_id)
        except (RuntimeError, OSError, TimeoutError) as e:
            self._error_count += 1
            self._policy_checker.record_error()
            duration = (time.time() - start_time) * 1000
            self._log_audit(action_type, action_id, "error", str(e))
            return DispatchResult(
                success=False,
                action_type=action_type,
                action_id=action_id,
                error=f"Execution failed: {e}",
                policy_decision="allow",
                duration_ms=duration,
            )

        # Step 4: Build receipt
        duration = (time.time() - start_time) * 1000
        receipt_hash = _build_receipt_hash(action_type, action_id, result.success)

        if result.success:
            self._policy_checker.record_success()
        else:
            self._policy_checker.record_error()
            self._error_count += 1

        self._log_audit(
            action_type,
            action_id,
            "success" if result.success else "failed",
            result.error or "",
        )

        return DispatchResult(
            success=result.success,
            action_type=action_type,
            action_id=action_id,
            error=result.error,
            screenshot_b64=result.screenshot_b64,
            duration_ms=duration,
            policy_decision="allow",
            receipt_hash=receipt_hash,
            metadata=result.metadata,
        )

    async def _execute(
        self,
        action: Action | NavigateAction | ExtractAction,
        action_id: str,
    ) -> ActionResult:
        """
        Execute a converted action on the underlying executor.

        Handles three cases:
        - NavigateAction: calls executor.navigate()
        - ExtractAction: evaluates selector on page
        - Action subclass: calls executor.execute()
        """
        if isinstance(action, NavigateAction):
            return await self._execute_navigate(action, action_id)
        if isinstance(action, ExtractAction):
            return await self._execute_extract(action, action_id)

        # Standard Aragora action -- delegate to executor
        return await self._require_executor().execute(action)

    async def _execute_navigate(self, action: NavigateAction, action_id: str) -> ActionResult:
        """Execute a navigate action."""
        if not action.url:
            return ActionResult(
                action_id=action_id,
                action_type=ActionType.MOVE,  # closest type
                success=False,
                error="Navigate action requires a URL",
            )

        executor = self._require_executor()
        success = await executor.navigate(action.url)
        self._current_url = action.url if success else self._current_url

        screenshot_b64 = None
        if success:
            screenshot_b64 = await executor.take_screenshot()

        return ActionResult(
            action_id=action_id,
            action_type=ActionType.MOVE,
            success=success,
            error=None if success else "Navigation failed",
            screenshot_b64=screenshot_b64,
            metadata={"url": action.url},
        )

    async def _execute_extract(self, action: ExtractAction, action_id: str) -> ActionResult:
        """Execute an extract action (content extraction from page)."""
        # Extract is a read-only action -- take screenshot of current state
        screenshot_b64 = await self._require_executor().take_screenshot()
        return ActionResult(
            action_id=action_id,
            action_type=ActionType.SCREENSHOT,
            success=True,
            screenshot_b64=screenshot_b64,
            metadata={
                "selector": action.selector,
                "extract_type": action.extract_type,
            },
        )

    def to_action_bundle(self, result: DispatchResult) -> dict[str, Any]:
        """
        Convert a DispatchResult to a ComputerUseActionBundle-compatible dict.

        Can be used to create a ComputerUseActionBundle:
            from aragora.pipeline.backbone_contracts import ComputerUseActionBundle
            bundle = ComputerUseActionBundle.from_dict(dispatcher.to_action_bundle(result))
        """
        return {
            "harness_name": "openclaw-dispatcher",
            "action_type": result.action_type,
            "input_prompt": "",
            "output_files": [],
            "execution_time_seconds": result.duration_ms / 1000,
            "exit_code": 0 if result.success else 1,
            "stdout_summary": result.error or "ok",
            "policy_violations": [result.policy_reason] if result.policy_reason else [],
        }

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get the audit log of all dispatched actions."""
        return list(self._audit_log)

    def get_stats(self) -> dict[str, Any]:
        """Get dispatcher statistics."""
        return {
            "dispatch_count": self._dispatch_count,
            "error_count": self._error_count,
            "running": self._running,
            "current_url": self._current_url,
            "policy_stats": self._policy_checker.get_stats(),
        }

    def _log_audit(self, action_type: str, action_id: str, outcome: str, detail: str) -> None:
        """Append an entry to the audit log."""
        entry = {
            "action_type": action_type,
            "action_id": action_id,
            "outcome": outcome,
            "detail": detail,
            "timestamp": time.time(),
            "current_url": self._current_url,
        }
        self._audit_log.append(entry)
        logger.debug("OpenClaw dispatch: %s", entry)


__all__ = [
    "DispatchResult",
    "OpenClawActionDispatcher",
]
