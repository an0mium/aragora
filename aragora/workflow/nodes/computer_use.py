"""
Computer-use task step for workflows.

Executes an end-to-end computer-use task using the ComputerUseOrchestrator.
This allows workflows to drive interactive browser and UI tasks with policy
controls and full auditability.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from aragora.workflow.step import BaseStep, WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class ComputerUseStepConfig:
    """Configuration for a computer-use task step."""

    goal: str = ""
    max_steps: int | None = None
    initial_context: str = ""

    # Model settings
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None

    # Executor/browser settings
    browser_type: str = "chromium"
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    start_url: str = "about:blank"

    # Timeouts
    action_timeout_seconds: float | None = None
    total_timeout_seconds: float | None = None

    # API key override
    api_key: str | None = None


class ComputerUseTaskStep(BaseStep):
    """
    Workflow step that executes a computer-use task.

    Config options:
        goal: str - Natural language goal for the computer-use task (required)
        max_steps: int - Max actions to attempt
        initial_context: str - Extra context for the model
        model: str - Model identifier (defaults to orchestrator config)
        max_tokens: int - Model token limit
        temperature: float - Model temperature
        browser_type: str - chromium/firefox/webkit
        headless: bool - Headless browser
        viewport_width: int - Browser width
        viewport_height: int - Browser height
        start_url: str - Start URL for the browser
        action_timeout_seconds: float - Per-action timeout
        total_timeout_seconds: float - Overall task timeout
        api_key: str - Optional model API key (falls back to ANTHROPIC_API_KEY)
    """

    step_type = "computer_use_task"

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        super().__init__(name, config)
        cfg = config or {}
        self._step_config = ComputerUseStepConfig(
            goal=cfg.get("goal", ""),
            max_steps=cfg.get("max_steps"),
            initial_context=cfg.get("initial_context", ""),
            model=cfg.get("model"),
            max_tokens=cfg.get("max_tokens"),
            temperature=cfg.get("temperature"),
            browser_type=cfg.get("browser_type", "chromium"),
            headless=cfg.get("headless", True),
            viewport_width=cfg.get("viewport_width", 1920),
            viewport_height=cfg.get("viewport_height", 1080),
            start_url=cfg.get("start_url", "about:blank"),
            action_timeout_seconds=cfg.get("action_timeout_seconds"),
            total_timeout_seconds=cfg.get("total_timeout_seconds"),
            api_key=cfg.get("api_key"),
        )

    def validate_config(self) -> bool:
        """Validate configuration before execution."""
        goal = self._step_config.goal
        if not goal:
            logger.error("computer_use_task requires 'goal' config")
            return False
        return True

    def _resolve_template(self, value: str, context: WorkflowContext) -> str:
        """Resolve template variables in a string value."""
        if not value or "{" not in value:
            return value

        import re

        def replace_var(match):
            var_path = match.group(1)
            parts = var_path.split(".")
            if parts[0] == "inputs" and len(parts) > 1:
                return str(context.get_input(parts[1], match.group(0)))
            if parts[0] == "steps" and len(parts) > 2:
                step_output = context.get_step_output(parts[1])
                if step_output and isinstance(step_output, dict):
                    return str(step_output.get(parts[2], match.group(0)))
            if parts[0] == "state" and len(parts) > 1:
                return str(context.get_state(parts[1], match.group(0)))
            return match.group(0)

        return re.sub(r"\{([^}]+)\}", replace_var, value)

    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute a computer-use task using the orchestrator."""
        cfg = {**self._config, **context.current_step_config}

        goal = self._resolve_template(str(cfg.get("goal", self._step_config.goal)), context)
        initial_context = self._resolve_template(
            str(cfg.get("initial_context", self._step_config.initial_context)), context
        )
        if not goal:
            return {"success": False, "error": "goal is required for computer_use_task"}

        api_key = cfg.get("api_key", self._step_config.api_key) or os.getenv("ANTHROPIC_API_KEY")

        try:
            from aragora.computer_use.executor import ExecutorConfig, PlaywrightActionExecutor
            from aragora.computer_use.orchestrator import ComputerUseConfig, ComputerUseOrchestrator
        except ImportError as e:
            return {
                "success": False,
                "error": f"Computer-use dependencies not available: {e}",
            }

        executor_config = ExecutorConfig(
            browser_type=cfg.get("browser_type", self._step_config.browser_type),
            headless=cfg.get("headless", self._step_config.headless),
            viewport_width=cfg.get("viewport_width", self._step_config.viewport_width),
            viewport_height=cfg.get("viewport_height", self._step_config.viewport_height),
            start_url=cfg.get("start_url", self._step_config.start_url),
        )

        computer_config = ComputerUseConfig()
        if cfg.get("model") is not None:
            computer_config.model = cfg["model"]
        if cfg.get("max_tokens") is not None:
            computer_config.max_tokens = int(cfg["max_tokens"])
        if cfg.get("temperature") is not None:
            computer_config.temperature = float(cfg["temperature"])
        if cfg.get("action_timeout_seconds") is not None:
            computer_config.action_timeout_seconds = float(cfg["action_timeout_seconds"])
        if cfg.get("total_timeout_seconds") is not None:
            computer_config.total_timeout_seconds = float(cfg["total_timeout_seconds"])
        if cfg.get("max_steps") is not None:
            computer_config.max_steps = int(cfg["max_steps"])

        try:
            from aragora.events.types import StreamEventType

            def _on_step_complete(step_result: Any) -> None:
                try:
                    context.emit_event(
                        StreamEventType.WORKFLOW_STEP_PROGRESS.value,
                        {
                            "workflow_id": context.workflow_id,
                            "definition_id": context.definition_id,
                            "step_id": context.current_step_id,
                            "step_name": self.name,
                            "computer_use_step": True,
                            "action_step": getattr(step_result, "step_number", None),
                            "status": getattr(step_result, "status", None).value
                            if getattr(step_result, "status", None)
                            else None,
                            "action": step_result.action.to_dict()
                            if getattr(step_result, "action", None)
                            else None,
                            "policy_check_passed": getattr(step_result, "policy_check_passed", None),
                            "policy_reason": getattr(step_result, "policy_reason", None),
                        },
                    )
                except (ImportError, AttributeError, TypeError) as exc:
                    logger.debug("Failed to emit computer_use step event: %s", exc)

            computer_config.on_step_complete = _on_step_complete
        except ImportError:
            pass

        executor = PlaywrightActionExecutor(executor_config)
        orchestrator = ComputerUseOrchestrator(
            executor=executor,
            config=computer_config,
            api_key=api_key,
        )

        try:
            async with executor:
                task_result = await orchestrator.run_task(
                    goal=goal,
                    max_steps=cfg.get("max_steps", self._step_config.max_steps),
                    initial_context=initial_context,
                    metadata={
                        "workflow_id": context.workflow_id,
                        "step_name": self.name,
                    },
                )

            result_dict = task_result.to_dict()
            return {
                "success": task_result.status.value == "completed",
                "status": task_result.status.value,
                "task": result_dict,
                "metrics": orchestrator.metrics.to_dict(),
            }

        except Exception as e:
            logger.error("Computer-use task failed: %s", e)
            return {"success": False, "error": str(e)}
