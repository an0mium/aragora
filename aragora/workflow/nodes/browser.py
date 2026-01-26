"""
Browser Automation Step for Workflows.

Provides browser automation capabilities in workflows using Playwright.
Supports navigation, interaction, screenshot capture, and data extraction.

Usage in workflow definition:
    steps:
      - id: "open_page"
        type: "browser"
        config:
          action: "navigate"
          url: "https://example.com"
          wait_until: "networkidle"

      - id: "fill_form"
        type: "browser"
        config:
          action: "fill"
          selector: "input[name='email']"
          value: "{inputs.email}"

      - id: "submit"
        type: "browser"
        config:
          action: "click"
          selector: "button[type='submit']"

      - id: "capture"
        type: "browser"
        config:
          action: "screenshot"
          full_page: true
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aragora.workflow.step import BaseStep, WorkflowContext
from aragora.workflow.types import StepResult

logger = logging.getLogger(__name__)


# Global connector instance for session persistence across steps
_browser_connector = None


async def _get_browser_connector(config: Dict[str, Any]):
    """Get or create browser connector with session persistence."""
    global _browser_connector

    # Check if we need a new connector
    force_new = config.get("new_session", False)
    if force_new and _browser_connector:
        await _browser_connector.close()
        _browser_connector = None

    if _browser_connector is None:
        from aragora.connectors.browser import PlaywrightConnector

        _browser_connector = PlaywrightConnector(
            headless=config.get("headless", True),
            browser_type=config.get("browser_type", "chromium"),
            allowed_domains=set(config.get("allowed_domains", [])),
            blocked_domains=set(config.get("blocked_domains", [])),
            timeout_ms=config.get("timeout_ms", 30000),
            viewport_width=config.get("viewport_width", 1280),
            viewport_height=config.get("viewport_height", 720),
            user_agent=config.get("user_agent"),
            proxy=config.get("proxy"),
            ignore_https_errors=config.get("ignore_https_errors", False),
        )
        await _browser_connector.initialize()

    return _browser_connector


async def _close_browser_connector():
    """Close the global browser connector."""
    global _browser_connector
    if _browser_connector:
        await _browser_connector.close()
        _browser_connector = None


@dataclass
class BrowserStepConfig:
    """Configuration for a browser step."""

    action: str  # navigate, click, fill, screenshot, etc.
    url: Optional[str] = None
    selector: Optional[str] = None
    value: Optional[str] = None
    wait_until: str = "load"
    timeout_ms: Optional[int] = None
    full_page: bool = False
    selectors: Optional[Dict[str, str]] = None  # For extract_data
    script: Optional[str] = None  # For execute_script
    attribute: Optional[str] = None  # For get_attribute
    state: str = "visible"  # For wait_for
    force: bool = False  # For click
    new_session: bool = False  # Start new browser session
    close_session: bool = False  # Close browser after step

    # Browser configuration (for new sessions)
    headless: bool = True
    browser_type: str = "chromium"
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: Optional[str] = None
    proxy: Optional[Dict[str, str]] = None
    ignore_https_errors: bool = False


class BrowserStep(BaseStep):
    """
    Workflow step for browser automation.

    Executes browser actions using Playwright connector.
    Maintains session across multiple steps for stateful navigation.
    """

    step_type = "browser"

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize browser step.

        Args:
            name: Step name
            config: Step configuration with action and parameters
        """
        super().__init__(name, config)
        self._config = BrowserStepConfig(
            action=config.get("action", "navigate"),
            url=config.get("url"),
            selector=config.get("selector"),
            value=config.get("value"),
            wait_until=config.get("wait_until", "load"),
            timeout_ms=config.get("timeout_ms"),
            full_page=config.get("full_page", False),
            selectors=config.get("selectors"),
            script=config.get("script"),
            attribute=config.get("attribute"),
            state=config.get("state", "visible"),
            force=config.get("force", False),
            new_session=config.get("new_session", False),
            close_session=config.get("close_session", False),
            headless=config.get("headless", True),
            browser_type=config.get("browser_type", "chromium"),
            allowed_domains=config.get("allowed_domains", []),
            blocked_domains=config.get("blocked_domains", []),
            viewport_width=config.get("viewport_width", 1280),
            viewport_height=config.get("viewport_height", 720),
            user_agent=config.get("user_agent"),
            proxy=config.get("proxy"),
            ignore_https_errors=config.get("ignore_https_errors", False),
        )

    def _resolve_template(self, value: Optional[str], context: WorkflowContext) -> Optional[str]:
        """Resolve template variables in a value."""
        if value is None:
            return None

        # Simple template resolution for {inputs.x} and {steps.x.y}
        import re

        def replace_var(match):
            var_path = match.group(1)
            parts = var_path.split(".")

            if parts[0] == "inputs" and len(parts) > 1:
                return str(context.get_input(parts[1], match.group(0)))
            elif parts[0] == "steps" and len(parts) > 2:
                step_output = context.get_step_output(parts[1])
                if step_output and isinstance(step_output, dict):
                    return str(step_output.get(parts[2], match.group(0)))
            elif parts[0] == "state" and len(parts) > 1:
                return str(context.get_state(parts[1], match.group(0)))

            return match.group(0)

        return re.sub(r"\{([^}]+)\}", replace_var, value)

    async def execute(self, context: WorkflowContext) -> StepResult:
        """
        Execute the browser action.

        Args:
            context: Workflow context with inputs and state

        Returns:
            StepResult with action outcome
        """
        config = self._config
        action = config.action.lower()

        try:
            # Get or create browser connector
            connector = await _get_browser_connector(self.config)

            # Resolve template variables
            url = self._resolve_template(config.url, context)
            selector = self._resolve_template(config.selector, context)
            value = self._resolve_template(config.value, context)
            script = self._resolve_template(config.script, context)

            result_data: Dict[str, Any] = {"action": action}

            # Execute action
            if action == "navigate":
                if not url:
                    return StepResult(
                        success=False,
                        output={"error": "URL is required for navigate action"},
                        error="URL is required for navigate action",
                    )
                state = await connector.navigate(
                    url,
                    wait_until=config.wait_until,
                    timeout_ms=config.timeout_ms,
                )
                result_data.update(state.to_dict())

            elif action == "click":
                if not selector:
                    return StepResult(
                        success=False,
                        output={"error": "Selector is required for click action"},
                        error="Selector is required for click action",
                    )
                result = await connector.click(
                    selector,
                    timeout_ms=config.timeout_ms,
                    force=config.force,
                )
                result_data.update(result.to_dict())
                if not result.success:
                    return StepResult(
                        success=False,
                        output=result_data,
                        error=result.error,
                    )

            elif action == "fill":
                if not selector or value is None:
                    return StepResult(
                        success=False,
                        output={"error": "Selector and value are required for fill action"},
                        error="Selector and value are required for fill action",
                    )
                result = await connector.fill(
                    selector,
                    value,
                    timeout_ms=config.timeout_ms,
                )
                result_data.update(result.to_dict())
                if not result.success:
                    return StepResult(
                        success=False,
                        output=result_data,
                        error=result.error,
                    )

            elif action == "select":
                if not selector or value is None:
                    return StepResult(
                        success=False,
                        output={"error": "Selector and value are required for select action"},
                        error="Selector and value are required for select action",
                    )
                result = await connector.select(
                    selector,
                    value,
                    timeout_ms=config.timeout_ms,
                )
                result_data.update(result.to_dict())
                if not result.success:
                    return StepResult(
                        success=False,
                        output=result_data,
                        error=result.error,
                    )

            elif action == "screenshot":
                screenshot_bytes = await connector.screenshot(
                    full_page=config.full_page,
                    selector=selector,
                )
                result_data["screenshot_base64"] = base64.b64encode(screenshot_bytes).decode()
                result_data["size_bytes"] = len(screenshot_bytes)

            elif action == "get_text":
                if not selector:
                    return StepResult(
                        success=False,
                        output={"error": "Selector is required for get_text action"},
                        error="Selector is required for get_text action",
                    )
                text = await connector.get_text(selector, timeout_ms=config.timeout_ms)
                result_data["text"] = text

            elif action == "get_attribute":
                if not selector or not config.attribute:
                    return StepResult(
                        success=False,
                        output={
                            "error": "Selector and attribute are required for get_attribute action"
                        },
                        error="Selector and attribute are required",
                    )
                attr_value = await connector.get_attribute(
                    selector,
                    config.attribute,
                    timeout_ms=config.timeout_ms,
                )
                result_data["attribute"] = config.attribute
                result_data["value"] = attr_value

            elif action == "get_html":
                html = await connector.get_html(selector)
                result_data["html"] = html

            elif action == "wait_for":
                if not selector:
                    return StepResult(
                        success=False,
                        output={"error": "Selector is required for wait_for action"},
                        error="Selector is required for wait_for action",
                    )
                found = await connector.wait_for(
                    selector,
                    state=config.state,
                    timeout_ms=config.timeout_ms,
                )
                result_data["found"] = found
                if not found:
                    return StepResult(
                        success=False,
                        output=result_data,
                        error=f"Element not found: {selector}",
                    )

            elif action == "execute_script":
                if not script:
                    return StepResult(
                        success=False,
                        output={"error": "Script is required for execute_script action"},
                        error="Script is required for execute_script action",
                    )
                script_result = await connector.execute_script(script)
                result_data["result"] = script_result

            elif action == "extract_data":
                if not config.selectors:
                    return StepResult(
                        success=False,
                        output={"error": "Selectors dict is required for extract_data action"},
                        error="Selectors dict is required",
                    )
                # Resolve templates in selectors
                resolved_selectors = {
                    k: self._resolve_template(v, context) or v for k, v in config.selectors.items()
                }
                data = await connector.extract_data(
                    resolved_selectors,
                    timeout_ms=config.timeout_ms,
                )
                result_data["data"] = data

            elif action == "get_cookies":
                cookies = await connector.get_cookies()
                result_data["cookies"] = cookies

            elif action == "clear_cookies":
                await connector.clear_cookies()
                result_data["cleared"] = True

            elif action == "reload":
                state = await connector.reload()
                result_data.update(state.to_dict())

            elif action == "go_back":
                state = await connector.go_back()
                result_data.update(state.to_dict())

            elif action == "go_forward":
                state = await connector.go_forward()
                result_data.update(state.to_dict())

            elif action == "close":
                await _close_browser_connector()
                result_data["closed"] = True

            else:
                return StepResult(
                    success=False,
                    output={"error": f"Unknown action: {action}"},
                    error=f"Unknown action: {action}",
                )

            # Add current URL to result
            if connector.is_initialized:
                result_data["current_url"] = connector.current_url

            # Close session if requested
            if config.close_session:
                await _close_browser_connector()
                result_data["session_closed"] = True

            return StepResult(
                success=True,
                output=result_data,
            )

        except Exception as e:
            logger.error(f"Browser step '{self.name}' failed: {e}")
            return StepResult(
                success=False,
                output={"error": str(e), "action": action},
                error=str(e),
            )


# Register step type
def register_browser_step():
    """Register the browser step type with the workflow engine."""
    from aragora.workflow.nodes import register_step_type

    register_step_type("browser", BrowserStep)
    logger.debug("Registered browser step type")
