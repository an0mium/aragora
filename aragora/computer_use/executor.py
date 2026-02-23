"""
Playwright-based Action Executor for Computer Use.

Pattern: Computer Use
Inspired by: Anthropic's computer_20241022 tool specification
Aragora adaptation: Policy-gated execution with RBAC approval and full audit logging.

Provides real browser automation via Playwright for executing computer-use actions:
- Screenshot capture with configurable resolution
- Mouse clicks (single, double, right-click)
- Keyboard input and special key presses
- Scrolling and mouse movement
- Drag operations

Usage:
    from aragora.computer_use.executor import PlaywrightActionExecutor

    async with PlaywrightActionExecutor() as executor:
        result = await executor.execute(ClickAction(x=100, y=200))
        screenshot = await executor.take_screenshot()
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any, cast

from aragora.computer_use.actions import (
    Action,
    ActionResult,
    ActionType,
    ClickAction,
    ClickButton,
    DragAction,
    KeyAction,
    MoveAction,
    ScrollAction,
    ScrollDirection,
    TypeAction,
    WaitAction,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutorConfig:
    """Configuration for the Playwright executor."""

    # Browser settings
    browser_type: str = "chromium"  # chromium, firefox, webkit
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080

    # Screenshot settings
    screenshot_type: str = "png"  # png, jpeg
    screenshot_quality: int = 80  # For jpeg only
    full_page: bool = False

    # Timeout settings
    action_timeout_ms: int = 10000
    navigation_timeout_ms: int = 30000

    # Startup URL
    start_url: str = "about:blank"

    # Mouse settings
    click_delay_ms: int = 50
    type_delay_ms: int = 50

    # Extra browser args (for sandbox environments)
    browser_args: list[str] = field(default_factory=list)


class PlaywrightActionExecutor:
    """
    Execute computer-use actions via Playwright browser automation.

    Implements the ActionExecutor protocol for real browser control.
    Supports all action types defined in actions.py.

    Usage:
        # Context manager (recommended)
        async with PlaywrightActionExecutor() as executor:
            await executor.execute(ClickAction(x=100, y=200))

        # Manual lifecycle
        executor = PlaywrightActionExecutor()
        await executor.start()
        try:
            await executor.execute(action)
        finally:
            await executor.stop()
    """

    def __init__(self, config: ExecutorConfig | None = None) -> None:
        """
        Initialize the executor.

        Args:
            config: Executor configuration (uses defaults if not provided)
        """
        self._config = config or ExecutorConfig()
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._running = False
        self._action_count = 0
        self._error_count = 0

    @property
    def is_running(self) -> bool:
        """Check if executor is running."""
        return self._running

    @property
    def action_count(self) -> int:
        """Get total action count."""
        return self._action_count

    @property
    def error_count(self) -> int:
        """Get error count."""
        return self._error_count

    async def __aenter__(self) -> PlaywrightActionExecutor:
        """Start executor as context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop executor on context exit."""
        await self.stop()

    async def start(self, start_url: str | None = None) -> None:
        """
        Start the Playwright browser.

        Args:
            start_url: Optional URL to navigate to on start
        """
        if self._running:
            logger.warning("Executor already running")
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ImportError(
                "Playwright is required for PlaywrightActionExecutor. "
                "Install with: pip install playwright && playwright install"
            ) from e

        self._playwright = await async_playwright().start()

        # Select browser type
        browser_type = getattr(self._playwright, self._config.browser_type, None)
        if not browser_type:
            raise ValueError(f"Unknown browser type: {self._config.browser_type}")

        # Build browser args
        browser_args = list(self._config.browser_args)
        if not browser_args:
            # Default args for sandboxed environments
            browser_args = [
                "--disable-blink-features=AutomationControlled",
            ]

        # Launch browser
        self._browser = await browser_type.launch(
            headless=self._config.headless,
            args=browser_args,
        )

        # Create context with viewport
        self._context = await self._browser.new_context(
            viewport={
                "width": self._config.viewport_width,
                "height": self._config.viewport_height,
            },
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        )

        # Set default timeouts
        self._context.set_default_timeout(self._config.action_timeout_ms)
        self._context.set_default_navigation_timeout(self._config.navigation_timeout_ms)

        # Create page
        self._page = await self._context.new_page()

        # Navigate to start URL
        url = start_url or self._config.start_url
        if url and url != "about:blank":
            await self._page.goto(url)

        self._running = True
        logger.info(
            "PlaywrightActionExecutor started (%s, headless=%s)", self._config.browser_type, self._config.headless
        )

    async def stop(self) -> None:
        """Stop the Playwright browser."""
        if not self._running:
            return

        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except (RuntimeError, OSError, TimeoutError) as e:
            logger.warning("Error stopping executor: %s", e)

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._running = False
        logger.info("PlaywrightActionExecutor stopped")

    async def execute(self, action: Action) -> ActionResult:
        """
        Execute an action and return the result.

        Args:
            action: The action to execute

        Returns:
            ActionResult with success/failure details
        """
        if not self._running:
            return ActionResult(
                action_id=action.action_id,
                action_type=action.action_type,
                success=False,
                error="Executor not running",
            )

        start_time = time.time()
        self._action_count += 1

        try:
            # Dispatch to specific handler based on action type
            # Type narrowing via match-case not fully recognized by mypy
            match action.action_type:
                case ActionType.SCREENSHOT:
                    return await self._execute_screenshot(action)
                case ActionType.CLICK:
                    return await self._execute_click(cast(ClickAction, action))
                case ActionType.DOUBLE_CLICK:
                    return await self._execute_double_click(cast(ClickAction, action))
                case ActionType.RIGHT_CLICK:
                    return await self._execute_right_click(cast(ClickAction, action))
                case ActionType.TYPE:
                    return await self._execute_type(cast(TypeAction, action))
                case ActionType.KEY:
                    return await self._execute_key(cast(KeyAction, action))
                case ActionType.SCROLL:
                    return await self._execute_scroll(cast(ScrollAction, action))
                case ActionType.MOVE:
                    return await self._execute_move(cast(MoveAction, action))
                case ActionType.DRAG:
                    return await self._execute_drag(cast(DragAction, action))
                case ActionType.WAIT:
                    return await self._execute_wait(cast(WaitAction, action))
                case _:
                    return ActionResult(
                        action_id=action.action_id,
                        action_type=action.action_type,
                        success=False,
                        error=f"Unknown action type: {action.action_type}",
                    )

        except (RuntimeError, OSError, TimeoutError) as e:
            self._error_count += 1
            logger.exception("Action %s failed: %s", action.action_id, e)
            return ActionResult(
                action_id=action.action_id,
                action_type=action.action_type,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def take_screenshot(self) -> str:
        """
        Take a screenshot and return base64-encoded image.

        Returns:
            Base64-encoded screenshot image
        """
        if not self._running or not self._page:
            return ""

        screenshot_bytes = await self._page.screenshot(
            type=self._config.screenshot_type,
            full_page=self._config.full_page,
            quality=self._config.screenshot_quality
            if self._config.screenshot_type == "jpeg"
            else None,
        )

        return base64.b64encode(screenshot_bytes).decode("utf-8")

    async def get_current_url(self) -> str | None:
        """
        Get the current browser URL.

        Returns:
            Current URL or None if not available
        """
        if not self._running or not self._page:
            return None

        return self._page.url

    async def navigate(self, url: str) -> bool:
        """
        Navigate to a URL.

        Args:
            url: URL to navigate to

        Returns:
            True if navigation succeeded
        """
        if not self._running or not self._page:
            return False

        try:
            await self._page.goto(url)
            return True
        except (RuntimeError, OSError, TimeoutError) as e:
            logger.error("Navigation failed: %s", e)
            return False

    # =========================================================================
    # Private action handlers
    # =========================================================================

    async def _execute_screenshot(self, action: Action) -> ActionResult:
        """Take a screenshot."""
        screenshot_b64 = await self.take_screenshot()
        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=True,
            screenshot_b64=screenshot_b64,
        )

    async def _execute_click(self, action: ClickAction) -> ActionResult:
        """Execute a click action."""
        await self._page.mouse.click(
            action.x,
            action.y,
            button=action.button.value if action.button != ClickButton.LEFT else "left",
            delay=self._config.click_delay_ms,
        )

        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=True,
            metadata={"x": action.x, "y": action.y, "button": action.button.value},
        )

    async def _execute_double_click(self, action: ClickAction) -> ActionResult:
        """Execute a double-click action."""
        await self._page.mouse.dblclick(
            action.x,
            action.y,
            delay=self._config.click_delay_ms,
        )

        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=True,
            metadata={"x": action.x, "y": action.y, "double_click": True},
        )

    async def _execute_right_click(self, action: ClickAction) -> ActionResult:
        """Execute a right-click action."""
        await self._page.mouse.click(
            action.x,
            action.y,
            button="right",
            delay=self._config.click_delay_ms,
        )

        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=True,
            metadata={"x": action.x, "y": action.y, "button": "right"},
        )

    async def _execute_type(self, action: TypeAction) -> ActionResult:
        """Execute a type action."""
        await self._page.keyboard.type(
            action.text,
            delay=self._config.type_delay_ms,
        )

        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=True,
            metadata={"text_length": len(action.text)},
        )

    async def _execute_key(self, action: KeyAction) -> ActionResult:
        """Execute a key press action."""
        # Handle key combinations (e.g., "ctrl+c", "alt+Tab")
        key = action.key

        if "+" in key:
            # Key combination
            parts = key.split("+")
            modifiers = parts[:-1]
            final_key = parts[-1]

            # Map modifier names to Playwright modifiers
            modifier_map = {
                "ctrl": "Control",
                "alt": "Alt",
                "shift": "Shift",
                "meta": "Meta",
                "cmd": "Meta",
            }

            # Press modifiers
            for mod in modifiers:
                mod_key = modifier_map.get(mod.lower(), mod)
                await self._page.keyboard.down(mod_key)

            # Press and release the final key
            await self._page.keyboard.press(final_key)

            # Release modifiers in reverse order
            for mod in reversed(modifiers):
                mod_key = modifier_map.get(mod.lower(), mod)
                await self._page.keyboard.up(mod_key)
        else:
            # Single key
            await self._page.keyboard.press(key)

        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=True,
            metadata={"key": action.key},
        )

    async def _execute_scroll(self, action: ScrollAction) -> ActionResult:
        """Execute a scroll action."""
        # Calculate scroll delta based on direction
        delta_x = 0
        delta_y = 0
        scroll_amount = action.amount * 100  # Convert to pixels

        match action.direction:
            case ScrollDirection.UP:
                delta_y = -scroll_amount
            case ScrollDirection.DOWN:
                delta_y = scroll_amount
            case ScrollDirection.LEFT:
                delta_x = -scroll_amount
            case ScrollDirection.RIGHT:
                delta_x = scroll_amount

        # Move to position if specified, otherwise use center
        x = action.x if action.x is not None else self._config.viewport_width // 2
        y = action.y if action.y is not None else self._config.viewport_height // 2

        await self._page.mouse.move(x, y)
        await self._page.mouse.wheel(delta_x, delta_y)

        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=True,
            metadata={
                "direction": action.direction.value,
                "amount": action.amount,
                "delta": {"x": delta_x, "y": delta_y},
            },
        )

    async def _execute_move(self, action: MoveAction) -> ActionResult:
        """Execute a mouse move action."""
        await self._page.mouse.move(action.x, action.y)

        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=True,
            metadata={"x": action.x, "y": action.y},
        )

    async def _execute_drag(self, action: DragAction) -> ActionResult:
        """Execute a drag action."""
        # Move to start position
        await self._page.mouse.move(action.start_x, action.start_y)
        # Press mouse button
        await self._page.mouse.down()
        # Move to end position
        await self._page.mouse.move(action.end_x, action.end_y)
        # Release mouse button
        await self._page.mouse.up()

        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=True,
            metadata={
                "start": {"x": action.start_x, "y": action.start_y},
                "end": {"x": action.end_x, "y": action.end_y},
            },
        )

    async def _execute_wait(self, action: WaitAction) -> ActionResult:
        """Execute a wait action."""
        await asyncio.sleep(action.duration_ms / 1000)

        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=True,
            metadata={
                "duration_ms": action.duration_ms,
                "wait_for": action.wait_for,
            },
        )


__all__ = [
    "ExecutorConfig",
    "PlaywrightActionExecutor",
]
