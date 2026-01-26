"""
Playwright Browser Connector.

Provides browser automation capabilities using Playwright for workflow integration.
Supports headless and headed browser operation with configurable security sandboxing.

Features:
- Page navigation with wait strategies
- Element interaction (click, fill, select)
- Screenshot and PDF capture
- Data extraction via selectors
- JavaScript execution
- Cookie and storage management
- Network request interception
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class BrowserAction(str, Enum):
    """Available browser actions."""

    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"
    SCREENSHOT = "screenshot"
    PDF = "pdf"
    GET_TEXT = "get_text"
    GET_ATTRIBUTE = "get_attribute"
    GET_HTML = "get_html"
    WAIT_FOR = "wait_for"
    WAIT_FOR_NAVIGATION = "wait_for_navigation"
    EXECUTE_SCRIPT = "execute_script"
    EXTRACT_DATA = "extract_data"
    SET_COOKIE = "set_cookie"
    GET_COOKIES = "get_cookies"
    CLEAR_COOKIES = "clear_cookies"
    RELOAD = "reload"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    CLOSE = "close"


@dataclass
class PageState:
    """Current state of a browser page."""

    url: str
    title: str
    content_type: str = "text/html"
    status_code: int = 200
    load_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content_type": self.content_type,
            "status_code": self.status_code,
            "load_time_ms": self.load_time_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class ActionResult:
    """Result of a browser action."""

    success: bool
    action: str
    selector: Optional[str] = None
    value: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "action": self.action,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }
        if self.selector:
            result["selector"] = self.selector
        if self.value is not None:
            result["value"] = self.value
        if self.error:
            result["error"] = self.error
        return result


class PlaywrightConnector:
    """
    Browser automation connector using Playwright.

    Provides a high-level interface for browser automation in workflows.
    Includes security sandboxing to limit allowed domains and actions.

    Usage:
        connector = PlaywrightConnector(
            headless=True,
            allowed_domains=["example.com", "api.example.com"],
        )
        await connector.initialize()

        state = await connector.navigate("https://example.com")
        result = await connector.click("button.submit")
        screenshot = await connector.screenshot()

        await connector.close()
    """

    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        allowed_domains: Optional[Set[str]] = None,
        blocked_domains: Optional[Set[str]] = None,
        timeout_ms: int = 30000,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        user_agent: Optional[str] = None,
        proxy: Optional[Dict[str, str]] = None,
        ignore_https_errors: bool = False,
    ):
        """
        Initialize the browser connector.

        Args:
            headless: Run browser in headless mode
            browser_type: Browser to use (chromium, firefox, webkit)
            allowed_domains: If set, only allow navigation to these domains
            blocked_domains: Domains to block (e.g., ad trackers)
            timeout_ms: Default timeout for operations
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            user_agent: Custom user agent string
            proxy: Proxy configuration {"server": "...", "username": "...", "password": "..."}
            ignore_https_errors: Ignore HTTPS certificate errors
        """
        self.headless = headless
        self.browser_type = browser_type
        self.allowed_domains = allowed_domains or set()
        self.blocked_domains = blocked_domains or set()
        self.timeout_ms = timeout_ms
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.user_agent = user_agent
        self.proxy = proxy
        self.ignore_https_errors = ignore_https_errors

        # Load defaults from environment
        if not self.allowed_domains:
            env_domains = os.environ.get("ARAGORA_BROWSER_ALLOWED_DOMAINS", "")
            if env_domains:
                self.allowed_domains = set(d.strip() for d in env_domains.split(","))

        if not self.blocked_domains:
            env_blocked = os.environ.get("ARAGORA_BROWSER_BLOCKED_DOMAINS", "")
            if env_blocked:
                self.blocked_domains = set(d.strip() for d in env_blocked.split(","))

        # Playwright objects (lazy initialized)
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Playwright and launch browser."""
        if self._initialized:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright is not installed. Install with: pip install playwright && playwright install chromium"
            )

        self._playwright = await async_playwright().start()

        # Select browser type
        if self.browser_type == "firefox":
            browser_launcher = self._playwright.firefox
        elif self.browser_type == "webkit":
            browser_launcher = self._playwright.webkit
        else:
            browser_launcher = self._playwright.chromium

        # Launch browser with proxy if configured
        launch_options = {
            "headless": self.headless,
        }
        if self.proxy:
            launch_options["proxy"] = self.proxy

        self._browser = await browser_launcher.launch(**launch_options)

        # Create browser context with viewport and user agent
        context_options = {
            "viewport": {"width": self.viewport_width, "height": self.viewport_height},
            "ignore_https_errors": self.ignore_https_errors,
        }
        if self.user_agent:
            context_options["user_agent"] = self.user_agent

        self._context = await self._browser.new_context(**context_options)
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self.timeout_ms)

        self._initialized = True
        logger.info(f"Browser connector initialized: {self.browser_type}, headless={self.headless}")

    async def close(self) -> None:
        """Close browser and cleanup resources."""
        if self._page:
            await self._page.close()
            self._page = None
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._initialized = False
        logger.info("Browser connector closed")

    def _check_domain_allowed(self, url: str) -> bool:
        """Check if URL domain is allowed."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove port if present
        if ":" in domain:
            domain = domain.split(":")[0]

        # Check blocked domains first
        for blocked in self.blocked_domains:
            if domain == blocked or domain.endswith(f".{blocked}"):
                logger.warning(f"Domain blocked: {domain}")
                return False

        # If allowed_domains is set, check if domain is in the list
        if self.allowed_domains:
            for allowed in self.allowed_domains:
                if domain == allowed or domain.endswith(f".{allowed}"):
                    return True
            logger.warning(f"Domain not in allowed list: {domain}")
            return False

        return True

    async def _ensure_initialized(self) -> None:
        """Ensure browser is initialized."""
        if not self._initialized:
            await self.initialize()

    async def navigate(
        self,
        url: str,
        wait_until: str = "load",
        timeout_ms: Optional[int] = None,
    ) -> PageState:
        """
        Navigate to a URL.

        Args:
            url: URL to navigate to
            wait_until: Wait strategy (load, domcontentloaded, networkidle, commit)
            timeout_ms: Override default timeout

        Returns:
            PageState with current page information
        """
        await self._ensure_initialized()

        if not self._check_domain_allowed(url):
            raise ValueError(f"Navigation to domain not allowed: {url}")

        import time

        start = time.perf_counter()

        response = await self._page.goto(
            url,
            wait_until=wait_until,
            timeout=timeout_ms or self.timeout_ms,
        )

        load_time = (time.perf_counter() - start) * 1000

        return PageState(
            url=self._page.url,
            title=await self._page.title(),
            status_code=response.status if response else 0,
            content_type=(
                response.headers.get("content-type", "text/html") if response else "text/html"
            ),
            load_time_ms=load_time,
        )

    async def click(
        self,
        selector: str,
        timeout_ms: Optional[int] = None,
        force: bool = False,
    ) -> ActionResult:
        """
        Click an element.

        Args:
            selector: CSS selector or XPath
            timeout_ms: Override default timeout
            force: Force click even if element is not visible

        Returns:
            ActionResult indicating success or failure
        """
        await self._ensure_initialized()

        import time

        start = time.perf_counter()

        try:
            await self._page.click(
                selector,
                timeout=timeout_ms or self.timeout_ms,
                force=force,
            )
            return ActionResult(
                success=True,
                action="click",
                selector=selector,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="click",
                selector=selector,
                error=str(e),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    async def fill(
        self,
        selector: str,
        value: str,
        timeout_ms: Optional[int] = None,
    ) -> ActionResult:
        """
        Fill a text input field.

        Args:
            selector: CSS selector or XPath
            value: Value to fill
            timeout_ms: Override default timeout

        Returns:
            ActionResult indicating success or failure
        """
        await self._ensure_initialized()

        import time

        start = time.perf_counter()

        try:
            await self._page.fill(
                selector,
                value,
                timeout=timeout_ms or self.timeout_ms,
            )
            return ActionResult(
                success=True,
                action="fill",
                selector=selector,
                value=value,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="fill",
                selector=selector,
                error=str(e),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    async def select(
        self,
        selector: str,
        value: str,
        timeout_ms: Optional[int] = None,
    ) -> ActionResult:
        """
        Select an option from a dropdown.

        Args:
            selector: CSS selector for select element
            value: Option value to select
            timeout_ms: Override default timeout

        Returns:
            ActionResult indicating success or failure
        """
        await self._ensure_initialized()

        import time

        start = time.perf_counter()

        try:
            await self._page.select_option(
                selector,
                value,
                timeout=timeout_ms or self.timeout_ms,
            )
            return ActionResult(
                success=True,
                action="select",
                selector=selector,
                value=value,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="select",
                selector=selector,
                error=str(e),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    async def screenshot(
        self,
        path: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
    ) -> bytes:
        """
        Capture a screenshot.

        Args:
            path: Optional file path to save screenshot
            full_page: Capture full scrollable page
            selector: If set, capture only this element

        Returns:
            Screenshot as PNG bytes
        """
        await self._ensure_initialized()

        if selector:
            element = await self._page.query_selector(selector)
            if element:
                screenshot = await element.screenshot()
            else:
                raise ValueError(f"Element not found: {selector}")
        else:
            screenshot = await self._page.screenshot(full_page=full_page)

        if path:
            with open(path, "wb") as f:
                f.write(screenshot)

        return screenshot

    async def get_text(
        self,
        selector: str,
        timeout_ms: Optional[int] = None,
    ) -> str:
        """
        Get text content of an element.

        Args:
            selector: CSS selector or XPath
            timeout_ms: Override default timeout

        Returns:
            Text content of the element
        """
        await self._ensure_initialized()

        await self._page.wait_for_selector(
            selector,
            timeout=timeout_ms or self.timeout_ms,
        )
        return await self._page.text_content(selector) or ""

    async def get_attribute(
        self,
        selector: str,
        attribute: str,
        timeout_ms: Optional[int] = None,
    ) -> Optional[str]:
        """
        Get attribute value of an element.

        Args:
            selector: CSS selector or XPath
            attribute: Attribute name
            timeout_ms: Override default timeout

        Returns:
            Attribute value or None
        """
        await self._ensure_initialized()

        await self._page.wait_for_selector(
            selector,
            timeout=timeout_ms or self.timeout_ms,
        )
        return await self._page.get_attribute(selector, attribute)

    async def wait_for(
        self,
        selector: str,
        state: str = "visible",
        timeout_ms: Optional[int] = None,
    ) -> bool:
        """
        Wait for an element to reach a state.

        Args:
            selector: CSS selector or XPath
            state: State to wait for (attached, detached, visible, hidden)
            timeout_ms: Override default timeout

        Returns:
            True if element reached state, False on timeout
        """
        await self._ensure_initialized()

        try:
            await self._page.wait_for_selector(
                selector,
                state=state,
                timeout=timeout_ms or self.timeout_ms,
            )
            return True
        except Exception:
            return False

    async def execute_script(
        self,
        script: str,
        arg: Any = None,
    ) -> Any:
        """
        Execute JavaScript in the page context.

        Args:
            script: JavaScript code to execute
            arg: Optional argument passed to the script

        Returns:
            Result of script execution
        """
        await self._ensure_initialized()

        if arg is not None:
            return await self._page.evaluate(script, arg)
        return await self._page.evaluate(script)

    async def extract_data(
        self,
        selectors: Dict[str, str],
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract data from multiple elements.

        Args:
            selectors: Dict mapping field names to CSS selectors
            timeout_ms: Override default timeout

        Returns:
            Dict with extracted data
        """
        await self._ensure_initialized()

        result = {}
        for field_name, selector in selectors.items():
            try:
                await self._page.wait_for_selector(
                    selector,
                    timeout=timeout_ms or self.timeout_ms,
                )
                result[field_name] = await self._page.text_content(selector) or ""
            except Exception as e:
                result[field_name] = None
                logger.warning(f"Failed to extract {field_name} ({selector}): {e}")

        return result

    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies for the current context."""
        await self._ensure_initialized()
        return await self._context.cookies()

    async def set_cookie(
        self,
        name: str,
        value: str,
        domain: Optional[str] = None,
        path: str = "/",
        secure: bool = False,
        http_only: bool = False,
    ) -> None:
        """Set a cookie in the browser context."""
        await self._ensure_initialized()

        cookie = {
            "name": name,
            "value": value,
            "path": path,
            "secure": secure,
            "httpOnly": http_only,
        }
        if domain:
            cookie["domain"] = domain
        else:
            # Use current page domain
            from urllib.parse import urlparse

            parsed = urlparse(self._page.url)
            cookie["domain"] = parsed.netloc

        await self._context.add_cookies([cookie])

    async def clear_cookies(self) -> None:
        """Clear all cookies in the browser context."""
        await self._ensure_initialized()
        await self._context.clear_cookies()

    async def reload(self) -> PageState:
        """Reload the current page."""
        await self._ensure_initialized()

        import time

        start = time.perf_counter()

        await self._page.reload()
        load_time = (time.perf_counter() - start) * 1000

        return PageState(
            url=self._page.url,
            title=await self._page.title(),
            load_time_ms=load_time,
        )

    async def go_back(self) -> PageState:
        """Navigate back in history."""
        await self._ensure_initialized()

        await self._page.go_back()
        return PageState(
            url=self._page.url,
            title=await self._page.title(),
        )

    async def go_forward(self) -> PageState:
        """Navigate forward in history."""
        await self._ensure_initialized()

        await self._page.go_forward()
        return PageState(
            url=self._page.url,
            title=await self._page.title(),
        )

    async def get_html(self, selector: Optional[str] = None) -> str:
        """
        Get HTML content.

        Args:
            selector: If set, get innerHTML of element. Otherwise full page HTML.

        Returns:
            HTML content
        """
        await self._ensure_initialized()

        if selector:
            element = await self._page.query_selector(selector)
            if element:
                return await element.inner_html()
            return ""
        return await self._page.content()

    @property
    def current_url(self) -> str:
        """Get current page URL."""
        if self._page:
            return self._page.url
        return ""

    @property
    def is_initialized(self) -> bool:
        """Check if connector is initialized."""
        return self._initialized

    async def __aenter__(self) -> "PlaywrightConnector":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
