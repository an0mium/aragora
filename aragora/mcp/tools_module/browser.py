"""
MCP Tools for Browser automation operations.

Provides tools for browser automation using Playwright:
- browser_navigate: Navigate to a URL
- browser_click: Click an element
- browser_fill: Fill a form field
- browser_screenshot: Capture a screenshot
- browser_extract: Extract data from page
- browser_execute_script: Run JavaScript
- browser_close: Close browser session
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Global connector instance for session persistence across tool calls
_browser_connector = None


async def _get_connector(
    headless: bool = True,
    browser_type: str = "chromium",
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
) -> Any:
    """Get or create browser connector."""
    global _browser_connector

    if _browser_connector is None:
        try:
            from aragora.connectors.browser import PlaywrightConnector

            _browser_connector = PlaywrightConnector(
                headless=headless,
                browser_type=browser_type,
                allowed_domains=set(allowed_domains or []),
                blocked_domains=set(blocked_domains or []),
            )
            await _browser_connector.initialize()
        except ImportError:
            raise RuntimeError(
                "Playwright connector not available. Install with: pip install playwright"
            )

    return _browser_connector


async def _close_connector() -> None:
    """Close the global browser connector."""
    global _browser_connector
    if _browser_connector:
        await _browser_connector.close()
        _browser_connector = None


async def browser_navigate_tool(
    url: str,
    wait_until: str = "load",
    timeout_ms: int = 30000,
    headless: bool = True,
    allowed_domains: str = "",
    blocked_domains: str = "",
) -> Dict[str, Any]:
    """
    Navigate the browser to a URL.

    Args:
        url: The URL to navigate to
        wait_until: Wait condition - load, domcontentloaded, networkidle
        timeout_ms: Navigation timeout in milliseconds
        headless: Run browser in headless mode
        allowed_domains: Comma-separated list of allowed domains (empty = all allowed)
        blocked_domains: Comma-separated list of blocked domains

    Returns:
        Dict with page state including URL, title, and status
    """
    try:
        allowed = (
            [d.strip() for d in allowed_domains.split(",") if d.strip()]
            if allowed_domains
            else None
        )
        blocked = (
            [d.strip() for d in blocked_domains.split(",") if d.strip()]
            if blocked_domains
            else None
        )

        connector = await _get_connector(
            headless=headless,
            allowed_domains=allowed,
            blocked_domains=blocked,
        )

        state = await connector.navigate(url, wait_until=wait_until, timeout_ms=timeout_ms)

        return {
            "success": True,
            "url": state.url,
            "title": state.title,
            "status": state.status,
            "content_type": state.content_type,
        }

    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Browser navigation failed: {e}")
        return {"success": False, "error": f"Navigation failed: {str(e)}"}


async def browser_click_tool(
    selector: str,
    timeout_ms: int = 30000,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Click an element on the page.

    Args:
        selector: CSS selector for the element to click
        timeout_ms: Timeout in milliseconds
        force: Force click even if element is not visible

    Returns:
        Dict with action result
    """
    try:
        connector = await _get_connector()
        result = await connector.click(selector, timeout_ms=timeout_ms, force=force)

        return {
            "success": result.success,
            "action": result.action.value if result.action else "click",
            "selector": selector,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"Browser click failed: {e}")
        return {"success": False, "error": f"Click failed: {str(e)}"}


async def browser_fill_tool(
    selector: str,
    value: str,
    timeout_ms: int = 30000,
) -> Dict[str, Any]:
    """
    Fill a form field with a value.

    Args:
        selector: CSS selector for the input field
        value: Value to fill
        timeout_ms: Timeout in milliseconds

    Returns:
        Dict with action result
    """
    try:
        connector = await _get_connector()
        result = await connector.fill(selector, value, timeout_ms=timeout_ms)

        return {
            "success": result.success,
            "action": result.action.value if result.action else "fill",
            "selector": selector,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"Browser fill failed: {e}")
        return {"success": False, "error": f"Fill failed: {str(e)}"}


async def browser_screenshot_tool(
    selector: str = "",
    full_page: bool = False,
) -> Dict[str, Any]:
    """
    Capture a screenshot of the page or element.

    Args:
        selector: Optional CSS selector for element screenshot
        full_page: Capture full page (ignored if selector provided)

    Returns:
        Dict with base64-encoded screenshot
    """
    try:
        connector = await _get_connector()
        screenshot_bytes = await connector.screenshot(
            full_page=full_page,
            selector=selector if selector else None,
        )

        return {
            "success": True,
            "screenshot_base64": base64.b64encode(screenshot_bytes).decode(),
            "size_bytes": len(screenshot_bytes),
            "full_page": full_page,
            "selector": selector if selector else None,
        }

    except Exception as e:
        logger.error(f"Browser screenshot failed: {e}")
        return {"success": False, "error": f"Screenshot failed: {str(e)}"}


async def browser_get_text_tool(
    selector: str,
    timeout_ms: int = 30000,
) -> Dict[str, Any]:
    """
    Get the text content of an element.

    Args:
        selector: CSS selector for the element
        timeout_ms: Timeout in milliseconds

    Returns:
        Dict with text content
    """
    try:
        connector = await _get_connector()
        text = await connector.get_text(selector, timeout_ms=timeout_ms)

        return {
            "success": True,
            "text": text,
            "selector": selector,
        }

    except Exception as e:
        logger.error(f"Browser get_text failed: {e}")
        return {"success": False, "error": f"Get text failed: {str(e)}"}


async def browser_extract_tool(
    selectors: str,
    timeout_ms: int = 30000,
) -> Dict[str, Any]:
    """
    Extract data from multiple elements using CSS selectors.

    Args:
        selectors: JSON string mapping field names to CSS selectors
                   e.g. '{"title": "h1", "price": ".price"}'
        timeout_ms: Timeout in milliseconds

    Returns:
        Dict with extracted data
    """
    import json as json_module

    try:
        # Parse selectors JSON
        try:
            selector_map = json_module.loads(selectors)
        except json_module.JSONDecodeError:
            return {"success": False, "error": "Invalid JSON in selectors parameter"}

        connector = await _get_connector()
        data = await connector.extract_data(selector_map, timeout_ms=timeout_ms)

        return {
            "success": True,
            "data": data,
            "selectors": selector_map,
        }

    except Exception as e:
        logger.error(f"Browser extract failed: {e}")
        return {"success": False, "error": f"Extract failed: {str(e)}"}


async def browser_execute_script_tool(
    script: str,
) -> Dict[str, Any]:
    """
    Execute JavaScript in the browser context.

    Args:
        script: JavaScript code to execute

    Returns:
        Dict with script result
    """
    try:
        connector = await _get_connector()
        result = await connector.execute_script(script)

        return {
            "success": True,
            "result": result,
        }

    except Exception as e:
        logger.error(f"Browser execute_script failed: {e}")
        return {"success": False, "error": f"Script execution failed: {str(e)}"}


async def browser_wait_for_tool(
    selector: str,
    state: str = "visible",
    timeout_ms: int = 30000,
) -> Dict[str, Any]:
    """
    Wait for an element to reach a specific state.

    Args:
        selector: CSS selector for the element
        state: State to wait for - visible, hidden, attached, detached
        timeout_ms: Timeout in milliseconds

    Returns:
        Dict with wait result
    """
    try:
        connector = await _get_connector()
        found = await connector.wait_for(selector, state=state, timeout_ms=timeout_ms)

        return {
            "success": True,
            "found": found,
            "selector": selector,
            "state": state,
        }

    except Exception as e:
        logger.error(f"Browser wait_for failed: {e}")
        return {"success": False, "error": f"Wait failed: {str(e)}"}


async def browser_get_html_tool(
    selector: str = "",
) -> Dict[str, Any]:
    """
    Get the HTML content of the page or an element.

    Args:
        selector: Optional CSS selector (empty = full page)

    Returns:
        Dict with HTML content
    """
    try:
        connector = await _get_connector()
        html = await connector.get_html(selector if selector else None)

        return {
            "success": True,
            "html": html,
            "selector": selector if selector else "full_page",
            "length": len(html),
        }

    except Exception as e:
        logger.error(f"Browser get_html failed: {e}")
        return {"success": False, "error": f"Get HTML failed: {str(e)}"}


async def browser_close_tool() -> Dict[str, Any]:
    """
    Close the browser session.

    Returns:
        Dict with close result
    """
    try:
        await _close_connector()
        return {
            "success": True,
            "message": "Browser session closed",
        }

    except Exception as e:
        logger.error(f"Browser close failed: {e}")
        return {"success": False, "error": f"Close failed: {str(e)}"}


async def browser_get_cookies_tool(
    url: str = "",
) -> Dict[str, Any]:
    """
    Get cookies from the browser session.

    Args:
        url: Optional URL to filter cookies (empty = all cookies)

    Returns:
        Dict with cookies
    """
    try:
        connector = await _get_connector()
        cookies = await connector.get_cookies()

        return {
            "success": True,
            "cookies": cookies,
            "count": len(cookies),
        }

    except Exception as e:
        logger.error(f"Browser get_cookies failed: {e}")
        return {"success": False, "error": f"Get cookies failed: {str(e)}"}


async def browser_clear_cookies_tool() -> Dict[str, Any]:
    """
    Clear all cookies from the browser session.

    Returns:
        Dict with clear result
    """
    try:
        connector = await _get_connector()
        await connector.clear_cookies()

        return {
            "success": True,
            "message": "Cookies cleared",
        }

    except Exception as e:
        logger.error(f"Browser clear_cookies failed: {e}")
        return {"success": False, "error": f"Clear cookies failed: {str(e)}"}


# Export all tools
__all__ = [
    "browser_navigate_tool",
    "browser_click_tool",
    "browser_fill_tool",
    "browser_screenshot_tool",
    "browser_get_text_tool",
    "browser_extract_tool",
    "browser_execute_script_tool",
    "browser_wait_for_tool",
    "browser_get_html_tool",
    "browser_close_tool",
    "browser_get_cookies_tool",
    "browser_clear_cookies_tool",
]
