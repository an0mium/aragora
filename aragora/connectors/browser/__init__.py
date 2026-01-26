"""
Browser Automation Connector Package.

Provides browser automation capabilities using Playwright for:
- Web page navigation and interaction
- Form filling and button clicking
- Screenshot capture
- Data extraction
- JavaScript execution

Usage:
    from aragora.connectors.browser import PlaywrightConnector

    connector = PlaywrightConnector()
    await connector.initialize()

    # Navigate and interact
    page = await connector.navigate("https://example.com")
    await connector.click("button.submit")
    screenshot = await connector.screenshot()

    await connector.close()
"""

from .playwright_connector import (
    ActionResult,
    BrowserAction,
    PageState,
    PlaywrightConnector,
)

__all__ = [
    "PlaywrightConnector",
    "PageState",
    "ActionResult",
    "BrowserAction",
]
