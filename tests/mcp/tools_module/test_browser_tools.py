"""Tests for MCP browser tools execution logic."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.browser import (
    browser_clear_cookies_tool,
    browser_click_tool,
    browser_close_tool,
    browser_execute_script_tool,
    browser_extract_tool,
    browser_fill_tool,
    browser_get_cookies_tool,
    browser_get_html_tool,
    browser_get_text_tool,
    browser_navigate_tool,
    browser_screenshot_tool,
    browser_wait_for_tool,
)

pytest.importorskip("mcp")


@pytest.fixture(autouse=True)
def reset_browser_connector():
    """Reset global browser connector between tests."""
    import aragora.mcp.tools_module.browser as browser_mod

    browser_mod._browser_connector = None
    yield
    browser_mod._browser_connector = None


class TestBrowserNavigateTool:
    """Tests for browser_navigate_tool."""

    @pytest.mark.asyncio
    async def test_navigate_success(self):
        """Test successful navigation."""
        mock_connector = AsyncMock()
        mock_state = MagicMock()
        mock_state.url = "https://example.com"
        mock_state.title = "Example"
        mock_state.status = 200
        mock_state.content_type = "text/html"
        mock_connector.navigate.return_value = mock_state

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_navigate_tool(url="https://example.com")

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["title"] == "Example"
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_navigate_with_domain_filters(self):
        """Test navigation with domain filtering."""
        mock_connector = AsyncMock()
        mock_state = MagicMock()
        mock_state.url = "https://allowed.com"
        mock_state.title = "Allowed"
        mock_state.status = 200
        mock_state.content_type = "text/html"
        mock_connector.navigate.return_value = mock_state

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_navigate_tool(
                url="https://allowed.com",
                allowed_domains="allowed.com,other.com",
                blocked_domains="blocked.com",
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_navigate_value_error(self):
        """Test navigation with invalid URL returns error."""
        mock_connector = AsyncMock()
        mock_connector.navigate.side_effect = ValueError("Invalid URL")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_navigate_tool(url="not-a-url")

        assert result["success"] is False
        assert "Invalid URL" in result["error"]

    @pytest.mark.asyncio
    async def test_navigate_generic_exception(self):
        """Test navigation with unexpected error."""
        mock_connector = AsyncMock()
        mock_connector.navigate.side_effect = RuntimeError("Connection failed")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_navigate_tool(url="https://example.com")

        assert result["success"] is False
        assert "Navigation failed" in result["error"]


class TestBrowserClickTool:
    """Tests for browser_click_tool."""

    @pytest.mark.asyncio
    async def test_click_success(self):
        """Test successful click."""
        mock_connector = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.action = MagicMock()
        mock_result.action.value = "click"
        mock_result.error = None
        mock_connector.click.return_value = mock_result

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_click_tool(selector="#button")

        assert result["success"] is True
        assert result["selector"] == "#button"
        assert result["action"] == "click"

    @pytest.mark.asyncio
    async def test_click_failure(self):
        """Test click on non-existent element."""
        mock_connector = AsyncMock()
        mock_connector.click.side_effect = Exception("Element not found")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_click_tool(selector="#nonexistent")

        assert result["success"] is False
        assert "Click failed" in result["error"]


class TestBrowserFillTool:
    """Tests for browser_fill_tool."""

    @pytest.mark.asyncio
    async def test_fill_success(self):
        """Test successful form fill."""
        mock_connector = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.action = MagicMock()
        mock_result.action.value = "fill"
        mock_result.error = None
        mock_connector.fill.return_value = mock_result

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_fill_tool(selector="#email", value="test@example.com")

        assert result["success"] is True
        assert result["selector"] == "#email"

    @pytest.mark.asyncio
    async def test_fill_failure(self):
        """Test fill on non-existent input."""
        mock_connector = AsyncMock()
        mock_connector.fill.side_effect = Exception("Input not found")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_fill_tool(selector="#missing", value="test")

        assert result["success"] is False
        assert "Fill failed" in result["error"]


class TestBrowserScreenshotTool:
    """Tests for browser_screenshot_tool."""

    @pytest.mark.asyncio
    async def test_screenshot_success(self):
        """Test successful screenshot capture."""
        mock_connector = AsyncMock()
        fake_bytes = b"\x89PNG\r\n\x1a\nfakeimage"
        mock_connector.screenshot.return_value = fake_bytes

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_screenshot_tool(full_page=True)

        assert result["success"] is True
        assert "screenshot_base64" in result
        assert result["size_bytes"] == len(fake_bytes)
        assert result["full_page"] is True
        # Verify base64 encoding
        decoded = base64.b64decode(result["screenshot_base64"])
        assert decoded == fake_bytes

    @pytest.mark.asyncio
    async def test_screenshot_with_selector(self):
        """Test screenshot of specific element."""
        mock_connector = AsyncMock()
        mock_connector.screenshot.return_value = b"elementimg"

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_screenshot_tool(selector="#header")

        assert result["success"] is True
        assert result["selector"] == "#header"

    @pytest.mark.asyncio
    async def test_screenshot_failure(self):
        """Test screenshot failure."""
        mock_connector = AsyncMock()
        mock_connector.screenshot.side_effect = Exception("No page loaded")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_screenshot_tool()

        assert result["success"] is False
        assert "Screenshot failed" in result["error"]


class TestBrowserGetTextTool:
    """Tests for browser_get_text_tool."""

    @pytest.mark.asyncio
    async def test_get_text_success(self):
        """Test successful text extraction."""
        mock_connector = AsyncMock()
        mock_connector.get_text.return_value = "Hello World"

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_get_text_tool(selector="h1")

        assert result["success"] is True
        assert result["text"] == "Hello World"
        assert result["selector"] == "h1"

    @pytest.mark.asyncio
    async def test_get_text_failure(self):
        """Test text extraction failure."""
        mock_connector = AsyncMock()
        mock_connector.get_text.side_effect = Exception("Selector not found")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_get_text_tool(selector="#missing")

        assert result["success"] is False


class TestBrowserExtractTool:
    """Tests for browser_extract_tool."""

    @pytest.mark.asyncio
    async def test_extract_success(self):
        """Test successful data extraction."""
        mock_connector = AsyncMock()
        mock_connector.extract_data.return_value = {"title": "Test Page", "price": "$9.99"}

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_extract_tool(selectors='{"title": "h1", "price": ".price"}')

        assert result["success"] is True
        assert result["data"]["title"] == "Test Page"
        assert result["data"]["price"] == "$9.99"

    @pytest.mark.asyncio
    async def test_extract_invalid_json(self):
        """Test extract with invalid JSON selectors."""
        result = await browser_extract_tool(selectors="not json")
        assert result["success"] is False
        assert "Invalid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_failure(self):
        """Test extract with connector error."""
        mock_connector = AsyncMock()
        mock_connector.extract_data.side_effect = Exception("Page not loaded")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_extract_tool(selectors='{"title": "h1"}')

        assert result["success"] is False
        assert "Extract failed" in result["error"]


class TestBrowserExecuteScriptTool:
    """Tests for browser_execute_script_tool."""

    @pytest.mark.asyncio
    async def test_execute_script_success(self):
        """Test successful script execution."""
        mock_connector = AsyncMock()
        mock_connector.execute_script.return_value = 42

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_execute_script_tool(script="return 21 * 2")

        assert result["success"] is True
        assert result["result"] == 42

    @pytest.mark.asyncio
    async def test_execute_script_failure(self):
        """Test script execution failure."""
        mock_connector = AsyncMock()
        mock_connector.execute_script.side_effect = Exception("Script error")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_execute_script_tool(script="invalid()")

        assert result["success"] is False
        assert "Script execution failed" in result["error"]


class TestBrowserWaitForTool:
    """Tests for browser_wait_for_tool."""

    @pytest.mark.asyncio
    async def test_wait_for_success(self):
        """Test successful wait for element."""
        mock_connector = AsyncMock()
        mock_connector.wait_for.return_value = True

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_wait_for_tool(selector="#loading", state="hidden")

        assert result["success"] is True
        assert result["found"] is True
        assert result["state"] == "hidden"

    @pytest.mark.asyncio
    async def test_wait_for_timeout(self):
        """Test wait for element timeout."""
        mock_connector = AsyncMock()
        mock_connector.wait_for.side_effect = Exception("Timeout")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_wait_for_tool(selector="#spinner", timeout_ms=100)

        assert result["success"] is False
        assert "Wait failed" in result["error"]


class TestBrowserGetHtmlTool:
    """Tests for browser_get_html_tool."""

    @pytest.mark.asyncio
    async def test_get_html_full_page(self):
        """Test getting full page HTML."""
        mock_connector = AsyncMock()
        mock_connector.get_html.return_value = "<html><body>Test</body></html>"

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_get_html_tool()

        assert result["success"] is True
        assert "<html>" in result["html"]
        assert result["selector"] == "full_page"
        assert result["length"] > 0

    @pytest.mark.asyncio
    async def test_get_html_with_selector(self):
        """Test getting element HTML."""
        mock_connector = AsyncMock()
        mock_connector.get_html.return_value = "<div>Content</div>"

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_get_html_tool(selector="#main")

        assert result["success"] is True
        assert result["selector"] == "#main"

    @pytest.mark.asyncio
    async def test_get_html_failure(self):
        """Test get HTML failure."""
        mock_connector = AsyncMock()
        mock_connector.get_html.side_effect = Exception("No page")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_get_html_tool()

        assert result["success"] is False


class TestBrowserCloseTool:
    """Tests for browser_close_tool."""

    @pytest.mark.asyncio
    async def test_close_success(self):
        """Test successful browser close."""
        import aragora.mcp.tools_module.browser as browser_mod

        mock_connector = AsyncMock()
        browser_mod._browser_connector = mock_connector

        result = await browser_close_tool()

        assert result["success"] is True
        assert "closed" in result["message"].lower()
        mock_connector.close.assert_awaited_once()
        assert browser_mod._browser_connector is None

    @pytest.mark.asyncio
    async def test_close_when_no_connector(self):
        """Test close when no browser is open."""
        result = await browser_close_tool()
        assert result["success"] is True


class TestBrowserCookieTools:
    """Tests for browser cookie tools."""

    @pytest.mark.asyncio
    async def test_get_cookies_success(self):
        """Test getting cookies."""
        mock_connector = AsyncMock()
        mock_connector.get_cookies.return_value = [
            {"name": "session", "value": "abc123", "domain": "example.com"},
        ]

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_get_cookies_tool()

        assert result["success"] is True
        assert result["count"] == 1
        assert result["cookies"][0]["name"] == "session"

    @pytest.mark.asyncio
    async def test_get_cookies_failure(self):
        """Test get cookies failure."""
        mock_connector = AsyncMock()
        mock_connector.get_cookies.side_effect = Exception("No context")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_get_cookies_tool()

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_clear_cookies_success(self):
        """Test clearing cookies."""
        mock_connector = AsyncMock()

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_clear_cookies_tool()

        assert result["success"] is True
        assert "cleared" in result["message"].lower()
        mock_connector.clear_cookies.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_clear_cookies_failure(self):
        """Test clear cookies failure."""
        mock_connector = AsyncMock()
        mock_connector.clear_cookies.side_effect = Exception("Failed")

        with patch(
            "aragora.mcp.tools_module.browser._get_connector",
            return_value=mock_connector,
        ):
            result = await browser_clear_cookies_tool()

        assert result["success"] is False
