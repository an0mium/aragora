"""
Tests for PlaywrightConnector browser automation.

Tests cover:
- Initialization and configuration
- Domain security sandboxing
- Navigation operations
- Element interactions (click, fill, select)
- Data extraction and screenshots
- Cookie management
- Error handling
"""

import pytest
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from aragora.connectors.browser.playwright_connector import (
    PlaywrightConnector,
    PageState,
    ActionResult,
    BrowserAction,
)

# Check if playwright is available
try:
    import playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Skip marker for tests requiring playwright
requires_playwright = pytest.mark.skipif(
    not PLAYWRIGHT_AVAILABLE, reason="playwright package not installed"
)


class TestPageState:
    """Tests for PageState dataclass."""

    def test_page_state_creation(self):
        """Test PageState creation with defaults."""
        state = PageState(url="https://example.com", title="Example")
        assert state.url == "https://example.com"
        assert state.title == "Example"
        assert state.content_type == "text/html"
        assert state.status_code == 200
        assert state.load_time_ms == 0.0
        assert state.timestamp  # Auto-generated

    def test_page_state_to_dict(self):
        """Test PageState serialization."""
        state = PageState(
            url="https://example.com",
            title="Example",
            content_type="application/json",
            status_code=201,
            load_time_ms=150.5,
        )
        data = state.to_dict()
        assert data["url"] == "https://example.com"
        assert data["title"] == "Example"
        assert data["content_type"] == "application/json"
        assert data["status_code"] == 201
        assert data["load_time_ms"] == 150.5


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_action_result_success(self):
        """Test successful ActionResult."""
        result = ActionResult(
            success=True,
            action="click",
            selector="button.submit",
            duration_ms=50.0,
        )
        assert result.success is True
        assert result.action == "click"
        assert result.selector == "button.submit"
        assert result.error is None

    def test_action_result_failure(self):
        """Test failed ActionResult."""
        result = ActionResult(
            success=False,
            action="fill",
            selector="input.missing",
            error="Element not found",
        )
        assert result.success is False
        assert result.error == "Element not found"

    def test_action_result_to_dict(self):
        """Test ActionResult serialization."""
        result = ActionResult(
            success=True,
            action="fill",
            selector="input.name",
            value="test value",
            duration_ms=25.0,
        )
        data = result.to_dict()
        assert data["success"] is True
        assert data["action"] == "fill"
        assert data["selector"] == "input.name"
        assert data["value"] == "test value"
        assert data["duration_ms"] == 25.0

    def test_action_result_to_dict_without_optional_fields(self):
        """Test ActionResult serialization excludes None fields."""
        result = ActionResult(success=True, action="screenshot")
        data = result.to_dict()
        assert "selector" not in data
        assert "value" not in data
        assert "error" not in data


class TestBrowserAction:
    """Tests for BrowserAction enum."""

    def test_browser_action_values(self):
        """Test BrowserAction enum has expected values."""
        assert BrowserAction.NAVIGATE.value == "navigate"
        assert BrowserAction.CLICK.value == "click"
        assert BrowserAction.FILL.value == "fill"
        assert BrowserAction.SCREENSHOT.value == "screenshot"
        assert BrowserAction.EXTRACT_DATA.value == "extract_data"


class TestPlaywrightConnectorInit:
    """Tests for PlaywrightConnector initialization."""

    def test_default_initialization(self):
        """Test default connector initialization."""
        connector = PlaywrightConnector()
        assert connector.headless is True
        assert connector.browser_type == "chromium"
        assert connector.timeout_ms == 30000
        assert connector.viewport_width == 1280
        assert connector.viewport_height == 720
        assert connector.is_initialized is False

    def test_custom_initialization(self):
        """Test connector with custom settings."""
        connector = PlaywrightConnector(
            headless=False,
            browser_type="firefox",
            timeout_ms=60000,
            viewport_width=1920,
            viewport_height=1080,
            user_agent="Custom Agent",
            ignore_https_errors=True,
        )
        assert connector.headless is False
        assert connector.browser_type == "firefox"
        assert connector.timeout_ms == 60000
        assert connector.viewport_width == 1920
        assert connector.viewport_height == 1080
        assert connector.user_agent == "Custom Agent"
        assert connector.ignore_https_errors is True

    def test_allowed_domains_initialization(self):
        """Test connector with allowed domains."""
        connector = PlaywrightConnector(allowed_domains={"example.com", "api.example.com"})
        assert "example.com" in connector.allowed_domains
        assert "api.example.com" in connector.allowed_domains

    def test_blocked_domains_initialization(self):
        """Test connector with blocked domains."""
        connector = PlaywrightConnector(blocked_domains={"ads.example.com", "tracker.com"})
        assert "ads.example.com" in connector.blocked_domains
        assert "tracker.com" in connector.blocked_domains

    def test_proxy_initialization(self):
        """Test connector with proxy configuration."""
        proxy = {"server": "http://proxy.example.com:8080"}
        connector = PlaywrightConnector(proxy=proxy)
        assert connector.proxy == proxy

    @patch.dict("os.environ", {"ARAGORA_BROWSER_ALLOWED_DOMAINS": "example.com,test.com"})
    def test_allowed_domains_from_env(self):
        """Test loading allowed domains from environment."""
        connector = PlaywrightConnector()
        assert "example.com" in connector.allowed_domains
        assert "test.com" in connector.allowed_domains

    @patch.dict("os.environ", {"ARAGORA_BROWSER_BLOCKED_DOMAINS": "ads.com,trackers.com"})
    def test_blocked_domains_from_env(self):
        """Test loading blocked domains from environment."""
        connector = PlaywrightConnector()
        assert "ads.com" in connector.blocked_domains
        assert "trackers.com" in connector.blocked_domains


class TestDomainSecurity:
    """Tests for domain security sandboxing."""

    def test_check_domain_allowed_no_restrictions(self):
        """Test domain check when no restrictions set."""
        connector = PlaywrightConnector()
        assert connector._check_domain_allowed("https://example.com") is True
        assert connector._check_domain_allowed("https://any-domain.org") is True

    def test_check_domain_allowed_with_allowed_list(self):
        """Test domain check with allowed domains."""
        connector = PlaywrightConnector(allowed_domains={"example.com"})
        assert connector._check_domain_allowed("https://example.com") is True
        assert connector._check_domain_allowed("https://sub.example.com") is True
        assert connector._check_domain_allowed("https://other.com") is False

    def test_check_domain_blocked(self):
        """Test domain check with blocked domains."""
        connector = PlaywrightConnector(blocked_domains={"blocked.com"})
        assert connector._check_domain_allowed("https://blocked.com") is False
        assert connector._check_domain_allowed("https://sub.blocked.com") is False
        assert connector._check_domain_allowed("https://allowed.com") is True

    def test_blocked_takes_priority(self):
        """Test that blocked domains take priority over allowed."""
        connector = PlaywrightConnector(
            allowed_domains={"example.com"},
            blocked_domains={"bad.example.com"},
        )
        assert connector._check_domain_allowed("https://example.com") is True
        assert connector._check_domain_allowed("https://good.example.com") is True
        assert connector._check_domain_allowed("https://bad.example.com") is False

    def test_domain_with_port(self):
        """Test domain check with port in URL."""
        connector = PlaywrightConnector(allowed_domains={"localhost"})
        assert connector._check_domain_allowed("http://localhost:3000") is True
        assert connector._check_domain_allowed("http://localhost:8080/path") is True


@requires_playwright
class TestPlaywrightConnectorMocked:
    """Tests for PlaywrightConnector with mocked Playwright."""

    @pytest.fixture
    def mock_playwright(self):
        """Create mock Playwright objects."""
        # Create mock chain
        mock_pw = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        # Setup browser launcher
        mock_pw.chromium = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw.firefox = MagicMock()
        mock_pw.firefox.launch = AsyncMock(return_value=mock_browser)
        mock_pw.webkit = MagicMock()
        mock_pw.webkit.launch = AsyncMock(return_value=mock_browser)
        mock_pw.stop = AsyncMock()

        # Setup browser context
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.cookies = AsyncMock(return_value=[])
        mock_context.add_cookies = AsyncMock()
        mock_context.clear_cookies = AsyncMock()
        mock_context.close = AsyncMock()

        # Setup page
        mock_page.set_default_timeout = MagicMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example Page")
        mock_page.content = AsyncMock(return_value="<html></html>")
        mock_page.text_content = AsyncMock(return_value="Test content")
        mock_page.get_attribute = AsyncMock(return_value="test-value")
        mock_page.close = AsyncMock()

        # Create the async_playwright context manager mock
        async_pw_mock = MagicMock()
        async_pw_mock.return_value.start = AsyncMock(return_value=mock_pw)

        with patch("playwright.async_api.async_playwright", async_pw_mock):
            yield {
                "mock": async_pw_mock,
                "playwright": mock_pw,
                "browser": mock_browser,
                "context": mock_context,
                "page": mock_page,
            }

    @pytest.mark.asyncio
    async def test_initialize(self, mock_playwright):
        """Test connector initialization."""
        connector = PlaywrightConnector()
        await connector.initialize()

        assert connector.is_initialized is True
        mock_playwright["playwright"].chromium.launch.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_firefox(self, mock_playwright):
        """Test initialization with Firefox."""
        connector = PlaywrightConnector(browser_type="firefox")
        await connector.initialize()

        mock_playwright["playwright"].firefox.launch.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_webkit(self, mock_playwright):
        """Test initialization with WebKit."""
        connector = PlaywrightConnector(browser_type="webkit")
        await connector.initialize()

        mock_playwright["playwright"].webkit.launch.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, mock_playwright):
        """Test connector close cleanup."""
        connector = PlaywrightConnector()
        await connector.initialize()
        await connector.close()

        assert connector.is_initialized is False
        mock_playwright["page"].close.assert_called_once()
        mock_playwright["context"].close.assert_called_once()
        mock_playwright["browser"].close.assert_called_once()

    @pytest.mark.asyncio
    async def test_navigate(self, mock_playwright):
        """Test page navigation."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_playwright["page"].goto = AsyncMock(return_value=mock_response)

        connector = PlaywrightConnector()
        await connector.initialize()
        state = await connector.navigate("https://example.com")

        assert state.url == "https://example.com"
        assert state.status_code == 200
        mock_playwright["page"].goto.assert_called_once()

    @pytest.mark.asyncio
    async def test_navigate_blocked_domain(self, mock_playwright):
        """Test navigation to blocked domain raises error."""
        connector = PlaywrightConnector(blocked_domains={"blocked.com"})
        await connector.initialize()

        with pytest.raises(ValueError, match="domain not allowed"):
            await connector.navigate("https://blocked.com")

    @pytest.mark.asyncio
    async def test_click_success(self, mock_playwright):
        """Test successful click action."""
        mock_playwright["page"].click = AsyncMock()

        connector = PlaywrightConnector()
        await connector.initialize()
        result = await connector.click("button.submit")

        assert result.success is True
        assert result.action == "click"
        assert result.selector == "button.submit"
        mock_playwright["page"].click.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_failure(self, mock_playwright):
        """Test click action failure."""
        mock_playwright["page"].click = AsyncMock(side_effect=Exception("Element not found"))

        connector = PlaywrightConnector()
        await connector.initialize()
        result = await connector.click("button.missing")

        assert result.success is False
        assert "Element not found" in result.error

    @pytest.mark.asyncio
    async def test_fill_success(self, mock_playwright):
        """Test successful fill action."""
        mock_playwright["page"].fill = AsyncMock()

        connector = PlaywrightConnector()
        await connector.initialize()
        result = await connector.fill("input.name", "Test Value")

        assert result.success is True
        assert result.value == "Test Value"
        mock_playwright["page"].fill.assert_called_once_with(
            "input.name", "Test Value", timeout=30000
        )

    @pytest.mark.asyncio
    async def test_select_success(self, mock_playwright):
        """Test successful select action."""
        mock_playwright["page"].select_option = AsyncMock()

        connector = PlaywrightConnector()
        await connector.initialize()
        result = await connector.select("select.country", "US")

        assert result.success is True
        assert result.value == "US"

    @pytest.mark.asyncio
    async def test_screenshot(self, mock_playwright):
        """Test screenshot capture."""
        mock_playwright["page"].screenshot = AsyncMock(return_value=b"PNG_DATA")

        connector = PlaywrightConnector()
        await connector.initialize()
        screenshot = await connector.screenshot()

        assert screenshot == b"PNG_DATA"
        mock_playwright["page"].screenshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_screenshot_full_page(self, mock_playwright):
        """Test full page screenshot."""
        mock_playwright["page"].screenshot = AsyncMock(return_value=b"FULL_PNG")

        connector = PlaywrightConnector()
        await connector.initialize()
        await connector.screenshot(full_page=True)

        mock_playwright["page"].screenshot.assert_called_once_with(full_page=True)

    @pytest.mark.asyncio
    async def test_screenshot_element(self, mock_playwright):
        """Test element screenshot."""
        mock_element = AsyncMock()
        mock_element.screenshot = AsyncMock(return_value=b"ELEMENT_PNG")
        mock_playwright["page"].query_selector = AsyncMock(return_value=mock_element)

        connector = PlaywrightConnector()
        await connector.initialize()
        screenshot = await connector.screenshot(selector="div.target")

        assert screenshot == b"ELEMENT_PNG"
        mock_playwright["page"].query_selector.assert_called_once_with("div.target")

    @pytest.mark.asyncio
    async def test_get_text(self, mock_playwright):
        """Test getting element text."""
        mock_playwright["page"].wait_for_selector = AsyncMock()

        connector = PlaywrightConnector()
        await connector.initialize()
        text = await connector.get_text("h1.title")

        assert text == "Test content"

    @pytest.mark.asyncio
    async def test_get_attribute(self, mock_playwright):
        """Test getting element attribute."""
        mock_playwright["page"].wait_for_selector = AsyncMock()

        connector = PlaywrightConnector()
        await connector.initialize()
        value = await connector.get_attribute("a.link", "href")

        assert value == "test-value"
        mock_playwright["page"].get_attribute.assert_called_once_with("a.link", "href")

    @pytest.mark.asyncio
    async def test_wait_for_success(self, mock_playwright):
        """Test wait for element success."""
        mock_playwright["page"].wait_for_selector = AsyncMock()

        connector = PlaywrightConnector()
        await connector.initialize()
        result = await connector.wait_for("div.content", state="visible")

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_timeout(self, mock_playwright):
        """Test wait for element timeout."""
        mock_playwright["page"].wait_for_selector = AsyncMock(side_effect=Exception("Timeout"))

        connector = PlaywrightConnector()
        await connector.initialize()
        result = await connector.wait_for("div.missing", timeout_ms=1000)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_script(self, mock_playwright):
        """Test JavaScript execution."""
        mock_playwright["page"].evaluate = AsyncMock(return_value=42)

        connector = PlaywrightConnector()
        await connector.initialize()
        result = await connector.execute_script("return 21 * 2")

        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_script_with_arg(self, mock_playwright):
        """Test JavaScript execution with argument."""
        mock_playwright["page"].evaluate = AsyncMock(return_value=10)

        connector = PlaywrightConnector()
        await connector.initialize()
        result = await connector.execute_script("arg => arg * 2", 5)

        mock_playwright["page"].evaluate.assert_called_once_with("arg => arg * 2", 5)

    @pytest.mark.asyncio
    async def test_extract_data(self, mock_playwright):
        """Test data extraction from multiple selectors."""
        mock_playwright["page"].wait_for_selector = AsyncMock()
        mock_playwright["page"].text_content = AsyncMock(side_effect=["Title", "Description", None])

        connector = PlaywrightConnector()
        await connector.initialize()
        data = await connector.extract_data(
            {
                "title": "h1.title",
                "desc": "p.description",
            }
        )

        assert data["title"] == "Title"
        assert data["desc"] == "Description"

    @pytest.mark.asyncio
    async def test_get_cookies(self, mock_playwright):
        """Test getting cookies."""
        mock_playwright["context"].cookies = AsyncMock(
            return_value=[{"name": "session", "value": "abc123"}]
        )

        connector = PlaywrightConnector()
        await connector.initialize()
        cookies = await connector.get_cookies()

        assert len(cookies) == 1
        assert cookies[0]["name"] == "session"

    @pytest.mark.asyncio
    async def test_set_cookie(self, mock_playwright):
        """Test setting a cookie."""
        connector = PlaywrightConnector()
        await connector.initialize()
        await connector.set_cookie("test", "value", domain="example.com")

        mock_playwright["context"].add_cookies.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cookies(self, mock_playwright):
        """Test clearing cookies."""
        connector = PlaywrightConnector()
        await connector.initialize()
        await connector.clear_cookies()

        mock_playwright["context"].clear_cookies.assert_called_once()

    @pytest.mark.asyncio
    async def test_reload(self, mock_playwright):
        """Test page reload."""
        mock_playwright["page"].reload = AsyncMock()

        connector = PlaywrightConnector()
        await connector.initialize()
        state = await connector.reload()

        assert state.url == "https://example.com"
        mock_playwright["page"].reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_go_back(self, mock_playwright):
        """Test navigation back."""
        mock_playwright["page"].go_back = AsyncMock()

        connector = PlaywrightConnector()
        await connector.initialize()
        await connector.go_back()

        mock_playwright["page"].go_back.assert_called_once()

    @pytest.mark.asyncio
    async def test_go_forward(self, mock_playwright):
        """Test navigation forward."""
        mock_playwright["page"].go_forward = AsyncMock()

        connector = PlaywrightConnector()
        await connector.initialize()
        await connector.go_forward()

        mock_playwright["page"].go_forward.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_html_full_page(self, mock_playwright):
        """Test getting full page HTML."""
        connector = PlaywrightConnector()
        await connector.initialize()
        html = await connector.get_html()

        assert html == "<html></html>"

    @pytest.mark.asyncio
    async def test_get_html_element(self, mock_playwright):
        """Test getting element HTML."""
        mock_element = AsyncMock()
        mock_element.inner_html = AsyncMock(return_value="<span>Content</span>")
        mock_playwright["page"].query_selector = AsyncMock(return_value=mock_element)

        connector = PlaywrightConnector()
        await connector.initialize()
        html = await connector.get_html("div.content")

        assert html == "<span>Content</span>"

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_playwright):
        """Test async context manager."""
        async with PlaywrightConnector() as connector:
            assert connector.is_initialized is True

        # After exiting context, should be closed
        mock_playwright["page"].close.assert_called()

    @pytest.mark.asyncio
    async def test_current_url_property(self, mock_playwright):
        """Test current_url property."""
        connector = PlaywrightConnector()
        await connector.initialize()

        assert connector.current_url == "https://example.com"

    @pytest.mark.asyncio
    async def test_current_url_not_initialized(self):
        """Test current_url when not initialized."""
        connector = PlaywrightConnector()
        assert connector.current_url == ""


@requires_playwright
class TestPlaywrightConnectorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_double_initialize(self):
        """Test that double initialization is safe."""
        mock_pw = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_pw.chromium = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_page.set_default_timeout = MagicMock()

        async_pw_mock = MagicMock()
        async_pw_mock.return_value.start = AsyncMock(return_value=mock_pw)

        with patch("playwright.async_api.async_playwright", async_pw_mock):
            connector = PlaywrightConnector()
            await connector.initialize()
            await connector.initialize()  # Should be no-op

            # Should only launch once
            assert mock_pw.chromium.launch.call_count == 1

    @pytest.mark.asyncio
    async def test_close_not_initialized(self):
        """Test closing when not initialized is safe."""
        connector = PlaywrightConnector()
        await connector.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_playwright_not_installed(self):
        """Test error message when Playwright not installed."""
        connector = PlaywrightConnector()

        # Remove playwright from sys.modules to simulate not installed
        original_modules = sys.modules.copy()

        # Remove playwright-related modules
        modules_to_remove = [k for k in sys.modules if k.startswith("playwright")]
        for mod in modules_to_remove:
            sys.modules.pop(mod, None)

        # Make the import raise ImportError
        with patch.dict("sys.modules", {"playwright": None, "playwright.async_api": None}):
            # Reset connector state
            connector._initialized = False

            with pytest.raises(ImportError, match="Playwright is not installed"):
                await connector.initialize()

        # Restore modules
        sys.modules.update(original_modules)

    @pytest.mark.asyncio
    async def test_screenshot_element_not_found(self):
        """Test screenshot when element not found."""
        mock_pw = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_pw.chromium = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_page.set_default_timeout = MagicMock()
        mock_page.query_selector = AsyncMock(return_value=None)

        async_pw_mock = MagicMock()
        async_pw_mock.return_value.start = AsyncMock(return_value=mock_pw)

        with patch("playwright.async_api.async_playwright", async_pw_mock):
            connector = PlaywrightConnector()
            await connector.initialize()

            with pytest.raises(ValueError, match="Element not found"):
                await connector.screenshot(selector="div.nonexistent")
