"""Tests for PlaywrightActionExecutor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.computer_use.actions import (
    ActionResult,
    ActionType,
    ClickAction,
    ClickButton,
    DragAction,
    KeyAction,
    MoveAction,
    ScreenshotAction,
    ScrollAction,
    ScrollDirection,
    TypeAction,
    WaitAction,
)
from aragora.computer_use.executor import ExecutorConfig, PlaywrightActionExecutor


class TestExecutorConfig:
    """Test ExecutorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExecutorConfig()
        assert config.browser_type == "chromium"
        assert config.headless is True
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080
        assert config.screenshot_type == "png"
        assert config.action_timeout_ms == 10000

    def test_custom_values(self):
        """Test custom configuration."""
        config = ExecutorConfig(
            browser_type="firefox",
            headless=False,
            viewport_width=1280,
            viewport_height=720,
        )
        assert config.browser_type == "firefox"
        assert config.headless is False
        assert config.viewport_width == 1280
        assert config.viewport_height == 720


class TestPlaywrightActionExecutor:
    """Test PlaywrightActionExecutor class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        executor = PlaywrightActionExecutor()
        assert executor._config.browser_type == "chromium"
        assert executor.is_running is False
        assert executor.action_count == 0
        assert executor.error_count == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ExecutorConfig(headless=False, viewport_width=1280)
        executor = PlaywrightActionExecutor(config=config)
        assert executor._config.headless is False
        assert executor._config.viewport_width == 1280

    @pytest.mark.asyncio
    async def test_execute_not_running(self):
        """Test execute returns error when not running."""
        executor = PlaywrightActionExecutor()
        action = ScreenshotAction()
        result = await executor.execute(action)

        assert result.success is False
        assert result.error == "Executor not running"
        assert result.action_type == ActionType.SCREENSHOT

    @pytest.mark.asyncio
    async def test_take_screenshot_not_running(self):
        """Test take_screenshot returns empty when not running."""
        executor = PlaywrightActionExecutor()
        screenshot = await executor.take_screenshot()
        assert screenshot == ""

    @pytest.mark.asyncio
    async def test_get_current_url_not_running(self):
        """Test get_current_url returns None when not running."""
        executor = PlaywrightActionExecutor()
        url = await executor.get_current_url()
        assert url is None

    @pytest.mark.asyncio
    async def test_navigate_not_running(self):
        """Test navigate returns False when not running."""
        executor = PlaywrightActionExecutor()
        success = await executor.navigate("https://example.com")
        assert success is False


class TestPlaywrightActionExecutorMocked:
    """Test PlaywrightActionExecutor with mocked Playwright.

    Note: These tests require mocking the playwright import inside the start() method.
    The mocking is done at the playwright.async_api module level.
    """

    @pytest.fixture
    def mock_playwright_setup(self):
        """Create mock Playwright instance and setup context manager mock."""
        mock_page = MagicMock()
        mock_page.url = "https://example.com"
        mock_page.screenshot = AsyncMock(return_value=b"fake_screenshot")
        mock_page.mouse = MagicMock()
        mock_page.mouse.click = AsyncMock()
        mock_page.mouse.dblclick = AsyncMock()
        mock_page.mouse.move = AsyncMock()
        mock_page.mouse.down = AsyncMock()
        mock_page.mouse.up = AsyncMock()
        mock_page.mouse.wheel = AsyncMock()
        mock_page.keyboard = MagicMock()
        mock_page.keyboard.type = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.keyboard.down = AsyncMock()
        mock_page.keyboard.up = AsyncMock()
        mock_page.close = AsyncMock()
        mock_page.goto = AsyncMock()

        mock_context = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.set_default_timeout = MagicMock()
        mock_context.set_default_navigation_timeout = MagicMock()
        mock_context.close = AsyncMock()

        mock_browser = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_browser_type = MagicMock()
        mock_browser_type.launch = AsyncMock(return_value=mock_browser)

        mock_pw_instance = MagicMock()
        mock_pw_instance.chromium = mock_browser_type
        mock_pw_instance.stop = AsyncMock()

        # Mock the async_playwright context manager
        mock_async_pw = MagicMock()
        mock_async_pw.start = AsyncMock(return_value=mock_pw_instance)

        # Ensure module import works even if playwright isn't installed
        import sys
        import types

        playwright_module = types.ModuleType("playwright")
        async_api_module = types.ModuleType("playwright.async_api")
        async_api_module.async_playwright = MagicMock(return_value=mock_async_pw)

        sys.modules.setdefault("playwright", playwright_module)
        sys.modules["playwright.async_api"] = async_api_module

        return mock_async_pw, mock_page

    @pytest.mark.asyncio
    async def test_start_and_stop(self, mock_playwright_setup):
        """Test start and stop lifecycle."""
        mock_async_pw, mock_page = mock_playwright_setup

        with patch("playwright.async_api.async_playwright", return_value=mock_async_pw):
            executor = PlaywrightActionExecutor()
            await executor.start()

            assert executor.is_running is True

            await executor.stop()
            assert executor.is_running is False

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_playwright_setup):
        """Test context manager usage."""
        mock_async_pw, mock_page = mock_playwright_setup

        with patch("playwright.async_api.async_playwright", return_value=mock_async_pw):
            async with PlaywrightActionExecutor() as executor:
                assert executor.is_running is True

            assert executor.is_running is False

    @pytest.mark.asyncio
    async def test_execute_screenshot(self, mock_playwright_setup):
        """Test executing screenshot action."""
        mock_async_pw, mock_page = mock_playwright_setup

        with patch("playwright.async_api.async_playwright", return_value=mock_async_pw):
            async with PlaywrightActionExecutor() as executor:
                action = ScreenshotAction()
                result = await executor.execute(action)

                assert result.success is True
                assert result.action_type == ActionType.SCREENSHOT
                assert result.screenshot_b64 is not None

    @pytest.mark.asyncio
    async def test_execute_click(self, mock_playwright_setup):
        """Test executing click action."""
        mock_async_pw, mock_page = mock_playwright_setup

        with patch("playwright.async_api.async_playwright", return_value=mock_async_pw):
            async with PlaywrightActionExecutor() as executor:
                action = ClickAction(x=100, y=200)
                result = await executor.execute(action)

                assert result.success is True
                assert result.action_type == ActionType.CLICK
                mock_page.mouse.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_type(self, mock_playwright_setup):
        """Test executing type action."""
        mock_async_pw, mock_page = mock_playwright_setup

        with patch("playwright.async_api.async_playwright", return_value=mock_async_pw):
            async with PlaywrightActionExecutor() as executor:
                action = TypeAction(text="Hello, World!")
                result = await executor.execute(action)

                assert result.success is True
                assert result.action_type == ActionType.TYPE
                mock_page.keyboard.type.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_key(self, mock_playwright_setup):
        """Test executing key press action."""
        mock_async_pw, mock_page = mock_playwright_setup

        with patch("playwright.async_api.async_playwright", return_value=mock_async_pw):
            async with PlaywrightActionExecutor() as executor:
                action = KeyAction(key="Enter")
                result = await executor.execute(action)

                assert result.success is True
                assert result.action_type == ActionType.KEY
                mock_page.keyboard.press.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_scroll(self, mock_playwright_setup):
        """Test executing scroll action."""
        mock_async_pw, mock_page = mock_playwright_setup

        with patch("playwright.async_api.async_playwright", return_value=mock_async_pw):
            async with PlaywrightActionExecutor() as executor:
                action = ScrollAction(direction=ScrollDirection.DOWN, amount=3)
                result = await executor.execute(action)

                assert result.success is True
                assert result.action_type == ActionType.SCROLL
                mock_page.mouse.wheel.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_wait(self, mock_playwright_setup):
        """Test executing wait action."""
        mock_async_pw, mock_page = mock_playwright_setup

        with patch("playwright.async_api.async_playwright", return_value=mock_async_pw):
            async with PlaywrightActionExecutor() as executor:
                action = WaitAction(duration_ms=100)
                result = await executor.execute(action)

                assert result.success is True
                assert result.action_type == ActionType.WAIT

    @pytest.mark.asyncio
    async def test_action_count_tracking(self, mock_playwright_setup):
        """Test that action count is tracked."""
        mock_async_pw, mock_page = mock_playwright_setup

        with patch("playwright.async_api.async_playwright", return_value=mock_async_pw):
            async with PlaywrightActionExecutor() as executor:
                assert executor.action_count == 0

                await executor.execute(ScreenshotAction())
                assert executor.action_count == 1

                await executor.execute(ClickAction(x=0, y=0))
                assert executor.action_count == 2
