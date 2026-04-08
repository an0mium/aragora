"""Shared fixtures for channel dock tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.channels.dock import ChannelCapability, ChannelDock, SendResult
from aragora.channels.normalized import MessageFormat, NormalizedMessage
from aragora.channels.registry import DockRegistry


class MockDock(ChannelDock):
    """Mock dock for registry tests."""

    PLATFORM = "mock"
    CAPABILITIES = ChannelCapability.RICH_TEXT | ChannelCapability.BUTTONS

    def __init__(self, config=None):
        self.config = config or {}
        self._initialized = False

    @property
    def is_initialized(self):
        return self._initialized

    async def initialize(self):
        self._initialized = True

    async def send_message(self, channel_id, message):
        return None

    async def send_result(self, channel_id, result, **kwargs):
        return None

    async def send_error(self, channel_id, error, **kwargs):
        return None


class VoiceDock(ChannelDock):
    """Mock dock with voice capability for registry tests."""

    PLATFORM = "voice_platform"
    CAPABILITIES = ChannelCapability.RICH_TEXT | ChannelCapability.VOICE

    def __init__(self, config=None):
        self.config = config or {}
        self._initialized = False

    @property
    def is_initialized(self):
        return self._initialized

    async def initialize(self):
        self._initialized = True

    async def send_message(self, channel_id, message):
        return None

    async def send_result(self, channel_id, result, **kwargs):
        return None

    async def send_error(self, channel_id, error, **kwargs):
        return None


class FakeSlackDock(ChannelDock):
    PLATFORM = "slack"
    CAPABILITIES = (
        ChannelCapability.RICH_TEXT | ChannelCapability.BUTTONS | ChannelCapability.THREADS
    )

    async def send_message(self, channel_id, message, **kwargs):
        return SendResult.ok(message_id="msg-1", platform=self.PLATFORM, channel_id=channel_id)


class FakeTelegramDock(ChannelDock):
    PLATFORM = "telegram"
    CAPABILITIES = (
        ChannelCapability.RICH_TEXT
        | ChannelCapability.BUTTONS
        | ChannelCapability.VOICE
        | ChannelCapability.FILES
    )

    async def send_message(self, channel_id, message, **kwargs):
        return SendResult.ok(message_id="msg-2", platform=self.PLATFORM, channel_id=channel_id)


class FakeEmailDock(ChannelDock):
    PLATFORM = "email"
    CAPABILITIES = ChannelCapability.RICH_TEXT | ChannelCapability.FILES

    async def send_message(self, channel_id, message, **kwargs):
        return SendResult.ok(platform=self.PLATFORM, channel_id=channel_id)


class FakeSimpleDock(ChannelDock):
    PLATFORM = "simple"
    CAPABILITIES = ChannelCapability.NONE

    async def send_message(self, channel_id, message, **kwargs):
        return SendResult.ok(platform=self.PLATFORM, channel_id=channel_id)


def _make_message(**kwargs):
    """Create a NormalizedMessage for testing."""
    return NormalizedMessage(
        content=kwargs.get("content", "Test message"),
        message_type=kwargs.get("message_type", "notification"),
        format=kwargs.get("format", MessageFormat.MARKDOWN),
        title=kwargs.get("title"),
        thread_id=kwargs.get("thread_id"),
        reply_to=kwargs.get("reply_to"),
        metadata=kwargs.get("metadata", {}),
    )


def _make_httpx_response(status_code=200, json_data=None, text="OK"):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = text.encode() if text else b""
    resp.json.return_value = json_data or {}
    return resp


@pytest.fixture
def dock_registry():
    """Create a fresh dock registry."""
    return DockRegistry()


@pytest.fixture
def dock_registry_factory():
    """Create fresh dock registries on demand."""

    def _build() -> DockRegistry:
        return DockRegistry()

    return _build


@pytest.fixture
def normalized_message_factory():
    """Expose the shared normalized message helper as a fixture."""
    return _make_message


@pytest.fixture
def httpx_response_factory():
    """Expose the shared httpx response helper as a fixture."""
    return _make_httpx_response


@pytest.fixture
def registered_mock_dock(dock_registry):
    """Register and return a mock dock for router tests."""
    mock_dock = MagicMock(spec=ChannelDock)
    mock_dock.is_initialized = True
    mock_dock.supports.return_value = False
    mock_dock.send_result = AsyncMock(
        return_value=SendResult.ok(platform="test", channel_id="123", message_id="msg-1")
    )
    mock_dock.send_error = AsyncMock(return_value=SendResult.ok(platform="test", channel_id="123"))
    dock_registry._dock_classes["test"] = lambda config=None: mock_dock
    dock_registry._dock_instances["test"] = mock_dock
    return dock_registry, mock_dock
