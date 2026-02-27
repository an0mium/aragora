"""MCP test fixtures — isolate external services."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_control_plane_coordinator(monkeypatch):
    """Prevent MCP control-plane tools from connecting to Redis."""
    try:
        import aragora.mcp.tools_module.control_plane as cp_mod

        async def _mock_get_coordinator():
            return None

        monkeypatch.setattr(cp_mod, "_get_coordinator", _mock_get_coordinator)
        monkeypatch.setattr(cp_mod, "_coordinator", None)
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _mock_knowledge_mound(monkeypatch):
    """Prevent MCP knowledge tools from requiring a live KnowledgeMound."""
    try:
        import aragora.mcp.tools_module.knowledge as km_mod

        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=[])
        mock_mound.ingest = AsyncMock(return_value=None)
        mock_mound.search = AsyncMock(return_value=[])

        async def _mock_get_mound():
            return mock_mound

        monkeypatch.setattr(km_mod, "_get_mound", _mock_get_mound)
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _isolate_notification_tokens(monkeypatch):
    """Remove real Slack tokens to prevent external API calls."""
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
