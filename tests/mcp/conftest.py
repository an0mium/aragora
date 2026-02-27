"""Shared fixtures for MCP tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _mock_control_plane_coordinator(monkeypatch):
    """Prevent ControlPlaneCoordinator from connecting to Redis."""
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
    """Prevent KnowledgeMound from requiring initialization."""
    try:
        from unittest.mock import AsyncMock, MagicMock

        import aragora.mcp.tools_module.knowledge as km_mod

        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=[])
        mock_mound.query_graph = AsyncMock(return_value=MagicMock(edges=[], nodes=[]))
        mock_mound.ingest = AsyncMock(return_value=None)

        async def _mock_get_mound():
            return mock_mound

        if hasattr(km_mod, "_get_mound"):
            monkeypatch.setattr(km_mod, "_get_mound", _mock_get_mound)
    except (ImportError, AttributeError):
        pass
