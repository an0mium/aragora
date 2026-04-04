"""
Tests for FastAPI memory route endpoints.

Covers:
- GET /api/v2/memory/stats - Continuum memory dashboard stats
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aragora.rbac.models import AuthorizationContext
from aragora.server.fastapi import create_app
from aragora.server.fastapi.dependencies.auth import require_authenticated


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    return create_app()


@pytest.fixture
def mock_continuum():
    """Create a mock continuum memory backend."""
    continuum = MagicMock()
    continuum.get_stats = MagicMock(
        return_value={
            "total_memories": 144,
            "by_tier": {
                "fast": {
                    "count": 12,
                    "avg_importance": 0.7,
                    "avg_surprise": 0.2,
                    "avg_consolidation": 0.1,
                },
                "medium": {
                    "count": 40,
                    "avg_importance": 0.6,
                    "avg_surprise": 0.3,
                    "avg_consolidation": 0.2,
                },
            },
        }
    )
    continuum.get_memory_pressure = MagicMock(return_value=0.42)
    return continuum


@pytest.fixture
def client(app, mock_continuum):
    """Create a test client with mocked memory context."""
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
        "continuum_memory": mock_continuum,
    }
    return TestClient(app, raise_server_exceptions=False)


def _override_auth(client: TestClient) -> None:
    """Override auth dependency with memory read permission."""
    auth_ctx = AuthorizationContext(
        user_id="user-1",
        org_id="org-1",
        workspace_id="ws-1",
        roles={"member"},
        permissions={"memory:read"},
    )
    client.app.dependency_overrides[require_authenticated] = lambda: auth_ctx


class TestMemoryStats:
    """Tests for GET /api/v2/memory/stats."""

    def test_memory_stats_require_auth(self, client):
        response = client.get("/api/v2/memory/stats")
        assert response.status_code == 401

    def test_memory_stats_return_dashboard_shape(self, client, mock_continuum):
        _override_auth(client)
        try:
            response = client.get("/api/v2/memory/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["total_entries"] == 144
            assert data["memory_pressure"] == 0.42
            assert data["status"] == "normal"
            assert data["tiers"]["fast"]["count"] == 12
            assert data["tiers"]["fast"]["limit"] == 100
            assert data["tiers"]["fast"]["utilization"] == 0.12
            mock_continuum.get_stats.assert_called_once()
            mock_continuum.get_memory_pressure.assert_called_once()
        finally:
            client.app.dependency_overrides.clear()
