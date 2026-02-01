"""
Tests for the /livez liveness probe endpoint.

Verifies that the /livez endpoint:
- Returns 200 with {"status": "alive"}
- Responds even when subsystems are not initialized
- Is registered in the health router
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app
from aragora.server.fastapi.routes.health import router


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a test client with initialized subsystems."""
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
    }
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def bare_client(app):
    """Create a test client with NO subsystems initialized."""
    app.state.context = {}
    return TestClient(app, raise_server_exceptions=False)


class TestLivezEndpoint:
    """Tests for the /livez liveness probe."""

    def test_livez_returns_200_with_alive_status(self, client):
        """/livez returns 200 with {"status": "alive"}."""
        response = client.get("/livez")
        assert response.status_code == 200
        data = response.json()
        assert data == {"status": "alive"}

    def test_livez_responds_without_subsystems(self, bare_client):
        """/livez responds even when no subsystems are initialized."""
        response = bare_client.get("/livez")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_livez_endpoint_exists_in_router(self):
        """The /livez endpoint is registered in the health router."""
        route_paths = [route.path for route in router.routes]
        assert "/livez" in route_paths
