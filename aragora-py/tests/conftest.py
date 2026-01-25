"""Test fixtures for Aragora Python SDK."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from aragora_client import AragoraClient


@pytest.fixture
def mock_response() -> MagicMock:
    """Create a factory for mock httpx responses."""

    def _create_response(
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        content: bytes = b"",
    ) -> MagicMock:
        response = MagicMock(spec=httpx.Response)
        response.status_code = status_code
        response.json.return_value = json_data or {}
        response.content = content or (
            b"{}" if json_data is None else str(json_data).encode()
        )
        return response

    return _create_response


@pytest.fixture
def mock_client(mock_response) -> AragoraClient:
    """Create a client with mocked HTTP client."""
    client = AragoraClient("http://localhost:8080", api_key="test-key")

    # Mock the internal HTTP client
    client._client = MagicMock(spec=httpx.AsyncClient)
    client._client.request = AsyncMock(return_value=mock_response(200, {}))
    client._client.aclose = AsyncMock()

    return client


@pytest.fixture
def auth_token_response() -> dict[str, Any]:
    """Standard auth token response."""
    return {
        "access_token": "test-access-token-123",
        "refresh_token": "test-refresh-token-456",
        "token_type": "bearer",
        "expires_in": 3600,
    }


@pytest.fixture
def user_response() -> dict[str, Any]:
    """Standard user profile response."""
    return {
        "id": "user-123",
        "email": "test@example.com",
        "name": "Test User",
        "email_verified": True,
        "mfa_enabled": False,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def tenant_response() -> dict[str, Any]:
    """Standard tenant response."""
    return {
        "id": "tenant-123",
        "name": "Test Tenant",
        "slug": "test-tenant",
        "owner_id": "user-123",
        "created_at": "2026-01-01T00:00:00Z",
        "status": "active",
    }


@pytest.fixture
def role_response() -> dict[str, Any]:
    """Standard role response."""
    return {
        "id": "role-123",
        "name": "Admin",
        "description": "Full administrative access",
        "permissions": ["admin:*", "debate:*", "user:*"],
        "is_system": True,
    }


@pytest.fixture
def tournament_response() -> dict[str, Any]:
    """Standard tournament response."""
    return {
        "id": "tournament-123",
        "name": "Test Tournament",
        "status": "active",
        "format": "round_robin",
        "participants": ["claude", "gpt4", "gemini"],
        "created_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def audit_event_response() -> dict[str, Any]:
    """Standard audit event response."""
    return {
        "id": "event-123",
        "event_type": "user.login",
        "actor_id": "user-123",
        "target_type": "session",
        "target_id": "session-456",
        "timestamp": "2026-01-01T00:00:00Z",
        "metadata": {"ip": "127.0.0.1"},
    }


@pytest.fixture
def onboarding_flow_response() -> dict[str, Any]:
    """Standard onboarding flow response."""
    return {
        "id": "flow-123",
        "name": "New User Onboarding",
        "steps": [
            {"id": "step-1", "title": "Welcome", "completed": True},
            {"id": "step-2", "title": "Setup Profile", "completed": False},
        ],
        "current_step": 1,
        "progress": 50,
    }


@pytest.fixture
def control_plane_health_response() -> dict[str, Any]:
    """Standard control plane health response."""
    return {
        "status": "healthy",
        "version": "2.1.14",
        "uptime": 86400,
        "agents": {
            "total": 10,
            "healthy": 10,
            "unhealthy": 0,
        },
    }
