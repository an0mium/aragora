"""Shared fixtures for FastAPI route tests.

These helpers keep route tests focused on endpoint behavior instead of
repeating the same app/context/auth bootstrap in every file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aragora.rbac.models import AuthorizationContext
from aragora.server.fastapi import create_app
from aragora.server.fastapi.dependencies.auth import require_authenticated


def _base_fastapi_context() -> dict[str, object]:
    """Build the default context shape expected by FastAPI route tests."""
    return {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
    }


@pytest.fixture
def build_fastapi_context() -> Callable[..., dict[str, object]]:
    """Create a standard app context with optional overrides."""

    def _build(**overrides: object) -> dict[str, object]:
        context = _base_fastapi_context()
        context.update(overrides)
        return context

    return _build


@pytest.fixture
def build_fastapi_client(
    build_fastapi_context: Callable[..., dict[str, object]],
) -> Iterator[Callable[..., TestClient]]:
    """Create disposable TestClient instances with the standard context wiring."""
    clients: list[TestClient] = []

    def _build(**context_overrides: object) -> TestClient:
        app = create_app()
        app.state.context = build_fastapi_context(**context_overrides)
        client = TestClient(app, raise_server_exceptions=False)
        clients.append(client)
        return client

    yield _build

    for client in clients:
        client.app.dependency_overrides.clear()
        client.close()


@pytest.fixture
def override_fastapi_auth() -> Callable[..., AuthorizationContext]:
    """Install a minimal authenticated RBAC context on a TestClient."""

    def _override(
        client: TestClient,
        *,
        user_id: str = "user-1",
        org_id: str = "org-1",
        workspace_id: str = "ws-1",
        roles: set[str] | None = None,
        permissions: set[str] | None = None,
    ) -> AuthorizationContext:
        auth_ctx = AuthorizationContext(
            user_id=user_id,
            org_id=org_id,
            workspace_id=workspace_id,
            roles=roles if roles is not None else {"member"},
            permissions=permissions if permissions is not None else set(),
        )
        client.app.dependency_overrides[require_authenticated] = lambda: auth_ctx
        return auth_ctx

    return _override
