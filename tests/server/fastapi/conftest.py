"""Shared fixtures for FastAPI route tests.

These helpers keep route tests focused on endpoint behavior instead of
repeating the same app/context/auth bootstrap in every file.
"""

from __future__ import annotations

from types import SimpleNamespace
from collections.abc import Callable, Iterator
from unittest.mock import AsyncMock, MagicMock

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
def fastapi_context(build_fastapi_context: Callable[..., dict[str, object]]) -> dict[str, object]:
    """Default FastAPI context for tests that only need the base shape."""
    return build_fastapi_context()


@pytest.fixture
def fastapi_app(fastapi_context: dict[str, object]):
    """Create a FastAPI app with the default mocked context."""
    app = create_app()
    app.state.context = fastapi_context
    return app


@pytest.fixture
def app(fastapi_app):
    """Compatibility alias for tests expecting a shared app fixture."""
    return fastapi_app


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
def fastapi_client(fastapi_app):
    """Create a disposable client for tests using the default context."""
    client = TestClient(fastapi_app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        fastapi_app.dependency_overrides.clear()
        client.close()


@pytest.fixture
def client(fastapi_client):
    """Compatibility alias for tests expecting a shared client fixture."""
    return fastapi_client


@pytest.fixture
def mock_app_client(fastapi_client):
    """Explicit alias for tests that name the shared app client differently."""
    return fastapi_client


@pytest.fixture
def fastapi_request_factory(
    build_fastapi_context: Callable[..., dict[str, object]],
) -> Callable[..., MagicMock]:
    """Create lightweight request doubles for direct route-function tests."""

    def _build(
        *,
        context: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        method: str = "GET",
        path: str = "/",
        body: bytes = b"",
    ) -> MagicMock:
        request = MagicMock()
        request.app.state.context = context if context is not None else build_fastapi_context()
        request.headers = headers or {}
        request.method = method
        request.url = SimpleNamespace(path=path)
        request.state = SimpleNamespace()
        request.body = AsyncMock(return_value=body)
        return request

    return _build


@pytest.fixture
def route_request_factory(fastapi_request_factory):
    """Short alias for direct route request helpers."""
    return fastapi_request_factory


@pytest.fixture
def fastapi_route_auth_factory() -> Callable[..., SimpleNamespace]:
    """Build lightweight auth objects for direct route-function tests."""

    def _build(
        *,
        user_id: str = "user-1",
        email: str = "user@example.com",
        org_id: str = "org-1",
        workspace_id: str = "ws-1",
        roles: set[str] | None = None,
        permissions: set[str] | None = None,
        **extra: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            user_id=user_id,
            email=email,
            org_id=org_id,
            workspace_id=workspace_id,
            roles=set(roles or {"admin"}),
            permissions=set(permissions or {"*"}),
            **extra,
        )

    return _build


@pytest.fixture
def route_auth_factory(fastapi_route_auth_factory):
    """Short alias for direct route auth helpers."""
    return fastapi_route_auth_factory


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
            roles=roles if roles is not None else {"admin"},
            permissions=permissions if permissions is not None else {"*"},
        )
        client.app.dependency_overrides[require_authenticated] = lambda: auth_ctx
        return auth_ctx

    return _override


@pytest.fixture
def override_auth(override_fastapi_auth):
    """Short alias for common auth override helper usage."""
    return override_fastapi_auth
