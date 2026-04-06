"""Admin handler test helpers layered on top of shared handler fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest


def _coerce_request_body(body: dict[str, Any] | list[Any] | bytes | str | None) -> bytes:
    if body is None:
        return b"{}"
    if isinstance(body, bytes):
        return body
    if isinstance(body, str):
        return body.encode("utf-8")
    return json.dumps(body).encode("utf-8")


@pytest.fixture(autouse=True)
def reset_admin_rate_limiter():
    """Reset the admin rate limiter before and after each test."""
    from aragora.server.handlers.admin.handler import _admin_limiter

    _admin_limiter._buckets.clear()
    yield
    _admin_limiter._buckets.clear()


@pytest.fixture
def admin_server_context(mock_server_context: dict[str, Any]) -> dict[str, Any]:
    """Provide a mutable admin-oriented context with nomic_dir available."""
    context = dict(mock_server_context)
    context.setdefault("nomic_dir", "/tmp/nomic")
    return context


@pytest.fixture
def admin_request_factory() -> Callable[..., MagicMock]:
    """Create admin request doubles with auth and body metadata populated."""

    def _build(
        *,
        method: str = "GET",
        path: str = "/api/v1/admin",
        body: dict[str, Any] | list[Any] | bytes | str | None = None,
        headers: dict[str, str] | None = None,
        user_id: str = "admin-1",
        roles: list[str] | None = None,
        permissions: set[str] | None = None,
        client_address: tuple[str, int] = ("127.0.0.1", 12345),
    ) -> MagicMock:
        body_bytes = _coerce_request_body(body)
        role_list = list(roles or ["admin", "owner"])
        permission_set = set(permissions or {"*"})
        handler = MagicMock()
        handler.headers = {"Content-Type": "application/json", **(headers or {})}
        handler.headers.setdefault("Content-Length", str(len(body_bytes)))
        handler.rfile = MagicMock()
        handler.rfile.read = MagicMock(return_value=body_bytes)
        handler.path = path
        handler.command = method
        handler.request_body = body_bytes
        handler._body = body_bytes
        handler.client_address = client_address
        handler._context = {"user": {"id": user_id, "roles": role_list}}
        handler._auth_context = MagicMock()
        handler._auth_context.user_id = user_id
        handler._auth_context.user_email = "admin@example.com"
        handler._auth_context.org_id = "org-1"
        handler._auth_context.workspace_id = "ws-1"
        handler._auth_context.roles = role_list
        handler._auth_context.permissions = permission_set
        return handler

    return _build


@pytest.fixture
def admin_request_handler(admin_request_factory) -> MagicMock:
    """Default admin request double."""
    return admin_request_factory()


@pytest.fixture
def mock_handler(admin_request_handler) -> MagicMock:
    """Compatibility alias for tests expecting a generic mock handler."""
    return admin_request_handler


@pytest.fixture
def admin_http_handler(admin_request_factory) -> Callable[..., MagicMock]:
    """Compatibility alias for tests naming the admin request builder differently."""
    return admin_request_factory


@pytest.fixture
def decode_admin_result() -> Callable[[Any], dict[str, Any]]:
    """Decode JSON HandlerResult bodies for admin tests."""

    def _decode(result: Any) -> dict[str, Any]:
        if result and getattr(result, "body", None):
            return json.loads(result.body.decode("utf-8"))
        return {}

    return _decode
