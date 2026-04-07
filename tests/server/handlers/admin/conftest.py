"""Shared fixtures for admin handler tests."""

from __future__ import annotations

import json
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
def admin_context_builder():
    """Build common server context dictionaries for admin handler tests."""

    def _build(**overrides: Any) -> dict[str, Any]:
        context = {
            "user_store": MagicMock(),
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "decision_service": MagicMock(),
        }
        context.update(overrides)
        return context

    return _build


@pytest.fixture
def admin_server_context(admin_context_builder):
    """Default admin server context with mocked dependencies."""
    return admin_context_builder()


@pytest.fixture
def mock_server_context(admin_server_context):
    """Compatibility alias for tests that expect ``mock_server_context``."""
    return admin_server_context


@pytest.fixture
def admin_auth_context() -> MagicMock:
    """Create a reusable authenticated admin auth context."""
    auth_ctx = MagicMock()
    auth_ctx.user_id = "admin-1"
    auth_ctx.user_email = "admin@example.com"
    auth_ctx.org_id = "org-1"
    auth_ctx.workspace_id = "ws-1"
    auth_ctx.roles = ["admin", "owner"]
    auth_ctx.permissions = {"*"}
    return auth_ctx


@pytest.fixture
def admin_request_factory(admin_auth_context):
    """Create request-handler doubles for admin handler unit tests."""

    def _build(
        *,
        method: str = "GET",
        path: str = "/api/v1/admin",
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | list[Any] | bytes | str | None = None,
        user_id: str | None = None,
        roles: list[str] | None = None,
    ) -> MagicMock:
        body_bytes = _coerce_request_body(body)
        role_list = list(roles or admin_auth_context.roles)
        current_user_id = user_id or admin_auth_context.user_id

        handler = MagicMock()
        handler.headers = {"Content-Type": "application/json", **(headers or {})}
        handler.headers.setdefault("Content-Length", str(len(body_bytes)))
        handler.request_body = body_bytes
        handler._body = body_bytes
        handler.rfile = MagicMock()
        handler.rfile.read = MagicMock(return_value=body_bytes)
        handler.path = path
        handler.command = method
        handler.client_address = ("127.0.0.1", 12345)
        handler._context = {"user": {"id": current_user_id, "roles": role_list}}

        request_auth_context = MagicMock()
        request_auth_context.user_id = current_user_id
        request_auth_context.user_email = admin_auth_context.user_email
        request_auth_context.org_id = admin_auth_context.org_id
        request_auth_context.workspace_id = admin_auth_context.workspace_id
        request_auth_context.roles = role_list
        request_auth_context.permissions = admin_auth_context.permissions
        handler._auth_context = request_auth_context
        return handler

    return _build


@pytest.fixture
def admin_request_handler(admin_request_factory):
    """Default admin request handler double."""
    return admin_request_factory()


@pytest.fixture
def mock_handler(admin_request_handler):
    """Compatibility alias for tests that expect ``mock_handler``."""
    return admin_request_handler


@pytest.fixture
def admin_handler_factory(admin_server_context):
    """Instantiate admin handlers with the shared server context."""

    def _build(handler_cls, *args: Any, **kwargs: Any):
        return handler_cls(admin_server_context, *args, **kwargs)

    return _build
