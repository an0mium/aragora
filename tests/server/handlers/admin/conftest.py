"""Admin handler test helpers layered on top of shared handler fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def admin_server_context(mock_server_context: dict[str, Any]) -> dict[str, Any]:
    """Provide a mutable admin-oriented context with nomic_dir available."""
    context = dict(mock_server_context)
    context.setdefault("nomic_dir", "/tmp/nomic")
    return context


@pytest.fixture
def admin_http_handler(mock_http_handler) -> Callable[..., MagicMock]:
    """Create admin handler request doubles with path and request_body fields."""

    def _build(
        *,
        method: str = "GET",
        path: str = "/",
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        client_address: tuple[str, int] = ("127.0.0.1", 12345),
    ) -> MagicMock:
        handler = mock_http_handler(
            method=method,
            body=body,
            headers=headers,
            client_address=client_address,
        )
        body_bytes = handler.rfile.read.return_value
        handler.path = path
        handler.request_body = body_bytes
        handler._body = body_bytes
        return handler

    return _build


@pytest.fixture
def decode_admin_result() -> Callable[[Any], dict[str, Any]]:
    """Decode JSON HandlerResult bodies for admin tests."""

    def _decode(result: Any) -> dict[str, Any]:
        if result and getattr(result, "body", None):
            return json.loads(result.body.decode("utf-8"))
        return {}

    return _decode
