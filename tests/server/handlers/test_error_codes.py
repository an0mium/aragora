"""Regression tests for top-level error codes added by ``@handle_errors``."""

from __future__ import annotations

import asyncio
import json

from aragora.server.handlers.utils.decorators import handle_errors


def _decode_error(result) -> dict[str, str]:
    return json.loads(result.body.decode("utf-8"))


def test_handle_errors_adds_not_found_error_code_without_nesting() -> None:
    @handle_errors("lookup")
    def handler():
        raise FileNotFoundError("gone")

    result = handler()
    body = _decode_error(result)

    assert result.status_code == 404
    assert body["error"] == "Resource not found"
    assert body["error_code"] == "NOT_FOUND"
    assert "X-Trace-Id" in result.headers


def test_handle_errors_adds_internal_error_code_without_changing_error_shape() -> None:
    @handle_errors("explode")
    def handler():
        raise RuntimeError("boom")

    result = handler()
    body = _decode_error(result)

    assert result.status_code == 500
    assert isinstance(body["error"], str)
    assert body["error_code"] == "INTERNAL_ERROR"


def test_async_handle_errors_adds_timeout_error_code() -> None:
    @handle_errors("slow op")
    async def handler():
        raise TimeoutError("too slow")

    result = asyncio.run(handler())
    body = _decode_error(result)

    assert result.status_code == 504
    assert body["error_code"] == "TIMEOUT"
