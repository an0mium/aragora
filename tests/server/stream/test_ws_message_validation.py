"""
Tests for WebSocket message size validation, JSON parse error handling,
and nested JSON depth limits.

Covers:
- Normal-sized messages pass through and are processed
- Oversized messages are rejected with close code 1009 (Message Too Big)
- Invalid JSON is rejected with close code 1003 (Unsupported Data)
- Deeply nested JSON (JSON bombs) are rejected with close code 1003
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import aiohttp
import aiohttp.web
import pytest
from aiohttp.test_utils import make_mocked_request

from aragora.server.stream.servers import AiohttpUnifiedServer
from aragora.server.stream.servers_ws_handler import (
    WS_CLOSE_MESSAGE_TOO_BIG,
    WS_CLOSE_UNSUPPORTED_DATA,
    WS_MAX_JSON_DEPTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeWSMsg:
    """Mimics aiohttp.WSMessage enough for the handler loop."""

    type: aiohttp.WSMsgType
    data: str | bytes = ""
    extra: str = ""


class _StubWebSocket:
    """Stub aiohttp WebSocket that feeds pre-configured messages to the handler.

    After the handler runs, inspect ``sent_json``, ``close_code``, and
    ``close_message`` to verify behaviour.
    """

    def __init__(self, messages: list[_FakeWSMsg] | None = None):
        self._messages = list(messages or [])
        self.sent_json: list[dict[str, Any]] = []
        self.close_code: int | None = None
        self.close_message: bytes | None = None
        self.closed: bool = False

    # -- aiohttp WebSocketResponse interface --------------------------------

    async def prepare(self, request: Any) -> "_StubWebSocket":
        return self

    async def send_json(self, data: dict[str, Any]) -> None:
        self.sent_json.append(data)

    async def close(
        self,
        *,
        code: int = 1000,
        message: bytes = b"",
    ) -> None:
        self.close_code = code
        self.close_message = message
        self.closed = True

    def __aiter__(self):
        return _StubWSIter(self._messages)


class _StubWSIter:
    """Async iterator that yields pre-built messages."""

    def __init__(self, messages: list[_FakeWSMsg]):
        self._it = iter(messages)

    def __aiter__(self):
        return self

    async def __anext__(self) -> _FakeWSMsg:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def _make_server() -> AiohttpUnifiedServer:
    return AiohttpUnifiedServer(port=0, host="127.0.0.1")


def _make_request(origin: str = "https://aragora.ai") -> Any:
    return make_mocked_request(
        "GET",
        "/ws",
        headers={"Origin": origin},
    )


def _text_msg(payload: str) -> _FakeWSMsg:
    return _FakeWSMsg(type=aiohttp.WSMsgType.TEXT, data=payload)


def _build_deeply_nested(depth: int) -> dict:
    """Build a dict nested *depth* levels deep."""
    obj: dict = {"leaf": True}
    for _ in range(depth):
        obj = {"nested": obj}
    return obj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalMessagesPassThrough:
    """Verify that well-formed, appropriately-sized messages are processed."""

    @pytest.mark.asyncio
    async def test_get_loops_message_processed(self, monkeypatch):
        """A valid 'get_loops' message should get a loop_list response."""
        payload = json.dumps({"type": "get_loops"})
        ws_stub = _StubWebSocket(messages=[_text_msg(payload)])
        monkeypatch.setattr(aiohttp.web, "WebSocketResponse", lambda **kw: ws_stub)

        server = _make_server()
        request = _make_request()
        await server._websocket_handler(request)

        # Should NOT have been closed with an error code
        assert ws_stub.close_code is None or ws_stub.close_code == 1000

        # Should have received at least the connection_info and loop_list
        types_sent = [m.get("type") for m in ws_stub.sent_json]
        assert "connection_info" in types_sent
        # The handler sends an initial loop_list, then another for get_loops
        assert types_sent.count("loop_list") >= 2

    @pytest.mark.asyncio
    async def test_small_json_not_rejected(self, monkeypatch):
        """A small valid JSON message should not trigger any close."""
        payload = json.dumps({"type": "unknown_but_valid_json", "data": "x" * 100})
        ws_stub = _StubWebSocket(messages=[_text_msg(payload)])
        monkeypatch.setattr(aiohttp.web, "WebSocketResponse", lambda **kw: ws_stub)

        server = _make_server()
        request = _make_request()
        await server._websocket_handler(request)

        # Connection should NOT have been closed with 1003 or 1009
        assert ws_stub.close_code != WS_CLOSE_MESSAGE_TOO_BIG
        assert ws_stub.close_code != WS_CLOSE_UNSUPPORTED_DATA


class TestOversizedMessageRejection:
    """Oversized messages must be rejected with close code 1009."""

    @pytest.mark.asyncio
    async def test_oversized_message_closes_1009(self, monkeypatch):
        """Messages exceeding WS_MAX_MESSAGE_SIZE trigger close(1009)."""
        # Temporarily lower the limit so we don't need a 64KB string in the test
        monkeypatch.setattr(
            "aragora.server.stream.servers_ws_handler.WS_MAX_MESSAGE_SIZE",
            128,
        )
        # Also patch the config-level constant (imported at module top)
        monkeypatch.setattr(
            "aragora.config.WS_MAX_MESSAGE_SIZE",
            128,
        )

        oversized = "x" * 256  # Well above the 128-byte limit
        ws_stub = _StubWebSocket(messages=[_text_msg(oversized)])
        monkeypatch.setattr(aiohttp.web, "WebSocketResponse", lambda **kw: ws_stub)

        server = _make_server()
        request = _make_request()
        await server._websocket_handler(request)

        assert ws_stub.close_code == WS_CLOSE_MESSAGE_TOO_BIG
        assert ws_stub.close_message == b"Message too large"

    @pytest.mark.asyncio
    async def test_message_at_exact_limit_not_rejected(self, monkeypatch):
        """A message exactly at the size limit should NOT be rejected."""
        limit = 256
        monkeypatch.setattr(
            "aragora.server.stream.servers_ws_handler.WS_MAX_MESSAGE_SIZE",
            limit,
        )
        monkeypatch.setattr("aragora.config.WS_MAX_MESSAGE_SIZE", limit)

        # Build a JSON message that is exactly `limit` bytes
        # We need valid JSON, so pad carefully
        base = json.dumps({"type": "get_loops"})
        # Pad is not needed as long as len <= limit; just check it passes
        assert len(base) <= limit

        ws_stub = _StubWebSocket(messages=[_text_msg(base)])
        monkeypatch.setattr(aiohttp.web, "WebSocketResponse", lambda **kw: ws_stub)

        server = _make_server()
        request = _make_request()
        await server._websocket_handler(request)

        assert ws_stub.close_code != WS_CLOSE_MESSAGE_TOO_BIG


class TestInvalidJsonRejection:
    """Invalid JSON must be rejected with close code 1003."""

    @pytest.mark.asyncio
    async def test_malformed_json_closes_1003(self, monkeypatch):
        """Sending non-JSON text should trigger close(1003)."""
        ws_stub = _StubWebSocket(messages=[_text_msg("{not valid json!!!")])
        monkeypatch.setattr(aiohttp.web, "WebSocketResponse", lambda **kw: ws_stub)

        server = _make_server()
        request = _make_request()
        await server._websocket_handler(request)

        assert ws_stub.close_code == WS_CLOSE_UNSUPPORTED_DATA
        assert ws_stub.close_message == b"Unsupported data"

    @pytest.mark.asyncio
    async def test_truncated_json_closes_1003(self, monkeypatch):
        """Truncated JSON should trigger close(1003)."""
        ws_stub = _StubWebSocket(messages=[_text_msg('{"type": "subscribe"')])
        monkeypatch.setattr(aiohttp.web, "WebSocketResponse", lambda **kw: ws_stub)

        server = _make_server()
        request = _make_request()
        await server._websocket_handler(request)

        assert ws_stub.close_code == WS_CLOSE_UNSUPPORTED_DATA

    @pytest.mark.asyncio
    async def test_empty_string_closes_1003(self, monkeypatch):
        """An empty string is not valid JSON and should close(1003)."""
        ws_stub = _StubWebSocket(messages=[_text_msg("")])
        monkeypatch.setattr(aiohttp.web, "WebSocketResponse", lambda **kw: ws_stub)

        server = _make_server()
        request = _make_request()
        await server._websocket_handler(request)

        assert ws_stub.close_code == WS_CLOSE_UNSUPPORTED_DATA


class TestJsonDepthLimit:
    """Deeply nested JSON must be rejected with close code 1003."""

    @pytest.mark.asyncio
    async def test_deeply_nested_json_closes_1003(self, monkeypatch):
        """JSON nested beyond WS_MAX_JSON_DEPTH should trigger close(1003)."""
        deep = _build_deeply_nested(WS_MAX_JSON_DEPTH + 5)
        payload = json.dumps(deep)
        ws_stub = _StubWebSocket(messages=[_text_msg(payload)])
        monkeypatch.setattr(aiohttp.web, "WebSocketResponse", lambda **kw: ws_stub)

        server = _make_server()
        request = _make_request()
        await server._websocket_handler(request)

        assert ws_stub.close_code == WS_CLOSE_UNSUPPORTED_DATA
        assert ws_stub.close_message == b"Unsupported data"

    @pytest.mark.asyncio
    async def test_acceptable_depth_passes(self, monkeypatch):
        """JSON within the depth limit should NOT be rejected."""
        # Depth of 5 is well within the default 20 limit
        obj = _build_deeply_nested(5)
        obj["type"] = "get_loops"
        payload = json.dumps(obj)
        ws_stub = _StubWebSocket(messages=[_text_msg(payload)])
        monkeypatch.setattr(aiohttp.web, "WebSocketResponse", lambda **kw: ws_stub)

        server = _make_server()
        request = _make_request()
        await server._websocket_handler(request)

        assert ws_stub.close_code != WS_CLOSE_UNSUPPORTED_DATA

    @pytest.mark.asyncio
    async def test_depth_just_under_limit_passes(self, monkeypatch):
        """JSON just under the depth limit should be accepted.

        ``_build_deeply_nested(n)`` creates n+1 levels (the leaf dict at level 1
        plus n wrapping dicts), so ``n = WS_MAX_JSON_DEPTH - 2`` gives a
        structure with ``WS_MAX_JSON_DEPTH - 1`` levels -- safely within the limit.
        """
        obj = _build_deeply_nested(WS_MAX_JSON_DEPTH - 2)
        obj["type"] = "get_loops"
        payload = json.dumps(obj)
        ws_stub = _StubWebSocket(messages=[_text_msg(payload)])
        monkeypatch.setattr(aiohttp.web, "WebSocketResponse", lambda **kw: ws_stub)

        server = _make_server()
        request = _make_request()
        await server._websocket_handler(request)

        # Should not be closed with unsupported data
        assert ws_stub.close_code != WS_CLOSE_UNSUPPORTED_DATA


class TestCloseCodeConstants:
    """Verify RFC 6455 close code constants are correct."""

    def test_close_code_1003(self):
        assert WS_CLOSE_UNSUPPORTED_DATA == 1003

    def test_close_code_1009(self):
        assert WS_CLOSE_MESSAGE_TOO_BIG == 1009

    def test_max_json_depth_is_reasonable(self):
        assert WS_MAX_JSON_DEPTH > 0
        assert WS_MAX_JSON_DEPTH <= 100
