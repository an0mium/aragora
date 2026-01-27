import importlib

import aiohttp
import pytest
from aiohttp.test_utils import make_mocked_request

from aragora.server.stream.servers import AiohttpUnifiedServer


class _StubWebSocket:
    def __init__(self, *args, **kwargs):
        self.closed = False
        self.sent = []

    async def prepare(self, request):
        return self

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self, *args, **kwargs):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_websocket_accepts_token_query_param(monkeypatch):
    """Token query param should be mapped to Authorization header for auth checks."""
    auth_module = importlib.import_module("aragora.server.auth")

    captured = {}

    def fake_check_auth(headers, *_args, **_kwargs):
        captured["auth_header"] = headers.get("Authorization")
        return True, 1

    # Force auth enabled and capture header usage
    monkeypatch.setattr(auth_module, "auth_config", type("C", (), {"enabled": True})())
    monkeypatch.setattr(auth_module, "check_auth", fake_check_auth)
    monkeypatch.setattr(aiohttp.web, "WebSocketResponse", _StubWebSocket)

    server = AiohttpUnifiedServer(port=0, host="127.0.0.1")
    request = make_mocked_request(
        "GET",
        "/ws?token=test-token",
        headers={"Origin": "https://aragora.ai"},
    )

    await server._websocket_handler(request)

    assert captured["auth_header"] == "Bearer test-token"
