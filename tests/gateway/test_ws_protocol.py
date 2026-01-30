"""
Tests for gateway WebSocket protocol adapter (Moltbot parity skeleton).
"""

import pytest

from aragora.gateway import (
    GatewayConfig,
    GatewayProtocolAdapter,
    GatewayWebSocketProtocol,
    LocalGateway,
)


@pytest.mark.asyncio
async def test_ws_protocol_session_flow():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)
    ws_protocol = GatewayWebSocketProtocol(adapter)

    opened = await ws_protocol.handle_message(
        {"type": "session.open", "user_id": "user-1", "device_id": "dev-1"}
    )
    assert opened["type"] == "session.opened"
    session_id = opened["session"]["session_id"]

    paused = await ws_protocol.handle_message(
        {"type": "presence.update", "session_id": session_id, "status": "paused"}
    )
    assert paused["type"] == "presence.updated"
    assert paused["session"]["status"] == "paused"

    closed = await ws_protocol.handle_message(
        {"type": "session.close", "session_id": session_id, "reason": "done"}
    )
    assert closed["type"] == "session.closed"
    assert closed["session"]["status"] == "ended"
    assert closed["session"]["end_reason"] == "done"


@pytest.mark.asyncio
async def test_ws_protocol_config_get():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False, port=9999))
    adapter = GatewayProtocolAdapter(gateway)
    ws_protocol = GatewayWebSocketProtocol(adapter)

    opened = await ws_protocol.handle_message(
        {"type": "session.open", "user_id": "user-1", "device_id": "dev-1"}
    )
    session_id = opened["session"]["session_id"]

    config_response = await ws_protocol.handle_message(
        {"type": "config.get", "session_id": session_id}
    )
    assert config_response["type"] == "config"
    assert config_response["config"]["port"] == 9999


@pytest.mark.asyncio
async def test_ws_protocol_errors():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)
    ws_protocol = GatewayWebSocketProtocol(adapter)

    missing = await ws_protocol.handle_message({"type": "session.open"})
    assert missing["type"] == "error"
    assert missing["error"]["code"] == "missing_fields"

    unknown = await ws_protocol.handle_message({"type": "does.not.exist"})
    assert unknown["type"] == "error"
    assert unknown["error"]["code"] == "unknown_type"
