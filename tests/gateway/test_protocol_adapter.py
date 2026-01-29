"""
Acceptance tests for gateway protocol adapter (Moltbot parity scaffold).
"""

import pytest

from aragora.gateway import GatewayConfig, GatewayProtocolAdapter, LocalGateway


@pytest.mark.asyncio
async def test_gateway_session_lifecycle():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    session = await adapter.open_session(user_id="user-1", device_id="dev-1")
    assert session.session_id
    assert session.status == "active"

    updated = await adapter.update_presence(session.session_id, status="paused")
    assert updated is not None
    assert updated.status == "paused"

    closed = await adapter.close_session(session.session_id, reason="done")
    assert closed is not None
    assert closed.status == "ended"
    assert closed.end_reason == "done"


@pytest.mark.asyncio
async def test_gateway_session_filters():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    s1 = await adapter.open_session(user_id="user-1", device_id="dev-1")
    s2 = await adapter.open_session(user_id="user-2", device_id="dev-2")
    await adapter.update_presence(s2.session_id, status="paused")

    user1_sessions = await adapter.list_sessions(user_id="user-1")
    assert len(user1_sessions) == 1
    assert user1_sessions[0].session_id == s1.session_id

    paused_sessions = await adapter.list_sessions(status="paused")
    assert len(paused_sessions) == 1
    assert paused_sessions[0].session_id == s2.session_id
