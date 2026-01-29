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


# --- get_config_for_session ---


@pytest.mark.asyncio
async def test_get_config_for_session_active():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    session = await adapter.open_session(user_id="u1", device_id="d1")
    cfg = await adapter.get_config_for_session(session.session_id)
    assert cfg is not None
    assert isinstance(cfg, GatewayConfig)


@pytest.mark.asyncio
async def test_get_config_for_session_ended_returns_none():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    session = await adapter.open_session(user_id="u1", device_id="d1")
    await adapter.close_session(session.session_id)
    cfg = await adapter.get_config_for_session(session.session_id)
    assert cfg is None


@pytest.mark.asyncio
async def test_get_config_for_session_unknown_returns_none():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    cfg = await adapter.get_config_for_session("nonexistent")
    assert cfg is None


# --- resume_session ---


@pytest.mark.asyncio
async def test_resume_session_success():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    session = await adapter.open_session(user_id="u1", device_id="d1")
    await adapter.update_presence(session.session_id, status="paused")

    resumed = await adapter.resume_session(session.session_id, device_id="d1")
    assert resumed is not None
    assert resumed.status == "active"


@pytest.mark.asyncio
async def test_resume_session_wrong_device():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    session = await adapter.open_session(user_id="u1", device_id="d1")
    await adapter.update_presence(session.session_id, status="paused")

    resumed = await adapter.resume_session(session.session_id, device_id="d2")
    assert resumed is None


@pytest.mark.asyncio
async def test_resume_session_not_paused():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    session = await adapter.open_session(user_id="u1", device_id="d1")
    # Session is active, not paused
    resumed = await adapter.resume_session(session.session_id, device_id="d1")
    assert resumed is None


@pytest.mark.asyncio
async def test_resume_session_unknown():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    resumed = await adapter.resume_session("nonexistent", device_id="d1")
    assert resumed is None


# --- bind_device_to_session ---


@pytest.mark.asyncio
async def test_bind_device_success():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    session = await adapter.open_session(user_id="u1", device_id="d1")
    ok = await adapter.bind_device_to_session(session.session_id, device_id="d2")
    assert ok is True

    fetched = await adapter.get_session(session.session_id)
    assert fetched is not None
    assert fetched.device_id == "d2"


@pytest.mark.asyncio
async def test_bind_device_ended_session():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    session = await adapter.open_session(user_id="u1", device_id="d1")
    await adapter.close_session(session.session_id)
    ok = await adapter.bind_device_to_session(session.session_id, device_id="d2")
    assert ok is False


@pytest.mark.asyncio
async def test_bind_device_unknown_session():
    gateway = LocalGateway(config=GatewayConfig(enable_auth=False))
    adapter = GatewayProtocolAdapter(gateway)

    ok = await adapter.bind_device_to_session("nonexistent", device_id="d2")
    assert ok is False
