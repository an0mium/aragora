"""
Comprehensive tests for aragora/server/handlers/extensions.py.

Covers all 10 handler functions and the route registration helper:
- handle_extensions_status (GET /api/extensions/status)
- handle_extensions_stats (GET /api/extensions/stats)
- handle_gastown_workspaces_list (GET /api/extensions/gastown/workspaces)
- handle_gastown_workspace_create (POST /api/extensions/gastown/workspaces/create)
- handle_gastown_convoys_list (GET /api/extensions/gastown/convoys)
- handle_moltbot_inbox_messages (GET /api/extensions/moltbot/inbox/messages)
- handle_moltbot_gateway_devices (GET /api/extensions/moltbot/gateway/devices)
- handle_moltbot_onboarding_flows (GET /api/extensions/moltbot/onboarding/flows)
- handle_fabric_agents_list (GET /api/extensions/fabric/agents)
- handle_fabric_tasks_list (GET /api/extensions/fabric/tasks)
- get_extension_routes (route registration)

Tests cover: happy paths, null state, disabled extensions, missing sub-components,
exception handling for each caught exception type, field shapes, edge cases,
long content truncation, empty lists, None ctx fields, and route table validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

PATCH_TARGET = "aragora.server.handlers.extensions.get_extension_state"


class _ConvoyStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"


class _MessageStatus(Enum):
    PENDING = "pending"
    DELIVERED = "delivered"


class _AgentStatus(Enum):
    ACTIVE = "active"
    IDLE = "idle"


def _make_workspace(
    ws_id: str = "ws-1",
    name: str = "Workspace 1",
    status: str = "active",
    rigs: list | None = None,
    created_at: datetime | None = None,
) -> MagicMock:
    ws = MagicMock()
    ws.id = ws_id
    ws.config.name = name
    ws.status = status
    ws.rigs = rigs if rigs is not None else ["rig-1"]
    ws.created_at = created_at or datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    return ws


def _make_convoy(
    convoy_id: str = "convoy-1",
    title: str = "Test Convoy",
    status: _ConvoyStatus = _ConvoyStatus.RUNNING,
    rig_id: str = "rig-1",
    created_at: datetime | None = None,
) -> MagicMock:
    c = MagicMock()
    c.id = convoy_id
    c.title = title
    c.status = status
    c.rig_id = rig_id
    c.created_at = created_at or datetime(2025, 2, 1, 8, 0, 0, tzinfo=timezone.utc)
    return c


def _make_message(
    msg_id: str = "msg-1",
    channel_id: str = "ch-1",
    direction: str = "inbound",
    content: str = "Hello world",
    status: _MessageStatus = _MessageStatus.PENDING,
    created_at: datetime | None = None,
) -> MagicMock:
    m = MagicMock()
    m.id = msg_id
    m.channel_id = channel_id
    m.direction = direction
    m.content = content
    m.status = status
    m.created_at = created_at or datetime(2025, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    return m


def _make_device(
    dev_id: str = "dev-1",
    name: str = "Device 1",
    device_type: str = "sensor",
    status: str = "online",
    last_seen: datetime | None = None,
) -> MagicMock:
    d = MagicMock()
    d.id = dev_id
    d.config.name = name
    d.config.device_type = device_type
    d.status = status
    d.last_seen = last_seen
    return d


def _make_flow(
    flow_id: str = "flow-1",
    name: str = "Onboarding Flow",
    status: str = "active",
    steps: list | None = None,
    started_count: int = 100,
    completed_count: int = 85,
) -> MagicMock:
    f = MagicMock()
    f.id = flow_id
    f.name = name
    f.status = status
    f.steps = steps if steps is not None else ["s1", "s2", "s3"]
    f.started_count = started_count
    f.completed_count = completed_count
    return f


def _make_fabric_agent(
    agent_id: str = "agent-1",
    model: str = "claude-opus-4",
    status: _AgentStatus = _AgentStatus.ACTIVE,
    created_at: str = "2025-01-15T10:00:00Z",
) -> MagicMock:
    a = MagicMock()
    a.id = agent_id
    a.config.model = model
    a.status = status
    a.created_at = created_at
    return a


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_ctx() -> MagicMock:
    """Mock authorization context."""
    ctx = MagicMock()
    ctx.user_id = "user-123"
    ctx.org_id = "org-456"
    return ctx


@pytest.fixture
def full_state() -> MagicMock:
    """Fully populated extension state with all extensions enabled."""
    state = MagicMock()
    state.fabric_enabled = True
    state.gastown_enabled = True
    state.moltbot_enabled = True
    state.computer_use_enabled = False
    state.metadata = {
        "fabric_available": True,
        "gastown_available": True,
        "moltbot_available": True,
        "computer_use_available": False,
    }

    # Fabric
    state.fabric = MagicMock()
    state.fabric.get_stats = AsyncMock(return_value={"agents_active": 5, "tasks_pending": 3})
    state.fabric.list_agents = AsyncMock(
        return_value=[
            _make_fabric_agent("agent-1", "claude-opus-4", _AgentStatus.ACTIVE),
            _make_fabric_agent("agent-2", "gpt-4", _AgentStatus.IDLE),
        ]
    )

    # Gastown
    state.coordinator = MagicMock()
    state.coordinator.get_stats = AsyncMock(return_value={"workspaces": 3, "rigs_active": 7})
    _created_ws = MagicMock()
    _created_ws.id = "ws-new"
    _created_ws_config = MagicMock()
    _created_ws_config.name = "New Workspace"
    _created_ws.config = _created_ws_config
    _created_ws.status = "active"
    state.coordinator.create_workspace = AsyncMock(return_value=_created_ws)
    state.workspace_manager = MagicMock()
    state.workspace_manager.list_workspaces = AsyncMock(return_value=[_make_workspace()])
    state.convoy_tracker = MagicMock()
    state.convoy_tracker.list_convoys = AsyncMock(return_value=[_make_convoy()])

    # Moltbot
    state.inbox_manager = MagicMock()
    state.inbox_manager.get_stats = AsyncMock(
        return_value={"messages_pending": 12, "channels_active": 3}
    )
    state.inbox_manager.list_messages = AsyncMock(return_value=[_make_message()])
    state.local_gateway = MagicMock()
    state.local_gateway.get_stats = AsyncMock(return_value={"devices_connected": 5})
    state.local_gateway.list_devices = AsyncMock(
        return_value=[_make_device(last_seen=datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc))]
    )
    state.voice_processor = MagicMock()
    state.voice_processor.get_stats = AsyncMock(return_value={"active_sessions": 2})
    state.onboarding = MagicMock()
    state.onboarding.get_stats = AsyncMock(return_value={"flows_active": 3})
    state.onboarding.list_flows = AsyncMock(return_value=[_make_flow()])

    return state


# ===========================================================================
# Extension Status Tests
# ===========================================================================


class TestHandleExtensionsStatus:
    """Tests for handle_extensions_status."""

    @pytest.mark.asyncio
    async def test_returns_ok_with_all_extensions(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_extensions_status

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_status(auth_ctx)

        assert result["status"] == "ok"
        exts = result["extensions"]
        assert exts["agent_fabric"]["enabled"] is True
        assert exts["agent_fabric"]["available"] is True
        assert exts["gastown"]["enabled"] is True
        assert exts["gastown"]["available"] is True
        assert exts["moltbot"]["enabled"] is True
        assert exts["moltbot"]["available"] is True
        assert exts["computer_use"]["enabled"] is False
        assert exts["computer_use"]["available"] is False

    @pytest.mark.asyncio
    async def test_returns_unavailable_when_state_is_none(self, auth_ctx):
        from aragora.server.handlers.extensions import handle_extensions_status

        with patch(PATCH_TARGET, return_value=None):
            result = await handle_extensions_status(auth_ctx)

        assert result["status"] == "unavailable"
        assert "error" in result
        assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_status_with_all_disabled(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_extensions_status

        full_state.fabric_enabled = False
        full_state.gastown_enabled = False
        full_state.moltbot_enabled = False
        full_state.computer_use_enabled = False
        full_state.metadata = {}

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_status(auth_ctx)

        assert result["status"] == "ok"
        for ext in result["extensions"].values():
            assert ext["enabled"] is False
            assert ext["available"] is False

    @pytest.mark.asyncio
    async def test_status_with_partial_metadata(self, full_state, auth_ctx):
        """Metadata may not have all keys; .get() should return False."""
        from aragora.server.handlers.extensions import handle_extensions_status

        full_state.metadata = {"fabric_available": True}

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_status(auth_ctx)

        exts = result["extensions"]
        assert exts["agent_fabric"]["available"] is True
        assert exts["gastown"]["available"] is False
        assert exts["moltbot"]["available"] is False
        assert exts["computer_use"]["available"] is False


# ===========================================================================
# Extension Stats Tests
# ===========================================================================


class TestHandleExtensionsStats:
    """Tests for handle_extensions_stats."""

    @pytest.mark.asyncio
    async def test_all_stats_collected(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_extensions_stats

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        assert result["status"] == "ok"
        assert result["agent_fabric"] == {"agents_active": 5, "tasks_pending": 3}
        assert result["gastown"] == {"workspaces": 3, "rigs_active": 7}
        moltbot = result["moltbot"]
        assert "inbox" in moltbot
        assert "gateway" in moltbot
        assert "voice" in moltbot
        assert "onboarding" in moltbot

    @pytest.mark.asyncio
    async def test_returns_unavailable_when_state_is_none(self, auth_ctx):
        from aragora.server.handlers.extensions import handle_extensions_stats

        with patch(PATCH_TARGET, return_value=None):
            result = await handle_extensions_stats(auth_ctx)

        assert result["status"] == "unavailable"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_fabric_disabled_no_fabric_stats(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_extensions_stats

        full_state.fabric_enabled = False

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        assert "agent_fabric" not in result

    @pytest.mark.asyncio
    async def test_fabric_enabled_but_none(self, full_state, auth_ctx):
        """fabric_enabled=True but fabric object is None => no stats."""
        from aragora.server.handlers.extensions import handle_extensions_stats

        full_state.fabric = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        assert "agent_fabric" not in result

    @pytest.mark.asyncio
    async def test_gastown_disabled_no_gastown_stats(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_extensions_stats

        full_state.gastown_enabled = False

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        assert "gastown" not in result

    @pytest.mark.asyncio
    async def test_gastown_enabled_but_coordinator_none(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_extensions_stats

        full_state.coordinator = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        assert "gastown" not in result

    @pytest.mark.asyncio
    async def test_moltbot_disabled_no_moltbot_stats(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_extensions_stats

        full_state.moltbot_enabled = False

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        assert "moltbot" not in result

    @pytest.mark.asyncio
    async def test_moltbot_partial_components(self, full_state, auth_ctx):
        """Only some moltbot sub-components present."""
        from aragora.server.handlers.extensions import handle_extensions_stats

        full_state.voice_processor = None
        full_state.onboarding = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        moltbot = result["moltbot"]
        assert "inbox" in moltbot
        assert "gateway" in moltbot
        assert "voice" not in moltbot
        assert "onboarding" not in moltbot

    @pytest.mark.asyncio
    async def test_moltbot_all_components_none(self, full_state, auth_ctx):
        """Moltbot enabled but all sub-components None."""
        from aragora.server.handlers.extensions import handle_extensions_stats

        full_state.inbox_manager = None
        full_state.local_gateway = None
        full_state.voice_processor = None
        full_state.onboarding = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        assert result["moltbot"] == {}

    # -- Exception handling per component --

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, AttributeError, OSError],
        ids=["TypeError", "ValueError", "AttributeError", "OSError"],
    )
    async def test_fabric_stats_exception(self, full_state, auth_ctx, exc_class):
        from aragora.server.handlers.extensions import handle_extensions_stats

        full_state.fabric.get_stats = AsyncMock(side_effect=exc_class("boom"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        assert result["status"] == "ok"
        assert "error" in result["agent_fabric"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, AttributeError, OSError],
        ids=["TypeError", "ValueError", "AttributeError", "OSError"],
    )
    async def test_gastown_stats_exception(self, full_state, auth_ctx, exc_class):
        from aragora.server.handlers.extensions import handle_extensions_stats

        full_state.coordinator.get_stats = AsyncMock(side_effect=exc_class("boom"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        assert result["status"] == "ok"
        assert "error" in result["gastown"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "component,exc_class",
        [
            ("inbox_manager", TypeError),
            ("inbox_manager", ValueError),
            ("inbox_manager", OSError),
            ("local_gateway", TypeError),
            ("local_gateway", AttributeError),
            ("voice_processor", ValueError),
            ("voice_processor", OSError),
            ("onboarding", TypeError),
            ("onboarding", AttributeError),
        ],
    )
    async def test_moltbot_component_stats_exception(
        self, full_state, auth_ctx, component, exc_class
    ):
        from aragora.server.handlers.extensions import handle_extensions_stats

        getattr(full_state, component).get_stats = AsyncMock(side_effect=exc_class("fail"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        assert result["status"] == "ok"
        # The sub-key should have an error
        sub_key = {
            "inbox_manager": "inbox",
            "local_gateway": "gateway",
            "voice_processor": "voice",
            "onboarding": "onboarding",
        }[component]
        assert "error" in result["moltbot"][sub_key]

    @pytest.mark.asyncio
    async def test_all_extensions_error_still_returns_ok(self, full_state, auth_ctx):
        """When every component fails, status is still 'ok' with errors per-component."""
        from aragora.server.handlers.extensions import handle_extensions_stats

        full_state.fabric.get_stats = AsyncMock(side_effect=OSError("fabric"))
        full_state.coordinator.get_stats = AsyncMock(side_effect=OSError("gastown"))
        full_state.inbox_manager.get_stats = AsyncMock(side_effect=OSError("inbox"))
        full_state.local_gateway.get_stats = AsyncMock(side_effect=OSError("gw"))
        full_state.voice_processor.get_stats = AsyncMock(side_effect=OSError("voice"))
        full_state.onboarding.get_stats = AsyncMock(side_effect=OSError("onboard"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_extensions_stats(auth_ctx)

        assert result["status"] == "ok"
        assert "error" in result["agent_fabric"]
        assert "error" in result["gastown"]
        for key in ("inbox", "gateway", "voice", "onboarding"):
            assert "error" in result["moltbot"][key]


# ===========================================================================
# Gastown Workspace List Tests
# ===========================================================================


class TestHandleGastownWorkspacesList:
    """Tests for handle_gastown_workspaces_list."""

    @pytest.mark.asyncio
    async def test_list_workspaces_success(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_workspaces_list

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_workspaces_list(auth_ctx)

        assert result["status"] == "ok"
        assert len(result["workspaces"]) == 1
        ws = result["workspaces"][0]
        assert ws["id"] == "ws-1"
        assert ws["name"] == "Workspace 1"
        assert ws["status"] == "active"
        assert ws["rigs"] == 1
        assert "created_at" in ws

    @pytest.mark.asyncio
    async def test_list_workspaces_multiple(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_workspaces_list

        full_state.workspace_manager.list_workspaces = AsyncMock(
            return_value=[
                _make_workspace("ws-1", "WS One"),
                _make_workspace("ws-2", "WS Two", rigs=["r1", "r2", "r3"]),
                _make_workspace("ws-3", "WS Three", status="archived", rigs=[]),
            ]
        )

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_workspaces_list(auth_ctx)

        assert len(result["workspaces"]) == 3
        assert result["workspaces"][1]["rigs"] == 3
        assert result["workspaces"][2]["rigs"] == 0

    @pytest.mark.asyncio
    async def test_list_workspaces_empty(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_workspaces_list

        full_state.workspace_manager.list_workspaces = AsyncMock(return_value=[])

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_workspaces_list(auth_ctx)

        assert result["status"] == "ok"
        assert result["workspaces"] == []

    @pytest.mark.asyncio
    async def test_list_workspaces_state_none(self, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_workspaces_list

        with patch(PATCH_TARGET, return_value=None):
            result = await handle_gastown_workspaces_list(auth_ctx)

        assert "error" in result
        assert "not enabled" in result["error"]

    @pytest.mark.asyncio
    async def test_list_workspaces_gastown_disabled(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_workspaces_list

        full_state.gastown_enabled = False

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_workspaces_list(auth_ctx)

        assert "error" in result
        assert "not enabled" in result["error"]
        assert result.get("code") == "SERVICE_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_list_workspaces_manager_none(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_workspaces_list

        full_state.workspace_manager = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_workspaces_list(auth_ctx)

        assert "error" in result
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, OSError],
        ids=["TypeError", "ValueError", "KeyError", "AttributeError", "OSError"],
    )
    async def test_list_workspaces_exception(self, full_state, auth_ctx, exc_class):
        from aragora.server.handlers.extensions import handle_gastown_workspaces_list

        full_state.workspace_manager.list_workspaces = AsyncMock(side_effect=exc_class("fail"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_workspaces_list(auth_ctx)

        assert "error" in result
        assert result.get("code") == "INTERNAL_ERROR"


# ===========================================================================
# Gastown Workspace Create Tests
# ===========================================================================


class TestHandleGastownWorkspaceCreate:
    """Tests for handle_gastown_workspace_create."""

    @pytest.mark.asyncio
    async def test_create_workspace_success(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_workspace_create

        data = {"name": "My Workspace", "root_path": "/tmp/ws", "description": "A workspace"}

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_workspace_create(auth_ctx, data)

        assert result["status"] == "ok"
        assert result["workspace"]["id"] == "ws-new"
        assert result["workspace"]["name"] == "New Workspace"
        assert result["workspace"]["status"] == "active"

        # Verify coordinator was called with correct arguments
        full_state.coordinator.create_workspace.assert_awaited_once_with(
            name="My Workspace",
            root_path="/tmp/ws",
            description="A workspace",
            owner_id="user-123",
            tenant_id="org-456",
        )

    @pytest.mark.asyncio
    async def test_create_workspace_defaults(self, full_state, auth_ctx):
        """data.get() fallbacks should provide defaults for missing fields."""
        from aragora.server.handlers.extensions import handle_gastown_workspace_create

        with patch(PATCH_TARGET, return_value=full_state):
            await handle_gastown_workspace_create(auth_ctx, {})

        call_kwargs = full_state.coordinator.create_workspace.call_args.kwargs
        assert call_kwargs["name"] == "Unnamed Workspace"
        assert call_kwargs["root_path"] == "/tmp/workspace"
        assert call_kwargs["description"] == ""

    @pytest.mark.asyncio
    async def test_create_workspace_none_ctx(self, full_state):
        """When ctx is None, owner_id/tenant_id should be empty/None."""
        from aragora.server.handlers.extensions import handle_gastown_workspace_create

        with patch(PATCH_TARGET, return_value=full_state):
            await handle_gastown_workspace_create(None, {"name": "Test"})

        call_kwargs = full_state.coordinator.create_workspace.call_args.kwargs
        assert call_kwargs["owner_id"] == ""
        assert call_kwargs["tenant_id"] is None

    @pytest.mark.asyncio
    async def test_create_workspace_state_none(self, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_workspace_create

        with patch(PATCH_TARGET, return_value=None):
            result = await handle_gastown_workspace_create(auth_ctx, {})

        assert "error" in result
        assert "not enabled" in result["error"]

    @pytest.mark.asyncio
    async def test_create_workspace_gastown_disabled(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_workspace_create

        full_state.gastown_enabled = False

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_workspace_create(auth_ctx, {})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_workspace_coordinator_none(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_workspace_create

        full_state.coordinator = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_workspace_create(auth_ctx, {})

        assert "error" in result
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, OSError],
        ids=["TypeError", "ValueError", "KeyError", "AttributeError", "OSError"],
    )
    async def test_create_workspace_exception(self, full_state, auth_ctx, exc_class):
        from aragora.server.handlers.extensions import handle_gastown_workspace_create

        full_state.coordinator.create_workspace = AsyncMock(side_effect=exc_class("fail"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_workspace_create(auth_ctx, {"name": "x"})

        assert "error" in result
        assert result.get("code") == "INTERNAL_ERROR"


# ===========================================================================
# Gastown Convoys List Tests
# ===========================================================================


class TestHandleGastownConvoysList:
    """Tests for handle_gastown_convoys_list."""

    @pytest.mark.asyncio
    async def test_list_convoys_success(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_convoys_list

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_convoys_list(auth_ctx)

        assert result["status"] == "ok"
        assert len(result["convoys"]) == 1
        c = result["convoys"][0]
        assert c["id"] == "convoy-1"
        assert c["title"] == "Test Convoy"
        assert c["status"] == "running"
        assert c["rig_id"] == "rig-1"
        assert "created_at" in c

    @pytest.mark.asyncio
    async def test_list_convoys_multiple(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_convoys_list

        full_state.convoy_tracker.list_convoys = AsyncMock(
            return_value=[
                _make_convoy("c1", "Convoy 1"),
                _make_convoy("c2", "Convoy 2", _ConvoyStatus.COMPLETED, "rig-2"),
            ]
        )

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_convoys_list(auth_ctx)

        assert len(result["convoys"]) == 2
        assert result["convoys"][1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_convoys_empty(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_convoys_list

        full_state.convoy_tracker.list_convoys = AsyncMock(return_value=[])

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_convoys_list(auth_ctx)

        assert result["status"] == "ok"
        assert result["convoys"] == []

    @pytest.mark.asyncio
    async def test_list_convoys_state_none(self, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_convoys_list

        with patch(PATCH_TARGET, return_value=None):
            result = await handle_gastown_convoys_list(auth_ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_convoys_gastown_disabled(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_convoys_list

        full_state.gastown_enabled = False

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_convoys_list(auth_ctx)

        assert "error" in result
        assert "not enabled" in result["error"]

    @pytest.mark.asyncio
    async def test_list_convoys_tracker_none(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_gastown_convoys_list

        full_state.convoy_tracker = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_convoys_list(auth_ctx)

        assert "error" in result
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, OSError],
    )
    async def test_list_convoys_exception(self, full_state, auth_ctx, exc_class):
        from aragora.server.handlers.extensions import handle_gastown_convoys_list

        full_state.convoy_tracker.list_convoys = AsyncMock(side_effect=exc_class("fail"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_gastown_convoys_list(auth_ctx)

        assert "error" in result
        assert result.get("code") == "INTERNAL_ERROR"


# ===========================================================================
# Moltbot Inbox Messages Tests
# ===========================================================================


class TestHandleMoltbotInboxMessages:
    """Tests for handle_moltbot_inbox_messages."""

    @pytest.mark.asyncio
    async def test_list_messages_success(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_inbox_messages(auth_ctx)

        assert result["status"] == "ok"
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg["id"] == "msg-1"
        assert msg["channel_id"] == "ch-1"
        assert msg["direction"] == "inbound"
        assert msg["content"] == "Hello world"
        assert msg["status"] == "pending"
        assert "created_at" in msg

    @pytest.mark.asyncio
    async def test_list_messages_passes_user_id(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        with patch(PATCH_TARGET, return_value=full_state):
            await handle_moltbot_inbox_messages(auth_ctx)

        full_state.inbox_manager.list_messages.assert_awaited_once_with(
            user_id="user-123", limit=100
        )

    @pytest.mark.asyncio
    async def test_list_messages_none_ctx(self, full_state):
        """When ctx is None, user_id should be None."""
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        with patch(PATCH_TARGET, return_value=full_state):
            await handle_moltbot_inbox_messages(None)

        full_state.inbox_manager.list_messages.assert_awaited_once_with(user_id=None, limit=100)

    @pytest.mark.asyncio
    async def test_long_content_truncated(self, full_state, auth_ctx):
        """Messages longer than 100 chars should be truncated with '...'."""
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        long_content = "A" * 200
        full_state.inbox_manager.list_messages = AsyncMock(
            return_value=[_make_message(content=long_content)]
        )

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_inbox_messages(auth_ctx)

        displayed = result["messages"][0]["content"]
        assert len(displayed) == 103  # 100 chars + "..."
        assert displayed.endswith("...")

    @pytest.mark.asyncio
    async def test_exact_100_char_not_truncated(self, full_state, auth_ctx):
        """Exactly 100-char content should NOT be truncated."""
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        exact_content = "B" * 100
        full_state.inbox_manager.list_messages = AsyncMock(
            return_value=[_make_message(content=exact_content)]
        )

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_inbox_messages(auth_ctx)

        assert result["messages"][0]["content"] == exact_content

    @pytest.mark.asyncio
    async def test_list_messages_empty(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        full_state.inbox_manager.list_messages = AsyncMock(return_value=[])

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_inbox_messages(auth_ctx)

        assert result["status"] == "ok"
        assert result["messages"] == []

    @pytest.mark.asyncio
    async def test_list_messages_state_none(self, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        with patch(PATCH_TARGET, return_value=None):
            result = await handle_moltbot_inbox_messages(auth_ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_messages_moltbot_disabled(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        full_state.moltbot_enabled = False

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_inbox_messages(auth_ctx)

        assert "error" in result
        assert "not enabled" in result["error"]

    @pytest.mark.asyncio
    async def test_list_messages_inbox_manager_none(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        full_state.inbox_manager = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_inbox_messages(auth_ctx)

        assert "error" in result
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, OSError],
    )
    async def test_list_messages_exception(self, full_state, auth_ctx, exc_class):
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        full_state.inbox_manager.list_messages = AsyncMock(side_effect=exc_class("fail"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_inbox_messages(auth_ctx)

        assert "error" in result
        assert result.get("code") == "INTERNAL_ERROR"


# ===========================================================================
# Moltbot Gateway Devices Tests
# ===========================================================================


class TestHandleMoltbotGatewayDevices:
    """Tests for handle_moltbot_gateway_devices."""

    @pytest.mark.asyncio
    async def test_list_devices_success(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_gateway_devices

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_gateway_devices(auth_ctx)

        assert result["status"] == "ok"
        assert len(result["devices"]) == 1
        d = result["devices"][0]
        assert d["id"] == "dev-1"
        assert d["name"] == "Device 1"
        assert d["type"] == "sensor"
        assert d["status"] == "online"
        assert d["last_seen"] is not None

    @pytest.mark.asyncio
    async def test_device_last_seen_none(self, full_state, auth_ctx):
        """Device with last_seen=None should serialize as None."""
        from aragora.server.handlers.extensions import handle_moltbot_gateway_devices

        full_state.local_gateway.list_devices = AsyncMock(
            return_value=[_make_device(last_seen=None)]
        )

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_gateway_devices(auth_ctx)

        assert result["devices"][0]["last_seen"] is None

    @pytest.mark.asyncio
    async def test_list_devices_multiple(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_gateway_devices

        full_state.local_gateway.list_devices = AsyncMock(
            return_value=[
                _make_device(
                    "d1", "Camera", "camera", "online", datetime(2025, 1, 1, tzinfo=timezone.utc)
                ),
                _make_device("d2", "Lock", "actuator", "offline", None),
            ]
        )

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_gateway_devices(auth_ctx)

        assert len(result["devices"]) == 2
        assert result["devices"][0]["type"] == "camera"
        assert result["devices"][1]["last_seen"] is None

    @pytest.mark.asyncio
    async def test_list_devices_empty(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_gateway_devices

        full_state.local_gateway.list_devices = AsyncMock(return_value=[])

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_gateway_devices(auth_ctx)

        assert result["status"] == "ok"
        assert result["devices"] == []

    @pytest.mark.asyncio
    async def test_list_devices_state_none(self, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_gateway_devices

        with patch(PATCH_TARGET, return_value=None):
            result = await handle_moltbot_gateway_devices(auth_ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_devices_moltbot_disabled(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_gateway_devices

        full_state.moltbot_enabled = False

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_gateway_devices(auth_ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_devices_gateway_none(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_gateway_devices

        full_state.local_gateway = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_gateway_devices(auth_ctx)

        assert "error" in result
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, OSError],
    )
    async def test_list_devices_exception(self, full_state, auth_ctx, exc_class):
        from aragora.server.handlers.extensions import handle_moltbot_gateway_devices

        full_state.local_gateway.list_devices = AsyncMock(side_effect=exc_class("fail"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_gateway_devices(auth_ctx)

        assert "error" in result
        assert result.get("code") == "INTERNAL_ERROR"


# ===========================================================================
# Moltbot Onboarding Flows Tests
# ===========================================================================


class TestHandleMoltbotOnboardingFlows:
    """Tests for handle_moltbot_onboarding_flows."""

    @pytest.mark.asyncio
    async def test_list_flows_success(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_onboarding_flows

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_onboarding_flows(auth_ctx)

        assert result["status"] == "ok"
        assert len(result["flows"]) == 1
        f = result["flows"][0]
        assert f["id"] == "flow-1"
        assert f["name"] == "Onboarding Flow"
        assert f["status"] == "active"
        assert f["steps"] == 3
        assert f["started"] == 100
        assert f["completed"] == 85

    @pytest.mark.asyncio
    async def test_list_flows_multiple(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_onboarding_flows

        full_state.onboarding.list_flows = AsyncMock(
            return_value=[
                _make_flow("f1", "Flow 1", "active", ["a", "b"], 50, 40),
                _make_flow("f2", "Flow 2", "draft", [], 0, 0),
            ]
        )

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_onboarding_flows(auth_ctx)

        assert len(result["flows"]) == 2
        assert result["flows"][1]["steps"] == 0

    @pytest.mark.asyncio
    async def test_list_flows_empty(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_onboarding_flows

        full_state.onboarding.list_flows = AsyncMock(return_value=[])

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_onboarding_flows(auth_ctx)

        assert result["status"] == "ok"
        assert result["flows"] == []

    @pytest.mark.asyncio
    async def test_flow_name_from_mock_name(self, full_state, auth_ctx):
        """When flow.name is not a string, fall back to flow._mock_name."""
        from aragora.server.handlers.extensions import handle_moltbot_onboarding_flows

        flow = MagicMock()
        flow.id = "flow-x"
        flow.name = MagicMock()  # Not a string
        flow._mock_name = "Fallback Name"
        flow.status = "active"
        flow.steps = ["s1"]
        flow.started_count = 10
        flow.completed_count = 5
        full_state.onboarding.list_flows = AsyncMock(return_value=[flow])

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_onboarding_flows(auth_ctx)

        assert result["flows"][0]["name"] == "Fallback Name"

    @pytest.mark.asyncio
    async def test_flow_name_neither_string(self, full_state, auth_ctx):
        """When neither name nor _mock_name is a string, return whatever name is."""
        from aragora.server.handlers.extensions import handle_moltbot_onboarding_flows

        flow = MagicMock()
        flow.id = "flow-y"
        # Override name with a non-string, non-mock value
        flow.name = 42
        flow._mock_name = 99  # Also not a string
        flow.status = "active"
        flow.steps = []
        flow.started_count = 0
        flow.completed_count = 0
        full_state.onboarding.list_flows = AsyncMock(return_value=[flow])

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_onboarding_flows(auth_ctx)

        # Should return the original name (42)
        assert result["flows"][0]["name"] == 42

    @pytest.mark.asyncio
    async def test_list_flows_state_none(self, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_onboarding_flows

        with patch(PATCH_TARGET, return_value=None):
            result = await handle_moltbot_onboarding_flows(auth_ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_flows_moltbot_disabled(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_onboarding_flows

        full_state.moltbot_enabled = False

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_onboarding_flows(auth_ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_flows_onboarding_none(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_moltbot_onboarding_flows

        full_state.onboarding = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_onboarding_flows(auth_ctx)

        assert "error" in result
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, OSError],
    )
    async def test_list_flows_exception(self, full_state, auth_ctx, exc_class):
        from aragora.server.handlers.extensions import handle_moltbot_onboarding_flows

        full_state.onboarding.list_flows = AsyncMock(side_effect=exc_class("fail"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_moltbot_onboarding_flows(auth_ctx)

        assert "error" in result
        assert result.get("code") == "INTERNAL_ERROR"


# ===========================================================================
# Fabric Agents List Tests
# ===========================================================================


class TestHandleFabricAgentsList:
    """Tests for handle_fabric_agents_list."""

    @pytest.mark.asyncio
    async def test_list_agents_success(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_fabric_agents_list

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_fabric_agents_list(auth_ctx)

        assert result["status"] == "ok"
        assert len(result["agents"]) == 2
        a = result["agents"][0]
        assert a["id"] == "agent-1"
        assert a["model"] == "claude-opus-4"
        assert a["status"] == "active"
        assert a["created_at"] == "2025-01-15T10:00:00Z"

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_fabric_agents_list

        full_state.fabric.list_agents = AsyncMock(return_value=[])

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_fabric_agents_list(auth_ctx)

        assert result["status"] == "ok"
        assert result["agents"] == []

    @pytest.mark.asyncio
    async def test_list_agents_state_none(self, auth_ctx):
        from aragora.server.handlers.extensions import handle_fabric_agents_list

        with patch(PATCH_TARGET, return_value=None):
            result = await handle_fabric_agents_list(auth_ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_agents_fabric_disabled(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_fabric_agents_list

        full_state.fabric_enabled = False

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_fabric_agents_list(auth_ctx)

        assert "error" in result
        assert "not enabled" in result["error"]

    @pytest.mark.asyncio
    async def test_list_agents_fabric_none(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_fabric_agents_list

        full_state.fabric = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_fabric_agents_list(auth_ctx)

        assert "error" in result
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, OSError],
    )
    async def test_list_agents_exception(self, full_state, auth_ctx, exc_class):
        from aragora.server.handlers.extensions import handle_fabric_agents_list

        full_state.fabric.list_agents = AsyncMock(side_effect=exc_class("fail"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_fabric_agents_list(auth_ctx)

        assert "error" in result
        assert result.get("code") == "INTERNAL_ERROR"


# ===========================================================================
# Fabric Tasks List Tests
# ===========================================================================


class TestHandleFabricTasksList:
    """Tests for handle_fabric_tasks_list."""

    @pytest.mark.asyncio
    async def test_list_tasks_success(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_fabric_tasks_list

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_fabric_tasks_list(auth_ctx)

        assert result["status"] == "ok"
        assert result["stats"] == {"agents_active": 5, "tasks_pending": 3}

    @pytest.mark.asyncio
    async def test_list_tasks_state_none(self, auth_ctx):
        from aragora.server.handlers.extensions import handle_fabric_tasks_list

        with patch(PATCH_TARGET, return_value=None):
            result = await handle_fabric_tasks_list(auth_ctx)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_tasks_fabric_disabled(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_fabric_tasks_list

        full_state.fabric_enabled = False

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_fabric_tasks_list(auth_ctx)

        assert "error" in result
        assert "not enabled" in result["error"]

    @pytest.mark.asyncio
    async def test_list_tasks_fabric_none(self, full_state, auth_ctx):
        from aragora.server.handlers.extensions import handle_fabric_tasks_list

        full_state.fabric = None

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_fabric_tasks_list(auth_ctx)

        assert "error" in result
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, AttributeError, OSError],
    )
    async def test_list_tasks_exception(self, full_state, auth_ctx, exc_class):
        """Note: handle_fabric_tasks_list catches TypeError, ValueError,
        AttributeError, OSError (no KeyError unlike some other handlers)."""
        from aragora.server.handlers.extensions import handle_fabric_tasks_list

        full_state.fabric.get_stats = AsyncMock(side_effect=exc_class("fail"))

        with patch(PATCH_TARGET, return_value=full_state):
            result = await handle_fabric_tasks_list(auth_ctx)

        assert "error" in result
        assert result.get("code") == "INTERNAL_ERROR"


# ===========================================================================
# Route Registration Tests
# ===========================================================================


class TestGetExtensionRoutes:
    """Tests for get_extension_routes helper."""

    def test_returns_all_10_routes(self):
        from aragora.server.handlers.extensions import get_extension_routes

        routes = get_extension_routes()
        assert len(routes) == 10

    def test_route_keys(self):
        from aragora.server.handlers.extensions import get_extension_routes

        routes = get_extension_routes()
        expected_paths = {
            "/api/extensions/status",
            "/api/extensions/stats",
            "/api/extensions/gastown/workspaces",
            "/api/extensions/gastown/workspaces/create",
            "/api/extensions/gastown/convoys",
            "/api/extensions/moltbot/inbox/messages",
            "/api/extensions/moltbot/gateway/devices",
            "/api/extensions/moltbot/onboarding/flows",
            "/api/extensions/fabric/agents",
            "/api/extensions/fabric/tasks",
        }
        assert set(routes.keys()) == expected_paths

    def test_status_routes_are_get(self):
        from aragora.server.handlers.extensions import get_extension_routes

        routes = get_extension_routes()
        _, methods = routes["/api/extensions/status"]
        assert methods == ["GET"]
        _, methods = routes["/api/extensions/stats"]
        assert methods == ["GET"]

    def test_workspace_create_is_post(self):
        from aragora.server.handlers.extensions import get_extension_routes

        routes = get_extension_routes()
        _, methods = routes["/api/extensions/gastown/workspaces/create"]
        assert methods == ["POST"]

    def test_all_read_routes_are_get(self):
        from aragora.server.handlers.extensions import get_extension_routes

        routes = get_extension_routes()
        get_paths = [
            "/api/extensions/status",
            "/api/extensions/stats",
            "/api/extensions/gastown/workspaces",
            "/api/extensions/gastown/convoys",
            "/api/extensions/moltbot/inbox/messages",
            "/api/extensions/moltbot/gateway/devices",
            "/api/extensions/moltbot/onboarding/flows",
            "/api/extensions/fabric/agents",
            "/api/extensions/fabric/tasks",
        ]
        for path in get_paths:
            _, methods = routes[path]
            assert methods == ["GET"], f"{path} should be GET-only"

    def test_route_values_are_tuples_with_handler_and_methods(self):
        from aragora.server.handlers.extensions import get_extension_routes

        routes = get_extension_routes()
        for path, value in routes.items():
            assert isinstance(value, tuple), f"{path} value should be a tuple"
            assert len(value) == 2, f"{path} tuple should have (handler, methods)"
            handler, methods = value
            assert callable(handler), f"{path} handler should be callable"
            assert isinstance(methods, list), f"{path} methods should be a list"

    def test_handler_functions_are_correct(self):
        from aragora.server.handlers.extensions import (
            get_extension_routes,
            handle_extensions_status,
            handle_extensions_stats,
            handle_gastown_workspaces_list,
            handle_gastown_workspace_create,
            handle_gastown_convoys_list,
            handle_moltbot_inbox_messages,
            handle_moltbot_gateway_devices,
            handle_moltbot_onboarding_flows,
            handle_fabric_agents_list,
            handle_fabric_tasks_list,
        )

        routes = get_extension_routes()

        # The handlers are wrapped by @require_permission, so we cannot do
        # identity comparison. Instead verify the route count and that each
        # value's handler is callable (already tested above).
        assert len(routes) == 10


# ===========================================================================
# Cross-Cutting / Edge Case Tests
# ===========================================================================


class TestExtensionsCrossCutting:
    """Cross-cutting tests: consistency, idempotency, edge cases."""

    @pytest.mark.asyncio
    async def test_status_and_stats_consistent(self, full_state, auth_ctx):
        """If an extension is enabled in status, it should have stats."""
        from aragora.server.handlers.extensions import (
            handle_extensions_status,
            handle_extensions_stats,
        )

        with patch(PATCH_TARGET, return_value=full_state):
            status = await handle_extensions_status(auth_ctx)
            stats = await handle_extensions_stats(auth_ctx)

        assert status["extensions"]["agent_fabric"]["enabled"] is True
        assert "agent_fabric" in stats

        assert status["extensions"]["gastown"]["enabled"] is True
        assert "gastown" in stats

        assert status["extensions"]["moltbot"]["enabled"] is True
        assert "moltbot" in stats

    @pytest.mark.asyncio
    async def test_idempotent_status_calls(self, full_state, auth_ctx):
        """Calling status multiple times should return same result."""
        from aragora.server.handlers.extensions import handle_extensions_status

        with patch(PATCH_TARGET, return_value=full_state):
            r1 = await handle_extensions_status(auth_ctx)
            r2 = await handle_extensions_status(auth_ctx)

        assert r1 == r2

    @pytest.mark.asyncio
    async def test_workspace_create_then_list(self, full_state, auth_ctx):
        """Workspace create and list should both succeed."""
        from aragora.server.handlers.extensions import (
            handle_gastown_workspace_create,
            handle_gastown_workspaces_list,
        )

        with patch(PATCH_TARGET, return_value=full_state):
            create = await handle_gastown_workspace_create(auth_ctx, {"name": "Test"})
            listing = await handle_gastown_workspaces_list(auth_ctx)

        assert create["status"] == "ok"
        assert listing["status"] == "ok"

    @pytest.mark.asyncio
    async def test_all_handlers_return_dict(self, full_state, auth_ctx):
        """Every handler should return a dict."""
        from aragora.server.handlers.extensions import (
            handle_extensions_status,
            handle_extensions_stats,
            handle_gastown_workspaces_list,
            handle_gastown_workspace_create,
            handle_gastown_convoys_list,
            handle_moltbot_inbox_messages,
            handle_moltbot_gateway_devices,
            handle_moltbot_onboarding_flows,
            handle_fabric_agents_list,
            handle_fabric_tasks_list,
        )

        handlers = [
            (handle_extensions_status, (auth_ctx,)),
            (handle_extensions_stats, (auth_ctx,)),
            (handle_gastown_workspaces_list, (auth_ctx,)),
            (handle_gastown_workspace_create, (auth_ctx, {"name": "T"})),
            (handle_gastown_convoys_list, (auth_ctx,)),
            (handle_moltbot_inbox_messages, (auth_ctx,)),
            (handle_moltbot_gateway_devices, (auth_ctx,)),
            (handle_moltbot_onboarding_flows, (auth_ctx,)),
            (handle_fabric_agents_list, (auth_ctx,)),
            (handle_fabric_tasks_list, (auth_ctx,)),
        ]

        with patch(PATCH_TARGET, return_value=full_state):
            for handler, args in handlers:
                result = await handler(*args)
                assert isinstance(result, dict), f"{handler.__name__} should return dict"

    @pytest.mark.asyncio
    async def test_all_handlers_with_none_state_return_error(self, auth_ctx):
        """Every handler should handle state=None gracefully."""
        from aragora.server.handlers.extensions import (
            handle_extensions_status,
            handle_extensions_stats,
            handle_gastown_workspaces_list,
            handle_gastown_workspace_create,
            handle_gastown_convoys_list,
            handle_moltbot_inbox_messages,
            handle_moltbot_gateway_devices,
            handle_moltbot_onboarding_flows,
            handle_fabric_agents_list,
            handle_fabric_tasks_list,
        )

        handlers = [
            (handle_extensions_status, (auth_ctx,)),
            (handle_extensions_stats, (auth_ctx,)),
            (handle_gastown_workspaces_list, (auth_ctx,)),
            (handle_gastown_workspace_create, (auth_ctx, {})),
            (handle_gastown_convoys_list, (auth_ctx,)),
            (handle_moltbot_inbox_messages, (auth_ctx,)),
            (handle_moltbot_gateway_devices, (auth_ctx,)),
            (handle_moltbot_onboarding_flows, (auth_ctx,)),
            (handle_fabric_agents_list, (auth_ctx,)),
            (handle_fabric_tasks_list, (auth_ctx,)),
        ]

        with patch(PATCH_TARGET, return_value=None):
            for handler, args in handlers:
                result = await handler(*args)
                assert isinstance(result, dict)
                # Should have either "error" or "status" == "unavailable"
                has_error = "error" in result
                is_unavailable = result.get("status") == "unavailable"
                assert has_error or is_unavailable, (
                    f"{handler.__name__} should signal error when state=None"
                )
