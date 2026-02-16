"""
Tests for ExtensionsHandler - Extension management endpoints.

Covers:
- Extension status and statistics
- Gastown workspace/convoy management
- Moltbot inbox/gateway/onboarding management
- Agent Fabric operations
- Route registration
- RBAC permission checks
- Error handling
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_extension_state():
    """Create a mock extension state."""
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

    # Fabric mock
    state.fabric = MagicMock()
    state.fabric.get_stats = AsyncMock(return_value={"agents_active": 5, "tasks_pending": 3})
    state.fabric.list_agents = AsyncMock(
        return_value=[
            MagicMock(
                id="agent-1",
                config=MagicMock(model="claude-opus-4"),
                status=MagicMock(value="active"),
                created_at="2024-01-15T10:00:00Z",
            ),
            MagicMock(
                id="agent-2",
                config=MagicMock(model="gpt-4"),
                status=MagicMock(value="idle"),
                created_at="2024-01-15T11:00:00Z",
            ),
        ]
    )

    # Gastown mocks
    state.coordinator = MagicMock()
    state.coordinator.get_stats = AsyncMock(return_value={"workspaces": 3, "rigs_active": 7})
    state.coordinator.create_workspace = AsyncMock(
        return_value=MagicMock(
            id="ws-new",
            config=MagicMock(name="New Workspace"),
            status="active",
        )
    )

    state.workspace_manager = MagicMock()
    state.workspace_manager.list_workspaces = AsyncMock(
        return_value=[
            MagicMock(
                id="ws-1",
                config=MagicMock(name="Workspace 1"),
                status="active",
                rigs=["rig-1", "rig-2"],
                created_at=datetime.now(timezone.utc),
            ),
        ]
    )

    state.convoy_tracker = MagicMock()
    state.convoy_tracker.list_convoys = AsyncMock(
        return_value=[
            MagicMock(
                id="convoy-1",
                title="Test Convoy",
                status=MagicMock(value="running"),
                rig_id="rig-1",
                created_at=datetime.now(timezone.utc),
            ),
        ]
    )

    # Moltbot mocks
    state.inbox_manager = MagicMock()
    state.inbox_manager.get_stats = AsyncMock(
        return_value={"messages_pending": 12, "channels_active": 3}
    )
    state.inbox_manager.list_messages = AsyncMock(
        return_value=[
            MagicMock(
                id="msg-1",
                channel_id="ch-1",
                direction="inbound",
                content="Test message content",
                status=MagicMock(value="pending"),
                created_at=datetime.now(timezone.utc),
            ),
        ]
    )

    state.local_gateway = MagicMock()
    state.local_gateway.get_stats = AsyncMock(
        return_value={"devices_connected": 5, "uptime_hours": 72}
    )
    state.local_gateway.list_devices = AsyncMock(
        return_value=[
            MagicMock(
                id="dev-1",
                config=MagicMock(name="Device 1", device_type="sensor"),
                status="online",
                last_seen=datetime.now(timezone.utc),
            ),
        ]
    )

    state.voice_processor = MagicMock()
    state.voice_processor.get_stats = AsyncMock(
        return_value={"active_sessions": 2, "processed_today": 150}
    )

    state.onboarding = MagicMock()
    state.onboarding.get_stats = AsyncMock(return_value={"flows_active": 3, "users_onboarded": 45})
    state.onboarding.list_flows = AsyncMock(
        return_value=[
            MagicMock(
                id="flow-1",
                name="New User Flow",
                status="active",
                steps=["step1", "step2", "step3"],
                started_count=100,
                completed_count=85,
            ),
        ]
    )

    return state


@pytest.fixture
def mock_auth_context():
    """Create a mock authorization context."""
    ctx = MagicMock()
    ctx.user_id = "user-123"
    ctx.org_id = "org-456"
    ctx.permissions = [
        "workspaces:read",
        "workspaces:write",
        "convoys:read",
        "inbox:read",
        "devices:read",
        "onboarding:read",
        "agents:read",
        "tasks:read",
    ]
    return ctx


# -----------------------------------------------------------------------------
# Extension Status Tests
# -----------------------------------------------------------------------------


class TestExtensionsStatus:
    """Tests for extension status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_all_extensions(self, mock_extension_state, mock_auth_context):
        """Test getting status for all extensions."""
        from aragora.server.handlers.extensions import handle_extensions_status

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_extensions_status(mock_auth_context)

        assert result["status"] == "ok"
        assert "extensions" in result
        assert result["extensions"]["agent_fabric"]["enabled"] is True
        assert result["extensions"]["gastown"]["enabled"] is True
        assert result["extensions"]["moltbot"]["enabled"] is True
        assert result["extensions"]["computer_use"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_get_status_extensions_not_initialized(self, mock_auth_context):
        """Test status when extensions not initialized."""
        from aragora.server.handlers.extensions import handle_extensions_status

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=None,
        ):
            result = await handle_extensions_status(mock_auth_context)

        assert result["status"] == "unavailable"
        assert "error" in result


class TestExtensionsStats:
    """Tests for extension statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_all_extensions(self, mock_extension_state, mock_auth_context):
        """Test getting statistics for all enabled extensions."""
        from aragora.server.handlers.extensions import handle_extensions_stats

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_extensions_stats(mock_auth_context)

        assert result["status"] == "ok"
        assert "agent_fabric" in result
        assert "gastown" in result
        assert "moltbot" in result
        assert result["agent_fabric"]["agents_active"] == 5

    @pytest.mark.asyncio
    async def test_get_stats_handles_errors(self, mock_extension_state, mock_auth_context):
        """Test stats endpoint handles individual extension errors gracefully."""
        from aragora.server.handlers.extensions import handle_extensions_stats

        mock_extension_state.fabric.get_stats = AsyncMock(
            side_effect=ValueError("Fabric unavailable")
        )

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_extensions_stats(mock_auth_context)

        assert result["status"] == "ok"
        assert "error" in result["agent_fabric"]


# -----------------------------------------------------------------------------
# Gastown Endpoint Tests
# -----------------------------------------------------------------------------


class TestGastownWorkspaces:
    """Tests for Gastown workspace endpoints."""

    @pytest.mark.asyncio
    async def test_list_workspaces(self, mock_extension_state, mock_auth_context):
        """Test listing workspaces."""
        from aragora.server.handlers.extensions import handle_gastown_workspaces_list

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_gastown_workspaces_list(mock_auth_context)

        assert result["status"] == "ok"
        assert "workspaces" in result
        assert len(result["workspaces"]) == 1
        assert result["workspaces"][0]["id"] == "ws-1"

    @pytest.mark.asyncio
    async def test_list_workspaces_gastown_disabled(self, mock_extension_state, mock_auth_context):
        """Test listing workspaces when Gastown is disabled."""
        from aragora.server.handlers.extensions import handle_gastown_workspaces_list

        mock_extension_state.gastown_enabled = False

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_gastown_workspaces_list(mock_auth_context)

        assert "error" in result
        assert "not enabled" in result["error"]

    @pytest.mark.asyncio
    async def test_create_workspace(self, mock_extension_state, mock_auth_context):
        """Test creating a workspace."""
        from aragora.server.handlers.extensions import handle_gastown_workspace_create

        data = {
            "name": "Test Workspace",
            "root_path": "/tmp/test",
            "description": "A test workspace",
        }

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_gastown_workspace_create(mock_auth_context, data)

        assert result["status"] == "ok"
        assert "workspace" in result
        assert result["workspace"]["id"] == "ws-new"

    @pytest.mark.asyncio
    async def test_create_workspace_no_coordinator(self, mock_extension_state, mock_auth_context):
        """Test creating workspace when coordinator not available."""
        from aragora.server.handlers.extensions import handle_gastown_workspace_create

        mock_extension_state.coordinator = None

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_gastown_workspace_create(mock_auth_context, {})

        assert "error" in result


class TestGastownConvoys:
    """Tests for Gastown convoy endpoints."""

    @pytest.mark.asyncio
    async def test_list_convoys(self, mock_extension_state, mock_auth_context):
        """Test listing convoys."""
        from aragora.server.handlers.extensions import handle_gastown_convoys_list

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_gastown_convoys_list(mock_auth_context)

        assert result["status"] == "ok"
        assert "convoys" in result
        assert len(result["convoys"]) == 1
        assert result["convoys"][0]["id"] == "convoy-1"

    @pytest.mark.asyncio
    async def test_list_convoys_no_tracker(self, mock_extension_state, mock_auth_context):
        """Test listing convoys when tracker not available."""
        from aragora.server.handlers.extensions import handle_gastown_convoys_list

        mock_extension_state.convoy_tracker = None

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_gastown_convoys_list(mock_auth_context)

        assert "error" in result


# -----------------------------------------------------------------------------
# Moltbot Endpoint Tests
# -----------------------------------------------------------------------------


class TestMoltbotInbox:
    """Tests for Moltbot inbox endpoints."""

    @pytest.mark.asyncio
    async def test_list_inbox_messages(self, mock_extension_state, mock_auth_context):
        """Test listing inbox messages."""
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_moltbot_inbox_messages(mock_auth_context)

        assert result["status"] == "ok"
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["id"] == "msg-1"

    @pytest.mark.asyncio
    async def test_list_inbox_moltbot_disabled(self, mock_extension_state, mock_auth_context):
        """Test inbox when Moltbot is disabled."""
        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        mock_extension_state.moltbot_enabled = False

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_moltbot_inbox_messages(mock_auth_context)

        assert "error" in result


class TestMoltbotGateway:
    """Tests for Moltbot gateway endpoints."""

    @pytest.mark.asyncio
    async def test_list_devices(self, mock_extension_state, mock_auth_context):
        """Test listing gateway devices."""
        from aragora.server.handlers.extensions import handle_moltbot_gateway_devices

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_moltbot_gateway_devices(mock_auth_context)

        assert result["status"] == "ok"
        assert "devices" in result
        assert len(result["devices"]) == 1
        assert result["devices"][0]["id"] == "dev-1"

    @pytest.mark.asyncio
    async def test_list_devices_no_gateway(self, mock_extension_state, mock_auth_context):
        """Test listing devices when gateway not available."""
        from aragora.server.handlers.extensions import handle_moltbot_gateway_devices

        mock_extension_state.local_gateway = None

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_moltbot_gateway_devices(mock_auth_context)

        assert "error" in result


class TestMoltbotOnboarding:
    """Tests for Moltbot onboarding endpoints."""

    @pytest.mark.asyncio
    async def test_list_onboarding_flows(self, mock_extension_state, mock_auth_context):
        """Test listing onboarding flows."""
        from aragora.server.handlers.extensions import handle_moltbot_onboarding_flows

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_moltbot_onboarding_flows(mock_auth_context)

        assert result["status"] == "ok"
        assert "flows" in result
        assert len(result["flows"]) == 1
        assert result["flows"][0]["name"] == "New User Flow"

    @pytest.mark.asyncio
    async def test_list_flows_no_onboarding(self, mock_extension_state, mock_auth_context):
        """Test listing flows when onboarding not available."""
        from aragora.server.handlers.extensions import handle_moltbot_onboarding_flows

        mock_extension_state.onboarding = None

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_moltbot_onboarding_flows(mock_auth_context)

        assert "error" in result


# -----------------------------------------------------------------------------
# Agent Fabric Endpoint Tests
# -----------------------------------------------------------------------------


class TestAgentFabric:
    """Tests for Agent Fabric endpoints."""

    @pytest.mark.asyncio
    async def test_list_agents(self, mock_extension_state, mock_auth_context):
        """Test listing fabric agents."""
        from aragora.server.handlers.extensions import handle_fabric_agents_list

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_fabric_agents_list(mock_auth_context)

        assert result["status"] == "ok"
        assert "agents" in result
        assert len(result["agents"]) == 2
        assert result["agents"][0]["id"] == "agent-1"

    @pytest.mark.asyncio
    async def test_list_agents_fabric_disabled(self, mock_extension_state, mock_auth_context):
        """Test listing agents when fabric is disabled."""
        from aragora.server.handlers.extensions import handle_fabric_agents_list

        mock_extension_state.fabric_enabled = False

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_fabric_agents_list(mock_auth_context)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_tasks(self, mock_extension_state, mock_auth_context):
        """Test listing fabric tasks."""
        from aragora.server.handlers.extensions import handle_fabric_tasks_list

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_fabric_tasks_list(mock_auth_context)

        assert result["status"] == "ok"
        assert "stats" in result


# -----------------------------------------------------------------------------
# Route Registration Tests
# -----------------------------------------------------------------------------


class TestExtensionRouteRegistration:
    """Tests for extension route registration."""

    def test_get_extension_routes(self):
        """Test getting all extension routes."""
        from aragora.server.handlers.extensions import get_extension_routes

        routes = get_extension_routes()

        assert "/api/extensions/status" in routes
        assert "/api/extensions/stats" in routes
        assert "/api/extensions/gastown/workspaces" in routes
        assert "/api/extensions/moltbot/inbox/messages" in routes
        assert "/api/extensions/fabric/agents" in routes

    def test_route_methods(self):
        """Test that routes have correct HTTP methods."""
        from aragora.server.handlers.extensions import get_extension_routes

        routes = get_extension_routes()

        # Status endpoints should be GET
        _, methods = routes["/api/extensions/status"]
        assert "GET" in methods

        # Create endpoints should be POST
        _, methods = routes["/api/extensions/gastown/workspaces/create"]
        assert "POST" in methods


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


class TestExtensionsErrorHandling:
    """Tests for error handling in extensions."""

    @pytest.mark.asyncio
    async def test_handles_exception_in_stats(self, mock_extension_state, mock_auth_context):
        """Test graceful handling of exceptions in stats collection."""
        from aragora.server.handlers.extensions import handle_extensions_stats

        # All components throw errors
        mock_extension_state.fabric.get_stats = AsyncMock(side_effect=ValueError("Fabric error"))
        mock_extension_state.coordinator.get_stats = AsyncMock(
            side_effect=ValueError("Gastown error")
        )
        mock_extension_state.inbox_manager.get_stats = AsyncMock(
            side_effect=ValueError("Inbox error")
        )

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_extensions_stats(mock_auth_context)

        # Should still return ok status with error details per extension
        assert result["status"] == "ok"
        assert "error" in result.get("agent_fabric", {})

    @pytest.mark.asyncio
    async def test_handles_missing_components(self, mock_extension_state, mock_auth_context):
        """Test handling of missing optional components."""
        from aragora.server.handlers.extensions import handle_extensions_stats

        # Remove some components
        mock_extension_state.voice_processor = None
        mock_extension_state.onboarding = None

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            result = await handle_extensions_stats(mock_auth_context)

        assert result["status"] == "ok"
        # Should still have moltbot stats without voice/onboarding
        assert "moltbot" in result


# -----------------------------------------------------------------------------
# Integration-style Tests
# -----------------------------------------------------------------------------


class TestExtensionsIntegration:
    """Integration-style tests for extensions handler."""

    @pytest.mark.asyncio
    async def test_full_status_workflow(self, mock_extension_state, mock_auth_context):
        """Test complete status check workflow."""
        from aragora.server.handlers.extensions import (
            handle_extensions_status,
            handle_extensions_stats,
        )

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            # First check status
            status = await handle_extensions_status(mock_auth_context)
            assert status["status"] == "ok"

            # Then get detailed stats
            stats = await handle_extensions_stats(mock_auth_context)
            assert stats["status"] == "ok"

            # Verify consistency
            for ext_name, ext_info in status["extensions"].items():
                if ext_info["enabled"]:
                    # Enabled extensions should have stats
                    assert ext_name.replace("_", "") in str(stats) or ext_name in stats

    @pytest.mark.asyncio
    async def test_workspace_creation_workflow(self, mock_extension_state, mock_auth_context):
        """Test workspace creation and listing workflow."""
        from aragora.server.handlers.extensions import (
            handle_gastown_workspaces_list,
            handle_gastown_workspace_create,
        )

        with patch(
            "aragora.server.handlers.extensions.get_extension_state",
            return_value=mock_extension_state,
        ):
            # List existing workspaces
            list_result = await handle_gastown_workspaces_list(mock_auth_context)
            assert list_result["status"] == "ok"
            initial_count = len(list_result["workspaces"])

            # Create new workspace
            create_result = await handle_gastown_workspace_create(
                mock_auth_context,
                {"name": "Test", "root_path": "/tmp/test"},
            )
            assert create_result["status"] == "ok"
            assert create_result["workspace"]["id"] == "ws-new"
