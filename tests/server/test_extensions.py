"""
Tests for server extension integration.

Tests extension initialization, lifecycle, and handlers.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.extensions import (
    ExtensionState,
    init_extensions,
    init_agent_fabric,
    init_gastown,
    init_moltbot,
    init_computer_use,
    shutdown_extensions,
    get_extension_state,
    get_extension_stats,
    FABRIC_AVAILABLE,
    GASTOWN_AVAILABLE,
    MOLTBOT_AVAILABLE,
)


class TestExtensionInitialization:
    """Tests for extension initialization."""

    def test_extension_state_defaults(self):
        """Test ExtensionState has correct defaults."""
        state = ExtensionState()

        assert state.fabric is None
        assert state.coordinator is None
        assert state.inbox_manager is None
        assert state.fabric_enabled is False
        assert state.gastown_enabled is False
        assert state.moltbot_enabled is False

    @pytest.mark.skipif(not FABRIC_AVAILABLE, reason="Agent Fabric not available")
    def test_init_agent_fabric(self, tmp_path: Path):
        """Test Agent Fabric initialization."""
        fabric, hooks = init_agent_fabric(tmp_path)

        if FABRIC_AVAILABLE:
            assert fabric is not None
        else:
            assert fabric is None

    @pytest.mark.skipif(not GASTOWN_AVAILABLE, reason="Gastown not available")
    def test_init_gastown(self, tmp_path: Path):
        """Test Gastown initialization."""
        coordinator, workspace_mgr, convoy_tracker, hooks = init_gastown(tmp_path)

        if GASTOWN_AVAILABLE:
            assert coordinator is not None
            assert workspace_mgr is not None
            assert convoy_tracker is not None

    @pytest.mark.skipif(not MOLTBOT_AVAILABLE, reason="Moltbot not available")
    def test_init_moltbot(self, tmp_path: Path):
        """Test Moltbot initialization."""
        inbox, gateway, voice, onboarding = init_moltbot(tmp_path)

        if MOLTBOT_AVAILABLE:
            assert inbox is not None
            assert gateway is not None
            assert voice is not None
            assert onboarding is not None

    def test_init_computer_use_disabled_by_default(self):
        """Test Computer Use is disabled by default."""
        orchestrator, policy = init_computer_use()

        # Computer Use is disabled by default
        assert orchestrator is None
        assert policy is None

    def test_init_extensions_creates_state(self, tmp_path: Path):
        """Test init_extensions creates global state."""
        state = init_extensions(tmp_path)

        assert state is not None
        assert isinstance(state, ExtensionState)

        # Check global state is set
        global_state = get_extension_state()
        assert global_state is state

    def test_init_extensions_creates_storage_dir(self, tmp_path: Path):
        """Test init_extensions creates storage directory."""
        storage_path = tmp_path / "extensions"
        assert not storage_path.exists()

        init_extensions(storage_path)

        assert storage_path.exists()

    def test_get_extension_stats_without_init(self):
        """Test get_extension_stats before initialization."""
        # Clear global state
        import aragora.server.extensions as ext_module

        ext_module._extension_state = None

        stats = get_extension_stats()
        assert "error" in stats


class TestExtensionLifecycle:
    """Tests for extension lifecycle management."""

    @pytest.mark.asyncio
    async def test_shutdown_extensions_without_init(self):
        """Test shutdown_extensions handles no initialization."""
        import aragora.server.extensions as ext_module

        ext_module._extension_state = None

        # Should not raise
        await shutdown_extensions()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MOLTBOT_AVAILABLE, reason="Moltbot not available")
    async def test_shutdown_stops_gateway(self, tmp_path: Path):
        """Test shutdown stops Moltbot gateway."""
        state = init_extensions(tmp_path)

        if state.local_gateway:
            # Start gateway first
            await state.local_gateway.start()

            # Shutdown should stop it
            await shutdown_extensions()

            # Gateway should be stopped (heartbeat task cancelled)
            assert state.local_gateway._heartbeat_task is None

    @pytest.mark.asyncio
    @pytest.mark.skipif(not GASTOWN_AVAILABLE, reason="Gastown not available")
    async def test_shutdown_persists_gastown(self, tmp_path: Path):
        """Test shutdown persists Gastown state."""
        state = init_extensions(tmp_path)

        if state.coordinator:
            # Create some data
            await state.coordinator.create_ledger_entry(
                workspace_id="test",
                entry_type="note",
                title="Test Entry",
            )

            # Shutdown should persist
            await shutdown_extensions()

            # Check persistence file exists
            ledger_path = tmp_path / "gastown" / "ledger.json"
            assert ledger_path.exists()


class TestExtensionFeatureFlags:
    """Tests for extension feature flags."""

    def test_fabric_disabled_by_env(self, tmp_path: Path, monkeypatch):
        """Test Agent Fabric can be disabled via env var."""
        monkeypatch.setenv("ARAGORA_ENABLE_AGENT_FABRIC", "false")

        # Re-import to pick up env var
        import importlib
        import aragora.server.extensions as ext_module

        importlib.reload(ext_module)

        fabric, hooks = ext_module.init_agent_fabric(tmp_path)
        assert fabric is None

    def test_gastown_disabled_by_env(self, tmp_path: Path, monkeypatch):
        """Test Gastown can be disabled via env var."""
        monkeypatch.setenv("ARAGORA_ENABLE_GASTOWN", "false")

        import importlib
        import aragora.server.extensions as ext_module

        importlib.reload(ext_module)

        result = ext_module.init_gastown(tmp_path)
        assert all(r is None for r in result)

    def test_moltbot_disabled_by_env(self, tmp_path: Path, monkeypatch):
        """Test Moltbot can be disabled via env var."""
        monkeypatch.setenv("ARAGORA_ENABLE_MOLTBOT", "false")

        import importlib
        import aragora.server.extensions as ext_module

        importlib.reload(ext_module)

        result = ext_module.init_moltbot(tmp_path)
        assert all(r is None for r in result)

    def test_computer_use_enabled_by_env(self, tmp_path: Path, monkeypatch):
        """Test Computer Use can be enabled via env var."""
        monkeypatch.setenv("ARAGORA_ENABLE_COMPUTER_USE", "true")

        import importlib
        import aragora.server.extensions as ext_module

        importlib.reload(ext_module)

        # Will only be non-None if module is available
        orchestrator, policy = ext_module.init_computer_use()
        # Just verify it doesn't error


class TestExtensionHandlers:
    """Tests for extension HTTP handlers."""

    @pytest.fixture
    def mock_ctx(self):
        """Create mock authorization context."""
        from aragora.rbac.models import AuthorizationContext

        return AuthorizationContext(
            user_id="test-user",
            tenant_id="test-tenant",
            roles=["admin"],  # Admin role has all permissions
            permissions=frozenset(
                [
                    "workspaces:read",
                    "workspaces:write",
                    "convoys:read",
                    "convoys:write",
                    "inbox:read",
                    "inbox:write",
                    "devices:read",
                    "devices:write",
                    "onboarding:read",
                    "onboarding:write",
                    "agents:read",
                    "agents:write",
                    "tasks:read",
                    "tasks:write",
                ]
            ),
        )

    @pytest.mark.asyncio
    async def test_handle_extensions_status_not_initialized(self, mock_ctx):
        """Test status handler when not initialized."""
        import aragora.server.extensions as ext_module

        ext_module._extension_state = None

        from aragora.server.handlers.extensions import handle_extensions_status

        result = await handle_extensions_status(mock_ctx)
        assert result["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_handle_extensions_status(self, mock_ctx, tmp_path: Path):
        """Test status handler returns extension info."""
        init_extensions(tmp_path)

        from aragora.server.handlers.extensions import handle_extensions_status

        result = await handle_extensions_status(mock_ctx)
        assert result["status"] == "ok"
        assert "extensions" in result
        assert "agent_fabric" in result["extensions"]
        assert "gastown" in result["extensions"]
        assert "moltbot" in result["extensions"]

    @pytest.mark.asyncio
    async def test_handle_extensions_stats(self, mock_ctx, tmp_path: Path):
        """Test stats handler returns metrics."""
        init_extensions(tmp_path)

        from aragora.server.handlers.extensions import handle_extensions_stats

        result = await handle_extensions_stats(mock_ctx)
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not GASTOWN_AVAILABLE, reason="Gastown not available")
    async def test_handle_gastown_workspaces_list(self, mock_ctx, tmp_path: Path):
        """Test Gastown workspaces list handler."""
        init_extensions(tmp_path)

        from aragora.server.handlers.extensions import handle_gastown_workspaces_list

        result = await handle_gastown_workspaces_list(mock_ctx)

        if get_extension_state().gastown_enabled:
            assert result["status"] == "ok"
            assert "workspaces" in result
        else:
            assert "error" in result

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MOLTBOT_AVAILABLE, reason="Moltbot not available")
    async def test_handle_moltbot_inbox_messages(self, mock_ctx, tmp_path: Path):
        """Test Moltbot inbox messages handler."""
        init_extensions(tmp_path)

        from aragora.server.handlers.extensions import handle_moltbot_inbox_messages

        result = await handle_moltbot_inbox_messages(mock_ctx)

        if get_extension_state().moltbot_enabled:
            assert result["status"] == "ok"
            assert "messages" in result
        else:
            assert "error" in result

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MOLTBOT_AVAILABLE, reason="Moltbot not available")
    async def test_handle_moltbot_gateway_devices(self, mock_ctx, tmp_path: Path):
        """Test Moltbot gateway devices handler."""
        init_extensions(tmp_path)

        from aragora.server.handlers.extensions import handle_moltbot_gateway_devices

        result = await handle_moltbot_gateway_devices(mock_ctx)

        if get_extension_state().moltbot_enabled:
            assert result["status"] == "ok"
            assert "devices" in result
        else:
            assert "error" in result


class TestExtensionRoutes:
    """Tests for extension route registration."""

    def test_get_extension_routes(self):
        """Test route registration helper."""
        from aragora.server.handlers.extensions import get_extension_routes

        routes = get_extension_routes()

        assert isinstance(routes, dict)
        assert "/api/extensions/status" in routes
        assert "/api/extensions/stats" in routes
        assert "/api/extensions/gastown/workspaces" in routes
        assert "/api/extensions/moltbot/inbox/messages" in routes
        assert "/api/extensions/fabric/agents" in routes

    def test_routes_have_handlers_and_methods(self):
        """Test each route has handler and methods."""
        from aragora.server.handlers.extensions import get_extension_routes

        routes = get_extension_routes()

        for path, (handler, methods) in routes.items():
            assert callable(handler), f"Handler for {path} is not callable"
            assert isinstance(methods, list), f"Methods for {path} is not a list"
            assert len(methods) > 0, f"No methods for {path}"
