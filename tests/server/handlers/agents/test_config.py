"""Tests for agent configuration endpoint handlers."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


@dataclass
class MockAgentConfig:
    """Mock agent configuration."""

    name: str
    model_type: str
    role: str = "proposer"
    priority: str = "normal"
    description: str = ""
    expertise_domains: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "model_type": self.model_type,
            "role": self.role,
            "priority": self.priority,
            "description": self.description,
            "expertise_domains": self.expertise_domains,
            "capabilities": self.capabilities,
            "tags": self.tags,
        }


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    import aragora.server.handlers.agents.config as config_mod

    config_mod._config_loader = None
    yield
    config_mod._config_loader = None


class TestAgentConfigHandlerRoutes:
    """Tests for AgentConfigHandler route configuration."""

    def test_routes_defined(self):
        """Test AgentConfigHandler has expected routes."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        routes = AgentConfigHandler.ROUTES

        assert "/api/v1/agents/configs" in routes
        assert "/api/v1/agents/configs/reload" in routes
        assert "/api/v1/agents/configs/search" in routes

    def test_can_handle_config_routes(self):
        """Test can_handle returns True for config routes."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        assert handler.can_handle("/api/v1/agents/configs") is True
        assert handler.can_handle("/api/v1/agents/configs/claude") is True
        assert handler.can_handle("/api/v1/agents/configs/reload") is True

    def test_can_handle_non_config_routes(self):
        """Test can_handle returns False for non-config routes."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        assert handler.can_handle("/api/agents") is False
        assert handler.can_handle("/api/v1/debates") is False


class TestAgentConfigHandlerAuth:
    """Tests for AgentConfigHandler authentication."""

    @pytest.mark.asyncio
    async def test_requires_authentication(self):
        """Test config endpoints require authentication."""
        from aragora.server.handlers.agents.config import AgentConfigHandler
        from aragora.server.handlers.secure import UnauthorizedError

        handler = AgentConfigHandler({"storage": None})
        mock_http_handler = MagicMock()

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("Not authenticated")
            result = await handler.handle("/api/v1/agents/configs", {}, mock_http_handler)

        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_reload_requires_admin(self):
        """Test reload endpoint requires admin role."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})
        mock_http_handler = MagicMock()
        mock_auth_context = MagicMock()
        mock_auth_context.has_any_role.return_value = False

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = mock_auth_context
            result = await handler.handle("/api/v1/agents/configs/reload", {}, mock_http_handler)

        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_create_requires_permission(self):
        """Test create endpoint requires agents.create permission."""
        from aragora.server.handlers.agents.config import AgentConfigHandler
        from aragora.server.handlers.secure import ForbiddenError

        handler = AgentConfigHandler({"storage": None})
        mock_http_handler = MagicMock()
        mock_auth_context = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.side_effect = ForbiddenError("Permission denied")
            result = await handler.handle(
                "/api/v1/agents/configs/claude/create", {}, mock_http_handler
            )

        assert result.status_code == 403


class TestListConfigs:
    """Tests for _list_configs method."""

    def test_list_configs_success(self):
        """Test list configs returns all configurations."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_config = MockAgentConfig(
            name="claude",
            model_type="anthropic",
            role="proposer",
            description="Claude AI agent",
        )

        mock_loader = MagicMock()
        mock_loader.list_configs.return_value = ["claude"]
        mock_loader.get_config.return_value = mock_config

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._list_configs({})

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert "configs" in body
        assert len(body["configs"]) == 1
        assert body["configs"][0]["name"] == "claude"

    def test_list_configs_loader_not_available(self):
        """Test list configs returns 503 when loader not available."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        with patch("aragora.server.handlers.agents.config.get_config_loader", return_value=None):
            result = handler._list_configs({})

        assert result.status_code == 503

    def test_list_configs_filter_by_priority(self):
        """Test list configs filters by priority."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_config_high = MockAgentConfig(name="claude", model_type="anthropic", priority="high")
        mock_config_normal = MockAgentConfig(name="gemini", model_type="google", priority="normal")

        mock_loader = MagicMock()
        mock_loader.list_configs.return_value = ["claude", "gemini"]

        def get_config(name):
            if name == "claude":
                return mock_config_high
            return mock_config_normal

        mock_loader.get_config.side_effect = get_config

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._list_configs({"priority": "high"})

        body = json.loads(result.body.decode("utf-8"))
        assert len(body["configs"]) == 1
        assert body["configs"][0]["name"] == "claude"

    def test_list_configs_filter_by_role(self):
        """Test list configs filters by role."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_config_proposer = MockAgentConfig(
            name="claude", model_type="anthropic", role="proposer"
        )
        mock_config_critic = MockAgentConfig(name="gemini", model_type="google", role="critic")

        mock_loader = MagicMock()
        mock_loader.list_configs.return_value = ["claude", "gemini"]

        def get_config(name):
            if name == "claude":
                return mock_config_proposer
            return mock_config_critic

        mock_loader.get_config.side_effect = get_config

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._list_configs({"role": "critic"})

        body = json.loads(result.body.decode("utf-8"))
        assert len(body["configs"]) == 1
        assert body["configs"][0]["name"] == "gemini"


class TestGetConfig:
    """Tests for _get_config method."""

    def test_get_config_success(self):
        """Test get config returns configuration."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_config = MockAgentConfig(
            name="claude",
            model_type="anthropic",
            description="Claude AI",
        )

        mock_loader = MagicMock()
        mock_loader.get_config.return_value = mock_config

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._get_config("claude")

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["config"]["name"] == "claude"

    def test_get_config_not_found(self):
        """Test get config returns 404 for missing config."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_loader = MagicMock()
        mock_loader.get_config.return_value = None

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._get_config("nonexistent")

        assert result.status_code == 404


class TestCreateAgentFromConfig:
    """Tests for _create_agent_from_config method."""

    def test_create_agent_success(self):
        """Test create agent from config succeeds."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_config = MockAgentConfig(
            name="claude",
            model_type="anthropic",
            role="proposer",
        )

        mock_agent = MagicMock()
        mock_agent.name = "claude"

        mock_loader = MagicMock()
        mock_loader.get_config.return_value = mock_config
        mock_loader.create_agent.return_value = mock_agent

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._create_agent_from_config("claude")

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["success"] is True
        assert body["agent"]["name"] == "claude"

    def test_create_agent_config_not_found(self):
        """Test create agent returns 404 for missing config."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_loader = MagicMock()
        mock_loader.get_config.return_value = None

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._create_agent_from_config("nonexistent")

        assert result.status_code == 404

    def test_create_agent_failure(self):
        """Test create agent returns 500 on creation error."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_config = MockAgentConfig(name="claude", model_type="anthropic")

        mock_loader = MagicMock()
        mock_loader.get_config.return_value = mock_config
        mock_loader.create_agent.side_effect = RuntimeError("Creation failed")

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._create_agent_from_config("claude")

        assert result.status_code == 500


class TestReloadConfigs:
    """Tests for _reload_configs method."""

    def test_reload_success(self):
        """Test reload configs succeeds."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_loader = MagicMock()
        mock_loader.reload_all.return_value = {"claude": {}, "gemini": {}}

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._reload_configs()

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["success"] is True
        assert body["reloaded"] == 2

    def test_reload_failure(self):
        """Test reload configs returns 500 on error."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_loader = MagicMock()
        mock_loader.reload_all.side_effect = RuntimeError("Reload failed")

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._reload_configs()

        assert result.status_code == 500


class TestSearchConfigs:
    """Tests for _search_configs method."""

    def test_search_by_expertise(self):
        """Test search configs by expertise domain."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_config = MockAgentConfig(
            name="claude",
            model_type="anthropic",
            expertise_domains=["coding"],
        )

        mock_loader = MagicMock()
        mock_loader.get_by_expertise.return_value = [mock_config]
        mock_loader.get_by_capability.return_value = []
        mock_loader.get_by_tag.return_value = []

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._search_configs({"expertise": "coding"})

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert len(body["results"]) == 1

    def test_search_by_capability(self):
        """Test search configs by capability."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_config = MockAgentConfig(
            name="claude",
            model_type="anthropic",
            capabilities=["code_generation"],
        )

        mock_loader = MagicMock()
        mock_loader.get_by_expertise.return_value = []
        mock_loader.get_by_capability.return_value = [mock_config]
        mock_loader.get_by_tag.return_value = []

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._search_configs({"capability": "code_generation"})

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert len(body["results"]) == 1

    def test_search_requires_parameter(self):
        """Test search requires at least one parameter."""
        from aragora.server.handlers.agents.config import AgentConfigHandler

        handler = AgentConfigHandler({"storage": None})

        mock_loader = MagicMock()

        with patch(
            "aragora.server.handlers.agents.config.get_config_loader", return_value=mock_loader
        ):
            result = handler._search_configs({})

        assert result.status_code == 400
