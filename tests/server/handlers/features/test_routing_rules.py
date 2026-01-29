"""Tests for Routing Rules Handler."""

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

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.routing_rules import (
    RoutingRulesHandler,
    _rules_store,
    _get_routing_engine,
)


@pytest.fixture(autouse=True)
def clear_rules():
    """Clear rules store between tests."""
    _rules_store.clear()
    yield


@pytest.fixture
def handler():
    """Create handler instance."""
    return RoutingRulesHandler({})


class TestRoutingRulesHandler:
    """Tests for RoutingRulesHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(RoutingRulesHandler, "ROUTES")
        routes = RoutingRulesHandler.ROUTES
        assert "/api/v1/routing-rules" in routes
        assert "/api/v1/routing-rules/evaluate" in routes
        assert "/api/v1/routing-rules/templates" in routes

    def test_can_handle_routing_rules(self, handler):
        """Test can_handle for routing rules routes."""
        assert handler.can_handle("/api/v1/routing-rules/") is True
        assert handler.can_handle("/api/v1/routing-rules/evaluate") is True

    def test_resource_type(self):
        """Test resource type for audit logging."""
        assert RoutingRulesHandler.RESOURCE_TYPE == "policy"


class TestRoutingRulesAuthentication:
    """Tests for routing rules authentication."""

    @pytest.mark.asyncio
    async def test_handle_requires_authentication(self):
        """Test handle_request requires authentication."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/v1/routing-rules"
        mock_request.args = {}

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            from aragora.server.handlers.secure import UnauthorizedError

            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            result = await handler.handle_request(mock_request)
            assert result["code"] == 401

    @pytest.mark.asyncio
    async def test_handle_checks_permission(self):
        """Test handle checks policies.read permission."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/v1/routing-rules"
        mock_request.args = {}

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            from aragora.server.handlers.secure import ForbiddenError

            mock_auth.return_value = MagicMock()
            mock_check.side_effect = ForbiddenError("Permission denied")

            result = await handler.handle_request(mock_request)
            assert result["code"] == 403


class TestListRules:
    """Tests for listing routing rules."""

    @pytest.mark.asyncio
    async def test_list_rules_empty(self):
        """Test listing rules when none exist."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.args = {}

        result = await handler._list_rules(mock_request)
        assert result["status"] == "success"
        assert result["rules"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_list_rules_with_data(self):
        """Test listing rules with existing rules."""
        _rules_store["rule1"] = {
            "id": "rule1",
            "name": "Test Rule",
            "enabled": True,
            "priority": 10,
            "tags": [],
            "conditions": [],
            "actions": [],
        }

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.args = {}

        with patch(
            "aragora.server.handlers.features.routing_rules.RoutingRule.from_dict"
        ) as mock_from_dict:
            mock_rule = MagicMock()
            mock_rule.enabled = True
            mock_rule.tags = []
            mock_rule.to_dict.return_value = {"id": "rule1", "name": "Test Rule", "priority": 10}
            mock_from_dict.return_value = mock_rule

            result = await handler._list_rules(mock_request)
            assert result["status"] == "success"
            assert result["count"] == 1


class TestCreateRule:
    """Tests for creating routing rules."""

    @pytest.mark.asyncio
    async def test_create_rule_missing_body(self):
        """Test create rule requires body."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = None

            result = await handler._create_rule(mock_request)
            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_create_rule_success(self):
        """Test successful rule creation."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        mock_rule = MagicMock()
        mock_rule.id = "rule123"
        mock_rule.to_dict.return_value = {"id": "rule123", "name": "Test Rule"}

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch("aragora.server.handlers.features.routing_rules.RoutingRule") as MockRule,
            patch("aragora.server.handlers.features.routing_rules.Condition") as MockCondition,
            patch("aragora.server.handlers.features.routing_rules.Action") as MockAction,
        ):
            mock_body.return_value = {
                "name": "Test Rule",
                "conditions": [],
                "actions": [],
            }
            MockRule.create.return_value = mock_rule

            result = await handler._create_rule(mock_request)
            assert result["status"] == "success"


class TestGetRule:
    """Tests for getting specific rules."""

    @pytest.mark.asyncio
    async def test_get_rule_not_found(self):
        """Test getting non-existent rule."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        result = await handler._get_rule(mock_request, "invalid-rule")
        assert result["status"] == "error"
        assert result["code"] == 404

    @pytest.mark.asyncio
    async def test_get_rule_success(self):
        """Test getting existing rule."""
        _rules_store["rule1"] = {"id": "rule1", "name": "Test Rule"}

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        result = await handler._get_rule(mock_request, "rule1")
        assert result["status"] == "success"
        assert result["rule"]["id"] == "rule1"


class TestUpdateRule:
    """Tests for updating routing rules."""

    @pytest.mark.asyncio
    async def test_update_rule_not_found(self):
        """Test updating non-existent rule."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        result = await handler._update_rule(mock_request, "invalid-rule")
        assert result["status"] == "error"
        assert result["code"] == 404

    @pytest.mark.asyncio
    async def test_update_rule_missing_body(self):
        """Test update rule requires body."""
        _rules_store["rule1"] = {"id": "rule1", "name": "Test Rule"}

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = None

            result = await handler._update_rule(mock_request, "rule1")
            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_update_rule_success(self):
        """Test successful rule update."""
        _rules_store["rule1"] = {
            "id": "rule1",
            "name": "Test Rule",
            "conditions": [],
            "actions": [],
        }

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"name": "Updated Rule"}

            result = await handler._update_rule(mock_request, "rule1")
            assert result["status"] == "success"
            assert result["rule"]["name"] == "Updated Rule"


class TestDeleteRule:
    """Tests for deleting routing rules."""

    @pytest.mark.asyncio
    async def test_delete_rule_not_found(self):
        """Test deleting non-existent rule."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        result = await handler._delete_rule(mock_request, "invalid-rule")
        assert result["status"] == "error"
        assert result["code"] == 404

    @pytest.mark.asyncio
    async def test_delete_rule_success(self):
        """Test successful rule deletion."""
        _rules_store["rule1"] = {"id": "rule1", "name": "Test Rule"}

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        result = await handler._delete_rule(mock_request, "rule1")
        assert result["status"] == "success"
        assert "rule1" not in _rules_store


class TestToggleRule:
    """Tests for toggling rule enabled state."""

    @pytest.mark.asyncio
    async def test_toggle_rule_not_found(self):
        """Test toggling non-existent rule."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        result = await handler._toggle_rule(mock_request, "invalid-rule")
        assert result["status"] == "error"
        assert result["code"] == 404

    @pytest.mark.asyncio
    async def test_toggle_rule_success(self):
        """Test successful rule toggle."""
        _rules_store["rule1"] = {"id": "rule1", "name": "Test Rule", "enabled": True}

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = None

            result = await handler._toggle_rule(mock_request, "rule1")
            assert result["status"] == "success"
            assert result["rule"]["enabled"] is False


class TestEvaluateRules:
    """Tests for evaluating routing rules."""

    @pytest.mark.asyncio
    async def test_evaluate_missing_body(self):
        """Test evaluate requires body."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = None

            result = await handler._evaluate_rules(mock_request)
            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_evaluate_missing_context(self):
        """Test evaluate requires context."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {}

            result = await handler._evaluate_rules(mock_request)
            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        """Test successful rule evaluation."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        mock_result = MagicMock()
        mock_result.rule = MagicMock(id="rule1", name="Test Rule")
        mock_result.matched = True
        mock_result.actions = []
        mock_result.execution_time_ms = 1.5

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch(
                "aragora.server.handlers.features.routing_rules._get_routing_engine"
            ) as mock_engine,
        ):
            mock_body.return_value = {"context": {"confidence": 0.8}}
            mock_engine.return_value = MagicMock(evaluate=lambda ctx, **kw: [mock_result])

            result = await handler._evaluate_rules(mock_request)
            assert result["status"] == "success"
            assert result["rules_evaluated"] == 1


class TestGetTemplates:
    """Tests for getting rule templates."""

    @pytest.mark.asyncio
    async def test_get_templates(self):
        """Test getting rule templates."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch(
            "aragora.server.handlers.features.routing_rules.RULE_TEMPLATES",
            {"template1": MagicMock(to_dict=lambda: {"name": "Template 1"})},
        ):
            result = await handler._get_templates(mock_request)
            assert result["status"] == "success"
            assert result["count"] == 1


class TestUtilityMethods:
    """Tests for utility methods."""

    @pytest.mark.asyncio
    async def test_get_json_body_callable(self):
        """Test JSON body extraction when json is callable."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={"test": "data"})

        result = await handler._get_json_body(mock_request)
        assert result == {"test": "data"}

    @pytest.mark.asyncio
    async def test_get_json_body_from_body(self):
        """Test JSON body extraction from body attribute."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock(spec=["body"])
        mock_request.body = b'{"test": "data"}'

        result = await handler._get_json_body(mock_request)
        assert result == {"test": "data"}

    def test_method_not_allowed(self):
        """Test method not allowed response."""
        handler = RoutingRulesHandler({})

        result = handler._method_not_allowed("PATCH", "/api/v1/routing-rules")
        assert result["status"] == "error"
        assert result["code"] == 405
