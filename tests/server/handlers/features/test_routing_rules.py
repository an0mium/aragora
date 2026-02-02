"""Tests for Routing Rules Handler.

Comprehensive test suite for the RoutingRulesHandler, covering:
- Authentication and authorization (RBAC)
- CRUD operations (Create, Read, Update, Delete)
- Rule evaluation
- Input validation
- Error handling
- Audit logging
- Rate limiting
"""

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
    MAX_ACTIONS,
    MAX_CONDITIONS,
    MAX_DESCRIPTION_LENGTH,
    MAX_RULE_NAME_LENGTH,
    MAX_TAG_LENGTH,
    MAX_TAGS,
    RoutingRulesHandler,
    VALID_MATCH_MODES,
    _get_routing_engine,
    _rules_store,
    _validate_rule_data,
    _validate_rule_id,
)


@pytest.fixture(autouse=True)
def clear_rules():
    """Clear rules store between tests."""
    _rules_store.clear()
    yield
    _rules_store.clear()


@pytest.fixture
def handler():
    """Create handler instance."""
    return RoutingRulesHandler({})


@pytest.fixture
def mock_auth_context():
    """Create a mock authorization context."""
    ctx = MagicMock()
    ctx.user_id = "test-user-123"
    ctx.workspace_id = "test-workspace"
    ctx.permissions = ["policies.read", "policies.create", "policies.update", "policies.delete"]
    return ctx


@pytest.fixture
def sample_rule_data():
    """Sample rule data for testing."""
    return {
        "id": "rule1",
        "name": "Test Rule",
        "description": "A test rule",
        "enabled": True,
        "priority": 10,
        "tags": ["test", "sample"],
        "conditions": [{"field": "confidence", "operator": "lt", "value": 0.7}],
        "actions": [{"type": "notify", "target": "admin"}],
        "match_mode": "all",
        "stop_processing": False,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


# =============================================================================
# Handler Basic Tests
# =============================================================================


class TestRoutingRulesHandler:
    """Tests for RoutingRulesHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None
        assert handler.ctx == {}

    def test_handler_creation_with_context(self):
        """Test creating handler with custom context."""
        ctx = {"custom": "data"}
        handler = RoutingRulesHandler(ctx)
        assert handler.ctx == ctx

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(RoutingRulesHandler, "ROUTES")
        routes = RoutingRulesHandler.ROUTES
        assert "/api/v1/routing-rules" in routes
        assert "/api/v1/routing-rules/evaluate" in routes
        assert "/api/v1/routing-rules/templates" in routes
        assert "/api/v1/routing-rules/{rule_id}" in routes
        assert "/api/v1/routing-rules/{rule_id}/toggle" in routes

    def test_can_handle_routing_rules(self, handler):
        """Test can_handle for routing rules routes."""
        assert handler.can_handle("/api/v1/routing-rules") is True
        assert handler.can_handle("/api/v1/routing-rules/") is True
        assert handler.can_handle("/api/v1/routing-rules/evaluate") is True
        assert handler.can_handle("/api/v1/routing-rules/templates") is True
        assert handler.can_handle("/api/v1/routing-rules/rule123") is True
        assert handler.can_handle("/api/v1/routing-rules/rule123/toggle") is True

    def test_can_handle_other_routes(self, handler):
        """Test can_handle returns False for other routes."""
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/agents") is False

    def test_resource_type(self):
        """Test resource type for audit logging."""
        assert RoutingRulesHandler.RESOURCE_TYPE == "policy"


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for input validation functions."""

    def test_validate_rule_id_valid(self):
        """Test valid rule IDs pass validation."""
        valid_ids = ["rule1", "my-rule", "rule_123", "ABC-123_xyz"]
        for rule_id in valid_ids:
            is_valid, error = _validate_rule_id(rule_id)
            assert is_valid is True, f"Expected {rule_id} to be valid"
            assert error is None

    def test_validate_rule_id_empty(self):
        """Test empty rule ID fails validation."""
        is_valid, error = _validate_rule_id("")
        assert is_valid is False
        assert "required" in error.lower()

    def test_validate_rule_id_invalid_format(self):
        """Test invalid rule IDs fail validation."""
        invalid_ids = ["rule with space", "rule@123", "../path", "a" * 100]
        for rule_id in invalid_ids:
            is_valid, error = _validate_rule_id(rule_id)
            assert is_valid is False, f"Expected {rule_id} to be invalid"
            assert error is not None

    def test_validate_rule_data_valid(self):
        """Test valid rule data passes validation."""
        data = {
            "name": "Test Rule",
            "description": "A test rule",
            "conditions": [{"field": "confidence", "operator": "lt", "value": 0.7}],
            "actions": [{"type": "notify", "target": "admin"}],
            "tags": ["test"],
            "match_mode": "all",
            "priority": 100,
        }
        is_valid, error = _validate_rule_data(data)
        assert is_valid is True
        assert error is None

    def test_validate_rule_data_name_too_long(self):
        """Test rule name exceeding max length fails."""
        data = {"name": "a" * (MAX_RULE_NAME_LENGTH + 1)}
        is_valid, error = _validate_rule_data(data)
        assert is_valid is False
        assert "name" in error.lower()

    def test_validate_rule_data_description_too_long(self):
        """Test description exceeding max length fails."""
        data = {"description": "a" * (MAX_DESCRIPTION_LENGTH + 1)}
        is_valid, error = _validate_rule_data(data)
        assert is_valid is False
        assert "description" in error.lower()

    def test_validate_rule_data_too_many_conditions(self):
        """Test too many conditions fails validation."""
        data = {
            "conditions": [
                {"field": f"f{i}", "operator": "eq", "value": i} for i in range(MAX_CONDITIONS + 1)
            ]
        }
        is_valid, error = _validate_rule_data(data)
        assert is_valid is False
        assert "conditions" in error.lower()

    def test_validate_rule_data_too_many_actions(self):
        """Test too many actions fails validation."""
        data = {"actions": [{"type": "log"} for _ in range(MAX_ACTIONS + 1)]}
        is_valid, error = _validate_rule_data(data)
        assert is_valid is False
        assert "actions" in error.lower()

    def test_validate_rule_data_too_many_tags(self):
        """Test too many tags fails validation."""
        data = {"tags": [f"tag{i}" for i in range(MAX_TAGS + 1)]}
        is_valid, error = _validate_rule_data(data)
        assert is_valid is False
        assert "tags" in error.lower()

    def test_validate_rule_data_tag_too_long(self):
        """Test tag exceeding max length fails."""
        data = {"tags": ["a" * (MAX_TAG_LENGTH + 1)]}
        is_valid, error = _validate_rule_data(data)
        assert is_valid is False
        assert "tag" in error.lower()

    def test_validate_rule_data_non_string_tag(self):
        """Test non-string tag fails validation."""
        data = {"tags": [123]}
        is_valid, error = _validate_rule_data(data)
        assert is_valid is False
        assert "string" in error.lower()

    def test_validate_rule_data_invalid_match_mode(self):
        """Test invalid match_mode fails validation."""
        data = {"match_mode": "invalid"}
        is_valid, error = _validate_rule_data(data)
        assert is_valid is False
        assert "match_mode" in error.lower()

    def test_validate_rule_data_valid_match_modes(self):
        """Test valid match_modes pass validation."""
        for mode in VALID_MATCH_MODES:
            data = {"match_mode": mode}
            is_valid, error = _validate_rule_data(data)
            assert is_valid is True, f"Expected {mode} to be valid"

    def test_validate_rule_data_non_integer_priority(self):
        """Test non-integer priority fails validation."""
        data = {"priority": "high"}
        is_valid, error = _validate_rule_data(data)
        assert is_valid is False
        assert "priority" in error.lower()

    def test_validate_rule_data_priority_out_of_range(self):
        """Test priority out of range fails validation."""
        data = {"priority": 2000}
        is_valid, error = _validate_rule_data(data)
        assert is_valid is False
        assert "priority" in error.lower()

        data = {"priority": -2000}
        is_valid, error = _validate_rule_data(data)
        assert is_valid is False


# =============================================================================
# Authentication Tests
# =============================================================================


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
            assert "Authentication" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_checks_read_permission(self):
        """Test handle checks policies.read permission for GET."""
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
            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_checks_create_permission(self, mock_auth_context):
        """Test handle checks policies.create permission for POST."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.path = "/api/v1/routing-rules"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            from aragora.server.handlers.secure import ForbiddenError

            mock_auth.return_value = mock_auth_context
            mock_check.side_effect = ForbiddenError("Permission denied")

            result = await handler.handle_request(mock_request)
            assert result["code"] == 403
            mock_check.assert_called_with(mock_auth_context, "policies.create")

    @pytest.mark.asyncio
    async def test_handle_checks_update_permission(self, mock_auth_context):
        """Test handle checks policies.update permission for PUT."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "PUT"
        mock_request.path = "/api/v1/routing-rules/rule1"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            from aragora.server.handlers.secure import ForbiddenError

            mock_auth.return_value = mock_auth_context
            mock_check.side_effect = ForbiddenError("Permission denied")

            result = await handler.handle_request(mock_request)
            assert result["code"] == 403
            mock_check.assert_called_with(mock_auth_context, "policies.update")

    @pytest.mark.asyncio
    async def test_handle_checks_delete_permission(self, mock_auth_context):
        """Test handle checks policies.delete permission for DELETE."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "DELETE"
        mock_request.path = "/api/v1/routing-rules/rule1"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            from aragora.server.handlers.secure import ForbiddenError

            mock_auth.return_value = mock_auth_context
            mock_check.side_effect = ForbiddenError("Permission denied")

            result = await handler.handle_request(mock_request)
            assert result["code"] == 403
            mock_check.assert_called_with(mock_auth_context, "policies.delete")

    @pytest.mark.asyncio
    async def test_evaluate_requires_read_permission(self, mock_auth_context):
        """Test evaluate endpoint uses read permission."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.path = "/api/v1/routing-rules/evaluate"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            from aragora.server.handlers.secure import ForbiddenError

            mock_auth.return_value = mock_auth_context
            mock_check.side_effect = ForbiddenError("Permission denied")

            result = await handler.handle_request(mock_request)
            assert result["code"] == 403
            mock_check.assert_called_with(mock_auth_context, "policies.read")


# =============================================================================
# List Rules Tests
# =============================================================================


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
    async def test_list_rules_with_data(self, sample_rule_data):
        """Test listing rules with existing rules."""
        _rules_store["rule1"] = sample_rule_data

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.args = {}

        with patch("aragora.core.routing_rules.RoutingRule.from_dict") as mock_from_dict:
            mock_rule = MagicMock()
            mock_rule.enabled = True
            mock_rule.tags = ["test"]
            mock_rule.to_dict.return_value = sample_rule_data
            mock_from_dict.return_value = mock_rule

            result = await handler._list_rules(mock_request)
            assert result["status"] == "success"
            assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_list_rules_filter_by_enabled(self, sample_rule_data):
        """Test listing rules filters by enabled status."""
        _rules_store["rule1"] = sample_rule_data
        disabled_rule = dict(sample_rule_data)
        disabled_rule["id"] = "rule2"
        disabled_rule["enabled"] = False
        _rules_store["rule2"] = disabled_rule

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.args = {"enabled_only": "true"}

        with patch("aragora.core.routing_rules.RoutingRule.from_dict") as mock_from_dict:

            def create_rule(data):
                rule = MagicMock()
                rule.enabled = data.get("enabled", True)
                rule.tags = data.get("tags", [])
                rule.to_dict.return_value = data
                return rule

            mock_from_dict.side_effect = create_rule

            result = await handler._list_rules(mock_request)
            assert result["status"] == "success"
            assert result["count"] == 1
            assert result["rules"][0]["id"] == "rule1"

    @pytest.mark.asyncio
    async def test_list_rules_filter_by_tags(self, sample_rule_data):
        """Test listing rules filters by tags."""
        sample_rule_data["tags"] = ["security"]
        _rules_store["rule1"] = sample_rule_data

        other_rule = dict(sample_rule_data)
        other_rule["id"] = "rule2"
        other_rule["tags"] = ["other"]
        _rules_store["rule2"] = other_rule

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.args = {"tags": "security"}

        with patch("aragora.core.routing_rules.RoutingRule.from_dict") as mock_from_dict:

            def create_rule(data):
                rule = MagicMock()
                rule.enabled = True
                rule.tags = data.get("tags", [])
                rule.to_dict.return_value = data
                return rule

            mock_from_dict.side_effect = create_rule

            result = await handler._list_rules(mock_request)
            assert result["status"] == "success"
            assert result["count"] == 1
            assert result["rules"][0]["tags"] == ["security"]

    @pytest.mark.asyncio
    async def test_list_rules_sorted_by_priority(self):
        """Test rules are sorted by priority descending."""
        _rules_store["rule1"] = {"id": "rule1", "priority": 10, "enabled": True, "tags": []}
        _rules_store["rule2"] = {"id": "rule2", "priority": 100, "enabled": True, "tags": []}
        _rules_store["rule3"] = {"id": "rule3", "priority": 50, "enabled": True, "tags": []}

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.args = {}

        with patch("aragora.core.routing_rules.RoutingRule.from_dict") as mock_from_dict:

            def create_rule(data):
                rule = MagicMock()
                rule.enabled = True
                rule.tags = []
                rule.to_dict.return_value = data
                return rule

            mock_from_dict.side_effect = create_rule

            result = await handler._list_rules(mock_request)
            assert result["status"] == "success"
            priorities = [r["priority"] for r in result["rules"]]
            assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_list_rules_handles_invalid_rule_data(self):
        """Test list rules skips invalid rule data gracefully."""
        _rules_store["rule1"] = {"invalid": "data"}

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.args = {}

        with patch("aragora.core.routing_rules.RoutingRule.from_dict") as mock_from_dict:
            mock_from_dict.side_effect = ValueError("Invalid rule data")

            result = await handler._list_rules(mock_request)
            assert result["status"] == "success"
            assert result["count"] == 0


# =============================================================================
# Create Rule Tests
# =============================================================================


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
            assert result["code"] == 400

    @pytest.mark.asyncio
    async def test_create_rule_validation_failure(self):
        """Test create rule with invalid data returns error."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"name": "a" * 300}  # Too long

            result = await handler._create_rule(mock_request)
            assert result["status"] == "error"
            assert result["code"] == 400

    @pytest.mark.asyncio
    async def test_create_rule_invalid_condition(self):
        """Test create rule with invalid condition returns error."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch("aragora.core.routing_rules.Condition.from_dict") as mock_condition,
        ):
            mock_body.return_value = {
                "name": "Test Rule",
                "conditions": [{"invalid": "data"}],
                "actions": [],
            }
            mock_condition.side_effect = KeyError("field")

            result = await handler._create_rule(mock_request)
            assert result["status"] == "error"
            assert result["code"] == 400
            assert "condition" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_rule_invalid_action(self):
        """Test create rule with invalid action returns error."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch("aragora.core.routing_rules.Condition.from_dict") as mock_condition,
            patch("aragora.core.routing_rules.Action.from_dict") as mock_action,
        ):
            mock_body.return_value = {
                "name": "Test Rule",
                "conditions": [],
                "actions": [{"invalid": "data"}],
            }
            mock_condition.side_effect = lambda x: MagicMock()
            mock_action.side_effect = ValueError("Invalid action type")

            result = await handler._create_rule(mock_request)
            assert result["status"] == "error"
            assert result["code"] == 400
            assert "action" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_rule_success(self):
        """Test successful rule creation."""
        handler = RoutingRulesHandler({})
        handler._auth_context = MagicMock(user_id="test-user")
        mock_request = MagicMock()

        mock_rule = MagicMock()
        mock_rule.id = "rule123"
        mock_rule.name = "Test Rule"
        mock_rule.to_dict.return_value = {"id": "rule123", "name": "Test Rule"}

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch("aragora.core.routing_rules.RoutingRule") as MockRule,
            patch("aragora.core.routing_rules.Condition") as MockCondition,
            patch("aragora.core.routing_rules.Action") as MockAction,
            patch.object(handler, "_audit_rule_change") as mock_audit,
        ):
            mock_body.return_value = {
                "name": "Test Rule",
                "conditions": [],
                "actions": [],
            }
            MockRule.create.return_value = mock_rule

            result = await handler._create_rule(mock_request)
            assert result["status"] == "success"
            assert result["rule"]["id"] == "rule123"
            mock_audit.assert_called_once_with("create", "rule123", "Test Rule")

    @pytest.mark.asyncio
    async def test_create_rule_stores_in_rules_store(self):
        """Test created rule is stored in _rules_store."""
        handler = RoutingRulesHandler({})
        handler._auth_context = MagicMock(user_id="test-user")
        mock_request = MagicMock()

        mock_rule = MagicMock()
        mock_rule.id = "new-rule"
        mock_rule.name = "New Rule"
        mock_rule.to_dict.return_value = {"id": "new-rule", "name": "New Rule"}

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch("aragora.core.routing_rules.RoutingRule") as MockRule,
            patch("aragora.core.routing_rules.Condition"),
            patch("aragora.core.routing_rules.Action"),
            patch.object(handler, "_audit_rule_change"),
        ):
            mock_body.return_value = {"name": "New Rule", "conditions": [], "actions": []}
            MockRule.create.return_value = mock_rule

            await handler._create_rule(mock_request)
            assert "new-rule" in _rules_store


# =============================================================================
# Get Rule Tests
# =============================================================================


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
    async def test_get_rule_success(self, sample_rule_data):
        """Test getting existing rule."""
        _rules_store["rule1"] = sample_rule_data

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        result = await handler._get_rule(mock_request, "rule1")
        assert result["status"] == "success"
        assert result["rule"]["id"] == "rule1"
        assert result["rule"]["name"] == "Test Rule"


# =============================================================================
# Update Rule Tests
# =============================================================================


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
    async def test_update_rule_missing_body(self, sample_rule_data):
        """Test update rule requires body."""
        _rules_store["rule1"] = sample_rule_data

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = None

            result = await handler._update_rule(mock_request, "rule1")
            assert result["status"] == "error"
            assert result["code"] == 400

    @pytest.mark.asyncio
    async def test_update_rule_validation_failure(self, sample_rule_data):
        """Test update rule with invalid data returns error."""
        _rules_store["rule1"] = sample_rule_data

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"priority": "invalid"}

            result = await handler._update_rule(mock_request, "rule1")
            assert result["status"] == "error"
            assert result["code"] == 400

    @pytest.mark.asyncio
    async def test_update_rule_invalid_condition(self, sample_rule_data):
        """Test update rule with invalid condition returns error."""
        _rules_store["rule1"] = sample_rule_data

        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch("aragora.core.routing_rules.Condition.from_dict") as mock_condition,
        ):
            mock_body.return_value = {"conditions": [{"invalid": "data"}]}
            mock_condition.side_effect = KeyError("field")

            result = await handler._update_rule(mock_request, "rule1")
            assert result["status"] == "error"
            assert result["code"] == 400

    @pytest.mark.asyncio
    async def test_update_rule_success(self, sample_rule_data):
        """Test successful rule update."""
        _rules_store["rule1"] = sample_rule_data

        handler = RoutingRulesHandler({})
        handler._auth_context = MagicMock(user_id="test-user")
        mock_request = MagicMock()

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch.object(handler, "_audit_rule_change") as mock_audit,
        ):
            mock_body.return_value = {"name": "Updated Rule"}

            result = await handler._update_rule(mock_request, "rule1")
            assert result["status"] == "success"
            assert result["rule"]["name"] == "Updated Rule"
            assert "updated_at" in result["rule"]
            mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_rule_partial_update(self, sample_rule_data):
        """Test partial update only changes specified fields."""
        _rules_store["rule1"] = sample_rule_data
        original_description = sample_rule_data["description"]

        handler = RoutingRulesHandler({})
        handler._auth_context = MagicMock(user_id="test-user")
        mock_request = MagicMock()

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch.object(handler, "_audit_rule_change"),
        ):
            mock_body.return_value = {"name": "Updated Name Only"}

            result = await handler._update_rule(mock_request, "rule1")
            assert result["status"] == "success"
            assert result["rule"]["name"] == "Updated Name Only"
            assert result["rule"]["description"] == original_description


# =============================================================================
# Delete Rule Tests
# =============================================================================


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
    async def test_delete_rule_success(self, sample_rule_data):
        """Test successful rule deletion."""
        _rules_store["rule1"] = sample_rule_data

        handler = RoutingRulesHandler({})
        handler._auth_context = MagicMock(user_id="test-user")
        mock_request = MagicMock()

        with patch.object(handler, "_audit_rule_change") as mock_audit:
            result = await handler._delete_rule(mock_request, "rule1")
            assert result["status"] == "success"
            assert "rule1" not in _rules_store
            mock_audit.assert_called_once_with("delete", "rule1", "Test Rule")


# =============================================================================
# Toggle Rule Tests
# =============================================================================


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
    async def test_toggle_rule_without_body(self, sample_rule_data):
        """Test toggle rule without body inverts enabled state."""
        _rules_store["rule1"] = sample_rule_data

        handler = RoutingRulesHandler({})
        handler._auth_context = MagicMock(user_id="test-user")
        mock_request = MagicMock()

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch.object(handler, "_audit_rule_change") as mock_audit,
        ):
            mock_body.return_value = None

            result = await handler._toggle_rule(mock_request, "rule1")
            assert result["status"] == "success"
            assert result["rule"]["enabled"] is False  # Was True, now False
            mock_audit.assert_called_once_with("disable", "rule1", "Test Rule")

    @pytest.mark.asyncio
    async def test_toggle_rule_with_explicit_value(self, sample_rule_data):
        """Test toggle rule with explicit enabled value."""
        sample_rule_data["enabled"] = True
        _rules_store["rule1"] = sample_rule_data

        handler = RoutingRulesHandler({})
        handler._auth_context = MagicMock(user_id="test-user")
        mock_request = MagicMock()

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch.object(handler, "_audit_rule_change") as mock_audit,
        ):
            mock_body.return_value = {"enabled": False}

            result = await handler._toggle_rule(mock_request, "rule1")
            assert result["status"] == "success"
            assert result["rule"]["enabled"] is False
            mock_audit.assert_called_once_with("disable", "rule1", "Test Rule")

    @pytest.mark.asyncio
    async def test_toggle_rule_enables_disabled_rule(self, sample_rule_data):
        """Test toggle rule enables a disabled rule."""
        sample_rule_data["enabled"] = False
        _rules_store["rule1"] = sample_rule_data

        handler = RoutingRulesHandler({})
        handler._auth_context = MagicMock(user_id="test-user")
        mock_request = MagicMock()

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch.object(handler, "_audit_rule_change") as mock_audit,
        ):
            mock_body.return_value = None

            result = await handler._toggle_rule(mock_request, "rule1")
            assert result["status"] == "success"
            assert result["rule"]["enabled"] is True
            mock_audit.assert_called_once_with("enable", "rule1", "Test Rule")


# =============================================================================
# Evaluate Rules Tests
# =============================================================================


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
            assert result["code"] == 400

    @pytest.mark.asyncio
    async def test_evaluate_missing_context(self):
        """Test evaluate requires context."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {}

            result = await handler._evaluate_rules(mock_request)
            assert result["status"] == "error"
            assert result["code"] == 400

    @pytest.mark.asyncio
    async def test_evaluate_context_not_dict(self):
        """Test evaluate requires context to be a dict."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"context": "not a dict"}

            result = await handler._evaluate_rules(mock_request)
            assert result["status"] == "error"
            assert result["code"] == 400
            assert "object" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        """Test successful rule evaluation."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        mock_action = MagicMock()
        mock_action.to_dict.return_value = {"type": "notify", "target": "admin"}

        mock_result = MagicMock()
        mock_result.rule = MagicMock(id="rule1", name="Test Rule")
        mock_result.matched = True
        mock_result.actions = [mock_action]
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
            assert result["rules_matched"] == 1
            assert len(result["matching_actions"]) == 1

    @pytest.mark.asyncio
    async def test_evaluate_no_matches(self):
        """Test evaluate when no rules match."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        mock_result = MagicMock()
        mock_result.rule = MagicMock(id="rule1", name="Test Rule")
        mock_result.matched = False
        mock_result.actions = []
        mock_result.execution_time_ms = 0.5

        with (
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
            patch(
                "aragora.server.handlers.features.routing_rules._get_routing_engine"
            ) as mock_engine,
        ):
            mock_body.return_value = {"context": {"confidence": 0.95}}
            mock_engine.return_value = MagicMock(evaluate=lambda ctx, **kw: [mock_result])

            result = await handler._evaluate_rules(mock_request)
            assert result["status"] == "success"
            assert result["rules_matched"] == 0
            assert result["matching_actions"] == []


# =============================================================================
# Get Templates Tests
# =============================================================================


class TestGetTemplates:
    """Tests for getting rule templates."""

    @pytest.mark.asyncio
    async def test_get_templates(self):
        """Test getting rule templates."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        mock_template = MagicMock()
        mock_template.to_dict.return_value = {"name": "Template 1", "description": "Test template"}

        with patch(
            "aragora.core.routing_rules.RULE_TEMPLATES",
            {"template1": mock_template},
        ):
            result = await handler._get_templates(mock_request)
            assert result["status"] == "success"
            assert result["count"] == 1
            assert result["templates"][0]["template_key"] == "template1"

    @pytest.mark.asyncio
    async def test_get_templates_import_error(self):
        """Test templates endpoint handles ImportError."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        # Mock the import to raise ImportError
        with patch.dict(
            "sys.modules",
            {"aragora.core.routing_rules": None},
        ):
            # Force reimport to trigger ImportError
            result = await handler._get_templates(mock_request)
            # When import fails, it should return an error response
            assert result["status"] == "error"
            assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_get_templates_multiple(self):
        """Test getting multiple templates."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()

        templates = {
            "low_confidence": MagicMock(to_dict=lambda: {"name": "Low Confidence"}),
            "security_route": MagicMock(to_dict=lambda: {"name": "Security Route"}),
            "high_priority": MagicMock(to_dict=lambda: {"name": "High Priority"}),
        }

        with patch("aragora.core.routing_rules.RULE_TEMPLATES", templates):
            result = await handler._get_templates(mock_request)
            assert result["status"] == "success"
            assert result["count"] == 3


# =============================================================================
# Utility Methods Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    @pytest.mark.asyncio
    async def test_get_json_body_callable(self):
        """Test JSON body extraction when json is callable."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={"test": "data"})

        with patch(
            "aragora.server.handlers.features.routing_rules.parse_json_body",
            new_callable=AsyncMock,
            return_value=({"test": "data"}, None),
        ):
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

    @pytest.mark.asyncio
    async def test_get_json_body_from_body_string(self):
        """Test JSON body extraction from string body."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock(spec=["body"])
        mock_request.body = '{"test": "data"}'

        result = await handler._get_json_body(mock_request)
        assert result == {"test": "data"}

    @pytest.mark.asyncio
    async def test_get_json_body_invalid_json(self):
        """Test JSON body extraction with invalid JSON."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock(spec=["body"])
        mock_request.body = b"not valid json"

        result = await handler._get_json_body(mock_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_json_body_empty(self):
        """Test JSON body extraction with empty body."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock(spec=["body"])
        mock_request.body = b""

        result = await handler._get_json_body(mock_request)
        assert result is None

    def test_method_not_allowed(self):
        """Test method not allowed response."""
        handler = RoutingRulesHandler({})

        result = handler._method_not_allowed("PATCH", "/api/v1/routing-rules")
        assert result["status"] == "error"
        assert result["code"] == 405
        assert "PATCH" in result["error"]

    def test_audit_rule_change_success(self):
        """Test audit logging for rule changes."""
        handler = RoutingRulesHandler({})
        handler._auth_context = MagicMock(user_id="test-user")

        with patch("aragora.server.handlers.features.routing_rules.audit_data") as mock_audit:
            handler._audit_rule_change("create", "rule1", "Test Rule")
            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["event_type"] == "routing_rule_create"
            assert call_kwargs["resource_id"] == "rule1"
            assert call_kwargs["actor_id"] == "test-user"

    def test_audit_rule_change_no_auth_context(self):
        """Test audit logging without auth context."""
        handler = RoutingRulesHandler({})

        with patch("aragora.server.handlers.features.routing_rules.audit_data") as mock_audit:
            handler._audit_rule_change("delete", "rule1", "Test Rule")
            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["actor_id"] == "unknown"

    def test_audit_rule_change_exception_handled(self):
        """Test audit logging handles exceptions gracefully."""
        handler = RoutingRulesHandler({})

        with patch(
            "aragora.server.handlers.features.routing_rules.audit_data",
            side_effect=Exception("Audit failed"),
        ):
            # Should not raise
            handler._audit_rule_change("update", "rule1", "Test Rule")


# =============================================================================
# Request Routing Tests
# =============================================================================


class TestRequestRouting:
    """Tests for request routing in handle_request."""

    @pytest.mark.asyncio
    async def test_route_to_list_rules(self, mock_auth_context):
        """Test routing GET /routing-rules to _list_rules."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/v1/routing-rules"
        mock_request.args = {}

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch.object(handler, "_list_rules", new_callable=AsyncMock) as mock_list,
        ):
            mock_auth.return_value = mock_auth_context
            mock_list.return_value = {"status": "success"}

            await handler.handle_request(mock_request)
            mock_list.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_route_to_create_rule(self, mock_auth_context):
        """Test routing POST /routing-rules to _create_rule."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.path = "/api/v1/routing-rules"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch.object(handler, "_create_rule", new_callable=AsyncMock) as mock_create,
        ):
            mock_auth.return_value = mock_auth_context
            mock_create.return_value = {"status": "success"}

            await handler.handle_request(mock_request)
            mock_create.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_route_to_get_templates(self, mock_auth_context):
        """Test routing GET /routing-rules/templates to _get_templates."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/v1/routing-rules/templates"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch.object(handler, "_get_templates", new_callable=AsyncMock) as mock_templates,
        ):
            mock_auth.return_value = mock_auth_context
            mock_templates.return_value = {"status": "success"}

            await handler.handle_request(mock_request)
            mock_templates.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_route_to_evaluate_rules(self, mock_auth_context):
        """Test routing POST /routing-rules/evaluate to _evaluate_rules."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.path = "/api/v1/routing-rules/evaluate"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch.object(handler, "_evaluate_rules", new_callable=AsyncMock) as mock_eval,
        ):
            mock_auth.return_value = mock_auth_context
            mock_eval.return_value = {"status": "success"}

            await handler.handle_request(mock_request)
            mock_eval.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_route_to_get_rule(self, mock_auth_context):
        """Test routing GET /routing-rules/{id} to _get_rule."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/v1/routing-rules/rule123"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch.object(handler, "_get_rule", new_callable=AsyncMock) as mock_get,
        ):
            mock_auth.return_value = mock_auth_context
            mock_get.return_value = {"status": "success"}

            await handler.handle_request(mock_request)
            mock_get.assert_called_once_with(mock_request, "rule123")

    @pytest.mark.asyncio
    async def test_route_to_toggle_rule(self, mock_auth_context):
        """Test routing POST /routing-rules/{id}/toggle to _toggle_rule."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.path = "/api/v1/routing-rules/rule123/toggle"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch.object(handler, "_toggle_rule", new_callable=AsyncMock) as mock_toggle,
        ):
            mock_auth.return_value = mock_auth_context
            mock_toggle.return_value = {"status": "success"}

            await handler.handle_request(mock_request)
            mock_toggle.assert_called_once_with(mock_request, "rule123")

    @pytest.mark.asyncio
    async def test_invalid_rule_id_rejected(self, mock_auth_context):
        """Test that invalid rule IDs are rejected."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/v1/routing-rules/invalid@id!"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = mock_auth_context

            result = await handler.handle_request(mock_request)
            assert result["code"] == 400
            assert "Invalid" in result["error"]

    @pytest.mark.asyncio
    async def test_method_not_allowed_for_invalid_route(self, mock_auth_context):
        """Test method not allowed for unsupported method/route combos."""
        handler = RoutingRulesHandler({})
        mock_request = MagicMock()
        mock_request.method = "PATCH"
        mock_request.path = "/api/v1/routing-rules/unknown/path"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = mock_auth_context

            result = await handler.handle_request(mock_request)
            assert result["code"] == 405


# =============================================================================
# Routing Engine Tests
# =============================================================================


class TestRoutingEngine:
    """Tests for the routing engine helper."""

    def test_get_routing_engine_empty(self):
        """Test getting engine with no rules."""
        engine = _get_routing_engine()
        assert engine is not None

    def test_get_routing_engine_with_rules(self, sample_rule_data):
        """Test getting engine loads rules from store."""
        _rules_store["rule1"] = sample_rule_data

        with patch("aragora.core.routing_rules.RoutingRule.from_dict") as mock_from_dict:
            mock_rule = MagicMock()
            mock_from_dict.return_value = mock_rule

            engine = _get_routing_engine()
            assert engine is not None
            mock_from_dict.assert_called_once()

    def test_get_routing_engine_handles_invalid_rules(self):
        """Test engine creation handles invalid rules gracefully."""
        _rules_store["bad_rule"] = {"invalid": "data"}

        with patch("aragora.core.routing_rules.RoutingRule.from_dict") as mock_from_dict:
            mock_from_dict.side_effect = ValueError("Invalid rule")

            # Should not raise
            engine = _get_routing_engine()
            assert engine is not None
