"""Tests for the shared inbox handler re-export module.

Tests the backward-compatibility re-export layer at
``aragora/server/handlers/_shared_inbox_handler.py`` (173 lines),
plus the underlying SharedInboxHandler class methods, validator functions,
rules engine logic, and handler functions for inbox/rule CRUD operations.

Covers:
- Re-export verification (all __all__ symbols importable)
- SharedInboxHandler routing (can_handle), init, and method dispatch
- POST/GET inbox handlers (create, list, get, messages)
- Message operations (assign, status update, tag)
- Routing rule CRUD (create, list, update, delete, test)
- Validator functions (regex safety, conditions, actions, tags, inbox input)
- Rules engine (evaluate_rule, get_matching_rules, apply_routing_rules)
- Rate limiter
- Error and edge-case paths
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import from the re-export layer under test
from aragora.server.handlers._shared_inbox_handler import (
    # Models
    MessageStatus,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionField,
    RuleConditionOperator,
    RoutingRule,
    SharedInbox,
    SharedInboxMessage,
    # Storage
    USE_PERSISTENT_STORAGE,
    _shared_inboxes,
    _inbox_messages,
    _routing_rules,
    _storage_lock,
    # Validators - Constants
    ALLOWED_RULE_CONDITION_FIELDS,
    REGEX_OPERATORS,
    MAX_RULE_NAME_LENGTH,
    MAX_RULE_DESCRIPTION_LENGTH,
    MAX_CONDITION_VALUE_LENGTH,
    MAX_REGEX_PATTERN_LENGTH,
    MAX_TAG_LENGTH,
    MAX_INBOX_NAME_LENGTH,
    MAX_INBOX_DESCRIPTION_LENGTH,
    MAX_CONDITIONS_PER_RULE,
    MAX_ACTIONS_PER_RULE,
    MAX_RULES_PER_WORKSPACE,
    RULE_RATE_LIMIT_WINDOW_SECONDS,
    RULE_RATE_LIMIT_MAX_REQUESTS,
    # Validators - Classes
    RateLimitEntry,
    RuleRateLimiter,
    RuleValidationResult,
    # Validators - Functions
    get_rule_rate_limiter,
    validate_safe_regex,
    validate_rule_condition_field,
    validate_rule_condition,
    validate_rule_action,
    detect_circular_routing,
    validate_routing_rule,
    validate_inbox_input,
    validate_tag,
    # Rules Engine
    MessageLike,
    get_matching_rules_for_email,
    apply_routing_rules_to_message,
    evaluate_rule_for_test,
    _evaluate_rule,
    # Handler Class
    SharedInboxHandler,
    # Handler Functions
    handle_create_shared_inbox,
    handle_list_shared_inboxes,
    handle_get_shared_inbox,
    handle_get_inbox_messages,
    handle_assign_message,
    handle_update_message_status,
    handle_add_message_tag,
    handle_add_message_to_inbox,
    handle_create_routing_rule,
    handle_list_routing_rules,
    handle_update_routing_rule,
    handle_delete_routing_rule,
    handle_test_routing_rule,
    # Backward compat extras
    _rule_rate_limiter,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_server_context():
    return {"user_store": MagicMock(), "nomic_dir": "/tmp/test"}


@pytest.fixture
def handler(mock_server_context):
    h = SharedInboxHandler(mock_server_context)
    h._get_user_id = MagicMock(return_value="test-user-001")
    return h


@pytest.fixture(autouse=True)
def _clean_in_memory_stores():
    """Clear in-memory caches between tests to avoid cross-contamination."""
    with _storage_lock:
        _shared_inboxes.clear()
        _inbox_messages.clear()
        _routing_rules.clear()
    yield
    with _storage_lock:
        _shared_inboxes.clear()
        _inbox_messages.clear()
        _routing_rules.clear()


# ===========================================================================
# 1. Re-export verification
# ===========================================================================


class TestReExports:
    """Verify every __all__ symbol is importable from the re-export module."""

    def test_all_model_types_exported(self):
        assert MessageStatus is not None
        assert RuleConditionField is not None
        assert RuleConditionOperator is not None
        assert RuleActionType is not None
        assert RuleCondition is not None
        assert RuleAction is not None
        assert RoutingRule is not None
        assert SharedInboxMessage is not None
        assert SharedInbox is not None

    def test_storage_symbols_exported(self):
        assert USE_PERSISTENT_STORAGE is not None
        assert _shared_inboxes is not None
        assert _inbox_messages is not None
        assert _routing_rules is not None
        assert _storage_lock is not None

    def test_validator_constants_exported(self):
        assert MAX_RULE_NAME_LENGTH == 200
        assert MAX_TAG_LENGTH == 100
        assert MAX_INBOX_NAME_LENGTH == 200
        assert MAX_CONDITIONS_PER_RULE == 20
        assert MAX_ACTIONS_PER_RULE == 10
        assert MAX_RULES_PER_WORKSPACE == 500
        assert RULE_RATE_LIMIT_WINDOW_SECONDS == 60
        assert RULE_RATE_LIMIT_MAX_REQUESTS == 10

    def test_validator_functions_exported(self):
        for fn in (
            validate_safe_regex,
            validate_rule_condition_field,
            validate_rule_condition,
            validate_rule_action,
            detect_circular_routing,
            validate_routing_rule,
            validate_inbox_input,
            validate_tag,
            get_rule_rate_limiter,
        ):
            assert callable(fn)

    def test_rules_engine_exports(self):
        assert callable(get_matching_rules_for_email)
        assert callable(apply_routing_rules_to_message)
        assert callable(evaluate_rule_for_test)
        assert callable(_evaluate_rule)

    def test_handler_class_exported(self):
        assert SharedInboxHandler is not None

    def test_handler_functions_exported(self):
        for fn in (
            handle_create_shared_inbox,
            handle_list_shared_inboxes,
            handle_get_shared_inbox,
            handle_get_inbox_messages,
            handle_assign_message,
            handle_update_message_status,
            handle_add_message_tag,
            handle_add_message_to_inbox,
            handle_create_routing_rule,
            handle_list_routing_rules,
            handle_update_routing_rule,
            handle_delete_routing_rule,
            handle_test_routing_rule,
        ):
            assert callable(fn)

    def test_backward_compat_rate_limiter(self):
        assert isinstance(_rule_rate_limiter, RuleRateLimiter)


# ===========================================================================
# 2. SharedInboxHandler routing and initialization
# ===========================================================================


class TestSharedInboxHandlerCanHandle:
    def test_exact_route_shared(self, handler):
        assert handler.can_handle("/api/v1/inbox/shared")

    def test_exact_route_rules(self, handler):
        assert handler.can_handle("/api/v1/inbox/routing/rules")

    def test_prefix_shared_inbox_id(self, handler):
        assert handler.can_handle("/api/v1/inbox/shared/abc-123")

    def test_prefix_shared_messages(self, handler):
        assert handler.can_handle("/api/v1/inbox/shared/abc/messages")

    def test_prefix_rules_id(self, handler):
        assert handler.can_handle("/api/v1/inbox/routing/rules/rule-99/test")

    def test_reject_unknown(self, handler):
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/inbox/other")
        assert not handler.can_handle("/unknown")

    def test_handle_returns_none(self, handler):
        """The base handle() stub returns None (not yet dispatched)."""
        result = handler.handle("/api/v1/inbox/shared", {}, MagicMock())
        assert result is None


class TestSharedInboxHandlerInit:
    def test_extends_base_handler(self, handler):
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)

    def test_routes_and_prefixes(self, handler):
        assert "/api/v1/inbox/shared" in handler.ROUTES
        assert "/api/v1/inbox/routing/rules" in handler.ROUTES
        assert "/api/v1/inbox/shared/" in handler.ROUTE_PREFIXES
        assert "/api/v1/inbox/routing/rules/" in handler.ROUTE_PREFIXES


# ===========================================================================
# 3. Validator tests
# ===========================================================================


class TestValidateTag:
    def test_valid_tag(self):
        ok, err = validate_tag("urgent")
        assert ok is True
        assert err is None

    def test_empty_tag(self):
        ok, err = validate_tag("")
        assert ok is False
        assert "empty" in err.lower()

    def test_tag_too_long(self):
        ok, err = validate_tag("a" * (MAX_TAG_LENGTH + 1))
        assert ok is False
        assert "maximum" in err.lower() or "exceeds" in err.lower()

    def test_tag_invalid_chars(self):
        ok, err = validate_tag("hello world!")
        assert ok is False

    def test_tag_with_hyphens_underscores(self):
        ok, err = validate_tag("my-tag_123")
        assert ok is True


class TestValidateInboxInput:
    def test_valid_inbox(self):
        ok, err = validate_inbox_input("Support", description="Help desk")
        assert ok is True

    def test_missing_name(self):
        ok, err = validate_inbox_input("")
        assert ok is False

    def test_name_too_long(self):
        ok, err = validate_inbox_input("x" * (MAX_INBOX_NAME_LENGTH + 1))
        assert ok is False

    def test_description_too_long(self):
        ok, err = validate_inbox_input("Ok", description="d" * (MAX_INBOX_DESCRIPTION_LENGTH + 1))
        assert ok is False

    def test_valid_email(self):
        ok, err = validate_inbox_input("Inbox", email_address="team@example.com")
        assert ok is True

    def test_invalid_email_no_at(self):
        ok, err = validate_inbox_input("Inbox", email_address="invalid")
        assert ok is False

    def test_invalid_email_no_domain_dot(self):
        ok, err = validate_inbox_input("Inbox", email_address="user@localhost")
        assert ok is False


class TestValidateRuleConditionField:
    def test_valid_field(self):
        ok, err = validate_rule_condition_field("from")
        assert ok is True

    def test_invalid_field(self):
        ok, err = validate_rule_condition_field("x_custom")
        assert ok is False
        assert "Allowed fields" in err or "Invalid field" in err


class TestValidateRuleCondition:
    def test_valid_condition(self):
        ok, err, sanitized = validate_rule_condition(
            {"field": "subject", "operator": "contains", "value": "help"}
        )
        assert ok is True
        assert sanitized is not None
        assert sanitized["field"] == "subject"

    def test_missing_field(self):
        ok, err, _ = validate_rule_condition({"operator": "contains", "value": "x"})
        assert ok is False

    def test_missing_operator(self):
        ok, err, _ = validate_rule_condition({"field": "subject", "value": "x"})
        assert ok is False

    def test_missing_value(self):
        ok, err, _ = validate_rule_condition({"field": "subject", "operator": "contains"})
        assert ok is False

    def test_invalid_operator(self):
        ok, err, _ = validate_rule_condition(
            {"field": "subject", "operator": "NOT_REAL", "value": "x"}
        )
        assert ok is False

    def test_value_too_long(self):
        ok, err, _ = validate_rule_condition(
            {
                "field": "subject",
                "operator": "contains",
                "value": "x" * (MAX_CONDITION_VALUE_LENGTH + 1),
            }
        )
        assert ok is False

    def test_not_dict(self):
        ok, err, _ = validate_rule_condition("bad")
        assert ok is False


class TestValidateRuleAction:
    def test_valid_action(self):
        ok, err, sanitized = validate_rule_action({"type": "assign", "target": "user-1"})
        assert ok is True
        assert sanitized["type"] == "assign"

    def test_missing_type(self):
        ok, err, _ = validate_rule_action({"target": "user-1"})
        assert ok is False

    def test_invalid_type(self):
        ok, err, _ = validate_rule_action({"type": "explode"})
        assert ok is False

    def test_target_too_long(self):
        ok, err, _ = validate_rule_action({"type": "assign", "target": "x" * 201})
        assert ok is False

    def test_not_dict(self):
        ok, err, _ = validate_rule_action(42)
        assert ok is False

    def test_params_not_dict(self):
        ok, err, _ = validate_rule_action({"type": "assign", "params": "bad"})
        assert ok is False


class TestValidateRoutingRule:
    def test_valid_rule(self):
        result = validate_routing_rule(
            name="Test rule",
            conditions=[{"field": "subject", "operator": "contains", "value": "help"}],
            actions=[{"type": "assign", "target": "user-1"}],
            workspace_id="ws-1",
        )
        assert result.is_valid is True
        assert result.sanitized_conditions is not None
        assert result.sanitized_actions is not None

    def test_empty_name(self):
        result = validate_routing_rule(
            name="",
            conditions=[{"field": "subject", "operator": "contains", "value": "x"}],
            actions=[{"type": "assign", "target": "u"}],
            workspace_id="ws-1",
        )
        assert result.is_valid is False

    def test_no_conditions(self):
        result = validate_routing_rule(
            name="rule",
            conditions=[],
            actions=[{"type": "assign", "target": "u"}],
            workspace_id="ws-1",
        )
        assert result.is_valid is False

    def test_no_actions(self):
        result = validate_routing_rule(
            name="rule",
            conditions=[{"field": "subject", "operator": "contains", "value": "x"}],
            actions=[],
            workspace_id="ws-1",
        )
        assert result.is_valid is False

    def test_too_many_conditions(self):
        conds = [
            {"field": "subject", "operator": "contains", "value": f"v{i}"}
            for i in range(MAX_CONDITIONS_PER_RULE + 1)
        ]
        result = validate_routing_rule(
            name="rule",
            conditions=conds,
            actions=[{"type": "assign", "target": "u"}],
            workspace_id="ws-1",
        )
        assert result.is_valid is False

    def test_too_many_actions(self):
        acts = [{"type": "assign", "target": f"u{i}"} for i in range(MAX_ACTIONS_PER_RULE + 1)]
        result = validate_routing_rule(
            name="rule",
            conditions=[{"field": "subject", "operator": "contains", "value": "x"}],
            actions=acts,
            workspace_id="ws-1",
        )
        assert result.is_valid is False

    def test_description_too_long(self):
        result = validate_routing_rule(
            name="rule",
            conditions=[{"field": "subject", "operator": "contains", "value": "x"}],
            actions=[{"type": "assign", "target": "u"}],
            workspace_id="ws-1",
            description="d" * (MAX_RULE_DESCRIPTION_LENGTH + 1),
        )
        assert result.is_valid is False


class TestValidateSafeRegex:
    def test_valid_regex(self):
        ok, err = validate_safe_regex(r"hello\.\*world")
        assert ok is True

    def test_empty_regex(self):
        ok, err = validate_safe_regex("")
        assert ok is False

    def test_regex_too_long(self):
        ok, err = validate_safe_regex("x" * (MAX_REGEX_PATTERN_LENGTH + 1))
        assert ok is False


class TestDetectCircularRouting:
    def test_no_forward_no_circular(self):
        has_circular, err = detect_circular_routing(
            [{"type": "assign", "target": "user-1"}],
            [],
            "ws-1",
        )
        assert has_circular is False

    def test_forward_without_cycle(self):
        has_circular, err = detect_circular_routing(
            [{"type": "forward", "target": "inbox-B"}],
            [],
            "ws-1",
        )
        assert has_circular is False


# ===========================================================================
# 4. Rate limiter
# ===========================================================================


class TestRuleRateLimiter:
    def test_initial_allowed(self):
        rl = RuleRateLimiter(window_seconds=60, max_requests=5)
        allowed, remaining = rl.is_allowed("ws-1")
        assert allowed is True
        assert remaining == 5

    def test_exceeds_limit(self):
        rl = RuleRateLimiter(window_seconds=60, max_requests=2)
        rl.record_request("ws-1")
        rl.record_request("ws-1")
        allowed, remaining = rl.is_allowed("ws-1")
        assert allowed is False
        assert remaining == 0

    def test_get_retry_after_zero_when_allowed(self):
        rl = RuleRateLimiter(window_seconds=60, max_requests=5)
        assert rl.get_retry_after("ws-1") == 0.0

    def test_get_retry_after_positive_when_exceeded(self):
        rl = RuleRateLimiter(window_seconds=60, max_requests=1)
        rl.record_request("ws-1")
        retry = rl.get_retry_after("ws-1")
        assert retry > 0

    def test_global_instance(self):
        rl = get_rule_rate_limiter()
        assert isinstance(rl, RuleRateLimiter)


# ===========================================================================
# 5. Rules engine
# ===========================================================================


class TestEvaluateRule:
    def _make_message(self, **overrides):
        defaults = {
            "from_address": "alice@example.com",
            "to_addresses": ["bob@example.com"],
            "subject": "Need help with billing",
            "priority": "normal",
        }
        defaults.update(overrides)

        class Msg:
            pass

        msg = Msg()
        for k, v in defaults.items():
            setattr(msg, k, v)
        return msg

    def _make_rule(self, conditions, logic="AND"):
        return RoutingRule(
            id="rule-1",
            name="test",
            workspace_id="ws-1",
            conditions=[RuleCondition.from_dict(c) for c in conditions],
            condition_logic=logic,
            actions=[RuleAction(type=RuleActionType.ASSIGN, target="user-1")],
        )

    def test_contains_match(self):
        rule = self._make_rule([{"field": "subject", "operator": "contains", "value": "billing"}])
        msg = self._make_message()
        assert _evaluate_rule(rule, msg) is True

    def test_contains_no_match(self):
        rule = self._make_rule([{"field": "subject", "operator": "contains", "value": "shipping"}])
        msg = self._make_message()
        assert _evaluate_rule(rule, msg) is False

    def test_equals_match(self):
        rule = self._make_rule(
            [{"field": "from", "operator": "equals", "value": "alice@example.com"}]
        )
        msg = self._make_message()
        assert _evaluate_rule(rule, msg) is True

    def test_starts_with(self):
        rule = self._make_rule([{"field": "subject", "operator": "starts_with", "value": "need"}])
        msg = self._make_message()
        assert _evaluate_rule(rule, msg) is True

    def test_ends_with(self):
        rule = self._make_rule([{"field": "subject", "operator": "ends_with", "value": "billing"}])
        msg = self._make_message()
        assert _evaluate_rule(rule, msg) is True

    def test_sender_domain(self):
        rule = self._make_rule(
            [{"field": "sender_domain", "operator": "equals", "value": "example.com"}]
        )
        msg = self._make_message()
        assert _evaluate_rule(rule, msg) is True

    def test_priority_field(self):
        rule = self._make_rule([{"field": "priority", "operator": "equals", "value": "high"}])
        msg = self._make_message(priority="high")
        assert _evaluate_rule(rule, msg) is True

    def test_and_logic_all_match(self):
        rule = self._make_rule(
            [
                {"field": "subject", "operator": "contains", "value": "billing"},
                {"field": "from", "operator": "contains", "value": "alice"},
            ],
            logic="AND",
        )
        msg = self._make_message()
        assert _evaluate_rule(rule, msg) is True

    def test_and_logic_partial_match(self):
        rule = self._make_rule(
            [
                {"field": "subject", "operator": "contains", "value": "billing"},
                {"field": "from", "operator": "contains", "value": "charlie"},
            ],
            logic="AND",
        )
        msg = self._make_message()
        assert _evaluate_rule(rule, msg) is False

    def test_or_logic_partial_match(self):
        rule = self._make_rule(
            [
                {"field": "subject", "operator": "contains", "value": "xyz"},
                {"field": "from", "operator": "contains", "value": "alice"},
            ],
            logic="OR",
        )
        msg = self._make_message()
        assert _evaluate_rule(rule, msg) is True

    def test_matches_regex(self):
        rule = self._make_rule(
            [{"field": "subject", "operator": "matches", "value": r"help.*billing"}]
        )
        msg = self._make_message()
        assert _evaluate_rule(rule, msg) is True


class TestEvaluateRuleForTest:
    def test_counts_matching_messages(self):
        now = datetime.now(timezone.utc)
        inbox = SharedInbox(id="inbox-1", workspace_id="ws-1", name="Test")
        msg1 = SharedInboxMessage(
            id="m1",
            inbox_id="inbox-1",
            email_id="e1",
            subject="Billing issue",
            from_address="a@b.com",
            to_addresses=["x@y.com"],
            snippet="...",
            received_at=now,
        )
        msg2 = SharedInboxMessage(
            id="m2",
            inbox_id="inbox-1",
            email_id="e2",
            subject="Shipping question",
            from_address="a@b.com",
            to_addresses=["x@y.com"],
            snippet="...",
            received_at=now,
        )
        with _storage_lock:
            _shared_inboxes["inbox-1"] = inbox
            _inbox_messages["inbox-1"] = {"m1": msg1, "m2": msg2}

        rule = RoutingRule(
            id="r1",
            name="billing",
            workspace_id="ws-1",
            conditions=[
                RuleCondition(RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "billing")
            ],
            condition_logic="AND",
            actions=[RuleAction(type=RuleActionType.ASSIGN, target="u1")],
        )
        count = evaluate_rule_for_test(rule, "ws-1")
        assert count == 1


# ===========================================================================
# 6. Model dataclass tests
# ===========================================================================


class TestModels:
    def test_rule_condition_round_trip(self):
        cond = RuleCondition(RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "test")
        d = cond.to_dict()
        assert d["field"] == "subject"
        restored = RuleCondition.from_dict(d)
        assert restored.field == RuleConditionField.SUBJECT

    def test_rule_action_round_trip(self):
        action = RuleAction(type=RuleActionType.ASSIGN, target="user-1", params={"foo": "bar"})
        d = action.to_dict()
        assert d["type"] == "assign"
        restored = RuleAction.from_dict(d)
        assert restored.target == "user-1"
        assert restored.params == {"foo": "bar"}

    def test_routing_rule_round_trip(self):
        rule = RoutingRule(
            id="r1",
            name="test",
            workspace_id="ws-1",
            conditions=[
                RuleCondition(RuleConditionField.FROM, RuleConditionOperator.EQUALS, "a@b.com")
            ],
            condition_logic="OR",
            actions=[RuleAction(type=RuleActionType.LABEL, target="urgent")],
            priority=3,
        )
        d = rule.to_dict()
        assert d["id"] == "r1"
        assert d["condition_logic"] == "OR"
        restored = RoutingRule.from_dict(d)
        assert restored.priority == 3
        assert len(restored.conditions) == 1

    def test_shared_inbox_to_dict(self):
        inbox = SharedInbox(id="i1", workspace_id="ws-1", name="Support")
        d = inbox.to_dict()
        assert d["name"] == "Support"
        assert d["team_members"] == []

    def test_shared_inbox_message_to_dict(self):
        now = datetime.now(timezone.utc)
        msg = SharedInboxMessage(
            id="m1",
            inbox_id="i1",
            email_id="e1",
            subject="Hi",
            from_address="a@b.com",
            to_addresses=["c@d.com"],
            snippet="...",
            received_at=now,
            tags=["important"],
        )
        d = msg.to_dict()
        assert d["tags"] == ["important"]
        assert d["status"] == "open"


# ===========================================================================
# 7. SharedInboxHandler async method tests
# ===========================================================================


class TestHandlerPostSharedInbox:
    @pytest.mark.asyncio
    async def test_missing_workspace_id(self, handler):
        result = await handler.handle_post_shared_inbox({"name": "Inbox"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_name(self, handler):
        result = await handler.handle_post_shared_inbox({"workspace_id": "ws-1"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_email(self, handler):
        result = await handler.handle_post_shared_inbox(
            {"workspace_id": "ws-1", "name": "Test", "email_address": "bad"}
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_create_shared_inbox",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = {"success": True, "inbox": {"id": "i1"}}
            result = await handler.handle_post_shared_inbox(
                {"workspace_id": "ws-1", "name": "Support"}
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_downstream_failure(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_create_shared_inbox",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = {"success": False, "error": "Duplicate name"}
            result = await handler.handle_post_shared_inbox(
                {"workspace_id": "ws-1", "name": "Support"}
            )
            assert _status(result) == 400


class TestHandlerGetSharedInboxes:
    @pytest.mark.asyncio
    async def test_missing_workspace_id(self, handler):
        result = await handler.handle_get_shared_inboxes({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_list_shared_inboxes",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = {"success": True, "inboxes": [], "total": 0}
            result = await handler.handle_get_shared_inboxes({"workspace_id": "ws-1"})
            assert _status(result) == 200


class TestHandlerGetSharedInbox:
    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_get_shared_inbox",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = {"success": True, "inbox": {"id": "i1"}}
            result = await handler.handle_get_shared_inbox({}, "i1")
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_not_found(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_get_shared_inbox",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = {"success": False, "error": "Inbox not found"}
            result = await handler.handle_get_shared_inbox({}, "bad-id")
            assert _status(result) == 404


class TestHandlerPostAssignMessage:
    @pytest.mark.asyncio
    async def test_missing_assigned_to(self, handler):
        result = await handler.handle_post_assign_message({}, "inbox-1", "msg-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_assigned_to_not_string(self, handler):
        result = await handler.handle_post_assign_message({"assigned_to": 123}, "inbox-1", "msg-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_assigned_to_too_long(self, handler):
        result = await handler.handle_post_assign_message(
            {"assigned_to": "x" * 201}, "inbox-1", "msg-1"
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_assign_message",
            new_callable=AsyncMock,
        ) as mock_assign:
            mock_assign.return_value = {"success": True, "message": {"id": "msg-1"}}
            result = await handler.handle_post_assign_message(
                {"assigned_to": "user-1"}, "inbox-1", "msg-1"
            )
            assert _status(result) == 200


class TestHandlerPostUpdateStatus:
    @pytest.mark.asyncio
    async def test_missing_status(self, handler):
        result = await handler.handle_post_update_status({}, "inbox-1", "msg-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_status(self, handler):
        result = await handler.handle_post_update_status({"status": "nonsense"}, "inbox-1", "msg-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_update_message_status",
            new_callable=AsyncMock,
        ) as mock_update:
            mock_update.return_value = {"success": True, "message": {"id": "msg-1"}}
            result = await handler.handle_post_update_status(
                {"status": "resolved"}, "inbox-1", "msg-1"
            )
            assert _status(result) == 200


class TestHandlerPostAddTag:
    @pytest.mark.asyncio
    async def test_missing_tag(self, handler):
        result = await handler.handle_post_add_tag({}, "inbox-1", "msg-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_tag(self, handler):
        result = await handler.handle_post_add_tag({"tag": "bad tag!"}, "inbox-1", "msg-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_add_message_tag",
            new_callable=AsyncMock,
        ) as mock_tag:
            mock_tag.return_value = {"success": True, "message": {"id": "msg-1"}}
            result = await handler.handle_post_add_tag({"tag": "urgent"}, "inbox-1", "msg-1")
            assert _status(result) == 200


class TestHandlerPostRoutingRule:
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, handler):
        result = await handler.handle_post_routing_rule({"workspace_id": "ws-1"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_create_routing_rule",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = {"success": True, "rule": {"id": "r1"}}
            result = await handler.handle_post_routing_rule(
                {
                    "workspace_id": "ws-1",
                    "name": "Test",
                    "conditions": [{"field": "subject", "operator": "contains", "value": "help"}],
                    "actions": [{"type": "assign", "target": "u1"}],
                }
            )
            assert _status(result) == 200


class TestHandlerGetRoutingRules:
    @pytest.mark.asyncio
    async def test_missing_workspace_id(self, handler):
        result = await handler.handle_get_routing_rules({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_list_routing_rules",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = {"success": True, "rules": [], "total": 0}
            result = await handler.handle_get_routing_rules({"workspace_id": "ws-1"})
            assert _status(result) == 200


class TestHandlerPatchRoutingRule:
    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_update_routing_rule",
            new_callable=AsyncMock,
        ) as mock_update:
            mock_update.return_value = {"success": True, "rule": {"id": "r1"}}
            result = await handler.handle_patch_routing_rule({"enabled": False}, "r1")
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_not_found(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_update_routing_rule",
            new_callable=AsyncMock,
        ) as mock_update:
            mock_update.return_value = {"success": False, "error": "Rule not found"}
            result = await handler.handle_patch_routing_rule({"enabled": True}, "bad")
            assert _status(result) == 400


class TestHandlerDeleteRoutingRule:
    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_delete_routing_rule",
            new_callable=AsyncMock,
        ) as mock_delete:
            mock_delete.return_value = {"success": True, "deleted": "r1"}
            result = await handler.handle_delete_routing_rule("r1")
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_not_found(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_delete_routing_rule",
            new_callable=AsyncMock,
        ) as mock_delete:
            mock_delete.return_value = {"success": False, "error": "Rule not found"}
            result = await handler.handle_delete_routing_rule("bad")
            assert _status(result) == 400


class TestHandlerPostTestRoutingRule:
    @pytest.mark.asyncio
    async def test_missing_workspace_id(self, handler):
        result = await handler.handle_post_test_routing_rule({}, "r1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_test_routing_rule",
            new_callable=AsyncMock,
        ) as mock_test:
            mock_test.return_value = {"success": True, "match_count": 3}
            result = await handler.handle_post_test_routing_rule({"workspace_id": "ws-1"}, "r1")
            assert _status(result) == 200


# ===========================================================================
# 8. Handler _get_user_id
# ===========================================================================


class TestGetUserId:
    def test_from_auth_context(self, mock_server_context):
        ctx_with_auth = dict(mock_server_context)
        auth = MagicMock()
        auth.user_id = "uid-42"
        ctx_with_auth["auth_context"] = auth
        h = SharedInboxHandler(ctx_with_auth)
        assert h._get_user_id() == "uid-42"

    def test_default_when_no_context(self, mock_server_context):
        h = SharedInboxHandler(mock_server_context)
        assert h._get_user_id() == "default"


# ===========================================================================
# 9. GetInboxMessages handler
# ===========================================================================


class TestHandlerGetInboxMessages:
    @pytest.mark.asyncio
    async def test_success(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_get_inbox_messages",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = {
                "success": True,
                "messages": [],
                "total": 0,
                "limit": 50,
                "offset": 0,
            }
            result = await handler.handle_get_inbox_messages({}, "inbox-1")
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_with_filters(self, handler):
        with patch(
            "aragora.server.handlers.shared_inbox.handler.handle_get_inbox_messages",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = {
                "success": True,
                "messages": [],
                "total": 0,
                "limit": 10,
                "offset": 0,
            }
            result = await handler.handle_get_inbox_messages(
                {"status": "open", "assigned_to": "u1", "tag": "urgent", "limit": "10"},
                "inbox-1",
            )
            assert _status(result) == 200
