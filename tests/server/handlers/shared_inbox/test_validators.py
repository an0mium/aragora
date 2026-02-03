"""Tests for shared inbox validators."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.server.handlers.shared_inbox.models import (
    RoutingRule,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionField,
    RuleConditionOperator,
)
from aragora.server.handlers.shared_inbox.validators import (
    RuleRateLimiter,
    detect_circular_routing,
    get_rule_rate_limiter,
    validate_inbox_input,
    validate_routing_rule,
    validate_rule_action,
    validate_rule_condition,
    validate_rule_condition_field,
    validate_safe_regex,
    validate_tag,
)


# =============================================================================
# Helper Factories
# =============================================================================


def _make_routing_rule(
    *,
    rule_id: str = "rule-1",
    name: str = "Test Rule",
    workspace_id: str = "ws-1",
    conditions: list[RuleCondition] | None = None,
    actions: list[RuleAction] | None = None,
    enabled: bool = True,
) -> RoutingRule:
    """Create a RoutingRule with sensible defaults for testing."""
    if conditions is None:
        conditions = [
            RuleCondition(
                field=RuleConditionField.SUBJECT,
                operator=RuleConditionOperator.CONTAINS,
                value="test",
            )
        ]
    if actions is None:
        actions = [RuleAction(type=RuleActionType.ASSIGN, target="user-1")]
    return RoutingRule(
        id=rule_id,
        name=name,
        workspace_id=workspace_id,
        conditions=conditions,
        condition_logic="AND",
        actions=actions,
        enabled=enabled,
    )


# =============================================================================
# TestValidateSafeRegex
# =============================================================================


class TestValidateSafeRegex:
    def test_valid_simple_pattern(self):
        is_safe, error = validate_safe_regex(r"^urgent.*$")
        assert is_safe is True
        assert error is None

    def test_valid_character_class(self):
        is_safe, error = validate_safe_regex(r"[a-z]+@[a-z]+\.com")
        assert is_safe is True
        assert error is None

    def test_valid_alternation(self):
        is_safe, error = validate_safe_regex(r"urgent|critical")
        assert is_safe is True
        assert error is None

    def test_valid_at_max_length(self):
        # The inner is_safe_regex_pattern has its own limit of 100 chars,
        # so a 200-char pattern passes the local length check but is
        # rejected by the security module. Use 100 chars instead.
        pattern = "a" * 100
        is_safe, error = validate_safe_regex(pattern)
        assert is_safe is True
        assert error is None

    def test_valid_quantifier(self):
        is_safe, error = validate_safe_regex(r"\d{1,5}")
        assert is_safe is True
        assert error is None

    def test_valid_backslash_sequences(self):
        is_safe, error = validate_safe_regex(r"\w+\s+\d+")
        assert is_safe is True
        assert error is None

    def test_empty_pattern_rejected(self):
        is_safe, error = validate_safe_regex("")
        assert is_safe is False
        assert error == "Empty regex pattern"

    def test_exceeds_max_length(self):
        pattern = "a" * 201
        is_safe, error = validate_safe_regex(pattern)
        assert is_safe is False
        assert "exceeds maximum length" in error

    def test_custom_max_length(self):
        is_safe, error = validate_safe_regex("abc", max_length=2)
        assert is_safe is False
        assert "exceeds maximum length of 2" in error

    def test_invalid_syntax(self):
        is_safe, error = validate_safe_regex("[unclosed")
        assert is_safe is False
        assert "Invalid regex syntax" in error

    def test_redos_nested_quantifier(self):
        is_safe, error = validate_safe_regex("(a+)+")
        assert is_safe is False
        assert error is not None

    def test_pattern_at_exact_custom_max(self):
        is_safe, error = validate_safe_regex("ab", max_length=2)
        assert is_safe is True
        assert error is None


# =============================================================================
# TestValidateRuleConditionField
# =============================================================================


class TestValidateRuleConditionField:
    @pytest.mark.parametrize(
        "field_value",
        ["from", "to", "subject", "body", "labels", "priority", "sender_domain"],
    )
    def test_all_allowed_fields(self, field_value: str):
        is_valid, error = validate_rule_condition_field(field_value)
        assert is_valid is True
        assert error is None

    def test_invalid_field(self):
        is_valid, error = validate_rule_condition_field("x_custom_header")
        assert is_valid is False
        assert "Invalid field" in error

    def test_empty_field(self):
        is_valid, error = validate_rule_condition_field("")
        assert is_valid is False
        assert error is not None

    def test_case_sensitive(self):
        is_valid, error = validate_rule_condition_field("FROM")
        assert is_valid is False
        assert error is not None


# =============================================================================
# TestValidateRuleCondition
# =============================================================================


class TestValidateRuleCondition:
    def test_valid_contains(self):
        condition = {"field": "subject", "operator": "contains", "value": "urgent"}
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is True
        assert error is None
        assert sanitized is not None
        assert sanitized["field"] == "subject"
        assert sanitized["operator"] == "contains"
        assert sanitized["value"] == "urgent"

    def test_valid_matches_safe_regex(self):
        condition = {"field": "from", "operator": "matches", "value": r"^admin@.*"}
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is True
        assert error is None
        assert sanitized is not None

    def test_not_a_dict(self):
        is_valid, error, sanitized = validate_rule_condition("string")
        assert is_valid is False
        assert error == "Condition must be a dictionary"
        assert sanitized is None

    def test_missing_field(self):
        condition = {"operator": "contains", "value": "x"}
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is False
        assert error == "Condition missing 'field'"
        assert sanitized is None

    def test_missing_operator(self):
        condition = {"field": "subject", "value": "x"}
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is False
        assert error == "Condition missing 'operator'"
        assert sanitized is None

    def test_missing_value(self):
        condition = {"field": "subject", "operator": "contains"}
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is False
        assert error == "Condition missing 'value'"
        assert sanitized is None

    def test_none_value(self):
        condition = {"field": "subject", "operator": "contains", "value": None}
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is False
        assert error == "Condition missing 'value'"
        assert sanitized is None

    def test_invalid_field(self):
        condition = {"field": "bad", "operator": "contains", "value": "x"}
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is False
        assert "Invalid field" in error
        assert sanitized is None

    def test_invalid_operator(self):
        condition = {"field": "subject", "operator": "bad", "value": "x"}
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is False
        assert "Invalid operator" in error
        assert sanitized is None

    def test_value_not_string(self):
        condition = {"field": "subject", "operator": "contains", "value": 42}
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is False
        assert error == "Condition value must be a string"
        assert sanitized is None

    def test_value_too_long(self):
        condition = {
            "field": "subject",
            "operator": "contains",
            "value": "a" * 501,
        }
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is False
        assert "exceeds maximum length" in error
        assert sanitized is None

    def test_value_at_max_length(self):
        condition = {
            "field": "subject",
            "operator": "contains",
            "value": "a" * 500,
        }
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is True
        assert error is None
        assert sanitized is not None

    def test_matches_unsafe_regex(self):
        condition = {"field": "subject", "operator": "matches", "value": "(a+)+"}
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is False
        assert "Unsafe regex" in error
        assert sanitized is None

    def test_sanitization_strips_control_chars(self):
        condition = {
            "field": "subject",
            "operator": "contains",
            "value": "hello\x00world",
        }
        is_valid, error, sanitized = validate_rule_condition(condition)
        assert is_valid is True
        assert error is None
        assert "\x00" not in sanitized["value"]


# =============================================================================
# TestValidateRuleAction
# =============================================================================


class TestValidateRuleAction:
    def test_valid_assign(self):
        action = {"type": "assign", "target": "user-42"}
        is_valid, error, sanitized = validate_rule_action(action)
        assert is_valid is True
        assert error is None
        assert sanitized["type"] == "assign"
        assert sanitized["target"] == "user-42"

    def test_valid_archive_no_target(self):
        action = {"type": "archive"}
        is_valid, error, sanitized = validate_rule_action(action)
        assert is_valid is True
        assert error is None
        assert sanitized == {"type": "archive", "target": None, "params": {}}

    def test_valid_forward(self):
        action = {"type": "forward", "target": "inbox-2"}
        is_valid, error, sanitized = validate_rule_action(action)
        assert is_valid is True
        assert error is None
        assert sanitized["type"] == "forward"
        assert sanitized["target"] == "inbox-2"

    def test_not_a_dict(self):
        is_valid, error, sanitized = validate_rule_action("bad")
        assert is_valid is False
        assert error == "Action must be a dictionary"
        assert sanitized is None

    def test_missing_type(self):
        action = {"target": "user-1"}
        is_valid, error, sanitized = validate_rule_action(action)
        assert is_valid is False
        assert error == "Action missing 'type'"
        assert sanitized is None

    def test_invalid_type(self):
        action = {"type": "delete_all"}
        is_valid, error, sanitized = validate_rule_action(action)
        assert is_valid is False
        assert "Invalid action type" in error
        assert sanitized is None

    def test_target_not_string(self):
        action = {"type": "assign", "target": 123}
        is_valid, error, sanitized = validate_rule_action(action)
        assert is_valid is False
        assert error == "Action target must be a string"
        assert sanitized is None

    def test_target_too_long(self):
        action = {"type": "assign", "target": "a" * 201}
        is_valid, error, sanitized = validate_rule_action(action)
        assert is_valid is False
        assert "exceeds maximum length of 200" in error
        assert sanitized is None

    def test_params_not_dict(self):
        action = {"type": "assign", "params": "bad"}
        is_valid, error, sanitized = validate_rule_action(action)
        assert is_valid is False
        assert error == "Action params must be a dictionary"
        assert sanitized is None

    def test_with_params(self):
        action = {"type": "notify", "params": {"channel": "slack"}}
        is_valid, error, sanitized = validate_rule_action(action)
        assert is_valid is True
        assert error is None
        assert sanitized["params"] == {"channel": "slack"}


# =============================================================================
# TestDetectCircularRouting
# =============================================================================


class TestDetectCircularRouting:
    def test_no_cycle_assign_action(self):
        """Non-forward actions never create cycles."""
        actions = [{"type": "assign", "target": "user-1"}]
        has_cycle, error = detect_circular_routing(actions, [], "ws-1")
        assert has_cycle is False
        assert error is None

    def test_no_cycle_forward_no_existing(self):
        """Forward with no existing rules cannot create a cycle."""
        actions = [{"type": "forward", "target": "inbox-b"}]
        has_cycle, error = detect_circular_routing(actions, [], "ws-1")
        assert has_cycle is False
        assert error is None

    def test_no_cycle_linear_chain(self):
        """A linear forward chain without loops should be fine."""
        rule = _make_routing_rule(
            actions=[RuleAction(type=RuleActionType.FORWARD, target="inbox-c")],
        )
        # Give the rule an inbox_id so it becomes source="inbox-b"
        rule.inbox_id = "inbox-b"  # type: ignore[attr-defined]

        actions = [{"type": "forward", "target": "inbox-b"}]
        has_cycle, error = detect_circular_routing(actions, [rule], "ws-1")
        # BFS from inbox-b: forward_graph["inbox-b"] = {"inbox-c"}.
        # "inbox-c" != "inbox-b" and != "global". forward_graph["inbox-c"] empty.
        assert has_cycle is False
        assert error is None

    def test_cycle_self_loop(self):
        """A rule that forwards to itself creates a cycle."""
        rule = _make_routing_rule(
            actions=[RuleAction(type=RuleActionType.FORWARD, target="inbox-b")],
        )
        rule.inbox_id = "inbox-b"  # type: ignore[attr-defined]

        # New rule forwards to inbox-b, and inbox-b already forwards to inbox-b
        actions = [{"type": "forward", "target": "inbox-b"}]
        has_cycle, error = detect_circular_routing(actions, [rule], "ws-1")
        # BFS from inbox-b: forward_graph["inbox-b"] = {"inbox-b"}.
        # next_target "inbox-b" == target "inbox-b" -> cycle
        assert has_cycle is True
        assert "Circular routing detected" in error

    def test_cycle_indirect(self):
        """A -> B -> A creates an indirect cycle."""
        rule_b = _make_routing_rule(
            rule_id="rule-b",
            actions=[RuleAction(type=RuleActionType.FORWARD, target="inbox-a")],
        )
        rule_b.inbox_id = "inbox-b"  # type: ignore[attr-defined]

        rule_a = _make_routing_rule(
            rule_id="rule-a",
            actions=[RuleAction(type=RuleActionType.FORWARD, target="inbox-b")],
        )
        rule_a.inbox_id = "inbox-a"  # type: ignore[attr-defined]

        # New rule forwards to inbox-a:
        # BFS from inbox-a: forward_graph["inbox-a"] = {"inbox-b"}.
        # Check inbox-b: forward_graph["inbox-b"] = {"inbox-a"}.
        # "inbox-a" == target "inbox-a" -> cycle
        actions = [{"type": "forward", "target": "inbox-a"}]
        has_cycle, error = detect_circular_routing(
            actions, [rule_a, rule_b], "ws-1"
        )
        assert has_cycle is True
        assert "Circular routing detected" in error

    def test_cycle_through_global(self):
        """Forwarding to 'global' is detected as a cycle."""
        rule = _make_routing_rule(
            actions=[RuleAction(type=RuleActionType.FORWARD, target="global")],
        )
        rule.inbox_id = "inbox-b"  # type: ignore[attr-defined]

        actions = [{"type": "forward", "target": "inbox-b"}]
        has_cycle, error = detect_circular_routing(actions, [rule], "ws-1")
        # BFS from inbox-b: forward_graph["inbox-b"] = {"global"}.
        # "global" triggers cycle detection
        assert has_cycle is True
        assert "Circular routing detected" in error

    def test_disabled_rule_ignored(self):
        """Disabled rules should not contribute to cycle detection."""
        rule = _make_routing_rule(
            actions=[RuleAction(type=RuleActionType.FORWARD, target="inbox-b")],
            enabled=False,
        )
        rule.inbox_id = "inbox-b"  # type: ignore[attr-defined]

        actions = [{"type": "forward", "target": "inbox-b"}]
        has_cycle, error = detect_circular_routing(actions, [rule], "ws-1")
        assert has_cycle is False
        assert error is None

    def test_different_workspace_ignored(self):
        """Rules from a different workspace should not affect cycle detection."""
        rule = _make_routing_rule(
            workspace_id="ws-other",
            actions=[RuleAction(type=RuleActionType.FORWARD, target="inbox-b")],
        )
        rule.inbox_id = "inbox-b"  # type: ignore[attr-defined]

        actions = [{"type": "forward", "target": "inbox-b"}]
        has_cycle, error = detect_circular_routing(actions, [rule], "ws-1")
        assert has_cycle is False
        assert error is None


# =============================================================================
# TestValidateRoutingRule
# =============================================================================


class TestValidateRoutingRule:
    """Tests for the top-level validate_routing_rule orchestrator."""

    VALID_CONDITIONS = [
        {"field": "subject", "operator": "contains", "value": "test"},
    ]
    VALID_ACTIONS = [
        {"type": "assign", "target": "user-1"},
    ]

    def test_valid_rule(self):
        result = validate_routing_rule(
            name="My Rule",
            conditions=self.VALID_CONDITIONS,
            actions=self.VALID_ACTIONS,
            workspace_id="ws-1",
        )
        assert result.is_valid is True
        assert result.error is None
        assert result.sanitized_conditions is not None
        assert result.sanitized_actions is not None

    def test_empty_name(self):
        result = validate_routing_rule(
            name="",
            conditions=self.VALID_CONDITIONS,
            actions=self.VALID_ACTIONS,
            workspace_id="ws-1",
        )
        assert result.is_valid is False
        assert result.error == "Rule name is required"

    def test_name_too_long(self):
        result = validate_routing_rule(
            name="a" * 201,
            conditions=self.VALID_CONDITIONS,
            actions=self.VALID_ACTIONS,
            workspace_id="ws-1",
        )
        assert result.is_valid is False
        assert "exceeds maximum length" in result.error

    def test_name_only_control_chars(self):
        result = validate_routing_rule(
            name="\x00\x01",
            conditions=self.VALID_CONDITIONS,
            actions=self.VALID_ACTIONS,
            workspace_id="ws-1",
        )
        assert result.is_valid is False
        assert "empty after sanitization" in result.error

    def test_description_too_long(self):
        result = validate_routing_rule(
            name="Rule",
            conditions=self.VALID_CONDITIONS,
            actions=self.VALID_ACTIONS,
            workspace_id="ws-1",
            description="a" * 1001,
        )
        assert result.is_valid is False
        assert "Description exceeds maximum length" in result.error

    def test_no_conditions(self):
        result = validate_routing_rule(
            name="Rule",
            conditions=[],
            actions=self.VALID_ACTIONS,
            workspace_id="ws-1",
        )
        assert result.is_valid is False
        assert "At least one condition" in result.error

    def test_too_many_conditions(self):
        conditions = [
            {"field": "subject", "operator": "contains", "value": f"v{i}"}
            for i in range(21)
        ]
        result = validate_routing_rule(
            name="Rule",
            conditions=conditions,
            actions=self.VALID_ACTIONS,
            workspace_id="ws-1",
        )
        assert result.is_valid is False
        assert "exceeds maximum" in result.error

    def test_no_actions(self):
        result = validate_routing_rule(
            name="Rule",
            conditions=self.VALID_CONDITIONS,
            actions=[],
            workspace_id="ws-1",
        )
        assert result.is_valid is False
        assert "At least one action" in result.error

    def test_too_many_actions(self):
        actions = [{"type": "assign", "target": f"u{i}"} for i in range(11)]
        result = validate_routing_rule(
            name="Rule",
            conditions=self.VALID_CONDITIONS,
            actions=actions,
            workspace_id="ws-1",
        )
        assert result.is_valid is False
        assert "exceeds maximum" in result.error

    def test_invalid_condition_propagates(self):
        bad_conditions = [{"field": "bad_field", "operator": "contains", "value": "x"}]
        result = validate_routing_rule(
            name="Rule",
            conditions=bad_conditions,
            actions=self.VALID_ACTIONS,
            workspace_id="ws-1",
        )
        assert result.is_valid is False
        assert result.error.startswith("Condition 1:")

    def test_invalid_action_propagates(self):
        bad_actions = [{"type": "delete_all"}]
        result = validate_routing_rule(
            name="Rule",
            conditions=self.VALID_CONDITIONS,
            actions=bad_actions,
            workspace_id="ws-1",
        )
        assert result.is_valid is False
        assert result.error.startswith("Action 1:")

    def test_circular_check_skipped(self):
        """When check_circular=False, circular routing is not checked."""
        # Create existing rules that would form a cycle
        rule = _make_routing_rule(
            actions=[RuleAction(type=RuleActionType.FORWARD, target="inbox-a")],
        )
        rule.inbox_id = "inbox-a"  # type: ignore[attr-defined]

        result = validate_routing_rule(
            name="Rule",
            conditions=self.VALID_CONDITIONS,
            actions=[{"type": "forward", "target": "inbox-a"}],
            workspace_id="ws-1",
            existing_rules=[rule],
            check_circular=False,
        )
        assert result.is_valid is True


# =============================================================================
# TestValidateInboxInput
# =============================================================================


class TestValidateInboxInput:
    def test_valid_name_only(self):
        is_valid, error = validate_inbox_input(name="Support Inbox")
        assert is_valid is True
        assert error is None

    def test_valid_with_email(self):
        is_valid, error = validate_inbox_input(
            name="Support", email_address="support@example.com"
        )
        assert is_valid is True
        assert error is None

    def test_valid_with_description(self):
        is_valid, error = validate_inbox_input(
            name="Support", description="Main support inbox"
        )
        assert is_valid is True
        assert error is None

    def test_empty_name(self):
        is_valid, error = validate_inbox_input(name="")
        assert is_valid is False
        assert error == "Inbox name is required"

    def test_name_too_long(self):
        is_valid, error = validate_inbox_input(name="a" * 201)
        assert is_valid is False
        assert "exceeds maximum length" in error

    def test_description_too_long(self):
        is_valid, error = validate_inbox_input(
            name="Support", description="a" * 1001
        )
        assert is_valid is False
        assert "exceeds maximum length" in error

    def test_invalid_email_no_at(self):
        is_valid, error = validate_inbox_input(
            name="Support", email_address="not-an-email"
        )
        assert is_valid is False
        assert "Invalid email" in error

    def test_invalid_email_no_domain_dot(self):
        is_valid, error = validate_inbox_input(
            name="Support", email_address="user@localhost"
        )
        assert is_valid is False
        assert "Invalid email" in error

    def test_valid_complex_email(self):
        is_valid, error = validate_inbox_input(
            name="Support", email_address="team+support@sub.example.co.uk"
        )
        assert is_valid is True
        assert error is None


# =============================================================================
# TestValidateTag
# =============================================================================


class TestValidateTag:
    def test_valid_alphanumeric(self):
        is_valid, error = validate_tag("urgent123")
        assert is_valid is True
        assert error is None

    def test_valid_hyphens_underscores(self):
        is_valid, error = validate_tag("high-priority_v2")
        assert is_valid is True
        assert error is None

    def test_empty_tag(self):
        is_valid, error = validate_tag("")
        assert is_valid is False
        assert error == "Tag cannot be empty"

    def test_too_long(self):
        is_valid, error = validate_tag("a" * 101)
        assert is_valid is False
        assert "exceeds maximum length" in error

    def test_spaces_rejected(self):
        is_valid, error = validate_tag("has spaces")
        assert is_valid is False
        assert "can only contain" in error

    def test_special_chars_rejected(self):
        is_valid, error = validate_tag("tag@#!")
        assert is_valid is False
        assert "can only contain" in error

    def test_unicode_letters(self):
        # Python \w matches unicode letters by default
        is_valid, error = validate_tag("priorite")
        assert is_valid is True
        assert error is None


# =============================================================================
# TestRuleRateLimiter
# =============================================================================


class TestRuleRateLimiter:
    def test_allows_first_request(self):
        limiter = RuleRateLimiter()
        allowed, remaining = limiter.is_allowed("ws-1")
        assert allowed is True
        assert remaining == 10

    def test_allows_up_to_max(self):
        limiter = RuleRateLimiter(window_seconds=60, max_requests=10)
        for _ in range(9):
            limiter.record_request("ws-1")
        allowed, remaining = limiter.is_allowed("ws-1")
        assert allowed is True
        assert remaining == 1

        limiter.record_request("ws-1")
        allowed, remaining = limiter.is_allowed("ws-1")
        assert allowed is False
        assert remaining == 0

    def test_different_workspaces_independent(self):
        limiter = RuleRateLimiter(window_seconds=60, max_requests=2)
        limiter.record_request("ws-a")
        limiter.record_request("ws-a")

        allowed_a, remaining_a = limiter.is_allowed("ws-a")
        assert allowed_a is False
        assert remaining_a == 0

        allowed_b, remaining_b = limiter.is_allowed("ws-b")
        assert allowed_b is True
        assert remaining_b == 2

    def test_window_expiry(self):
        limiter = RuleRateLimiter(window_seconds=60, max_requests=2)

        with patch("aragora.server.handlers.shared_inbox.validators.time") as mock_time:
            mock_time.time.return_value = 1000.0
            limiter.record_request("ws-1")
            limiter.record_request("ws-1")

            allowed, _ = limiter.is_allowed("ws-1")
            assert allowed is False

            # Advance past the window
            mock_time.time.return_value = 1061.0
            allowed, remaining = limiter.is_allowed("ws-1")
            assert allowed is True
            assert remaining == 2

    def test_record_and_check(self):
        limiter = RuleRateLimiter(window_seconds=60, max_requests=10)
        for _ in range(3):
            limiter.record_request("ws-1")
        allowed, remaining = limiter.is_allowed("ws-1")
        assert allowed is True
        assert remaining == 7

    def test_get_retry_after_zero(self):
        limiter = RuleRateLimiter(window_seconds=60, max_requests=10)
        limiter.record_request("ws-1")
        retry_after = limiter.get_retry_after("ws-1")
        assert retry_after == 0.0

    def test_get_retry_after_positive(self):
        limiter = RuleRateLimiter(window_seconds=60, max_requests=2)

        with patch("aragora.server.handlers.shared_inbox.validators.time") as mock_time:
            mock_time.time.return_value = 1000.0
            limiter.record_request("ws-1")
            limiter.record_request("ws-1")

            mock_time.time.return_value = 1030.0
            retry_after = limiter.get_retry_after("ws-1")
            # Oldest timestamp is 1000.0. Retry after = (1000 + 60) - 1030 = 30.0
            assert retry_after == pytest.approx(30.0)

    def test_custom_config(self):
        limiter = RuleRateLimiter(window_seconds=10, max_requests=2)
        limiter.record_request("ws-1")
        limiter.record_request("ws-1")
        allowed, _ = limiter.is_allowed("ws-1")
        assert allowed is False

    def test_get_rule_rate_limiter(self):
        limiter = get_rule_rate_limiter()
        assert isinstance(limiter, RuleRateLimiter)
