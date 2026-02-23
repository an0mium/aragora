"""
Tests for Shared Inbox Rules Engine (aragora/server/handlers/shared_inbox/rules_engine.py).

Covers all 4 public functions:
- _evaluate_rule          Synchronous rule evaluation against a message
- get_matching_rules_for_email  Async matching rules lookup
- apply_routing_rules_to_message  Async rule application with side effects
- evaluate_rule_for_test  Synchronous test evaluation against inbox messages

Test categories:
- _evaluate_rule: condition fields (FROM, TO, SUBJECT, SENDER_DOMAIN, PRIORITY),
  operators (CONTAINS, EQUALS, STARTS_WITH, ENDS_WITH, MATCHES), logic (AND/OR),
  edge cases, case insensitivity, regex timeout, empty conditions
- get_matching_rules_for_email: store primary path, in-memory fallback,
  store failure handling, workspace filtering, priority sorting, merge logic
- apply_routing_rules_to_message: action types (assign, label, escalate, archive),
  no matching rules, multiple rules, rule stats, store failure
- evaluate_rule_for_test: workspace filtering, inbox matching, no messages,
  multiple messages
- Security: ReDoS protection, path traversal in fields, injection attempts
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.shared_inbox.models import (
    MessageStatus,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionField,
    RuleConditionOperator,
    RoutingRule,
    SharedInbox,
    SharedInboxMessage,
)
from aragora.server.handlers.shared_inbox.rules_engine import (
    _evaluate_rule,
    apply_routing_rules_to_message,
    evaluate_rule_for_test,
    get_matching_rules_for_email,
)
from aragora.server.handlers.shared_inbox.storage import (
    _inbox_messages,
    _routing_rules,
    _shared_inboxes,
    _storage_lock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODULE = "aragora.server.handlers.shared_inbox.rules_engine"


def _make_condition(
    field_type: RuleConditionField = RuleConditionField.SUBJECT,
    operator: RuleConditionOperator = RuleConditionOperator.CONTAINS,
    value: str = "urgent",
) -> RuleCondition:
    return RuleCondition(field=field_type, operator=operator, value=value)


def _make_rule(
    rule_id: str = "rule_1",
    workspace_id: str = "ws_test",
    name: str = "Test Rule",
    conditions: list[RuleCondition] | None = None,
    condition_logic: str = "AND",
    actions: list[RuleAction] | None = None,
    priority: int = 5,
    enabled: bool = True,
    **kwargs: Any,
) -> RoutingRule:
    return RoutingRule(
        id=rule_id,
        workspace_id=workspace_id,
        name=name,
        conditions=[_make_condition()] if conditions is None else conditions,
        condition_logic=condition_logic,
        actions=[RuleAction(type=RuleActionType.ASSIGN, target="support-team")]
        if actions is None
        else actions,
        priority=priority,
        enabled=enabled,
        created_at=kwargs.get("created_at", datetime.now(timezone.utc)),
        updated_at=kwargs.get("updated_at", datetime.now(timezone.utc)),
        description=kwargs.get("description"),
        created_by=kwargs.get("created_by"),
        stats=kwargs.get("stats", {}),
    )


@dataclass
class _FakeMessage:
    """Minimal message-like object for rule evaluation."""

    from_address: str = "user@example.com"
    to_addresses: list[str] = field(default_factory=lambda: ["inbox@company.com"])
    subject: str = "Urgent: system down"
    priority: str | None = None


def _make_inbox_message(
    msg_id: str = "msg_1",
    inbox_id: str = "inbox_1",
    **kwargs: Any,
) -> SharedInboxMessage:
    return SharedInboxMessage(
        id=msg_id,
        inbox_id=inbox_id,
        email_id=kwargs.get("email_id", "email_1"),
        subject=kwargs.get("subject", "Urgent request"),
        from_address=kwargs.get("from_address", "sender@example.com"),
        to_addresses=kwargs.get("to_addresses", ["inbox@company.com"]),
        snippet=kwargs.get("snippet", "Please help"),
        received_at=kwargs.get("received_at", datetime.now(timezone.utc)),
        status=kwargs.get("status", MessageStatus.OPEN),
        assigned_to=kwargs.get("assigned_to"),
        tags=kwargs.get("tags", []),
        priority=kwargs.get("priority"),
    )


def _make_shared_inbox(
    inbox_id: str = "inbox_1",
    workspace_id: str = "ws_test",
) -> SharedInbox:
    return SharedInbox(
        id=inbox_id,
        workspace_id=workspace_id,
        name="Support Inbox",
    )


# ---------------------------------------------------------------------------
# Autouse fixture: clean shared state before/after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_shared_state():
    """Reset in-memory stores between tests."""
    with _storage_lock:
        _routing_rules.clear()
        _inbox_messages.clear()
        _shared_inboxes.clear()
    yield
    with _storage_lock:
        _routing_rules.clear()
        _inbox_messages.clear()
        _shared_inboxes.clear()


# ============================================================================
# _evaluate_rule -- condition field tests
# ============================================================================


class TestEvaluateRuleConditionFields:
    """Test _evaluate_rule with different condition fields."""

    def test_from_field_contains(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.FROM, RuleConditionOperator.CONTAINS, "user")
            ]
        )
        msg = _FakeMessage(from_address="user@example.com")
        assert _evaluate_rule(rule, msg) is True

    def test_from_field_no_match(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.FROM, RuleConditionOperator.CONTAINS, "admin")
            ]
        )
        msg = _FakeMessage(from_address="user@example.com")
        assert _evaluate_rule(rule, msg) is False

    def test_to_field_contains(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.TO, RuleConditionOperator.CONTAINS, "inbox")
            ]
        )
        msg = _FakeMessage(to_addresses=["inbox@company.com", "cc@company.com"])
        assert _evaluate_rule(rule, msg) is True

    def test_to_field_no_match(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.TO, RuleConditionOperator.CONTAINS, "sales")
            ]
        )
        msg = _FakeMessage(to_addresses=["inbox@company.com"])
        assert _evaluate_rule(rule, msg) is False

    def test_subject_field_contains(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "urgent"
                )
            ]
        )
        msg = _FakeMessage(subject="Urgent: server issues")
        assert _evaluate_rule(rule, msg) is True

    def test_subject_field_no_match(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "billing"
                )
            ]
        )
        msg = _FakeMessage(subject="Urgent: server issues")
        assert _evaluate_rule(rule, msg) is False

    def test_sender_domain_field(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SENDER_DOMAIN, RuleConditionOperator.EQUALS, "example.com"
                )
            ]
        )
        msg = _FakeMessage(from_address="user@example.com")
        assert _evaluate_rule(rule, msg) is True

    def test_sender_domain_no_at_sign(self):
        """When from_address has no @, sender_domain is empty string."""
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SENDER_DOMAIN, RuleConditionOperator.EQUALS, "example.com"
                )
            ]
        )
        msg = _FakeMessage(from_address="localuser")
        assert _evaluate_rule(rule, msg) is False

    def test_sender_domain_no_at_equals_empty(self):
        """No @ in address yields empty string; rule matching empty succeeds."""
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.SENDER_DOMAIN, RuleConditionOperator.EQUALS, "")
            ]
        )
        msg = _FakeMessage(from_address="localuser")
        assert _evaluate_rule(rule, msg) is True

    def test_priority_field(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.PRIORITY, RuleConditionOperator.EQUALS, "high")
            ]
        )
        msg = _FakeMessage(priority="high")
        assert _evaluate_rule(rule, msg) is True

    def test_priority_field_none(self):
        """When message.priority is None, it becomes empty string."""
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.PRIORITY, RuleConditionOperator.EQUALS, "high")
            ]
        )
        msg = _FakeMessage(priority=None)
        assert _evaluate_rule(rule, msg) is False

    def test_priority_field_none_matches_empty(self):
        """None priority resolves to empty string; equals '' matches."""
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.PRIORITY, RuleConditionOperator.EQUALS, "")
            ]
        )
        msg = _FakeMessage(priority=None)
        assert _evaluate_rule(rule, msg) is True


# ============================================================================
# _evaluate_rule -- operator tests
# ============================================================================


class TestEvaluateRuleOperators:
    """Test _evaluate_rule with different operators."""

    def test_contains_operator(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "alert")
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(subject="Security alert!")) is True
        assert _evaluate_rule(rule, _FakeMessage(subject="Hello world")) is False

    def test_equals_operator(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.SUBJECT, RuleConditionOperator.EQUALS, "hello")
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(subject="Hello")) is True
        assert _evaluate_rule(rule, _FakeMessage(subject="Hello World")) is False

    def test_starts_with_operator(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.STARTS_WITH, "re:"
                )
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(subject="Re: meeting")) is True
        assert _evaluate_rule(rule, _FakeMessage(subject="Fwd: Re: meeting")) is False

    def test_ends_with_operator(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.ENDS_WITH, "please"
                )
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(subject="Help me please")) is True
        assert _evaluate_rule(rule, _FakeMessage(subject="Please help me")) is False

    def test_matches_operator_simple_regex(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.MATCHES, r"\d{3,}"
                )
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(subject="Ticket 12345")) is True
        assert _evaluate_rule(rule, _FakeMessage(subject="No digits")) is False

    def test_matches_operator_invalid_regex_returns_false(self):
        """Invalid regex pattern returns None from execute_regex_with_timeout -> no match."""
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.MATCHES, "[invalid"
                )
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(subject="anything")) is False

    def test_matches_operator_regex_timeout(self):
        """Regex that times out returns None -> no match."""
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.SUBJECT, RuleConditionOperator.MATCHES, r"test")
            ]
        )
        with patch(f"{MODULE}.execute_regex_with_timeout", return_value=None):
            assert _evaluate_rule(rule, _FakeMessage(subject="test")) is False


# ============================================================================
# _evaluate_rule -- case insensitivity
# ============================================================================


class TestEvaluateRuleCaseInsensitivity:
    """Test that rule evaluation is case-insensitive."""

    def test_from_case_insensitive(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.FROM, RuleConditionOperator.CONTAINS, "USER")
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(from_address="user@test.com")) is True

    def test_subject_case_insensitive(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.SUBJECT, RuleConditionOperator.EQUALS, "URGENT")
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(subject="urgent")) is True

    def test_domain_case_insensitive(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SENDER_DOMAIN, RuleConditionOperator.EQUALS, "EXAMPLE.COM"
                )
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(from_address="u@example.com")) is True

    def test_to_case_insensitive(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.TO, RuleConditionOperator.CONTAINS, "INBOX")
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(to_addresses=["inbox@co.com"])) is True


# ============================================================================
# _evaluate_rule -- condition logic (AND / OR)
# ============================================================================


class TestEvaluateRuleConditionLogic:
    """Test AND/OR condition logic."""

    def test_and_logic_all_match(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "urgent"
                ),
                _make_condition(RuleConditionField.FROM, RuleConditionOperator.CONTAINS, "user"),
            ],
            condition_logic="AND",
        )
        msg = _FakeMessage(subject="Urgent issue", from_address="user@test.com")
        assert _evaluate_rule(rule, msg) is True

    def test_and_logic_one_fails(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "urgent"
                ),
                _make_condition(RuleConditionField.FROM, RuleConditionOperator.CONTAINS, "admin"),
            ],
            condition_logic="AND",
        )
        msg = _FakeMessage(subject="Urgent issue", from_address="user@test.com")
        assert _evaluate_rule(rule, msg) is False

    def test_or_logic_one_matches(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "urgent"
                ),
                _make_condition(RuleConditionField.FROM, RuleConditionOperator.CONTAINS, "admin"),
            ],
            condition_logic="OR",
        )
        msg = _FakeMessage(subject="Urgent issue", from_address="user@test.com")
        assert _evaluate_rule(rule, msg) is True

    def test_or_logic_none_match(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "billing"
                ),
                _make_condition(RuleConditionField.FROM, RuleConditionOperator.CONTAINS, "admin"),
            ],
            condition_logic="OR",
        )
        msg = _FakeMessage(subject="Hello", from_address="user@test.com")
        assert _evaluate_rule(rule, msg) is False

    def test_and_logic_empty_conditions_returns_false(self):
        rule = _make_rule(conditions=[], condition_logic="AND")
        assert _evaluate_rule(rule, _FakeMessage()) is False

    def test_or_logic_empty_conditions_returns_false(self):
        rule = _make_rule(conditions=[], condition_logic="OR")
        assert _evaluate_rule(rule, _FakeMessage()) is False


# ============================================================================
# _evaluate_rule -- edge cases
# ============================================================================


class TestEvaluateRuleEdgeCases:
    """Edge cases for rule evaluation."""

    def test_empty_subject(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "test")
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(subject="")) is False

    def test_empty_from_address(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.FROM, RuleConditionOperator.CONTAINS, "test")
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(from_address="")) is False

    def test_empty_to_addresses(self):
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.TO, RuleConditionOperator.CONTAINS, "test")
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(to_addresses=[])) is False

    def test_multiple_to_addresses_joined(self):
        """TO field joins all addresses with space."""
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.TO, RuleConditionOperator.CONTAINS, "second")
            ]
        )
        msg = _FakeMessage(to_addresses=["first@co.com", "second@co.com"])
        assert _evaluate_rule(rule, msg) is True

    def test_single_condition_matches(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "urgent"
                )
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(subject="Urgent")) is True

    def test_sender_domain_multiple_at_signs(self):
        """Multiple @ signs -- split('@')[-1] gets the last part."""
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SENDER_DOMAIN, RuleConditionOperator.EQUALS, "domain.com"
                )
            ]
        )
        msg = _FakeMessage(from_address="user@weird@domain.com")
        assert _evaluate_rule(rule, msg) is True

    def test_matches_with_dot_star(self):
        """Regex .* should match anything."""
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.SUBJECT, RuleConditionOperator.MATCHES, ".*")
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(subject="anything")) is True

    def test_matches_with_anchored_pattern(self):
        """Anchored regex ^...$ for exact matching."""
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.MATCHES, r"^hello$"
                )
            ]
        )
        assert _evaluate_rule(rule, _FakeMessage(subject="hello")) is True
        assert _evaluate_rule(rule, _FakeMessage(subject="hello world")) is False


# ============================================================================
# get_matching_rules_for_email -- basic tests
# ============================================================================


class TestGetMatchingRulesForEmail:
    """Test get_matching_rules_for_email."""

    @pytest.mark.asyncio
    async def test_no_rules_returns_empty(self):
        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={"from_address": "a@b.com", "to_addresses": [], "subject": "test"},
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_matching_rule_in_memory(self):
        rule = _make_rule(
            rule_id="r1",
            workspace_id="ws_1",
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "urgent"
                )
            ],
        )
        with _storage_lock:
            _routing_rules["r1"] = rule

        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={
                    "from_address": "user@test.com",
                    "to_addresses": ["inbox@co.com"],
                    "subject": "Urgent: help",
                    "snippet": "Please help",
                },
                workspace_id="ws_1",
            )
        assert len(result) == 1
        assert result[0]["id"] == "r1"

    @pytest.mark.asyncio
    async def test_non_matching_rule_excluded(self):
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "billing"
                )
            ],
        )
        with _storage_lock:
            _routing_rules["r1"] = rule

        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={"from_address": "a@b.com", "to_addresses": [], "subject": "hello"},
                workspace_id="ws_test",
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_disabled_rule_excluded(self):
        rule = _make_rule(enabled=False)
        with _storage_lock:
            _routing_rules["r1"] = rule

        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={"from_address": "a@b.com", "to_addresses": [], "subject": "Urgent"},
                workspace_id="ws_test",
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_workspace_filtering(self):
        rule_a = _make_rule(rule_id="r_a", workspace_id="ws_a")
        rule_b = _make_rule(rule_id="r_b", workspace_id="ws_b")
        with _storage_lock:
            _routing_rules["r_a"] = rule_a
            _routing_rules["r_b"] = rule_b

        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={
                    "from_address": "a@b.com",
                    "to_addresses": [],
                    "subject": "Urgent",
                },
                workspace_id="ws_a",
            )
        ids = [r["id"] for r in result]
        assert "r_a" in ids
        assert "r_b" not in ids

    @pytest.mark.asyncio
    async def test_workspace_none_returns_all(self):
        """When workspace_id is None, all rules are considered."""
        rule = _make_rule(rule_id="r1", workspace_id="ws_any")
        with _storage_lock:
            _routing_rules["r1"] = rule

        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={
                    "from_address": "a@b.com",
                    "to_addresses": [],
                    "subject": "Urgent",
                },
                workspace_id=None,
            )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_rules_sorted_by_priority(self):
        """Matching rules are sorted by priority ascending."""
        rule_high = _make_rule(rule_id="rh", priority=10)
        rule_low = _make_rule(rule_id="rl", priority=1)
        with _storage_lock:
            _routing_rules["rh"] = rule_high
            _routing_rules["rl"] = rule_low

        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={
                    "from_address": "a@b.com",
                    "to_addresses": [],
                    "subject": "Urgent",
                },
                workspace_id="ws_test",
            )
        assert len(result) == 2
        assert result[0]["priority"] <= result[1]["priority"]


# ============================================================================
# get_matching_rules_for_email -- store interactions
# ============================================================================


class TestGetMatchingRulesStoreInteraction:
    """Test store fallback logic in get_matching_rules_for_email."""

    @pytest.mark.asyncio
    async def test_rules_store_primary(self):
        """Uses RulesStore when available."""
        rules_store = MagicMock()
        rules_store.get_matching_rules.return_value = [
            {"id": "rs_1", "name": "Store Rule", "priority": 1}
        ]

        with patch(f"{MODULE}._get_rules_store", return_value=rules_store):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={"from_address": "a@b.com", "to_addresses": [], "subject": "test"},
                workspace_id="ws_1",
            )
        assert len(result) == 1
        assert result[0]["id"] == "rs_1"

    @pytest.mark.asyncio
    async def test_rules_store_failure_falls_back_to_memory(self):
        """Falls back to in-memory evaluation when RulesStore fails."""
        rules_store = MagicMock()
        rules_store.get_matching_rules.side_effect = RuntimeError("db down")

        rule = _make_rule(rule_id="mem_1")
        with _storage_lock:
            _routing_rules["mem_1"] = rule

        with patch(f"{MODULE}._get_rules_store", return_value=rules_store):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={
                    "from_address": "a@b.com",
                    "to_addresses": [],
                    "subject": "Urgent",
                },
                workspace_id="ws_test",
            )
        assert len(result) == 1
        assert result[0]["id"] == "mem_1"

    @pytest.mark.asyncio
    async def test_store_oserror_falls_back(self):
        rules_store = MagicMock()
        rules_store.get_matching_rules.side_effect = OSError("disk error")

        with patch(f"{MODULE}._get_rules_store", return_value=rules_store):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={"from_address": "a@b.com", "to_addresses": [], "subject": "test"},
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_store_valueerror_falls_back(self):
        rules_store = MagicMock()
        rules_store.get_matching_rules.side_effect = ValueError("bad data")

        with patch(f"{MODULE}._get_rules_store", return_value=rules_store):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={"from_address": "a@b.com", "to_addresses": [], "subject": "test"},
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_store_keyerror_falls_back(self):
        rules_store = MagicMock()
        rules_store.get_matching_rules.side_effect = KeyError("missing")

        with patch(f"{MODULE}._get_rules_store", return_value=rules_store):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={"from_address": "a@b.com", "to_addresses": [], "subject": "test"},
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_merge_store_and_memory_results(self):
        """When both store and memory return results, they are merged."""
        rules_store = MagicMock()
        rules_store.get_matching_rules.return_value = [
            {"id": "store_r1", "name": "Store Rule", "priority": 2}
        ]

        mem_rule = _make_rule(rule_id="mem_r1", priority=1)
        with _storage_lock:
            _routing_rules["mem_r1"] = mem_rule

        with patch(f"{MODULE}._get_rules_store", return_value=rules_store):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={
                    "from_address": "a@b.com",
                    "to_addresses": [],
                    "subject": "Urgent",
                },
                workspace_id="ws_test",
            )
        ids = {r["id"] for r in result}
        assert "store_r1" in ids
        assert "mem_r1" in ids

    @pytest.mark.asyncio
    async def test_merge_deduplicates_by_id(self):
        """Same rule ID in store and memory keeps the memory version in combined."""
        rules_store = MagicMock()
        rules_store.get_matching_rules.return_value = [
            {"id": "r1", "name": "Store Version", "priority": 5}
        ]

        mem_rule = _make_rule(rule_id="r1", priority=5, name="Mem Version")
        with _storage_lock:
            _routing_rules["r1"] = mem_rule

        with patch(f"{MODULE}._get_rules_store", return_value=rules_store):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={
                    "from_address": "a@b.com",
                    "to_addresses": [],
                    "subject": "Urgent",
                },
                workspace_id="ws_test",
            )
        # Memory version overwrites store version in combined dict
        assert len(result) == 1
        assert result[0]["name"] == "Mem Version"

    @pytest.mark.asyncio
    async def test_store_returns_results_no_memory_matches(self):
        """Store results returned when memory has no matching rules."""
        rules_store = MagicMock()
        rules_store.get_matching_rules.return_value = [
            {"id": "s1", "name": "Store Only", "priority": 1}
        ]

        with patch(f"{MODULE}._get_rules_store", return_value=rules_store):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={
                    "from_address": "a@b.com",
                    "to_addresses": [],
                    "subject": "No match",
                },
                workspace_id="ws_1",
            )
        assert len(result) == 1
        assert result[0]["id"] == "s1"

    @pytest.mark.asyncio
    async def test_merged_results_sorted_by_priority(self):
        """Merged results are sorted by priority."""
        rules_store = MagicMock()
        rules_store.get_matching_rules.return_value = [
            {"id": "store_r", "name": "S", "priority": 10}
        ]

        mem_rule = _make_rule(rule_id="mem_r", priority=1)
        with _storage_lock:
            _routing_rules["mem_r"] = mem_rule

        with patch(f"{MODULE}._get_rules_store", return_value=rules_store):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={
                    "from_address": "a@b.com",
                    "to_addresses": [],
                    "subject": "Urgent",
                },
                workspace_id="ws_test",
            )
        assert result[0]["priority"] <= result[1]["priority"]

    @pytest.mark.asyncio
    async def test_email_data_defaults(self):
        """Missing email_data keys default gracefully."""
        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={},
            )
        assert result == []


# ============================================================================
# apply_routing_rules_to_message -- action types
# ============================================================================


class TestApplyRoutingRulesActions:
    """Test apply_routing_rules_to_message action application."""

    @pytest.mark.asyncio
    async def test_no_matching_rules(self):
        msg = _make_inbox_message(subject="boring email")
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=[]),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is False
        assert result["rules_matched"] == 0
        assert result["actions"] == []

    @pytest.mark.asyncio
    async def test_assign_action(self):
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "assign", "target": "agent-42"}],
            }
        ]
        msg = _make_inbox_message(status=MessageStatus.OPEN)
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is True
        assert msg.assigned_to == "agent-42"
        assert msg.status == MessageStatus.ASSIGNED
        assert msg.assigned_at is not None
        assert {"type": "assign", "target": "agent-42"} in result["actions"]
        assert result["changes"]["assigned_to"] == "agent-42"

    @pytest.mark.asyncio
    async def test_assign_action_non_open_status_unchanged(self):
        """Assign does not change status if message is not OPEN."""
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "assign", "target": "agent-1"}],
            }
        ]
        msg = _make_inbox_message(status=MessageStatus.IN_PROGRESS)
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is True
        assert msg.assigned_to == "agent-1"
        assert msg.status == MessageStatus.IN_PROGRESS  # unchanged

    @pytest.mark.asyncio
    async def test_assign_without_target_skipped(self):
        """Assign action without target is skipped."""
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "assign", "target": None}],
            }
        ]
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is False
        assert msg.assigned_to is None

    @pytest.mark.asyncio
    async def test_label_action(self):
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "label", "target": "important"}],
            }
        ]
        msg = _make_inbox_message(tags=[])
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is True
        assert "important" in msg.tags
        assert {"type": "label", "target": "important"} in result["actions"]

    @pytest.mark.asyncio
    async def test_label_action_no_duplicate(self):
        """Label is not added if already present."""
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "label", "target": "existing"}],
            }
        ]
        msg = _make_inbox_message(tags=["existing"])
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert msg.tags.count("existing") == 1

    @pytest.mark.asyncio
    async def test_label_without_target_skipped(self):
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "label", "target": None}],
            }
        ]
        msg = _make_inbox_message(tags=[])
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert msg.tags == []

    @pytest.mark.asyncio
    async def test_escalate_action(self):
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "escalate"}],
            }
        ]
        msg = _make_inbox_message(priority="low")
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is True
        assert msg.priority == "high"
        assert result["changes"]["priority"] == "high"
        assert {"type": "escalate"} in result["actions"]

    @pytest.mark.asyncio
    async def test_archive_action(self):
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "archive"}],
            }
        ]
        msg = _make_inbox_message(status=MessageStatus.OPEN)
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is True
        assert msg.status == MessageStatus.CLOSED
        assert result["changes"]["status"] == "closed"
        assert {"type": "archive"} in result["actions"]

    @pytest.mark.asyncio
    async def test_multiple_actions_in_one_rule(self):
        matching = [
            {
                "id": "r1",
                "actions": [
                    {"type": "assign", "target": "agent-1"},
                    {"type": "label", "target": "vip"},
                    {"type": "escalate"},
                ],
            }
        ]
        msg = _make_inbox_message(status=MessageStatus.OPEN, tags=[])
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is True
        assert result["rules_matched"] == 1
        assert len(result["actions"]) == 3
        assert msg.assigned_to == "agent-1"
        assert "vip" in msg.tags
        assert msg.priority == "high"

    @pytest.mark.asyncio
    async def test_multiple_rules_applied_in_order(self):
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "label", "target": "first"}],
            },
            {
                "id": "r2",
                "actions": [{"type": "label", "target": "second"}],
            },
        ]
        msg = _make_inbox_message(tags=[])
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["rules_matched"] == 2
        assert "first" in msg.tags
        assert "second" in msg.tags

    @pytest.mark.asyncio
    async def test_unknown_action_type_ignored(self):
        """Unknown action types are silently ignored."""
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "teleport", "target": "mars"}],
            }
        ]
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is False
        assert result["actions"] == []

    @pytest.mark.asyncio
    async def test_action_params_reserved(self):
        """Action params are read but not currently used."""
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "assign", "target": "t1", "params": {"key": "val"}}],
            }
        ]
        msg = _make_inbox_message(status=MessageStatus.OPEN)
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is True


# ============================================================================
# apply_routing_rules_to_message -- rule stats
# ============================================================================


class TestApplyRoutingRulesStats:
    """Test rule stats increment in apply_routing_rules_to_message."""

    @pytest.mark.asyncio
    async def test_stats_incremented_per_rule(self):
        matching = [
            {"id": "r1", "actions": [{"type": "escalate"}]},
            {"id": "r2", "actions": [{"type": "escalate"}]},
        ]
        rules_store = MagicMock()
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=rules_store),
        ):
            await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert rules_store.increment_rule_stats.call_count == 2

    @pytest.mark.asyncio
    async def test_stats_increment_failure_ignored(self):
        matching = [{"id": "r1", "actions": [{"type": "escalate"}]}]
        rules_store = MagicMock()
        rules_store.increment_rule_stats.side_effect = RuntimeError("fail")
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=rules_store),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        # Should still succeed despite stats failure
        assert result["applied"] is True

    @pytest.mark.asyncio
    async def test_stats_oserror_ignored(self):
        matching = [{"id": "r1", "actions": [{"type": "escalate"}]}]
        rules_store = MagicMock()
        rules_store.increment_rule_stats.side_effect = OSError("disk")
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=rules_store),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is True

    @pytest.mark.asyncio
    async def test_stats_valueerror_ignored(self):
        matching = [{"id": "r1", "actions": [{"type": "escalate"}]}]
        rules_store = MagicMock()
        rules_store.increment_rule_stats.side_effect = ValueError("bad")
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=rules_store),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is True

    @pytest.mark.asyncio
    async def test_stats_keyerror_ignored(self):
        matching = [{"id": "r1", "actions": [{"type": "escalate"}]}]
        rules_store = MagicMock()
        rules_store.increment_rule_stats.side_effect = KeyError("missing")
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=rules_store),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is True

    @pytest.mark.asyncio
    async def test_no_stats_when_store_absent(self):
        matching = [{"id": "r1", "actions": [{"type": "escalate"}]}]
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is True


# ============================================================================
# apply_routing_rules_to_message -- email_data construction
# ============================================================================


class TestApplyRoutingRulesEmailData:
    """Test that email_data is properly constructed from the message."""

    @pytest.mark.asyncio
    async def test_email_data_fields_passed(self):
        msg = _make_inbox_message(
            from_address="sender@test.com",
            to_addresses=["to@test.com"],
            subject="Subject line",
            snippet="Body snippet",
            priority="medium",
        )
        captured_email_data = {}

        async def capture_matching(inbox_id, email_data, workspace_id=None):
            captured_email_data.update(email_data)
            return []

        with patch(f"{MODULE}.get_matching_rules_for_email", side_effect=capture_matching):
            await apply_routing_rules_to_message("inbox_1", msg, "ws_test")

        assert captured_email_data["from_address"] == "sender@test.com"
        assert captured_email_data["to_addresses"] == ["to@test.com"]
        assert captured_email_data["subject"] == "Subject line"
        assert captured_email_data["snippet"] == "Body snippet"
        assert captured_email_data["priority"] == "medium"


# ============================================================================
# evaluate_rule_for_test
# ============================================================================


class TestEvaluateRuleForTest:
    """Test evaluate_rule_for_test."""

    def test_no_messages_returns_zero(self):
        rule = _make_rule(workspace_id="ws_1")
        count = evaluate_rule_for_test(rule, workspace_id="ws_1")
        assert count == 0

    def test_matching_message_counted(self):
        inbox = _make_shared_inbox(inbox_id="inbox_1", workspace_id="ws_1")
        msg = _make_inbox_message(
            msg_id="msg_1",
            inbox_id="inbox_1",
            subject="Urgent request",
        )
        with _storage_lock:
            _shared_inboxes["inbox_1"] = inbox
            _inbox_messages["inbox_1"] = {"msg_1": msg}

        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "urgent"
                )
            ],
        )
        count = evaluate_rule_for_test(rule, workspace_id="ws_1")
        assert count == 1

    def test_non_matching_message_not_counted(self):
        inbox = _make_shared_inbox(inbox_id="inbox_1", workspace_id="ws_1")
        msg = _make_inbox_message(
            msg_id="msg_1",
            inbox_id="inbox_1",
            subject="Regular email",
        )
        with _storage_lock:
            _shared_inboxes["inbox_1"] = inbox
            _inbox_messages["inbox_1"] = {"msg_1": msg}

        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "urgent"
                )
            ],
        )
        count = evaluate_rule_for_test(rule, workspace_id="ws_1")
        assert count == 0

    def test_wrong_workspace_excluded(self):
        inbox = _make_shared_inbox(inbox_id="inbox_1", workspace_id="ws_other")
        msg = _make_inbox_message(
            msg_id="msg_1",
            inbox_id="inbox_1",
            subject="Urgent request",
        )
        with _storage_lock:
            _shared_inboxes["inbox_1"] = inbox
            _inbox_messages["inbox_1"] = {"msg_1": msg}

        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "urgent"
                )
            ],
        )
        count = evaluate_rule_for_test(rule, workspace_id="ws_1")
        assert count == 0

    def test_multiple_inboxes_multiple_messages(self):
        inbox_a = _make_shared_inbox(inbox_id="inbox_a", workspace_id="ws_1")
        inbox_b = _make_shared_inbox(inbox_id="inbox_b", workspace_id="ws_1")
        msg_a1 = _make_inbox_message(msg_id="a1", inbox_id="inbox_a", subject="Urgent A")
        msg_a2 = _make_inbox_message(msg_id="a2", inbox_id="inbox_a", subject="Normal")
        msg_b1 = _make_inbox_message(msg_id="b1", inbox_id="inbox_b", subject="Urgent B")

        with _storage_lock:
            _shared_inboxes["inbox_a"] = inbox_a
            _shared_inboxes["inbox_b"] = inbox_b
            _inbox_messages["inbox_a"] = {"a1": msg_a1, "a2": msg_a2}
            _inbox_messages["inbox_b"] = {"b1": msg_b1}

        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "urgent"
                )
            ],
        )
        count = evaluate_rule_for_test(rule, workspace_id="ws_1")
        assert count == 2

    def test_inbox_not_in_shared_inboxes_skipped(self):
        """Messages in inbox_messages without corresponding shared_inbox are skipped."""
        msg = _make_inbox_message(msg_id="msg_1", inbox_id="orphan_inbox", subject="Urgent")
        with _storage_lock:
            _inbox_messages["orphan_inbox"] = {"msg_1": msg}

        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "urgent"
                )
            ],
        )
        count = evaluate_rule_for_test(rule, workspace_id="ws_1")
        assert count == 0


# ============================================================================
# Security tests
# ============================================================================


class TestSecurityEdgeCases:
    """Security-related edge cases."""

    def test_regex_timeout_protection(self):
        """Regex that times out is handled gracefully."""
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.MATCHES, r"(a+)+$"
                )
            ]
        )
        # Simulate timeout by patching execute_regex_with_timeout
        with patch(f"{MODULE}.execute_regex_with_timeout", return_value=None):
            result = _evaluate_rule(rule, _FakeMessage(subject="aaaaaaaaaaaaaaaaaaaab"))
        assert result is False

    def test_path_traversal_in_from_address(self):
        """Path traversal attempts in from_address don't cause issues."""
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.FROM, RuleConditionOperator.CONTAINS, "../../../etc/passwd"
                )
            ]
        )
        msg = _FakeMessage(from_address="../../../etc/passwd@evil.com")
        # Should just do string matching, no file access
        result = _evaluate_rule(rule, msg)
        assert isinstance(result, bool)

    def test_null_bytes_in_subject(self):
        """Null bytes in subject are handled."""
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "test")
            ]
        )
        msg = _FakeMessage(subject="test\x00injection")
        result = _evaluate_rule(rule, msg)
        assert result is True

    def test_unicode_normalization(self):
        """Unicode characters don't break evaluation."""
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "cafe")
            ]
        )
        msg = _FakeMessage(subject="Welcome to the caf\u00e9")
        # "cafe" is not in "caf\u00e9" because \u00e9 != e
        result = _evaluate_rule(rule, msg)
        assert result is False

    def test_very_long_subject(self):
        """Very long subjects don't cause errors."""
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "needle"
                )
            ]
        )
        msg = _FakeMessage(subject="x" * 100000 + "needle")
        result = _evaluate_rule(rule, msg)
        assert result is True

    def test_very_long_condition_value(self):
        """Very long condition values don't cause errors."""
        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "x" * 100000
                )
            ]
        )
        msg = _FakeMessage(subject="short")
        result = _evaluate_rule(rule, msg)
        assert result is False

    def test_sql_injection_in_from(self):
        """SQL injection attempts are just string-matched."""
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.FROM, RuleConditionOperator.CONTAINS, "admin")
            ]
        )
        msg = _FakeMessage(from_address="'; DROP TABLE users; --@evil.com")
        result = _evaluate_rule(rule, msg)
        assert result is False

    def test_html_injection_in_subject(self):
        """HTML injection in subject is just string-matched."""
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "alert")
            ]
        )
        msg = _FakeMessage(subject="<script>alert('xss')</script>")
        result = _evaluate_rule(rule, msg)
        assert result is True

    @pytest.mark.asyncio
    async def test_injection_in_email_data_keys(self):
        """Extra keys in email_data don't affect evaluation."""
        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "test")
            ]
        )
        with _storage_lock:
            _routing_rules["r1"] = rule

        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await get_matching_rules_for_email(
                inbox_id="inbox_1",
                email_data={
                    "from_address": "a@b.com",
                    "to_addresses": [],
                    "subject": "test",
                    "__class__": "exploit",
                    "constructor": {"prototype": "attack"},
                },
                workspace_id="ws_test",
            )
        assert len(result) == 1


# ============================================================================
# Additional edge cases and integration-like tests
# ============================================================================


class TestIntegrationScenarios:
    """Integration-like test scenarios combining multiple functions."""

    @pytest.mark.asyncio
    async def test_full_pipeline_assign_and_label(self):
        """Full flow: rule matches email -> assigns + labels message."""
        rule = _make_rule(
            rule_id="r1",
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "support"
                )
            ],
            actions=[
                RuleAction(type=RuleActionType.ASSIGN, target="agent-support"),
                RuleAction(type=RuleActionType.LABEL, target="needs-attention"),
            ],
        )
        with _storage_lock:
            _routing_rules["r1"] = rule

        msg = _make_inbox_message(
            subject="Support request: login issue",
            status=MessageStatus.OPEN,
            tags=[],
        )

        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")

        assert result["applied"] is True
        assert msg.assigned_to == "agent-support"
        assert msg.status == MessageStatus.ASSIGNED
        assert "needs-attention" in msg.tags

    @pytest.mark.asyncio
    async def test_full_pipeline_escalate_and_archive(self):
        """Full flow: escalate then archive."""
        rule = _make_rule(
            rule_id="r1",
            conditions=[
                _make_condition(
                    RuleConditionField.FROM, RuleConditionOperator.ENDS_WITH, "spam.com"
                )
            ],
            actions=[
                RuleAction(type=RuleActionType.ESCALATE),
                RuleAction(type=RuleActionType.ARCHIVE),
            ],
        )
        with _storage_lock:
            _routing_rules["r1"] = rule

        msg = _make_inbox_message(
            from_address="bot@spam.com",
            status=MessageStatus.OPEN,
            priority="low",
        )

        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")

        assert result["applied"] is True
        assert msg.priority == "high"
        assert msg.status == MessageStatus.CLOSED

    @pytest.mark.asyncio
    async def test_rule_with_or_logic_partial_match(self):
        """OR logic matches when only one condition is true."""
        rule = _make_rule(
            rule_id="r1",
            conditions=[
                _make_condition(RuleConditionField.SUBJECT, RuleConditionOperator.CONTAINS, "vip"),
                _make_condition(RuleConditionField.PRIORITY, RuleConditionOperator.EQUALS, "high"),
            ],
            condition_logic="OR",
            actions=[RuleAction(type=RuleActionType.LABEL, target="important")],
        )
        with _storage_lock:
            _routing_rules["r1"] = rule

        msg = _make_inbox_message(
            subject="Regular subject",
            priority="high",
            tags=[],
        )

        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")

        assert result["applied"] is True
        assert "important" in msg.tags

    @pytest.mark.asyncio
    async def test_multiple_rules_different_priorities(self):
        """Multiple rules applied in priority order."""
        rule_low = _make_rule(
            rule_id="r_low",
            priority=1,
            actions=[RuleAction(type=RuleActionType.LABEL, target="first")],
        )
        rule_high = _make_rule(
            rule_id="r_high",
            priority=10,
            actions=[RuleAction(type=RuleActionType.LABEL, target="second")],
        )
        with _storage_lock:
            _routing_rules["r_low"] = rule_low
            _routing_rules["r_high"] = rule_high

        msg = _make_inbox_message(subject="Urgent: please help", tags=[])

        with patch(f"{MODULE}._get_rules_store", return_value=None):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")

        assert result["rules_matched"] == 2
        assert "first" in msg.tags
        assert "second" in msg.tags

    def test_evaluate_for_test_with_from_condition(self):
        """evaluate_rule_for_test works with FROM condition."""
        inbox = _make_shared_inbox(inbox_id="inbox_1", workspace_id="ws_1")
        msg = _make_inbox_message(
            msg_id="m1",
            inbox_id="inbox_1",
            from_address="vip@company.com",
        )
        with _storage_lock:
            _shared_inboxes["inbox_1"] = inbox
            _inbox_messages["inbox_1"] = {"m1": msg}

        rule = _make_rule(
            conditions=[
                _make_condition(RuleConditionField.FROM, RuleConditionOperator.CONTAINS, "vip")
            ]
        )
        count = evaluate_rule_for_test(rule, workspace_id="ws_1")
        assert count == 1

    def test_evaluate_for_test_with_regex_condition(self):
        """evaluate_rule_for_test works with MATCHES operator."""
        inbox = _make_shared_inbox(inbox_id="inbox_1", workspace_id="ws_1")
        msg = _make_inbox_message(
            msg_id="m1",
            inbox_id="inbox_1",
            subject="Ticket-12345 assigned",
        )
        with _storage_lock:
            _shared_inboxes["inbox_1"] = inbox
            _inbox_messages["inbox_1"] = {"m1": msg}

        rule = _make_rule(
            conditions=[
                _make_condition(
                    RuleConditionField.SUBJECT, RuleConditionOperator.MATCHES, r"ticket-\d+"
                )
            ]
        )
        count = evaluate_rule_for_test(rule, workspace_id="ws_1")
        assert count == 1

    @pytest.mark.asyncio
    async def test_actions_without_type_key_skipped(self):
        """Actions missing the 'type' key are silently skipped."""
        matching = [
            {
                "id": "r1",
                "actions": [{"target": "someone"}],
            }
        ]
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is False

    @pytest.mark.asyncio
    async def test_actions_empty_list(self):
        """Rule with empty actions list doesn't crash."""
        matching = [
            {
                "id": "r1",
                "actions": [],
            }
        ]
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is False

    @pytest.mark.asyncio
    async def test_rule_without_actions_key(self):
        """Rule dict missing 'actions' key defaults to empty list."""
        matching = [{"id": "r1"}]
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        assert result["applied"] is False

    @pytest.mark.asyncio
    async def test_action_missing_target_key(self):
        """Action dict missing 'target' key defaults to None."""
        matching = [
            {
                "id": "r1",
                "actions": [{"type": "assign"}],
            }
        ]
        msg = _make_inbox_message()
        with (
            patch(f"{MODULE}.get_matching_rules_for_email", return_value=matching),
            patch(f"{MODULE}._get_rules_store", return_value=None),
        ):
            result = await apply_routing_rules_to_message("inbox_1", msg, "ws_test")
        # assign with target=None is skipped
        assert result["applied"] is False
