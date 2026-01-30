"""Tests for the Rules Store service (SQLite-backed)."""

from __future__ import annotations

import os
import tempfile

import pytest

from aragora.services.rules_store import RulesStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def store(tmp_db):
    """Create a RulesStore backed by a temporary database."""
    s = RulesStore(db_path=tmp_db)
    yield s
    s.close()


def _make_rule_data(**overrides) -> dict:
    defaults = {
        "id": "rule_1",
        "name": "Test Rule",
        "workspace_id": "ws_1",
        "conditions": [{"field": "from", "operator": "contains", "value": "boss"}],
        "actions": [{"action": "tag", "value": "important"}],
        "priority": 5,
        "enabled": True,
    }
    defaults.update(overrides)
    return defaults


def _make_inbox_data(**overrides) -> dict:
    defaults = {
        "id": "inbox_1",
        "workspace_id": "ws_1",
        "name": "Support Inbox",
        "description": "Customer support",
        "email_address": "support@example.com",
    }
    defaults.update(overrides)
    return defaults


def _make_message_data(**overrides) -> dict:
    defaults = {
        "id": "msg_1",
        "inbox_id": "inbox_1",
        "email_id": "email_123",
        "subject": "Help needed",
        "from_address": "customer@example.com",
        "to_addresses": ["support@example.com"],
        "received_at": "2025-07-01T10:00:00",
        "status": "open",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestRulesStoreInit:
    def test_creates_database(self, tmp_db):
        store = RulesStore(db_path=tmp_db)
        assert os.path.exists(tmp_db)
        store.close()

    def test_schema_initialized(self, store):
        assert store._initialized is True


# ---------------------------------------------------------------------------
# Routing Rules CRUD
# ---------------------------------------------------------------------------


class TestRulesCRUD:
    def test_create_rule(self, store):
        data = _make_rule_data()
        result = store.create_rule(data)
        assert result["id"] == "rule_1"
        assert result["name"] == "Test Rule"

    def test_get_rule(self, store):
        store.create_rule(_make_rule_data())
        rule = store.get_rule("rule_1")
        assert rule is not None
        assert rule["name"] == "Test Rule"
        assert rule["workspace_id"] == "ws_1"
        assert rule["enabled"] is True

    def test_get_rule_not_found(self, store):
        rule = store.get_rule("nonexistent")
        assert rule is None

    def test_update_rule(self, store):
        store.create_rule(_make_rule_data())
        updated = store.update_rule("rule_1", {"name": "Updated Rule"})
        assert updated is not None
        assert updated["name"] == "Updated Rule"

    def test_update_rule_not_found(self, store):
        result = store.update_rule("nonexistent", {"name": "X"})
        assert result is None

    def test_delete_rule(self, store):
        store.create_rule(_make_rule_data())
        deleted = store.delete_rule("rule_1")
        assert deleted is True
        assert store.get_rule("rule_1") is None

    def test_delete_rule_not_found(self, store):
        deleted = store.delete_rule("nonexistent")
        assert deleted is False

    def test_list_rules_all(self, store):
        store.create_rule(_make_rule_data(id="r1"))
        store.create_rule(_make_rule_data(id="r2", name="Rule 2"))
        rules = store.list_rules()
        assert len(rules) == 2

    def test_list_rules_by_workspace(self, store):
        store.create_rule(_make_rule_data(id="r1", workspace_id="ws_1"))
        store.create_rule(_make_rule_data(id="r2", workspace_id="ws_2"))
        rules = store.list_rules(workspace_id="ws_1")
        assert len(rules) == 1
        assert rules[0]["workspace_id"] == "ws_1"

    def test_list_rules_enabled_only(self, store):
        store.create_rule(_make_rule_data(id="r1", enabled=True))
        store.create_rule(_make_rule_data(id="r2", enabled=False))
        rules = store.list_rules(enabled_only=True)
        assert len(rules) == 1

    def test_count_rules(self, store):
        store.create_rule(_make_rule_data(id="r1"))
        store.create_rule(_make_rule_data(id="r2"))
        assert store.count_rules() == 2

    def test_count_rules_by_workspace(self, store):
        store.create_rule(_make_rule_data(id="r1", workspace_id="ws_1"))
        store.create_rule(_make_rule_data(id="r2", workspace_id="ws_2"))
        assert store.count_rules(workspace_id="ws_1") == 1

    def test_increment_rule_stats(self, store):
        store.create_rule(_make_rule_data())
        store.increment_rule_stats("rule_1", matched=3, applied=2)
        rule = store.get_rule("rule_1")
        assert rule["stats"]["matched"] == 3
        assert rule["stats"]["applied"] == 2

    def test_increment_rule_stats_accumulates(self, store):
        store.create_rule(_make_rule_data())
        store.increment_rule_stats("rule_1", matched=3)
        store.increment_rule_stats("rule_1", matched=2)
        rule = store.get_rule("rule_1")
        assert rule["stats"]["matched"] == 5

    def test_rule_conditions_stored_as_json(self, store):
        conditions = [
            {"field": "from", "operator": "contains", "value": "boss"},
            {"field": "subject", "operator": "contains", "value": "urgent"},
        ]
        store.create_rule(_make_rule_data(conditions=conditions))
        rule = store.get_rule("rule_1")
        assert len(rule["conditions"]) == 2
        assert rule["conditions"][0]["field"] == "from"


# ---------------------------------------------------------------------------
# Shared Inboxes CRUD
# ---------------------------------------------------------------------------


class TestInboxCRUD:
    def test_create_inbox(self, store):
        data = _make_inbox_data()
        result = store.create_inbox(data)
        assert result["id"] == "inbox_1"

    def test_get_inbox(self, store):
        store.create_inbox(_make_inbox_data())
        inbox = store.get_inbox("inbox_1")
        assert inbox is not None
        assert inbox["name"] == "Support Inbox"

    def test_get_inbox_not_found(self, store):
        assert store.get_inbox("nonexistent") is None

    def test_update_inbox(self, store):
        store.create_inbox(_make_inbox_data())
        updated = store.update_inbox("inbox_1", {"name": "VIP Support"})
        assert updated["name"] == "VIP Support"

    def test_update_inbox_not_found(self, store):
        result = store.update_inbox("nonexistent", {"name": "X"})
        assert result is None

    def test_delete_inbox(self, store):
        store.create_inbox(_make_inbox_data())
        deleted = store.delete_inbox("inbox_1")
        assert deleted is True
        assert store.get_inbox("inbox_1") is None

    def test_list_inboxes_all(self, store):
        store.create_inbox(_make_inbox_data(id="i1"))
        store.create_inbox(_make_inbox_data(id="i2", name="Sales"))
        inboxes = store.list_inboxes()
        assert len(inboxes) == 2

    def test_list_inboxes_by_workspace(self, store):
        store.create_inbox(_make_inbox_data(id="i1", workspace_id="ws_1"))
        store.create_inbox(_make_inbox_data(id="i2", workspace_id="ws_2"))
        inboxes = store.list_inboxes(workspace_id="ws_1")
        assert len(inboxes) == 1


# ---------------------------------------------------------------------------
# Inbox Messages CRUD
# ---------------------------------------------------------------------------


class TestMessageCRUD:
    def _setup_inbox(self, store):
        store.create_inbox(_make_inbox_data())

    def test_create_message(self, store):
        self._setup_inbox(store)
        data = _make_message_data()
        result = store.create_message(data)
        assert result["id"] == "msg_1"

    def test_get_message(self, store):
        self._setup_inbox(store)
        store.create_message(_make_message_data())
        msg = store.get_message("msg_1")
        assert msg is not None
        assert msg["subject"] == "Help needed"

    def test_get_message_not_found(self, store):
        assert store.get_message("nonexistent") is None

    def test_update_message(self, store):
        self._setup_inbox(store)
        store.create_message(_make_message_data())
        updated = store.update_message("msg_1", {"status": "resolved"})
        assert updated["status"] == "resolved"

    def test_update_message_not_found(self, store):
        result = store.update_message("nonexistent", {"status": "x"})
        assert result is None

    def test_list_messages(self, store):
        self._setup_inbox(store)
        store.create_message(_make_message_data(id="m1"))
        store.create_message(_make_message_data(id="m2", subject="Another"))
        msgs = store.list_messages("inbox_1")
        assert len(msgs) == 2

    def test_list_messages_by_status(self, store):
        self._setup_inbox(store)
        store.create_message(_make_message_data(id="m1", status="open"))
        store.create_message(_make_message_data(id="m2", status="resolved"))
        msgs = store.list_messages("inbox_1", status="open")
        assert len(msgs) == 1

    def test_message_increments_inbox_count(self, store):
        self._setup_inbox(store)
        store.create_message(_make_message_data())
        inbox = store.get_inbox("inbox_1")
        assert inbox["message_count"] == 1


# ---------------------------------------------------------------------------
# Rule Matching / Evaluation
# ---------------------------------------------------------------------------


class TestRuleMatching:
    def _setup_workspace_with_rule(self, store, conditions, condition_logic="AND", **kw):
        rule_data = _make_rule_data(
            conditions=conditions,
            condition_logic=condition_logic,
            **kw,
        )
        store.create_rule(rule_data)

    def test_match_contains(self, store):
        self._setup_workspace_with_rule(
            store,
            conditions=[{"field": "from", "operator": "contains", "value": "boss"}],
        )
        email_data = {"from_address": "boss@company.com", "subject": "Hello"}
        matches = store.get_matching_rules("inbox_1", email_data, workspace_id="ws_1")
        assert len(matches) == 1

    def test_no_match(self, store):
        self._setup_workspace_with_rule(
            store,
            conditions=[{"field": "from", "operator": "contains", "value": "ceo"}],
        )
        email_data = {"from_address": "intern@company.com"}
        matches = store.get_matching_rules("inbox_1", email_data, workspace_id="ws_1")
        assert len(matches) == 0

    def test_match_equals(self, store):
        self._setup_workspace_with_rule(
            store,
            conditions=[{"field": "subject", "operator": "equals", "value": "urgent"}],
        )
        email_data = {"subject": "Urgent"}
        matches = store.get_matching_rules("inbox_1", email_data, workspace_id="ws_1")
        assert len(matches) == 1

    def test_match_starts_with(self, store):
        self._setup_workspace_with_rule(
            store,
            conditions=[{"field": "subject", "operator": "starts_with", "value": "re:"}],
        )
        email_data = {"subject": "RE: Follow up"}
        matches = store.get_matching_rules("inbox_1", email_data, workspace_id="ws_1")
        assert len(matches) == 1

    def test_match_regex(self, store):
        self._setup_workspace_with_rule(
            store,
            conditions=[{"field": "subject", "operator": "matches", "value": r"ticket[-#]\d+"}],
        )
        email_data = {"subject": "RE: ticket-12345 update"}
        matches = store.get_matching_rules("inbox_1", email_data, workspace_id="ws_1")
        assert len(matches) == 1

    def test_match_and_logic(self, store):
        self._setup_workspace_with_rule(
            store,
            conditions=[
                {"field": "from", "operator": "contains", "value": "boss"},
                {"field": "subject", "operator": "contains", "value": "urgent"},
            ],
            condition_logic="AND",
        )
        # Only from matches, not subject
        email_data = {"from_address": "boss@company.com", "subject": "Hello"}
        matches = store.get_matching_rules("inbox_1", email_data, workspace_id="ws_1")
        assert len(matches) == 0

    def test_match_or_logic(self, store):
        self._setup_workspace_with_rule(
            store,
            conditions=[
                {"field": "from", "operator": "contains", "value": "boss"},
                {"field": "subject", "operator": "contains", "value": "urgent"},
            ],
            condition_logic="OR",
        )
        # Only from matches
        email_data = {"from_address": "boss@company.com", "subject": "Hello"}
        matches = store.get_matching_rules("inbox_1", email_data, workspace_id="ws_1")
        assert len(matches) == 1

    def test_match_sender_domain(self, store):
        self._setup_workspace_with_rule(
            store,
            conditions=[{"field": "sender_domain", "operator": "equals", "value": "vip.com"}],
        )
        email_data = {"from_address": "ceo@vip.com"}
        matches = store.get_matching_rules("inbox_1", email_data, workspace_id="ws_1")
        assert len(matches) == 1

    def test_empty_conditions_no_match(self, store):
        self._setup_workspace_with_rule(store, conditions=[])
        email_data = {"from_address": "anyone@example.com"}
        matches = store.get_matching_rules("inbox_1", email_data, workspace_id="ws_1")
        assert len(matches) == 0


# ---------------------------------------------------------------------------
# Close / cleanup
# ---------------------------------------------------------------------------


class TestRulesStoreClose:
    def test_close(self, store):
        store.close()
        # After close, connection should be None
        assert getattr(store._local, "connection", None) is None
