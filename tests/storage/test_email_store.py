"""Comprehensive tests for EmailStore.

Covers:
- User email configuration CRUD
- VIP sender management
- Shared inbox CRUD
- Shared inbox message management (save, list, status, tags, stats)
- Routing rule CRUD and match tracking
- Prioritization decision audit trail and feedback
- Singleton lifecycle
"""

import json
import pytest
import tempfile
from pathlib import Path

from aragora.storage.email_store import EmailStore, get_email_store, reset_email_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_email_store.db")


@pytest.fixture
def store(temp_db_path):
    """Create a fresh EmailStore for testing."""
    reset_email_store()
    s = EmailStore(temp_db_path)
    yield s
    reset_email_store()


@pytest.fixture
def store_with_inbox(store):
    """Store with a pre-created shared inbox."""
    store.create_shared_inbox(
        inbox_id="inbox_1",
        workspace_id="ws_1",
        name="Support",
        description="Main support inbox",
        email_address="support@example.com",
        members=["user_a", "user_b"],
        settings={"auto_assign": True},
    )
    return store


@pytest.fixture
def store_with_messages(store_with_inbox):
    """Store with inbox and several messages."""
    for i in range(5):
        store_with_inbox.save_message(
            message_id=f"msg_{i}",
            inbox_id="inbox_1",
            workspace_id="ws_1",
            subject=f"Subject {i}",
            from_address=f"sender{i}@example.com",
            snippet=f"Snippet body text {i}",
            status="open" if i % 2 == 0 else "assigned",
            priority="high" if i == 0 else "normal",
            assigned_to="agent_x" if i % 2 == 1 else None,
            tags=["urgent"] if i == 0 else [],
            metadata={"thread_id": f"t_{i}"},
            external_id=f"ext_{i}",
        )
    return store_with_inbox


# ===========================================================================
# User Email Configurations
# ===========================================================================


class TestUserConfig:
    """Tests for user email configuration CRUD."""

    def test_save_and_get_config(self, store):
        """Save a config and retrieve it by user/workspace."""
        config = {"vip_domains": ["ceo.com"], "tier_1_threshold": 0.9}
        config_id = store.save_user_config("user_1", "ws_1", config)

        assert config_id == "cfg_user_1_ws_1"

        retrieved = store.get_user_config("user_1", "ws_1")
        assert retrieved is not None
        assert retrieved["vip_domains"] == ["ceo.com"]
        assert retrieved["tier_1_threshold"] == 0.9

    def test_get_config_not_found(self, store):
        """Retrieving a non-existent config returns None."""
        assert store.get_user_config("no_user", "no_ws") is None

    def test_upsert_config(self, store):
        """Saving a config for the same user/workspace updates it."""
        store.save_user_config("user_1", "ws_1", {"version": 1})
        store.save_user_config("user_1", "ws_1", {"version": 2})

        config = store.get_user_config("user_1", "ws_1")
        assert config["version"] == 2

    def test_delete_config(self, store):
        """Delete an existing config returns True; missing returns False."""
        store.save_user_config("user_1", "ws_1", {"a": 1})
        assert store.delete_user_config("user_1", "ws_1") is True
        assert store.get_user_config("user_1", "ws_1") is None
        assert store.delete_user_config("user_1", "ws_1") is False

    def test_list_workspace_configs(self, store):
        """List all configs within a workspace."""
        store.save_user_config("user_a", "ws_1", {"a": 1})
        store.save_user_config("user_b", "ws_1", {"b": 2})
        store.save_user_config("user_c", "ws_2", {"c": 3})

        ws1_configs = store.list_workspace_configs("ws_1")
        assert len(ws1_configs) == 2
        user_ids = {c["user_id"] for c in ws1_configs}
        assert user_ids == {"user_a", "user_b"}

        ws2_configs = store.list_workspace_configs("ws_2")
        assert len(ws2_configs) == 1
        assert ws2_configs[0]["config"]["c"] == 3

    def test_list_workspace_configs_empty(self, store):
        """Listing configs for an empty workspace returns an empty list."""
        assert store.list_workspace_configs("ws_empty") == []


# ===========================================================================
# VIP Senders
# ===========================================================================


class TestVIPSenders:
    """Tests for VIP sender management."""

    def test_add_and_get_vip_sender(self, store):
        """Add a VIP sender and retrieve the list."""
        store.add_vip_sender(
            "user_1", "ws_1", "CEO@BigCo.com", sender_name="The CEO", notes="Important"
        )

        vips = store.get_vip_senders("user_1", "ws_1")
        assert len(vips) == 1
        # Email should be lowercased on storage
        assert vips[0]["sender_email"] == "ceo@bigco.com"
        assert vips[0]["sender_name"] == "The CEO"
        assert vips[0]["notes"] == "Important"

    def test_is_vip_sender(self, store):
        """Check VIP status for known and unknown senders."""
        store.add_vip_sender("user_1", "ws_1", "VIP@domain.com")

        assert store.is_vip_sender("user_1", "ws_1", "vip@domain.com") is True
        assert store.is_vip_sender("user_1", "ws_1", "VIP@DOMAIN.COM") is True
        assert store.is_vip_sender("user_1", "ws_1", "nobody@domain.com") is False

    def test_is_vip_sender_wrong_user(self, store):
        """VIP status is scoped to user and workspace."""
        store.add_vip_sender("user_1", "ws_1", "vip@domain.com")

        # Different user
        assert store.is_vip_sender("user_2", "ws_1", "vip@domain.com") is False
        # Different workspace
        assert store.is_vip_sender("user_1", "ws_2", "vip@domain.com") is False

    def test_remove_vip_sender(self, store):
        """Remove a VIP sender."""
        store.add_vip_sender("user_1", "ws_1", "vip@domain.com")
        assert store.remove_vip_sender("user_1", "ws_1", "vip@domain.com") is True
        assert store.is_vip_sender("user_1", "ws_1", "vip@domain.com") is False

    def test_remove_vip_sender_not_found(self, store):
        """Removing a non-existent VIP returns False."""
        assert store.remove_vip_sender("user_1", "ws_1", "nobody@x.com") is False

    def test_add_vip_sender_upsert(self, store):
        """Adding the same VIP email again updates name/notes (upsert)."""
        store.add_vip_sender("user_1", "ws_1", "vip@domain.com", sender_name="V1")
        store.add_vip_sender("user_1", "ws_1", "vip@domain.com", sender_name="V2", notes="Updated")

        vips = store.get_vip_senders("user_1", "ws_1")
        assert len(vips) == 1
        assert vips[0]["sender_name"] == "V2"
        assert vips[0]["notes"] == "Updated"


# ===========================================================================
# Shared Inboxes
# ===========================================================================


class TestSharedInboxes:
    """Tests for shared inbox CRUD."""

    def test_create_and_get_inbox(self, store):
        """Create a shared inbox and retrieve it."""
        store.create_shared_inbox(
            inbox_id="inbox_1",
            workspace_id="ws_1",
            name="Sales",
            description="Sales inbox",
            email_address="sales@co.com",
            members=["alice", "bob"],
            settings={"notify": True},
        )

        inbox = store.get_shared_inbox("inbox_1")
        assert inbox is not None
        assert inbox["name"] == "Sales"
        assert inbox["description"] == "Sales inbox"
        assert inbox["email_address"] == "sales@co.com"
        assert inbox["members"] == ["alice", "bob"]
        assert inbox["settings"]["notify"] is True

    def test_get_inbox_not_found(self, store):
        """Getting a non-existent inbox returns None."""
        assert store.get_shared_inbox("nope") is None

    def test_create_inbox_defaults(self, store):
        """Create inbox with minimal params uses defaults for members/settings."""
        store.create_shared_inbox("inbox_min", "ws_1", "Minimal")

        inbox = store.get_shared_inbox("inbox_min")
        assert inbox["members"] == []
        assert inbox["settings"] == {}
        assert inbox["description"] is None
        assert inbox["email_address"] is None

    def test_list_shared_inboxes(self, store):
        """List inboxes for a workspace."""
        store.create_shared_inbox("inbox_a", "ws_1", "A")
        store.create_shared_inbox("inbox_b", "ws_1", "B")
        store.create_shared_inbox("inbox_c", "ws_2", "C")

        ws1 = store.list_shared_inboxes("ws_1")
        assert len(ws1) == 2

        ws2 = store.list_shared_inboxes("ws_2")
        assert len(ws2) == 1
        assert ws2[0]["name"] == "C"

    def test_list_shared_inboxes_by_member(self, store):
        """Filter inboxes by member_id."""
        store.create_shared_inbox("inbox_a", "ws_1", "A", members=["alice", "bob"])
        store.create_shared_inbox("inbox_b", "ws_1", "B", members=["alice"])
        store.create_shared_inbox("inbox_c", "ws_1", "C", members=["charlie"])

        alice_inboxes = store.list_shared_inboxes("ws_1", member_id="alice")
        assert len(alice_inboxes) == 2

        charlie_inboxes = store.list_shared_inboxes("ws_1", member_id="charlie")
        assert len(charlie_inboxes) == 1
        assert charlie_inboxes[0]["id"] == "inbox_c"

    def test_update_shared_inbox(self, store_with_inbox):
        """Update allowed fields on a shared inbox."""
        updated = store_with_inbox.update_shared_inbox(
            "inbox_1",
            name="Updated Support",
            description="New description",
            members=["user_a", "user_b", "user_c"],
            settings={"auto_assign": False, "escalation": True},
        )
        assert updated is True

        inbox = store_with_inbox.get_shared_inbox("inbox_1")
        assert inbox["name"] == "Updated Support"
        assert inbox["description"] == "New description"
        assert len(inbox["members"]) == 3
        assert inbox["settings"]["escalation"] is True

    def test_update_shared_inbox_ignores_unknown_fields(self, store_with_inbox):
        """Unknown fields in update are silently ignored."""
        result = store_with_inbox.update_shared_inbox("inbox_1", unknown_field="val")
        assert result is False  # no valid fields -> no update

    def test_update_nonexistent_inbox(self, store):
        """Updating a non-existent inbox returns False."""
        assert store.update_shared_inbox("nope", name="X") is False

    def test_delete_shared_inbox(self, store_with_inbox):
        """Delete removes the inbox."""
        assert store_with_inbox.delete_shared_inbox("inbox_1") is True
        assert store_with_inbox.get_shared_inbox("inbox_1") is None

    def test_delete_shared_inbox_not_found(self, store):
        """Deleting a non-existent inbox returns False."""
        assert store.delete_shared_inbox("nope") is False


# ===========================================================================
# Shared Inbox Messages
# ===========================================================================


class TestMessages:
    """Tests for shared inbox message management."""

    def test_save_and_get_message(self, store_with_inbox):
        """Save a message and retrieve it by ID."""
        store_with_inbox.save_message(
            message_id="msg_1",
            inbox_id="inbox_1",
            workspace_id="ws_1",
            subject="Hello",
            from_address="alice@example.com",
            snippet="Hi there",
            status="open",
            priority="high",
            tags=["vip"],
            metadata={"thread": "t1"},
            external_id="ext_1",
        )

        msg = store_with_inbox.get_message("msg_1")
        assert msg is not None
        assert msg["subject"] == "Hello"
        assert msg["from_address"] == "alice@example.com"
        assert msg["status"] == "open"
        assert msg["priority"] == "high"
        assert msg["tags"] == ["vip"]
        assert msg["metadata"]["thread"] == "t1"
        assert msg["external_id"] == "ext_1"

    def test_get_message_not_found(self, store):
        """Getting a non-existent message returns None."""
        assert store.get_message("nope") is None

    def test_save_message_upsert(self, store_with_inbox):
        """Saving with the same message_id updates fields."""
        store_with_inbox.save_message("msg_u", "inbox_1", "ws_1", subject="V1")
        store_with_inbox.save_message("msg_u", "inbox_1", "ws_1", subject="V2", priority="high")

        msg = store_with_inbox.get_message("msg_u")
        assert msg["subject"] == "V2"
        assert msg["priority"] == "high"

    def test_list_inbox_messages(self, store_with_messages):
        """List all messages in an inbox."""
        msgs = store_with_messages.list_inbox_messages("inbox_1")
        assert len(msgs) == 5

    def test_list_inbox_messages_filter_status(self, store_with_messages):
        """Filter messages by status."""
        open_msgs = store_with_messages.list_inbox_messages("inbox_1", status="open")
        # msg_0, msg_2, msg_4 are open
        assert len(open_msgs) == 3

        assigned_msgs = store_with_messages.list_inbox_messages("inbox_1", status="assigned")
        assert len(assigned_msgs) == 2

    def test_list_inbox_messages_filter_assigned_to(self, store_with_messages):
        """Filter messages by assignee."""
        agent_msgs = store_with_messages.list_inbox_messages("inbox_1", assigned_to="agent_x")
        # msg_1, msg_3 are assigned to agent_x
        assert len(agent_msgs) == 2

    def test_list_inbox_messages_pagination(self, store_with_messages):
        """Paginate through messages."""
        page1 = store_with_messages.list_inbox_messages("inbox_1", limit=2, offset=0)
        page2 = store_with_messages.list_inbox_messages("inbox_1", limit=2, offset=2)
        page3 = store_with_messages.list_inbox_messages("inbox_1", limit=2, offset=4)

        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1

        all_ids = [m["id"] for m in page1 + page2 + page3]
        assert len(set(all_ids)) == 5  # no duplicates

    def test_update_message_status(self, store_with_messages):
        """Update status of a message."""
        result = store_with_messages.update_message_status("msg_0", "in_progress")
        assert result is True

        msg = store_with_messages.get_message("msg_0")
        assert msg["status"] == "in_progress"

    def test_update_message_status_with_assignee(self, store_with_messages):
        """Update status and assignee in one call."""
        store_with_messages.update_message_status("msg_0", "assigned", assigned_to="agent_y")

        msg = store_with_messages.get_message("msg_0")
        assert msg["status"] == "assigned"
        assert msg["assigned_to"] == "agent_y"

    def test_update_message_status_nonexistent(self, store):
        """Updating a non-existent message returns False."""
        assert store.update_message_status("nope", "closed") is False

    def test_add_message_tag(self, store_with_messages):
        """Add a tag to a message."""
        result = store_with_messages.add_message_tag("msg_1", "billing")
        assert result is True

        msg = store_with_messages.get_message("msg_1")
        assert "billing" in msg["tags"]

    def test_add_message_tag_idempotent(self, store_with_messages):
        """Adding a duplicate tag does not create duplicates."""
        store_with_messages.add_message_tag("msg_0", "urgent")  # already has "urgent"
        msg = store_with_messages.get_message("msg_0")
        assert msg["tags"].count("urgent") == 1

    def test_add_message_tag_nonexistent_message(self, store):
        """Adding a tag to a non-existent message returns False."""
        assert store.add_message_tag("nope", "tag") is False

    def test_get_inbox_stats(self, store_with_messages):
        """Get aggregate stats for a shared inbox."""
        # msg_0, msg_2, msg_4 are open; msg_1, msg_3 are assigned
        stats = store_with_messages.get_inbox_stats("inbox_1")

        assert stats["total"] == 5
        assert stats["open"] == 3
        assert stats["assigned"] == 2
        assert stats["in_progress"] == 0
        assert stats["resolved"] == 0

    def test_get_inbox_stats_empty(self, store_with_inbox):
        """Stats for an inbox with no messages returns all zeros."""
        stats = store_with_inbox.get_inbox_stats("inbox_1")
        assert stats["total"] == 0
        assert stats["open"] == 0

    def test_get_inbox_stats_after_status_update(self, store_with_messages):
        """Stats reflect status changes."""
        store_with_messages.update_message_status("msg_0", "resolved")
        stats = store_with_messages.get_inbox_stats("inbox_1")

        assert stats["open"] == 2
        assert stats["resolved"] == 1


# ===========================================================================
# Routing Rules
# ===========================================================================


class TestRoutingRules:
    """Tests for routing rule CRUD and match tracking."""

    def _make_rule(self, store, rule_id="rule_1", **overrides):
        """Helper to create a routing rule with defaults."""
        defaults = dict(
            rule_id=rule_id,
            workspace_id="ws_1",
            name="Auto-assign VIP",
            conditions=[{"field": "from_domain", "op": "equals", "value": "vip.com"}],
            actions=[{"type": "assign", "agent_id": "agent_1"}],
            description="Route VIP emails",
            inbox_id="inbox_1",
            priority=10,
            enabled=True,
        )
        defaults.update(overrides)
        return store.create_routing_rule(**defaults)

    def test_create_and_get_rule(self, store):
        """Create a routing rule and retrieve it."""
        self._make_rule(store)

        rule = store.get_routing_rule("rule_1")
        assert rule is not None
        assert rule["name"] == "Auto-assign VIP"
        assert rule["priority"] == 10
        assert rule["enabled"] is True
        assert len(rule["conditions"]) == 1
        assert rule["conditions"][0]["field"] == "from_domain"
        assert len(rule["actions"]) == 1
        assert rule["match_count"] == 0
        assert rule["last_matched_at"] is None

    def test_get_rule_not_found(self, store):
        """Getting a non-existent rule returns None."""
        assert store.get_routing_rule("nope") is None

    def test_list_routing_rules(self, store):
        """List rules for a workspace, ordered by priority DESC."""
        self._make_rule(store, rule_id="r_low", priority=1, name="Low")
        self._make_rule(store, rule_id="r_high", priority=100, name="High")
        self._make_rule(store, rule_id="r_mid", priority=50, name="Mid")

        rules = store.list_routing_rules("ws_1")
        assert len(rules) == 3
        # Highest priority first
        assert rules[0]["name"] == "High"
        assert rules[1]["name"] == "Mid"
        assert rules[2]["name"] == "Low"

    def test_list_routing_rules_by_inbox(self, store):
        """Filter rules by inbox_id (includes rules with NULL inbox_id)."""
        self._make_rule(store, rule_id="r_inbox", inbox_id="inbox_1")
        self._make_rule(store, rule_id="r_global", inbox_id=None)
        self._make_rule(store, rule_id="r_other", inbox_id="inbox_2")

        rules = store.list_routing_rules("ws_1", inbox_id="inbox_1")
        rule_ids = {r["id"] for r in rules}
        assert "r_inbox" in rule_ids
        assert "r_global" in rule_ids
        assert "r_other" not in rule_ids

    def test_list_routing_rules_enabled_only(self, store):
        """Filter to only enabled rules."""
        self._make_rule(store, rule_id="r_on", enabled=True)
        self._make_rule(store, rule_id="r_off", enabled=False)

        enabled_rules = store.list_routing_rules("ws_1", enabled_only=True)
        assert len(enabled_rules) == 1
        assert enabled_rules[0]["id"] == "r_on"

    def test_update_routing_rule(self, store):
        """Update various fields on a routing rule."""
        self._make_rule(store)

        updated = store.update_routing_rule(
            "rule_1",
            name="Updated Rule",
            priority=99,
            enabled=False,
            conditions=[{"field": "subject", "op": "contains", "value": "urgent"}],
        )
        assert updated is True

        rule = store.get_routing_rule("rule_1")
        assert rule["name"] == "Updated Rule"
        assert rule["priority"] == 99
        assert rule["enabled"] is False
        assert rule["conditions"][0]["field"] == "subject"

    def test_update_routing_rule_no_valid_fields(self, store):
        """Updating with only unknown fields returns False."""
        self._make_rule(store)
        assert store.update_routing_rule("rule_1", bad_field="x") is False

    def test_delete_routing_rule(self, store):
        """Delete a routing rule."""
        self._make_rule(store)
        assert store.delete_routing_rule("rule_1") is True
        assert store.get_routing_rule("rule_1") is None

    def test_delete_routing_rule_not_found(self, store):
        """Deleting a non-existent rule returns False."""
        assert store.delete_routing_rule("nope") is False

    def test_increment_rule_match_count(self, store):
        """Incrementing match count updates count and last_matched_at."""
        self._make_rule(store)

        store.increment_rule_match_count("rule_1")
        store.increment_rule_match_count("rule_1")
        store.increment_rule_match_count("rule_1")

        rule = store.get_routing_rule("rule_1")
        assert rule["match_count"] == 3
        assert rule["last_matched_at"] is not None


# ===========================================================================
# Prioritization Decision Audit Trail
# ===========================================================================


class TestPrioritizationDecisions:
    """Tests for prioritization decision recording and feedback."""

    def test_record_decision(self, store):
        """Record a prioritization decision with all fields."""
        dec_id = store.record_prioritization_decision(
            decision_id="dec_1",
            user_id="user_1",
            workspace_id="ws_1",
            email_id="email_100",
            tier_used=1,
            priority="high",
            confidence=0.92,
            score=8.5,
            rationale="VIP sender detected",
            factors={"vip_boost": 3.0, "recency": 1.5},
            context_boosts={"calendar_overlap": 0.5},
        )
        assert dec_id == "dec_1"

        # Verify via direct fetch
        row = store.fetch_one(
            "SELECT * FROM prioritization_decisions WHERE id = ?",
            ("dec_1",),
        )
        assert row is not None
        assert row[1] == "user_1"  # user_id
        assert row[4] == 1  # tier_used
        assert row[5] == "high"  # priority
        assert abs(row[6] - 0.92) < 0.001  # confidence
        assert abs(row[7] - 8.5) < 0.001  # score

        factors = json.loads(row[9])
        assert factors["vip_boost"] == 3.0

    def test_record_decision_minimal(self, store):
        """Record a decision with only required fields."""
        store.record_prioritization_decision(
            decision_id="dec_min",
            user_id="user_1",
            workspace_id="ws_1",
            email_id="email_50",
            tier_used=3,
            priority="low",
            confidence=0.4,
            score=2.0,
        )

        row = store.fetch_one(
            "SELECT rationale, factors_json, context_boosts_json FROM prioritization_decisions WHERE id = ?",
            ("dec_min",),
        )
        assert row[0] is None  # rationale
        assert row[1] is None  # factors_json
        assert row[2] is None  # context_boosts_json

    def test_record_user_feedback(self, store):
        """Record user feedback on a prioritization decision."""
        store.record_prioritization_decision(
            "dec_fb", "user_1", "ws_1", "email_1", 1, "high", 0.9, 8.0
        )

        result = store.record_user_feedback("email_1", "user_1", "ws_1", is_correct=True)
        assert result is True

        row = store.fetch_one(
            "SELECT user_feedback FROM prioritization_decisions WHERE id = ?",
            ("dec_fb",),
        )
        assert row[0] == 1  # True -> 1

    def test_record_negative_feedback(self, store):
        """Record negative feedback."""
        store.record_prioritization_decision(
            "dec_neg", "user_1", "ws_1", "email_2", 2, "normal", 0.6, 5.0
        )

        store.record_user_feedback("email_2", "user_1", "ws_1", is_correct=False)

        row = store.fetch_one(
            "SELECT user_feedback FROM prioritization_decisions WHERE id = ?",
            ("dec_neg",),
        )
        assert row[0] == 0  # False -> 0

    def test_feedback_stats_empty(self, store):
        """Feedback stats for a user with no feedback returns zeros."""
        stats = store.get_feedback_stats("user_1", "ws_1")
        assert stats["total_feedback"] == 0
        assert stats["accuracy"] == 0.0
        assert stats["by_tier"] == {}
        assert stats["by_priority"] == {}

    def test_feedback_stats_with_data(self, store):
        """Feedback stats aggregate correctly across tiers and priorities."""
        # Record several decisions with feedback
        decisions = [
            ("d1", "email_1", 1, "high", 0.9, 8.0, True),
            ("d2", "email_2", 1, "high", 0.85, 7.5, True),
            ("d3", "email_3", 1, "high", 0.7, 6.0, False),
            ("d4", "email_4", 2, "normal", 0.6, 5.0, True),
            ("d5", "email_5", 2, "normal", 0.5, 4.0, False),
        ]

        for dec_id, email_id, tier, priority, conf, score, correct in decisions:
            store.record_prioritization_decision(
                dec_id, "user_1", "ws_1", email_id, tier, priority, conf, score
            )
            store.record_user_feedback(email_id, "user_1", "ws_1", is_correct=correct)

        stats = store.get_feedback_stats("user_1", "ws_1", days=30)

        assert stats["total_feedback"] == 5
        # 3 correct out of 5
        assert abs(stats["accuracy"] - 0.6) < 0.001

        # Tier 1: 2 correct out of 3
        assert stats["by_tier"][1]["total"] == 3
        assert stats["by_tier"][1]["correct"] == 2

        # Tier 2: 1 correct out of 2
        assert stats["by_tier"][2]["total"] == 2
        assert stats["by_tier"][2]["correct"] == 1

        # Priority breakdown
        assert stats["by_priority"]["high"]["total"] == 3
        assert stats["by_priority"]["normal"]["total"] == 2


# ===========================================================================
# Singleton Lifecycle
# ===========================================================================


class TestSingleton:
    """Tests for get_email_store / reset_email_store singleton."""

    def test_get_email_store_creates_instance(self, temp_db_path):
        """get_email_store creates an instance when none exists."""
        reset_email_store()
        s = get_email_store(db_path=temp_db_path)
        assert isinstance(s, EmailStore)
        reset_email_store()

    def test_get_email_store_returns_same_instance(self, temp_db_path):
        """get_email_store returns the same singleton on repeated calls."""
        reset_email_store()
        s1 = get_email_store(db_path=temp_db_path)
        s2 = get_email_store(db_path=temp_db_path)
        assert s1 is s2
        reset_email_store()

    def test_reset_clears_singleton(self, temp_db_path):
        """reset_email_store allows a new instance to be created."""
        reset_email_store()
        s1 = get_email_store(db_path=temp_db_path)
        reset_email_store()

        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = str(Path(tmpdir) / "new_email_store.db")
            s2 = get_email_store(db_path=new_path)
            assert s1 is not s2
            reset_email_store()


# ===========================================================================
# Edge Cases and Error Paths
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_message_with_no_optional_fields(self, store_with_inbox):
        """Save a message with only required fields."""
        store_with_inbox.save_message(
            message_id="msg_bare",
            inbox_id="inbox_1",
            workspace_id="ws_1",
        )

        msg = store_with_inbox.get_message("msg_bare")
        assert msg is not None
        assert msg["subject"] is None
        assert msg["from_address"] is None
        assert msg["tags"] == []
        assert msg["metadata"] == {}
        assert msg["status"] == "open"
        assert msg["priority"] == "normal"

    def test_config_with_complex_json(self, store):
        """Configs can store arbitrarily nested JSON."""
        complex_config = {
            "rules": [
                {"domain": "a.com", "priority": 1},
                {"domain": "b.com", "priority": 2},
            ],
            "nested": {"deep": {"value": [1, 2, 3]}},
        }
        store.save_user_config("user_1", "ws_1", complex_config)

        retrieved = store.get_user_config("user_1", "ws_1")
        assert retrieved["rules"][1]["domain"] == "b.com"
        assert retrieved["nested"]["deep"]["value"] == [1, 2, 3]

    def test_fts_escape_query_strips_dangerous_chars(self, store):
        """FTS query escaping removes all dangerous characters."""
        escaped = store._escape_fts_query("test*query^with:special(chars)")
        for char in ["*", "^", ":", "(", ")"]:
            assert char not in escaped
        # Should produce quoted terms
        assert '"test"' in escaped
        assert '"query"' in escaped

    def test_fts_escape_empty_query(self, store):
        """FTS escaping of an empty string returns empty."""
        assert store._escape_fts_query("") == ""

    def test_multiple_workspaces_isolated(self, store):
        """Data across different workspaces is isolated."""
        store.save_user_config("user_1", "ws_a", {"workspace": "a"})
        store.save_user_config("user_1", "ws_b", {"workspace": "b"})

        config_a = store.get_user_config("user_1", "ws_a")
        config_b = store.get_user_config("user_1", "ws_b")

        assert config_a["workspace"] == "a"
        assert config_b["workspace"] == "b"

    def test_schema_version(self, store):
        """Store reports correct schema version after initialization."""
        version = store.get_schema_version()
        assert version == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
