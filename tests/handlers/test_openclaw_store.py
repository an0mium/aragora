"""Tests for OpenClaw Gateway in-memory store.

Covers:
- Session CRUD (create, get, list, update, delete)
- Action CRUD (create, get, update)
- Credential management (store, get, list, delete, rotate)
- Audit log (add, query with filters)
- Metrics computation
- Session filtering by user, tenant, status
- Session idle timeout / cleanup
- Credential filtering by type
- Audit log eviction
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest


from aragora.server.handlers.openclaw.store import OpenClawGatewayStore
from aragora.server.handlers.openclaw.models import (
    ActionStatus,
    CredentialType,
    SessionStatus,
)


# ============================================================================
# Session CRUD
# ============================================================================


class TestSessionCRUD:
    """Test session create/read/update/delete."""

    def test_create_session(self):
        store = OpenClawGatewayStore()
        session = store.create_session(user_id="user1", tenant_id="tenant1")
        assert session.id is not None
        assert session.user_id == "user1"
        assert session.tenant_id == "tenant1"
        assert session.status == SessionStatus.ACTIVE

    def test_get_session(self):
        store = OpenClawGatewayStore()
        created = store.create_session(user_id="user1")
        fetched = store.get_session(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    def test_get_session_not_found(self):
        store = OpenClawGatewayStore()
        assert store.get_session("nonexistent") is None

    def test_update_session_status(self):
        store = OpenClawGatewayStore()
        session = store.create_session(user_id="user1")
        updated = store.update_session_status(session.id, SessionStatus.CLOSED)
        assert updated is not None
        assert updated.status == SessionStatus.CLOSED

    def test_update_session_not_found(self):
        store = OpenClawGatewayStore()
        result = store.update_session_status("nonexistent", SessionStatus.CLOSED)
        assert result is None

    def test_delete_session(self):
        store = OpenClawGatewayStore()
        session = store.create_session(user_id="user1")
        assert store.delete_session(session.id) is True
        assert store.get_session(session.id) is None

    def test_delete_session_not_found(self):
        store = OpenClawGatewayStore()
        assert store.delete_session("nonexistent") is False

    def test_create_session_with_metadata(self):
        store = OpenClawGatewayStore()
        session = store.create_session(
            user_id="user1",
            config={"model": "gpt-4"},
            metadata={"source": "api"},
        )
        assert session.config == {"model": "gpt-4"}
        assert session.metadata == {"source": "api"}


# ============================================================================
# Session Listing & Filtering
# ============================================================================


class TestSessionListing:
    """Test session listing with filters."""

    def test_list_all_sessions(self):
        store = OpenClawGatewayStore()
        store.create_session(user_id="user1")
        store.create_session(user_id="user2")
        sessions, total = store.list_sessions()
        assert total == 2
        assert len(sessions) == 2

    def test_list_filter_by_user(self):
        store = OpenClawGatewayStore()
        store.create_session(user_id="user1")
        store.create_session(user_id="user2")
        store.create_session(user_id="user1")
        sessions, total = store.list_sessions(user_id="user1")
        assert total == 2

    def test_list_filter_by_tenant(self):
        store = OpenClawGatewayStore()
        store.create_session(user_id="user1", tenant_id="t1")
        store.create_session(user_id="user2", tenant_id="t2")
        sessions, total = store.list_sessions(tenant_id="t1")
        assert total == 1
        assert sessions[0].tenant_id == "t1"

    def test_list_filter_by_status(self):
        store = OpenClawGatewayStore()
        s1 = store.create_session(user_id="user1")
        store.create_session(user_id="user2")
        store.update_session_status(s1.id, SessionStatus.CLOSED)
        sessions, total = store.list_sessions(status=SessionStatus.ACTIVE)
        assert total == 1

    def test_list_pagination(self):
        store = OpenClawGatewayStore()
        for i in range(5):
            store.create_session(user_id=f"user{i}")
        sessions, total = store.list_sessions(limit=2, offset=0)
        assert total == 5
        assert len(sessions) == 2

    def test_list_empty(self):
        store = OpenClawGatewayStore()
        sessions, total = store.list_sessions()
        assert total == 0
        assert sessions == []


# ============================================================================
# Session Expiration
# ============================================================================


class TestSessionExpiration:
    """Test session idle timeout cleanup."""

    def test_cleanup_expired_sessions(self):
        store = OpenClawGatewayStore(session_idle_timeout=60)
        session = store.create_session(user_id="user1")
        # Manually make session appear old
        session.last_activity_at = datetime.now(timezone.utc) - timedelta(seconds=120)
        closed = store.cleanup_expired_sessions()
        assert closed == 1
        assert store.get_session(session.id).status == SessionStatus.CLOSED

    def test_cleanup_skips_active_sessions(self):
        store = OpenClawGatewayStore(session_idle_timeout=3600)
        store.create_session(user_id="user1")
        closed = store.cleanup_expired_sessions()
        assert closed == 0

    def test_cleanup_disabled_with_zero_timeout(self):
        store = OpenClawGatewayStore(session_idle_timeout=0)
        session = store.create_session(user_id="user1")
        session.last_activity_at = datetime.now(timezone.utc) - timedelta(days=30)
        closed = store.cleanup_expired_sessions()
        assert closed == 0

    def test_cleanup_skips_already_closed(self):
        store = OpenClawGatewayStore(session_idle_timeout=60)
        session = store.create_session(user_id="user1")
        store.update_session_status(session.id, SessionStatus.CLOSED)
        session.last_activity_at = datetime.now(timezone.utc) - timedelta(seconds=120)
        closed = store.cleanup_expired_sessions()
        assert closed == 0


# ============================================================================
# Action CRUD
# ============================================================================


class TestActionCRUD:
    """Test action create/read/update."""

    def test_create_action(self):
        store = OpenClawGatewayStore()
        session = store.create_session(user_id="user1")
        action = store.create_action(
            session_id=session.id,
            action_type="search",
            input_data={"query": "test"},
        )
        assert action.id is not None
        assert action.session_id == session.id
        assert action.action_type == "search"
        assert action.status == ActionStatus.PENDING

    def test_get_action(self):
        store = OpenClawGatewayStore()
        session = store.create_session(user_id="user1")
        created = store.create_action(
            session_id=session.id,
            action_type="search",
            input_data={},
        )
        fetched = store.get_action(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    def test_get_action_not_found(self):
        store = OpenClawGatewayStore()
        assert store.get_action("nonexistent") is None

    def test_update_action_status(self):
        store = OpenClawGatewayStore()
        session = store.create_session(user_id="user1")
        action = store.create_action(
            session_id=session.id,
            action_type="search",
            input_data={},
        )
        updated = store.update_action(action.id, status=ActionStatus.RUNNING)
        assert updated.status == ActionStatus.RUNNING
        assert updated.started_at is not None

    def test_update_action_completed(self):
        store = OpenClawGatewayStore()
        session = store.create_session(user_id="user1")
        action = store.create_action(
            session_id=session.id,
            action_type="search",
            input_data={},
        )
        store.update_action(action.id, status=ActionStatus.RUNNING)
        updated = store.update_action(
            action.id,
            status=ActionStatus.COMPLETED,
            output_data={"results": [1, 2, 3]},
        )
        assert updated.status == ActionStatus.COMPLETED
        assert updated.completed_at is not None
        assert updated.output_data == {"results": [1, 2, 3]}

    def test_update_action_failed(self):
        store = OpenClawGatewayStore()
        session = store.create_session(user_id="user1")
        action = store.create_action(
            session_id=session.id,
            action_type="search",
            input_data={},
        )
        updated = store.update_action(
            action.id,
            status=ActionStatus.FAILED,
            error="Timeout",
        )
        assert updated.status == ActionStatus.FAILED
        assert updated.error == "Timeout"

    def test_update_action_not_found(self):
        store = OpenClawGatewayStore()
        result = store.update_action("nonexistent", status=ActionStatus.RUNNING)
        assert result is None


# ============================================================================
# Credential Management
# ============================================================================


class TestCredentialManagement:
    """Test credential CRUD and rotation."""

    def test_store_credential(self):
        store = OpenClawGatewayStore()
        cred = store.store_credential(
            name="my-api-key",
            credential_type=CredentialType.API_KEY,
            secret_value="sk-test-123",
            user_id="user1",
        )
        assert cred.id is not None
        assert cred.name == "my-api-key"
        assert cred.credential_type == CredentialType.API_KEY

    def test_get_credential(self):
        store = OpenClawGatewayStore()
        created = store.store_credential(
            name="token",
            credential_type=CredentialType.API_KEY,
            secret_value="secret",
            user_id="user1",
        )
        fetched = store.get_credential(created.id)
        assert fetched is not None
        assert fetched.name == "token"

    def test_get_credential_not_found(self):
        store = OpenClawGatewayStore()
        assert store.get_credential("nonexistent") is None

    def test_list_credentials(self):
        store = OpenClawGatewayStore()
        store.store_credential(
            name="key1",
            credential_type=CredentialType.API_KEY,
            secret_value="s1",
            user_id="user1",
        )
        store.store_credential(
            name="key2",
            credential_type=CredentialType.API_KEY,
            secret_value="s2",
            user_id="user2",
        )
        creds, total = store.list_credentials()
        assert total == 2

    def test_list_credentials_filter_by_user(self):
        store = OpenClawGatewayStore()
        store.store_credential(
            name="k1",
            credential_type=CredentialType.API_KEY,
            secret_value="s1",
            user_id="user1",
        )
        store.store_credential(
            name="k2",
            credential_type=CredentialType.API_KEY,
            secret_value="s2",
            user_id="user2",
        )
        creds, total = store.list_credentials(user_id="user1")
        assert total == 1

    def test_delete_credential(self):
        store = OpenClawGatewayStore()
        cred = store.store_credential(
            name="tmp",
            credential_type=CredentialType.API_KEY,
            secret_value="s",
            user_id="user1",
        )
        assert store.delete_credential(cred.id) is True
        assert store.get_credential(cred.id) is None

    def test_delete_credential_not_found(self):
        store = OpenClawGatewayStore()
        assert store.delete_credential("nonexistent") is False

    def test_rotate_credential(self):
        store = OpenClawGatewayStore()
        cred = store.store_credential(
            name="key",
            credential_type=CredentialType.API_KEY,
            secret_value="old_secret",
            user_id="user1",
        )
        rotated = store.rotate_credential(cred.id, "new_secret")
        assert rotated is not None
        assert rotated.last_rotated_at is not None
        # Verify new secret stored
        assert store._credential_secrets[cred.id] == "new_secret"

    def test_rotate_credential_not_found(self):
        store = OpenClawGatewayStore()
        result = store.rotate_credential("nonexistent", "new")
        assert result is None


# ============================================================================
# Audit Log
# ============================================================================


class TestAuditLog:
    """Test audit log operations."""

    def test_add_audit_entry(self):
        store = OpenClawGatewayStore()
        entry = store.add_audit_entry(
            action="session.create",
            actor_id="user1",
            resource_type="session",
            resource_id="s1",
        )
        assert entry.id is not None
        assert entry.action == "session.create"
        assert entry.result == "success"

    def test_get_audit_log(self):
        store = OpenClawGatewayStore()
        store.add_audit_entry(
            action="session.create",
            actor_id="user1",
            resource_type="session",
        )
        store.add_audit_entry(
            action="action.run",
            actor_id="user2",
            resource_type="action",
        )
        entries, total = store.get_audit_log()
        assert total == 2

    def test_get_audit_log_filter_by_action(self):
        store = OpenClawGatewayStore()
        store.add_audit_entry(
            action="session.create",
            actor_id="user1",
            resource_type="session",
        )
        store.add_audit_entry(
            action="action.run",
            actor_id="user1",
            resource_type="action",
        )
        entries, total = store.get_audit_log(action="session.create")
        assert total == 1

    def test_get_audit_log_filter_by_actor(self):
        store = OpenClawGatewayStore()
        store.add_audit_entry(
            action="a",
            actor_id="user1",
            resource_type="session",
        )
        store.add_audit_entry(
            action="b",
            actor_id="user2",
            resource_type="session",
        )
        entries, total = store.get_audit_log(actor_id="user1")
        assert total == 1

    def test_audit_log_eviction(self):
        store = OpenClawGatewayStore()
        # Add more than 10000 entries
        for i in range(10002):
            store.add_audit_entry(
                action="bulk",
                actor_id="bot",
                resource_type="test",
            )
        assert len(store._audit_log) == 10000


# ============================================================================
# Metrics
# ============================================================================


class TestMetrics:
    """Test gateway metrics computation."""

    def test_empty_metrics(self):
        store = OpenClawGatewayStore()
        metrics = store.get_metrics()
        assert metrics["sessions"]["total"] == 0
        assert metrics["sessions"]["active"] == 0
        assert metrics["actions"]["total"] == 0
        assert metrics["credentials"]["total"] == 0
        assert metrics["audit_log_entries"] == 0

    def test_metrics_with_data(self):
        store = OpenClawGatewayStore()
        s1 = store.create_session(user_id="user1")
        s2 = store.create_session(user_id="user2")
        store.update_session_status(s2.id, SessionStatus.CLOSED)
        store.create_action(session_id=s1.id, action_type="search", input_data={})
        store.store_credential(
            name="k",
            credential_type=CredentialType.API_KEY,
            secret_value="s",
            user_id="user1",
        )
        store.add_audit_entry(action="test", actor_id="user1", resource_type="test")

        metrics = store.get_metrics()
        assert metrics["sessions"]["total"] == 2
        assert metrics["sessions"]["active"] == 1
        assert metrics["actions"]["total"] == 1
        assert metrics["actions"]["pending"] == 1
        assert metrics["credentials"]["total"] == 1
        assert metrics["audit_log_entries"] == 1
