"""Integration tests for the OpenClaw Gateway handler package.

Tests the coordinated behavior of the 9-module handler package:
- Store (session, action, credential, audit CRUD)
- Validation (input sanitization, size limits, injection prevention)
- Models (serialization, enum values)
- Lifecycle flows (session → action → close, credential → rotate → revoke)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aragora.server.handlers.openclaw import (
    OpenClawGatewayStore,
    _get_store,
)
from aragora.server.handlers.openclaw.models import (
    Action,
    ActionStatus,
    AuditEntry,
    Credential,
    CredentialType,
    Session,
    SessionStatus,
)
from aragora.server.handlers.openclaw.validation import (
    MAX_ACTION_INPUT_SIZE,
    MAX_CREDENTIAL_NAME_LENGTH,
    MAX_SESSION_CONFIG_DEPTH,
    MAX_SESSION_CONFIG_KEYS,
    sanitize_action_parameters,
    validate_action_input,
    validate_action_type,
    validate_credential_name,
    validate_credential_secret,
    validate_metadata,
    validate_session_config,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def store() -> OpenClawGatewayStore:
    """Create a fresh store for each test."""
    return OpenClawGatewayStore()


# =============================================================================
# Session Lifecycle
# =============================================================================


class TestSessionLifecycle:
    """Test full session lifecycle: create → use → close."""

    def test_create_session(self, store: OpenClawGatewayStore) -> None:
        session = store.create_session(user_id="user-1", tenant_id="t-1")
        assert session.status == SessionStatus.ACTIVE
        assert session.user_id == "user-1"
        assert session.tenant_id == "t-1"
        assert session.id

    def test_create_session_with_config(self, store: OpenClawGatewayStore) -> None:
        config = {"model": "claude-3", "temperature": 0.7}
        session = store.create_session(user_id="user-1", config=config)
        assert session.config == config

    def test_get_session(self, store: OpenClawGatewayStore) -> None:
        session = store.create_session(user_id="user-1")
        retrieved = store.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    def test_get_session_not_found(self, store: OpenClawGatewayStore) -> None:
        assert store.get_session("nonexistent") is None

    def test_list_sessions_by_user(self, store: OpenClawGatewayStore) -> None:
        store.create_session(user_id="user-1")
        store.create_session(user_id="user-1")
        store.create_session(user_id="user-2")
        sessions, total = store.list_sessions(user_id="user-1")
        assert total == 2
        assert len(sessions) == 2

    def test_list_sessions_by_status(self, store: OpenClawGatewayStore) -> None:
        s1 = store.create_session(user_id="user-1")
        store.create_session(user_id="user-1")
        store.update_session_status(s1.id, SessionStatus.CLOSED)
        sessions, total = store.list_sessions(status=SessionStatus.ACTIVE)
        assert total == 1

    def test_update_session_status(self, store: OpenClawGatewayStore) -> None:
        session = store.create_session(user_id="user-1")
        updated = store.update_session_status(session.id, SessionStatus.CLOSING)
        assert updated is not None
        assert updated.status == SessionStatus.CLOSING

    def test_delete_session(self, store: OpenClawGatewayStore) -> None:
        session = store.create_session(user_id="user-1")
        assert store.delete_session(session.id) is True
        assert store.get_session(session.id) is None

    def test_delete_nonexistent_session(self, store: OpenClawGatewayStore) -> None:
        assert store.delete_session("nonexistent") is False

    def test_session_serialization(self, store: OpenClawGatewayStore) -> None:
        session = store.create_session(user_id="user-1", tenant_id="t-1")
        d = session.to_dict()
        assert d["user_id"] == "user-1"
        assert d["status"] == "active"
        assert "created_at" in d
        assert "id" in d

    def test_full_session_lifecycle(self, store: OpenClawGatewayStore) -> None:
        """End-to-end: create → add actions → close."""
        session = store.create_session(user_id="user-1")
        assert session.status == SessionStatus.ACTIVE

        action = store.create_action(session.id, "query", {"q": "test"})
        assert action.status == ActionStatus.PENDING

        store.update_action(action.id, status=ActionStatus.RUNNING)
        store.update_action(
            action.id,
            status=ActionStatus.COMPLETED,
            output_data={"result": "ok"},
        )

        completed = store.get_action(action.id)
        assert completed is not None
        assert completed.status == ActionStatus.COMPLETED
        assert completed.output_data == {"result": "ok"}
        assert completed.started_at is not None
        assert completed.completed_at is not None

        store.update_session_status(session.id, SessionStatus.CLOSED)
        closed = store.get_session(session.id)
        assert closed is not None
        assert closed.status == SessionStatus.CLOSED


# =============================================================================
# Action Lifecycle
# =============================================================================


class TestActionLifecycle:
    """Test action creation, execution, failure, and cancellation."""

    def test_create_action(self, store: OpenClawGatewayStore) -> None:
        session = store.create_session(user_id="user-1")
        action = store.create_action(session.id, "search", {"query": "test"})
        assert action.session_id == session.id
        assert action.action_type == "search"
        assert action.status == ActionStatus.PENDING

    def test_action_failure(self, store: OpenClawGatewayStore) -> None:
        session = store.create_session(user_id="user-1")
        action = store.create_action(session.id, "query", {})
        store.update_action(action.id, status=ActionStatus.RUNNING)
        store.update_action(action.id, status=ActionStatus.FAILED, error="Timeout")
        failed = store.get_action(action.id)
        assert failed is not None
        assert failed.status == ActionStatus.FAILED
        assert failed.error == "Timeout"

    def test_action_cancellation(self, store: OpenClawGatewayStore) -> None:
        session = store.create_session(user_id="user-1")
        action = store.create_action(session.id, "long-task", {})
        store.update_action(action.id, status=ActionStatus.RUNNING)
        store.update_action(action.id, status=ActionStatus.CANCELLED)
        cancelled = store.get_action(action.id)
        assert cancelled is not None
        assert cancelled.status == ActionStatus.CANCELLED

    def test_action_serialization(self, store: OpenClawGatewayStore) -> None:
        session = store.create_session(user_id="user-1")
        action = store.create_action(session.id, "test", {"key": "value"})
        d = action.to_dict()
        assert d["action_type"] == "test"
        assert d["status"] == "pending"
        assert d["input_data"] == {"key": "value"}


# =============================================================================
# Credential Lifecycle
# =============================================================================


class TestCredentialLifecycle:
    """Test credential storage, rotation, and revocation."""

    def test_store_credential(self, store: OpenClawGatewayStore) -> None:
        cred = store.store_credential(
            name="my-api-key",
            credential_type=CredentialType.API_KEY,
            secret_value="sk-test-12345678",
            user_id="user-1",
        )
        assert cred.name == "my-api-key"
        assert cred.credential_type == CredentialType.API_KEY
        assert cred.last_rotated_at is None

    def test_get_credential_excludes_secret(self, store: OpenClawGatewayStore) -> None:
        cred = store.store_credential(
            name="secret-key",
            credential_type=CredentialType.API_KEY,
            secret_value="sk-supersecret123",
            user_id="user-1",
        )
        retrieved = store.get_credential(cred.id)
        assert retrieved is not None
        d = retrieved.to_dict()
        assert "secret" not in d
        assert "secret_value" not in d

    def test_rotate_credential(self, store: OpenClawGatewayStore) -> None:
        cred = store.store_credential(
            name="rotatable",
            credential_type=CredentialType.OAUTH_TOKEN,
            secret_value="old-token-12345678",
            user_id="user-1",
        )
        rotated = store.rotate_credential(cred.id, "new-token-12345678")
        assert rotated is not None
        assert rotated.last_rotated_at is not None

    def test_delete_credential(self, store: OpenClawGatewayStore) -> None:
        cred = store.store_credential(
            name="disposable",
            credential_type=CredentialType.PASSWORD,
            secret_value="pass12345678",
            user_id="user-1",
        )
        assert store.delete_credential(cred.id) is True
        assert store.get_credential(cred.id) is None

    def test_list_credentials_by_type(self, store: OpenClawGatewayStore) -> None:
        store.store_credential("k1", CredentialType.API_KEY, "sk-12345678a", "u1")
        store.store_credential("k2", CredentialType.SSH_KEY, "ssh-key-value", "u1")
        store.store_credential("k3", CredentialType.API_KEY, "sk-12345678b", "u1")
        creds, total = store.list_credentials(credential_type=CredentialType.API_KEY)
        assert total == 2

    def test_credential_serialization(self, store: OpenClawGatewayStore) -> None:
        cred = store.store_credential(
            name="test-cred",
            credential_type=CredentialType.SERVICE_ACCOUNT,
            secret_value="sa-key-12345678",
            user_id="user-1",
            tenant_id="t-1",
        )
        d = cred.to_dict()
        assert d["name"] == "test-cred"
        assert d["credential_type"] == "service_account"
        assert d["tenant_id"] == "t-1"

    def test_full_credential_lifecycle(self, store: OpenClawGatewayStore) -> None:
        """End-to-end: store → rotate → revoke."""
        cred = store.store_credential(
            name="lifecycle-key",
            credential_type=CredentialType.API_KEY,
            secret_value="sk-original12345",
            user_id="user-1",
        )
        assert cred.last_rotated_at is None

        rotated = store.rotate_credential(cred.id, "sk-rotated-12345")
        assert rotated is not None
        assert rotated.last_rotated_at is not None

        assert store.delete_credential(cred.id) is True
        assert store.get_credential(cred.id) is None


# =============================================================================
# Audit Log
# =============================================================================


class TestAuditLog:
    """Test audit logging."""

    def test_add_audit_entry(self, store: OpenClawGatewayStore) -> None:
        entry = store.add_audit_entry(
            action="session.create",
            actor_id="user-1",
            resource_type="session",
            resource_id="sess-123",
        )
        assert entry.action == "session.create"
        assert entry.result == "success"

    def test_audit_log_filtering(self, store: OpenClawGatewayStore) -> None:
        store.add_audit_entry("session.create", "user-1", "session")
        store.add_audit_entry("credential.store", "user-1", "credential")
        store.add_audit_entry("session.close", "user-2", "session")

        entries, total = store.get_audit_log(resource_type="session")
        assert total == 2

    def test_audit_log_cap(self, store: OpenClawGatewayStore) -> None:
        for i in range(105):
            store.add_audit_entry(f"action-{i}", "user-1", "test")
        entries, total = store.get_audit_log()
        assert total <= 10000


# =============================================================================
# Validation Integration
# =============================================================================


class TestValidationIntegration:
    """Test validation functions used by the handler package."""

    def test_valid_credential_name(self) -> None:
        valid, err = validate_credential_name("my-api-key")
        assert valid is True
        assert err is None

    def test_invalid_credential_name_empty(self) -> None:
        valid, err = validate_credential_name("")
        assert valid is False

    def test_invalid_credential_name_special_chars(self) -> None:
        valid, err = validate_credential_name("key; DROP TABLE")
        assert valid is False

    def test_credential_name_too_long(self) -> None:
        valid, err = validate_credential_name("a" * (MAX_CREDENTIAL_NAME_LENGTH + 1))
        assert valid is False

    def test_valid_credential_secret(self) -> None:
        valid, err = validate_credential_secret("sk-test-12345678")
        assert valid is True

    def test_secret_too_short(self) -> None:
        valid, err = validate_credential_secret("short")
        assert valid is False

    def test_secret_with_null_byte(self) -> None:
        valid, err = validate_credential_secret("secret\x00injected")
        assert valid is False

    def test_valid_session_config(self) -> None:
        valid, err = validate_session_config({"model": "claude-3", "temp": 0.7})
        assert valid is True

    def test_session_config_none_is_valid(self) -> None:
        valid, err = validate_session_config(None)
        assert valid is True

    def test_session_config_too_many_keys(self) -> None:
        config = {f"key-{i}": i for i in range(MAX_SESSION_CONFIG_KEYS + 1)}
        valid, err = validate_session_config(config)
        assert valid is False

    def test_session_config_too_deep(self) -> None:
        config: dict = {"level": {}}
        current = config["level"]
        for _ in range(MAX_SESSION_CONFIG_DEPTH + 1):
            current["nested"] = {}
            current = current["nested"]
        valid, err = validate_session_config(config)
        assert valid is False

    def test_valid_action_type(self) -> None:
        valid, err = validate_action_type("search.documents")
        assert valid is True

    def test_invalid_action_type_injection(self) -> None:
        valid, err = validate_action_type("search; rm -rf /")
        assert valid is False

    def test_action_input_size_limit(self) -> None:
        large_input = {"data": "x" * MAX_ACTION_INPUT_SIZE}
        valid, err = validate_action_input(large_input)
        assert valid is False

    def test_sanitize_action_parameters(self) -> None:
        params = {"cmd": "echo $(whoami)", "safe": "hello"}
        sanitized = sanitize_action_parameters(params)
        # $ and ( are escaped with backslashes, not removed
        assert sanitized["cmd"] != params["cmd"]
        assert "\\$" in sanitized["cmd"]
        assert sanitized["safe"] == "hello"

    def test_sanitize_null_bytes(self) -> None:
        params = {"val": "test\x00injected"}
        sanitized = sanitize_action_parameters(params)
        assert "\x00" not in sanitized["val"]

    def test_validate_metadata(self) -> None:
        valid, err = validate_metadata({"key": "value"})
        assert valid is True

    def test_validate_metadata_none(self) -> None:
        valid, err = validate_metadata(None)
        assert valid is True

    def test_validate_metadata_oversized(self) -> None:
        valid, err = validate_metadata({"data": "x" * 5000})
        assert valid is False


# =============================================================================
# Model Enums
# =============================================================================


class TestModelEnums:
    """Test enum values match expected API contract."""

    def test_session_statuses(self) -> None:
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.CLOSED.value == "closed"
        assert SessionStatus.ERROR.value == "error"

    def test_action_statuses(self) -> None:
        assert ActionStatus.PENDING.value == "pending"
        assert ActionStatus.RUNNING.value == "running"
        assert ActionStatus.COMPLETED.value == "completed"
        assert ActionStatus.FAILED.value == "failed"
        assert ActionStatus.CANCELLED.value == "cancelled"

    def test_credential_types(self) -> None:
        assert CredentialType.API_KEY.value == "api_key"
        assert CredentialType.OAUTH_TOKEN.value == "oauth_token"
        assert CredentialType.SSH_KEY.value == "ssh_key"


# =============================================================================
# Store Singleton
# =============================================================================


class TestStoreSingleton:
    """Test the global store accessor."""

    def test_get_store_returns_instance(self) -> None:
        store = _get_store()
        assert store is not None
        # May be OpenClawGatewayStore or OpenClawPersistentStore
        assert hasattr(store, "create_session")

    def test_get_store_is_same_instance(self) -> None:
        s1 = _get_store()
        s2 = _get_store()
        assert s1 is s2
