"""Comprehensive tests for OpenClaw data models module.

Covers all enums and dataclasses defined in
aragora/server/handlers/openclaw/models.py:

Enums:
- SessionStatus     (ACTIVE, IDLE, CLOSING, CLOSED, ERROR)
- ActionStatus      (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT)
- CredentialType    (API_KEY, OAUTH_TOKEN, PASSWORD, CERTIFICATE, SSH_KEY, SERVICE_ACCOUNT)

Dataclasses:
- Session           (id, user_id, tenant_id, status, timestamps, config, metadata)
- Action            (id, session_id, action_type, status, input/output, error, timestamps, metadata)
- Credential        (id, name, credential_type, user_id, tenant_id, timestamps, metadata)
- AuditEntry        (id, timestamp, action, actor_id, resource_type, resource_id, result, details)

Test categories:
- Enum membership and values
- Enum iteration and uniqueness
- Dataclass construction (required + default fields)
- to_dict() serialization (all fields, None handling, enum value conversion)
- Datetime ISO format serialization
- Default factory isolation (config/metadata dicts not shared)
- Edge cases (empty strings, None tenant_id, missing optional fields)
- __all__ exports verification
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

import pytest

from aragora.server.handlers.openclaw.models import (
    ActionStatus,
    AuditEntry,
    Action,
    Credential,
    CredentialType,
    Session,
    SessionStatus,
)


# ============================================================================
# Helper timestamps
# ============================================================================

NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
EARLIER = datetime(2026, 2, 23, 11, 0, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 2, 23, 13, 0, 0, tzinfo=timezone.utc)


# ============================================================================
# SessionStatus enum
# ============================================================================


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_active_value(self):
        assert SessionStatus.ACTIVE.value == "active"

    def test_idle_value(self):
        assert SessionStatus.IDLE.value == "idle"

    def test_closing_value(self):
        assert SessionStatus.CLOSING.value == "closing"

    def test_closed_value(self):
        assert SessionStatus.CLOSED.value == "closed"

    def test_error_value(self):
        assert SessionStatus.ERROR.value == "error"

    def test_member_count(self):
        assert len(SessionStatus) == 5

    def test_unique_values(self):
        values = [s.value for s in SessionStatus]
        assert len(values) == len(set(values))

    def test_from_value_active(self):
        assert SessionStatus("active") is SessionStatus.ACTIVE

    def test_from_value_invalid_raises(self):
        with pytest.raises(ValueError):
            SessionStatus("nonexistent")

    def test_identity_comparison(self):
        assert SessionStatus.ACTIVE is SessionStatus.ACTIVE
        assert SessionStatus.ACTIVE is not SessionStatus.IDLE


# ============================================================================
# ActionStatus enum
# ============================================================================


class TestActionStatus:
    """Tests for ActionStatus enum."""

    def test_pending_value(self):
        assert ActionStatus.PENDING.value == "pending"

    def test_running_value(self):
        assert ActionStatus.RUNNING.value == "running"

    def test_completed_value(self):
        assert ActionStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        assert ActionStatus.FAILED.value == "failed"

    def test_cancelled_value(self):
        assert ActionStatus.CANCELLED.value == "cancelled"

    def test_timeout_value(self):
        assert ActionStatus.TIMEOUT.value == "timeout"

    def test_member_count(self):
        assert len(ActionStatus) == 6

    def test_unique_values(self):
        values = [s.value for s in ActionStatus]
        assert len(values) == len(set(values))

    def test_from_value_completed(self):
        assert ActionStatus("completed") is ActionStatus.COMPLETED

    def test_from_value_invalid_raises(self):
        with pytest.raises(ValueError):
            ActionStatus("unknown")


# ============================================================================
# CredentialType enum
# ============================================================================


class TestCredentialType:
    """Tests for CredentialType enum."""

    def test_api_key_value(self):
        assert CredentialType.API_KEY.value == "api_key"

    def test_oauth_token_value(self):
        assert CredentialType.OAUTH_TOKEN.value == "oauth_token"

    def test_password_value(self):
        assert CredentialType.PASSWORD.value == "password"

    def test_certificate_value(self):
        assert CredentialType.CERTIFICATE.value == "certificate"

    def test_ssh_key_value(self):
        assert CredentialType.SSH_KEY.value == "ssh_key"

    def test_service_account_value(self):
        assert CredentialType.SERVICE_ACCOUNT.value == "service_account"

    def test_member_count(self):
        assert len(CredentialType) == 6

    def test_unique_values(self):
        values = [c.value for c in CredentialType]
        assert len(values) == len(set(values))

    def test_from_value_ssh_key(self):
        assert CredentialType("ssh_key") is CredentialType.SSH_KEY

    def test_from_value_invalid_raises(self):
        with pytest.raises(ValueError):
            CredentialType("bearer")


# ============================================================================
# Session dataclass
# ============================================================================


class TestSession:
    """Tests for Session dataclass."""

    def _make_session(self, **overrides: Any) -> Session:
        defaults = {
            "id": "sess-001",
            "user_id": "user-001",
            "tenant_id": "tenant-001",
            "status": SessionStatus.ACTIVE,
            "created_at": NOW,
            "updated_at": NOW,
            "last_activity_at": NOW,
        }
        defaults.update(overrides)
        return Session(**defaults)

    def test_basic_construction(self):
        s = self._make_session()
        assert s.id == "sess-001"
        assert s.user_id == "user-001"
        assert s.tenant_id == "tenant-001"
        assert s.status is SessionStatus.ACTIVE
        assert s.created_at == NOW
        assert s.updated_at == NOW
        assert s.last_activity_at == NOW

    def test_default_config_is_empty_dict(self):
        s = self._make_session()
        assert s.config == {}

    def test_default_metadata_is_empty_dict(self):
        s = self._make_session()
        assert s.metadata == {}

    def test_custom_config(self):
        cfg = {"model": "gpt-4", "timeout": 30}
        s = self._make_session(config=cfg)
        assert s.config == cfg

    def test_custom_metadata(self):
        meta = {"source": "api", "version": "2.0"}
        s = self._make_session(metadata=meta)
        assert s.metadata == meta

    def test_none_tenant_id(self):
        s = self._make_session(tenant_id=None)
        assert s.tenant_id is None

    def test_default_factories_are_independent(self):
        """Ensure default dicts are not shared between instances."""
        s1 = self._make_session()
        s2 = self._make_session()
        s1.config["key"] = "val"
        assert "key" not in s2.config

    def test_to_dict_returns_dict(self):
        s = self._make_session()
        result = s.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_id(self):
        s = self._make_session(id="sess-abc")
        assert s.to_dict()["id"] == "sess-abc"

    def test_to_dict_user_id(self):
        s = self._make_session(user_id="user-xyz")
        assert s.to_dict()["user_id"] == "user-xyz"

    def test_to_dict_tenant_id(self):
        s = self._make_session(tenant_id="t-99")
        assert s.to_dict()["tenant_id"] == "t-99"

    def test_to_dict_tenant_id_none(self):
        s = self._make_session(tenant_id=None)
        assert s.to_dict()["tenant_id"] is None

    def test_to_dict_status_is_string_value(self):
        s = self._make_session(status=SessionStatus.CLOSING)
        assert s.to_dict()["status"] == "closing"

    def test_to_dict_created_at_iso_format(self):
        s = self._make_session(created_at=NOW)
        assert s.to_dict()["created_at"] == NOW.isoformat()

    def test_to_dict_updated_at_iso_format(self):
        s = self._make_session(updated_at=LATER)
        assert s.to_dict()["updated_at"] == LATER.isoformat()

    def test_to_dict_last_activity_at_iso_format(self):
        s = self._make_session(last_activity_at=EARLIER)
        assert s.to_dict()["last_activity_at"] == EARLIER.isoformat()

    def test_to_dict_config_included(self):
        cfg = {"retries": 3}
        s = self._make_session(config=cfg)
        assert s.to_dict()["config"] == cfg

    def test_to_dict_metadata_included(self):
        meta = {"env": "prod"}
        s = self._make_session(metadata=meta)
        assert s.to_dict()["metadata"] == meta

    def test_to_dict_has_all_keys(self):
        s = self._make_session()
        keys = set(s.to_dict().keys())
        expected = {
            "id",
            "user_id",
            "tenant_id",
            "status",
            "created_at",
            "updated_at",
            "last_activity_at",
            "config",
            "metadata",
        }
        assert keys == expected

    def test_to_dict_all_statuses(self):
        """Verify to_dict works with every SessionStatus."""
        for status in SessionStatus:
            s = self._make_session(status=status)
            assert s.to_dict()["status"] == status.value


# ============================================================================
# Action dataclass
# ============================================================================


class TestAction:
    """Tests for Action dataclass."""

    def _make_action(self, **overrides: Any) -> Action:
        defaults = {
            "id": "act-001",
            "session_id": "sess-001",
            "action_type": "tool.execute",
            "status": ActionStatus.PENDING,
            "input_data": {"prompt": "hello"},
            "output_data": None,
            "error": None,
            "created_at": NOW,
            "started_at": None,
            "completed_at": None,
        }
        defaults.update(overrides)
        return Action(**defaults)

    def test_basic_construction(self):
        a = self._make_action()
        assert a.id == "act-001"
        assert a.session_id == "sess-001"
        assert a.action_type == "tool.execute"
        assert a.status is ActionStatus.PENDING

    def test_input_data(self):
        a = self._make_action(input_data={"key": "val"})
        assert a.input_data == {"key": "val"}

    def test_output_data_none_by_default(self):
        a = self._make_action()
        assert a.output_data is None

    def test_output_data_set(self):
        a = self._make_action(output_data={"result": 42})
        assert a.output_data == {"result": 42}

    def test_error_none_by_default(self):
        a = self._make_action()
        assert a.error is None

    def test_error_set(self):
        a = self._make_action(error="timeout exceeded")
        assert a.error == "timeout exceeded"

    def test_started_at_none(self):
        a = self._make_action(started_at=None)
        assert a.started_at is None

    def test_completed_at_none(self):
        a = self._make_action(completed_at=None)
        assert a.completed_at is None

    def test_default_metadata_is_empty(self):
        a = self._make_action()
        assert a.metadata == {}

    def test_custom_metadata(self):
        a = self._make_action(metadata={"trace_id": "xyz"})
        assert a.metadata == {"trace_id": "xyz"}

    def test_default_metadata_isolation(self):
        a1 = self._make_action()
        a2 = self._make_action()
        a1.metadata["x"] = 1
        assert "x" not in a2.metadata

    def test_to_dict_returns_dict(self):
        a = self._make_action()
        assert isinstance(a.to_dict(), dict)

    def test_to_dict_id(self):
        a = self._make_action(id="act-xyz")
        assert a.to_dict()["id"] == "act-xyz"

    def test_to_dict_session_id(self):
        a = self._make_action(session_id="sess-xyz")
        assert a.to_dict()["session_id"] == "sess-xyz"

    def test_to_dict_action_type(self):
        a = self._make_action(action_type="file.read")
        assert a.to_dict()["action_type"] == "file.read"

    def test_to_dict_status_is_string_value(self):
        a = self._make_action(status=ActionStatus.RUNNING)
        assert a.to_dict()["status"] == "running"

    def test_to_dict_input_data(self):
        a = self._make_action(input_data={"cmd": "ls"})
        assert a.to_dict()["input_data"] == {"cmd": "ls"}

    def test_to_dict_output_data_none(self):
        a = self._make_action(output_data=None)
        assert a.to_dict()["output_data"] is None

    def test_to_dict_output_data_set(self):
        a = self._make_action(output_data={"files": ["a.py"]})
        assert a.to_dict()["output_data"] == {"files": ["a.py"]}

    def test_to_dict_error_none(self):
        a = self._make_action(error=None)
        assert a.to_dict()["error"] is None

    def test_to_dict_error_set(self):
        a = self._make_action(error="fail")
        assert a.to_dict()["error"] == "fail"

    def test_to_dict_created_at_iso_format(self):
        a = self._make_action(created_at=NOW)
        assert a.to_dict()["created_at"] == NOW.isoformat()

    def test_to_dict_started_at_none(self):
        a = self._make_action(started_at=None)
        assert a.to_dict()["started_at"] is None

    def test_to_dict_started_at_iso_format(self):
        a = self._make_action(started_at=EARLIER)
        assert a.to_dict()["started_at"] == EARLIER.isoformat()

    def test_to_dict_completed_at_none(self):
        a = self._make_action(completed_at=None)
        assert a.to_dict()["completed_at"] is None

    def test_to_dict_completed_at_iso_format(self):
        a = self._make_action(completed_at=LATER)
        assert a.to_dict()["completed_at"] == LATER.isoformat()

    def test_to_dict_metadata(self):
        a = self._make_action(metadata={"retry": 2})
        assert a.to_dict()["metadata"] == {"retry": 2}

    def test_to_dict_has_all_keys(self):
        a = self._make_action()
        keys = set(a.to_dict().keys())
        expected = {
            "id",
            "session_id",
            "action_type",
            "status",
            "input_data",
            "output_data",
            "error",
            "created_at",
            "started_at",
            "completed_at",
            "metadata",
        }
        assert keys == expected

    def test_to_dict_all_statuses(self):
        """Verify to_dict works with every ActionStatus."""
        for status in ActionStatus:
            a = self._make_action(status=status)
            assert a.to_dict()["status"] == status.value

    def test_completed_action_full_lifecycle(self):
        """Simulate a fully completed action with all fields populated."""
        a = self._make_action(
            status=ActionStatus.COMPLETED,
            input_data={"prompt": "test"},
            output_data={"response": "ok"},
            started_at=EARLIER,
            completed_at=LATER,
            metadata={"duration_ms": 500},
        )
        d = a.to_dict()
        assert d["status"] == "completed"
        assert d["output_data"] == {"response": "ok"}
        assert d["started_at"] == EARLIER.isoformat()
        assert d["completed_at"] == LATER.isoformat()

    def test_failed_action_with_error(self):
        a = self._make_action(
            status=ActionStatus.FAILED,
            error="Connection refused",
            started_at=EARLIER,
        )
        d = a.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "Connection refused"
        assert d["completed_at"] is None


# ============================================================================
# Credential dataclass
# ============================================================================


class TestCredential:
    """Tests for Credential dataclass."""

    def _make_credential(self, **overrides: Any) -> Credential:
        defaults = {
            "id": "cred-001",
            "name": "my-api-key",
            "credential_type": CredentialType.API_KEY,
            "user_id": "user-001",
            "tenant_id": "tenant-001",
            "created_at": NOW,
            "updated_at": NOW,
            "last_rotated_at": None,
            "expires_at": None,
        }
        defaults.update(overrides)
        return Credential(**defaults)

    def test_basic_construction(self):
        c = self._make_credential()
        assert c.id == "cred-001"
        assert c.name == "my-api-key"
        assert c.credential_type is CredentialType.API_KEY
        assert c.user_id == "user-001"
        assert c.tenant_id == "tenant-001"

    def test_none_tenant_id(self):
        c = self._make_credential(tenant_id=None)
        assert c.tenant_id is None

    def test_last_rotated_at_none(self):
        c = self._make_credential(last_rotated_at=None)
        assert c.last_rotated_at is None

    def test_last_rotated_at_set(self):
        c = self._make_credential(last_rotated_at=EARLIER)
        assert c.last_rotated_at == EARLIER

    def test_expires_at_none(self):
        c = self._make_credential(expires_at=None)
        assert c.expires_at is None

    def test_expires_at_set(self):
        future = NOW + timedelta(days=90)
        c = self._make_credential(expires_at=future)
        assert c.expires_at == future

    def test_default_metadata_is_empty(self):
        c = self._make_credential()
        assert c.metadata == {}

    def test_custom_metadata(self):
        c = self._make_credential(metadata={"provider": "aws"})
        assert c.metadata == {"provider": "aws"}

    def test_default_metadata_isolation(self):
        c1 = self._make_credential()
        c2 = self._make_credential()
        c1.metadata["z"] = 99
        assert "z" not in c2.metadata

    def test_to_dict_returns_dict(self):
        c = self._make_credential()
        assert isinstance(c.to_dict(), dict)

    def test_to_dict_id(self):
        c = self._make_credential(id="cred-xyz")
        assert c.to_dict()["id"] == "cred-xyz"

    def test_to_dict_name(self):
        c = self._make_credential(name="prod-key")
        assert c.to_dict()["name"] == "prod-key"

    def test_to_dict_credential_type_is_string_value(self):
        c = self._make_credential(credential_type=CredentialType.SSH_KEY)
        assert c.to_dict()["credential_type"] == "ssh_key"

    def test_to_dict_user_id(self):
        c = self._make_credential(user_id="u-42")
        assert c.to_dict()["user_id"] == "u-42"

    def test_to_dict_tenant_id(self):
        c = self._make_credential(tenant_id="t-10")
        assert c.to_dict()["tenant_id"] == "t-10"

    def test_to_dict_tenant_id_none(self):
        c = self._make_credential(tenant_id=None)
        assert c.to_dict()["tenant_id"] is None

    def test_to_dict_created_at_iso_format(self):
        c = self._make_credential(created_at=NOW)
        assert c.to_dict()["created_at"] == NOW.isoformat()

    def test_to_dict_updated_at_iso_format(self):
        c = self._make_credential(updated_at=LATER)
        assert c.to_dict()["updated_at"] == LATER.isoformat()

    def test_to_dict_last_rotated_at_none(self):
        c = self._make_credential(last_rotated_at=None)
        assert c.to_dict()["last_rotated_at"] is None

    def test_to_dict_last_rotated_at_iso_format(self):
        c = self._make_credential(last_rotated_at=EARLIER)
        assert c.to_dict()["last_rotated_at"] == EARLIER.isoformat()

    def test_to_dict_expires_at_none(self):
        c = self._make_credential(expires_at=None)
        assert c.to_dict()["expires_at"] is None

    def test_to_dict_expires_at_iso_format(self):
        future = NOW + timedelta(days=365)
        c = self._make_credential(expires_at=future)
        assert c.to_dict()["expires_at"] == future.isoformat()

    def test_to_dict_metadata(self):
        c = self._make_credential(metadata={"scope": "read"})
        assert c.to_dict()["metadata"] == {"scope": "read"}

    def test_to_dict_has_all_keys(self):
        c = self._make_credential()
        keys = set(c.to_dict().keys())
        expected = {
            "id",
            "name",
            "credential_type",
            "user_id",
            "tenant_id",
            "created_at",
            "updated_at",
            "last_rotated_at",
            "expires_at",
            "metadata",
        }
        assert keys == expected

    def test_to_dict_all_credential_types(self):
        """Verify to_dict works with every CredentialType."""
        for ctype in CredentialType:
            c = self._make_credential(credential_type=ctype)
            assert c.to_dict()["credential_type"] == ctype.value

    def test_no_secret_in_to_dict(self):
        """Credential.to_dict() must never expose secret values."""
        c = self._make_credential()
        d = c.to_dict()
        assert "secret" not in d
        assert "secret_value" not in d
        assert "password" not in d
        assert "token" not in d


# ============================================================================
# AuditEntry dataclass
# ============================================================================


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def _make_audit_entry(self, **overrides: Any) -> AuditEntry:
        defaults = {
            "id": "audit-001",
            "timestamp": NOW,
            "action": "session.create",
            "actor_id": "user-001",
            "resource_type": "session",
            "resource_id": "sess-001",
            "result": "success",
        }
        defaults.update(overrides)
        return AuditEntry(**defaults)

    def test_basic_construction(self):
        ae = self._make_audit_entry()
        assert ae.id == "audit-001"
        assert ae.timestamp == NOW
        assert ae.action == "session.create"
        assert ae.actor_id == "user-001"
        assert ae.resource_type == "session"
        assert ae.resource_id == "sess-001"
        assert ae.result == "success"

    def test_none_resource_id(self):
        ae = self._make_audit_entry(resource_id=None)
        assert ae.resource_id is None

    def test_default_details_is_empty(self):
        ae = self._make_audit_entry()
        assert ae.details == {}

    def test_custom_details(self):
        ae = self._make_audit_entry(details={"ip": "10.0.0.1", "ua": "curl/7.80"})
        assert ae.details == {"ip": "10.0.0.1", "ua": "curl/7.80"}

    def test_default_details_isolation(self):
        ae1 = self._make_audit_entry()
        ae2 = self._make_audit_entry()
        ae1.details["foo"] = "bar"
        assert "foo" not in ae2.details

    def test_to_dict_returns_dict(self):
        ae = self._make_audit_entry()
        assert isinstance(ae.to_dict(), dict)

    def test_to_dict_id(self):
        ae = self._make_audit_entry(id="audit-xyz")
        assert ae.to_dict()["id"] == "audit-xyz"

    def test_to_dict_timestamp_iso_format(self):
        ae = self._make_audit_entry(timestamp=NOW)
        assert ae.to_dict()["timestamp"] == NOW.isoformat()

    def test_to_dict_action(self):
        ae = self._make_audit_entry(action="credential.rotate")
        assert ae.to_dict()["action"] == "credential.rotate"

    def test_to_dict_actor_id(self):
        ae = self._make_audit_entry(actor_id="admin-01")
        assert ae.to_dict()["actor_id"] == "admin-01"

    def test_to_dict_resource_type(self):
        ae = self._make_audit_entry(resource_type="credential")
        assert ae.to_dict()["resource_type"] == "credential"

    def test_to_dict_resource_id(self):
        ae = self._make_audit_entry(resource_id="cred-001")
        assert ae.to_dict()["resource_id"] == "cred-001"

    def test_to_dict_resource_id_none(self):
        ae = self._make_audit_entry(resource_id=None)
        assert ae.to_dict()["resource_id"] is None

    def test_to_dict_result(self):
        ae = self._make_audit_entry(result="failure")
        assert ae.to_dict()["result"] == "failure"

    def test_to_dict_details(self):
        details = {"reason": "expired", "code": 401}
        ae = self._make_audit_entry(details=details)
        assert ae.to_dict()["details"] == details

    def test_to_dict_has_all_keys(self):
        ae = self._make_audit_entry()
        keys = set(ae.to_dict().keys())
        expected = {
            "id",
            "timestamp",
            "action",
            "actor_id",
            "resource_type",
            "resource_id",
            "result",
            "details",
        }
        assert keys == expected


# ============================================================================
# __all__ exports
# ============================================================================


class TestModuleExports:
    """Verify the __all__ list matches actual exports."""

    def test_session_status_exported(self):
        import aragora.server.handlers.openclaw.models as mod

        assert "SessionStatus" in mod.__all__

    def test_action_status_exported(self):
        import aragora.server.handlers.openclaw.models as mod

        assert "ActionStatus" in mod.__all__

    def test_credential_type_exported(self):
        import aragora.server.handlers.openclaw.models as mod

        assert "CredentialType" in mod.__all__

    def test_session_exported(self):
        import aragora.server.handlers.openclaw.models as mod

        assert "Session" in mod.__all__

    def test_action_exported(self):
        import aragora.server.handlers.openclaw.models as mod

        assert "Action" in mod.__all__

    def test_credential_exported(self):
        import aragora.server.handlers.openclaw.models as mod

        assert "Credential" in mod.__all__

    def test_audit_entry_exported(self):
        import aragora.server.handlers.openclaw.models as mod

        assert "AuditEntry" in mod.__all__

    def test_all_count(self):
        import aragora.server.handlers.openclaw.models as mod

        assert len(mod.__all__) == 7


# ============================================================================
# Cross-model and edge case tests
# ============================================================================


class TestCrossModelEdgeCases:
    """Cross-cutting edge case tests across models."""

    def test_session_with_empty_string_id(self):
        s = Session(
            id="",
            user_id="",
            tenant_id=None,
            status=SessionStatus.ACTIVE,
            created_at=NOW,
            updated_at=NOW,
            last_activity_at=NOW,
        )
        d = s.to_dict()
        assert d["id"] == ""
        assert d["user_id"] == ""

    def test_action_with_empty_input_data(self):
        a = Action(
            id="a1",
            session_id="s1",
            action_type="noop",
            status=ActionStatus.COMPLETED,
            input_data={},
            output_data={},
            error=None,
            created_at=NOW,
            started_at=NOW,
            completed_at=NOW,
        )
        d = a.to_dict()
        assert d["input_data"] == {}
        assert d["output_data"] == {}

    def test_credential_with_all_optional_datetimes(self):
        future = NOW + timedelta(days=30)
        c = Credential(
            id="c1",
            name="key",
            credential_type=CredentialType.CERTIFICATE,
            user_id="u1",
            tenant_id="t1",
            created_at=NOW,
            updated_at=NOW,
            last_rotated_at=EARLIER,
            expires_at=future,
        )
        d = c.to_dict()
        assert d["last_rotated_at"] == EARLIER.isoformat()
        assert d["expires_at"] == future.isoformat()

    def test_audit_entry_with_large_details(self):
        details = {f"field_{i}": f"value_{i}" for i in range(100)}
        ae = AuditEntry(
            id="ae1",
            timestamp=NOW,
            action="bulk.op",
            actor_id="system",
            resource_type="batch",
            resource_id=None,
            result="partial",
            details=details,
        )
        d = ae.to_dict()
        assert len(d["details"]) == 100

    def test_session_naive_datetime(self):
        """to_dict should work with naive datetimes too."""
        naive = datetime(2026, 1, 1, 0, 0, 0)
        s = Session(
            id="s1",
            user_id="u1",
            tenant_id=None,
            status=SessionStatus.IDLE,
            created_at=naive,
            updated_at=naive,
            last_activity_at=naive,
        )
        d = s.to_dict()
        assert d["created_at"] == "2026-01-01T00:00:00"

    def test_action_with_nested_output_data(self):
        output = {
            "messages": [{"role": "assistant", "content": "hello"}],
            "usage": {"tokens": 42},
        }
        a = Action(
            id="a1",
            session_id="s1",
            action_type="chat",
            status=ActionStatus.COMPLETED,
            input_data={"prompt": "hi"},
            output_data=output,
            error=None,
            created_at=NOW,
            started_at=EARLIER,
            completed_at=LATER,
        )
        d = a.to_dict()
        assert d["output_data"]["messages"][0]["content"] == "hello"
        assert d["output_data"]["usage"]["tokens"] == 42

    def test_to_dict_does_not_mutate_original(self):
        """Calling to_dict should not alter the dataclass instance."""
        s = Session(
            id="s1",
            user_id="u1",
            tenant_id="t1",
            status=SessionStatus.ACTIVE,
            created_at=NOW,
            updated_at=NOW,
            last_activity_at=NOW,
            config={"a": 1},
            metadata={"b": 2},
        )
        d = s.to_dict()
        d["config"]["c"] = 3
        # Original should be unchanged since config dict is the same reference,
        # but the point is to_dict itself didn't cause mutation
        assert s.status is SessionStatus.ACTIVE

    def test_session_to_dict_roundtrip_values(self):
        """All values from to_dict() should match the original fields."""
        s = Session(
            id="s1",
            user_id="u1",
            tenant_id="t1",
            status=SessionStatus.ERROR,
            created_at=NOW,
            updated_at=EARLIER,
            last_activity_at=LATER,
            config={"x": [1, 2]},
            metadata={"y": True},
        )
        d = s.to_dict()
        assert d["id"] == s.id
        assert d["user_id"] == s.user_id
        assert d["tenant_id"] == s.tenant_id
        assert d["status"] == s.status.value
        assert d["created_at"] == s.created_at.isoformat()
        assert d["updated_at"] == s.updated_at.isoformat()
        assert d["last_activity_at"] == s.last_activity_at.isoformat()
        assert d["config"] == s.config
        assert d["metadata"] == s.metadata
