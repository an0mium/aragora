"""
Tests for security audit logging.

Tests cover:
- Authentication audit functions (success, failure, sessions, tokens)
- RBAC audit functions (decisions, role changes)
- Encryption audit functions (encrypt/decrypt, key rotation)
- Secret access audit functions
- Migration audit functions
- Security incident audit functions
- Query functions
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.observability.security_audit import (
    # Constants
    SECURITY_EVENTS,
    # Auth functions
    audit_auth_success,
    audit_auth_failure,
    audit_session_created,
    audit_token_issued,
    # RBAC functions
    audit_rbac_decision,
    audit_role_change,
    # Encryption functions
    audit_encryption_operation,
    audit_key_rotation,
    audit_key_generated,
    # Secret functions
    audit_secret_access,
    audit_secret_modified,
    # Migration functions
    audit_migration_started,
    audit_migration_completed,
    # Incident functions
    audit_security_incident,
    audit_security_alert,
    audit_request_blocked,
    # Query functions
    get_security_events,
    get_auth_failures,
    get_security_incidents,
)


class TestSecurityEvents:
    """Tests for security event constants."""

    def test_auth_events_defined(self):
        """Test authentication event types are defined."""
        assert "auth_success" in SECURITY_EVENTS
        assert "auth_failure" in SECURITY_EVENTS
        assert "auth_logout" in SECURITY_EVENTS
        assert "session_created" in SECURITY_EVENTS
        assert "session_invalidated" in SECURITY_EVENTS
        assert "token_issued" in SECURITY_EVENTS
        assert "token_revoked" in SECURITY_EVENTS

    def test_rbac_events_defined(self):
        """Test RBAC event types are defined."""
        assert "rbac_granted" in SECURITY_EVENTS
        assert "rbac_denied" in SECURITY_EVENTS
        assert "permission_checked" in SECURITY_EVENTS
        assert "role_assigned" in SECURITY_EVENTS
        assert "role_revoked" in SECURITY_EVENTS

    def test_encryption_events_defined(self):
        """Test encryption event types are defined."""
        assert "encryption_success" in SECURITY_EVENTS
        assert "encryption_failure" in SECURITY_EVENTS
        assert "decryption_success" in SECURITY_EVENTS
        assert "decryption_failure" in SECURITY_EVENTS

    def test_key_events_defined(self):
        """Test key management event types are defined."""
        assert "key_generated" in SECURITY_EVENTS
        assert "key_rotated" in SECURITY_EVENTS
        assert "key_accessed" in SECURITY_EVENTS
        assert "key_deleted" in SECURITY_EVENTS

    def test_secret_events_defined(self):
        """Test secret access event types are defined."""
        assert "secret_accessed" in SECURITY_EVENTS
        assert "secret_created" in SECURITY_EVENTS
        assert "secret_updated" in SECURITY_EVENTS
        assert "secret_deleted" in SECURITY_EVENTS

    def test_migration_events_defined(self):
        """Test migration event types are defined."""
        assert "migration_started" in SECURITY_EVENTS
        assert "migration_completed" in SECURITY_EVENTS
        assert "migration_record_encrypted" in SECURITY_EVENTS
        assert "migration_error" in SECURITY_EVENTS

    def test_incident_events_defined(self):
        """Test security incident event types are defined."""
        assert "incident_detected" in SECURITY_EVENTS
        assert "incident_escalated" in SECURITY_EVENTS
        assert "incident_resolved" in SECURITY_EVENTS
        assert "alert_triggered" in SECURITY_EVENTS
        assert "request_blocked" in SECURITY_EVENTS

    def test_event_type_format(self):
        """Test all event types follow security.* format."""
        for event_name, event_type in SECURITY_EVENTS.items():
            assert event_type.startswith("security."), f"{event_name} should start with security."


@pytest.fixture
def mock_audit_log():
    """Create mock audit log."""
    mock_entry = MagicMock()
    mock_entry.id = "test-entry-id"
    mock_entry.event_type = "security.auth.success"
    mock_entry.actor = "user_123"
    mock_entry.details = {}

    mock_log = MagicMock()
    mock_log.append = AsyncMock(return_value=mock_entry)
    mock_log.query = AsyncMock(return_value=[mock_entry])

    return mock_log


@pytest.fixture
def mock_metrics():
    """Create mock metrics functions."""
    with (
        patch("aragora.observability.security_audit.record_auth_attempt") as mock_auth_attempt,
        patch("aragora.observability.security_audit.record_auth_failure") as mock_auth_failure,
        patch("aragora.observability.security_audit.record_rbac_decision") as mock_rbac,
        patch("aragora.observability.security_audit.record_encryption_operation") as mock_encrypt,
        patch("aragora.observability.security_audit.record_key_rotation") as mock_key_rotate,
        patch("aragora.observability.security_audit.record_secret_access") as mock_secret,
        patch("aragora.observability.security_audit.record_security_incident") as mock_incident,
        patch("aragora.observability.security_audit.record_security_alert") as mock_alert,
        patch("aragora.observability.security_audit.record_blocked_request") as mock_blocked,
    ):
        yield {
            "auth_attempt": mock_auth_attempt,
            "auth_failure": mock_auth_failure,
            "rbac": mock_rbac,
            "encrypt": mock_encrypt,
            "key_rotate": mock_key_rotate,
            "secret": mock_secret,
            "incident": mock_incident,
            "alert": mock_alert,
            "blocked": mock_blocked,
        }


class TestAuditAuthSuccess:
    """Tests for audit_auth_success function."""

    @pytest.mark.asyncio
    async def test_logs_successful_auth(self, mock_audit_log, mock_metrics):
        """Test successful authentication is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_auth_success(
                user_id="user_123",
                method="jwt",
                ip_address="192.168.1.1",
            )

            assert entry is not None
            mock_audit_log.append.assert_called_once()
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["auth_success"]
            assert call_kwargs["actor"] == "user_123"
            assert call_kwargs["resource_id"] == "jwt"
            assert call_kwargs["ip_address"] == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_records_metrics(self, mock_audit_log, mock_metrics):
        """Test authentication metrics are recorded."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_auth_success(user_id="user_123", method="jwt")

            mock_metrics["auth_attempt"].assert_called_once_with("jwt", success=True)

    @pytest.mark.asyncio
    async def test_includes_user_agent(self, mock_audit_log, mock_metrics):
        """Test user agent is included."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_auth_success(
                user_id="user_123",
                method="jwt",
                user_agent="Mozilla/5.0",
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["user_agent"] == "Mozilla/5.0"

    @pytest.mark.asyncio
    async def test_includes_workspace_id(self, mock_audit_log, mock_metrics):
        """Test workspace ID is included."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_auth_success(
                user_id="user_123",
                method="jwt",
                workspace_id="ws_456",
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["workspace_id"] == "ws_456"

    @pytest.mark.asyncio
    async def test_includes_extra_details(self, mock_audit_log, mock_metrics):
        """Test extra details are passed through."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_auth_success(
                user_id="user_123",
                method="jwt",
                custom_field="custom_value",
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["details"]["custom_field"] == "custom_value"


class TestAuditAuthFailure:
    """Tests for audit_auth_failure function."""

    @pytest.mark.asyncio
    async def test_logs_failed_auth(self, mock_audit_log, mock_metrics):
        """Test failed authentication is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_auth_failure(
                user_id="user_123",
                method="jwt",
                reason="invalid_token",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["auth_failure"]
            assert call_kwargs["details"]["reason"] == "invalid_token"
            assert call_kwargs["details"]["success"] is False

    @pytest.mark.asyncio
    async def test_records_failure_metrics(self, mock_audit_log, mock_metrics):
        """Test failure metrics are recorded."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_auth_failure(
                user_id="user_123",
                method="jwt",
                reason="expired",
            )

            mock_metrics["auth_attempt"].assert_called_once_with("jwt", success=False)
            mock_metrics["auth_failure"].assert_called_once_with("jwt", "expired")

    @pytest.mark.asyncio
    async def test_handles_unknown_user(self, mock_audit_log, mock_metrics):
        """Test handling of unknown user."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_auth_failure(
                user_id=None,
                method="jwt",
                reason="invalid_credentials",
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["actor"] == "unknown"


class TestAuditSessionCreated:
    """Tests for audit_session_created function."""

    @pytest.mark.asyncio
    async def test_logs_session_creation(self, mock_audit_log, mock_metrics):
        """Test session creation is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_session_created(
                session_id="sess_123",
                user_id="user_456",
                session_type="jwt",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["session_created"]
            assert call_kwargs["resource_id"] == "sess_123"
            assert call_kwargs["actor"] == "user_456"

    @pytest.mark.asyncio
    async def test_includes_expiry(self, mock_audit_log, mock_metrics):
        """Test expiry is included."""
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_session_created(
                session_id="sess_123",
                user_id="user_456",
                session_type="jwt",
                expiry=expiry,
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["details"]["expiry"] == expiry.isoformat()


class TestAuditTokenIssued:
    """Tests for audit_token_issued function."""

    @pytest.mark.asyncio
    async def test_logs_token_issuance(self, mock_audit_log, mock_metrics):
        """Test token issuance is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_token_issued(
                token_id="tok_abc123",
                user_id="user_789",
                token_type="access",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["token_issued"]
            assert call_kwargs["resource_id"] == "tok_abc123"
            assert call_kwargs["details"]["token_type"] == "access"

    @pytest.mark.asyncio
    async def test_includes_scopes(self, mock_audit_log, mock_metrics):
        """Test scopes are included."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_token_issued(
                token_id="tok_abc123",
                user_id="user_789",
                token_type="access",
                scopes=["read", "write"],
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["details"]["scopes"] == ["read", "write"]


class TestAuditRbacDecision:
    """Tests for audit_rbac_decision function."""

    @pytest.mark.asyncio
    async def test_logs_granted_decision(self, mock_audit_log, mock_metrics):
        """Test granted RBAC decision is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_rbac_decision(
                user_id="user_123",
                permission="debates:create",
                granted=True,
                resource_type="debate",
                resource_id="debate_456",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["rbac_granted"]
            assert call_kwargs["details"]["permission"] == "debates:create"
            assert call_kwargs["details"]["granted"] is True

    @pytest.mark.asyncio
    async def test_logs_denied_decision(self, mock_audit_log, mock_metrics):
        """Test denied RBAC decision is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_rbac_decision(
                user_id="user_123",
                permission="admin:delete",
                granted=False,
                resource_type="user",
                resource_id="user_456",
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["rbac_denied"]
            assert call_kwargs["details"]["granted"] is False

    @pytest.mark.asyncio
    async def test_records_rbac_metrics(self, mock_audit_log, mock_metrics):
        """Test RBAC metrics are recorded."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_rbac_decision(
                user_id="user_123",
                permission="debates:read",
                granted=True,
                resource_type="debate",
                resource_id="debate_789",
            )

            mock_metrics["rbac"].assert_called_once_with("debates:read", True)


class TestAuditRoleChange:
    """Tests for audit_role_change function."""

    @pytest.mark.asyncio
    async def test_logs_role_assignment(self, mock_audit_log, mock_metrics):
        """Test role assignment is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_role_change(
                target_user_id="user_target",
                actor_id="user_admin",
                role="admin",
                action="assign",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["role_assigned"]
            assert call_kwargs["details"]["role"] == "admin"

    @pytest.mark.asyncio
    async def test_logs_role_revocation(self, mock_audit_log, mock_metrics):
        """Test role revocation is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_role_change(
                target_user_id="user_target",
                actor_id="user_admin",
                role="admin",
                action="revoke",
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["role_revoked"]


class TestAuditEncryptionOperation:
    """Tests for audit_encryption_operation function."""

    @pytest.mark.asyncio
    async def test_logs_encrypt_success(self, mock_audit_log, mock_metrics):
        """Test successful encryption is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_encryption_operation(
                actor="user_123",
                operation="encrypt",
                success=True,
                store="secrets",
                field="api_key",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["encryption_success"]
            assert call_kwargs["details"]["success"] is True

    @pytest.mark.asyncio
    async def test_logs_decrypt_failure(self, mock_audit_log, mock_metrics):
        """Test failed decryption is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_encryption_operation(
                actor="user_123",
                operation="decrypt",
                success=False,
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["decryption_failure"]

    @pytest.mark.asyncio
    async def test_records_encryption_metrics(self, mock_audit_log, mock_metrics):
        """Test encryption metrics are recorded."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_encryption_operation(
                actor="user_123",
                operation="encrypt",
                success=True,
                latency_ms=50.0,
            )

            mock_metrics["encrypt"].assert_called_once_with("encrypt", True, 0.05)

    @pytest.mark.asyncio
    async def test_system_actor_type(self, mock_audit_log, mock_metrics):
        """Test system actors are correctly typed."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_encryption_operation(
                actor="system:encryption",
                operation="encrypt",
                success=True,
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["actor_type"] == "system"


class TestAuditKeyRotation:
    """Tests for audit_key_rotation function."""

    @pytest.mark.asyncio
    async def test_logs_key_rotation(self, mock_audit_log, mock_metrics):
        """Test key rotation is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_key_rotation(
                actor="system:key_manager",
                key_id="master_key",
                old_version=1,
                new_version=2,
                success=True,
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["key_rotated"]
            assert call_kwargs["details"]["old_version"] == 1
            assert call_kwargs["details"]["new_version"] == 2

    @pytest.mark.asyncio
    async def test_includes_records_count(self, mock_audit_log, mock_metrics):
        """Test re-encrypted records count is included."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_key_rotation(
                actor="system:key_manager",
                key_id="master_key",
                old_version=1,
                new_version=2,
                success=True,
                records_re_encrypted=1500,
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["details"]["records_re_encrypted"] == 1500


class TestAuditKeyGenerated:
    """Tests for audit_key_generated function."""

    @pytest.mark.asyncio
    async def test_logs_key_generation(self, mock_audit_log, mock_metrics):
        """Test key generation is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_key_generated(
                actor="system:init",
                key_id="new_key_123",
                key_type="master",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["key_generated"]
            assert call_kwargs["details"]["key_type"] == "master"


class TestAuditSecretAccess:
    """Tests for audit_secret_access function."""

    @pytest.mark.asyncio
    async def test_logs_secret_access(self, mock_audit_log, mock_metrics):
        """Test secret access is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_secret_access(
                actor="user_123",
                secret_type="api_key",
                store="integrations",
                operation="decrypt",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["secret_accessed"]
            assert call_kwargs["details"]["secret_type"] == "api_key"

    @pytest.mark.asyncio
    async def test_records_secret_metrics(self, mock_audit_log, mock_metrics):
        """Test secret access metrics are recorded."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_secret_access(
                actor="user_123",
                secret_type="oauth_token",
                store="auth",
                operation="read",
            )

            mock_metrics["secret"].assert_called_once_with("oauth_token", "read")


class TestAuditSecretModified:
    """Tests for audit_secret_modified function."""

    @pytest.mark.asyncio
    async def test_logs_secret_creation(self, mock_audit_log, mock_metrics):
        """Test secret creation is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_secret_modified(
                actor="user_admin",
                secret_type="api_key",
                store="integrations",
                action="create",
                secret_id="secret_123",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["secret_created"]

    @pytest.mark.asyncio
    async def test_logs_secret_deletion(self, mock_audit_log, mock_metrics):
        """Test secret deletion is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_secret_modified(
                actor="user_admin",
                secret_type="api_key",
                store="integrations",
                action="delete",
                secret_id="secret_123",
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["secret_deleted"]


class TestAuditMigrationStarted:
    """Tests for audit_migration_started function."""

    @pytest.mark.asyncio
    async def test_logs_migration_start(self, mock_audit_log, mock_metrics):
        """Test migration start is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_migration_started(
                actor="user_admin",
                migration_type="encrypt_secrets",
                stores=["integrations", "oauth"],
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["migration_started"]
            assert call_kwargs["details"]["stores"] == ["integrations", "oauth"]

    @pytest.mark.asyncio
    async def test_dry_run_flag(self, mock_audit_log, mock_metrics):
        """Test dry run flag is included."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_migration_started(
                actor="user_admin",
                migration_type="encrypt_secrets",
                stores=["test"],
                dry_run=True,
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["details"]["dry_run"] is True


class TestAuditMigrationCompleted:
    """Tests for audit_migration_completed function."""

    @pytest.mark.asyncio
    async def test_logs_migration_completion(self, mock_audit_log, mock_metrics):
        """Test migration completion is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_migration_completed(
                actor="user_admin",
                migration_type="encrypt_secrets",
                success=True,
                records_migrated=1500,
                errors=[],
                duration_seconds=120.5,
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["migration_completed"]
            assert call_kwargs["details"]["records_migrated"] == 1500

    @pytest.mark.asyncio
    async def test_truncates_errors(self, mock_audit_log, mock_metrics):
        """Test errors are truncated to 10."""
        many_errors = [f"error_{i}" for i in range(20)]
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_migration_completed(
                actor="user_admin",
                migration_type="encrypt_secrets",
                success=False,
                records_migrated=100,
                errors=many_errors,
                duration_seconds=30.0,
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert len(call_kwargs["details"]["errors"]) == 10
            assert call_kwargs["details"]["error_count"] == 20


class TestAuditSecurityIncident:
    """Tests for audit_security_incident function."""

    @pytest.mark.asyncio
    async def test_logs_security_incident(self, mock_audit_log, mock_metrics):
        """Test security incident is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_security_incident(
                severity="high",
                incident_type="unauthorized_access",
                description="Repeated failed login attempts",
                ip_address="10.0.0.1",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["incident_detected"]
            assert call_kwargs["details"]["severity"] == "high"
            assert call_kwargs["details"]["incident_type"] == "unauthorized_access"

    @pytest.mark.asyncio
    async def test_records_incident_metrics(self, mock_audit_log, mock_metrics):
        """Test incident metrics are recorded."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_security_incident(
                severity="critical",
                incident_type="data_breach",
                description="Sensitive data exposed",
            )

            mock_metrics["incident"].assert_called_once_with("critical", "data_breach")


class TestAuditSecurityAlert:
    """Tests for audit_security_alert function."""

    @pytest.mark.asyncio
    async def test_logs_security_alert(self, mock_audit_log, mock_metrics):
        """Test security alert is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_security_alert(
                alert_type="brute_force_detection",
                destination="siem",
                triggered_by="rate_limiter",
                severity="high",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["alert_triggered"]
            assert call_kwargs["details"]["destination"] == "siem"

    @pytest.mark.asyncio
    async def test_records_alert_metrics(self, mock_audit_log, mock_metrics):
        """Test alert metrics are recorded."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_security_alert(
                alert_type="intrusion_detection",
                destination="email",
                triggered_by="ids",
                severity="critical",
            )

            mock_metrics["alert"].assert_called_once_with("intrusion_detection", "email")


class TestAuditRequestBlocked:
    """Tests for audit_request_blocked function."""

    @pytest.mark.asyncio
    async def test_logs_blocked_request(self, mock_audit_log, mock_metrics):
        """Test blocked request is logged."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entry = await audit_request_blocked(
                reason="rate_limit",
                ip_address="192.168.1.100",
                path="/api/v1/debates",
                method="POST",
            )

            assert entry is not None
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == SECURITY_EVENTS["request_blocked"]
            assert call_kwargs["details"]["reason"] == "rate_limit"

    @pytest.mark.asyncio
    async def test_records_blocked_metrics(self, mock_audit_log, mock_metrics):
        """Test blocked request metrics are recorded."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_request_blocked(
                reason="unauthorized",
                ip_address="10.0.0.5",
            )

            mock_metrics["blocked"].assert_called_once_with("unauthorized", "10.0.0.5")

    @pytest.mark.asyncio
    async def test_handles_user_id(self, mock_audit_log, mock_metrics):
        """Test user ID is included when known."""
        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await audit_request_blocked(
                reason="forbidden",
                ip_address="192.168.1.1",
                user_id="user_123",
            )

            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["actor"] == "user_123"
            assert call_kwargs["actor_type"] == "user"


class TestGetSecurityEvents:
    """Tests for get_security_events query function."""

    @pytest.mark.asyncio
    async def test_queries_security_events(self, mock_audit_log):
        """Test querying security events."""
        mock_entry = MagicMock()
        mock_entry.details = {"severity": "high"}
        mock_audit_log.query.return_value = [mock_entry]

        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entries = await get_security_events(limit=50)

            assert len(entries) == 1
            mock_audit_log.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_converts_event_names(self, mock_audit_log):
        """Test event names are converted to full types."""
        mock_audit_log.query.return_value = []

        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            await get_security_events(event_types=["auth_success", "auth_failure"])

            call_kwargs = mock_audit_log.query.call_args[1]
            assert SECURITY_EVENTS["auth_success"] in call_kwargs["event_types"]
            assert SECURITY_EVENTS["auth_failure"] in call_kwargs["event_types"]

    @pytest.mark.asyncio
    async def test_filters_by_severity(self, mock_audit_log):
        """Test filtering by severity."""
        mock_entries = [
            MagicMock(details={"severity": "high"}),
            MagicMock(details={"severity": "low"}),
        ]
        mock_audit_log.query.return_value = mock_entries

        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entries = await get_security_events(severity="high")

            assert len(entries) == 1
            assert entries[0].details["severity"] == "high"


class TestGetAuthFailures:
    """Tests for get_auth_failures query function."""

    @pytest.mark.asyncio
    async def test_queries_auth_failures(self, mock_audit_log):
        """Test querying auth failures."""
        mock_entry = MagicMock()
        mock_entry.ip_address = "192.168.1.1"
        mock_audit_log.query.return_value = [mock_entry]

        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entries = await get_auth_failures()

            assert len(entries) == 1
            call_kwargs = mock_audit_log.query.call_args[1]
            assert call_kwargs["event_types"] == [SECURITY_EVENTS["auth_failure"]]

    @pytest.mark.asyncio
    async def test_filters_by_ip(self, mock_audit_log):
        """Test filtering by IP address."""
        mock_entries = [
            MagicMock(ip_address="192.168.1.1"),
            MagicMock(ip_address="192.168.1.2"),
        ]
        mock_audit_log.query.return_value = mock_entries

        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entries = await get_auth_failures(ip_address="192.168.1.1")

            assert len(entries) == 1
            assert entries[0].ip_address == "192.168.1.1"


class TestGetSecurityIncidents:
    """Tests for get_security_incidents query function."""

    @pytest.mark.asyncio
    async def test_queries_incidents(self, mock_audit_log):
        """Test querying security incidents."""
        mock_entry = MagicMock()
        mock_entry.details = {"severity": "critical"}
        mock_audit_log.query.return_value = [mock_entry]

        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entries = await get_security_incidents()

            assert len(entries) == 1
            call_kwargs = mock_audit_log.query.call_args[1]
            assert SECURITY_EVENTS["incident_detected"] in call_kwargs["event_types"]

    @pytest.mark.asyncio
    async def test_filters_by_severity(self, mock_audit_log):
        """Test filtering incidents by severity."""
        mock_entries = [
            MagicMock(details={"severity": "critical"}),
            MagicMock(details={"severity": "low"}),
        ]
        mock_audit_log.query.return_value = mock_entries

        with patch(
            "aragora.observability.security_audit.get_audit_log",
            return_value=mock_audit_log,
        ):
            entries = await get_security_incidents(severity="critical")

            assert len(entries) == 1
            assert entries[0].details["severity"] == "critical"
