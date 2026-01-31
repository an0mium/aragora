"""
Tests for Credential Proxy - Secure credential mediation for external systems.

Tests the credential proxy functionality including:
- Credential registration and management
- Scope validation
- Rate limiting per external service
- Tenant isolation
- Usage history and auditing
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.gateway.credential_proxy import (
    CredentialProxy,
    CredentialUsage,
    ExternalCredential,
    CredentialNotFoundError,
    CredentialExpiredError,
    ScopeError,
    RateLimitExceededError,
    TenantIsolationError,
    get_credential_proxy,
    set_credential_proxy,
    reset_credential_proxy,
)


# =============================================================================
# ExternalCredential Tests
# =============================================================================


class TestExternalCredential:
    """Test ExternalCredential dataclass."""

    def test_credential_creation(self):
        cred = ExternalCredential(
            credential_id="cred-1",
            external_service="slack",
            tenant_id="tenant-a",
            api_key="sk-123",
            scopes=["read", "write"],
        )
        assert cred.credential_id == "cred-1"
        assert cred.external_service == "slack"
        assert cred.tenant_id == "tenant-a"
        assert cred.api_key == "sk-123"
        assert cred.scopes == ["read", "write"]
        assert cred.is_expired is False

    def test_credential_not_expired_no_expiry(self):
        cred = ExternalCredential(
            credential_id="cred-1",
            external_service="slack",
            tenant_id="tenant-a",
            expires_at=None,
        )
        assert cred.is_expired is False

    def test_credential_not_expired_future(self):
        cred = ExternalCredential(
            credential_id="cred-1",
            external_service="slack",
            tenant_id="tenant-a",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        assert cred.is_expired is False

    def test_credential_expired(self):
        cred = ExternalCredential(
            credential_id="cred-1",
            external_service="slack",
            tenant_id="tenant-a",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert cred.is_expired is True

    def test_credential_to_dict(self):
        cred = ExternalCredential(
            credential_id="cred-1",
            external_service="slack",
            tenant_id="tenant-a",
            api_key="secret",
            scopes=["read"],
        )
        data = cred.to_dict()
        assert data["credential_id"] == "cred-1"
        assert data["external_service"] == "slack"
        assert data["tenant_id"] == "tenant-a"
        assert data["scopes"] == ["read"]
        # Should NOT include sensitive data
        assert "api_key" not in data
        assert "oauth_token" not in data

    def test_credential_with_oauth(self):
        cred = ExternalCredential(
            credential_id="cred-1",
            external_service="github",
            tenant_id="tenant-a",
            oauth_token="gho_xxx",
            refresh_token="ghr_xxx",
            scopes=["repo", "user"],
        )
        assert cred.oauth_token == "gho_xxx"
        assert cred.refresh_token == "ghr_xxx"


# =============================================================================
# CredentialUsage Tests
# =============================================================================


class TestCredentialUsage:
    """Test CredentialUsage dataclass."""

    def test_usage_creation(self):
        usage = CredentialUsage(
            credential_id="cred-1",
            external_service="slack",
            operation="send_message",
            scopes_used=["write"],
            timestamp=datetime.utcnow(),
            tenant_id="tenant-a",
            user_id="user-1",
            success=True,
        )
        assert usage.credential_id == "cred-1"
        assert usage.external_service == "slack"
        assert usage.operation == "send_message"
        assert usage.success is True
        assert usage.error_message == ""

    def test_usage_with_error(self):
        usage = CredentialUsage(
            credential_id="cred-1",
            external_service="slack",
            operation="send_message",
            scopes_used=["write"],
            timestamp=datetime.utcnow(),
            tenant_id="tenant-a",
            user_id="user-1",
            success=False,
            error_message="Rate limit exceeded",
        )
        assert usage.success is False
        assert usage.error_message == "Rate limit exceeded"

    def test_usage_to_dict(self):
        ts = datetime.utcnow()
        usage = CredentialUsage(
            credential_id="cred-1",
            external_service="slack",
            operation="send_message",
            scopes_used=["write"],
            timestamp=ts,
            tenant_id="tenant-a",
            user_id="user-1",
            success=True,
            duration_ms=150.5,
            decision_id="decision-123",
        )
        data = usage.to_dict()
        assert data["credential_id"] == "cred-1"
        assert data["external_service"] == "slack"
        assert data["operation"] == "send_message"
        assert data["scopes_used"] == ["write"]
        assert data["timestamp"] == ts.isoformat()
        assert data["tenant_id"] == "tenant-a"
        assert data["user_id"] == "user-1"
        assert data["success"] is True
        assert data["duration_ms"] == 150.5
        assert data["decision_id"] == "decision-123"


# =============================================================================
# CredentialProxy Tests
# =============================================================================


class TestCredentialProxy:
    """Test CredentialProxy class."""

    @pytest.fixture
    def proxy(self):
        return CredentialProxy(default_rate_limit=60, audit_enabled=True)

    @pytest.fixture
    def sample_credential(self):
        return ExternalCredential(
            credential_id="cred-1",
            external_service="slack",
            tenant_id="tenant-a",
            api_key="sk-123",
            scopes=["read", "write", "admin"],
        )

    # -------------------------------------------------------------------------
    # Credential Registration Tests
    # -------------------------------------------------------------------------

    def test_register_credential(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)
        assert proxy.get_credential("cred-1") is not None
        assert proxy.get_credential("cred-1").api_key == "sk-123"

    def test_unregister_credential(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)
        assert proxy.unregister_credential("cred-1") is True
        assert proxy.get_credential("cred-1") is None

    def test_unregister_nonexistent_credential(self, proxy):
        assert proxy.unregister_credential("nonexistent") is False

    def test_list_credentials(self, proxy):
        cred1 = ExternalCredential(
            credential_id="cred-1",
            external_service="slack",
            tenant_id="tenant-a",
        )
        cred2 = ExternalCredential(
            credential_id="cred-2",
            external_service="github",
            tenant_id="tenant-a",
        )
        cred3 = ExternalCredential(
            credential_id="cred-3",
            external_service="slack",
            tenant_id="tenant-b",
        )
        proxy.register_credential(cred1)
        proxy.register_credential(cred2)
        proxy.register_credential(cred3)

        # List all
        all_creds = proxy.list_credentials()
        assert len(all_creds) == 3

        # Filter by tenant
        tenant_a_creds = proxy.list_credentials(tenant_id="tenant-a")
        assert len(tenant_a_creds) == 2

        # Filter by service
        slack_creds = proxy.list_credentials(external_service="slack")
        assert len(slack_creds) == 2

        # Filter by both
        filtered = proxy.list_credentials(tenant_id="tenant-a", external_service="slack")
        assert len(filtered) == 1
        assert filtered[0].credential_id == "cred-1"

    # -------------------------------------------------------------------------
    # Rate Limit Tests
    # -------------------------------------------------------------------------

    def test_set_rate_limit(self, proxy):
        proxy.set_rate_limit("slack", 100)
        assert proxy.get_rate_limit("slack") == 100

    def test_set_rate_limit_invalid(self, proxy):
        with pytest.raises(ValueError, match="Rate limit must be positive"):
            proxy.set_rate_limit("slack", 0)

    def test_get_rate_limit_default(self, proxy):
        assert proxy.get_rate_limit("unknown_service") == 60

    def test_get_current_request_count(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)
        # Initially zero
        assert proxy.get_current_request_count("slack") == 0

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)
        proxy.set_rate_limit("slack", 2)  # Only 2 requests per minute

        async def mock_operation(cred):
            return "success"

        # First two should succeed
        for _ in range(2):
            await proxy.execute_with_credential(
                credential_id="cred-1",
                external_service="slack",
                operation="test",
                execute_fn=mock_operation,
                required_scopes=["read"],
                tenant_id="tenant-a",
                user_id="user-1",
            )

        # Third should fail
        with pytest.raises(RateLimitExceededError, match="Rate limit exceeded"):
            await proxy.execute_with_credential(
                credential_id="cred-1",
                external_service="slack",
                operation="test",
                execute_fn=mock_operation,
                required_scopes=["read"],
                tenant_id="tenant-a",
                user_id="user-1",
            )

    # -------------------------------------------------------------------------
    # execute_with_credential Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_success(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)

        async def mock_operation(cred):
            return f"used key: {cred.api_key[:5]}..."

        result = await proxy.execute_with_credential(
            credential_id="cred-1",
            external_service="slack",
            operation="send_message",
            execute_fn=mock_operation,
            required_scopes=["write"],
            tenant_id="tenant-a",
            user_id="user-1",
        )

        assert result == "used key: sk-12..."
        # Check last_used_at was updated
        assert proxy.get_credential("cred-1").last_used_at is not None

    @pytest.mark.asyncio
    async def test_execute_credential_not_found(self, proxy):
        async def mock_operation(cred):
            return "success"

        with pytest.raises(CredentialNotFoundError, match="Credential not found"):
            await proxy.execute_with_credential(
                credential_id="nonexistent",
                external_service="slack",
                operation="test",
                execute_fn=mock_operation,
                required_scopes=["read"],
                tenant_id="tenant-a",
                user_id="user-1",
            )

    @pytest.mark.asyncio
    async def test_execute_credential_expired(self, proxy):
        expired_cred = ExternalCredential(
            credential_id="cred-expired",
            external_service="slack",
            tenant_id="tenant-a",
            api_key="sk-123",
            scopes=["read"],
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        proxy.register_credential(expired_cred)

        async def mock_operation(cred):
            return "success"

        with pytest.raises(CredentialExpiredError, match="expired"):
            await proxy.execute_with_credential(
                credential_id="cred-expired",
                external_service="slack",
                operation="test",
                execute_fn=mock_operation,
                required_scopes=["read"],
                tenant_id="tenant-a",
                user_id="user-1",
            )

    @pytest.mark.asyncio
    async def test_execute_tenant_isolation(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)  # tenant-a

        async def mock_operation(cred):
            return "success"

        with pytest.raises(TenantIsolationError, match="different tenant"):
            await proxy.execute_with_credential(
                credential_id="cred-1",
                external_service="slack",
                operation="test",
                execute_fn=mock_operation,
                required_scopes=["read"],
                tenant_id="tenant-b",  # Wrong tenant!
                user_id="user-1",
            )

    @pytest.mark.asyncio
    async def test_execute_missing_scopes(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)  # Has read, write, admin

        async def mock_operation(cred):
            return "success"

        with pytest.raises(ScopeError, match="Missing scopes"):
            await proxy.execute_with_credential(
                credential_id="cred-1",
                external_service="slack",
                operation="test",
                execute_fn=mock_operation,
                required_scopes=["read", "superadmin"],  # superadmin not available
                tenant_id="tenant-a",
                user_id="user-1",
            )

    @pytest.mark.asyncio
    async def test_execute_with_decision_id(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)

        async def mock_operation(cred):
            return "success"

        await proxy.execute_with_credential(
            credential_id="cred-1",
            external_service="slack",
            operation="send_message",
            execute_fn=mock_operation,
            required_scopes=["write"],
            tenant_id="tenant-a",
            user_id="user-1",
            decision_id="decision-123",
        )

        # Check decision_id is in usage history
        history = proxy.get_usage_history(credential_id="cred-1")
        assert len(history) == 1
        assert history[0].decision_id == "decision-123"

    @pytest.mark.asyncio
    async def test_execute_logs_failure(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)

        async def failing_operation(cred):
            raise ValueError("API error")

        with pytest.raises(ValueError, match="API error"):
            await proxy.execute_with_credential(
                credential_id="cred-1",
                external_service="slack",
                operation="test",
                execute_fn=failing_operation,
                required_scopes=["read"],
                tenant_id="tenant-a",
                user_id="user-1",
            )

        # Check failure was logged
        history = proxy.get_usage_history(success_only=False)
        assert len(history) == 1
        assert history[0].success is False
        assert "API error" in history[0].error_message

    # -------------------------------------------------------------------------
    # Usage History Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_usage_history(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)

        async def mock_operation(cred):
            return "success"

        # Execute multiple operations
        for i in range(3):
            await proxy.execute_with_credential(
                credential_id="cred-1",
                external_service="slack",
                operation=f"operation-{i}",
                execute_fn=mock_operation,
                required_scopes=["read"],
                tenant_id="tenant-a",
                user_id="user-1",
            )

        history = proxy.get_usage_history()
        assert len(history) == 3
        # Most recent first
        assert history[0].operation == "operation-2"

    @pytest.mark.asyncio
    async def test_get_usage_history_filters(self, proxy):
        cred1 = ExternalCredential(
            credential_id="cred-1",
            external_service="slack",
            tenant_id="tenant-a",
            scopes=["read"],
        )
        cred2 = ExternalCredential(
            credential_id="cred-2",
            external_service="github",
            tenant_id="tenant-b",
            scopes=["repo"],
        )
        proxy.register_credential(cred1)
        proxy.register_credential(cred2)

        async def mock_operation(cred):
            return "success"

        # Use both credentials
        await proxy.execute_with_credential(
            credential_id="cred-1",
            external_service="slack",
            operation="test",
            execute_fn=mock_operation,
            required_scopes=["read"],
            tenant_id="tenant-a",
            user_id="user-1",
        )
        await proxy.execute_with_credential(
            credential_id="cred-2",
            external_service="github",
            operation="test",
            execute_fn=mock_operation,
            required_scopes=["repo"],
            tenant_id="tenant-b",
            user_id="user-2",
        )

        # Filter by credential_id
        history = proxy.get_usage_history(credential_id="cred-1")
        assert len(history) == 1
        assert history[0].credential_id == "cred-1"

        # Filter by external_service
        history = proxy.get_usage_history(external_service="github")
        assert len(history) == 1
        assert history[0].external_service == "github"

        # Filter by tenant_id
        history = proxy.get_usage_history(tenant_id="tenant-a")
        assert len(history) == 1
        assert history[0].tenant_id == "tenant-a"

        # Filter by user_id
        history = proxy.get_usage_history(user_id="user-2")
        assert len(history) == 1
        assert history[0].user_id == "user-2"

    @pytest.mark.asyncio
    async def test_get_usage_history_since(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)

        async def mock_operation(cred):
            return "success"

        await proxy.execute_with_credential(
            credential_id="cred-1",
            external_service="slack",
            operation="test",
            execute_fn=mock_operation,
            required_scopes=["read"],
            tenant_id="tenant-a",
            user_id="user-1",
        )

        # Query since future = no results
        future = datetime.utcnow() + timedelta(hours=1)
        history = proxy.get_usage_history(since=future)
        assert len(history) == 0

        # Query since past = has results
        past = datetime.utcnow() - timedelta(hours=1)
        history = proxy.get_usage_history(since=past)
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_get_usage_history_limit(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)

        async def mock_operation(cred):
            return "success"

        # Create 10 entries
        for i in range(10):
            await proxy.execute_with_credential(
                credential_id="cred-1",
                external_service="slack",
                operation=f"op-{i}",
                execute_fn=mock_operation,
                required_scopes=["read"],
                tenant_id="tenant-a",
                user_id="user-1",
            )

        history = proxy.get_usage_history(limit=5)
        assert len(history) == 5

    def test_clear_usage_history(self, proxy):
        # Manually add some history
        proxy._usage_history.append(
            CredentialUsage(
                credential_id="cred-1",
                external_service="slack",
                operation="test",
                scopes_used=["read"],
                timestamp=datetime.utcnow(),
                tenant_id="tenant-a",
                user_id="user-1",
                success=True,
            )
        )
        assert len(proxy._usage_history) == 1

        count = proxy.clear_usage_history()
        assert count == 1
        assert len(proxy._usage_history) == 0

    # -------------------------------------------------------------------------
    # Audit Callback Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_audit_callback(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)

        callback_called = []

        async def audit_callback(usage: CredentialUsage):
            callback_called.append(usage)

        proxy.add_audit_callback(audit_callback)

        async def mock_operation(cred):
            return "success"

        await proxy.execute_with_credential(
            credential_id="cred-1",
            external_service="slack",
            operation="test",
            execute_fn=mock_operation,
            required_scopes=["read"],
            tenant_id="tenant-a",
            user_id="user-1",
        )

        assert len(callback_called) == 1
        assert callback_called[0].credential_id == "cred-1"
        assert callback_called[0].success is True

    @pytest.mark.asyncio
    async def test_audit_callback_on_failure(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)

        callback_called = []

        async def audit_callback(usage: CredentialUsage):
            callback_called.append(usage)

        proxy.add_audit_callback(audit_callback)

        async def failing_operation(cred):
            raise RuntimeError("Test failure")

        with pytest.raises(RuntimeError):
            await proxy.execute_with_credential(
                credential_id="cred-1",
                external_service="slack",
                operation="test",
                execute_fn=failing_operation,
                required_scopes=["read"],
                tenant_id="tenant-a",
                user_id="user-1",
            )

        assert len(callback_called) == 1
        assert callback_called[0].success is False
        assert "Test failure" in callback_called[0].error_message

    @pytest.mark.asyncio
    async def test_audit_callback_error_does_not_break_execution(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)

        async def failing_callback(usage: CredentialUsage):
            raise Exception("Callback failed")

        proxy.add_audit_callback(failing_callback)

        async def mock_operation(cred):
            return "success"

        # Should not raise despite callback failure
        result = await proxy.execute_with_credential(
            credential_id="cred-1",
            external_service="slack",
            operation="test",
            execute_fn=mock_operation,
            required_scopes=["read"],
            tenant_id="tenant-a",
            user_id="user-1",
        )
        assert result == "success"

    @pytest.mark.asyncio
    async def test_audit_disabled(self, sample_credential):
        proxy = CredentialProxy(audit_enabled=False)
        proxy.register_credential(sample_credential)

        async def mock_operation(cred):
            return "success"

        await proxy.execute_with_credential(
            credential_id="cred-1",
            external_service="slack",
            operation="test",
            execute_fn=mock_operation,
            required_scopes=["read"],
            tenant_id="tenant-a",
            user_id="user-1",
        )

        # No history when audit disabled
        assert len(proxy._usage_history) == 0

    # -------------------------------------------------------------------------
    # Stats Tests
    # -------------------------------------------------------------------------

    def test_get_stats(self, proxy, sample_credential):
        proxy.register_credential(sample_credential)
        proxy.set_rate_limit("slack", 100)

        stats = proxy.get_stats()
        assert stats["credentials_registered"] == 1
        assert stats["usage_history_size"] == 0
        assert stats["rate_limits"]["slack"] == 100
        assert stats["default_rate_limit"] == 60
        assert stats["audit_enabled"] is True
        assert stats["audit_callbacks_count"] == 0


# =============================================================================
# Global Instance Tests
# =============================================================================


class TestGlobalInstance:
    """Test global credential proxy instance management."""

    def setup_method(self):
        """Reset global instance before each test."""
        reset_credential_proxy()

    def teardown_method(self):
        """Clean up after each test."""
        reset_credential_proxy()

    def test_get_credential_proxy_creates_instance(self):
        proxy = get_credential_proxy()
        assert proxy is not None
        assert isinstance(proxy, CredentialProxy)

    def test_get_credential_proxy_returns_same_instance(self):
        proxy1 = get_credential_proxy()
        proxy2 = get_credential_proxy()
        assert proxy1 is proxy2

    def test_set_credential_proxy(self):
        custom_proxy = CredentialProxy(default_rate_limit=120)
        set_credential_proxy(custom_proxy)
        assert get_credential_proxy() is custom_proxy
        assert get_credential_proxy().default_rate_limit == 120

    def test_reset_credential_proxy(self):
        proxy1 = get_credential_proxy()
        reset_credential_proxy()
        proxy2 = get_credential_proxy()
        assert proxy1 is not proxy2


# =============================================================================
# Integration Tests
# =============================================================================


class TestCredentialProxyIntegration:
    """Integration tests for credential proxy with simulated external calls."""

    @pytest.fixture
    def proxy(self):
        return CredentialProxy(default_rate_limit=100)

    @pytest.mark.asyncio
    async def test_multiple_services_concurrent(self, proxy):
        """Test using credentials for multiple services."""
        slack_cred = ExternalCredential(
            credential_id="slack-cred",
            external_service="slack",
            tenant_id="tenant-a",
            api_key="xoxb-slack",
            scopes=["chat:write"],
        )
        github_cred = ExternalCredential(
            credential_id="github-cred",
            external_service="github",
            tenant_id="tenant-a",
            oauth_token="ghp_xxx",
            scopes=["repo", "user"],
        )
        proxy.register_credential(slack_cred)
        proxy.register_credential(github_cred)

        # Set different rate limits
        proxy.set_rate_limit("slack", 50)
        proxy.set_rate_limit("github", 30)

        results = []

        async def slack_call(cred):
            return f"Slack: {cred.api_key[:4]}"

        async def github_call(cred):
            return f"GitHub: {cred.oauth_token[:4]}"

        # Execute both
        slack_result = await proxy.execute_with_credential(
            credential_id="slack-cred",
            external_service="slack",
            operation="post_message",
            execute_fn=slack_call,
            required_scopes=["chat:write"],
            tenant_id="tenant-a",
            user_id="user-1",
        )
        github_result = await proxy.execute_with_credential(
            credential_id="github-cred",
            external_service="github",
            operation="create_pr",
            execute_fn=github_call,
            required_scopes=["repo"],
            tenant_id="tenant-a",
            user_id="user-1",
        )

        assert slack_result == "Slack: xoxb"
        assert github_result == "GitHub: ghp_"

        # Verify separate rate limit tracking
        assert proxy.get_current_request_count("slack") == 1
        assert proxy.get_current_request_count("github") == 1

    @pytest.mark.asyncio
    async def test_credential_refresh_workflow(self, proxy):
        """Test credential that needs refresh."""
        cred = ExternalCredential(
            credential_id="cred-1",
            external_service="oauth_service",
            tenant_id="tenant-a",
            oauth_token="old_token",
            refresh_token="refresh_xxx",
            scopes=["read"],
            expires_at=datetime.utcnow() - timedelta(minutes=1),  # Expired
        )
        proxy.register_credential(cred)

        async def api_call(cred):
            return "success"

        # Should fail due to expiry
        with pytest.raises(CredentialExpiredError):
            await proxy.execute_with_credential(
                credential_id="cred-1",
                external_service="oauth_service",
                operation="test",
                execute_fn=api_call,
                required_scopes=["read"],
                tenant_id="tenant-a",
                user_id="user-1",
            )

        # Simulate refresh: unregister old, register new
        proxy.unregister_credential("cred-1")
        new_cred = ExternalCredential(
            credential_id="cred-1",
            external_service="oauth_service",
            tenant_id="tenant-a",
            oauth_token="new_token",
            refresh_token="new_refresh",
            scopes=["read"],
            expires_at=datetime.utcnow() + timedelta(hours=1),  # Not expired
        )
        proxy.register_credential(new_cred)

        # Now should succeed
        result = await proxy.execute_with_credential(
            credential_id="cred-1",
            external_service="oauth_service",
            operation="test",
            execute_fn=api_call,
            required_scopes=["read"],
            tenant_id="tenant-a",
            user_id="user-1",
        )
        assert result == "success"

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, proxy):
        """Test that credentials are properly isolated between tenants."""
        cred_a = ExternalCredential(
            credential_id="shared-name",  # Same ID but different tenants
            external_service="api",
            tenant_id="tenant-a",
            api_key="key-a",
            scopes=["read"],
        )
        cred_b = ExternalCredential(
            credential_id="shared-name-b",
            external_service="api",
            tenant_id="tenant-b",
            api_key="key-b",
            scopes=["read"],
        )
        proxy.register_credential(cred_a)
        proxy.register_credential(cred_b)

        async def api_call(cred):
            return cred.api_key

        # Tenant A can only use their credential
        result_a = await proxy.execute_with_credential(
            credential_id="shared-name",
            external_service="api",
            operation="test",
            execute_fn=api_call,
            required_scopes=["read"],
            tenant_id="tenant-a",
            user_id="user-1",
        )
        assert result_a == "key-a"

        # Tenant B cannot use tenant A's credential
        with pytest.raises(TenantIsolationError):
            await proxy.execute_with_credential(
                credential_id="shared-name",
                external_service="api",
                operation="test",
                execute_fn=api_call,
                required_scopes=["read"],
                tenant_id="tenant-b",  # Wrong tenant
                user_id="user-2",
            )


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestCredentialProxyErrors:
    """Test error handling in credential proxy."""

    @pytest.fixture
    def proxy(self):
        return CredentialProxy()

    @pytest.mark.asyncio
    async def test_execute_fn_exception_propagates(self, proxy):
        cred = ExternalCredential(
            credential_id="cred-1",
            external_service="api",
            tenant_id="tenant-a",
            scopes=["read"],
        )
        proxy.register_credential(cred)

        class CustomError(Exception):
            pass

        async def failing_fn(cred):
            raise CustomError("Custom failure")

        with pytest.raises(CustomError, match="Custom failure"):
            await proxy.execute_with_credential(
                credential_id="cred-1",
                external_service="api",
                operation="test",
                execute_fn=failing_fn,
                required_scopes=["read"],
                tenant_id="tenant-a",
                user_id="user-1",
            )

    @pytest.mark.asyncio
    async def test_all_validation_errors_are_logged(self, proxy):
        """Test that validation errors are still logged even when they raise."""

        # Test CredentialNotFoundError
        async def mock_fn(cred):
            return "success"

        with pytest.raises(CredentialNotFoundError):
            await proxy.execute_with_credential(
                credential_id="nonexistent",
                external_service="api",
                operation="test",
                execute_fn=mock_fn,
                required_scopes=["read"],
                tenant_id="tenant-a",
                user_id="user-1",
            )

        # Error should be in history
        history = proxy.get_usage_history(success_only=False)
        assert len(history) == 1
        assert history[0].success is False
        assert "not found" in history[0].error_message.lower()
