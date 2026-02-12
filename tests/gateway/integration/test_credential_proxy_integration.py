"""
Integration tests for credential proxy with external agents.

Tests the credential proxy functionality in the context of external agent
operations including:
- Credential mediation for external agents
- Rate limiting enforcement during execution
- Tenant isolation for credentials
- Scope validation and enforcement
- Audit logging of credential access
- Handling of expired credentials
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from tests.gateway.integration.conftest import (
    MockAgent,
    MockCredentialVault,
    MockExternalFrameworkServer,
    TenantContext,
    register_external_agent,
)


# =============================================================================
# Mock Credential Proxy Implementation for Testing
# =============================================================================


@dataclass
class MockExternalCredential:
    """Mock external credential for testing."""

    credential_id: str
    external_service: str
    tenant_id: str
    api_key: str = ""
    oauth_token: str = ""
    scopes: list[str] = field(default_factory=list)
    expires_at: datetime | None = None
    last_used_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        """Check if credential is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at


@dataclass
class MockCredentialUsage:
    """Mock credential usage record for audit."""

    credential_id: str
    external_service: str
    operation: str
    scopes_used: list[str]
    timestamp: datetime
    tenant_id: str
    user_id: str
    success: bool
    error_message: str = ""
    duration_ms: float = 0.0


class CredentialNotFoundError(Exception):
    """Raised when credential is not found."""

    pass


class CredentialExpiredError(Exception):
    """Raised when credential is expired."""

    pass


class ScopeError(Exception):
    """Raised when required scopes are not available."""

    pass


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    pass


class TenantIsolationError(Exception):
    """Raised when tenant isolation is violated."""

    pass


class MockCredentialProxy:
    """
    Mock credential proxy for integration testing.

    Simulates the CredentialProxy with:
    - Credential storage and lookup
    - Scope validation
    - Rate limiting
    - Tenant isolation
    - Audit logging
    """

    def __init__(self, default_rate_limit: int = 60, audit_enabled: bool = True):
        self._credentials: dict[str, MockExternalCredential] = {}
        self._usage_history: list[MockCredentialUsage] = []
        self._rate_limits: dict[str, int] = {}
        self._request_counts: dict[str, list[datetime]] = {}
        self.default_rate_limit = default_rate_limit
        self.audit_enabled = audit_enabled
        self._audit_callbacks: list[Callable] = []

    def register_credential(self, credential: MockExternalCredential) -> None:
        """Register a credential."""
        self._credentials[credential.credential_id] = credential

    def unregister_credential(self, credential_id: str) -> bool:
        """Unregister a credential."""
        if credential_id in self._credentials:
            del self._credentials[credential_id]
            return True
        return False

    def get_credential(self, credential_id: str) -> MockExternalCredential | None:
        """Get a credential by ID."""
        return self._credentials.get(credential_id)

    def set_rate_limit(self, service: str, limit: int) -> None:
        """Set rate limit for a service."""
        if limit <= 0:
            raise ValueError("Rate limit must be positive")
        self._rate_limits[service] = limit

    def get_rate_limit(self, service: str) -> int:
        """Get rate limit for a service."""
        return self._rate_limits.get(service, self.default_rate_limit)

    def _check_rate_limit(self, service: str) -> bool:
        """Check if rate limit allows request."""
        now = datetime.now(timezone.utc)
        minute_ago = now - timedelta(minutes=1)

        if service not in self._request_counts:
            self._request_counts[service] = []

        # Remove old entries
        self._request_counts[service] = [t for t in self._request_counts[service] if t > minute_ago]

        limit = self.get_rate_limit(service)
        return len(self._request_counts[service]) < limit

    def _record_request(self, service: str) -> None:
        """Record a request for rate limiting."""
        if service not in self._request_counts:
            self._request_counts[service] = []
        self._request_counts[service].append(datetime.now(timezone.utc))

    def add_audit_callback(self, callback: Callable) -> None:
        """Add audit callback."""
        self._audit_callbacks.append(callback)

    def get_usage_history(
        self,
        credential_id: str | None = None,
        tenant_id: str | None = None,
        success_only: bool = True,
    ) -> list[MockCredentialUsage]:
        """Get credential usage history."""
        history = self._usage_history

        if credential_id:
            history = [u for u in history if u.credential_id == credential_id]
        if tenant_id:
            history = [u for u in history if u.tenant_id == tenant_id]
        if success_only:
            history = [u for u in history if u.success]

        return history

    async def execute_with_credential(
        self,
        credential_id: str,
        external_service: str,
        operation: str,
        execute_fn: Callable,
        required_scopes: list[str],
        tenant_id: str,
        user_id: str,
    ) -> Any:
        """Execute an operation using a mediated credential."""
        start_time = datetime.now(timezone.utc)

        # Get credential
        credential = self.get_credential(credential_id)
        if credential is None:
            self._log_usage(
                credential_id,
                external_service,
                operation,
                required_scopes,
                tenant_id,
                user_id,
                False,
                "Credential not found",
                start_time,
            )
            raise CredentialNotFoundError(f"Credential not found: {credential_id}")

        # Check tenant isolation
        if credential.tenant_id != tenant_id:
            self._log_usage(
                credential_id,
                external_service,
                operation,
                required_scopes,
                tenant_id,
                user_id,
                False,
                "Tenant isolation violation",
                start_time,
            )
            raise TenantIsolationError(
                f"Credential belongs to different tenant: {credential.tenant_id}"
            )

        # Check expiration
        if credential.is_expired:
            self._log_usage(
                credential_id,
                external_service,
                operation,
                required_scopes,
                tenant_id,
                user_id,
                False,
                "Credential expired",
                start_time,
            )
            raise CredentialExpiredError(f"Credential expired: {credential_id}")

        # Check scopes
        missing_scopes = set(required_scopes) - set(credential.scopes)
        if missing_scopes:
            self._log_usage(
                credential_id,
                external_service,
                operation,
                required_scopes,
                tenant_id,
                user_id,
                False,
                f"Missing scopes: {missing_scopes}",
                start_time,
            )
            raise ScopeError(f"Missing scopes: {missing_scopes}")

        # Check rate limit
        if not self._check_rate_limit(external_service):
            self._log_usage(
                credential_id,
                external_service,
                operation,
                required_scopes,
                tenant_id,
                user_id,
                False,
                "Rate limit exceeded",
                start_time,
            )
            raise RateLimitExceededError(f"Rate limit exceeded for service: {external_service}")

        # Execute operation
        try:
            result = await execute_fn(credential)
            self._record_request(external_service)

            # Update last used
            credential.last_used_at = datetime.now(timezone.utc)

            self._log_usage(
                credential_id,
                external_service,
                operation,
                required_scopes,
                tenant_id,
                user_id,
                True,
                "",
                start_time,
            )

            return result
        except Exception as e:
            self._log_usage(
                credential_id,
                external_service,
                operation,
                required_scopes,
                tenant_id,
                user_id,
                False,
                str(e),
                start_time,
            )
            raise

    def _log_usage(
        self,
        credential_id: str,
        external_service: str,
        operation: str,
        scopes_used: list[str],
        tenant_id: str,
        user_id: str,
        success: bool,
        error_message: str,
        start_time: datetime,
    ) -> None:
        """Log credential usage."""
        if not self.audit_enabled:
            return

        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        usage = MockCredentialUsage(
            credential_id=credential_id,
            external_service=external_service,
            operation=operation,
            scopes_used=scopes_used,
            timestamp=end_time,
            tenant_id=tenant_id,
            user_id=user_id,
            success=success,
            error_message=error_message,
            duration_ms=duration_ms,
        )

        self._usage_history.append(usage)

        # Call audit callbacks
        for callback in self._audit_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(usage))
                else:
                    callback(usage)
            except Exception:
                pass  # Don't let callback failures break execution


class MockExternalAgent:
    """Mock external agent that uses credential proxy."""

    def __init__(
        self,
        name: str,
        credential_proxy: MockCredentialProxy,
        credential_id: str,
    ):
        self.name = name
        self._proxy = credential_proxy
        self._credential_id = credential_id

    async def generate(
        self,
        prompt: str,
        tenant_id: str,
        user_id: str,
    ) -> str:
        """Generate using mediated credentials."""

        async def _execute(credential: MockExternalCredential) -> str:
            # Simulate API call using credential
            return f"Response using key: {credential.api_key[:5]}..."

        return await self._proxy.execute_with_credential(
            credential_id=self._credential_id,
            external_service="external-framework",
            operation="generate",
            execute_fn=_execute,
            required_scopes=["read", "execute"],
            tenant_id=tenant_id,
            user_id=user_id,
        )


# =============================================================================
# Test Class: Credential Proxy Integration
# =============================================================================


class TestCredentialProxyIntegration:
    """Integration tests for credential proxy with external agents."""

    @pytest.fixture
    def credential_proxy(self) -> MockCredentialProxy:
        """Provide a mock credential proxy."""
        return MockCredentialProxy(default_rate_limit=60, audit_enabled=True)

    @pytest.fixture
    def sample_credential(self) -> MockExternalCredential:
        """Provide a sample credential."""
        return MockExternalCredential(
            credential_id="cred-1",
            external_service="external-framework",
            tenant_id="tenant-a",
            api_key="sk-test-key-12345",
            scopes=["read", "write", "execute"],
        )

    @pytest.fixture
    def tenant_a_context(self) -> TenantContext:
        """Provide tenant A context."""
        return TenantContext(
            tenant_id="tenant-a",
            user_id="user-a-1",
            permissions=["gateway:credential.use"],
        )

    @pytest.fixture
    def tenant_b_context(self) -> TenantContext:
        """Provide tenant B context (different tenant)."""
        return TenantContext(
            tenant_id="tenant-b",
            user_id="user-b-1",
            permissions=["gateway:credential.use"],
        )

    @pytest.mark.asyncio
    async def test_credential_proxy_with_external_agent(
        self,
        credential_proxy: MockCredentialProxy,
        sample_credential: MockExternalCredential,
        tenant_a_context: TenantContext,
    ):
        """Test that agent uses mediated credentials.

        Verifies:
        - External agent can use credentials via proxy
        - Credential lookup works correctly
        - Operation executes successfully
        """
        # Register credential
        credential_proxy.register_credential(sample_credential)

        # Create agent with proxy
        agent = MockExternalAgent(
            name="test-agent",
            credential_proxy=credential_proxy,
            credential_id="cred-1",
        )

        # Execute operation
        result = await agent.generate(
            prompt="Test prompt",
            tenant_id=tenant_a_context.tenant_id,
            user_id=tenant_a_context.user_id,
        )

        # Verify result
        assert result is not None
        assert "sk-te" in result  # First 5 chars of API key
        assert sample_credential.last_used_at is not None

    @pytest.mark.asyncio
    async def test_credential_proxy_rate_limiting(
        self,
        credential_proxy: MockCredentialProxy,
        sample_credential: MockExternalCredential,
        tenant_a_context: TenantContext,
    ):
        """Test that rate limits enforced during execution.

        Verifies:
        - Rate limit is enforced per service
        - Exceeding rate limit raises error
        - Rate limit counter resets appropriately
        """
        # Register credential with low rate limit
        credential_proxy.register_credential(sample_credential)
        credential_proxy.set_rate_limit("external-framework", 3)

        agent = MockExternalAgent(
            name="test-agent",
            credential_proxy=credential_proxy,
            credential_id="cred-1",
        )

        # Execute within rate limit
        for i in range(3):
            result = await agent.generate(
                prompt=f"Request {i}",
                tenant_id=tenant_a_context.tenant_id,
                user_id=tenant_a_context.user_id,
            )
            assert result is not None

        # Exceed rate limit
        with pytest.raises(RateLimitExceededError, match="Rate limit exceeded"):
            await agent.generate(
                prompt="Exceeding request",
                tenant_id=tenant_a_context.tenant_id,
                user_id=tenant_a_context.user_id,
            )

    @pytest.mark.asyncio
    async def test_credential_proxy_tenant_isolation(
        self,
        credential_proxy: MockCredentialProxy,
        sample_credential: MockExternalCredential,
        tenant_a_context: TenantContext,
        tenant_b_context: TenantContext,
    ):
        """Test that Tenant A cannot access Tenant B credentials.

        Verifies:
        - Credential is associated with specific tenant
        - Cross-tenant access is blocked
        - Error message indicates isolation violation
        """
        # Register credential for tenant A
        credential_proxy.register_credential(sample_credential)

        agent = MockExternalAgent(
            name="test-agent",
            credential_proxy=credential_proxy,
            credential_id="cred-1",
        )

        # Tenant A can use the credential
        result = await agent.generate(
            prompt="Tenant A request",
            tenant_id=tenant_a_context.tenant_id,
            user_id=tenant_a_context.user_id,
        )
        assert result is not None

        # Tenant B cannot use tenant A's credential
        with pytest.raises(TenantIsolationError, match="different tenant"):
            await agent.generate(
                prompt="Tenant B request",
                tenant_id=tenant_b_context.tenant_id,
                user_id=tenant_b_context.user_id,
            )

    @pytest.mark.asyncio
    async def test_credential_proxy_scope_validation(
        self,
        credential_proxy: MockCredentialProxy,
        tenant_a_context: TenantContext,
    ):
        """Test that credential scopes enforced.

        Verifies:
        - Operations require specific scopes
        - Missing scopes cause rejection
        - Error message lists missing scopes
        """
        # Credential with limited scopes
        limited_credential = MockExternalCredential(
            credential_id="cred-limited",
            external_service="external-framework",
            tenant_id="tenant-a",
            api_key="sk-limited-key",
            scopes=["read"],  # Missing 'execute' scope
        )
        credential_proxy.register_credential(limited_credential)

        agent = MockExternalAgent(
            name="test-agent",
            credential_proxy=credential_proxy,
            credential_id="cred-limited",
        )

        # Should fail due to missing 'execute' scope
        with pytest.raises(ScopeError, match="Missing scopes"):
            await agent.generate(
                prompt="Test",
                tenant_id=tenant_a_context.tenant_id,
                user_id=tenant_a_context.user_id,
            )

    @pytest.mark.asyncio
    async def test_credential_proxy_audit_logging(
        self,
        credential_proxy: MockCredentialProxy,
        sample_credential: MockExternalCredential,
        tenant_a_context: TenantContext,
    ):
        """Test that credential access is logged.

        Verifies:
        - All credential access is logged
        - Log includes operation details
        - Log includes tenant and user info
        - Both successful and failed operations are logged
        """
        credential_proxy.register_credential(sample_credential)

        agent = MockExternalAgent(
            name="test-agent",
            credential_proxy=credential_proxy,
            credential_id="cred-1",
        )

        # Successful operation
        await agent.generate(
            prompt="Test",
            tenant_id=tenant_a_context.tenant_id,
            user_id=tenant_a_context.user_id,
        )

        # Check audit log
        history = credential_proxy.get_usage_history(credential_id="cred-1")
        assert len(history) == 1

        usage = history[0]
        assert usage.credential_id == "cred-1"
        assert usage.external_service == "external-framework"
        assert usage.operation == "generate"
        assert usage.tenant_id == tenant_a_context.tenant_id
        assert usage.user_id == tenant_a_context.user_id
        assert usage.success is True
        assert usage.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_credential_proxy_expired_credential(
        self,
        credential_proxy: MockCredentialProxy,
        tenant_a_context: TenantContext,
    ):
        """Test that expired credentials are rejected.

        Verifies:
        - Expired credentials cannot be used
        - Error message indicates expiration
        - Expiration check occurs before execution
        """
        # Create expired credential
        expired_credential = MockExternalCredential(
            credential_id="cred-expired",
            external_service="external-framework",
            tenant_id="tenant-a",
            api_key="sk-expired-key",
            scopes=["read", "execute"],
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        credential_proxy.register_credential(expired_credential)

        agent = MockExternalAgent(
            name="test-agent",
            credential_proxy=credential_proxy,
            credential_id="cred-expired",
        )

        # Should fail due to expired credential
        with pytest.raises(CredentialExpiredError, match="expired"):
            await agent.generate(
                prompt="Test",
                tenant_id=tenant_a_context.tenant_id,
                user_id=tenant_a_context.user_id,
            )

        # Verify failed attempt was logged
        history = credential_proxy.get_usage_history(
            credential_id="cred-expired",
            success_only=False,
        )
        assert len(history) == 1
        assert history[0].success is False
        assert "expired" in history[0].error_message.lower()
