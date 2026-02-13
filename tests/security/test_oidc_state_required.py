"""
Security tests for OIDC state parameter enforcement.

Verifies that the OIDC authentication flow requires the state parameter
on the callback endpoint to prevent CSRF attacks. The state parameter
must never be optional.

Finding 5: OIDC CSRF bypass - State parameter validation.
"""

import time
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from aragora.auth.oidc import OIDCProvider, OIDCConfig, OIDCError
from aragora.auth.sso import SSOAuthenticationError, SSOProviderType


def _make_oidc_provider() -> OIDCProvider:
    """Create a minimal OIDCProvider for testing."""
    config = OIDCConfig(
        provider_type=SSOProviderType.OIDC,
        entity_id="test-client-id",
        client_id="test-client-id",
        client_secret="test-client-secret",
        issuer_url="https://issuer.example.com",
        callback_url="https://app.example.com/auth/callback",
        authorization_endpoint="https://issuer.example.com/authorize",
        token_endpoint="https://issuer.example.com/token",
        userinfo_endpoint="https://issuer.example.com/userinfo",
        jwks_uri="https://issuer.example.com/.well-known/jwks.json",
        validate_tokens=True,
    )
    with patch(
        "aragora.auth.oidc.validate_oidc_security_settings"
    ):
        return OIDCProvider(config)


class TestOIDCStateRequired:
    """Verify that state parameter is mandatory on OIDC callback."""

    @pytest.mark.asyncio
    async def test_authenticate_rejects_none_state(self):
        """authenticate() must reject state=None with a clear error."""
        provider = _make_oidc_provider()

        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(code="auth-code-123", state=None)

        assert "state" in str(exc_info.value).lower()
        assert "MISSING_STATE" in str(exc_info.value.details.get("code", ""))

    @pytest.mark.asyncio
    async def test_authenticate_rejects_empty_string_state(self):
        """authenticate() must reject state='' with a clear error."""
        provider = _make_oidc_provider()

        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(code="auth-code-123", state="")

        assert "state" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_authenticate_rejects_missing_code(self):
        """authenticate() must still reject missing authorization code."""
        provider = _make_oidc_provider()

        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(code=None, state="some-state")

        assert "authorization code" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_authenticate_rejects_invalid_state(self):
        """authenticate() must reject a state that was never issued."""
        provider = _make_oidc_provider()

        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(code="auth-code-123", state="forged-state-value")

        assert "INVALID_STATE" in str(exc_info.value.details.get("code", ""))

    @pytest.mark.asyncio
    async def test_authenticate_accepts_valid_state(self):
        """authenticate() should proceed past state validation with a valid state.

        We mock _exchange_code to isolate state validation from network calls.
        The mock raises a distinctive error so we can confirm we got past
        state validation and into the token exchange phase.
        """
        provider = _make_oidc_provider()

        # Register a valid state
        valid_state = provider.generate_state()

        # Mock _exchange_code to raise a known error so we can confirm
        # that state validation passed and execution reached token exchange.
        sentinel_msg = "MOCK_TOKEN_EXCHANGE_REACHED"
        provider._exchange_code = AsyncMock(
            side_effect=SSOAuthenticationError(sentinel_msg)
        )

        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(code="auth-code-123", state=valid_state)

        # The error should be our sentinel, NOT a state-related error
        assert sentinel_msg in str(exc_info.value)
        error_str = str(exc_info.value).lower()
        assert "missing state" not in error_str
        assert "invalid_state" not in error_str.replace(" ", "_")

    @pytest.mark.asyncio
    async def test_authenticate_rejects_expired_state(self):
        """authenticate() must reject a state that has expired (>10 min)."""
        provider = _make_oidc_provider()

        # Register a state but backdate it beyond the 10-minute window
        expired_state = "expired-state-token"
        provider._state_store[expired_state] = time.time() - 700  # 11+ minutes ago

        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(code="auth-code-123", state=expired_state)

        assert "INVALID_STATE" in str(exc_info.value.details.get("code", ""))


class TestOIDCStateCoversFalsy:
    """Ensure all falsy state values are caught."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("falsy_state", [None, "", 0, False])
    async def test_falsy_state_values_rejected(self, falsy_state):
        """All falsy state values must be rejected."""
        provider = _make_oidc_provider()

        with pytest.raises(SSOAuthenticationError):
            await provider.authenticate(code="auth-code-123", state=falsy_state)
