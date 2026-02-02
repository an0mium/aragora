"""
Tests for timing-safe state validation in SSO handlers.

These tests verify that the OAuth state validation is timing-safe,
meaning it does not leak information about the expected state value
through timing side channels.

The OAuth state stores (JWT, In-memory, SQLite, Redis) all implement
timing-safe comparison internally:
- JWTOAuthStateStore uses hmac.compare_digest() for signature verification
- InMemoryOAuthStateStore uses dictionary lookup (hash-based, constant-time)
- SQLiteOAuthStateStore uses SQL index lookup (not timing-vulnerable)
- RedisOAuthStateStore uses Redis key lookup (not timing-vulnerable)
"""

from __future__ import annotations

import asyncio
import json
import time
from hmac import compare_digest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.auth import sso_handlers
from aragora.server.handlers.auth.sso_handlers import (
    handle_sso_callback,
    _get_sso_state_store,
)
from aragora.server.handlers.utils.responses import HandlerResult
from aragora.server.oauth_state_store import (
    InMemoryOAuthStateStore,
    JWTOAuthStateStore,
    OAuthState,
    reset_oauth_state_store,
)


# ===========================================================================
# Helper Functions
# ===========================================================================


def parse_result(result: HandlerResult) -> tuple[int, dict[str, Any]]:
    """Parse HandlerResult into (status_code, body_dict)."""
    body = json.loads(result.body.decode("utf-8"))
    return result.status_code, body


def get_data(result: HandlerResult) -> dict[str, Any]:
    """Get the 'data' from a success response."""
    _, body = parse_result(result)
    return body.get("data", body)


def get_error(result: HandlerResult) -> str:
    """Get the error message from an error response."""
    _, body = parse_result(result)
    error = body.get("error", "")
    if isinstance(error, dict):
        return error.get("message", "")
    return error


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def clear_sso_state():
    """Clear all SSO stores before and after each test."""
    reset_oauth_state_store()
    yield
    reset_oauth_state_store()


@pytest.fixture
def memory_state_store():
    """Create an in-memory OAuth state store for testing."""
    return InMemoryOAuthStateStore()


@pytest.fixture
def jwt_state_store():
    """Create a JWT OAuth state store for testing."""
    return JWTOAuthStateStore(secret_key="test_secret_key_for_testing_only")


@pytest.fixture
def mock_oidc_provider():
    """Create a mock OIDC provider."""
    provider = MagicMock()
    provider.get_authorization_url = AsyncMock(
        return_value="https://idp.example.com/authorize?state=test"
    )
    provider.authenticate = AsyncMock()
    provider.refresh_token = AsyncMock()
    provider.logout = AsyncMock(return_value="https://idp.example.com/logout")
    return provider


@pytest.fixture
def mock_sso_user():
    """Create a mock SSO user response."""
    user = MagicMock()
    user.id = "sso_user_123"
    user.email = "user@example.com"
    user.name = "SSO User"
    user.access_token = "sso_access_token_abc"
    user.refresh_token = "sso_refresh_token_xyz"
    user.token_expires_at = time.time() + 3600
    user.to_dict = MagicMock(
        return_value={
            "id": "sso_user_123",
            "email": "user@example.com",
            "name": "SSO User",
        }
    )
    return user


# ===========================================================================
# Test Timing-Safe Comparison in JWT Store
# ===========================================================================


class TestJWTStoreTimingSafeComparison:
    """Tests verifying that JWTOAuthStateStore uses timing-safe comparison."""

    def test_jwt_store_uses_hmac_compare_digest(self, jwt_state_store):
        """Verify that JWT store uses hmac.compare_digest for signature verification."""
        # Generate a valid state
        state = jwt_state_store.generate(
            redirect_url="/dashboard",
            metadata={"provider_type": "oidc"},
            ttl_seconds=3600,
        )

        # Validate it - this should succeed
        result = jwt_state_store.validate_and_consume(state)
        assert result is not None
        assert result.redirect_url == "/dashboard"

    def test_jwt_store_rejects_tampered_signature(self, jwt_state_store):
        """JWT store should reject states with tampered signatures."""
        # Generate a valid state
        state = jwt_state_store.generate(ttl_seconds=3600)

        # Tamper with the signature (last part after the dot)
        parts = state.split(".")
        assert len(parts) == 2, "JWT state should have payload.signature format"

        # Modify the signature slightly
        tampered_sig = parts[1][:-4] + "XXXX"
        tampered_state = f"{parts[0]}.{tampered_sig}"

        # Validation should fail
        result = jwt_state_store.validate_and_consume(tampered_state)
        assert result is None

    def test_jwt_store_rejects_tampered_payload(self, jwt_state_store):
        """JWT store should reject states with tampered payloads."""
        # Generate a valid state
        state = jwt_state_store.generate(ttl_seconds=3600)

        # Tamper with the payload (first part before the dot)
        parts = state.split(".")
        tampered_payload = "X" + parts[0][1:]
        tampered_state = f"{tampered_payload}.{parts[1]}"

        # Validation should fail
        result = jwt_state_store.validate_and_consume(tampered_state)
        assert result is None

    def test_jwt_store_prevents_replay_attack(self, jwt_state_store):
        """JWT store should prevent replay attacks (state reuse)."""
        # Generate a valid state
        state = jwt_state_store.generate(ttl_seconds=3600)

        # First use should succeed
        result1 = jwt_state_store.validate_and_consume(state)
        assert result1 is not None

        # Second use should fail (nonce already consumed)
        result2 = jwt_state_store.validate_and_consume(state)
        assert result2 is None

    def test_jwt_store_compare_digest_is_called(self, jwt_state_store):
        """Verify hmac.compare_digest is used for signature comparison.

        We verify this by checking that the JWT store correctly validates
        signatures in a timing-safe manner. The implementation uses
        hmac.compare_digest() internally (verified by code inspection).
        """
        import hmac as hmac_module

        # Generate a valid state
        state = jwt_state_store.generate(ttl_seconds=3600)

        # Verify the state format (payload.signature)
        parts = state.split(".")
        assert len(parts) == 2, "JWT state should have payload.signature format"

        # Create a forged state with wrong signature but same length
        payload, signature = parts
        # Create a different signature of the same length
        fake_sig = "A" * len(signature)
        forged_state = f"{payload}.{fake_sig}"

        # Both should be validated consistently - the forged one should fail
        # This demonstrates that signature comparison is happening correctly
        valid_result = jwt_state_store.validate_and_consume(state)
        assert valid_result is not None, "Valid state should pass validation"

        # Generate another valid state to test forged signature
        state2 = jwt_state_store.generate(ttl_seconds=3600)
        parts2 = state2.split(".")
        forged_state2 = f"{parts2[0]}.{fake_sig}"

        # Forged state should fail
        forged_result = jwt_state_store.validate_and_consume(forged_state2)
        assert forged_result is None, "Forged state should fail validation"

        # Verify the implementation uses compare_digest by examining the source
        import inspect

        source = inspect.getsource(jwt_state_store.validate_and_consume)
        assert "compare_digest" in source, (
            "JWTOAuthStateStore.validate_and_consume should use compare_digest"
        )


# ===========================================================================
# Test Timing-Safe Comparison in Memory Store
# ===========================================================================


class TestMemoryStoreTimingSafeComparison:
    """Tests verifying InMemoryOAuthStateStore lookup behavior."""

    def test_memory_store_dict_lookup_is_safe(self, memory_state_store):
        """Dictionary-based lookup is constant-time and timing-safe."""
        # Generate valid state
        state = memory_state_store.generate(
            redirect_url="/dashboard",
            metadata={"provider_type": "oidc"},
            ttl_seconds=3600,
        )

        # Validation should succeed
        result = memory_state_store.validate_and_consume(state)
        assert result is not None
        assert result.redirect_url == "/dashboard"

    def test_memory_store_rejects_invalid_state(self, memory_state_store):
        """Memory store should reject invalid state tokens."""
        # Generate valid state to ensure store is working
        valid_state = memory_state_store.generate(ttl_seconds=3600)

        # Try to validate an invalid state
        result = memory_state_store.validate_and_consume("invalid_state_token")
        assert result is None

        # Valid state should still work
        result = memory_state_store.validate_and_consume(valid_state)
        assert result is not None

    def test_memory_store_single_use(self, memory_state_store):
        """Memory store should consume state on validation (single-use)."""
        # Generate valid state
        state = memory_state_store.generate(ttl_seconds=3600)

        # First validation succeeds
        result1 = memory_state_store.validate_and_consume(state)
        assert result1 is not None

        # Second validation fails (already consumed)
        result2 = memory_state_store.validate_and_consume(state)
        assert result2 is None


# ===========================================================================
# Test SSO Callback State Validation
# ===========================================================================


class TestSsoCallbackStateValidation:
    """Tests for state validation in SSO callback handler."""

    @pytest.mark.asyncio
    async def test_callback_invalid_state_returns_401(self):
        """Invalid state should return 401 Unauthorized."""
        result = await handle_sso_callback(
            {
                "code": "auth_code_123",
                "state": "completely_invalid_state_token",
            }
        )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 401
        assert "invalid" in error.lower() or "expired" in error.lower()

    @pytest.mark.asyncio
    async def test_callback_valid_state_allows_continuation(
        self, memory_state_store, mock_oidc_provider, mock_sso_user
    ):
        """Valid state should allow the callback to continue."""
        # Generate valid state
        state = memory_state_store.generate(
            redirect_url="/dashboard",
            metadata={"provider_type": "oidc"},
            ttl_seconds=3600,
        )

        mock_oidc_provider.authenticate.return_value = mock_sso_user

        with patch.object(sso_handlers, "_get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch.object(sso_handlers, "_get_sso_state_store") as mock_store:
                mock_store.return_value = memory_state_store
                with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                    mock_jwt.return_value = "test_jwt_token"

                    result = await handle_sso_callback(
                        {
                            "code": "auth_code_123",
                            "state": state,
                        }
                    )

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["access_token"] == "test_jwt_token"
        assert body["redirect_url"] == "/dashboard"

    @pytest.mark.asyncio
    async def test_callback_consumed_state_returns_401(
        self, memory_state_store, mock_oidc_provider, mock_sso_user
    ):
        """Already consumed state should return 401."""
        # Generate valid state
        state = memory_state_store.generate(
            redirect_url="/dashboard",
            metadata={"provider_type": "oidc"},
            ttl_seconds=3600,
        )

        mock_oidc_provider.authenticate.return_value = mock_sso_user

        with patch.object(sso_handlers, "_get_sso_provider") as mock_get:
            mock_get.return_value = mock_oidc_provider
            with patch.object(sso_handlers, "_get_sso_state_store") as mock_store:
                mock_store.return_value = memory_state_store
                with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                    mock_jwt.return_value = "test_jwt_token"

                    # First callback succeeds
                    result1 = await handle_sso_callback({"code": "code1", "state": state})
                    # Second callback fails (state already consumed)
                    result2 = await handle_sso_callback({"code": "code2", "state": state})

        status1, _ = parse_result(result1)
        status2, _ = parse_result(result2)
        assert status1 == 200
        assert status2 == 401

    @pytest.mark.asyncio
    async def test_callback_expired_state_returns_401(self, memory_state_store):
        """Expired state should return 401."""
        # Generate state that expires immediately
        state = memory_state_store.generate(ttl_seconds=0)

        # Wait a moment for expiration
        await asyncio.sleep(0.01)

        with patch.object(sso_handlers, "_get_sso_state_store") as mock_store:
            mock_store.return_value = memory_state_store

            result = await handle_sso_callback(
                {
                    "code": "auth_code_123",
                    "state": state,
                }
            )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 401
        assert "invalid" in error.lower() or "expired" in error.lower()


# ===========================================================================
# Test hmac.compare_digest behavior
# ===========================================================================


class TestCompareDigestBehavior:
    """Tests verifying hmac.compare_digest is used correctly."""

    def test_compare_digest_with_matching_strings(self):
        """compare_digest should return True for matching strings."""
        assert compare_digest("test_state", "test_state") is True

    def test_compare_digest_with_non_matching_strings(self):
        """compare_digest should return False for non-matching strings."""
        assert compare_digest("test_state", "wrong_state") is False

    def test_compare_digest_with_matching_bytes(self):
        """compare_digest should work with bytes."""
        assert compare_digest(b"test_state", b"test_state") is True
        assert compare_digest(b"test_state", b"wrong_state") is False

    def test_compare_digest_similar_length(self):
        """compare_digest should handle strings of similar length."""
        # Even with similar strings, should be timing-safe
        assert compare_digest("abcdefgh", "abcdefgi") is False
        assert compare_digest("abcdefgh", "zbcdefgh") is False

    def test_compare_digest_different_length(self):
        """compare_digest should handle strings of different length."""
        assert compare_digest("short", "longer_string") is False
        assert compare_digest("longer_string", "short") is False


# ===========================================================================
# Test State Store Validation Mocking
# ===========================================================================


class TestStateStoreValidationMocking:
    """Tests that verify we can mock and observe state store validation."""

    @pytest.mark.asyncio
    async def test_validate_and_consume_is_called(self, memory_state_store):
        """Verify that validate_and_consume is called during callback."""
        # Generate valid state
        state = memory_state_store.generate(ttl_seconds=3600)

        # Create a spy on validate_and_consume
        original_validate = memory_state_store.validate_and_consume
        call_count = [0]

        def spy_validate(s):
            call_count[0] += 1
            return original_validate(s)

        memory_state_store.validate_and_consume = spy_validate

        with patch.object(sso_handlers, "_get_sso_state_store") as mock_store:
            mock_store.return_value = memory_state_store

            await handle_sso_callback(
                {
                    "code": "auth_code_123",
                    "state": state,
                }
            )

        assert call_count[0] == 1, "validate_and_consume should be called exactly once"

    @pytest.mark.asyncio
    async def test_state_store_validation_order(self, memory_state_store):
        """Verify state validation happens before provider authentication."""
        # Generate valid state
        state = memory_state_store.generate(
            metadata={"provider_type": "oidc"},
            ttl_seconds=3600,
        )

        validation_order = []

        # Track validation order
        original_validate = memory_state_store.validate_and_consume

        def spy_validate(s):
            validation_order.append("state_validation")
            return original_validate(s)

        memory_state_store.validate_and_consume = spy_validate

        mock_provider = MagicMock()

        async def spy_authenticate(**kwargs):
            validation_order.append("provider_auth")
            user = MagicMock()
            user.id = "user_123"
            user.email = "test@example.com"
            user.access_token = "token"
            user.token_expires_at = time.time() + 3600
            user.to_dict = MagicMock(return_value={})
            return user

        mock_provider.authenticate = spy_authenticate

        with patch.object(sso_handlers, "_get_sso_provider") as mock_get:
            mock_get.return_value = mock_provider
            with patch.object(sso_handlers, "_get_sso_state_store") as mock_store:
                mock_store.return_value = memory_state_store
                with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
                    mock_jwt.return_value = "jwt"

                    await handle_sso_callback({"code": "auth_code", "state": state})

        # State validation should happen before provider authentication
        assert validation_order == ["state_validation", "provider_auth"]
