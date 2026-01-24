"""
Tests for TeamsSSO - Microsoft Teams Single Sign-On authentication.

Tests cover:
- TeamsTokenInfo dataclass methods
- TeamsSSO initialization
- Activity-based authentication
- Azure AD token validation
- Token exchange (OBO flow)
- Graph API user fetching
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.auth.teams_sso import (
    TeamsSSO,
    TeamsTokenInfo,
    get_teams_sso,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def teams_sso():
    """Create a TeamsSSO instance for testing."""
    return TeamsSSO(
        client_id="test-client-id",
        client_secret="test-client-secret",
        tenant_id="test-tenant-id",
    )


@pytest.fixture
def sample_activity():
    """Create a sample Bot Framework activity."""
    return {
        "type": "message",
        "id": "activity-123",
        "serviceUrl": "https://smba.trafficmanager.net/teams/",
        "from": {
            "id": "29:user-id-123",
            "name": "Test User",
            "aadObjectId": "aad-object-id-456",
        },
        "conversation": {
            "id": "conv-id-789",
            "tenantId": "tenant-id-abc",
        },
        "channelData": {
            "tenant": {"id": "tenant-id-abc"},
            "channel": {"id": "channel-id-xyz"},
        },
        "recipient": {
            "id": "28:bot-id-000",
            "name": "Aragora Bot",
        },
    }


@pytest.fixture
def sample_token_claims():
    """Create sample JWT claims."""
    return {
        "oid": "aad-object-id-456",
        "tid": "tenant-id-abc",
        "upn": "user@example.com",
        "email": "user@example.com",
        "name": "Test User",
        "given_name": "Test",
        "family_name": "User",
        "preferred_username": "user@example.com",
        "iss": "https://login.microsoftonline.com/tenant-id-abc/v2.0",
        "aud": "test-client-id",
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
    }


# ===========================================================================
# TeamsTokenInfo Tests
# ===========================================================================


class TestTeamsTokenInfo:
    """Tests for TeamsTokenInfo dataclass."""

    def test_is_expired_when_expired(self):
        """Test is_expired returns True for expired token."""
        token_info = TeamsTokenInfo(
            oid="oid-123",
            tid="tid-456",
            exp=int(time.time()) - 100,  # Expired 100 seconds ago
        )

        assert token_info.is_expired() is True

    def test_is_expired_when_valid(self):
        """Test is_expired returns False for valid token."""
        token_info = TeamsTokenInfo(
            oid="oid-123",
            tid="tid-456",
            exp=int(time.time()) + 3600,  # Expires in 1 hour
        )

        assert token_info.is_expired() is False

    def test_is_expired_when_no_exp(self):
        """Test is_expired returns False when exp is None."""
        token_info = TeamsTokenInfo(
            oid="oid-123",
            tid="tid-456",
            exp=None,
        )

        assert token_info.is_expired() is False

    def test_default_raw_claims(self):
        """Test default raw_claims is empty dict."""
        token_info = TeamsTokenInfo(
            oid="oid-123",
            tid="tid-456",
        )

        assert token_info.raw_claims == {}

    def test_raw_claims_from_init(self):
        """Test raw_claims can be set in __init__."""
        claims = {"custom": "value"}
        token_info = TeamsTokenInfo(
            oid="oid-123",
            tid="tid-456",
            raw_claims=claims,
        )

        assert token_info.raw_claims == claims

    def test_all_fields_populated(self, sample_token_claims):
        """Test all optional fields can be populated."""
        token_info = TeamsTokenInfo(
            oid=sample_token_claims["oid"],
            tid=sample_token_claims["tid"],
            upn=sample_token_claims["upn"],
            email=sample_token_claims["email"],
            name=sample_token_claims["name"],
            given_name=sample_token_claims["given_name"],
            family_name=sample_token_claims["family_name"],
            preferred_username=sample_token_claims["preferred_username"],
            iss=sample_token_claims["iss"],
            aud=sample_token_claims["aud"],
            exp=sample_token_claims["exp"],
            iat=sample_token_claims["iat"],
            raw_claims=sample_token_claims,
        )

        assert token_info.oid == "aad-object-id-456"
        assert token_info.email == "user@example.com"
        assert token_info.name == "Test User"


# ===========================================================================
# TeamsSSO Initialization Tests
# ===========================================================================


class TestTeamsSSOInit:
    """Tests for TeamsSSO initialization."""

    def test_init_with_explicit_params(self):
        """Test initialization with explicit parameters."""
        sso = TeamsSSO(
            client_id="my-client-id",
            client_secret="my-secret",
            tenant_id="my-tenant",
        )

        assert sso.client_id == "my-client-id"
        assert sso.client_secret == "my-secret"
        assert sso.tenant_id == "my-tenant"

    @patch("aragora.auth.teams_sso.AZURE_AD_CLIENT_ID", "env-client-id")
    @patch("aragora.auth.teams_sso.AZURE_AD_CLIENT_SECRET", "env-secret")
    @patch("aragora.auth.teams_sso.AZURE_AD_TENANT_ID", "env-tenant")
    def test_init_with_env_vars(self):
        """Test initialization uses environment variables."""
        sso = TeamsSSO()

        assert sso.client_id == "env-client-id"
        assert sso.client_secret == "env-secret"
        assert sso.tenant_id == "env-tenant"

    def test_init_lazy_loads_identity_bridge(self, teams_sso):
        """Test identity bridge is lazy-loaded."""
        assert teams_sso._identity_bridge is None

    def test_init_lazy_loads_jwks_cache(self, teams_sso):
        """Test JWKS cache is lazy-loaded."""
        assert teams_sso._jwks_cache is None
        assert teams_sso._jwks_cache_time == 0


# ===========================================================================
# Authenticate from Activity Tests
# ===========================================================================


class TestTeamsSSOAuthenticateFromActivity:
    """Tests for authenticate_from_activity method."""

    @pytest.mark.asyncio
    async def test_authenticate_missing_aad_id(self, teams_sso):
        """Test returns None when aadObjectId is missing."""
        activity = {
            "from": {"id": "user-id", "name": "Test User"},
            "conversation": {"tenantId": "tenant-123"},
        }

        result = await teams_sso.authenticate_from_activity(activity)

        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_with_aad_id_user_exists(self, teams_sso, sample_activity):
        """Test returns SSOUser when user exists in identity bridge."""
        mock_bridge = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "aragora-user-123"
        mock_bridge.resolve_user = AsyncMock(return_value=mock_user)

        with patch.object(teams_sso, "_get_identity_bridge", return_value=mock_bridge):
            result = await teams_sso.authenticate_from_activity(sample_activity)

        assert result is not None
        assert result.id == "aragora-user-123"
        mock_bridge.resolve_user.assert_awaited_once_with(
            "aad-object-id-456",
            "tenant-id-abc",
        )

    @pytest.mark.asyncio
    async def test_authenticate_creates_user_when_missing(self, teams_sso, sample_activity):
        """Test creates user when not found and create_user_if_missing=True."""
        mock_bridge = MagicMock()
        mock_bridge.resolve_user = AsyncMock(return_value=None)

        mock_user = MagicMock()
        mock_user.id = "new-user-123"
        mock_bridge.sync_user_from_teams = AsyncMock(return_value=mock_user)

        with patch.object(teams_sso, "_get_identity_bridge", return_value=mock_bridge):
            with patch("aragora.auth.teams_sso.TeamsUserInfo") as mock_user_info_class:
                mock_user_info = MagicMock()
                mock_user_info_class.return_value = mock_user_info

                result = await teams_sso.authenticate_from_activity(
                    sample_activity, create_user_if_missing=True
                )

        assert result is not None
        assert result.id == "new-user-123"
        mock_bridge.sync_user_from_teams.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_authenticate_no_create_when_disabled(self, teams_sso, sample_activity):
        """Test returns None when user not found and creation disabled."""
        mock_bridge = MagicMock()
        mock_bridge.resolve_user = AsyncMock(return_value=None)

        with patch.object(teams_sso, "_get_identity_bridge", return_value=mock_bridge):
            result = await teams_sso.authenticate_from_activity(
                sample_activity, create_user_if_missing=False
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_uses_conversation_tenant_id(self, teams_sso):
        """Test extracts tenant ID from conversation."""
        activity = {
            "from": {"id": "user-id", "aadObjectId": "aad-123"},
            "conversation": {"tenantId": "conv-tenant-id"},
            "channelData": {},
        }

        mock_bridge = MagicMock()
        mock_bridge.resolve_user = AsyncMock(return_value=None)

        with patch.object(teams_sso, "_get_identity_bridge", return_value=mock_bridge):
            await teams_sso.authenticate_from_activity(activity, create_user_if_missing=False)

        mock_bridge.resolve_user.assert_awaited_once_with("aad-123", "conv-tenant-id")

    @pytest.mark.asyncio
    async def test_authenticate_uses_channel_data_tenant_id(self, teams_sso):
        """Test extracts tenant ID from channelData when not in conversation."""
        activity = {
            "from": {"id": "user-id", "aadObjectId": "aad-123"},
            "conversation": {},
            "channelData": {"tenant": {"id": "channel-tenant-id"}},
        }

        mock_bridge = MagicMock()
        mock_bridge.resolve_user = AsyncMock(return_value=None)

        with patch.object(teams_sso, "_get_identity_bridge", return_value=mock_bridge):
            await teams_sso.authenticate_from_activity(activity, create_user_if_missing=False)

        mock_bridge.resolve_user.assert_awaited_once_with("aad-123", "channel-tenant-id")


# ===========================================================================
# Token Validation Tests
# ===========================================================================


class TestTeamsSSOValidateToken:
    """Tests for validate_token method."""

    @pytest.mark.asyncio
    async def test_validate_token_success(self, teams_sso, sample_token_claims):
        """Test successful token validation."""
        mock_key = MagicMock()

        with (
            patch("jwt.decode") as mock_decode,
            patch("jwt.get_unverified_header") as mock_header,
            patch.object(teams_sso, "_get_signing_key", new_callable=AsyncMock) as mock_get_key,
        ):
            mock_header.return_value = {"kid": "key-id-123"}
            mock_get_key.return_value = mock_key
            mock_decode.side_effect = [
                sample_token_claims,  # First call (unverified)
                sample_token_claims,  # Second call (verified)
            ]

            result = await teams_sso.validate_token("test-token")

        assert result is not None
        assert result.oid == "aad-object-id-456"
        assert result.tid == "tenant-id-abc"
        assert result.email == "user@example.com"

    @pytest.mark.asyncio
    async def test_validate_token_expired(self, teams_sso):
        """Test returns None for expired token."""
        with patch("jwt.decode") as mock_decode:
            import jwt

            mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")

            result = await teams_sso.validate_token("expired-token")

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_token_invalid(self, teams_sso):
        """Test returns None for invalid token."""
        with patch("jwt.decode") as mock_decode:
            import jwt

            mock_decode.side_effect = jwt.InvalidTokenError("Invalid token")

            result = await teams_sso.validate_token("invalid-token")

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_token_missing_kid(self, teams_sso, sample_token_claims):
        """Test returns None when token header missing kid."""
        with patch("jwt.decode") as mock_decode, patch("jwt.get_unverified_header") as mock_header:
            mock_decode.return_value = sample_token_claims
            mock_header.return_value = {}  # No kid

            result = await teams_sso.validate_token("test-token")

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_token_no_signing_key(self, teams_sso, sample_token_claims):
        """Test returns None when signing key not found."""
        with (
            patch("jwt.decode") as mock_decode,
            patch("jwt.get_unverified_header") as mock_header,
            patch.object(teams_sso, "_get_signing_key", new_callable=AsyncMock) as mock_get_key,
        ):
            mock_decode.return_value = sample_token_claims
            mock_header.return_value = {"kid": "unknown-key"}
            mock_get_key.return_value = None

            result = await teams_sso.validate_token("test-token")

        assert result is None


# ===========================================================================
# Token Exchange (OBO) Tests
# ===========================================================================


class TestTeamsSSOExchangeToken:
    """Tests for exchange_token method (On-Behalf-Of flow)."""

    @pytest.mark.asyncio
    async def test_exchange_token_success(self, teams_sso):
        """Test successful token exchange."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await teams_sso.exchange_token("original-token")

        assert result is not None
        assert result["access_token"] == "new-access-token"

    @pytest.mark.asyncio
    async def test_exchange_token_with_scopes(self, teams_sso):
        """Test token exchange with custom scopes."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "token"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            await teams_sso.exchange_token(
                "original-token",
                scopes=["User.Read", "Mail.Read"],
            )

            call_args = mock_instance.post.call_args
            assert "User.Read Mail.Read" in str(call_args)

    @pytest.mark.asyncio
    async def test_exchange_token_missing_credentials(self):
        """Test returns None when credentials not configured."""
        sso = TeamsSSO(client_id="", client_secret="")

        result = await sso.exchange_token("token")

        assert result is None

    @pytest.mark.asyncio
    async def test_exchange_token_failure(self, teams_sso):
        """Test returns None on exchange failure."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "invalid_grant"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await teams_sso.exchange_token("invalid-token")

        assert result is None


# ===========================================================================
# Graph API Tests
# ===========================================================================


class TestTeamsSSOGetUserFromGraph:
    """Tests for get_user_from_graph method."""

    @pytest.mark.asyncio
    async def test_get_user_success(self, teams_sso):
        """Test successful user fetch from Graph API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "user-id",
            "displayName": "Test User",
            "mail": "user@example.com",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.get = AsyncMock(return_value=mock_response)

            result = await teams_sso.get_user_from_graph("access-token")

        assert result is not None
        assert result["displayName"] == "Test User"

    @pytest.mark.asyncio
    async def test_get_user_unauthorized(self, teams_sso):
        """Test returns None on 401 unauthorized."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.get = AsyncMock(return_value=mock_response)

            result = await teams_sso.get_user_from_graph("invalid-token")

        assert result is None


# ===========================================================================
# Singleton Tests
# ===========================================================================


class TestTeamsSSOSingleton:
    """Tests for singleton pattern."""

    def test_get_teams_sso_returns_singleton(self):
        """Test get_teams_sso returns the same instance."""
        import aragora.auth.teams_sso as module

        # Reset singleton
        module._sso = None

        sso1 = get_teams_sso()
        sso2 = get_teams_sso()

        assert sso1 is sso2

        # Cleanup
        module._sso = None


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestTeamsSSOErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_validate_token_handles_import_error(self, teams_sso):
        """Test gracefully handles missing jwt library."""
        with patch.dict("sys.modules", {"jwt": None}):
            with patch("builtins.__import__", side_effect=ImportError("No jwt")):
                # This should not raise, just return None
                result = await teams_sso.validate_token("token")
                # Note: actual behavior depends on how the import is done

    @pytest.mark.asyncio
    async def test_exchange_token_handles_network_error(self, teams_sso):
        """Test handles network errors gracefully."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(side_effect=Exception("Network error"))

            result = await teams_sso.exchange_token("token")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_handles_network_error(self, teams_sso):
        """Test handles Graph API network errors gracefully."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.get = AsyncMock(side_effect=Exception("Network error"))

            result = await teams_sso.get_user_from_graph("token")

        assert result is None
