"""
Tests for aragora.server.middleware.user_auth - Supabase JWT Authentication Middleware.

Tests cover:
- User, Workspace, APIKey dataclasses
- SupabaseAuthValidator JWT validation
- APIKeyValidator API key validation
- Token extraction from headers
- authenticate_request function
- get_current_user function
- require_user decorator
- require_admin decorator
- require_plan decorator
- Error handling (401, 403 responses)
- Edge cases and caching behavior
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.middleware.user_auth import (
    User,
    Workspace,
    APIKey,
    SupabaseAuthValidator,
    APIKeyValidator,
    get_jwt_validator,
    get_api_key_validator,
    authenticate_request,
    get_current_user,
    extract_auth_token,
    require_user,
    require_admin,
    require_plan,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockHandler:
    """Mock HTTP handler for testing."""

    headers: dict[str, str]
    client_address: tuple[str, int] = ("127.0.0.1", 12345)


@pytest.fixture
def mock_handler():
    """Create a mock handler with no auth."""
    return MockHandler(headers={})


@pytest.fixture
def mock_handler_with_bearer():
    """Create a mock handler with Bearer token."""
    return MockHandler(headers={"Authorization": "Bearer test-jwt-token"})


@pytest.fixture
def mock_handler_with_api_key():
    """Create a mock handler with API key."""
    return MockHandler(headers={"Authorization": "ApiKey ara_test_key_12345"})


@pytest.fixture
def sample_jwt_payload():
    """Create a sample Supabase JWT payload."""
    return {
        "sub": "user-123",
        "email": "test@example.com",
        "role": "user",
        "aud": "authenticated",
        "exp": time.time() + 3600,  # 1 hour from now
        "iat": time.time(),
        "user_metadata": {"name": "Test User"},
        "app_metadata": {"plan": "pro", "workspace_id": "ws-456"},
    }


@pytest.fixture
def expired_jwt_payload():
    """Create an expired JWT payload."""
    return {
        "sub": "user-123",
        "email": "test@example.com",
        "role": "user",
        "aud": "authenticated",
        "exp": time.time() - 3600,  # 1 hour ago
        "iat": time.time() - 7200,
    }


@pytest.fixture
def admin_user():
    """Create an admin user for testing."""
    return User(
        id="admin-123",
        email="admin@example.com",
        role="admin",
        plan="enterprise",
    )


@pytest.fixture
def pro_user():
    """Create a pro user for testing."""
    return User(
        id="pro-user-123",
        email="pro@example.com",
        role="user",
        plan="pro",
    )


@pytest.fixture
def free_user():
    """Create a free user for testing."""
    return User(
        id="free-user-123",
        email="free@example.com",
        role="user",
        plan="free",
    )


def create_unsigned_jwt(payload: dict) -> str:
    """Create an unsigned JWT for testing (unsafe decode path)."""
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    signature_b64 = base64.urlsafe_b64encode(b"fake_signature").rstrip(b"=").decode()
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def get_status(result) -> int:
    """Extract status code from result."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple):
        return result[1]
    return 0


def get_body(result) -> dict:
    """Extract body from result."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        if isinstance(body, str):
            return json.loads(body)
        return body
    if isinstance(result, tuple):
        body = result[0]
        if isinstance(body, dict):
            return body
        return json.loads(body)
    return {}


# ===========================================================================
# Test User Dataclass
# ===========================================================================


class TestUserDataclass:
    """Tests for User dataclass."""

    def test_user_defaults(self):
        """User should have sensible defaults."""
        user = User(id="user-123", email="test@example.com")

        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.role == "user"
        assert user.metadata == {}
        assert user.plan == "free"
        assert user.workspace_id is None
        assert user.created_at is None
        assert user.last_sign_in is None

    def test_user_is_admin_property(self):
        """is_admin should return True only for admin role."""
        regular_user = User(id="u1", email="user@test.com", role="user")
        admin_user = User(id="u2", email="admin@test.com", role="admin")

        assert regular_user.is_admin is False
        assert admin_user.is_admin is True

    def test_user_is_pro_property(self):
        """is_pro should return True for pro, team, and enterprise plans."""
        free_user = User(id="u1", email="free@test.com", plan="free")
        pro_user = User(id="u2", email="pro@test.com", plan="pro")
        team_user = User(id="u3", email="team@test.com", plan="team")
        enterprise_user = User(id="u4", email="enterprise@test.com", plan="enterprise")

        assert free_user.is_pro is False
        assert pro_user.is_pro is True
        assert team_user.is_pro is True
        assert enterprise_user.is_pro is True

    def test_user_to_dict(self):
        """to_dict should return all key fields."""
        user = User(
            id="user-123",
            email="test@example.com",
            role="admin",
            plan="pro",
            workspace_id="ws-456",
        )

        result = user.to_dict()

        assert result["id"] == "user-123"
        assert result["email"] == "test@example.com"
        assert result["role"] == "admin"
        assert result["plan"] == "pro"
        assert result["workspace_id"] == "ws-456"
        assert result["is_admin"] is True
        assert result["is_pro"] is True

    def test_user_with_metadata(self):
        """User should store metadata correctly."""
        metadata = {"name": "Test User", "avatar": "https://example.com/avatar.png"}
        user = User(id="user-123", email="test@example.com", metadata=metadata)

        assert user.metadata == metadata
        assert user.metadata["name"] == "Test User"

    def test_user_service_role(self):
        """Service role users should not be admin but should work correctly."""
        service_user = User(id="service-123", email="service@example.com", role="service")

        assert service_user.is_admin is False
        assert service_user.role == "service"


# ===========================================================================
# Test Workspace Dataclass
# ===========================================================================


class TestWorkspaceDataclass:
    """Tests for Workspace dataclass."""

    def test_workspace_defaults(self):
        """Workspace should have sensible defaults."""
        workspace = Workspace(id="ws-123", name="Test Workspace", owner_id="user-123")

        assert workspace.id == "ws-123"
        assert workspace.name == "Test Workspace"
        assert workspace.owner_id == "user-123"
        assert workspace.plan == "free"
        assert workspace.max_debates == 50
        assert workspace.max_agents == 2
        assert workspace.max_members == 1
        assert workspace.member_ids == []
        assert workspace.settings == {}

    def test_workspace_to_dict(self):
        """to_dict should return all key fields including member_count."""
        workspace = Workspace(
            id="ws-123",
            name="Pro Workspace",
            owner_id="user-123",
            plan="pro",
            max_debates=500,
            member_ids=["user-456", "user-789"],
        )

        result = workspace.to_dict()

        assert result["id"] == "ws-123"
        assert result["name"] == "Pro Workspace"
        assert result["owner_id"] == "user-123"
        assert result["plan"] == "pro"
        assert result["max_debates"] == 500
        assert result["member_count"] == 3  # owner + 2 members

    def test_workspace_member_count_with_no_members(self):
        """Workspace member count should include only owner when no members."""
        workspace = Workspace(id="ws-123", name="Solo", owner_id="user-123")

        assert workspace.to_dict()["member_count"] == 1

    def test_workspace_with_custom_settings(self):
        """Workspace should store custom settings."""
        settings = {"theme": "dark", "notifications": True}
        workspace = Workspace(id="ws-123", name="Custom", owner_id="user-123", settings=settings)

        assert workspace.settings == settings


# ===========================================================================
# Test APIKey Dataclass
# ===========================================================================


class TestAPIKeyDataclass:
    """Tests for APIKey dataclass."""

    def test_api_key_defaults(self):
        """APIKey should have sensible defaults."""
        api_key = APIKey(
            id="key-123",
            user_id="user-123",
            workspace_id="ws-123",
            name="My API Key",
            key_hash="abc123hash",
            prefix="ara_xxxx",
        )

        assert api_key.id == "key-123"
        assert api_key.user_id == "user-123"
        assert api_key.workspace_id == "ws-123"
        assert api_key.name == "My API Key"
        assert api_key.key_hash == "abc123hash"
        assert api_key.prefix == "ara_xxxx"
        assert api_key.scopes == ["read", "write"]
        assert api_key.created_at is None
        assert api_key.last_used_at is None
        assert api_key.expires_at is None

    def test_api_key_to_dict_excludes_hash(self):
        """to_dict should not include the key_hash (security)."""
        api_key = APIKey(
            id="key-123",
            user_id="user-123",
            workspace_id="ws-123",
            name="My API Key",
            key_hash="abc123hash",
            prefix="ara_xxxx",
            scopes=["read"],
        )

        result = api_key.to_dict()

        assert "key_hash" not in result
        assert "user_id" not in result  # Also not included
        assert result["id"] == "key-123"
        assert result["name"] == "My API Key"
        assert result["prefix"] == "ara_xxxx"
        assert result["scopes"] == ["read"]

    def test_api_key_custom_scopes(self):
        """APIKey should accept custom scopes."""
        api_key = APIKey(
            id="key-123",
            user_id="user-123",
            workspace_id="ws-123",
            name="Read Only Key",
            key_hash="hash",
            prefix="ara_read",
            scopes=["read"],
        )

        assert api_key.scopes == ["read"]

    def test_api_key_with_expiration(self):
        """APIKey should store expiration date."""
        api_key = APIKey(
            id="key-123",
            user_id="user-123",
            workspace_id="ws-123",
            name="Expiring Key",
            key_hash="hash",
            prefix="ara_exp",
            expires_at="2025-12-31T23:59:59Z",
        )

        assert api_key.expires_at == "2025-12-31T23:59:59Z"


# ===========================================================================
# Test SupabaseAuthValidator
# ===========================================================================


class TestSupabaseAuthValidator:
    """Tests for SupabaseAuthValidator class."""

    def test_init_defaults(self):
        """Should initialize with environment variables."""
        with patch.dict(
            "os.environ",
            {"SUPABASE_JWT_SECRET": "test-secret", "SUPABASE_URL": "https://test.supabase.co"},
        ):
            validator = SupabaseAuthValidator()

            assert validator.jwt_secret == "test-secret"
            assert validator.supabase_url == "https://test.supabase.co"

    def test_init_explicit_values(self):
        """Should accept explicit constructor values."""
        validator = SupabaseAuthValidator(
            jwt_secret="explicit-secret",
            supabase_url="https://explicit.supabase.co",
        )

        assert validator.jwt_secret == "explicit-secret"
        assert validator.supabase_url == "https://explicit.supabase.co"

    def test_validate_jwt_empty_token(self):
        """Should return None for empty token."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        assert validator.validate_jwt("") is None
        assert validator.validate_jwt(None) is None

    def test_validate_jwt_with_pyjwt(self, sample_jwt_payload):
        """Should validate JWT using PyJWT when available."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                mock_jwt.decode.return_value = sample_jwt_payload

                user = validator.validate_jwt("test-token")

                assert user is not None
                assert user.id == "user-123"
                assert user.email == "test@example.com"
                assert user.role == "user"
                assert user.plan == "pro"
                assert user.workspace_id == "ws-456"
                mock_jwt.decode.assert_called_once_with(
                    "test-token",
                    "secret",
                    algorithms=["HS256"],
                    audience="authenticated",
                )

    def test_validate_jwt_caching(self, sample_jwt_payload):
        """Should cache validated tokens."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                mock_jwt.decode.return_value = sample_jwt_payload

                # First call
                user1 = validator.validate_jwt("cached-token")
                # Second call should use cache
                user2 = validator.validate_jwt("cached-token")

                assert user1 is user2  # Same object from cache
                assert mock_jwt.decode.call_count == 1  # Only called once

    def test_validate_jwt_cache_expiration(self, sample_jwt_payload):
        """Should re-validate after cache expires."""
        validator = SupabaseAuthValidator(jwt_secret="secret")
        validator._cache_ttl = 0  # Immediate expiration

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                mock_jwt.decode.return_value = sample_jwt_payload

                # First call
                validator.validate_jwt("expiring-token")
                # Wait a moment for cache to expire
                time.sleep(0.01)
                # Second call should re-validate
                validator.validate_jwt("expiring-token")

                assert mock_jwt.decode.call_count == 2

    def test_validate_jwt_expired_token(self):
        """Should return None for expired token."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                from aragora.server.middleware.user_auth import ExpiredSignatureError

                mock_jwt.decode.side_effect = ExpiredSignatureError("Token expired")

                user = validator.validate_jwt("expired-token")

                assert user is None

    def test_validate_jwt_invalid_signature(self):
        """Should return None for invalid signature."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                from aragora.server.middleware.user_auth import InvalidSignatureError

                mock_jwt.decode.side_effect = InvalidSignatureError("Bad signature")

                user = validator.validate_jwt("invalid-sig-token")

                assert user is None

    def test_validate_jwt_invalid_audience(self):
        """Should return None for invalid audience."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                from aragora.server.middleware.user_auth import InvalidAudienceError

                mock_jwt.decode.side_effect = InvalidAudienceError("Wrong audience")

                user = validator.validate_jwt("wrong-audience-token")

                assert user is None

    def test_validate_jwt_decode_error(self):
        """Should return None for malformed token."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                from aragora.server.middleware.user_auth import DecodeError

                mock_jwt.decode.side_effect = DecodeError("Malformed")

                user = validator.validate_jwt("malformed-token")

                assert user is None

    def test_validate_jwt_generic_invalid_token(self):
        """Should return None for generic InvalidTokenError."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                from aragora.server.middleware.user_auth import InvalidTokenError

                mock_jwt.decode.side_effect = InvalidTokenError("Invalid")

                user = validator.validate_jwt("invalid-token")

                assert user is None

    def test_validate_jwt_production_no_pyjwt(self):
        """Should reject tokens in production when PyJWT unavailable."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", False):
            with patch.dict("os.environ", {"ARAGORA_ENVIRONMENT": "production"}):
                user = validator.validate_jwt("test-token")

                assert user is None

    def test_validate_jwt_production_no_secret(self, sample_jwt_payload):
        """Should reject tokens in production without JWT secret."""
        validator = SupabaseAuthValidator(jwt_secret=None)

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch.dict("os.environ", {"ARAGORA_ENVIRONMENT": "production"}):
                user = validator.validate_jwt("test-token")

                assert user is None

    def test_validate_jwt_development_fallback(self, sample_jwt_payload):
        """Should use unsafe decode in development without PyJWT."""
        validator = SupabaseAuthValidator(jwt_secret=None)
        token = create_unsigned_jwt(sample_jwt_payload)

        with patch("aragora.server.middleware.user_auth.HAS_JWT", False):
            with patch.dict("os.environ", {"ARAGORA_ENVIRONMENT": "development"}):
                user = validator.validate_jwt(token)

                assert user is not None
                assert user.id == "user-123"
                assert user.email == "test@example.com"

    def test_decode_jwt_unsafe_expired(self, expired_jwt_payload):
        """Unsafe decode should still check expiration."""
        validator = SupabaseAuthValidator(jwt_secret=None)
        token = create_unsigned_jwt(expired_jwt_payload)

        result = validator._decode_jwt_unsafe(token)

        assert result is None

    def test_decode_jwt_unsafe_malformed(self):
        """Unsafe decode should handle malformed tokens."""
        validator = SupabaseAuthValidator(jwt_secret=None)

        assert validator._decode_jwt_unsafe("not.a.valid.jwt") is None
        assert validator._decode_jwt_unsafe("invalid") is None
        assert validator._decode_jwt_unsafe("") is None

    def test_payload_to_user(self, sample_jwt_payload):
        """Should correctly convert JWT payload to User."""
        validator = SupabaseAuthValidator()

        user = validator._payload_to_user(sample_jwt_payload)

        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.role == "user"
        assert user.metadata == {"name": "Test User"}
        assert user.plan == "pro"
        assert user.workspace_id == "ws-456"

    def test_clear_cache(self, sample_jwt_payload):
        """clear_cache should empty the cache."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                mock_jwt.decode.return_value = sample_jwt_payload

                validator.validate_jwt("token1")
                validator.validate_jwt("token2")

                assert len(validator._cache) == 2

                validator.clear_cache()

                assert len(validator._cache) == 0

    def test_cached_token_expiration_check(self, sample_jwt_payload):
        """Cached tokens should be invalidated when token expires."""
        validator = SupabaseAuthValidator(jwt_secret="secret")
        validator._cache_ttl = 3600  # Long cache TTL

        # Create a token that expires soon
        nearly_expired_payload = sample_jwt_payload.copy()
        nearly_expired_payload["exp"] = time.time() + 0.01  # Expires in 10ms

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                mock_jwt.decode.return_value = nearly_expired_payload

                # First call - caches the result
                user1 = validator.validate_jwt("expiring-soon-token")
                assert user1 is not None

                # Wait for token to expire
                time.sleep(0.02)

                # Second call - cache entry exists but token expired
                user2 = validator.validate_jwt("expiring-soon-token")
                assert user2 is None  # Token expired

    def test_validate_jwt_key_error_handling(self):
        """Should handle KeyError in JWT payload."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                # Payload that causes KeyError during processing
                mock_jwt.decode.side_effect = KeyError("missing_key")

                user = validator.validate_jwt("problematic-token")
                assert user is None

    def test_validate_jwt_value_error_handling(self):
        """Should handle ValueError in JWT payload."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                mock_jwt.decode.side_effect = ValueError("invalid value")

                user = validator.validate_jwt("problematic-token")
                assert user is None


# ===========================================================================
# Test APIKeyValidator
# ===========================================================================


class TestAPIKeyValidator:
    """Tests for APIKeyValidator class."""

    def test_init_defaults(self):
        """Should initialize with None storage."""
        validator = APIKeyValidator()

        assert validator._storage is None
        assert validator._cache == {}
        assert validator._cache_ttl == 300

    def test_init_with_storage(self):
        """Should accept storage parameter."""
        mock_storage = MagicMock()
        validator = APIKeyValidator(storage=mock_storage)

        assert validator._storage is mock_storage

    @pytest.mark.asyncio
    async def test_validate_key_empty(self):
        """Should return None for empty key."""
        validator = APIKeyValidator()

        assert await validator.validate_key("") is None
        assert await validator.validate_key(None) is None

    @pytest.mark.asyncio
    async def test_validate_key_wrong_prefix(self):
        """Should return None for non-ara_ prefix."""
        validator = APIKeyValidator()

        assert await validator.validate_key("wrong_prefix_key") is None
        assert await validator.validate_key("sk_live_12345") is None
        assert await validator.validate_key("api_key_12345") is None

    @pytest.mark.asyncio
    async def test_validate_key_no_storage(self):
        """Should return None when no storage configured."""
        validator = APIKeyValidator()

        result = await validator.validate_key("ara_test_key_12345")

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_key_from_storage(self):
        """Should validate key from storage."""
        mock_storage = AsyncMock()
        mock_user = User(id="user-123", email="test@example.com")
        mock_storage.get_api_key_by_hash.return_value = {
            "id": "key-123",
            "user_id": "user-123",
        }
        mock_storage.get_user.return_value = mock_user

        validator = APIKeyValidator(storage=mock_storage)
        key = "ara_test_key_12345"

        result = await validator.validate_key(key)

        assert result is mock_user
        expected_hash = hashlib.sha256(key.encode()).hexdigest()
        mock_storage.get_api_key_by_hash.assert_called_once_with(expected_hash)
        mock_storage.update_api_key_usage.assert_called_once_with("key-123")

    @pytest.mark.asyncio
    async def test_validate_key_caching(self):
        """Should cache validated keys."""
        mock_storage = AsyncMock()
        mock_user = User(id="user-123", email="test@example.com")
        mock_storage.get_api_key_by_hash.return_value = {
            "id": "key-123",
            "user_id": "user-123",
        }
        mock_storage.get_user.return_value = mock_user

        validator = APIKeyValidator(storage=mock_storage)
        key = "ara_cached_key_12345"

        # First call
        result1 = await validator.validate_key(key)
        # Second call should use cache
        result2 = await validator.validate_key(key)

        assert result1 is result2
        assert mock_storage.get_api_key_by_hash.call_count == 1

    @pytest.mark.asyncio
    async def test_validate_key_cache_expiration(self):
        """Should re-validate after cache expires."""
        mock_storage = AsyncMock()
        mock_user = User(id="user-123", email="test@example.com")
        mock_storage.get_api_key_by_hash.return_value = {
            "id": "key-123",
            "user_id": "user-123",
        }
        mock_storage.get_user.return_value = mock_user

        validator = APIKeyValidator(storage=mock_storage)
        validator._cache_ttl = 0  # Immediate expiration
        key = "ara_expiring_key"

        # First call
        await validator.validate_key(key)
        # Wait for cache to expire
        time.sleep(0.01)
        # Second call should re-validate
        await validator.validate_key(key)

        assert mock_storage.get_api_key_by_hash.call_count == 2

    @pytest.mark.asyncio
    async def test_validate_key_not_found(self):
        """Should return None when key not found."""
        mock_storage = AsyncMock()
        mock_storage.get_api_key_by_hash.return_value = None

        validator = APIKeyValidator(storage=mock_storage)

        result = await validator.validate_key("ara_nonexistent_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_key_user_not_found(self):
        """Should return None when user not found."""
        mock_storage = AsyncMock()
        mock_storage.get_api_key_by_hash.return_value = {
            "id": "key-123",
            "user_id": "user-123",
        }
        mock_storage.get_user.return_value = None

        validator = APIKeyValidator(storage=mock_storage)

        result = await validator.validate_key("ara_orphaned_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_key_storage_error(self):
        """Should return None on storage error."""
        mock_storage = AsyncMock()
        mock_storage.get_api_key_by_hash.side_effect = Exception("DB error")

        validator = APIKeyValidator(storage=mock_storage)

        result = await validator.validate_key("ara_error_key")

        assert result is None


# ===========================================================================
# Test Global Validators
# ===========================================================================


class TestGlobalValidators:
    """Tests for global validator singletons."""

    def test_get_jwt_validator_singleton(self):
        """get_jwt_validator should return singleton instance."""
        # Reset global state
        import aragora.server.middleware.user_auth as auth_module

        auth_module._jwt_validator = None

        v1 = get_jwt_validator()
        v2 = get_jwt_validator()

        assert v1 is v2
        assert isinstance(v1, SupabaseAuthValidator)

    def test_get_api_key_validator_singleton(self):
        """get_api_key_validator should return singleton instance."""
        # Reset global state
        import aragora.server.middleware.user_auth as auth_module

        auth_module._api_key_validator = None

        v1 = get_api_key_validator()
        v2 = get_api_key_validator()

        assert v1 is v2
        assert isinstance(v1, APIKeyValidator)


# ===========================================================================
# Test Token Extraction
# ===========================================================================


class TestExtractAuthToken:
    """Tests for extract_auth_token function."""

    def test_extract_none_handler(self):
        """Should return None for None handler."""
        assert extract_auth_token(None) is None

    def test_extract_no_headers_attribute(self):
        """Should return None for handler without headers."""
        handler = MagicMock(spec=[])  # No headers attribute
        assert extract_auth_token(handler) is None

    def test_extract_no_auth_header(self, mock_handler):
        """Should return None when no Authorization header."""
        assert extract_auth_token(mock_handler) is None

    def test_extract_bearer_token(self, mock_handler_with_bearer):
        """Should extract Bearer token."""
        token = extract_auth_token(mock_handler_with_bearer)
        assert token == "test-jwt-token"

    def test_extract_api_key(self, mock_handler_with_api_key):
        """Should extract API key."""
        token = extract_auth_token(mock_handler_with_api_key)
        assert token == "ara_test_key_12345"

    def test_extract_raw_token(self):
        """Should return raw token if no prefix."""
        handler = MockHandler(headers={"Authorization": "raw-token-value"})
        token = extract_auth_token(handler)
        assert token == "raw-token-value"

    def test_extract_empty_header(self):
        """Should return None for empty Authorization header."""
        handler = MockHandler(headers={"Authorization": ""})
        token = extract_auth_token(handler)
        assert token is None

    def test_extract_bearer_case_sensitive(self):
        """Bearer prefix is case sensitive."""
        handler = MockHandler(headers={"Authorization": "bearer test-token"})
        token = extract_auth_token(handler)
        # Returns raw token since "bearer" != "Bearer"
        assert token == "bearer test-token"


# ===========================================================================
# Test authenticate_request
# ===========================================================================


class TestAuthenticateRequest:
    """Tests for authenticate_request function."""

    @pytest.mark.asyncio
    async def test_no_token(self, mock_handler):
        """Should return None when no token provided."""
        result = await authenticate_request(mock_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_jwt_authentication(self, mock_handler_with_bearer):
        """Should authenticate with valid JWT."""
        mock_user = User(id="user-123", email="test@example.com")

        with patch.object(SupabaseAuthValidator, "validate_jwt", return_value=mock_user):
            result = await authenticate_request(mock_handler_with_bearer)

            assert result is mock_user

    @pytest.mark.asyncio
    async def test_api_key_authentication(self, mock_handler_with_api_key):
        """Should authenticate with valid API key when JWT fails."""
        mock_user = User(id="user-123", email="test@example.com")

        with patch.object(SupabaseAuthValidator, "validate_jwt", return_value=None):
            with patch.object(
                APIKeyValidator, "validate_key", new_callable=AsyncMock, return_value=mock_user
            ):
                result = await authenticate_request(mock_handler_with_api_key)

                assert result is mock_user

    @pytest.mark.asyncio
    async def test_fallback_from_jwt_to_api_key(self):
        """Should try API key when JWT validation fails."""
        handler = MockHandler(headers={"Authorization": "Bearer ara_test_key"})
        mock_user = User(id="user-123", email="test@example.com")

        with patch.object(SupabaseAuthValidator, "validate_jwt", return_value=None):
            with patch.object(
                APIKeyValidator, "validate_key", new_callable=AsyncMock, return_value=mock_user
            ):
                result = await authenticate_request(handler)

                assert result is mock_user

    @pytest.mark.asyncio
    async def test_both_fail(self, mock_handler_with_bearer):
        """Should return None when both auth methods fail."""
        with patch.object(SupabaseAuthValidator, "validate_jwt", return_value=None):
            with patch.object(
                APIKeyValidator, "validate_key", new_callable=AsyncMock, return_value=None
            ):
                result = await authenticate_request(mock_handler_with_bearer)

                assert result is None

    @pytest.mark.asyncio
    async def test_jwt_takes_priority_over_api_key(self, mock_handler_with_bearer):
        """JWT should be validated first and take priority."""
        jwt_user = User(id="jwt-user", email="jwt@example.com")
        api_user = User(id="api-user", email="api@example.com")

        with patch.object(SupabaseAuthValidator, "validate_jwt", return_value=jwt_user):
            with patch.object(
                APIKeyValidator, "validate_key", new_callable=AsyncMock, return_value=api_user
            ):
                result = await authenticate_request(mock_handler_with_bearer)

                # JWT user should be returned, API key not even tried
                assert result is jwt_user


# ===========================================================================
# Test get_current_user
# ===========================================================================


class TestGetCurrentUser:
    """Tests for get_current_user function (sync version)."""

    def test_no_token(self, mock_handler):
        """Should return None when no token provided."""
        result = get_current_user(mock_handler)
        assert result is None

    def test_valid_jwt(self, mock_handler_with_bearer):
        """Should return user for valid JWT."""
        mock_user = User(id="user-123", email="test@example.com")

        with patch.object(SupabaseAuthValidator, "validate_jwt", return_value=mock_user):
            result = get_current_user(mock_handler_with_bearer)

            assert result is mock_user

    def test_invalid_jwt(self, mock_handler_with_bearer):
        """Should return None for invalid JWT."""
        with patch.object(SupabaseAuthValidator, "validate_jwt", return_value=None):
            result = get_current_user(mock_handler_with_bearer)

            assert result is None

    def test_none_handler(self):
        """Should handle None handler gracefully."""
        result = get_current_user(None)
        assert result is None


# ===========================================================================
# Test require_user Decorator
# ===========================================================================


class TestRequireUserDecorator:
    """Tests for require_user decorator."""

    def test_no_handler_returns_500(self):
        """Should return 500 when no handler provided."""

        @require_user
        def endpoint():
            return {"success": True}

        result = endpoint()

        assert get_status(result) == 500

    def test_unauthenticated_returns_401(self, mock_handler):
        """Should return 401 when not authenticated."""
        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=None):

            @require_user
            def endpoint(handler):
                return {"success": True}

            result = endpoint(handler=mock_handler)

            assert get_status(result) == 401
            body = get_body(result)
            assert "Authentication required" in str(body)

    def test_authenticated_allows_access(self, mock_handler_with_bearer):
        """Should allow access with valid authentication."""
        mock_user = User(id="user-123", email="test@example.com")

        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=mock_user):

            @require_user
            def endpoint(handler, user):
                return {"success": True, "user_id": user.id}

            result = endpoint(handler=mock_handler_with_bearer)

            assert result["success"] is True
            assert result["user_id"] == "user-123"

    def test_handler_extraction_from_args(self):
        """Should extract handler from positional args."""
        mock_user = User(id="user-123", email="test@example.com")
        handler = MockHandler(headers={"Authorization": "Bearer test"})

        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=mock_user):

            @require_user
            def endpoint(self_arg, handler, user):
                return {"success": True, "user_id": user.id}

            result = endpoint(object(), handler)

            assert result["success"] is True
            assert result["user_id"] == "user-123"

    def test_user_injected_into_kwargs(self, mock_handler_with_bearer):
        """User should be injected into kwargs."""
        mock_user = User(id="user-123", email="test@example.com", role="admin")

        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=mock_user):

            @require_user
            def endpoint(handler, user):
                return {"is_admin": user.is_admin, "email": user.email}

            result = endpoint(handler=mock_handler_with_bearer)

            assert result["is_admin"] is True
            assert result["email"] == "test@example.com"


# ===========================================================================
# Test require_admin Decorator
# ===========================================================================


class TestRequireAdminDecorator:
    """Tests for require_admin decorator."""

    def test_no_handler_returns_500(self):
        """Should return 500 when no handler provided."""

        @require_admin
        def endpoint():
            return {"success": True}

        result = endpoint()

        assert get_status(result) == 500

    def test_unauthenticated_returns_401(self, mock_handler):
        """Should return 401 when not authenticated."""
        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=None):

            @require_admin
            def endpoint(handler):
                return {"success": True}

            result = endpoint(handler=mock_handler)

            assert get_status(result) == 401

    def test_non_admin_returns_403(self, mock_handler_with_bearer):
        """Should return 403 for non-admin users."""
        regular_user = User(id="user-123", email="test@example.com", role="user")

        with patch(
            "aragora.server.middleware.user_auth.get_current_user", return_value=regular_user
        ):

            @require_admin
            def endpoint(handler, user):
                return {"success": True}

            result = endpoint(handler=mock_handler_with_bearer)

            assert get_status(result) == 403
            body = get_body(result)
            assert "Admin access required" in str(body)

    def test_admin_allows_access(self, mock_handler_with_bearer):
        """Should allow access for admin users."""
        admin_user = User(id="admin-123", email="admin@example.com", role="admin")

        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=admin_user):

            @require_admin
            def endpoint(handler, user):
                return {"success": True, "is_admin": user.is_admin}

            result = endpoint(handler=mock_handler_with_bearer)

            assert result["success"] is True
            assert result["is_admin"] is True

    def test_service_role_not_admin(self, mock_handler_with_bearer):
        """Service role should not be considered admin."""
        service_user = User(id="service-123", email="service@example.com", role="service")

        with patch(
            "aragora.server.middleware.user_auth.get_current_user", return_value=service_user
        ):

            @require_admin
            def endpoint(handler, user):
                return {"success": True}

            result = endpoint(handler=mock_handler_with_bearer)

            assert get_status(result) == 403


# ===========================================================================
# Test require_plan Decorator
# ===========================================================================


class TestRequirePlanDecorator:
    """Tests for require_plan decorator."""

    def test_unauthenticated_returns_401(self, mock_handler):
        """Should return 401 when not authenticated."""
        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=None):

            @require_plan("pro")
            def endpoint(handler):
                return {"success": True}

            result = endpoint(handler=mock_handler)

            assert get_status(result) == 401

    def test_insufficient_plan_returns_403(self, mock_handler_with_bearer):
        """Should return 403 when plan is insufficient."""
        free_user = User(id="user-123", email="test@example.com", plan="free")

        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=free_user):

            @require_plan("pro")
            def endpoint(handler, user):
                return {"success": True}

            result = endpoint(handler=mock_handler_with_bearer)

            assert get_status(result) == 403
            body = get_body(result)
            assert "pro plan" in str(body).lower()

    def test_matching_plan_allows_access(self, mock_handler_with_bearer):
        """Should allow access when plan matches requirement."""
        pro_user = User(id="user-123", email="test@example.com", plan="pro")

        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=pro_user):

            @require_plan("pro")
            def endpoint(handler, user):
                return {"success": True, "plan": user.plan}

            result = endpoint(handler=mock_handler_with_bearer)

            assert result["success"] is True
            assert result["plan"] == "pro"

    def test_higher_plan_allows_access(self, mock_handler_with_bearer):
        """Should allow access when plan exceeds requirement."""
        enterprise_user = User(id="user-123", email="test@example.com", plan="enterprise")

        with patch(
            "aragora.server.middleware.user_auth.get_current_user", return_value=enterprise_user
        ):

            @require_plan("pro")
            def endpoint(handler, user):
                return {"success": True, "plan": user.plan}

            result = endpoint(handler=mock_handler_with_bearer)

            assert result["success"] is True
            assert result["plan"] == "enterprise"

    def test_plan_hierarchy(self, mock_handler_with_bearer):
        """Should respect plan hierarchy: free < pro < team < enterprise."""
        test_cases = [
            ("free", "free", True),
            ("free", "pro", True),
            ("free", "team", True),
            ("free", "enterprise", True),
            ("pro", "free", False),
            ("pro", "pro", True),
            ("pro", "team", True),
            ("pro", "enterprise", True),
            ("team", "free", False),
            ("team", "pro", False),
            ("team", "team", True),
            ("team", "enterprise", True),
            ("enterprise", "free", False),
            ("enterprise", "pro", False),
            ("enterprise", "team", False),
            ("enterprise", "enterprise", True),
        ]

        for required_plan, user_plan, should_succeed in test_cases:
            user = User(id="user-123", email="test@example.com", plan=user_plan)

            with patch("aragora.server.middleware.user_auth.get_current_user", return_value=user):

                @require_plan(required_plan)
                def endpoint(handler, user):
                    return {"success": True}

                result = endpoint(handler=mock_handler_with_bearer)

                if should_succeed:
                    assert result.get("success") is True, (
                        f"{user_plan} should access {required_plan}"
                    )
                else:
                    assert get_status(result) == 403, (
                        f"{user_plan} should not access {required_plan}"
                    )

    def test_unknown_plan_treated_as_zero(self, mock_handler_with_bearer):
        """Unknown plans should be treated as level 0."""
        unknown_plan_user = User(id="user-123", email="test@example.com", plan="unknown")

        with patch(
            "aragora.server.middleware.user_auth.get_current_user", return_value=unknown_plan_user
        ):

            @require_plan("free")
            def endpoint(handler, user):
                return {"success": True}

            result = endpoint(handler=mock_handler_with_bearer)
            assert result["success"] is True  # Unknown >= free (both level 0)

    def test_require_free_plan(self, mock_handler_with_bearer):
        """Free plan requirement should allow all authenticated users."""
        free_user = User(id="user-123", email="test@example.com", plan="free")

        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=free_user):

            @require_plan("free")
            def endpoint(handler, user):
                return {"success": True}

            result = endpoint(handler=mock_handler_with_bearer)
            assert result["success"] is True


# ===========================================================================
# Test Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for user_auth middleware."""

    def test_handler_extraction_from_first_arg_with_headers(self):
        """Should find handler in first positional arg."""

        @require_user
        def endpoint(handler, user):
            return {"user_id": user.id}

        handler = MockHandler(headers={"Authorization": "Bearer token"})
        mock_user = User(id="user-123", email="test@example.com")

        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=mock_user):
            result = endpoint(handler)

            assert result["user_id"] == "user-123"

    def test_user_metadata_preserved(self, sample_jwt_payload):
        """Should preserve user_metadata from JWT payload."""
        validator = SupabaseAuthValidator()

        user = validator._payload_to_user(sample_jwt_payload)

        assert user.metadata == {"name": "Test User"}

    def test_missing_optional_jwt_fields(self):
        """Should handle missing optional JWT fields."""
        validator = SupabaseAuthValidator()
        minimal_payload = {
            "sub": "user-123",
            "email": "test@example.com",
        }

        user = validator._payload_to_user(minimal_payload)

        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.role == "user"  # Default
        assert user.plan == "free"  # Default
        assert user.metadata == {}  # Default

    def test_api_key_hash_consistency(self):
        """API key hash should be consistent."""
        key = "ara_test_key_12345"
        hash1 = hashlib.sha256(key.encode()).hexdigest()
        hash2 = hashlib.sha256(key.encode()).hexdigest()

        assert hash1 == hash2

    def test_workspace_member_count_includes_owner(self):
        """Workspace member_count should include owner."""
        workspace = Workspace(
            id="ws-123",
            name="Test",
            owner_id="owner-123",
            member_ids=[],
        )

        assert workspace.to_dict()["member_count"] == 1  # Just owner

        workspace.member_ids = ["user-1", "user-2", "user-3"]
        assert workspace.to_dict()["member_count"] == 4  # Owner + 3 members

    def test_jwt_structure_error_handling(self):
        """Should handle JWT payload structure errors gracefully."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                # Missing required fields
                mock_jwt.decode.return_value = {}

                user = validator.validate_jwt("test-token")

                # Should still create user with empty/default values
                assert user is not None
                assert user.id == ""
                assert user.email == ""

    def test_system_error_in_production(self):
        """Should raise system errors in production (fail closed)."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                mock_jwt.decode.side_effect = RuntimeError("System error")

                with patch.dict("os.environ", {"ARAGORA_ENVIRONMENT": "production"}):
                    with pytest.raises(RuntimeError):
                        validator.validate_jwt("test-token")

    def test_system_error_in_development(self):
        """Should return None for system errors in development."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                mock_jwt.decode.side_effect = RuntimeError("System error")

                with patch.dict("os.environ", {"ARAGORA_ENVIRONMENT": "development"}):
                    user = validator.validate_jwt("test-token")
                    assert user is None

    def test_type_error_in_jwt_validation(self):
        """Should handle TypeError in JWT validation."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                mock_jwt.decode.side_effect = TypeError("type error")

                user = validator.validate_jwt("test-token")
                assert user is None


# ===========================================================================
# Test Module Exports
# ===========================================================================


class TestModuleExports:
    """Tests for module's __all__ exports."""

    def test_all_exports_importable(self):
        """All items in __all__ can be imported."""
        from aragora.server.middleware import user_auth

        for name in user_auth.__all__:
            assert hasattr(user_auth, name), f"Missing export: {name}"

    def test_exported_items(self):
        """Key items are exported in __all__."""
        from aragora.server.middleware.user_auth import __all__

        expected = [
            "User",
            "Workspace",
            "APIKey",
            "SupabaseAuthValidator",
            "APIKeyValidator",
            "get_jwt_validator",
            "get_api_key_validator",
            "authenticate_request",
            "get_current_user",
            "extract_auth_token",
            "require_user",
            "require_admin",
            "require_plan",
        ]

        for item in expected:
            assert item in __all__, f"Expected {item} in __all__"


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_jwt_authentication_flow(self, sample_jwt_payload):
        """Test complete JWT authentication from request to user."""
        handler = MockHandler(headers={"Authorization": "Bearer test-jwt"})
        mock_user = User(id="user-123", email="test@example.com", plan="pro")

        # Mock at the validator level for integration test
        with patch.object(SupabaseAuthValidator, "validate_jwt", return_value=mock_user):
            user = await authenticate_request(handler)

            assert user is not None
            assert user.id == "user-123"
            assert user.plan == "pro"

    def test_decorator_chain(self, mock_handler_with_bearer):
        """Test that decorators can be used together logically."""
        admin_user = User(
            id="admin-123", email="admin@example.com", role="admin", plan="enterprise"
        )

        with patch("aragora.server.middleware.user_auth.get_current_user", return_value=admin_user):
            # require_admin already implies require_user
            @require_admin
            def admin_endpoint(handler, user):
                return {"admin": True, "plan": user.plan}

            result = admin_endpoint(handler=mock_handler_with_bearer)
            assert result["admin"] is True
            assert result["plan"] == "enterprise"

    def test_cache_isolation_between_validators(self, sample_jwt_payload):
        """Different validators should have isolated caches."""
        jwt_validator1 = SupabaseAuthValidator(jwt_secret="secret1")
        jwt_validator2 = SupabaseAuthValidator(jwt_secret="secret2")

        with patch("aragora.server.middleware.user_auth.HAS_JWT", True):
            with patch("aragora.server.middleware.user_auth._jwt_module") as mock_jwt:
                mock_jwt.decode.return_value = sample_jwt_payload

                jwt_validator1.validate_jwt("token1")
                jwt_validator2.validate_jwt("token2")

                assert "token1" in jwt_validator1._cache
                assert "token1" not in jwt_validator2._cache
                assert "token2" in jwt_validator2._cache
                assert "token2" not in jwt_validator1._cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
