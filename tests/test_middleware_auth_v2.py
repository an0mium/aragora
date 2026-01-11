"""
Tests for aragora.server.middleware.auth_v2 module.

Tests JWT validation, API key validation, and authentication decorators.
"""

import base64
import json
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.server.middleware.auth_v2 import (
    User,
    Workspace,
    APIKey,
    SupabaseAuthValidator,
    APIKeyValidator,
    extract_auth_token,
    authenticate_request,
    get_current_user,
    require_user,
    require_admin,
    require_plan,
    get_jwt_validator,
    get_api_key_validator,
)


class TestUserModel:
    """Tests for User dataclass."""

    def test_user_defaults(self):
        """User has expected default values."""
        user = User(id="user-1", email="test@example.com")

        assert user.id == "user-1"
        assert user.email == "test@example.com"
        assert user.role == "user"
        assert user.plan == "free"
        assert user.metadata == {}
        assert user.workspace_id is None

    def test_user_is_admin(self):
        """is_admin returns True for admin role."""
        admin = User(id="admin-1", email="admin@example.com", role="admin")
        regular = User(id="user-1", email="user@example.com", role="user")

        assert admin.is_admin is True
        assert regular.is_admin is False

    def test_user_is_pro(self):
        """is_pro returns True for pro, team, or enterprise plans."""
        free_user = User(id="1", email="free@example.com", plan="free")
        pro_user = User(id="2", email="pro@example.com", plan="pro")
        team_user = User(id="3", email="team@example.com", plan="team")
        enterprise_user = User(id="4", email="ent@example.com", plan="enterprise")

        assert free_user.is_pro is False
        assert pro_user.is_pro is True
        assert team_user.is_pro is True
        assert enterprise_user.is_pro is True

    def test_user_to_dict(self):
        """to_dict() returns correct representation."""
        user = User(
            id="user-1",
            email="test@example.com",
            role="admin",
            plan="pro",
            workspace_id="ws-1",
        )
        data = user.to_dict()

        assert data["id"] == "user-1"
        assert data["email"] == "test@example.com"
        assert data["role"] == "admin"
        assert data["plan"] == "pro"
        assert data["workspace_id"] == "ws-1"
        assert data["is_admin"] is True
        assert data["is_pro"] is True


class TestWorkspaceModel:
    """Tests for Workspace dataclass."""

    def test_workspace_defaults(self):
        """Workspace has expected default values."""
        ws = Workspace(id="ws-1", name="Test Workspace", owner_id="user-1")

        assert ws.id == "ws-1"
        assert ws.name == "Test Workspace"
        assert ws.owner_id == "user-1"
        assert ws.plan == "free"
        assert ws.max_debates == 50
        assert ws.max_agents == 2
        assert ws.max_members == 1
        assert ws.member_ids == []

    def test_workspace_to_dict(self):
        """to_dict() includes member count."""
        ws = Workspace(
            id="ws-1",
            name="Test Workspace",
            owner_id="user-1",
            member_ids=["user-2", "user-3"],
        )
        data = ws.to_dict()

        assert data["id"] == "ws-1"
        assert data["member_count"] == 3  # owner + 2 members


class TestAPIKeyModel:
    """Tests for APIKey dataclass."""

    def test_api_key_defaults(self):
        """APIKey has expected default values."""
        key = APIKey(
            id="key-1",
            user_id="user-1",
            workspace_id="ws-1",
            name="Test Key",
            key_hash="abc123",
            prefix="ara_xxxx",
        )

        assert key.id == "key-1"
        assert key.scopes == ["read", "write"]
        assert key.expires_at is None

    def test_api_key_to_dict(self):
        """to_dict() excludes sensitive key_hash."""
        key = APIKey(
            id="key-1",
            user_id="user-1",
            workspace_id="ws-1",
            name="Test Key",
            key_hash="secret_hash",
            prefix="ara_xxxx",
        )
        data = key.to_dict()

        assert data["id"] == "key-1"
        assert data["name"] == "Test Key"
        assert data["prefix"] == "ara_xxxx"
        assert "key_hash" not in data


class TestSupabaseAuthValidator:
    """Tests for SupabaseAuthValidator class."""

    def test_validator_init(self):
        """Validator initializes with secrets from env or params."""
        validator = SupabaseAuthValidator(
            jwt_secret="test-secret",
            supabase_url="https://test.supabase.co",
        )

        assert validator.jwt_secret == "test-secret"
        assert validator.supabase_url == "https://test.supabase.co"
        assert validator._cache == {}

    def test_validate_jwt_empty_token(self):
        """validate_jwt returns None for empty token."""
        validator = SupabaseAuthValidator(jwt_secret="secret")
        result = validator.validate_jwt("")

        assert result is None

    def test_validate_jwt_none_token(self):
        """validate_jwt returns None for None token."""
        validator = SupabaseAuthValidator(jwt_secret="secret")
        result = validator.validate_jwt(None)

        assert result is None

    def test_validate_jwt_caches_result(self):
        """validate_jwt caches valid tokens."""
        # Create a valid JWT-like token structure for testing
        payload = {
            "sub": "user-123",
            "email": "test@example.com",
            "role": "user",
            "exp": time.time() + 3600,
        }
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        token = f"header.{payload_b64}.signature"

        validator = SupabaseAuthValidator()

        with patch.dict('os.environ', {'ARAGORA_ENVIRONMENT': 'development'}):
            # First call
            user1 = validator.validate_jwt(token)
            assert user1 is not None
            assert user1.id == "user-123"

            # Second call should use cache
            user2 = validator.validate_jwt(token)
            assert user2 is user1  # Same object from cache

    def test_validate_jwt_rejects_expired_cached_token(self):
        """validate_jwt rejects expired tokens from cache."""
        validator = SupabaseAuthValidator(jwt_secret="secret")

        # Manually add expired token to cache
        expired_user = User(id="old", email="old@example.com")
        expired_time = time.time() - 3600  # 1 hour ago
        validator._cache["old-token"] = (expired_user, time.time(), expired_time)

        result = validator.validate_jwt("old-token")
        assert result is None

    def test_decode_jwt_unsafe_valid_token(self):
        """_decode_jwt_unsafe decodes valid JWT parts."""
        validator = SupabaseAuthValidator()

        # Create a mock JWT
        payload = {
            "sub": "user-123",
            "email": "test@example.com",
            "exp": time.time() + 3600,
        }
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        token = f"header.{payload_b64}.signature"

        result = validator._decode_jwt_unsafe(token)

        assert result is not None
        assert result["sub"] == "user-123"
        assert result["email"] == "test@example.com"

    def test_decode_jwt_unsafe_expired_token(self):
        """_decode_jwt_unsafe rejects expired tokens."""
        validator = SupabaseAuthValidator()

        # Create expired JWT
        payload = {
            "sub": "user-123",
            "exp": time.time() - 3600,  # Expired 1 hour ago
        }
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        token = f"header.{payload_b64}.signature"

        result = validator._decode_jwt_unsafe(token)

        assert result is None

    def test_decode_jwt_unsafe_invalid_format(self):
        """_decode_jwt_unsafe rejects invalid token format."""
        validator = SupabaseAuthValidator()

        # Invalid tokens
        assert validator._decode_jwt_unsafe("invalid") is None
        assert validator._decode_jwt_unsafe("only.two") is None
        assert validator._decode_jwt_unsafe("") is None

    def test_payload_to_user(self):
        """_payload_to_user creates User from JWT payload."""
        validator = SupabaseAuthValidator()

        payload = {
            "sub": "user-123",
            "email": "test@example.com",
            "role": "admin",
            "user_metadata": {"display_name": "Test User"},
            "app_metadata": {"plan": "pro", "workspace_id": "ws-1"},
            "iat": 1234567890,
        }

        user = validator._payload_to_user(payload)

        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.role == "admin"
        assert user.plan == "pro"
        assert user.workspace_id == "ws-1"
        assert user.metadata == {"display_name": "Test User"}

    def test_clear_cache(self):
        """clear_cache empties the token cache."""
        validator = SupabaseAuthValidator()
        validator._cache["token"] = (User(id="1", email="t@t.com"), time.time(), time.time() + 3600)

        validator.clear_cache()

        assert validator._cache == {}


class TestAPIKeyValidator:
    """Tests for APIKeyValidator class."""

    def test_validator_init(self):
        """Validator initializes with storage."""
        storage = MagicMock()
        validator = APIKeyValidator(storage=storage)

        assert validator._storage is storage
        assert validator._cache == {}

    @pytest.mark.asyncio
    async def test_validate_key_empty_key(self):
        """validate_key returns None for empty key."""
        validator = APIKeyValidator()
        result = await validator.validate_key("")

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_key_invalid_prefix(self):
        """validate_key returns None for invalid prefix."""
        validator = APIKeyValidator()
        result = await validator.validate_key("invalid_prefix")

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_key_uses_cache(self):
        """validate_key returns cached user on second call."""
        validator = APIKeyValidator()

        # Add to cache
        cached_user = User(id="cached", email="cached@example.com")
        validator._cache["ara_testkey123"] = (cached_user, time.time())

        result = await validator.validate_key("ara_testkey123")

        assert result is cached_user

    @pytest.mark.asyncio
    async def test_validate_key_expired_cache(self):
        """validate_key ignores expired cache entries."""
        validator = APIKeyValidator()
        validator._cache_ttl = 300

        # Add expired cache entry
        old_user = User(id="old", email="old@example.com")
        validator._cache["ara_testkey123"] = (old_user, time.time() - 600)

        result = await validator.validate_key("ara_testkey123")

        # Should return None since cache expired and no storage
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_key_with_storage(self):
        """validate_key queries storage for valid key."""
        storage = AsyncMock()
        storage.get_api_key_by_hash.return_value = {"id": "key-1", "user_id": "user-1"}
        storage.get_user.return_value = User(id="user-1", email="test@example.com")
        storage.update_api_key_usage.return_value = None

        validator = APIKeyValidator(storage=storage)
        result = await validator.validate_key("ara_testkey123")

        assert result is not None
        assert result.id == "user-1"
        storage.update_api_key_usage.assert_called_once_with("key-1")


class TestExtractAuthToken:
    """Tests for extract_auth_token function."""

    def test_extract_bearer_token(self):
        """extract_auth_token extracts Bearer token."""
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer jwt-token-here"}

        token = extract_auth_token(handler)

        assert token == "jwt-token-here"

    def test_extract_api_key_token(self):
        """extract_auth_token extracts ApiKey token."""
        handler = MagicMock()
        handler.headers = {"Authorization": "ApiKey ara_abc123"}

        token = extract_auth_token(handler)

        assert token == "ara_abc123"

    def test_extract_raw_token(self):
        """extract_auth_token returns raw auth header if no prefix."""
        handler = MagicMock()
        handler.headers = {"Authorization": "some-token"}

        token = extract_auth_token(handler)

        assert token == "some-token"

    def test_extract_no_header(self):
        """extract_auth_token returns None for missing header."""
        handler = MagicMock()
        handler.headers = {}

        token = extract_auth_token(handler)

        assert token is None

    def test_extract_none_handler(self):
        """extract_auth_token returns None for None handler."""
        token = extract_auth_token(None)

        assert token is None


class TestAuthenticateRequest:
    """Tests for authenticate_request function."""

    @pytest.mark.asyncio
    async def test_authenticate_no_token(self):
        """authenticate_request returns None without token."""
        handler = MagicMock()
        handler.headers = {}

        result = await authenticate_request(handler)

        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_jwt_success(self):
        """authenticate_request returns user for valid JWT."""
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer valid-jwt"}

        mock_user = User(id="user-1", email="test@example.com")

        with patch('aragora.server.middleware.auth_v2.get_jwt_validator') as mock_get:
            mock_validator = MagicMock()
            mock_validator.validate_jwt.return_value = mock_user
            mock_get.return_value = mock_validator

            result = await authenticate_request(handler)

            assert result is mock_user
            mock_validator.validate_jwt.assert_called_once_with("valid-jwt")

    @pytest.mark.asyncio
    async def test_authenticate_api_key_fallback(self):
        """authenticate_request tries API key if JWT fails."""
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer ara_apikey123"}

        mock_user = User(id="user-1", email="test@example.com")

        with patch('aragora.server.middleware.auth_v2.get_jwt_validator') as mock_jwt:
            mock_jwt.return_value.validate_jwt.return_value = None

            with patch('aragora.server.middleware.auth_v2.get_api_key_validator') as mock_api:
                mock_api.return_value.validate_key = AsyncMock(return_value=mock_user)

                result = await authenticate_request(handler)

                assert result is mock_user


class TestGetCurrentUser:
    """Tests for get_current_user function."""

    def test_get_current_user_valid_jwt(self):
        """get_current_user returns user for valid JWT."""
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer valid-jwt"}

        mock_user = User(id="user-1", email="test@example.com")

        with patch('aragora.server.middleware.auth_v2.get_jwt_validator') as mock_get:
            mock_validator = MagicMock()
            mock_validator.validate_jwt.return_value = mock_user
            mock_get.return_value = mock_validator

            result = get_current_user(handler)

            assert result is mock_user

    def test_get_current_user_no_token(self):
        """get_current_user returns None without token."""
        handler = MagicMock()
        handler.headers = {}

        result = get_current_user(handler)

        assert result is None


class TestRequireUserDecorator:
    """Tests for require_user decorator."""

    def test_require_user_authenticated(self):
        """require_user passes for authenticated user."""
        @require_user
        def endpoint(handler, user):
            return {"user_id": user.id}

        mock_user = User(id="user-1", email="test@example.com")
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer token"}

        with patch('aragora.server.middleware.auth_v2.get_current_user', return_value=mock_user):
            result = endpoint(handler)

            assert result == {"user_id": "user-1"}

    def test_require_user_unauthenticated(self):
        """require_user returns 401 for unauthenticated request."""
        @require_user
        def endpoint(handler, user):
            return {"user_id": user.id}

        handler = MagicMock()
        handler.headers = {}

        with patch('aragora.server.middleware.auth_v2.get_current_user', return_value=None):
            result = endpoint(handler)

            assert result.status_code == 401
            assert b"Authentication required" in result.body

    def test_require_user_no_handler(self):
        """require_user returns 500 for missing handler."""
        @require_user
        def endpoint():
            return {"ok": True}

        result = endpoint()

        assert result.status_code == 500


class TestRequireAdminDecorator:
    """Tests for require_admin decorator."""

    def test_require_admin_admin_user(self):
        """require_admin passes for admin user."""
        @require_admin
        def endpoint(handler, user):
            return {"admin": user.is_admin}

        admin_user = User(id="admin-1", email="admin@example.com", role="admin")
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer token"}

        with patch('aragora.server.middleware.auth_v2.get_current_user', return_value=admin_user):
            result = endpoint(handler)

            assert result == {"admin": True}

    def test_require_admin_regular_user(self):
        """require_admin returns 403 for non-admin user."""
        @require_admin
        def endpoint(handler, user):
            return {"admin": user.is_admin}

        regular_user = User(id="user-1", email="user@example.com", role="user")
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer token"}

        with patch('aragora.server.middleware.auth_v2.get_current_user', return_value=regular_user):
            result = endpoint(handler)

            assert result.status_code == 403
            assert b"Admin access required" in result.body


class TestRequirePlanDecorator:
    """Tests for require_plan decorator."""

    def test_require_plan_sufficient_plan(self):
        """require_plan passes when user has sufficient plan."""
        @require_plan("pro")
        def endpoint(handler, user):
            return {"plan": user.plan}

        pro_user = User(id="user-1", email="pro@example.com", plan="pro")
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer token"}

        with patch('aragora.server.middleware.auth_v2.get_current_user', return_value=pro_user):
            result = endpoint(handler)

            assert result == {"plan": "pro"}

    def test_require_plan_insufficient_plan(self):
        """require_plan returns 403 when user has lower plan."""
        @require_plan("pro")
        def endpoint(handler, user):
            return {"plan": user.plan}

        free_user = User(id="user-1", email="free@example.com", plan="free")
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer token"}

        with patch('aragora.server.middleware.auth_v2.get_current_user', return_value=free_user):
            result = endpoint(handler)

            assert result.status_code == 403
            assert b"pro plan" in result.body

    def test_require_plan_hierarchy(self):
        """require_plan respects plan hierarchy."""
        @require_plan("team")
        def endpoint(handler, user):
            return {"plan": user.plan}

        # Enterprise is higher than team, should pass
        enterprise_user = User(id="user-1", email="ent@example.com", plan="enterprise")
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer token"}

        with patch('aragora.server.middleware.auth_v2.get_current_user', return_value=enterprise_user):
            result = endpoint(handler)

            assert result == {"plan": "enterprise"}


class TestGlobalValidators:
    """Tests for global validator singleton getters."""

    def test_get_jwt_validator_singleton(self):
        """get_jwt_validator returns same instance."""
        # Reset global
        import aragora.server.middleware.auth_v2 as auth_module
        auth_module._jwt_validator = None

        v1 = get_jwt_validator()
        v2 = get_jwt_validator()

        assert v1 is v2
        assert isinstance(v1, SupabaseAuthValidator)

    def test_get_api_key_validator_singleton(self):
        """get_api_key_validator returns same instance."""
        # Reset global
        import aragora.server.middleware.auth_v2 as auth_module
        auth_module._api_key_validator = None

        v1 = get_api_key_validator()
        v2 = get_api_key_validator()

        assert v1 is v2
        assert isinstance(v1, APIKeyValidator)


class TestJWTWithPyJWT:
    """Tests for JWT validation with PyJWT library."""

    @pytest.fixture
    def jwt_available(self):
        """Ensure jwt module is available for these tests."""
        try:
            import jwt
            return True
        except ImportError:
            pytest.skip("PyJWT not installed")

    def test_validate_jwt_with_pyjwt(self, jwt_available):
        """validate_jwt uses PyJWT when available with secret."""
        import jwt as pyjwt

        secret = "test-secret-key"
        validator = SupabaseAuthValidator(jwt_secret=secret)

        # Create a valid JWT
        payload = {
            "sub": "user-123",
            "email": "test@example.com",
            "role": "user",
            "exp": time.time() + 3600,
            "aud": "authenticated",
        }
        token = pyjwt.encode(payload, secret, algorithm="HS256")

        user = validator.validate_jwt(token)

        assert user is not None
        assert user.id == "user-123"
        assert user.email == "test@example.com"

    def test_validate_jwt_invalid_signature(self, jwt_available):
        """validate_jwt rejects tokens with invalid signature."""
        import jwt as pyjwt

        secret = "correct-secret"
        wrong_secret = "wrong-secret"
        validator = SupabaseAuthValidator(jwt_secret=secret)

        # Create JWT with wrong secret
        payload = {
            "sub": "user-123",
            "exp": time.time() + 3600,
            "aud": "authenticated",
        }
        token = pyjwt.encode(payload, wrong_secret, algorithm="HS256")

        user = validator.validate_jwt(token)

        assert user is None

    def test_validate_jwt_expired(self, jwt_available):
        """validate_jwt rejects expired tokens."""
        import jwt as pyjwt

        secret = "test-secret"
        validator = SupabaseAuthValidator(jwt_secret=secret)

        # Create expired JWT
        payload = {
            "sub": "user-123",
            "exp": time.time() - 3600,  # Expired 1 hour ago
            "aud": "authenticated",
        }
        token = pyjwt.encode(payload, secret, algorithm="HS256")

        user = validator.validate_jwt(token)

        assert user is None
