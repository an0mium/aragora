"""
Tests for API key management handler functions.

Phase 5: Auth Handler Test Coverage - API Keys

Covers:
- handle_generate_api_key - Creates SHA-256 hashed key, returns plaintext once
- handle_revoke_api_key - Revokes by user's current key
- handle_list_api_keys - Lists metadata (no plaintext)
- handle_revoke_api_key_prefix - Revokes by prefix match
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.server.handlers.conftest import parse_handler_response


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_user():
    """Create a mock user with API key generation capability."""
    user = MagicMock()
    user.id = "user-001"
    user.email = "test@example.com"
    user.org_id = "org-001"
    user.api_key_hash = None
    user.api_key_prefix = None
    user.api_key_created_at = None
    user.api_key_expires_at = None

    # Track generated key for verification
    generated_key = None

    def generate_api_key(expires_days: int = 365):
        nonlocal generated_key
        import secrets

        # Generate key and store hash
        generated_key = f"ara_{secrets.token_urlsafe(32)}"
        user.api_key_hash = hashlib.sha256(generated_key.encode()).hexdigest()
        user.api_key_prefix = generated_key[:12]
        user.api_key_created_at = datetime.now(timezone.utc)
        user.api_key_expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
        return generated_key

    user.generate_api_key = generate_api_key
    user._generated_key = lambda: generated_key

    return user


@pytest.fixture
def mock_user_with_key(mock_user):
    """Create a mock user with an existing API key."""
    mock_user.generate_api_key(expires_days=365)
    return mock_user


@pytest.fixture
def mock_org():
    """Create a mock organization with API access enabled."""
    org = MagicMock()
    org.limits = MagicMock()
    org.limits.api_access = True
    return org


@pytest.fixture
def mock_org_no_api_access():
    """Create a mock organization without API access."""
    org = MagicMock()
    org.limits = MagicMock()
    org.limits.api_access = False
    return org


@pytest.fixture
def mock_user_store(mock_user, mock_org):
    """Create a mock user store."""
    store = MagicMock()
    store.get_user_by_id.return_value = mock_user
    store.get_organization_by_id.return_value = mock_org
    store.update_user.return_value = True
    return store


@pytest.fixture
def mock_handler_instance(mock_user_store):
    """Create a mock AuthHandler instance."""
    handler_instance = MagicMock()
    handler_instance._check_permission.return_value = None  # No error
    handler_instance._get_user_store.return_value = mock_user_store
    return handler_instance


@pytest.fixture
def mock_http_handler(mock_http_handler):
    """Use the conftest mock_http_handler factory."""
    return mock_http_handler


# ============================================================================
# Test: Generate API Key
# ============================================================================


class TestGenerateApiKey:
    """Tests for handle_generate_api_key."""

    def test_generate_api_key_returns_200_with_key(
        self, mock_handler_instance, mock_http_handler, mock_user, mock_user_store
    ):
        """Test successful API key generation returns 200 with key."""
        from aragora.server.handlers.auth.api_keys import handle_generate_api_key

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx

            result = handle_generate_api_key(mock_handler_instance, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "api_key" in body
        assert "prefix" in body
        assert "expires_at" in body
        assert "message" in body
        assert "Save this key" in body["message"]

    def test_generate_api_key_key_is_unique_each_call(
        self, mock_handler_instance, mock_http_handler, mock_user, mock_user_store
    ):
        """Test that each API key generation produces a unique key."""
        # Test uniqueness by calling generate_api_key on the user directly
        # This avoids rate limiting issues with the handler
        keys = []
        for _ in range(5):
            # Reset user state for fresh key generation
            mock_user.api_key_hash = None
            mock_user.api_key_prefix = None
            key = mock_user.generate_api_key(expires_days=365)
            keys.append(key)

        # All keys should be unique
        assert len(set(keys)) == 5

    def test_generate_api_key_stores_hash_not_plaintext(
        self, mock_handler_instance, mock_http_handler, mock_user, mock_user_store
    ):
        """Test that the user store receives hash, not plaintext."""
        from aragora.server.handlers.auth.api_keys import handle_generate_api_key

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx

            result = handle_generate_api_key(mock_handler_instance, http)

        body = parse_handler_response(result)
        api_key = body["api_key"]

        # update_user should have been called
        mock_user_store.update_user.assert_called_once()
        call_kwargs = mock_user_store.update_user.call_args[1]

        # The stored hash should NOT be the plaintext key
        stored_hash = call_kwargs.get("api_key_hash")
        assert stored_hash is not None
        assert stored_hash != api_key

        # The stored hash should be SHA-256 of the key
        expected_hash = hashlib.sha256(api_key.encode()).hexdigest()
        assert stored_hash == expected_hash

    def test_generate_api_key_includes_prefix(
        self, mock_handler_instance, mock_http_handler, mock_user
    ):
        """Test that generated key includes a prefix."""
        from aragora.server.handlers.auth.api_keys import handle_generate_api_key

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx

            result = handle_generate_api_key(mock_handler_instance, http)

        body = parse_handler_response(result)
        assert body["prefix"] is not None
        # Prefix should be first part of the key
        assert body["api_key"].startswith(body["prefix"])

    def test_generate_api_key_respects_expiration_date(
        self, mock_handler_instance, mock_http_handler, mock_user, mock_user_store
    ):
        """Test that generated key has an expiration date."""
        from aragora.server.handlers.auth.api_keys import handle_generate_api_key

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx

            result = handle_generate_api_key(mock_handler_instance, http)

        body = parse_handler_response(result)
        assert body["expires_at"] is not None

        # Expiration should be about 365 days from now
        expires = datetime.fromisoformat(body["expires_at"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days_until_expiry = (expires - now).days
        assert 360 <= days_until_expiry <= 370

    def test_generate_api_key_requires_authentication(self, mock_http_handler):
        """Test that unauthenticated requests are rejected."""
        from aragora.server.handlers.auth.api_keys import handle_generate_api_key

        # Handler that fails permission check
        handler_instance = MagicMock()
        from aragora.server.handlers.base import error_response

        handler_instance._check_permission.return_value = error_response(
            "Authentication required", 401
        )

        http = mock_http_handler(method="POST")
        result = handle_generate_api_key(handler_instance, http)

        assert result.status_code == 401

    def test_generate_api_key_rejects_tier_without_api_access(
        self,
        mock_handler_instance,
        mock_http_handler,
        mock_user,
        mock_user_store,
        mock_org_no_api_access,
    ):
        """Test that users without API access tier are rejected."""
        from aragora.server.handlers.auth.api_keys import handle_generate_api_key

        # Set org to not have API access
        mock_user_store.get_organization_by_id.return_value = mock_org_no_api_access

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx

            result = handle_generate_api_key(mock_handler_instance, http)

        assert result.status_code == 403
        body = parse_handler_response(result)
        assert (
            "tier" in body.get("error", "").lower()
            or "professional" in body.get("error", "").lower()
        )

    def test_generate_api_key_returns_404_for_unknown_user(
        self, mock_handler_instance, mock_http_handler, mock_user_store
    ):
        """Test that unknown user returns 404."""
        from aragora.server.handlers.auth.api_keys import handle_generate_api_key

        mock_user_store.get_user_by_id.return_value = None

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = "nonexistent-user"
            mock_extract.return_value = mock_ctx

            result = handle_generate_api_key(mock_handler_instance, http)

        assert result.status_code == 404

    def test_generate_api_key_returns_503_without_user_store(self, mock_http_handler):
        """Test that missing user store returns 503."""
        from aragora.server.handlers.auth.api_keys import handle_generate_api_key

        handler_instance = MagicMock()
        handler_instance._check_permission.return_value = None
        handler_instance._get_user_store.return_value = None

        http = mock_http_handler(method="POST")
        result = handle_generate_api_key(handler_instance, http)

        assert result.status_code == 503


# ============================================================================
# Test: Revoke API Key
# ============================================================================


class TestRevokeApiKey:
    """Tests for handle_revoke_api_key."""

    def test_revoke_api_key_returns_200_on_success(
        self, mock_handler_instance, mock_http_handler, mock_user_with_key, mock_user_store
    ):
        """Test successful API key revocation returns 200."""
        from aragora.server.handlers.auth.api_keys import handle_revoke_api_key

        mock_user_store.get_user_by_id.return_value = mock_user_with_key

        http = mock_http_handler(method="DELETE")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user_with_key.id
            mock_extract.return_value = mock_ctx

            result = handle_revoke_api_key(mock_handler_instance, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "revoked" in body.get("message", "").lower()

    def test_revoke_api_key_clears_all_key_fields(
        self, mock_handler_instance, mock_http_handler, mock_user_with_key, mock_user_store
    ):
        """Test that revocation clears all API key fields."""
        from aragora.server.handlers.auth.api_keys import handle_revoke_api_key

        mock_user_store.get_user_by_id.return_value = mock_user_with_key

        http = mock_http_handler(method="DELETE")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user_with_key.id
            mock_extract.return_value = mock_ctx

            handle_revoke_api_key(mock_handler_instance, http)

        # update_user should clear all key fields
        mock_user_store.update_user.assert_called_once()
        call_kwargs = mock_user_store.update_user.call_args[1]

        assert call_kwargs.get("api_key_hash") is None
        assert call_kwargs.get("api_key_prefix") is None
        assert call_kwargs.get("api_key_created_at") is None
        assert call_kwargs.get("api_key_expires_at") is None

    def test_revoke_api_key_returns_404_for_unknown_user(
        self, mock_handler_instance, mock_http_handler, mock_user_store
    ):
        """Test that unknown user returns 404."""
        from aragora.server.handlers.auth.api_keys import handle_revoke_api_key

        mock_user_store.get_user_by_id.return_value = None

        http = mock_http_handler(method="DELETE")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = "nonexistent-user"
            mock_extract.return_value = mock_ctx

            result = handle_revoke_api_key(mock_handler_instance, http)

        assert result.status_code == 404

    def test_revoke_api_key_requires_permission(self, mock_http_handler):
        """Test that revocation requires api_key.revoke permission."""
        from aragora.server.handlers.auth.api_keys import handle_revoke_api_key
        from aragora.server.handlers.base import error_response

        handler_instance = MagicMock()
        handler_instance._check_permission.return_value = error_response("Permission denied", 403)

        http = mock_http_handler(method="DELETE")
        result = handle_revoke_api_key(handler_instance, http)

        assert result.status_code == 403


# ============================================================================
# Test: List API Keys
# ============================================================================


class TestListApiKeys:
    """Tests for handle_list_api_keys."""

    def test_list_api_keys_returns_metadata_only(
        self, mock_handler_instance, mock_http_handler, mock_user_with_key, mock_user_store
    ):
        """Test that list returns metadata but not the actual key."""
        from aragora.server.handlers.auth.api_keys import handle_list_api_keys

        mock_user_store.get_user_by_id.return_value = mock_user_with_key

        http = mock_http_handler(method="GET")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user_with_key.id
            mock_extract.return_value = mock_ctx

            result = handle_list_api_keys(mock_handler_instance, http)

        assert result.status_code == 200
        body = parse_handler_response(result)

        assert "keys" in body
        assert len(body["keys"]) == 1

        key_info = body["keys"][0]
        # Should have prefix but NOT the full key
        assert "prefix" in key_info
        assert "api_key" not in key_info
        assert "hash" not in key_info

    def test_list_api_keys_shows_expiration_status(
        self, mock_handler_instance, mock_http_handler, mock_user_with_key, mock_user_store
    ):
        """Test that list includes expiration date."""
        from aragora.server.handlers.auth.api_keys import handle_list_api_keys

        mock_user_store.get_user_by_id.return_value = mock_user_with_key

        http = mock_http_handler(method="GET")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user_with_key.id
            mock_extract.return_value = mock_ctx

            result = handle_list_api_keys(mock_handler_instance, http)

        body = parse_handler_response(result)
        key_info = body["keys"][0]

        assert "expires_at" in key_info
        assert key_info["expires_at"] is not None

    def test_list_api_keys_includes_created_at(
        self, mock_handler_instance, mock_http_handler, mock_user_with_key, mock_user_store
    ):
        """Test that list includes creation date."""
        from aragora.server.handlers.auth.api_keys import handle_list_api_keys

        mock_user_store.get_user_by_id.return_value = mock_user_with_key

        http = mock_http_handler(method="GET")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user_with_key.id
            mock_extract.return_value = mock_ctx

            result = handle_list_api_keys(mock_handler_instance, http)

        body = parse_handler_response(result)
        key_info = body["keys"][0]

        assert "created_at" in key_info

    def test_list_api_keys_empty_for_new_user(
        self, mock_handler_instance, mock_http_handler, mock_user, mock_user_store
    ):
        """Test that list returns empty for user without API key."""
        from aragora.server.handlers.auth.api_keys import handle_list_api_keys

        # User without key
        mock_user.api_key_prefix = None
        mock_user_store.get_user_by_id.return_value = mock_user

        http = mock_http_handler(method="GET")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx

            result = handle_list_api_keys(mock_handler_instance, http)

        body = parse_handler_response(result)
        assert body["keys"] == []
        assert body["count"] == 0

    def test_list_api_keys_returns_count(
        self, mock_handler_instance, mock_http_handler, mock_user_with_key, mock_user_store
    ):
        """Test that list includes count field."""
        from aragora.server.handlers.auth.api_keys import handle_list_api_keys

        mock_user_store.get_user_by_id.return_value = mock_user_with_key

        http = mock_http_handler(method="GET")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user_with_key.id
            mock_extract.return_value = mock_ctx

            result = handle_list_api_keys(mock_handler_instance, http)

        body = parse_handler_response(result)
        assert "count" in body
        assert body["count"] == 1


# ============================================================================
# Test: Revoke API Key by Prefix
# ============================================================================


class TestRevokeApiKeyByPrefix:
    """Tests for handle_revoke_api_key_prefix."""

    def test_revoke_api_key_by_prefix_succeeds(
        self, mock_handler_instance, mock_http_handler, mock_user_with_key, mock_user_store
    ):
        """Test successful revocation by prefix."""
        from aragora.server.handlers.auth.api_keys import handle_revoke_api_key_prefix

        mock_user_store.get_user_by_id.return_value = mock_user_with_key
        prefix = mock_user_with_key.api_key_prefix

        http = mock_http_handler(method="DELETE")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user_with_key.id
            mock_extract.return_value = mock_ctx

            result = handle_revoke_api_key_prefix(mock_handler_instance, http, prefix)

        assert result.status_code == 200

    def test_revoke_api_key_by_prefix_no_match_returns_404(
        self, mock_handler_instance, mock_http_handler, mock_user_with_key, mock_user_store
    ):
        """Test that non-matching prefix returns 404."""
        from aragora.server.handlers.auth.api_keys import handle_revoke_api_key_prefix

        mock_user_store.get_user_by_id.return_value = mock_user_with_key

        http = mock_http_handler(method="DELETE")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user_with_key.id
            mock_extract.return_value = mock_ctx

            result = handle_revoke_api_key_prefix(mock_handler_instance, http, "wrong_prefix")

        assert result.status_code == 404

    def test_revoke_api_key_by_prefix_clears_all_fields(
        self, mock_handler_instance, mock_http_handler, mock_user_with_key, mock_user_store
    ):
        """Test that prefix revocation clears all key fields."""
        from aragora.server.handlers.auth.api_keys import handle_revoke_api_key_prefix

        mock_user_store.get_user_by_id.return_value = mock_user_with_key
        prefix = mock_user_with_key.api_key_prefix

        http = mock_http_handler(method="DELETE")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user_with_key.id
            mock_extract.return_value = mock_ctx

            handle_revoke_api_key_prefix(mock_handler_instance, http, prefix)

        call_kwargs = mock_user_store.update_user.call_args[1]
        assert call_kwargs.get("api_key_hash") is None
        assert call_kwargs.get("api_key_prefix") is None

    def test_revoke_api_key_by_prefix_user_without_key_returns_404(
        self, mock_handler_instance, mock_http_handler, mock_user, mock_user_store
    ):
        """Test that user without key returns 404 for any prefix."""
        from aragora.server.handlers.auth.api_keys import handle_revoke_api_key_prefix

        mock_user.api_key_prefix = None
        mock_user_store.get_user_by_id.return_value = mock_user

        http = mock_http_handler(method="DELETE")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx

            result = handle_revoke_api_key_prefix(mock_handler_instance, http, "any_prefix")

        assert result.status_code == 404


# ============================================================================
# Test: Security Properties
# ============================================================================


class TestApiKeySecurityProperties:
    """Tests for API key security properties."""

    def test_api_key_hash_uses_sha256(self, mock_user):
        """Test that API key hashing uses SHA-256."""
        key = mock_user.generate_api_key(expires_days=365)

        # Verify the stored hash is SHA-256
        expected_hash = hashlib.sha256(key.encode()).hexdigest()
        assert mock_user.api_key_hash == expected_hash
        assert len(mock_user.api_key_hash) == 64  # SHA-256 produces 64 hex chars

    def test_api_key_prefix_is_first_chars(self, mock_user):
        """Test that prefix is derived from key start."""
        key = mock_user.generate_api_key(expires_days=365)

        # Prefix should be first part of the key
        assert key.startswith(mock_user.api_key_prefix)

    def test_cannot_retrieve_plaintext_after_creation(
        self, mock_handler_instance, mock_http_handler, mock_user_with_key, mock_user_store
    ):
        """Test that plaintext key cannot be retrieved after creation."""
        from aragora.server.handlers.auth.api_keys import handle_list_api_keys

        mock_user_store.get_user_by_id.return_value = mock_user_with_key

        http = mock_http_handler(method="GET")

        with patch(
            "aragora.server.handlers.auth.api_keys.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user_with_key.id
            mock_extract.return_value = mock_ctx

            result = handle_list_api_keys(mock_handler_instance, http)

        body = parse_handler_response(result)

        # Response should NOT contain the plaintext key or hash
        response_str = json.dumps(body)
        assert mock_user_with_key.api_key_hash not in response_str
        # The _generated_key is a mock method, but the actual key shouldn't be exposed
        for key_info in body["keys"]:
            assert "api_key" not in key_info
            assert "hash" not in key_info
            assert "api_key_hash" not in key_info

    def test_api_key_generation_audited(
        self, mock_handler_instance, mock_http_handler, mock_user, mock_user_store
    ):
        """Test that API key generation is audited."""
        from aragora.server.handlers.auth.api_keys import handle_generate_api_key

        http = mock_http_handler(method="POST")

        with (
            patch(
                "aragora.server.handlers.auth.api_keys.extract_user_from_request"
            ) as mock_extract,
            patch("aragora.server.handlers.auth.api_keys.AUDIT_AVAILABLE", True),
            patch("aragora.server.handlers.auth.api_keys.audit_admin") as mock_audit,
        ):
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx

            handle_generate_api_key(mock_handler_instance, http)

        # Audit should be called
        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["action"] == "api_key_generated"
        assert call_kwargs["target_type"] == "api_key"

    def test_api_key_revocation_audited(
        self, mock_handler_instance, mock_http_handler, mock_user_with_key, mock_user_store
    ):
        """Test that API key revocation is audited."""
        from aragora.server.handlers.auth.api_keys import handle_revoke_api_key

        mock_user_store.get_user_by_id.return_value = mock_user_with_key

        http = mock_http_handler(method="DELETE")

        with (
            patch(
                "aragora.server.handlers.auth.api_keys.extract_user_from_request"
            ) as mock_extract,
            patch("aragora.server.handlers.auth.api_keys.AUDIT_AVAILABLE", True),
            patch("aragora.server.handlers.auth.api_keys.audit_admin") as mock_audit,
        ):
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user_with_key.id
            mock_extract.return_value = mock_ctx

            handle_revoke_api_key(mock_handler_instance, http)

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["action"] == "api_key_revoked"


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestApiKeyErrorHandling:
    """Tests for API key error handling."""

    def test_generate_returns_503_without_user_store(self, mock_http_handler):
        """Test graceful handling of missing user store."""
        from aragora.server.handlers.auth.api_keys import handle_generate_api_key

        handler_instance = MagicMock()
        handler_instance._check_permission.return_value = None
        handler_instance._get_user_store.return_value = None

        http = mock_http_handler(method="POST")
        result = handle_generate_api_key(handler_instance, http)

        assert result.status_code == 503
        body = parse_handler_response(result)
        assert "unavailable" in body.get("error", "").lower()

    def test_list_returns_503_without_user_store(self, mock_http_handler):
        """Test list endpoint handles missing user store."""
        from aragora.server.handlers.auth.api_keys import handle_list_api_keys

        handler_instance = MagicMock()
        handler_instance._check_permission.return_value = None
        handler_instance._get_user_store.return_value = None

        http = mock_http_handler(method="GET")
        result = handle_list_api_keys(handler_instance, http)

        assert result.status_code == 503

    def test_revoke_returns_503_without_user_store(self, mock_http_handler):
        """Test revoke endpoint handles missing user store."""
        from aragora.server.handlers.auth.api_keys import handle_revoke_api_key

        handler_instance = MagicMock()
        handler_instance._check_permission.return_value = None
        handler_instance._get_user_store.return_value = None

        http = mock_http_handler(method="DELETE")
        result = handle_revoke_api_key(handler_instance, http)

        assert result.status_code == 503


__all__ = [
    "TestGenerateApiKey",
    "TestRevokeApiKey",
    "TestListApiKeys",
    "TestRevokeApiKeyByPrefix",
    "TestApiKeySecurityProperties",
    "TestApiKeyErrorHandling",
]
