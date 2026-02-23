"""Tests for API key handler functions (aragora/server/handlers/auth/api_keys.py).

Covers all four API key endpoints:
- POST /api/auth/api-key      -> handle_generate_api_key
- DELETE /api/auth/api-key     -> handle_revoke_api_key
- GET /api/auth/api-keys       -> handle_list_api_keys
- DELETE /api/auth/api-keys/:p -> handle_revoke_api_key_prefix

Tests exercise: success paths, permission checks, user-not-found, service unavailable,
tier restrictions, audit logging, edge cases (no prefix, wrong prefix, multiple keys),
routing through the AuthHandler.handle() dispatcher, SDK aliases, unsupported methods,
and security tests (path traversal, injection).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.auth.api_keys import (
    handle_generate_api_key,
    handle_revoke_api_key,
    handle_list_api_keys,
    handle_revoke_api_key_prefix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Lightweight mock HTTP request handler."""

    def __init__(self, body: dict | None = None, method: str = "POST"):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {
            "User-Agent": "test-agent",
            "Authorization": "Bearer test-token-abc",
        }
        self.rfile = MagicMock()
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


@dataclass
class MockUser:
    """Mock user for user-store interactions."""

    id: str = "user-001"
    email: str = "test@example.com"
    name: str = "Test User"
    org_id: str | None = "org-001"
    role: str = "admin"
    is_active: bool = True
    api_key_hash: str | None = None
    api_key_prefix: str | None = None
    api_key_created_at: datetime | None = None
    api_key_expires_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "org_id": self.org_id,
            "role": self.role,
        }

    def generate_api_key(self, expires_days: int = 365) -> str:
        """Simulate generating an API key."""
        self.api_key_hash = "sha256_hash_value"
        self.api_key_prefix = "ara_test1234"
        self.api_key_created_at = datetime.now(timezone.utc)
        from datetime import timedelta

        self.api_key_expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
        return "ara_full_api_key_plaintext_value"


@dataclass
class MockAuthCtx:
    """Mock auth context from extract_user_from_request."""

    is_authenticated: bool = True
    user_id: str = "user-001"
    email: str = "test@example.com"
    org_id: str = "org-001"
    role: str = "admin"
    client_ip: str = "127.0.0.1"


@dataclass
class MockOrg:
    """Mock organization."""

    id: str = "org-001"
    name: str = "Test Org"

    @dataclass
    class Limits:
        api_access: bool = True

    limits: Limits = field(default_factory=Limits)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name}


def _make_user_store(user: MockUser | None = None):
    """Create a mock user store with standard methods."""
    store = MagicMock()
    u = user or MockUser()
    store.get_user_by_id.return_value = u
    store.update_user.return_value = None
    store.get_organization_by_id.return_value = None
    return store


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_api_key_deps(monkeypatch):
    """Patch dependencies common to all API key handler functions."""
    mock_auth_ctx = MockAuthCtx()

    # Patch extract_user_from_request used by the proxy in api_keys.py
    monkeypatch.setattr(
        "aragora.server.handlers.auth.api_keys.extract_user_from_request",
        lambda handler, user_store: mock_auth_ctx,
    )

    # Patch emit_handler_event to no-op
    monkeypatch.setattr(
        "aragora.server.handlers.auth.api_keys.emit_handler_event",
        lambda *args, **kwargs: None,
    )

    # Patch audit_admin to no-op and mark available
    monkeypatch.setattr("aragora.server.handlers.auth.api_keys.AUDIT_AVAILABLE", True)
    monkeypatch.setattr(
        "aragora.server.handlers.auth.api_keys.audit_admin",
        lambda **kwargs: None,
    )


@pytest.fixture
def handler_instance():
    """Create an AuthHandler-like object with mocked methods."""
    from aragora.server.handlers.auth.handler import AuthHandler

    store = _make_user_store()
    h = AuthHandler(server_context={"user_store": store})
    # Always grant permissions
    h._check_permission = MagicMock(return_value=None)
    return h, store


@pytest.fixture
def http():
    """Factory for creating mock HTTP handlers."""

    def _create(body: dict | None = None, method: str = "POST") -> MockHTTPHandler:
        return MockHTTPHandler(body=body, method=method)

    return _create


# =========================================================================
# handle_generate_api_key
# =========================================================================


class TestGenerateAPIKey:
    """POST /api/auth/api-key."""

    def test_success(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = handle_generate_api_key(hi, http())
        assert _status(result) == 200
        body = _body(result)
        assert "api_key" in body
        assert "prefix" in body
        assert "expires_at" in body
        assert "message" in body
        assert "Save this key" in body["message"]

    def test_returns_plaintext_key(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = handle_generate_api_key(hi, http())
        body = _body(result)
        assert body["api_key"].startswith("ara_")

    def test_returns_prefix(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = handle_generate_api_key(hi, http())
        body = _body(result)
        assert body["prefix"] == "ara_test1234"

    def test_returns_expiry_iso(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = handle_generate_api_key(hi, http())
        body = _body(result)
        # Expiry should be an ISO format date string
        assert body["expires_at"] is not None
        datetime.fromisoformat(body["expires_at"])

    def test_persists_hashed_key(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        handle_generate_api_key(hi, http())
        store.update_user.assert_called_once()
        call_kwargs = store.update_user.call_args
        assert call_kwargs[0][0] == user.id
        assert "api_key_hash" in call_kwargs[1]
        assert "api_key_prefix" in call_kwargs[1]
        assert "api_key_created_at" in call_kwargs[1]
        assert "api_key_expires_at" in call_kwargs[1]

    def test_permission_denied(self, handler_instance, http):
        from aragora.server.handlers.base import error_response

        hi, store = handler_instance
        hi._check_permission = MagicMock(
            return_value=error_response("Permission denied", 403)
        )
        result = handle_generate_api_key(hi, http())
        assert _status(result) == 403

    def test_user_not_found(self, handler_instance, http):
        hi, store = handler_instance
        store.get_user_by_id.return_value = None
        result = handle_generate_api_key(hi, http())
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_user_store_unavailable(self, handler_instance, http):
        hi, store = handler_instance
        hi.ctx = {}  # No user_store in context
        result = handle_generate_api_key(hi, http())
        assert _status(result) == 503
        assert "unavailable" in _body(result).get("error", "").lower()

    def test_tier_restricts_api_access(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(org_id="org-001")
        store.get_user_by_id.return_value = user
        org = MockOrg(limits=MockOrg.Limits(api_access=False))
        store.get_organization_by_id.return_value = org
        result = handle_generate_api_key(hi, http())
        assert _status(result) == 403
        assert "Professional" in _body(result).get("error", "")

    def test_tier_allows_api_access(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(org_id="org-001")
        store.get_user_by_id.return_value = user
        org = MockOrg(limits=MockOrg.Limits(api_access=True))
        store.get_organization_by_id.return_value = org
        result = handle_generate_api_key(hi, http())
        assert _status(result) == 200

    def test_no_org_allows_api_access(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(org_id=None)
        store.get_user_by_id.return_value = user
        result = handle_generate_api_key(hi, http())
        assert _status(result) == 200

    def test_org_exists_but_no_limits_restriction(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(org_id="org-001")
        store.get_user_by_id.return_value = user
        # org is None (not found), so no limits check
        store.get_organization_by_id.return_value = None
        result = handle_generate_api_key(hi, http())
        assert _status(result) == 200

    def test_audit_logged(self, handler_instance, http, monkeypatch):
        audit_calls = []
        monkeypatch.setattr(
            "aragora.server.handlers.auth.api_keys.audit_admin",
            lambda **kwargs: audit_calls.append(kwargs),
        )
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        handle_generate_api_key(hi, http())
        assert len(audit_calls) == 1
        assert audit_calls[0]["action"] == "api_key_generated"
        assert audit_calls[0]["target_type"] == "api_key"

    def test_audit_not_logged_when_unavailable(self, handler_instance, http, monkeypatch):
        monkeypatch.setattr("aragora.server.handlers.auth.api_keys.AUDIT_AVAILABLE", False)
        audit_calls = []
        monkeypatch.setattr(
            "aragora.server.handlers.auth.api_keys.audit_admin",
            lambda **kwargs: audit_calls.append(kwargs),
        )
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        handle_generate_api_key(hi, http())
        assert len(audit_calls) == 0

    def test_event_emitted(self, handler_instance, http, monkeypatch):
        events = []
        monkeypatch.setattr(
            "aragora.server.handlers.auth.api_keys.emit_handler_event",
            lambda *args, **kwargs: events.append((args, kwargs)),
        )
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        handle_generate_api_key(hi, http())
        assert len(events) == 1
        assert events[0][0][0] == "auth"  # resource_type

    def test_user_with_expires_none(self, handler_instance, http):
        """Edge case: generate_api_key sets expires_at to None."""
        hi, store = handler_instance
        user = MockUser()

        original_gen = user.generate_api_key

        def patched_gen(expires_days=365):
            key = original_gen(expires_days)
            user.api_key_expires_at = None
            return key

        user.generate_api_key = patched_gen
        store.get_user_by_id.return_value = user
        result = handle_generate_api_key(hi, http())
        assert _status(result) == 200
        body = _body(result)
        assert body["expires_at"] is None


# =========================================================================
# handle_revoke_api_key
# =========================================================================


class TestRevokeAPIKey:
    """DELETE /api/auth/api-key."""

    def test_success(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key(hi, http(method="DELETE"))
        assert _status(result) == 200
        body = _body(result)
        assert "revoked" in body["message"].lower()

    def test_clears_all_key_fields(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(
            api_key_hash="some_hash",
            api_key_prefix="ara_pre12345",
        )
        store.get_user_by_id.return_value = user
        handle_revoke_api_key(hi, http(method="DELETE"))
        store.update_user.assert_called_once_with(
            user.id,
            api_key_hash=None,
            api_key_prefix=None,
            api_key_created_at=None,
            api_key_expires_at=None,
        )

    def test_permission_denied(self, handler_instance, http):
        from aragora.server.handlers.base import error_response

        hi, store = handler_instance
        hi._check_permission = MagicMock(
            return_value=error_response("Permission denied", 403)
        )
        result = handle_revoke_api_key(hi, http(method="DELETE"))
        assert _status(result) == 403

    def test_user_not_found(self, handler_instance, http):
        hi, store = handler_instance
        store.get_user_by_id.return_value = None
        result = handle_revoke_api_key(hi, http(method="DELETE"))
        assert _status(result) == 404

    def test_user_store_unavailable(self, handler_instance, http):
        hi, store = handler_instance
        hi.ctx = {}
        result = handle_revoke_api_key(hi, http(method="DELETE"))
        assert _status(result) == 503

    def test_audit_logged(self, handler_instance, http, monkeypatch):
        audit_calls = []
        monkeypatch.setattr(
            "aragora.server.handlers.auth.api_keys.audit_admin",
            lambda **kwargs: audit_calls.append(kwargs),
        )
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        handle_revoke_api_key(hi, http(method="DELETE"))
        assert len(audit_calls) == 1
        assert audit_calls[0]["action"] == "api_key_revoked"
        assert audit_calls[0]["target_type"] == "api_key"
        assert audit_calls[0]["target_id"] == user.id

    def test_audit_not_logged_when_unavailable(self, handler_instance, http, monkeypatch):
        monkeypatch.setattr("aragora.server.handlers.auth.api_keys.AUDIT_AVAILABLE", False)
        audit_calls = []
        monkeypatch.setattr(
            "aragora.server.handlers.auth.api_keys.audit_admin",
            lambda **kwargs: audit_calls.append(kwargs),
        )
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        handle_revoke_api_key(hi, http(method="DELETE"))
        assert len(audit_calls) == 0

    def test_revoke_when_no_key_exists(self, handler_instance, http):
        """Revoking when user has no key should still succeed (idempotent clear)."""
        hi, store = handler_instance
        user = MockUser(api_key_hash=None, api_key_prefix=None)
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key(hi, http(method="DELETE"))
        assert _status(result) == 200
        body = _body(result)
        assert "revoked" in body["message"].lower()


# =========================================================================
# handle_list_api_keys
# =========================================================================


class TestListAPIKeys:
    """GET /api/auth/api-keys."""

    def test_success_with_key(self, handler_instance, http):
        hi, store = handler_instance
        now = datetime.now(timezone.utc)
        user = MockUser(
            api_key_prefix="ara_test1234",
            api_key_created_at=now,
            api_key_expires_at=now,
        )
        store.get_user_by_id.return_value = user
        result = handle_list_api_keys(hi, http(method="GET"))
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1
        assert len(body["keys"]) == 1
        assert body["keys"][0]["prefix"] == "ara_test1234"

    def test_success_no_key(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(api_key_prefix=None)
        store.get_user_by_id.return_value = user
        result = handle_list_api_keys(hi, http(method="GET"))
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 0
        assert body["keys"] == []

    def test_key_includes_created_at(self, handler_instance, http):
        hi, store = handler_instance
        now = datetime.now(timezone.utc)
        user = MockUser(
            api_key_prefix="ara_test1234",
            api_key_created_at=now,
            api_key_expires_at=now,
        )
        store.get_user_by_id.return_value = user
        result = handle_list_api_keys(hi, http(method="GET"))
        body = _body(result)
        assert body["keys"][0]["created_at"] == now.isoformat()

    def test_key_includes_expires_at(self, handler_instance, http):
        hi, store = handler_instance
        now = datetime.now(timezone.utc)
        user = MockUser(
            api_key_prefix="ara_test1234",
            api_key_created_at=now,
            api_key_expires_at=now,
        )
        store.get_user_by_id.return_value = user
        result = handle_list_api_keys(hi, http(method="GET"))
        body = _body(result)
        assert body["keys"][0]["expires_at"] == now.isoformat()

    def test_key_with_null_timestamps(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(
            api_key_prefix="ara_test1234",
            api_key_created_at=None,
            api_key_expires_at=None,
        )
        store.get_user_by_id.return_value = user
        result = handle_list_api_keys(hi, http(method="GET"))
        body = _body(result)
        assert body["keys"][0]["created_at"] is None
        assert body["keys"][0]["expires_at"] is None

    def test_permission_denied(self, handler_instance, http):
        from aragora.server.handlers.base import error_response

        hi, store = handler_instance
        hi._check_permission = MagicMock(
            return_value=error_response("Permission denied", 403)
        )
        result = handle_list_api_keys(hi, http(method="GET"))
        assert _status(result) == 403

    def test_user_not_found(self, handler_instance, http):
        hi, store = handler_instance
        store.get_user_by_id.return_value = None
        result = handle_list_api_keys(hi, http(method="GET"))
        assert _status(result) == 404

    def test_user_store_unavailable(self, handler_instance, http):
        hi, store = handler_instance
        hi.ctx = {}
        result = handle_list_api_keys(hi, http(method="GET"))
        assert _status(result) == 503

    def test_does_not_return_full_key(self, handler_instance, http):
        """List should only return prefix, never the full key."""
        hi, store = handler_instance
        user = MockUser(
            api_key_prefix="ara_test1234",
            api_key_hash="sha256_hash_value_of_full_key",
            api_key_created_at=datetime.now(timezone.utc),
            api_key_expires_at=datetime.now(timezone.utc),
        )
        store.get_user_by_id.return_value = user
        result = handle_list_api_keys(hi, http(method="GET"))
        body = _body(result)
        # Should not contain the hash or full key
        body_str = json.dumps(body)
        assert "sha256_hash_value" not in body_str
        assert "api_key_hash" not in body_str


# =========================================================================
# handle_revoke_api_key_prefix
# =========================================================================


class TestRevokeAPIKeyPrefix:
    """DELETE /api/auth/api-keys/:prefix."""

    def test_success(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_test1234")
        assert _status(result) == 200
        body = _body(result)
        assert "revoked" in body["message"].lower()

    def test_clears_all_key_fields(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234", api_key_hash="some_hash")
        store.get_user_by_id.return_value = user
        handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_test1234")
        store.update_user.assert_called_once_with(
            user.id,
            api_key_hash=None,
            api_key_prefix=None,
            api_key_created_at=None,
            api_key_expires_at=None,
        )

    def test_prefix_not_found(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_wrong9999")
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_no_key_set(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(api_key_prefix=None)
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_any_prefix")
        assert _status(result) == 404

    def test_permission_denied(self, handler_instance, http):
        from aragora.server.handlers.base import error_response

        hi, store = handler_instance
        hi._check_permission = MagicMock(
            return_value=error_response("Permission denied", 403)
        )
        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_test1234")
        assert _status(result) == 403

    def test_user_not_found(self, handler_instance, http):
        hi, store = handler_instance
        store.get_user_by_id.return_value = None
        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_test1234")
        assert _status(result) == 404

    def test_user_store_unavailable(self, handler_instance, http):
        hi, store = handler_instance
        hi.ctx = {}
        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_test1234")
        assert _status(result) == 503

    def test_audit_logged(self, handler_instance, http, monkeypatch):
        audit_calls = []
        monkeypatch.setattr(
            "aragora.server.handlers.auth.api_keys.audit_admin",
            lambda **kwargs: audit_calls.append(kwargs),
        )
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_test1234")
        assert len(audit_calls) == 1
        assert audit_calls[0]["action"] == "api_key_revoked"
        assert audit_calls[0]["target_id"] == "ara_test1234"

    def test_audit_not_logged_when_unavailable(self, handler_instance, http, monkeypatch):
        monkeypatch.setattr("aragora.server.handlers.auth.api_keys.AUDIT_AVAILABLE", False)
        audit_calls = []
        monkeypatch.setattr(
            "aragora.server.handlers.auth.api_keys.audit_admin",
            lambda **kwargs: audit_calls.append(kwargs),
        )
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_test1234")
        assert len(audit_calls) == 0

    def test_empty_prefix(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "")
        assert _status(result) == 404


# =========================================================================
# AuthHandler.handle() routing tests
# =========================================================================


class TestHandleRouting:
    """Test routing through AuthHandler.handle() for API key endpoints."""

    @pytest.fixture(autouse=True)
    def _patch_routing_deps(self, monkeypatch):
        """Patch dependencies for routing tests."""
        mock_auth_ctx = MockAuthCtx()
        monkeypatch.setattr(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            lambda handler, user_store: mock_auth_ctx,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.auth.handler.validate_refresh_token",
            lambda token: None,
        )

    @pytest.mark.asyncio
    async def test_post_api_key_routes_to_generate(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = await hi.handle("/api/auth/api-key", {}, http(), method="POST")
        assert _status(result) == 200
        body = _body(result)
        assert "api_key" in body

    @pytest.mark.asyncio
    async def test_delete_api_key_routes_to_revoke(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = await hi.handle("/api/auth/api-key", {}, http(method="DELETE"), method="DELETE")
        assert _status(result) == 200
        body = _body(result)
        assert "revoked" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_get_api_keys_routes_to_list(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = await hi.handle("/api/auth/api-keys", {}, http(method="GET"), method="GET")
        assert _status(result) == 200
        body = _body(result)
        assert "keys" in body

    @pytest.mark.asyncio
    async def test_post_api_keys_routes_to_generate(self, handler_instance, http):
        """POST to /api/auth/api-keys also generates a key."""
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = await hi.handle("/api/auth/api-keys", {}, http(), method="POST")
        assert _status(result) == 200
        body = _body(result)
        assert "api_key" in body

    @pytest.mark.asyncio
    async def test_delete_api_keys_routes_to_revoke(self, handler_instance, http):
        """DELETE to /api/auth/api-keys revokes (same as /api/auth/api-key DELETE)."""
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = await hi.handle("/api/auth/api-keys", {}, http(method="DELETE"), method="DELETE")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_delete_api_keys_prefix_routes_to_prefix_revoke(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = await hi.handle(
            "/api/auth/api-keys/ara_test1234", {}, http(method="DELETE"), method="DELETE"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_path_normalization(self, handler_instance, http):
        """Versioned paths (/api/v1/auth/...) should be normalized."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = await hi.handle(
            "/api/v1/auth/api-keys", {}, http(method="GET"), method="GET"
        )
        assert _status(result) == 200
        body = _body(result)
        assert "keys" in body

    @pytest.mark.asyncio
    async def test_sdk_alias_keys_get(self, handler_instance, http):
        """/api/keys should alias to /api/auth/api-keys GET."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_xyz12345")
        store.get_user_by_id.return_value = user
        result = await hi.handle("/api/keys", {}, http(method="GET"), method="GET")
        assert _status(result) == 200
        body = _body(result)
        assert "keys" in body

    @pytest.mark.asyncio
    async def test_sdk_alias_keys_post(self, handler_instance, http):
        """/api/keys POST should alias to /api/auth/api-keys POST (generate)."""
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = await hi.handle("/api/keys", {}, http(), method="POST")
        assert _status(result) == 200
        body = _body(result)
        assert "api_key" in body

    @pytest.mark.asyncio
    async def test_sdk_alias_keys_prefix_delete(self, handler_instance, http):
        """/api/keys/:prefix DELETE should alias to /api/auth/api-keys/:prefix DELETE."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = await hi.handle(
            "/api/keys/ara_test1234", {}, http(method="DELETE"), method="DELETE"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unsupported_method_api_key(self, handler_instance, http):
        """GET on /api/auth/api-key should return 405."""
        hi, store = handler_instance
        result = await hi.handle("/api/auth/api-key", {}, http(method="GET"), method="GET")
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_unsupported_method_put(self, handler_instance, http):
        """PUT on /api/auth/api-key should return 405."""
        hi, store = handler_instance
        result = await hi.handle("/api/auth/api-key", {}, http(method="PUT"), method="PUT")
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_unsupported_method_patch(self, handler_instance, http):
        """PATCH on /api/auth/api-keys should return 405."""
        hi, store = handler_instance
        result = await hi.handle("/api/auth/api-keys", {}, http(method="PATCH"), method="PATCH")
        assert _status(result) == 405


# =========================================================================
# can_handle() tests
# =========================================================================


class TestCanHandle:
    """Test AuthHandler.can_handle() for API key paths."""

    def test_api_key_path(self):
        from aragora.server.handlers.auth.handler import AuthHandler

        h = AuthHandler()
        assert h.can_handle("/api/auth/api-key") is True

    def test_api_keys_path(self):
        from aragora.server.handlers.auth.handler import AuthHandler

        h = AuthHandler()
        assert h.can_handle("/api/auth/api-keys") is True

    def test_api_keys_prefix_path(self):
        from aragora.server.handlers.auth.handler import AuthHandler

        h = AuthHandler()
        assert h.can_handle("/api/auth/api-keys/ara_test1234") is True

    def test_versioned_api_key_path(self):
        from aragora.server.handlers.auth.handler import AuthHandler

        h = AuthHandler()
        assert h.can_handle("/api/v1/auth/api-key") is True

    def test_versioned_api_keys_path(self):
        from aragora.server.handlers.auth.handler import AuthHandler

        h = AuthHandler()
        assert h.can_handle("/api/v1/auth/api-keys") is True

    def test_sdk_alias_keys(self):
        from aragora.server.handlers.auth.handler import AuthHandler

        h = AuthHandler()
        assert h.can_handle("/api/keys") is True

    def test_sdk_alias_keys_prefix(self):
        from aragora.server.handlers.auth.handler import AuthHandler

        h = AuthHandler()
        assert h.can_handle("/api/keys/ara_test1234") is True

    def test_unrelated_path(self):
        from aragora.server.handlers.auth.handler import AuthHandler

        h = AuthHandler()
        assert h.can_handle("/api/auth/nonexistent-endpoint-xyz") is False


# =========================================================================
# Security tests
# =========================================================================


class TestSecurity:
    """Security-focused tests for API key endpoints."""

    def test_path_traversal_prefix(self, handler_instance, http):
        """Path traversal in prefix should not work."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "../../etc/passwd")
        assert _status(result) == 404

    def test_sql_injection_prefix(self, handler_instance, http):
        """SQL injection in prefix should not work."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key_prefix(
            hi, http(method="DELETE"), "'; DROP TABLE users; --"
        )
        assert _status(result) == 404

    def test_xss_prefix(self, handler_instance, http):
        """XSS attempt in prefix should be treated as 404."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key_prefix(
            hi, http(method="DELETE"), "<script>alert('xss')</script>"
        )
        assert _status(result) == 404

    def test_null_byte_prefix(self, handler_instance, http):
        """Null byte in prefix should not match."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key_prefix(
            hi, http(method="DELETE"), "ara_test1234\x00extra"
        )
        assert _status(result) == 404

    def test_very_long_prefix(self, handler_instance, http):
        """Very long prefix should be handled gracefully."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key_prefix(
            hi, http(method="DELETE"), "A" * 10000
        )
        assert _status(result) == 404

    def test_unicode_prefix(self, handler_instance, http):
        """Unicode in prefix should not cause errors."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key_prefix(
            hi, http(method="DELETE"), "ara_\u00e9\u00e8\u00ea"
        )
        assert _status(result) == 404


# =========================================================================
# Edge cases and integration tests
# =========================================================================


class TestEdgeCases:
    """Edge cases and additional coverage."""

    def test_generate_then_list(self, handler_instance, http):
        """After generating a key, listing should show it."""
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user

        # Generate
        result = handle_generate_api_key(hi, http())
        assert _status(result) == 200

        # Now user has api_key_prefix set from generate_api_key
        assert user.api_key_prefix is not None

        # List
        result = handle_list_api_keys(hi, http(method="GET"))
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1
        assert body["keys"][0]["prefix"] == user.api_key_prefix

    def test_generate_revoke_list(self, handler_instance, http):
        """After revoking, listing should show empty."""
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user

        # Generate
        handle_generate_api_key(hi, http())
        assert user.api_key_prefix is not None

        # Revoke - clears fields
        handle_revoke_api_key(hi, http(method="DELETE"))

        # Simulate the user store clearing the fields
        user.api_key_prefix = None
        user.api_key_hash = None
        user.api_key_created_at = None
        user.api_key_expires_at = None

        # List
        result = handle_list_api_keys(hi, http(method="GET"))
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 0

    def test_revoke_by_prefix_exact_match(self, handler_instance, http):
        """Revoke by prefix must be exact match."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user

        # Partial match should fail
        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_test")
        assert _status(result) == 404

        # Exact match should succeed
        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_test1234")
        assert _status(result) == 200

    def test_revoke_by_prefix_case_sensitive(self, handler_instance, http):
        """Prefix matching should be case-sensitive."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_Test1234")
        store.get_user_by_id.return_value = user

        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_test1234")
        assert _status(result) == 404

        result = handle_revoke_api_key_prefix(hi, http(method="DELETE"), "ara_Test1234")
        assert _status(result) == 200

    def test_permission_check_receives_correct_key(self, handler_instance, http):
        """Verify the correct permission key is passed to _check_permission."""
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user

        # Generate checks api_key.create
        h = http()
        handle_generate_api_key(hi, h)
        hi._check_permission.assert_called_with(h, "api_key.create")

    def test_revoke_permission_key(self, handler_instance, http):
        """Verify revoke passes api_key.revoke permission."""
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        h = http(method="DELETE")
        handle_revoke_api_key(hi, h)
        hi._check_permission.assert_called_with(h, "api_key.revoke")

    def test_list_permission_key(self, handler_instance, http):
        """Verify list passes api_key.create permission (reuse for self-service)."""
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        h = http(method="GET")
        handle_list_api_keys(hi, h)
        hi._check_permission.assert_called_with(h, "api_key.create")

    def test_prefix_revoke_permission_key(self, handler_instance, http):
        """Verify prefix revoke passes api_key.revoke permission."""
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        h = http(method="DELETE")
        handle_revoke_api_key_prefix(hi, h, "ara_test1234")
        hi._check_permission.assert_called_with(h, "api_key.revoke")

    @pytest.mark.asyncio
    async def test_handle_returns_result_not_none(self, handler_instance, http, monkeypatch):
        """All valid routes should return a HandlerResult, not None."""
        mock_auth_ctx = MockAuthCtx()
        monkeypatch.setattr(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            lambda handler, user_store: mock_auth_ctx,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.auth.handler.validate_refresh_token",
            lambda token: None,
        )
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user

        # POST /api/auth/api-key
        result = await hi.handle("/api/auth/api-key", {}, http(), method="POST")
        assert result is not None

        # DELETE /api/auth/api-key
        result = await hi.handle("/api/auth/api-key", {}, http(method="DELETE"), method="DELETE")
        assert result is not None

        # GET /api/auth/api-keys
        result = await hi.handle("/api/auth/api-keys", {}, http(method="GET"), method="GET")
        assert result is not None

    def test_handler_result_has_json_content_type(self, handler_instance, http):
        """All API key responses should be JSON."""
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = handle_generate_api_key(hi, http())
        assert result.content_type == "application/json"

    def test_list_handler_result_has_json_content_type(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(api_key_prefix="ara_test1234")
        store.get_user_by_id.return_value = user
        result = handle_list_api_keys(hi, http(method="GET"))
        assert result.content_type == "application/json"

    def test_revoke_handler_result_has_json_content_type(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser()
        store.get_user_by_id.return_value = user
        result = handle_revoke_api_key(hi, http(method="DELETE"))
        assert result.content_type == "application/json"

    def test_error_responses_have_json_content_type(self, handler_instance, http):
        hi, store = handler_instance
        store.get_user_by_id.return_value = None  # user not found
        result = handle_generate_api_key(hi, http())
        assert result.content_type == "application/json"
        assert _status(result) == 404
