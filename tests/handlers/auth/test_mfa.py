"""Tests for MFA handler functions (aragora/server/handlers/auth/mfa.py).

Covers all five MFA endpoints:
- POST /api/auth/mfa/setup     -> handle_mfa_setup
- POST /api/auth/mfa/enable    -> handle_mfa_enable
- POST /api/auth/mfa/disable   -> handle_mfa_disable
- POST /api/auth/mfa/verify    -> handle_mfa_verify
- POST /api/auth/mfa/backup-codes -> handle_mfa_backup_codes

Tests exercise: success paths, permission checks, missing pyotp, user-not-found,
already-enabled/disabled states, invalid codes, backup code flows, audit logging,
edge cases (whitespace, empty bodies, etc.).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.auth.mfa import (
    handle_mfa_backup_codes,
    handle_mfa_disable,
    handle_mfa_enable,
    handle_mfa_setup,
    handle_mfa_verify,
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
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    mfa_backup_codes: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "org_id": self.org_id,
            "role": self.role,
        }

    def verify_password(self, password: str) -> bool:
        return password == "correct-password"


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
class MockTokenPair:
    """Mock token pair from create_token_pair."""

    access_token: str = "new-access-token"
    refresh_token: str = "new-refresh-token"

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": "bearer",
            "expires_in": 3600,
        }


@dataclass
class MockPendingPayload:
    """Mock payload from validate_mfa_pending_token."""

    sub: str = "user-001"


def _make_user_store(user: MockUser | None = None):
    """Create a mock user store with standard methods."""
    store = MagicMock()
    u = user or MockUser()
    store.get_user_by_id.return_value = u
    store.update_user.return_value = None
    store.increment_token_version.return_value = 2
    return store


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_mfa_deps(monkeypatch):
    """Patch dependencies common to all MFA handler functions."""
    mock_auth_ctx = MockAuthCtx()

    # Patch extract_user_from_request used by the proxy in mfa.py
    monkeypatch.setattr(
        "aragora.server.handlers.auth.mfa.extract_user_from_request",
        lambda handler, user_store: mock_auth_ctx,
    )

    # Patch emit_handler_event to no-op
    monkeypatch.setattr(
        "aragora.server.handlers.auth.mfa.emit_handler_event",
        lambda *args, **kwargs: None,
    )

    # Patch audit_security to no-op and mark available
    monkeypatch.setattr("aragora.server.handlers.auth.mfa.AUDIT_AVAILABLE", True)
    monkeypatch.setattr(
        "aragora.server.handlers.auth.mfa.audit_security",
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
# handle_mfa_setup
# =========================================================================


class TestMFASetup:
    """POST /api/auth/mfa/setup."""

    def test_setup_success(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(mfa_enabled=False)
        store.get_user_by_id.return_value = user
        h = http()
        result = handle_mfa_setup(hi, h)
        assert _status(result) == 200
        body = _body(result)
        assert "secret" in body
        assert "provisioning_uri" in body
        assert "message" in body
        store.update_user.assert_called_once()

    def test_setup_returns_provisioning_uri_with_email(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(email="alice@example.com", mfa_enabled=False)
        store.get_user_by_id.return_value = user
        result = handle_mfa_setup(hi, http())
        body = _body(result)
        # Email is URL-encoded in the provisioning URI
        assert "alice" in body["provisioning_uri"]
        assert "example.com" in body["provisioning_uri"]
        assert "Aragora" in body["provisioning_uri"]

    def test_setup_permission_denied(self, handler_instance, http):
        from aragora.server.handlers.base import error_response

        hi, store = handler_instance
        hi._check_permission = MagicMock(
            return_value=error_response("Permission denied", 403)
        )
        result = handle_mfa_setup(hi, http())
        assert _status(result) == 403

    def test_setup_pyotp_not_installed(self, handler_instance, http, monkeypatch):
        hi, store = handler_instance
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyotp":
                raise ImportError("No module named 'pyotp'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = handle_mfa_setup(hi, http())
        assert _status(result) == 503
        assert "not available" in _body(result)["error"].lower()

    def test_setup_user_not_found(self, handler_instance, http):
        hi, store = handler_instance
        store.get_user_by_id.return_value = None
        result = handle_mfa_setup(hi, http())
        assert _status(result) == 404

    def test_setup_mfa_already_enabled(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(mfa_enabled=True)
        store.get_user_by_id.return_value = user
        result = handle_mfa_setup(hi, http())
        assert _status(result) == 400
        assert "already enabled" in _body(result)["error"].lower()

    def test_setup_stores_secret(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(mfa_enabled=False)
        store.get_user_by_id.return_value = user
        result = handle_mfa_setup(hi, http())
        assert _status(result) == 200
        # Verify that update_user was called with a mfa_secret kwarg
        call_kwargs = store.update_user.call_args
        assert "mfa_secret" in call_kwargs[1]
        assert len(call_kwargs[1]["mfa_secret"]) > 0

    def test_setup_secret_matches_response(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(mfa_enabled=False)
        store.get_user_by_id.return_value = user
        result = handle_mfa_setup(hi, http())
        body = _body(result)
        stored_secret = store.update_user.call_args[1]["mfa_secret"]
        assert body["secret"] == stored_secret

    def test_setup_message_mentions_enable(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(mfa_enabled=False)
        store.get_user_by_id.return_value = user
        result = handle_mfa_setup(hi, http())
        body = _body(result)
        assert "/api/auth/mfa/enable" in body["message"]


# =========================================================================
# handle_mfa_enable
# =========================================================================


class TestMFAEnable:
    """POST /api/auth/mfa/enable."""

    def _make_valid_code(self, secret: str) -> str:
        import pyotp

        return pyotp.TOTP(secret).now()

    def test_enable_success(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=False, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        result = handle_mfa_enable(hi, http(body={"code": code}))
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "MFA enabled successfully"
        assert len(body["backup_codes"]) == 10
        assert body["sessions_invalidated"] is True
        assert "warning" in body

    def test_enable_returns_10_backup_codes(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=False, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        result = handle_mfa_enable(hi, http(body={"code": code}))
        body = _body(result)
        assert len(body["backup_codes"]) == 10
        # Each backup code is a hex string (8 chars = 4 bytes hex)
        for bc in body["backup_codes"]:
            assert len(bc) == 8
            int(bc, 16)  # should not raise

    def test_enable_stores_hashed_backup_codes(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=False, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        result = handle_mfa_enable(hi, http(body={"code": code}))
        body = _body(result)

        # Check that hashes of backup codes are stored, not the raw codes
        call_kwargs = store.update_user.call_args[1]
        stored_hashes = json.loads(call_kwargs["mfa_backup_codes"])
        for bc in body["backup_codes"]:
            expected_hash = hashlib.sha256(bc.encode()).hexdigest()
            assert expected_hash in stored_hashes

    def test_enable_sets_mfa_enabled_true(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=False, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        handle_mfa_enable(hi, http(body={"code": code}))
        call_kwargs = store.update_user.call_args[1]
        assert call_kwargs["mfa_enabled"] is True

    def test_enable_invalidates_sessions(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=False, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        handle_mfa_enable(hi, http(body={"code": code}))
        store.increment_token_version.assert_called_once_with(user.id)

    def test_enable_permission_denied(self, handler_instance, http):
        from aragora.server.handlers.base import error_response

        hi, store = handler_instance
        hi._check_permission = MagicMock(
            return_value=error_response("Permission denied", 403)
        )
        result = handle_mfa_enable(hi, http(body={"code": "123456"}))
        assert _status(result) == 403

    def test_enable_pyotp_not_installed(self, handler_instance, http, monkeypatch):
        hi, _ = handler_instance
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyotp":
                raise ImportError("No module named 'pyotp'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = handle_mfa_enable(hi, http(body={"code": "123456"}))
        assert _status(result) == 503

    def test_enable_invalid_json_body(self, handler_instance):
        hi, _ = handler_instance
        h = MockHTTPHandler()
        h.rfile.read.return_value = b"not json"
        h.headers["Content-Length"] = "8"
        result = handle_mfa_enable(hi, h)
        assert _status(result) == 400
        assert "json" in _body(result)["error"].lower() or "Invalid" in _body(result)["error"]

    def test_enable_missing_code(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_enable(hi, http(body={}))
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    def test_enable_empty_code(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_enable(hi, http(body={"code": ""}))
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    def test_enable_whitespace_code(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_enable(hi, http(body={"code": "   "}))
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    def test_enable_user_not_found(self, handler_instance, http):
        hi, store = handler_instance
        store.get_user_by_id.return_value = None
        result = handle_mfa_enable(hi, http(body={"code": "123456"}))
        assert _status(result) == 404

    def test_enable_mfa_already_enabled(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(mfa_enabled=True, mfa_secret="SOME_SECRET")
        store.get_user_by_id.return_value = user
        result = handle_mfa_enable(hi, http(body={"code": "123456"}))
        assert _status(result) == 400
        assert "already enabled" in _body(result)["error"].lower()

    def test_enable_no_mfa_secret(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(mfa_enabled=False, mfa_secret=None)
        store.get_user_by_id.return_value = user
        result = handle_mfa_enable(hi, http(body={"code": "123456"}))
        assert _status(result) == 400
        assert "not set up" in _body(result)["error"].lower()

    def test_enable_invalid_code(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=False, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        result = handle_mfa_enable(hi, http(body={"code": "000000"}))
        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    def test_enable_audit_called(self, handler_instance, http, monkeypatch):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=False, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)

        audit_calls = []
        monkeypatch.setattr(
            "aragora.server.handlers.auth.mfa.audit_security",
            lambda **kwargs: audit_calls.append(kwargs),
        )
        handle_mfa_enable(hi, http(body={"code": code}))
        assert len(audit_calls) == 1
        assert audit_calls[0]["reason"] == "mfa_enabled"

    def test_enable_audit_not_called_when_unavailable(self, handler_instance, http, monkeypatch):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=False, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)

        monkeypatch.setattr("aragora.server.handlers.auth.mfa.AUDIT_AVAILABLE", False)
        audit_calls = []
        monkeypatch.setattr(
            "aragora.server.handlers.auth.mfa.audit_security",
            lambda **kwargs: audit_calls.append(kwargs),
        )
        handle_mfa_enable(hi, http(body={"code": code}))
        assert len(audit_calls) == 0


# =========================================================================
# handle_mfa_disable
# =========================================================================


class TestMFADisable:
    """POST /api/auth/mfa/disable."""

    def _make_valid_code(self, secret: str) -> str:
        import pyotp

        return pyotp.TOTP(secret).now()

    def test_disable_with_code(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        result = handle_mfa_disable(hi, http(body={"code": code}))
        assert _status(result) == 200
        assert "disabled" in _body(result)["message"].lower()

    def test_disable_with_password(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        result = handle_mfa_disable(hi, http(body={"password": "correct-password"}))
        assert _status(result) == 200
        assert "disabled" in _body(result)["message"].lower()

    def test_disable_clears_mfa_fields(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        handle_mfa_disable(hi, http(body={"code": code}))
        call_kwargs = store.update_user.call_args[1]
        assert call_kwargs["mfa_enabled"] is False
        assert call_kwargs["mfa_secret"] is None
        assert call_kwargs["mfa_backup_codes"] is None

    def test_disable_permission_denied(self, handler_instance, http):
        from aragora.server.handlers.base import error_response

        hi, _ = handler_instance
        hi._check_permission = MagicMock(
            return_value=error_response("Permission denied", 403)
        )
        result = handle_mfa_disable(hi, http(body={"code": "123456"}))
        assert _status(result) == 403

    def test_disable_pyotp_not_installed(self, handler_instance, http, monkeypatch):
        hi, _ = handler_instance
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyotp":
                raise ImportError("No module named 'pyotp'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = handle_mfa_disable(hi, http(body={"code": "123456"}))
        assert _status(result) == 503

    def test_disable_invalid_json_body(self, handler_instance):
        hi, _ = handler_instance
        h = MockHTTPHandler()
        h.rfile.read.return_value = b"not json"
        h.headers["Content-Length"] = "8"
        result = handle_mfa_disable(hi, h)
        assert _status(result) == 400

    def test_disable_no_code_no_password(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_disable(hi, http(body={}))
        assert _status(result) == 400
        assert "code or password" in _body(result)["error"].lower()

    def test_disable_empty_code_empty_password(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_disable(hi, http(body={"code": "", "password": ""}))
        assert _status(result) == 400

    def test_disable_whitespace_code_whitespace_password(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_disable(hi, http(body={"code": "  ", "password": "  "}))
        assert _status(result) == 400

    def test_disable_user_not_found(self, handler_instance, http):
        hi, store = handler_instance
        store.get_user_by_id.return_value = None
        result = handle_mfa_disable(hi, http(body={"code": "123456"}))
        assert _status(result) == 404

    def test_disable_mfa_not_enabled(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(mfa_enabled=False)
        store.get_user_by_id.return_value = user
        result = handle_mfa_disable(hi, http(body={"code": "123456"}))
        assert _status(result) == 400
        assert "not enabled" in _body(result)["error"].lower()

    def test_disable_invalid_code(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        result = handle_mfa_disable(hi, http(body={"code": "000000"}))
        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    def test_disable_wrong_password(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        result = handle_mfa_disable(hi, http(body={"password": "wrong-password"}))
        assert _status(result) == 400
        assert "invalid password" in _body(result)["error"].lower()

    def test_disable_audit_called(self, handler_instance, http, monkeypatch):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)

        audit_calls = []
        monkeypatch.setattr(
            "aragora.server.handlers.auth.mfa.audit_security",
            lambda **kwargs: audit_calls.append(kwargs),
        )
        handle_mfa_disable(hi, http(body={"code": code}))
        assert len(audit_calls) == 1
        assert audit_calls[0]["reason"] == "mfa_disabled"

    def test_disable_audit_not_called_when_unavailable(self, handler_instance, http, monkeypatch):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)

        monkeypatch.setattr("aragora.server.handlers.auth.mfa.AUDIT_AVAILABLE", False)
        audit_calls = []
        monkeypatch.setattr(
            "aragora.server.handlers.auth.mfa.audit_security",
            lambda **kwargs: audit_calls.append(kwargs),
        )
        handle_mfa_disable(hi, http(body={"code": code}))
        assert len(audit_calls) == 0

    def test_disable_code_takes_priority_over_password(self, handler_instance, http):
        """When both code and password are provided, code is checked first."""
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        result = handle_mfa_disable(
            hi, http(body={"code": code, "password": "wrong-password"})
        )
        # Code is valid, so should succeed even though password is wrong
        assert _status(result) == 200


# =========================================================================
# handle_mfa_verify
# =========================================================================


class TestMFAVerify:
    """POST /api/auth/mfa/verify."""

    JWT_MODULE = "aragora.billing.jwt_auth"

    def _make_valid_code(self, secret: str) -> str:
        import pyotp

        return pyotp.TOTP(secret).now()

    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_totp_success(self, mock_validate, mock_create, mock_bl, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        mock_validate.return_value = MockPendingPayload()
        mock_create.return_value = MockTokenPair()
        bl = MagicMock()
        mock_bl.return_value = bl

        code = self._make_valid_code(secret)
        result = handle_mfa_verify(
            hi, http(body={"code": code, "pending_token": "pt-valid"})
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "MFA verification successful"
        assert body["tokens"]["access_token"] == "new-access-token"
        assert "user" in body

    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_blacklists_pending_token(self, mock_validate, mock_create, mock_bl, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        mock_validate.return_value = MockPendingPayload()
        mock_create.return_value = MockTokenPair()
        bl = MagicMock()
        mock_bl.return_value = bl

        code = self._make_valid_code(secret)
        handle_mfa_verify(hi, http(body={"code": code, "pending_token": "pt-valid"}))
        bl.revoke_token.assert_called_once_with("pt-valid")

    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_backup_code(self, mock_validate, mock_create, mock_bl, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        backup_code = "abcd1234"
        backup_hash = hashlib.sha256(backup_code.encode()).hexdigest()
        other_hash = hashlib.sha256(b"other_code").hexdigest()
        user = MockUser(
            mfa_enabled=True,
            mfa_secret=secret,
            mfa_backup_codes=json.dumps([backup_hash, other_hash]),
        )
        store.get_user_by_id.return_value = user
        mock_validate.return_value = MockPendingPayload()
        mock_create.return_value = MockTokenPair()
        bl = MagicMock()
        mock_bl.return_value = bl

        result = handle_mfa_verify(
            hi, http(body={"code": backup_code, "pending_token": "pt-valid"})
        )
        assert _status(result) == 200
        body = _body(result)
        assert "backup code used" in body["message"].lower()
        assert body["backup_codes_remaining"] == 1

    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_backup_code_removes_used_hash(self, mock_validate, mock_create, mock_bl, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        backup_code = "abcd1234"
        backup_hash = hashlib.sha256(backup_code.encode()).hexdigest()
        other_hash = hashlib.sha256(b"other_code").hexdigest()
        user = MockUser(
            mfa_enabled=True,
            mfa_secret=secret,
            mfa_backup_codes=json.dumps([backup_hash, other_hash]),
        )
        store.get_user_by_id.return_value = user
        mock_validate.return_value = MockPendingPayload()
        mock_create.return_value = MockTokenPair()
        bl = MagicMock()
        mock_bl.return_value = bl

        handle_mfa_verify(
            hi, http(body={"code": backup_code, "pending_token": "pt-valid"})
        )
        call_kwargs = store.update_user.call_args[1]
        remaining_hashes = json.loads(call_kwargs["mfa_backup_codes"])
        assert backup_hash not in remaining_hashes
        assert other_hash in remaining_hashes

    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_backup_code_warning_when_low(self, mock_validate, mock_create, mock_bl, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        backup_code = "abcd1234"
        backup_hash = hashlib.sha256(backup_code.encode()).hexdigest()
        # Only 2 remaining after using this one (below 5 threshold)
        other_hashes = [hashlib.sha256(f"code{i}".encode()).hexdigest() for i in range(2)]
        user = MockUser(
            mfa_enabled=True,
            mfa_secret=secret,
            mfa_backup_codes=json.dumps([backup_hash] + other_hashes),
        )
        store.get_user_by_id.return_value = user
        mock_validate.return_value = MockPendingPayload()
        mock_create.return_value = MockTokenPair()
        bl = MagicMock()
        mock_bl.return_value = bl

        result = handle_mfa_verify(
            hi, http(body={"code": backup_code, "pending_token": "pt-valid"})
        )
        body = _body(result)
        assert body["warning"] is not None
        assert "2 remaining" in body["warning"]

    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_backup_code_no_warning_when_enough(self, mock_validate, mock_create, mock_bl, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        backup_code = "abcd1234"
        backup_hash = hashlib.sha256(backup_code.encode()).hexdigest()
        # 6 remaining after using this one (>= 5 threshold)
        other_hashes = [hashlib.sha256(f"code{i}".encode()).hexdigest() for i in range(6)]
        user = MockUser(
            mfa_enabled=True,
            mfa_secret=secret,
            mfa_backup_codes=json.dumps([backup_hash] + other_hashes),
        )
        store.get_user_by_id.return_value = user
        mock_validate.return_value = MockPendingPayload()
        mock_create.return_value = MockTokenPair()
        bl = MagicMock()
        mock_bl.return_value = bl

        result = handle_mfa_verify(
            hi, http(body={"code": backup_code, "pending_token": "pt-valid"})
        )
        body = _body(result)
        assert body.get("warning") is None

    def test_verify_pyotp_not_installed(self, handler_instance, http, monkeypatch):
        hi, _ = handler_instance
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyotp":
                raise ImportError("No module named 'pyotp'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = handle_mfa_verify(
            hi, http(body={"code": "123456", "pending_token": "pt"})
        )
        assert _status(result) == 503

    def test_verify_invalid_json_body(self, handler_instance):
        hi, _ = handler_instance
        h = MockHTTPHandler()
        h.rfile.read.return_value = b"not json"
        h.headers["Content-Length"] = "8"
        result = handle_mfa_verify(hi, h)
        assert _status(result) == 400

    def test_verify_missing_code(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_verify(
            hi, http(body={"pending_token": "pt-valid"})
        )
        assert _status(result) == 400
        assert "code" in _body(result)["error"].lower()

    def test_verify_empty_code(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_verify(
            hi, http(body={"code": "", "pending_token": "pt-valid"})
        )
        assert _status(result) == 400

    def test_verify_missing_pending_token(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_verify(hi, http(body={"code": "123456"}))
        assert _status(result) == 400
        assert "pending token" in _body(result)["error"].lower()

    def test_verify_empty_pending_token(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_verify(
            hi, http(body={"code": "123456", "pending_token": ""})
        )
        assert _status(result) == 400

    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_invalid_pending_token(self, mock_validate, handler_instance, http):
        hi, _ = handler_instance
        mock_validate.return_value = None
        result = handle_mfa_verify(
            hi, http(body={"code": "123456", "pending_token": "bad-token"})
        )
        assert _status(result) == 401
        assert "expired" in _body(result)["error"].lower() or "invalid" in _body(result)["error"].lower()

    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_user_not_found(self, mock_validate, handler_instance, http):
        hi, store = handler_instance
        mock_validate.return_value = MockPendingPayload()
        store.get_user_by_id.return_value = None
        result = handle_mfa_verify(
            hi, http(body={"code": "123456", "pending_token": "pt-valid"})
        )
        assert _status(result) == 404

    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_mfa_not_enabled(self, mock_validate, handler_instance, http):
        hi, store = handler_instance
        mock_validate.return_value = MockPendingPayload()
        user = MockUser(mfa_enabled=False, mfa_secret=None)
        store.get_user_by_id.return_value = user
        result = handle_mfa_verify(
            hi, http(body={"code": "123456", "pending_token": "pt-valid"})
        )
        assert _status(result) == 400
        assert "not enabled" in _body(result)["error"].lower()

    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_mfa_enabled_no_secret(self, mock_validate, handler_instance, http):
        hi, store = handler_instance
        mock_validate.return_value = MockPendingPayload()
        user = MockUser(mfa_enabled=True, mfa_secret=None)
        store.get_user_by_id.return_value = user
        result = handle_mfa_verify(
            hi, http(body={"code": "123456", "pending_token": "pt-valid"})
        )
        assert _status(result) == 400

    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_invalid_totp_no_backup(self, mock_validate, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret, mfa_backup_codes=None)
        store.get_user_by_id.return_value = user
        mock_validate.return_value = MockPendingPayload()
        result = handle_mfa_verify(
            hi, http(body={"code": "000000", "pending_token": "pt-valid"})
        )
        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_invalid_totp_and_invalid_backup(self, mock_validate, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        other_hash = hashlib.sha256(b"some_other_code").hexdigest()
        user = MockUser(
            mfa_enabled=True,
            mfa_secret=secret,
            mfa_backup_codes=json.dumps([other_hash]),
        )
        store.get_user_by_id.return_value = user
        mock_validate.return_value = MockPendingPayload()
        result = handle_mfa_verify(
            hi, http(body={"code": "wrong_code", "pending_token": "pt-valid"})
        )
        assert _status(result) == 400

    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_no_user_store(self, mock_validate, http):
        from aragora.server.handlers.auth.handler import AuthHandler

        hi = AuthHandler(server_context={})
        hi._check_permission = MagicMock(return_value=None)
        mock_validate.return_value = MockPendingPayload()
        result = handle_mfa_verify(
            hi, http(body={"code": "123456", "pending_token": "pt-valid"})
        )
        assert _status(result) == 503

    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_backup_code_blacklists_token(self, mock_validate, mock_create, mock_bl, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        backup_code = "dead1234"
        backup_hash = hashlib.sha256(backup_code.encode()).hexdigest()
        user = MockUser(
            mfa_enabled=True,
            mfa_secret=secret,
            mfa_backup_codes=json.dumps([backup_hash]),
        )
        store.get_user_by_id.return_value = user
        mock_validate.return_value = MockPendingPayload()
        mock_create.return_value = MockTokenPair()
        bl = MagicMock()
        mock_bl.return_value = bl

        handle_mfa_verify(
            hi, http(body={"code": backup_code, "pending_token": "pt-pending"})
        )
        bl.revoke_token.assert_called_once_with("pt-pending")


# =========================================================================
# handle_mfa_backup_codes
# =========================================================================


class TestMFABackupCodes:
    """POST /api/auth/mfa/backup-codes."""

    def _make_valid_code(self, secret: str) -> str:
        import pyotp

        return pyotp.TOTP(secret).now()

    def test_backup_codes_success(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        result = handle_mfa_backup_codes(hi, http(body={"code": code}))
        assert _status(result) == 200
        body = _body(result)
        assert len(body["backup_codes"]) == 10
        assert "warning" in body

    def test_backup_codes_stores_hashes(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        result = handle_mfa_backup_codes(hi, http(body={"code": code}))
        body = _body(result)

        call_kwargs = store.update_user.call_args[1]
        stored_hashes = json.loads(call_kwargs["mfa_backup_codes"])
        assert len(stored_hashes) == 10
        for bc in body["backup_codes"]:
            expected_hash = hashlib.sha256(bc.encode()).hexdigest()
            assert expected_hash in stored_hashes

    def test_backup_codes_unique(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        result = handle_mfa_backup_codes(hi, http(body={"code": code}))
        body = _body(result)
        # All 10 backup codes should be unique
        assert len(set(body["backup_codes"])) == 10

    def test_backup_codes_permission_denied(self, handler_instance, http):
        from aragora.server.handlers.base import error_response

        hi, _ = handler_instance
        hi._check_permission = MagicMock(
            return_value=error_response("Permission denied", 403)
        )
        result = handle_mfa_backup_codes(hi, http(body={"code": "123456"}))
        assert _status(result) == 403

    def test_backup_codes_pyotp_not_installed(self, handler_instance, http, monkeypatch):
        hi, _ = handler_instance
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyotp":
                raise ImportError("No module named 'pyotp'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = handle_mfa_backup_codes(hi, http(body={"code": "123456"}))
        assert _status(result) == 503

    def test_backup_codes_invalid_json(self, handler_instance):
        hi, _ = handler_instance
        h = MockHTTPHandler()
        h.rfile.read.return_value = b"not json"
        h.headers["Content-Length"] = "8"
        result = handle_mfa_backup_codes(hi, h)
        assert _status(result) == 400

    def test_backup_codes_missing_code(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_backup_codes(hi, http(body={}))
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    def test_backup_codes_empty_code(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_backup_codes(hi, http(body={"code": ""}))
        assert _status(result) == 400

    def test_backup_codes_whitespace_code(self, handler_instance, http):
        hi, _ = handler_instance
        result = handle_mfa_backup_codes(hi, http(body={"code": "   "}))
        assert _status(result) == 400

    def test_backup_codes_user_not_found(self, handler_instance, http):
        hi, store = handler_instance
        store.get_user_by_id.return_value = None
        result = handle_mfa_backup_codes(hi, http(body={"code": "123456"}))
        assert _status(result) == 404

    def test_backup_codes_mfa_not_enabled(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(mfa_enabled=False, mfa_secret=None)
        store.get_user_by_id.return_value = user
        result = handle_mfa_backup_codes(hi, http(body={"code": "123456"}))
        assert _status(result) == 400
        assert "not enabled" in _body(result)["error"].lower()

    def test_backup_codes_mfa_enabled_no_secret(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(mfa_enabled=True, mfa_secret=None)
        store.get_user_by_id.return_value = user
        result = handle_mfa_backup_codes(hi, http(body={"code": "123456"}))
        assert _status(result) == 400

    def test_backup_codes_invalid_code(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        result = handle_mfa_backup_codes(hi, http(body={"code": "000000"}))
        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    def test_backup_codes_warning_message(self, handler_instance, http):
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user
        code = self._make_valid_code(secret)
        result = handle_mfa_backup_codes(hi, http(body={"code": code}))
        body = _body(result)
        assert "save" in body["warning"].lower() or "securely" in body["warning"].lower()


# =========================================================================
# Cross-cutting / integration-style tests
# =========================================================================


class TestMFAFlowIntegration:
    """Tests simulating multi-step MFA flows."""

    def test_setup_then_enable_flow(self, handler_instance, http):
        """Setup generates secret, enable uses it to activate MFA."""
        import pyotp

        hi, store = handler_instance
        user = MockUser(mfa_enabled=False, mfa_secret=None)
        store.get_user_by_id.return_value = user

        # Step 1: Setup
        setup_result = handle_mfa_setup(hi, http())
        assert _status(setup_result) == 200
        secret = _body(setup_result)["secret"]

        # Simulate the secret being stored
        user.mfa_secret = secret

        # Step 2: Enable with valid TOTP
        code = pyotp.TOTP(secret).now()
        enable_result = handle_mfa_enable(hi, http(body={"code": code}))
        assert _status(enable_result) == 200
        assert len(_body(enable_result)["backup_codes"]) == 10

    def test_setup_already_enabled_user_cannot_setup_again(self, handler_instance, http):
        hi, store = handler_instance
        user = MockUser(mfa_enabled=True, mfa_secret="EXISTING_SECRET")
        store.get_user_by_id.return_value = user
        result = handle_mfa_setup(hi, http())
        assert _status(result) == 400

    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_returns_user_dict(self, mock_validate, mock_create, mock_bl, handler_instance, http):
        """Verify response includes user dict from user.to_dict()."""
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret, email="verify@test.com")
        store.get_user_by_id.return_value = user
        mock_validate.return_value = MockPendingPayload()
        mock_create.return_value = MockTokenPair()
        bl = MagicMock()
        mock_bl.return_value = bl

        code = pyotp.TOTP(secret).now()
        result = handle_mfa_verify(
            hi, http(body={"code": code, "pending_token": "pt-valid"})
        )
        body = _body(result)
        assert body["user"]["email"] == "verify@test.com"
        assert body["user"]["id"] == "user-001"

    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    @patch("aragora.billing.jwt_auth.validate_mfa_pending_token")
    def test_verify_creates_token_pair_with_user_fields(self, mock_validate, mock_create, mock_bl, handler_instance, http):
        """Verify that create_token_pair is called with the correct user fields."""
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(
            id="u-42", email="token@test.com", org_id="org-99",
            role="editor", mfa_enabled=True, mfa_secret=secret,
        )
        store.get_user_by_id.return_value = user
        mock_validate.return_value = MockPendingPayload(sub="u-42")
        mock_create.return_value = MockTokenPair()
        bl = MagicMock()
        mock_bl.return_value = bl

        code = pyotp.TOTP(secret).now()
        handle_mfa_verify(hi, http(body={"code": code, "pending_token": "pt"}))
        mock_create.assert_called_once_with(
            user_id="u-42",
            email="token@test.com",
            org_id="org-99",
            role="editor",
        )

    def test_disable_then_setup_again(self, handler_instance, http):
        """After disabling MFA, user can set it up again."""
        import pyotp

        hi, store = handler_instance
        secret = pyotp.random_base32()
        user = MockUser(mfa_enabled=True, mfa_secret=secret)
        store.get_user_by_id.return_value = user

        # Disable first
        code = pyotp.TOTP(secret).now()
        disable_result = handle_mfa_disable(hi, http(body={"code": code}))
        assert _status(disable_result) == 200

        # Simulate the disable having cleared fields
        user.mfa_enabled = False
        user.mfa_secret = None

        # Setup again
        setup_result = handle_mfa_setup(hi, http())
        assert _status(setup_result) == 200
        assert "secret" in _body(setup_result)


# =========================================================================
# Cross-cutting / module-level
# =========================================================================


class TestModuleExports:
    """Verify __all__ exports."""

    def test_all_exports(self):
        from aragora.server.handlers.auth import mfa

        assert "handle_mfa_setup" in mfa.__all__
        assert "handle_mfa_enable" in mfa.__all__
        assert "handle_mfa_disable" in mfa.__all__
        assert "handle_mfa_verify" in mfa.__all__
        assert "handle_mfa_backup_codes" in mfa.__all__

    def test_all_exports_count(self):
        from aragora.server.handlers.auth import mfa

        assert len(mfa.__all__) == 5
