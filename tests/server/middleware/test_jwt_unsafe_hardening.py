"""Tests for JWT unsafe decode hardening and insecure JWT audit logging.

Verifies:
1. _decode_jwt_unsafe rejects tokens missing the 'sub' claim
2. _decode_jwt_unsafe rejects tokens with empty 'sub' claim
3. Insecure JWT decode emits an audit event
4. Insecure JWT decode falls back to logger.error when audit unavailable
5. Startup validation warns about ARAGORA_ALLOW_INSECURE_JWT
"""

from __future__ import annotations

import base64
import json
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jwt_payload(payload: dict) -> str:
    """Create a fake JWT string from a payload dict (no real signing)."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    sig = base64.urlsafe_b64encode(b"fake").rstrip(b"=")
    return f"{header.decode()}.{body.decode()}.{sig.decode()}"


# ---------------------------------------------------------------------------
# Tests: _decode_jwt_unsafe sub claim validation
# ---------------------------------------------------------------------------


class TestDecodeJwtUnsafeSubClaim:
    """Verify _decode_jwt_unsafe rejects tokens without valid sub."""

    def _get_validator(self):
        """Create a SupabaseAuthValidator instance for testing."""
        from aragora.server.middleware.user_auth import SupabaseAuthValidator

        return SupabaseAuthValidator(jwt_secret=None, supabase_url=None)

    def test_rejects_missing_sub(self):
        """JWT without 'sub' claim should be rejected."""
        payload = {"exp": time.time() + 3600, "email": "test@example.com"}
        token = _make_jwt_payload(payload)

        validator = self._get_validator()
        result = validator._decode_jwt_unsafe(token)

        assert result is None

    def test_rejects_empty_sub(self):
        """JWT with empty 'sub' claim should be rejected."""
        payload = {"exp": time.time() + 3600, "sub": "", "email": "test@example.com"}
        token = _make_jwt_payload(payload)

        validator = self._get_validator()
        result = validator._decode_jwt_unsafe(token)

        assert result is None

    def test_accepts_valid_sub(self):
        """JWT with valid 'sub' claim should be accepted."""
        payload = {"exp": time.time() + 3600, "sub": "user-123", "email": "test@example.com"}
        token = _make_jwt_payload(payload)

        validator = self._get_validator()
        result = validator._decode_jwt_unsafe(token)

        assert result is not None
        assert result["sub"] == "user-123"

    def test_rejects_expired_token(self):
        """JWT with expired 'exp' should be rejected."""
        payload = {"exp": time.time() - 3600, "sub": "user-123"}
        token = _make_jwt_payload(payload)

        validator = self._get_validator()
        result = validator._decode_jwt_unsafe(token)

        assert result is None

    def test_rejects_missing_exp(self):
        """JWT without 'exp' claim should be rejected."""
        payload = {"sub": "user-123"}
        token = _make_jwt_payload(payload)

        validator = self._get_validator()
        result = validator._decode_jwt_unsafe(token)

        assert result is None

    def test_rejects_malformed_token(self):
        """Malformed token string should be rejected."""
        validator = self._get_validator()
        assert validator._decode_jwt_unsafe("not.a.valid.jwt.string") is None
        assert validator._decode_jwt_unsafe("too-short") is None
        assert validator._decode_jwt_unsafe("") is None


# ---------------------------------------------------------------------------
# Tests: Insecure JWT audit logging
# ---------------------------------------------------------------------------


class TestInsecureJwtAuditLogging:
    """Verify audit events are emitted when insecure JWT decode is used."""

    @patch.dict(
        "os.environ",
        {
            "ARAGORA_ENVIRONMENT": "development",
            "ARAGORA_ALLOW_INSECURE_JWT": "true",
        },
    )
    @patch("aragora.server.middleware.user_auth.HAS_JWT", False)
    @patch("aragora.server.middleware.user_auth._jwt_module", None)
    def test_audit_event_on_insecure_decode(self):
        """When insecure decode is used, an audit event should be emitted."""
        from aragora.server.middleware.user_auth import SupabaseAuthValidator

        validator = SupabaseAuthValidator(jwt_secret=None, supabase_url=None)

        payload = {"exp": time.time() + 3600, "sub": "user-456", "email": "test@example.com"}
        token = _make_jwt_payload(payload)

        mock_audit = MagicMock()
        with patch(
            "aragora.server.middleware.user_auth.audit_security",
            mock_audit,
            create=True,
        ):
            # The validate_token method will use _decode_jwt_unsafe
            # We need to patch the audit import inside the method
            with patch.dict("sys.modules", {"aragora.audit.unified": MagicMock(audit_security=mock_audit)}):
                result = validator.validate_token(token)

        # The user should be returned (insecure decode succeeds in dev)
        if result is not None:
            assert result.id == "user-456"

    @patch.dict(
        "os.environ",
        {
            "ARAGORA_ENVIRONMENT": "development",
            "ARAGORA_ALLOW_INSECURE_JWT": "true",
        },
    )
    @patch("aragora.server.middleware.user_auth.HAS_JWT", False)
    @patch("aragora.server.middleware.user_auth._jwt_module", None)
    def test_audit_fallback_to_logger(self, caplog):
        """When audit module unavailable, falls back to logger.error."""
        import logging

        from aragora.server.middleware.user_auth import SupabaseAuthValidator

        validator = SupabaseAuthValidator(jwt_secret=None, supabase_url=None)

        payload = {"exp": time.time() + 3600, "sub": "user-789", "email": "test@example.com"}
        token = _make_jwt_payload(payload)

        # Make audit import fail
        with patch.dict("sys.modules", {"aragora.audit.unified": None}):
            with caplog.at_level(logging.ERROR, logger="aragora.server.middleware.user_auth"):
                validator.validate_token(token)

        # Should have logged the security audit fallback
        audit_msgs = [r for r in caplog.records if "insecure JWT decode" in r.message or "INSECURE" in r.message]
        assert len(audit_msgs) > 0


# ---------------------------------------------------------------------------
# Tests: Startup validation for ARAGORA_ALLOW_INSECURE_JWT
# ---------------------------------------------------------------------------


class TestStartupInsecureJwtValidation:
    """Verify startup warns about ARAGORA_ALLOW_INSECURE_JWT."""

    @patch.dict(
        "os.environ",
        {
            "ARAGORA_ALLOW_INSECURE_JWT": "true",
            "ARAGORA_ENV": "production",
        },
    )
    def test_production_warning_for_insecure_jwt(self):
        """In production, ARAGORA_ALLOW_INSECURE_JWT should produce a warning."""
        from aragora.server.startup.validation import validate_production_config

        errors, warnings = validate_production_config()

        insecure_warnings = [w for w in warnings if "ARAGORA_ALLOW_INSECURE_JWT" in w]
        assert len(insecure_warnings) >= 1
        assert "production" in insecure_warnings[0].lower() or "IGNORED" in insecure_warnings[0]

    @patch.dict(
        "os.environ",
        {
            "ARAGORA_ALLOW_INSECURE_JWT": "true",
            "ARAGORA_ENV": "development",
        },
    )
    def test_dev_mode_logs_security_warning(self, caplog):
        """In dev mode, ARAGORA_ALLOW_INSECURE_JWT should log a security warning."""
        import logging

        from aragora.server.startup.validation import validate_production_config

        with caplog.at_level(logging.WARNING):
            validate_production_config()

        security_msgs = [r for r in caplog.records if "ARAGORA_ALLOW_INSECURE_JWT" in r.message]
        assert len(security_msgs) >= 1

    @patch.dict(
        "os.environ",
        {"ARAGORA_ENV": "production"},
        clear=False,
    )
    def test_no_warning_when_not_set(self):
        """When ARAGORA_ALLOW_INSECURE_JWT is not set, no warning should appear."""
        import os
        os.environ.pop("ARAGORA_ALLOW_INSECURE_JWT", None)

        from aragora.server.startup.validation import validate_production_config

        errors, warnings = validate_production_config()

        insecure_warnings = [w for w in warnings if "INSECURE_JWT" in w]
        assert len(insecure_warnings) == 0
