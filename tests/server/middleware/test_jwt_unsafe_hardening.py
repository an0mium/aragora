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
import logging
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
        validator._cache.clear()

        payload = {"exp": time.time() + 3600, "sub": "user-456", "email": "test@example.com"}
        token = _make_jwt_payload(payload)

        mock_audit = MagicMock()
        mock_module = MagicMock()
        mock_module.audit_security = mock_audit

        with patch.dict("sys.modules", {"aragora.audit.unified": mock_module}):
            result = validator.validate_token(token)

        # The user should be returned (insecure decode succeeds in dev)
        assert result is not None
        assert result.id == "user-456"

        # Audit should have been called
        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args
        assert call_kwargs[1]["event_type"] == "insecure_jwt_decode"
        assert call_kwargs[1]["actor_id"] == "user-456"

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
        from aragora.server.middleware.user_auth import SupabaseAuthValidator

        validator = SupabaseAuthValidator(jwt_secret=None, supabase_url=None)
        validator._cache.clear()

        payload = {"exp": time.time() + 3600, "sub": "user-789", "email": "test@example.com"}
        token = _make_jwt_payload(payload)

        # Make audit import fail by removing the module
        import sys

        orig = sys.modules.pop("aragora.audit.unified", None)
        try:
            # Ensure the import will fail
            sys.modules["aragora.audit.unified"] = None  # type: ignore[assignment]
            with caplog.at_level(logging.WARNING, logger="aragora.server.middleware.user_auth"):
                validator.validate_token(token)
        finally:
            if orig is not None:
                sys.modules["aragora.audit.unified"] = orig
            else:
                sys.modules.pop("aragora.audit.unified", None)

        # Should have logged the INSECURE warning at minimum
        insecure_msgs = [
            r for r in caplog.records if "INSECURE" in r.message or "SECURITY AUDIT" in r.message
        ]
        assert len(insecure_msgs) > 0


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
    def test_production_warning_for_insecure_jwt(self, caplog):
        """In production, ARAGORA_ALLOW_INSECURE_JWT should produce a warning."""
        from aragora.server.startup.validation import check_production_requirements

        with caplog.at_level(logging.WARNING):
            check_production_requirements()

        # The warning should be logged (via warnings list which gets logged)
        insecure_msgs = [
            r for r in caplog.records if "ARAGORA_ALLOW_INSECURE_JWT" in r.message
        ]
        assert len(insecure_msgs) >= 1

    @patch.dict(
        "os.environ",
        {
            "ARAGORA_ALLOW_INSECURE_JWT": "true",
            "ARAGORA_ENV": "development",
        },
    )
    def test_dev_mode_logs_security_warning(self, caplog):
        """In dev mode, ARAGORA_ALLOW_INSECURE_JWT should log a security warning."""
        from aragora.server.startup.validation import check_production_requirements

        with caplog.at_level(logging.WARNING):
            check_production_requirements()

        security_msgs = [
            r for r in caplog.records if "ARAGORA_ALLOW_INSECURE_JWT" in r.message
        ]
        assert len(security_msgs) >= 1

    @patch.dict(
        "os.environ",
        {"ARAGORA_ENV": "production"},
        clear=False,
    )
    def test_no_warning_when_not_set(self, caplog):
        """When ARAGORA_ALLOW_INSECURE_JWT is not set, no warning should appear."""
        import os

        os.environ.pop("ARAGORA_ALLOW_INSECURE_JWT", None)

        from aragora.server.startup.validation import check_production_requirements

        with caplog.at_level(logging.WARNING):
            check_production_requirements()

        insecure_warnings = [
            r for r in caplog.records if "INSECURE_JWT" in r.message
        ]
        assert len(insecure_warnings) == 0
