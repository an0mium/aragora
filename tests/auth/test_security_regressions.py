"""
Security regression tests for authentication and audit modules.

These tests verify critical security fixes remain in place:
1. SAML signature validation - simplified parser requires explicit opt-in
2. JWT timing attack protection - hmac.compare_digest() used for signature comparison
3. OIDC token validation - ID token validation enforced in production
4. Audit signing key requirement - production requires persistent key
"""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest

from aragora.auth.saml import SAMLConfig, SAMLProvider
from aragora.auth.sso import SSOConfigurationError, SSOProviderType


def make_saml_config(**kwargs) -> SAMLConfig:
    """Helper to create SAMLConfig with sensible defaults."""
    defaults = {
        "provider_type": SSOProviderType.SAML,
        "entity_id": "https://aragora.example.com/saml/metadata",
        "callback_url": "https://aragora.example.com/saml/acs",
        "idp_entity_id": "https://idp.example.com/metadata",
        "idp_sso_url": "https://idp.example.com/sso",
        "idp_certificate": "-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----",
    }
    defaults.update(kwargs)
    return SAMLConfig(**defaults)


# ============================================================================
# SAML Signature Validation Security Tests
# ============================================================================


class TestSAMLSignatureValidationSecurity:
    """Tests for SAML signature validation security requirements."""

    def test_saml_requires_library_in_production(self):
        """SECURITY: SAML must require python3-saml in production."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
                with pytest.raises(SSOConfigurationError) as exc:
                    SAMLProvider(config)

                assert "python3-saml" in str(exc.value)

    def test_saml_requires_library_in_staging(self):
        """SECURITY: SAML must require python3-saml in staging."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(os.environ, {"ARAGORA_ENV": "staging"}):
                with pytest.raises(SSOConfigurationError) as exc:
                    SAMLProvider(config)

                assert "python3-saml" in str(exc.value)

    def test_saml_requires_explicit_opt_in_for_unsafe_parser(self):
        """SECURITY: Simplified SAML parser requires explicit opt-in."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
                # Without explicit opt-in, should fail
                with pytest.raises(SSOConfigurationError) as exc:
                    SAMLProvider(config)

                assert "ARAGORA_ALLOW_UNSAFE_SAML" in str(exc.value)

    def test_saml_explicit_opt_in_works_in_dev(self):
        """Test explicit opt-in allows unsafe parser in development."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                os.environ,
                {"ARAGORA_ENV": "development", "ARAGORA_ALLOW_UNSAFE_SAML": "true"},
            ):
                # With explicit opt-in, should work
                provider = SAMLProvider(config)
                assert provider is not None

    def test_saml_explicit_opt_in_blocked_in_production(self):
        """SECURITY: Explicit opt-in must NOT work in production."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                os.environ,
                {"ARAGORA_ENV": "production", "ARAGORA_ALLOW_UNSAFE_SAML": "true"},
            ):
                # Even with opt-in, production must require the library
                with pytest.raises(SSOConfigurationError) as exc:
                    SAMLProvider(config)

                assert "python3-saml" in str(exc.value)


# ============================================================================
# JWT Timing Attack Protection Tests
# ============================================================================


class TestJWTTimingAttackProtection:
    """Tests for JWT timing attack protection."""

    def test_jwt_decode_uses_compare_digest(self):
        """SECURITY: JWT signature comparison must use hmac.compare_digest."""
        # This test verifies the code uses the secure comparison method
        # by inspecting the source code
        import inspect

        from aragora.billing.auth import tokens

        source = inspect.getsource(tokens.decode_jwt)

        # Must use hmac.compare_digest for signature comparison
        assert "hmac.compare_digest" in source, (
            "JWT signature comparison must use hmac.compare_digest() to prevent timing attacks"
        )

        # Must NOT use simple == comparison for signatures
        # (The code does use == for algorithm string comparison, which is fine,
        # so we check specifically for signature comparison context)
        assert "signature_valid = hmac.compare_digest" in source, (
            "Signature validation must use hmac.compare_digest()"
        )


# ============================================================================
# OIDC Token Validation Security Tests
# ============================================================================


class TestOIDCTokenValidationSecurity:
    """Tests for OIDC token validation security requirements."""

    def test_oidc_enforces_token_validation_in_production(self):
        """SECURITY: OIDC must enforce ID token validation in production."""
        import inspect

        from aragora.auth import oidc

        source = inspect.getsource(oidc.OIDCProvider._get_user_info)

        # Must check for dev auth fallback mode and raise on validation failure
        # The implementation uses _allow_dev_auth_fallback() which checks both
        # ARAGORA_ENV and ARAGORA_ALLOW_DEV_AUTH_FALLBACK
        assert "_allow_dev_auth_fallback()" in source or "_is_production_mode()" in source, (
            "OIDC must check environment mode for token validation enforcement"
        )
        assert "raise SSOAuthenticationError" in source, (
            "OIDC must raise error on token validation failure in production"
        )


# ============================================================================
# Audit Signing Key Security Tests
# ============================================================================


class TestAuditSigningKeySecurity:
    """Tests for audit signing key security requirements."""

    def test_audit_requires_signing_key_in_production(self):
        """SECURITY: Audit signing key must be required in production."""
        from aragora.rbac import audit

        # Reset the global key state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=False):
            # Remove the signing key if present
            env_copy = dict(os.environ)
            env_copy.pop("ARAGORA_AUDIT_SIGNING_KEY", None)
            env_copy["ARAGORA_ENV"] = "production"

            with patch.dict(os.environ, env_copy, clear=True):
                # Reset again inside the context
                with audit._signing_key_lock:
                    audit._AUDIT_SIGNING_KEY = None

                with pytest.raises(RuntimeError) as exc:
                    audit.get_audit_signing_key()

                assert "ARAGORA_AUDIT_SIGNING_KEY required" in str(exc.value)

        # Cleanup - reset global state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None

    def test_audit_requires_signing_key_in_staging(self):
        """SECURITY: Audit signing key must be required in staging."""
        from aragora.rbac import audit

        # Reset the global key state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None

        env_copy = {"ARAGORA_ENV": "staging"}

        with patch.dict(os.environ, env_copy, clear=True):
            # Reset again inside the context
            with audit._signing_key_lock:
                audit._AUDIT_SIGNING_KEY = None

            with pytest.raises(RuntimeError) as exc:
                audit.get_audit_signing_key()

            assert "ARAGORA_AUDIT_SIGNING_KEY required" in str(exc.value)

        # Cleanup - reset global state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None

    def test_audit_generates_ephemeral_key_in_development(self):
        """Test audit generates ephemeral key in development (allowed)."""
        from aragora.rbac import audit

        # Reset the global key state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None

        env_copy = {"ARAGORA_ENV": "development"}

        with patch.dict(os.environ, env_copy, clear=True):
            # Reset again inside the context
            with audit._signing_key_lock:
                audit._AUDIT_SIGNING_KEY = None

            # Should succeed in development
            key = audit.get_audit_signing_key()
            assert key is not None
            assert len(key) >= 32

        # Cleanup - reset global state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None

    def test_audit_loads_signing_key_from_env(self):
        """Test audit correctly loads signing key from environment."""
        from aragora.rbac import audit

        # Reset the global key state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None

        # 32 bytes = 64 hex characters
        test_key_hex = "a" * 64
        env_copy = {
            "ARAGORA_ENV": "production",
            "ARAGORA_AUDIT_SIGNING_KEY": test_key_hex,
        }

        with patch.dict(os.environ, env_copy, clear=True):
            # Reset again inside the context
            with audit._signing_key_lock:
                audit._AUDIT_SIGNING_KEY = None

            key = audit.get_audit_signing_key()
            assert key == bytes.fromhex(test_key_hex)

        # Cleanup - reset global state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None

    def test_audit_rejects_short_signing_key(self):
        """SECURITY: Audit must reject signing keys shorter than 32 bytes."""
        from aragora.rbac import audit

        # Reset the global key state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None

        # 16 bytes = 32 hex characters (too short)
        short_key_hex = "a" * 32
        env_copy = {
            "ARAGORA_ENV": "production",
            "ARAGORA_AUDIT_SIGNING_KEY": short_key_hex,
        }

        with patch.dict(os.environ, env_copy, clear=True):
            # Reset again inside the context
            with audit._signing_key_lock:
                audit._AUDIT_SIGNING_KEY = None

            with pytest.raises(RuntimeError) as exc:
                audit.get_audit_signing_key()

            assert "at least 32 bytes" in str(exc.value)

        # Cleanup - reset global state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None

    def test_audit_rejects_invalid_hex_signing_key(self):
        """SECURITY: Audit must reject invalid hex format signing keys."""
        from aragora.rbac import audit

        # Reset the global key state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None

        # Invalid hex (contains 'g')
        invalid_key_hex = "g" * 64
        env_copy = {
            "ARAGORA_ENV": "production",
            "ARAGORA_AUDIT_SIGNING_KEY": invalid_key_hex,
        }

        with patch.dict(os.environ, env_copy, clear=True):
            # Reset again inside the context
            with audit._signing_key_lock:
                audit._AUDIT_SIGNING_KEY = None

            with pytest.raises(RuntimeError) as exc:
                audit.get_audit_signing_key()

            assert "Invalid" in str(exc.value) or "format" in str(exc.value).lower()

        # Cleanup - reset global state
        with audit._signing_key_lock:
            audit._AUDIT_SIGNING_KEY = None


# ============================================================================
# Cross-cutting Security Tests
# ============================================================================


class TestSecurityDefenseInDepth:
    """Tests for defense-in-depth security patterns."""

    def test_no_bare_except_in_auth_modules(self):
        """SECURITY: Auth modules should not silently catch all exceptions."""
        import ast
        from pathlib import Path

        auth_dir = Path(__file__).parent.parent.parent / "aragora" / "auth"

        bare_excepts = []

        for py_file in auth_dir.glob("*.py"):
            content = py_file.read_text()
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    # Check for bare except or except Exception without handling
                    if node.type is None:  # bare except:
                        bare_excepts.append(f"{py_file.name}:{node.lineno}")
                    elif (
                        isinstance(node.type, ast.Name)
                        and node.type.id == "Exception"
                        and len(node.body) == 1
                        and isinstance(node.body[0], ast.Pass)
                    ):
                        # except Exception: pass
                        bare_excepts.append(f"{py_file.name}:{node.lineno}")

        assert len(bare_excepts) == 0, (
            f"Found bare except handlers that may silently swallow security errors: {bare_excepts}"
        )
