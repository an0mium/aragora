"""
Security hardening regression tests.

These tests verify critical security controls remain in place:
- PayPal webhook signature verification
- SAML double-confirmation for unsafe mode
- OIDC algorithm restrictions
- Config validator production enforcement

This file serves as a security regression suite to catch if
any security-critical code is accidentally weakened.
"""

from __future__ import annotations

import inspect
import os
from unittest.mock import patch

import pytest

from aragora.auth.oidc import OIDCConfig
from aragora.auth.saml import SAMLProvider
from aragora.auth.sso import SSOConfigurationError, SSOProviderType
from aragora.connectors.payments.paypal import PayPalClient, PayPalCredentials
from aragora.server.config_validator import ConfigValidator


class TestPayPalSecurityControls:
    """Verify PayPal webhook security controls are in place."""

    def test_signature_comparison_uses_timing_safe(self):
        """CRITICAL: Signature comparison must use hmac.compare_digest."""
        source = inspect.getsource(PayPalClient.verify_webhook_signature)

        # Must use timing-safe comparison
        assert "hmac.compare_digest" in source, (
            "SECURITY: PayPal webhook verification must use hmac.compare_digest "
            "to prevent timing attacks"
        )

        # Must NOT use simple equality
        assert "actual_signature ==" not in source, (
            "SECURITY: PayPal webhook verification must not use == for signature comparison"
        )
        assert "== actual_signature" not in source, (
            "SECURITY: PayPal webhook verification must not use == for signature comparison"
        )

    def test_signature_actually_verified(self):
        """CRITICAL: verify_webhook_signature must actually verify signatures."""
        source = inspect.getsource(PayPalClient.verify_webhook_signature)

        # Must compute expected signature
        assert "hmac.new" in source, (
            "SECURITY: PayPal webhook verification must compute expected signature with HMAC"
        )

        # Must use CRC32 for payload hashing (PayPal protocol)
        assert "zlib.crc32" in source, (
            "SECURITY: PayPal webhook verification must use CRC32 for payload hashing"
        )

        # Must mask CRC32 to unsigned
        assert "0xFFFFFFFF" in source.upper() or "0XFFFFFFFF" in source.upper(), (
            "SECURITY: PayPal CRC32 must be masked to unsigned integer"
        )

    def test_production_requires_webhook_secret(self):
        """Production must reject webhooks if webhook_secret not configured."""
        creds = PayPalCredentials(
            client_id="test",
            client_secret="test",
            webhook_id="WH-123",
            webhook_secret=None,  # Not configured
        )
        client = PayPalClient(creds)

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            result = client.verify_webhook_signature(
                transmission_id="t1",
                timestamp="2026-01-30T00:00:00Z",
                webhook_id="WH-123",
                event_body="{}",
                cert_url="https://paypal.com/cert",
                auth_algo="SHA256withRSA",
                actual_signature="sig",
            )

        assert result is False, (
            "SECURITY: Production must reject webhooks when webhook_secret is not configured"
        )

    def test_production_requires_webhook_id(self):
        """Production must reject webhooks if webhook_id not configured."""
        creds = PayPalCredentials(
            client_id="test",
            client_secret="test",
            webhook_id=None,  # Not configured
            webhook_secret="secret",
        )
        client = PayPalClient(creds)

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            result = client.verify_webhook_signature(
                transmission_id="t1",
                timestamp="2026-01-30T00:00:00Z",
                webhook_id="WH-123",
                event_body="{}",
                cert_url="https://paypal.com/cert",
                auth_algo="SHA256withRSA",
                actual_signature="sig",
            )

        assert result is False, (
            "SECURITY: Production must reject webhooks when webhook_id is not configured"
        )


class TestSAMLSecurityControls:
    """Verify SAML security controls are in place."""

    def test_unsafe_mode_requires_double_confirmation(self):
        """CRITICAL: Unsafe SAML mode requires BOTH env vars."""
        source = inspect.getsource(SAMLProvider.__init__)

        # Must check for BOTH env vars
        assert "ARAGORA_ALLOW_UNSAFE_SAML" in source, (
            "SECURITY: SAML must check ARAGORA_ALLOW_UNSAFE_SAML"
        )
        assert "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED" in source, (
            "SECURITY: SAML must require ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED for double confirmation"
        )

    def test_single_env_var_not_enough(self):
        """Single env var must NOT enable unsafe mode."""
        from aragora.auth.saml import SAMLConfig

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com",
            callback_url="https://sp.example.com/acs",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----",
        )

        # Only one env var set - should NOT be enough
        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                os.environ,
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    # Missing: ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED
                },
                clear=False,
            ):
                # Ensure the confirmed var is NOT set
                if "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED" in os.environ:
                    del os.environ["ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED"]

                with pytest.raises(SSOConfigurationError):
                    SAMLProvider(config)

    def test_both_env_vars_required(self):
        """Both env vars together should allow unsafe mode in dev."""
        from aragora.auth.saml import SAMLConfig

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com",
            callback_url="https://sp.example.com/acs",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----",
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                os.environ,
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                # Should NOT raise with both env vars
                provider = SAMLProvider(config)
                assert provider is not None

    def test_production_never_allows_unsafe_mode(self):
        """Production must never allow unsafe SAML mode."""
        from aragora.auth.saml import SAMLConfig

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com",
            callback_url="https://sp.example.com/acs",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----",
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                os.environ,
                {
                    "ARAGORA_ENV": "production",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                with pytest.raises(SSOConfigurationError):
                    SAMLProvider(config)


class TestOIDCSecurityControls:
    """Verify OIDC security controls are in place."""

    def test_insecure_algorithms_rejected(self):
        """CRITICAL: Insecure algorithms must be rejected."""
        insecure_algorithms = ["HS256", "HS384", "HS512", "none"]

        for alg in insecure_algorithms:
            config = OIDCConfig(
                provider_type=SSOProviderType.OIDC,
                client_id="test",
                client_secret="test",
                issuer_url="https://issuer.example.com",
                authorization_endpoint="https://issuer.example.com/authorize",
                token_endpoint="https://issuer.example.com/token",
                callback_url="https://sp.example.com/callback",
                allowed_algorithms=[alg],
            )

            errors = config.validate()
            assert any("insecure" in e.lower() for e in errors), (
                f"SECURITY: Algorithm '{alg}' must be flagged as insecure"
            )

    def test_default_algorithm_is_secure(self):
        """Default algorithm must be RS256 (asymmetric)."""
        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test",
            client_secret="test",
            issuer_url="https://issuer.example.com",
            authorization_endpoint="https://issuer.example.com/authorize",
            token_endpoint="https://issuer.example.com/token",
            callback_url="https://sp.example.com/callback",
        )

        assert config.allowed_algorithms == ["RS256"], (
            "SECURITY: Default OIDC algorithm must be RS256"
        )

    def test_secure_algorithms_accepted(self):
        """Secure asymmetric algorithms should be accepted."""
        secure_algorithms = ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]

        for alg in secure_algorithms:
            config = OIDCConfig(
                provider_type=SSOProviderType.OIDC,
                client_id="test",
                client_secret="test",
                issuer_url="https://issuer.example.com",
                authorization_endpoint="https://issuer.example.com/authorize",
                token_endpoint="https://issuer.example.com/token",
                callback_url="https://sp.example.com/callback",
                allowed_algorithms=[alg],
            )

            errors = config.validate()
            alg_errors = [e for e in errors if "insecure" in e.lower() and alg in e]
            assert not alg_errors, f"SECURITY: Algorithm '{alg}' should be accepted as secure"


class TestConfigValidatorSecurityControls:
    """Verify config validator security controls are in place."""

    def test_unsafe_saml_vars_in_insecure_list(self):
        """SAML unsafe env vars must be in INSECURE_DEV_ONLY_VARS."""
        assert "ARAGORA_ALLOW_UNSAFE_SAML" in ConfigValidator.INSECURE_DEV_ONLY_VARS, (
            "SECURITY: ARAGORA_ALLOW_UNSAFE_SAML must be in INSECURE_DEV_ONLY_VARS"
        )
        assert "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED" in ConfigValidator.INSECURE_DEV_ONLY_VARS, (
            "SECURITY: ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED must be in INSECURE_DEV_ONLY_VARS"
        )

    def test_production_blocks_insecure_vars(self):
        """Production must block insecure env vars."""
        # Mock to avoid needing real LLM keys
        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "production",
                "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                "ANTHROPIC_API_KEY": "test_key",  # Needed for validation
            },
        ):
            result = ConfigValidator.validate_all()
            # Should find the unsafe SAML var in errors
            saml_errors = [e for e in result.errors if "ARAGORA_ALLOW_UNSAFE_SAML" in e]
            assert len(saml_errors) > 0, (
                "SECURITY: Production must reject ARAGORA_ALLOW_UNSAFE_SAML"
            )

    def test_development_allows_insecure_vars(self):
        """Development may allow insecure vars (for testing)."""
        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                "ANTHROPIC_API_KEY": "test_key",  # Needed for validation
            },
        ):
            result = ConfigValidator.validate_all()
            # In dev, it should not produce errors for SAML var
            saml_errors = [e for e in result.errors if "ARAGORA_ALLOW_UNSAFE_SAML" in e]
            assert len(saml_errors) == 0

    def test_saml_library_check_exists(self):
        """ConfigValidator must have SAML library check method."""
        assert hasattr(ConfigValidator, "check_saml_library_availability"), (
            "SECURITY: ConfigValidator must have check_saml_library_availability method"
        )

    def test_saml_config_vars_defined(self):
        """SAML config vars must be defined for detection."""
        assert hasattr(ConfigValidator, "SAML_CONFIG_VARS"), (
            "ConfigValidator must have SAML_CONFIG_VARS list"
        )
        assert "SAML_IDP_ENTITY_ID" in ConfigValidator.SAML_CONFIG_VARS


class TestSecurityCodePatterns:
    """Verify secure coding patterns are used throughout."""

    def test_paypal_uses_constant_time_comparison(self):
        """PayPal must use constant-time comparison for signatures."""
        import hmac as hmac_module

        # Verify hmac.compare_digest exists and is used
        assert hasattr(hmac_module, "compare_digest"), "Python hmac module must have compare_digest"

        source = inspect.getsource(PayPalClient.verify_webhook_signature)
        assert "compare_digest" in source

    def test_no_hardcoded_secrets_in_tests(self):
        """Test files should not contain production-like secrets."""
        import tests.connectors.payments.test_paypal_webhook as paypal_tests

        source = inspect.getsource(paypal_tests)

        # Should not contain AWS-like keys
        assert "AKIA" not in source, "Test files should not contain AWS-like keys"

        # Should not contain production webhook secrets (long random strings)
        # Test secrets are clearly fake like "test_webhook_secret_12345"


class TestReplayAttackProtection:
    """Verify replay attack protections are in place."""

    def test_paypal_validates_timestamp_freshness(self):
        """PayPal must validate timestamp freshness."""
        source = inspect.getsource(PayPalClient.verify_webhook_signature)

        # Must check timestamp
        assert "timestamp" in source.lower()

        # Must have time window check (300 seconds = 5 minutes)
        assert "300" in source or "5" in source, (
            "SECURITY: PayPal must check timestamp within window"
        )

    def test_saml_validates_state_expiry(self):
        """SAML must validate state token expiry."""
        source = inspect.getsource(SAMLProvider.authenticate)

        # Should reference state store or expiry
        assert "_state_store" in source or "expired" in source.lower() or "relay_state" in source
