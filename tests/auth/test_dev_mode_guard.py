"""Tests for token validation dev-mode guard edge cases.

Supplements tests/auth/test_oidc_security.py with additional edge cases:
- Whitespace in environment variable values
- ARAGORA_ENV values that look production-like but differ
- Startup validation with empty string edge cases
- Double-opt-in invariant: production + fallback always blocked
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from aragora.auth.oidc import (
    _allow_dev_auth_fallback,
    _is_production_mode,
    validate_oidc_security_settings,
)
from aragora.auth.sso import SSOConfigurationError


# ---------------------------------------------------------------------------
# _is_production_mode edge cases
# ---------------------------------------------------------------------------


class TestProductionModeEdgeCases:
    """Edge-case inputs for _is_production_mode."""

    def test_empty_string_treated_as_production(self):
        """Empty ARAGORA_ENV is not in the dev allow-list, so production."""
        with patch.dict(os.environ, {"ARAGORA_ENV": ""}, clear=True):
            assert _is_production_mode() is True

    def test_whitespace_only_treated_as_production(self):
        with patch.dict(os.environ, {"ARAGORA_ENV": "  "}, clear=True):
            assert _is_production_mode() is True

    def test_staging_treated_as_production(self):
        """Staging environments should use production-safe defaults."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "staging"}, clear=True):
            assert _is_production_mode() is True

    def test_prod_shorthand_treated_as_production(self):
        with patch.dict(os.environ, {"ARAGORA_ENV": "prod"}, clear=True):
            assert _is_production_mode() is True

    def test_uppercase_dev_normalised(self):
        """DEV, DEVELOPMENT, LOCAL, TEST should all be dev mode (case-insensitive)."""
        for val in ["DEV", "DEVELOPMENT", "LOCAL", "TEST", "Dev", "dEv"]:
            with patch.dict(os.environ, {"ARAGORA_ENV": val}, clear=True):
                assert _is_production_mode() is False, f"Expected dev mode for {val!r}"


# ---------------------------------------------------------------------------
# _allow_dev_auth_fallback edge cases
# ---------------------------------------------------------------------------


class TestDevAuthFallbackEdgeCases:
    """Edge-case inputs for _allow_dev_auth_fallback."""

    def test_fallback_flag_with_whitespace(self):
        """Whitespace around the flag value should be treated as disabled."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ENV": "development", "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": " 1 "},
            clear=True,
        ):
            # " 1 ".lower() is " 1 " which is NOT in ("1", "true", "yes")
            assert _allow_dev_auth_fallback() is False

    def test_fallback_flag_TRUE_uppercase(self):
        with patch.dict(
            os.environ,
            {"ARAGORA_ENV": "development", "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "TRUE"},
            clear=True,
        ):
            assert _allow_dev_auth_fallback() is True

    def test_fallback_flag_Yes_mixed_case(self):
        with patch.dict(
            os.environ,
            {"ARAGORA_ENV": "development", "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "Yes"},
            clear=True,
        ):
            assert _allow_dev_auth_fallback() is True

    def test_fallback_disabled_with_random_string(self):
        with patch.dict(
            os.environ,
            {"ARAGORA_ENV": "development", "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "enabled"},
            clear=True,
        ):
            assert _allow_dev_auth_fallback() is False


# ---------------------------------------------------------------------------
# validate_oidc_security_settings edge cases
# ---------------------------------------------------------------------------


class TestStartupValidationEdgeCases:
    """Edge-case inputs for validate_oidc_security_settings."""

    def test_production_with_empty_fallback_raises(self):
        """Even an empty string for the fallback flag should trigger rejection."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ENV": "production", "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": ""},
            clear=True,
        ):
            with pytest.raises(SSOConfigurationError):
                validate_oidc_security_settings()

    def test_production_with_false_fallback_still_raises(self):
        """Setting the var to 'false' in production is still a misconfiguration."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ENV": "production", "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "false"},
            clear=True,
        ):
            with pytest.raises(SSOConfigurationError):
                validate_oidc_security_settings()

    def test_no_env_vars_at_all_succeeds(self):
        """Completely clean environment should default to production and pass."""
        with patch.dict(os.environ, {}, clear=True):
            # No ARAGORA_ALLOW_DEV_AUTH_FALLBACK present → no rejection
            validate_oidc_security_settings()

    def test_dev_mode_without_fallback_flag_succeeds(self):
        with patch.dict(os.environ, {"ARAGORA_ENV": "dev"}, clear=True):
            validate_oidc_security_settings()

    def test_test_mode_without_fallback_flag_succeeds(self):
        with patch.dict(os.environ, {"ARAGORA_ENV": "test"}, clear=True):
            validate_oidc_security_settings()


# ---------------------------------------------------------------------------
# Double-opt-in invariant
# ---------------------------------------------------------------------------


class TestDoubleOptInInvariant:
    """The fallback should NEVER be active unless BOTH conditions are met."""

    @pytest.mark.parametrize(
        "env_val,flag_val,expected",
        [
            ("production", "1", False),
            ("production", None, False),
            ("development", None, False),
            ("development", "0", False),
            ("development", "1", True),
            ("development", "true", True),
            ("dev", "yes", True),
            ("local", "1", True),
            ("test", "true", True),
            ("staging", "1", False),
            ("prod", "1", False),
        ],
    )
    def test_double_opt_in(self, env_val, flag_val, expected):
        env = {"ARAGORA_ENV": env_val}
        if flag_val is not None:
            env["ARAGORA_ALLOW_DEV_AUTH_FALLBACK"] = flag_val
        with patch.dict(os.environ, env, clear=True):
            assert _allow_dev_auth_fallback() is expected, (
                f"env={env_val!r} flag={flag_val!r} expected={expected}"
            )
