"""
Tests for credential vault salt configuration.

Tests cover:
- Production mode requires ARAGORA_CREDENTIAL_VAULT_SALT to be set
- Development mode allows default salt with warning
- Custom salt is used when provided
- Ephemeral key behavior when no vault key is set
"""

from __future__ import annotations

import hashlib
import os
from unittest.mock import patch

import pytest

from aragora.gateway.security.credential_vault import CredentialVault


class TestCredentialVaultSaltConfiguration:
    """Tests for credential vault salt configuration behavior."""

    def test_production_mode_fails_without_salt(self) -> None:
        """Test production mode fails if ARAGORA_CREDENTIAL_VAULT_SALT is not set."""
        env = {
            "ARAGORA_CREDENTIAL_VAULT_KEY": "test-vault-key",
            "ARAGORA_ENV": "production",
        }
        # Ensure salt is not set
        with patch.dict(os.environ, env, clear=False):
            # Remove salt if present
            os.environ.pop("ARAGORA_CREDENTIAL_VAULT_SALT", None)

            with pytest.raises(RuntimeError) as exc_info:
                CredentialVault()

            assert "ARAGORA_CREDENTIAL_VAULT_SALT must be set in production" in str(exc_info.value)
            assert "python -c" in str(exc_info.value)  # Contains generation command

    def test_production_mode_prod_alias_fails_without_salt(self) -> None:
        """Test 'prod' alias for production mode also fails without salt."""
        env = {
            "ARAGORA_CREDENTIAL_VAULT_KEY": "test-vault-key",
            "ARAGORA_ENV": "prod",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("ARAGORA_CREDENTIAL_VAULT_SALT", None)

            with pytest.raises(RuntimeError) as exc_info:
                CredentialVault()

            assert "ARAGORA_CREDENTIAL_VAULT_SALT must be set in production" in str(exc_info.value)

    def test_staging_mode_fails_without_salt(self) -> None:
        """Test staging mode also fails without salt."""
        env = {
            "ARAGORA_CREDENTIAL_VAULT_KEY": "test-vault-key",
            "ARAGORA_ENV": "staging",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("ARAGORA_CREDENTIAL_VAULT_SALT", None)

            with pytest.raises(RuntimeError) as exc_info:
                CredentialVault()

            assert "ARAGORA_CREDENTIAL_VAULT_SALT must be set in production" in str(exc_info.value)

    def test_development_mode_allows_default_salt_with_warning(self, caplog) -> None:
        """Test development mode allows default salt but logs a warning."""
        env = {
            "ARAGORA_CREDENTIAL_VAULT_KEY": "test-vault-key",
            "ARAGORA_ENV": "development",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("ARAGORA_CREDENTIAL_VAULT_SALT", None)

            import logging

            with caplog.at_level(logging.WARNING):
                vault = CredentialVault()

            # Verify warning was logged
            assert any(
                "Using default credential vault salt" in record.message
                and "NOT SAFE FOR PRODUCTION" in record.message
                for record in caplog.records
            )

            # Verify key was derived with default salt
            expected_key = hashlib.pbkdf2_hmac(
                "sha256",
                b"test-vault-key",
                b"aragora-vault-default-salt",
                iterations=600_000,
            )
            assert vault._encryption_key == expected_key

    def test_no_env_set_allows_default_salt(self, caplog) -> None:
        """Test when ARAGORA_ENV is not set, default salt is allowed."""
        env = {
            "ARAGORA_CREDENTIAL_VAULT_KEY": "test-vault-key",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("ARAGORA_CREDENTIAL_VAULT_SALT", None)
            os.environ.pop("ARAGORA_ENV", None)

            import logging

            with caplog.at_level(logging.WARNING):
                vault = CredentialVault()

            # Verify warning was logged
            assert any(
                "Using default credential vault salt" in record.message for record in caplog.records
            )

            # Vault should be created successfully
            assert vault._encryption_key is not None

    def test_test_env_allows_default_salt(self, caplog) -> None:
        """Test 'test' environment allows default salt."""
        env = {
            "ARAGORA_CREDENTIAL_VAULT_KEY": "test-vault-key",
            "ARAGORA_ENV": "test",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("ARAGORA_CREDENTIAL_VAULT_SALT", None)

            import logging

            with caplog.at_level(logging.WARNING):
                vault = CredentialVault()

            # Vault should be created successfully
            assert vault._encryption_key is not None

    def test_custom_salt_is_used_when_provided(self) -> None:
        """Test custom salt is used when ARAGORA_CREDENTIAL_VAULT_SALT is set."""
        custom_salt = "my-custom-secure-salt-value-12345"
        env = {
            "ARAGORA_CREDENTIAL_VAULT_KEY": "test-vault-key",
            "ARAGORA_CREDENTIAL_VAULT_SALT": custom_salt,
            "ARAGORA_ENV": "production",  # Should work in production with custom salt
        }
        with patch.dict(os.environ, env, clear=False):
            vault = CredentialVault()

            # Verify key was derived with custom salt
            expected_key = hashlib.pbkdf2_hmac(
                "sha256",
                b"test-vault-key",
                custom_salt.encode(),
                iterations=600_000,
            )
            assert vault._encryption_key == expected_key

    def test_custom_salt_in_development_mode(self) -> None:
        """Test custom salt works in development mode too."""
        custom_salt = "dev-custom-salt"
        env = {
            "ARAGORA_CREDENTIAL_VAULT_KEY": "test-vault-key",
            "ARAGORA_CREDENTIAL_VAULT_SALT": custom_salt,
            "ARAGORA_ENV": "development",
        }
        with patch.dict(os.environ, env, clear=False):
            vault = CredentialVault()

            expected_key = hashlib.pbkdf2_hmac(
                "sha256",
                b"test-vault-key",
                custom_salt.encode(),
                iterations=600_000,
            )
            assert vault._encryption_key == expected_key

    def test_ephemeral_key_when_no_vault_key(self, caplog) -> None:
        """Test ephemeral key is used when no vault key is set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARAGORA_CREDENTIAL_VAULT_KEY", None)
            os.environ.pop("ARAGORA_CREDENTIAL_VAULT_SALT", None)
            os.environ.pop("ARAGORA_ENV", None)

            import logging

            with caplog.at_level(logging.WARNING):
                vault = CredentialVault()

            # Verify warning about ephemeral key
            assert any(
                "Using ephemeral credential vault key" in record.message
                and "credentials will not persist" in record.message
                for record in caplog.records
            )

            # Key should be 32 bytes
            assert len(vault._encryption_key) == 32

    def test_ephemeral_key_is_random(self) -> None:
        """Test ephemeral keys are random on each instantiation."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARAGORA_CREDENTIAL_VAULT_KEY", None)

            vault1 = CredentialVault()
            vault2 = CredentialVault()

            # Each vault should have a different ephemeral key
            assert vault1._encryption_key != vault2._encryption_key

    def test_case_insensitive_env_check(self) -> None:
        """Test ARAGORA_ENV check is case insensitive."""
        env = {
            "ARAGORA_CREDENTIAL_VAULT_KEY": "test-vault-key",
            "ARAGORA_ENV": "PRODUCTION",  # Uppercase
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("ARAGORA_CREDENTIAL_VAULT_SALT", None)

            with pytest.raises(RuntimeError) as exc_info:
                CredentialVault()

            assert "ARAGORA_CREDENTIAL_VAULT_SALT must be set in production" in str(exc_info.value)

    def test_production_with_salt_succeeds(self) -> None:
        """Test production mode succeeds when salt is properly set."""
        env = {
            "ARAGORA_CREDENTIAL_VAULT_KEY": "production-vault-key",
            "ARAGORA_CREDENTIAL_VAULT_SALT": "production-secure-salt-abc123",
            "ARAGORA_ENV": "production",
        }
        with patch.dict(os.environ, env, clear=False):
            vault = CredentialVault()

            # Should create successfully
            assert vault._encryption_key is not None
            assert len(vault._encryption_key) == 32

            # Should be able to encrypt/decrypt
            encrypted = vault._encrypt("test-secret")
            decrypted = vault._decrypt(encrypted)
            assert decrypted == "test-secret"
