"""
Tests for credential vault key derivation function (KDF) upgrade.

Tests cover:
- PBKDF2-HMAC-SHA256 key derivation produces different keys than SHA-256
- PBKDF2 key derivation is deterministic
- Different salts produce different keys
- Default salt used when environment variable not set
- Custom salt from environment variable
- Ephemeral key generation when no env key
- Ephemeral key logs warning
- Migration from legacy SHA-256 to PBKDF2 encrypted data
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from unittest.mock import patch

import pytest

from aragora.gateway.security.credential_vault import (
    CredentialVault,
)


def _make_env(key: str | None = None, salt: str | None = None) -> dict[str, str]:
    """Build a clean environment dict with only the specified vault vars."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("ARAGORA_CREDENTIAL_VAULT_KEY", "ARAGORA_CREDENTIAL_VAULT_SALT")
    }
    if key is not None:
        env["ARAGORA_CREDENTIAL_VAULT_KEY"] = key
    if salt is not None:
        env["ARAGORA_CREDENTIAL_VAULT_SALT"] = salt
    return env


# =============================================================================
# PBKDF2 Key Derivation Tests
# =============================================================================


class TestCredentialVaultKDF:
    """Tests for PBKDF2-HMAC-SHA256 key derivation."""

    def test_pbkdf2_produces_different_key_than_sha256(self) -> None:
        """Test that PBKDF2 key derivation produces a different key than plain SHA-256."""
        env_key = "test-secret-key"

        # Legacy SHA-256 derivation
        sha256_key = hashlib.sha256(env_key.encode()).digest()

        # New PBKDF2 derivation (with default salt)
        pbkdf2_key = hashlib.pbkdf2_hmac(
            "sha256",
            env_key.encode(),
            b"aragora-vault-default-salt",
            iterations=600_000,
        )

        assert sha256_key != pbkdf2_key
        assert len(sha256_key) == 32
        assert len(pbkdf2_key) == 32

    def test_pbkdf2_key_is_deterministic(self) -> None:
        """Test that PBKDF2 produces the same key for the same input."""
        env = _make_env(key="deterministic-test-key")

        with patch.dict(os.environ, env, clear=True):
            vault1 = CredentialVault()
            vault2 = CredentialVault()

        assert vault1._encryption_key == vault2._encryption_key

    def test_different_salts_produce_different_keys(self) -> None:
        """Test that different salts produce different derived keys."""
        env_key = "same-password"

        key_salt_a = hashlib.pbkdf2_hmac(
            "sha256",
            env_key.encode(),
            b"salt-alpha",
            iterations=600_000,
        )

        key_salt_b = hashlib.pbkdf2_hmac(
            "sha256",
            env_key.encode(),
            b"salt-beta",
            iterations=600_000,
        )

        assert key_salt_a != key_salt_b

        # Also verify via the vault itself
        env_a = _make_env(key=env_key, salt="salt-alpha")
        with patch.dict(os.environ, env_a, clear=True):
            vault_a = CredentialVault()

        env_b = _make_env(key=env_key, salt="salt-beta")
        with patch.dict(os.environ, env_b, clear=True):
            vault_b = CredentialVault()

        assert vault_a._encryption_key != vault_b._encryption_key

    def test_default_salt_used_when_env_not_set(self) -> None:
        """Test that the default salt is used when ARAGORA_CREDENTIAL_VAULT_SALT is not set."""
        env_key = "test-key-for-default-salt"

        expected = hashlib.pbkdf2_hmac(
            "sha256",
            env_key.encode(),
            b"aragora-vault-default-salt",
            iterations=600_000,
        )

        env = _make_env(key=env_key)
        with patch.dict(os.environ, env, clear=True):
            vault = CredentialVault()

        assert vault._encryption_key == expected

    def test_custom_salt_from_env(self) -> None:
        """Test that a custom salt from the environment variable is used."""
        env_key = "test-key-custom-salt"
        custom_salt = "my-custom-production-salt"

        expected = hashlib.pbkdf2_hmac(
            "sha256",
            env_key.encode(),
            custom_salt.encode(),
            iterations=600_000,
        )

        env = _make_env(key=env_key, salt=custom_salt)
        with patch.dict(os.environ, env, clear=True):
            vault = CredentialVault()

        assert vault._encryption_key == expected

    def test_ephemeral_key_when_no_env_key(self) -> None:
        """Test that an ephemeral key is generated when no env key is set."""
        env = _make_env()  # No key, no salt
        with patch.dict(os.environ, env, clear=True):
            vault = CredentialVault()

        assert vault._encryption_key is not None
        assert len(vault._encryption_key) == 32
        assert isinstance(vault._encryption_key, bytes)

        # Ephemeral keys should be different each time
        with patch.dict(os.environ, env, clear=True):
            vault2 = CredentialVault()

        assert vault._encryption_key != vault2._encryption_key

    def test_ephemeral_key_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that using an ephemeral key logs a warning."""
        env = _make_env()  # No key, no salt

        with caplog.at_level(logging.WARNING, logger="aragora.gateway.security.credential_vault"):
            with patch.dict(os.environ, env, clear=True):
                CredentialVault()

        assert any(
            "ephemeral credential vault key" in record.message.lower() for record in caplog.records
        ), f"Expected warning about ephemeral key, got: {[r.message for r in caplog.records]}"

    def test_migration_from_sha256_to_pbkdf2(self) -> None:
        """Test that data encrypted with legacy SHA-256 key can be decrypted via migration helper."""
        env_key = "migration-test-key"
        secret_value = "super-secret-api-key"

        # Simulate legacy encryption: create a vault with SHA-256 derived key
        legacy_key = hashlib.sha256(env_key.encode()).digest()
        legacy_vault = CredentialVault(encryption_key=legacy_key)
        encrypted_data = legacy_vault._encrypt(secret_value)

        # Now create a vault with the new PBKDF2 key derivation
        env = _make_env(key=env_key)
        with patch.dict(os.environ, env, clear=True):
            new_vault = CredentialVault()

        # Verify the new key is different from legacy
        assert new_vault._encryption_key != legacy_key

        # The new PBKDF2 key should NOT be able to decrypt legacy data directly
        with pytest.raises(Exception):
            new_vault._decrypt(encrypted_data)

        # But the migration helper should succeed by falling back to legacy key
        with patch.dict(os.environ, {"ARAGORA_CREDENTIAL_VAULT_KEY": env_key}):
            result = new_vault._try_decrypt_with_migration(
                encrypted_data, new_vault._encryption_key
            )

        assert result == secret_value
