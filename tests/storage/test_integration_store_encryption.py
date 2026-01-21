"""
Tests for integration store encryption functionality.

Tests cover:
- Encryption roundtrip for sensitive fields
- Backward compatibility with unencrypted data
- Graceful degradation when crypto unavailable
- Field-level encryption (only sensitive keys encrypted)
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from aragora.storage.integration_store import (
    _encrypt_settings,
    _decrypt_settings,
    SENSITIVE_KEYS,
    IntegrationConfig,
    SQLiteIntegrationStore,
)


class TestEncryptionHelpers:
    """Tests for _encrypt_settings and _decrypt_settings."""

    def test_encrypt_sensitive_keys_only(self):
        """Only sensitive keys are encrypted."""
        settings = {
            "api_key": "secret123",
            "username": "testuser",
            "webhook_url": "https://hook.example.com/secret",
            "enabled": True,
        }

        encrypted = _encrypt_settings(settings)

        # Non-sensitive keys unchanged
        assert encrypted["username"] == "testuser"
        assert encrypted["enabled"] is True

        # Sensitive keys should be dicts with _encrypted marker (if crypto available)
        from aragora.security.encryption import CRYPTO_AVAILABLE
        if CRYPTO_AVAILABLE:
            assert isinstance(encrypted["api_key"], dict)
            assert encrypted["api_key"].get("_encrypted") is True
            assert isinstance(encrypted["webhook_url"], dict)
            assert encrypted["webhook_url"].get("_encrypted") is True

    def test_decrypt_roundtrip(self):
        """Encrypted settings decrypt to original values."""
        settings = {
            "api_key": "my-secret-api-key",
            "access_token": "oauth-token-123",
            "username": "admin",
            "port": 8080,
        }

        encrypted = _encrypt_settings(settings)
        decrypted = _decrypt_settings(encrypted)

        assert decrypted == settings

    def test_decrypt_legacy_unencrypted(self):
        """Unencrypted legacy data passes through unchanged."""
        legacy_settings = {
            "api_key": "plain-text-key",
            "username": "testuser",
        }

        # No _encrypted markers, should return as-is
        result = _decrypt_settings(legacy_settings)
        assert result == legacy_settings

    def test_encrypt_empty_settings(self):
        """Empty settings return empty dict."""
        assert _encrypt_settings({}) == {}
        assert _encrypt_settings(None) is None

    def test_decrypt_empty_settings(self):
        """Empty settings return empty dict."""
        assert _decrypt_settings({}) == {}
        assert _decrypt_settings(None) is None

    def test_all_sensitive_keys_encrypted(self):
        """All defined sensitive keys are encrypted."""
        settings = {key: f"value-{key}" for key in SENSITIVE_KEYS}
        settings["safe_key"] = "not-encrypted"

        encrypted = _encrypt_settings(settings)

        from aragora.security.encryption import CRYPTO_AVAILABLE
        if CRYPTO_AVAILABLE:
            for key in SENSITIVE_KEYS:
                assert isinstance(encrypted[key], dict), f"{key} should be encrypted"
                assert encrypted[key].get("_encrypted") is True

            assert encrypted["safe_key"] == "not-encrypted"


class TestEncryptionGracefulDegradation:
    """Tests for graceful degradation when crypto unavailable."""

    def test_encrypt_without_crypto(self):
        """Settings pass through when crypto unavailable."""
        settings = {"api_key": "secret", "username": "test"}

        with patch("aragora.storage.integration_store.CRYPTO_AVAILABLE", False):
            result = _encrypt_settings(settings)
            assert result == settings

    def test_decrypt_without_crypto(self):
        """Settings pass through when crypto unavailable."""
        settings = {"api_key": "secret", "username": "test"}

        with patch("aragora.storage.integration_store.CRYPTO_AVAILABLE", False):
            result = _decrypt_settings(settings)
            assert result == settings


class TestIntegrationConfigEncryption:
    """Tests for IntegrationConfig encryption integration."""

    @pytest.fixture
    def config(self):
        """Sample integration config."""
        return IntegrationConfig(
            type="slack",
            user_id="user-456",
            settings={
                "api_key": "xoxb-secret-token",
                "channel": "#general",
                "bot_token": "xoxb-bot-token",
            },
        )

    def test_settings_contain_sensitive_data(self, config):
        """Config settings contain sensitive data that needs encryption."""
        assert "api_key" in config.settings
        assert "bot_token" in config.settings
        assert config.settings["api_key"] == "xoxb-secret-token"

    def test_encrypt_config_settings(self, config):
        """Settings can be encrypted."""
        encrypted = _encrypt_settings(config.settings)
        from aragora.security.encryption import CRYPTO_AVAILABLE
        if CRYPTO_AVAILABLE:
            assert isinstance(encrypted["api_key"], dict)
            assert encrypted["api_key"].get("_encrypted") is True


class TestSQLiteStoreEncryptionIntegration:
    """Tests for SQLite store encryption integration."""

    @pytest.fixture
    def store(self, tmp_path):
        """SQLite store with temp database."""
        db_path = tmp_path / "test_integrations.db"
        return SQLiteIntegrationStore(str(db_path))

    @pytest.fixture
    def config(self):
        """Sample config with sensitive data."""
        return IntegrationConfig(
            type="github",
            user_id="user-1",
            settings={
                "access_token": "ghp_secret_token_12345",
                "webhook_url": "https://api.github.com/webhook/secret",
                "repo": "org/repo",
            },
        )

    @pytest.mark.asyncio
    async def test_save_encrypts_settings(self, store, config):
        """Settings are encrypted when saved."""
        await store.save(config)

        # Read raw from database
        import sqlite3
        conn = sqlite3.connect(str(store.db_path))
        cursor = conn.execute(
            "SELECT settings_json FROM integrations WHERE integration_type = ? AND user_id = ?",
            (config.type, config.user_id)
        )
        row = cursor.fetchone()
        conn.close()

        stored_settings = json.loads(row[0])

        from aragora.security.encryption import CRYPTO_AVAILABLE
        if CRYPTO_AVAILABLE:
            # Sensitive fields should be encrypted
            assert isinstance(stored_settings.get("access_token"), dict)
            assert stored_settings["access_token"].get("_encrypted") is True
            # Non-sensitive unchanged
            assert stored_settings["repo"] == "org/repo"

    @pytest.mark.asyncio
    async def test_load_decrypts_settings(self, store, config):
        """Settings are decrypted when loaded."""
        await store.save(config)
        loaded = await store.get(config.type, config.user_id)

        assert loaded is not None
        assert loaded.settings["access_token"] == "ghp_secret_token_12345"
        assert loaded.settings["webhook_url"] == "https://api.github.com/webhook/secret"
        assert loaded.settings["repo"] == "org/repo"

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_all_fields(self, store, config):
        """Full save/load roundtrip preserves all data."""
        await store.save(config)
        loaded = await store.get(config.type, config.user_id)

        assert loaded.type == config.type
        assert loaded.user_id == config.user_id
        assert loaded.settings == config.settings
