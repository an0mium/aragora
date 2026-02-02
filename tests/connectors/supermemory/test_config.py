"""Tests for Supermemory configuration."""

import os
from unittest.mock import patch

import pytest

from aragora.connectors.supermemory.config import SupermemoryConfig


class TestSupermemoryConfig:
    """Test SupermemoryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SupermemoryConfig(api_key="sm_test_key")

        assert config.api_key == "sm_test_key"
        assert config.base_url is None
        assert config.timeout_seconds == 30.0
        assert config.sync_threshold == 0.7
        assert config.privacy_filter_enabled is True
        assert config.container_tag == "aragora"
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SupermemoryConfig(
            api_key="sm_custom_key",
            base_url="https://custom.api.supermemory.ai",
            timeout_seconds=60.0,
            sync_threshold=0.5,
            privacy_filter_enabled=False,
            container_tag="custom_tag",
            max_retries=5,
        )

        assert config.api_key == "sm_custom_key"
        assert config.base_url == "https://custom.api.supermemory.ai"
        assert config.timeout_seconds == 60.0
        assert config.sync_threshold == 0.5
        assert config.privacy_filter_enabled is False
        assert config.container_tag == "custom_tag"
        assert config.max_retries == 5

    def test_container_tags_default(self):
        """Test default container tags."""
        config = SupermemoryConfig(api_key="sm_test_key")

        assert "debate_outcomes" in config.container_tags
        assert "consensus" in config.container_tags
        assert "agent_performance" in config.container_tags
        assert "patterns" in config.container_tags
        assert "errors" in config.container_tags

    def test_get_container_tag_existing(self):
        """Test getting an existing container tag."""
        config = SupermemoryConfig(api_key="sm_test_key")

        assert config.get_container_tag("debate_outcomes") == "aragora_debates"
        assert config.get_container_tag("consensus") == "aragora_consensus"

    def test_get_container_tag_fallback(self):
        """Test getting a non-existent container tag falls back to default."""
        config = SupermemoryConfig(api_key="sm_test_key")

        assert config.get_container_tag("unknown_category") == "aragora"

    def test_should_sync_above_threshold(self):
        """Test should_sync returns True for high importance."""
        config = SupermemoryConfig(api_key="sm_test_key", sync_threshold=0.7)

        assert config.should_sync(0.8) is True
        assert config.should_sync(0.7) is True
        assert config.should_sync(1.0) is True

    def test_should_sync_below_threshold(self):
        """Test should_sync returns False for low importance."""
        config = SupermemoryConfig(api_key="sm_test_key", sync_threshold=0.7)

        assert config.should_sync(0.6) is False
        assert config.should_sync(0.5) is False
        assert config.should_sync(0.0) is False


class TestSupermemoryConfigFromEnv:
    """Test SupermemoryConfig.from_env()."""

    def test_from_env_no_api_key(self):
        """Test from_env returns None when no API key is set."""
        with patch.dict(os.environ, {}, clear=True):
            config = SupermemoryConfig.from_env()
            assert config is None

    def test_from_env_with_api_key(self):
        """Test from_env creates config with API key."""
        env = {"SUPERMEMORY_API_KEY": "sm_env_test_key"}
        with patch.dict(os.environ, env, clear=True):
            config = SupermemoryConfig.from_env()

            assert config is not None
            assert config.api_key == "sm_env_test_key"
            assert config.timeout_seconds == 30.0  # Default

    def test_from_env_with_all_vars(self):
        """Test from_env reads all environment variables."""
        env = {
            "SUPERMEMORY_API_KEY": "sm_full_env_key",
            "SUPERMEMORY_BASE_URL": "https://custom.url",
            "SUPERMEMORY_TIMEOUT": "60",
            "SUPERMEMORY_SYNC_THRESHOLD": "0.5",
            "SUPERMEMORY_PRIVACY_FILTER": "false",
            "SUPERMEMORY_CONTAINER_TAG": "custom_container",
        }
        with patch.dict(os.environ, env, clear=True):
            config = SupermemoryConfig.from_env()

            assert config is not None
            assert config.api_key == "sm_full_env_key"
            assert config.base_url == "https://custom.url"
            assert config.timeout_seconds == 60.0
            assert config.sync_threshold == 0.5
            assert config.privacy_filter_enabled is False
            assert config.container_tag == "custom_container"

    def test_from_env_privacy_filter_true(self):
        """Test privacy filter defaults to true."""
        env = {
            "SUPERMEMORY_API_KEY": "sm_test_key",
            "SUPERMEMORY_PRIVACY_FILTER": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            config = SupermemoryConfig.from_env()

            assert config.privacy_filter_enabled is True
